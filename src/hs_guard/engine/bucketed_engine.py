# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

SESSION_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
TOKEN_BUCKETS = (1, 2, 4, 8, 16, 32, 64)


def bucket(value, choices):
    for choice in choices:
        if value <= choice:
            return choice
    raise ValueError(f"{value} exceeds the largest bucket {choices[-1]}")


class BucketedStreamEngine:
    session_buckets = SESSION_BUCKETS

    def __init__(self, model, max_slots, window=512, gdn_kernel="recurrent", use_graphs=True,
                 roles=("user", "assistant")):
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        self.rope = apply_rotary_pos_emb
        self.model, self.backbone = model, model.backbone
        cfg = self.backbone.config
        self.layers = list(self.backbone.layers[:cfg.num_hidden_layers])
        types = list(cfg.layer_types[:cfg.num_hidden_layers])
        self.kind = [(t, sum(1 for u in types[:i] if u == t)) for i, t in enumerate(types)]
        self.kernel = {"chunk": chunk_gated_delta_rule, "recurrent": fused_recurrent_gated_delta_rule}[gdn_kernel]
        self.window, self.max_slots, self.roles, self.use_graphs = window, max_slots, roles, use_graphs
        gdn = next(layer.linear_attn for layer in self.layers if hasattr(layer, "linear_attn"))
        att = next(layer.self_attn for layer in self.layers if hasattr(layer, "self_attn"))
        n_gdn, n_att = types.count("linear_attention"), types.count("full_attention")
        slots, dev, bf16 = max_slots + 1, "cuda", torch.bfloat16
        self.rec = torch.zeros(n_gdn, slots, gdn.num_v_heads, gdn.head_k_dim, gdn.head_v_dim, dtype=torch.float32, device=dev)
        self.conv = torch.zeros(n_gdn, slots, gdn.conv_kernel_size - 1, gdn.conv_dim, dtype=bf16, device=dev)
        self.ring_k = torch.zeros(n_att, slots, window + 1, cfg.num_key_value_heads, att.head_dim, dtype=bf16, device=dev)
        self.ring_v = torch.zeros_like(self.ring_k)
        self.ring_pos = torch.full((slots, window + 1), -1, dtype=torch.long, device=dev)
        self.length = [0] * max_slots
        self.graphs, self.pool = {}, None
        self.stats = {"graph_replays": 0, "eager_steps": 0, "captures": 0}

    def state_bytes_per_session(self):
        tensors = (self.rec, self.conv, self.ring_k, self.ring_v, self.ring_pos)
        return sum(t.element_size() * t.numel() for t in tensors) // (self.max_slots + 1)

    def reset(self):
        for t in (self.rec, self.conv, self.ring_k, self.ring_v):
            t.zero_()
        self.ring_pos.fill_(-1)
        self.length = [0] * self.max_slots


    def _forward(self, ids, lens, slots, starts):
        nb, lb = ids.shape
        w = self.window
        steps = torch.arange(lb, device=ids.device)
        valid = steps[None] < lens[:, None]
        pos = starts[:, None] + steps[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = self.backbone.embed_tokens(ids)
            cos, sin = self.backbone.rotary_emb(hidden, pos[None].expand(3, nb, lb))
            ring_pos = self.ring_pos[slots, :w]
            key_pos = torch.cat([ring_pos, torch.where(valid, pos, -1)], 1)
            qq, kk = pos[:, :, None], key_pos[:, None, :]
            allowed = ((kk >= 0) & (kk <= qq) & (kk > qq - w)) | ~valid[:, :, None]
            allowed = allowed[:, None]
            ring_col = torch.where(valid, pos % w, w)
            for layer, (kind, index) in zip(self.layers, self.kind):
                residual = hidden
                normed = layer.input_layernorm(hidden)
                if kind == "linear_attention":
                    mixed = self._gdn(layer.linear_attn, normed, index, slots, lens, valid)
                else:
                    mixed = self._attention(layer.self_attn, normed, index, slots, cos, sin, allowed, ring_col)
                hidden = residual + mixed
                hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
            hidden = self.backbone.norm(hidden)
            probs = torch.stack([torch.softmax(self.model.readout(hidden, role)[0].float(), -1)
                                 for role in self.roles])
        self.ring_pos[slots[:, None], ring_col] = torch.where(valid, pos, -1)
        return probs

    def _gdn(self, mod, x, g_index, slots, lens, valid):
        nb, lb, _ = x.shape
        qkv = mod.in_proj_qkv(x)
        z = mod.in_proj_z(x).reshape(nb, lb, mod.num_v_heads, mod.head_v_dim)
        b, a = mod.in_proj_b(x), mod.in_proj_a(x)
        keep = mod.conv_kernel_size - 1
        ext = torch.cat([self.conv[g_index, slots].to(qkv.dtype), qkv], 1)
        conv = F.conv1d(ext.transpose(1, 2), mod.conv1d.weight, groups=mod.conv_dim)
        conv = F.silu(conv.float()).to(qkv.dtype).transpose(1, 2)
        index = (lens[:, None] + torch.arange(keep, device=x.device)[None])[:, :, None].expand(-1, -1, ext.shape[2])
        self.conv[g_index, slots] = torch.gather(ext, 1, index).to(self.conv.dtype)
        q, k, v = torch.split(conv, [mod.key_dim, mod.key_dim, mod.value_dim], dim=-1)
        q = q.reshape(nb, lb, -1, mod.head_k_dim).contiguous()
        k = k.reshape(nb, lb, -1, mod.head_k_dim).contiguous()
        v = v.reshape(nb, lb, -1, mod.head_v_dim).contiguous()
        mask = valid[:, :, None]
        beta = b.sigmoid() * mask
        g = (-mod.A_log.float().exp() * F.softplus(a.float() + mod.dt_bias)) * mask
        if mod.num_v_heads // mod.num_k_heads > 1:
            q = q.repeat_interleave(mod.num_v_heads // mod.num_k_heads, dim=2)
            k = k.repeat_interleave(mod.num_v_heads // mod.num_k_heads, dim=2)
        out, final = self.kernel(q, k, v, g=g, beta=beta, initial_state=self.rec[g_index, slots],
                                 output_final_state=True, use_qk_l2norm_in_kernel=True)
        self.rec[g_index, slots] = final.float()
        out = mod.norm(out.reshape(-1, mod.head_v_dim), z.reshape(-1, mod.head_v_dim)).reshape(nb, lb, -1)
        return mod.out_proj(out)

    def _attention(self, mod, x, a_index, slots, cos, sin, allowed, ring_col):
        nb, lb, _ = x.shape
        w = self.window
        query, gate = torch.chunk(mod.q_proj(x).view(nb, lb, -1, mod.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(nb, lb, -1)
        query = mod.q_norm(query).transpose(1, 2)
        key = mod.k_norm(mod.k_proj(x).view(nb, lb, -1, mod.head_dim)).transpose(1, 2)
        value = mod.v_proj(x).view(nb, lb, -1, mod.head_dim)
        query, key = self.rope(query, key, cos, sin)
        key = key.transpose(1, 2)
        keys = torch.cat([self.ring_k[a_index, slots, :w], key.to(self.ring_k.dtype)], 1).transpose(1, 2)
        values = torch.cat([self.ring_v[a_index, slots, :w], value.to(self.ring_v.dtype)], 1).transpose(1, 2)
        out = F.scaled_dot_product_attention(query, keys, values, attn_mask=allowed, scale=mod.scaling,
                                             enable_gqa=query.shape[1] != keys.shape[1])
        self.ring_k[a_index, slots[:, None], ring_col] = key.to(self.ring_k.dtype)
        self.ring_v[a_index, slots[:, None], ring_col] = value.to(self.ring_v.dtype)
        out = out.transpose(1, 2).reshape(nb, lb, -1) * torch.sigmoid(gate)
        return mod.o_proj(out)


    def _static(self, nb, lb):
        key = (nb, lb)
        if key not in self.graphs:
            dev = "cuda"
            static = {"ids": torch.zeros(nb, lb, dtype=torch.long, device=dev),
                      "lens": torch.zeros(nb, dtype=torch.long, device=dev),
                      "slots": torch.full((nb,), self.max_slots, dtype=torch.long, device=dev),
                      "starts": torch.zeros(nb, dtype=torch.long, device=dev)}
            graph = None
            if self.use_graphs:

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        self._forward(**static)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=self.pool):
                    static["out"] = self._forward(**static)
                self.pool = graph.pool()
                self.stats["captures"] += 1
            self.graphs[key] = (graph, static)
        return self.graphs[key]

    @torch.inference_mode()
    def step(self, requests):
        """requests: list of (slot, token_ids), each slot at most once. Returns {role: [list of [L_i, 3]]}."""
        if len({s for s, _ in requests}) != len(requests):
            raise ValueError("a session may appear once per tick")
        largest = TOKEN_BUCKETS[-1]
        results = {role: [[] for _ in requests] for role in self.roles}
        pending = [(i, s, list(ids)) for i, (s, ids) in enumerate(requests)]
        while pending:
            batch = [(i, s, ids[:largest]) for i, s, ids in pending]
            for start in range(0, len(batch), self.session_buckets[-1]):
                part = batch[start:start + self.session_buckets[-1]]
                probs = self._run(part)
                for role_index, role in enumerate(self.roles):
                    for row, (i, _, ids) in enumerate(part):
                        results[role][i].append(probs[role_index, row, :len(ids)])
            pending = [(i, s, ids[largest:]) for i, s, ids in pending if len(ids) > largest]
        return {role: [torch.cat(chunks) for chunks in rows] for role, rows in results.items()}

    def _run(self, part):
        nb = bucket(len(part), self.session_buckets)
        lb = bucket(max(len(ids) for _, _, ids in part), TOKEN_BUCKETS)
        graph, static = self._static(nb, lb)
        ids = np.zeros((nb, lb), dtype=np.int64)
        lens = np.zeros(nb, dtype=np.int64)
        slots = np.full(nb, self.max_slots, dtype=np.int64)
        starts = np.zeros(nb, dtype=np.int64)
        for row, (_, s, chunk) in enumerate(part):
            ids[row, :len(chunk)] = chunk
            lens[row], slots[row], starts[row] = len(chunk), s, self.length[s]
        for name, value in (("ids", ids), ("lens", lens), ("slots", slots), ("starts", starts)):
            static[name].copy_(torch.from_numpy(value))
        if graph is not None:
            graph.replay()
            probs = static["out"]
            self.stats["graph_replays"] += 1
        else:
            probs = self._forward(static["ids"], static["lens"], static["slots"], static["starts"])
            self.stats["eager_steps"] += 1
        for _, s, chunk in part:
            self.length[s] += len(chunk)
        return probs.clone()
