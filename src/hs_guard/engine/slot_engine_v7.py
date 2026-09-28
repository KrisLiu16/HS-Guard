# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np


def pack_inputs(chunks, slots, nb, lb, length, max_slots, out):
    """Pure (numpy): fill out [nb*lb + 3*nb] int64 with ids [nb, lb] (right-padded with 0), lens, slots (padded rows ->
    max_slots) and starts (length[slot], padded rows 0), the layout of the bucket's static buffer. Returns lens [n]."""
    n = len(chunks)
    lens = np.fromiter((len(c) for c in chunks), dtype=np.int64, count=n)
    ids = out[:nb * lb].reshape(nb, lb)
    ids.fill(0)
    if lb == 1:
        ids[:n, 0] = [c[0] for c in chunks]
    else:
        ids[:n][np.arange(lb)[None] < lens[:, None]] = np.concatenate([np.asarray(c, dtype=np.int64) for c in chunks])
    o = nb * lb
    out[o:o + nb] = 0
    out[o:o + n] = lens
    out[o + nb:o + 2 * nb] = max_slots
    out[o + nb:o + nb + n] = slots
    out[o + 2 * nb:o + 3 * nb] = 0
    out[o + 2 * nb:o + 2 * nb + n] = length[slots]
    return lens


try:
    import torch
    import torch.nn.functional as F
    from .bucketed_engine import TOKEN_BUCKETS, bucket
    from .slot_engine_v4 import SlotStreamEngineV4
except ImportError:
    torch = None

if torch is not None:
    class SlotStreamEngineV7(SlotStreamEngineV4):
        def __init__(self, model, max_slots, **kwargs):
            kwargs.setdefault("ring", "inplace")
            super().__init__(model, max_slots, **kwargs)
            if self.ring_mode != "inplace":
                raise ValueError("v7 keeps only the in-place ring")
            self.length = np.zeros(max_slots, dtype=np.int64)
            self.head_weights = {}
            for role in self.roles:
                head = model.heads[role]
                linear, norm = head["projection"][0], head["projection"][1]
                self.head_weights[role] = (linear.weight.detach().to(torch.bfloat16),
                                           linear.bias.detach().to(torch.bfloat16), norm,
                                           head["risk"].weight.detach().to(torch.bfloat16),
                                           head["risk"].bias.detach().to(torch.bfloat16))
            self.host = {}

        def reset(self):
            super().reset()
            self.length = np.zeros(self.max_slots, dtype=np.int64)

        def _risk(self, hidden, role):
            """Risk logits as Classifier.readout(...)[0] under autocast: Linear -> LayerNorm (fp32) -> SiLU -> Linear."""
            w1, b1, norm, w2, b2 = self.head_weights[role]
            return F.linear(F.silu(norm(F.linear(hidden, w1, b1))), w2, b2)

        def _forward(self, ids, lens, slots, starts):
            self._lens, self._starts = lens, starts
            nb, lb = ids.shape
            w = self.window
            steps = torch.arange(lb, device=ids.device)
            valid = steps[None] < lens[:, None]
            pos = starts[:, None] + steps[None]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = self.backbone.embed_tokens(ids)
                cos, sin = self.backbone.rotary_emb(hidden, pos[None].expand(3, nb, lb))
                ring_col = torch.where(valid, pos % w, w)
                for layer, (kind, index) in zip(self.layers, self.kind):
                    residual = hidden
                    normed = layer.input_layernorm(hidden)
                    if kind == "linear_attention":
                        mixed = self._gdn(layer.linear_attn, normed, index, slots, lens, valid)
                    else:
                        mixed = self._attention(layer.self_attn, normed, index, slots, cos, sin, None, ring_col)
                    hidden = residual + mixed
                    hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
                hidden = self.backbone.norm(hidden)
                probs = torch.stack([torch.softmax(self._risk(hidden, role).float(), -1) for role in self.roles])
            self.ring_pos[slots[:, None], ring_col] = torch.where(valid, pos, -1)
            return probs

        def _static(self, nb, lb):
            key = (nb, lb)
            if key not in self.graphs:
                size = nb * lb + 3 * nb
                buffer = torch.zeros(size, dtype=torch.long, device="cuda")
                o = nb * lb
                static = {"buffer": buffer, "ids": buffer[:o].view(nb, lb), "lens": buffer[o:o + nb],
                          "slots": buffer[o + nb:o + 2 * nb], "starts": buffer[o + 2 * nb:]}
                static["slots"].fill_(self.max_slots)
                inputs = {k: static[k] for k in ("ids", "lens", "slots", "starts")}
                graph = None
                if self.use_graphs:
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(2):
                            self._forward(**inputs)
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=self.pool):
                        static["out"] = self._forward(**inputs)
                    self.pool = graph.pool()
                    self.stats["captures"] += 1
                host_in = torch.zeros(size, dtype=torch.long, pin_memory=True)
                host_out = torch.zeros(len(self.roles), nb, lb, 3, dtype=torch.float32, pin_memory=True)
                self.host[key] = (host_in, host_in.numpy(), host_out, host_out.numpy())
                self.graphs[key] = (graph, static)
            return self.graphs[key]

        def _run_arrays(self, chunks, slots):
            """One tick (<= largest session bucket rows, <= 64 tokens each). Returns ([R, n, lb, 3] numpy copy, lens)."""
            n = len(chunks)
            nb = bucket(n, self.session_buckets)
            lb = bucket(max(len(c) for c in chunks), TOKEN_BUCKETS)
            graph, static = self._static(nb, lb)
            host_in, host_np, host_out, out_np = self.host[(nb, lb)]
            lens = pack_inputs(chunks, slots, nb, lb, self.length, self.max_slots, host_np)
            static["buffer"].copy_(host_in, non_blocking=True)
            if graph is not None:
                graph.replay()
                probs = static["out"]
                self.stats["graph_replays"] += 1
            else:
                probs = self._forward(static["ids"], static["lens"], static["slots"], static["starts"])
                self.stats["eager_steps"] += 1
            host_out.copy_(probs, non_blocking=True)
            torch.cuda.current_stream().synchronize()
            self.length[slots] += lens
            return out_np[:, :n].copy(), lens

        @torch.inference_mode()
        def step_packed(self, requests):
            """One tick without per-row objects: requests as step(), each <= 64 tokens and at most the largest session
            bucket of them. Returns (probs [R, n, lb, 3] numpy, lens [n]); row r holds request r's first lens[r]."""
            slots = np.fromiter((s for s, _ in requests), dtype=np.int64, count=len(requests))
            if len(np.unique(slots)) != len(slots):
                raise ValueError("a session may appear once per tick")
            return self._run_arrays([c for _, c in requests], slots)

        @torch.inference_mode()
        def step(self, requests):
            """As v6: requests (slot, token_ids), each slot at most once; returns {role: [CPU tensor [L_i, 3]]}."""
            if len({s for s, _ in requests}) != len(requests):
                raise ValueError("a session may appear once per tick")
            largest, rows = TOKEN_BUCKETS[-1], self.session_buckets[-1]
            results = {role: [[] for _ in requests] for role in self.roles}
            pending = [(i, s, list(ids)) for i, (s, ids) in enumerate(requests)]
            while pending:
                batch = [(i, s, ids[:largest]) for i, s, ids in pending]
                for start in range(0, len(batch), rows):
                    part = batch[start:start + rows]
                    probs, lens = self._run_arrays([c for _, _, c in part],
                                                   np.fromiter((s for _, s, _ in part), dtype=np.int64, count=len(part)))
                    probs = torch.from_numpy(probs)
                    for role_index, role in enumerate(self.roles):
                        for row, (i, _, _) in enumerate(part):
                            results[role][i].append(probs[role_index, row, :lens[row]])
                pending = [(i, s, ids[largest:]) for i, s, ids in pending if len(ids) > largest]
            return {role: [chunks[0] if len(chunks) == 1 else torch.cat(chunks) for chunks in rows_]
                    for role, rows_ in results.items()}
