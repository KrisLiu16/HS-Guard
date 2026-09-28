# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from .fused_kernels import (add_square, conv_silu_update, gated_finish, gdn_slot_v8, norm_rope, ring_attention_v8,
                           rms_finish, silu_mul, square_rows)
from .slot_engine_v7 import SlotStreamEngineV7


class SlotStreamEngineV8(SlotStreamEngineV7):
    def __init__(self, model, max_slots, **kwargs):
        super().__init__(model, max_slots, **kwargs)
        self.norm_in = [(1.0 + l.input_layernorm.weight.float(), l.input_layernorm.eps) for l in self.layers]
        self.norm_post = [(1.0 + l.post_attention_layernorm.weight.float(), l.post_attention_layernorm.eps)
                          for l in self.layers]
        self.norm_final = (1.0 + self.backbone.norm.weight.float(), self.backbone.norm.eps)
        self.conv_w = {}
        self.qk_w1 = {}
        for layer, (kind, index) in zip(self.layers, self.kind):
            if kind == "linear_attention":
                self.conv_w[index] = layer.linear_attn.conv1d.weight.detach().to(torch.bfloat16).contiguous()
            else:
                att = layer.self_attn
                self.qk_w1[index] = (1.0 + att.q_norm.weight.float(), att.q_norm.eps,
                                     1.0 + att.k_norm.weight.float(), att.k_norm.eps)
        cfg = self.backbone.config
        self.hq, self.hkv = cfg.num_attention_heads, cfg.num_key_value_heads

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
            cos, sin = cos.contiguous(), sin.contiguous()
            ring_col = torch.where(valid, pos % w, w)
            _, sq = add_square(hidden)
            normed = rms_finish(hidden, sq.mean(-1, keepdim=True), *self.norm_in[0])
            for i, (layer, (kind, index)) in enumerate(zip(self.layers, self.kind)):
                if kind == "linear_attention":
                    mixed = self._gdn8(layer.linear_attn, normed, index, slots, lens)
                else:
                    mixed = self._attention8(layer.self_attn, normed, index, slots, cos, sin, ring_col)
                hidden, sq = add_square(hidden, mixed)
                normed = rms_finish(hidden, sq.mean(-1, keepdim=True), *self.norm_post[i])
                mlp = layer.mlp
                act = silu_mul(mlp.gate_proj(normed), mlp.up_proj(normed))
                hidden, sq = add_square(hidden, mlp.down_proj(act))
                nxt = self.norm_in[i + 1] if i + 1 < len(self.layers) else self.norm_final
                normed = rms_finish(hidden, sq.mean(-1, keepdim=True), *nxt)
            probs = torch.stack([torch.softmax(self._risk(normed, role).float(), -1) for role in self.roles])
        self.ring_pos[slots[:, None], ring_col] = torch.where(valid, pos, -1)
        return probs

    def _gdn8(self, mod, x, g_index, slots, lens):
        nb, lb, _ = x.shape
        bv, warps = self.gdn_tiles.get(nb, self.default_tile)
        qkv = mod.in_proj_qkv(x)
        z = mod.in_proj_z(x)
        b, a = mod.in_proj_b(x), mod.in_proj_a(x)
        packed = conv_silu_update(qkv, self.conv[g_index], self.conv_w[g_index], slots, lens)
        o, sq = gdn_slot_v8(packed, mod.key_dim, mod.num_k_heads, mod.head_k_dim, mod.num_v_heads, mod.head_v_dim,
                            a, b, mod.A_log, mod.dt_bias, self.rec[g_index], slots, lens, bv=bv, num_warps=warps)
        v = mod.head_v_dim
        out = gated_finish(o.view(-1, v), sq.view(-1, v).mean(-1, keepdim=True), mod.norm.weight,
                           z.view(-1, v), mod.norm.variance_epsilon)
        return mod.out_proj(out.view(nb, lb, -1))

    def _attention8(self, mod, x, a_index, slots, cos, sin, ring_col):
        nb, lb, _ = x.shape
        d, hq, hkv = mod.head_dim, self.hq, self.hkv
        qw1, qeps, kw1, keps = self.qk_w1[a_index]
        qp, kp, vp = mod.q_proj(x), mod.k_proj(x), mod.v_proj(x)
        var_q = square_rows(qp, nb * lb, hq, hq * 2 * d, 2 * d, d).view(nb, lb, hq, d).mean(-1, keepdim=True)
        var_k = square_rows(kp, nb * lb, hkv, hkv * d, d, d).view(nb, lb, hkv, d).mean(-1, keepdim=True)
        q = norm_rope(qp, var_q, qw1, cos, sin, qeps, nb, lb, hq, hq // hkv, hq * 2 * d, 2 * d, d)
        k = norm_rope(kp, var_k, kw1, cos, sin, keps, nb, lb, hkv, 1, hkv * d, d, d)
        write = lb == 1
        out = ring_attention_v8(q, self.ring_k[a_index], self.ring_v[a_index], k, vp, qp, self.ring_pos, slots,
                                self._starts, self._lens, mod.scaling, self.window, write)
        if not write:
            self.ring_k[a_index, slots[:, None], :, ring_col] = k.transpose(1, 2)
            self.ring_v[a_index, slots[:, None], :, ring_col] = vp.view(nb, lb, hkv, d)
        return mod.o_proj(out)
