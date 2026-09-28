# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
import torch.nn.functional as F

from .bucketed_engine import BucketedStreamEngine
from .gdn_slot_kernel import gdn_slot_recurrent


class SlotStreamEngine(BucketedStreamEngine):
    def __init__(self, model, max_slots, window=512, use_graphs=True, roles=("user", "assistant")):
        super().__init__(model, max_slots, window=window, gdn_kernel="recurrent", use_graphs=use_graphs, roles=roles)
        n_att, slots, _, hkv, d = self.ring_k.shape
        del self.ring_k, self.ring_v
        self.ring_k = torch.zeros(n_att, slots, hkv, window + 1, d, dtype=torch.bfloat16, device="cuda")
        self.ring_v = torch.zeros_like(self.ring_k)
        self.gdn_bv, self.gdn_warps = 8, 1

    def _gdn(self, mod, x, g_index, slots, lens, valid):
        nb, lb, _ = x.shape
        qkv = mod.in_proj_qkv(x)
        z = mod.in_proj_z(x).reshape(nb, lb, mod.num_v_heads, mod.head_v_dim)
        b, a = mod.in_proj_b(x).contiguous(), mod.in_proj_a(x).contiguous()
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
        out = gdn_slot_recurrent(q, k, v, a, b, mod.A_log, mod.dt_bias, self.rec[g_index], slots, lens,
                                 bv=self.gdn_bv, num_warps=self.gdn_warps)
        out = mod.norm(out.reshape(-1, mod.head_v_dim), z.reshape(-1, mod.head_v_dim)).reshape(nb, lb, -1)
        return mod.out_proj(out)

    def _attention(self, mod, x, a_index, slots, cos, sin, allowed, ring_col):
        nb, lb, _ = x.shape
        w = self.window
        query, gate = torch.chunk(mod.q_proj(x).view(nb, lb, -1, mod.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(nb, lb, -1)
        query = mod.q_norm(query).transpose(1, 2)
        key = mod.k_norm(mod.k_proj(x).view(nb, lb, -1, mod.head_dim)).transpose(1, 2)
        value = mod.v_proj(x).view(nb, lb, -1, mod.head_dim).transpose(1, 2)
        query, key = self.rope(query, key, cos, sin)
        hq, hkv, d = query.shape[1], key.shape[1], query.shape[3]
        group = hq // hkv
        q = query.reshape(nb, hkv, group * lb, d)
        ring_k = self.ring_k[a_index, slots, :, :w]
        ring_v = self.ring_v[a_index, slots, :, :w]
        key_b, value_b = key.to(ring_k.dtype), value.to(ring_v.dtype)
        scores = torch.cat([torch.matmul(q, ring_k.transpose(-1, -2)), torch.matmul(q, key_b.transpose(-1, -2))], -1)
        scores = scores.float() * mod.scaling
        mask = allowed.expand(nb, 1, lb, w + lb).repeat(1, 1, group, 1)
        probs = torch.softmax(scores.masked_fill(~mask, float("-inf")), -1).to(ring_v.dtype)
        out = torch.matmul(probs[..., :w], ring_v) + torch.matmul(probs[..., w:], value_b)
        out = out.reshape(nb, hq, lb, d)
        self.ring_k[a_index, slots[:, None], :, ring_col] = key_b.transpose(1, 2)
        self.ring_v[a_index, slots[:, None], :, ring_col] = value_b.transpose(1, 2)
        out = out.transpose(1, 2).reshape(nb, lb, -1) * torch.sigmoid(gate)
        return mod.o_proj(out)
