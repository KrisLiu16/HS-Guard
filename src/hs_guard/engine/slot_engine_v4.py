# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from .ring_attention_kernel import ring_attention
from .slot_engine import SlotStreamEngine

STATE_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


class SlotStreamEngineV4(SlotStreamEngine):
    def __init__(self, model, max_slots, window=512, use_graphs=True, roles=("user", "assistant"),
                 state_dtype="float32", ring="gather", gdn_bv=8, gdn_warps=1, gdn_tiles=None, session_buckets=None):
        if session_buckets is not None:
            self.session_buckets = tuple(session_buckets)
        super().__init__(model, max_slots, window=window, use_graphs=use_graphs, roles=roles)
        if ring not in ("gather", "inplace"):
            raise ValueError("ring must be gather or inplace")
        self.ring_mode, self.gdn_bv, self.gdn_warps = ring, gdn_bv, gdn_warps
        self.default_tile, self.gdn_tiles = (gdn_bv, gdn_warps), dict(gdn_tiles or {})
        if STATE_DTYPES[state_dtype] != self.rec.dtype:
            shape = self.rec.shape
            del self.rec
            self.rec = torch.zeros(shape, dtype=STATE_DTYPES[state_dtype], device="cuda")
        self._lens = self._starts = None

    def _forward(self, ids, lens, slots, starts):
        self._lens, self._starts = lens, starts
        return super()._forward(ids, lens, slots, starts)

    def _gdn(self, mod, x, g_index, slots, lens, valid):
        self.gdn_bv, self.gdn_warps = self.gdn_tiles.get(x.shape[0], self.default_tile)
        return super()._gdn(mod, x, g_index, slots, lens, valid)

    def _attention(self, mod, x, a_index, slots, cos, sin, allowed, ring_col):
        if self.ring_mode == "gather":
            return super()._attention(mod, x, a_index, slots, cos, sin, allowed, ring_col)
        nb, lb, _ = x.shape
        query, gate = torch.chunk(mod.q_proj(x).view(nb, lb, -1, mod.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(nb, lb, -1)
        query = mod.q_norm(query).transpose(1, 2)
        key = mod.k_norm(mod.k_proj(x).view(nb, lb, -1, mod.head_dim)).transpose(1, 2)
        value = mod.v_proj(x).view(nb, lb, -1, mod.head_dim).transpose(1, 2)
        query, key = self.rope(query, key, cos, sin)
        hq, hkv, d = query.shape[1], key.shape[1], query.shape[3]
        dtype = self.ring_k.dtype
        q = query.to(dtype).reshape(nb, hkv, (hq // hkv) * lb, d).contiguous()
        key_b, value_b = key.to(dtype).contiguous(), value.to(dtype).contiguous()
        out = ring_attention(q, self.ring_k[a_index], self.ring_v[a_index], key_b, value_b, self.ring_pos,
                             slots, self._starts, self._lens, mod.scaling, self.window)
        out = out.reshape(nb, hq, lb, d)
        self.ring_k[a_index, slots[:, None], :, ring_col] = key_b.transpose(1, 2)
        self.ring_v[a_index, slots[:, None], :, ring_col] = value_b.transpose(1, 2)
        out = out.transpose(1, 2).reshape(nb, lb, -1) * torch.sigmoid(gate)
        return mod.o_proj(out)
