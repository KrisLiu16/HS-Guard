# SPDX-License-Identifier: Apache-2.0
# Gated-delta-rule portions derive from FLA (MIT); see third_party_licenses/FLA-MIT.txt.
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gdn_slot_recurrent_kernel(q, k, v, a, b, A_log, dt_bias, o, state, slots, lens, scale,
                               L: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
                               V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    n_tok = tl.load(lens + i_n).to(tl.int32)
    slot = tl.load(slots + i_n).to(tl.int64)
    bos = i_n.to(tl.int64) * L
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k, mask_v = o_k < K, o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :] & (n_tok > 0)
    p_h = state + (slot * HV + i_hv) * K * V + o_k[:, None] * V + o_v[None, :]
    b_h = tl.load(p_h, mask=mask_h, other=0).to(tl.float32)
    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_a = a + bos * HV + i_hv
    p_b = b + bos * HV + i_hv
    p_o = o + (bos * HV + i_hv) * V + o_v
    b_A = tl.exp(tl.load(A_log + i_hv).to(tl.float32))
    b_dt = tl.load(dt_bias + i_hv).to(tl.float32)
    for _ in range(0, n_tok):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6) * scale
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        x = tl.load(p_a).to(tl.float32) + b_dt
        b_g = -b_A * tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
        b_beta = tl.sigmoid(tl.load(p_b).to(tl.float32))
        b_h *= tl.exp(b_g)
        b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        p_q += H * K
        p_k += H * K
        p_v += HV * V
        p_a += HV
        p_b += HV
        p_o += HV * V
    tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=mask_h)


def gdn_slot_recurrent(q, k, v, a, b, A_log, dt_bias, state, slots, lens, scale=None, bv=8, num_warps=1):
    """q, k: [N, L, H, K]; v: [N, L, HV, V]; a, b: [N, L, HV] raw projections; state: [S, HV, K, V]
    pool (fp32, bf16 or fp16; the recurrence runs in fp32) updated in place at rows slots[n]; lens: [N]
    real lengths. Returns o [N, L, HV, V] (zeros at padded positions). Slots must be unique among rows
    with lens > 0. bv / num_warps: tile width over V and warps per program (v3 used 8 and 1)."""
    n, length, h, kd = k.shape
    hv, vd = v.shape[2], v.shape[3]
    for tensor in (q, k, v, a, b, state):
        assert tensor.is_contiguous()
    assert state.dtype in (torch.float32, torch.bfloat16, torch.float16) and state.shape[1:] == (hv, kd, vd)
    scale = kd ** -0.5 if scale is None else scale
    bk, bv = triton.next_power_of_2(kd), min(bv, triton.next_power_of_2(vd))
    o = torch.zeros_like(v)
    grid = (triton.cdiv(vd, bv), n * hv)
    _gdn_slot_recurrent_kernel[grid](q, k, v, a, b, A_log, dt_bias, o, state, slots, lens, scale,
                                     L=length, H=h, HV=hv, K=kd, V=vd, BK=bk, BV=bv, num_warps=num_warps, num_stages=3)
    return o
