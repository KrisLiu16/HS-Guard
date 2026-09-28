# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch


def ring_attention_reference(q, ring_k, ring_v, new_k, new_v, ring_pos, slots, starts, lens, scale, window):
    """Torch reference with the kernel's contract. q: [N, Hkv, G*Lb, D]; ring_k/v: [S, Hkv, W+1, D];
    new_k/v: [N, Hkv, Lb, D]; ring_pos: [S, W+1]; slots/starts/lens: [N]. Returns [N, Hkv, G*Lb, D]."""
    n, hkv, m, d = q.shape
    lb = new_k.shape[2]
    rows = torch.arange(m, device=q.device)
    l_idx = rows % lb
    qpos = starts[:, None] + l_idx[None]
    q_valid = l_idx[None] < lens[:, None]
    new_pos = torch.where(torch.arange(lb, device=q.device)[None] < lens[:, None],
                          starts[:, None] + torch.arange(lb, device=q.device)[None], -1)
    kpos = torch.cat([ring_pos[slots, :window], new_pos], 1)
    ok = (kpos[:, None] >= 0) & (kpos[:, None] <= qpos[..., None]) & (kpos[:, None] > qpos[..., None] - window)
    ok = ok & q_valid[..., None]
    keys = torch.cat([ring_k[slots, :, :window], new_k], 2).float()
    values = torch.cat([ring_v[slots, :, :window], new_v], 2).float()
    scores = torch.matmul(q.float(), keys.transpose(-1, -2)) * scale
    scores = scores.masked_fill(~ok[:, None], float("-inf"))
    probs = torch.softmax(scores, -1).nan_to_num(0.0)
    return torch.matmul(probs, values).to(q.dtype)


try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

if triton is not None:
    @triton.jit
    def _ring_attention_kernel(Q, RK, RV, NK, NV, RPOS, SLOTS, STARTS, LENS, O, scale,
                               HKV: tl.constexpr, M: tl.constexpr, LB: tl.constexpr, W: tl.constexpr,
                               D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
        n, h, mb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        slot = tl.load(SLOTS + n).to(tl.int64)
        start = tl.load(STARTS + n).to(tl.int64)
        length = tl.load(LENS + n).to(tl.int64)
        rows = mb * BM + tl.arange(0, BM)
        row_ok = rows < M
        qpos = start + (rows % LB).to(tl.int64)
        q_valid = row_ok & ((rows % LB).to(tl.int64) < length)
        d = tl.arange(0, D)
        head = (n * HKV + h).to(tl.int64)
        q = tl.load(Q + (head * M + rows[:, None]) * D + d[None, :], mask=row_ok[:, None], other=0.0)
        m_i = tl.full([BM], float("-inf"), tl.float32)
        l_i = tl.zeros([BM], tl.float32)
        acc = tl.zeros([BM, D], tl.float32)
        ring = (slot * HKV + h) * (W + 1)
        n_ring = W * (length > 0).to(tl.int32)
        for j in range(0, n_ring, BN):
            cols = j + tl.arange(0, BN)
            kpos = tl.load(RPOS + slot * (W + 1) + cols)
            k = tl.load(RK + (ring + cols[:, None]) * D + d[None, :])
            s = tl.dot(q, tl.trans(k)) * scale
            ok = (kpos[None, :] >= 0) & (kpos[None, :] <= qpos[:, None]) & (kpos[None, :] > qpos[:, None] - W)
            s = tl.where(ok & q_valid[:, None], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.exp(s - m_safe[:, None])
            alpha = tl.exp(m_i - m_safe)
            v = tl.load(RV + (ring + cols[:, None]) * D + d[None, :])
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_new
        for j in range(0, LB, BN):
            cols = j + tl.arange(0, BN)
            col_ok = cols < LB
            kpos = tl.where(cols.to(tl.int64) < length, start + cols.to(tl.int64), -1)
            k = tl.load(NK + (head * LB + cols[:, None]) * D + d[None, :], mask=col_ok[:, None], other=0.0)
            s = tl.dot(q, tl.trans(k)) * scale
            ok = col_ok[None, :] & (kpos[None, :] >= 0) & (kpos[None, :] <= qpos[:, None]) & (kpos[None, :] > qpos[:, None] - W)
            s = tl.where(ok & q_valid[:, None], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.exp(s - m_safe[:, None])
            alpha = tl.exp(m_i - m_safe)
            v = tl.load(NV + (head * LB + cols[:, None]) * D + d[None, :], mask=col_ok[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_new
        out = tl.where(l_i[:, None] > 0, acc / tl.where(l_i[:, None] > 0, l_i[:, None], 1.0), 0.0)
        tl.store(O + (head * M + rows[:, None]) * D + d[None, :], out.to(O.dtype.element_ty), mask=row_ok[:, None])


def ring_attention(q, ring_k, ring_v, new_k, new_v, ring_pos, slots, starts, lens, scale, window,
                   block_m=16, block_n=32, num_warps=4, num_stages=1):
    """Same contract as ring_attention_reference; ring_k/v are one layer's pool [S, Hkv, W+1, D] (read only)."""
    n, hkv, m, d = q.shape
    lb = new_k.shape[2]
    assert window % block_n == 0 and d & (d - 1) == 0 and block_m >= 16 and block_n >= 16
    assert ring_k.shape[1:] == (hkv, window + 1, d) and new_k.shape == (n, hkv, lb, d)
    for tensor in (q, ring_k, ring_v, new_k, new_v, ring_pos):
        assert tensor.is_contiguous()
    out = torch.empty_like(q)
    grid = (n, hkv, triton.cdiv(m, block_m))
    _ring_attention_kernel[grid](q, ring_k, ring_v, new_k, new_v, ring_pos, slots, starts, lens, out, scale,
                                 HKV=hkv, M=m, LB=lb, W=window, D=d, BM=block_m, BN=block_n,
                                 num_warps=num_warps, num_stages=num_stages)
    return out
