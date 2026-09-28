# SPDX-License-Identifier: Apache-2.0
# Gated-delta-rule portions derive from FLA (MIT); see third_party_licenses/FLA-MIT.txt.
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _div_rn(a, b):
    """IEEE fp32 division without flush-to-zero (PyTorch's '/' in CUDA kernels)."""
    return tl.inline_asm_elementwise("div.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _add_rn(a, b):
    """fp32 addition that the compiler cannot fuse with a preceding multiply."""
    return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _silu(x):
    """x / (1 + exp(-x)) in fp32, as PyTorch's silu kernel."""
    return _div_rn(x, 1.0 + libdevice.exp(-x))



@triton.jit
def _add_square_kernel(res, mix, out, sq, N: tl.constexpr, HAS_RES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(res + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_RES:
        m = tl.load(mix + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        h = _add_rn(x, m).to(tl.bfloat16)
        tl.store(out + row * N + cols, h, mask=mask)
        x = h.to(tl.float32)
    tl.store(sq + row * N + cols, x * x, mask=mask)


def add_square(residual, mixed=None):
    """(residual + mixed as bf16, its fp32 squares); mixed None: squares of residual only (returns residual)."""
    n = residual.shape[-1]
    rows = residual.numel() // n
    sq = torch.empty(residual.shape, dtype=torch.float32, device=residual.device)
    out = residual if mixed is None else torch.empty_like(residual)
    _add_square_kernel[(rows,)](residual, residual if mixed is None else mixed, out, sq, N=n, HAS_RES=mixed is not None,
                                BLOCK=triton.next_power_of_2(n), num_warps=4)
    return out, sq



@triton.jit
def _rms_finish_kernel(x, var, w1, out, eps, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    h = tl.load(x + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    r = libdevice.rsqrt(tl.load(var + row) + eps)
    y = (h * r) * tl.load(w1 + cols, mask=mask, other=0.0)
    tl.store(out + row * N + cols, y.to(tl.bfloat16), mask=mask)


def rms_finish(x, var, w1, eps):
    """Qwen3_5RMSNorm given var = x.float().pow(2).mean(-1, keepdim=True) and w1 = 1.0 + weight.float()."""
    n = x.shape[-1]
    out = torch.empty_like(x)
    _rms_finish_kernel[(x.numel() // n,)](x, var, w1, out, eps, N=n, BLOCK=triton.next_power_of_2(n), num_warps=4)
    return out



@triton.jit
def _silu_mul_kernel(gate, up, out, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    g = tl.load(gate + offs, mask=mask, other=0.0).to(tl.float32)
    s = _silu(g).to(tl.bfloat16)
    u = tl.load(up + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out + offs, (s.to(tl.float32) * u).to(tl.bfloat16), mask=mask)


def silu_mul(gate, up):
    out = torch.empty_like(gate)
    n = gate.numel()
    _silu_mul_kernel[(triton.cdiv(n, 2048),)](gate, up, out, n, BLOCK=2048, num_warps=4)
    return out



@triton.jit
def _conv_silu_kernel(qkv, pool, weight, out, slots, lens, C: tl.constexpr, L: tl.constexpr, KS: tl.constexpr,
                      BLOCK: tl.constexpr):
    n = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < C
    slot = tl.load(slots + n).to(tl.int64)
    length = tl.load(lens + n)
    KEEP: tl.constexpr = KS - 1
    base = qkv + n * L * C
    for t in range(L):
        acc = tl.zeros([BLOCK], tl.float32)
        for k in tl.static_range(KS):
            j = t + k
            old = tl.load(pool + (slot * KEEP + tl.minimum(j, KEEP - 1)) * C + cols, mask=mask & (j < KEEP), other=0.0)
            new = tl.load(base + tl.maximum(j - KEEP, 0) * C + cols, mask=mask & (j >= KEEP), other=0.0)
            e = tl.where(j < KEEP, old, new).to(tl.float32)
            w = tl.load(weight + cols * KS + k, mask=mask, other=0.0).to(tl.float32)
            acc = acc + w * e
        c = acc.to(tl.bfloat16).to(tl.float32)
        s = _silu(c)
        tl.store(out + (n * L + t) * C + cols, s.to(tl.bfloat16), mask=mask)
    if length > 0:


        for i in tl.static_range(KEEP):
            j = length + i
            old = tl.load(pool + (slot * KEEP + tl.minimum(j, KEEP - 1)) * C + cols, mask=mask & (j < KEEP), other=0.0)
            new = tl.load(base + tl.maximum(j - KEEP, 0) * C + cols, mask=mask & (j >= KEEP), other=0.0)
            tl.store(pool + (slot * KEEP + i) * C + cols, tl.where(j < KEEP, old, new), mask=mask)


def conv_silu_update(qkv, pool, weight, slots, lens):
    """qkv [N, L, C] bf16; pool [S, K-1, C] bf16 (one layer, updated in place at rows slots[n] with lens[n] > 0);
    weight [C, 1, K] bf16. Returns silu(causal depthwise conv) [N, L, C] bf16, as SlotStreamEngine._gdn."""
    n, length, c = qkv.shape
    ks = weight.shape[-1]
    assert qkv.is_contiguous() and pool.is_contiguous() and weight.is_contiguous() and pool.shape[1:] == (ks - 1, c)
    out = torch.empty_like(qkv)
    block = 256
    _conv_silu_kernel[(n, triton.cdiv(c, block))](qkv, pool, weight, out, slots, lens, C=c, L=length, KS=ks, BLOCK=block,
                                                  num_warps=4)
    return out



@triton.jit
def _gdn_slot_v8_kernel(qkv, a, b, A_log, dt_bias, o, sq, state, slots, lens, scale,
                        C: tl.constexpr, KOFF: tl.constexpr, VOFF: tl.constexpr,
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
    p_q = qkv + bos * C + i_h * K + o_k
    p_k = qkv + bos * C + KOFF + i_h * K + o_k
    p_v = qkv + bos * C + VOFF + i_hv * V + o_v
    p_a = a + bos * HV + i_hv
    p_b = b + bos * HV + i_hv
    p_o = o + (bos * HV + i_hv) * V + o_v
    p_s = sq + (bos * HV + i_hv) * V + o_v
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
        b_o = tl.sum(b_h * b_q[:, None], 0).to(p_o.dtype.element_ty)
        tl.store(p_o, b_o, mask=mask_v)
        b_of = b_o.to(tl.float32)
        tl.store(p_s, b_of * b_of, mask=mask_v)
        p_q += C
        p_k += C
        p_v += C
        p_a += HV
        p_b += HV
        p_o += HV * V
        p_s += HV * V
    for _ in range(n_tok, L):
        tl.store(p_o, tl.zeros([BV], p_o.dtype.element_ty), mask=mask_v)
        tl.store(p_s, tl.zeros([BV], tl.float32), mask=mask_v)
        p_o += HV * V
        p_s += HV * V
    tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=mask_h)


def gdn_slot_v8(packed, key_dim, heads, head_k, value_heads, head_v, a, b, A_log, dt_bias, state, slots, lens,
                bv=32, num_warps=2):
    """packed [N, L, 2*key_dim + value_dim] (conv_silu_update output); a, b [N, L, HV]; state [S, HV, K, V] pool.
    Returns (o [N, L, HV, V] in packed's dtype, zeros at padded positions; o.float() ** 2 as fp32)."""
    n, length, c = packed.shape
    for tensor in (packed, a, b, state):
        assert tensor.is_contiguous()
    o = torch.empty(n, length, value_heads, head_v, dtype=packed.dtype, device=packed.device)
    sq = torch.empty(o.shape, dtype=torch.float32, device=packed.device)
    bk, bv = triton.next_power_of_2(head_k), min(bv, triton.next_power_of_2(head_v))
    grid = (triton.cdiv(head_v, bv), n * value_heads)
    _gdn_slot_v8_kernel[grid](packed, a, b, A_log, dt_bias, o, sq, state, slots, lens, head_k ** -0.5,
                              C=c, KOFF=key_dim, VOFF=2 * key_dim, L=length, H=heads, HV=value_heads, K=head_k, V=head_v,
                              BK=bk, BV=bv, num_warps=num_warps, num_stages=3)
    return o, sq



@triton.jit
def _gated_finish_kernel(x, var, w, z, out, eps, N: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, N)
    h = tl.load(x + row * N + cols).to(tl.float32)
    r = libdevice.rsqrt(tl.load(var + row) + eps)
    hb = (h * r).to(tl.bfloat16).to(tl.float32)
    t = (tl.load(w + cols).to(tl.float32) * hb).to(tl.bfloat16)
    g = tl.load(z + row * N + cols).to(tl.float32)
    s = _silu(g)
    tl.store(out + row * N + cols, (t.to(tl.float32) * s).to(tl.bfloat16))


def gated_finish(x, var, weight, z, eps):
    """Qwen3_5RMSNormGated(x, z) over the last dim given var = x.float().pow(2).mean(-1, keepdim=True)."""
    n = x.shape[-1]
    out = torch.empty_like(x)
    _gated_finish_kernel[(x.numel() // n,)](x, var, weight, z, out, eps, N=n, num_warps=1)
    return out



@triton.jit
def _square_rows_kernel(src, sq, heads, row_stride, head_stride, D: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, D)
    x = tl.load(src + (r // heads) * row_stride + (r % heads) * head_stride + cols).to(tl.float32)
    tl.store(sq + r * D + cols, x * x)


def square_rows(src, tokens, heads, row_stride, head_stride, dim):
    sq = torch.empty(tokens, heads, dim, dtype=torch.float32, device=src.device)
    _square_rows_kernel[(tokens * heads,)](src, sq, heads, row_stride, head_stride, D=dim, num_warps=2)
    return sq



@triton.jit
def _norm_rope_kernel(src, var, w1, cos, sin, out, eps, heads, lb, groups, row_stride, head_stride,
                      out_n, out_h, out_row, D: tl.constexpr, ROT: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    j = r % heads
    token = r // heads
    n, l = token // lb, token % lb
    d = tl.arange(0, D)
    half: tl.constexpr = ROT // 2
    partner = tl.where(d < half, d + half, tl.where(d < ROT, d - half, d))
    rr = libdevice.rsqrt(tl.load(var + r) + eps)
    base = src + token * row_stride + j * head_stride
    x = ((tl.load(base + d).to(tl.float32) * rr) * tl.load(w1 + d)).to(tl.bfloat16).to(tl.float32)
    xp = ((tl.load(base + partner).to(tl.float32) * rr) * tl.load(w1 + partner)).to(tl.bfloat16).to(tl.float32)
    rot_d = tl.minimum(d, ROT - 1)
    c = tl.load(cos + token * ROT + rot_d).to(tl.float32)
    s = tl.load(sin + token * ROT + tl.where(d < ROT, rot_d, 0)).to(tl.float32)
    rh = tl.where(d < half, -xp, xp)
    first = (x * c).to(tl.bfloat16).to(tl.float32)
    second = (rh * s).to(tl.bfloat16).to(tl.float32)
    y = tl.where(d < ROT, _add_rn(first, second).to(tl.bfloat16), x.to(tl.bfloat16))
    h, g = j // groups, j % groups
    tl.store(out + n * out_n + h * out_h + (g * lb + l) * out_row + d, y)


def norm_rope(src, var, w1, cos, sin, eps, n, lb, heads, groups, row_stride, head_stride, dim):
    """Qwen3_5RMSNorm per head (given var) then apply_rotary_pos_emb, written as [N, heads // groups, groups * lb, dim]:
    the grouped query layout of the ring kernel (groups = 1 gives the key layout [N, Hkv, lb, dim])."""
    rot = cos.shape[-1]
    out = torch.empty(n, heads // groups, groups * lb, dim, dtype=torch.bfloat16, device=src.device)
    _norm_rope_kernel[(n * lb * heads,)](src, var, w1, cos, sin, out, eps, heads, lb, groups, row_stride, head_stride,
                                         out.stride(0), out.stride(1), out.stride(2), D=dim, ROT=rot, num_warps=2)
    return out



@triton.jit
def _ring_v8_kernel(Q, RK, RV, NK, NV, RPOS, SLOTS, STARTS, LENS, GATE, O, scale,
                    nv_n, nv_l, gate_n, gate_l, o_n, o_l,
                    HKV: tl.constexpr, G: tl.constexpr, M: tl.constexpr, LB: tl.constexpr, W: tl.constexpr,
                    D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, WRITE_RING: tl.constexpr):
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
    nv_base = NV + n.to(tl.int64) * nv_n + h * D
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
        v = tl.load(nv_base + cols[:, None] * nv_l + d[None, :], mask=col_ok[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new
    out = tl.where(l_i[:, None] > 0, acc / tl.where(l_i[:, None] > 0, l_i[:, None], 1.0), 0.0)
    out = out.to(tl.bfloat16).to(tl.float32)
    g_idx, l_idx = rows // LB, rows % LB
    col = (h * G + g_idx)[:, None] * D + d[None, :]
    gate = tl.load(GATE + n.to(tl.int64) * gate_n + l_idx[:, None] * gate_l + (h * G + g_idx)[:, None] * (2 * D) + D + d[None, :],
                   mask=row_ok[:, None], other=0.0).to(tl.float32)
    sig = _div_rn(tl.zeros_like(gate) + 1.0, 1.0 + libdevice.exp(-gate)).to(tl.bfloat16).to(tl.float32)
    tl.store(O + n.to(tl.int64) * o_n + l_idx[:, None] * o_l + col, (out * sig).to(tl.bfloat16), mask=row_ok[:, None])
    if WRITE_RING:
        if length > 0:
            wcol = start % W
            k_new = tl.load(NK + head * D + d)
            v_new = tl.load(nv_base + d)
            tl.store(RK + (ring + wcol) * D + d, k_new)
            tl.store(RV + (ring + wcol) * D + d, v_new)


def ring_attention_v8(q, ring_k, ring_v, new_k, value, gate, ring_pos, slots, starts, lens, scale, window, write_ring,
                      block_m=16, block_n=32, num_warps=4, num_stages=1):
    """q [N, Hkv, G*Lb, D] and new_k [N, Hkv, Lb, D] (norm_rope layouts); value [N, Lb, Hkv*D] (v_proj output);
    gate: the q_proj output [N, Lb, Hq*2D] (gate = second half of each head). Returns out * sigmoid(gate) as
    [N, Lb, Hq*D] bf16 (the o_proj input). write_ring (Lb == 1 only): also writes this tick's key/value to the ring."""
    n, hkv, m, d = q.shape
    lb = new_k.shape[2]
    groups = m // lb
    assert window % block_n == 0 and d & (d - 1) == 0 and block_m >= 16 and block_n >= 16
    assert not write_ring or (lb == 1 and m <= block_m)
    for tensor in (q, ring_k, ring_v, new_k, value, gate, ring_pos):
        assert tensor.is_contiguous()
    out = torch.empty(n, lb, hkv * groups * d, dtype=torch.bfloat16, device=q.device)
    grid = (n, hkv, triton.cdiv(m, block_m))
    _ring_v8_kernel[grid](q, ring_k, ring_v, new_k, value, ring_pos, slots, starts, lens, gate, out, scale,
                          value.stride(0), value.stride(1), gate.stride(0), gate.stride(1), out.stride(0), out.stride(1),
                          HKV=hkv, G=groups, M=m, LB=lb, W=window, D=d, BM=block_m, BN=block_n, WRITE_RING=write_ring,
                          num_warps=num_warps, num_stages=num_stages)
    return out
