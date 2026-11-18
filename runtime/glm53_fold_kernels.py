#!/usr/bin/env python3
"""
glm53_fold_kernels.py  [Round 3.2, Claude]
==========================================
Fused CUDA kernels compiled at runtime with NVRTC through `torch.cuda._compile_kernel` (no Triton, no nvcc, no MSVC
needed on Windows). Every kernel has a pure-PyTorch reference used by the unit tests and as the fallback path.

Kernels (all decode-oriented, N <= 4 tokens; launches are CUDA-graph capturable):
  * mhc_prepare : RMS norm -> 24-way mix -> sigmoid/softmax -> Sinkhorn (hc_sinkhorn_iters) -> stream collapse,
                  one launch per token block (replaces ~130 tiny kernels per mHC site).
  * mhc_merge   : out = post (x) y + comb^T @ residual, one launch.
  * fp8_gemv    : y = x @ (W_fp8 * block_scale_128x128)^T with the dequant done in registers (W read once, 1 B/weight;
                  replaces "dequantize the whole matrix to BF16 on every call").
  * rvq_gemv    : 2x8 additive-codebook LUT-GEMV straight from uint8 codes stored in an expert slot pool
                  (see glm53_fold_expert_pool.py); codebooks + rotated activations live in shared memory.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Optional, Tuple

import torch

_CACHE: Dict[str, object] = {}


def _compile(name: str, src: str, shared: int = 0):
    key = name
    if key not in _CACHE:
        k = torch.cuda._compile_kernel(src, name)
        if shared >= 48 * 1024:
            k.set_shared_memory_config(shared)
        _CACHE[key] = k
    return _CACHE[key]


def precompile(fp8_rows: bool = False) -> float:
    """[R3.4-2h] JIT the NVRTC kernels of the TCQ runtime now (start-up) instead of inside the first request: decode
    kernels (the graph capture's warm-up compiles them too) and the prefill-only tcq_dense. Same names / shared-memory
    configs as the call sites (the cache is keyed by name). Returns seconds.
    [R3.4 review] fp8_rows=True also compiles fp8_gemv_rows (a trimmed FP8 output head, E2; off = unchanged default)."""
    import time as _t
    t0 = _t.perf_counter()
    _compile("mhc_prepare_kernel", _MHC_PREPARE_SRC)
    _compile("mhc_merge_kernel", _MHC_MERGE_SRC)
    _compile("fp8_gemv_kernel", _FP8_GEMV_SRC)
    _compile("tcq_gemv_v2_kernel", _TCQ_GEMV_V2_SRC, shared=64 * 1024)
    _compile("tcq_sparse_add_kernel", _TCQ_SPARSE_SRC)
    _compile("tcq_dense_kernel", _TCQ_DENSE_SRC, shared=3 * 4096 * 4)
    if fp8_rows:
        _compile("fp8_gemv_rows_kernel", _FP8_GEMV_ROWS_SRC)
    return _t.perf_counter() - t0


def _e4m3_table(device: torch.device) -> torch.Tensor:
    """float32[256] decode table for float8_e4m3fn bytes (bit-exact with torch's conversion)."""
    key = f"e4m3_{device}"
    if key not in _CACHE:
        b = torch.arange(256, dtype=torch.uint8)
        _CACHE[key] = b.view(torch.float8_e4m3fn).to(torch.float32).nan_to_num(0.0).to(device)
    return _CACHE[key]


# ----------------------------------------------------------------------------------------------------------------
# mHC prepare / merge
# ----------------------------------------------------------------------------------------------------------------
_MHC_PREPARE_SRC = r"""
extern "C" __global__ void mhc_prepare_kernel(
    const float* __restrict__ h, const float* __restrict__ fn, const float* __restrict__ base,
    const float* __restrict__ scale, float* __restrict__ post_out, float* __restrict__ comb_out,
    float* __restrict__ xin_out, int D, double eps_d, double hc_eps_d, int iters)
{
    const int HC = 4;
    const int NMIX = 24;
    const int n = HC * D;
    const int t = blockIdx.x;
    const float* ht = h + (size_t)t * n;
    const float eps = (float)eps_d;
    const float hc_eps = (float)hc_eps_d;

    __shared__ float s_part[32][NMIX + 1];
    __shared__ float s_mix[NMIX];
    __shared__ float s_pre[HC];
    __shared__ float s_rms;

    float acc[NMIX];
    #pragma unroll
    for (int j = 0; j < NMIX; ++j) acc[j] = 0.f;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = ht[i];
        ss += v * v;
        #pragma unroll
        for (int j = 0; j < NMIX; ++j) acc[j] += fn[(size_t)j * n + i] * v;
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarp = (blockDim.x + 31) >> 5;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        ss += __shfl_down_sync(0xffffffff, ss, off);
        #pragma unroll
        for (int j = 0; j < NMIX; ++j) acc[j] += __shfl_down_sync(0xffffffff, acc[j], off);
    }
    if (lane == 0) {
        s_part[warp][NMIX] = ss;
        #pragma unroll
        for (int j = 0; j < NMIX; ++j) s_part[warp][j] = acc[j];
    }
    __syncthreads();
    if (threadIdx.x <= NMIX) {
        float tot = 0.f;
        for (int w = 0; w < nwarp; ++w) tot += s_part[w][threadIdx.x];
        if (threadIdx.x == NMIX) s_rms = rsqrtf(tot / (float)n + eps);
        else s_mix[threadIdx.x] = tot;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const float r = s_rms;
        float pre[HC], post[HC], c[HC][HC];
        for (int k = 0; k < HC; ++k) {
            float pw = s_mix[k] * r * scale[0] + base[k];
            pre[k] = 1.f / (1.f + expf(-pw)) + hc_eps;
            float qw = s_mix[HC + k] * r * scale[1] + base[HC + k];
            post[k] = 2.f / (1.f + expf(-qw));
        }
        for (int a = 0; a < HC; ++a) {
            float mx = -1e30f;
            for (int b = 0; b < HC; ++b) {
                c[a][b] = s_mix[2 * HC + a * HC + b] * r * scale[2] + base[2 * HC + a * HC + b];
                mx = fmaxf(mx, c[a][b]);
            }
            float se = 0.f;
            for (int b = 0; b < HC; ++b) { c[a][b] = expf(c[a][b] - mx); se += c[a][b]; }
            for (int b = 0; b < HC; ++b) c[a][b] = c[a][b] / se + hc_eps;
        }
        // column normalisation (sum over rows), then (iters-1) x [row, column]
        for (int b = 0; b < HC; ++b) {
            float cs = 0.f; for (int a = 0; a < HC; ++a) cs += c[a][b];
            for (int a = 0; a < HC; ++a) c[a][b] = c[a][b] / (cs + hc_eps);
        }
        for (int it = 0; it < iters - 1; ++it) {
            for (int a = 0; a < HC; ++a) {
                float rs = 0.f; for (int b = 0; b < HC; ++b) rs += c[a][b];
                for (int b = 0; b < HC; ++b) c[a][b] = c[a][b] / (rs + hc_eps);
            }
            for (int b = 0; b < HC; ++b) {
                float cs = 0.f; for (int a = 0; a < HC; ++a) cs += c[a][b];
                for (int a = 0; a < HC; ++a) c[a][b] = c[a][b] / (cs + hc_eps);
            }
        }
        for (int k = 0; k < HC; ++k) { s_pre[k] = pre[k]; post_out[t * HC + k] = post[k]; }
        for (int a = 0; a < HC; ++a) for (int b = 0; b < HC; ++b) comb_out[(t * HC + a) * HC + b] = c[a][b];
    }
    __syncthreads();
    const float p0 = s_pre[0], p1 = s_pre[1], p2 = s_pre[2], p3 = s_pre[3];
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        xin_out[(size_t)t * D + d] = p0 * ht[d] + p1 * ht[D + d] + p2 * ht[2 * D + d] + p3 * ht[3 * D + d];
    }
}
"""

_MHC_MERGE_SRC = r"""
extern "C" __global__ void mhc_merge_kernel(
    const float* __restrict__ post, const float* __restrict__ comb, const float* __restrict__ y,
    const float* __restrict__ res, float* __restrict__ out, int D, int T)
{
    const int HC = 4;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)T * D;
    if (idx >= total) return;
    int t = (int)(idx / D), d = (int)(idx % D);
    const float* cb = comb + t * HC * HC;
    const float* rt = res + (size_t)t * HC * D;
    float r0 = rt[d], r1 = rt[D + d], r2 = rt[2 * D + d], r3 = rt[3 * D + d];
    float yv = y[(size_t)t * D + d];
    for (int s = 0; s < HC; ++s) {
        float v = post[t * HC + s] * yv + cb[0 * HC + s] * r0 + cb[1 * HC + s] * r1 + cb[2 * HC + s] * r2 + cb[3 * HC + s] * r3;
        out[((size_t)t * HC + s) * D + d] = v;
    }
}
"""


def mhc_prepare_ref(h, fn, base, scale, eps, hc_eps, iters):
    """Exact copy of RealGLM53Backbone.mhc_prepare (HF Glm5NextTextHyperConnection) for testing."""
    hc = h.shape[-2]
    flat = h.flatten(-2).float()
    flat = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + eps)
    mix = torch.nn.functional.linear(flat, fn.float())
    pre_w, post_w, comb_w = mix.split([hc, hc, hc * hc], dim=-1)
    pre_b, post_b, comb_b = base.float().view(-1).split([hc, hc, hc * hc])
    sc = scale.float().view(-1)
    pre = torch.sigmoid(pre_w * sc[0] + pre_b) + hc_eps
    post = 2.0 * torch.sigmoid(post_w * sc[1] + post_b)
    comb = torch.softmax(comb_w.view(*comb_w.shape[:-1], hc, hc) * sc[2] + comb_b.view(hc, hc), dim=-1) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(int(iters) - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    x_in = (pre.unsqueeze(-1) * h.float()).sum(dim=-2)
    return post, comb, x_in


def mhc_prepare(h: torch.Tensor, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor,
                eps: float, hc_eps: float, iters: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """h [..., HC=4, D] fp32 (contiguous) -> post [..., 4], comb [..., 4, 4], x_in [..., D]."""
    if os.environ.get('R36_FUSE_MHC', '0') == '1':
        from glm53_fold_fused_r36 import mhc_prepare as fused
        return fused(h, fn, base, scale, eps, hc_eps, iters)
    lead = h.shape[:-2]
    D = h.shape[-1]
    hf = h.reshape(-1, 4, D).float().contiguous()
    T = hf.shape[0]
    post = torch.empty((T, 4), dtype=torch.float32, device=h.device)
    comb = torch.empty((T, 4, 4), dtype=torch.float32, device=h.device)
    xin = torch.empty((T, D), dtype=torch.float32, device=h.device)
    k = _compile("mhc_prepare_kernel", _MHC_PREPARE_SRC)
    k(grid=(T, 1, 1), block=(512, 1, 1),
      args=[hf, fn.float().contiguous(), base.float().contiguous(), scale.float().contiguous(), post, comb, xin,
            int(D), float(eps), float(hc_eps), int(iters)])
    return post.view(*lead, 4), comb.view(*lead, 4, 4), xin.view(*lead, D)


def mhc_merge(post: torch.Tensor, comb: torch.Tensor, y: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """post [..., 4], comb [..., 4, 4], y [..., D], residual [..., 4, D] -> [..., 4, D] (fp32)."""
    lead = residual.shape[:-2]
    D = residual.shape[-1]
    T = int(math.prod(lead)) if len(lead) else 1
    out = torch.empty((T, 4, D), dtype=torch.float32, device=residual.device)
    k = _compile("mhc_merge_kernel", _MHC_MERGE_SRC)
    n = T * D
    k(grid=((n + 255) // 256, 1, 1), block=(256, 1, 1),
      args=[post.float().contiguous(), comb.float().contiguous(), y.float().contiguous(),
            residual.float().contiguous(), out, int(D), int(T)])
    return out.view(*lead, 4, D)


# ----------------------------------------------------------------------------------------------------------------
# FP8 (e4m3, 128x128 block scales) weight x FP32 activation GEMV, N <= 4 tokens
# ----------------------------------------------------------------------------------------------------------------
_FP8_GEMV_SRC = r"""
extern "C" __global__ void fp8_gemv_kernel(
    const unsigned char* __restrict__ W, const float* __restrict__ sinv, const float* __restrict__ lut_g,
    const float* __restrict__ x, float* __restrict__ y, int O, int I, int N, int sc_cols, int bs)
{
    __shared__ float lut[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) lut[i] = lut_g[i];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int warps = blockDim.x >> 5;
    const int o = blockIdx.x * warps + (threadIdx.x >> 5);
    if (o >= O) return;
    const unsigned char* row = W + (size_t)o * I;
    const float* srow = sinv + (size_t)(o / bs) * sc_cols;
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    // each lane handles 16 consecutive bytes per step: a 16-byte chunk never crosses a 128-wide scale block
    for (int c = lane * 16; c < I; c += 32 * 16) {
        uint4 q = *reinterpret_cast<const uint4*>(row + c);
        const unsigned char* qb = reinterpret_cast<const unsigned char*>(&q);
        float s = srow[c / bs];
        float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
        #pragma unroll
        for (int u = 0; u < 16; ++u) {
            float w = lut[qb[u]];
            p0 += w * x[c + u];
            if (N > 1) p1 += w * x[I + c + u];
            if (N > 2) p2 += w * x[2 * I + c + u];
            if (N > 3) p3 += w * x[3 * I + c + u];
        }
        acc0 += p0 * s; acc1 += p1 * s; acc2 += p2 * s; acc3 += p3 * s;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc0 += __shfl_down_sync(0xffffffff, acc0, off);
        acc1 += __shfl_down_sync(0xffffffff, acc1, off);
        acc2 += __shfl_down_sync(0xffffffff, acc2, off);
        acc3 += __shfl_down_sync(0xffffffff, acc3, off);
    }
    if (lane == 0) {
        y[o] = acc0;
        if (N > 1) y[O + o] = acc1;
        if (N > 2) y[2 * O + o] = acc2;
        if (N > 3) y[3 * O + o] = acc3;
    }
}
"""


def fp8_gemv(qweight: torch.Tensor, scale_inv: torch.Tensor, x: torch.Tensor, block: int = 128) -> torch.Tensor:
    """qweight float8_e4m3fn [O, I], scale_inv fp32 [O/bs, I/bs], x [N<=4, I] -> fp32 [N, O]."""
    if os.environ.get('R36_FP8_GEMV', 'v1') == 'v2':
        from glm53_fold_fused_r36 import fp8_gemv as fused
        return fused(qweight, scale_inv, x, block)
    O, I = qweight.shape
    N = x.shape[0]
    assert N <= 4 and I % 16 == 0 and I % block == 0 and O % block == 0
    xf = x.float().contiguous()
    y = torch.empty((N, O), dtype=torch.float32, device=x.device)
    k = _compile("fp8_gemv_kernel", _FP8_GEMV_SRC)
    warps = 8
    k(grid=((O + warps - 1) // warps, 1, 1), block=(32 * warps, 1, 1),
      args=[qweight.view(torch.uint8), scale_inv.float().contiguous(), _e4m3_table(x.device), xf, y,
            int(O), int(I), int(N), int(scale_inv.shape[1]), int(block)])
    return y


def fp8_gemv_ref(qweight: torch.Tensor, scale_inv: torch.Tensor, x: torch.Tensor, block: int = 128) -> torch.Tensor:
    O, I = qweight.shape
    so, si = scale_inv.shape
    w = qweight.to(torch.float32).view(so, block, si, block) * scale_inv.float()[:, None, :, None]
    return x.float() @ w.view(O, I).T


# [R3.4-V] row-sliced FP8 head (vocabulary trim): one scale row per OUTPUT row. Generated from _FP8_GEMV_SRC by changing
# only the kernel name and the scale-row index, so each output row's arithmetic (lane split, FMA order, warp reduction)
# is the same as fp8_gemv's -> a trimmed head's logits equal the full head's logits of the kept rows bit for bit.
_FP8_SROW_OLD = "const float* srow = sinv + (size_t)(o / bs) * sc_cols;"
_FP8_SROW_NEW = "const float* srow = sinv + (size_t)o * sc_cols;"
assert _FP8_GEMV_SRC.count(_FP8_SROW_OLD) == 1 and _FP8_GEMV_SRC.count("fp8_gemv_kernel(") == 1
_FP8_GEMV_ROWS_SRC = _FP8_GEMV_SRC.replace("fp8_gemv_kernel(", "fp8_gemv_rows_kernel(").replace(_FP8_SROW_OLD, _FP8_SROW_NEW)


def fp8_gemv_rows(qweight: torch.Tensor, scale_inv: torch.Tensor, x: torch.Tensor, block: int = 128) -> torch.Tensor:
    """qweight float8_e4m3fn [O, I], scale_inv fp32 [O, I/bs] (per-row block scales), x [N<=4, I] -> fp32 [N, O]."""
    if os.environ.get('R36_FP8_GEMV', 'v1') == 'v2':
        from glm53_fold_fused_r36 import fp8_gemv as fused
        return fused(qweight, scale_inv, x, block, row_scales=True)
    O, I = qweight.shape
    N = x.shape[0]
    assert N <= 4 and I % 16 == 0 and I % block == 0 and scale_inv.shape[0] == O and scale_inv.shape[1] == I // block
    xf = x.float().contiguous()
    y = torch.empty((N, O), dtype=torch.float32, device=x.device)
    k = _compile("fp8_gemv_rows_kernel", _FP8_GEMV_ROWS_SRC)
    warps = 8
    k(grid=((O + warps - 1) // warps, 1, 1), block=(32 * warps, 1, 1),
      args=[qweight.view(torch.uint8), scale_inv.float().contiguous(), _e4m3_table(x.device), xf, y,
            int(O), int(I), int(N), int(scale_inv.shape[1]), int(block)])
    return y


def fp8_gemv_rows_ref(qweight: torch.Tensor, scale_inv: torch.Tensor, x: torch.Tensor, block: int = 128) -> torch.Tensor:
    O, I = qweight.shape
    w = qweight.to(torch.float32).view(O, I // block, block) * scale_inv.float()[:, :, None]
    return x.float() @ w.view(O, I).T


# ----------------------------------------------------------------------------------------------------------------
# 2x8 (up to 3-stage) additive-codebook LUT-GEMV over slab rows (expert slot pool or a VRAM-resident layer slab)
# ----------------------------------------------------------------------------------------------------------------
_RVQ_GEMV_SRC = r"""
#include <cuda_fp16.h>
extern "C" __global__ void rvq_gemv_kernel(
    const unsigned char* __restrict__ rows, long long row_bytes, const int* __restrict__ slots,
    int off_s, int off_i0, int off_i1, int off_i2,
    const float* __restrict__ cb_g,     // [3, 256, 8] fp32 (stages 0..2)
    const float* __restrict__ x_g,      // [x_count, d_in] fp32, rotated + permuted activations
    int x_per_expert, float* __restrict__ y,  // [K, d_out]
    int d_out, int d_in, int f0, int f1, int f2, int G, int fpg)
{
    extern __shared__ float smem[];
    float* cb = smem;                    // 3 * 256 * 8
    float* xs = smem + 3 * 256 * 8;      // d_in
    const int e = blockIdx.y;
    const float* xg = x_g + (size_t)(x_per_expert ? e : 0) * d_in;
    for (int i = threadIdx.x; i < 3 * 256 * 8; i += blockDim.x) cb[i] = cb_g[i];
    for (int i = threadIdx.x; i < d_in; i += blockDim.x) xs[i] = xg[i];
    __syncthreads();
    const unsigned char* base = rows + (long long)slots[e] * row_bytes;
    const __half* S = reinterpret_cast<const __half*>(base + off_s);
    const unsigned char* I0 = base + off_i0;
    const unsigned char* I1 = base + off_i1;
    const unsigned char* I2 = base + off_i2;
    const int lane = threadIdx.x & 31;
    const int warps = blockDim.x >> 5;
    for (int r = blockIdx.x * warps + (threadIdx.x >> 5); r < d_out; r += gridDim.x * warps) {
        float acc = 0.f;
        const __half* Sr = S + (size_t)r * G;
        for (int j = lane; j < f0; j += 32) {
            const float4* xv = reinterpret_cast<const float4*>(xs + 8 * j);
            float4 xa = xv[0], xb = xv[1];
            const float4* c0 = reinterpret_cast<const float4*>(cb + 8 * (int)I0[(size_t)r * f0 + j]);
            float4 a = c0[0], b = c0[1];
            if (j < f1) {
                const float4* c1 = reinterpret_cast<const float4*>(cb + 2048 + 8 * (int)I1[(size_t)r * f1 + j]);
                float4 a1 = c1[0], b1 = c1[1];
                a.x += a1.x; a.y += a1.y; a.z += a1.z; a.w += a1.w; b.x += b1.x; b.y += b1.y; b.z += b1.z; b.w += b1.w;
            }
            if (j < f2) {
                const float4* c2 = reinterpret_cast<const float4*>(cb + 4096 + 8 * (int)I2[(size_t)r * f2 + j]);
                float4 a2 = c2[0], b2 = c2[1];
                a.x += a2.x; a.y += a2.y; a.z += a2.z; a.w += a2.w; b.x += b2.x; b.y += b2.y; b.z += b2.z; b.w += b2.w;
            }
            float p = a.x * xa.x + a.y * xa.y + a.z * xa.z + a.w * xa.w + b.x * xb.x + b.y * xb.y + b.z * xb.z + b.w * xb.w;
            acc += p * __half2float(Sr[j / fpg]);
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
        if (lane == 0) y[(size_t)e * d_out + r] = acc;
    }
}
"""


class RvqProjLayout:
    """Byte offsets of one projection's fields inside a slab row + its tier geometry (from preset.slab_layout())."""

    def __init__(self, preset, prefix: str, hdr):
        layout, row_bytes = preset.slab_layout()
        self.row_bytes = int(row_bytes)
        self.off_s = int(layout[f"{prefix}_S_blk"][0])
        self.off_i0 = int(layout[f"{prefix}_idx_s0"][0])
        self.off_i1 = int(layout[f"{prefix}_idx_s1"][0]) if f"{prefix}_idx_s1" in layout else 0
        self.off_i2 = int(layout[f"{prefix}_idx_s2"][0]) if f"{prefix}_idx_s2" in layout else 0
        self.d_out = int(hdr.d_out)
        self.d_in = int(hdr.c2)
        self.f0 = int(hdr.c2) // 8
        self.f1 = int(hdr.c1) // 8 if f"{prefix}_idx_s1" in layout else 0
        self.f2 = int(hdr.c0) // 8 if f"{prefix}_idx_s2" in layout else 0
        self.G = int(layout[f"{prefix}_S_blk"][2][-1])
        self.fpg = self.f0 // self.G
        assert getattr(hdr, "r_shared", 0) == 0, "rvq_gemv: shared subspace (r_shared > 0) not supported"
        assert hdr.stages_t0 <= 3, "rvq_gemv: stage 3 not supported"
        self.cb = hdr.codebooks[:3].float().contiguous()  # [3, 256, 8]


def rvq_gemv(rows: torch.Tensor, slots: torch.Tensor, lay: RvqProjLayout, x_rot: torch.Tensor,
             x_per_expert: bool, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """rows uint8 [S, row_bytes] (CUDA), slots int32 [K], x_rot fp32 [1 or K, d_in] -> fp32 [K, d_out]."""
    K = int(slots.numel())
    if out is None:
        out = torch.empty((K, lay.d_out), dtype=torch.float32, device=rows.device)
    shared = (3 * 256 * 8 + lay.d_in) * 4
    k = _compile("rvq_gemv_kernel", _RVQ_GEMV_SRC, shared=max(shared, 64 * 1024))
    cb = lay.cb if lay.cb.device == rows.device else lay.cb.to(rows.device)
    lay.cb = cb
    warps = 8
    gx = min((lay.d_out + warps - 1) // warps, 128)
    k(grid=(gx, K, 1), block=(32 * warps, 1, 1), shared_mem=shared,
      args=[rows, int(lay.row_bytes), slots.to(torch.int32), lay.off_s, lay.off_i0, lay.off_i1, lay.off_i2,
            cb, x_rot.float().contiguous(), 1 if x_per_expert else 0, out,
            lay.d_out, lay.d_in, lay.f0, lay.f1, lay.f2, lay.G, lay.fpg])
    return out


# ----------------------------------------------------------------------------------------------------------------
# TCQ (bitshift trellis, L=12) LUT-GEMV over slab rows: weight(r, t) = table_k[ bits[k t : k t + 12) ] * scale[r]
# ----------------------------------------------------------------------------------------------------------------
_TCQ_GEMV_SRC = r"""
#include <cuda_fp16.h>
extern "C" __global__ void tcq_gemv_kernel(
    const unsigned char* __restrict__ rows, long long row_bytes, const int* __restrict__ slots,
    int off_map, int off_scale, int off_c1, int off_c2, int off_c3,
    int n1, int n2, int n3, int rb1, int rb2, int rb3,
    const float* __restrict__ tab_g,    // [3, 4096] (k = 1, 2, 3)
    const float* __restrict__ x_g, int x_per_pair, float* __restrict__ y, int d_out, int T)
{
    extern __shared__ float smem[];
    float* tab = smem;              // 3 * 4096
    float* xs = smem + 3 * 4096;    // T
    const int pr = blockIdx.y;
    const float* xg = x_g + (size_t)(x_per_pair ? pr : 0) * T;
    for (int i = threadIdx.x; i < 3 * 4096; i += blockDim.x) tab[i] = tab_g[i];
    for (int i = threadIdx.x; i < T; i += blockDim.x) xs[i] = xg[i];
    __syncthreads();
    const unsigned char* base = rows + (long long)slots[pr] * row_bytes;
    const unsigned short* rmap = reinterpret_cast<const unsigned short*>(base + off_map);
    const __half* scale = reinterpret_cast<const __half*>(base + off_scale);
    const int lane = threadIdx.x & 31;
    const int warps = blockDim.x >> 5;
    for (int r = blockIdx.x * warps + (threadIdx.x >> 5); r < d_out; r += gridDim.x * warps) {
        int k; const unsigned char* p;
        if (r < n1) { k = 1; p = base + off_c1 + (size_t)r * rb1; }
        else if (r < n1 + n2) { k = 2; p = base + off_c2 + (size_t)(r - n1) * rb2; }
        else { k = 3; p = base + off_c3 + (size_t)(r - n1 - n2) * rb3; }
        const float* tk = tab + (k - 1) * 4096;
        float acc = 0.f;
        for (int t = lane; t < T; t += 32) {
            int bit = k * t;
            int byte = bit >> 3, off = bit & 7;
            unsigned int w24 = ((unsigned int)p[byte] << 16) | ((unsigned int)p[byte + 1] << 8) | (unsigned int)p[byte + 2];
            int idx = (w24 >> (12 - off)) & 4095;
            acc += tk[idx] * xs[t];
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
        if (lane == 0) y[(size_t)pr * d_out + rmap[r]] = acc * __half2float(scale[r]);
    }
}
"""


class TcqProjLayout:
    """Byte offsets of one projection inside a TCQ slab row (glm53_fold_tcq_format.TcqPreset.layout())."""

    def __init__(self, preset, prefix: str, spec, tables: torch.Tensor):
        lay, row_bytes = preset.layout()
        self.row_bytes = int(row_bytes)
        self.off = {f: int(lay[f"{prefix}_{f}"][0]) for f in ("map", "scale", "c1", "c2", "c3", "sp")}
        self.n1, self.n2, self.n3 = int(spec.n1), int(spec.n2), int(spec.n3)
        b1, b2, b3 = spec.code_bytes()
        self.rb = (int(b1), int(b2), int(b3))
        self.d_out, self.T = int(spec.d_out), int(spec.d_in)
        self.k_sparse = int(spec.k_sparse)
        self.tab = tables.float().contiguous()  # [3, 4096]


def tcq_gemv(rows: torch.Tensor, slots: torch.Tensor, lay: TcqProjLayout, x_rot: torch.Tensor, x_per_pair: bool,
             out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """rows uint8 [S, row_bytes] CUDA; slots int32 [P]; x_rot fp32 [1 or P, T] -> fp32 [P, d_out] (rotated-domain part)."""
    P = int(slots.numel())
    if out is None:
        out = torch.empty((P, lay.d_out), dtype=torch.float32, device=rows.device)
    shared = (3 * 4096 + lay.T) * 4
    k = _compile("tcq_gemv_kernel", _TCQ_GEMV_SRC, shared=max(shared, 64 * 1024))
    if lay.tab.device != rows.device:
        lay.tab = lay.tab.to(rows.device)
    warps = 8
    gx = min((lay.d_out + warps - 1) // warps, 128)
    k(grid=(gx, P, 1), block=(32 * warps, 1, 1), shared_mem=shared,
      args=[rows, int(lay.row_bytes), slots.to(torch.int32), lay.off["map"], lay.off["scale"], lay.off["c1"],
            lay.off["c2"], lay.off["c3"], lay.n1, lay.n2, lay.n3, lay.rb[0], lay.rb[1], lay.rb[2], lay.tab,
            x_rot.float().contiguous(), 1 if x_per_pair else 0, out, lay.d_out, lay.T])
    return out


def tcq_sparse_add_ref(rows: torch.Tensor, slots: torch.Tensor, lay: TcqProjLayout, x_orig: torch.Tensor,
                       x_per_pair: bool, out: torch.Tensor) -> torch.Tensor:
    """[R3.2 original, kept as reference] 13 small kernels ending in a float-atomic scatter_add_: the summation order
    of several outliers that hit the same output row is not fixed (run-to-run drift)."""
    if lay.k_sparse == 0:
        return out
    P = int(slots.numel())
    K = lay.k_sparse
    sp = rows[:, lay.off["sp"]: lay.off["sp"] + 6 * K].index_select(0, slots.long()).reshape(P, K, 6)
    r = sp[..., 0:2].contiguous().view(torch.int16).view(P, K).long() & 0xFFFF
    c = sp[..., 2:4].contiguous().view(torch.int16).view(P, K).long() & 0xFFFF
    v = sp[..., 4:6].contiguous().view(torch.float16).view(P, K).float()
    xp = x_orig if x_per_pair else x_orig.expand(P, -1)
    contrib = v * torch.gather(xp.float(), 1, c)
    out.scatter_add_(1, r, contrib)
    return out


# [R3.4-2a] deterministic sparse add: ONE launch, one block per pair, one thread per slot, no atomics. Each output row's
# outliers are added in slot (index) order with __fmul_rn / __fadd_rn (never contracted to FMA): for a row with one outlier
# this is exactly the old chain's fp32 product + fp32 add; for a row with several outliers it is the old chain's sum in
# one fixed order. The thread holding a row's FIRST slot owns the row: it loads the base value, adds every slot j >= its
# own that targets the row in index order, and writes the row once (no other thread writes it).
_TCQ_SPARSE_SRC = r"""
#include <cuda_fp16.h>
extern "C" __global__ void tcq_sparse_add_kernel(
    const unsigned char* __restrict__ rows, int row_bytes, const int* __restrict__ slots, int off_sp, int K,
    const float* __restrict__ x, int x_stride, float* __restrict__ out, int out_stride, int P)
{
    extern __shared__ int sm[];
    int* s_row = sm;                                        // [K] output row of each slot
    float* s_prod = reinterpret_cast<float*>(sm + K);       // [K] fp32 value * x[col]
    const int pr = blockIdx.x;
    if (pr >= P) return;
    const int tid = threadIdx.x, nt = blockDim.x;
    const unsigned short* sp = reinterpret_cast<const unsigned short*>(rows + (long long)slots[pr] * (long long)row_bytes + off_sp);
    const float* xp = x + (long long)pr * (long long)x_stride;   // x_stride 0 = one shared activation row
    float* op = out + (long long)pr * (long long)out_stride;
    for (int k = tid; k < K; k += nt) {
        const int c = (int)sp[3 * k + 1];
        s_row[k] = (int)sp[3 * k];
        s_prod[k] = __fmul_rn(__half2float(__ushort_as_half(sp[3 * k + 2])), xp[c]);
    }
    __syncthreads();
    for (int k = tid; k < K; k += nt) {
        const int r = s_row[k];
        float acc = op[r];
        bool first = true;
        for (int j = 0; j < K; ++j) {                       // uniform loop: s_row[j] / s_prod[j] are broadcast reads
            const int rj = s_row[j];
            const float pj = s_prod[j];
            if (rj == r) {
                if (j < k) first = false;
                else acc = __fadd_rn(acc, pj);
            }
        }
        if (first) op[r] = acc;
    }
}
"""
_SPARSE_DET = [True]   # [R3.4-2a] False = old torch chain (reference / A-B only)


def tcq_sparse_add(rows: torch.Tensor, slots: torch.Tensor, lay: TcqProjLayout, x_orig: torch.Tensor,
                   x_per_pair: bool, out: torch.Tensor) -> torch.Tensor:
    """Adds the super-weight / outlier slots (original domain): out[p, row] += val * x[p, col]  (in place, returns out).
    [R3.4-2a] one deterministic NVRTC launch (see _TCQ_SPARSE_SRC); falls back to the reference chain off-GPU."""
    if lay.k_sparse == 0:
        return out
    if not _SPARSE_DET[0] or not rows.is_cuda:
        return tcq_sparse_add_ref(rows, slots, lay, x_orig, x_per_pair, out)
    P = int(slots.numel())
    K = int(lay.k_sparse)
    off = int(lay.off["sp"])
    assert off % 2 == 0 and int(lay.row_bytes) % 2 == 0 and rows.stride(0) == int(lay.row_bytes)
    assert out.dtype == torch.float32 and out.dim() == 2 and out.stride(1) == 1 and out.shape[0] == P
    xf = x_orig.float()
    if xf.dim() != 2 or xf.stride(1) != 1:
        xf = xf.reshape(-1, xf.shape[-1]).contiguous()
    k = _compile("tcq_sparse_add_kernel", _TCQ_SPARSE_SRC)
    nt = min(256, ((K + 31) // 32) * 32)
    k(grid=(P, 1, 1), block=(nt, 1, 1), shared_mem=8 * K,
      args=[rows, int(lay.row_bytes), slots.to(torch.int32).contiguous(), off, K, xf,
            int(xf.stride(0)) if x_per_pair else 0, out, int(out.stride(0)), P])
    return out


# v2: four consecutive weights per lane-iteration share one 32-bit window read (4 byte loads instead of 12)
_TCQ_GEMV_V2_SRC = r"""
#include <cuda_fp16.h>
extern "C" __global__ void tcq_gemv_v2_kernel(
    const unsigned char* __restrict__ rows, long long row_bytes, const int* __restrict__ slots,
    int off_map, int off_scale, int off_c1, int off_c2, int off_c3,
    int n1, int n2, int n3, int rb1, int rb2, int rb3,
    const float* __restrict__ tab_g, const float* __restrict__ x_g, int x_per_pair, float* __restrict__ y,
    int d_out, int T)
{
    extern __shared__ float smem[];
    float* tab = smem;
    float* xs = smem + 3 * 4096;
    const int pr = blockIdx.y;
    const float* xg = x_g + (size_t)(x_per_pair ? pr : 0) * T;
    for (int i = threadIdx.x; i < 3 * 4096; i += blockDim.x) tab[i] = tab_g[i];
    for (int i = threadIdx.x; i < T; i += blockDim.x) xs[i] = xg[i];
    __syncthreads();
    const unsigned char* base = rows + (long long)slots[pr] * row_bytes;
    const unsigned short* rmap = reinterpret_cast<const unsigned short*>(base + off_map);
    const __half* scale = reinterpret_cast<const __half*>(base + off_scale);
    const int lane = threadIdx.x & 31;
    const int warps = blockDim.x >> 5;
    for (int r = blockIdx.x * warps + (threadIdx.x >> 5); r < d_out; r += gridDim.x * warps) {
        int k; const unsigned char* p;
        if (r < n1) { k = 1; p = base + off_c1 + (size_t)r * rb1; }
        else if (r < n1 + n2) { k = 2; p = base + off_c2 + (size_t)(r - n1) * rb2; }
        else { k = 3; p = base + off_c3 + (size_t)(r - n1 - n2) * rb3; }
        const float* tk = tab + (k - 1) * 4096;
        float acc = 0.f;
        for (int t0 = lane * 4; t0 < T; t0 += 128) {
            int bit = k * t0;
            int byte = bit >> 3, off = bit & 7;
            unsigned int w32 = ((unsigned int)p[byte] << 24) | ((unsigned int)p[byte + 1] << 16) |
                               ((unsigned int)p[byte + 2] << 8) | (unsigned int)p[byte + 3];
            float4 xv = *reinterpret_cast<const float4*>(xs + t0);
            int s0 = 20 - off;
            acc += tk[(w32 >> s0) & 4095] * xv.x;
            acc += tk[(w32 >> (s0 - k)) & 4095] * xv.y;
            acc += tk[(w32 >> (s0 - 2 * k)) & 4095] * xv.z;
            acc += tk[(w32 >> (s0 - 3 * k)) & 4095] * xv.w;
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
        if (lane == 0) y[(size_t)pr * d_out + rmap[r]] = acc * __half2float(scale[r]);
    }
}
"""
_TCQ_V2 = [True]
_TCQ_GX = [32]  # blocks per pair along d_out (amortises the 48 KB table load per block)


def tcq_gemv_v2(rows, slots, lay, x_rot, x_per_pair, out=None, gx=None):
    P = int(slots.numel())
    if out is None:
        out = torch.empty((P, lay.d_out), dtype=torch.float32, device=rows.device)
    shared = (3 * 4096 + lay.T) * 4
    k = _compile("tcq_gemv_v2_kernel", _TCQ_GEMV_V2_SRC, shared=max(shared, 64 * 1024))
    if lay.tab.device != rows.device:
        lay.tab = lay.tab.to(rows.device)
    warps = 8
    if gx is None:
        gx = _TCQ_GX[0]
    gx = min((lay.d_out + warps - 1) // warps, int(gx))
    k(grid=(gx, P, 1), block=(32 * warps, 1, 1), shared_mem=shared,
      args=[rows, int(lay.row_bytes), slots.to(torch.int32), lay.off["map"], lay.off["scale"], lay.off["c1"],
            lay.off["c2"], lay.off["c3"], lay.n1, lay.n2, lay.n3, lay.rb[0], lay.rb[1], lay.rb[2], lay.tab,
            x_rot.float().contiguous(), 1 if x_per_pair else 0, out, lay.d_out, lay.T])
    return out


# ----------------------------------------------------------------------------------------------------------------------
# [R3.3] dense decode of TCQ projections (rotated domain) -> bf16 [E, d_out, T], for GEMM-based prefill / calibration.
# Same bit-window logic as tcq_gemv_v2: weight t of a k-bit row is LUT_k[(12-bit window at bit k*t)] * row_scale,
# written at its ORIGINAL row (row map). Sparse outliers are not included (add tcq_sparse_add on the original domain).
_TCQ_DENSE_SRC = r"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>
extern "C" __global__ void tcq_dense_kernel(
    const unsigned char* __restrict__ rows, long long row_bytes, const int* __restrict__ slots,
    int off_map, int off_scale, int off_c1, int off_c2, int off_c3,
    int n1, int n2, int n3, int rb1, int rb2, int rb3,
    const float* __restrict__ tab_g, __nv_bfloat16* __restrict__ w_out, int d_out, int T)
{
    __shared__ float tab[3 * 4096];
    for (int i = threadIdx.x; i < 3 * 4096; i += blockDim.x) tab[i] = tab_g[i];
    __syncthreads();
    const int e = blockIdx.y;
    const unsigned char* base = rows + (long long)slots[e] * row_bytes;
    const unsigned short* rmap = reinterpret_cast<const unsigned short*>(base + off_map);
    const __half* scale = reinterpret_cast<const __half*>(base + off_scale);
    const int lane = threadIdx.x & 31;
    const int warps = blockDim.x >> 5;
    for (int r = blockIdx.x * warps + (threadIdx.x >> 5); r < d_out; r += gridDim.x * warps) {
        int k; const unsigned char* p;
        if (r < n1) { k = 1; p = base + off_c1 + (size_t)r * rb1; }
        else if (r < n1 + n2) { k = 2; p = base + off_c2 + (size_t)(r - n1) * rb2; }
        else { k = 3; p = base + off_c3 + (size_t)(r - n1 - n2) * rb3; }
        const float* tk = tab + (k - 1) * 4096;
        const float sc = __half2float(scale[r]);
        __nv_bfloat16* o = w_out + ((size_t)e * d_out + rmap[r]) * T;
        for (int t0 = lane * 4; t0 < T; t0 += 128) {
            int bit = k * t0;
            int byte = bit >> 3, off = bit & 7;
            unsigned int w32 = ((unsigned int)p[byte] << 24) | ((unsigned int)p[byte + 1] << 16) |
                               ((unsigned int)p[byte + 2] << 8) | (unsigned int)p[byte + 3];
            int s0 = 20 - off;
            o[t0]     = __float2bfloat16(tk[(w32 >> s0) & 4095] * sc);
            o[t0 + 1] = __float2bfloat16(tk[(w32 >> (s0 - k)) & 4095] * sc);
            o[t0 + 2] = __float2bfloat16(tk[(w32 >> (s0 - 2 * k)) & 4095] * sc);
            o[t0 + 3] = __float2bfloat16(tk[(w32 >> (s0 - 3 * k)) & 4095] * sc);
        }
    }
}
"""


def tcq_dense(rows: torch.Tensor, slots: torch.Tensor, lay: "TcqProjLayout", out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """rows uint8 [S, row_bytes] CUDA; slots int32 [E] -> bf16 [E, d_out, T] rotated-domain weights (scales applied)."""
    E = int(slots.numel())
    if out is None:
        out = torch.empty((E, lay.d_out, lay.T), dtype=torch.bfloat16, device=rows.device)
    k = _compile("tcq_dense_kernel", _TCQ_DENSE_SRC, shared=3 * 4096 * 4)
    if lay.tab.device != rows.device:
        lay.tab = lay.tab.to(rows.device)
    warps = 8
    gx = min((lay.d_out + warps - 1) // warps, 64)
    k(grid=(gx, E, 1), block=(32 * warps, 1, 1), shared_mem=0,
      args=[rows, int(lay.row_bytes), slots.to(torch.int32), lay.off["map"], lay.off["scale"], lay.off["c1"],
            lay.off["c2"], lay.off["c3"], lay.n1, lay.n2, lay.n3, lay.rb[0], lay.rb[1], lay.rb[2], lay.tab,
            out, lay.d_out, lay.T])
    return out


# ----------------------------------------------------------------------------------------------------------------------
# [R3.5-M1] device-side conditional copy (the N=2 verify step's in-graph rollback): dst[j] <- src[j] for every tensor pair
# j of a pointer table, but only when *flag == when (else every block returns at once). One launch inside the captured
# final graph of the N=2 set: on a rejected draft it restores the 34 KDA recurrent states (4 MiB each) and conv states
# from their post-t_last copies; on an accepted draft it costs one near-empty launch. Plain copies -> bit-exact.
# Scalar arguments are 32-bit ints (torch.cuda._compile_kernel passes Python ints as C int); the flag is a device int64.
_COND_COPY_SRC = r"""
extern "C" __global__ void cond_copy_kernel(
    const long long* __restrict__ dst, const long long* __restrict__ src, const long long* __restrict__ n4,
    const long long* __restrict__ flag, int when)
{
    if (flag[0] != (long long)when) return;
    const int j = blockIdx.y;
    float4* d = reinterpret_cast<float4*>(dst[j]);
    const float4* s = reinterpret_cast<const float4*>(src[j]);
    const long long n = n4[j];
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        d[i] = s[i];
}
"""


def cond_copy_table(dsts, srcs, gx: int = 32) -> Optional[Dict[str, object]]:
    """[R3.5-M1] pointer table for cond_copy(): fp32, contiguous, 16-byte aligned CUDA tensors with numel % 4 == 0 and
    equal sizes per pair (static buffers: their addresses never change). None when any pair does not qualify (the
    caller then uses cond_copy_torch)."""
    if not dsts or len(dsts) != len(srcs):
        return None
    for d, s in zip(dsts, srcs):
        for t in (d, s):
            if not (t.is_cuda and t.dtype == torch.float32 and t.is_contiguous() and t.numel() % 4 == 0
                    and t.data_ptr() % 16 == 0):
                return None
        if d.numel() != s.numel() or d.device != s.device:
            return None
    dev = dsts[0].device
    return {"dst": torch.tensor([t.data_ptr() for t in dsts], dtype=torch.long, device=dev),
            "src": torch.tensor([t.data_ptr() for t in srcs], dtype=torch.long, device=dev),
            "n4": torch.tensor([t.numel() // 4 for t in dsts], dtype=torch.long, device=dev),
            "n": len(dsts), "gx": int(gx), "keep": (list(dsts), list(srcs))}


def cond_copy(tab: Dict[str, object], flag: torch.Tensor, when: int = 0) -> None:
    """[R3.5-M1] dst[j] <- src[j] for all pairs of `tab` iff flag[0] == when (flag: int64 CUDA tensor). Capturable."""
    k = _compile("cond_copy_kernel", _COND_COPY_SRC)
    k(grid=(int(tab["gx"]), int(tab["n"]), 1), block=(256, 1, 1),
      args=[tab["dst"], tab["src"], tab["n4"], flag, int(when)])


def cond_copy_torch(dsts, srcs, flag: torch.Tensor, when: int = 0) -> None:
    """[R3.5-M1] reference / fallback of cond_copy with torch ops (capturable, also on CPU): reads and writes every
    tensor on both outcomes (select, no data-dependent control flow)."""
    m = flag.reshape(-1)[:1] == int(when)
    for d, s in zip(dsts, srcs):
        d.copy_(torch.where(m, s, d))
