#!/usr/bin/env python3
"""
r39_l27_qtip_2d_ldlq_lab.py  [Round 3.9 Single-Layer L27 Gate Validation]
=========================================================================
End-to-end single-layer validation harness comparing Round 3.3/3.7 `TCQ33` (L=12 explicit LUT,
1D BlockLDLQ g=256, uncompensated FP8 H_down) against Round 3.9 `QTIP` L=16 trellises on Layer 27
of GLM-5.3-Flash (288 routed experts, held-out `qbench` / `eval` tokens from `r33_capture`).

Controlled Ablation Ladder Implemented in This Script:
  1. `TCQ33_A_best` (Baseline):
       Exact reproduction of `r33_layer_lab.py` `A_best` (`L=12` 4096-state explicit LUT, 1D `g=256`
       BlockLDLQ, `scale_mode="hunb"`, `n0=8192`, `damp=0.10`, exact-FP8 `H_down`, `2.0216 bpw`).
  2. `QTIP_L16_mul1_1D`:
       Swaps `L=12` explicit LUT for lookup-free `L=16, V=1` `mul1` (`0 B` LUT) + 2-pass collar
       tail-biting (`0` start-state bits, `2.0112 bpw`), keeping 1D `g=256` BlockLDLQ.
  3. `QTIP_L16_HYB_1D`:
       Swaps `mul1` for `L=16, V=2, Q=9` bijective `HYB-mul1` (`2 KiB` `half2[512]` LUT, dual sign
       flip via `0x80008000u` XOR mask, data-free Viterbi-Lloyd trained), keeping 1D `g=256`.
  4. `QTIP_L16_HYB_2D16`:
       Upgrades 1D `g=256` BlockLDLQ to QTIP `16 x 16` 2D-BlockLDLQ (`g=16` Cholesky column feedback,
       16x tighter error bound, RMS-grouped 16-row tiles with exact `rms^2` tile row weighting) AND
       verifies the NVRTC `qtip_tile16x16_gemv_kernel` (`20 KiB` shared mem) against reference decode.
  5. `QTIP_L16_HYB_2D16_QZ`:
       Adds Cross-Covariance Quantized-`gate/up` `H_down` Compensation:
         W_d* = W_d + W_d (H_{z, z_hat} - H_{z_hat, z_hat}) (H_{z_hat, z_hat} + qz_damp * mu * I)^{-1}
       quantized under `H_{z_hat, z_hat}` (fixing R3.3 `down_from_quantized`, which updated `Hz`
       without updating the target weight matrix `W_d` to cancel `W_d (z - z_hat)`).
  6. `QTIP_L16_HYB_2D16_QZ_Tune`:
       Adds fast held-out-validated (`seq % 5 == 0`) joint tuning of the layer-shared `2 KiB` `HYB`
       codebooks (`[2, 512, 2]` FP16) + per-expert in-place FP16 row `scale` (`0 B` slab overhead).
  7. `QTIP_Sub70_GU125_2D16_QZ_Tune`:
       Evaluates the `< 70 GB` (`1.917 bpw` routed expert) candidate (`n1 = 256` `k=1` lowest-`hnorm`
       rows on `gate_proj` and `up_proj` only; `down_proj` kept 100% at `k=2`) vs `TCQ33` `L190_best`.

Typical Usage on Your RTX 5090 Workstation (Windows or WSL2):
  # 1) Fast synthetic NVRTC + bijection + 2D-LDLQ + QZ self-test (~8s, no model files required):
  python r39_l27_qtip_2d_ldlq_lab.py --selftest

  # 2) Full 288-expert Layer 27 validation on qbench tokens (matches RESULTS-095 `27_qb_calct`):
  python r39_l27_qtip_2d_ldlq_lab.py --layer 27 --score-split qb --cal-split ct

  # 3) Rapid 32-expert smoke gate on Layer 27 (~30s) before all 288 experts:
  python r39_l27_qtip_2d_ldlq_lab.py --layer 27 --score-split qb --cal-split mix --max-experts 32
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

LIM = 10.0  # Exact GLM-5.3-Flash SwiGLU clamp: silu(min(g, 10.0)) * clamp(u, -10.0, 10.0)


# ================================================================================================================
# Path Resolution & Optional APEX-FOLD R3.2/R3.3 Module Discovery
# ================================================================================================================
def _first_existing(candidates: List[str], fallback: str) -> str:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return fallback


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PKG = _first_existing(
    [
        os.path.join(os.path.dirname(HERE), "runtime"),
        os.path.join(HERE, "claude_r32_apex_fold"),
        os.path.join(os.path.dirname(HERE), "claude_r32_apex_fold"),
        r"C:\models\GLM53-APEX-FOLD-TCQ\claude_r32_apex_fold",
        "/mnt/c/models/GLM53-APEX-FOLD-TCQ/claude_r32_apex_fold",
    ],
    os.path.join(os.path.dirname(HERE), "claude_r32_apex_fold"),
)
DEFAULT_RANGE = _first_existing(
    [r"C:\models\GLM53-RANGE-EXPERTS", "/mnt/c/models/GLM53-RANGE-EXPERTS"],
    r"C:\models\GLM53-RANGE-EXPERTS",
)
DEFAULT_CAP = _first_existing(
    [r"C:\models\r33_capture", "/mnt/c/models/r33_capture"],
    r"C:\models\r33_capture",
)
DEFAULT_OUT = os.path.join(HERE, "out_r32", "r39_l27_qtip_lab.json")


# ================================================================================================================
# Self-Contained FWHT + Rademacher Signs + Forward Block-LDL (Bit-Exact with glm53_fold_math / tcq_ldlq)
# ================================================================================================================
_HAD_CACHE: Dict[Tuple[int, str], torch.Tensor] = {}


def _hadamard(n: int, device: torch.device) -> torch.Tensor:
    key = (n, str(device))
    if key not in _HAD_CACHE:
        h = np.array([[1.0]], dtype=np.float32)
        while h.shape[0] < n:
            h = np.block([[h, h], [h, -h]])
        h = h * (1.0 / math.sqrt(float(n)))
        _HAD_CACHE[key] = torch.from_numpy(h).to(device=device, dtype=torch.float32).contiguous()
    return _HAD_CACHE[key]


def fwht_full(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal Kronecker Fast Walsh-Hadamard Transform across last dim d_in (power of 2)."""
    orig_shape = x.shape
    d_in = orig_shape[-1]
    assert d_in > 0 and (d_in & (d_in - 1)) == 0, f"d_in={d_in} must be power of two"
    xf = x.float()
    if d_in <= 64:
        return torch.matmul(xf, _hadamard(d_in, xf.device)).to(x.dtype)
    log_d = int(round(math.log2(d_in)))
    d1 = 1 << (log_d // 2)
    d2 = d_in // d1
    v2d = xf.reshape(-1, d1, d2)
    out = torch.matmul(_hadamard(d1, xf.device), torch.matmul(v2d, _hadamard(d2, xf.device))).reshape(orig_shape)
    return out.to(x.dtype)


def generate_rademacher_signs(d_in: int, seed: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) & 0xFFFFFFFF)
    bits = torch.randint(0, 2, (d_in,), generator=gen, dtype=torch.int8)
    return (bits * 2 - 1).to(device=device)


def rotate_hessian(H: torch.Tensor, signs_f32: torch.Tensor) -> torch.Tensor:
    Hs = H * signs_f32.view(1, -1, 1) * signs_f32.view(1, 1, -1)
    Hs = fwht_full(Hs)
    return fwht_full(Hs.transpose(1, 2)).transpose(1, 2)


def block_ldl_forward(H: torch.Tensor, g: int) -> torch.Tensor:
    """H [B, n, n] SPD -> Uf [B, n, n] block-unit-upper-triangular (block size g) s.t. H = Uf Dblk Uf^T."""
    B, n, _ = H.shape
    assert n % g == 0
    nb = n // g
    J = torch.arange(n - 1, -1, -1, device=H.device)
    Hr = H[:, J][:, :, J].double()
    L = torch.linalg.cholesky(Hr)
    Lb = L.view(B, nb, g, nb, g)
    diag = torch.stack([Lb[:, b, :, b, :] for b in range(nb)], dim=1)  # [B, nb, g, g]
    eye = torch.eye(g, device=H.device, dtype=torch.float64).expand(B, nb, g, g)
    dinv = torch.linalg.solve_triangular(diag, eye, upper=False)
    Lu = torch.empty_like(L).view(B, nb, g, nb, g)
    for c in range(nb):
        Lu[:, :, :, c, :] = torch.einsum("bpij,bjk->bpik", Lb[:, :, :, c, :], dinv[:, c])
    Lu = Lu.view(B, n, n)
    return Lu[:, J][:, :, J].float()


# ================================================================================================================
# NVRTC CUDA Kernels: Register-Resident L=16 Viterbi (HYB V=2 & mul1 V=1) + 16x16 Tile GEMV
# ================================================================================================================
_NVRTC_CACHE: Dict[str, object] = {}


def _compile_nvrtc(name: str, src: str, shared_bytes: int = 0):
    if name not in _NVRTC_CACHE:
        # [R3.9 fix 1, 2026-11-16] The two Viterbi kernels launch 1,024 threads per block, which allows at most 64
        # registers per thread (65,536 per SM). As shipped they compiled to 128 and 222 registers and every launch
        # failed with "too many resources requested for launch". The cap changes register allocation only.
        kern = torch.cuda._compile_kernel(src, name, nvcc_options=[f"--maxrregcount={int(os.environ.get('R39_MAXRREG', '64'))}"])
        if shared_bytes >= 48 * 1024:
            kern.set_shared_memory_config(shared_bytes)
        _NVRTC_CACHE[name] = kern
    return _NVRTC_CACHE[name]


_QTIP_L16_CUDA_SRC = r"""
#include <cuda_fp16.h>

// Bijective mul1-hashed HYB (L=16, V=2, Q=9):
//   1) h = 34038481u * s16 + 76625530u  (bijective mod 2^32, uniform across all 512 buckets)
//   2) idx = (h >> 17) & 511u           (9 bits disjoint from sign bits 15 & 31)
//   3) raw ^ (h & 0x80008000u)          (single lop3 XOR flips both half2 signs independently)
__device__ __forceinline__ float2 qtip_hyb_mul1_pair(unsigned int s16, const unsigned int* __restrict__ lut_u32) {
    unsigned int h = 34038481u * s16 + 76625530u;
    unsigned int raw = lut_u32[(h >> 17) & 511u];
    unsigned int signed_bits = raw ^ (h & 0x80008000u);
    return __half22float2(*reinterpret_cast<const __half2*>(&signed_bits));
}

// QTIP Algorithm 1 (1MAD / EXL3 mul1, L=16, V=1): 0 B LUT
__device__ __forceinline__ float qtip_mul1_scalar(unsigned int s16) {
    unsigned int x = 34038481u * s16 + 76625530u;
    unsigned int bsum = (x & 255u) + ((x >> 8) & 255u) + ((x >> 16) & 255u) + (x >> 24);
    return ((float)bsum - 510.0f) * (1.0f / 147.8f);
}

// Circular MSB-first 16-bit state reader for power-of-two byte buffer rb (32, 64, 256, 512, 1024 B)
__device__ __forceinline__ unsigned int read_tailbite_state16(
    const unsigned char* __restrict__ p, int rb, int bit_pos)
{
    int b0 = (bit_pos >> 3) & (rb - 1);
    int b1 = (b0 + 1) & (rb - 1);
    int b2 = (b0 + 2) & (rb - 1);
    int off = bit_pos & 7;
    unsigned int w24 = ((unsigned int)p[b0] << 16) | ((unsigned int)p[b1] << 8) | (unsigned int)p[b2];
    return (w24 >> (8 - off)) & 0xFFFFu;
}

// ------------------------------------------------------------------------------------------------
// Kernel 1: Register-Resident L=16, V=2 HYB Viterbi for k=2 (NG = 4096, NCHOICE = 16, 20 KiB smem)
// 1024 threads/CTA; each thread holds 64 states in 32x __half2 registers (ZERO global cost_buf!).
// ------------------------------------------------------------------------------------------------
extern "C" __global__ void qtip_viterbi_hyb_k2_kernel(
    const float* __restrict__ X,                 // [R, 2 * M]
    const float* __restrict__ row_w,             // [R, M] optional per-step weight (nullptr = 1.0f)
    const unsigned int* __restrict__ lut_g,      // [512] packed half2
    unsigned char* __restrict__ bp,              // [R, M, 4096]
    const int* __restrict__ start_state,         // [R] (-1 = free)
    const int* __restrict__ end_state,           // [R] (-1 = free)
    int* __restrict__ states_out,                // [R, M]
    int M, int has_row_w)
{
    __shared__ float sm_best[4096];              // 16 KiB FP32
    __shared__ unsigned int s_lut[512];          // 2 KiB
    __shared__ int sm_arg[1024];                 // 4 KiB (used only when end_state < 0)

    const int r = blockIdx.x;
    const int tid = threadIdx.x;                 // 0..1023
    const float* xr = X + (size_t)r * (2 * M);
    const float* rw_ptr = has_row_w ? (row_w + (size_t)r * M) : nullptr;
    const size_t bp_row = (size_t)r * (size_t)M * 4096ULL;

    if (tid < 512) s_lut[tid] = lut_g[tid];
    __syncthreads();

    const int s_in = start_state[r];
    __half2 reg_c[4][8];                         // u=0..3 (q = tid + 1024*u), p=0..7 (j = 2*p, 2*p+1)

    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        int q = tid + 1024 * u;
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            int s0 = ((2 * p) << 12) | q;
            int s1 = ((2 * p + 1) << 12) | q;
            float v0 = (s_in < 0 || s0 == s_in) ? 0.0f : 16384.0f;
            float v1 = (s_in < 0 || s1 == s_in) ? 0.0f : 16384.0f;
            reg_c[u][p] = __floats2half2_rn(v0, v1);
        }
    }

    for (int m = 0; m < M; ++m) {
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            int q = tid + 1024 * u;
            float2 f0 = __half22float2(reg_c[u][0]);
            float best = f0.x;
            int arg = 0;
            if (f0.y < best) { best = f0.y; arg = 1; }
            #pragma unroll
            for (int p = 1; p < 8; ++p) {
                float2 fp = __half22float2(reg_c[u][p]);
                if (fp.x < best) { best = fp.x; arg = 2 * p; }
                if (fp.y < best) { best = fp.y; arg = 2 * p + 1; }
            }
            bp[bp_row + (size_t)m * 4096ULL + (size_t)q] = (unsigned char)arg;
            sm_best[q] = best;
        }
        __syncthreads();

        const float x0 = xr[2 * m];
        const float x1 = xr[2 * m + 1];
        const float rw = has_row_w ? rw_ptr[m] : 1.0f;
        const float s_norm = (m >= 8) ? sm_best[0] : 0.0f;

        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            unsigned int q = (unsigned int)(tid + 1024 * u);
            #pragma unroll
            for (int p = 0; p < 8; ++p) {
                unsigned int s0 = ((unsigned int)(2 * p) << 12) | q;
                unsigned int s1 = ((unsigned int)(2 * p + 1) << 12) | q;
                float2 w0 = qtip_hyb_mul1_pair(s0, s_lut);
                float2 w1 = qtip_hyb_mul1_pair(s1, s_lut);
                float d0 = (x0 - w0.x) * (x0 - w0.x) + (x1 - w0.y) * (x1 - w0.y);
                float d1 = (x0 - w1.x) * (x0 - w1.x) + (x1 - w1.y) * (x1 - w1.y);
                float n0 = fminf(fmaxf(sm_best[s0 >> 4] + rw * d0 - s_norm, -8192.0f), 32768.0f);
                float n1 = fminf(fmaxf(sm_best[s1 >> 4] + rw * d1 - s_norm, -8192.0f), 32768.0f);
                reg_c[u][p] = __floats2half2_rn(n0, n1);
            }
        }
        __syncthreads();
    }

    int s_end = end_state[r];
    if (s_end < 0) {
        float loc_best = 3.4e38f;
        int loc_arg = 0;
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            int q = tid + 1024 * u;
            #pragma unroll
            for (int p = 0; p < 8; ++p) {
                float2 fp = __half22float2(reg_c[u][p]);
                if (fp.x < loc_best) { loc_best = fp.x; loc_arg = ((2 * p) << 12) | q; }
                if (fp.y < loc_best) { loc_best = fp.y; loc_arg = ((2 * p + 1) << 12) | q; }
            }
        }
        sm_best[tid] = loc_best;
        sm_arg[tid] = loc_arg;
        __syncthreads();
        for (int off = 512; off > 0; off >>= 1) {
            if (tid < off && sm_best[tid + off] < sm_best[tid]) {
                sm_best[tid] = sm_best[tid + off];
                sm_arg[tid] = sm_arg[tid + off];
            }
            __syncthreads();
        }
        s_end = sm_arg[0];
    }

    if (tid == 0) {
        int s = s_end;
        int* so = states_out + (size_t)r * M;
        for (int m = M - 1; m >= 0; --m) {
            so[m] = s;
            int q = s >> 4;
            int j = (int)bp[bp_row + (size_t)m * 4096ULL + (size_t)q];
            s = (j << 12) | q;
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Kernel 2: Register-Resident L=16 4-Branch Viterbi (Shared by HYB V=2 k=1 AND mul1 V=1 k=2!)
// NG = 16384, NCHOICE = 4, SHIFT = 14, STEP_BITS = 2; uses 66 KiB dynamic shared memory.
// mode == 0: HYB V=2 k=1 (consumes 2 floats per step); mode == 1: mul1 V=1 k=2 (1 float per step).
// ------------------------------------------------------------------------------------------------
extern "C" __global__ void qtip_viterbi_4branch_kernel(
    const float* __restrict__ X,                 // [R, V * M]
    const float* __restrict__ row_w,             // [R, M] optional per-step weight
    const unsigned int* __restrict__ lut_g,      // [512] packed half2 (unused when mode == 1)
    unsigned char* __restrict__ bp,              // [R, M, 16384]
    const int* __restrict__ start_state,
    const int* __restrict__ end_state,
    int* __restrict__ states_out,
    int M, int has_row_w, int mode)
{
    extern __shared__ float sm_dyn[];
    float* sm_best = sm_dyn;                                                // 16384 floats = 64 KiB
    unsigned int* s_lut = reinterpret_cast<unsigned int*>(sm_dyn + 16384);  // 512 uint32 = 2 KiB
    int* sm_arg = reinterpret_cast<int*>(s_lut + 512);                      // 1024 int = 4 KiB

    const int r = blockIdx.x;
    const int tid = threadIdx.x;                 // 0..1023
    const int V = (mode == 0) ? 2 : 1;
    const float* xr = X + (size_t)r * (size_t)(V * M);
    const float* rw_ptr = has_row_w ? (row_w + (size_t)r * M) : nullptr;
    const size_t bp_row = (size_t)r * (size_t)M * 16384ULL;

    if (mode == 0 && tid < 512) s_lut[tid] = lut_g[tid];
    __syncthreads();

    const int s_in = start_state[r];
    __half2 reg_c[16][2];                        // u=0..15 (q = tid + 1024*u), p=0..1 (j = 0..3)

    #pragma unroll
    for (int u = 0; u < 16; ++u) {
        int q = tid + 1024 * u;
        #pragma unroll
        for (int p = 0; p < 2; ++p) {
            int s0 = ((2 * p) << 14) | q;
            int s1 = ((2 * p + 1) << 14) | q;
            float v0 = (s_in < 0 || s0 == s_in) ? 0.0f : 16384.0f;
            float v1 = (s_in < 0 || s1 == s_in) ? 0.0f : 16384.0f;
            reg_c[u][p] = __floats2half2_rn(v0, v1);
        }
    }

    for (int m = 0; m < M; ++m) {
        #pragma unroll
        for (int u = 0; u < 16; ++u) {
            int q = tid + 1024 * u;
            float2 f0 = __half22float2(reg_c[u][0]);
            float2 f1 = __half22float2(reg_c[u][1]);
            float best = f0.x; int arg = 0;
            if (f0.y < best) { best = f0.y; arg = 1; }
            if (f1.x < best) { best = f1.x; arg = 2; }
            if (f1.y < best) { best = f1.y; arg = 3; }
            bp[bp_row + (size_t)m * 16384ULL + (size_t)q] = (unsigned char)arg;
            sm_best[q] = best;
        }
        __syncthreads();

        const float x0 = xr[V * m];
        const float x1 = (mode == 0) ? xr[2 * m + 1] : 0.0f;
        const float rw = has_row_w ? rw_ptr[m] : 1.0f;
        const float s_norm = (m >= 8) ? sm_best[0] : 0.0f;

        #pragma unroll
        for (int u = 0; u < 16; ++u) {
            unsigned int q = (unsigned int)(tid + 1024 * u);
            #pragma unroll
            for (int p = 0; p < 2; ++p) {
                unsigned int s0 = ((unsigned int)(2 * p) << 14) | q;
                unsigned int s1 = ((unsigned int)(2 * p + 1) << 14) | q;
                float d0, d1;
                if (mode == 0) {
                    float2 w0 = qtip_hyb_mul1_pair(s0, s_lut);
                    float2 w1 = qtip_hyb_mul1_pair(s1, s_lut);
                    d0 = (x0 - w0.x) * (x0 - w0.x) + (x1 - w0.y) * (x1 - w0.y);
                    d1 = (x0 - w1.x) * (x0 - w1.x) + (x1 - w1.y) * (x1 - w1.y);
                } else {
                    float w0 = qtip_mul1_scalar(s0);
                    float w1 = qtip_mul1_scalar(s1);
                    d0 = (x0 - w0) * (x0 - w0);
                    d1 = (x0 - w1) * (x0 - w1);
                }
                float n0 = fminf(fmaxf(sm_best[s0 >> 2] + rw * d0 - s_norm, -8192.0f), 32768.0f);
                float n1 = fminf(fmaxf(sm_best[s1 >> 2] + rw * d1 - s_norm, -8192.0f), 32768.0f);
                reg_c[u][p] = __floats2half2_rn(n0, n1);
            }
        }
        __syncthreads();
    }

    int s_end = end_state[r];
    if (s_end < 0) {
        float loc_best = 3.4e38f;
        int loc_arg = 0;
        #pragma unroll
        for (int u = 0; u < 16; ++u) {
            int q = tid + 1024 * u;
            #pragma unroll
            for (int p = 0; p < 2; ++p) {
                float2 fp = __half22float2(reg_c[u][p]);
                if (fp.x < loc_best) { loc_best = fp.x; loc_arg = ((2 * p) << 14) | q; }
                if (fp.y < loc_best) { loc_best = fp.y; loc_arg = ((2 * p + 1) << 14) | q; }
            }
        }
        sm_best[tid] = loc_best;
        sm_arg[tid] = loc_arg;
        __syncthreads();
        for (int off = 512; off > 0; off >>= 1) {
            if (tid < off && sm_best[tid + off] < sm_best[tid]) {
                sm_best[tid] = sm_best[tid + off];
                sm_arg[tid] = sm_arg[tid + off];
            }
            __syncthreads();
        }
        s_end = sm_arg[0];
    }

    if (tid == 0) {
        int s = s_end;
        int* so = states_out + (size_t)r * M;
        for (int m = M - 1; m >= 0; --m) {
            so[m] = s;
            int q = s >> 2;
            int j = (int)bp[bp_row + (size_t)m * 16384ULL + (size_t)q];
            s = (j << 14) | q;
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Kernel 3: Coalesced 16x16 2D-Tile HYB Decode GEMV on RTX 5090 (20 KiB shared memory)
// One warp (32 lanes = 16 rows x 2 half-columns) decodes a full 16-row tile with 16x x_rot reuse
// and a single 64-byte sector read per [16, 16] k=2 tile!
// ------------------------------------------------------------------------------------------------
extern "C" __global__ void qtip_tile16x16_hyb_gemv_kernel(
    const unsigned short* __restrict__ rmap_all, // [E, d_out]
    const __half* __restrict__ scale_all,        // [E, d_out]
    const unsigned char* __restrict__ c1_all,    // [E, n1/16, T/16, 32]
    const unsigned char* __restrict__ c2_all,    // [E, n2/16, T/16, 64]
    const int* __restrict__ slots,               // [P] expert indices
    int n1, int n2,
    const unsigned int* __restrict__ hyb_lut_g,  // [2, 512] packed half2 (k=1 and k=2)
    const float* __restrict__ x_g,               // [P or 1, T] rotated inputs
    int x_per_pair,
    float* __restrict__ y,                       // [P, d_out]
    int d_out, int T)
{
    extern __shared__ unsigned int smem_u32[];
    unsigned int* s_lut = smem_u32;                      // 1024 uint32 = 4 KiB
    float* xs = reinterpret_cast<float*>(smem_u32 + 1024);

    const int pr = blockIdx.y;
    const int e = slots[pr];
    const float* xg = x_g + (size_t)(x_per_pair ? pr : 0) * T;

    for (int i = threadIdx.x; i < 1024; i += blockDim.x) s_lut[i] = hyb_lut_g[i];
    for (int i = threadIdx.x; i < T; i += blockDim.x) xs[i] = xg[i];
    __syncthreads();

    const unsigned short* rmap = rmap_all + (size_t)e * d_out;
    const __half* scale = scale_all + (size_t)e * d_out;

    const int lane = threadIdx.x & 31;
    const int r_local = lane >> 1;               // 0..15 row inside 16x16 tile
    const int half_col = lane & 1;               // 0 -> cols 0..7 (pairs 0..3); 1 -> cols 8..15 (pairs 4..7)
    const int warps = blockDim.x >> 5;
    const int n_row_tiles = d_out >> 4;
    const int nb_col = T >> 4;
    const int n1_tiles = n1 >> 4;
    const int n2_tiles = n2 >> 4;

    const unsigned char* e_c1 = c1_all + (size_t)e * (size_t)n1_tiles * (size_t)nb_col * 32ULL;
    const unsigned char* e_c2 = c2_all + (size_t)e * (size_t)n2_tiles * (size_t)nb_col * 64ULL;

    for (int rt = blockIdx.x * warps + (threadIdx.x >> 5); rt < n_row_tiles; rt += gridDim.x * warps) {
        int k, tile_bytes;
        const unsigned char* row_tile_base;
        if (rt < n1_tiles) {
            k = 1; tile_bytes = 32;
            row_tile_base = e_c1 + (size_t)rt * (size_t)nb_col * 32ULL;
        } else {
            k = 2; tile_bytes = 64;
            row_tile_base = e_c2 + (size_t)(rt - n1_tiles) * (size_t)nb_col * 64ULL;
        }
        const unsigned int* tk = s_lut + (k - 1) * 512;
        const int step_bits = k << 1;
        const int m_base = r_local * 8 + half_col * 4;

        float acc = 0.0f;
        for (int ct = 0; ct < nb_col; ++ct) {
            const unsigned char* tp = row_tile_base + (size_t)ct * (size_t)tile_bytes;
            const float* xp = xs + (ct << 4) + (half_col << 3);
            float4 xv0 = *reinterpret_cast<const float4*>(xp);
            float4 xv1 = *reinterpret_cast<const float4*>(xp + 4);

            unsigned int s0 = read_tailbite_state16(tp, tile_bytes, (m_base + 0) * step_bits);
            unsigned int s1 = read_tailbite_state16(tp, tile_bytes, (m_base + 1) * step_bits);
            unsigned int s2 = read_tailbite_state16(tp, tile_bytes, (m_base + 2) * step_bits);
            unsigned int s3 = read_tailbite_state16(tp, tile_bytes, (m_base + 3) * step_bits);

            float2 w0 = qtip_hyb_mul1_pair(s0, tk);
            float2 w1 = qtip_hyb_mul1_pair(s1, tk);
            float2 w2 = qtip_hyb_mul1_pair(s2, tk);
            float2 w3 = qtip_hyb_mul1_pair(s3, tk);

            acc = fmaf(w0.x, xv0.x, acc);
            acc = fmaf(w0.y, xv0.y, acc);
            acc = fmaf(w1.x, xv0.z, acc);
            acc = fmaf(w1.y, xv0.w, acc);
            acc = fmaf(w2.x, xv1.x, acc);
            acc = fmaf(w2.y, xv1.y, acc);
            acc = fmaf(w3.x, xv1.z, acc);
            acc = fmaf(w3.y, xv1.w, acc);
        }

        acc += __shfl_xor_sync(0xffffffff, acc, 1);
        if (half_col == 0) {
            int stored_r = (rt << 4) + r_local;
            y[(size_t)pr * d_out + rmap[stored_r]] = acc * __half2float(scale[stored_r]);
        }
    }
}
"""


# ================================================================================================================
# Python Wrappers for QTIP Tail-Biting Bitpacking, Viterbi, Codebook Training & 2D-BlockLDLQ
# ================================================================================================================
@torch.no_grad()
def eval_hyb_all_states_torch(lut_fp16: torch.Tensor) -> torch.Tensor:
    """lut_fp16: [512, 2] float16 -> [65536, 2] float32 exact match to qtip_hyb_mul1_pair."""
    dev = lut_fp16.device
    s = torch.arange(65536, device=dev, dtype=torch.int64)
    h = (34038481 * s + 76625530) & 0xFFFFFFFF
    idx = (h >> 17) & 511
    sx = torch.where((h & 0x00008000) != 0, -1.0, 1.0)
    sy = torch.where((h & 0x80000000) != 0, -1.0, 1.0)
    base = lut_fp16.float()[idx]
    return torch.stack([base[:, 0] * sx, base[:, 1] * sy], dim=-1)


@torch.no_grad()
def eval_mul1_all_states_torch(device: torch.device) -> torch.Tensor:
    """Returns [65536] float32 lookup-free mul1 table for fast reference weight reconstruction."""
    s = torch.arange(65536, device=device, dtype=torch.int64)
    x = (34038481 * s + 76625530) & 0xFFFFFFFF
    bsum = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24)
    return (bsum.float() - 510.0) / 147.8


@torch.no_grad()
def pack_tailbite_l16(states: torch.Tensor, step_bits: int) -> torch.Tensor:
    """states: [R, M] int32 (16-bit state after each step m, with start_state == states[:, M-1])
    -> uint8 [R, M * step_bits // 8] exact circular bitstream (0 start-state overhead)."""
    R, M = states.shape
    total_bits = M * step_bits
    assert total_bits % 8 == 0 and total_bits >= 16
    rb = total_bits // 8
    hist = 16 // step_bits
    dev = states.device
    bits = torch.zeros((R, total_bits), dtype=torch.uint8, device=dev)
    s0 = states[:, 0]
    for i in range(16):
        bits[:, i] = ((s0 >> (15 - i)) & 1).to(torch.uint8)
    if M > hist:
        low = states[:, 1 : M - hist + 1] & ((1 << step_bits) - 1)
        for i in range(step_bits):
            bits[:, 16 + i : total_bits : step_bits] = ((low >> (step_bits - 1 - i)) & 1).to(torch.uint8)
    weights = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=dev)
    return (bits.view(R, rb, 8) * weights).sum(dim=-1).to(torch.uint8)


@torch.no_grad()
def unpack_tailbite_l16(packed: torch.Tensor, M: int, step_bits: int) -> torch.Tensor:
    """Vectorized PyTorch match to CUDA read_tailbite_state16(p, rb, m * step_bits)."""
    R, rb = packed.shape
    assert (rb & (rb - 1)) == 0, f"rb={rb} must be power of two"
    dev = packed.device
    m = torch.arange(M, device=dev, dtype=torch.long)
    bit_pos = m * step_bits
    b0 = (bit_pos >> 3) & (rb - 1)
    b1 = (b0 + 1) & (rb - 1)
    b2 = (b0 + 2) & (rb - 1)
    off = bit_pos & 7
    p = packed.long()
    w24 = (p[:, b0] << 16) | (p[:, b1] << 8) | p[:, b2]
    return ((w24 >> (8 - off)) & 0xFFFF).to(torch.int32)


@torch.no_grad()
def viterbi_qtip_tailbite_cuda(
    X: torch.Tensor,
    k: int,
    mode: str = "hyb",
    hyb_lut_fp16: Optional[torch.Tensor] = None,
    row_w: Optional[torch.Tensor] = None,
    rows_per_launch: int = 512,
    collar_steps: int = 16,
) -> torch.Tensor:
    """Register-resident L=16 2-pass collar tail-biting Viterbi on GPU.
    mode == 'hyb': V=2 (k in {1, 2}); mode == 'mul1': V=1 (k == 2).
    Returns states [R, M] int32 where M = Nw // V and states[:, -1] == start_state."""
    assert X.is_cuda and X.dtype == torch.float32
    R, Nw = X.shape
    V = 2 if mode == "hyb" else 1
    M = Nw // V
    step_bits = k * V
    assert step_bits in (2, 4), f"Unsupported (mode={mode}, k={k})"
    w_pad = min(collar_steps, M // 2)
    assert w_pad >= 8, f"Sequence M={M} too short for L=16 mixing"

    if mode == "hyb":
        assert hyb_lut_fp16 is not None and hyb_lut_fp16.shape == (512, 2)
        lut_u32 = hyb_lut_fp16.to(device=X.device, dtype=torch.float16).contiguous().view(torch.int32)
    else:
        lut_u32 = torch.zeros((512,), dtype=torch.int32, device=X.device)

    has_rw = 1 if row_w is not None else 0
    X_collar = torch.cat([X[:, -V * w_pad :], X[:, : V * w_pad]], dim=1).contiguous()
    rw_collar = torch.cat([row_w[:, -w_pad:], row_w[:, :w_pad]], dim=1).contiguous() if has_rw else X.new_empty(0)
    rw_full = row_w.contiguous() if has_rw else X.new_empty(0)

    states_out = torch.empty((R, M), dtype=torch.int32, device=X.device)
    free_pin = torch.full((rows_per_launch,), -1, dtype=torch.int32, device=X.device)

    if step_bits == 4:
        NG = 4096
        kern = _compile_nvrtc("qtip_viterbi_hyb_k2_kernel", _QTIP_L16_CUDA_SRC)
        smem = 0
    else:
        NG = 16384
        smem = 72 * 1024
        kern = _compile_nvrtc("qtip_viterbi_4branch_kernel", _QTIP_L16_CUDA_SRC, shared_bytes=smem)
        kern_mode = 0 if mode == "hyb" else 1

    bp = torch.empty((rows_per_launch, M, NG), dtype=torch.uint8, device=X.device)
    st_collar = torch.empty((rows_per_launch, 2 * w_pad), dtype=torch.int32, device=X.device)

    for r0 in range(0, R, rows_per_launch):
        r1 = min(R, r0 + rows_per_launch)
        B = r1 - r0
        if step_bits == 4:
            kern(
                grid=(B, 1, 1),
                block=(1024, 1, 1),
                args=[
                    X_collar[r0:r1],
                    rw_collar[r0:r1] if has_rw else rw_collar,
                    lut_u32,
                    bp[:B, : 2 * w_pad],
                    free_pin[:B],
                    free_pin[:B],
                    st_collar[:B],
                    int(2 * w_pad),
                    int(has_rw),
                ],
            )
            seam = st_collar[:B, w_pad - 1].contiguous()
            kern(
                grid=(B, 1, 1),
                block=(1024, 1, 1),
                args=[
                    X[r0:r1].contiguous(),
                    rw_full[r0:r1] if has_rw else rw_full,
                    lut_u32,
                    bp[:B],
                    seam,
                    seam,
                    states_out[r0:r1],
                    int(M),
                    int(has_rw),
                ],
            )
        else:
            kern(
                grid=(B, 1, 1),
                block=(1024, 1, 1),
                shared_mem=smem,
                args=[
                    X_collar[r0:r1],
                    rw_collar[r0:r1] if has_rw else rw_collar,
                    lut_u32,
                    bp[:B, : 2 * w_pad],
                    free_pin[:B],
                    free_pin[:B],
                    st_collar[:B],
                    int(2 * w_pad),
                    int(has_rw),
                    int(kern_mode),
                ],
            )
            seam = st_collar[:B, w_pad - 1].contiguous()
            kern(
                grid=(B, 1, 1),
                block=(1024, 1, 1),
                shared_mem=smem,
                args=[
                    X[r0:r1].contiguous(),
                    rw_full[r0:r1] if has_rw else rw_full,
                    lut_u32,
                    bp[:B],
                    seam,
                    seam,
                    states_out[r0:r1],
                    int(M),
                    int(has_rw),
                    int(kern_mode),
                ],
            )
    return states_out


@torch.no_grad()
def train_hyb_codebooks_l16_v2(
    device: torch.device,
    cache_path: Optional[str] = None,
    n_rows: int = 512,
    Nw: int = 512,
    iters: int = 12,
    seed: int = 20261116,
) -> Dict[int, torch.Tensor]:
    """Data-free Viterbi-Lloyd training of [512, 2] FP16 HYB-mul1 codebooks for k=1 and k=2 on i.i.d. N(0, 1)."""
    if cache_path and os.path.exists(cache_path):
        loaded = torch.load(cache_path, map_location=device)
        return {1: loaded[1].to(device=device, dtype=torch.float16), 2: loaded[2].to(device=device, dtype=torch.float16)}

    gen_cpu = torch.Generator(device="cpu").manual_seed(seed)
    q32 = 0.5 + 0.5 * (torch.arange(32, dtype=torch.float64) + 0.5) / 32.0
    q16 = 0.5 + 0.5 * (torch.arange(16, dtype=torch.float64) + 0.5) / 16.0
    v32 = math.sqrt(2.0) * torch.erfinv(2.0 * q32 - 1.0)
    v16 = math.sqrt(2.0) * torch.erfinv(2.0 * q16 - 1.0)
    gx, gy = torch.meshgrid(v32, v16, indexing="ij")
    init_grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1).float()
    perm = torch.randperm(512, generator=gen_cpu)
    lut_init = init_grid[perm].to(device=device, dtype=torch.float16).contiguous()

    gen_gpu = torch.Generator(device=device).manual_seed(seed + 7)
    X = torch.randn((n_rows, Nw), generator=gen_gpu, device=device, dtype=torch.float32)

    s_all = torch.arange(65536, device=device, dtype=torch.int64)
    h_all = (34038481 * s_all + 76625530) & 0xFFFFFFFF
    idx_all = (h_all >> 17) & 511
    sx_all = torch.where((h_all & 0x00008000) != 0, -1.0, 1.0)
    sy_all = torch.where((h_all & 0x80000000) != 0, -1.0, 1.0)

    x0_flat = X[:, 0::2].reshape(-1)
    x1_flat = X[:, 1::2].reshape(-1)
    ones = torch.ones_like(x0_flat)

    out: Dict[int, torch.Tensor] = {}
    for k in (1, 2):
        lut = lut_init.clone()
        for _ in range(iters):
            st = viterbi_qtip_tailbite_cuda(X, k=k, mode="hyb", hyb_lut_fp16=lut)
            sf = st.reshape(-1).long()
            b_idx = idx_all[sf]
            cnt = torch.zeros(512, device=device).index_add_(0, b_idx, ones)
            sum0 = torch.zeros(512, device=device).index_add_(0, b_idx, sx_all[sf] * x0_flat)
            sum1 = torch.zeros(512, device=device).index_add_(0, b_idx, sy_all[sf] * x1_flat)
            upd = cnt > 0
            lf = lut.float()
            lf[:, 0] = torch.where(upd, sum0 / cnt.clamp_min(1.0), lf[:, 0])
            lf[:, 1] = torch.where(upd, sum1 / cnt.clamp_min(1.0), lf[:, 1])
            lut = lf.to(torch.float16).contiguous()
        out[k] = lut

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({1: out[1].cpu(), 2: out[2].cpu()}, cache_path)
    return out


# ================================================================================================================
# QTIP Projection Specs, 1D vs 16x16 2D-BlockLDLQ Encoder, Cross-Covariance H_down & Held-Out Joint Tuning
# ================================================================================================================
@dataclass(frozen=True)
class QtipProjSpec:
    name: str
    d_out: int
    d_in: int
    n1: int = 0
    k_sparse: int = 64
    unbiased: bool = False

    @property
    def n2(self) -> int:
        return self.d_out - self.n1

    def code_bytes_per_row(self) -> Tuple[int, int]:
        return (self.d_in * 1) // 8, (self.d_in * 2) // 8

    def expert_bytes(self) -> int:
        b1, b2 = self.code_bytes_per_row()
        return 4 * self.d_out + self.n1 * b1 + self.n2 * b2 + 6 * self.k_sparse


@dataclass(frozen=True)
class QtipPreset:
    name: str
    gate: QtipProjSpec
    up: QtipProjSpec
    down: QtipProjSpec
    mode: str = "hyb"
    tile_mode: str = "2d_16x16"

    def bpw(self, n_experts: int = 288) -> float:
        sign_hdr = sum(sp.d_in for sp in (self.gate, self.up, self.down))
        lut_hdr = (3 * 2 * 2048) if self.mode == "hyb" else 0
        expert_b = sum(sp.expert_bytes() for sp in (self.gate, self.up, self.down))
        tot_b = sign_hdr + lut_hdr + n_experts * expert_b
        tot_w = n_experts * sum(sp.d_out * sp.d_in for sp in (self.gate, self.up, self.down))
        return (tot_b * 8.0) / float(tot_w)


def make_qtip_preset(rule: str = "A", mode: str = "hyb", tile_mode: str = "2d_16x16") -> QtipPreset:
    if rule == "A":
        n1_gu = 0
    elif rule == "GU125":
        n1_gu = 256
    else:
        raise KeyError(rule)
    g = QtipProjSpec("gate", 2048, 4096, n1=n1_gu, k_sparse=64, unbiased=True)
    u = QtipProjSpec("up", 2048, 4096, n1=n1_gu, k_sparse=64, unbiased=True)
    d = QtipProjSpec("down", 4096, 2048, n1=0, k_sparse=256, unbiased=False)
    return QtipPreset(f"qtip_{rule}_{mode}_{tile_mode}", g, u, d, mode=mode, tile_mode=tile_mode)


@torch.no_grad()
def encode_projection_qtip_ldlq(
    W: torch.Tensor,
    sp: QtipProjSpec,
    signs: torch.Tensor,
    device: torch.device,
    hessian_and_target: Callable[[int, int], Tuple[torch.Tensor, Optional[torch.Tensor]]],
    mode: str = "hyb",
    hyb_luts: Optional[Dict[int, torch.Tensor]] = None,
    tile_mode: str = "2d_16x16",
    damp: float = 0.10,
    alloc: str = "hnorm",
    scale_mode: str = "hunb",
    expert_batch: int = 8,
) -> Dict[str, torch.Tensor]:
    E, d_out, d_in = W.shape
    assert sp.n1 % 16 == 0 and sp.n2 % 16 == 0 and d_in % 16 == 0
    b1, b2 = sp.code_bytes_per_row()

    if tile_mode == "2d_16x16":
        c1_shape = (E, sp.n1 // 16, d_in // 16, 32)
        c2_shape = (E, sp.n2 // 16, d_in // 16, 64)
        g_blk = 16
    else:
        c1_shape = (E, sp.n1, b1)
        c2_shape = (E, sp.n2, b2)
        g_blk = 256

    out = {
        "map": torch.empty((E, d_out), dtype=torch.int16),
        "scale": torch.empty((E, d_out), dtype=torch.float16),
        "c1": torch.empty(c1_shape, dtype=torch.uint8),
        "c2": torch.empty(c2_shape, dtype=torch.uint8),
        "sp": torch.zeros((E, sp.k_sparse, 6), dtype=torch.uint8),
        "tile_mode": tile_mode,
        "mode": mode,
    }

    s_f = signs.to(device=device, dtype=torch.float32)
    mul1_tab = eval_mul1_all_states_torch(device) if mode == "mul1" else None
    hyb_pair_tabs = {k: eval_hyb_all_states_torch(hyb_luts[k]) for k in (1, 2)} if mode == "hyb" else None

    sig = err = hsig = herr = 0.0
    nb = d_in // g_blk

    for e0 in range(0, E, expert_batch):
        e1 = min(E, e0 + expert_batch)
        B = e1 - e0
        H, W_star = hessian_and_target(e0, e1)
        Wb_true = W[e0:e1].to(device=device, dtype=torch.float32)
        Wb = W_star.to(device=device, dtype=torch.float32) if W_star is not None else Wb_true.clone()
        H = H.to(device=device, dtype=torch.float32)

        if sp.k_sparse > 0:
            flat = Wb.reshape(B, -1)
            top = torch.topk(flat.abs(), sp.k_sparse, dim=1).indices
            vals = torch.gather(flat, 1, top)
            rows, cols = top // d_in, top % d_in
            flat.scatter_(1, top, 0.0)
            spb = torch.zeros((B, sp.k_sparse, 6), dtype=torch.uint8, device=device)
            spb[..., 0:2] = rows.to(torch.int16).view(B, sp.k_sparse, 1).view(torch.uint8).view(B, sp.k_sparse, 2)
            spb[..., 2:4] = cols.to(torch.int16).view(B, sp.k_sparse, 1).view(torch.uint8).view(B, sp.k_sparse, 2)
            spb[..., 4:6] = vals.to(torch.float16).view(B, sp.k_sparse, 1).view(torch.uint8).view(B, sp.k_sparse, 2)
            out["sp"][e0:e1] = spb.cpu()
            dense = flat.view(B, d_out, d_in)
        else:
            dense = Wb

        U = fwht_full(dense * s_f)
        Hr = rotate_hessian(H, s_f)
        lam = damp * torch.diagonal(Hr, dim1=1, dim2=2).mean(-1).clamp_min(1e-12)
        Hr = Hr + lam.view(B, 1, 1) * torch.eye(d_in, device=device).view(1, d_in, d_in)

        if alloc == "hnorm":
            score = torch.einsum("bri,bij,brj->br", U, Hr, U)
        else:
            score = U.pow(2).sum(-1)
        order = torch.argsort(score, dim=1)
        ks = torch.full((B, d_out), 2, dtype=torch.long, device=device)
        if sp.n1 > 0:
            ks.scatter_(1, order[:, : sp.n1], 1)

        rms_raw = U.pow(2).mean(-1).sqrt().clamp_min(1e-12)
        rms_rank = torch.argsort(torch.argsort(rms_raw, dim=1), dim=1)
        sort_key = ks * (d_out + 1) + rms_rank
        stored = torch.argsort(sort_key, dim=1)
        out["map"][e0:e1] = stored.to(torch.int16).cpu()

        Us = torch.gather(U, 1, stored.unsqueeze(-1).expand(B, d_out, d_in))
        rms = Us.pow(2).mean(-1).sqrt().clamp_min(1e-12)

        Uf = block_ldl_forward(Hr, g=g_blk)
        Uh = torch.zeros_like(Us)
        Eacc = torch.zeros_like(Us)
        groups = [(k, a, b) for k, a, b in ((1, 0, sp.n1), (2, sp.n1, d_out)) if b > a]

        if tile_mode == "2d_16x16":
            rw_groups: Dict[int, torch.Tensor] = {}
            for k, a, b in groups:
                nt = (b - a) // 16
                rms_t = rms[:, a:b].reshape(B * nt, 16)
                w16 = rms_t.pow(2) / rms_t.pow(2).mean(dim=1, keepdim=True).clamp_min(1e-24)
                V_step = 2 if mode == "hyb" else 1
                steps_per_row = 16 // V_step
                rw_groups[k] = w16.repeat_interleave(steps_per_row, dim=1).contiguous()

            for blk in range(nb):
                c0, c1 = blk * 16, (blk + 1) * 16
                v = Us[:, :, c0:c1].clone()
                if blk > 0:
                    v.add_(torch.bmm(Eacc[:, :, :c0], Uf[:, :c0, c0:c1]))
                xn = v / rms.unsqueeze(-1)

                for k, a, b in groups:
                    nt = (b - a) // 16
                    X_tile = xn[:, a:b, :].reshape(B * nt, 256).contiguous()
                    st = viterbi_qtip_tailbite_cuda(
                        X_tile,
                        k=k,
                        mode=mode,
                        hyb_lut_fp16=hyb_luts[k] if mode == "hyb" else None,
                        row_w=rw_groups[k],
                    )
                    step_bits = k * (2 if mode == "hyb" else 1)
                    packed = pack_tailbite_l16(st, step_bits=step_bits).view(B, nt, -1)
                    out[f"c{k}"][e0:e1, :, blk, :] = packed.cpu()

                    if mode == "hyb":
                        rec_tile = hyb_pair_tabs[k][st.long()].reshape(B, b - a, 16)
                    else:
                        rec_tile = mul1_tab[st.long()].reshape(B, b - a, 16)
                    Uh[:, a:b, c0:c1] = rec_tile * rms[:, a:b].unsqueeze(-1)

                Eacc[:, :, c0:c1] = Us[:, :, c0:c1] - Uh[:, :, c0:c1]
        else:
            for blk in range(nb):
                c0, c1 = blk * g_blk, (blk + 1) * g_blk
                v = Us[:, :, c0:c1].clone()
                if blk > 0:
                    v.add_(torch.bmm(Eacc[:, :, :c0], Uf[:, :c0, c0:c1]))
                xn = v / rms.unsqueeze(-1)
                for k, a, b in groups:
                    Xk = xn[:, a:b, :].reshape(B * (b - a), g_blk).contiguous()
                    st = viterbi_qtip_tailbite_cuda(
                        Xk,
                        k=k,
                        mode=mode,
                        hyb_lut_fp16=hyb_luts[k] if mode == "hyb" else None,
                    )
                    if mode == "hyb":
                        rec_seg = hyb_pair_tabs[k][st.long()].reshape(B, b - a, g_blk)
                    else:
                        rec_seg = mul1_tab[st.long()].reshape(B, b - a, g_blk)
                    Uh[:, a:b, c0:c1] = rec_seg * rms[:, a:b].unsqueeze(-1)
                Eacc[:, :, c0:c1] = Us[:, :, c0:c1] - Uh[:, :, c0:c1]

            Uf_strict = Uf - torch.eye(d_in, device=device).unsqueeze(0)
            V_full = (Us + torch.bmm(Eacc, Uf_strict)) / rms.unsqueeze(-1)
            for k, a, b in groups:
                X_full = V_full[:, a:b, :].reshape(B * (b - a), d_in).contiguous()
                st_f = viterbi_qtip_tailbite_cuda(
                    X_full,
                    k=k,
                    mode=mode,
                    hyb_lut_fp16=hyb_luts[k] if mode == "hyb" else None,
                    rows_per_launch=128,
                )
                step_bits = k * (2 if mode == "hyb" else 1)
                out[f"c{k}"][e0:e1] = pack_tailbite_l16(st_f, step_bits=step_bits).view(B, b - a, -1).cpu()
                if mode == "hyb":
                    rec_f = hyb_pair_tabs[k][st_f.long()].reshape(B, b - a, d_in)
                else:
                    rec_f = mul1_tab[st_f.long()].reshape(B, b - a, d_in)
                Uh[:, a:b, :] = rec_f * rms[:, a:b].unsqueeze(-1)

        HU = torch.bmm(Uh, Hr)
        num = (Us * HU).sum(-1)
        den = (Uh * HU).sum(-1).clamp_min(1e-30)
        s = num / den
        if scale_mode == "hunb" and sp.unbiased:
            HUs = torch.bmm(Us, Hr)
            beta = (s * (Uh * HUs).sum(-1)) / (Us * HUs).sum(-1).clamp_min(1e-30)
            s = s / beta.clamp_min(1e-6)
        Uh = Uh * s.unsqueeze(-1)
        out["scale"][e0:e1] = (rms * s).to(torch.float16).cpu()

        D_err = Us - Uh
        sig += float(Us.double().pow(2).sum())
        err += float(D_err.double().pow(2).sum())
        hsig += float(torch.einsum("bri,bij,brj->", Us, Hr, Us).double())
        herr += float(torch.einsum("bri,bij,brj->", D_err, Hr, D_err).double())

    out["_snr_dense_db"] = round(10.0 * math.log10(sig / max(err, 1e-30)), 3)
    out["_snr_h_db"] = round(10.0 * math.log10(hsig / max(herr, 1e-30)), 3)
    return out


@torch.no_grad()
def _slice_fields(fields: Dict[str, torch.Tensor], e0: int, e1: int) -> Dict[str, torch.Tensor]:
    """Expert rows e0:e1 of an encoded projection (tensor fields sliced on dim 0, the rest passed through)."""
    return {k: (v[e0:e1] if (torch.is_tensor(v) and k in ("map", "scale", "c1", "c2", "sp")) else v) for k, v in fields.items()}


DECODE_CHUNK = int(os.environ.get("R39_DECODE_CHUNK", "16"))


def decode_projection_qtip(
    fields: Dict[str, torch.Tensor],
    sp: QtipProjSpec,
    signs: torch.Tensor,
    device: torch.device,
    hyb_luts: Optional[Dict[int, torch.Tensor]] = None,
    include_sparse: bool = True,
) -> torch.Tensor:
    """[R3.9 fix 3, 2026-11-16] The shipped decode built every expert of the layer on the GPU at once: with 288 experts
    that is about 60-70 GiB of fp32/int64 temporaries per projection on a 32 GiB card (on Windows it spilled 90 GiB into
    shared memory and exhausted 128 GB of RAM). Experts are independent, so more than DECODE_CHUNK experts are now
    decoded chunk by chunk with the unchanged arithmetic and returned as a CPU fp32 tensor; callers index one expert
    at a time and move it to the device. At or below DECODE_CHUNK the behaviour is as shipped (GPU tensor)."""
    E_all = int(fields["map"].shape[0])
    if E_all <= DECODE_CHUNK:
        return _decode_projection_qtip_gpu(fields, sp, signs, device, hyb_luts=hyb_luts, include_sparse=include_sparse)
    out = torch.empty((E_all, sp.d_out, sp.d_in), dtype=torch.float32)
    for e0 in range(0, E_all, DECODE_CHUNK):
        e1 = min(E_all, e0 + DECODE_CHUNK)
        out[e0:e1] = _decode_projection_qtip_gpu(
            _slice_fields(fields, e0, e1), sp, signs, device, hyb_luts=hyb_luts, include_sparse=include_sparse
        ).cpu()
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def _decode_projection_qtip_gpu(
    fields: Dict[str, torch.Tensor],
    sp: QtipProjSpec,
    signs: torch.Tensor,
    device: torch.device,
    hyb_luts: Optional[Dict[int, torch.Tensor]] = None,
    include_sparse: bool = True,
) -> torch.Tensor:
    mode = fields["mode"]
    tile_mode = fields["tile_mode"]
    rmap = fields["map"].to(device).long() & 0xFFFF
    sc = fields["scale"].to(device).float()
    E, d_out = rmap.shape
    d_in = sp.d_in

    mul1_tab = eval_mul1_all_states_torch(device) if mode == "mul1" else None
    hyb_tabs = {k: eval_hyb_all_states_torch(hyb_luts[k]) for k in (1, 2)} if mode == "hyb" else None

    Us = torch.empty((E, d_out, d_in), dtype=torch.float32, device=device)
    for k, a, b in ((1, 0, sp.n1), (2, sp.n1, d_out)):
        if b <= a:
            continue
        V = 2 if mode == "hyb" else 1
        step_bits = k * V
        if tile_mode == "2d_16x16":
            nt = (b - a) // 16
            nb = d_in // 16
            packed = fields[f"c{k}"].to(device).reshape(E * nt * nb, 32 * k)
            M_tile = 256 // V
            st = unpack_tailbite_l16(packed, M=M_tile, step_bits=step_bits)
            vals = hyb_tabs[k][st.long()] if mode == "hyb" else mul1_tab[st.long()]
            tiles = vals.reshape(E, nt, nb, 16, 16).permute(0, 1, 3, 2, 4).reshape(E, b - a, d_in)
            Us[:, a:b, :] = tiles
        else:
            packed = fields[f"c{k}"].to(device).reshape(E * (b - a), (d_in * k) // 8)
            M_row = d_in // V
            st = unpack_tailbite_l16(packed, M=M_row, step_bits=step_bits)
            vals = hyb_tabs[k][st.long()] if mode == "hyb" else mul1_tab[st.long()]
            Us[:, a:b, :] = vals.reshape(E, b - a, d_in)

    Us.mul_(sc.unsqueeze(-1))
    U = torch.zeros_like(Us)
    U.scatter_(1, rmap.unsqueeze(-1).expand(E, d_out, d_in), Us)
    W_hat = fwht_full(U) * signs.to(device=device, dtype=torch.float32)

    if include_sparse and sp.k_sparse > 0:
        spb = fields["sp"].to(device)
        rows = spb[..., 0:2].contiguous().view(torch.int16).long().view(E, -1) & 0xFFFF
        cols = spb[..., 2:4].contiguous().view(torch.int16).long().view(E, -1) & 0xFFFF
        vals = spb[..., 4:6].contiguous().view(torch.float16).float().view(E, -1)
        W_hat.view(E, -1).scatter_add_(1, rows * d_in + cols, vals)
    return W_hat


@torch.no_grad()
def verify_nvrtc_tile16x16_gemv(
    fields: Dict[str, torch.Tensor],
    sp: QtipProjSpec,
    signs: torch.Tensor,
    hyb_luts: Dict[int, torch.Tensor],
    x_sample: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    E = min(fields["map"].shape[0], 8)
    P = min(x_sample.shape[0], 8)
    slots = torch.arange(P, device=device, dtype=torch.int32) % E
    x_sub = x_sample[:P].to(device).float().contiguous()
    x_rot = fwht_full(x_sub * signs.to(device=device, dtype=torch.float32)).contiguous()

    rmap_d = fields["map"][:E].to(device).contiguous()
    scale_d = fields["scale"][:E].to(device).contiguous()
    c1_d = fields["c1"][:E].to(device).contiguous()
    c2_d = fields["c2"][:E].to(device).contiguous()
    lut_stack = torch.stack([hyb_luts[1], hyb_luts[2]], dim=0).to(device=device, dtype=torch.float16).contiguous()
    lut_u32 = lut_stack.view(torch.int32)

    shared = (1024 + sp.d_in) * 4
    kern = _compile_nvrtc("qtip_tile16x16_hyb_gemv_kernel", _QTIP_L16_CUDA_SRC, shared_bytes=shared)
    y_kern = torch.empty((P, sp.d_out), dtype=torch.float32, device=device)
    warps = 8
    gx = min(((sp.d_out // 16) + warps - 1) // warps, 64)

    for _ in range(3):
        kern(
            grid=(gx, P, 1),
            block=(32 * warps, 1, 1),
            shared_mem=shared,
            args=[rmap_d, scale_d, c1_d, c2_d, slots, int(sp.n1), int(sp.n2), lut_u32, x_rot, 1, y_kern, int(sp.d_out), int(sp.d_in)],
        )
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(50):
        kern(
            grid=(gx, P, 1),
            block=(32 * warps, 1, 1),
            shared_mem=shared,
            args=[rmap_d, scale_d, c1_d, c2_d, slots, int(sp.n1), int(sp.n2), lut_u32, x_rot, 1, y_kern, int(sp.d_out), int(sp.d_in)],
        )
    ev1.record()
    torch.cuda.synchronize()
    us_per_launch = (ev0.elapsed_time(ev1) * 1000.0) / 50.0

    sub_f = {k: (v[:E] if torch.is_tensor(v) else v) for k, v in fields.items()}
    W_dense_ref = decode_projection_qtip(sub_f, sp, signs, device, hyb_luts=hyb_luts, include_sparse=False)
    y_ref = torch.bmm(x_sub.unsqueeze(1), W_dense_ref[slots.long()].transpose(1, 2)).squeeze(1)
    cos = float(F.cosine_similarity(y_kern.flatten(), y_ref.flatten(), dim=0))
    max_rel = float((y_kern - y_ref).abs().max() / y_ref.abs().max().clamp_min(1e-8))
    return {"gemv_cos_vs_ref": round(cos, 7), "gemv_max_rel_err": round(max_rel, 6), "gemv_8pair_us": round(us_per_launch, 2)}


# ================================================================================================================
# Calibration Hessians + Cross-Covariance Quantized-Gate/Up H_down Compensation + Held-Out Joint Tuning
# ================================================================================================================
class LayerCalHessiansR39:
    def __init__(self, cal: Dict[str, torch.Tensor], dev: torch.device, n0: float = 8192.0):
        keep = cal["pos"] >= 2
        self.x = cal["x"][keep].to(dev)
        self.idx = cal["idx"][keep].to(dev).long()
        self.w = cal["w"][keep].to(dev).float()
        self.seq = cal["seq"][keep].to(dev)
        self.dev = dev
        self.n0 = float(n0)

        H = torch.zeros((self.x.shape[1], self.x.shape[1]), dtype=torch.float32, device=dev)
        for s in range(0, self.x.shape[0], 32768):
            xb = self.x[s : s + 32768].float()
            H.addmm_(xb.T, xb)
        self.Hbar = H / max(int(self.x.shape[0]), 1)

    def tokens(self, e: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tk, sl = torch.where(self.idx == e)
        return tk, self.w[tk, sl].pow(2)

    @torch.no_grad()
    def gu_hess(self, e0: int, e1: int) -> Tuple[torch.Tensor, None]:
        out = []
        for e in range(e0, e1):
            tk, c = self.tokens(e)
            if tk.numel() == 0:
                out.append(self.Hbar.clone())
                continue
            xe = self.x[tk].float()
            cm = c.mean().clamp_min(1e-12)
            He = (xe * c.unsqueeze(-1)).T @ xe
            out.append((He + self.n0 * cm * self.Hbar) / (c.sum() + self.n0 * cm))
        return torch.stack(out, dim=0), None

    @torch.no_grad()
    def down_hess_and_target(
        self,
        e0: int,
        e1: int,
        Wg: torch.Tensor,
        Wu: torch.Tensor,
        Wd: torch.Tensor,
        Wg_hat: Optional[torch.Tensor] = None,
        Wu_hat: Optional[torch.Tensor] = None,
        cross_cov_compensate: bool = False,
        qz_damp: float = 0.15,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        H_list = []
        W_list = []
        d_inter = Wd.shape[2]
        eye = torch.eye(d_inter, device=self.dev, dtype=torch.float64)

        for e in range(e0, e1):
            tk, c = self.tokens(e)
            We_d = Wd[e].to(self.dev).float()
            if tk.numel() == 0:
                H_list.append(torch.eye(d_inter, device=self.dev))
                W_list.append(We_d)
                continue

            if cross_cov_compensate:
                tr_mask = (self.seq[tk] % 5) != 0
                if int(tr_mask.sum()) >= 32:
                    tk = tk[tr_mask]
                    c = c[tr_mask]

            xe = self.x[tk].float()
            cw = c.unsqueeze(-1)
            cm = c.mean().clamp_min(1e-12)
            Ge = Wg[e].to(self.dev).float()
            Ue = Wu[e].to(self.dev).float()
            z_ex = F.silu((xe @ Ge.T).clamp(max=LIM)) * (xe @ Ue.T).clamp(-LIM, LIM)

            if Wg_hat is None or Wu_hat is None:
                C_zz = (z_ex * cw).T @ z_ex
                diag_m = torch.diagonal(C_zz).mean() / max(float(c.sum()), 1e-12)
                prior = torch.eye(d_inter, device=self.dev) * diag_m
                H_list.append((C_zz + self.n0 * cm * prior) / (c.sum() + self.n0 * cm))
                W_list.append(We_d)
                continue

            Gh = Wg_hat[e].to(self.dev).float()
            Uh = Wu_hat[e].to(self.dev).float()
            z_qh = F.silu((xe @ Gh.T).clamp(max=LIM)) * (xe @ Uh.T).clamp(-LIM, LIM)

            C_qq = (z_qh * cw).T @ z_qh
            diag_m = torch.diagonal(C_qq).mean() / max(float(c.sum()), 1e-12)
            prior = torch.eye(d_inter, device=self.dev) * diag_m
            denom = c.sum() + self.n0 * cm
            H_qq = (C_qq + self.n0 * cm * prior) / denom
            H_list.append(H_qq)

            if cross_cov_compensate:
                C_eq = (z_ex * cw).T @ z_qh
                H_eq = (C_eq + self.n0 * cm * prior) / denom
                H_qq64 = H_qq.double()
                diff64 = (H_eq - H_qq).double()
                lam = float(qz_damp) * float(torch.diagonal(H_qq64).mean().clamp_min(1e-12))
                chol = torch.linalg.cholesky(H_qq64 + lam * eye)
                delta_m_T = torch.cholesky_solve(diff64.T, chol)
                W_star = We_d.double() + We_d.double() @ delta_m_T.T
                W_list.append(W_star.float())
            else:
                W_list.append(We_d)

        return torch.stack(H_list, dim=0), (torch.stack(W_list, dim=0) if cross_cov_compensate else None)


def tune_qtip_hyb_and_scales(
    preset: QtipPreset,
    fg: Dict[str, torch.Tensor],
    fu: Dict[str, torch.Tensor],
    fd: Dict[str, torch.Tensor],
    signs: Dict[str, torch.Tensor],
    hyb_luts: Dict[str, Dict[int, torch.Tensor]],
    Wg: torch.Tensor,
    Wu: torch.Tensor,
    Wd: torch.Tensor,
    ch: LayerCalHessiansR39,
    dev: torch.device,
    lut_iters: int = 35,
    scale_iters: int = 60,
) -> Dict[str, float]:
    E = Wg.shape[0]

    for p_key, fields, sp in (("g", fg, preset.gate), ("u", fu, preset.up), ("d", fd, preset.down)):
        lut_dict = hyb_luts[p_key]
        params = {k: lut_dict[k].float().clone().detach().requires_grad_(True) for k in (1, 2)}
        opt_lut = torch.optim.Adam(list(params.values()), lr=2.5e-3)
        n_sub = min(E, 64)
        # [R3.9 fix 3] only the n_sub experts used below go to the GPU (shipped: all E, 9.7 GiB per projection at 288)
        W_ref = {"g": Wg, "u": Wu, "d": Wd}[p_key][:n_sub].to(dev).float()
        s_f = signs[p_key].to(dev).float()

        W_sp = torch.zeros((n_sub, sp.d_out, sp.d_in), device=dev)
        if sp.k_sparse > 0:
            spb = fields["sp"][:n_sub].to(dev)
            r_idx = spb[..., 0:2].contiguous().view(torch.int16).long().view(n_sub, -1) & 0xFFFF
            c_idx = spb[..., 2:4].contiguous().view(torch.int16).long().view(n_sub, -1) & 0xFFFF
            vals = spb[..., 4:6].contiguous().view(torch.float16).float().view(n_sub, -1)
            W_sp.view(n_sub, -1).scatter_add_(1, r_idx * sp.d_in + c_idx, vals)
        U_tgt = fwht_full((W_ref[:n_sub] - W_sp) * s_f)
        rmap = fields["map"][:n_sub].to(dev).long() & 0xFFFF
        Us_tgt = torch.gather(U_tgt, 1, rmap.unsqueeze(-1).expand(n_sub, sp.d_out, sp.d_in))
        sc_fixed = fields["scale"][:n_sub].to(dev).float().unsqueeze(-1)

        st_groups = {}
        for k, a, b in ((1, 0, sp.n1), (2, sp.n1, sp.d_out)):
            if b <= a:
                continue
            nt, nb = (b - a) // 16, sp.d_in // 16
            packed = fields[f"c{k}"][:n_sub].to(dev).reshape(n_sub * nt * nb, 32 * k)
            st_groups[(k, a, b)] = unpack_tailbite_l16(packed, M=128, step_bits=2 * k).long()

        s_idx_all = torch.arange(65536, device=dev, dtype=torch.int64)
        h_all = (34038481 * s_idx_all + 76625530) & 0xFFFFFFFF
        b_all = (h_all >> 17) & 511
        sx_all = torch.where((h_all & 0x00008000) != 0, -1.0, 1.0)
        sy_all = torch.where((h_all & 0x80000000) != 0, -1.0, 1.0)

        with torch.enable_grad():
            for _ in range(lut_iters):
                opt_lut.zero_grad()
                loss_lut = torch.tensor(0.0, device=dev)
                for (k, a, b), st in st_groups.items():
                    nt, nb = (b - a) // 16, sp.d_in // 16
                    base = params[k][b_all]
                    pair_tab = torch.stack([base[:, 0] * sx_all, base[:, 1] * sy_all], dim=-1)
                    rec = pair_tab[st].reshape(n_sub, nt, nb, 16, 16).permute(0, 1, 3, 2, 4).reshape(n_sub, b - a, sp.d_in)
                    diff = rec * sc_fixed[:, a:b, :] - Us_tgt[:, a:b, :]
                    loss_lut = loss_lut + diff.pow(2).mean()
                loss_lut.backward()
                opt_lut.step()

        for k in (1, 2):
            lut_dict[k] = params[k].detach().to(torch.float16).contiguous()
        del W_ref, W_sp, U_tgt, Us_tgt, st_groups, sc_fixed, rmap, opt_lut
        torch.cuda.empty_cache()

    # [R3.9 fix 3] The shipped code decoded all E experts six times on the GPU before the loop (58 GiB at 288). Each
    # expert is now decoded when the loop reaches it; an expert's scales are only edited after its own decode, so the
    # values are the same.
    def _dec1(f, sp_, key, e, with_sparse):
        return _decode_projection_qtip_gpu(_slice_fields(f, e, e + 1), sp_, signs[key], dev, hyb_luts=hyb_luts[key], include_sparse=with_sparse)[0]

    tot_before = tot_after = tot_sig = 0.0
    n_tuned = 0
    for e in range(E):
        tk, c = ch.tokens(e)
        if int(tk.numel()) < 192:
            continue
        x = ch.x[tk].float()
        hold = (ch.seq[tk] % 5) == 0
        tr, va = ~hold, hold
        if not bool(tr.any()):
            continue

        Ge, Ue, De = Wg[e].to(dev).float(), Wu[e].to(dev).float(), Wd[e].to(dev).float()
        y_ex = (F.silu((x @ Ge.T).clamp(max=LIM)) * (x @ Ue.T).clamp(-LIM, LIM)) @ De.T

        Gq = _dec1(fg, preset.gate, "g", e, False)
        Gs = _dec1(fg, preset.gate, "g", e, True) - Gq
        Uq = _dec1(fu, preset.up, "u", e, False)
        Us_m = _dec1(fu, preset.up, "u", e, True) - Uq
        Dq = _dec1(fd, preset.down, "d", e, False)
        Ds = _dec1(fd, preset.down, "d", e, True) - Dq

        gq, gs = x @ Gq.T, x @ Gs.T
        uq, us = x @ Uq.T, x @ Us_m.T

        la_g = torch.zeros(Gq.shape[0], device=dev, requires_grad=True)
        la_u = torch.zeros(Uq.shape[0], device=dev, requires_grad=True)
        la_d = torch.zeros(Dq.shape[0], device=dev, requires_grad=True)
        opt_s = torch.optim.Adam([la_g, la_u, la_d], lr=3e-3)

        eval_mask = va if bool(va.any()) else tr
        sig_tr = float((c[tr].unsqueeze(-1) * y_ex[tr].pow(2)).sum()) + 1e-30
        sig_va = float((c[eval_mask].unsqueeze(-1) * y_ex[eval_mask].pow(2)).sum()) + 1e-30

        def _loss(m_bool: torch.Tensor, denom: float) -> torch.Tensor:
            g = torch.exp(la_g) * gq[m_bool] + gs[m_bool]
            u = torch.exp(la_u) * uq[m_bool] + us[m_bool]
            z = F.silu(g.clamp(max=LIM)) * u.clamp(-LIM, LIM)
            yh = torch.exp(la_d) * (z @ Dq.T) + z @ Ds.T
            reg = 1e-3 * (la_g.pow(2).mean() + la_u.pow(2).mean() + la_d.pow(2).mean())
            return (c[m_bool].unsqueeze(-1) * (yh - y_ex[m_bool]).pow(2)).sum() / denom + reg

        with torch.enable_grad():
            start_v = float(_loss(eval_mask, sig_va))
            best = (start_v, None)
            for it in range(scale_iters):
                opt_s.zero_grad()
                lo = _loss(tr, sig_tr)
                lo.backward()
                opt_s.step()
                if (it + 1) % 10 == 0:
                    cur_v = float(_loss(eval_mask, sig_va))
                    if cur_v < best[0]:
                        best = (cur_v, (la_g.detach().clone(), la_u.detach().clone(), la_d.detach().clone()))

        if best[1] is None:
            continue
        ag, au, ad = (torch.exp(t) for t in best[1])
        for f, a_vec in ((fg, ag), (fu, au), (fd, ad)):
            m_row = f["map"][e].long() & 0xFFFF
            f["scale"][e] = (f["scale"][e].float() * a_vec.cpu()[m_row]).to(torch.float16)
        n_tuned += 1
        tot_before += start_v * sig_va
        tot_after += best[0] * sig_va
        tot_sig += sig_va

    return {
        "experts_tuned": n_tuned,
        "val_rel_err_before": round(tot_before / max(tot_sig, 1e-30), 5),
        "val_rel_err_after": round(tot_after / max(tot_sig, 1e-30), 5),
    }


# ================================================================================================================
# Routed Output Scoring (Bit-Identical Metric Definitions to r33_layer_lab.py)
# ================================================================================================================
@torch.no_grad()
def forward_routed_from_weights(
    Wg: torch.Tensor,
    Wu: torch.Tensor,
    Wd: torch.Tensor,
    X: torch.Tensor,
    idx: torch.Tensor,
    w: torch.Tensor,
    dev: torch.device,
) -> torch.Tensor:
    N, D = X.shape
    Xd = X.to(dev).float()
    y = torch.zeros((N, D), dtype=torch.float32, device=dev)
    idx_d = idx.to(dev).long()
    w_d = w.to(dev).float()
    n_avail = Wg.shape[0]
    for e in torch.unique(idx_d).tolist():
        if e < 0 or e >= n_avail:
            continue
        tk, sl = torch.where(idx_d == e)
        xx = Xd[tk]
        Ge = Wg[e].to(dev).float()
        Ue = Wu[e].to(dev).float()
        De = Wd[e].to(dev).float()
        z = F.silu((xx @ Ge.T).clamp(max=LIM)) * (xx @ Ue.T).clamp(-LIM, LIM)
        y.index_add_(0, tk, (z @ De.T) * w_d[tk, sl].unsqueeze(-1))
    return y


def score_routed_output(y_exact: torch.Tensor, y_cand: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    a = y_exact[mask].double()
    b = y_cand[mask].double()
    sig = float(a.pow(2).sum())
    err = float((a - b).pow(2).sum())
    snr = 10.0 * math.log10(max(sig, 1e-30) / max(err, 1e-30))
    norm_ratio = float(b.norm() / a.norm().clamp_min(1e-30))
    cos = float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))
    return {
        "content_snr_db": round(snr, 3),
        "norm_ratio": round(norm_ratio, 4),
        "cos": round(cos, 5),
    }


# ================================================================================================================
# Synthetic Self-Test Mode (--selftest: Runs on Any CUDA GPU in ~8s Without Model Shards)
# ================================================================================================================
def run_selftest() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("--selftest requires a CUDA GPU for NVRTC kernel compilation.")
    dev = torch.device("cuda")
    print(f"[SELFTEST] Device: {torch.cuda.get_device_name(0)}", flush=True)

    t0 = time.perf_counter()
    hyb_base = train_hyb_codebooks_l16_v2(dev, cache_path=None, n_rows=256, Nw=256, iters=10)
    X_gauss = torch.randn((128, 256), device=dev, dtype=torch.float32)

    st_mul1 = viterbi_qtip_tailbite_cuda(X_gauss, k=2, mode="mul1")
    pk_mul1 = pack_tailbite_l16(st_mul1, step_bits=2)
    assert torch.equal(st_mul1, unpack_tailbite_l16(pk_mul1, M=256, step_bits=2)), "mul1 bitpack mismatch!"
    rec_mul1 = eval_mul1_all_states_torch(dev)[st_mul1.long()]
    snr_mul1 = 10.0 * math.log10(float(X_gauss.pow(2).sum()) / float((X_gauss - rec_mul1).pow(2).sum()))

    for k in (1, 2):
        st_h = viterbi_qtip_tailbite_cuda(X_gauss, k=k, mode="hyb", hyb_lut_fp16=hyb_base[k])
        pk_h = pack_tailbite_l16(st_h, step_bits=2 * k)
        assert torch.equal(st_h, unpack_tailbite_l16(pk_h, M=128, step_bits=2 * k)), f"HYB k={k} bitpack mismatch!"
        rec_h = eval_hyb_all_states_torch(hyb_base[k])[st_h.long()].reshape_as(X_gauss)
        snr_h = 10.0 * math.log10(float(X_gauss.pow(2).sum()) / float((X_gauss - rec_h).pow(2).sum()))
        print(f"  [GATE-1] HYB L=16 V=2 k={k}: bitpack=EXACT, Gaussian SNR={snr_h:.3f} dB", flush=True)
    print(f"  [GATE-1] mul1 L=16 V=1 k=2: bitpack=EXACT, Gaussian SNR={snr_mul1:.3f} dB ({time.perf_counter()-t0:.1f}s)", flush=True)

    E_s, Dm, Di = 4, 512, 256
    gen = torch.Generator(device=dev).manual_seed(99)
    eig = torch.logspace(1.2, -1.2, Dm, device=dev)
    X_cal = torch.randn((2048, Dm), generator=gen, device=dev) * eig.sqrt()
    X_ev = torch.randn((1024, Dm), generator=gen, device=dev) * eig.sqrt()
    idx_cal = torch.randint(0, E_s, (2048, 2), generator=gen, device=dev)
    w_cal = torch.softmax(torch.randn((2048, 2), generator=gen, device=dev), dim=-1)
    idx_ev = torch.randint(0, E_s, (1024, 2), generator=gen, device=dev)
    w_ev = torch.softmax(torch.randn((1024, 2), generator=gen, device=dev), dim=-1)

    Wg = (torch.randn((E_s, Di, Dm), generator=gen, device=dev) / math.sqrt(Dm)).bfloat16().cpu()
    Wu = (torch.randn((E_s, Di, Dm), generator=gen, device=dev) / math.sqrt(Dm)).bfloat16().cpu()
    Wd = (torch.randn((E_s, Dm, Di), generator=gen, device=dev) / math.sqrt(Di)).bfloat16().cpu()

    cal_dict = {
        "x": X_cal.bfloat16(),
        "idx": idx_cal,
        "w": w_cal,
        "pos": torch.full((2048,), 4, device=dev),
        "seq": torch.arange(2048, device=dev) // 64,
    }
    ch = LayerCalHessiansR39(cal_dict, dev, n0=256.0)
    mask_ev = torch.ones(1024, dtype=torch.bool, device=dev)
    y_ex = forward_routed_from_weights(Wg, Wu, Wd, X_ev, idx_ev, w_ev, dev)

    sp_g = QtipProjSpec("gate", Di, Dm, n1=0, k_sparse=16, unbiased=True)
    sp_u = QtipProjSpec("up", Di, Dm, n1=0, k_sparse=16, unbiased=True)
    sp_d = QtipProjSpec("down", Dm, Di, n1=0, k_sparse=32, unbiased=False)
    signs = {
        "g": generate_rademacher_signs(Dm, 101, dev),
        "u": generate_rademacher_signs(Dm, 102, dev),
        "d": generate_rademacher_signs(Di, 103, dev),
    }

    for tag, tmode, use_qz in (("HYB_1D_g256", "1d_row", False), ("HYB_2D_16x16", "2d_16x16", False), ("HYB_2D_16x16_QZ", "2d_16x16", True)):
        luts = {p: {1: hyb_base[1].clone(), 2: hyb_base[2].clone()} for p in ("g", "u", "d")}
        fg = encode_projection_qtip_ldlq(Wg, sp_g, signs["g"], dev, ch.gu_hess, mode="hyb", hyb_luts=luts["g"], tile_mode=tmode)
        fu = encode_projection_qtip_ldlq(Wu, sp_u, signs["u"], dev, ch.gu_hess, mode="hyb", hyb_luts=luts["u"], tile_mode=tmode)
        Gh = decode_projection_qtip(fg, sp_g, signs["g"], dev, hyb_luts=luts["g"])
        Uh = decode_projection_qtip(fu, sp_u, signs["u"], dev, hyb_luts=luts["u"])
        fd = encode_projection_qtip_ldlq(
            Wd,
            sp_d,
            signs["d"],
            dev,
            lambda a, b: ch.down_hess_and_target(a, b, Wg, Wu, Wd, Gh if use_qz else None, Uh if use_qz else None, cross_cov_compensate=use_qz),
            mode="hyb",
            hyb_luts=luts["d"],
            tile_mode=tmode,
        )
        Dh = decode_projection_qtip(fd, sp_d, signs["d"], dev, hyb_luts=luts["d"])
        sc = score_routed_output(y_ex, forward_routed_from_weights(Gh, Uh, Dh, X_ev, idx_ev, w_ev, dev), mask_ev)
        extra = ""
        if tmode == "2d_16x16":
            gv = verify_nvrtc_tile16x16_gemv(fg, sp_g, signs["g"], luts["g"], X_ev, dev)
            assert gv["gemv_cos_vs_ref"] > 0.99999, f"NVRTC GEMV mismatch: {gv}"
            extra = f" | NVRTC GEMV cos={gv['gemv_cos_vs_ref']}"
        print(f"  [GATE-2] {tag:18s}: routed SNR={sc['content_snr_db']:.3f} dB, cos={sc['cos']:.5f}{extra}", flush=True)

    print("[SELFTEST PASSED] All NVRTC kernels, bitpack bijections, 2D-BlockLDLQ, and QZ solvers verified.", flush=True)


# ================================================================================================================
# Main Layer 27 Benchmark Entrypoint (Matches r33_layer_lab.py Splits & Checkpoints)
# ================================================================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Round 3.9 Single-Layer L27 QTIP 2D-BlockLDLQ + Quantized-H_down Gate")
    ap.add_argument("--layer", type=int, default=27, help="MoE layer index (default: 27, R3.3 bottleneck reference)")
    ap.add_argument("--score-split", default="qb", choices=["eval", "qb", "cal"], help="Held-out evaluation token split")
    ap.add_argument("--cal-split", default="ct", choices=["cal", "ct", "mix"], help="Calibration split ('ct' matches 27_qb_calct)")
    ap.add_argument("--max-experts", type=int, default=288, help="Number of experts to encode (use 32 for rapid smoke gate)")
    ap.add_argument("--n0", type=float, default=8192.0, help="Empirical Bayes Hessian shrinkage pseudo-tokens (R3.3 A_best=8192)")
    ap.add_argument("--damp", type=float, default=0.10, help="Rotated Hessian diagonal damping (R3.3 A_best=0.10)")
    ap.add_argument("--qz-damp", type=float, default=0.15, help="Tikhonov damping for cross-covariance W_d* compensation")
    ap.add_argument(
        "--variants",
        default="TCQ33_A_best,QTIP_L16_mul1_1D,QTIP_L16_HYB_1D,QTIP_L16_HYB_2D16,QTIP_L16_HYB_2D16_QZ,QTIP_L16_HYB_2D16_QZ_Tune,QTIP_Sub70_GU125_2D16_QZ_Tune",
        help="Comma-separated ablation variants to evaluate",
    )
    ap.add_argument("--pkg-dir", default=DEFAULT_PKG, help="Path to claude_r32_apex_fold package directory")
    ap.add_argument("--range-dir", default=DEFAULT_RANGE, help="Path to GLM53-RANGE-EXPERTS checkpoint")
    ap.add_argument("--cap-dir", default=DEFAULT_CAP, help="Path to r33_capture directory containing L27_*.pt")
    ap.add_argument("--out-json", "--out", dest="out_json", default=DEFAULT_OUT, help="Output JSON report path")
    ap.add_argument("--baseline-snr-db", type=float, default=None,
                    help="[R3.9 fix 3] TCQ33_A_best content SNR measured in an earlier run of this script on the same "
                         "splits; lets the QTIP arms run without repeating the TCQ33 arm")
    ap.add_argument("--selftest", action="store_true", help="Run standalone synthetic CUDA/NVRTC self-test and exit")
    args = ap.parse_args()

    if args.selftest:
        run_selftest()
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required to run Layer 27 NVRTC validation.")
    dev = torch.device("cuda")

    if os.path.isdir(args.pkg_dir) and args.pkg_dir not in sys.path:
        sys.path.insert(0, args.pkg_dir)

    import glm53_fold_backbone as FB  # type: ignore
    import glm53_fold_tcq_format as TF  # type: ignore
    import glm53_fold_tcq_ldlq as LQ  # type: ignore

    l = args.layer
    E_run = min(int(args.max_experts), 288)
    t_start = time.perf_counter()

    ev_path = os.path.join(args.cap_dir, f"L{l:02d}_{args.score_split}.pt")
    ev = torch.load(ev_path, map_location="cpu")
    if args.score_split == "cal":
        ev = {k: (v[:2688] if torch.is_tensor(v) and v.dim() > 0 else v) for k, v in ev.items()}
    X_ev, idx_ev, w_ev, pos_ev = ev["x"], ev["idx"], ev["w"], ev["pos"]
    if E_run < 288:
        keep_tok = (pos_ev >= 2) & ((idx_ev < E_run).any(dim=-1))
        mask_ev = keep_tok.to(dev)
    else:
        mask_ev = (pos_ev >= 2).to(dev)

    parts = {"cal": ["cal"], "ct": ["ct"], "mix": ["cal", "ct"]}[args.cal_split]
    loaded_cal = [torch.load(os.path.join(args.cap_dir, f"L{l:02d}_{p_}.pt"), map_location="cpu") for p_ in parts]
    if len(loaded_cal) == 1:
        cal = loaded_cal[0]
    else:
        off = 0
        seqs = []
        for d_ in loaded_cal:
            seqs.append(d_["seq"] + off)
            off += int(d_["seq"].max()) + 1
        cal = {k: torch.cat([d_[k] for d_ in loaded_cal]) for k in ("x", "idx", "w", "pos")}
        cal["seq"] = torch.cat(seqs)

    ck = FB.RealGLM53Checkpoint(args.range_dir)
    Wg, Wu, Wd = ck.load_all_experts_torch(l, n_experts=E_run, dtype=torch.bfloat16)
    y_exact = forward_routed_from_weights(Wg, Wu, Wd, X_ev, idx_ev, w_ev, dev)
    ch = LayerCalHessiansR39(cal, dev, n0=args.n0)
    print(
        f"[L{l:02d}] Loaded {E_run} exact experts + {X_ev.shape[0]} '{args.score_split}' tokens "
        f"({int(mask_ev.sum())} scored content tokens) in {time.perf_counter() - t_start:.1f}s",
        flush=True,
    )

    hyb_cache = os.path.join(os.path.dirname(args.out_json), "qtip_hyb_L16_V2.pt")
    hyb_base = train_hyb_codebooks_l16_v2(dev, cache_path=hyb_cache)
    signs = {
        "g": generate_rademacher_signs(4096, seed=1000 * l + 11, device=dev),
        "u": generate_rademacher_signs(4096, seed=1000 * l + 12, device=dev),
        "d": generate_rademacher_signs(2048, seed=1000 * l + 13, device=dev),
    }

    Xg_test = torch.randn((128, 256), generator=torch.Generator(device=dev).manual_seed(777), device=dev)
    st_m = viterbi_qtip_tailbite_cuda(Xg_test, k=2, mode="mul1")
    snr_m = 10.0 * math.log10(float(Xg_test.pow(2).sum()) / float((Xg_test - eval_mul1_all_states_torch(dev)[st_m.long()]).pow(2).sum()))
    st_h2 = viterbi_qtip_tailbite_cuda(Xg_test, k=2, mode="hyb", hyb_lut_fp16=hyb_base[2])
    rec_h2 = eval_hyb_all_states_torch(hyb_base[2])[st_h2.long()].reshape_as(Xg_test)
    snr_h2 = 10.0 * math.log10(float(Xg_test.pow(2).sum()) / float((Xg_test - rec_h2).pow(2).sum()))
    print(f"[CODEBOOK GATE] Held-out N(0,1) k=2 SNR: QTIP L=16 mul1 = {snr_m:.3f} dB | QTIP L=16 HYB = {snr_h2:.3f} dB", flush=True)

    results: Dict[str, dict] = {}
    baseline_snr: Optional[float] = args.baseline_snr_db

    for vname in [v.strip() for v in args.variants.split(",") if v.strip()]:
        t_v = time.perf_counter()

        if vname == "TCQ33_A_best":
            tcq_tab_candidates = [
                os.path.join(os.path.dirname(args.out_json), "tcq_tables_L12.pt"),
                os.path.join(HERE, "out_r32", "tcq_tables_L12.pt"),
                os.path.join(os.path.dirname(args.pkg_dir), "build_run", "out_r32", "tcq_tables_L12.pt"),
            ]
            tab_path = _first_existing(tcq_tab_candidates, "")
            if tab_path and os.path.exists(tab_path):
                tabs = torch.load(tab_path, map_location=dev)
                tables = {k: tabs[k].float().half().float().to(dev) for k in (1, 2, 3)}
            else:
                import glm53_fold_tcq as TQ  # type: ignore

                tables = {k: TQ.train_codebook(k, device=dev).half().float() for k in (1, 2, 3)}

            preset_tcq = TF.make_preset("A")
            kw = {"scale_mode": "hunb", "damp": args.damp}
            fg = LQ.encode_projection_ldlq(Wg, preset_tcq.gate, signs["g"], tables, dev, lambda a, b: ch.gu_hess(a, b)[0], **kw)
            fu = LQ.encode_projection_ldlq(Wu, preset_tcq.up, signs["u"], tables, dev, lambda a, b: ch.gu_hess(a, b)[0], **kw)
            fd = LQ.encode_projection_ldlq(
                Wd,
                preset_tcq.down,
                signs["d"],
                tables,
                dev,
                lambda a, b: ch.down_hess_and_target(a, b, Wg, Wu, Wd, cross_cov_compensate=False)[0],
                **kw,
            )
            Gh = TF.decode_projection(fg, preset_tcq.gate, signs["g"], tables, dev)
            Uh = TF.decode_projection(fu, preset_tcq.up, signs["u"], tables, dev)
            Dh = TF.decode_projection(fd, preset_tcq.down, signs["d"], tables, dev)
            y_cand = forward_routed_from_weights(Gh, Uh, Dh, X_ev, idx_ev, w_ev, dev)
            rec = score_routed_output(y_exact, y_cand, mask_ev)
            rec.update(
                {
                    "bpw": round(preset_tcq.bpw(), 4),
                    "weight_snr_db": {p: round(float(f["_snr_dense_db"]), 3) for p, f in (("g", fg), ("u", fu), ("d", fd))},
                    "seconds": round(time.perf_counter() - t_v, 1),
                }
            )
            baseline_snr = rec["content_snr_db"]
            results[vname] = rec
            print(f"[RESULT] {vname:30s}: {json.dumps(rec)}", flush=True)
            del fg, fu, fd, Gh, Uh, Dh, y_cand
            torch.cuda.empty_cache()
            continue

        rule = "GU125" if "Sub70" in vname else "A"
        mode = "mul1" if "mul1" in vname else "hyb"
        tile_mode = "2d_16x16" if "2D16" in vname else "1d_row"
        use_qz = "_QZ" in vname
        do_tune = "_Tune" in vname

        preset_q = make_qtip_preset(rule=rule, mode=mode, tile_mode=tile_mode)
        luts = {p: {1: hyb_base[1].clone(), 2: hyb_base[2].clone()} for p in ("g", "u", "d")}

        fg = encode_projection_qtip_ldlq(
            Wg,
            preset_q.gate,
            signs["g"],
            dev,
            ch.gu_hess,
            mode=mode,
            hyb_luts=luts["g"] if mode == "hyb" else None,
            tile_mode=tile_mode,
            damp=args.damp,
            alloc="hnorm",
            scale_mode="hunb",
        )
        fu = encode_projection_qtip_ldlq(
            Wu,
            preset_q.up,
            signs["u"],
            dev,
            ch.gu_hess,
            mode=mode,
            hyb_luts=luts["u"] if mode == "hyb" else None,
            tile_mode=tile_mode,
            damp=args.damp,
            alloc="hnorm",
            scale_mode="hunb",
        )

        Gh = decode_projection_qtip(fg, preset_q.gate, signs["g"], dev, hyb_luts=luts["g"] if mode == "hyb" else None)
        Uh = decode_projection_qtip(fu, preset_q.up, signs["u"], dev, hyb_luts=luts["u"] if mode == "hyb" else None)

        fd = encode_projection_qtip_ldlq(
            Wd,
            preset_q.down,
            signs["d"],
            dev,
            lambda a, b: ch.down_hess_and_target(
                a,
                b,
                Wg,
                Wu,
                Wd,
                Wg_hat=Gh if use_qz else None,
                Wu_hat=Uh if use_qz else None,
                cross_cov_compensate=use_qz,
                qz_damp=args.qz_damp,
            ),
            mode=mode,
            hyb_luts=luts["d"] if mode == "hyb" else None,
            tile_mode=tile_mode,
            damp=args.damp,
            alloc="hnorm",
            scale_mode="hls",
        )

        tune_stats = None
        if do_tune and mode == "hyb" and tile_mode == "2d_16x16":
            tune_stats = tune_qtip_hyb_and_scales(preset_q, fg, fu, fd, signs, luts, Wg, Wu, Wd, ch, dev)
            Gh = decode_projection_qtip(fg, preset_q.gate, signs["g"], dev, hyb_luts=luts["g"])
            Uh = decode_projection_qtip(fu, preset_q.up, signs["u"], dev, hyb_luts=luts["u"])

        Dh = decode_projection_qtip(fd, preset_q.down, signs["d"], dev, hyb_luts=luts["d"] if mode == "hyb" else None)
        y_cand = forward_routed_from_weights(Gh, Uh, Dh, X_ev, idx_ev, w_ev, dev)
        rec = score_routed_output(y_exact, y_cand, mask_ev)
        if baseline_snr is not None:
            rec["delta_vs_tcq33_db"] = round(rec["content_snr_db"] - baseline_snr, 3)
        rec.update(
            {
                "bpw": round(preset_q.bpw(), 4),
                "weight_snr_db": {"g": fg["_snr_dense_db"], "u": fu["_snr_dense_db"], "d": fd["_snr_dense_db"]},
                "h_snr_db": {"g": fg["_snr_h_db"], "u": fu["_snr_h_db"], "d": fd["_snr_h_db"]},
                "seconds": round(time.perf_counter() - t_v, 1),
            }
        )
        if mode == "hyb" and tile_mode == "2d_16x16":
            # [R3.9 fix 2, 2026-11-16] X_ev is a CPU tensor and mask_ev a CUDA one: the shipped line raised
            # "indices should be either on cpu or on the same device" after the first 2D arm had been encoded.
            rec["nvrtc_gemv_check"] = verify_nvrtc_tile16x16_gemv(fg, preset_q.gate, signs["g"], luts["g"], X_ev[mask_ev.cpu()], dev)
        if tune_stats is not None:
            rec["joint_tune"] = tune_stats

        results[vname] = rec
        print(f"[RESULT] {vname:30s}: {json.dumps(rec)}", flush=True)
        del fg, fu, fd, Gh, Uh, Dh, y_cand
        gc.collect()
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    all_saved = json.load(open(args.out_json, encoding="utf-8")) if os.path.exists(args.out_json) else {}
    run_key = f"L{l:02d}_{args.score_split}_cal{args.cal_split}_E{E_run}"
    all_saved.setdefault(run_key, {}).update(results)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(all_saved, fh, indent=2)
    print(f"\n[DONE] Saved full Layer {l} ablation report to {args.out_json} (key='{run_key}')", flush=True)


if __name__ == "__main__":
    main()
