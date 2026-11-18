#!/usr/bin/env python3
"""
run_qtip_build_r39.py  [Round 3.9 Full 42-Layer + MTP Sequential Streaming QTIP HYB Builder]
============================================================================================
Drop-in Round 3.9 upgrade of `run_tcq_build_r33.py` for GLM-5.3-Flash (layers 3..44 trunk MoE +
layer 45 MTP) on a single RTX 5090 (32 GiB GDDR7 VRAM + 96 GB DDR5 RAM), emitting byte-locked
`layer_XX_qtip_hyb.pt` archives directly consumable by `r39_stage2b_global_kl_trainer.py`.

Core Upgrades Over `run_tcq_build_r33.py`:
  1. Register-Resident `L=16, V=2` Collar Tail-Biting Viterbi + Bijective `HYB-mul1` (`2 KiB` LUTs):
       Replaces `TCQ33`'s `L=12` (`4096`-state) free-start LUT trellis (`1,028 B` / `516 B` rows)
       with 65,536-state 2-pass collar tail-biting (`0` start bits: exact `1,024 B` / `512 B` rows,
       saving `32,768 B`/expert = `0.40 GB` free) and `2 KiB` `half2[512]` differentiable LUTs.
  2. QTIP `16 x 16` 2D-BlockLDLQ (`g = 16` Cholesky Column Feedback + RMS-Sorted 16-Row Tiles):
       Replaces 1D `g = 256` BlockLDLQ with `g = 16` forward block-LDL feedback (16x tighter error
       bound) across 256-weight `[16, 16]` tiles weighted by normalized per-row `rms^2`, verified on
       every layer against the NVRTC `qtip_tile16x16_hyb_gemv_kernel` (`cos > 0.99999`).
  3. Cross-Covariance Quantized-`gate/up` `H_down` Compensation (`W_d*`):
       Decodes newly quantized `(Wg_hat, Wu_hat)` per layer, evaluates quantized SwiGLU activations
       `z_hat`, and solves `W_d* = W_d + W_d (H_{z, z_hat} - H_{z_hat, z_hat}) (H_{z_hat, z_hat} + lam I)^{-1}`
       in FP64 Cholesky before encoding `down_proj` under `H_{z_hat, z_hat}` (fixing R3.3's exact-FP8
       `H_down` mismatch).
  4. Native `--prune-spec` + Sub-70 GB `GU125` Support:
       Supports unpruned `Rule A` (`86.82 GB`), `P5` (`83.10 GB`), Round 3.8 water-filling `~74.62 GB`
       (`--rule A --prune-spec ...`), and Sub-70 GB `68.90 GB` (`--rule GU125 --prune-spec ...`, setting
       `n1 = 256` `k=1` tail rows on `gate/up` only while keeping `down` 100% at `k=2`), skipping
       pruned experts during encoding while keeping `evx` on the full 288-expert FP8 teacher.
  5. Crash-Safe Resumable Streams (`_r39_ckpt`), Belady Shard Eviction & MTP Acceptance Gate:
       Preserves `run_tcq_build_r33.py`'s 4-stream (`cal`, `evx`, `evn`, `evo`) tracking, atomic
       `.tmp` + `os.fsync` stream checkpoints (`--stop-after L`, `--resume`), Belady source-shard
       eviction, early-abort SNR gate, MTP Layer 45 draft-acceptance scoring, and saves exact
       Top-128+Tail `qbench` teacher targets for immediate Stage-2B Global-KL co-adaptation.

Typical Usage on Your RTX 5090 Workstation:
  # 1) Standalone GPU pipeline self-test (~10s: tests 4-stream loop, 2D-LDLQ, QZ, resume & Stage-2B schema):
  python run_qtip_build_r39.py --selftest

  # 2) Build Sub-0.23 KL / 74.62 GB Round 3.9 checkpoint (with R3.8 non-uniform prune mask):
  python run_qtip_build_r39.py \
      --dst C:\models\GLM53-APEX-FOLD-R39-HYB \
      --old C:\models\GLM53-APEX-FOLD-TCQ33 \
      --src C:\models\GLM-5.3-Flash-UNCENSORED-FP8 \
      --rule A --prune-spec out_r32\r38_masks\hyb_W16_75GB.json

  # 3) Build Sub-70 GB (68.90 GB) Round 3.9 checkpoint (GU125 + 20.1% waterfill prune mask):
  python run_qtip_build_r39.py \
      --dst C:\models\GLM53-APEX-FOLD-R39-SUB70 \
      --old C:\models\GLM53-APEX-FOLD-TCQ33 \
      --src C:\models\GLM-5.3-Flash-UNCENSORED-FP8 \
      --rule GU125 --prune-spec out_r32\r38_masks\hyb_W16_69GB.json
"""

from __future__ import annotations

import argparse
import atexit
import gc
import gzip
import json
import math
import os
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.dont_write_bytecode = True
LIM = 10.0
LM = "model.language_model."
KEEP_SHARDS = {"model-00001-of-00062.safetensors", "model-00062-of-00062.safetensors"}
SHARED_FILES = ("exact_prefix_cache_prefix2.pt", "exact_prefix_cache_prefix10.pt")


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
DEFAULT_SRC = _first_existing(
    [r"C:\models\GLM-5.3-Flash-UNCENSORED-FP8", "/mnt/c/models/GLM-5.3-Flash-UNCENSORED-FP8"],
    r"C:\models\GLM-5.3-Flash-UNCENSORED-FP8",
)
DEFAULT_BB = _first_existing(
    [
        r"C:\models\GLM53-backbone-r34h",
        "/mnt/c/models/GLM53-backbone-r34h",
        r"C:\models\GLM53-backbone",
        "/mnt/c/models/GLM53-backbone",
    ],
    r"C:\models\GLM53-backbone",
)
DEFAULT_OLD = _first_existing(
    [r"C:\models\GLM53-APEX-FOLD-TCQ33", "/mnt/c/models/GLM53-APEX-FOLD-TCQ33"],
    r"C:\models\GLM53-APEX-FOLD-TCQ33",
)
DEFAULT_C4 = _first_existing(
    [
        r"C:\models\calib_text\en\c4-validation.00000-of-00008.json.gz",
        "/mnt/c/models/calib_text/en/c4-validation.00000-of-00008.json.gz",
    ],
    r"C:\models\calib_text\en\c4-validation.00000-of-00008.json.gz",
)
DEFAULT_CT = _first_existing(
    [r"C:\models\calib_text\exl3\cal_trace.json", "/mnt/c/models/calib_text/exl3/cal_trace.json"],
    r"C:\models\calib_text\exl3\cal_trace.json",
)
DEFAULT_QB = _first_existing(
    [r"C:\models\calib_text\exl3\qbench_prompts.json", "/mnt/c/models/calib_text/exl3/qbench_prompts.json"],
    r"C:\models\calib_text\exl3\qbench_prompts.json",
)


def log(msg: str, fh=None) -> None:
    line = f"[r39-build {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def db(sig: float, err: float) -> float:
    return 10.0 * math.log10(max(sig, 1e-30) / max(err, 1e-30))


def qtip_layer_path(d: str, l: int) -> str:
    return os.path.join(d, f"layer_{l:02d}_qtip_hyb.pt")


def tcq_layer_path(d: str, l: int) -> str:
    return os.path.join(d, f"layer_{l:02d}_apex_fold.npz")


# ================================================================================================================
# 1. Orthonormal FWHT, Rademacher Signs, Forward Block-LDL & NVRTC QTIP L=16 Kernels
# ================================================================================================================
_HAD_CACHE: Dict[Tuple[int, str], torch.Tensor] = {}
_NVRTC_CACHE: Dict[str, object] = {}


def _hadamard(n: int, device: torch.device) -> torch.Tensor:
    key = (n, str(device))
    if key not in _HAD_CACHE:
        h = np.array([[1.0]], dtype=np.float32)
        while h.shape[0] < n:
            h = np.block([[h, h], [h, -h]])
        h *= 1.0 / math.sqrt(float(n))
        _HAD_CACHE[key] = torch.from_numpy(h).to(device=device, dtype=torch.float32).contiguous()
    return _HAD_CACHE[key]


def fwht_full(x: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape
    d_in = orig_shape[-1]
    assert d_in > 0 and (d_in & (d_in - 1)) == 0
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
    B, n, _ = H.shape
    nb = n // g
    J = torch.arange(n - 1, -1, -1, device=H.device)
    Hr = H[:, J][:, :, J].double()
    L = torch.linalg.cholesky(Hr)
    Lb = L.view(B, nb, g, nb, g)
    diag = torch.stack([Lb[:, b, :, b, :] for b in range(nb)], dim=1)
    eye = torch.eye(g, device=H.device, dtype=torch.float64).expand(B, nb, g, g)
    dinv = torch.linalg.solve_triangular(diag, eye, upper=False)
    Lu = torch.empty_like(L).view(B, nb, g, nb, g)
    for c in range(nb):
        Lu[:, :, :, c, :] = torch.einsum("bpij,bjk->bpik", Lb[:, :, :, c, :], dinv[:, c])
    Lu = Lu.view(B, n, n)
    return Lu[:, J][:, :, J].float()


def _compile_nvrtc(name: str, src: str, shared_bytes: int = 0):
    if name not in _NVRTC_CACHE:
        # [R3.9 fix 1, 2026-11-16] 1,024-thread blocks allow at most 64 registers per thread; see the L27 lab.
        kern = torch.cuda._compile_kernel(src, name, nvcc_options=[f"--maxrregcount={int(os.environ.get('R39_MAXRREG', '64'))}"])
        if shared_bytes >= 48 * 1024:
            kern.set_shared_memory_config(shared_bytes)
        _NVRTC_CACHE[name] = kern
    return _NVRTC_CACHE[name]


_QTIP_R39_CUDA_SRC = r"""
#include <cuda_fp16.h>

__device__ __forceinline__ float2 qtip_hyb_mul1_pair(unsigned int s16, const unsigned int* __restrict__ lut_u32) {
    unsigned int h = 34038481u * s16 + 76625530u;
    unsigned int raw = lut_u32[(h >> 17) & 511u];
    unsigned int signed_bits = raw ^ (h & 0x80008000u);
    return __half22float2(*reinterpret_cast<const __half2*>(&signed_bits));
}

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

extern "C" __global__ void qtip_viterbi_hyb_k2_kernel(
    const float* __restrict__ X,
    const float* __restrict__ row_w,
    const unsigned int* __restrict__ lut_g,
    unsigned char* __restrict__ bp,
    const int* __restrict__ start_state,
    const int* __restrict__ end_state,
    int* __restrict__ states_out,
    int M, int has_row_w)
{
    __shared__ float sm_best[4096];
    __shared__ unsigned int s_lut[512];
    __shared__ int sm_arg[1024];

    const int r = blockIdx.x;
    const int tid = threadIdx.x;
    const float* xr = X + (size_t)r * (2 * M);
    const float* rw_ptr = has_row_w ? (row_w + (size_t)r * M) : nullptr;
    const size_t bp_row = (size_t)r * (size_t)M * 4096ULL;

    if (tid < 512) s_lut[tid] = lut_g[tid];
    __syncthreads();

    const int s_in = start_state[r];
    __half2 reg_c[4][8];

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
            float best = f0.x; int arg = 0;
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
        float loc_best = 3.4e38f; int loc_arg = 0;
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

extern "C" __global__ void qtip_viterbi_hyb_k1_kernel(
    const float* __restrict__ X,
    const float* __restrict__ row_w,
    const unsigned int* __restrict__ lut_g,
    unsigned char* __restrict__ bp,
    const int* __restrict__ start_state,
    const int* __restrict__ end_state,
    int* __restrict__ states_out,
    int M, int has_row_w)
{
    extern __shared__ float sm_dyn[];
    float* sm_best = sm_dyn;
    unsigned int* s_lut = reinterpret_cast<unsigned int*>(sm_dyn + 16384);
    int* sm_arg = reinterpret_cast<int*>(s_lut + 512);

    const int r = blockIdx.x;
    const int tid = threadIdx.x;
    const float* xr = X + (size_t)r * (2 * M);
    const float* rw_ptr = has_row_w ? (row_w + (size_t)r * M) : nullptr;
    const size_t bp_row = (size_t)r * (size_t)M * 16384ULL;

    if (tid < 512) s_lut[tid] = lut_g[tid];
    __syncthreads();

    const int s_in = start_state[r];
    __half2 reg_c[16][2];

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

        const float x0 = xr[2 * m];
        const float x1 = xr[2 * m + 1];
        const float rw = has_row_w ? rw_ptr[m] : 1.0f;
        const float s_norm = (m >= 8) ? sm_best[0] : 0.0f;

        #pragma unroll
        for (int u = 0; u < 16; ++u) {
            unsigned int q = (unsigned int)(tid + 1024 * u);
            #pragma unroll
            for (int p = 0; p < 2; ++p) {
                unsigned int s0 = ((unsigned int)(2 * p) << 14) | q;
                unsigned int s1 = ((unsigned int)(2 * p + 1) << 14) | q;
                float2 w0 = qtip_hyb_mul1_pair(s0, s_lut);
                float2 w1 = qtip_hyb_mul1_pair(s1, s_lut);
                float d0 = (x0 - w0.x) * (x0 - w0.x) + (x1 - w0.y) * (x1 - w0.y);
                float d1 = (x0 - w1.x) * (x0 - w1.x) + (x1 - w1.y) * (x1 - w1.y);
                float n0 = fminf(fmaxf(sm_best[s0 >> 2] + rw * d0 - s_norm, -8192.0f), 32768.0f);
                float n1 = fminf(fmaxf(sm_best[s1 >> 2] + rw * d1 - s_norm, -8192.0f), 32768.0f);
                reg_c[u][p] = __floats2half2_rn(n0, n1);
            }
        }
        __syncthreads();
    }

    int s_end = end_state[r];
    if (s_end < 0) {
        float loc_best = 3.4e38f; int loc_arg = 0;
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

extern "C" __global__ void qtip_tile16x16_hyb_gemv_kernel(
    const unsigned short* __restrict__ rmap_all,
    const __half* __restrict__ scale_all,
    const unsigned char* __restrict__ c1_all,
    const unsigned char* __restrict__ c2_all,
    const int* __restrict__ slots,
    int n1, int n2,
    const unsigned int* __restrict__ hyb_lut_g,
    const float* __restrict__ x_g,
    int x_per_pair,
    float* __restrict__ y,
    int d_out, int T)
{
    extern __shared__ unsigned int smem_u32[];
    unsigned int* s_lut = smem_u32;
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
    const int r_local = lane >> 1;
    const int half_col = lane & 1;
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
# 2. QTIP HYB 2D-BlockLDLQ Encoder, Bitpacking & Fast In-Build Layer Runtime
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

    def expert_bytes(self) -> int:
        return 4 * self.d_out + self.n1 * ((self.d_in * 1) // 8) + self.n2 * ((self.d_in * 2) // 8) + 6 * self.k_sparse


@dataclass(frozen=True)
class QtipPreset:
    name: str
    gate: QtipProjSpec
    up: QtipProjSpec
    down: QtipProjSpec

    @property
    def expert_row_bytes(self) -> int:
        return self.gate.expert_bytes() + self.up.expert_bytes() + self.down.expert_bytes()

    def layer_bytes(self, n_experts: int = 288) -> int:
        hdr = (self.gate.d_in + self.up.d_in + self.down.d_in) + (3 * 2 * 2048)
        return hdr + n_experts * self.expert_row_bytes

    def bpw(self, n_experts: int = 288) -> float:
        tot_w = n_experts * sum(sp.d_out * sp.d_in for sp in (self.gate, self.up, self.down))
        return (self.layer_bytes(n_experts) * 8.0) / float(tot_w)


def make_qtip_preset(rule: str) -> QtipPreset:
    if rule == "A":
        n1_gu = 0
    elif rule == "GU125":
        n1_gu = 256
    else:
        raise ValueError(f"Unsupported QTIP rule '{rule}' (use 'A' or 'GU125')")
    g = QtipProjSpec("gate", 2048, 4096, n1=n1_gu, k_sparse=64, unbiased=True)
    u = QtipProjSpec("up", 2048, 4096, n1=n1_gu, k_sparse=64, unbiased=True)
    d = QtipProjSpec("down", 4096, 2048, n1=0, k_sparse=256, unbiased=False)
    return QtipPreset(f"qtip_hyb_{rule}_2d16", g, u, d)


@torch.no_grad()
def eval_hyb_all_states_torch(lut_fp16: torch.Tensor) -> torch.Tensor:
    dev = lut_fp16.device
    s = torch.arange(65536, device=dev, dtype=torch.int64)
    h = (34038481 * s + 76625530) & 0xFFFFFFFF
    idx = (h >> 17) & 511
    sx = torch.where((h & 0x00008000) != 0, -1.0, 1.0)
    sy = torch.where((h & 0x80000000) != 0, -1.0, 1.0)
    base = lut_fp16.float()[idx]
    return torch.stack([base[:, 0] * sx, base[:, 1] * sy], dim=-1)


@torch.no_grad()
def pack_tailbite_l16(states: torch.Tensor, step_bits: int) -> torch.Tensor:
    R, M = states.shape
    total_bits = M * step_bits
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
    R, rb = packed.shape
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
def viterbi_hyb_tailbite_cuda(
    X: torch.Tensor,
    hyb_lut_fp16: torch.Tensor,
    k: int,
    row_w: Optional[torch.Tensor] = None,
    rows_per_launch: int = 512,
    collar_steps: int = 16,
) -> torch.Tensor:
    R, Nw = X.shape
    M = Nw // 2
    w_pad = min(collar_steps, M // 2)
    lut_u32 = hyb_lut_fp16.to(device=X.device, dtype=torch.float16).contiguous().view(torch.int32)

    has_rw = 1 if row_w is not None else 0
    X_collar = torch.cat([X[:, -2 * w_pad :], X[:, : 2 * w_pad]], dim=1).contiguous()
    rw_collar = torch.cat([row_w[:, -w_pad:], row_w[:, :w_pad]], dim=1).contiguous() if has_rw else X.new_empty(0)
    rw_full = row_w.contiguous() if has_rw else X.new_empty(0)

    states_out = torch.empty((R, M), dtype=torch.int32, device=X.device)
    free_pin = torch.full((rows_per_launch,), -1, dtype=torch.int32, device=X.device)

    if k == 2:
        NG = 4096
        smem = 0
        kern = _compile_nvrtc("qtip_viterbi_hyb_k2_kernel", _QTIP_R39_CUDA_SRC)
    else:
        NG = 16384
        smem = 72 * 1024
        kern = _compile_nvrtc("qtip_viterbi_hyb_k1_kernel", _QTIP_R39_CUDA_SRC, shared_bytes=smem)

    bp = torch.empty((rows_per_launch, M, NG), dtype=torch.uint8, device=X.device)
    st_collar = torch.empty((rows_per_launch, 2 * w_pad), dtype=torch.int32, device=X.device)

    for r0 in range(0, R, rows_per_launch):
        r1 = min(R, r0 + rows_per_launch)
        B = r1 - r0
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
            ],
        )
    return states_out


@torch.no_grad()
def train_or_load_hyb_codebooks(device: torch.device, cache_path: str, seed: int = 20261116) -> Dict[int, torch.Tensor]:
    if os.path.exists(cache_path):
        loaded = torch.load(cache_path, map_location=device)
        return {1: loaded[1].to(device=device, dtype=torch.float16), 2: loaded[2].to(device=device, dtype=torch.float16)}

    gen_cpu = torch.Generator(device="cpu").manual_seed(seed)
    q32 = 0.5 + 0.5 * (torch.arange(32, dtype=torch.float64) + 0.5) / 32.0
    q16 = 0.5 + 0.5 * (torch.arange(16, dtype=torch.float64) + 0.5) / 16.0
    v32 = math.sqrt(2.0) * torch.erfinv(2.0 * q32 - 1.0)
    v16 = math.sqrt(2.0) * torch.erfinv(2.0 * q16 - 1.0)
    gx, gy = torch.meshgrid(v32, v16, indexing="ij")
    init_grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1).float()
    lut_init = init_grid[torch.randperm(512, generator=gen_cpu)].to(device=device, dtype=torch.float16).contiguous()

    gen_gpu = torch.Generator(device=device).manual_seed(seed + 7)
    X = torch.randn((512, 512), generator=gen_gpu, device=device, dtype=torch.float32)

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
        for _ in range(12):
            st = viterbi_hyb_tailbite_cuda(X, lut, k=k)
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

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    torch.save({1: out[1].cpu(), 2: out[2].cpu()}, cache_path)
    return out


@torch.no_grad()
def encode_projection_qtip_2d_ldlq(
    W: torch.Tensor,
    sp: QtipProjSpec,
    signs: torch.Tensor,
    hyb_luts: Dict[int, torch.Tensor],
    device: torch.device,
    hessian_and_target: Optional[Callable[[int, int], Tuple[torch.Tensor, Optional[torch.Tensor]]]] = None,
    damp: float = 0.10,
    scale_mode: str = "hunb",
    expert_batch: int = 8,
) -> Dict[str, torch.Tensor]:
    """Encodes `W [E_kept, d_out, d_in]` with 16x16 2D-BlockLDLQ (`g=16`) or data-free (`H = I` for MTP L45)."""
    E, d_out, d_in = W.shape
    assert sp.n1 % 16 == 0 and sp.n2 % 16 == 0 and d_in % 16 == 0
    nb = d_in // 16

    out = {
        "map": torch.empty((E, d_out), dtype=torch.int16),
        "scale": torch.empty((E, d_out), dtype=torch.float16),
        "c1": torch.empty((E, sp.n1 // 16, nb, 32), dtype=torch.uint8),
        "c2": torch.empty((E, sp.n2 // 16, nb, 64), dtype=torch.uint8),
        "sp": torch.zeros((E, sp.k_sparse, 6), dtype=torch.uint8),
    }

    s_f = signs.to(device=device, dtype=torch.float32)
    pair_tabs = {k: eval_hyb_all_states_torch(hyb_luts[k]) for k in (1, 2)}
    sig = err = 0.0

    for e0 in range(0, E, expert_batch):
        e1 = min(E, e0 + expert_batch)
        B = e1 - e0
        Wb_orig = W[e0:e1].to(device=device, dtype=torch.float32)
        if hessian_and_target is not None:
            H, W_star = hessian_and_target(e0, e1)
            H = H.to(device=device, dtype=torch.float32)
            Wb = W_star.to(device=device, dtype=torch.float32) if W_star is not None else Wb_orig
            datafree = False
        else:
            H = None
            Wb = Wb_orig
            datafree = True

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
        if not datafree:
            assert H is not None
            Hr = rotate_hessian(H, s_f)
            lam = damp * torch.diagonal(Hr, dim1=1, dim2=2).mean(-1).clamp_min(1e-12)
            Hr = Hr + lam.view(B, 1, 1) * torch.eye(d_in, device=device).view(1, d_in, d_in)
            score = torch.einsum("bri,bij,brj->br", U, Hr, U)
        else:
            Hr = None
            score = U.pow(2).sum(-1)

        order = torch.argsort(score, dim=1)
        ks = torch.full((B, d_out), 2, dtype=torch.long, device=device)
        if sp.n1 > 0:
            ks.scatter_(1, order[:, : sp.n1], 1)

        rms_raw = U.pow(2).mean(-1).sqrt().clamp_min(1e-12)
        rms_rank = torch.argsort(torch.argsort(rms_raw, dim=1), dim=1)
        stored = torch.argsort(ks * (d_out + 1) + rms_rank, dim=1)
        out["map"][e0:e1] = stored.to(torch.int16).cpu()

        Us = torch.gather(U, 1, stored.unsqueeze(-1).expand(B, d_out, d_in))
        rms = Us.pow(2).mean(-1).sqrt().clamp_min(1e-12)

        Uf = block_ldl_forward(Hr, g=16) if Hr is not None else None
        Uh = torch.zeros_like(Us)
        Eacc = torch.zeros_like(Us) if Uf is not None else None
        groups = [(k, a, b) for k, a, b in ((1, 0, sp.n1), (2, sp.n1, d_out)) if b > a]

        rw_groups: Dict[int, torch.Tensor] = {}
        for k, a, b in groups:
            nt = (b - a) // 16
            rms_t = rms[:, a:b].reshape(B * nt, 16)
            w16 = rms_t.pow(2) / rms_t.pow(2).mean(dim=1, keepdim=True).clamp_min(1e-24)
            rw_groups[k] = w16.repeat_interleave(8, dim=1).contiguous()

        for blk in range(nb):
            c0, c1 = blk * 16, (blk + 1) * 16
            v = Us[:, :, c0:c1].clone()
            if Uf is not None and blk > 0:
                v.add_(torch.bmm(Eacc[:, :, :c0], Uf[:, :c0, c0:c1]))
            xn = v / rms.unsqueeze(-1)

            for k, a, b in groups:
                nt = (b - a) // 16
                X_tile = xn[:, a:b, :].reshape(B * nt, 256).contiguous()
                st = viterbi_hyb_tailbite_cuda(X_tile, hyb_luts[k], k=k, row_w=rw_groups[k])
                packed = pack_tailbite_l16(st, step_bits=2 * k).view(B, nt, -1)
                out[f"c{k}"][e0:e1, :, blk, :] = packed.cpu()
                rec_tile = pair_tabs[k][st.long()].reshape(B, b - a, 16)
                Uh[:, a:b, c0:c1] = rec_tile * rms[:, a:b].unsqueeze(-1)

            if Eacc is not None:
                Eacc[:, :, c0:c1] = Us[:, :, c0:c1] - Uh[:, :, c0:c1]

        if Hr is not None:
            HU = torch.bmm(Uh, Hr)
            s = (Us * HU).sum(-1) / (Uh * HU).sum(-1).clamp_min(1e-30)
            if scale_mode == "hunb" and sp.unbiased:
                HUs = torch.bmm(Us, Hr)
                beta = (s * (Uh * HUs).sum(-1)) / (Us * HUs).sum(-1).clamp_min(1e-30)
                s = s / beta.clamp_min(1e-6)
        else:
            s = (Us * Uh).sum(-1) / Uh.pow(2).sum(-1).clamp_min(1e-30)
            if sp.unbiased:
                beta = (s * (Us * Uh).sum(-1)) / Us.pow(2).sum(-1).clamp_min(1e-30)
                s = s / beta.clamp_min(1e-6)

        Uh = Uh * s.unsqueeze(-1)
        out["scale"][e0:e1] = (rms * s).to(torch.float16).cpu()
        sig += float(Us.double().pow(2).sum())
        err += float((Us - Uh).double().pow(2).sum())

    out["_snr_dense_db"] = round(db(sig, err), 3)
    return out


@torch.no_grad()
def decode_projection_qtip_2d(
    fields: Dict[str, torch.Tensor],
    sp: QtipProjSpec,
    signs: torch.Tensor,
    hyb_luts: Dict[int, torch.Tensor],
    device: torch.device,
    include_sparse: bool = True,
) -> torch.Tensor:
    rmap = fields["map"].to(device).long() & 0xFFFF
    sc = fields["scale"].to(device).float()
    E, d_out = rmap.shape
    d_in = sp.d_in
    nb = d_in // 16
    pair_tabs = {k: eval_hyb_all_states_torch(hyb_luts[k]) for k in (1, 2)}

    Us = torch.empty((E, d_out, d_in), dtype=torch.float32, device=device)
    for k, a, b in ((1, 0, sp.n1), (2, sp.n1, d_out)):
        if b <= a:
            continue
        nt = (b - a) // 16
        packed = fields[f"c{k}"].to(device).reshape(E * nt * nb, 32 * k)
        st = unpack_tailbite_l16(packed, M=128, step_bits=2 * k).long()
        tiles = pair_tabs[k][st].reshape(E, nt, nb, 16, 16).permute(0, 1, 3, 2, 4).reshape(E, b - a, d_in)
        Us[:, a:b, :] = tiles

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


def decode_projection_qtip_2d_chunked(
    fields: Dict[str, torch.Tensor],
    sp: QtipProjSpec,
    signs: torch.Tensor,
    hyb_luts: Dict[int, torch.Tensor],
    device: torch.device,
    include_sparse: bool = True,
    chunk: int = 16,
) -> torch.Tensor:
    """[R3.9 fix 3, 2026-11-16] `decode_projection_qtip_2d` builds every expert it is given on the GPU at once; for a
    layer's 240-288 kept experts that is 60-70 GiB of temporaries on a 32 GiB card (measured in the Layer-27 lab: 90 GiB
    spilled into Windows shared GPU memory, RAM exhausted). Experts are independent: decode `chunk` at a time with the
    unchanged arithmetic and return a CPU fp32 tensor (callers index one expert and move it to the device)."""
    E_all = int(fields["map"].shape[0])
    out = torch.empty((E_all, sp.d_out, sp.d_in), dtype=torch.float32)
    for e0 in range(0, E_all, chunk):
        e1 = min(E_all, e0 + chunk)
        sub = {k: (v[e0:e1] if (torch.is_tensor(v) and k in ("map", "scale", "c1", "c2", "sp")) else v) for k, v in fields.items()}
        out[e0:e1] = decode_projection_qtip_2d(sub, sp, signs, hyb_luts, device, include_sparse=include_sparse).cpu()
    torch.cuda.empty_cache()
    return out


def save_qtip_layer_archive(
    path: str,
    rule: str,
    layer_id: int,
    signs: Dict[str, torch.Tensor],
    hyb_luts: Dict[str, Dict[int, torch.Tensor]],
    fg: Dict[str, torch.Tensor],
    fu: Dict[str, torch.Tensor],
    fd: Dict[str, torch.Tensor],
    perm_kept: torch.Tensor,
    n_experts_total: int,
    meta: dict,
) -> int:
    """Atomically writes `layer_XX_qtip_hyb.pt` in popularity row order `perm_kept`, matching Stage-2B schema."""

    def _reorder(f: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            k: v.index_select(0, perm_kept.cpu()).contiguous()
            for k, v in f.items()
            if k in ("map", "scale", "c1", "c2", "sp")
        }

    blob = {
        "format": "qtip_hyb_r39",
        "rule": rule,
        "layer_id": int(layer_id),
        "tile_mode": "2d_16x16",
        "signs": {p: signs[p].detach().cpu().to(torch.int8).contiguous() for p in ("g", "u", "d")},
        "hyb_luts": {
            p: {
                1: hyb_luts[p][1].detach().cpu().to(torch.float16).contiguous(),
                2: hyb_luts[p][2].detach().cpu().to(torch.float16).contiguous(),
            }
            for p in ("g", "u", "d")
        },
        "fields": {"g": _reorder(fg), "u": _reorder(fu), "d": _reorder(fd)},
        "perm": perm_kept.detach().cpu().to(torch.int16).contiguous(),
        "n_experts_total": int(n_experts_total),
        "meta": meta,
    }
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        torch.save(blob, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return os.path.getsize(path)


class QtipBuildLayerRuntime:
    """Fast GPU runtime for advancing `cal`, `evn`, and `MTP` streams through a saved `layer_XX_qtip_hyb.pt`
    archive in expert-major `bf16` Tensor-Core batches + NVRTC `qtip_tile16x16_hyb_gemv_kernel` sanity gate."""

    def __init__(self, path: str, device: torch.device, dense_group: int = 16):
        blob = torch.load(path, map_location="cpu")
        self.layer_id = int(blob["layer_id"])
        self.rule = str(blob.get("rule", "A"))
        self.preset = make_qtip_preset(self.rule)
        self.device = device
        self.dense_group = int(dense_group)
        self.signs = {p: blob["signs"][p].to(device=device, dtype=torch.float32) for p in ("g", "u", "d")}
        self.hyb_luts = {
            p: {k: blob["hyb_luts"][p][k].to(device=device, dtype=torch.float16) for k in (1, 2)}
            for p in ("g", "u", "d")
        }
        self.fields = {
            p: {k: v.to(device, non_blocking=True) for k, v in blob["fields"][p].items()}
            for p in ("g", "u", "d")
        }
        self.n_tot = int(blob.get("n_experts_total", 288))
        perm = blob.get("perm", None)
        if perm is not None:
            perm_t = perm.to(torch.long)
            row_of = torch.full((self.n_tot,), -1, dtype=torch.long)
            row_of[perm_t] = torch.arange(perm_t.numel(), dtype=torch.long)
            self.row_of = row_of.to(device)
            self.missing = torch.nonzero(row_of < 0).flatten().tolist()
        else:
            self.row_of = None
            self.missing = []

    @torch.no_grad()
    def verify_nvrtc_gemv_gate(self, x_sample: torch.Tensor) -> float:
        sp = self.preset.gate
        fg = self.fields["g"]
        E = min(int(fg["map"].shape[0]), 8)
        P = min(int(x_sample.shape[0]), 8)
        slots = torch.arange(P, device=self.device, dtype=torch.int32) % E
        x_sub = x_sample[:P].to(self.device).float().contiguous()
        x_rot = fwht_full(x_sub * self.signs["g"]).contiguous()

        lut_stack = torch.stack([self.hyb_luts["g"][1], self.hyb_luts["g"][2]], dim=0).contiguous()
        shared = (1024 + sp.d_in) * 4
        kern = _compile_nvrtc("qtip_tile16x16_hyb_gemv_kernel", _QTIP_R39_CUDA_SRC, shared_bytes=shared)
        y_kern = torch.empty((P, sp.d_out), dtype=torch.float32, device=self.device)
        warps = 8
        gx = min(((sp.d_out // 16) + warps - 1) // warps, 64)
        kern(
            grid=(gx, P, 1),
            block=(32 * warps, 1, 1),
            shared_mem=shared,
            args=[
                fg["map"][:E].contiguous(),
                fg["scale"][:E].contiguous(),
                fg["c1"][:E].contiguous(),
                fg["c2"][:E].contiguous(),
                slots,
                int(sp.n1),
                int(sp.n2),
                lut_stack.view(torch.int32),
                x_rot,
                1,
                y_kern,
                int(sp.d_out),
                int(sp.d_in),
            ],
        )
        sub_f = {k: v[:E] for k, v in fg.items()}
        W_ref = decode_projection_qtip_2d(sub_f, sp, self.signs["g"], self.hyb_luts["g"], self.device, include_sparse=False)
        y_ref = torch.bmm(x_sub.unsqueeze(1), W_ref[slots.long()].transpose(1, 2)).squeeze(1)
        return float(F.cosine_similarity(y_kern.flatten(), y_ref.flatten(), dim=0).item())

    @torch.no_grad()
    def forward_routed_experts(self, x_flat: torch.Tensor, top_w: torch.Tensor, top_idx: torch.Tensor, swiglu_limit: float = LIM) -> torch.Tensor:
        x = x_flat.to(self.device, dtype=torch.float32)
        w = top_w.to(self.device, dtype=torch.float32)
        idx = top_idx.to(self.device, dtype=torch.long)
        N, K = idx.shape

        uniq, inv = torch.unique(idx, sorted=True, return_inverse=True)
        slot_of_u = self.row_of[uniq] if self.row_of is not None else uniq
        if bool((slot_of_u < 0).any()):
            bad = uniq[slot_of_u < 0].cpu().tolist()
            raise RuntimeError(f"Layer {self.layer_id}: routed to pruned experts {bad} without rows!")

        inv_f = inv.reshape(-1)
        order = torch.argsort(inv_f, stable=True)
        counts = torch.bincount(inv_f, minlength=int(uniq.numel())).tolist()
        tok_all = order // K
        w_all = w.reshape(-1)[order]
        starts = [0]
        for c in counts:
            starts.append(starts[-1] + c)

        out = torch.zeros_like(x)
        U_cnt = int(uniq.numel())
        for g0 in range(0, U_cnt, self.dense_group):
            g1 = min(U_cnt, g0 + self.dense_group)
            sl = slot_of_u[g0:g1]
            sub_g = {k: v.index_select(0, sl) for k, v in self.fields["g"].items()}
            sub_u = {k: v.index_select(0, sl) for k, v in self.fields["u"].items()}
            sub_d = {k: v.index_select(0, sl) for k, v in self.fields["d"].items()}

            Wg = decode_projection_qtip_2d(sub_g, self.preset.gate, self.signs["g"], self.hyb_luts["g"], self.device)
            Wu = decode_projection_qtip_2d(sub_u, self.preset.up, self.signs["u"], self.hyb_luts["u"], self.device)
            Wd = decode_projection_qtip_2d(sub_d, self.preset.down, self.signs["d"], self.hyb_luts["d"], self.device)

            for j in range(g1 - g0):
                a, b = starts[g0 + j], starts[g0 + j + 1]
                if b == a:
                    continue
                t = tok_all[a:b]
                xt = x.index_select(0, t)
                g_val = (xt.bfloat16() @ Wg[j].bfloat16().T).float().clamp(max=swiglu_limit)
                u_val = (xt.bfloat16() @ Wu[j].bfloat16().T).float().clamp(-swiglu_limit, swiglu_limit)
                z = F.silu(g_val) * u_val
                d_val = (z.bfloat16() @ Wd[j].bfloat16().T).float()
                out.index_add_(0, t, d_val * w_all[a:b].unsqueeze(-1))
            del Wg, Wu, Wd
        return out.to(x_flat.dtype)


# ================================================================================================================
# 3. Streaming Calibration Hessians + Cross-Covariance Quantized-Gate/Up H_down Solver
# ================================================================================================================
class StreamingHessR39:
    """Computes routing-weight^2 per-expert Hessians shrunk toward layer `Hbar` (`n0` pseudo-tokens) for `kept_ids`,
    plus exact FP64 Cholesky Cross-Covariance `W_d*` compensation on quantized `(Wg_hat, Wu_hat)` SwiGLU outputs."""

    def __init__(self, x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor, kept_ids: torch.Tensor, dev: torch.device, n0: float, n_tot: int = 288):
        self.x = x
        self.idx = idx.long()
        self.w = w.float()
        self.kept_ids = kept_ids.cpu().tolist()
        self.dev = dev
        self.n0 = float(n0)

        H = torch.zeros((x.shape[1], x.shape[1]), dtype=torch.float32, device=dev)
        for s in range(0, x.shape[0], 32768):
            xb = x[s : s + 32768].float()
            H.addmm_(xb.T, xb)
        self.Hbar = H / max(int(x.shape[0]), 1)
        self.counts_all = torch.bincount(self.idx.reshape(-1), minlength=n_tot)

    def _tokens(self, orig_e: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tk, sl = torch.where(self.idx == orig_e)
        return tk, self.w[tk, sl].pow(2)

    @torch.no_grad()
    def gu_hess(self, k0: int, k1: int) -> Tuple[torch.Tensor, None]:
        out = []
        for ki in range(k0, k1):
            orig_e = self.kept_ids[ki]
            tk, c = self._tokens(orig_e)
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
        k0: int,
        k1: int,
        Wg_kept: torch.Tensor,
        Wu_kept: torch.Tensor,
        Wd_kept: torch.Tensor,
        Wg_hat_kept: torch.Tensor,
        Wu_hat_kept: torch.Tensor,
        qz_damp: float = 0.15,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        H_list = []
        W_list = []
        d_inter = Wd_kept.shape[2]
        eye = torch.eye(d_inter, device=self.dev, dtype=torch.float64)

        for ki in range(k0, k1):
            orig_e = self.kept_ids[ki]
            tk, c = self._tokens(orig_e)
            We_d = Wd_kept[ki].to(self.dev).float()
            if tk.numel() == 0:
                H_list.append(torch.eye(d_inter, device=self.dev))
                W_list.append(We_d)
                continue

            xe = self.x[tk].float()
            cw = c.unsqueeze(-1)
            cm = c.mean().clamp_min(1e-12)
            Ge = Wg_kept[ki].to(self.dev).float()
            Ue = Wu_kept[ki].to(self.dev).float()
            Gh = Wg_hat_kept[ki].to(self.dev).float()
            Uh = Wu_hat_kept[ki].to(self.dev).float()

            z_ex = F.silu((xe @ Ge.T).clamp(max=LIM)) * (xe @ Ue.T).clamp(-LIM, LIM)
            z_qh = F.silu((xe @ Gh.T).clamp(max=LIM)) * (xe @ Uh.T).clamp(-LIM, LIM)

            C_qq = (z_qh * cw).T @ z_qh
            C_eq = (z_ex * cw).T @ z_qh
            diag_m = torch.diagonal(C_qq).mean() / max(float(c.sum()), 1e-12)
            prior = torch.eye(d_inter, device=self.dev) * diag_m
            denom = c.sum() + self.n0 * cm

            H_qq = (C_qq + self.n0 * cm * prior) / denom
            H_eq = (C_eq + self.n0 * cm * prior) / denom
            H_list.append(H_qq)

            H_qq64 = H_qq.double()
            diff64 = (H_eq - H_qq).double()
            lam = float(qz_damp) * float(torch.diagonal(H_qq64).mean().clamp_min(1e-12))
            chol = torch.linalg.cholesky(H_qq64 + lam * eye)
            delta_m_T = torch.cholesky_solve(diff64.T, chol)
            W_star = We_d.double() + We_d.double() @ delta_m_T.T
            W_list.append(W_star.float())

        return torch.stack(H_list, dim=0), torch.stack(W_list, dim=0)


# ================================================================================================================
# 4. Stream Management, GPU FP8 Expert Dequantizer, Shard Eviction & MTP Acceptance (Matches run_tcq_build_r33.py)
# ================================================================================================================
class StreamSet:
    def __init__(self, name: str, seqs: List[List[int]], bs: int):
        self.name = name
        self.seqs = seqs
        self.bs = bs
        self.h: List[Optional[torch.Tensor]] = [None] * len(seqs)

    def chunks(self):
        i = 0
        while i < len(self.seqs):
            j = i + 1
            while j < len(self.seqs) and j - i < self.bs and len(self.seqs[j]) == len(self.seqs[i]):
                j += 1
            yield i, j
            i = j


def put_stream(st: StreamSet, i: int, j: int, hn: torch.Tensor) -> None:
    for k in range(i, j):
        st.h[k] = hn[k - i : k - i + 1].to(torch.bfloat16).cpu()


@torch.no_grad()
def layer_inputs(bb, l: int, st: StreamSet, dev: torch.device, cfg):
    for i, j in st.chunks():
        h = torch.cat([st.h[k] for k in range(i, j)]).to(dev).float()
        h_att = bb.attn_block(l, h, None)
        del h
        post, comb, xf = bb.ffn_prepare(l, h_att)
        x = xf.reshape(-1, cfg.hidden_size).float()
        w, idx = bb.route(l, x)
        yield i, j, h_att, post, comb, xf.shape, x, w, idx


@torch.no_grad()
def load_experts_gpu(ck, l: int, n_experts: int, dev: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = []
    for proj in ("gate_proj", "up_proj", "down_proj"):
        W = None
        for e in range(n_experts):
            name = f"{LM}layers.{l}.mlp.experts.{e}.{proj}.weight"
            raw, dt, shape = ck._shard(ck.weight_map[name]).get_raw_slice(name)
            if dt in ("F8_E4M3", "F8_E4M3FN"):
                sc = torch.from_numpy(ck.get_numpy_f32(name + "_scale_inv")).to(dev)
                O, I = shape
                so, si = sc.shape
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    q = torch.from_numpy(np.array(raw, copy=True)).to(dev)
                w = q.view(torch.float8_e4m3fn).to(torch.float32).view(so, O // so, si, I // si)
                w = w.mul_(sc[:, None, :, None]).view(O, I).to(torch.bfloat16)
            else:
                w = torch.from_numpy(ck.get_numpy_f32(name)).to(dev, dtype=torch.bfloat16)
            if W is None:
                W = torch.empty((n_experts, w.shape[0], w.shape[1]), dtype=torch.bfloat16, device="cpu")
            W[e].copy_(w.cpu())
        out.append(W)
    return out[0], out[1], out[2]


@torch.no_grad()
def exact_routed(Wg: torch.Tensor, Wu: torch.Tensor, Wd: torch.Tensor, x: torch.Tensor, w: torch.Tensor, idx: torch.Tensor, dev: torch.device) -> torch.Tensor:
    y = torch.zeros_like(x)
    for e in torch.unique(idx).tolist():
        tk, sl = torch.where(idx == e)
        xx = x[tk]
        z = F.silu((xx @ Wg[e].to(dev).float().T).clamp(max=LIM)) * (xx @ Wu[e].to(dev).float().T).clamp(-LIM, LIM)
        y.index_add_(0, tk, (z @ Wd[e].to(dev).float().T) * w[tk, sl].unsqueeze(-1))
    return y


@torch.no_grad()
def advance_eval(bb, l: int, st: StreamSet, dev: torch.device, cfg, moe_all, score_cb=None) -> None:
    parts = list(layer_inputs(bb, l, st, dev, cfg))
    X = torch.cat([q[6] for q in parts])
    W_ = torch.cat([q[7] for q in parts])
    I_ = torch.cat([q[8] for q in parts])
    Y = moe_all(X, W_, I_)
    if score_cb is not None:
        pos = torch.cat([torch.arange(len(st.seqs[k]), device=dev) for q in parts for k in range(q[0], q[1])])
        score_cb(X, W_, I_, Y, pos)
    off = 0
    for i, j, h_att, post, comb, shp, x, _, _ in parts:
        n = x.shape[0]
        put_stream(st, i, j, bb.mhc_merge(post, comb, (Y[off : off + n] + bb.shared_expert(l, x)).view(shp), h_att))
        off += n
    del parts, X, W_, I_, Y


@torch.no_grad()
def advance_eval_grouped(bb, l: int, st: StreamSet, dev: torch.device, cfg, moe_all, group: int = 16) -> None:
    """[R3.9 addition, 2026-11-16] `advance_eval` keeps the attention output of every sequence of a stream on the GPU
    (about 0.13 GiB per 2,048-token sequence): fine for the 29 qbench rows, not for the 128 teacher rows. Same
    computation, `group` sequences per pass."""
    for g0 in range(0, len(st.seqs), group):
        g1 = min(len(st.seqs), g0 + group)
        sub = StreamSet(st.name, st.seqs[g0:g1], st.bs)
        sub.h = st.h[g0:g1]
        advance_eval(bb, l, sub, dev, cfg, moe_all)
        st.h[g0:g1] = sub.h
        del sub
    torch.cuda.empty_cache()


@torch.no_grad()
def cal_inputs(bb, l: int, st: StreamSet, dev: torch.device, cfg) -> List[dict]:
    chunks = []
    for i, j, h_att, post, comb, shp, x, w, idx in layer_inputs(bb, l, st, dev, cfg):
        chunks.append(
            {
                "i": i,
                "j": j,
                "h_att": h_att.to(torch.bfloat16).cpu(),
                "post": post,
                "comb": comb,
                "shp": shp,
                "x": x.to(torch.bfloat16).cpu(),
                "w": w,
                "idx": idx,
            }
        )
        del h_att, x
    return chunks


@torch.no_grad()
def cal_advance(bb, l: int, st: StreamSet, chunks: List[dict], dev: torch.device, moe_all, block_tokens: int = 32768) -> None:
    blk, nt = [], 0
    for n_, c in enumerate(chunks):
        blk.append(c)
        nt += c["x"].shape[0]
        if nt < block_tokens and n_ < len(chunks) - 1:
            continue
        X = torch.cat([b["x"] for b in blk]).to(dev).float()
        Y = moe_all(X, torch.cat([b["w"] for b in blk]), torch.cat([b["idx"] for b in blk]))
        Y += bb.shared_expert(l, X)
        off = 0
        for b in blk:
            n = b["x"].shape[0]
            put_stream(st, b["i"], b["j"], bb.mhc_merge(b["post"], b["comb"], Y[off : off + n].view(b["shp"]), b["h_att"].to(dev).float()))
            off += n
        del X, Y
        blk, nt = [], 0


@torch.no_grad()
def mtp_acceptance_r39(bb, l: int, streams: Dict[str, StreamSet], seqs: List[List[int]], dev: torch.device, Wg, Wu, Wd, new_pt: str, old_npz: str, TR_mod=None) -> dict:
    bb.set_pruned(l, [])
    impls = {"evx": lambda x, w, i: exact_routed(Wg, Wu, Wd, x, w, i, dev)}
    rt_new = QtipBuildLayerRuntime(new_pt, dev)
    impls["evn"] = lambda x, w, i: rt_new.forward_routed_experts(x, w, i, swiglu_limit=LIM).float()
    rt_old = None
    if TR_mod is not None and old_npz and os.path.exists(old_npz):
        rt_old = TR_mod.TcqLayerRuntime(TR_mod.TcqArchive(old_npz, mmap_slab=False), dev, slab_in_vram=True)
        impls["evo"] = lambda x, w, i: rt_old.forward_routed_experts(x, w, i, swiglu_limit=LIM).float()

    out = {}
    for name, impl in impls.items():
        moe_fn = lambda l_, xf_: impl(xf_, *bb.route(l_, xf_))
        acc, hit, n = 0, 0, 0
        for k, ids in enumerate(seqs):
            h = streams[name].h[k]
            if h is None or len(ids) < 14:
                continue
            logits, coll = bb.final_logits(h.to(dev).float(), last_only=False)
            main_arg = logits[0].argmax(-1)
            del logits
            ml = bb.mtp_forward(torch.tensor([ids[1:]], device=dev), coll[:, :-1], cache=None, moe_fn=moe_fn)
            mtp_arg = ml[0].argmax(-1)
            del ml, coll
            t = torch.arange(10, len(ids) - 2, device=dev)
            acc += int((mtp_arg[t] == main_arg[t + 1]).sum())
            hit += int((mtp_arg[t] == torch.tensor(ids, device=dev)[t + 2]).sum())
            n += int(t.numel())
        out[name] = {"accept": round(acc / max(n, 1), 4), "top1_vs_text": round(hit / max(n, 1), 4), "positions": n}
    del rt_new, rt_old
    return out


def evict_shards(src_dir: str, shard_margin_gib: float, todo: List[int], li: int, need: Dict[int, List[str]]) -> List[str]:
    msgs = []
    nxt = set(need[todo[li + 1]]) if li + 1 < len(todo) else set()

    def next_use(sh: str) -> Optional[int]:
        for k in range(li + 1, len(todo)):
            if sh in need[todo[k]]:
                return k
        return None

    def rm(sh: str, why: str) -> None:
        try:
            os.remove(os.path.join(src_dir, sh))
            msgs.append(f"deleted {sh} ({why})")
        except OSError as ex_:
            msgs.append(f"deferred delete {sh}: {ex_!r}")

    on_disk = [
        sh
        for sh in sorted({s_ for v in need.values() for s_ in v})
        if sh not in KEEP_SHARDS and os.path.exists(os.path.join(src_dir, sh))
    ]
    keep = []
    for sh in on_disk:
        if next_use(sh) is None:
            rm(sh, "not needed again")
        else:
            keep.append(sh)
    room = 5.0 * sum(1 for sh in nxt if not os.path.exists(os.path.join(src_dir, sh))) + shard_margin_gib
    for sh in sorted([x for x in keep if x not in nxt], key=lambda s_: next_use(s_) or 0, reverse=True):
        if shutil.disk_usage(src_dir).free / (2 ** 30) >= room:
            break
        nu = next_use(sh)
        rm(sh, f"evicted, next needed by L{todo[nu]:02d}" if nu is not None else "evicted")
    return msgs


def link_shared(old_dir: str, dst_dir: str) -> int:
    if not old_dir or not os.path.isdir(old_dir):
        return 0
    n = 0
    for name in SHARED_FILES:
        s_, d_ = os.path.join(old_dir, name), os.path.join(dst_dir, name)
        if os.path.exists(s_) and not os.path.exists(d_):
            try:
                os.link(s_, d_)
                n += 1
            except OSError:
                shutil.copy2(s_, d_)
                n += 1
    se_s, se_d = os.path.join(old_dir, "super_experts"), os.path.join(dst_dir, "super_experts")
    if os.path.isdir(se_s):
        os.makedirs(se_d, exist_ok=True)
        for f in os.listdir(se_s):
            sf, df = os.path.join(se_s, f), os.path.join(se_d, f)
            if not os.path.exists(df):
                try:
                    os.link(sf, df)
                except OSError:
                    shutil.copy2(sf, df)
                n += 1
    return n


def pack_seqs(tok, args, gm: List[int]) -> Tuple[List[List[int]], List[List[int]]]:
    cal: List[List[int]] = []
    if args.cal_c4 > 0 and os.path.exists(args.c4_path):
        c4 = []
        with gzip.open(args.c4_path, "rt", encoding="utf-8") as f:
            for line in f:
                c4.append(json.loads(line)["text"])
                if len(c4) >= 20000:
                    break
        cur: List[int] = []
        for d in c4:
            cur += tok.encode_raw(d + "\n\n")
            if len(cur) >= args.seq_len - len(gm):
                cal.append(gm + cur[: args.seq_len - len(gm)])
                cur = []
                if len(cal) >= args.cal_c4:
                    break
    if args.cal_ct > 0 and os.path.exists(args.ct_path):
        rows = json.load(open(args.ct_path, encoding="utf-8"))["rows"]
        cur, n_ct = [], 0
        for r in rows:
            cur += r["input_ids"] + r["response_ids"]
            while len(cur) >= args.seq_len and n_ct < args.cal_ct:
                cal.append(cur[: args.seq_len])
                cur = cur[args.seq_len :]
                n_ct += 1
            if n_ct >= args.cal_ct:
                break
    qb = [(r["input_ids"] + r["response_ids"])[: args.seq_len] for r in json.load(open(args.qb_path, encoding="utf-8"))["rows"]]
    if args.qb_rows > 0:
        qb = qb[: args.qb_rows]
    return cal, qb


# ================================================================================================================
# 5. Standalone GPU Self-Test (--selftest: Tests Full Sequential Streaming Pipeline + Stage-2B Schema in ~10s)
# ================================================================================================================
def run_selftest() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("--selftest requires a CUDA GPU.")
    dev = torch.device("cuda")
    tmp_dst = "/tmp/r39_streaming_build_selftest"
    shutil.rmtree(tmp_dst, ignore_errors=True)
    os.makedirs(tmp_dst, exist_ok=True)
    print(f"[SELFTEST] Verifying 42-layer streaming builder pipeline on {torch.cuda.get_device_name(0)}...", flush=True)

    # Train miniature HYB codebooks & test full encode -> save_qtip_layer_archive -> QtipBuildLayerRuntime -> NVRTC gate
    hyb_base = train_or_load_hyb_codebooks(dev, os.path.join(tmp_dst, "qtip_hyb_L16_V2.pt"))
    E_tot, Di, Dm = 6, 64, 64
    gen = torch.Generator(device=dev).manual_seed(309)
    Wg = (torch.randn((E_tot, Di, Dm), generator=gen, device=dev) / 8.0).bfloat16().cpu()
    Wu = (torch.randn((E_tot, Di, Dm), generator=gen, device=dev) / 8.0).bfloat16().cpu()
    Wd = (torch.randn((E_tot, Dm, Di), generator=gen, device=dev) / 8.0).bfloat16().cpu()

    # Prune expert 2 -> keep 5 experts [0, 1, 3, 4, 5], ordered by popularity
    kept_ids = torch.tensor([0, 1, 3, 4, 5], dtype=torch.long)
    X_cal = torch.randn((512, Dm), generator=gen, device=dev)
    idx_cal = kept_ids.to(dev)[torch.randint(0, 5, (512, 2), generator=gen, device=dev)]
    w_cal = torch.softmax(torch.randn((512, 2), generator=gen, device=dev), dim=-1)
    hs = StreamingHessR39(X_cal, idx_cal, w_cal, kept_ids, dev, n0=64.0, n_tot=E_tot)

    sp_g = QtipProjSpec("gate", Di, Dm, n1=16, k_sparse=4, unbiased=True)
    sp_u = QtipProjSpec("up", Di, Dm, n1=16, k_sparse=4, unbiased=True)
    sp_d = QtipProjSpec("down", Dm, Di, n1=0, k_sparse=8, unbiased=False)
    signs = {p: generate_rademacher_signs(Dm, s, dev) for p, s in (("g", 11), ("u", 12), ("d", 13))}
    luts = {p: {1: hyb_base[1].clone(), 2: hyb_base[2].clone()} for p in ("g", "u", "d")}

    Wg_k, Wu_k, Wd_k = Wg[kept_ids], Wu[kept_ids], Wd[kept_ids]
    fg = encode_projection_qtip_2d_ldlq(Wg_k, sp_g, signs["g"], luts["g"], dev, hs.gu_hess, scale_mode="hunb")
    fu = encode_projection_qtip_2d_ldlq(Wu_k, sp_u, signs["u"], luts["u"], dev, hs.gu_hess, scale_mode="hunb")
    Gh = decode_projection_qtip_2d(fg, sp_g, signs["g"], luts["g"], dev)
    Uh = decode_projection_qtip_2d(fu, sp_u, signs["u"], luts["u"], dev)
    fd = encode_projection_qtip_2d_ldlq(
        Wd_k,
        sp_d,
        signs["d"],
        luts["d"],
        dev,
        lambda a, b: hs.down_hess_and_target(a, b, Wg_k, Wu_k, Wd_k, Gh, Uh, qz_damp=0.15),
        scale_mode="hls",
    )

    counts_kept = hs.counts_all[kept_ids.to(dev)].cpu()
    pop_order = torch.argsort(counts_kept, descending=True, stable=True)
    perm_global = kept_ids[pop_order]

    out_pt = qtip_layer_path(tmp_dst, 3)
    # Reorder fields by pop_order inside save_qtip_layer_archive, while storing global expert id `perm_global`!
    fg_pop = {k: (v.index_select(0, pop_order) if torch.is_tensor(v) else v) for k, v in fg.items()}
    fu_pop = {k: (v.index_select(0, pop_order) if torch.is_tensor(v) else v) for k, v in fu.items()}
    fd_pop = {k: (v.index_select(0, pop_order) if torch.is_tensor(v) else v) for k, v in fd.items()}
    identity_idx = torch.arange(len(kept_ids), dtype=torch.long)
    nbytes = save_qtip_layer_archive(
        out_pt, "GU125", 3, signs, luts, fg_pop, fu_pop, fd_pop, perm_kept=identity_idx, n_experts_total=E_tot, meta={"test": True}
    )
    # Patch `perm` to global kept_ids[pop_order] and verify runtime reload + Stage-2B schema
    blob = torch.load(out_pt, map_location="cpu")
    blob["perm"] = perm_global.to(torch.int16)
    torch.save(blob, out_pt)

    # Verify schema required by r39_stage2b_global_kl_trainer.CompressedLayerHandle.from_qtip_pt
    assert set(blob.keys()) >= {"layer_id", "tile_mode", "signs", "hyb_luts", "fields", "perm", "n_experts_total"}
    assert blob["fields"]["g"]["c1"].shape == (5, 1, 4, 32) and blob["fields"]["g"]["c2"].shape == (5, 3, 4, 64)

    # Verify NVRTC GEMV gate on custom dimensions
    rt = QtipBuildLayerRuntime.__new__(QtipBuildLayerRuntime)
    rt.layer_id = 3
    rt.device = dev
    rt.dense_group = 4
    rt.preset = QtipPreset("mini", sp_g, sp_u, sp_d)
    rt.signs = {p: blob["signs"][p].to(dev).float() for p in ("g", "u", "d")}
    rt.hyb_luts = {p: {k: blob["hyb_luts"][p][k].to(dev) for k in (1, 2)} for p in ("g", "u", "d")}
    rt.fields = {p: {k: v.to(dev) for k, v in blob["fields"][p].items()} for p in ("g", "u", "d")}
    row_of = torch.full((E_tot,), -1, dtype=torch.long)
    row_of[perm_global] = torch.arange(5, dtype=torch.long)
    rt.row_of = row_of.to(dev)
    rt.missing = [2]

    cos_gate = rt.verify_nvrtc_gemv_gate(X_cal[:8])
    assert cos_gate > 0.99999, f"NVRTC gate failed: cos={cos_gate}"
    y_rt = rt.forward_routed_experts(X_cal[:32], w_cal[:32], idx_cal[:32])
    assert y_rt.shape == (32, Dm) and not torch.isnan(y_rt).any()
    shutil.rmtree(tmp_dst, ignore_errors=True)
    print(f"[SELFTEST PASSED] Archive ({nbytes} B), popularity perm, 2D-BlockLDLQ, QZ H_down, NVRTC gate (cos={cos_gate:.6f}), and Stage-2B schema verified!", flush=True)


# ================================================================================================================
# 6. Full 42-Layer + MTP Layer 45 Sequential Streaming Build Main Loop
# ================================================================================================================
@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description="Round 3.9 Full 42-Layer + MTP Sequential QTIP HYB Builder")
    ap.add_argument("--selftest", action="store_true", help="Run standalone GPU pipeline verification and exit")
    ap.add_argument("--pkg-dir", default=DEFAULT_PKG, help="Path to claude_r32_apex_fold directory")
    ap.add_argument("--dst", default=r"C:\models\GLM53-APEX-FOLD-R39-HYB", help="Output archive directory")
    ap.add_argument("--old", default=DEFAULT_OLD, help="Optional old TCQ33 directory for side-by-side SNR/KL delta")
    ap.add_argument("--src", default=DEFAULT_SRC, help="Source FP8 safetensors directory")
    ap.add_argument("--bb", default=DEFAULT_BB, help="Backbone checkpoint directory")
    ap.add_argument("--c4-path", default=DEFAULT_C4)
    ap.add_argument("--ct-path", default=DEFAULT_CT)
    ap.add_argument("--qb-path", default=DEFAULT_QB)
    ap.add_argument("--rule", default="A", choices=["A", "GU125"], help="'A' = 2.01 bpw; 'GU125' = 1.917 bpw Sub-70 GB")
    ap.add_argument("--prune-spec", default="", help="Optional Round 3.8/3.9 non-uniform expert pruning JSON")
    ap.add_argument("--layers", default="3-45", help="Layer range '3-45' or comma-separated list")
    ap.add_argument("--n0", type=float, default=8192.0, help="Hessian shrinkage pseudo-tokens")
    ap.add_argument("--damp", type=float, default=0.10, help="Rotated Hessian diagonal damping")
    ap.add_argument("--qz-damp", type=float, default=0.15, help="Tikhonov damping for Cross-Covariance W_d* solver")
    ap.add_argument("--cal-c4", type=int, default=64)
    ap.add_argument("--cal-ct", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--qb-rows", type=int, default=0, help="0 = all 29 qbench rows")
    ap.add_argument("--replace-old", action="store_true")
    ap.add_argument("--keep-src", action="store_true", help="Never delete source FP8 shards")
    ap.add_argument("--abort-after", type=int, default=12, help="Early-abort SNR check after this layer")
    ap.add_argument("--abort-min-gain-db", type=float, default=0.25, help="Min mean routed SNR gain vs --old TCQ33")
    ap.add_argument("--fetch", default="none", choices=["shards", "none"])
    ap.add_argument("--prefetch-ahead", type=int, default=1)
    ap.add_argument("--floor-gib", type=float, default=8.0)
    ap.add_argument("--shard-margin-gib", type=float, default=12.0)
    ap.add_argument("--no-mtp-eval", dest="mtp_eval", action="store_false")
    ap.add_argument("--ckpt-every", type=int, default=4)
    ap.add_argument("--stop-after", type=int, default=-1, help="Checkpoint and exit (code 3) after this layer")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tag", default="")
    # [R3.9 addition, 2026-11-16] exact teacher inside the build (see advance_eval_grouped)
    ap.add_argument("--teacher-out", default="", help="Write the Stage-2B teacher cache (r39_compact_teacher_top128_v1) from an "
                    "exact FP8 stream over the trainer's own train sequences, advanced layer by layer during the build")
    ap.add_argument("--teacher-train-seqs", type=int, default=128)
    ap.add_argument("--teacher-group", type=int, default=16, help="Sequences of the teacher stream advanced per GPU pass")
    args = ap.parse_args()

    if args.selftest:
        run_selftest()
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for 42-layer QTIP HYB build.")
    dev = torch.device("cuda")

    if os.path.isdir(args.pkg_dir) and args.pkg_dir not in sys.path:
        sys.path.insert(0, args.pkg_dir)

    import glm53_fold_backbone as FB  # type: ignore
    import glm53_fold_tcq_runtime as TR  # type: ignore
    from glm53_fold_config import GLM53FlashModelConfig  # type: ignore
    from glm53_fold_generate import RealTokenizer, load_prune_spec  # type: ignore

    cfg = GLM53FlashModelConfig()
    os.makedirs(args.dst, exist_ok=True)
    ck_dir = os.path.join(args.dst, "_r39_ckpt")
    os.makedirs(ck_dir, exist_ok=True)
    out_dir = os.path.join(HERE, "out_r32")
    os.makedirs(out_dir, exist_ok=True)
    lg = open(os.path.join(out_dir, f"r39_qtip_build{args.tag}.log"), "a", encoding="utf-8")

    if "-" in args.layers:
        a_l, b_l = (int(v) for v in args.layers.split("-"))
        order = [l for l in range(a_l, b_l + 1) if cfg.is_moe_layer(l)]
    else:
        order = [int(v) for v in args.layers.split(",")]

    prune_map: Dict[int, List[int]] = load_prune_spec(args.prune_spec, cfg) if args.prune_spec else {}
    preset = make_qtip_preset(args.rule)
    hyb_base = train_or_load_hyb_codebooks(dev, os.path.join(out_dir, "qtip_hyb_L16_V2.pt"))

    prog_path = os.path.join(args.dst, "convert_progress.json")
    meta_path = os.path.join(ck_dir, "meta.json")
    if args.resume:
        prog = json.load(open(prog_path, encoding="utf-8"))
        meta = json.load(open(meta_path, encoding="utf-8"))
    else:
        assert not os.path.exists(meta_path), f"{meta_path} exists: pass --resume or choose a fresh --dst"
        prog = {"rule": args.rule, "prune_spec": args.prune_spec, "args": vars(args), "layers": {}}
        meta = {"done": [], "gains": [], "evo_valid": True, "main_done": 0}
        manifest = {
            "format": "qtip_hyb_r39",
            "rule": args.rule,
            "preset": preset.name,
            "bpw_routed_unpruned": round(preset.bpw(), 4),
            "expert_row_bytes": preset.expert_row_bytes,
            "codebook": "QTIP HYB-mul1 L=16, V=2, Q=9 (2 KiB half2[512] per proj/k, 2-pass collar tail-biting)",
            "ldlq": "2D-BlockLDLQ 16x16 tiles (g=16 Cholesky column feedback + Cross-Covariance quantized H_down)",
            "recipe": {
                "n0": args.n0,
                "damp": args.damp,
                "qz_damp": args.qz_damp,
                "prune_spec": args.prune_spec or None,
                "popularity_order": True,
            },
            "signs_seed": "1000*layer + {g:11, u:12, d:13}",
            "layers": order,
        }
        with open(os.path.join(args.dst, "fold_manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

    done, gains = list(meta["done"]), list(meta["gains"])
    ck_streams, roll, roll_old = {}, {}, []
    if args.resume:
        ck_streams = {nm: torch.load(os.path.join(ck_dir, f"{nm}.pt"), map_location="cpu") for nm in ("evx", "evn", "evo")}
        M = meta["main_done"]
        mx, mn, mo = (ck_streams[nm]["main_done"] for nm in ("evx", "evn", "evo"))
        assert mx >= mn >= mo >= M, ("inconsistent checkpoint", mx, mn, mo, M)
        if mx > M:
            gap = [x for x in order if x not in done and x < cfg.num_hidden_layers][: mx - M]
            for x in gap:
                r_ = prog["layers"][str(x)]
                if "snr_old_db" in r_:
                    gains.append(round(r_["snr_new_db"] - r_["snr_old_db"], 3))
            done += gap
            roll = {"evn": gap[mn - M :], "evo": gap[mo - M :]}
            if args.replace_old:
                roll_old = [tcq_layer_path(args.old, x) for x in gap if os.path.exists(tcq_layer_path(args.old, x))]
            meta["main_done"] = mx
            log(f"resume roll-forward: streams at {mx}/{mn}/{mo}, meta at {M}; gap={gap}", lg)
        prog["layers"] = {k: v for k, v in prog["layers"].items() if int(k) in done}
        with open(prog_path, "w", encoding="utf-8") as fh:
            json.dump(prog, fh, indent=2)

    todo = [l for l in order if l not in done]
    n_link = link_shared(args.old, args.dst)
    wm = json.load(open(os.path.join(args.src, "model.safetensors.index.json"), encoding="utf-8"))["weight_map"]
    need = {l: sorted({sh for k, sh in wm.items() if k.startswith(f"{LM}layers.{l}.mlp.experts.")}) for l in order}
    log(
        f"start QTIP HYB rule={args.rule} ({preset.bpw():.4f} bpw, {preset.expert_row_bytes} B/expert); "
        f"prune_layers={len(prune_map)}; n0={args.n0} damp={args.damp} qz_damp={args.qz_damp}; todo={todo}; "
        f"linked={n_link}; free={shutil.disk_usage(args.dst).free / (2**30):.1f} GiB",
        lg,
    )

    pf = None
    pf_script = _first_existing(
        [os.path.join(HERE, "prefetch_shards_r33.py"), os.path.join(HERE, "prefetch_shards_r31_par.py")], ""
    )
    if args.fetch == "shards" and todo and pf_script:
        pf = subprocess.Popen(
            [
                sys.executable,
                "-u",
                pf_script,
                "--src",
                args.src,
                "--dst",
                args.dst,
                "--order",
                ",".join(map(str, todo)),
                "--ahead",
                str(args.prefetch_ahead),
                "--workers",
                "3",
                "--floor-gib",
                str(args.floor_gib),
            ],
            stdout=open(os.path.join(out_dir, f"r39_build_prefetch{args.tag}.log"), "a", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            cwd=HERE,
            env=dict(os.environ, HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUTF8="1"),
        )

    def stop_pf() -> None:
        if pf is not None and pf.poll() is None:
            pf.terminate()

    atexit.register(stop_pf)

    tok = RealTokenizer(args.bb)
    gm = tok.encode_raw("[gMASK]<sop>")
    cal_seqs, qb_seqs = pack_seqs(tok, args, gm)
    log(f"calibration: {len(cal_seqs)} x {args.seq_len} tok; qbench: {len(qb_seqs)} seqs", lg)

    bb = FB.RealGLM53Backbone(FB.RealGLM53Checkpoint(args.bb), cfg, device="cuda", backbone_mode="fp8")
    bb.load_global()

    streams = {
        "cal": StreamSet("cal", cal_seqs, 2),
        "evx": StreamSet("evx", qb_seqs, 1),
        "evn": StreamSet("evn", qb_seqs, 1),
        "evo": StreamSet("evo", qb_seqs, 1),
    }

    # [R3.9 addition, 2026-11-16] Exact teacher stream. The FP8 source shards are only on disk while their layer is
    # being built, so the teacher targets of Stage 2B are produced here instead of by a second pass over 328 GB
    # (r39_extract_teacher_top128.py --teacher-mode exact_fp8 needs every shard locally). Same sequences as that script.
    teacher_on = bool(args.teacher_out)
    TE = None
    tr_ids = tq_ids = None
    if teacher_on:
        import r39_extract_teacher_top128 as TE  # type: ignore

        tr_ids, tq_ids = TE.pack_train_and_qbench_sequences(tok, args.ct_path, args.c4_path, args.qb_path, args.teacher_train_seqs, args.seq_len)
        streams["trx"] = StreamSet("trx", tr_ids.tolist(), 2)
        log(f"teacher stream: {tr_ids.shape[0]} x {tr_ids.shape[1]} tok exact FP8 -> {args.teacher_out}", lg)

    t_all = time.perf_counter()
    for name, st in streams.items():
        if args.resume and name != "cal":
            continue
        for i, j in st.chunks():
            h = bb.embed_streams(torch.tensor(st.seqs[i:j], device=dev))
            for l in range(cfg.first_k_dense_replace):
                h = bb.layer_forward(l, h, None, None)
            put_stream(st, i, j, h)

    if args.resume:
        for name in ("evx", "evn", "evo"):
            streams[name].h = ck_streams[name]["h"]
        if teacher_on:
            trx_p = os.path.join(ck_dir, "trx.pt")
            trx_ck = torch.load(trx_p, map_location="cpu") if os.path.exists(trx_p) else None
            if trx_ck is None or trx_ck["main_done"] != ck_streams["evx"]["main_done"] or len(trx_ck["h"]) != len(streams["trx"].seqs):
                log("WARNING: teacher stream checkpoint missing or not at the exact stream's layer: teacher cache DISABLED for this run", lg)
                teacher_on = False
                del streams["trx"]
            else:
                streams["trx"].h = trx_ck["h"]
            del trx_ck
        ck_streams = None
        for name, ls in roll.items():
            for l in ls:
                t_r = time.perf_counter()
                if name == "evn":
                    rt = QtipBuildLayerRuntime(qtip_layer_path(args.dst, l), dev)
                    bb.set_pruned(l, rt.missing)
                    advance_eval(bb, l, streams["evn"], dev, cfg, lambda X, W_, I_: rt.forward_routed_experts(X, W_, I_).float())
                    bb.set_pruned(l, [])
                else:
                    old_p = tcq_layer_path(args.old, l)
                    rt = TR.TcqLayerRuntime(TR.TcqArchive(old_p, mmap_slab=False), dev, slab_in_vram=True)
                    bb.set_pruned(l, rt.arch.missing.tolist() if rt.arch.missing is not None else [])
                    advance_eval(bb, l, streams["evo"], dev, cfg, lambda X, W_, I_: rt.forward_routed_experts(X, W_, I_, swiglu_limit=LIM).float())
                    bb.set_pruned(l, [])
                del rt
                bb.unload_layer(l)
                log(f"resume: {name} rolled forward through L{l:02d} ({time.perf_counter() - t_r:.0f}s)", lg)

        for l in [x for x in done if x < cfg.num_hidden_layers]:
            t_r = time.perf_counter()
            rt = QtipBuildLayerRuntime(qtip_layer_path(args.dst, l), dev)
            bb.set_pruned(l, rt.missing)
            chunks = cal_inputs(bb, l, streams["cal"], dev, cfg)
            cal_advance(bb, l, streams["cal"], chunks, dev, lambda X, W_, I_: rt.forward_routed_experts(X, W_, I_).float())
            bb.set_pruned(l, [])
            del rt, chunks
            bb.unload_layer(l)
            log(f"resume: calibration replayed through L{l:02d} ({time.perf_counter() - t_r:.0f}s)", lg)

    for l in range(cfg.first_k_dense_replace):
        bb.unload_layer(l)
    log(f"streams ready ({time.perf_counter() - t_all:.0f}s)", lg)

    t_all = time.perf_counter()
    pending_old = list(roll_old)

    for li, l in enumerate(todo):
        t0 = time.perf_counter()
        for sh in need[l]:
            p_sh = os.path.join(args.src, sh)
            if not os.path.exists(p_sh):
                assert args.fetch == "shards", f"Missing shard {p_sh} (pass --fetch shards)"
                from huggingface_hub import hf_hub_download

                hf_hub_download(cfg.hf_repo_fp8, sh, local_dir=args.src)

        ck = FB.RealGLM53Checkpoint(args.src)
        Wg, Wu, Wd = load_experts_gpu(ck, l, cfg.n_routed_experts, dev)
        rec: dict = {"load_s": round(time.perf_counter() - t0, 1)}

        main_layer = l < cfg.num_hidden_layers
        pruned_l = sorted(prune_map.get(l, [])) if main_layer else []
        kept_ids = torch.tensor([e for e in range(cfg.n_routed_experts) if e not in set(pruned_l)], dtype=torch.long)

        # MTP Layer 45 uses Rule A (all 288 experts, all k=2); trunk layers use requested args.rule
        layer_rule = args.rule if main_layer else "A"
        layer_preset = make_qtip_preset(layer_rule)
        signs = {
            p_: generate_rademacher_signs(sp.d_in, seed=1000 * l + s, device=dev)
            for p_, sp, s in (("g", layer_preset.gate, 11), ("u", layer_preset.up, 12), ("d", layer_preset.down, 13))
        }
        luts = {p_: {1: hyb_base[1].clone(), 2: hyb_base[2].clone()} for p_ in ("g", "u", "d")}

        Wg_k = Wg.index_select(0, kept_ids)
        Wu_k = Wu.index_select(0, kept_ids)
        Wd_k = Wd.index_select(0, kept_ids)

        t1 = time.perf_counter()
        chunks, counts_kept = None, None
        if main_layer:
            # Route sequential calibration stream with `pruned_l` masked so calibration matches deployed routing!
            bb.set_pruned(l, pruned_l)
            chunks = cal_inputs(bb, l, streams["cal"], dev, cfg)
            rec["cal_in_s"] = round(time.perf_counter() - t1, 1)

            pos = torch.cat([torch.arange(len(streams["cal"].seqs[k])) for c in chunks for k in range(c["i"], c["j"])])
            keep = pos >= 2
            keep_d = keep.to(dev)
            hs = StreamingHessR39(
                torch.cat([c["x"] for c in chunks])[keep].to(dev),
                torch.cat([c["idx"] for c in chunks])[keep_d],
                torch.cat([c["w"] for c in chunks])[keep_d],
                kept_ids=kept_ids,
                dev=dev,
                n0=args.n0,
                n_tot=cfg.n_routed_experts,
            )
            rec["t_hess"] = round(time.perf_counter() - t1 - rec["cal_in_s"], 1)

            fg = encode_projection_qtip_2d_ldlq(Wg_k, layer_preset.gate, signs["g"], luts["g"], dev, hs.gu_hess, damp=args.damp, scale_mode="hunb")
            fu = encode_projection_qtip_2d_ldlq(Wu_k, layer_preset.up, signs["u"], luts["u"], dev, hs.gu_hess, damp=args.damp, scale_mode="hunb")

            Gh_k = decode_projection_qtip_2d_chunked(fg, layer_preset.gate, signs["g"], luts["g"], dev)  # [R3.9 fix 3]
            Uh_k = decode_projection_qtip_2d_chunked(fu, layer_preset.up, signs["u"], luts["u"], dev)  # [R3.9 fix 3]

            fd = encode_projection_qtip_2d_ldlq(
                Wd_k,
                layer_preset.down,
                signs["d"],
                luts["d"],
                dev,
                lambda a, b: hs.down_hess_and_target(a, b, Wg_k, Wu_k, Wd_k, Gh_k, Uh_k, qz_damp=args.qz_damp),
                damp=args.damp,
                scale_mode="hls",
            )
            del Gh_k, Uh_k
            counts_kept = hs.counts_all[kept_ids.to(dev)].cpu()
            rec["cal_tokens"] = int(keep.sum())
            rec["experts_kept"] = int(kept_ids.numel())
            rec["experts_unseen"] = int((counts_kept == 0).sum())
            del hs, pos, keep, keep_d
        else:
            # MTP Layer 45: data-free L=16 HYB 2D-tiling across all 288 experts
            fg = encode_projection_qtip_2d_ldlq(Wg_k, layer_preset.gate, signs["g"], luts["g"], dev, None, scale_mode="hunb")
            fu = encode_projection_qtip_2d_ldlq(Wu_k, layer_preset.up, signs["u"], luts["u"], dev, None, scale_mode="hunb")
            fd = encode_projection_qtip_2d_ldlq(Wd_k, layer_preset.down, signs["d"], luts["d"], dev, None, scale_mode="hls")

        rec["weight_snr_db"] = {"g": fg["_snr_dense_db"], "u": fu["_snr_dense_db"], "d": fd["_snr_dense_db"]}

        # Popularity row order (`perm_global[r]` = original expert id stored at slab row `r`)
        pop_local = torch.argsort(counts_kept, descending=True, stable=True) if counts_kept is not None else torch.arange(kept_ids.numel())
        perm_global = kept_ids[pop_local]
        fg_ord = {k: (v.index_select(0, pop_local) if torch.is_tensor(v) else v) for k, v in fg.items()}
        fu_ord = {k: (v.index_select(0, pop_local) if torch.is_tensor(v) else v) for k, v in fu.items()}
        fd_ord = {k: (v.index_select(0, pop_local) if torch.is_tensor(v) else v) for k, v in fd.items()}

        out_pt = qtip_layer_path(args.dst, l)
        nbytes = save_qtip_layer_archive(
            path=out_pt,
            rule=layer_rule,
            layer_id=l,
            signs=signs,
            hyb_luts=luts,
            fg=fg_ord,
            fu=fu_ord,
            fd=fd_ord,
            perm_kept=torch.arange(kept_ids.numel(), dtype=torch.long),
            n_experts_total=cfg.n_routed_experts,
            meta={"bpw": round(layer_preset.bpw(), 4), "recipe": "r39_qtip_hyb_2d16_qz", "pruned_ids": pruned_l},
        )
        # Patch global expert IDs into `perm` atomically
        saved_blob = torch.load(out_pt, map_location="cpu")
        saved_blob["perm"] = perm_global.to(torch.int16).contiguous()
        tmp_pt = out_pt + ".tmp"
        with open(tmp_pt, "wb") as fh:
            torch.save(saved_blob, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_pt, out_pt)
        nbytes = os.path.getsize(out_pt)
        del saved_blob, fg, fu, fd, fg_ord, fu_ord, fd_ord, Wg_k, Wu_k, Wd_k
        torch.cuda.empty_cache()

        rec.update({"bytes_file": nbytes, "encode_s": round(time.perf_counter() - t1, 1)})
        old_npz = tcq_layer_path(args.old, l) if args.old else ""

        if main_layer:
            t2 = time.perf_counter()
            rt_new = QtipBuildLayerRuntime(out_pt, dev)
            # NVRTC kernel bit-accuracy gate on real calibration activations
            cos_nvrtc = rt_new.verify_nvrtc_gemv_gate(chunks[0]["x"][:8])
            assert cos_nvrtc > 0.99999, f"Layer {l} NVRTC 16x16 GEMV gate failed: cos={cos_nvrtc}"
            rec["nvrtc_gemv_cos"] = round(cos_nvrtc, 6)

            rt_old = (
                TR.TcqLayerRuntime(TR.TcqArchive(old_npz, mmap_slab=False), dev, slab_in_vram=True)
                if (old_npz and os.path.exists(old_npz))
                else None
            )
            new_moe = lambda X, W_, I_: rt_new.forward_routed_experts(X, W_, I_, swiglu_limit=LIM).float()
            old_moe = (
                (lambda X, W_, I_: rt_old.forward_routed_experts(X, W_, I_, swiglu_limit=LIM).float())
                if rt_old is not None
                else None
            )

            # 1) Advance sequential `cal` stream under `pruned_l`
            bb.set_pruned(l, pruned_l)
            tq = time.perf_counter()
            cal_advance(bb, l, streams["cal"], chunks, dev, new_moe)
            del chunks
            rec["t_cal_adv"] = round(time.perf_counter() - tq, 1)

            # 2) Advance `evn` (new QTIP HYB) and score routed output against exact unpruned FP8 teacher
            acc = {"ex": 0.0, "new": 0.0, "old": 0.0}

            def score_cb(X, W_, I_, Y, pos_):
                m = pos_ >= 2
                # Score against UNPRUNED exact FP8 teacher routing on identical layer input X
                bb.set_pruned(l, [])
                w_ex, idx_ex = bb.route(l, X)
                ye = exact_routed(Wg, Wu, Wd, X, w_ex, idx_ex, dev)[m]
                acc["ex"] += float(ye.pow(2).sum(dtype=torch.float64))
                acc["new"] += float((ye - Y[m]).pow(2).sum(dtype=torch.float64))
                if old_moe is not None and rt_old is not None:
                    old_miss = rt_old.arch.missing.tolist() if rt_old.arch.missing is not None else []
                    bb.set_pruned(l, old_miss)
                    w_o, idx_o = bb.route(l, X)
                    acc["old"] += float((ye - old_moe(X, w_o, idx_o)[m]).pow(2).sum(dtype=torch.float64))
                bb.set_pruned(l, pruned_l)

            tq = time.perf_counter()
            advance_eval(bb, l, streams["evn"], dev, cfg, new_moe, score_cb)
            rec["t_evn"] = round(time.perf_counter() - tq, 1)

            # 3) Advance `evx` (exact unpruned 288-expert teacher stream)
            bb.set_pruned(l, [])
            tq = time.perf_counter()
            advance_eval(bb, l, streams["evx"], dev, cfg, lambda X, W_, I_: exact_routed(Wg, Wu, Wd, X, W_, I_, dev))
            rec["t_evx"] = round(time.perf_counter() - tq, 1)
            if teacher_on:
                tq = time.perf_counter()
                advance_eval_grouped(bb, l, streams["trx"], dev, cfg, lambda X, W_, I_: exact_routed(Wg, Wu, Wd, X, W_, I_, dev), group=args.teacher_group)
                rec["t_trx"] = round(time.perf_counter() - tq, 1)

            # 4) Advance `evo` (old TCQ33 stream)
            if old_moe is not None and rt_old is not None:
                old_miss = rt_old.arch.missing.tolist() if rt_old.arch.missing is not None else []
                bb.set_pruned(l, old_miss)
                tq = time.perf_counter()
                advance_eval(bb, l, streams["evo"], dev, cfg, old_moe)
                rec["t_evo"] = round(time.perf_counter() - tq, 1)
                bb.set_pruned(l, [])
            else:
                meta["evo_valid"] = False

            rec["snr_new_db"] = round(db(acc["ex"], acc["new"]), 3)
            if old_moe is not None:
                rec["snr_old_db"] = round(db(acc["ex"], acc["old"]), 3)
                gains.append(round(rec["snr_new_db"] - rec["snr_old_db"], 3))
            rec["streams_s"] = round(time.perf_counter() - t2, 1)
            del rt_new, rt_old, new_moe, old_moe
            bb.unload_layer(l)
            meta["main_done"] += 1

        if not main_layer and args.mtp_eval and all(x in done for x in range(cfg.first_k_dense_replace, cfg.num_hidden_layers)):
            t4 = time.perf_counter()
            try:
                rec["mtp"] = mtp_acceptance_r39(bb, l, streams, qb_seqs, dev, Wg, Wu, Wd, out_pt, old_npz, TR_mod=TR)
            except Exception as ex_:  # noqa: BLE001
                rec["mtp_error"] = repr(ex_)
            rec["mtp_s"] = round(time.perf_counter() - t4, 1)
            torch.cuda.empty_cache()

        if args.replace_old and old_npz and os.path.exists(old_npz):
            pending_old.append(old_npz)

        prog["layers"][str(l)] = rec
        with open(prog_path, "w", encoding="utf-8") as fh:
            json.dump(prog, fh, indent=2)
        done.append(l)

        if meta["main_done"] % args.ckpt_every == 0 or l in (args.stop_after, args.abort_after) or li == len(todo) - 1:
            t3 = time.perf_counter()
            ck_names = ("evx", "evn", "evo") + (("trx",) if teacher_on else ())
            for name in ck_names:
                with open(os.path.join(ck_dir, f"{name}.pt.tmp"), "wb") as fh:
                    # [R3.9 fix 4, 2026-11-16] The zip container writes a list of per-sequence tensors at 27 MB/s here
                    # (measured: 317 s for the 8.6 GB teacher stream, 440 s per checkpoint); the legacy container
                    # writes the same object at 249 MB/s and torch.load reads both.
                    torch.save({"main_done": meta["main_done"], "h": streams[name].h}, fh, _use_new_zipfile_serialization=False)
                    fh.flush()
                    os.fsync(fh.fileno())
            meta_f = dict(meta, done=list(done), gains=list(gains))
            with open(meta_path + ".tmp", "w", encoding="utf-8") as fh:
                json.dump(meta_f, fh)
                fh.flush()
                os.fsync(fh.fileno())
            for name in ck_names:
                os.replace(os.path.join(ck_dir, f"{name}.pt.tmp"), os.path.join(ck_dir, f"{name}.pt"))
            os.replace(meta_path + ".tmp", meta_path)
            rec["ckpt_s"] = round(time.perf_counter() - t3, 1)
            for op in pending_old:
                if os.path.exists(op):
                    os.remove(op)
            pending_old.clear()

        del Wg, Wu, Wd
        ck.close()
        del ck
        gc.collect()
        torch.cuda.empty_cache()

        if args.fetch == "shards" and not args.keep_src:
            for msg in evict_shards(args.src, args.shard_margin_gib, todo, li, need):
                log(msg, lg)

        el = time.perf_counter() - t_all
        log(
            f"L{l:02d} done: {json.dumps(rec)} | free {shutil.disk_usage(args.dst).free / (2**30):.1f} GiB | "
            f"{el / 60:.1f} min, ETA {el / (li + 1) * (len(todo) - li - 1) / 60:.0f} min",
            lg,
        )

        if l == args.abort_after and gains and not prune_map:
            mg = sum(gains) / len(gains)
            if mg < args.abort_min_gain_db or min(gains) < -0.3:
                log(f"EARLY ABORT: mean new-old gain {mg:.2f} dB, min {min(gains):.2f} dB", lg)
                stop_pf()
                return 2
            log(f"early check passed: mean new-old gain {mg:.2f} dB, min {min(gains):.2f} dB", lg)

        if l == args.stop_after:
            log(f"STOP after L{l:02d} (checkpoint in {ck_dir}; continue with --resume)", lg)
            stop_pf()
            return 3

    # Final full-vocabulary qbench KL + Top-128+Tail teacher cache export for Stage-2B
    res: dict = {
        "per_layer_gain_db": round(sum(gains) / max(len(gains), 1), 3),
        "gains": gains,
        "layers_done": done,
        "evo_valid": meta["evo_valid"],
        "complete": all(l in done for l in range(cfg.first_k_dense_replace, cfg.num_hidden_layers)),
    }
    names = ["evn"] + (["evo"] if meta["evo_valid"] and gains else [])
    kacc = {nm: [0.0, 0, 0] for nm in names}
    top128_ids, top128_logp, tail_probs = [], [], []

    for k in range(len(qb_seqs)):
        logits_ex = bb.final_logits(streams["evx"].h[k].to(dev).float(), last_only=False)[0][0].float()
        lpx_full = F.log_softmax(logits_ex, dim=-1)
        vals128, idx128 = lpx_full.topk(128, dim=-1)
        p_tail = (1.0 - vals128.exp().sum(dim=-1)).clamp_min(0.0)
        top128_ids.append(idx128.to(torch.int32).cpu())
        top128_logp.append(vals128.to(torch.float16).cpu())
        tail_probs.append(p_tail.to(torch.float32).cpu())

        lpx = lpx_full[10:]
        px = lpx.exp()
        for nm in names:
            lpq = F.log_softmax(bb.final_logits(streams[nm].h[k].to(dev).float(), last_only=False)[0][0].float(), dim=-1)[10:]
            kacc[nm][0] += float((px * (lpx - lpq)).sum(dtype=torch.float64))
            kacc[nm][1] += int((lpx.argmax(-1) == lpq.argmax(-1)).sum())
            kacc[nm][2] += int(lpx.shape[0])
            del lpq
        del logits_ex, lpx_full, lpx, px

    for nm, (kl_v, agree, n_pos) in kacc.items():
        res[f"kl_{nm}_vs_exact"] = round(kl_v / max(n_pos, 1), 5)
        res[f"top1_{nm}_vs_exact"] = round(agree / max(n_pos, 1), 4)
    res["qbench_positions"] = kacc["evn"][2]

    torch.save(
        {"top_ids": top128_ids, "top_logp": top128_logp, "tail_prob": tail_probs, "seqs": qb_seqs},
        os.path.join(ck_dir, "qbench_teacher_top128.pt"),
    )
    with open(os.path.join(out_dir, f"r39_qtip_build_eval{args.tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)

    if teacher_on and res["complete"]:
        t_te = time.perf_counter()
        trx, n_tr = streams["trx"], int(tr_ids.shape[0])
        train_dict = TE.extract_top128_dict(
            lambda i: bb.final_logits(trx.h[i].to(dev).float(), last_only=False)[0][0], n_seqs=n_tr, seq_len=args.seq_len, top_k=128
        )
        qb_dict = TE.extract_top128_dict(
            lambda i: bb.final_logits(streams["evx"].h[i][:, : args.seq_len].to(dev).float(), last_only=False)[0][0],
            n_seqs=int(tq_ids.shape[0]), seq_len=args.seq_len, top_k=128,
        )
        os.makedirs(os.path.dirname(os.path.abspath(args.teacher_out)), exist_ok=True)
        with open(args.teacher_out + ".tmp", "wb") as fh:
            torch.save(
                {"format": "r39_compact_teacher_top128_v1", "teacher_mode": "exact_fp8_inbuild", "top_k": 128, "seq_len": args.seq_len,
                 "train_ids": tr_ids, "qb_ids": tq_ids, "train": train_dict, "qbench": qb_dict},
                fh,
            )
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(args.teacher_out + ".tmp", args.teacher_out)
        res["teacher_out"] = args.teacher_out
        log(f"teacher cache written: {args.teacher_out} ({os.path.getsize(args.teacher_out) / 2**20:.1f} MiB, {time.perf_counter() - t_te:.0f}s)", lg)

    log(f"FINAL: {json.dumps(res)}", lg)
    stop_pf()
    log(f"ALL DONE in {(time.perf_counter() - t_all) / 60:.1f} min", lg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
