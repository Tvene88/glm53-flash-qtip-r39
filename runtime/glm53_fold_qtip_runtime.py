#!/usr/bin/env python3
"""
glm53_fold_qtip_runtime.py  [Round 3.9 Production QTIP HYB Runtime, Pinned Pool & CUDA-Graph Serving]
=====================================================================================================
Complete drop-in serving runtime for Round 3.9 `GLM-5.3-Flash` (`layer_XX_qtip_hyb.pt` archives from
`run_qtip_build_r39.py` + `merged_routers_bf16.pt` from `r39_stage2b_global_kl_trainer.py`) on a single
NVIDIA RTX 5090 (32 GiB GDDR7 VRAM + 96 GB DDR5 RAM).

Implements the EXACT contracts expected by `glm53_fold_graphs.GraphDecoder` and `glm53_fold_serve`:
  1. `QtipArchive` & Byte-Locked Flat Slab (`expert_slab_u8` `[E_kept, row_bytes]`):
       Packs `(map, scale, c1, c2, sp)` across `(gate, up, down)` into a 16-byte-aligned contiguous
       `uint8` slab (`6,326,528 B`/expert for `Rule A`; `6,064,384 B`/expert for Sub-70 GB `GU125`)
       with optional `.slab` binary caching so cold rows stream via zero-copy `cudaHostRegister` DMA.
  2. `QtipLayerRuntime` (Graph-Capturable `decode_core` + Rotated Tensor-Core `dense_core`):
       - `decode_core` (`N <= 4`, captured inside `GraphDecoder` `b{l}` and used by `N = 2` MTP verify):
         Launches NVRTC `qtip_tile16x16_hyb_slab_gemv_kernel` (`20 KiB` shared memory -> 4 CTAs / 32
         warps per SM on `sm_120`, 16x `x_rot` reuse, single 64-byte sector load per `[16, 16]` tile)
         + deterministic atomic-free `qtip_sparse_add_kernel`.
       - `dense_core` (`N > 4` prompt prefill):
         Unpacks active experts once into rotated `bf16` `[G, d_out, d_in]` via NVRTC
         `qtip_tile16x16_dense_bf16_kernel` and executes batched Tensor-Core `bf16` GEMMs (`x_rot @ U^T`)
         without running FWHT on weight matrices.
  3. Cross-Platform `QtipPinnedSlotPool` (Windows Win32 + Linux/WSL2 `cudaHostRegister` + Async Prefetch):
       Locks cold expert slabs in host DDR5 RAM via `cudaHostRegister` (Win32 `CreateFileMappingW` or
       POSIX `mmap` + `libcudart.so`, with automatic `pin_memory` fallback), caches hot experts in an
       `18.0-19.5 GiB` GDDR7 VRAM slot pool, overlaps next-layer `route_ids` DMA on a dedicated copy
       stream, and writes resolved slot IDs straight into `GraphDecoder`'s pinned mailbox (`out=`).
  4. In-Place Stage-2B Router Integration (`apply_merged_routers`) & `QtipMoEModelRunner`:
       Patches `merged_routers_bf16.pt` (`AnchoredRouterLoRA` + `e_score_correction_bias`) in place
       into `RealGLM53Backbone` prior to `GraphDecoder.capture()` (`0 B` extra VRAM, `0.0 us` overhead),
       strictly masking all pruned/archive-missing expert indices to `-inf`.

Typical Usage:
  # 1) Standalone GPU self-test (~8s: tests slab packing, pinned pool, CUDAGraph capture & eager==graph parity):
  python glm53_fold_qtip_runtime.py --selftest

  # 2) Live CUDA-graph generation with Round 3.9 QTIP HYB + Stage-2B merged routers:
  python glm53_fold_qtip_runtime.py \
      --backbone-dir C:\models\GLM53-backbone-r34h \
      --fold-dir     C:\models\GLM53-APEX-FOLD-R39-HYB-FT \
      --merged-routers C:\models\GLM53-APEX-FOLD-R39-HYB-FT\merged_routers_bf16.pt \
      --expert-pool-gib 18.5 --register-gib 42.0 --graphs \
      --prompt "Derive the Shannon rate-distortion lower bound for a unit-variance Gaussian source."

[R3.9S] PORT INTO THE ROUND 3.8 PRODUCTION RUNTIME (claude_r39_speed, 2026-11-17). Every changed region carries the tag
[R3.9S]. NOT CHANGED: the three NVRTC kernels and their launch wrappers, the slab layout / packing, the HYB LUT, row
scales, sign vectors + Hadamard (this file's own fwht_full), the SwiGLU clamp (10.0) and every bf16 / FP16 cast.
What changed, in short:
  * QtipLayerRuntime has TcqLayerRuntime's Round 3.6-3.8 interface: decode_core(..., pair_output=), the prompt-pass
    overrides R37_DENSE_MIN_PAIRS / R37_MAX_PAIRS / R37_DENSE_GROUP, prefill_cold (pool v6 cold inserts), the masked
    pair branch of the routing policies (r36_route) and the attribute masked_pairs that glm53_fold_route_r36 /
    glm53_fold_overlap_r36 accept in place of "rule-A TCQ kernel v3".
  * QtipMoEModelRunner is built ON the Round 3.8 runner (glm53_fold_generate.ApexFoldMoEModelRunner): it inherits
    generate() / moe_dispatch() / warmup() / the serve hooks / the two-decoder selection, and only the archive
    loading, the merged routers and the pool choice are its own. The class is created on first use
    (qtip_runner_class), so `--selftest` still imports nothing but torch.
  * Expert pool: R39_POOL=v6 (default when R32_POOL=6, i.e. under the production environment) uses the production
    pool glm53_fold_expert_pool_v6 with all its R35_POOL_* / R37_POOL_REG_* flags and R33_REGISTER_*; R39_POOL=qtip
    (default otherwise) keeps this file's QtipPinnedSlotPool. Its R37_POOL_REG_TOPK handling is fixed (it cut the
    prediction list by expert id; now it is the registered-tier gate of pool v6 and needs scores).
  * New self-test of the port only: `python glm53_fold_qtip_runtime.py --selftest-port` (GPU).

[R3.9K] EXPERT KERNEL v2 (2026-11-17; GPU results: RESULTS.md, section 6). Every changed region carries the tag
[R3.9K]. R39_QTIP_KERNEL unset or `v1` = the code path above, unchanged (the v1 kernel source, its wrappers and the
v1 statements of decode_core are untouched). R39_QTIP_KERNEL=v2 selects section 2b:
  * GEMV kernel v2: per output element the same floating-point operations in the same order as v1 (same states, hash,
    LUT word, sign flip, half2 -> float2, eight fmaf per tile and half-row lane, lane-pair sum, row scale). Changed:
    one load per lane and tile + warp shuffles instead of twelve byte loads, constant shifts (separate k = 1 / k = 2
    loops), load-ahead of the next tiles' code words, only the used HYB table staged, `int row_bytes`.
  * gate and up run in ONE GEMV launch and ONE sparse-add launch (4 launches per layer instead of 6).
  * Knobs (v2 only, all bit-identical): R39_QTIP_WARPS (8), R39_QTIP_GX (64), R39_QTIP_PF (load-ahead in tiles, 4;
    0 = off), R39_QTIP_XSRC (sh = activations staged in shared memory, default; g = read from global memory),
    R39_QTIP_FUSE (1; 0 = gate and up in separate launches).
  * `python glm53_fold_qtip_runtime.py --selftest-kernel [--fold DIR] [--kbench-sweep]` (GPU): v1 == v2 bitwise on
    synthetic and real rows, and a cold micro-benchmark (eager and CUDA-graph replay).

[R3.9M] (2026-11-17, written without a GPU: NOT YET RUN ON ONE) R39_QTIP_N2_WARPS / R39_QTIP_N2_PF / R39_QTIP_N2_GX:
with kernel v2, decode_core calls for more than one token (the N = 2 verify step of MTP decoding) take these launch
settings in place of R39_QTIP_WARPS / _PF / _GX. Launch geometry and load order only, so every output bit is the
same; unset = one setting for all calls, as before. See qtip_kernel_cfg_n2.
"""

from __future__ import annotations

import argparse
import atexit
import ctypes
import glob
import inspect
import json
import math
import mmap
import os
import queue
import shutil
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

LIM = 10.0


def _first_existing(candidates: List[str], fallback: str) -> str:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return fallback


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PKG = _first_existing(
    [
        HERE,
        os.path.join(os.path.dirname(HERE), "runtime"),
        os.path.join(HERE, "claude_r32_apex_fold"),
        os.path.join(os.path.dirname(HERE), "claude_r32_apex_fold"),
        r"C:\models\GLM53-APEX-FOLD-TCQ\claude_r32_apex_fold",
        "/mnt/c/models/GLM53-APEX-FOLD-TCQ/claude_r32_apex_fold",
    ],
    os.path.join(os.path.dirname(HERE), "claude_r32_apex_fold"),
)
DEFAULT_BB = _first_existing(
    [
        r"C:\models\GLM53-backbone-r34h",
        "/mnt/c/models/GLM53-backbone-r34h",
        r"C:\models\GLM53-backbone",
        "/mnt/c/models/GLM53-backbone",
    ],
    r"C:\models\GLM53-backbone-r34h",
)
DEFAULT_FOLD = _first_existing(
    [
        r"C:\models\GLM53-APEX-FOLD-R39-HYB-FT",
        "/mnt/c/models/GLM53-APEX-FOLD-R39-HYB-FT",
        r"C:\models\GLM53-APEX-FOLD-R39-HYB",
        "/mnt/c/models/GLM53-APEX-FOLD-R39-HYB",
        r"C:\models\GLM53-APEX-FOLD-R39-HYB-W16",        # [R3.9S] the archive that exists on this machine (untuned W16)
        "/mnt/c/models/GLM53-APEX-FOLD-R39-HYB-W16",
    ],
    r"C:\models\GLM53-APEX-FOLD-R39-HYB-FT",
)


# ================================================================================================================
# 1. Self-Contained Graph-Safe FWHT & NVRTC CUDA Kernels (Decode Slab GEMV + Rotated BF16 Dense + Sparse Add)
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
    """Graph-capturable Kronecker FWHT across last dim `d_in` (zero host syncs)."""
    orig_shape = x.shape
    d_in = orig_shape[-1]
    xf = x.float()
    if d_in <= 64:
        return torch.matmul(xf, _hadamard(d_in, xf.device)).to(x.dtype)
    log_d = int(round(math.log2(d_in)))
    d1 = 1 << (log_d // 2)
    d2 = d_in // d1
    v2d = xf.reshape(-1, d1, d2)
    out = torch.matmul(_hadamard(d1, xf.device), torch.matmul(v2d, _hadamard(d2, xf.device))).reshape(orig_shape)
    return out.to(x.dtype)


def _compile_nvrtc(name: str, src: str, shared_bytes: int = 0):
    if name not in _NVRTC_CACHE:
        kern = torch.cuda._compile_kernel(src, name)
        if shared_bytes >= 48 * 1024:
            kern.set_shared_memory_config(shared_bytes)
        _NVRTC_CACHE[name] = kern
    return _NVRTC_CACHE[name]


_QTIP_SERVE_CUDA_SRC = r"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>

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

// ------------------------------------------------------------------------------------------------
// 1) Graph-Capturable 16x16 Tile HYB Slab GEMV (Reads directly from uint8[n_slots, row_bytes] pool!)
//    Shared memory: 4 KiB HYB LUT + 16 KiB x_rot = 20 KiB total -> 4 CTAs (32 warps) per SM on sm_120!
// ------------------------------------------------------------------------------------------------
extern "C" __global__ void qtip_tile16x16_hyb_slab_gemv_kernel(
    const unsigned char* __restrict__ rows,
    long long row_bytes,
    const int* __restrict__ slots,
    int off_map,
    int off_scale,
    int off_c1,
    int off_c2,
    int n1,
    int n2,
    const unsigned int* __restrict__ hyb_lut_g,  // [2, 512] packed half2 (k=1 and k=2)
    const float* __restrict__ x_g,               // [1 or P, T] rotated input
    int x_per_pair,
    float* __restrict__ y,                       // [P, d_out]
    int d_out,
    int T)
{
    extern __shared__ unsigned int smem_u32[];
    unsigned int* s_lut = smem_u32;                      // 1024 uint32 = 4 KiB
    float* xs = reinterpret_cast<float*>(smem_u32 + 1024);

    const int pr = blockIdx.y;
    const int sl_id = slots[pr];
    if (sl_id < 0) {
        for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < d_out; r += gridDim.x * blockDim.x) {
            y[(size_t)pr * d_out + r] = 0.0f;
        }
        return;
    }
    const float* xg = x_g + (size_t)(x_per_pair ? pr : 0) * T;
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) s_lut[i] = hyb_lut_g[i];
    for (int i = threadIdx.x; i < T; i += blockDim.x) xs[i] = xg[i];
    __syncthreads();

    const unsigned char* base = rows + (long long)sl_id * row_bytes;
    const unsigned short* rmap = reinterpret_cast<const unsigned short*>(base + off_map);
    const __half* scale = reinterpret_cast<const __half*>(base + off_scale);

    const int lane = threadIdx.x & 31;
    const int r_local = lane >> 1;               // 0..15 row inside 16x16 tile
    const int half_col = lane & 1;               // 0 -> cols 0..7; 1 -> cols 8..15
    const int warps = blockDim.x >> 5;
    const int n_row_tiles = d_out >> 4;
    const int nb_col = T >> 4;
    const int n1_tiles = n1 >> 4;

    const unsigned char* base_c1 = base + off_c1;
    const unsigned char* base_c2 = base + off_c2;

    for (int rt = blockIdx.x * warps + (threadIdx.x >> 5); rt < n_row_tiles; rt += gridDim.x * warps) {
        int k, tile_bytes;
        const unsigned char* row_tile_base;
        if (rt < n1_tiles) {
            k = 1; tile_bytes = 32;
            row_tile_base = base_c1 + (size_t)rt * (size_t)nb_col * 32ULL;
        } else {
            k = 2; tile_bytes = 64;
            row_tile_base = base_c2 + (size_t)(rt - n1_tiles) * (size_t)nb_col * 64ULL;
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

// ------------------------------------------------------------------------------------------------
// 2) Grouped Prefill Dequantizer: Unpacks G active experts' rotated weights straight to BF16 [G, d_out, T]
//    so prompt prefill executes Tensor-Core BF16 GEMMs (`x_rot @ U^T`) with zero weight FWHT!
// ------------------------------------------------------------------------------------------------
extern "C" __global__ void qtip_tile16x16_dense_bf16_kernel(
    const unsigned char* __restrict__ rows,
    long long row_bytes,
    const int* __restrict__ slots,               // [G]
    int off_map,
    int off_scale,
    int off_c1,
    int off_c2,
    int n1,
    int n2,
    const unsigned int* __restrict__ hyb_lut_g,  // [2, 512]
    __nv_bfloat16* __restrict__ W_rot_out,       // [G, d_out, T]
    int d_out,
    int T)
{
    __shared__ unsigned int s_lut[1024];
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) s_lut[i] = hyb_lut_g[i];
    __syncthreads();

    const int g_idx = blockIdx.y;
    const unsigned char* base = rows + (long long)slots[g_idx] * row_bytes;
    const unsigned short* rmap = reinterpret_cast<const unsigned short*>(base + off_map);
    const __half* scale = reinterpret_cast<const __half*>(base + off_scale);

    const int nb_col = T >> 4;
    const int n_row_tiles = d_out >> 4;
    const int n1_tiles = n1 >> 4;
    const int total_tiles = n_row_tiles * nb_col;

    // 128 threads/warp-quad: each thread unpacks one 2D pair (m = 0..127) inside tile `tile_id`
    const int m = threadIdx.x & 127;             // 0..127 pair inside 16x16 tile
    const int sub_tiles_per_cta = blockDim.x >> 7;
    const int sub_id = threadIdx.x >> 7;

    for (int tile_id = blockIdx.x * sub_tiles_per_cta + sub_id;
         tile_id < total_tiles;
         tile_id += gridDim.x * sub_tiles_per_cta)
    {
        int rt = tile_id / nb_col;
        int ct = tile_id - rt * nb_col;
        int k, tile_bytes;
        const unsigned char* tp;
        if (rt < n1_tiles) {
            k = 1; tile_bytes = 32;
            tp = base + off_c1 + ((size_t)rt * (size_t)nb_col + (size_t)ct) * 32ULL;
        } else {
            k = 2; tile_bytes = 64;
            tp = base + off_c2 + ((size_t)(rt - n1_tiles) * (size_t)nb_col + (size_t)ct) * 64ULL;
        }
        const unsigned int* tk = s_lut + (k - 1) * 512;
        unsigned int s16 = read_tailbite_state16(tp, tile_bytes, m * (k << 1));
        float2 w = qtip_hyb_mul1_pair(s16, tk);

        int r_local = m >> 3;
        int c_pair = (m & 7) << 1;
        int stored_r = (rt << 4) + r_local;
        int orig_r = (int)rmap[stored_r];
        float sc = __half2float(scale[stored_r]);

        __nv_bfloat16* dst = W_rot_out + ((size_t)g_idx * (size_t)d_out + (size_t)orig_r) * (size_t)T + (ct << 4) + c_pair;
        dst[0] = __float2bfloat16(w.x * sc);
        dst[1] = __float2bfloat16(w.y * sc);
    }
}

// ------------------------------------------------------------------------------------------------
// 3) Deterministic Atomic-Free Sparse Outlier Add (Identical IEEE-754 order to glm53_fold_kernels.py)
// ------------------------------------------------------------------------------------------------
extern "C" __global__ void qtip_sparse_add_kernel(
    const unsigned char* __restrict__ rows,
    long long row_bytes,
    const int* __restrict__ slots,
    int off_sp,
    int K,
    const float* __restrict__ x,
    int x_stride,
    float* __restrict__ out,
    int out_stride,
    int P)
{
    extern __shared__ int sm[];
    int* s_row = sm;
    float* s_prod = reinterpret_cast<float*>(sm + K);
    const int pr = blockIdx.x;
    if (pr >= P) return;
    const int sl_id = slots[pr];
    if (sl_id < 0) return;
    const int tid = threadIdx.x, nt = blockDim.x;
    const unsigned short* sp = reinterpret_cast<const unsigned short*>(
        rows + (long long)sl_id * row_bytes + off_sp
    );
    const float* xp = x + (long long)pr * (long long)x_stride;
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
        for (int j = 0; j < K; ++j) {
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


def precompile_qtip_kernels() -> float:
    t0 = time.perf_counter()
    _compile_nvrtc("qtip_tile16x16_hyb_slab_gemv_kernel", _QTIP_SERVE_CUDA_SRC, shared_bytes=20 * 1024)
    _compile_nvrtc("qtip_tile16x16_dense_bf16_kernel", _QTIP_SERVE_CUDA_SRC)
    _compile_nvrtc("qtip_sparse_add_kernel", _QTIP_SERVE_CUDA_SRC)
    # [R3.9K] R39_QTIP_KERNEL=v2: compile the v2 kernels here too, so that no compile falls into a graph capture, and
    # leave one line in the load log naming the kernel that runs. Unset / v1: qtip_kernel_cfg() is None, nothing added.
    _k2 = qtip_kernel_cfg()
    if _k2 is not None:
        _k2.precompile()
        print(f"[R3.9K] QTIP expert kernel v2 selected ({_k2.describe()})", flush=True)
    # [R3.9M] R39_QTIP_N2_*: the launch settings of two-token (verify) calls, compiled here as well so that no compile
    # falls into a graph capture. Unset: qtip_kernel_cfg_n2() is None, nothing added.
    _k2n = qtip_kernel_cfg_n2()
    if _k2n is not None and _k2n is not _k2:
        _k2n.precompile()
        print(f"[R3.9M] QTIP expert kernel v2, two-token calls ({_k2n.describe()})", flush=True)
    return time.perf_counter() - t0


# ================================================================================================================
# 2. Byte-Locked QTIP Slab Layout, Archive Loader & NVRTC Launch Helpers
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

    def code_bytes(self) -> Tuple[int, int]:
        nb = self.d_in // 16
        return (self.n1 // 16) * nb * 32, (self.n2 // 16) * nb * 64


@dataclass(frozen=True)
class QtipPreset:
    name: str
    gate: QtipProjSpec
    up: QtipProjSpec
    down: QtipProjSpec

    def layout(self) -> Tuple[Dict[str, Tuple[int, int]], int]:
        """Byte offset & byte length of every field inside one 16-byte-aligned expert slab row."""
        lay: Dict[str, Tuple[int, int]] = {}
        cur = 0
        for p, sp in (("g", self.gate), ("u", self.up), ("d", self.down)):
            c1_b, c2_b = sp.code_bytes()
            for fname, nb in (
                ("map", 2 * sp.d_out),
                ("scale", 2 * sp.d_out),
                ("c1", c1_b),
                ("c2", c2_b),
                ("sp", 6 * sp.k_sparse),
            ):
                lay[f"{p}_{fname}"] = (cur, nb)
                cur += nb
                cur += (-cur) % 2
        cur += (-cur) % 16
        return lay, cur

    @property
    def expert_row_bytes(self) -> int:
        return self.layout()[1]

    def layer_bytes(self, n_experts: int = 288) -> int:
        hdr = (self.gate.d_in + self.up.d_in + self.down.d_in) + (3 * 2 * 2048)
        return hdr + n_experts * self.expert_row_bytes

    def bpw(self, n_experts: int = 288) -> float:
        tot_w = n_experts * sum(sp.d_out * sp.d_in for sp in (self.gate, self.up, self.down))
        return (self.layer_bytes(n_experts) * 8.0) / float(tot_w)


def make_qtip_preset(rule: str, custom_specs: Optional[Tuple[QtipProjSpec, QtipProjSpec, QtipProjSpec]] = None) -> QtipPreset:
    if custom_specs is not None:
        return QtipPreset(f"qtip_hyb_{rule}_custom", *custom_specs)
    if rule == "A":
        n1_gu = 0
    elif rule == "GU125":
        n1_gu = 256
    else:
        raise ValueError(f"Unsupported QTIP rule '{rule}'")
    g = QtipProjSpec("gate", 2048, 4096, n1=n1_gu, k_sparse=64, unbiased=True)
    u = QtipProjSpec("up", 2048, 4096, n1=n1_gu, k_sparse=64, unbiased=True)
    d = QtipProjSpec("down", 4096, 2048, n1=0, k_sparse=256, unbiased=False)
    return QtipPreset(f"qtip_hyb_{rule}_2d16", g, u, d)


def pack_qtip_slab(preset: QtipPreset, fields: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
    """Packs `fields['g'|'u'|'d']` into a contiguous byte-locked `uint8[E_kept, expert_row_bytes]` slab."""
    lay, rb = preset.layout()
    E = int(fields["g"]["map"].shape[0])
    slab = torch.zeros((E, rb), dtype=torch.uint8)
    for p in ("g", "u", "d"):
        fp = fields[p]
        for fname in ("map", "scale", "c1", "c2", "sp"):
            off, nb = lay[f"{p}_{fname}"]
            if nb == 0:
                continue
            t = fp[fname].contiguous()
            b = t.view(torch.uint8).reshape(E, -1) if t.dtype != torch.uint8 else t.reshape(E, -1)
            assert b.shape[1] == nb, f"Field {p}_{fname} byte mismatch: {b.shape[1]} vs {nb}"
            slab[:, off : off + nb] = b
    return slab


class QtipProjLayout:
    """Precomputed byte offsets + static GPU `half2[2, 512]` LUT view for CUDA-graph-capturable launches."""

    def __init__(self, preset: QtipPreset, prefix: str, spec: QtipProjSpec, hyb_lut_dict: Dict[int, torch.Tensor], device: torch.device):
        lay, row_bytes = preset.layout()
        self.row_bytes = int(row_bytes)
        self.off = {f: int(lay[f"{prefix}_{f}"][0]) for f in ("map", "scale", "c1", "c2", "sp")}
        self.n1 = int(spec.n1)
        self.n2 = int(spec.n2)
        self.d_out = int(spec.d_out)
        self.T = int(spec.d_in)
        self.k_sparse = int(spec.k_sparse)
        lut_stack = torch.stack(
            [hyb_lut_dict[1].to(device=device, dtype=torch.float16), hyb_lut_dict[2].to(device=device, dtype=torch.float16)],
            dim=0,
        ).contiguous()
        self.lut_u32 = lut_stack.view(torch.int32)  # [2, 512] static GPU address for CUDA graphs


def qtip_gemv(
    rows: torch.Tensor,
    slots: torch.Tensor,
    lay: QtipProjLayout,
    x_rot: torch.Tensor,
    x_per_pair: bool,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    P = int(slots.numel())
    if out is None:
        out = torch.empty((P, lay.d_out), dtype=torch.float32, device=rows.device)
    shared = (1024 + lay.T) * 4  # 20,480 B for T=4096
    kern = _compile_nvrtc("qtip_tile16x16_hyb_slab_gemv_kernel", _QTIP_SERVE_CUDA_SRC, shared_bytes=shared)
    warps = int(os.environ.get("R39_QTIP_WARPS", "8"))
    gx_cap = int(os.environ.get("R39_QTIP_GX", "64"))
    gx = min(((lay.d_out // 16) + warps - 1) // warps, gx_cap)
    kern(
        grid=(gx, P, 1),
        block=(32 * warps, 1, 1),
        shared_mem=shared,
        args=[
            rows,
            int(rows.stride(0)),
            slots,
            lay.off["map"],
            lay.off["scale"],
            lay.off["c1"],
            lay.off["c2"],
            lay.n1,
            lay.n2,
            lay.lut_u32,
            x_rot.float().contiguous(),
            1 if x_per_pair else 0,
            out,
            lay.d_out,
            lay.T,
        ],
    )
    return out


def qtip_dense_bf16(rows: torch.Tensor, slots: torch.Tensor, lay: QtipProjLayout) -> torch.Tensor:
    G = int(slots.numel())
    out = torch.empty((G, lay.d_out, lay.T), dtype=torch.bfloat16, device=rows.device)
    kern = _compile_nvrtc("qtip_tile16x16_dense_bf16_kernel", _QTIP_SERVE_CUDA_SRC)
    total_tiles = (lay.d_out // 16) * (lay.T // 16)
    gx = min((total_tiles + 1) // 2, 256)
    kern(
        grid=(gx, G, 1),
        block=(256, 1, 1),
        args=[
            rows,
            int(rows.stride(0)),
            slots,
            lay.off["map"],
            lay.off["scale"],
            lay.off["c1"],
            lay.off["c2"],
            lay.n1,
            lay.n2,
            lay.lut_u32,
            out,
            lay.d_out,
            lay.T,
        ],
    )
    return out


def qtip_sparse_add(
    rows: torch.Tensor,
    slots: torch.Tensor,
    lay: QtipProjLayout,
    x_orig: torch.Tensor,
    x_per_pair: bool,
    out: torch.Tensor,
) -> torch.Tensor:
    if lay.k_sparse == 0:
        return out
    P = int(slots.numel())
    K = int(lay.k_sparse)
    xf = x_orig.float()
    if xf.dim() != 2 or xf.stride(1) != 1:
        xf = xf.reshape(-1, xf.shape[-1]).contiguous()
    kern = _compile_nvrtc("qtip_sparse_add_kernel", _QTIP_SERVE_CUDA_SRC)
    nt = min(256, ((K + 31) // 32) * 32)
    kern(
        grid=(P, 1, 1),
        block=(nt, 1, 1),
        shared_mem=8 * K,
        args=[
            rows,
            int(rows.stride(0)),
            slots,
            int(lay.off["sp"]),
            K,
            xf,
            int(xf.stride(0)) if x_per_pair else 0,
            out,
            int(out.stride(0)),
            P,
        ],
    )
    return out


# ================================================================================================================
# 2b. [R3.9K] QTIP expert kernels v2 (R39_QTIP_KERNEL=v2). With the flag unset or v1 nothing in this section runs.
# ================================================================================================================
# [R3.9K] Reason for the section: bring the Round 3.7 kernel work (glm53_fold_tcq_kernels_v4, `e_sh16`) to the QTIP
# GEMV without changing one output bit. What carries over from e_sh16: word loads instead of byte loads, loops with
# constant shifts, a C `int` row stride (torch packs a Python int as a C int), __launch_bounds__(512), layout checks
# before any graph capture. What does NOT carry over: e_sh16's main gain was occupancy (64 KiB -> 8 KiB of shared
# memory with 16,384 one-row warps queued per launch). A QTIP launch has only 1,024 (gate / up) or 2,048 (down)
# warps, each a serial chain of T/16 tiles, so it is latency-bound, not occupancy-bound. Hence the two additions
# that have no TCQ counterpart: load-ahead of the code words, and gate + up in one launch (twice the warps in
# flight, one launch less; the sparse add of gate + up is fused the same way).
_R39K_GEMV_SRC = r"""
#include <cuda_fp16.h>

// [R3.9K] qtip_tile16x16_hyb_slab_gemv_kernel, version 2. Per output element the floating-point work is v1's: the
// same 16-bit states, the same hash and LUT word, the same sign flip, the same half2 -> float2 conversion, the same
// eight fmaf per 16-column tile in the same order for each of the two half-row lanes, the same lane-pair sum and the
// same product with the row scale. What differs is how the code bytes reach a lane and how the work is launched:
//   * one load per lane and tile (k = 2: one 16-bit word, k = 1: one byte) and the neighbour lanes' words by warp
//     shuffle, instead of twelve byte loads; a warp reads every code byte of a tile exactly once;
//   * separate k = 1 and k = 2 loops, so every shift is a constant;
//   * load-ahead (this build: @PF@ tiles; 0 = none): the words of the next group of tiles are requested before the
//     current group is decoded, so the memory latency overlaps with the decode;
//   * one launch may carry two projections of the same shape (gate and up): blockIdx.y >= P selects the second set
//     of offsets / LUT / activations / output;
//   * only the HYB table(s) the projection uses are staged; row_bytes is an int (the wrapper passes a C int).
__device__ __forceinline__ float2 r39k_hyb_pair(unsigned int s16, const unsigned int* __restrict__ lut_u32) {
    unsigned int h = 34038481u * s16 + 76625530u;
    unsigned int raw = lut_u32[(h >> 17) & 511u];
    unsigned int signed_bits = raw ^ (h & 0x80008000u);
    return __half22float2(*reinterpret_cast<const __half2*>(&signed_bits));
}

// code word of tile I for this lane: rp is the row tile's code as 16-bit words (k = 2, 32 per tile) or bytes (k = 1,
// 32 per tile). R39K_LDC clamps I to the last tile of the row tile, so a load-ahead never leaves the row tile.
#define R39K_LD(I) ((unsigned int)rp[(size_t)(I) * 32 + lane])
#define R39K_LDC(I) R39K_LD(((I) < nb_col) ? (I) : (nb_col - 1))

extern "C" __global__ void __launch_bounds__(512) @NAME@(
    const unsigned char* __restrict__ rows,
    int row_bytes,
    const int* __restrict__ slots,
    int off_map_a, int off_scale_a, int off_c1_a, int off_c2_a, int n1_a,
    int off_map_b, int off_scale_b, int off_c1_b, int off_c2_b, int n1_b,
    const unsigned int* __restrict__ lut_a,      // [2, 512] packed half2 (k=1 and k=2), first projection
    const unsigned int* __restrict__ lut_b,      // the same for the second projection (blockIdx.y >= P)
    const float* __restrict__ x_a,               // [1 or P, T] rotated input, first projection
    const float* __restrict__ x_b,
    int x_per_pair,
    float* __restrict__ y_a,                     // [P, d_out]
    float* __restrict__ y_b,
    int d_out,
    int T,
    int P)
{
    extern __shared__ unsigned int smem_u32[];
    unsigned int* s_lut = smem_u32;                      // 1024 uint32 = 4 KiB (k = 1 table, then k = 2 table)

    const int by = blockIdx.y;
    const int second = (by >= P) ? 1 : 0;
    const int pr = by - second * P;
    float* y = second ? y_b : y_a;
    const int sl_id = slots[pr];
    if (sl_id < 0) {
        for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < d_out; r += gridDim.x * blockDim.x) {
            y[(size_t)pr * d_out + r] = 0.0f;
        }
        return;
    }
    const int off_map = second ? off_map_b : off_map_a;
    const int off_scale = second ? off_scale_b : off_scale_a;
    const int off_c1 = second ? off_c1_b : off_c1_a;
    const int off_c2 = second ? off_c2_b : off_c2_a;
    const int n1 = second ? n1_b : n1_a;
    const unsigned int* lut_g = second ? lut_b : lut_a;
    const float* xg = (second ? x_b : x_a) + (size_t)(x_per_pair ? pr : 0) * T;

    const int n_row_tiles = d_out >> 4;
    const int nb_col = T >> 4;
    const int n1_tiles = n1 >> 4;

    // stage the k = 1 table only if the projection has k = 1 row tiles, the k = 2 table only if it has k = 2 ones
    const int lut_lo = (n1_tiles > 0) ? 0 : 512;
    const int lut_hi = (n1_tiles < n_row_tiles) ? 1024 : 512;
    for (int i = lut_lo + threadIdx.x; i < lut_hi; i += blockDim.x) s_lut[i] = lut_g[i];
@X_STAGE@
    __syncthreads();

    const unsigned char* base = rows + (long long)sl_id * (long long)row_bytes;
    const unsigned short* rmap = reinterpret_cast<const unsigned short*>(base + off_map);
    const __half* scale = reinterpret_cast<const __half*>(base + off_scale);
    const unsigned char* base_c1 = base + off_c1;
    const unsigned char* base_c2 = base + off_c2;

    const int lane = threadIdx.x & 31;
    const int r_local = lane >> 1;               // 0..15 row inside 16x16 tile
    const int half_col = lane & 1;               // 0 -> cols 0..7; 1 -> cols 8..15
    const int warps = blockDim.x >> 5;
    const int nx1 = (lane + 1) & 31;             // lane that holds the next code word of the tile (wraps: tail-biting)
    const int nx2 = (lane + 2) & 31;
    const float* xh = xs + (half_col << 3);      // this lane's 8 columns inside every 16-column tile
    const unsigned int* tk1 = s_lut;
    const unsigned int* tk2 = s_lut + 512;

    for (int rt = blockIdx.x * warps + (threadIdx.x >> 5); rt < n_row_tiles; rt += gridDim.x * warps) {
        float acc = 0.0f;
        if (rt < n1_tiles) {
            // k = 1: 32 code bytes per tile; this lane's four states lie in bytes lane, lane+1, lane+2 (mod 32)
            const unsigned char* rp = base_c1 + (size_t)rt * (size_t)nb_col * 32ULL;
@LOOP_K1@
        } else {
            // k = 2: 64 code bytes per tile; this lane's four states lie in bytes 2*lane .. 2*lane+3 (mod 64)
            const unsigned short* rp = reinterpret_cast<const unsigned short*>(
                base_c2 + (size_t)(rt - n1_tiles) * (size_t)nb_col * 64ULL);
@LOOP_K2@
        }

        acc += __shfl_xor_sync(0xffffffff, acc, 1);
        if (half_col == 0) {
            int stored_r = (rt << 4) + r_local;
            y[(size_t)pr * d_out + rmap[stored_r]] = acc * __half2float(scale[stored_r]);
        }
    }
}
"""

# [R3.9K] qtip_sparse_add_kernel for two projections in one launch (gate: blocks 0..P-1, up: blocks P..2P-1). The
# statements of one pair are v1's (__fmul_rn / __fadd_rn, same entry order); only the block index decoding is new.
_R39K_SPARSE2_SRC = r"""
#include <cuda_fp16.h>

extern "C" __global__ void qtip_r39k_sparse_add2_kernel(
    const unsigned char* __restrict__ rows,
    int row_bytes,
    const int* __restrict__ slots,
    int off_sp_a,
    int off_sp_b,
    int K,
    const float* __restrict__ x,
    int x_stride,
    float* __restrict__ out_a,
    float* __restrict__ out_b,
    int out_stride,
    int P)
{
    extern __shared__ int sm[];
    int* s_row = sm;
    float* s_prod = reinterpret_cast<float*>(sm + K);
    const int bx = blockIdx.x;
    const int second = (bx >= P) ? 1 : 0;
    const int pr = bx - second * P;
    if (pr >= P) return;
    const int sl_id = slots[pr];
    if (sl_id < 0) return;
    const int tid = threadIdx.x, nt = blockDim.x;
    const unsigned short* sp = reinterpret_cast<const unsigned short*>(
        rows + (long long)sl_id * (long long)row_bytes + (second ? off_sp_b : off_sp_a)
    );
    const float* xp = x + (long long)pr * (long long)x_stride;
    float* op = (second ? out_b : out_a) + (long long)pr * (long long)out_stride;

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
        for (int j = 0; j < K; ++j) {
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


def _r39k_tile(k: int, cw: str, ct: str, ind: str) -> str:
    """[R3.9K] C statements of one 16-column tile for one lane: the four states out of the lane's code word `cw`
    and its neighbour lanes' words, then v1's two float4 loads, four LUT pairs and eight fmaf in v1's order."""
    if k == 2:
        # v1: state j = 16 bits at tile bit 16*lane + 4*j, bytes taken modulo 64. bw = those 32 bits, MSB first.
        head = [
            f"const unsigned int nw = __shfl_sync(0xffffffffu, {cw}, nx1);",
            f"const unsigned int bw = __byte_perm({cw} | (nw << 16), 0u, 0x0123u);",
            "const unsigned int s0 = bw >> 16;",
            "const unsigned int s1 = (bw >> 12) & 0xFFFFu;",
            "const unsigned int s2 = (bw >> 8) & 0xFFFFu;",
            "const unsigned int s3 = (bw >> 4) & 0xFFFFu;",
        ]
        tk = "tk2"
    else:
        # v1: state j = 16 bits at tile bit 8*lane + 2*j, bytes taken modulo 32. bw = those 24 bits, MSB first.
        head = [
            f"const unsigned int na = __shfl_sync(0xffffffffu, {cw}, nx1);",
            f"const unsigned int nb = __shfl_sync(0xffffffffu, {cw}, nx2);",
            f"const unsigned int bw = ({cw} << 16) | (na << 8) | nb;",
            "const unsigned int s0 = bw >> 8;",
            "const unsigned int s1 = (bw >> 6) & 0xFFFFu;",
            "const unsigned int s2 = (bw >> 4) & 0xFFFFu;",
            "const unsigned int s3 = (bw >> 2) & 0xFFFFu;",
        ]
        tk = "tk1"
    body = head + [
        f"const float* xp = xh + (({ct}) << 4);",
        "const float4 xv0 = *reinterpret_cast<const float4*>(xp);",
        "const float4 xv1 = *reinterpret_cast<const float4*>(xp + 4);",
        f"const float2 w0 = r39k_hyb_pair(s0, {tk});",
        f"const float2 w1 = r39k_hyb_pair(s1, {tk});",
        f"const float2 w2 = r39k_hyb_pair(s2, {tk});",
        f"const float2 w3 = r39k_hyb_pair(s3, {tk});",
        "acc = fmaf(w0.x, xv0.x, acc);",
        "acc = fmaf(w0.y, xv0.y, acc);",
        "acc = fmaf(w1.x, xv0.z, acc);",
        "acc = fmaf(w1.y, xv0.w, acc);",
        "acc = fmaf(w2.x, xv1.x, acc);",
        "acc = fmaf(w2.y, xv1.y, acc);",
        "acc = fmaf(w3.x, xv1.z, acc);",
        "acc = fmaf(w3.y, xv1.w, acc);",
    ]
    return "\n".join(ind + ln for ln in body)


def _r39k_col_loop(k: int, pf: int) -> str:
    """[R3.9K] the loop over the column tiles of one row tile. pf = 0: one load, one tile. pf > 0: tiles in groups
    of pf; the words of the NEXT group (e*) are loaded before the current group (q*) is decoded. The tiles are
    still accumulated in ascending order, so the fmaf chain of every lane is v1's."""
    i12, i16, i20 = " " * 12, " " * 16, " " * 20
    if pf == 0:
        return "\n".join([
            i12 + "for (int ct = 0; ct < nb_col; ++ct) {",
            i16 + "const unsigned int cw = R39K_LD(ct);",
            _r39k_tile(k, "cw", "ct", i16),
            i12 + "}",
        ])
    out = [i12 + "unsigned int " + ", ".join(f"q{j} = R39K_LDC({j})" for j in range(pf)) + ";"]
    out.append(i12 + f"for (int ct = 0; ct < nb_col; ct += {pf}) {{")
    out.append(i16 + "unsigned int " + ", ".join(f"e{j} = 0u" for j in range(pf)) + ";")
    out.append(i16 + f"if (ct + {pf} < nb_col) {{")
    for j in range(pf):
        out.append(i20 + f"e{j} = R39K_LDC(ct + {pf + j});")
    out.append(i16 + "}")
    for j in range(pf):
        out.append(i16 + ("{" if j == 0 else f"if (ct + {j} < nb_col) {{"))
        out.append(_r39k_tile(k, f"q{j}", "ct" if j == 0 else f"ct + {j}", i20))
        out.append(i16 + "}")
    out.append(i16 + " ".join(f"q{j} = e{j};" for j in range(pf)))
    out.append(i12 + "}")
    return "\n".join(out)


def qtip_gemv_v2_source(pf: int, x_shared: bool) -> Tuple[str, str]:
    """[R3.9K] (kernel name, CUDA source) of one build of the v2 GEMV. CPU only (string work)."""
    pf = int(pf)
    name = f"qtip_r39k_gemv_pf{pf}_{'xs' if x_shared else 'xg'}_kernel"
    if x_shared:
        x_stage = (
            "    float* xs_w = reinterpret_cast<float*>(smem_u32 + 1024);   // [T] rotated activations, as v1\n"
            "    for (int i = threadIdx.x; i < T; i += blockDim.x) xs_w[i] = xg[i];\n"
            "    const float* xs = xs_w;"
        )
    else:
        x_stage = "    const float* xs = xg;                                      // activations read from global memory"
    src = (
        _R39K_GEMV_SRC.replace("@NAME@", name)
        .replace("@PF@", str(pf))
        .replace("@X_STAGE@", x_stage)
        .replace("@LOOP_K1@", _r39k_col_loop(1, pf))
        .replace("@LOOP_K2@", _r39k_col_loop(2, pf))
    )
    if "@" in src:
        raise RuntimeError("unreplaced marker in the R3.9K GEMV source")
    return name, src


class QtipKernelV2:
    """[R3.9K] launch state of the v2 expert kernels: geometry, the compiled kernels, the launch wrappers. One
    object serves every layer (qtip_kernel_cfg caches it). The launch path does no host sync and no allocation
    other than the output buffer, which v1 allocates too, so it is usable inside a CUDA graph capture."""

    def __init__(self, warps: int = 8, gx: int = 64, pf: int = 4, x_shared: bool = True, fuse_gu: bool = True):
        self.warps, self.gx, self.pf = int(warps), int(gx), int(pf)
        self.x_shared, self.fuse_gu = bool(x_shared), bool(fuse_gu)
        if not 1 <= self.warps <= 16:
            raise ValueError("R39_QTIP_WARPS must be in 1..16 for kernel v2 (built for at most 512 threads per block)")
        if self.gx < 1:
            raise ValueError("R39_QTIP_GX must be positive")
        if not 0 <= self.pf <= 16:
            raise ValueError("R39_QTIP_PF must be in 0..16")
        self.name, self.src = qtip_gemv_v2_source(self.pf, self.x_shared)
        self._kern = None
        self._kern_sp2 = None

    def describe(self) -> str:
        return (f"warps={self.warps} gx={self.gx} pf={self.pf} x={'shared' if self.x_shared else 'global'} "
                f"fuse_gu={int(self.fuse_gu)}")

    def check_layout(self, lay: "QtipProjLayout") -> None:
        """CPU-only. Raises ValueError if v2 cannot run this projection; called at load, before any graph capture."""
        if lay.d_out % 16 or lay.T % 16 or lay.n1 % 16 or not 0 <= lay.n1 <= lay.d_out:
            raise ValueError(f"QTIP kernel v2: d_out={lay.d_out}, T={lay.T}, n1={lay.n1} are not 16x16-tile sizes")
        if lay.off["map"] % 2 or lay.off["scale"] % 2 or lay.off["c2"] % 2 or lay.off["sp"] % 2 or lay.row_bytes % 2:
            raise ValueError("QTIP kernel v2: the row layout is not 2-byte aligned")
        if lay.row_bytes >= 2 ** 31:
            raise ValueError("QTIP kernel v2: the row size does not fit a 32-bit int")
        if self.x_shared and (1024 + lay.T) * 4 >= 48 * 1024:
            raise ValueError(f"QTIP kernel v2: T={lay.T} needs more than 48 KiB of shared memory; set R39_QTIP_XSRC=g")

    def precompile(self) -> float:
        """Compiles both kernels (needs CUDA). Call before a graph capture."""
        t0 = time.perf_counter()
        self._gemv_kernel()
        self._sparse2_kernel()
        return time.perf_counter() - t0

    def _gemv_kernel(self):
        k = self._kern
        if k is None:
            k = self._kern = _compile_nvrtc(self.name, self.src)
        return k

    def _sparse2_kernel(self):
        k = self._kern_sp2
        if k is None:
            k = self._kern_sp2 = _compile_nvrtc("qtip_r39k_sparse_add2_kernel", _R39K_SPARSE2_SRC)
        return k

    def _launch(self, rows, slots, P, nproj, la, lb, xa, xb, x_per_pair, ya, yb) -> None:
        """One GEMV launch for nproj (1 or 2) projections of the same shape. Host-side checks read attributes only."""
        T, d_out = la.T, la.d_out
        stride = int(rows.stride(0))            # the pool's row stride (may exceed the layer's row size)
        if stride % 2 or stride >= 2 ** 31 or rows.data_ptr() % 2:
            raise ValueError("QTIP kernel v2: rows must be 2-byte aligned with a row stride below 2 GiB")
        if slots.dtype != torch.int32:
            raise ValueError("QTIP kernel v2: slots must be int32")
        need = (P if x_per_pair else 1) * T
        if xa.shape[-1] != T or xb.shape[-1] != T or xa.numel() < need or xb.numel() < need:
            raise ValueError(f"QTIP kernel v2: activation shapes {tuple(xa.shape)} / {tuple(xb.shape)} do not match "
                             f"P={P}, T={T}, per_pair={bool(x_per_pair)}")
        if nproj * P > 65535:
            raise ValueError("QTIP kernel v2: more than 65,535 pair blocks in one launch (lower R37_MAX_PAIRS)")
        if self.x_shared:
            shared = (1024 + T) * 4             # v1's 20,480 B for T = 4096
        else:
            shared = 4096
            if xa.data_ptr() % 16 or xb.data_ptr() % 16:    # float4 loads straight from the activation buffer
                raise ValueError("QTIP kernel v2 (R39_QTIP_XSRC=g): activations must be 16-byte aligned")
        warps = self.warps
        gx = min(((d_out // 16) + warps - 1) // warps, self.gx)
        self._gemv_kernel()(
            grid=(gx, nproj * P, 1),
            block=(32 * warps, 1, 1),
            shared_mem=shared,
            args=[
                rows,
                stride,
                slots,
                la.off["map"], la.off["scale"], la.off["c1"], la.off["c2"], la.n1,
                lb.off["map"], lb.off["scale"], lb.off["c1"], lb.off["c2"], lb.n1,
                la.lut_u32,
                lb.lut_u32,
                xa,
                xb,
                1 if x_per_pair else 0,
                ya,
                yb,
                d_out,
                T,
                P,
            ],
        )

    def gemv(self, rows, slots, lay, x_rot, x_per_pair, out=None):
        """Same contract as qtip_gemv: rows uint8 [S, stride], slots int32 [P], x_rot [1 or P, T] -> fp32 [P, d_out];
        a negative slot gives a zero row."""
        P = int(slots.numel())
        if out is None:
            out = torch.empty((P, lay.d_out), dtype=torch.float32, device=rows.device)
        if P == 0:
            return out
        xf = x_rot.float().contiguous()
        self._launch(rows, slots, P, 1, lay, lay, xf, xf, x_per_pair, out, out)
        return out

    def gemv_gu(self, rows, slots, lg, lu, xg_rot, xu_rot, x_per_pair):
        """Gate and up GEMV. One launch when the two projections have the same shape (they do in both presets),
        else two v2 launches. Returns (g, u), fp32 [P, d_out] each (two halves of one buffer when fused)."""
        P = int(slots.numel())
        if not (self.fuse_gu and lg.d_out == lu.d_out and lg.T == lu.T and 0 < 2 * P <= 65535):
            return self.gemv(rows, slots, lg, xg_rot, x_per_pair), self.gemv(rows, slots, lu, xu_rot, x_per_pair)
        gu = torch.empty((2 * P, lg.d_out), dtype=torch.float32, device=rows.device)
        g, u = gu[:P], gu[P:]
        self._launch(rows, slots, P, 2, lg, lu, xg_rot.float().contiguous(), xu_rot.float().contiguous(), x_per_pair, g, u)
        return g, u

    def gate_up(self, rows, slots, lg, lu, xg_rot, xu_rot, x_orig, x_per_pair):
        """decode_core's gate / up block: GEMV + sparse add of both projections. v1 makes four launches (GEMV g,
        sparse g, GEMV u, sparse u); here it is two when the projections can share launches. Every pair's
        arithmetic is v1's, and gate and up never read each other's output, so the order between them is free."""
        g, u = self.gemv_gu(rows, slots, lg, lu, xg_rot, xu_rot, x_per_pair)
        P = int(slots.numel())
        if P == 0:
            return g, u
        K = int(lg.k_sparse)
        if not (self.fuse_gu and K > 0 and K == int(lu.k_sparse) and int(g.stride(0)) == int(u.stride(0))):
            g = qtip_sparse_add(rows, slots, lg, x_orig, x_per_pair, g)     # v1 launches, in place
            u = qtip_sparse_add(rows, slots, lu, x_orig, x_per_pair, u)
            return g, u
        xf = x_orig.float()
        if xf.dim() != 2 or xf.stride(1) != 1:
            xf = xf.reshape(-1, xf.shape[-1]).contiguous()
        nt = min(256, ((K + 31) // 32) * 32)
        self._sparse2_kernel()(
            grid=(2 * P, 1, 1),
            block=(nt, 1, 1),
            shared_mem=8 * K,
            args=[
                rows,
                int(rows.stride(0)),
                slots,
                int(lg.off["sp"]),
                int(lu.off["sp"]),
                K,
                xf,
                int(xf.stride(0)) if x_per_pair else 0,
                g,
                u,
                int(g.stride(0)),
                P,
            ],
        )
        return g, u


_R39K_CFGS: Dict[Tuple[int, int, int, bool, bool], QtipKernelV2] = {}


def qtip_kernel_cfg(variant: Optional[str] = None, **override) -> Optional[QtipKernelV2]:
    """[R3.9K] the expert-kernel selection. variant None reads R39_QTIP_KERNEL: unset / `v1` -> None (the v1 code
    path), `v2` -> the QtipKernelV2 for R39_QTIP_WARPS (8), R39_QTIP_GX (64), R39_QTIP_PF (4), R39_QTIP_XSRC (sh | g)
    and R39_QTIP_FUSE (1). `override` replaces single fields (used by --selftest-kernel). CPU only."""
    v = (os.environ.get("R39_QTIP_KERNEL", "") if variant is None else variant).strip().lower()
    if v in ("", "v1"):
        return None
    if v != "v2":
        raise ValueError(f"R39_QTIP_KERNEL={v!r}: expected v1 or v2")
    xsrc = os.environ.get("R39_QTIP_XSRC", "sh").strip().lower()
    if xsrc not in ("sh", "g"):
        raise ValueError(f"R39_QTIP_XSRC={xsrc!r}: expected sh or g")
    par = {
        "warps": int(os.environ.get("R39_QTIP_WARPS", "8")),
        "gx": int(os.environ.get("R39_QTIP_GX", "64")),
        "pf": int(os.environ.get("R39_QTIP_PF", "4")),
        "x_shared": xsrc == "sh",
        "fuse_gu": os.environ.get("R39_QTIP_FUSE", "1").strip() != "0",
    }
    par.update(override)
    key = (int(par["warps"]), int(par["gx"]), int(par["pf"]), bool(par["x_shared"]), bool(par["fuse_gu"]))
    cfg = _R39K_CFGS.get(key)
    if cfg is None:
        cfg = _R39K_CFGS[key] = QtipKernelV2(**par)
    return cfg


def qtip_kernel_cfg_n2() -> Optional[QtipKernelV2]:
    """[R3.9M] launch settings of decode_core calls with n_tok > 1 (the N = 2 verify step of MTP decoding; also the
    prompt batches that take the decode kernel), kernel v2 only: R39_QTIP_N2_WARPS / R39_QTIP_N2_PF / R39_QTIP_N2_GX
    replace R39_QTIP_WARPS / _PF / _GX for those calls.
    Reason: a verify call launches twice the thread blocks of an N = 1 call (16 pairs), so the block size and the
    load-ahead that are best for 8 pairs need not be best for 16; the N = 1 graphs keep their own settings. All three
    are launch geometry / load order only: the per-element arithmetic of the v2 kernel does not depend on them (the
    kernel self-test compares every such variant with v1 bit for bit). None when no R39_QTIP_N2_* is set or the kernel
    is v1 (then decode_core is unchanged). CPU only."""
    ov = {}
    for name, field in (("R39_QTIP_N2_WARPS", "warps"), ("R39_QTIP_N2_PF", "pf"), ("R39_QTIP_N2_GX", "gx")):
        v = os.environ.get(name, "").strip()
        if v:
            ov[field] = int(v)
    if not ov:
        return None
    return qtip_kernel_cfg(None, **ov)


class QtipArchive:
    """Loads `layer_XX_qtip_hyb.pt` and exposes the exact same interface as `TcqArchive` (`preset`, `signs`,
    `perm`, `row_of`, `missing`, `slab_file`, `expert_slab_u8`). Caches a raw `.slab` sidecar file when
    `mmap_slab=True` so restarts memory-map 60 GB of cold experts in `< 0.3s` with zero RAM duplication."""

    def __init__(self, pt_path: str, mmap_slab: bool = True):
        blob = torch.load(pt_path, map_location="cpu")
        self.layer_id = int(blob["layer_id"])
        self.rule = str(blob.get("rule", "A"))
        self.meta = blob.get("meta", {})
        self.signs = {p: blob["signs"][p].to(torch.int8).contiguous() for p in ("g", "u", "d")}
        self.hyb_luts = {
            p: {k: blob["hyb_luts"][p][k].to(torch.float16).contiguous() for k in (1, 2)}
            for p in ("g", "u", "d")
        }

        # Infer exact projection dimensions from saved tensors (supports both full 4096/2048 and mini self-test shapes)
        specs = []
        for p, pname, unb in (("g", "gate", True), ("u", "up", True), ("d", "down", False)):
            fp = blob["fields"][p]
            d_out = int(fp["map"].shape[1])
            d_in = int(self.signs[p].numel())
            n1 = int(fp["c1"].shape[1]) * 16
            ks = int(fp["sp"].shape[1])
            specs.append(QtipProjSpec(pname, d_out, d_in, n1=n1, k_sparse=ks, unbiased=unb))
        self.preset = make_qtip_preset(self.rule, custom_specs=(specs[0], specs[1], specs[2]))

        self.n_experts_total = int(blob.get("n_experts_total", 288))
        perm_raw = blob.get("perm", None)
        self.perm = perm_raw.to(torch.long) if perm_raw is not None else None
        self.row_of: Optional[torch.Tensor] = None
        self.missing = torch.empty(0, dtype=torch.long)
        if self.perm is not None:
            if torch.unique(self.perm).numel() != self.perm.numel():
                raise ValueError(f"{pt_path}: duplicate expert id in perm")
            self.row_of = torch.full((self.n_experts_total,), -1, dtype=torch.long)
            self.row_of[self.perm] = torch.arange(self.perm.numel(), dtype=torch.long)
            self.missing = torch.nonzero(self.row_of < 0).flatten()

        E_kept = int(blob["fields"]["g"]["map"].shape[0])
        rb = self.preset.expert_row_bytes
        slab_sidecar = pt_path + ".slab"
        # [R3.9S] R39_SLAB_DIR (unset = unchanged: the sidecar is written next to the .pt): another folder for the
        # 66 GB of sidecars, so a read-only archive folder stays untouched. Same bytes, same file name.
        _slab_dir = os.environ.get("R39_SLAB_DIR", "")
        if _slab_dir:
            os.makedirs(_slab_dir, exist_ok=True)
            slab_sidecar = os.path.join(_slab_dir, os.path.basename(pt_path) + ".slab")
        self.slab_file: Optional[Tuple[str, int]] = None

        if mmap_slab:
            need_write = (
                not os.path.exists(slab_sidecar)
                or os.path.getsize(slab_sidecar) != E_kept * rb
                or os.path.getmtime(slab_sidecar) < os.path.getmtime(pt_path)
            )
            if need_write:
                slab_t = pack_qtip_slab(self.preset, blob["fields"])
                tmp_s = slab_sidecar + ".tmp"
                with open(tmp_s, "wb") as fh:
                    fh.write(slab_t.numpy().tobytes())
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_s, slab_sidecar)
                del slab_t
            mm = np.memmap(slab_sidecar, dtype=np.uint8, mode="r", shape=(E_kept, rb))
            self.expert_slab_u8 = torch.from_numpy(mm)
            self.slab_file = (slab_sidecar, 0)
        else:
            self.expert_slab_u8 = pack_qtip_slab(self.preset, blob["fields"])


# ================================================================================================================
# 3. QtipLayerRuntime (100% Drop-In Compatible with GraphDecoder & ApexFoldMoEModelRunner)
# ================================================================================================================
class QtipLayerRuntime:
    """Per-layer Round 3.9 QTIP HYB serving engine matching `TcqLayerRuntime`'s exact runner & `GraphDecoder` API."""

    def __init__(self, arch: QtipArchive, device: torch.device, slab_in_vram: bool = False):
        self.arch = arch
        self.preset = arch.preset
        self.device = device
        self.layer_id = arch.layer_id
        self.slab_in_vram = slab_in_vram and device.type == "cuda"
        self.expert_slab_u8 = arch.expert_slab_u8.to(device) if self.slab_in_vram else arch.expert_slab_u8
        self.signs = {p: arch.signs[p].to(device=device, dtype=torch.float32) for p in ("g", "u", "d")}
        self.lay = {
            "g": QtipProjLayout(self.preset, "g", self.preset.gate, arch.hyb_luts["g"], device),
            "u": QtipProjLayout(self.preset, "u", self.preset.up, arch.hyb_luts["u"], device),
            "d": QtipProjLayout(self.preset, "d", self.preset.down, arch.hyb_luts["d"], device),
        }
        self.pool = None
        self.use_kernels = True
        self.gain = {"g": 1.0, "u": 1.0, "d": 1.0}
        self.max_pairs_per_launch = 4096
        self.dense_min_pairs = 512
        self.dense_group = 32
        # [R3.9S] the Round 3.7 prompt-pass overrides of TcqLayerRuntime, same names and rules (unset = the three
        # values above, unchanged). R37_DENSE_MIN_PAIRS: pairs (tokens x 8) from which a batch takes the bf16 dense
        # path; a very large value keeps every prompt batch on the decode kernel (the production serving profiles set
        # 1000000000 to bound the prompt pass's memory). R37_MAX_PAIRS: pairs per decode-kernel launch.
        # R37_DENSE_GROUP: experts decoded to dense bf16 at once.
        if os.environ.get("R37_DENSE_MIN_PAIRS", ""):
            self.dense_min_pairs = max(2, int(os.environ["R37_DENSE_MIN_PAIRS"]))
        if os.environ.get("R37_MAX_PAIRS", ""):
            self.max_pairs_per_launch = max(8, int(os.environ["R37_MAX_PAIRS"]))
        if os.environ.get("R37_DENSE_GROUP", ""):
            self.dense_group = max(1, int(os.environ["R37_DENSE_GROUP"]))
        self.n_experts = int(self.expert_slab_u8.shape[0])
        self.row_of = arch.row_of.to(device) if arch.row_of is not None else None
        self._partial = arch.missing.numel() > 0
        # [R3.9S] interface of the Round 3.6-3.8 runner / graphs / route / overlap modules:
        # masked_pairs: a pair whose slot is -1 gives an all-zero output and never addresses a slab row (the GEMV
        #   kernel zero-fills its row, the sparse add returns early) - what glm53_fold_route_r36 (Q) and
        #   glm53_fold_overlap_r36 (K3) otherwise require "rule-A TCQ kernel v3" for.
        # r36_route: set by glm53_fold_route_r36.Router (routed ids may be -1). prefill_cold: set by the runner while
        #   a prompt pass runs with R35_PREFILL_COLD=1 (pool v6: ensure_cold).
        self.masked_pairs = True
        self.r36_route = False
        self.prefill_cold = False
        # [R3.9K] expert kernel selection, read once at load as R37_TCQ_KERNEL is: None (R39_QTIP_KERNEL unset / v1) =
        # the v1 statements of decode_core; a QtipKernelV2 = section 2b. The layout checks are CPU-only and raise
        # here, before any graph is captured. dense_core (bf16 prompt path) is not affected.
        self.r39k = qtip_kernel_cfg()
        if self.r39k is not None:
            for _lay in self.lay.values():
                self.r39k.check_layout(_lay)
        # [R3.9M] optional separate launch settings for two-token calls (R39_QTIP_N2_*; None = use r39k, unchanged)
        self.r39k_n2 = qtip_kernel_cfg_n2() if self.r39k is not None else None
        if self.r39k_n2 is not None:
            for _lay in self.lay.values():
                self.r39k_n2.check_layout(_lay)

    def _check_rows(self, uniq: torch.Tensor) -> None:
        if self.row_of is not None and bool((self.row_of[uniq.to(self.row_of.device)] < 0).any()):
            bad = sorted(set(uniq.cpu().tolist()) & set(self.arch.missing.tolist()))
            raise RuntimeError(f"Layer {self.layer_id}: routed to pruned experts {bad} without slab rows!")

    def decode_core(
        self,
        x: torch.Tensor,
        rows: torch.Tensor,
        pair_slots: torch.Tensor,
        w_pairs: torch.Tensor,
        n_tok: int,
        lim: float,
        row_exact: bool = False,
        pair_output: bool = False,
    ) -> torch.Tensor:
        """CUDA-graph-capturable decode core (`N = 1` inside `GraphDecoder._seg_b`, `N = 2` for MTP verification).
        [R3.9S] pair_output=True (TcqLayerRuntime's Round 3.6 K3 contract, used by glm53_fold_overlap_r36): return the
        per-pair down outputs [N*K, D] BEFORE the routing-weight product and the sum over K; the caller does both.
        False (default) = the unchanged statements."""
        P = int(pair_slots.numel())
        K = P // n_tok
        lg, lu, ld = self.lay["g"], self.lay["u"], self.lay["d"]
        if row_exact and n_tok > 1:
            xg = torch.cat([fwht_full(x[t : t + 1] * self.signs["g"]) for t in range(n_tok)], dim=0)
            xu = torch.cat([fwht_full(x[t : t + 1] * self.signs["u"]) for t in range(n_tok)], dim=0)
        else:
            xg = fwht_full(x * self.signs["g"])
            xu = fwht_full(x * self.signs["u"])

        # [R3.9K] k2 is None with R39_QTIP_KERNEL unset / v1: the four v1 branches below run as before. With v2 the
        # gate / up block is one GEMV launch + one sparse-add launch (same inputs, same per-pair arithmetic).
        k2 = getattr(self, "r39k", None)
        if k2 is not None and n_tok > 1:         # [R3.9M] R39_QTIP_N2_*: own launch settings for two-token calls
            k2 = getattr(self, "r39k_n2", None) or k2
        if k2 is not None:
            if n_tok == 1:
                g, u = k2.gate_up(rows, pair_slots, lg, lu, xg, xu, x, False)
            else:
                g, u = k2.gate_up(rows, pair_slots, lg, lu, xg.repeat_interleave(K, 0), xu.repeat_interleave(K, 0),
                                  x.repeat_interleave(K, 0), True)
        elif n_tok == 1:
            g = qtip_gemv(rows, pair_slots, lg, xg, False)
            g = qtip_sparse_add(rows, pair_slots, lg, x, False, g)
            u = qtip_gemv(rows, pair_slots, lu, xu, False)
            u = qtip_sparse_add(rows, pair_slots, lu, x, False, u)
        else:
            xp = x.repeat_interleave(K, 0)
            g = qtip_gemv(rows, pair_slots, lg, xg.repeat_interleave(K, 0), True)
            g = qtip_sparse_add(rows, pair_slots, lg, xp, True, g)
            u = qtip_gemv(rows, pair_slots, lu, xu.repeat_interleave(K, 0), True)
            u = qtip_sparse_add(rows, pair_slots, lu, xp, True, u)

        if self.gain["g"] != 1.0:
            g = g * self.gain["g"]
        if self.gain["u"] != 1.0:
            u = u * self.gain["u"]

        z = F.silu(g.clamp(max=lim)) * u.clamp(min=-lim, max=lim)
        if row_exact and n_tok > 1:
            zr = torch.cat([fwht_full(z[t * K : (t + 1) * K] * self.signs["d"]) for t in range(n_tok)], dim=0)
        else:
            zr = fwht_full(z * self.signs["d"])
        if k2 is not None:                       # [R3.9K] v2 GEMV for the down projection; its sparse add stays v1's
            d = k2.gemv(rows, pair_slots, ld, zr, True)
        else:
            d = qtip_gemv(rows, pair_slots, ld, zr, True)
        d = qtip_sparse_add(rows, pair_slots, ld, z, True, d)
        if self.gain["d"] != 1.0:
            d = d * self.gain["d"]
        if pair_output:  # [R3.9S] K3 contract: the caller merges hit / miss pairs, then does the same reduction once
            return d
        return (d * w_pairs.view(-1, 1)).view(n_tok, K, -1).sum(dim=1)

    def dense_core(
        self,
        x: torch.Tensor,
        rows: torch.Tensor,
        slot_of_u: torch.Tensor,
        uniq: torch.Tensor,
        inv: torch.Tensor,
        w: torch.Tensor,
        lim: float,
    ) -> torch.Tensor:
        """Grouped rotated-domain BF16 Tensor-Core prefill (`N * K >= 512`)."""
        N, K = inv.shape
        lg, lu, ld = self.lay["g"], self.lay["u"], self.lay["d"]
        xg = fwht_full(x * self.signs["g"]).to(torch.bfloat16)
        xu = fwht_full(x * self.signs["u"]).to(torch.bfloat16)
        out = torch.zeros_like(x)

        inv_f = inv.reshape(-1)
        order = torch.argsort(inv_f, stable=True)
        counts = torch.bincount(inv_f, minlength=int(uniq.numel())).tolist()
        tok_all = order // K
        w_all = w.reshape(-1)[order]
        starts = [0]
        for c in counts:
            starts.append(starts[-1] + c)

        U_cnt = int(uniq.numel())
        for g0 in range(0, U_cnt, self.dense_group):
            g1 = min(U_cnt, g0 + self.dense_group)
            sl = slot_of_u[g0:g1].to(torch.int32)
            Wg = qtip_dense_bf16(rows, sl, lg)
            Wu = qtip_dense_bf16(rows, sl, lu)
            Wd = qtip_dense_bf16(rows, sl, ld)
            for j in range(g1 - g0):
                a, b = starts[g0 + j], starts[g0 + j + 1]
                if b == a:
                    continue
                t = tok_all[a:b]
                n = b - a
                sj = sl[j : j + 1].expand(n).contiguous()
                g = (xg[t] @ Wg[j].T).float()
                g = qtip_sparse_add(rows, sj, lg, x[t], True, g)
                u = (xu[t] @ Wu[j].T).float()
                u = qtip_sparse_add(rows, sj, lu, x[t], True, u)
                if self.gain["g"] != 1.0:
                    g = g * self.gain["g"]
                if self.gain["u"] != 1.0:
                    u = u * self.gain["u"]
                z = F.silu(g.clamp(max=lim)) * u.clamp(min=-lim, max=lim)
                zr = fwht_full(z * self.signs["d"]).to(torch.bfloat16)
                d = (zr @ Wd[j].T).float()
                d = qtip_sparse_add(rows, sj, ld, z, True, d)
                if self.gain["d"] != 1.0:
                    d = d * self.gain["d"]
                out.index_add_(0, t, d * w_all[a:b].unsqueeze(-1))
            del Wg, Wu, Wd
        return out

    def forward_routed_experts(
        self,
        x_flat: torch.Tensor,
        top_weights: torch.Tensor,
        top_idx: torch.Tensor,
        swiglu_limit: float = LIM,
        hobbit_tau: float = 1.0,
    ) -> torch.Tensor:
        orig = x_flat.dtype
        x = x_flat.to(self.device, dtype=torch.float32)
        w = top_weights.to(self.device, dtype=torch.float32)
        i = top_idx.to(self.device, dtype=torch.long)
        N, K = i.shape

        if getattr(self, "r36_route", False):
            # [R3.9S] routing policies of glm53_fold_route_r36 (dropped pairs carry id -1 and weight 0): the statements
            # of TcqLayerRuntime.forward_routed_experts. Never taken unless a Router was installed (R36_TOPK etc.).
            if N <= 4:
                flat = i.reshape(-1).cpu().tolist()
                active = sorted({e for e in flat if e >= 0})
                fetched = self.pool.ensure(self.layer_id, active)
                lookup = {e: j for j, e in enumerate(active)}
                gather = torch.tensor([lookup.get(e, 0) for e in flat], device=self.device, dtype=torch.long)
                slots = fetched.index_select(0, gather)
                slots = torch.where(i.reshape(-1) >= 0, slots, -1).to(torch.int32)   # masked pairs address no slab
                return self.decode_core(x, self.pool.pool, slots, w.reshape(-1), N, swiglu_limit).to(orig)
            # prompt batches: a zero-weight duplicate of a kept expert stands in for every dropped id
            first = i.gather(1, (i >= 0).to(torch.int32).argmax(1, keepdim=True))
            i = torch.where(i >= 0, i, first)

        if N == 1:
            uniq, inv = i[0], torch.arange(K, device=self.device).view(1, K)
        else:
            uniq, inv = torch.unique(i, sorted=True, return_inverse=True)
            inv = inv.view(N, K)

        if self.pool is not None:
            rows = self.pool.pool
            if getattr(self, "prefill_cold", False):
                # [R3.9S] Round 3.5 item 10b (R35_PREFILL_COLD=1, pool v6 only): prompt-only experts go to the pool's
                # cold end. The runner sets the flag only when the pool has ensure_cold.
                slot_of_u = self.pool.ensure_cold(self.layer_id, uniq.cpu().tolist())
            else:
                slot_of_u = self.pool.ensure(self.layer_id, uniq.cpu().tolist())
        elif self.slab_in_vram:
            if self._partial:
                self._check_rows(uniq)
            rows = self.expert_slab_u8
            slot_of_u = (self.row_of[uniq] if self.row_of is not None else uniq).to(torch.int32)
        else:
            if self._partial:
                self._check_rows(uniq)
            ur = (self.row_of[uniq] if self.row_of is not None else uniq).cpu()
            rows = self.expert_slab_u8.index_select(0, ur).to(self.device, non_blocking=True)
            slot_of_u = torch.arange(int(uniq.numel()), device=self.device, dtype=torch.int32)

        if N > 1 and N * K >= self.dense_min_pairs and self.device.type == "cuda":
            return self.dense_core(x, rows, slot_of_u, uniq, inv, w, swiglu_limit).to(orig)

        pair_slots = slot_of_u.index_select(0, inv.reshape(-1)).to(torch.int32)
        w_pairs = w.reshape(-1)
        if N * K <= self.max_pairs_per_launch:
            return self.decode_core(x, rows, pair_slots, w_pairs, N, swiglu_limit).to(orig)

        out = torch.empty((N, x.shape[1]), dtype=torch.float32, device=self.device)
        step = max(1, self.max_pairs_per_launch // K)
        for t0 in range(0, N, step):
            t1 = min(N, t0 + step)
            out[t0:t1] = self.decode_core(
                x[t0:t1], rows, pair_slots[t0 * K : t1 * K], w_pairs[t0 * K : t1 * K], t1 - t0, swiglu_limit
            )
        return out.to(orig)


# ================================================================================================================
# 4. Cross-Platform Pinned DDR5 + VRAM Slot Pool (`QtipPinnedSlotPool`: Win32 + Linux/WSL2 `cudaHostRegister`)
# ================================================================================================================
class _CrossPlatformRegisteredHostTier:
    """Locks cold expert slab rows in physical DDR5 RAM using `cudaHostRegister` across BOTH Windows (`Win32`
    `CreateFileMappingW` + `cudart64_*.dll`) and Linux/WSL2 (`POSIX mmap` + `libcudart.so`), enabling direct
    54+ GB/s PCIe 5.0 x16 `cudaMemcpyAsync` without staging bounce buffers or OS page-cache eviction."""

    def __init__(self) -> None:
        self.cud = self._load_cudart()
        self.is_win = os.name == "nt"
        self.views: Dict[int, dict] = {}
        if self.cud is not None:
            self.cud.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
            self.cud.cudaHostUnregister.argtypes = [ctypes.c_void_p]
            self.cud.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            self.cud.cudaMemcpyAsync.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_void_p,
            ]
        if self.is_win:
            from ctypes import wintypes

            k = ctypes.windll.kernel32
            k.CreateFileW.restype = wintypes.HANDLE
            k.CreateFileW.argtypes = [
                wintypes.LPCWSTR,
                wintypes.DWORD,
                wintypes.DWORD,
                ctypes.c_void_p,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.HANDLE,
            ]
            k.CreateFileMappingW.restype = wintypes.HANDLE
            k.CreateFileMappingW.argtypes = [
                wintypes.HANDLE,
                ctypes.c_void_p,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.LPCWSTR,
            ]
            k.MapViewOfFile.restype = ctypes.c_void_p
            k.MapViewOfFile.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.c_size_t]
            k.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
            self.k32 = k

    @staticmethod
    def _load_cudart():
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        patterns = (
            ["cudart64_*.dll", "cudart*.dll"]
            if os.name == "nt"
            else ["libcudart.so*", "/usr/local/cuda/lib64/libcudart.so*"]
        )
        for pat in patterns:
            full_pat = pat if os.path.isabs(pat) else os.path.join(torch_lib, pat)
            for cand in sorted(glob.glob(full_pat), reverse=True):
                try:
                    return ctypes.WinDLL(cand) if os.name == "nt" else ctypes.CDLL(cand)
                except OSError:
                    continue
        return None

    @staticmethod
    def avail_ram_gib() -> float:
        try:
            if os.name == "nt":
                from ctypes import wintypes

                class MEMSTAT(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", wintypes.DWORD),
                        ("dwMemoryLoad", wintypes.DWORD),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                m = MEMSTAT()
                m.dwLength = ctypes.sizeof(MEMSTAT)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
                return m.ullAvailPhys / (1024.0 ** 3)
            with open("/proc/meminfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) / (1024.0 ** 2)
        except Exception:
            pass
        return 64.0

    def register_layer(self, l: int, path: str, slab_off: int, nbytes: int, scratch_gpu: torch.Tensor) -> bool:
        if self.cud is None or not os.path.exists(path):
            return False
        file_sz = os.path.getsize(path)
        lock_sz = min(file_sz, ((int(slab_off) + int(nbytes) + 65535) // 65536) * 65536)

        if self.is_win:
            hf = self.k32.CreateFileW(path, 0x80000000, 7, None, 3, 0, None)
            hm = self.k32.CreateFileMappingW(hf, None, 2, 0, 0, None)
            base_ptr = self.k32.MapViewOfFile(hm, 4, 0, 0, 0)
            if not base_ptr:
                self.k32.CloseHandle(hm)
                self.k32.CloseHandle(hf)
                return False
            # flag 8 = cudaHostRegisterReadOnly (falls back to 1 = cudaHostRegisterPortable)
            rc = self.cud.cudaHostRegister(base_ptr, lock_sz, 8)
            if rc != 0:
                rc = self.cud.cudaHostRegister(base_ptr, lock_sz, 1)
            if rc != 0:
                self.k32.UnmapViewOfFile(base_ptr)
                self.k32.CloseHandle(hm)
                self.k32.CloseHandle(hf)
                return False
            entry = {"ptr": int(base_ptr), "size": lock_sz, "off": int(slab_off), "hf": hf, "hm": hm}
        else:
            fh = open(path, "rb")
            mm = mmap.mmap(fh.fileno(), lock_sz, access=mmap.ACCESS_READ)
            buf_arr = np.frombuffer(mm, dtype=np.uint8)
            base_ptr = int(buf_arr.ctypes.data)
            rc = self.cud.cudaHostRegister(ctypes.c_void_p(base_ptr), lock_sz, 8)
            if rc != 0:
                rc = self.cud.cudaHostRegister(ctypes.c_void_p(base_ptr), lock_sz, 1)
            if rc != 0:
                mm.close()
                fh.close()
                return False
            entry = {"ptr": base_ptr, "size": lock_sz, "off": int(slab_off), "fh": fh, "mm": mm, "arr": buf_arr}

        # Touch-lock pages once via sequential H2D DMA (`cudaMemcpyHostToDevice == 1`)
        ch = int(scratch_gpu.numel())
        for off in range(0, lock_sz, ch):
            self.cud.cudaMemcpy(
                ctypes.c_void_p(scratch_gpu.data_ptr()),
                ctypes.c_void_p(entry["ptr"] + off),
                min(ch, lock_sz - off),
                1,
            )
        self.views[l] = entry
        return True

    def copy_row_async(self, l: int, slab_row: int, row_bytes: int, dst_ptr: int, stream_handle: int) -> None:
        ent = self.views[l]
        src_addr = ent["ptr"] + ent["off"] + int(slab_row) * int(row_bytes)
        rc = self.cud.cudaMemcpyAsync(
            ctypes.c_void_p(dst_ptr),
            ctypes.c_void_p(src_addr),
            int(row_bytes),
            1,
            ctypes.c_void_p(stream_handle),
        )
        if rc != 0:
            raise RuntimeError(f"cudaMemcpyAsync failed rc={rc} on layer {l} row {slab_row}")

    def close(self) -> None:
        for ent in self.views.values():
            if self.cud is not None:
                self.cud.cudaHostUnregister(ctypes.c_void_p(ent["ptr"]))
            if self.is_win:
                self.k32.UnmapViewOfFile(ctypes.c_void_p(ent["ptr"]))
                self.k32.CloseHandle(ent["hm"])
                self.k32.CloseHandle(ent["hf"])
            else:
                del ent["arr"]
                ent["mm"].close()
                ent["fh"].close()
        self.views.clear()


class QtipPinnedSlotPool:
    """Two-tier GDDR7 VRAM (`pool.pool`) + `cudaHostRegister`-locked DDR5 RAM expert cache implementing the exact
    `ensure(l, experts, out=None)` and `prefetch(l, experts)` contracts required by `GraphDecoder`."""

    def __init__(
        self,
        sources: Dict[int, torch.Tensor],
        row_bytes: int,
        n_slots: int,
        device: torch.device,
        staging_rows: int = 48,
        files: Optional[Dict[int, Tuple[str, int]]] = None,
        row_of: Optional[Dict[int, torch.Tensor]] = None,
        register_gib: float = 0.0,
        register_floor_gib: float = 10.0,
        prewarm_hot_per_layer: int = 0,
    ):
        self.device = device
        self.src_row_bytes = {l: int(t.shape[1]) for l, t in sources.items()}
        self.row_bytes = max([int(row_bytes)] + list(self.src_row_bytes.values()))
        self.n_slots = int(n_slots)
        # [R3.9S] zero-filled (shipped: torch.empty). GraphDecoder's capture warm-up runs decode_core on slot 0 before
        # any expert was fetched; the GEMV kernel writes at y[row_map[r]], so an indeterminate row map could write
        # outside the output. A zero row maps every row to 0 and scales it by 0. Bytes of never-written slots only.
        self.pool = torch.zeros((self.n_slots, self.row_bytes), dtype=torch.uint8, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.sources = sources
        self.supports_release = False
        self.supports_cold = False
        self.reg_topk = int(os.environ.get("R39_POOL_REG_TOPK", os.environ.get("R37_POOL_REG_TOPK", "0")) or 0)
        # [R3.9S] the registered-tier gate needs the prediction scores (as pool v6's does): with reg_topk > 0 the
        # callers (GraphDecoder p{l}, the runner's moe_dispatch) compute and hand them over (prefetch_scored).
        # Shipped: wants_scores was always False and prefetch() cut the id-sorted list to reg_topk entries, i.e. it
        # dropped the predicted experts with the highest ids whatever their rank or tier.
        self.wants_scores = self.reg_topk > 0
        self.row_of = {l: [int(r) for r in v.tolist()] for l, v in (row_of or {}).items()}
        self._missing = {l: {e for e, r in enumerate(ro) if r < 0} for l, ro in self.row_of.items()}
        self._missing = {l: s for l, s in self._missing.items() if s}

        self.slot_of: Dict[Tuple[int, int], int] = {}
        self.lru: "OrderedDict[Tuple[int, int], int]" = OrderedDict()
        self.pinned: Set[Tuple[int, int]] = set()
        self.free: List[int] = list(range(self.n_slots))
        self.lock = threading.Lock()

        self.R = int(staging_rows)
        self.pin_a = torch.empty((self.R, self.row_bytes), dtype=torch.uint8, pin_memory=True)
        self.stg_a = torch.empty((self.R, self.row_bytes), dtype=torch.uint8, device=device)
        self.ev_a: Optional[torch.cuda.Event] = None

        self.pin_b = torch.empty((self.R, self.row_bytes), dtype=torch.uint8, pin_memory=True)
        self.stg_b = torch.empty((self.R, self.row_bytes), dtype=torch.uint8, device=device)
        self.copy_stream = torch.cuda.Stream(device=device)
        self.pending: Dict[Tuple[int, int], threading.Event] = {}
        self.ready_cuda: Dict[Tuple[int, int], torch.cuda.Event] = {}
        self.q: "queue.Queue" = queue.Queue()
        self._prefetched: Set[Tuple[int, int]] = set()

        self.stats = {
            "hits": 0,
            "misses": 0,
            "prefetch_issued": 0,
            "prefetch_useful": 0,
            "prefetch_skipped_reg": 0,   # [R3.9S] registered predictions the reg_topk gate did not prefetch
            "bytes_h2d": 0,
            "bytes_dma_registered": 0,
            "registered_layers": 0,
            "registered_rows": 0,
            "registered_gib": 0.0,
            "t_ensure": 0.0,
            "t_prefetch": 0.0,
        }

        self.reg: Optional[_CrossPlatformRegisteredHostTier] = None
        self.reg_rows: Dict[int, int] = {}
        if files and register_gib > 0.0:
            t0 = time.perf_counter()
            self.reg = _CrossPlatformRegisteredHostTier()
            scratch = torch.empty((64 << 20,), dtype=torch.uint8, device=device)
            n_rows_map = {l: int(self.sources[l].shape[0]) for l in files}
            rows_budget = int(register_gib * (1024 ** 3) // self.row_bytes)
            per_layer_quota = max(1, rows_budget // max(len(files), 1))
            locked_bytes = 0
            for l in sorted(files):
                rb_l = self.src_row_bytes[l]
                k_lock = min(n_rows_map[l], per_layer_quota)
                gib_l = (k_lock * rb_l) / (1024.0 ** 3)
                if self.reg.avail_ram_gib() - gib_l < register_floor_gib:
                    break
                path, off = files[l]
                if self.reg.register_layer(l, path, off, k_lock * rb_l, scratch):
                    self.reg_rows[l] = k_lock
                    locked_bytes += k_lock * rb_l
            del scratch
            torch.cuda.synchronize(device)
            self.stats["registered_layers"] = len(self.reg_rows)
            self.stats["registered_rows"] = int(sum(self.reg_rows.values()))
            self.stats["registered_gib"] = round(locked_bytes / (1024.0 ** 3), 2)
            self.stats["register_s"] = round(time.perf_counter() - t0, 2)
            atexit.register(self.reg.close)

        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

        # Pre-seed popularity-ranked top experts (`slab_row = 0..K_hot-1`) into VRAM slots at startup
        if prewarm_hot_per_layer > 0:
            for l in sorted(self.sources):
                ro = self.row_of.get(l)
                if ro is not None:
                    hot_e = [e for e, r in enumerate(ro) if 0 <= r < prewarm_hot_per_layer]
                else:
                    hot_e = list(range(min(prewarm_hot_per_layer, int(self.sources[l].shape[0]))))
                if hot_e and len(self.free) >= len(hot_e):
                    self.ensure(l, hot_e)

    def _evict_one(self, locked: set) -> int:
        for key in self.lru:
            if key in self.pinned or key in locked or key in self.pending:
                continue
            ev = self.ready_cuda.get(key)
            if ev is not None and not ev.query():
                continue
            slot = self.lru.pop(key)
            del self.slot_of[key]
            self._prefetched.discard(key)
            self.ready_cuda.pop(key, None)
            return slot
        raise RuntimeError("QtipPinnedSlotPool: no evictable VRAM slot available")

    def _alloc(self, key: Tuple[int, int], locked: set) -> int:
        slot = self.free.pop() if self.free else self._evict_one(locked)
        self.slot_of[key] = slot
        self.lru[key] = slot
        return slot

    def _check_rows(self, l: int, keys: List[Tuple[int, int]]) -> None:
        ms = self._missing.get(l)
        if ms:
            bad = sorted({e for _, e in keys if e in ms})
            if bad:
                raise RuntimeError(f"QtipPinnedSlotPool: layer {l} routed to pruned experts {bad} (row_of = -1)!")

    def _gather_cpu(self, l: int, experts: List[int], pin_buf: torch.Tensor) -> None:
        src = self.sources[l]
        rb_l = self.src_row_bytes[l]
        ro = self.row_of.get(l)
        rows_idx = [ro[e] for e in experts] if ro is not None else experts
        torch.index_select(src, 0, torch.tensor(rows_idx, dtype=torch.long), out=pin_buf[: len(experts), :rb_l])

    def _sync_fetch(self, l: int, experts: List[int], slots: List[int]) -> None:
        rb_l = self.src_row_bytes[l]
        for c0 in range(0, len(experts), self.R):
            ex, sl = experts[c0 : c0 + self.R], slots[c0 : c0 + self.R]
            m = len(ex)
            if self.ev_a is not None:
                self.ev_a.synchronize()
            src = self.sources[l]
            kreg = self.reg_rows.get(l, 0)
            if kreg and self.reg is not None:
                stream_h = torch.cuda.current_stream(self.device).cuda_stream
                ro = self.row_of.get(l)
                rest_e, rest_s = [], []
                for e, s_ in zip(ex, sl):
                    r_ = ro[e] if ro is not None else e
                    if r_ < kreg:
                        self.reg.copy_row_async(l, r_, rb_l, self.pool[s_].data_ptr(), stream_h)
                    else:
                        rest_e.append(e)
                        rest_s.append(s_)
                nd = m - len(rest_e)
                self.stats["bytes_dma_registered"] += nd * rb_l
                self.stats["bytes_h2d"] += nd * rb_l
                if not rest_e:
                    continue
                ex, sl, m = rest_e, rest_s, len(rest_e)

            slot_t = torch.tensor(sl, dtype=torch.long, pin_memory=True).to(self.device, non_blocking=True)
            if src.is_cuda:
                ro = self.row_of.get(l)
                exr = [ro[e] for e in ex] if ro is not None else ex
                picked = src.index_select(0, torch.tensor(exr, dtype=torch.long, device=self.device))
                if rb_l == self.row_bytes:
                    self.pool.index_copy_(0, slot_t, picked)
                else:
                    self.pool[:, :rb_l].index_copy_(0, slot_t, picked)
            else:
                self._gather_cpu(l, ex, self.pin_a)
                self.stg_a[:m, :rb_l].copy_(self.pin_a[:m, :rb_l], non_blocking=True)
                if rb_l == self.row_bytes:
                    self.pool.index_copy_(0, slot_t, self.stg_a[:m])
                else:
                    self.pool[:, :rb_l].index_copy_(0, slot_t, self.stg_a[:m, :rb_l])
            self.ev_a = torch.cuda.Event()
            self.ev_a.record()
            self.stats["bytes_h2d"] += m * rb_l

    def _worker_loop(self) -> None:
        torch.cuda.set_device(self.device.index if self.device.index is not None else torch.cuda.current_device())
        while True:
            item = self.q.get()
            if item is None:
                self.q.task_done()
                return
            try:
                l, experts, slots, keys = item
                rb_l = self.src_row_bytes[l]
                for c0 in range(0, len(experts), self.R):
                    ex, sl, ks = experts[c0 : c0 + self.R], slots[c0 : c0 + self.R], keys[c0 : c0 + self.R]
                    m = len(ex)
                    src = self.sources[l]
                    with torch.cuda.stream(self.copy_stream):
                        kreg = self.reg_rows.get(l, 0)
                        if kreg and self.reg is not None:
                            stream_h = self.copy_stream.cuda_stream
                            ro = self.row_of.get(l)
                            rest_e, rest_s = [], []
                            for e, s_ in zip(ex, sl):
                                r_ = ro[e] if ro is not None else e
                                if r_ < kreg:
                                    self.reg.copy_row_async(l, r_, rb_l, self.pool[s_].data_ptr(), stream_h)
                                else:
                                    rest_e.append(e)
                                    rest_s.append(s_)
                            self.stats["bytes_dma_registered"] += (m - len(rest_e)) * rb_l
                            ex, sl = rest_e, rest_s
                        mm = len(ex)
                        if mm > 0:
                            slot_t = torch.tensor(sl, dtype=torch.long, pin_memory=True).to(self.device, non_blocking=True)
                            if src.is_cuda:
                                ro_ = self.row_of.get(l)
                                exr = [ro_[e] for e in ex] if ro_ is not None else ex
                                picked = src.index_select(0, torch.tensor(exr, dtype=torch.long, device=self.device))
                                if rb_l == self.row_bytes:
                                    self.pool.index_copy_(0, slot_t, picked)
                                else:
                                    self.pool[:, :rb_l].index_copy_(0, slot_t, picked)
                            else:
                                self._gather_cpu(l, ex, self.pin_b)
                                self.stg_b[:mm, :rb_l].copy_(self.pin_b[:mm, :rb_l], non_blocking=True)
                                if rb_l == self.row_bytes:
                                    self.pool.index_copy_(0, slot_t, self.stg_b[:mm])
                                else:
                                    self.pool[:, :rb_l].index_copy_(0, slot_t, self.stg_b[:mm, :rb_l])
                        ev = torch.cuda.Event()
                        ev.record(self.copy_stream)
                    with self.lock:
                        for k in ks:
                            self.ready_cuda[k] = ev
                            e_ev = self.pending.pop(k, None)
                            if e_ev is not None:
                                e_ev.set()
                    self.stats["bytes_h2d"] += m * rb_l
                    ev.synchronize()
            finally:
                self.q.task_done()

    def ensure(self, l: int, experts: Iterable[int], out=None):
        """Resolves VRAM slot indices for `(l, experts)`. When `out` is passed (`GraphDecoder.slots_np[l]`),
        writes int32 slot IDs directly into the pinned host mailbox without allocating a tensor."""
        t0 = time.perf_counter()
        keys = [(l, int(e)) for e in experts]
        if self._missing:
            self._check_rows(l, keys)
        locked = set(keys)
        miss_e: List[int] = []
        miss_s: List[int] = []
        waits: List[threading.Event] = []
        cuda_waits: List[torch.cuda.Event] = []
        slots: List[int] = []

        with self.lock:
            for key in keys:
                if key in self.slot_of:
                    self.stats["hits"] += 1
                    if key in self._prefetched:
                        self.stats["prefetch_useful"] += 1
                        self._prefetched.discard(key)
                    self.lru.move_to_end(key)
                    if key in self.pending:
                        waits.append(self.pending[key])
                    elif key in self.ready_cuda:
                        cuda_waits.append(self.ready_cuda.pop(key))
                    slots.append(self.slot_of[key])
                else:
                    self.stats["misses"] += 1
                    s = self._alloc(key, locked)
                    miss_e.append(key[1])
                    miss_s.append(s)
                    slots.append(s)

        if miss_e:
            self._sync_fetch(l, miss_e, miss_s)
        if waits:
            for w in waits:
                w.wait()
            with self.lock:
                for key in keys:
                    ev = self.ready_cuda.pop(key, None)
                    if ev is not None:
                        cuda_waits.append(ev)

        cur_s = torch.cuda.current_stream(self.device)
        for ev in cuda_waits:
            if not ev.query():
                cur_s.wait_event(ev)

        if out is not None:
            out[: len(slots)] = slots
            res = out
        else:
            res = torch.tensor(slots, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        self.stats["t_ensure"] += time.perf_counter() - t0
        return res

    def _is_registered(self, l: int, e: int) -> bool:
        """[R3.9S] expert e of layer l lies in the page-locked rows (its miss is one direct DMA, no host gather)."""
        ro = self.row_of.get(l)
        r = ro[e] if ro is not None else e
        return 0 <= r < self.reg_rows.get(l, 0)

    def prefetch(self, l: int, experts: Iterable[int], protect: Iterable[Tuple[int, int]] = (),
                 scores=None, score_ref=None) -> int:
        t0 = time.perf_counter()
        exp_list = [int(e) for e in experts]
        # [R3.9S] FIX of the shipped reg_topk handling (it cut the id-sorted list: 2 of 8 predictions lost by expert
        # id). Now the rule of pool v6's registered-tier gate (R37_POOL_REG_TOPK, glm53_fold_expert_pool_v6._rank_all):
        # with scores, the distinct predictions are ranked by score (minus score_ref when given); a prediction that
        # lies in the REGISTERED rows is prefetched only when its rank is below reg_topk (its miss, if it is routed
        # after all, is a cheap direct DMA inside ensure()); predictions outside the registered rows are always
        # prefetched. Without scores, or with reg_topk = 0, every prediction is prefetched.
        skip_reg: Set[int] = set()
        if self.reg_topk > 0 and scores is not None:
            sc = [float(v) for v in scores]
            if len(sc) != len(exp_list):
                raise ValueError(f"prefetch: {len(sc)} scores for {len(exp_list)} experts")
            if score_ref is None:
                refs = None
            elif isinstance(score_ref, (int, float)):
                refs = [float(score_ref)] * len(sc)
            else:
                refs = [float(v) for v in score_ref]
            best: Dict[int, float] = {}
            for j, e in enumerate(exp_list):
                v = sc[j] - refs[j] if refs is not None else sc[j]
                if e not in best or v > best[e]:
                    best[e] = v
            exp_list = sorted(best, key=lambda e: -best[e])
            skip_reg = {e for rank, e in enumerate(exp_list) if rank >= self.reg_topk and self._is_registered(l, e)}
        keys = [(l, e) for e in exp_list]
        if self._missing:
            self._check_rows(l, keys)
        locked = set(keys) | set(protect)
        ex, sl, ks = [], [], []
        with self.lock:
            for key in keys:
                if key in self.slot_of:
                    continue
                if key[1] in skip_reg:                     # [R3.9S] registered-tier gate (see above)
                    self.stats["prefetch_skipped_reg"] += 1
                    continue
                s = self._alloc(key, locked)
                self.pending[key] = threading.Event()
                self._prefetched.add(key)
                ex.append(key[1])
                sl.append(s)
                ks.append(key)
        if ex:
            self.q.put((l, ex, sl, ks))
            self.stats["prefetch_issued"] += len(ex)
        self.stats["t_prefetch"] += time.perf_counter() - t0
        return len(ex)

    def prefetch_scored(self, l: int, experts: Iterable[int], scores=None, ref_score=None, protect: Iterable[Tuple[int, int]] = ()) -> int:
        # [R3.9S] the scores are used now (registered-tier gate); shipped: dropped here
        return self.prefetch(l, experts, protect=protect, scores=scores, score_ref=ref_score)

    def quiesce(self, timeout: float = 120.0) -> float:
        t0 = time.perf_counter()
        while not self.q.empty() or self.pending:
            if time.perf_counter() - t0 > timeout:
                raise RuntimeError(f"QtipPinnedSlotPool.quiesce timed out after {timeout:.0f}s")
            time.sleep(0.001)
        if self.device.type == "cuda":
            self.copy_stream.synchronize()
            torch.cuda.synchronize(self.device)
        return time.perf_counter() - t0

    def hit_rate(self) -> float:
        tot = self.stats["hits"] + self.stats["misses"]
        return self.stats["hits"] / tot if tot else 0.0

    def close(self) -> None:
        self.q.put(None)
        if self.reg is not None:
            self.reg.close()


# ================================================================================================================
# 5. In-Place Stage-2B Router Patcher & Complete `QtipMoEModelRunner` (Prefill + GraphDecode + N=2 MTP)
# ================================================================================================================
@torch.no_grad()
def apply_merged_routers(
    bb,
    merged_routers_path: str,
    archive_missing: Dict[int, List[int]],
    extra_prune_spec: Optional[Dict[int, List[int]]] = None,
) -> int:
    """Loads `merged_routers_bf16.pt` (from `r39_stage2b_global_kl_trainer.py`) and patches `gate_w` & `gate_b`
    IN PLACE inside `RealGLM53Backbone` before `GraphDecoder.capture()` (`0 B` extra VRAM, `0.0 us` overhead)."""
    state = torch.load(merged_routers_path, map_location="cpu") if (merged_routers_path and os.path.exists(merged_routers_path)) else {}
    n_patched = 0
    all_moe = sorted(set(archive_missing.keys()) | set((extra_prune_spec or {}).keys()) | {
        int(k.split(".")[3]) for k in state.keys() if k.startswith("model.language_model.layers.")
    })
    for l in all_moe:
        if l >= bb.cfg.num_hidden_layers + getattr(bb.cfg, "num_nextn_predict_layers", 1):
            continue
        w = bb.layer(l)
        if "gate_w" not in w:
            continue
        wk = f"model.language_model.layers.{l}.mlp.gate.weight"
        bk = f"model.language_model.layers.{l}.mlp.gate.e_score_correction_bias"
        if wk in state:
            w["gate_w"].copy_(state[wk].to(device=w["gate_w"].device, dtype=w["gate_w"].dtype))
            n_patched += 1
        if bk in state:
            b_src = state[bk].to(device=w["gate_b"].device, dtype=w["gate_b"].dtype)
            # Preserve finite tuned biases in `_gate_b0`, then apply `-inf` prune mask cleanly via `set_pruned`
            finite_b = torch.where(torch.isfinite(b_src), b_src, torch.zeros_like(b_src))
            bb._gate_b0[l] = finite_b.clone()
            w["gate_b"].copy_(finite_b)
        masked = sorted(set(archive_missing.get(l, [])) | set((extra_prune_spec or {}).get(l, [])))
        bb.set_pruned(l, masked)
    return n_patched


class _QtipRunnerMixin:
    """Production Round 3.9 serving runner combining `RealGLM53Backbone`, 42+1 `QtipLayerRuntime` archives,
    in-place `merged_routers_bf16.pt`, an expert pool, `GraphDecoder` (N=1 & N=2 MTP), and `SERVE_HOOKS`.

    [R3.9S] REWRITTEN AS A LAYER ON THE ROUND 3.8 RUNNER. The shipped class was a stand-alone copy of the Round 3.5
    runner (its own generate / moe_dispatch / warmup / perplexity). Here only what is specific to the QTIP archive is
    kept (archive loading, merged routers, pool choice, page-cache warm-up of the sidecars); generate(), moe_dispatch()
    (scored prediction for the pool gates, prefill prefetch, routing policies), warmup(), set_prune(), the serve hooks
    H1-H6, the two-decoder selection (R37_DUAL_GRAPHS) and evaluate_perplexity() are the Round 3.8 methods of
    glm53_fold_generate.ApexFoldMoEModelRunner, which qtip_runner_class() joins below this mixin. The constructor does
    not call the base constructor (that one loads `.npz` archives); it sets every attribute the base methods read."""

    SERVE_HOOKS = ("state", "skip_prefill")

    def __init__(
        self,
        backbone_dir: str,
        fold_dir: str,
        merged_routers_path: str = "",
        device: str = "cuda",
        backbone_mode: str = "fp8",
        expert_pool_gib: float = 18.0,
        register_gib: Optional[float] = None,
        prefetch: bool = True,
        capture_graphs: bool = True,
        t_max: int = 2048,
        load_mtp: bool = False,
        vocab_keep=None,
        prune_spec=None,
        prefix_cache: bool = True,
        pkg_dir: str = DEFAULT_PKG,
        sync_timing: bool = False,
        fp8_diet: bool = False,
        max_layers: Optional[int] = None,
        hobbit_tau: float = 1.0,
        pool_kind: Optional[str] = None,
        fast: bool = True,
        **_r38_only,
    ):
        # [R3.9S] signature: the shipped keywords in the shipped order (register_gib=None now means "R33_REGISTER_GIB
        # if set, else the shipped 42.0"), then the Round 3.8 runner's keywords that glm53_fold_generate.main() passes
        # (sync_timing, fp8_diet, max_layers, hobbit_tau, fast); pool_kind = "v6" | "qtip" | None (env, see below);
        # _r38_only swallows the ones without a meaning here (vram_budget_gib, cold_mode, host_cache_gib).
        if os.path.isdir(pkg_dir) and pkg_dir not in sys.path:
            sys.path.insert(0, pkg_dir)
        if not fast:
            raise ValueError("QtipMoEModelRunner has no --r31-runtime path (QTIP layers exist only as NVRTC kernels)")
        if float(expert_pool_gib) <= 0.0:
            raise ValueError("QtipMoEModelRunner needs the expert pool (--expert-pool-gib > 0)")

        import glm53_fold_backbone as FB  # type: ignore
        import glm53_fold_kernels as KR  # type: ignore
        from glm53_fold_config import GLM53FlashModelConfig  # type: ignore
        from glm53_fold_graphs import GraphDecoder  # type: ignore

        # [R3.9S] helpers of the module that defines the base class (glm53_fold_generate, or __main__ when that file
        # is the entry point: importing it by name there would load a second copy)
        _gen = sys.modules[type(self)._R38_BASE.__module__]
        load_prune_spec, load_vocab_keep = _gen.load_prune_spec, _gen.load_vocab_keep

        self._FB = FB
        self.cfg = GLM53FlashModelConfig()
        self.device = torch.device(device)
        self.prefetch = bool(prefetch)
        self.load_mtp = bool(load_mtp)
        self.moe_time_s = 0.0
        # [R3.9S] attributes the Round 3.8 base methods read (ApexFoldMoEModelRunner.__init__ sets them; same values).
        # max_layers: shipped cfg.num_hidden_layers; None gives the same layer count in bb.forward and is what the
        # base uses. sync_timing: shipped moe_dispatch had no device syncs = False.
        self.max_layers = max_layers
        self.hobbit_tau = hobbit_tau
        self.fast = True
        self.sync_timing = bool(sync_timing)
        self.expert_pool_gib = float(expert_pool_gib)
        self.host_cache_gib = 0.0
        self.vocab_info = None
        self.pool = None
        self.shared_dma = None
        self.tcq_rule = None
        self.graph_decoder = None
        self.graph_decoder_std = None           # R37_DUAL_GRAPHS: the window decoder
        self.graph_decoder_long = None          # R37_DUAL_GRAPHS: the long-context decoder
        self.prefill_prefetch = False           # R35_PREFILL_PREFETCH (set below with the pool)
        self.prefill_pf_min = 1
        self.prefill_pf_max = 0
        self._pf_cnt_host = None
        self._pf_stats = {"layers": 0, "predicted": 0, "queued": 0}
        self.prefill_cold = False               # R35_PREFILL_COLD (set below with the pool)
        self.continue_past_tmax = os.environ.get("R35_CONTINUE_PAST_TMAX", "0") == "1"
        self._prefill_probe = None
        self.r36_router = None

        KR.precompile(fp8_rows=(vocab_keep is not None))
        precompile_qtip_kernels()

        t0 = time.perf_counter()
        self.ckpt = FB.RealGLM53Checkpoint(backbone_dir)
        self.bb = FB.RealGLM53Backbone(self.ckpt, self.cfg, device=device, backbone_mode=backbone_mode)
        self.bb.fp8_diet = bool(fp8_diet)       # [R3.9S] --fp8-diet of the Round 3.8 CLI (shipped: never set = False)
        self.bb.load_global()

        if vocab_keep is not None:
            head_rows = int(getattr(self.bb.head, "qweight", self.bb.head).shape[0])
            self.vocab_info = self.bb.set_vocab_keep(
                load_vocab_keep(vocab_keep, vocab_rows=head_rows, tokenizer_json=os.path.join(backbone_dir, "tokenizer.json"))
            )

        n_main = self.cfg.num_hidden_layers
        n_load = n_main + (1 if self.load_mtp else 0)
        if max_layers is not None:              # [R3.9S] --max-layers of the Round 3.8 CLI (debug runs), as the base
            n_load = int(max_layers)
        for l in range(n_load):
            if l < n_main or os.path.exists(os.path.join(fold_dir, f"layer_{l:02d}_qtip_hyb.pt")):
                self.bb.layer(l)
        self.backbone_load_s = time.perf_counter() - t0

        # Load QtipArchive + QtipLayerRuntime across all MoE layers (3..44 + optional 45 MTP)
        self.moe_layers: Dict[int, QtipLayerRuntime] = {}
        self.archive_missing: Dict[int, List[int]] = {}
        for l in self.cfg.moe_layer_ids:
            if not self.load_mtp and l >= n_main:
                continue
            if max_layers is not None and l >= int(max_layers) and l != n_main:
                continue
            pt_p = os.path.join(fold_dir, f"layer_{l:02d}_qtip_hyb.pt")
            if not os.path.exists(pt_p):
                continue
            qarch = QtipArchive(pt_p, mmap_slab=True)
            if int(qarch.layer_id) != int(l):
                # [R3.9S] the pools are keyed by the file's layer number, the layer runtime asks with the archive's
                # own layer_id: they must agree (the base runner overwrites rt.layer_id; here a mismatch is an error)
                raise ValueError(f"{pt_p}: the archive says layer_id {qarch.layer_id}")
            self.moe_layers[l] = QtipLayerRuntime(qarch, self.device, slab_in_vram=False)
            if qarch.missing.numel() > 0:
                self.archive_missing[l] = [int(e) for e in qarch.missing.tolist()]

        if not self.moe_layers:
            raise FileNotFoundError(f"No layer_XX_qtip_hyb.pt archives found in {fold_dir}")

        first_rt = next(iter(self.moe_layers.values()))
        self.preset = first_rt.preset
        self.qtip_rule = str(getattr(first_rt.arch, "rule", "A"))
        max_row_bytes = max(rt.preset.expert_row_bytes for rt in self.moe_layers.values())
        # [R3.9S] read by glm53_fold_generate.main()'s report (the base computes it from the .npz preset)
        self.placement = {
            "vram_resident_layers": [],
            "vram_slabs_gib": 0.0,
            "host_slabs_gib": sum(int(rt.expert_slab_u8.shape[0]) * int(rt.expert_slab_u8.shape[1])
                                  for rt in self.moe_layers.values()) / (1024.0 ** 3),
        }

        # Load exact prefix caches (`exact_prefix_cache_prefix10.pt` / `prefix2.pt`) when present
        self.prefix_caches: List[dict] = []
        if prefix_cache:
            for fn in ("exact_prefix_cache_prefix10.pt", "exact_prefix_cache_prefix2.pt"):
                p_pc = os.path.join(fold_dir, fn)
                if os.path.exists(p_pc):
                    self.prefix_caches.append(torch.load(p_pc, map_location="cpu"))

        # Apply Stage-2B merged routers + archive-missing & optional extra prune masks BEFORE graph capture
        auto_routers = merged_routers_path or os.path.join(fold_dir, "merged_routers_bf16.pt")
        extra_p = load_prune_spec(prune_spec, self.cfg) if prune_spec else {}
        n_patched = apply_merged_routers(self.bb, auto_routers, self.archive_missing, extra_p)
        self.prune_masks = {l: sorted(set(self.archive_missing.get(l, [])) | set(extra_p.get(l, []))) for l in self.moe_layers}

        self.next_moe: Dict[int, int] = {}
        ordered = sorted(self.moe_layers.keys())
        for a, b in zip(ordered[:-1], ordered[1:]):
            if b < self.cfg.num_hidden_layers:
                self.next_moe[a] = b

        # ---- expert pool -------------------------------------------------------------------------------------------
        # [R3.9S] two pools. "v6" = the production pool glm53_fold_expert_pool_v6 (Round 3.5-3.7: R35_POOL_SIGNAL /
        # DEMAND / DEMAND_STREAM / GATHER / EVICT / COLD_TOPK / COLD_MARGIN, R37_POOL_REG_TOPK / _MARGIN, registration
        # by R33_REGISTER_GIB + R33_REGISTER_MODE); it moves bytes only and takes the QTIP sidecars as it takes the
        # `.npz` slabs (sources [rows, row_bytes] uint8, files (path, offset 0), row_of). "qtip" = this file's
        # QtipPinnedSlotPool, the pool of the shipped Round 3.9 runtime (the only one that holds rows of different
        # sizes, i.e. a GU125 trunk next to a rule-A MTP layer). Choice: pool_kind argument, else R39_POOL, else
        # "v6" when R32_POOL=6 (the production environment) and "qtip" otherwise (the shipped behaviour).
        n_slots = int(expert_pool_gib * (1024 ** 3) // max_row_bytes)
        files = {l: rt.arch.slab_file for l, rt in self.moe_layers.items() if rt.arch.slab_file is not None}
        row_of = {l: rt.arch.row_of for l, rt in self.moe_layers.items() if rt.arch.row_of is not None}
        sources = {l: rt.expert_slab_u8 for l, rt in self.moe_layers.items()}
        kind = (pool_kind or os.environ.get("R39_POOL", "")
                or ("v6" if os.environ.get("R32_POOL", "") == "6" else "qtip")).strip().lower()
        if kind not in ("v6", "qtip"):
            raise ValueError(f"R39_POOL / pool_kind must be v6 or qtip, not {kind!r}")
        reg_gib = float(os.environ.get("R33_REGISTER_GIB", "") or 42.0) if register_gib is None else float(register_gib)
        prewarm_env = os.environ.get("R39_POOL_PREWARM", "")
        self.pool_kind = kind
        if kind == "v6":
            row_sizes = sorted({int(rt.preset.expert_row_bytes) for rt in self.moe_layers.values()})
            if len(row_sizes) != 1:
                raise ValueError(f"pool v6 holds rows of one size; this archive has {row_sizes} bytes per expert "
                                 "(GU125 trunk + rule-A MTP layer?): run with R39_POOL=qtip or --no-mtp-layer")
            from glm53_fold_expert_pool_v6 import ExpertPool  # type: ignore
            if reg_gib > 0 and os.environ.get("R33_REGISTER_MODE", "layers") != "rows":
                print("[r39s] WARNING: pool v6 registers WHOLE LAYERS in layer order unless R33_REGISTER_MODE=rows "
                      "(production and the shipped QTIP pool lock the hottest rows of every layer)", flush=True)
            self.pool = ExpertPool(sources, max_row_bytes, n_slots, self.device, files=files, row_of=row_of,
                                   register_gib=reg_gib)
            # zero the slots once, before the first fetch (pool v6 allocates with torch.empty): the capture warm-up
            # decodes slot 0 before any expert is resident and the QTIP GEMV writes at y[row_map[r]]; see the same
            # note in QtipPinnedSlotPool. Synchronized, so no later copy on another stream can run before the fill.
            self.pool.pool.zero_()
            torch.cuda.synchronize(self.device)
            if getattr(self.pool, "reg_rows", None):
                print(f"[r39s] registered tier ({os.environ.get('R33_REGISTER_MODE', 'layers')}): "
                      f"{self.pool.stats['registered_layers']} layers, {self.pool.stats.get('registered_rows', '?')} rows, "
                      f"{self.pool.stats['registered_gib']} GiB locked in {self.pool.stats['register_s']} s", flush=True)
            print(f"[r39s] pool v6: {self.pool.flags_str()}", flush=True)
            # pool v6 has no start-up pre-seeding (production fills it by use); opt-in here, off by default
            self._prewarm_pool_v6(int(prewarm_env or 0), n_slots)
        else:
            # Initialize Two-Tier GDDR7 VRAM + cudaHostRegister DDR5 Slot Pool (stride-aware across GU125 + Rule A MTP)
            hot_per_layer = max(8, (n_slots - 128) // max(len(self.moe_layers), 1))
            if prewarm_env:                      # [R3.9S] R39_POOL_PREWARM overrides the shipped automatic count
                hot_per_layer = max(0, int(prewarm_env))
            self.pool = QtipPinnedSlotPool(
                sources=sources,
                row_bytes=max_row_bytes,
                n_slots=n_slots,
                device=self.device,
                files=files,
                row_of=row_of,
                register_gib=reg_gib,
                prewarm_hot_per_layer=hot_per_layer,
            )
        for rt in self.moe_layers.values():
            rt.pool = self.pool
        self._pool_prev = dict(self.pool.stats)
        self._pred_host = torch.empty((self.cfg.num_experts_per_tok * 4,), dtype=torch.int64, pin_memory=True)
        # [R3.9S] the rest of the base constructor's pool block: score row of the eager prediction (pool gates that
        # rank predictions), R35_PREFILL_COLD, R35_PREFILL_PREFETCH (both need pool v6; off otherwise, as in the base)
        if getattr(self.pool, "wants_scores", False):
            self._pred_sc_host = torch.empty((self.cfg.num_experts_per_tok * 4,), dtype=torch.float32, pin_memory=True)
        self.prefill_cold = (os.environ.get("R35_PREFILL_COLD", "0") == "1"
                             and bool(getattr(self.pool, "supports_cold", False)))
        if os.environ.get("R35_PREFILL_PREFETCH", "0") == "1":
            self.set_ttft_options(prefetch=self.prefetch,
                                  pf_min=int(os.environ.get("R35_PREFILL_PREFETCH_MIN", "1") or 1),
                                  pf_max=int(os.environ.get("R35_PREFILL_PREFETCH_MAX", "0") or 0))
        # [R3.9S] Round 3.6 routing policies (R36_TOPK / R36_MIN_WEIGHT / R36_SKIP_COLD_TAU / R36_CACHE_BIAS; all
        # inactive by default -> None). After the masks and the merged routers, before any graph capture.
        from glm53_fold_route_r36 import install as install_r36_route  # type: ignore
        self.r36_router = install_r36_route(self)

        # ---- CUDA graphs -------------------------------------------------------------------------------------------
        # [R3.9S] as glm53_fold_generate.main(): one decoder as the environment says (window, or long context with
        # R36_LONGCTX=1), plus the window decoder when R37_DUAL_GRAPHS=1 asks for both. capture_graphs=False leaves
        # all of it to the caller (main() of glm53_fold_generate builds and captures the decoders itself).
        if capture_graphs:
            self.graph_decoder = GraphDecoder(self, t_max=t_max)
            if os.environ.get("R37_DUAL_GRAPHS", "0") == "1" and getattr(self.graph_decoder, "r36_longctx", False):
                self.graph_decoder_long = self.graph_decoder
                self.graph_decoder_std = GraphDecoder(self, t_max=t_max, longctx=False)
            self.graph_decoder.capture()
            if self.graph_decoder_std is not None:
                self.graph_decoder_std.capture()
            if (self.load_mtp and (self.cfg.num_hidden_layers in self.moe_layers)
                    and os.environ.get("R35_MTP_GRAPHS", "0") == "1"
                    and not getattr(self.graph_decoder, "r36_longctx", False)):
                if callable(getattr(self.graph_decoder, "capture_mtp", None)):
                    self.graph_decoder.capture_mtp()

        print(
            f"[r39-ready] Loaded {len(self.moe_layers)} QTIP HYB layers ({self.preset.name}, {self.preset.expert_row_bytes} B/row), "
            f"patched_routers={n_patched}, pool[{kind}]={n_slots} VRAM slots ({expert_pool_gib:.1f} GiB) + "
            f"{self.pool.stats.get('registered_gib', 0.0):.1f} GiB cudaHostRegister DDR5, graphs={capture_graphs}, mtp={self.load_mtp} in {time.perf_counter()-t0:.1f}s",
            flush=True,
        )

    # ---------------------------------------------------------------------------------------------- [R3.9S] helpers
    def _prewarm_pool_v6(self, rows_per_layer: int, n_slots: int) -> int:
        """[R3.9S] R39_POOL_PREWARM=N with pool v6 (default 0 = off, production behaviour): fetch the N most popular
        rows of every layer before the first request (the shipped QTIP pool pre-seeds itself; pool v6 does not).
        Bytes only. Stops before the pool's last 128 slots. Returns the number of experts fetched."""
        if rows_per_layer <= 0:
            return 0
        done = 0
        for l in sorted(self.moe_layers):
            rt = self.moe_layers[l]
            ro = rt.arch.row_of
            if ro is not None:
                hot = [e for e, r in enumerate(ro.tolist()) if 0 <= r < rows_per_layer]
            else:
                hot = list(range(min(rows_per_layer, int(rt.expert_slab_u8.shape[0]))))
            if not hot or done + len(hot) > n_slots - 128:
                break
            self.pool.ensure(l, hot)
            done += len(hot)
        self.pool.release()
        torch.cuda.synchronize(self.device)
        print(f"[r39s] pool v6 pre-seeded with {done} experts ({rows_per_layer} rows per layer, R39_POOL_PREWARM)", flush=True)
        return done

    def _graph_tokens(self) -> int:
        tot, seen = 0, set()
        for gd in (self.graph_decoder, self.graph_decoder_std, self.graph_decoder_long):
            if gd is not None and id(gd) not in seen:
                seen.add(id(gd))
                tot += int((getattr(gd, "stats", None) or {}).get("tokens", 0))
        return tot

    # ------------------------------------------------------------------------------ [R3.9S] overrides of the base
    def set_gain(self, spec: str) -> None:
        """[R3.9S] --gain-correct of the Round 3.8 CLI: not applied (as for TCQ archives; the base's formula is for
        the RVQ presets)."""
        print(f"[r39s] gain correction {spec}: skipped for the QTIP archive (gate/up are encoded unbiased)", flush=True)

    def warm_page_cache(self, fold_dir: Optional[str] = None, threads: int = 4) -> float:
        """[R3.9S] --warm-cache of the Round 3.8 CLI for a QTIP archive: sequential 64 MiB reads of every layer's
        `.slab` sidecar (the file the pool reads and registers; the base reads `layer_XX_apex_fold.npz`)."""
        from concurrent.futures import ThreadPoolExecutor

        paths = [rt.arch.slab_file[0] for _, rt in sorted(self.moe_layers.items()) if rt.arch.slab_file is not None]

        def rd(p: str) -> int:
            n = 0
            buf = memoryview(bytearray(64 << 20))
            with open(p, "rb", buffering=0) as fh:
                while True:
                    k = fh.readinto(buf)
                    if not k:
                        return n
                    n += k

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=threads) as ex:
            tot = sum(ex.map(rd, paths))
        dt = time.perf_counter() - t0
        print(f"[r39s] page-cache warm-up: {tot / 2**30:.1f} GiB of sidecars in {dt:.1f}s", flush=True)
        return dt

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 128,
        eos_ids: Optional[Set[int]] = None,
        use_mtp: bool = False,
        serve_state=None,
    ) -> Dict[str, object]:
        """[R3.9S] the Round 3.8 generate() (ApexFoldMoEModelRunner.generate: N=1 graphs, the two-decoder selection,
        graph / eager MTP, serve hooks H1-H6, continuation past the graph window), plus the result keys only the
        shipped Round 3.9 generate() had: ids, tok_per_s, pool_hit_rate, used_graphs."""
        g0 = self._graph_tokens()
        res = super().generate(prompt_ids, max_new_tokens=max_new_tokens, eos_ids=eos_ids, use_mtp=use_mtp,
                               serve_state=serve_state)
        res["ids"] = res["generated_ids"]
        res["tok_per_s"] = res["decode_tok_s"]
        res["pool_hit_rate"] = round(self.pool.hit_rate(), 4) if callable(getattr(self.pool, "hit_rate", None)) else None
        res["used_graphs"] = bool(self._graph_tokens() > g0 or res.get("mtp_graph") is not None)
        return res


# [R3.9S] QtipMoEModelRunner = _QtipRunnerMixin + the Round 3.8 runner class. Built on first use, so that importing this
# module (and `--selftest`) needs torch only; `glm53_fold_qtip_runtime.QtipMoEModelRunner` resolves through the module
# __getattr__ below and stays the drop-in name the shipped tools use.
_RUNNER_CLASSES: Dict[type, type] = {}


def qtip_runner_class(base: Optional[type] = None) -> type:
    """[R3.9S] the runner class on `base` (default: glm53_fold_generate.ApexFoldMoEModelRunner of this folder).
    glm53_fold_generate.main() passes its own class, so that no second copy of that module is loaded when it runs as
    the entry point."""
    if base is None:
        if os.path.isdir(HERE) and HERE not in sys.path:
            sys.path.insert(0, HERE)
        import glm53_fold_generate as _G  # type: ignore
        base = _G.ApexFoldMoEModelRunner
    cls = _RUNNER_CLASSES.get(base)
    if cls is None:
        cls = type("QtipMoEModelRunner", (_QtipRunnerMixin, base),
                   {"__module__": __name__, "__doc__": _QtipRunnerMixin.__doc__, "_R38_BASE": base})
        _RUNNER_CLASSES[base] = cls
    return cls


def __getattr__(name: str):
    if name == "QtipMoEModelRunner":
        return qtip_runner_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ================================================================================================================
# 6. Standalone GPU Self-Test (--selftest: Verifies Slab, Pinned Pool, Router Merge & Real CUDAGraph Replay!)
# ================================================================================================================
def run_selftest() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("--selftest requires a CUDA GPU.")
    dev = torch.device("cuda")
    tmp_dir = "/tmp/r39_qtip_runtime_selftest"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    print(f"[SELFTEST] Testing QtipArchive + QtipPinnedSlotPool + CUDAGraph capture/replay on {torch.cuda.get_device_name(0)}...", flush=True)

    # 1. Verify exact 16-byte alignment & byte counts of production Rule A and Rule GU125 slabs
    p_a = make_qtip_preset("A")
    p_sub70 = make_qtip_preset("GU125")
    assert p_a.expert_row_bytes == 6326528 and p_a.expert_row_bytes % 16 == 0
    assert p_sub70.expert_row_bytes == 6064384 and p_sub70.expert_row_bytes % 16 == 0
    print(f"  [GATE-1] Production slab byte locks verified: Rule A = {p_a.expert_row_bytes:,} B | Rule GU125 = {p_sub70.expert_row_bytes:,} B", flush=True)

    # 2. Synthesize 2 pruned QTIP HYB MoE layer archives (E_tot=8, E_kept=6, D=64, Di=64) + merged_routers_bf16.pt
    gen_cpu = torch.Generator(device="cpu").manual_seed(77)
    for l in (1, 2):
        perm = torch.tensor([4, 0, 7, 2, 5, 1], dtype=torch.int16)  # experts [3, 6] pruned!
        fields = {}
        for p in ("g", "u", "d"):
            fields[p] = {
                "map": torch.stack([torch.randperm(64, generator=gen_cpu).to(torch.int16) for _ in range(6)]),
                "scale": (torch.rand((6, 64), generator=gen_cpu) * 0.4 + 0.1).to(torch.float16),
                "c1": torch.randint(0, 256, (6, 1, 4, 32), generator=gen_cpu, dtype=torch.uint8),
                "c2": torch.randint(0, 256, (6, 3, 4, 64), generator=gen_cpu, dtype=torch.uint8),
                "sp": torch.zeros((6, 8, 6), dtype=torch.uint8),
            }
            # Populate deterministic non-zero sparse outliers to exercise qtip_sparse_add_kernel!
            for e in range(6):
                for k_s in range(8):
                    r_s = (e * 7 + k_s * 5) % 64
                    c_s = (e * 11 + k_s * 3) % 64
                    val_s = torch.tensor([0.15 * (1 if k_s % 2 == 0 else -1)], dtype=torch.float16).view(torch.uint8)
                    fields[p]["sp"][e, k_s, 0:2] = torch.tensor([r_s], dtype=torch.int16).view(torch.uint8)
                    fields[p]["sp"][e, k_s, 2:4] = torch.tensor([c_s], dtype=torch.int16).view(torch.uint8)
                    fields[p]["sp"][e, k_s, 4:6] = val_s

        blob = {
            "format": "qtip_hyb_r39",
            "rule": "GU125",
            "layer_id": l,
            "tile_mode": "2d_16x16",
            "signs": {p: (torch.randint(0, 2, (64,), generator=gen_cpu, dtype=torch.int8) * 2 - 1) for p in ("g", "u", "d")},
            "hyb_luts": {
                p: {
                    1: (torch.rand((512, 2), generator=gen_cpu) + 0.1).to(torch.float16),
                    2: (torch.rand((512, 2), generator=gen_cpu) + 0.1).to(torch.float16),
                }
                for p in ("g", "u", "d")
            },
            "fields": fields,
            "perm": perm,
            "n_experts_total": 8,
        }
        torch.save(blob, os.path.join(tmp_dir, f"layer_{l:02d}_qtip_hyb.pt"))

    # 3. Load QtipArchive + QtipPinnedSlotPool (testing POSIX/Win32 cudaHostRegister + async prefetch + out= mailbox)
    arch1 = QtipArchive(os.path.join(tmp_dir, "layer_01_qtip_hyb.pt"), mmap_slab=True)
    arch2 = QtipArchive(os.path.join(tmp_dir, "layer_02_qtip_hyb.pt"), mmap_slab=True)
    rt1 = QtipLayerRuntime(arch1, dev, slab_in_vram=False)
    rt2 = QtipLayerRuntime(arch2, dev, slab_in_vram=False)

    pool = QtipPinnedSlotPool(
        sources={1: rt1.expert_slab_u8, 2: rt2.expert_slab_u8},
        row_bytes=rt1.preset.expert_row_bytes,
        n_slots=8,  # Intentionally smaller than 12 total experts so LRU eviction + prefetch are actively exercised!
        device=dev,
        files={1: arch1.slab_file, 2: arch2.slab_file},
        row_of={1: arch1.row_of, 2: arch2.row_of},
        register_gib=0.1,
        register_floor_gib=0.0,
    )
    assert "out" in inspect.signature(pool.ensure).parameters
    rt1.pool = pool
    rt2.pool = pool

    # 4. Verify dense_core (prefill BF16 kernel) vs decode_core (slab GEMV kernel) equivalence!
    x_multi = torch.randn((32, 64), device=dev, dtype=torch.float32)
    alive_ids = torch.tensor([0, 1, 2, 4, 5, 7], device=dev)
    top_i = alive_ids[torch.randint(0, 6, (32, 2), device=dev)]
    top_w = torch.softmax(torch.randn((32, 2), device=dev), dim=-1) * 2.0

    rt1.dense_min_pairs = 999999
    y_gemv = rt1.forward_routed_experts(x_multi, top_w, top_i)
    rt1.dense_min_pairs = 4
    y_dense = rt1.forward_routed_experts(x_multi, top_w, top_i)
    cos_pd = float(F.cosine_similarity(y_gemv.flatten(), y_dense.flatten(), dim=0).item())
    assert cos_pd > 0.9999, f"Prefill dense_core vs decode_core mismatch: cos={cos_pd}"
    print(f"  [GATE-2] Prefill rotated-BF16 dense_core vs NVRTC slab GEMV cos = {cos_pd:.6f}", flush=True)

    # 5. Capture real PyTorch CUDAGraph matching GraphDecoder._seg_b (pinned slot mailbox -> H2D -> decode_core)
    x_static = torch.randn((1, 64), device=dev, dtype=torch.float32)
    w_static = torch.tensor([1.2, 0.8], device=dev, dtype=torch.float32)
    slots_pin = torch.zeros((2,), dtype=torch.int32, pin_memory=True)
    slots_np = slots_pin.numpy()
    slots_dev = torch.zeros((2,), dtype=torch.int32, device=dev)

    pool.ensure(1, [4, 0], out=slots_np)
    s_cap = torch.cuda.Stream(device=dev)
    s_cap.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(s_cap):
        slots_dev.copy_(slots_pin, non_blocking=True)
        _ = rt1.decode_core(x_static, pool.pool, slots_dev, w_static, 1, LIM)
    torch.cuda.current_stream(dev).wait_stream(s_cap)
    torch.cuda.synchronize(dev)

    g_b = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_b):
        slots_dev.copy_(slots_pin, non_blocking=True)
        y_graph = rt1.decode_core(x_static, pool.pool, slots_dev, w_static, 1, LIM)

    # Replay graph across changing expert pairs & async prefetches and verify bit-identical match to eager!
    for pair in ([7, 2], [5, 1], [4, 7]):
        pool.prefetch(2, [0, 2])
        x_new = torch.randn((1, 64), device=dev, dtype=torch.float32)
        x_static.copy_(x_new)
        pool.ensure(1, pair, out=slots_np)
        g_b.replay()
        torch.cuda.synchronize(dev)

        y_eager = rt1.forward_routed_experts(x_new, w_static.view(1, 2), torch.tensor([pair], device=dev))
        max_diff = float((y_graph - y_eager).abs().max().item())
        assert max_diff == 0.0, f"CUDAGraph vs eager diff = {max_diff}!"

    pool.close()
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("[SELFTEST PASSED] CUDAGraph replay is 0.0-ULP bit-identical to eager decode with live pinned-pool DMA & prefetch!", flush=True)


# ================================================================================================================
# 6b. [R3.9S] GPU self-test of the port only (--selftest-port). run_selftest() above is the shipped one, unchanged.
# ================================================================================================================
def _port_test_archives(tmp_dir: str) -> None:
    """[R3.9S] the two synthetic pruned layers of run_selftest() step 2 (same generator, same seed), as files."""
    gen_cpu = torch.Generator(device="cpu").manual_seed(77)
    for l in (1, 2):
        perm = torch.tensor([4, 0, 7, 2, 5, 1], dtype=torch.int16)  # experts [3, 6] pruned
        fields = {}
        for p in ("g", "u", "d"):
            fields[p] = {
                "map": torch.stack([torch.randperm(64, generator=gen_cpu).to(torch.int16) for _ in range(6)]),
                "scale": (torch.rand((6, 64), generator=gen_cpu) * 0.4 + 0.1).to(torch.float16),
                "c1": torch.randint(0, 256, (6, 1, 4, 32), generator=gen_cpu, dtype=torch.uint8),
                "c2": torch.randint(0, 256, (6, 3, 4, 64), generator=gen_cpu, dtype=torch.uint8),
                "sp": torch.zeros((6, 8, 6), dtype=torch.uint8),
            }
            for e in range(6):
                for k_s in range(8):
                    r_s = (e * 7 + k_s * 5) % 64
                    c_s = (e * 11 + k_s * 3) % 64
                    val_s = torch.tensor([0.15 * (1 if k_s % 2 == 0 else -1)], dtype=torch.float16).view(torch.uint8)
                    fields[p]["sp"][e, k_s, 0:2] = torch.tensor([r_s], dtype=torch.int16).view(torch.uint8)
                    fields[p]["sp"][e, k_s, 2:4] = torch.tensor([c_s], dtype=torch.int16).view(torch.uint8)
                    fields[p]["sp"][e, k_s, 4:6] = val_s
        blob = {
            "format": "qtip_hyb_r39",
            "rule": "GU125",
            "layer_id": l,
            "tile_mode": "2d_16x16",
            "signs": {p: (torch.randint(0, 2, (64,), generator=gen_cpu, dtype=torch.int8) * 2 - 1) for p in ("g", "u", "d")},
            "hyb_luts": {
                p: {
                    1: (torch.rand((512, 2), generator=gen_cpu) + 0.1).to(torch.float16),
                    2: (torch.rand((512, 2), generator=gen_cpu) + 0.1).to(torch.float16),
                }
                for p in ("g", "u", "d")
            },
            "fields": fields,
            "perm": perm,
            "n_experts_total": 8,
        }
        torch.save(blob, os.path.join(tmp_dir, f"layer_{l:02d}_qtip_hyb.pt"))


def run_selftest_port() -> None:
    """[R3.9S] checks of what the port added, on the synthetic layers of the shipped self-test (D = 64, K = 2):
      P1 decode_core(pair_output=True) + the caller's reduction == decode_core()                       (bit-exact)
      P2 a pair with slot -1 gives zeros and leaves the other pair's output unchanged                    (bit-exact)
      P3 forward_routed_experts with a routed id of -1 (r36_route) == the same call with a zero-weight
         duplicate in its place                                                                          (bit-exact)
      P4 QtipPinnedSlotPool: R37_POOL_REG_TOPK no longer cuts an unscored prediction list; with scores it
         skips only REGISTERED predictions ranked at or below the limit
      P5 production pool v6 (glm53_fold_expert_pool_v6) on the QTIP sidecars: eager == QTIP pool, and a captured
         CUDA graph reading v6's pinned slot mailbox == eager                                            (bit-exact)"""
    if not torch.cuda.is_available():
        raise RuntimeError("--selftest-port requires a CUDA GPU.")
    import tempfile

    dev = torch.device("cuda")
    tmp_dir = os.path.join(tempfile.gettempdir(), "r39s_qtip_port_selftest")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    print(f"[SELFTEST-PORT] Round 3.9S port checks on {torch.cuda.get_device_name(0)}...", flush=True)
    _port_test_archives(tmp_dir)
    arch1 = QtipArchive(os.path.join(tmp_dir, "layer_01_qtip_hyb.pt"), mmap_slab=True)
    arch2 = QtipArchive(os.path.join(tmp_dir, "layer_02_qtip_hyb.pt"), mmap_slab=True)
    rt1 = QtipLayerRuntime(arch1, dev, slab_in_vram=False)
    rt2 = QtipLayerRuntime(arch2, dev, slab_in_vram=False)
    rb = rt1.preset.expert_row_bytes
    srcs = {1: rt1.expert_slab_u8, 2: rt2.expert_slab_u8}
    files = {1: arch1.slab_file, 2: arch2.slab_file}
    row_of = {1: arch1.row_of, 2: arch2.row_of}
    pool = QtipPinnedSlotPool(sources=srcs, row_bytes=rb, n_slots=12, device=dev, files=files, row_of=row_of,
                              register_gib=0.1, register_floor_gib=0.0)
    rt1.pool = pool
    rt2.pool = pool

    def mx(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a - b).abs().max().item())

    x = torch.randn((1, 64), device=dev, dtype=torch.float32)
    w = torch.tensor([1.2, 0.8], device=dev, dtype=torch.float32)

    # P1 pair_output
    sl = pool.ensure(1, [4, 0])
    y_ref = rt1.decode_core(x, pool.pool, sl, w, 1, LIM)
    d = rt1.decode_core(x, pool.pool, sl, w, 1, LIM, pair_output=True)
    y_po = (d * w.view(-1, 1)).view(1, 2, -1).sum(dim=1)
    assert tuple(d.shape) == (2, 64) and mx(y_ref, y_po) == 0.0, f"P1 pair_output: diff {mx(y_ref, y_po)}"
    print("  [P1] decode_core(pair_output=True) + reduction == decode_core(): 0.0", flush=True)

    # P2 masked pair
    sl_m = sl.clone()
    sl_m[1] = -1
    d_m = rt1.decode_core(x, pool.pool, sl_m, w, 1, LIM, pair_output=True)
    assert float(d_m[1].abs().max().item()) == 0.0, "P2: a masked pair must give zeros"
    assert mx(d_m[0], d[0]) == 0.0, f"P2: the unmasked pair changed by {mx(d_m[0], d[0])}"
    print("  [P2] slot -1 -> zero pair, the other pair unchanged: 0.0", flush=True)

    # P3 routed id -1 (routing policies)
    w_m = torch.tensor([[1.2, 0.0]], device=dev, dtype=torch.float32)
    rt1.r36_route = True
    y_route = rt1.forward_routed_experts(x, w_m, torch.tensor([[4, -1]], device=dev))
    rt1.r36_route = False
    y_dup = rt1.forward_routed_experts(x, w_m, torch.tensor([[4, 4]], device=dev))
    assert mx(y_route, y_dup) == 0.0, f"P3 r36_route: diff {mx(y_route, y_dup)}"
    print("  [P3] routed id -1 == zero-weight duplicate: 0.0", flush=True)

    # P4 reg_topk
    pool.quiesce()
    pool.reg_topk = 1
    n_plain = pool.prefetch(2, [0, 2, 4])
    assert n_plain == 3, f"P4: an unscored prediction list was cut ({n_plain} of 3 queued)"
    pool.quiesce()
    reg5 = pool._is_registered(2, 5)
    sk0 = pool.stats["prefetch_skipped_reg"]
    n_sc = pool.prefetch_scored(2, [5, 7], [0.1, 0.9])
    assert n_sc == (1 if reg5 else 2), f"P4: scored prefetch queued {n_sc} (expert 5 registered: {reg5})"
    assert pool.stats["prefetch_skipped_reg"] - sk0 == (1 if reg5 else 0)
    pool.reg_topk = 0
    pool.quiesce()
    print(f"  [P4] reg_topk gate: unscored list kept (3 of 3), scored list queued {n_sc} of 2 "
          f"(rank-1 expert registered: {reg5}; registered layers: {len(pool.reg_rows)})", flush=True)

    # P5 production pool v6 on the QTIP sidecars
    pairs = ([4, 0], [7, 2], [5, 1], [4, 7], [0, 2])
    xs = [torch.randn((1, 64), device=dev, dtype=torch.float32) for _ in pairs]
    refs = [rt1.forward_routed_experts(xv, w.view(1, 2), torch.tensor([pr], device=dev)).clone()
            for pr, xv in zip(pairs, xs)]
    torch.cuda.synchronize(dev)
    pool.close()
    if os.path.isdir(HERE) and HERE not in sys.path:
        sys.path.insert(0, HERE)
    from glm53_fold_expert_pool_v6 import ExpertPool  # type: ignore

    try:
        pool6 = ExpertPool(srcs, rb, 10, dev, files=files, row_of=row_of, register_gib=0.1, register_floor_gib=0.0)
    except Exception as exc:  # registration needs cudart64_12.dll; the copy paths are tested without it
        print(f"  [P5] note: pool v6 registration failed ({type(exc).__name__}: {exc}); retrying unregistered", flush=True)
        pool6 = ExpertPool(srcs, rb, 10, dev, files=files, row_of=row_of, register_gib=0.0)
    assert int(pool6.pool.stride(0)) == rb, "P5: pool v6 row stride != QTIP row bytes"
    rt1.pool = pool6
    rt2.pool = pool6
    for pr, xv, yr in zip(pairs, xs, refs):
        pool6.prefetch(2, [0, 2])
        y6 = rt1.forward_routed_experts(xv, w.view(1, 2), torch.tensor([pr], device=dev))
        assert mx(y6, yr) == 0.0, f"P5 eager: pool v6 vs QTIP pool diff {mx(y6, yr)} for experts {pr}"
    pool6.quiesce()

    x_static = xs[0].clone()
    slots_pin = torch.zeros((2,), dtype=torch.int32, pin_memory=True)
    slots_np = slots_pin.numpy()
    slots_dev = torch.zeros((2,), dtype=torch.int32, device=dev)
    pool6.ensure(1, pairs[0], out=slots_np)
    s_cap = torch.cuda.Stream(device=dev)
    s_cap.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(s_cap):
        slots_dev.copy_(slots_pin, non_blocking=True)
        _ = rt1.decode_core(x_static, pool6.pool, slots_dev, w, 1, LIM)
    torch.cuda.current_stream(dev).wait_stream(s_cap)
    torch.cuda.synchronize(dev)
    g_b = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_b):
        slots_dev.copy_(slots_pin, non_blocking=True)
        y_graph = rt1.decode_core(x_static, pool6.pool, slots_dev, w, 1, LIM)
    for pr, xv, yr in zip(pairs, xs, refs):
        pool6.prefetch_scored(2, [0, 2], [0.9, 0.1])
        x_static.copy_(xv)
        pool6.ensure(1, pr, out=slots_np)
        g_b.replay()                               # queued before the next ensure() on another layer (v6 lifetime rule)
        torch.cuda.synchronize(dev)
        assert mx(y_graph, yr) == 0.0, f"P5 graph: pool v6 mailbox vs eager diff {mx(y_graph, yr)} for experts {pr}"
    st6 = dict(pool6.stats)
    pool6.release()
    pool6.close()
    print(f"  [P5] pool v6 on QTIP sidecars: eager == QTIP pool and graph == eager over {len(pairs)} expert sets: 0.0 "
          f"(registered layers {st6.get('registered_layers', 0)}, hits {st6['hits']}, misses {st6['misses']}; "
          f"flags: {pool6.flags_str()})", flush=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("[SELFTEST-PORT PASSED] pair_output, masked pairs, routed -1, reg_topk gate and pool v6 are bit-exact / as specified.", flush=True)


# ================================================================================================================
# 6c. [R3.9K] GPU self-test and cold micro-benchmark of the v2 expert kernels (--selftest-kernel). New code only.
# ================================================================================================================
def _r39k_byte_perm(x: int, y: int, s: int) -> int:
    """[R3.9K] CPU model of CUDA's __byte_perm(x, y, s): result byte i is byte number (nibble i of s) & 7 of the
    eight bytes x (0..3) and y (4..7)."""
    v = (x & 0xFFFFFFFF) | ((y & 0xFFFFFFFF) << 32)
    out = 0
    for i in range(4):
        out |= ((v >> (8 * ((s >> (4 * i)) & 7))) & 0xFF) << (8 * i)
    return out


def _r39k_cpu_bit_windows(trials: int = 32) -> int:
    """[R3.9K] CPU-only check of the one thing v2 computes differently before the floating-point part: the four
    16-bit states of a lane. Reference = v1's read_tailbite_state16 (three byte loads per state, byte index modulo
    the tile size); v2 = the lane's own code word plus its neighbour lanes' words, as _r39k_tile writes it.
    Returns the number of states compared; raises AssertionError on the first difference."""
    rng = np.random.default_rng(39)
    n = 0
    for k, tile_bytes in ((1, 32), (2, 64)):
        step_bits = k << 1
        for _ in range(trials):
            tp = [int(b) for b in rng.integers(0, 256, size=tile_bytes, dtype=np.uint8)]
            for lane in range(32):
                m_base = (lane >> 1) * 8 + (lane & 1) * 4
                ref = []
                for j in range(4):                                   # v1, statement by statement
                    bit_pos = (m_base + j) * step_bits
                    b0 = (bit_pos >> 3) & (tile_bytes - 1)
                    b1 = (b0 + 1) & (tile_bytes - 1)
                    b2 = (b0 + 2) & (tile_bytes - 1)
                    off = bit_pos & 7
                    w24 = (tp[b0] << 16) | (tp[b1] << 8) | tp[b2]
                    ref.append((w24 >> (8 - off)) & 0xFFFF)
                nx1, nx2 = (lane + 1) & 31, (lane + 2) & 31
                if k == 2:                                           # v2: little-endian 16-bit word per lane
                    cw = tp[2 * lane] | (tp[2 * lane + 1] << 8)
                    nw = tp[2 * nx1] | (tp[2 * nx1 + 1] << 8)
                    bw = _r39k_byte_perm(cw | (nw << 16), 0, 0x0123)
                    got = [bw >> 16, (bw >> 12) & 0xFFFF, (bw >> 8) & 0xFFFF, (bw >> 4) & 0xFFFF]
                else:                                                # v2: one byte per lane
                    bw = (tp[lane] << 16) | (tp[nx1] << 8) | tp[nx2]
                    got = [bw >> 8, (bw >> 6) & 0xFFFF, (bw >> 4) & 0xFFFF, (bw >> 2) & 0xFFFF]
                if got != ref:
                    raise AssertionError(f"R3.9K bit windows: k={k} lane={lane}: v2 states {got} != v1 states {ref}")
                n += 4
    return n


def _r39k_fake_arch(rule: str, n_rows: int, seed: int):
    """[R3.9K] an in-memory stand-in for a QtipArchive with the production row layout of `rule` (A or GU125) and
    random content: random code bytes, a valid row permutation and finite scales per projection, valid sparse
    entries, random LUTs and signs. Nothing is written to disk. Timing does not depend on the code values."""
    from types import SimpleNamespace

    preset = make_qtip_preset(rule)
    lay, rb = preset.layout()
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    slab = torch.randint(0, 256, (n_rows, rb), generator=gen, dtype=torch.uint8)
    signs, luts = {}, {}
    for p, sp in (("g", preset.gate), ("u", preset.up), ("d", preset.down)):
        off, nb = lay[f"{p}_map"]
        m = torch.stack([torch.randperm(sp.d_out, generator=gen) for _ in range(n_rows)]).to(torch.int16)
        slab[:, off:off + nb] = m.view(torch.uint8).reshape(n_rows, -1)
        off, nb = lay[f"{p}_scale"]
        s = (torch.rand((n_rows, sp.d_out), generator=gen) * 0.4 + 0.1).to(torch.float16)
        slab[:, off:off + nb] = s.view(torch.uint8).reshape(n_rows, -1)
        off, nb = lay[f"{p}_sp"]
        if nb:
            ent = torch.empty((n_rows, sp.k_sparse, 3), dtype=torch.int16)
            ent[:, :, 0] = torch.randint(0, sp.d_out, (n_rows, sp.k_sparse), generator=gen).to(torch.int16)
            ent[:, :, 1] = torch.randint(0, sp.d_in, (n_rows, sp.k_sparse), generator=gen).to(torch.int16)
            val = ((torch.rand((n_rows, sp.k_sparse), generator=gen) - 0.5) * 0.3).to(torch.float16)
            ent[:, :, 2] = val.view(torch.int16)
            slab[:, off:off + nb] = ent.view(torch.uint8).reshape(n_rows, -1)
        signs[p] = (torch.randint(0, 2, (sp.d_in,), generator=gen, dtype=torch.int8) * 2 - 1)
        luts[p] = {k: (torch.rand((512, 2), generator=gen) - 0.5).to(torch.float16) for k in (1, 2)}
    return SimpleNamespace(preset=preset, layer_id=0, expert_slab_u8=slab, signs=signs, hyb_luts=luts,
                           row_of=None, missing=torch.empty(0, dtype=torch.long))


def run_selftest_kernel(fold: str = "", calls: int = 2000, bench_rows: int = 64, layer: int = -1,
                        sweep: bool = False, _dev: Optional[torch.device] = None) -> None:
    """[R3.9K] v1 against v2 of the expert kernels on identical inputs.
      E1 CPU: the v2 code-word windows give v1's states (no GPU).
      E2 bitwise equality, v1 == v2 (the configured v2 and its launch variants: fusion off, load-ahead 0 / 1 / other,
         activations from global memory, other block geometries), for every projection's GEMV alone and for
         decode_core with N = 1 (8 experts), N = 1 with masked pairs + pair_output, N = 2, N = 2 row_exact and a
         prefill-like batch (24 tokens), on:
           a) the synthetic layers of --selftest (D = 64, k = 1 and k = 2 row tiles), also with a row stride larger
              than the row (the pool convention);
           b) random slabs with the production layouts of rule A and GU125;
           c) with --fold: the first rows of one real layer file (packed in RAM, nothing written).
      E3 the same equality for outputs of captured CUDA graphs after a replay.
      B  micro-benchmark on b) and c): median microseconds per call over `calls` calls after warm-up, each call
         between two torch.cuda.synchronize(); plus the back-to-back mean. Timed: decode_core in a captured CUDA
         graph (what production replays), the three GEMVs alone in a graph, and eager decode_core (dominated by
         Python). COLD by construction: consecutive calls address different slab rows (bench_rows rows in sets of
         8 or 16), as in production, where every layer reads rows that are not in the GPU's L2.
      With sweep=True: decode_core N = 1 in a graph for a grid of v2 settings.
    `_dev` is a test hook (a device other than cuda, with stubbed kernels); leave it None."""
    n_states = _r39k_cpu_bit_windows()
    print(f"[SELFTEST-KERNEL] E1 CPU: the v2 code-word windows give v1's states in all {n_states} cases "
          f"(k = 1 and k = 2, 32 lanes, 4 states)", flush=True)
    if _dev is None and not torch.cuda.is_available():
        raise RuntimeError("--selftest-kernel requires a CUDA GPU.")
    import gc
    import tempfile
    from types import SimpleNamespace

    dev = _dev if _dev is not None else torch.device("cuda")
    calls = max(1, int(calls))
    n_moe = 42
    gen = torch.Generator(device="cpu").manual_seed(3939)
    cfg = qtip_kernel_cfg("v2")
    variants: List[QtipKernelV2] = [cfg]
    for kw in ({"fuse_gu": not cfg.fuse_gu}, {"pf": 0}, {"pf": 1}, {"pf": 2 if cfg.pf == 8 else 8},
               {"x_shared": not cfg.x_shared}, {"warps": 4 if cfg.warps == 16 else 16}, {"warps": 2, "gx": 4}):
        c = qtip_kernel_cfg("v2", **kw)
        if c not in variants:
            variants.append(c)
    t_c = sum(c.precompile() for c in variants)
    print(f"[SELFTEST-KERNEL] device: {torch.cuda.get_device_name(0) if dev.type == 'cuda' else dev}; v1 launches "
          f"with R39_QTIP_WARPS={os.environ.get('R39_QTIP_WARPS', '8')} R39_QTIP_GX={os.environ.get('R39_QTIP_GX', '64')}; "
          f"v2 = [{cfg.describe()}]; {len(variants)} v2 launch variants compiled in {t_c:.1f}s", flush=True)

    def sync() -> None:
        torch.cuda.synchronize(dev)

    def bits_eq(a, b) -> bool:
        if isinstance(a, (tuple, list)):
            return len(a) == len(b) and all(bits_eq(p, q) for p, q in zip(a, b))
        return a.shape == b.shape and bool(
            torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32)))

    def rnd(*shape) -> torch.Tensor:
        return torch.randn(shape, generator=gen).to(dev)

    def check_layer(tag: str, rt: QtipLayerRuntime, rows: torch.Tensor, n_rows: int, K: int) -> int:
        lg, lu, ld = rt.lay["g"], rt.lay["u"], rt.lay["d"]
        for c in variants:
            for lay in rt.lay.values():
                c.check_layout(lay)
        n_cmp = 0
        for pname, lay in (("gate", lg), ("up", lu), ("down", ld)):          # the GEMV alone
            for per_pair in (False, True):
                P = 2 * K
                sl = torch.randint(0, n_rows, (P,), generator=gen).to(torch.int32)
                sl[P - 1] = -1                                              # one masked pair
                sl = sl.to(dev)
                xr = rnd(P if per_pair else 1, lay.T)
                y1 = qtip_gemv(rows, sl, lay, xr, per_pair)
                for c in variants:
                    y2 = c.gemv(rows, sl, lay, xr, per_pair)
                    if not bits_eq(y1, y2):
                        raise AssertionError(f"{tag}: GEMV {pname} (per_pair={per_pair}) differs from v1 with v2 "
                                             f"[{c.describe()}]: max |diff| {float((y1 - y2).abs().max())}")
                    n_cmp += 1
        cases = (("N=1", 1, {}, False), ("N=1 masked pairs, pair_output", 1, {"pair_output": True}, True),
                 ("N=2", 2, {}, False), ("N=2 row_exact", 2, {"row_exact": True}, False),
                 ("prefill-like N=24", 24, {}, False))
        for name, n_tok, kw, masked in cases:                               # the whole routed core
            P = n_tok * K
            sl = torch.stack([torch.randperm(n_rows, generator=gen)[:K] for _ in range(n_tok)]).reshape(-1)
            sl = sl.to(torch.int32)
            if masked:
                sl[1::2] = -1
            sl = sl.to(dev)
            x = rnd(n_tok, lg.T)
            w = torch.rand((P,), generator=gen).to(dev)
            rt.r39k = None
            y1 = rt.decode_core(x, rows, sl, w, n_tok, LIM, **kw).clone()
            if not bool(torch.isfinite(y1).all()) or float(y1.abs().max()) == 0.0:
                raise AssertionError(f"{tag}: v1 decode_core gave a non-finite or all-zero output for {name}; "
                                     f"the comparison would say nothing")
            for c in variants:
                rt.r39k = c
                y2 = rt.decode_core(x, rows, sl, w, n_tok, LIM, **kw)
                if not bits_eq(y1, y2):
                    raise AssertionError(f"{tag}: decode_core {name} differs from v1 with v2 [{c.describe()}]: "
                                         f"max |diff| {float((y1 - y2).abs().max())}")
                n_cmp += 1
        rt.r39k = None
        sync()
        print(f"  [E2] {tag}: {n_cmp} comparisons v1 == v2 bitwise (3 GEMVs x 2 activation modes + "
              f"{len(cases)} decode_core cases, {len(variants)} v2 variants)", flush=True)
        return n_cmp

    s_cap = torch.cuda.Stream(device=dev)

    def capture(fn):
        """warm-up on a side stream, then one captured graph of fn() (the pattern of run_selftest above)"""
        s_cap.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(s_cap):
            fn()
        torch.cuda.current_stream(dev).wait_stream(s_cap)
        sync()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fn()
        return g, out

    def time_calls(fns, n: int):
        for i in range(min(n, 256)):                                        # warm-up
            fns[i % len(fns)]()
        sync()
        per = np.empty(n, dtype=np.float64)
        for i in range(n):
            f = fns[i % len(fns)]
            sync()
            t0 = time.perf_counter()
            f()
            sync()
            per[i] = time.perf_counter() - t0
        sync()
        t0 = time.perf_counter()
        for i in range(n):
            fns[i % len(fns)]()
        sync()
        return per, (time.perf_counter() - t0) / n

    def bench_layer(tag: str, rt: QtipLayerRuntime, rows: torch.Tensor, n_rows: int, labelled, n_toks=(1, 2),
                    kinds=("core", "gemv"), eager: bool = True, max_sets: int = 16, quiet: bool = False) -> Dict:
        """labelled = [(label, None | QtipKernelV2), ...]; the first entry is the reference (v1). Returns
        {(n_tok, kind, label): (median us per call, back-to-back mean us per call)}."""
        K = 8
        lg, lu, ld = rt.lay["g"], rt.lay["u"], rt.lay["d"]
        ref = labelled[0][0]
        result: Dict = {}
        for n_tok in n_toks:
            P = n_tok * K
            n_sets = min(max_sets, n_rows // P)
            if n_sets < 1:
                print(f"  [B] {tag}: N={n_tok} skipped ({n_rows} rows, {P} needed)", flush=True)
                continue
            sets = [torch.arange(i * P, (i + 1) * P, dtype=torch.int32).to(dev) for i in range(n_sets)]
            per_pair = n_tok > 1
            x = rnd(n_tok, lg.T)
            w = torch.rand((P,), generator=gen).to(dev)
            xg, xu, zr = rnd(P if per_pair else 1, lg.T), rnd(P if per_pair else 1, lu.T), rnd(P, ld.T)

            def core(c, sl):
                rt.r39k = c
                return rt.decode_core(x, rows, sl, w, n_tok, LIM)

            def gemv3(c, sl):
                if c is None:
                    return (qtip_gemv(rows, sl, lg, xg, per_pair), qtip_gemv(rows, sl, lu, xu, per_pair),
                            qtip_gemv(rows, sl, ld, zr, True))
                g, u = c.gemv_gu(rows, sl, lg, lu, xg, xu, per_pair)
                return (g, u, c.gemv(rows, sl, ld, zr, True))

            fn_of = {"core": core, "gemv": gemv3}
            graphs = {(kind, label): [capture(lambda c=c, sl=sl, f=fn_of[kind]: f(c, sl)) for sl in sets]
                      for kind in kinds for label, c in labelled}
            n_eq = 0
            for kind in kinds:                                              # E3: replayed graphs, v1 == v2
                for i in range(n_sets):
                    for label, _ in labelled:
                        graphs[(kind, label)][i][0].replay()
                    sync()
                    for label, _ in labelled[1:]:
                        if not bits_eq(graphs[(kind, ref)][i][1], graphs[(kind, label)][i][1]):
                            raise AssertionError(f"{tag}: graph replay N={n_tok} {kind} set {i}: [{label}] differs "
                                                 f"from {ref}")
                        n_eq += 1
            print(f"  [E3] {tag}: N={n_tok}: {n_eq} replayed graph outputs equal to {ref} bitwise "
                  f"({n_sets} row sets of {P})", flush=True)
            per = {key: [] for key in graphs}
            b2b = {key: [] for key in graphs}
            for _ in range(2):                                              # two alternating rounds
                for kind in kinds:
                    for label, _c in labelled:
                        a, m = time_calls([g.replay for g, _o in graphs[(kind, label)]], calls)
                        per[(kind, label)].append(a)
                        b2b[(kind, label)].append(m)
            for kind in kinds:
                what = "decode_core" if kind == "core" else "3 GEMVs only"
                for label, _c in labelled:
                    a = np.concatenate(per[(kind, label)]) * 1e6
                    med = float(np.median(a))
                    bb = float(np.mean(b2b[(kind, label)])) * 1e6
                    result[(n_tok, kind, label)] = (med, bb)
                    r0 = result[(n_tok, kind, ref)]
                    rel = "" if label == ref else f" | {ref}/this: x{r0[0] / med:.3f} (median), x{r0[1] / bb:.3f} (back-to-back)"
                    tok = (f" | x{n_moe} layers = {med * n_moe / 1000.0:.2f} ms/token" if n_tok == 1 else "")
                    if quiet:
                        continue
                    print(f"  [B] {tag}: graph replay, N={n_tok}, {what:13s} [{label}]: median {med:8.1f} us "
                          f"(p10 {float(np.percentile(a, 10)):.1f}, p90 {float(np.percentile(a, 90)):.1f}; rounds "
                          f"{', '.join(f'{float(np.median(r)) * 1e6:.1f}' for r in per[(kind, label)])}), "
                          f"back-to-back mean {bb:8.1f} us over 2 x {calls} calls{tok}{rel}", flush=True)
            del graphs
            if eager and n_tok == 1:
                for label, c in labelled:
                    a, m = time_calls([(lambda c=c, sl=sl: core(c, sl)) for sl in sets], calls)
                    a = a * 1e6
                    result[(n_tok, "eager", label)] = (float(np.median(a)), m * 1e6)
                    print(f"  [B] {tag}: eager (Python + launches), N=1, decode_core [{label}]: median "
                          f"{float(np.median(a)):8.1f} us, back-to-back mean {m * 1e6:8.1f} us over {calls} calls",
                          flush=True)
        rt.r39k = None
        sync()
        gc.collect()
        torch.cuda.empty_cache()
        return result

    def sweep_layer(tag: str, rt: QtipLayerRuntime, rows: torch.Tensor, n_rows: int) -> None:
        grid = [qtip_kernel_cfg("v2", warps=wp, pf=pf, x_shared=xs, fuse_gu=True)
                for wp in (4, 8, 16) for pf in (0, 2, 4, 8) for xs in (True, False)]
        grid.append(qtip_kernel_cfg("v2", fuse_gu=False))
        labelled = [("v1", None)] + [(c.describe(), c) for c in dict.fromkeys(grid)]
        for _lbl, c in labelled[1:]:
            c.precompile()
        print(f"  [SWEEP] {tag}: decode_core N=1 in a graph, {len(labelled) - 1} v2 settings", flush=True)
        res = bench_layer(tag + " sweep", rt, rows, n_rows, labelled, n_toks=(1,), kinds=("core",), eager=False,
                          max_sets=4, quiet=True)
        order = sorted(((v[0], v[1], k[2]) for k, v in res.items()), key=lambda t: t[0])
        v1_med = res[(1, "core", "v1")][0]
        print(f"  [SWEEP] {tag}: sorted by median us per decode_core, 2 x {calls} calls each (v1 = {v1_med:.1f} us; "
              f"x{n_moe} layers = {v1_med * n_moe / 1000.0:.2f} ms/token):", flush=True)
        for med, bb, label in order:
            print(f"      {med:8.1f} us  (back-to-back {bb:8.1f})  v1/this x{v1_med / med:.3f}  "
                  f"{med * n_moe / 1000.0:6.2f} ms/token  {label}", flush=True)

    labelled = [("v1", None), ("v2", cfg)]
    n_total = 0

    # a) the synthetic layers of --selftest: D = 64, one k = 1 row tile and three k = 2 row tiles per projection, K = 2
    tmp_dir = os.path.join(tempfile.gettempdir(), "r39k_qtip_kernel_selftest")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    _port_test_archives(tmp_dir)
    arch = QtipArchive(os.path.join(tmp_dir, "layer_01_qtip_hyb.pt"), mmap_slab=False)
    rt = QtipLayerRuntime(arch, dev, slab_in_vram=True)
    rows = rt.expert_slab_u8.to(dev)
    n_small = int(rows.shape[0])
    n_total += check_layer("synthetic D=64 (as --selftest)", rt, rows, n_small, 2)
    wide = torch.zeros((n_small, int(rows.shape[1]) + 48), dtype=torch.uint8, device=dev)
    wide[:, : int(rows.shape[1])] = rows
    n_total += check_layer("synthetic D=64, row stride = row + 48 B (pool convention)", rt, wide, n_small, 2)
    del rt, rows, wide, arch
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # b) random slabs with the production row layouts
    n_rows = max(16, int(bench_rows))
    for rule in ("A", "GU125"):
        fa = _r39k_fake_arch(rule, n_rows, seed=3900 + len(rule))
        rb = fa.preset.expert_row_bytes
        rt = QtipLayerRuntime(fa, dev, slab_in_vram=True)
        rows = rt.expert_slab_u8.to(dev)
        fa.expert_slab_u8 = None                                            # keep one copy only
        tag = f"random slab, rule {rule} ({rb:,} B/row, {n_rows} rows)"
        n_total += check_layer(tag, rt, rows, n_rows, 8)
        bench_layer(tag, rt, rows, n_rows, labelled)
        if sweep and rule == "A" and not fold:
            sweep_layer(tag, rt, rows, n_rows)
        del rt, rows, fa
        gc.collect()
        torch.cuda.empty_cache()

    # c) real rows of one layer file
    if fold:
        pts = sorted(glob.glob(os.path.join(fold, "layer_*_qtip_hyb.pt")))
        if not pts:
            raise RuntimeError(f"--fold {fold}: no layer_*_qtip_hyb.pt file")
        pick = pts[len(pts) // 2] if layer < 0 else os.path.join(fold, f"layer_{int(layer):02d}_qtip_hyb.pt")
        if not os.path.exists(pick):
            raise RuntimeError(f"--kbench-layer {layer}: {pick} does not exist")
        t0 = time.perf_counter()
        arch = QtipArchive(pick, mmap_slab=False)       # packed in RAM: no .slab sidecar is read or written
        n_real = min(n_rows, int(arch.expert_slab_u8.shape[0]))
        if n_real < 16:
            raise RuntimeError(f"{pick}: {n_real} rows, the test needs at least 16")
        sub = SimpleNamespace(preset=arch.preset, layer_id=arch.layer_id,
                              expert_slab_u8=arch.expert_slab_u8[:n_real].contiguous(), signs=arch.signs,
                              hyb_luts=arch.hyb_luts, row_of=None, missing=torch.empty(0, dtype=torch.long))
        rule_real, rb = arch.rule, arch.preset.expert_row_bytes
        del arch
        rt = QtipLayerRuntime(sub, dev, slab_in_vram=True)
        rows = rt.expert_slab_u8.to(dev)
        sub.expert_slab_u8 = None
        tag = f"REAL rows, {os.path.basename(pick)} (rule {rule_real}, {rb:,} B/row, first {n_real} rows)"
        print(f"  loaded {pick} in {time.perf_counter() - t0:.1f}s", flush=True)
        n_total += check_layer(tag, rt, rows, n_real, 8)
        bench_layer(tag, rt, rows, n_real, labelled)
        if sweep:
            sweep_layer(tag, rt, rows, n_real)
        del rt, rows, sub
        gc.collect()
        torch.cuda.empty_cache()

    print(f"[SELFTEST-KERNEL PASSED] {n_total} eager comparisons and every replayed graph output: v2 == v1 bitwise. "
          f"The timings above are per layer and cold; they do not include the backbone or the pool traffic.", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Round 3.9 QTIP HYB CUDA-Graph Serving Runtime, CLI & OpenAI HTTP Server")
    ap.add_argument("--selftest", action="store_true", help="Run standalone GPU CUDAGraph + pinned pool self-test")
    # [R3.9S] second self-test: only what the port added (the shipped --selftest stays as it was)
    ap.add_argument("--selftest-port", action="store_true",
                    help="[R3.9S] GPU self-test of the port only: pair_output, masked pairs, routed ids of -1, the "
                         "reg_topk fix, production pool v6 on QTIP sidecars (graph == eager, v6 == QTIP pool)")
    # [R3.9K] third self-test: expert kernel v1 against v2 (bitwise) + cold micro-benchmark; see run_selftest_kernel
    ap.add_argument("--selftest-kernel", action="store_true",
                    help="[R3.9K] GPU: expert kernel v2 == v1 bitwise (synthetic slabs; real rows with --fold) and a "
                         "cold micro-benchmark, eager and CUDA-graph replay")
    ap.add_argument("--fold", default="",
                    help="[R3.9K] with --selftest-kernel: archive folder; adds real rows of one layer file to the test")
    ap.add_argument("--kbench-calls", type=int, default=2000, help="[R3.9K] timed calls per measurement (at least 2000)")
    ap.add_argument("--kbench-rows", type=int, default=64,
                    help="[R3.9K] slab rows per test layer held in VRAM (64 rows = 0.4 GiB; the timed calls walk them)")
    ap.add_argument("--kbench-layer", type=int, default=-1, help="[R3.9K] layer id of the real file (-1 = the middle one)")
    ap.add_argument("--kbench-sweep", action="store_true",
                    help="[R3.9K] also time a grid of v2 settings (warps x load-ahead x activation source)")
    ap.add_argument("--pkg-dir", default=DEFAULT_PKG, help="Path to runtime directory (claude_r39_apex_fold / runtime)")
    ap.add_argument("--backbone-dir", default=DEFAULT_BB, help="Path to GLM53-backbone-r34h")
    ap.add_argument("--fold-dir", default=DEFAULT_FOLD, help="Directory containing layer_XX_qtip_hyb.pt archives")
    ap.add_argument("--merged-routers", default="", help="Optional path to merged_routers_bf16.pt (auto-detected in fold-dir)")
    ap.add_argument("--prune-spec", default="", help="Optional extra prune spec JSON")
    ap.add_argument("--vocab-keep", default="", help="Optional R3.4-V vocab keep JSON")
    ap.add_argument("--expert-pool-gib", type=float, default=18.0, help="VRAM hot expert slot pool size in GiB (18.0 CLI / 14.0 65k serve)")
    # [R3.9S] default None = R33_REGISTER_GIB when set (the production variable), else the shipped 48.0
    ap.add_argument("--register-gib", type=float, default=None, help="Host DDR5 cudaHostRegister lock budget in GiB")
    ap.add_argument("--graphs", action="store_true", default=True, help="Capture & run GraphDecoder")
    ap.add_argument("--no-graphs", dest="graphs", action="store_false")
    ap.add_argument("--mtp", action="store_true", help="Load MTP Layer 45 and enable speculative decoding")
    ap.add_argument("--no-prefix-cache", dest="prefix_cache", action="store_false", default=True)
    ap.add_argument("--t-max", type=int, default=2048, help="Max sequence length for static GraphDecoder caches")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--thinking", action="store_true", help="Enable <think> reasoning mode by default")
    ap.add_argument("--serve", action="store_true", help="Launch multi-turn OpenAI HTTP server (glm53_fold_serve)")
    ap.add_argument("--port", type=int, default=8088, help="HTTP server port when --serve is active")
    ap.add_argument("--prompt", default="Explain why tail-biting trellis quantization approaches the Shannon bound.")

    # Dynamically attach multi-turn OpenAI server flags when glm53_fold_serve is available
    for cand in (HERE, os.path.join(os.path.dirname(HERE), "runtime"), DEFAULT_PKG):
        if cand and os.path.isdir(cand) and cand not in sys.path:
            sys.path.insert(0, cand)
    try:
        import glm53_fold_serve as _S  # type: ignore
        _S.add_serve_args(ap)
    except Exception:
        _S = None

    args = ap.parse_args()

    if args.selftest:
        run_selftest()
        return 0
    if args.selftest_port:                       # [R3.9S]
        run_selftest_port()
        return 0
    if args.selftest_kernel:                     # [R3.9K] kernel v1 against v2; needs the GPU
        run_selftest_kernel(fold=args.fold, calls=max(2000, int(args.kbench_calls)), bench_rows=args.kbench_rows,
                            layer=args.kbench_layer, sweep=args.kbench_sweep)
        return 0

    if os.path.isdir(args.pkg_dir) and args.pkg_dir not in sys.path:
        sys.path.insert(0, args.pkg_dir)
    from glm53_fold_generate import RealTokenizer  # type: ignore

    tok = RealTokenizer(args.backbone_dir)
    # [R3.9S] the class is built on the Round 3.8 runner at first use (a bare name does not exist in this module)
    runner = qtip_runner_class()(
        backbone_dir=args.backbone_dir,
        fold_dir=args.fold_dir,
        merged_routers_path=args.merged_routers,
        expert_pool_gib=args.expert_pool_gib,
        register_gib=(args.register_gib if args.register_gib is not None
                      else float(os.environ.get("R33_REGISTER_GIB", "") or 48.0)),   # [R3.9S] see --register-gib
        capture_graphs=args.graphs,
        t_max=args.t_max,
        load_mtp=args.mtp,
        vocab_keep=args.vocab_keep or None,
        prune_spec=args.prune_spec or None,
        prefix_cache=args.prefix_cache,
        pkg_dir=args.pkg_dir,
    )

    if args.serve:
        import glm53_fold_serve as S_mod  # type: ignore
        S_mod.serve_main(runner, tok, args)
        return 0

    prompt_ids = tok.encode_chat(args.prompt, thinking=args.thinking)
    res = runner.generate(prompt_ids, max_new_tokens=args.max_new_tokens, eos_ids=tok.eos_ids, use_mtp=args.mtp)
    text = tok.decode(res["generated_ids"])
    print(f"\n[OUTPUT]\n{text}\n\n[STATS] {json.dumps({k: v for k, v in res.items() if k not in ('ids', 'generated_ids')})}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
