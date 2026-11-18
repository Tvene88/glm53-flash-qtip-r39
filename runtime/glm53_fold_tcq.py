#!/usr/bin/env python3
"""
glm53_fold_tcq.py  [Round 3.2, Claude]
=====================================
Data-free trellis-coded quantization (QTIP-style bitshift trellis) for Hadamard-rotated, row-normalised weights.

Code
  * State = last L = 12 bits of the row bit-stream (4096 states). Emitting one weight shifts in k bits (k = 2 or 3):
        s_next = ((s << k) | b) & 4095
    and the reconstructed (normalised) weight is cb_k[s_next] (a 4096-entry scalar table per k).
  * Free start: the first state's 12 bits are stored, then k bits per further weight, so a row of T weights costs
    exactly 12 + k (T - 1) bits (byte-locked: fixed per row length and k).
  * Random-access decode: weight t = cb_k[ bits[k t : k t + 12] ] (MSB-first window), i.e. a table lookup per weight.

Encoder: exact Viterbi (min-sum) per row on the GPU (NVRTC kernel). All 2^k predecessors of a state share its lower
(12 - k) bits, so each step is 2^(12-k) min-of-2^k reductions plus 4096 branch metrics; back-pointers are k bits per
predecessor group per step. Pure weight MSE: no Hessian, no data.

Codebooks: Lloyd iterations on synthetic N(0, 1) sequences (data-free), starting from Gaussian quantiles placed by a
fixed pseudo-random permutation of the states.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import torch

L_BITS = 12
N_STATES = 1 << L_BITS
_CACHE: Dict[str, object] = {}

_VITERBI_SRC = r"""
extern "C" __global__ void tcq_viterbi_kernel(
    const float* __restrict__ X, const float* __restrict__ cbk, unsigned char* __restrict__ bp,
    int* __restrict__ states_out, float* __restrict__ cost_out, int T, int k)
{
    // one block per row; 256 threads; cost double-buffered in shared memory
    extern __shared__ float sm[];
    float* cb = sm;                 // 4096
    float* c0 = sm + 4096;          // 4096
    float* c1 = sm + 8192;          // 4096
    const int r = blockIdx.x;
    const float* x = X + (size_t)r * T;
    const int NG = 4096 >> k;       // predecessor groups
    const int shift = 12 - k;
    const int nchoice = 1 << k;
    const size_t bp_row = (size_t)r * T * NG;
    for (int s = threadIdx.x; s < 4096; s += blockDim.x) {
        cb[s] = cbk[s];
    }
    __syncthreads();
    float x0 = x[0];
    for (int s = threadIdx.x; s < 4096; s += blockDim.x) {
        float d = x0 - cb[s];
        c0[s] = d * d;
    }
    __syncthreads();
    float* cur = c0;
    float* nxt = c1;
    for (int t = 1; t < T; ++t) {
        float xt = x[t];
        for (int q = threadIdx.x; q < NG; q += blockDim.x) {
            float best = cur[q];
            int arg = 0;
            for (int j = 1; j < nchoice; ++j) {
                float v = cur[(j << shift) | q];
                if (v < best) { best = v; arg = j; }
            }
            bp[bp_row + (size_t)t * NG + q] = (unsigned char)arg;
            for (int b = 0; b < nchoice; ++b) {
                int s = (q << k) | b;
                float d = xt - cb[s];
                nxt[s] = best + d * d;
            }
        }
        __syncthreads();
        float* tmp = cur; cur = nxt; nxt = tmp;
    }
    // final argmin over states
    __shared__ float bv[256];
    __shared__ int bi[256];
    float best = 3.4e38f; int arg = 0;
    for (int s = threadIdx.x; s < 4096; s += blockDim.x) {
        if (cur[s] < best) { best = cur[s]; arg = s; }
    }
    bv[threadIdx.x] = best; bi[threadIdx.x] = arg;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off && bv[threadIdx.x + off] < bv[threadIdx.x]) {
            bv[threadIdx.x] = bv[threadIdx.x + off]; bi[threadIdx.x] = bi[threadIdx.x + off];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        cost_out[r] = bv[0];
        // traceback: states_out[r, t]
        int s = bi[0];
        int* so = states_out + (size_t)r * T;
        so[T - 1] = s;
        for (int t = T - 1; t >= 1; --t) {
            int q = s >> k;
            int j = bp[bp_row + (size_t)t * NG + q];
            s = (j << shift) | q;
            so[t - 1] = s;
        }
    }
}
"""


def _kernel():
    if "viterbi" not in _CACHE:
        k = torch.cuda._compile_kernel(_VITERBI_SRC, "tcq_viterbi_kernel")
        k.set_shared_memory_config(3 * 4096 * 4)
        _CACHE["viterbi"] = k
    return _CACHE["viterbi"]


@torch.no_grad()
def viterbi_states(X: torch.Tensor, cb: torch.Tensor, k: int, rows_per_launch: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    """X fp32 [R, T] (normalised), cb fp32 [4096] -> (states int32 [R, T], cost fp32 [R]). Exact min-sum Viterbi."""
    assert X.is_cuda and X.dtype == torch.float32
    R, T = X.shape
    NG = N_STATES >> k
    states = torch.empty((R, T), dtype=torch.int32, device=X.device)
    cost = torch.empty((R,), dtype=torch.float32, device=X.device)
    kern = _kernel()
    for r0 in range(0, R, rows_per_launch):
        r1 = min(R, r0 + rows_per_launch)
        n = r1 - r0
        bp = torch.empty((n, T, NG), dtype=torch.uint8, device=X.device)
        kern(grid=(n, 1, 1), block=(256, 1, 1), shared_mem=3 * 4096 * 4,
             args=[X[r0:r1].contiguous(), cb.float().contiguous(), bp, states[r0:r1], cost[r0:r1], int(T), int(k)])
        del bp
    return states, cost


def row_bits(T: int, k: int) -> int:
    return L_BITS + k * (T - 1)


def row_bytes(T: int, k: int) -> int:
    """Byte-locked row size (+2 padding bytes so the 3-byte window read never leaves the row)."""
    return (row_bits(T, k) + 7) // 8 + 2


@torch.no_grad()
def pack_states(states: torch.Tensor, k: int) -> torch.Tensor:
    """states int32 [R, T] -> uint8 [R, row_bytes(T, k)] MSB-first stream: s_0 (12 bits), then the low k bits of s_t."""
    R, T = states.shape
    nb = row_bits(T, k)
    bits = torch.zeros((R, nb), dtype=torch.uint8, device=states.device)
    s0 = states[:, 0]
    for i in range(L_BITS):
        bits[:, i] = ((s0 >> (L_BITS - 1 - i)) & 1).to(torch.uint8)
    low = states[:, 1:] & ((1 << k) - 1)
    for i in range(k):
        bits[:, L_BITS + i: nb: k] = ((low >> (k - 1 - i)) & 1).to(torch.uint8)
    pad = (-nb) % 8
    if pad:
        bits = torch.cat([bits, torch.zeros((R, pad), dtype=torch.uint8, device=bits.device)], dim=1)
    w = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=bits.device)
    packed = (bits.view(R, -1, 8) * w).sum(-1).to(torch.uint8)
    out = torch.zeros((R, row_bytes(T, k)), dtype=torch.uint8, device=states.device)
    out[:, : packed.shape[1]] = packed
    return out


@torch.no_grad()
def unpack_indices(packed: torch.Tensor, T: int, k: int) -> torch.Tensor:
    """uint8 [R, row_bytes] -> int64 [R, T] table indices (the 12-bit windows), vectorised random-access decode."""
    R = packed.shape[0]
    t = torch.arange(T, device=packed.device, dtype=torch.long)
    bitpos = k * t
    byte = bitpos >> 3
    off = bitpos & 7
    p = packed.long()
    w24 = (p[:, byte] << 16) | (p[:, byte + 1] << 8) | p[:, byte + 2]
    return (w24 >> (24 - L_BITS - off)) & (N_STATES - 1)


@torch.no_grad()
def train_codebook(k: int, n_rows: int = 384, T: int = 2048, iters: int = 12, seed: int = 1234,
                   device: torch.device = torch.device("cuda")) -> torch.Tensor:
    """Data-free Lloyd training of the 4096-entry table for k bits/weight on synthetic N(0, 1) rows."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(N_STATES, generator=g)
    q = (torch.arange(N_STATES, dtype=torch.float64) + 0.5) / N_STATES
    gauss = math.sqrt(2.0) * torch.erfinv(2.0 * q - 1.0)
    cb = torch.empty(N_STATES, dtype=torch.float64)
    cb[perm] = gauss
    cb = cb.float().to(device)
    gd = torch.Generator(device=device).manual_seed(seed + 1)
    X = torch.randn((n_rows, T), generator=gd, device=device)
    for _ in range(iters):
        st, _ = viterbi_states(X, cb, k)
        flat_s = st.reshape(-1).long()
        sums = torch.zeros(N_STATES, device=device).index_add_(0, flat_s, X.reshape(-1))
        cnt = torch.zeros(N_STATES, device=device).index_add_(0, flat_s, torch.ones_like(X.reshape(-1)))
        upd = cnt > 0
        cb = torch.where(upd, sums / cnt.clamp_min(1.0), cb)
    return cb


@torch.no_grad()
def mse_db(X: torch.Tensor, cb: torch.Tensor, k: int) -> float:
    st, _ = viterbi_states(X, cb, k)
    rec = cb[st.long()]
    mse = float((X - rec).pow(2).mean())
    return -10.0 * math.log10(mse / float(X.pow(2).mean()))
