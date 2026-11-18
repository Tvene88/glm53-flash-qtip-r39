"""R3.6 K1: rule-A TCQ GEMV, compact table, masked pairs, two-row union.

Import is CPU-only. Kernels are compiled/loaded only by explicit CUDA wrappers.
The packed archive is unchanged. Rule B uses the existing v2 implementation.
The arithmetic order of each output matches v2; GPU exactness remains a gate.
"""
from __future__ import annotations

import os
import numpy as np


TCQ_V3_SRC = r'''
#include <cuda_fp16.h>
extern "C" __global__ void tcq_gemv_v3_kernel(
    const unsigned char* __restrict__ rows, int row_bytes,
    const int* __restrict__ slots, const float* __restrict__ table,
    const float* __restrict__ x, float* __restrict__ y,
    int off_map, int off_scale, int off_code, int code_stride,
    int O, int T, int P, int per_pair, int use_union, int preserve)
{
    // One 8 KiB LUT, rather than v2's three 16 KiB LUTs plus shared x.
    // Archive tables are FP16, so this conversion is lossless (checked on CPU).
    __shared__ __half tab[4096];
    const int p0 = blockIdx.y;
    const int slot = slots[p0];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nw = blockDim.x >> 5;
    if (slot < 0) {
        if (!preserve)
            for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < O;
                 r += gridDim.x * blockDim.x) y[(size_t)p0 * O + r] = 0.f;
        return;
    }
    int p1 = -1;
    if (use_union) {
        // Each token's top-k has distinct experts, so N=2 has at most two
        // occurrences. Only the first occurrence owns both output rows.
        int count=0;
        for(int p=0;p<P;++p) if(slots[p]==slot) ++count;
        // Capture warm-up may initialise every slot to zero. Fall back to
        // independent pairs when more than two occurrences exist.
        if(count<=2) {
            for (int p = 0; p < p0; ++p) if (slots[p] == slot) return;
            for (int p = p0 + 1; p < P; ++p) if (slots[p] == slot) { p1 = p; break; }
        }
    }
    for (int i = threadIdx.x; i < 4096; i += blockDim.x)
        tab[i] = __float2half_rn(table[i]);
    __syncthreads();
    const unsigned char* base = rows + (long long)slot * row_bytes;
    const unsigned short* rmap = (const unsigned short*)(base + off_map);
    const __half* scale = (const __half*)(base + off_scale);
    const float* x0 = x + (size_t)(per_pair ? p0 : 0) * T;
    const float* x1 = x + (size_t)(per_pair && p1 >= 0 ? p1 : 0) * T;
    for (int r = blockIdx.x * nw + warp; r < O; r += gridDim.x * nw) {
        const unsigned char* p = base + off_code + (size_t)r * code_stride;
        float a0 = 0.f, a1 = 0.f;
        for (int t = lane * 4; t < T; t += 128) {
            // k=2, t divisible by four: every bit window is byte aligned.
            // All fields/row strides are 2-byte aligned, not necessarily 4.
            // Odd byte offsets use byte loads: an unaligned uint16 is unsafe.
            int byte = t / 4;
            unsigned int b;
            if ((byte & 1) == 0) {
                const unsigned short* hp = (const unsigned short*)(p + byte);
                unsigned int lo = hp[0], hi = hp[1];
                b = ((lo & 255u) << 24) | ((lo >> 8) << 16)
                  | ((hi & 255u) << 8) | (hi >> 8);
            } else {
                b = ((unsigned int)p[byte] << 24) | ((unsigned int)p[byte+1] << 16)
                  | ((unsigned int)p[byte+2] << 8)
                  | (byte+3 < code_stride ? p[byte+3] : 0);
            }
            float w0 = __half2float(tab[(b >> 20) & 4095]);
            float w1 = __half2float(tab[(b >> 18) & 4095]);
            float w2 = __half2float(tab[(b >> 16) & 4095]);
            float w3 = __half2float(tab[(b >> 14) & 4095]);
            float4 v0 = *(const float4*)(x0 + t);
            a0 += w0 * v0.x; a0 += w1 * v0.y;
            a0 += w2 * v0.z; a0 += w3 * v0.w;
            if (p1 >= 0) {
                float4 v1 = *(const float4*)(x1 + t);
                a1 += w0 * v1.x; a1 += w1 * v1.y;
                a1 += w2 * v1.z; a1 += w3 * v1.w;
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            a0 += __shfl_down_sync(0xffffffff, a0, off);
            a1 += __shfl_down_sync(0xffffffff, a1, off);
        }
        if (lane == 0) {
            float s = __half2float(scale[r]);
            y[(size_t)p0 * O + rmap[r]] = a0 * s;
            if (p1 >= 0) y[(size_t)p1 * O + rmap[r]] = a1 * s;
        }
    }
}
'''


FWHT_SRC = r'''
extern "C" __global__ void tcq_fwht_r36_kernel(
    const float* x, const float* signs, float* y, int D, double norm_d)
{
    const float norm = (float)norm_d; // torch CUDAKernel packs Python floats as doubles
    extern __shared__ float v[];
    int row = blockIdx.x;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        v[i] = x[(size_t)row * D + i] * signs[i];
    __syncthreads();
    for (int h = 1; h < D; h <<= 1) {
        for (int p = threadIdx.x; p < D / 2; p += blockDim.x) {
            int a = (p / h) * (2 * h) + (p % h), b = a + h;
            float va = v[a], vb = v[b];
            v[a] = va + vb; v[b] = va - vb;
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        y[(size_t)row * D + i] = v[i] * norm;
}
'''


def rule_a(lay):
    return lay.n1 == 0 and lay.n3 == 0 and lay.n2 == lay.d_out and lay.T % 128 == 0


def prepare_layouts(layouts, cpu_tables):
    """Call at model load, before graph capture. CPU metadata check, no slab read."""
    a = np.asarray(cpu_tables, dtype=np.float32)
    if a.shape != (3, 4096) or not np.isfinite(a).all():
        raise ValueError('R36 TCQ requires finite [3,4096] archive tables')
    if not np.array_equal(a, a.astype(np.float16).astype(np.float32)):
        raise ValueError('R36 compact LUT requires the archive FP16 table invariant')
    for lay in layouts.values():
        lay.r36_table_validated = True


def gemv(rows, slots, lay, x_rot, x_per_pair, out=None, gx=None, *, union=False, preserve=False):
    import torch
    import glm53_fold_kernels as KR
    if not rule_a(lay):
        # Rule-B (mixed bit rate) stays on the original arithmetic and layout.
        if preserve:
            raise ValueError('R36 overlap requires rule-A TCQ')
        return KR.tcq_gemv_v2(rows, slots, lay, x_rot, x_per_pair, out=out, gx=gx)
    if not getattr(lay, 'r36_table_validated', False):
        raise ValueError('R36 TCQ layout must be validated at load before capture')
    if lay.off['c2'] % 2 or lay.rb[1] % 2 or lay.row_bytes % 2:
        raise ValueError('R36 TCQ code fields must be halfword aligned')
    P = slots.numel()
    if out is None:
        out = torch.empty((P, lay.d_out), dtype=torch.float32, device=rows.device)
    if P == 0:
        return out
    if union and P > 16:
        raise ValueError('R36 union is limited to two top-8 token rows')
    warps = int(os.environ.get('R36_TCQ_WARPS', '4'))
    if warps not in (4, 8):
        raise ValueError('R36_TCQ_WARPS must be 4 or 8')
    gx = min((lay.d_out + warps - 1) // warps,
             int(gx if gx is not None else os.environ.get('R36_TCQ_GX', '64')))
    if gx <= 0:
        raise ValueError('R36_TCQ_GX must be positive')
    kernel = KR._compile('tcq_gemv_v3_kernel', TCQ_V3_SRC)
    kernel(grid=(gx, P, 1), block=(32 * warps, 1, 1), args=[
        rows, lay.row_bytes, slots.to(torch.int32), lay.tab[1],
        x_rot.float().contiguous(), out, lay.off['map'], lay.off['scale'],
        lay.off['c2'], lay.rb[1], lay.d_out, lay.T, P, int(x_per_pair), int(union), int(preserve)])
    return out


def sparse_source(original):
    src = original.replace('tcq_sparse_add_kernel(', 'tcq_sparse_add_r36_kernel(')
    # All threads of a pair have the same slot; return before any slab address.
    anchor = 'if (pr >= P) return;'
    if src.count(anchor) != 1:
        raise RuntimeError('R36 sparse kernel anchor changed; update the reviewed integration')
    src = src.replace(anchor, anchor + '\n    if (slots[pr] < 0) return;')
    return src


def sparse_add(rows, slots, lay, x_orig, x_per_pair, out):
    """Original deterministic sparse arithmetic, with negative-slot early return."""
    import torch
    import glm53_fold_kernels as KR
    if lay.k_sparse == 0 or slots.numel() == 0:
        return out
    src = sparse_source(KR._TCQ_SPARSE_SRC)
    kernel = KR._compile('tcq_sparse_add_r36_kernel', src)
    xf = x_orig.float().contiguous()
    kernel(grid=(slots.numel(), 1, 1), block=(min(256, ((lay.k_sparse+31)//32)*32), 1, 1),
           shared_mem=8 * lay.k_sparse, args=[
        rows, int(lay.row_bytes), slots.to(torch.int32), lay.off['sp'], lay.k_sparse,
        xf, int(xf.stride(0)) if x_per_pair else 0, out, int(out.stride(0)), slots.numel()])
    return out


def fwht(x, signs):
    """Fused sign and normalized butterfly. ROUNDING; separate opt-in flag."""
    import torch
    import glm53_fold_kernels as KR
    D = x.shape[-1]
    if D < 1 or D > 8192 or D & (D - 1):
        raise ValueError('R36 FWHT requires a power-of-two width <= 8192')
    y = torch.empty_like(x, dtype=torch.float32)
    kernel = KR._compile('tcq_fwht_r36_kernel', FWHT_SRC)
    kernel(grid=(x.numel() // D, 1, 1), block=(256, 1, 1), shared_mem=D * 4,
           args=[x.float().contiguous(), signs.float().contiguous(), y, D, float(D ** -0.5)])
    return y


def extract_states_reference(codes, T, k=2):
    """Independent MSB-first bitstream definition (no GPU / archive access)."""
    c = np.asarray(codes, dtype=np.uint8)
    bits = np.unpackbits(c, axis=-1, bitorder='big')
    if bits.shape[-1] < k * (T - 1) + 12:
        raise ValueError('truncated TCQ code row')
    out = np.zeros(c.shape[:-1] + (T,), dtype=np.uint16)
    for j in range(12):
        out |= bits[..., j:j + k * T:k].astype(np.uint16) << (11 - j)
    return out


def extract_states_v3(codes, T):
    """CPU simulation of the little-endian reads and four bit windows."""
    c = np.asarray(codes, dtype=np.uint8)
    if T % 128 or c.shape[-1] < (2 * T + 11 + 7) // 8:
        raise ValueError('rule-A width or packed length invalid')
    out = np.empty(c.shape[:-1] + (T,), dtype=np.uint16)
    for t in range(0, T, 4):
        b = t // 4
        lo = c[..., b].astype(np.uint32) | (c[..., b + 1].astype(np.uint32) << 8)
        last = c[..., b + 3].astype(np.uint32) if b + 3 < c.shape[-1] else np.uint32(0)
        hi = c[..., b + 2].astype(np.uint32) | (last << 8)
        w = ((lo & 255) << 24) | ((lo >> 8) << 16) | ((hi & 255) << 8) | (hi >> 8)
        for j in range(4):
            out[..., t + j] = (w >> (20 - 2 * j)) & 4095
    return out


def gemv_reference(codes, table, x, scales, row_map, *, window='definition', fma=True):
    """Simulate v2's lane accumulation and shuffle tree on small CPU inputs.

    codes [O,bytes], x [T]. Independent bit extraction is the default oracle.
    fma=True models NVRTC contraction with FP64 product+addition rounded to FP32.
    """
    x = np.asarray(x, dtype=np.float32)
    get = extract_states_reference if window == 'definition' else extract_states_v3
    states = get(codes, x.size)
    weights = np.asarray(table, dtype=np.float32)[states]
    O, T = weights.shape
    lanes = np.zeros((O, 32), dtype=np.float32)
    for t0 in range(0, T, 128):
        for j in range(4):
            idx = t0 + np.arange(32) * 4 + j
            if fma:
                lanes = (weights[:, idx].astype(np.float64) * x[idx].astype(np.float64)
                         + lanes.astype(np.float64)).astype(np.float32)
            else:
                lanes = (lanes + weights[:, idx] * x[idx]).astype(np.float32)
    for off in (16, 8, 4, 2, 1):
        old = lanes.copy()
        lanes[:, :32-off] = (old[:, :32-off] + old[:, off:]).astype(np.float32)
    result = np.empty(O, dtype=np.float32)
    result[np.asarray(row_map)] = lanes[:, 0] * np.asarray(scales, dtype=np.float16).astype(np.float32)
    return result


def union_plan(slots, token_rows=2):
    """CPU oracle: each physical slot may occur once per token row."""
    if token_rows != 2 or len(slots) % token_rows or len(slots) > 16:
        raise ValueError('union requires two equal top-k rows (k <= 8)')
    K = len(slots) // token_rows
    for row in (slots[:K], slots[K:]):
        live = [s for s in row if s >= 0]
        if len(live) != len(set(live)):
            raise ValueError('duplicate live slot in a single token row')
    groups = {}
    for p, s in enumerate(slots):
        if s >= 0:
            groups.setdefault(int(s), []).append(p)
    return list(groups.values())


def fwht_reference(x, signs):
    a = np.asarray(x, np.float32) * np.asarray(signs, np.float32)
    D = a.shape[-1]
    if D < 1 or D & (D - 1):
        raise ValueError('width must be a power of two')
    a = a.copy()
    h = 1
    while h < D:
        v = a.reshape(*a.shape[:-1], -1, 2 * h)
        left, right = v[..., :h].copy(), v[..., h:].copy()
        v[..., :h], v[..., h:] = left + right, left - right
        h *= 2
    return a * np.float32(D ** -0.5)
