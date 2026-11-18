"""R3.7: routed-expert TCQ GEMV candidates for rule-A archives (k = 2, 12-bit state).

Import is CPU-only (numpy). CUDA is touched only inside `prepare` and `gemv`.
The packed archive is unchanged. Nothing here is on by default: the kernel lab
(build_run/kernel_lab_r37.py) measures every variant against tcq_gemv_v2 on real
archive rows, and only a variant that passes its gate is wired into decode_core.

Two families, selected by the variant name:

  e_<lut>          EXACT candidates. Same bit windows, the same four-product chain per
                   lane, the same shuffle reduction and the same epilogue as v2 / v3, so
                   the output is expected to be bit-identical to v2. What changes is how
                   bytes reach the lane: two aligned 32-bit words instead of four byte
                   loads, and one k = 2 table instead of three.

  r_<lut>_a<n>     ROUNDING candidates. Each lane decodes a whole 32-bit word (16 weights)
                   per step; a warp reads 128 contiguous code bytes per step and gets the
                   ten look-ahead bits from its neighbour lane by shuffle. <n> running
                   sums per lane (1, 2 or 4). The fp32 summation order differs from v2.

  <lut> is where the 4,096-entry table lives:
      sh16  shared memory, half precision (8 KiB per block)
      sh32  shared memory, single precision (16 KiB per block)
      g16   global memory, half precision (served from the SM's data cache)
      g32   global memory, single precision
      p14   global memory, a 16,384-entry table of weight PAIRS (64 KiB): one lookup
            yields two consecutive weights (ROUNDING family only)

Bit layout (glm53_fold_tcq.pack_states): a row is an MSB-first stream; weight t's state is
the 12-bit window starting at stream bit 2t. A 32-bit big-endian word m therefore holds
the first bits of weights 16m .. 16m+15, and the last of them needs 10 bits of word m+1.
Rows have T/4 + 4 code bytes, i.e. T/16 + 1 words, so word m+1 always exists.
"""
from __future__ import annotations

import numpy as np

LUTS_EXACT = ("sh16", "sh32", "g16", "g32")
LUTS_ROUND = ("sh16", "sh32", "g16", "g32", "p14")


def variants():
    """Every variant name the lab can ask for."""
    out = [f"e_{lut}" for lut in LUTS_EXACT]
    for lut in LUTS_ROUND:
        for n in (1, 2, 4):
            out.append(f"r_{lut}_a{n}")
    return out


def parse(variant):
    parts = variant.split("_")
    if parts[0] == "e" and len(parts) == 2 and parts[1] in LUTS_EXACT:
        return {"family": "e", "lut": parts[1], "nacc": 1}
    if (parts[0] == "r" and len(parts) == 3 and parts[1] in LUTS_ROUND
            and parts[2] in ("a1", "a2", "a4")):
        return {"family": "r", "lut": parts[1], "nacc": int(parts[2][1:])}
    raise ValueError(f"unknown R3.7 TCQ kernel variant {variant!r}")


def is_exact_candidate(variant):
    return parse(variant)["family"] == "e"


# ----------------------------------------------------------------------------------------------------------------------
# CUDA source
# ----------------------------------------------------------------------------------------------------------------------
_HEAD = r'''
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(512) @NAME@(
    const unsigned char* __restrict__ rows, int row_bytes,
    const int* __restrict__ slots, const float* __restrict__ table,
    const __half* __restrict__ table16, const __half2* __restrict__ table14,
    const float* __restrict__ x, float* __restrict__ y,
    int off_map, int off_scale, int off_code, int code_stride,
    int O, int T, int per_pair)
{
@LUT_DECL@
    const int pr = blockIdx.y;
    const int slot = slots[pr];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    if (slot < 0) {                                  // masked pair: defined (zero) output, no slab access
        for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < O; r += gridDim.x * blockDim.x)
            y[(size_t)pr * O + r] = 0.f;
        return;
    }
@LUT_LOAD@
    const unsigned char* base = rows + (long long)slot * row_bytes;
    const unsigned short* rmap = (const unsigned short*)(base + off_map);
    const __half* scale = (const __half*)(base + off_scale);
    const float* xp = x + (size_t)(per_pair ? pr : 0) * T;
    for (int r = blockIdx.x * nw + warp; r < O; r += gridDim.x * nw) {
        const unsigned int* pw = (const unsigned int*)(base + off_code + (size_t)r * code_stride);
@BODY@
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
        if (lane == 0) y[(size_t)pr * O + rmap[r]] = acc * __half2float(scale[r]);
    }
}
'''

_LUT = {
    "sh16": ("    __shared__ __half tab[4096];",
             "    for (int i = threadIdx.x; i < 4096; i += blockDim.x) tab[i] = __float2half_rn(table[i]);\n"
             "    __syncthreads();",
             "__half2float(tab[%s])"),
    "sh32": ("    __shared__ float tab[4096];",
             "    for (int i = threadIdx.x; i < 4096; i += blockDim.x) tab[i] = table[i];\n"
             "    __syncthreads();",
             "tab[%s]"),
    "g16": ("", "", "__half2float(table16[%s])"),
    "g32": ("", "", "table[%s]"),
    "p14": ("", "", None),
}

# EXACT family: v2's arithmetic. Lane L owns weights 4L + 128i .. +3, i.e. the 18 bits that start at byte L + 32i.
_BODY_EXACT = r'''        float acc = 0.f;
        for (int t = lane * 4; t < T; t += 128) {
            const int byte = t >> 2;
            const int q = byte >> 2, sh = (byte & 3) << 3;
            const unsigned long long reg =
                ((unsigned long long)__byte_perm(pw[q], 0, 0x0123) << 32) | __byte_perm(pw[q + 1], 0, 0x0123);
            const unsigned int b = (unsigned int)(reg >> (32 - sh));       // the 32 bits that start at `byte`
            const float4 v = *(const float4*)(xp + t);
            acc += @W0@ * v.x;
            acc += @W1@ * v.y;
            acc += @W2@ * v.z;
            acc += @W3@ * v.w;
        }'''

# ROUNDING family: lane L owns word L + 32i (16 weights); a warp step reads 128 contiguous code bytes.
_BODY_ROUND_HEAD = r'''        @ACC_DECL@
        const int nwords = T >> 4;
        for (int m = lane; m < nwords; m += 32) {
            const unsigned int w = __byte_perm(pw[m], 0, 0x0123);
            unsigned int nx = __shfl_down_sync(0xffffffff, w, 1);          // word m+1 lives in the next lane
            if (lane == 31) nx = __byte_perm(pw[m + 1], 0, 0x0123);
            const unsigned long long reg = ((unsigned long long)w << 32) | nx;
            const float4* xv = (const float4*)(xp + (m << 4));
            const float4 v0 = xv[0], v1 = xv[1], v2 = xv[2], v3 = xv[3];
'''
_BODY_ROUND_TAIL = r'''        }
        @ACC_SUM@'''

_XNAME = ["v0.x", "v0.y", "v0.z", "v0.w", "v1.x", "v1.y", "v1.z", "v1.w",
          "v2.x", "v2.y", "v2.z", "v2.w", "v3.x", "v3.y", "v3.z", "v3.w"]


def kernel_name(variant):
    return "tcq_gemv_r37_" + variant + "_kernel"


def source(variant):
    """Returns (kernel_name, CUDA source) for one variant."""
    v = parse(variant)
    decl, load, lut = _LUT[v["lut"]]
    if v["family"] == "e":
        body = _BODY_EXACT
        for j, shift in enumerate((20, 18, 16, 14)):
            body = body.replace(f"@W{j}@", lut % f"(b >> {shift}) & 4095u")
    else:
        n = v["nacc"]
        lines = []
        if v["lut"] == "p14":
            # one lookup per pair of weights: 14-bit window starting at stream bit 4j of the word
            for j in range(8):
                lines.append(f"            const float2 p{j} = __half22float2(table14[(unsigned int)(reg >> {50 - 4 * j}) & 16383u]);")
            for i in range(16):
                comp = "x" if i % 2 == 0 else "y"
                lines.append(f"            a{i % n} += p{i // 2}.{comp} * {_XNAME[i]};")
        else:
            for i in range(16):
                idx = f"(unsigned int)(reg >> {52 - 2 * i}) & 4095u"
                lines.append(f"            a{i % n} += {lut % idx} * {_XNAME[i]};")
        acc_decl = "float " + ", ".join(f"a{i} = 0.f" for i in range(n)) + ";"
        acc_sum = {1: "float acc = a0;", 2: "float acc = a0 + a1;", 4: "float acc = (a0 + a1) + (a2 + a3);"}[n]
        body = (_BODY_ROUND_HEAD.replace("@ACC_DECL@", acc_decl) + "\n".join(lines) + "\n"
                + _BODY_ROUND_TAIL.replace("@ACC_SUM@", acc_sum))
    name = kernel_name(variant)
    src = (_HEAD.replace("@NAME@", name).replace("@LUT_DECL@", decl).replace("@LUT_LOAD@", load)
           .replace("@BODY@", body))
    if "@" in src:
        raise RuntimeError("unreplaced marker in R3.7 TCQ kernel source for " + variant)
    return name, src


# ----------------------------------------------------------------------------------------------------------------------
# CUDA wrappers (torch imported lazily)
# ----------------------------------------------------------------------------------------------------------------------
def rule_a(lay):
    return lay.n1 == 0 and lay.n3 == 0 and lay.n2 == lay.d_out


def check_layout(lay, variant, rows=None):
    """Raises ValueError if this layout cannot run `variant`. CPU-only checks."""
    v = parse(variant)
    if not rule_a(lay):
        raise ValueError("R3.7 TCQ kernels need a rule-A (all k = 2) archive")
    need = 512 if v["family"] == "r" else 128
    if lay.T % need:
        raise ValueError(f"{variant}: row length {lay.T} is not a multiple of {need}")
    rb = int(lay.rb[1])
    if rb < lay.T // 4 + 4:
        raise ValueError(f"{variant}: code rows have {rb} bytes, need at least {lay.T // 4 + 4}")
    if lay.off["c2"] % 4 or rb % 4 or int(lay.row_bytes) % 4 or lay.off["map"] % 2 or lay.off["scale"] % 2:
        raise ValueError(f"{variant}: code fields are not 4-byte aligned")
    if rows is not None and int(rows.stride(0)) != int(lay.row_bytes):
        raise ValueError(f"{variant}: rows stride differs from the layout's row size")


def prepare(lay):
    """Derived tables for one projection layout, on the layout's device. Call before graph capture."""
    import torch
    cached = getattr(lay, "r37_tables", None)
    if cached is not None and cached["tab32"].device == lay.tab.device:
        return cached
    tab32 = lay.tab[1].contiguous()                       # the k = 2 table, fp32 [4096]
    tab16 = tab32.to(torch.float16).contiguous()
    if not torch.equal(tab16.float(), tab32):
        raise ValueError("R3.7 TCQ kernels need a table that round-trips through fp16")
    idx = torch.arange(16384, device=tab32.device)
    tab14 = torch.stack([tab16[idx >> 2], tab16[idx & 4095]], dim=1).contiguous()   # [16384, 2] half
    lay.r37_tables = {"tab32": tab32, "tab16": tab16, "tab14": tab14}
    return lay.r37_tables


def gemv(rows, slots, lay, x_rot, x_per_pair, out=None, *, variant="e_sh16", warps=8, gx=64):
    """Same contract as glm53_fold_kernels.tcq_gemv_v2: rows uint8 [S, row_bytes] CUDA, slots int32 [P],
    x_rot fp32 [1 or P, T] -> fp32 [P, d_out]. Negative slots give zero rows."""
    import torch
    import glm53_fold_kernels as KR
    check_layout(lay, variant, rows)
    if lay.tab.device != rows.device:
        lay.tab = lay.tab.to(rows.device)
    tabs = prepare(lay)
    P = int(slots.numel())
    if out is None:
        out = torch.empty((P, lay.d_out), dtype=torch.float32, device=rows.device)
    elif (out.dtype != torch.float32 or tuple(out.shape) != (P, lay.d_out) or not out.is_contiguous()):
        raise ValueError(f"{variant}: out must be a contiguous fp32 [{P}, {lay.d_out}] tensor")
    if P == 0:
        return out
    xf = x_rot.float().contiguous()
    if xf.dim() != 2 or xf.shape[1] != lay.T or xf.shape[0] != (P if x_per_pair else 1):
        raise ValueError(f"{variant}: activation shape {tuple(xf.shape)} does not match P={P}, T={lay.T}, "
                         f"per_pair={bool(x_per_pair)}")
    # the kernels load 32-bit code words and float4 activations: a sliced view could break both
    if rows.data_ptr() % 4 or xf.data_ptr() % 16:
        raise ValueError(f"{variant}: rows must be 4-byte aligned and the activations 16-byte aligned")
    if int(lay.row_bytes) >= 2 ** 31:
        raise ValueError(f"{variant}: row size does not fit a 32-bit int")
    warps = int(warps)
    if warps < 1 or warps > 16:
        raise ValueError("warps must be in 1..16 (the kernels are built for at most 512 threads per block)")
    gx = max(1, min((lay.d_out + warps - 1) // warps, int(gx)))
    name, src = source(variant)
    kernel = KR._compile(name, src)
    kernel(grid=(gx, P, 1), block=(32 * warps, 1, 1), args=[
        rows, int(lay.row_bytes), slots.to(torch.int32), tabs["tab32"], tabs["tab16"], tabs["tab14"],
        xf, out, lay.off["map"], lay.off["scale"], lay.off["c2"], int(lay.rb[1]),
        lay.d_out, lay.T, 1 if x_per_pair else 0])
    return out


# ----------------------------------------------------------------------------------------------------------------------
# CPU references (numpy only): the bit windows each family reads
# ----------------------------------------------------------------------------------------------------------------------
def _be_words(codes):
    """[..., bytes] uint8 -> [..., bytes // 4] uint64 big-endian words, as __byte_perm(p[m], 0, 0x0123) yields."""
    c = np.asarray(codes, dtype=np.uint8)
    n = c.shape[-1] // 4
    w = c[..., :4 * n].reshape(c.shape[:-1] + (n, 4)).astype(np.uint64)
    return (w[..., 0] << np.uint64(24)) | (w[..., 1] << np.uint64(16)) | (w[..., 2] << np.uint64(8)) | w[..., 3]


def extract_states_round(codes, T):
    """Simulates the ROUNDING family: word m with word m+1 as look-ahead, sixteen 12-bit windows."""
    words = _be_words(codes)
    if T % 16 or words.shape[-1] < T // 16 + 1:
        raise ValueError("row length or packed length invalid for the word-streaming kernel")
    out = np.empty(words.shape[:-1] + (T,), dtype=np.uint16)
    for m in range(T // 16):
        reg = (words[..., m] << np.uint64(32)) | words[..., m + 1]
        for i in range(16):
            out[..., 16 * m + i] = ((reg >> np.uint64(52 - 2 * i)) & np.uint64(4095)).astype(np.uint16)
    return out


def extract_pairs_round(codes, T):
    """Simulates the pair-table variant: eight 14-bit windows per word. Returns [..., T // 2] indices."""
    words = _be_words(codes)
    if T % 16 or words.shape[-1] < T // 16 + 1:
        raise ValueError("row length or packed length invalid for the word-streaming kernel")
    out = np.empty(words.shape[:-1] + (T // 2,), dtype=np.uint16)
    for m in range(T // 16):
        reg = (words[..., m] << np.uint64(32)) | words[..., m + 1]
        for j in range(8):
            out[..., 8 * m + j] = ((reg >> np.uint64(50 - 4 * j)) & np.uint64(16383)).astype(np.uint16)
    return out


def extract_states_exact(codes, T):
    """Simulates the EXACT family: for the 4 weights at byte L + 32i, two aligned words and one shift."""
    words = _be_words(codes)
    if T % 4 or words.shape[-1] < T // 16 + 1:
        raise ValueError("row length or packed length invalid for the aligned-word kernel")
    out = np.empty(words.shape[:-1] + (T,), dtype=np.uint16)
    for t in range(0, T, 4):
        byte = t >> 2
        q, sh = byte >> 2, (byte & 3) << 3
        reg = (words[..., q] << np.uint64(32)) | words[..., q + 1]
        b = (reg >> np.uint64(32 - sh)) & np.uint64(0xFFFFFFFF)
        for j, shift in enumerate((20, 18, 16, 14)):
            out[..., t + j] = ((b >> np.uint64(shift)) & np.uint64(4095)).astype(np.uint16)
    return out
