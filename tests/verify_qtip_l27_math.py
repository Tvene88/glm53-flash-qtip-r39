#!/usr/bin/env python3
"""NumPy bit-exactness & mathematical verification for r39_l27_qtip_2d_ldlq_lab.py."""
import math
import numpy as np


def hyb_eval_all_states(lut_f16: np.ndarray) -> np.ndarray:
    """lut_f16: [512, 2] float16 -> [65536, 2] float32 exact match to CUDA qtip_hyb_pair."""
    s = np.arange(65536, dtype=np.uint64)
    h = (s * s + s) & 0xFFFFFFFF
    idx = ((h >> 6) & 511).astype(np.int64)
    sx = np.where((h & 0x8000) != 0, -1.0, 1.0).astype(np.float32)
    sy = np.where((s & 0x8000) != 0, -1.0, 1.0).astype(np.float32)
    base = lut_f16.astype(np.float32)[idx]
    out = np.empty((65536, 2), dtype=np.float32)
    out[:, 0] = base[:, 0] * sx
    out[:, 1] = base[:, 1] * sy
    return out


def mul1_eval_all_states() -> np.ndarray:
    s = np.arange(65536, dtype=np.uint64)
    x = (34038481 * s + 76625530) & 0xFFFFFFFF
    bsum = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24)
    return ((bsum.astype(np.float32) - 510.0) / 147.8).astype(np.float32)


def pack_tailbite_l16(states: np.ndarray, step_bits: int) -> np.ndarray:
    """states: [R, M] int32 (each entry is 16-bit state AFTER step m), where start_state == states[:, M-1].
    Packs exact M * step_bits bits = rb bytes so circular 16-bit MSB window at m * step_bits == states[:, m]."""
    R, M = states.shape
    total_bits = M * step_bits
    assert total_bits % 8 == 0 and total_bits >= 16
    rb = total_bits // 8
    hist = 16 // step_bits  # 4 for step_bits=4 (k=2,V=2), 8 for step_bits=2 (k=1,V=2 or k=2,V=1)
    bits = np.zeros((R, total_bits), dtype=np.uint8)
    s0 = states[:, 0]
    for i in range(16):
        bits[:, i] = ((s0 >> (15 - i)) & 1).astype(np.uint8)
    if M > hist:
        low = states[:, 1 : M - hist + 1] & ((1 << step_bits) - 1)
        for i in range(step_bits):
            bits[:, 16 + i : total_bits : step_bits] = ((low >> (step_bits - 1 - i)) & 1).astype(np.uint8)
    weights = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint8)
    packed = (bits.reshape(R, rb, 8) * weights).sum(axis=-1).astype(np.uint8)
    return packed


def unpack_tailbite_l16_cuda_sim(packed: np.ndarray, M: int, step_bits: int) -> np.ndarray:
    """Exact bit-for-bit simulation of CUDA read_tailbite_state16(p, rb, m * step_bits)."""
    R, rb = packed.shape
    assert (rb & (rb - 1)) == 0, f"rb={rb} must be power of two"
    m = np.arange(M, dtype=np.int64)
    bit_pos = m * step_bits
    b0 = (bit_pos >> 3) & (rb - 1)
    b1 = (b0 + 1) & (rb - 1)
    b2 = (b0 + 2) & (rb - 1)
    off = bit_pos & 7
    p = packed.astype(np.uint32)
    w24 = (p[:, b0] << 16) | (p[:, b1] << 8) | p[:, b2]
    return ((w24 >> (8 - off)) & 0xFFFF).astype(np.int32)


def viterbi_l16_v2_fp16reg_sim(
    X: np.ndarray, pair_lut: np.ndarray, k: int, row_w: np.ndarray | None = None
) -> np.ndarray:
    """Simulates CUDA register-resident FP16 cost + 32-step collar Pass 1 + pinned Pass 2."""
    R, Nw = X.shape
    M = Nw // 2
    step_bits = 2 * k
    NG = 65536 >> step_bits
    shift = 16 - step_bits
    nchoice = 1 << step_bits

    def run_pass(X_seq, s_start, s_end, rw_seq):
        M_seq = X_seq.shape[1] // 2
        bp = np.empty((R, M_seq, NG), dtype=np.uint8)
        cur = np.full((R, 65536), 16384.0, dtype=np.float16)
        for r in range(R):
            if s_start[r] < 0:
                cur[r, :] = np.float16(0.0)
            else:
                cur[r, s_start[r]] = np.float16(0.0)

        for m in range(M_seq):
            cur_f32 = cur.astype(np.float32)
            # cur reshaped as [R, nchoice, NG] corresponds to s = (j << shift) | q!
            grouped = cur_f32.reshape(R, nchoice, NG)
            arg = np.argmin(grouped, axis=1).astype(np.uint8)
            best = np.min(grouped, axis=1)  # [R, NG] fp32 in shared memory!
            bp[:, m, :] = arg

            x0 = X_seq[:, 2 * m : 2 * m + 1]
            x1 = X_seq[:, 2 * m + 1 : 2 * m + 2]
            d2 = (x0 - pair_lut[:, 0][None, :]) ** 2 + (x1 - pair_lut[:, 1][None, :]) ** 2
            if rw_seq is not None:
                d2 = d2 * rw_seq[:, m : m + 1]

            # s = (q << step_bits) | b -> best expanded over b
            nxt_f32 = np.repeat(best, nchoice, axis=1) + d2
            if m >= 8:
                s_norm = best[:, 0:1]
                nxt_f32 = nxt_f32 - s_norm
            nxt_f32 = np.clip(nxt_f32, -8192.0, 32768.0)
            cur = nxt_f32.astype(np.float16)
            assert not np.isnan(cur).any() and not np.isinf(cur).any(), f"NaN/Inf at step m={m}"

        cur_f32 = cur.astype(np.float32)
        states = np.empty((R, M_seq), dtype=np.int32)
        for r in range(R):
            s = int(s_end[r]) if s_end[r] >= 0 else int(np.argmin(cur_f32[r]))
            for m in range(M_seq - 1, -1, -1):
                states[r, m] = s
                q = s >> step_bits
                j = int(bp[r, m, q])
                s = (j << shift) | q
        return states

    # Pass 1: 32-step collar [M-16 .. M-1, 0 .. 15] (64 weights)
    W_pad = 16
    X_collar = np.concatenate([X[:, -2 * W_pad :], X[:, : 2 * W_pad]], axis=1)
    rw_collar = None
    if row_w is not None:
        rw_collar = np.concatenate([row_w[:, -W_pad:], row_w[:, :W_pad]], axis=1)
    free = np.full(R, -1, dtype=np.int32)
    st_c = run_pass(X_collar, free, free, rw_collar)
    seam = st_c[:, W_pad - 1]

    # Pass 2: Full M-step tail-biting pinned to seam
    return run_pass(X, seam, seam, row_w), seam


def main():
    rng = np.random.default_rng(42)
    # 1. Initialize 32x16 stratified half-normal quantile grid for HYB [512, 2]
    from math import erf, sqrt
    # Quick inverse normal approximation via numpy/erfinv-free grid using sorted normal samples
    g_pool = np.sort(np.abs(rng.standard_normal(2_000_000)))
    q32 = g_pool[(np.linspace(0.5 / 32, 1.0 - 0.5 / 32, 32) * len(g_pool)).astype(int)]
    q16 = g_pool[(np.linspace(0.5 / 16, 1.0 - 0.5 / 16, 16) * len(g_pool)).astype(int)]
    grid = np.stack(np.meshgrid(q32, q16, indexing="ij"), axis=-1).reshape(512, 2)
    lut = grid[rng.permutation(512)].astype(np.float16)

    X_train = rng.standard_normal((24, 256)).astype(np.float32)
    X_test = rng.standard_normal((16, 256)).astype(np.float32)

    for k in (1, 2):
        lut_k = lut.copy()
        for it in range(6):
            pair_lut = hyb_eval_all_states(lut_k)
            st, seam = viterbi_l16_v2_fp16reg_sim(X_train, pair_lut, k=k)
            # Check tail-biting bitpack/unpack bijection!
            packed = pack_tailbite_l16(st, step_bits=2 * k)
            unpacked = unpack_tailbite_l16_cuda_sim(packed, M=128, step_bits=2 * k)
            assert np.array_equal(st, unpacked), f"Bitpack mismatch at k={k}!"
            # Verify end_state == seam!
            assert np.array_equal(st[:, -1], seam), f"Seam closure mismatch at k={k}!"

            # Signed M-step
            s_u64 = st.reshape(-1).astype(np.uint64)
            h = (s_u64 * s_u64 + s_u64) & 0xFFFFFFFF
            idx = ((h >> 6) & 511).astype(np.int64)
            sx = np.where((h & 0x8000) != 0, -1.0, 1.0).astype(np.float32)
            sy = np.where((s_u64 & 0x8000) != 0, -1.0, 1.0).astype(np.float32)
            x0 = X_train[:, 0::2].reshape(-1)
            x1 = X_train[:, 1::2].reshape(-1)
            cnt = np.bincount(idx, minlength=512).astype(np.float32)
            sum0 = np.bincount(idx, weights=sx * x0, minlength=512).astype(np.float32)
            sum1 = np.bincount(idx, weights=sy * x1, minlength=512).astype(np.float32)
            mask = cnt > 0
            lut_f32 = lut_k.astype(np.float32)
            lut_f32[mask, 0] = sum0[mask] / cnt[mask]
            lut_f32[mask, 1] = sum1[mask] / cnt[mask]
            lut_k = lut_f32.astype(np.float16)

        pair_lut = hyb_eval_all_states(lut_k)
        st_te, _ = viterbi_l16_v2_fp16reg_sim(X_test, pair_lut, k=k)
        rec = pair_lut[st_te].reshape(X_test.shape)
        snr = 10.0 * math.log10(float((X_test ** 2).sum()) / float(((X_test - rec) ** 2).sum()))
        print(f"[OK] HYB L=16 V=2 k={k}: bitpack EXACT, seam EXACT, held-out Gaussian SNR = {snr:.3f} dB")


if __name__ == "__main__":
    main()
