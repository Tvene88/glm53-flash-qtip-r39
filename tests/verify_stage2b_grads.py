#!/usr/bin/env python3
"""Finite-difference verification of exact analytical gradients for QtipHybRoutedMLPFunction:
  1) grad_x (propagating upstream through W_down -> SwiGLU clamp -> W_gate, W_up)
  2) grad_top_w (into ST-Top8 AnchoredRouterLoRA + e_score_correction_bias)
  3) grad_scale (in-place FP16 row scales)
  4) grad_hyb_lut (layer-shared 2 KiB [512, 2] HYB codebooks).
"""
import math
import numpy as np


def fwht_np(x: np.ndarray) -> np.ndarray:
    n = x.shape[-1]
    h = np.array([[1.0]], dtype=np.float64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    h /= math.sqrt(n)
    return x @ h


def silu_np(x: np.ndarray) -> np.ndarray:
    sig = 1.0 / (1.0 + np.exp(-x))
    return x * sig


def silu_grad_np(x: np.ndarray) -> np.ndarray:
    sig = 1.0 / (1.0 + np.exp(-x))
    return sig * (1.0 + x * (1.0 - sig))


def decode_proj_np(rmap, scale, st_tiles, lut, signs, W_sparse, d_out, d_in):
    # st_tiles: [d_out//4, d_in//4, 8] (using 4x4 mini-tiles = 16 weights = 8 pairs for fast finite-diff test)
    s = np.arange(65536, dtype=np.uint64)
    h = (34038481 * s + 76625530) & 0xFFFFFFFF
    idx = ((h >> 17) & 511).astype(np.int64)
    sx = np.where((h & 0x00008000) != 0, -1.0, 1.0)
    sy = np.where((h & 0x80000000) != 0, -1.0, 1.0)
    pair_tab = np.stack([lut[idx, 0] * sx, lut[idx, 1] * sy], axis=-1)  # [65536, 2]

    U_unit = pair_tab[st_tiles].reshape(d_out // 4, d_in // 4, 4, 4).transpose(0, 2, 1, 3).reshape(d_out, d_in)
    Us = U_unit * scale[:, None]
    U = np.zeros_like(Us)
    U[rmap] = Us
    W_dense = fwht_np(U) * signs[None, :]
    return W_dense + W_sparse, U_unit


def main():
    rng = np.random.default_rng(123)
    N, D, Di = 6, 16, 16
    lim = 1.5  # tight limit so clamp boundaries are actively tested too (away from exact kink)

    x = rng.standard_normal((N, D)) * 0.6
    # Ensure no sample sits within 1e-4 of +/-lim kink for clean central finite difference
    top_w = rng.uniform(0.2, 0.8, size=(N,))
    dy = rng.standard_normal((N, D))

    def make_proj(d_out, d_in):
        rmap = rng.permutation(d_out)
        scale = rng.uniform(0.3, 1.2, size=(d_out,))
        st_tiles = rng.integers(0, 65536, size=(d_out // 4, d_in // 4, 8), dtype=np.int32)
        lut = rng.uniform(0.1, 1.5, size=(512, 2))
        signs = rng.choice([-1.0, 1.0], size=(d_in,))
        W_sp = np.zeros((d_out, d_in))
        W_sp[0, 1] = 0.45
        return rmap, scale, st_tiles, lut, signs, W_sp

    pg = list(make_proj(Di, D))
    pu = list(make_proj(Di, D))
    pd = list(make_proj(D, Di))

    def forward_loss(x_in, w_in, pg_in, pu_in, pd_in):
        Wg, _ = decode_proj_np(*pg_in, Di, D)
        Wu, _ = decode_proj_np(*pu_in, Di, D)
        Wd, _ = decode_proj_np(*pd_in, D, Di)
        g_pre = x_in @ Wg.T
        u_pre = x_in @ Wu.T
        g_c = np.minimum(g_pre, lim)
        u_c = np.clip(u_pre, -lim, lim)
        z = silu_np(g_c) * u_c
        d_out_vec = z @ Wd.T
        y = d_out_vec * w_in[:, None]
        return float(np.sum(y * dy))

    # Analytical forward + backward
    Wg, Ug_unit = decode_proj_np(*pg, Di, D)
    Wu, Uu_unit = decode_proj_np(*pu, Di, D)
    Wd, Ud_unit = decode_proj_np(*pd, D, Di)

    g_pre = x @ Wg.T
    u_pre = x @ Wu.T
    assert np.min(np.abs(g_pre - lim)) > 1e-3 and np.min(np.abs(np.abs(u_pre) - lim)) > 1e-3
    g_c = np.minimum(g_pre, lim)
    u_c = np.clip(u_pre, -lim, lim)
    silu_g = silu_np(g_c)
    z = silu_g * u_c
    d_vec = z @ Wd.T

    # 1) grad_top_w
    grad_w = np.sum(dy * d_vec, axis=-1)
    grad_d_vec = dy * top_w[:, None]

    # 2) Backprop through down_proj
    grad_Wd = grad_d_vec.T @ z  # [D, Di]
    grad_z = grad_d_vec @ Wd    # [N, Di]

    # 3) Backprop through clamped SwiGLU: z = silu(min(g_pre, lim)) * clip(u_pre, -lim, lim)
    grad_g_c = grad_z * u_c * silu_grad_np(g_c)
    grad_g_pre = np.where(g_pre <= lim, grad_g_c, 0.0)
    grad_u_pre = np.where((u_pre >= -lim) & (u_pre <= lim), grad_z * silu_g, 0.0)

    grad_Wg = grad_g_pre.T @ x
    grad_Wu = grad_u_pre.T @ x
    grad_x = grad_g_pre @ Wg + grad_u_pre @ Wu

    # 4) Projection helper for grad_scale and grad_lut from grad_W
    s_all = np.arange(65536, dtype=np.uint64)
    h_all = (34038481 * s_all + 76625530) & 0xFFFFFFFF
    idx_all = ((h_all >> 17) & 511).astype(np.int64)
    sx_all = np.where((h_all & 0x00008000) != 0, -1.0, 1.0)
    sy_all = np.where((h_all & 0x80000000) != 0, -1.0, 1.0)

    def proj_param_grads(grad_W, U_unit, proj, d_out, d_in):
        rmap, scale, st_tiles, lut, signs, _ = proj
        # W_dense = fwht(U) * signs -> dL/dU = fwht(grad_W * signs) because fwht is symmetric orthonormal!
        grad_U = fwht_np(grad_W * signs[None, :])
        grad_Us = grad_U[rmap]  # [d_out, d_in]
        grad_scale = np.sum(grad_Us * U_unit, axis=-1)

        grad_U_unit = grad_Us * scale[:, None]
        tiles = grad_U_unit.reshape(d_out // 4, 4, d_in // 4, 4).transpose(0, 2, 1, 3).reshape(-1, 8, 2)
        st_f = st_tiles.reshape(-1)
        g_flat = tiles.reshape(-1, 2)
        b_idx = idx_all[st_f]
        grad_lut = np.zeros_like(lut)
        np.add.at(grad_lut[:, 0], b_idx, g_flat[:, 0] * sx_all[st_f])
        np.add.at(grad_lut[:, 1], b_idx, g_flat[:, 1] * sy_all[st_f])
        return grad_scale, grad_lut

    gs_g, glut_g = proj_param_grads(grad_Wg, Ug_unit, pg, Di, D)
    gs_d, glut_d = proj_param_grads(grad_Wd, Ud_unit, pd, D, Di)

    # Central finite difference checks
    eps = 1e-6
    # Check grad_x
    fd_x = np.zeros_like(x)
    for i in range(N):
        for j in range(D):
            xp, xm = x.copy(), x.copy()
            xp[i, j] += eps; xm[i, j] -= eps
            fd_x[i, j] = (forward_loss(xp, top_w, pg, pu, pd) - forward_loss(xm, top_w, pg, pu, pd)) / (2 * eps)
    assert np.max(np.abs(grad_x - fd_x)) < 1e-8, f"grad_x err: {np.max(np.abs(grad_x - fd_x))}"

    # Check grad_w
    fd_w = np.zeros_like(top_w)
    for i in range(N):
        wp, wm = top_w.copy(), top_w.copy()
        wp[i] += eps; wm[i] -= eps
        fd_w[i] = (forward_loss(x, wp, pg, pu, pd) - forward_loss(x, wm, pg, pu, pd)) / (2 * eps)
    assert np.max(np.abs(grad_w - fd_w)) < 1e-8, f"grad_w err: {np.max(np.abs(grad_w - fd_w))}"

    # Check grad_scale (gate & down)
    for label, p_list, gs_ana in (("gate", pg, gs_g), ("down", pd, gs_d)):
        fd_s = np.zeros_like(gs_ana)
        for r in range( len(gs_ana) ):
            sp, sm = p_list[1].copy(), p_list[1].copy()
            sp[r] += eps; sm[r] -= eps
            p_p = list(p_list); p_p[1] = sp
            p_m = list(p_list); p_m[1] = sm
            if label == "gate":
                fd_s[r] = (forward_loss(x, top_w, p_p, pu, pd) - forward_loss(x, top_w, p_m, pu, pd)) / (2 * eps)
            else:
                fd_s[r] = (forward_loss(x, top_w, pg, pu, p_p) - forward_loss(x, top_w, pg, pu, p_m)) / (2 * eps)
        assert np.max(np.abs(gs_ana - fd_s)) < 1e-8, f"grad_scale {label} err: {np.max(np.abs(gs_ana - fd_s))}"

    # Check grad_lut on active buckets
    active_b = np.unique(idx_all[pg[2].reshape(-1)])[:10]
    for b in active_b:
        for c in (0, 1):
            lp, lm = pg[3].copy(), pg[3].copy()
            lp[b, c] += eps; lm[b, c] -= eps
            p_p = list(pg); p_p[3] = lp
            p_m = list(pg); p_m[3] = lm
            fd_val = (forward_loss(x, top_w, p_p, pu, pd) - forward_loss(x, top_w, p_m, pu, pd)) / (2 * eps)
            assert abs(glut_g[b, c] - fd_val) < 1e-8, f"grad_lut mismatch: {glut_g[b,c]} vs {fd_val}"

    print("[PASS] Exact analytical gradients match central finite differences to < 1e-8 across x, router w, scale, and HYB LUT!")


if __name__ == "__main__":
    main()
