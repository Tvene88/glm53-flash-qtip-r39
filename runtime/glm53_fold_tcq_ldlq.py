#!/usr/bin/env python3
"""
glm53_fold_tcq_ldlq.py  [Round 3.3, Claude]
==========================================
Calibration-aware (Hessian / LDLQ) encoding for the row-trellis TCQ format of glm53_fold_tcq_format. The output
fields are byte-compatible with encode_projection (same slab layout, decoder, kernels and runtime).

Proxy loss per projection: tr((W - W_hat) H (W - W_hat)^T), H = sum_t c_t x_t x_t^T over calibration inputs x_t of
that expert (c_t = routing weight^2 or 1). Everything runs in the rotated domain: U = F D W^T-rows (D = random signs,
F = orthonormal Walsh-Hadamard), H_rot = F D H D F, so W x = U (F D x).

LDLQ with trellis blocks (forward order): H_rot = (I + U_l) Dblk (I + U_l)^T with (I + U_l) block-unit-UPPER
triangular (block size g) and Dblk block-diagonal. Columns are coded in blocks b = 0..nb-1:
    v_b      = U_b + sum_{c<b} E_c (U_l)_{c,b}             (E = U - U_hat on finished blocks)
    U_hat_b  = segment Viterbi of v_b / row_scale, START STATE = last state of the row's previous segment
so each row stays ONE continuous bitshift-trellis sequence (12 + k(T-1) bits, unchanged format). With that feedback
the loss equals sum_b tr(eta_b Dblk_b eta_b^T) where eta_b is block b's own rounding error (QuIP / QuIP# / QTIP).
After coding, each row's scale is refit under the H metric; gate/up may take the unbiased (signal-gain-1) scale.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple

import torch

import glm53_fold_tcq as TQ
from glm53_fold_math import fwht_full
from glm53_fold_tcq_format import TcqProjSpec

_SEG_SRC = r"""
extern "C" __global__ void tcq_viterbi_seg_kernel(
    const float* __restrict__ X, const float* __restrict__ cbk, unsigned char* __restrict__ bp,
    const int* __restrict__ start, int* __restrict__ states_out, float* __restrict__ cost_out, int T, int k)
{
    // one block per row segment; start[r] < 0: free start (first segment); else the trellis continues from that
    // state (the state at the step BEFORE this segment), so every step of the segment is a transition.
    extern __shared__ float sm[];
    float* cb = sm;
    float* c0 = sm + 4096;
    float* c1 = sm + 8192;
    const int r = blockIdx.x;
    const float* x = X + (size_t)r * T;
    const int NG = 4096 >> k;
    const int shift = 12 - k;
    const int nchoice = 1 << k;
    const size_t bp_row = (size_t)r * T * NG;
    const int s_in = start[r];
    for (int s = threadIdx.x; s < 4096; s += blockDim.x) cb[s] = cbk[s];
    __syncthreads();
    int t_begin;
    if (s_in < 0) {
        float x0 = x[0];
        for (int s = threadIdx.x; s < 4096; s += blockDim.x) { float d = x0 - cb[s]; c0[s] = d * d; }
        t_begin = 1;
    } else {
        for (int s = threadIdx.x; s < 4096; s += blockDim.x) c0[s] = (s == s_in) ? 0.f : 3.0e38f;
        t_begin = 0;
    }
    __syncthreads();
    float* cur = c0;
    float* nxt = c1;
    for (int t = t_begin; t < T; ++t) {
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
_SEG = [None]


def _seg_kernel():
    if _SEG[0] is None:
        kern = torch.cuda._compile_kernel(_SEG_SRC, "tcq_viterbi_seg_kernel")
        kern.set_shared_memory_config(3 * 4096 * 4)
        _SEG[0] = kern
    return _SEG[0]


def viterbi_segments(X: torch.Tensor, cb: torch.Tensor, k: int, start: torch.Tensor,
                     rows_per_launch: int = 1024) -> Tuple[torch.Tensor, torch.Tensor]:
    """X fp32 [R, g] normalised targets; start int32 [R] (-1 = free) -> states int32 [R, g], cost fp32 [R]."""
    R, g = X.shape
    NG = 4096 >> k
    states = torch.empty((R, g), dtype=torch.int32, device=X.device)
    cost = torch.empty((R,), dtype=torch.float32, device=X.device)
    kern = _seg_kernel()
    for r0 in range(0, R, rows_per_launch):
        r1 = min(R, r0 + rows_per_launch)
        bp = torch.empty((r1 - r0, g, NG), dtype=torch.uint8, device=X.device)
        kern(grid=(r1 - r0, 1, 1), block=(256, 1, 1), shared_mem=3 * 4096 * 4,
             args=[X[r0:r1].contiguous(), cb.float().contiguous(), bp, start[r0:r1].to(torch.int32).contiguous(),
                   states[r0:r1], cost[r0:r1], int(g), int(k)])
        del bp
    return states, cost


def rotate_hessian(H: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """H [B, n, n] (original input domain) -> F D H D F (same rotation as the weights' input dim)."""
    s = signs.to(H.dtype)
    Hs = H * s.view(1, -1, 1) * s.view(1, 1, -1)
    Hs = fwht_full(Hs)                      # along the last dim: H D F^T
    return fwht_full(Hs.transpose(1, 2)).transpose(1, 2)


def block_ldl_forward(H: torch.Tensor, g: int) -> torch.Tensor:
    """H [B, n, n] SPD -> Uf [B, n, n] block-unit-upper-triangular with H = Uf Dblk Uf^T (Dblk block-diagonal).
    Via reversal: J H J = L L^T (Cholesky), Lu = L blockdiag(L_bb)^-1 is block-unit-lower, Uf = J Lu J."""
    B, n, _ = H.shape
    nb = n // g
    J = torch.arange(n - 1, -1, -1, device=H.device)
    Hr = H[:, J][:, :, J].double()
    L = torch.linalg.cholesky(Hr)
    Lb = L.view(B, nb, g, nb, g)
    diag = torch.stack([Lb[:, b, :, b, :] for b in range(nb)], 1)          # [B, nb, g, g] lower
    dinv = torch.linalg.solve_triangular(diag, torch.eye(g, device=H.device, dtype=L.dtype).expand(B, nb, g, g),
                                         upper=False)
    Lu = torch.empty_like(L).view(B, nb, g, nb, g)
    for c in range(nb):                       # column block c: L[:, :, c] @ dinv[c]
        Lu[:, :, :, c, :] = torch.einsum("bpij,bjk->bpik", Lb[:, :, :, c, :], dinv[:, c])
    Lu = Lu.view(B, n, n)
    return Lu[:, J][:, :, J].float()


@torch.no_grad()
def encode_projection_ldlq(W: torch.Tensor, sp: TcqProjSpec, signs: torch.Tensor, tables: Dict[int, torch.Tensor],
                           device: torch.device, hessian: Callable[[int, int], Optional[torch.Tensor]],
                           g: int = 256, damp: float = 0.01, alloc: str = "norm", sparse_mode: str = "abs",
                           scale_mode: str = "hls", expert_batch: int = 4,
                           log: Optional[Callable[[str], None]] = None) -> Dict[str, torch.Tensor]:
    """W [E, d_out, d_in] -> fields (same as glm53_fold_tcq_format.encode_projection) with Hessian-aware coding.
    hessian(e0, e1) -> H [e1-e0, d_in, d_in] fp32 on device in the ORIGINAL input domain (or None = identity).
    alloc: 'norm' (row L2 norm, data-free) | 'hnorm' (row output energy u H u^T).
    sparse_mode: 'abs' (largest |w|) | 'hw' (largest w^2 * H_jj).
    scale_mode: 'hls' (per-row least squares under H) | 'hunb' (H-metric unbiased: gain 1 along the signal)."""
    E, d_out, d_in = W.shape
    assert d_in % g == 0
    nb = d_in // g
    b1, b2, b3 = sp.code_bytes()
    out = {"map": torch.empty((E, d_out), dtype=torch.int16), "scale": torch.empty((E, d_out), dtype=torch.float16),
           "c1": torch.empty((E, sp.n1, b1), dtype=torch.uint8), "c2": torch.empty((E, sp.n2, b2), dtype=torch.uint8),
           "c3": torch.empty((E, sp.n3, b3), dtype=torch.uint8), "sp": torch.zeros((E, sp.k_sparse, 6), dtype=torch.uint8)}
    s_f = signs.to(device=device, dtype=torch.float32)
    sig = err = hsig = herr = 0.0
    for e0 in range(0, E, expert_batch):
        e1 = min(E, e0 + expert_batch)
        B = e1 - e0
        Wb = W[e0:e1].to(device=device, dtype=torch.float32)
        H = hessian(e0, e1)
        if H is None:
            H = torch.eye(d_in, device=device).expand(B, d_in, d_in).contiguous()
        H = H.float()
        # 1) sparse outliers (original domain)
        if sp.k_sparse > 0:
            flat = Wb.reshape(B, -1)
            if sparse_mode == "hw":
                hd = torch.diagonal(H, dim1=1, dim2=2)                         # [B, d_in]
                imp = (Wb.pow(2) * hd.unsqueeze(1)).reshape(B, -1)
            else:
                imp = flat.abs()
            top = torch.topk(imp, sp.k_sparse, dim=1).indices
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
        # 2) rotate weights and Hessian; damping (rotation-invariant)
        U = fwht_full(dense * s_f)
        Hr = rotate_hessian(H, s_f)
        lam = damp * torch.diagonal(Hr, dim1=1, dim2=2).mean(-1).clamp_min(1e-12)
        Hr = Hr + lam.view(B, 1, 1) * torch.eye(d_in, device=device).view(1, d_in, d_in)
        # 3) bit allocation per row
        if alloc == "hnorm":
            score = torch.einsum("bri,bij,brj->br", U, Hr, U)
        else:
            score = U.pow(2).sum(-1)
        order = torch.argsort(score, dim=1)
        ks = torch.full((B, d_out), 2, dtype=torch.long, device=device)
        if sp.n1:
            ks.scatter_(1, order[:, : sp.n1], 1)
        if sp.n3:
            ks.scatter_(1, order[:, d_out - sp.n3:], 3)
        key = ks * (d_out + 1) + torch.arange(d_out, device=device).view(1, -1)
        stored = torch.argsort(key, dim=1)
        out["map"][e0:e1] = stored.to(torch.int16).cpu()
        Us = torch.gather(U, 1, stored.unsqueeze(-1).expand(B, d_out, d_in))
        rms = Us.pow(2).mean(-1).sqrt().clamp_min(1e-12)                     # [B, d_out]
        # 4) LDLQ over column blocks with the trellis as the block quantizer
        Uf = block_ldl_forward(Hr, g)                                           # [B, n, n]
        Uh = torch.zeros_like(Us)                                               # reconstruction (unit table * rms)
        Eacc = torch.zeros_like(Us)                                             # E = Us - Uh on finished blocks
        states_full = torch.empty((B, d_out, d_in), dtype=torch.int32, device=device)
        groups = [(k, a, b) for k, a, b in ((1, 0, sp.n1), (2, sp.n1, sp.n1 + sp.n2), (3, sp.n1 + sp.n2, d_out)) if b > a]
        last_state = torch.full((B, d_out), -1, dtype=torch.int32, device=device)
        for blk in range(nb):
            c0, c1 = blk * g, (blk + 1) * g
            v = Us[:, :, c0:c1].clone()
            if blk > 0:
                v += torch.bmm(Eacc[:, :, :c0], Uf[:, :c0, c0:c1])
            xn = v / rms.unsqueeze(-1)
            for k, a, b in groups:
                Xk = xn[:, a:b, :].reshape(-1, g).contiguous()
                st, _ = viterbi_segments(Xk, tables[k], k, last_state[:, a:b].reshape(-1))
                st = st.view(B, b - a, g)
                states_full[:, a:b, c0:c1] = st
                last_state[:, a:b] = st[:, :, -1]
                Uh[:, a:b, c0:c1] = tables[k][st.long()] * rms[:, a:b].unsqueeze(-1)
            Eacc[:, :, c0:c1] = Us[:, :, c0:c1] - Uh[:, :, c0:c1]
        # 5) per-row scale under the H metric (Hr in stored-row-independent input space)
        HU = torch.bmm(Uh, Hr)                                                  # [B, d_out, n]
        num = (Us * HU).sum(-1)
        den = (Uh * HU).sum(-1).clamp_min(1e-30)
        s = num / den                                                           # LS factor on top of rms
        if scale_mode == "hunb" and sp.unbiased:
            HUs = torch.bmm(Us, Hr)
            beta = (s * (Uh * HUs).sum(-1)) / (Us * HUs).sum(-1).clamp_min(1e-30)
            s = s / beta.clamp_min(1e-6)
        Uh = Uh * s.unsqueeze(-1)
        out["scale"][e0:e1] = (rms * s).to(torch.float16).cpu()
        for k, a, b in groups:
            packed = TQ.pack_states(states_full[:, a:b, :].reshape(B * (b - a), d_in), k).view(B, b - a, -1)
            out[f"c{k}"][e0:e1] = packed.cpu()
        D = Us - Uh
        sig += float(Us.double().pow(2).sum())
        err += float(D.double().pow(2).sum())
        hsig += float(torch.einsum("bri,bij,brj->", Us, Hr, Us).double())
        herr += float(torch.einsum("bri,bij,brj->", D, Hr, D).double())
        if log is not None and e0 == 0:
            log(f"    {sp.name}: experts {e0}-{e1 - 1} weight SNR {10 * math.log10(sig / max(err, 1e-30)):.2f} dB, "
                f"H-weighted SNR {10 * math.log10(hsig / max(herr, 1e-30)):.2f} dB")
        del Wb, H, U, Hr, Us, Uh, Eacc, Uf, states_full, HU
    out["_snr_dense_db"] = torch.tensor(10 * math.log10(sig / max(err, 1e-30)))
    out["_snr_h_db"] = torch.tensor(10 * math.log10(hsig / max(herr, 1e-30)))
    return out
