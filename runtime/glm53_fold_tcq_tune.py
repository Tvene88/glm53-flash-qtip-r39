#!/usr/bin/env python3
"""
glm53_fold_tcq_tune.py  [Round 3.3, Claude]
==========================================
Recovery tuning with the codes frozen ("T-s"): per expert, multiplicative factors on the per-row scales of gate, up
and down (2048 + 2048 + 4096 numbers) are fitted so the quantized expert's OUTPUT matches the exact expert's output
on that expert's calibration tokens (routing-weighted MSE), which targets the gate*up*down error compounding directly.
Sparse outliers stay as they are (they are added outside the scaled trellis part, exactly as the runtime does).
Early stopping on a held-out 20% of the expert's calibration tokens (split by calibration sequence). The result is
written back into the fields' fp16 `scale` (stored row order), so the slab format is unchanged.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

import glm53_fold_tcq_format as TF


def _sparse_dense(fields: Dict[str, torch.Tensor], e: int, d_out: int, d_in: int, device) -> torch.Tensor:
    spb = fields["sp"][e].to(device)
    K = spb.shape[0]
    S = torch.zeros((d_out, d_in), device=device)
    if K == 0:
        return S
    rows = spb[:, 0:2].contiguous().view(torch.int16).long().view(-1) & 0xFFFF
    cols = spb[:, 2:4].contiguous().view(torch.int16).long().view(-1) & 0xFFFF
    vals = spb[:, 4:6].contiguous().view(torch.float16).float().view(-1)
    S.view(-1).index_put_((rows * d_in + cols,), vals, accumulate=True)
    return S


def _one(fields, e, sp, signs, tables, device):
    sub = {k: (v[e:e + 1] if torch.is_tensor(v) and v.dim() > 0 else v) for k, v in fields.items()}
    W = TF.decode_projection(sub, sp, signs, tables, device)[0]          # dense + sparse, original domain
    S = _sparse_dense(fields, e, sp.d_out, sp.d_in, device)
    return W - S, S                                                      # trellis part (scaled by row scale), sparse part


@torch.no_grad()
def tune_scales(preset, fg, fu, fd, signs, tables, Wg, Wu, Wd, cal, device, lim: float = 10.0, iters: int = 80,
                lr: float = 3e-3, min_tokens: int = 256, weight: str = "aff", log=None) -> Dict[str, float]:
    """cal: object with .x [N, 4096] bf16 (device), .idx [N, 8] long, .w [N, 8] float, .seq [N] (calibration sequence
    ids of the kept tokens). Modifies fg/fu/fd['scale'] in place. Returns summary stats."""
    E = Wg.shape[0]
    tot_before = tot_after = tot_sig = 0.0
    n_tuned = 0
    for e in range(E):
        tk, sl = torch.where(cal.idx == e)
        n = int(tk.numel())
        if n < min_tokens:
            continue
        x = cal.x[tk].float()
        c = cal.w[tk, sl].pow(2) if weight == "aff" else torch.ones(n, device=device)
        hold = (cal.seq[tk] % 5) == 0                                     # ~20% of sequences held out
        G = Wg[e].to(device).float(); U = Wu[e].to(device).float(); D = Wd[e].to(device).float()
        y = (F.silu((x @ G.T).clamp(max=lim)) * (x @ U.T).clamp(-lim, lim)) @ D.T
        Gq, Gs = _one(fg, e, preset.gate, signs["g"], tables, device)
        Uq, Us = _one(fu, e, preset.up, signs["u"], tables, device)
        Dq, Ds = _one(fd, e, preset.down, signs["d"], tables, device)
        gq, gs = x @ Gq.T, x @ Gs.T
        uq, us = x @ Uq.T, x @ Us.T
        la_g = torch.zeros(Gq.shape[0], device=device, requires_grad=True)
        la_u = torch.zeros(Uq.shape[0], device=device, requires_grad=True)
        la_d = torch.zeros(Dq.shape[0], device=device, requires_grad=True)
        opt = torch.optim.Adam([la_g, la_u, la_d], lr=lr)
        tr, va = ~hold, hold
        sig_tr = float((c[tr].unsqueeze(-1) * y[tr].pow(2)).sum()) + 1e-30
        sig_va = float((c[va].unsqueeze(-1) * y[va].pow(2)).sum()) + 1e-30

        def forward(m):
            g = torch.exp(la_g) * gq[m] + gs[m]
            u = torch.exp(la_u) * uq[m] + us[m]
            z = F.silu(g.clamp(max=lim)) * u.clamp(-lim, lim)
            return torch.exp(la_d) * (z @ Dq.T) + z @ Ds.T

        def loss(m, sig):
            return (c[m].unsqueeze(-1) * (forward(m) - y[m]).pow(2)).sum() / sig

        with torch.enable_grad():
            best = (float(loss(va, sig_va)) if va.any() else float(loss(tr, sig_tr)), None)
            start = best[0]
            for it in range(iters):
                opt.zero_grad()
                lo = loss(tr, sig_tr)
                lo.backward()
                opt.step()
                if (it + 1) % 10 == 0 and va.any():
                    v = float(loss(va, sig_va))
                    if v < best[0]:
                        best = (v, (la_g.detach().clone(), la_u.detach().clone(), la_d.detach().clone()))
            if not va.any():
                best = (float(loss(tr, sig_tr)), (la_g.detach().clone(), la_u.detach().clone(), la_d.detach().clone()))
        if best[1] is None:
            continue                                                       # no improvement on held-out tokens
        ag, au, ad = (torch.exp(t) for t in best[1])
        for f, a in ((fg, ag), (fu, au), (fd, ad)):
            m = f["map"][e].long() & 0xFFFF                                # stored row j -> original row m[j]
            f["scale"][e] = (f["scale"][e].float() * a.cpu()[m]).to(torch.float16)
        n_tuned += 1
        tot_before += start * sig_va
        tot_after += best[0] * sig_va
        tot_sig += sig_va
    out = {"experts_tuned": n_tuned,
           "heldout_rel_err_before": round(tot_before / max(tot_sig, 1e-30), 5),
           "heldout_rel_err_after": round(tot_after / max(tot_sig, 1e-30), 5)}
    if log:
        log(f"    T-s: tuned {n_tuned} experts; held-out relative error {out['heldout_rel_err_before']:.4f} -> "
            f"{out['heldout_rel_err_after']:.4f}")
    return out
