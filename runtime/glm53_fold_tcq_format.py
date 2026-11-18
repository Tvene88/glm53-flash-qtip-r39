#!/usr/bin/env python3
"""
glm53_fold_tcq_format.py  [Round 3.2, Claude]
============================================
Byte-locked single-slab archive for trellis-coded (TCQ, L=12) routed experts.

Per layer (header, tiny): per projection p in (g, u, d): Rademacher signs [d_in] (int8); shared TCQ tables k=1,2,3
[3, 4096] (fp16); per-projection gain multipliers.
Per expert (one fixed-size row of the layer slab, identical for all 288 experts):
  for p in (g, u, d):
    row_map  uint16 [d_out]   stored-row -> original output row (rows are stored grouped by bit-width k)
    scale    fp16   [d_out]   per stored row: RMS * refit (LS, or unbiased for gate/up)
    codes_k1 uint8  [n1, rb(T,1)] / codes_k2 [n2, rb(T,2)] / codes_k3 [n3, rb(T,3)]  (TCQ bit-streams)
    sparse   [K] x (uint16 row, uint16 col, fp16 value)  super-weight / outlier slots (original domain)
with n1 + n2 + n3 = d_out FIXED per projection (rows ranked per expert by their own L2 norm: data-free reverse
water-filling), so every expert row has exactly the same byte size (asserted).

Decode of a weight (row r, column t): window = bits[k t : k t + 12) of the row stream -> table_k[window] * scale[r]
in the Hadamard-rotated domain; plus the sparse slots in the original domain.
"""

from __future__ import annotations

import math
import os
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import glm53_fold_tcq as TQ
from glm53_fold_math import fwht_full, generate_rademacher_signs


@dataclass(frozen=True)
class TcqProjSpec:
    name: str
    d_out: int
    d_in: int
    n1: int
    n3: int
    k_sparse: int
    unbiased: bool = False

    @property
    def n2(self) -> int:
        return self.d_out - self.n1 - self.n3

    def code_bytes(self) -> Tuple[int, int, int]:
        T = self.d_in
        return TQ.row_bytes(T, 1), TQ.row_bytes(T, 2), TQ.row_bytes(T, 3)

    def bytes(self) -> int:
        b1, b2, b3 = self.code_bytes()
        return 2 * self.d_out + 2 * self.d_out + self.n1 * b1 + self.n2 * b2 + self.n3 * b3 + 6 * self.k_sparse


@dataclass(frozen=True)
class TcqPreset:
    name: str
    gate: TcqProjSpec
    up: TcqProjSpec
    down: TcqProjSpec
    bias: bool = False          # [R3.3] per-expert g/u/o bias fields (16 KiB per expert)

    def layout(self) -> Tuple[Dict[str, Tuple[int, int]], int]:
        """field -> (byte offset, byte length) inside one expert row; total row bytes (16-byte aligned)."""
        lay: Dict[str, Tuple[int, int]] = {}
        cur = 0
        for p, sp in (("g", self.gate), ("u", self.up), ("d", self.down)):
            b1, b2, b3 = sp.code_bytes()
            for fname, nbytes in (("map", 2 * sp.d_out), ("scale", 2 * sp.d_out), ("c1", sp.n1 * b1),
                                  ("c2", sp.n2 * b2), ("c3", sp.n3 * b3), ("sp", 6 * sp.k_sparse)):
                lay[f"{p}_{fname}"] = (cur, nbytes)
                cur += nbytes
                cur += (-cur) % 2
        if self.bias:
            for fname, n in (("g_bias", self.gate.d_out), ("u_bias", self.up.d_out), ("o_bias", self.down.d_out)):
                lay[fname] = (cur, 2 * n)
                cur += 2 * n
        cur += (-cur) % 16
        return lay, cur

    @property
    def expert_row_bytes(self) -> int:
        return self.layout()[1]

    def layer_bytes(self, n_experts: int = 288) -> int:
        hdr = 2 * 3 * 4096 + sum(sp.d_in for sp in (self.gate, self.up, self.down))
        return hdr + n_experts * self.expert_row_bytes

    def bpw(self, n_experts: int = 288) -> float:
        w = n_experts * sum(sp.d_out * sp.d_in for sp in (self.gate, self.up, self.down))
        return self.layer_bytes(n_experts) * 8.0 / w


def make_preset(rule: str = "A") -> TcqPreset:
    """A = all rows 2 bits; B = gate/up 10% lowest-norm rows at 1 bit + down 20% highest-norm rows at 3 bits.
    [R3.3] suffix "+bias" adds the per-expert bias fields (e.g. "A+bias")."""
    rule, _, ext = rule.partition("+")
    bias = "bias" in ext.split("+")
    fr = {"A": (0.0, 0.0), "B": (0.10, 0.20), "C": (0.05, 0.10)}[rule]
    n1 = int(round(fr[0] * 2048 / 8)) * 8
    n3 = int(round(fr[1] * 4096 / 8)) * 8
    g = TcqProjSpec("gate", 2048, 4096, n1=n1, n3=0, k_sparse=64, unbiased=True)
    u = TcqProjSpec("up", 2048, 4096, n1=n1, n3=0, k_sparse=64, unbiased=True)
    d = TcqProjSpec("down", 4096, 2048, n1=0, n3=n3, k_sparse=256, unbiased=False)
    return TcqPreset(f"tcq_balanced_{rule}{'_bias' if bias else ''}", g, u, d, bias)


# ------------------------------------------------------------------------------------------------------------------
# encoder
# ------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def encode_projection(W: torch.Tensor, sp: TcqProjSpec, signs: torch.Tensor, tables: Dict[int, torch.Tensor],
                      device: torch.device, expert_batch: int = 16) -> Dict[str, torch.Tensor]:
    """W [E, d_out, d_in] (any float dtype, CPU or GPU) -> per-expert fields (CPU tensors):
    map [E, d_out] uint16, scale [E, d_out] fp16, c1/c2/c3 [E, n_k, rb] uint8, sp [E, K, 6] uint8; plus SNR bookkeeping."""
    E, d_out, d_in = W.shape
    b1, b2, b3 = sp.code_bytes()
    out = {"map": torch.empty((E, d_out), dtype=torch.int16), "scale": torch.empty((E, d_out), dtype=torch.float16),
           "c1": torch.empty((E, sp.n1, b1), dtype=torch.uint8), "c2": torch.empty((E, sp.n2, b2), dtype=torch.uint8),
           "c3": torch.empty((E, sp.n3, b3), dtype=torch.uint8), "sp": torch.zeros((E, sp.k_sparse, 6), dtype=torch.uint8)}
    sig = err = 0.0
    s_f = signs.to(device=device, dtype=torch.float32)
    for e0 in range(0, E, expert_batch):
        e1 = min(E, e0 + expert_batch)
        Wb = W[e0:e1].to(device=device, dtype=torch.float32)
        B = Wb.shape[0]
        # 1) super-weight / outlier slots: K largest |w| per expert (original domain), removed from the dense part
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
        # 2) rotate (Hadamard + signs), row norms, per-expert data-free bit allocation
        U = fwht_full(dense * s_f)
        norms = U.pow(2).sum(-1)                                  # [B, d_out]
        order = torch.argsort(norms, dim=1)                       # ascending
        ks = torch.full((B, d_out), 2, dtype=torch.long, device=device)
        if sp.n1:
            ks.scatter_(1, order[:, : sp.n1], 1)
        if sp.n3:
            ks.scatter_(1, order[:, d_out - sp.n3:], 3)
        # stored order: k=1 rows, then k=2, then k=3 (ascending original index inside each group)
        key = ks * (d_out + 1) + torch.arange(d_out, device=device).view(1, -1)
        stored = torch.argsort(key, dim=1)                        # [B, d_out] original row index per stored row
        out["map"][e0:e1] = stored.to(torch.int16).cpu()
        Us = torch.gather(U, 1, stored.unsqueeze(-1).expand(B, d_out, d_in))
        rms = Us.pow(2).mean(-1).sqrt().clamp_min(1e-12)          # [B, d_out]
        Xn = (Us / rms.unsqueeze(-1)).reshape(B * d_out, d_in)
        scale = torch.empty((B, d_out), dtype=torch.float32, device=device)
        rec = torch.empty_like(Xn)
        groups = ((1, 0, sp.n1), (2, sp.n1, sp.n1 + sp.n2), (3, sp.n1 + sp.n2, d_out))
        for k, a, b in groups:
            if b <= a:
                continue
            rows_idx = (torch.arange(B, device=device).view(-1, 1) * d_out + torch.arange(a, b, device=device).view(1, -1)).reshape(-1)
            Xk = Xn.index_select(0, rows_idx).contiguous()
            st, _ = TQ.viterbi_states(Xk, tables[k], k)
            q = tables[k][st.long()]
            s_ls = (Xk * q).sum(-1) / q.pow(2).sum(-1).clamp_min(1e-12)
            r = q * s_ls.unsqueeze(-1)
            if sp.unbiased:
                beta = (Xk * r).sum(-1) / Xk.pow(2).sum(-1).clamp_min(1e-12)
                s_ls = s_ls / beta.clamp_min(1e-6)
                r = q * s_ls.unsqueeze(-1)
            rec.index_copy_(0, rows_idx, r)
            scale.view(-1).index_copy_(0, rows_idx, s_ls)
            packed = TQ.pack_states(st.view(B * (b - a), d_in), k).view(B, b - a, -1)
            out[f"c{k}"][e0:e1] = packed.cpu()
        out["scale"][e0:e1] = (scale * rms).to(torch.float16).cpu()
        Us_hat = rec.view(B, d_out, d_in) * rms.unsqueeze(-1)
        sig += float(Us.double().pow(2).sum())
        err += float((Us - Us_hat).double().pow(2).sum())
        del Wb, U, Us, Xn, rec
    out["_snr_dense_db"] = torch.tensor(10 * math.log10(sig / max(err, 1e-30)))
    return out


@torch.no_grad()
def decode_projection(fields: Dict[str, torch.Tensor], sp: TcqProjSpec, signs: torch.Tensor,
                      tables: Dict[int, torch.Tensor], device: torch.device, rotated: bool = False,
                      out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Reference decode of U experts -> W_hat [U, d_out, d_in] (original domain) or U_hat (rotated) + sparse slots."""
    m = fields["map"].to(device).long() & 0xFFFF
    sc = fields["scale"].to(device).float()
    Ue, d_out = m.shape
    d_in = sp.d_in
    Us = torch.empty((Ue, d_out, d_in), dtype=torch.float32, device=device)
    a = 0
    for k, n in ((1, sp.n1), (2, sp.n2), (3, sp.n3)):
        if n == 0:
            continue
        codes = fields[f"c{k}"].to(device).reshape(Ue * n, -1)
        idx = TQ.unpack_indices(codes, d_in, k)
        Us[:, a:a + n, :] = tables[k][idx].view(Ue, n, d_in)
        a += n
    Us.mul_(sc.unsqueeze(-1))
    U = torch.zeros_like(Us)
    U.scatter_(1, m.unsqueeze(-1).expand(Ue, d_out, d_in), Us)
    if not rotated:
        s_f = signs.to(device=device, dtype=torch.float32)
        W = fwht_full(U) * s_f
        if sp.k_sparse > 0:
            spb = fields["sp"].to(device)
            rows = spb[..., 0:2].contiguous().view(torch.int16).long().view(Ue, -1) & 0xFFFF
            cols = spb[..., 2:4].contiguous().view(torch.int16).long().view(Ue, -1) & 0xFFFF
            vals = spb[..., 4:6].contiguous().view(torch.float16).float().view(Ue, -1)
            flat = W.view(Ue, -1)
            flat.scatter_add_(1, rows * d_in + cols, vals)
        return W.to(out_dtype)
    return U.to(out_dtype)


def pack_slab(preset: TcqPreset, fg: Dict[str, torch.Tensor], fu: Dict[str, torch.Tensor],
              fd: Dict[str, torch.Tensor], biases: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
    """Per-projection encoder fields -> uint8 slab [E, expert_row_bytes] (byte-locked; asserted)."""
    lay, rb = preset.layout()
    E = fg["map"].shape[0]
    slab = torch.zeros((E, rb), dtype=torch.uint8)
    for p, f in (("g", fg), ("u", fu), ("d", fd)):
        for name in ("map", "scale", "c1", "c2", "c3", "sp"):
            off, nb = lay[f"{p}_{name}"]
            if nb == 0:
                continue
            t = f[name].contiguous()
            b = t.view(torch.uint8).reshape(E, -1) if t.dtype != torch.uint8 else t.reshape(E, -1)
            assert b.shape[1] == nb, (p, name, b.shape, nb)
            slab[:, off: off + nb] = b
    if preset.bias:                                         # [R3.3] g_bias / u_bias / o_bias, fp16
        assert biases is not None, "preset has bias fields: pass biases"
        for name in ("g_bias", "u_bias", "o_bias"):
            off, nb = lay[name]
            b = biases[name].to(torch.float16).contiguous().view(torch.uint8).reshape(E, -1)
            assert b.shape[1] == nb, (name, b.shape, nb)
            slab[:, off: off + nb] = b
    assert slab.shape[1] == preset.expert_row_bytes
    return slab


def unpack_fields(preset: TcqPreset, rows: torch.Tensor, p: str) -> Dict[str, torch.Tensor]:
    """uint8 [U, row_bytes] -> one projection's fields (views) for decode_projection."""
    lay, _ = preset.layout()
    sp = {"g": preset.gate, "u": preset.up, "d": preset.down}[p]
    U = rows.shape[0]
    b1, b2, b3 = sp.code_bytes()
    def f(name):
        off, nb = lay[f"{p}_{name}"]
        return rows[:, off: off + nb]
    return {"map": f("map").contiguous().view(torch.int16).view(U, sp.d_out),
            "scale": f("scale").contiguous().view(torch.float16).view(U, sp.d_out),
            "c1": f("c1").reshape(U, sp.n1, b1) if sp.n1 else torch.empty((U, 0, b1), dtype=torch.uint8),
            "c2": f("c2").reshape(U, sp.n2, b2) if sp.n2 else torch.empty((U, 0, b2), dtype=torch.uint8),
            "c3": f("c3").reshape(U, sp.n3, b3) if sp.n3 else torch.empty((U, 0, b3), dtype=torch.uint8),
            "sp": f("sp").reshape(U, sp.k_sparse, 6)}
