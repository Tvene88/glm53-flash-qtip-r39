#!/usr/bin/env python3
"""
glm53_fold_tcq_runtime.py  [Round 3.2, Claude]
=============================================
Archive I/O + per-layer runtime for trellis-coded (TCQ) layers. Same runner-facing interface as
glm53_fold_runtime.ApexFoldLayerRuntime (expert_slab_u8, pool, layer_id, use_kernels, gain, forward_routed_experts,
decode_core for the CUDA-graph decoder).

Archive (.npz, uncompressed, zero-copy mmap of the slab): preset_rule, layer_id, signs_g/u/d (int8), tables [3,4096]
(fp16), expert_slab_u8 [288, row_bytes]. The slab is byte-locked: row_bytes == TcqPreset.expert_row_bytes.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

import glm53_fold_kernels as KR
import glm53_fold_tcq_format as TF
from glm53_fold_math import fwht_full


def save_layer(path: str, rule: str, layer_id: int, signs: Dict[str, torch.Tensor], tables: torch.Tensor,
               slab: torch.Tensor, meta: Optional[dict] = None, perm: Optional[torch.Tensor] = None) -> int:
    tmp = path + ".tmp.npz"
    extra = {} if perm is None else {"perm": perm.cpu().numpy().astype(np.int16)}   # [R3.3] row r holds expert perm[r]
    np.savez(tmp, preset_rule=np.array(rule), layer_id=np.array(layer_id), **extra,
             signs_g=signs["g"].cpu().numpy().astype(np.int8), signs_u=signs["u"].cpu().numpy().astype(np.int8),
             signs_d=signs["d"].cpu().numpy().astype(np.int8), tables=tables.cpu().to(torch.float16).numpy(),
             meta=np.array(json.dumps(meta or {})), expert_slab_u8=slab.cpu().numpy())
    os.replace(tmp, path)
    return os.path.getsize(path)


class TcqArchive:
    def __init__(self, path: str, mmap_slab: bool = True):
        raw = np.load(path, allow_pickle=False)
        self.rule = str(raw["preset_rule"])
        self.layer_id = int(raw["layer_id"])
        self.preset = TF.make_preset(self.rule)
        self.signs = {p: torch.from_numpy(raw[f"signs_{p}"].astype(np.int8)) for p in ("g", "u", "d")}
        self.tables = torch.from_numpy(raw["tables"].astype(np.float32))
        self.meta = json.loads(str(raw["meta"]))
        # [R3.3] optional popularity order: slab row r holds expert perm[r]; row_of[e] = its row
        self.perm = torch.from_numpy(raw["perm"].astype(np.int64)) if "perm" in raw.files else None
        # [R3.4-S2] a pruned file holds only R < n_experts_total rows: row_of[e] = -1 for experts without a row (argsort(perm)
        # mapped them to WRONG rows); identical to argsort(perm) for a full permutation. `missing` = experts without a row
        # (the runner masks them out of routing; the pools refuse them).
        self.n_experts_total = int(self.meta.get("n_experts_total", 288))
        self.row_of = None
        self.missing = torch.empty(0, dtype=torch.long)
        if self.perm is not None:               # [R3.4 review] explicit raises (an assert vanishes under python -O)
            if not (self.perm.numel() <= self.n_experts_total and int(self.perm.min()) >= 0
                    and int(self.perm.max()) < self.n_experts_total):
                raise ValueError(f"{path}: perm outside [0, {self.n_experts_total})")
            if torch.unique(self.perm).numel() != self.perm.numel():
                raise ValueError(f"{path}: perm repeats an expert")
            self.row_of = torch.full((self.n_experts_total,), -1, dtype=torch.long)
            self.row_of[self.perm] = torch.arange(self.perm.numel(), dtype=torch.long)
            self.missing = torch.nonzero(self.row_of < 0).flatten()
        self.slab_file = None
        if mmap_slab:
            # zero-copy read-only memory map of the uncompressed slab inside the .npz (shared helper)
            from glm53_fold_encoder import ApexFoldDiskStore
            mm = ApexFoldDiskStore._npz_memmaps(path)["expert_slab_u8"]
            self.slab_file = (path, int(mm.offset))  # [R3.2-1.6e] for the pool's file-read gather
            self.expert_slab_u8 = torch.from_numpy(mm)
        else:
            self.expert_slab_u8 = torch.from_numpy(np.ascontiguousarray(raw["expert_slab_u8"]))
        assert self.expert_slab_u8.shape[1] == self.preset.expert_row_bytes, "byte lock violated"
        if self.perm is not None and self.perm.numel() != self.expert_slab_u8.shape[0]:   # [R3.4-S2] (review: raise)
            raise ValueError(f"{path}: perm has {self.perm.numel()} entries for {self.expert_slab_u8.shape[0]} slab rows")


class TcqLayerRuntime:
    def __init__(self, arch: TcqArchive, device: torch.device, slab_in_vram: bool = False):
        self.arch = arch
        self.preset = arch.preset
        self.device = device
        self.layer_id = arch.layer_id
        self.slab_in_vram = slab_in_vram and device.type == "cuda"
        self.expert_slab_u8 = arch.expert_slab_u8.to(device) if self.slab_in_vram else arch.expert_slab_u8
        self.signs = {p: arch.signs[p].to(device=device, dtype=torch.float32) for p in ("g", "u", "d")}
        tab3 = arch.tables.to(device)
        self.lay = {"g": KR.TcqProjLayout(self.preset, "g", self.preset.gate, tab3),
                    "u": KR.TcqProjLayout(self.preset, "u", self.preset.up, tab3),
                    "d": KR.TcqProjLayout(self.preset, "d", self.preset.down, tab3)}
        self.pool = None
        self.use_kernels = True
        self.gain = {"g": 1.0, "u": 1.0, "d": 1.0}
        self.max_pairs_per_launch = 4096
        self.dense_min_pairs = 512   # [R3.3] batches this large decode each used expert once + bf16 GEMMs
        self.dense_group = 32
        # [R3.7 Stage 7] opt-in overrides for prompt processing (unset = the three values above, unchanged).
        # R37_DENSE_MIN_PAIRS: pairs (tokens x 8) from which a batch takes the dense path; a very large value keeps
        # every batch on the pair path, i.e. on the decode kernel (R37_TCQ_KERNEL), which the dense path never got.
        # R37_MAX_PAIRS: pairs per pair-path launch (bounds its activation copies). R37_DENSE_GROUP: experts decoded
        # to dense values at once (32 = 1.5 GiB of temporaries).
        if os.environ.get('R37_DENSE_MIN_PAIRS', ''):
            self.dense_min_pairs = max(2, int(os.environ['R37_DENSE_MIN_PAIRS']))
        if os.environ.get('R37_MAX_PAIRS', ''):
            self.max_pairs_per_launch = max(8, int(os.environ['R37_MAX_PAIRS']))
        if os.environ.get('R37_DENSE_GROUP', ''):
            self.dense_group = max(1, int(os.environ['R37_DENSE_GROUP']))
        self.n_experts = int(self.expert_slab_u8.shape[0])
        lay_all, _ = self.preset.layout()
        self.bias_off = ({k: int(lay_all[k][0]) for k in ("g_bias", "u_bias", "o_bias")}
                         if getattr(self.preset, "bias", False) else None)          # [R3.3]
        self.row_of = arch.row_of.to(device) if getattr(arch, "row_of", None) is not None else None
        miss = getattr(arch, "missing", None)
        self._partial = miss is not None and miss.numel() > 0     # [R3.4-S2] pruned file: guard the non-pool paths
        # [R3.6 K1] opt-in, validated before capture. Default keeps R3.5 calls.
        self.r36_tcq = os.environ.get('R36_TCQ_KERNEL', 'v2') == 'v3'
        self.r36_fwht = os.environ.get('R36_TCQ_FWHT', '0') == '1'
        if self.r36_tcq:
            import glm53_fold_tcq_kernels_v3 as K36
            K36.prepare_layouts(self.lay, arch.tables.numpy())
        # [R3.7] opt-in routed-expert kernel candidate for rule-A archives; empty = the R3.5 / R3.6 calls above.
        # R37_TCQ_KERNEL names a variant of glm53_fold_tcq_kernels_v4 (e_* = bit-exact candidates, r_* = rounding).
        # It replaces only the GEMV; the sparse add stays the R3.5 / R3.6 kernel. N = 2 pairs are computed
        # independently (no union mode yet).
        self.r37_tcq = os.environ.get('R37_TCQ_KERNEL', '')
        self.r37_warps = int(os.environ.get('R37_TCQ_WARPS', '8'))
        self.r37_gx = int(os.environ.get('R37_TCQ_GX', '64'))
        if self.r37_tcq:
            import glm53_fold_tcq_kernels_v4 as K37
            for lay in self.lay.values():
                K37.check_layout(lay, self.r37_tcq)       # raises here, before any graph is captured
                if device.type == 'cuda':
                    K37.prepare(lay)

    def _check_rows(self, uniq: torch.Tensor) -> None:
        """[R3.4-S2] non-pool paths of a pruned file: a routed expert without a row would read a wrong / out-of-bounds row.
        Only pruned files pay for this check (one device sync)."""
        if bool((self.row_of[uniq.to(self.row_of.device)] < 0).any()):
            bad = sorted(set(uniq.cpu().tolist()) & set(self.arch.missing.tolist()))
            raise RuntimeError(f"layer {self.layer_id}: routed experts {bad} have no row in this pruned archive "
                               f"(router mask missing)")

    def _bias(self, rows: torch.Tensor, slots: torch.Tensor, name: str, n: int) -> torch.Tensor:
        o = self.bias_off[name]
        # [R3.7, review K1 M4] the masked variant costs about four extra graph nodes per call; slots can only be
        # negative when hit-first scheduling (K3) or a routing policy (Q) is active, so K1 alone keeps the one-liner.
        if getattr(self, 'r36_tcq', False) and (getattr(self, 'r36_route', False)
                                                 or os.environ.get('R36_OVERLAP', '0') == '1'):
            # Masked pairs never index -1; zero the bias for their unused output.
            b = rows[:, o:o + 2 * n].index_select(0, slots.clamp_min(0).long()).contiguous().view(torch.float16).float()
            return torch.where((slots >= 0).view(-1, 1), b, 0.)
        return rows[:, o:o + 2 * n].index_select(0, slots.long()).contiguous().view(torch.float16).float()

    # ---------------------------------------------------------------- core (graph-capturable for fixed shapes)
    def decode_core(self, x: torch.Tensor, rows: torch.Tensor, pair_slots: torch.Tensor, w_pairs: torch.Tensor,
                    n_tok: int, lim: float, row_exact: bool = False, pair_output: bool = False) -> torch.Tensor:
        """x [N, D] fp32; pair_slots int32 [N*K]; w_pairs [N*K] routing weights -> routed output [N, D].
        [R3.5-M1] row_exact=True (pair mode, n_tok > 1; used by the N=2 verify graphs): the three FWHTs (cuBLAS GEMMs
        whose kernel choice depends on the row count) and the final weighted sum over the K experts run once per token on
        that token's own rows, with the n_tok = 1 shapes; the TCQ GEMV / sparse-add kernels compute every pair
        independently anyway. Each token's output is then bit-identical to an n_tok = 1 call, while the expert kernels
        still run once for all N*K pairs. Default False = the unchanged code path."""
        P = int(pair_slots.numel())
        K = P // n_tok
        lg, lu, ld = self.lay["g"], self.lay["u"], self.lay["d"]
        rx = bool(row_exact) and n_tok > 1
        gemv, sparse = KR.tcq_gemv_v2, KR.tcq_sparse_add
        if getattr(self, 'r36_tcq', False):
            import glm53_fold_tcq_kernels_v3 as K36
            def gemv(*args):
                return K36.gemv(*args, union=n_tok == 2)
            sparse = K36.sparse_add
        if getattr(self, 'r37_tcq', ''):                 # [R3.7] candidate GEMV; same call contract as v2
            import glm53_fold_tcq_kernels_v4 as K37
            _v37, _w37, _g37 = self.r37_tcq, self.r37_warps, self.r37_gx
            def gemv(rows_, slots_, lay_, x_, per_pair_):
                return K37.gemv(rows_, slots_, lay_, x_, per_pair_, variant=_v37, warps=_w37, gx=_g37)
        if getattr(self, 'r36_fwht', False):
            import glm53_fold_tcq_kernels_v3 as K36
            xg, xu = K36.fwht(x, self.signs['g']), K36.fwht(x, self.signs['u'])
        elif rx:
            xs_g, xs_u = x * self.signs["g"], x * self.signs["u"]
            xg = torch.cat([fwht_full(xs_g[t:t + 1]) for t in range(n_tok)], 0)
            xu = torch.cat([fwht_full(xs_u[t:t + 1]) for t in range(n_tok)], 0)
        else:
            xg = fwht_full(x * self.signs["g"])
            xu = fwht_full(x * self.signs["u"])
        if n_tok == 1:
            g = gemv(rows, pair_slots, lg, xg, False)
            g = sparse(rows, pair_slots, lg, x, False, g)
            u = gemv(rows, pair_slots, lu, xu, False)
            u = sparse(rows, pair_slots, lu, x, False, u)
            if self.bias_off is not None:
                g = g + self._bias(rows, pair_slots, "g_bias", lg.d_out)
                u = u + self._bias(rows, pair_slots, "u_bias", lu.d_out)
        else:
            xp = x.repeat_interleave(K, 0)
            g = gemv(rows, pair_slots, lg, xg.repeat_interleave(K, 0), True)
            g = sparse(rows, pair_slots, lg, xp, True, g)
            u = gemv(rows, pair_slots, lu, xu.repeat_interleave(K, 0), True)
            u = sparse(rows, pair_slots, lu, xp, True, u)
            if self.bias_off is not None:
                g = g + self._bias(rows, pair_slots, "g_bias", lg.d_out)
                u = u + self._bias(rows, pair_slots, "u_bias", lu.d_out)
        if self.gain["g"] != 1.0:
            g = g * self.gain["g"]
        if self.gain["u"] != 1.0:
            u = u * self.gain["u"]
        z = F.silu(g.clamp(max=lim)) * u.clamp(min=-lim, max=lim)
        if getattr(self, 'r36_fwht', False):
            zr = K36.fwht(z, self.signs['d'])
        elif rx:                                 # [R3.5-M1] one [K, d_inter] FWHT per token (the n_tok = 1 shape)
            zs = z * self.signs["d"]
            zr = torch.cat([fwht_full(zs[t * K:(t + 1) * K]) for t in range(n_tok)], 0)
        else:
            zr = fwht_full(z * self.signs["d"])
        d = gemv(rows, pair_slots, ld, zr, True)
        d = sparse(rows, pair_slots, ld, z, True, d)
        if self.bias_off is not None:
            d = d + self._bias(rows, pair_slots, "o_bias", ld.d_out)
        if self.gain["d"] != 1.0:
            d = d * self.gain["d"]
        if pair_output:                           # [R3.6 K3] combine arrivals before the original fixed reduction
            return d
        if rx:                                   # [R3.5-M1] per-token [1, K, D] sum (the n_tok = 1 reduction shape)
            dw = d * w_pairs.view(-1, 1)
            return torch.cat([dw[t * K:(t + 1) * K].view(1, K, -1).sum(dim=1) for t in range(n_tok)], 0)
        return (d * w_pairs.view(-1, 1)).view(n_tok, K, -1).sum(dim=1)

    # ---------------------------------------------------------------- [R3.3] dense prefill path
    def dense_core(self, x: torch.Tensor, rows: torch.Tensor, slot_of_u: torch.Tensor, uniq: torch.Tensor,
                   inv: torch.Tensor, w: torch.Tensor, lim: float) -> torch.Tensor:
        """x [N, D] fp32; uniq [U] expert ids, slot_of_u [U] int32 slots in `rows`, inv [N, K] index into uniq.
        Decodes each used expert ONCE to bf16 (rotated domain) and runs per-expert bf16 GEMMs over its tokens.
        Same math as decode_core up to bf16 weight rounding (~2e-3 relative, far below the code's own error)."""
        N, K = inv.shape
        lg, lu, ld = self.lay["g"], self.lay["u"], self.lay["d"]
        xg = fwht_full(x * self.signs["g"]).to(torch.bfloat16)
        xu = fwht_full(x * self.signs["u"]).to(torch.bfloat16)
        out = torch.zeros_like(x)
        inv_f = inv.reshape(-1)
        order = torch.argsort(inv_f, stable=True)
        counts = torch.bincount(inv_f, minlength=int(uniq.numel())).tolist()
        tok_all = (order // K)
        w_all = w.reshape(-1)[order]
        starts = [0]
        for c in counts:
            starts.append(starts[-1] + c)
        U = int(uniq.numel())
        for g0 in range(0, U, self.dense_group):
            g1 = min(U, g0 + self.dense_group)
            sl = slot_of_u[g0:g1].to(torch.int32)
            Wg = KR.tcq_dense(rows, sl, lg)
            Wu = KR.tcq_dense(rows, sl, lu)
            Wd = KR.tcq_dense(rows, sl, ld)
            for j in range(g1 - g0):
                a, b = starts[g0 + j], starts[g0 + j + 1]
                if b == a:
                    continue
                t = tok_all[a:b]
                n = b - a
                sj = sl[j:j + 1].expand(n).contiguous()
                g = (xg[t] @ Wg[j].T).float()
                g = KR.tcq_sparse_add(rows, sj, lg, x[t], True, g)
                u = (xu[t] @ Wu[j].T).float()
                u = KR.tcq_sparse_add(rows, sj, lu, x[t], True, u)
                if self.bias_off is not None:
                    g = g + self._bias(rows, sj[:1], "g_bias", lg.d_out)
                    u = u + self._bias(rows, sj[:1], "u_bias", lu.d_out)
                if self.gain["g"] != 1.0:
                    g = g * self.gain["g"]
                if self.gain["u"] != 1.0:
                    u = u * self.gain["u"]
                z = F.silu(g.clamp(max=lim)) * u.clamp(min=-lim, max=lim)
                zr = fwht_full(z * self.signs["d"]).to(torch.bfloat16)
                d = (zr @ Wd[j].T).float()
                d = KR.tcq_sparse_add(rows, sj, ld, z, True, d)
                if self.bias_off is not None:
                    d = d + self._bias(rows, sj[:1], "o_bias", ld.d_out)
                if self.gain["d"] != 1.0:
                    d = d * self.gain["d"]
                out.index_add_(0, t, d * w_all[a:b].unsqueeze(-1))
            del Wg, Wu, Wd
        return out

    # ---------------------------------------------------------------- runner interface
    def forward_routed_experts(self, x_flat, top_weights, top_idx, swiglu_limit: float = 10.0, hobbit_tau: float = 1.0):
        orig = x_flat.dtype
        x = x_flat.to(self.device, dtype=torch.float32)
        w = top_weights.to(self.device, dtype=torch.float32)
        i = top_idx.to(self.device, dtype=torch.long)
        N, K = i.shape
        if getattr(self,'r36_route',False):
            if N <= 4:
                flat=i.reshape(-1).cpu().tolist()
                active=sorted({e for e in flat if e>=0})
                fetched=self.pool.ensure(self.layer_id,active)
                # gather only active slots; masked pairs never address a slab.
                lookup={e:j for j,e in enumerate(active)}
                gather=torch.tensor([lookup.get(e,0) for e in flat],device=self.device,dtype=torch.long)
                slots=fetched.index_select(0,gather)
                slots=torch.where(i.reshape(-1)>=0,slots,-1).to(torch.int32)
                return self.decode_core(x,self.pool.pool,slots,w.reshape(-1),N,swiglu_limit).to(orig)
            # Large prefill uses existing dense grouping. A zero-weight duplicate
            # of a live expert preserves the route and avoids fetching dropped IDs.
            first=i.gather(1,(i>=0).to(torch.int32).argmax(1,keepdim=True))
            i=torch.where(i>=0,i,first)
        if N == 1:
            uniq, inv = i[0], torch.arange(K, device=self.device).view(1, K)
        else:
            uniq, inv = torch.unique(i, sorted=True, return_inverse=True)
            inv = inv.view(N, K)
        if self.pool is not None:
            rows = self.pool.pool
            if getattr(self, "prefill_cold", False):   # [R3.5-INT item 10b] prefill-only experts: the pool's cold end
                slot_of_u = self.pool.ensure_cold(self.layer_id, uniq.cpu().tolist())   # (set only with pool v6)
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
            return self.dense_core(x, rows, slot_of_u, uniq, inv, w, swiglu_limit).to(orig)  # [R3.3]
        pair_slots = slot_of_u.index_select(0, inv.reshape(-1))
        w_pairs = w.reshape(-1)
        if N * K <= self.max_pairs_per_launch:
            return self.decode_core(x, rows, pair_slots, w_pairs, N, swiglu_limit).to(orig)
        out = torch.empty((N, x.shape[1]), dtype=torch.float32, device=self.device)
        step = max(1, self.max_pairs_per_launch // K)
        for t0 in range(0, N, step):
            t1 = min(N, t0 + step)
            out[t0:t1] = self.decode_core(x[t0:t1], rows, pair_slots[t0 * K: t1 * K], w_pairs[t0 * K: t1 * K],
                                          t1 - t0, swiglu_limit)
        return out.to(orig)
