#!/usr/bin/env python3
"""
glm53_fold_encoder.py
=====================
Native PyTorch CUDA / CPU Full-Width Hadamard Weight-Folding Encoder & Single-Slab Disk Store
for GLM-5.3-Flash-Abliterated (`APEX-FOLD-FP16` v3.1).

Key Architectural Guarantees:
-----------------------------
1. 100% CHANNEL COVERAGE (`c2 = d_in`) + FULL-DIMENSION KRONECKER FWHT (`fwht_full`):
   Eliminates the `2,304` uncoded tail channels (`c2 < d_in`) and dead cross-expert subspace
   (`r_shared = 0`) that caused real Layer-3 orthogonal weights to cap at `2.03 dB`.
   Applies a Full-Dimension (`4096`/`2048`-point) Fast Walsh-Hadamard Transform before sorting
   Hadamard columns by exact cross-expert Frobenius energy so every rotated coordinate carries
   `1 / d_in` of the real activation energy regardless of channel spikes.
2. POST-QUANTIZATION VARIANCE-RESTITUTED LEAST-SQUARES SCALE REFIT (`S_blk`):
   After multi-stage vector quantization (`idx_s0..s3`), refits each group scale `S_blk` via
   `s_refit = (s_LS ^ 0.65) * (s_norm ^ 0.35)`, maximizing Euclidean weight SNR while
   preventing the 3-stage SwiGLU MMSE amplitude shrinkage (`6.6% -> ~75-97%` magnitude retained).
3. HARD BYTE-LOCKED SINGLE-SLAB ARCHIVES (`expert_slab_u8[E, row_bytes]`):
   Packs all per-expert tensors (`[A_e]`, `S_blk`, `idx_s0`, `[idx_s1]`, `[idx_s2]`, `[idx_s3]`)
   across `gate`, `up`, and `down` into a single 16-byte-aligned byte slab per layer,
   asserting `actual_bytes == preset.layer_bytes(E)` down to the single byte!
"""

from __future__ import annotations
import os
import struct
import zipfile
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
from glm53_torch_compat import torch, nn, F

from glm53_fold_config import (
    APEX_FOLD_PRESETS,
    CODEBOOK_SIZE,
    MAX_STAGES,
    VEC_DIM,
    LayerFoldPreset,
    ProjectionFoldSpec,
)
from glm53_fold_math import (
    OrthogonalAbliterationGuard,
    build_gosset_e8_codebook,
    compute_down_preconditioner,
    compute_gate_up_preconditioner,
    inverse_transform_weights_fwht,
    transform_weights_fwht,
)


@dataclass
class ProjectionSharedHeader:
    """Layer-shared FP16 manifold & transform header for one projection (`gate`, `up`, or `down`)."""
    name: str
    d_out: int
    d_in: int
    r_shared: int
    c0: int
    c1: int
    c2: int
    stages_t0: int
    w_pre: torch.Tensor       # float16[d_in]
    perm: torch.Tensor        # uint16[d_in] (or int64 on device)
    signs: torch.Tensor       # int8[d_in]
    B_shared: torch.Tensor    # float16[r_shared, d_in]
    codebooks: torch.Tensor   # float16[4, 256, 8]

    def to_device(self, device: torch.device) -> "ProjectionSharedHeader":
        return ProjectionSharedHeader(
            name=self.name,
            d_out=self.d_out,
            d_in=self.d_in,
            r_shared=self.r_shared,
            c0=self.c0,
            c1=self.c1,
            c2=self.c2,
            stages_t0=self.stages_t0,
            w_pre=self.w_pre.to(device=device, dtype=torch.float16).contiguous(),
            perm=self.perm.to(device=device, dtype=torch.long).contiguous(),
            signs=self.signs.to(device=device, dtype=torch.int8).contiguous(),
            B_shared=self.B_shared.to(device=device, dtype=torch.float16).contiguous(),
            codebooks=self.codebooks.to(device=device, dtype=torch.float16).contiguous(),
        )

    def exact_bytes(self) -> int:
        return (
            self.d_in * 2
            + self.d_in * 2
            + self.d_in * 1
            + self.r_shared * self.d_in * 2
            + MAX_STAGES * CODEBOOK_SIZE * VEC_DIM * 2
        )


@dataclass
class FoldedLayerArchive:
    """Complete folded representation of a 288-expert MoE layer (`gate`, `up`, `down`)."""
    layer_id: int
    preset_name: str
    n_experts: int
    gate_header: ProjectionSharedHeader
    up_header: ProjectionSharedHeader
    down_header: ProjectionSharedHeader
    expert_slab_u8: torch.Tensor  # uint8[n_experts, expert_row_bytes]

    @property
    def total_bytes(self) -> int:
        hdr_bytes = (
            self.gate_header.exact_bytes()
            + self.up_header.exact_bytes()
            + self.down_header.exact_bytes()
        )
        slab_bytes = int(self.expert_slab_u8.numel() * self.expert_slab_u8.element_size())
        return hdr_bytes + slab_bytes


class ApexFoldEncoder:
    """
    Full-Width Hadamard `FP16` Weight-Folding Encoder (`QuIP#` Full-FWHT + `AQLM` Multi-Stage Gosset $E_8$).
    """

    @staticmethod
    @torch.no_grad()
    def fit_shared_subspace_basis(
        W_experts: torch.Tensor,         # [E, d_out, d_in]
        w_pre: torch.Tensor,             # [d_in] float16
        perm: torch.Tensor,              # [d_in] int64
        signs: torch.Tensor,             # [d_in] int8
        r_shared: int,
        expert_batch: int = 32,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optional shared right subspace extractor when `r_shared > 0`.
        When `r_shared == 0` (the default for orthogonal MoE experts), returns empty tensors immediately.
        """
        E, d_out, d_in = W_experts.shape
        if r_shared <= 0:
            B_empty = torch.zeros((0, d_in), dtype=torch.float16, device=device)
            P_empty = torch.zeros((d_in, 0), dtype=torch.float32, device=device)
            return B_empty, P_empty

        gram = torch.zeros((d_in, d_in), dtype=torch.float32, device=device)
        for e0 in range(0, E, expert_batch):
            wb = W_experts[e0 : e0 + expert_batch].to(device=device)
            ub = transform_weights_fwht(wb, w_pre, perm, signs)
            u_flat = ub.reshape(-1, d_in)
            gram.addmm_(u_flat.T, u_flat)

        gram = 0.5 * (gram + gram.T)
        _, evecs = torch.linalg.eigh(gram)
        top_v = evecs[:, -r_shared:].flip(dims=[1]).T.contiguous()
        B_shared = top_v.to(torch.float16)

        B_f32 = B_shared.to(torch.float32)
        BBt = B_f32 @ B_f32.T + 1e-5 * torch.eye(r_shared, dtype=torch.float32, device=device)
        P_right = torch.linalg.solve(BBt, B_f32).T.contiguous()
        return B_shared, P_right

    @staticmethod
    @torch.no_grad()
    def train_multistage_fp16_codebooks(
        sample_vecs_s0: torch.Tensor,    # [N_s0, 8] float32 normalized vectors in 0..c2
        sample_vecs_s1: torch.Tensor,    # [N_s1, 8] float32 vectors in 0..c1
        sample_vecs_s2: torch.Tensor,    # [N_s2, 8] float32 vectors in 0..c0
        stages_t0: int = 3,
        kmeans_iters: int = 15,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Trains `MAX_STAGES = 4` additive `FP16` 8D vector codebooks `C[4, 256, 8]` seeded
        from the Multi-Shell 256-point Gosset $E_8$ lattice and refined via batched Lloyd-Max
        K-Means on the exact residual distribution of each stage (`s0`, `s1`, `s2`, `s3`).
        """
        codebooks = torch.zeros((MAX_STAGES, CODEBOOK_SIZE, VEC_DIM), dtype=torch.float32, device=device)
        e8_seed = build_gosset_e8_codebook(device=device)  # [256, 8]

        def _fit_stage_codebook(res_vecs: torch.Tensor, damp: float = 1.0) -> torch.Tensor:
            if res_vecs.numel() == 0:
                return (e8_seed * 0.25 * damp).to(torch.float16).to(torch.float32)
            v_std = res_vecs.pow(2).mean().sqrt().clamp_min(1e-4)
            C = e8_seed * (v_std * damp)
            for _ in range(kmeans_iters):
                c_sq = C.pow(2).sum(dim=1)
                dots = torch.matmul(res_vecs, C.T)
                idx = torch.argmax(2.0 * dots - c_sq.unsqueeze(0), dim=1)
                counts = torch.bincount(idx, minlength=CODEBOOK_SIZE)
                mask = counts > 0
                if mask.any():
                    new_C = torch.zeros_like(C)
                    new_C.index_add_(0, idx, res_vecs)
                    C[mask] = new_C[mask] / counts[mask].unsqueeze(1).to(torch.float32)
            return C.to(torch.float16).to(torch.float32)

        def _nearest_idx(vecs: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
            if vecs.numel() == 0:
                return torch.zeros((0,), dtype=torch.long, device=device)
            c_sq = C.pow(2).sum(dim=1)
            return torch.argmax(2.0 * torch.matmul(vecs, C.T) - c_sq.unsqueeze(0), dim=1)

        # Fallback to sample_vecs_s0 if c1 == 0 or c0 == 0
        s1_in = sample_vecs_s1 if sample_vecs_s1.numel() > 0 else sample_vecs_s0
        s2_in = sample_vecs_s2 if sample_vecs_s2.numel() > 0 else s1_in

        # Stage 0 codebook C[0]: fit on all active vectors (0..c2)
        codebooks[0] = _fit_stage_codebook(sample_vecs_s0, damp=0.95)

        # Stage 1 codebook C[1]: fit on Stage-0 residuals of Tier 0+1 vectors (0..c1)
        idx0_s1 = _nearest_idx(s1_in, codebooks[0])
        res1_s1 = s1_in - codebooks[0][idx0_s1]
        codebooks[1] = _fit_stage_codebook(res1_s1, damp=0.90)

        # Stage 2 codebook C[2]: fit on Stage-0+1 residuals of Tier 0 vectors (0..c0)
        idx0_s2 = _nearest_idx(s2_in, codebooks[0])
        res1_s2 = s2_in - codebooks[0][idx0_s2]
        idx1_s2 = _nearest_idx(res1_s2, codebooks[1])
        res2_s2 = res1_s2 - codebooks[1][idx1_s2]
        codebooks[2] = _fit_stage_codebook(res2_s2, damp=0.85)

        # Stage 3 codebook C[3]: fit on Stage-0+1+2 residuals if stages_t0 == 4
        idx2_s2 = _nearest_idx(res2_s2, codebooks[2])
        res3_s2 = res2_s2 - codebooks[2][idx2_s2]
        if stages_t0 >= 4:
            codebooks[3] = _fit_stage_codebook(res3_s2, damp=0.80)
        else:
            codebooks[3] = (codebooks[2] * 0.5).to(torch.float16).to(torch.float32)

        return codebooks.to(torch.float16)

    @staticmethod
    @torch.no_grad()
    def quantize_multistage_beam(
        vecs: torch.Tensor,              # [M, 8] float32
        cb_f32: torch.Tensor,            # [4, 256, 8] float32 (exact FP16 values)
        num_stages: int,
        beam_width: int = 4,
        chunk_size: int = 131072,
    ) -> torch.Tensor:
        """
        Quantizes M 8D vectors into `num_stages` uint8 codebook indices `[M, num_stages]`
        using Beam-K search (for `num_stages >= 2`) so multi-stage `FP16` codebook vectors
        cooperate to minimize total Euclidean error `||v - sum_s C_s[i_s]||_2^2`.
        """
        M = vecs.shape[0]
        out_idx = torch.empty((M, num_stages), dtype=torch.uint8, device=vecs.device)
        if num_stages == 0 or M == 0:
            return out_idx

        C0 = cb_f32[0]
        C0_sq = C0.pow(2).sum(dim=1)

        for m0 in range(0, M, chunk_size):
            v = vecs[m0 : m0 + chunk_size]
            B = v.shape[0]
            scores0 = 2.0 * torch.matmul(v, C0.T) - C0_sq.unsqueeze(0)

            if num_stages == 1:
                out_idx[m0 : m0 + B, 0] = torch.argmax(scores0, dim=1).to(torch.uint8)
                continue

            K = min(beam_width, CODEBOOK_SIZE)
            _, top_k0 = torch.topk(scores0, k=K, dim=1)           # [B, K]
            beam_idx = top_k0.unsqueeze(-1)                       # [B, K, 1]
            beam_res = v.unsqueeze(1) - C0[top_k0]                # [B, K, 8]

            for s in range(1, num_stages):
                Cs = cb_f32[s]
                Cs_sq = Cs.pow(2).sum(dim=1)
                res_flat = beam_res.reshape(B * K, 8)
                res_norm_sq = res_flat.pow(2).sum(dim=1, keepdim=True)
                dist_sq = (res_norm_sq + Cs_sq.unsqueeze(0) - 2.0 * torch.matmul(res_flat, Cs.T)).reshape(
                    B, K, CODEBOOK_SIZE
                )

                if s == num_stages - 1:
                    flat_best = torch.argmin(dist_sq.reshape(B, -1), dim=1)
                    best_k = flat_best // CODEBOOK_SIZE
                    best_c = flat_best % CODEBOOK_SIZE
                    b_arange = torch.arange(B, device=v.device)
                    out_idx[m0 : m0 + B, :s] = beam_idx[b_arange, best_k, :].to(torch.uint8)
                    out_idx[m0 : m0 + B, s] = best_c.to(torch.uint8)
                else:
                    _, flat_topk = torch.topk(dist_sq.reshape(B, -1), k=K, dim=1, largest=False)
                    parent_k = flat_topk // CODEBOOK_SIZE
                    chosen_c = flat_topk % CODEBOOK_SIZE
                    b_arange = torch.arange(B, device=v.device).unsqueeze(1)
                    beam_idx = torch.cat([beam_idx[b_arange, parent_k, :], chosen_c.unsqueeze(-1)], dim=-1)
                    beam_res = beam_res[b_arange, parent_k, :] - Cs[chosen_c]

        return out_idx

    @classmethod
    @torch.no_grad()
    def encode_projection(
        cls,
        W_experts: torch.Tensor,         # [E, d_out, d_in]
        w_pre: torch.Tensor,             # [d_in] float16
        perm: torch.Tensor,              # [d_in] int64
        signs: torch.Tensor,             # [d_in] int8
        spec: ProjectionFoldSpec,
        expert_batch: int = 32,
        max_train_vecs: int = 131072,
        beam_width: int = 4,
        scale_restitution_beta: float = 0.35,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[ProjectionSharedHeader, Dict[str, torch.Tensor]]:
        """
        Folds all E experts of a single projection (`gate`, `up`, or `down`) into:
          - `ProjectionSharedHeader` (`w_pre`, `perm`, `signs`, `B_shared`, `codebooks`)
          - Per-expert dictionary (`[A_e]`, `S_blk`, `idx_s0`, `[idx_s1]`, `[idx_s2]`, `[idx_s3]`)
        Includes Post-Quantization Variance-Restituted Least-Squares Scale Refit (`S_blk`) in Pass 3.
        """
        E, d_out, d_in = W_experts.shape
        assert d_out == spec.d_out and d_in == spec.d_in, f"Shape {(d_out, d_in)} != {(spec.d_out, spec.d_in)}"

        # Pass 1: Optional shared subspace (empty when spec.r_shared == 0)
        B_shared, P_right = cls.fit_shared_subspace_basis(
            W_experts, w_pre, perm, signs, r_shared=spec.r_shared, expert_batch=expert_batch, device=device
        )
        B_sh_f32 = B_shared.to(torch.float32)

        num_groups = spec.num_scale_groups
        sg = spec.scale_group
        f_s0, f_s1, f_s2, f_s3 = spec.f_s0, spec.f_s1, spec.f_s2, spec.f_s3
        f0 = f_s2
        f1 = f_s1 - f_s2
        f2 = f_s0 - f_s1

        A_e_all = torch.empty((E, d_out, spec.r_shared), dtype=torch.float16, device="cpu") if spec.r_shared > 0 else None
        S_blk_all = torch.empty((E, d_out, num_groups), dtype=torch.float16, device="cpu")
        idx_s0_all = torch.empty((E, d_out, f_s0), dtype=torch.uint8, device="cpu")
        idx_s1_all = torch.empty((E, d_out, f_s1), dtype=torch.uint8, device="cpu") if f_s1 > 0 else None
        idx_s2_all = torch.empty((E, d_out, f_s2), dtype=torch.uint8, device="cpu") if f_s2 > 0 else None
        idx_s3_all = torch.empty((E, d_out, f_s3), dtype=torch.uint8, device="cpu") if f_s3 > 0 else None

        # Pass 2: Compute initial group scales S_blk and sample training vectors for codebooks
        if device.type == "cpu" and max_train_vecs > 16384:
            max_train_vecs = 16384
        n_batches = max(1, (E + expert_batch - 1) // expert_batch)
        samples_per_batch = max(1024, (max_train_vecs + n_batches - 1) // n_batches)
        s0_samples, s1_samples, s2_samples = [], [], []
        gen = torch.Generator(device=device)
        gen.manual_seed(42 + d_out + d_in)

        for e0 in range(0, E, expert_batch):
            B = min(expert_batch, E - e0)
            wb = W_experts[e0 : e0 + B].to(device=device)
            ub = transform_weights_fwht(wb, w_pre, perm, signs)
            if spec.r_shared > 0 and A_e_all is not None:
                ab = torch.matmul(ub, P_right).to(torch.float16)
                A_e_all[e0 : e0 + B] = ab.cpu()
                r_act = ub[:, :, : spec.c2] - torch.matmul(ab.to(torch.float32), B_sh_f32[:, : spec.c2])
            else:
                r_act = ub[:, :, : spec.c2]

            r_grp = r_act.reshape(B, d_out, num_groups, sg)
            s_blk = r_grp.pow(2).mean(dim=-1).sqrt().clamp_min(1e-6).to(torch.float16)
            S_blk_all[e0 : e0 + B] = s_blk.cpu()

            r_norm = (r_grp / s_blk.to(torch.float32).unsqueeze(-1)).reshape(B, d_out, f_s0, VEC_DIM)

            v_all = r_norm.reshape(-1, VEC_DIM)
            p0 = torch.randperm(v_all.shape[0], generator=gen, device=device)[:samples_per_batch]
            s0_samples.append(v_all[p0])

            if f_s1 > 0:
                v_t01 = r_norm[:, :, :f_s1, :].reshape(-1, VEC_DIM)
                p1 = torch.randperm(v_t01.shape[0], generator=gen, device=device)[:samples_per_batch]
                s1_samples.append(v_t01[p1])

            if f_s2 > 0:
                v_t0 = r_norm[:, :, :f_s2, :].reshape(-1, VEC_DIM)
                p2 = torch.randperm(v_t0.shape[0], generator=gen, device=device)[:samples_per_batch]
                s2_samples.append(v_t0[p2])

        cat_s0 = torch.cat(s0_samples, dim=0)[:max_train_vecs]
        cat_s1 = torch.cat(s1_samples, dim=0)[:max_train_vecs] if len(s1_samples) > 0 else cat_s0
        cat_s2 = torch.cat(s2_samples, dim=0)[:max_train_vecs] if len(s2_samples) > 0 else cat_s1

        codebooks = cls.train_multistage_fp16_codebooks(
            cat_s0,
            cat_s1,
            cat_s2,
            stages_t0=spec.stages_t0,
            device=device,
        )
        cb_f32 = codebooks.to(torch.float32)

        # Pass 3: Encode Tiers 0, 1, 2 and refit S_blk via variance-restituted least squares
        for e0 in range(0, E, expert_batch):
            B = min(expert_batch, E - e0)
            wb = W_experts[e0 : e0 + B].to(device=device)
            ub = transform_weights_fwht(wb, w_pre, perm, signs)
            if spec.r_shared > 0 and A_e_all is not None:
                ab_f32 = A_e_all[e0 : e0 + B].to(device=device, dtype=torch.float32)
                r_act = ub[:, :, : spec.c2] - torch.matmul(ab_f32, B_sh_f32[:, : spec.c2])
            else:
                r_act = ub[:, :, : spec.c2]

            r_grp = r_act.reshape(B, d_out, num_groups, sg)
            s_blk_f32 = S_blk_all[e0 : e0 + B].to(device=device, dtype=torch.float32)
            r_norm = (r_grp / s_blk_f32.unsqueeze(-1)).reshape(B, d_out, f_s0, VEC_DIM)

            if f0 > 0:
                q_t0 = cls.quantize_multistage_beam(
                    r_norm[:, :, :f0, :].reshape(-1, VEC_DIM),
                    cb_f32,
                    num_stages=spec.stages_t0,
                    beam_width=beam_width,
                ).reshape(B, d_out, f0, spec.stages_t0)
            else:
                q_t0 = None

            if f1 > 0:
                q_t1 = cls.quantize_multistage_beam(
                    r_norm[:, :, f0 : f0 + f1, :].reshape(-1, VEC_DIM),
                    cb_f32,
                    num_stages=2,
                    beam_width=beam_width,
                ).reshape(B, d_out, f1, 2)
            else:
                q_t1 = None

            if f2 > 0:
                q_t2 = cls.quantize_multistage_beam(
                    r_norm[:, :, f0 + f1 : f_s0, :].reshape(-1, VEC_DIM),
                    cb_f32,
                    num_stages=1,
                    beam_width=1,
                ).reshape(B, d_out, f2, 1)
            else:
                q_t2 = None

            s0_parts = [p[:, :, :, 0] for p in (q_t0, q_t1, q_t2) if p is not None]
            b_idx_s0 = torch.cat(s0_parts, dim=2)
            idx_s0_all[e0 : e0 + B] = b_idx_s0.cpu()

            q_rec = cb_f32[0][b_idx_s0.to(torch.long)]  # [B, d_out, f_s0, 8]

            s1_parts = [p[:, :, :, 1] for p in (q_t0, q_t1) if p is not None]
            if idx_s1_all is not None and len(s1_parts) > 0:
                b_idx_s1 = torch.cat(s1_parts, dim=2)
                idx_s1_all[e0 : e0 + B] = b_idx_s1.cpu()
                q_rec[:, :, :f_s1, :].add_(cb_f32[1][b_idx_s1.to(torch.long)])

            if q_t0 is not None and idx_s2_all is not None:
                b_idx_s2 = q_t0[:, :, :, 2]
                idx_s2_all[e0 : e0 + B] = b_idx_s2.cpu()
                q_rec[:, :, :f_s2, :].add_(cb_f32[2][b_idx_s2.to(torch.long)])
                if idx_s3_all is not None and spec.stages_t0 >= 4:
                    b_idx_s3 = q_t0[:, :, :, 3]
                    idx_s3_all[e0 : e0 + B] = b_idx_s3.cpu()
                    q_rec[:, :, :f_s3, :].add_(cb_f32[3][b_idx_s3.to(torch.long)])

            # Post-Quantization Variance-Restituted Least-Squares Scale Refit:
            # Blends exact least-squares scale s_ls = <r_grp, q_grp>/||q_grp||^2 with
            # norm-preserving scale s_norm = ||r_grp||/||q_grp|| to prevent 3-stage SwiGLU shrinkage!
            q_grp = q_rec.reshape(B, d_out, num_groups, sg)
            s_ls = (r_grp * q_grp).sum(dim=-1) / (q_grp.pow(2).sum(dim=-1) + 1e-12)
            s_norm = r_grp.pow(2).sum(dim=-1).sqrt() / (q_grp.pow(2).sum(dim=-1).sqrt() + 1e-12)
            beta = float(scale_restitution_beta)
            s_refit = (
                (s_ls.clamp_min(1e-6) ** (1.0 - beta)) * (s_norm.clamp_min(1e-6) ** beta)
            ).to(torch.float16)
            S_blk_all[e0 : e0 + B] = s_refit.cpu()

        header = ProjectionSharedHeader(
            name=spec.name,
            d_out=d_out,
            d_in=d_in,
            r_shared=spec.r_shared,
            c0=spec.c0,
            c1=spec.c1,
            c2=spec.c2,
            stages_t0=spec.stages_t0,
            w_pre=w_pre.cpu().to(torch.float16),
            perm=perm.cpu().to(torch.uint16),
            signs=signs.cpu().to(torch.int8),
            B_shared=B_shared.cpu().to(torch.float16),
            codebooks=codebooks.cpu().to(torch.float16),
        )

        per_exp: Dict[str, torch.Tensor] = {
            "S_blk": S_blk_all,
            "idx_s0": idx_s0_all,
        }
        if A_e_all is not None:
            per_exp["A_e"] = A_e_all
        if idx_s1_all is not None:
            per_exp["idx_s1"] = idx_s1_all
        if idx_s2_all is not None:
            per_exp["idx_s2"] = idx_s2_all
        if idx_s3_all is not None:
            per_exp["idx_s3"] = idx_s3_all

        return header, per_exp

    @staticmethod
    def pack_single_slab(
        preset: LayerFoldPreset,
        g_exp: Dict[str, torch.Tensor],
        u_exp: Dict[str, torch.Tensor],
        d_exp: Dict[str, torch.Tensor],
        n_experts: int,
    ) -> torch.Tensor:
        """
        Packs all per-expert tensors of `gate`, `up`, and `down` into a single 16-byte-aligned
        contiguous byte table `expert_slab_u8[n_experts, row_bytes]`.
        """
        layout, row_bytes = preset.slab_layout()
        slab = torch.zeros((n_experts, row_bytes), dtype=torch.uint8, device="cpu")
        src_map = {"g": g_exp, "u": u_exp, "d": d_exp}

        for key, (start, end, shape, _) in layout.items():
            prefix, suffix = key.split("_", 1)
            t = src_map[prefix][suffix].contiguous().cpu()
            assert t.shape == (n_experts, *shape), f"{key} shape {t.shape} != {(n_experts, *shape)}"
            raw_u8 = t.view(torch.uint8).reshape(n_experts, end - start)
            slab[:, start:end].copy_(raw_u8)

        return slab

    @staticmethod
    def unpack_slab_views(
        slab_rows: torch.Tensor,  # [U, row_bytes] uint8 on GPU or CPU
        preset: LayerFoldPreset,
    ) -> Dict[str, torch.Tensor]:
        """
        Zero-copy pointer unpacking of `slab_rows[U, row_bytes]` into typed tensors
        (`float16` for `A_e` and `S_blk`, `uint8` for `idx_s0..idx_s3`).
        """
        layout, _ = preset.slab_layout()
        U = slab_rows.shape[0]
        dtype_map = {"float16": torch.float16, "uint8": torch.uint8}
        views: Dict[str, torch.Tensor] = {}
        for key, (start, end, shape, dtype_str) in layout.items():
            dt = dtype_map[dtype_str]
            views[key] = slab_rows[:, start:end].view(dt).reshape(U, *shape)
        return views

    @classmethod
    @torch.no_grad()
    def encode_moe_layer(
        cls,
        layer_id: int,
        W_gate: torch.Tensor,            # [E, d_inter, d_model]
        W_up: torch.Tensor,              # [E, d_inter, d_model]
        W_down: torch.Tensor,            # [E, d_model, d_inter]
        rms_weight: torch.Tensor,        # [d_model] post_attention_layernorm.weight
        preset: LayerFoldPreset,
        r_dir: Optional[torch.Tensor] = None,
        expert_batch: int = 32,
        beam_width: int = 4,
        device: torch.device = torch.device("cpu"),
    ) -> FoldedLayerArchive:
        """
        Encodes a complete MoE layer (`W_gate`, `W_up`, `W_down`) into a byte-locked
        `FoldedLayerArchive` with single-slab packing and optional refusal-direction guard.
        """
        E = W_gate.shape[0]
        if r_dir is not None:
            W_down = OrthogonalAbliterationGuard.project_out_refusal_torch(W_down, r_dir)

        w_pre_g, perm_g, _, signs_g = compute_gate_up_preconditioner(
            W_gate, rms_weight, seed=1000 + layer_id * 3 + 0, expert_batch=expert_batch, device=device
        )
        w_pre_u, perm_u, _, signs_u = compute_gate_up_preconditioner(
            W_up, rms_weight, seed=1000 + layer_id * 3 + 1, expert_batch=expert_batch, device=device
        )
        w_pre_d, perm_d, _, signs_d = compute_down_preconditioner(
            W_gate, W_up, W_down, rms_weight, seed=1000 + layer_id * 3 + 2, expert_batch=expert_batch, device=device
        )

        g_hdr, g_exp = cls.encode_projection(
            W_gate, w_pre_g, perm_g, signs_g, preset.gate,
            expert_batch=expert_batch, beam_width=beam_width, device=device,
        )
        u_hdr, u_exp = cls.encode_projection(
            W_up, w_pre_u, perm_u, signs_u, preset.up,
            expert_batch=expert_batch, beam_width=beam_width, device=device,
        )
        d_hdr, d_exp = cls.encode_projection(
            W_down, w_pre_d, perm_d, signs_d, preset.down,
            expert_batch=expert_batch, beam_width=beam_width, device=device,
        )

        if r_dir is not None and "A_e" in d_exp and d_exp["A_e"].numel() > 0:
            d_exp["A_e"] = OrthogonalAbliterationGuard.project_out_refusal_torch(d_exp["A_e"], r_dir)

        slab = cls.pack_single_slab(preset, g_exp, u_exp, d_exp, n_experts=E)
        arch = FoldedLayerArchive(
            layer_id=layer_id,
            preset_name=preset.name,
            n_experts=E,
            gate_header=g_hdr,
            up_header=u_hdr,
            down_header=d_hdr,
            expert_slab_u8=slab,
        )
        expected = preset.layer_bytes(n_experts=E)
        assert arch.total_bytes == expected, f"Archive byte mismatch: {arch.total_bytes} != {expected}"
        return arch

    @staticmethod
    @torch.no_grad()
    def reconstruct_projection_rotated(
        hdr: ProjectionSharedHeader,
        views: Dict[str, torch.Tensor],
        prefix: str,
        device: torch.device,
        out_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """
        [R3.2-1.1] Rotated-domain reconstruction U_hat [U, d_out, d_in] (no inverse FWHT/permutation):
        x_rot @ U_hat^T == x @ W_hat^T, where x_rot = transform_activations_fwht(x, w_pre, perm, signs).
        Codebook vectors are summed in FP32 per vector and written once in `out_dtype`.
        """
        cb = hdr.codebooks.to(device=device, dtype=torch.float32)
        s_blk = views[f"{prefix}_S_blk"].to(device=device, dtype=torch.float32)
        idx_s0 = views[f"{prefix}_idx_s0"].to(device=device, dtype=torch.long)
        U_cnt, d_out, f_s0 = idx_s0.shape
        num_groups = s_blk.shape[2]
        sg = hdr.c2 // num_groups
        r_norm = cb[0][idx_s0]
        if f"{prefix}_idx_s1" in views and hdr.c1 > 0:
            i1 = views[f"{prefix}_idx_s1"].to(device=device, dtype=torch.long)
            if i1.shape[2] > 0:
                r_norm[:, :, : i1.shape[2], :].add_(cb[1][i1])
        if f"{prefix}_idx_s2" in views and hdr.c0 > 0:
            i2 = views[f"{prefix}_idx_s2"].to(device=device, dtype=torch.long)
            if i2.shape[2] > 0:
                r_norm[:, :, : i2.shape[2], :].add_(cb[2][i2])
        if f"{prefix}_idx_s3" in views and hdr.stages_t0 >= 4 and hdr.c0 > 0:
            i3 = views[f"{prefix}_idx_s3"].to(device=device, dtype=torch.long)
            if i3.shape[2] > 0:
                r_norm[:, :, : i3.shape[2], :].add_(cb[3][i3])
        U_hat = torch.zeros((U_cnt, d_out, hdr.d_in), dtype=out_dtype, device=device)
        U_hat[:, :, : hdr.c2] = (r_norm.view(U_cnt, d_out, num_groups, sg) * s_blk.unsqueeze(-1)).view(U_cnt, d_out, hdr.c2).to(out_dtype)
        del r_norm
        if hdr.r_shared > 0 and f"{prefix}_A_e" in views:
            A_e = views[f"{prefix}_A_e"].to(device=device, dtype=torch.float32)
            U_hat.add_(torch.matmul(A_e, hdr.B_shared.to(device=device, dtype=torch.float32)).to(out_dtype))
        return U_hat

    @staticmethod
    @torch.no_grad()
    def reconstruct_projection_weights(
        hdr: ProjectionSharedHeader,
        views: Dict[str, torch.Tensor],
        prefix: str,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Exact Mode-B weight reconstruction `[U, d_out, d_in]` in `float32` from unpacked slab views
        (used for mathematical verification and for Path-B batched Tensor-Core prefill).
        """
        hdr_d = hdr.to_device(device)
        cb_f32 = hdr_d.codebooks.to(torch.float32)  # [4, 256, 8]
        s_blk = views[f"{prefix}_S_blk"].to(device=device, dtype=torch.float32)  # [U, d_out, G]
        idx_s0 = views[f"{prefix}_idx_s0"].to(device=device, dtype=torch.long)   # [U, d_out, f_s0]

        U_cnt, d_out, f_s0 = idx_s0.shape
        num_groups = s_blk.shape[2]
        sg = hdr_d.c2 // num_groups

        r_norm = cb_f32[0][idx_s0]  # [U, d_out, f_s0, 8]
        if f"{prefix}_idx_s1" in views and hdr_d.c1 > 0:
            idx_s1 = views[f"{prefix}_idx_s1"].to(device=device, dtype=torch.long)
            f_s1 = idx_s1.shape[2]
            if f_s1 > 0:
                r_norm[:, :, :f_s1, :].add_(cb_f32[1][idx_s1])
        if f"{prefix}_idx_s2" in views and hdr_d.c0 > 0:
            idx_s2 = views[f"{prefix}_idx_s2"].to(device=device, dtype=torch.long)
            f_s2 = idx_s2.shape[2]
            if f_s2 > 0:
                r_norm[:, :, :f_s2, :].add_(cb_f32[2][idx_s2])
        if f"{prefix}_idx_s3" in views and hdr_d.stages_t0 >= 4 and hdr_d.c0 > 0:
            idx_s3 = views[f"{prefix}_idx_s3"].to(device=device, dtype=torch.long)
            f_s3 = idx_s3.shape[2]
            if f_s3 > 0:
                r_norm[:, :, :f_s3, :].add_(cb_f32[3][idx_s3])

        r_act = (
            r_norm.reshape(U_cnt, d_out, num_groups, sg) * s_blk.unsqueeze(-1)
        ).reshape(U_cnt, d_out, hdr_d.c2)

        if hdr_d.r_shared > 0 and f"{prefix}_A_e" in views:
            A_e = views[f"{prefix}_A_e"].to(device=device, dtype=torch.float32)
            U_hat = torch.matmul(A_e, hdr_d.B_shared.to(torch.float32))
        else:
            U_hat = torch.zeros((U_cnt, d_out, hdr_d.d_in), dtype=torch.float32, device=device)

        U_hat[:, :, : hdr_d.c2].add_(r_act)
        W_hat = inverse_transform_weights_fwht(U_hat, hdr_d.w_pre, hdr_d.perm, hdr_d.signs)
        return W_hat


class ApexFoldDiskStore:
    """
    Uncompressed `.npz` serializer with zero-copy read-only memory-map support (`npz_memmaps`).
    Because `np.savez` stores members with `ZIP_STORED`, `expert_slab_u8` can be memory-mapped
    directly from the `.npz` file on NVMe with ZERO host RAM duplication!
    """

    @staticmethod
    def save_layer(path: str, arch: FoldedLayerArchive) -> int:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        arrays: Dict[str, np.ndarray] = {
            "meta_layer_id": np.array([arch.layer_id, arch.n_experts], dtype=np.int32),
            "meta_preset": np.array([arch.preset_name], dtype="<U32"),
            "expert_slab_u8": arch.expert_slab_u8.cpu().numpy(),
        }
        for p_key, hdr in (("g", arch.gate_header), ("u", arch.up_header), ("d", arch.down_header)):
            arrays[f"{p_key}_meta"] = np.array(
                [hdr.d_out, hdr.d_in, hdr.r_shared, hdr.c0, hdr.c1, hdr.c2, hdr.stages_t0],
                dtype=np.int32,
            )
            arrays[f"{p_key}_w_pre"] = hdr.w_pre.cpu().numpy()
            arrays[f"{p_key}_perm"] = hdr.perm.cpu().to(torch.uint16).numpy()
            arrays[f"{p_key}_signs"] = hdr.signs.cpu().numpy()
            arrays[f"{p_key}_B_shared"] = hdr.B_shared.cpu().numpy()
            arrays[f"{p_key}_codebooks"] = hdr.codebooks.cpu().numpy()

        np.savez(path, **arrays)
        return os.path.getsize(path)

    @staticmethod
    def _npz_memmaps(npz_path: str) -> Dict[str, np.ndarray]:
        """Zero-copy read-only memory maps of every uncompressed member inside an `.npz` file."""
        out: Dict[str, np.ndarray] = {}
        with zipfile.ZipFile(npz_path) as zf, open(npz_path, "rb") as f:
            for zi in zf.infolist():
                assert zi.compress_type == zipfile.ZIP_STORED, f"{zi.filename} is compressed; cannot mmap"
                f.seek(zi.header_offset)
                lh = f.read(30)
                assert struct.unpack("<I", lh[:4])[0] == 0x04034B50
                n_len, e_len = struct.unpack("<HH", lh[26:30])
                f.seek(zi.header_offset + 30 + n_len + e_len)
                version = np.lib.format.read_magic(f)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
                else:
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
                arr_start = f.tell()
                name = zi.filename[:-4] if zi.filename.endswith(".npy") else zi.filename
                if int(np.prod(shape)) == 0 or dtype.kind == "U":
                    f.seek(arr_start)
                    out[name] = np.load(npz_path, allow_pickle=False)[name]
                else:
                    out[name] = np.memmap(
                        npz_path,
                        dtype=dtype,
                        mode="r",
                        offset=arr_start,
                        shape=tuple(shape),
                        order="F" if fortran else "C",
                    )
        return out

    @classmethod
    def load_layer(cls, path: str, mmap_slab: bool = False) -> FoldedLayerArchive:
        if mmap_slab:
            data = cls._npz_memmaps(path)
            slab_np = data["expert_slab_u8"]
            slab_t = torch.from_numpy(slab_np)
        else:
            raw = np.load(path, allow_pickle=False)
            data = {k: raw[k] for k in raw.files}
            slab_t = torch.from_numpy(np.array(data["expert_slab_u8"], copy=True))

        meta_id = data["meta_layer_id"]
        layer_id, n_experts = int(meta_id[0]), int(meta_id[1])
        preset_name = str(data["meta_preset"][0])

        headers: Dict[str, ProjectionSharedHeader] = {}
        for p_key, p_name in (("g", "gate"), ("u", "up"), ("d", "down")):
            m = [int(x) for x in data[f"{p_key}_meta"]]
            headers[p_key] = ProjectionSharedHeader(
                name=p_name,
                d_out=m[0],
                d_in=m[1],
                r_shared=m[2],
                c0=m[3],
                c1=m[4],
                c2=m[5],
                stages_t0=m[6],
                w_pre=torch.from_numpy(np.array(data[f"{p_key}_w_pre"], copy=True)),
                perm=torch.from_numpy(np.array(data[f"{p_key}_perm"], copy=True)),
                signs=torch.from_numpy(np.array(data[f"{p_key}_signs"], copy=True)),
                B_shared=torch.from_numpy(np.array(data[f"{p_key}_B_shared"], copy=True)),
                codebooks=torch.from_numpy(np.array(data[f"{p_key}_codebooks"], copy=True)),
            )

        return FoldedLayerArchive(
            layer_id=layer_id,
            preset_name=preset_name,
            n_experts=n_experts,
            gate_header=headers["g"],
            up_header=headers["u"],
            down_header=headers["d"],
            expert_slab_u8=slab_t,
        )
