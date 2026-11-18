#!/usr/bin/env python3
"""
glm53_fold_runtime.py
=====================
High-Speed Zero-Allocation GPU/CPU Runtime Engine for Round-3.1 `APEX-FOLD-FP16`:

Includes `PATCH R3-03` (`swiglu_limit = 10.0`) plus the Zero-Transpose Contiguous
`dim=0` / `dim=1` LUT-GEMV Gather Engine:
---------------------------------------------------------------------------------
1. ZERO-TRANSPOSE CONTIGUOUS PRE-TABULATED LUT-GEMV (`Path A`, `N <= 4`):
   In Round-3.0, `_forward_proj_pretabulated` called `idx_s0.index_select(0, flat_inv).transpose(1, 2).to(torch.long)`,
   which created a non-contiguous strided `[1, 512, 16384]` tensor before `torch.gather` and
   another strided transpose after `torch.gather`.
   In Round-3.1:
   - `_pretabulate_codebook_luts` produces `T` directly in shape `[4, M_tok, 256, f_s0]`
     (`einsum("mfd,skd->smkf", x_act, cb)`).
   - For `N == 1` (`per_expert_input=False`), `torch.gather(T[s, 0], dim=0, index=...)`
     operates directly on the `U` unique experts `views[f"{prefix}_idx_s{s}"]` (`[U * d_out, f_s]`,
     100% contiguous in memory, zero `.index_select(0, flat_inv)` copy, zero `.transpose(1, 2)`!),
     and sums adjacent `f_per_g` elements in the same cache line!
   - For `N > 1` or `per_expert_input=True`, `torch.gather(T[s], dim=1, index=...)` gathers
     along `dim=1` (`256`) directly into `[NK, d_out, f_s0]` with zero pre- or post-transpose!

2. SINGLE-SLAB COALESCED PINNED DMA (`PinnedSlabDMAEngine`):
   Coalesces all active experts of a host-mapped layer into a SINGLE PCIe transfer per layer.

3. DUAL-PATH TOKEN-GROUPED EXECUTION (`N <= 4` LUT-GEMV vs `N > 4` Tensor-Core GEMM).
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
from glm53_torch_compat import torch, nn, F

from glm53_fold_config import (
    APEX_FOLD_PRESETS,
    GLM53FlashModelConfig,
    VEC_DIM,
    LayerFoldPreset,
)
from glm53_fold_encoder import (
    ApexFoldEncoder,
    FoldedLayerArchive,
    ProjectionSharedHeader,
)
from glm53_fold_math import transform_activations_fwht
import glm53_fold_kernels as KR  # [R3.2]


def hobbit_filter_experts(
    top_weights: torch.Tensor,  # [N, K]
    top_idx: torch.Tensor,      # [N, K]
    tau: float = 1.0,
    routed_scaling_factor: float = 2.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    HOBBIT dynamic cumulative-score expert pruning: retains the minimal prefix of
    sorted top-k experts per token whose cumulative score reaches `tau` of the row sum,
    zeroing negligible tail experts while preserving exact `routed_scaling_factor` sum.
    """
    if tau >= 0.9999:
        return top_weights, top_idx
    sorted_w, sort_order = torch.sort(top_weights, dim=-1, descending=True)
    sorted_idx = torch.gather(top_idx, dim=-1, index=sort_order)
    row_sums = sorted_w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    csum = torch.cumsum(sorted_w, dim=-1) / row_sums
    keep_mask = torch.ones_like(sorted_w, dtype=torch.bool)
    keep_mask[:, 1:] = csum[:, :-1] < tau
    filtered_w = torch.where(keep_mask, sorted_w, torch.zeros_like(sorted_w))
    new_sums = filtered_w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    filtered_w = (filtered_w / new_sums) * routed_scaling_factor
    return filtered_w, sorted_idx


class PinnedSlabDMAEngine:
    """
    Pre-allocated pinned CPU staging buffer + GPU work buffer that coalesces all
    active experts of a host-mapped layer into a SINGLE PCIe transfer per layer.
    """

    def __init__(self, max_unique_experts: int, row_bytes: int, device: torch.device):
        self.max_u = max_unique_experts
        self.row_bytes = row_bytes
        self.device = device
        self.is_cuda = device.type == "cuda"
        self.cpu_staging = torch.empty(
            (max_unique_experts, row_bytes),
            dtype=torch.uint8,
            pin_memory=self.is_cuda,
        )
        self.gpu_buffer = torch.empty(
            (max_unique_experts, row_bytes),
            dtype=torch.uint8,
            device=device,
        )

    def fetch_unique_rows(self, slab_u8: torch.Tensor, uniq_e: torch.Tensor) -> torch.Tensor:
        if slab_u8.device == self.device:
            return slab_u8.index_select(0, uniq_e)

        U = int(uniq_e.numel())
        uniq_cpu = uniq_e.to("cpu")
        if U <= self.max_u:
            staging = self.cpu_staging[:U]
            torch.index_select(slab_u8, 0, uniq_cpu, out=staging)
            gpu_out = self.gpu_buffer[:U]
            gpu_out.copy_(staging, non_blocking=self.is_cuda)
            return gpu_out

        rows_cpu = slab_u8.index_select(0, uniq_cpu)
        if self.is_cuda:
            rows_cpu = rows_cpu.pin_memory()
        return rows_cpu.to(self.device, non_blocking=self.is_cuda)


class ApexFoldLayerRuntime(nn.Module):
    """
    High-Speed Round-3.1 MoE Layer Runtime for `APEX-FOLD-FP16`.
    Supports both VRAM-resident slabs and Host-RAM / zero-copy `.npz` mmap slabs.
    """

    def __init__(
        self,
        archive: FoldedLayerArchive,
        preset: LayerFoldPreset,
        device: torch.device = torch.device("cuda"),
        slab_in_vram: bool = True,
        dma_engine: Optional[PinnedSlabDMAEngine] = None,
        prefill_expert_chunk: int = 8,  # [R3.2-1.1] was 32 (FP32, ~8.7 GiB of temporaries)
    ):
        super().__init__()
        self.layer_id = archive.layer_id
        self.preset = preset
        self.n_experts = archive.n_experts
        self.device = device
        self.slab_in_vram = slab_in_vram and (device.type == "cuda")
        self.use_kernels = device.type == "cuda"  # [R3.2-1.4] NVRTC LUT-GEMV decode path
        self._kr_layouts = None
        self.pool = None          # [R3.2-1.6] set by the runner when the global expert cache is enabled
        self.gain = {"g": 1.0, "u": 1.0, "d": 1.0}  # [R3.2-1Q.1] per-projection output multipliers
        self.max_kernel_pairs = 4096  # [R3.2-1.1b] prefill through the pool kernels up to N*K pairs
        self.layer_id = int(getattr(archive, "layer_id", -1))
        self.prefill_expert_chunk = prefill_expert_chunk

        self.gate_hdr = archive.gate_header.to_device(device)
        self.up_hdr = archive.up_header.to_device(device)
        self.down_hdr = archive.down_header.to_device(device)

        if self.slab_in_vram:
            self.expert_slab_u8 = archive.expert_slab_u8.to(device=device, dtype=torch.uint8).contiguous()
            self.dma_engine = None
        else:
            self.expert_slab_u8 = archive.expert_slab_u8
            self.dma_engine = dma_engine or PinnedSlabDMAEngine(
                max_unique_experts=min(self.n_experts, max(32, prefill_expert_chunk)),
                row_bytes=preset.expert_row_bytes,
                device=device,
            )

    def _fetch_and_unpack(self, uniq_e: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.slab_in_vram or self.expert_slab_u8.device == self.device:
            slab_rows = self.expert_slab_u8.index_select(0, uniq_e)
        else:
            assert self.dma_engine is not None
            slab_rows = self.dma_engine.fetch_unique_rows(self.expert_slab_u8, uniq_e)
        return ApexFoldEncoder.unpack_slab_views(slab_rows, self.preset)

    @staticmethod
    def _pretabulate_codebook_luts(
        x_rot: torch.Tensor,              # [M_tok, d_in] float16/float32
        hdr: ProjectionSharedHeader,
    ) -> torch.Tensor:
        """
        Pre-computes the Activation-Codebook Inner-Product Lookup Table `T` for all 4 stages
        directly in pre-transposed layout `[4, M_tok, 256, f_s0]` so `idx_s0..s3` (`[..., d_out, f_s0]`)
        never need to be transposed before or after `torch.gather`!
        """
        work_dtype = torch.float16 if x_rot.is_cuda else torch.float32
        M_tok = x_rot.shape[0]
        f_s0 = hdr.c2 // VEC_DIM
        x_act = x_rot[:, : hdr.c2].to(work_dtype).reshape(M_tok, f_s0, VEC_DIM)
        cb = hdr.codebooks.to(work_dtype)  # [4, 256, 8]
        return torch.einsum("mfd,skd->smkf", x_act, cb).contiguous()

    def _forward_proj_pretabulated(
        self,
        x_raw: torch.Tensor,              # [N, d_in] (shared across K) or [N, K, d_in] (per-expert input)
        hdr: ProjectionSharedHeader,
        views: Dict[str, torch.Tensor],
        prefix: str,
        inv_idx: torch.Tensor,            # [N, K] mapping (n, k) -> u in 0..U-1
        per_expert_input: bool = False,
    ) -> torch.Tensor:
        """
        Executes the Zero-Transpose Pre-Tabulated `CodeGEMM / AQLM` LUT-GEMV for `N <= 4` tokens.
        """
        work_dtype = torch.float16 if x_raw.is_cuda else torch.float32
        N, K = inv_idx.shape
        d_out = hdr.d_out
        f_s0 = hdr.c2 // VEC_DIM
        f_s1 = hdr.c1 // VEC_DIM
        f_s2 = hdr.c0 // VEC_DIM
        s_blk_view = views[f"{prefix}_S_blk"]
        U = s_blk_view.shape[0]
        num_groups = s_blk_view.shape[-1]
        f_per_g = f_s0 // num_groups
        flat_inv = inv_idx.reshape(-1)    # [N * K]

        if not per_expert_input:
            x_rot = transform_activations_fwht(x_raw, hdr.w_pre, hdr.perm, hdr.signs).to(work_dtype)  # [N, d_in]
            T = self._pretabulate_codebook_luts(x_rot, hdr)  # [4, N, 256, f_s0]

            if N == 1:
                # Fast path for single-token decode (N = 1): operate directly on the U unique experts
                # with ZERO index_select copy on idx_s0..s3 and ZERO transpose!
                idx_s0_u = views[f"{prefix}_idx_s0"].reshape(U * d_out, f_s0).to(torch.long)  # [PATCH R3.1-P01]
                acc_f = torch.gather(T[0, 0], dim=0, index=idx_s0_u)  # [U * d_out, f_s0]

                if f_s1 > 0 and f"{prefix}_idx_s1" in views:
                    idx_s1_u = views[f"{prefix}_idx_s1"].reshape(U * d_out, f_s1).to(torch.long)  # [PATCH R3.1-P01]
                    acc_f[:, :f_s1].add_(torch.gather(T[1, 0, :, :f_s1], dim=0, index=idx_s1_u))

                if f_s2 > 0 and f"{prefix}_idx_s2" in views:
                    idx_s2_u = views[f"{prefix}_idx_s2"].reshape(U * d_out, f_s2).to(torch.long)  # [PATCH R3.1-P01]
                    acc_f[:, :f_s2].add_(torch.gather(T[2, 0, :, :f_s2], dim=0, index=idx_s2_u))

                if hdr.stages_t0 >= 4 and f"{prefix}_idx_s3" in views and f_s2 > 0:
                    idx_s3_u = views[f"{prefix}_idx_s3"].reshape(U * d_out, f_s2).to(torch.long)  # [PATCH R3.1-P01]
                    acc_f[:, :f_s2].add_(torch.gather(T[3, 0, :, :f_s2], dim=0, index=idx_s3_u))

                acc_grp = acc_f.view(U, d_out, num_groups, f_per_g).sum(dim=-1)
                y_u = (acc_grp * s_blk_view.to(work_dtype)).sum(dim=-1)  # [U, d_out]

                if hdr.r_shared > 0 and f"{prefix}_A_e" in views:
                    z_sh = F.linear(x_rot, hdr.B_shared.to(work_dtype))  # [1, r_shared]
                    A_u = views[f"{prefix}_A_e"].to(work_dtype)          # [U, d_out, r_shared]
                    y_u.add_(torch.matmul(A_u, z_sh[0]))

                return y_u.index_select(0, flat_inv).view(1, K, d_out).to(torch.float32)

            # Batched N = 2..4 path (e.g. N=2 MTP verification): gather on dim=1 with zero transpose
            idx_s0 = views[f"{prefix}_idx_s0"].index_select(0, flat_inv).view(N, K * d_out, f_s0).to(torch.long)
            acc_f = torch.gather(T[0], dim=1, index=idx_s0)  # [N, K * d_out, f_s0]

            if f_s1 > 0 and f"{prefix}_idx_s1" in views:
                idx_s1 = views[f"{prefix}_idx_s1"].index_select(0, flat_inv).view(N, K * d_out, f_s1).to(torch.long)
                acc_f[:, :, :f_s1].add_(torch.gather(T[1, :, :, :f_s1], dim=1, index=idx_s1))

            if f_s2 > 0 and f"{prefix}_idx_s2" in views:
                idx_s2 = views[f"{prefix}_idx_s2"].index_select(0, flat_inv).view(N, K * d_out, f_s2).to(torch.long)
                acc_f[:, :, :f_s2].add_(torch.gather(T[2, :, :, :f_s2], dim=1, index=idx_s2))

            if hdr.stages_t0 >= 4 and f"{prefix}_idx_s3" in views and f_s2 > 0:
                idx_s3 = views[f"{prefix}_idx_s3"].index_select(0, flat_inv).view(N, K * d_out, f_s2).to(torch.long)
                acc_f[:, :, :f_s2].add_(torch.gather(T[3, :, :, :f_s2], dim=1, index=idx_s3))

            acc_grp = acc_f.view(N, K, d_out, num_groups, f_per_g).sum(dim=-1)
            s_blk = s_blk_view.index_select(0, flat_inv).to(work_dtype).view(N, K, d_out, num_groups)
            y_out = (acc_grp * s_blk).sum(dim=-1)

            if hdr.r_shared > 0 and f"{prefix}_A_e" in views:
                z_sh = F.linear(x_rot, hdr.B_shared.to(work_dtype))
                A_nk = views[f"{prefix}_A_e"].index_select(0, flat_inv).to(work_dtype).view(
                    N, K, d_out, hdr.r_shared
                )
                y_out.add_(torch.einsum("nr,nkor->nko", z_sh, A_nk))

            return y_out.to(torch.float32)

        else:
            # per_expert_input=True (down_proj where x_raw is [N, K, d_in])
            NK = N * K
            x_flat = x_raw.reshape(NK, hdr.d_in)
            x_rot = transform_activations_fwht(x_flat, hdr.w_pre, hdr.perm, hdr.signs).to(work_dtype)
            T = self._pretabulate_codebook_luts(x_rot, hdr)  # [4, NK, 256, f_s0]

            idx_s0 = views[f"{prefix}_idx_s0"].index_select(0, flat_inv).to(torch.long)  # [NK, d_out, f_s0]
            acc_f = torch.gather(T[0], dim=1, index=idx_s0)  # [NK, d_out, f_s0]

            if f_s1 > 0 and f"{prefix}_idx_s1" in views:
                idx_s1 = views[f"{prefix}_idx_s1"].index_select(0, flat_inv).to(torch.long)
                acc_f[:, :, :f_s1].add_(torch.gather(T[1, :, :, :f_s1], dim=1, index=idx_s1))

            if f_s2 > 0 and f"{prefix}_idx_s2" in views:
                idx_s2 = views[f"{prefix}_idx_s2"].index_select(0, flat_inv).to(torch.long)
                acc_f[:, :, :f_s2].add_(torch.gather(T[2, :, :, :f_s2], dim=1, index=idx_s2))

            if hdr.stages_t0 >= 4 and f"{prefix}_idx_s3" in views and f_s2 > 0:
                idx_s3 = views[f"{prefix}_idx_s3"].index_select(0, flat_inv).to(torch.long)
                acc_f[:, :, :f_s2].add_(torch.gather(T[3, :, :, :f_s2], dim=1, index=idx_s3))

            acc_grp = acc_f.view(NK, d_out, num_groups, f_per_g).sum(dim=-1)
            s_blk = s_blk_view.index_select(0, flat_inv).to(work_dtype)
            y_out = (acc_grp * s_blk).sum(dim=-1).view(N, K, d_out)

            if hdr.r_shared > 0 and f"{prefix}_A_e" in views:
                z_sh = F.linear(x_rot, hdr.B_shared.to(work_dtype))
                A_nk = views[f"{prefix}_A_e"].index_select(0, flat_inv).to(work_dtype)
                y_out.add_(torch.bmm(A_nk, z_sh.unsqueeze(-1)).squeeze(-1).view(N, K, d_out))

            return y_out.to(torch.float32)

    @torch.no_grad()
    def forward_routed_experts(
        self,
        x_flat: torch.Tensor,       # [N, d_model]
        top_weights: torch.Tensor,  # [N, K]
        top_idx: torch.Tensor,      # [N, K]
        swiglu_limit: float = 10.0,  # [PATCH R3-03] checkpoint swiglu_limit
        hobbit_tau: float = 1.0,
    ) -> torch.Tensor:
        """
        Dual-Path Routed Expert Execution:
          - Path A (`N <= 4`): Zero-Transpose Pre-Tabulated `T_m` scalar gather in one coalesced slab fetch.
          - Path B (`N > 4` Prefill): Token-grouped unique-expert reconstruction + Tensor-Core GEMM.
        """
        orig_dtype = x_flat.dtype
        x_dev = x_flat.to(self.device)
        N, d_model = x_dev.shape

        w_top = top_weights.to(self.device, dtype=torch.float32)
        i_top = top_idx.to(self.device, dtype=torch.long)
        if hobbit_tau < 0.9999:
            w_top, i_top = hobbit_filter_experts(w_top, i_top, tau=hobbit_tau)

        K = i_top.shape[1]
        if N == 1 and hobbit_tau >= 0.9999:
            # [R3.2-1.2] a single token's top-k experts are distinct: skip torch.unique (it forces a host sync)
            uniq_e = i_top[0]
            inv_idx = torch.arange(K, device=self.device, dtype=torch.long).view(1, K)
        else:
            uniq_e, inv_idx = torch.unique(i_top, sorted=True, return_inverse=True)
            inv_idx = inv_idx.view(N, K)

        if self.use_kernels and x_dev.is_cuda and (
                N <= 4 or (self.pool is not None and N * K <= self.max_kernel_pairs)):
            # [R3.2-1.1b] with the expert pool, prefill also runs the per-(token, expert) LUT-GEMV kernels:
            # only pool misses are transferred and the pool is warmed for the decode that follows
            return self._forward_kernel_path(x_dev, w_top, uniq_e, inv_idx, swiglu_limit).to(orig_dtype)

        if N <= 4:
            views = self._fetch_and_unpack(uniq_e)
            gate_out = self._forward_proj_pretabulated(
                x_dev, self.gate_hdr, views, "g", inv_idx, per_expert_input=False
            )
            up_out = self._forward_proj_pretabulated(
                x_dev, self.up_hdr, views, "u", inv_idx, per_expert_input=False
            )

            gate_c = gate_out.clamp(max=swiglu_limit)
            up_c = up_out.clamp(min=-swiglu_limit, max=swiglu_limit)
            z_mid = F.silu(gate_c) * up_c

            down_out = self._forward_proj_pretabulated(
                z_mid, self.down_hdr, views, "d", inv_idx, per_expert_input=True
            )

            y_routed = (down_out * w_top.unsqueeze(-1)).sum(dim=1)
            return y_routed.to(orig_dtype)

        # [R3.2-1.1] Path B (prefill) in the ROTATED domain: x_rot @ U_hat^T == x @ W_hat^T exactly
        # (orthogonal transform), so expert weights are never inverse-transformed; codebook vectors are
        # summed straight into a reusable BF16 buffer of at most `prefill_expert_chunk` experts.
        gemm_dtype = torch.bfloat16 if x_dev.is_cuda else torch.float32
        out_accum = torch.zeros((N, d_model), dtype=torch.float32, device=self.device)
        xg_rot = transform_activations_fwht(
            x_dev.float(), self.gate_hdr.w_pre, self.gate_hdr.perm, self.gate_hdr.signs
        ).to(gemm_dtype)
        xu_rot = transform_activations_fwht(
            x_dev.float(), self.up_hdr.w_pre, self.up_hdr.perm, self.up_hdr.signs
        ).to(gemm_dtype)

        U_total = int(uniq_e.numel())
        flat_inv = inv_idx.reshape(-1)
        sorted_u, sort_perm = torch.sort(flat_inv)
        tok_indices = sort_perm // K
        weights_sorted = w_top.reshape(-1).index_select(0, sort_perm)

        u_counts = torch.bincount(sorted_u, minlength=U_total)
        u_offsets = torch.zeros(U_total + 1, dtype=torch.long, device=self.device)
        torch.cumsum(u_counts, dim=0, out=u_offsets[1:])
        offsets_list = u_offsets.tolist()

        chunk = max(1, int(self.prefill_expert_chunk))
        for c_start in range(0, U_total, chunk):
            c_end = min(c_start + chunk, U_total)
            uniq_chunk = uniq_e[c_start:c_end]
            views = self._fetch_and_unpack(uniq_chunk)

            U_g = ApexFoldEncoder.reconstruct_projection_rotated(self.gate_hdr, views, "g", self.device, gemm_dtype)
            U_u = ApexFoldEncoder.reconstruct_projection_rotated(self.up_hdr, views, "u", self.device, gemm_dtype)
            U_d = ApexFoldEncoder.reconstruct_projection_rotated(self.down_hdr, views, "d", self.device, gemm_dtype)

            for local_u in range(c_end - c_start):
                global_u = c_start + local_u
                beg, end = offsets_list[global_u], offsets_list[global_u + 1]
                if beg == end:
                    continue
                routed_toks = tok_indices[beg:end]
                routed_w = weights_sorted[beg:end]
                nonzero = routed_w > 0
                if not nonzero.any():
                    continue
                if not nonzero.all():
                    routed_toks = routed_toks[nonzero]
                    routed_w = routed_w[nonzero]

                g_e = (torch.mm(xg_rot.index_select(0, routed_toks), U_g[local_u].T).to(torch.float32) * self.gain["g"]).clamp(max=swiglu_limit)
                u_e = (torch.mm(xu_rot.index_select(0, routed_toks), U_u[local_u].T).to(torch.float32) * self.gain["u"]).clamp(min=-swiglu_limit, max=swiglu_limit)
                z_e = F.silu(g_e) * u_e
                z_rot = transform_activations_fwht(z_e, self.down_hdr.w_pre, self.down_hdr.perm, self.down_hdr.signs).to(gemm_dtype)
                d_e = torch.mm(z_rot, U_d[local_u].T).to(torch.float32) * self.gain["d"]

                out_accum.index_add_(0, routed_toks, d_e * routed_w.unsqueeze(-1))
            del U_g, U_u, U_d, views

        return out_accum.to(orig_dtype)

    @torch.no_grad()
    def _layouts(self):
        if self._kr_layouts is None:
            self._kr_layouts = {
                "g": KR.RvqProjLayout(self.preset, "g", self.gate_hdr),
                "u": KR.RvqProjLayout(self.preset, "u", self.up_hdr),
                "d": KR.RvqProjLayout(self.preset, "d", self.down_hdr),
            }
        return self._kr_layouts

    def _forward_kernel_path(self, x_dev, w_top, uniq_e, inv_idx, swiglu_limit):
        """[R3.2-1.4] Decode (N <= 4): LUT-GEMV straight from uint8 codes in the slab rows (one launch per
        projection for all active experts), activations rotated once; numerically equivalent to Path A."""
        N, K = inv_idx.shape
        if self.pool is not None:
            # [R3.2-1.6] global VRAM expert cache: rows live in pool slots (exact archive bytes)
            rows = self.pool.pool
            slot_of_u = self.pool.ensure(self.layer_id, uniq_e.cpu().tolist())
        elif self.slab_in_vram or self.expert_slab_u8.device == self.device:
            rows = self.expert_slab_u8
            slot_of_u = uniq_e.to(torch.int32)
        else:
            rows = self.dma_engine.fetch_unique_rows(self.expert_slab_u8, uniq_e)
            slot_of_u = torch.arange(int(uniq_e.numel()), device=self.device, dtype=torch.int32)
        pair_slots = slot_of_u.index_select(0, inv_idx.reshape(-1))
        lay = self._layouts()
        xg = transform_activations_fwht(x_dev.float(), self.gate_hdr.w_pre, self.gate_hdr.perm, self.gate_hdr.signs)
        xu = transform_activations_fwht(x_dev.float(), self.up_hdr.w_pre, self.up_hdr.perm, self.up_hdr.signs)
        if N == 1:
            g = KR.rvq_gemv(rows, pair_slots, lay["g"], xg, False)
            u = KR.rvq_gemv(rows, pair_slots, lay["u"], xu, False)
        else:
            g = KR.rvq_gemv(rows, pair_slots, lay["g"], xg.repeat_interleave(K, 0), True)
            u = KR.rvq_gemv(rows, pair_slots, lay["u"], xu.repeat_interleave(K, 0), True)
        if self.gain["g"] != 1.0:
            g = g * self.gain["g"]
        if self.gain["u"] != 1.0:
            u = u * self.gain["u"]
        z = F.silu(g.clamp(max=swiglu_limit)) * u.clamp(min=-swiglu_limit, max=swiglu_limit)
        zr = transform_activations_fwht(z, self.down_hdr.w_pre, self.down_hdr.perm, self.down_hdr.signs)
        d = KR.rvq_gemv(rows, pair_slots, lay["d"], zr, True)
        if self.gain["d"] != 1.0:
            d = d * self.gain["d"]
        return (d.view(N, K, -1) * w_top.unsqueeze(-1)).sum(dim=1)

    def decode_core(self, x, rows, pair_slots, w_pairs, n_tok: int, lim: float):
        """[R3.2] format-agnostic core used by the CUDA-graph decoder: x [N, D] fp32, pair_slots int32 [N*K]."""
        P = int(pair_slots.numel())
        K = P // n_tok
        lay = self._layouts()
        xg = transform_activations_fwht(x, self.gate_hdr.w_pre, self.gate_hdr.perm, self.gate_hdr.signs)
        xu = transform_activations_fwht(x, self.up_hdr.w_pre, self.up_hdr.perm, self.up_hdr.signs)
        if n_tok == 1:
            g = KR.rvq_gemv(rows, pair_slots, lay["g"], xg, False)
            u = KR.rvq_gemv(rows, pair_slots, lay["u"], xu, False)
        else:
            g = KR.rvq_gemv(rows, pair_slots, lay["g"], xg.repeat_interleave(K, 0), True)
            u = KR.rvq_gemv(rows, pair_slots, lay["u"], xu.repeat_interleave(K, 0), True)
        if self.gain["g"] != 1.0:
            g = g * self.gain["g"]
        if self.gain["u"] != 1.0:
            u = u * self.gain["u"]
        z = F.silu(g.clamp(max=lim)) * u.clamp(min=-lim, max=lim)
        zr = transform_activations_fwht(z, self.down_hdr.w_pre, self.down_hdr.perm, self.down_hdr.signs)
        d = KR.rvq_gemv(rows, pair_slots, lay["d"], zr, True)
        if self.gain["d"] != 1.0:
            d = d * self.gain["d"]
        return (d * w_pairs.view(-1, 1)).view(n_tok, K, -1).sum(dim=1)

    def forward_mode_b_reference(
        self,
        x_flat: torch.Tensor,       # [N, d_model]
        top_weights: torch.Tensor,  # [N, K]
        top_idx: torch.Tensor,      # [N, K]
        swiglu_limit: float = 10.0,  # [PATCH R3-03] checkpoint swiglu_limit
    ) -> torch.Tensor:
        """
        Mode-B Explicit Weight Dequantization Reference Path (verifies that Path-A Pre-Tabulated
        LUT-GEMV matches explicit matrix reconstruction to floating-point machine precision!).
        """
        orig_dtype = x_flat.dtype
        x_f32 = x_flat.to(self.device, dtype=torch.float32)
        w_top = top_weights.to(self.device, dtype=torch.float32)
        i_top = top_idx.to(self.device, dtype=torch.long)

        N, K = i_top.shape
        uniq_e, inv_idx = torch.unique(i_top, sorted=True, return_inverse=True)
        views = self._fetch_and_unpack(uniq_e)

        W_g_u = ApexFoldEncoder.reconstruct_projection_weights(self.gate_hdr, views, "g", device=self.device)
        W_u_u = ApexFoldEncoder.reconstruct_projection_weights(self.up_hdr, views, "u", device=self.device)
        W_d_u = ApexFoldEncoder.reconstruct_projection_weights(self.down_hdr, views, "d", device=self.device)

        out = torch.zeros_like(x_f32)
        for n in range(N):
            for k in range(K):
                u_id = int(inv_idx.view(N, K)[n, k].item())
                wt = w_top[n, k]
                g = torch.mv(W_g_u[u_id], x_f32[n]).clamp(max=swiglu_limit)
                u = torch.mv(W_u_u[u_id], x_f32[n]).clamp(min=-swiglu_limit, max=swiglu_limit)
                z = F.silu(g) * u
                d = torch.mv(W_d_u[u_id], z)
                out[n] += wt * d
        return out.to(orig_dtype)


class ApexFoldPlacementManager:
    """
    Deterministic VRAM / Host-RAM placement planner that calls the exact same
    `compute_exact_hardware_budget` function as `--doctor` so VRAM never spills.
    """

    @staticmethod
    def plan_placement(
        preset_name: str,
        available_layers: List[int],
        backbone_vram_gib: float,
        max_vram_budget_gib: float = 30.6,
        kv_cache_gib: float = 1.20,
    ) -> Dict[str, object]:
        cfg = GLM53FlashModelConfig()
        preset = APEX_FOLD_PRESETS[preset_name]
        n_exp = cfg.n_routed_experts

        shared_per_layer_gib = preset.shared_basis_bytes_per_layer() / (1024.0 ** 3)
        slab_per_layer_gib = preset.slab_bytes_per_layer(n_exp) / (1024.0 ** 3)
        dma_staging_gib = (32 * preset.expert_row_bytes * 2) / (1024.0 ** 3)

        all_shared_gib = len(available_layers) * shared_per_layer_gib
        fixed_vram_gib = backbone_vram_gib + kv_cache_gib + all_shared_gib + dma_staging_gib
        avail_for_slabs_gib = max(0.0, max_vram_budget_gib - fixed_vram_gib)

        max_vram_slabs = min(len(available_layers), int(avail_for_slabs_gib // slab_per_layer_gib))

        prio: List[int] = []
        for l in (3, 4, 5, 6, 45):
            if l in available_layers and l not in prio:
                prio.append(l)
        for l in cfg.dsa_layers:
            if l in available_layers and l not in prio:
                prio.append(l)
        for l in sorted(available_layers):
            if l not in prio:
                prio.append(l)

        vram_resident_layers = sorted(prio[:max_vram_slabs])
        host_resident_layers = sorted(prio[max_vram_slabs:])

        return {
            "preset": preset_name,
            "backbone_vram_gib": round(backbone_vram_gib, 3),
            "shared_headers_vram_gib": round(all_shared_gib, 3),
            "vram_slabs_gib": round(len(vram_resident_layers) * slab_per_layer_gib, 3),
            "total_planned_vram_gib": round(fixed_vram_gib + len(vram_resident_layers) * slab_per_layer_gib, 3),
            "host_slabs_gib": round(len(host_resident_layers) * slab_per_layer_gib, 3),
            "vram_resident_layers": vram_resident_layers,
            "host_resident_layers": host_resident_layers,
        }
