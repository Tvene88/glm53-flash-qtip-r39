#!/usr/bin/env python3
"""
r39_stage2b_global_kl_trainer.py  [Round 3.9 Stage-2B Layer-Major Global-KL Co-Adaptation Trainer]
==================================================================================================
Single-RTX-5090 (32 GiB GDDR7 VRAM + 96 GB DDR5 RAM) global teacher-KL co-adaptation engine for
GLM-5.3-Flash (320.8B, 45 trunk layers: 3 dense + 42 routed MoE, 4-stream mHC hyper-connections).

Jointly optimizes THREE zero-byte-overhead parameter families against end-to-end full-model teacher
KL divergence while keeping all packed trellis bitstreams (`c1`, `c2`, `c3`) 100% frozen:
  1. `AnchoredRouterLoRA` + `delta_gate_b` (`ST-Top8` straight-through sigmoid router + correction bias):
       Heals non-uniform cross-layer LOO expert-pruning holes (`R3.8` / `R3.9`) and steers sensitive
       tokens away from `k=1` tail rows or high-residual directions. Merges in place into existing
       `mlp.gate.weight` and `mlp.gate.e_score_correction_bias` at export (`0 B` serving overhead).
  2. Layer-Shared `2 KiB` `QTIP HYB` Codebooks (`[512, 2]` FP16 per `(layer, proj, k)` — or `TCQ33`
     `[3, 4096]` FP16 tables when `--archive-format tcq33`):
       Adapts the marginal 2D reconstruction levels to non-Gaussian heavy-tailed MoE residuals and
       post-routing token covariance shift (`0 B` slab overhead).
  3. In-Place Per-Expert FP16 Row Scales (`off_scale` in `expert_slab_u8` / `fields['scale']`):
       Multiplicative log-scale factors `s = fp16(s_0 * exp(lambda))` anchored to initial LDLQ scales
       `s_0`, stored in host-paged AdamW state (`~0.5 GiB` DDR5 RAM across all 42 layers, `0 B` idle
       VRAM) and written straight back into `off_scale` in place (`0 B` serving overhead).

Single-5090 Layer-Major Windowed Execution Schedule (`W = 4` Layers/Window, `11` Host Checkpoints):
  * Forward Sweep (`torch.no_grad()`, `use_kernels=True`):
      Streams macro-batch (`N_seq = 32 x 2048` tokens) layer-by-layer `l = 0..44`, saving 11 `bf16`
      boundary checkpoints `[N_seq, S, 4, 4096]` in DDR5 host RAM (`22.0 GiB` total).
  * Head Loss & Top Gradient (`l = 44 -> dL / dh_45`):
      Computes exact Top-128 + Residual-Tail Teacher KL (`pos >= 10`) in 512-token head chunks
      (`< 0.4 GiB` VRAM) and writes `grad_h` (`2.0 GiB` `bf16` host buffer).
  * Reverse Window Sweep (`w = 10` down to `0`, e.g. `[40..44], [36..39], ..., [0..3]`):
      Loads ONLY the `<= 5` layers of the active window into VRAM (`~6.4 GiB` slabs + `~0.9 GiB`
      backbone), toggles `_DifferentiableBackboneScope` (`use_kernels=False`, `_FUSED_MERGE=False`)
      so autograd flows cleanly through mHC Sinkhorn / KDA WY / DSA MLA, backpropagates micro-batches
      (`micro_bs = 1..2` seqs) via `CompressedMoERoutedFunction` (exact bilinear `G^T X` Tensor-Core
      gradients without unpacking full-layer FP16 weights), steps optimizers, patches `off_scale` in
      place, and passes `h_in.grad` to the window below. Peak VRAM: `~11.8 GiB / 32.0 GiB`.

Typical Usage:
  # 1) Standalone GPU self-test (~8s: verifies mHC + ST-TopK + HYB + off_scale autograd & 0-byte invariance):
  python r39_stage2b_global_kl_trainer.py --selftest

  # 2) Train on Round 3.9 QTIP HYB or Round 3.8 pruned TCQ33 archive + export merged artifacts:
  python r39_stage2b_global_kl_trainer.py \
      --backbone-dir C:\models\GLM53-backbone-r34h \
      --archive-dir  C:\models\GLM53-APEX-FOLD-R39-HYB \
      --teacher-cache C:\models\r39_teacher_top128\train_cal_top128.pt \
      --qbench-teacher-evx C:\models\GLM53-APEX-FOLD-TCQ33\_r33_ckpt\evx.pt \
      --prune-spec out_r32\r38_masks\hyb_W16_75GB.json \
      --export-dir C:\models\GLM53-APEX-FOLD-R39-HYB-FT \
      --sweeps 6 --macro-seqs 32 --micro-bs 1 --lora-rank 16
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import glob
import gzip
import json
import math
import gc
import os
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LIM = 10.0  # GLM-5.3-Flash SwiGLU clamp limit


def _first_existing(candidates: List[str], fallback: str) -> str:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return fallback


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PKG = _first_existing(
    [
        os.path.join(os.path.dirname(HERE), "runtime"),
        os.path.join(HERE, "claude_r32_apex_fold"),
        os.path.join(os.path.dirname(HERE), "claude_r32_apex_fold"),
        r"C:\models\GLM53-APEX-FOLD-TCQ\claude_r32_apex_fold",
        "/mnt/c/models/GLM53-APEX-FOLD-TCQ/claude_r32_apex_fold",
    ],
    os.path.join(os.path.dirname(HERE), "claude_r32_apex_fold"),
)
DEFAULT_BB = _first_existing(
    [
        r"C:\models\GLM53-backbone-r34h",
        "/mnt/c/models/GLM53-backbone-r34h",
        r"C:\models\GLM53-backbone",
        "/mnt/c/models/GLM53-backbone",
    ],
    r"C:\models\GLM53-backbone-r34h",
)
DEFAULT_ARCH = _first_existing(
    [r"C:\models\GLM53-APEX-FOLD-TCQ33", "/mnt/c/models/GLM53-APEX-FOLD-TCQ33"],
    r"C:\models\GLM53-APEX-FOLD-TCQ33",
)
DEFAULT_CT = _first_existing(
    [r"C:\models\calib_text\exl3\cal_trace.json", "/mnt/c/models/calib_text/exl3/cal_trace.json"],
    r"C:\models\calib_text\exl3\cal_trace.json",
)
DEFAULT_C4 = _first_existing(
    [
        r"C:\models\calib_text\en\c4-validation.00000-of-00008.json.gz",
        "/mnt/c/models/calib_text/en/c4-validation.00000-of-00008.json.gz",
    ],
    r"C:\models\calib_text\en\c4-validation.00000-of-00008.json.gz",
)
DEFAULT_QB = _first_existing(
    [r"C:\models\calib_text\exl3\qbench_prompts.json", "/mnt/c/models/calib_text/exl3/qbench_prompts.json"],
    r"C:\models\calib_text\exl3\qbench_prompts.json",
)


# ================================================================================================================
# 1. Self-Contained Orthonormal FWHT & QTIP Tail-Biting Bitstream Unpacker
# ================================================================================================================
_HAD_CACHE: Dict[Tuple[int, str], torch.Tensor] = {}
_HYB_HASH_CACHE: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _hadamard(n: int, device: torch.device) -> torch.Tensor:
    key = (n, str(device))
    if key not in _HAD_CACHE:
        h = np.array([[1.0]], dtype=np.float32)
        while h.shape[0] < n:
            h = np.block([[h, h], [h, -h]])
        h *= 1.0 / math.sqrt(float(n))
        _HAD_CACHE[key] = torch.from_numpy(h).to(device=device, dtype=torch.float32).contiguous()
    return _HAD_CACHE[key]


def fwht_full(x: torch.Tensor) -> torch.Tensor:
    """Symmetric orthonormal Fast Walsh-Hadamard Transform across last dimension d_in (self-adjoint in autograd)."""
    orig_shape = x.shape
    d_in = orig_shape[-1]
    xf = x.float()
    if d_in <= 64:
        return torch.matmul(xf, _hadamard(d_in, xf.device)).to(x.dtype)
    log_d = int(round(math.log2(d_in)))
    d1 = 1 << (log_d // 2)
    d2 = d_in // d1
    v2d = xf.reshape(-1, d1, d2)
    out = torch.matmul(_hadamard(d1, xf.device), torch.matmul(v2d, _hadamard(d2, xf.device))).reshape(orig_shape)
    return out.to(x.dtype)


def _hyb_hash_tables(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns cached (bucket_idx [65536] long, sx [65536] fp32, sy [65536] fp32) for bijective HYB-mul1."""
    key = str(device)
    if key not in _HYB_HASH_CACHE:
        s = torch.arange(65536, device=device, dtype=torch.int64)
        h = (34038481 * s + 76625530) & 0xFFFFFFFF
        idx = (h >> 17) & 511
        sx = torch.where((h & 0x00008000) != 0, -1.0, 1.0).float()
        sy = torch.where((h & 0x80000000) != 0, -1.0, 1.0).float()
        _HYB_HASH_CACHE[key] = (idx, sx, sy)
    return _HYB_HASH_CACHE[key]


def unpack_tailbite_l16(packed: torch.Tensor, M: int, step_bits: int) -> torch.Tensor:
    """Vectorized circular MSB-first 16-bit window reader matching CUDA read_tailbite_state16."""
    R, rb = packed.shape
    m = torch.arange(M, device=packed.device, dtype=torch.long)
    bit_pos = m * step_bits
    b0 = (bit_pos >> 3) & (rb - 1)
    b1 = (b0 + 1) & (rb - 1)
    b2 = (b0 + 2) & (rb - 1)
    off = bit_pos & 7
    p = packed.long()
    w24 = (p[:, b0] << 16) | (p[:, b1] << 8) | p[:, b2]
    return ((w24 >> (8 - off)) & 0xFFFF).to(torch.int32)


def unpack_tcq33_l12(packed: torch.Tensor, T: int, k: int) -> torch.Tensor:
    """Vectorized 12-bit window reader matching glm53_fold_tcq.unpack_indices."""
    t = torch.arange(T, device=packed.device, dtype=torch.long)
    bitpos = k * t
    byte = bitpos >> 3
    off = bitpos & 7
    p = packed.long()
    w24 = (p[:, byte] << 16) | (p[:, byte + 1] << 8) | p[:, byte + 2]
    return (w24 >> (12 - off)) & 4095


def _mem_line(device=None) -> str:
    """[R3.9 addition, 2026-11-17] host and GPU memory of this process, for the per-phase / per-window log lines."""
    try:
        import psutil  # type: ignore
        mi = psutil.Process().memory_info()
        vm = psutil.virtual_memory()
        host = f"RSS {mi.rss / 2**30:.1f} private {getattr(mi, 'private', mi.vms) / 2**30:.1f} RAM-free {vm.available / 2**30:.1f} GiB"
    except Exception:
        host = "host n/a"
    try:
        free_b, tot_b = torch.cuda.mem_get_info()
        gpu = (f"VRAM alloc {torch.cuda.memory_allocated() / 2**30:.1f} peak {torch.cuda.max_memory_allocated() / 2**30:.1f} "
               f"reserved {torch.cuda.memory_reserved() / 2**30:.1f} dev-free {free_b / 2**30:.1f} GiB")
    except Exception:
        gpu = "gpu n/a"
    return host + " | " + gpu


def silu_backward(x: torch.Tensor, grad_out: torch.Tensor) -> torch.Tensor:
    sig = torch.sigmoid(x)
    return grad_out * sig * (1.0 + x * (1.0 - sig))


# ================================================================================================================
# 2. Host-Paged AdamW State for In-Place Per-Expert FP16 Row Scales (0 B Idle VRAM)
# ================================================================================================================
class HostPagedScaleAdamW:
    """Maintains multiplicative log-scale parameters `la` (`s = fp16(s0 * exp(la))`) + AdamW moments in host RAM
    (`bfloat16` moments + `float16` `s0` = ~12 MiB per layer, ~500 MiB across all 42 layers, 0 B idle VRAM!).
    Writes updated `fp16` scales straight back into the archive's `off_scale` storage after every window step."""

    def __init__(
        self,
        init_scales: Dict[str, torch.Tensor],  # p in ('g','u','d') -> CPU fp16 [E_kept, d_out]
        lr: float = 4e-4,
        betas: Tuple[float, float] = (0.9, 0.995),
        eps: float = 1e-8,
        anchor_wd: float = 2e-3,
        max_log_delta: float = 0.30,
    ):
        self.lr = float(lr)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.anchor_wd = float(anchor_wd)
        self.max_log_delta = float(max_log_delta)
        self.step_count = 0

        self.s0: Dict[str, torch.Tensor] = {}
        self.la: Dict[str, torch.Tensor] = {}
        self.m: Dict[str, torch.Tensor] = {}
        self.v: Dict[str, torch.Tensor] = {}
        for p, sc in init_scales.items():
            # [R3.9 fix 5, 2026-11-16] `.cpu().to(fp16).contiguous()` returns the SAME tensor for a CPU fp16 input, so
            # s0 aliased `fields_cpu[p]["scale"]`; `commit_scales_in_place` then overwrote s0 with s0*exp(la) and every
            # step compounded (s = s0*exp(sum of all past la)), outside `max_log_delta` and the anchor. Own copy.
            sc_cpu = sc.detach().to("cpu", torch.float16).clone().contiguous()
            self.s0[p] = sc_cpu
            self.la[p] = torch.zeros_like(sc_cpu, dtype=torch.float32)
            self.m[p] = torch.zeros_like(sc_cpu, dtype=torch.bfloat16)
            self.v[p] = torch.zeros_like(sc_cpu, dtype=torch.bfloat16)

    @torch.no_grad()
    def current_scale_fp32(self, p: str, device: torch.device) -> torch.Tensor:
        s0_d = self.s0[p].to(device=device, dtype=torch.float32)
        la_d = self.la[p].to(device=device, dtype=torch.float32)
        # Round through FP16 in forward so training sees the exact quantized FP16 scale that ships in off_scale!
        s_fp16 = (s0_d * torch.exp(la_d)).to(torch.float16)
        return s_fp16.float()

    @torch.no_grad()
    def step_and_pack_fp16(
        self, grad_scale_raw: Dict[str, torch.Tensor], device: torch.device, grad_is_log: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Takes dL/ds [E_kept, d_out] for p in ('g','u','d'), converts via chain rule dL/d(la) = dL/ds * s,
        applies decoupled anchor decay toward la=0 (s -> s0), updates moments, and returns updated CPU FP16 scales.
        [R3.9 fix 12] grad_is_log=True: the input already is dL/d(la) (dense reverse path)."""
        self.step_count += 1
        bc1 = 1.0 - (self.beta1 ** self.step_count)
        bc2 = 1.0 - (self.beta2 ** self.step_count)
        updated_fp16: Dict[str, torch.Tensor] = {}

        for p in ("g", "u", "d"):
            s0_d = self.s0[p].to(device=device, dtype=torch.float32)
            la_d = self.la[p].to(device=device, dtype=torch.float32)
            m_d = self.m[p].to(device=device, dtype=torch.float32)
            v_d = self.v[p].to(device=device, dtype=torch.float32)

            s_cur = s0_d * torch.exp(la_d)
            g_la = grad_scale_raw[p].to(device=device, dtype=torch.float32)
            if not grad_is_log:
                g_la = g_la * s_cur

            # Only step rows that actually received non-zero routing gradient (idle experts stay exact s0!)
            active = g_la.abs() > 0.0
            if bool(active.any()):
                g_eff = g_la + self.anchor_wd * la_d
                m_new = torch.where(active, self.beta1 * m_d + (1.0 - self.beta1) * g_eff, m_d)
                v_new = torch.where(active, self.beta2 * v_d + (1.0 - self.beta2) * (g_eff * g_eff), v_d)
                m_hat = m_new / bc1
                v_hat = v_new / bc2
                step_delta = self.lr * m_hat / (v_hat.sqrt() + self.eps)
                la_new = torch.where(active, (la_d - step_delta).clamp(-self.max_log_delta, self.max_log_delta), la_d)
                self.la[p].copy_(la_new.cpu())
                self.m[p].copy_(m_new.to(torch.bfloat16).cpu())
                self.v[p].copy_(v_new.to(torch.bfloat16).cpu())
                s_new_fp16 = (s0_d * torch.exp(la_new)).to(torch.float16).cpu()
            else:
                s_new_fp16 = (s0_d * torch.exp(la_d)).to(torch.float16).cpu()

            updated_fp16[p] = s_new_fp16
        return updated_fp16

    def state_dict(self) -> dict:
        return {"step_count": self.step_count, "la": {p: v.clone() for p, v in self.la.items()}}

    def load_state_dict(self, sd: dict) -> None:
        self.step_count = int(sd["step_count"])
        for p, v in sd["la"].items():
            self.la[p].copy_(v)


# ================================================================================================================
# 3. Unified Compressed Layer Archive Handle (Supports BOTH Round 3.9 QTIP HYB and Round 3.4/3.8 TCQ33 .npz)
# ================================================================================================================
@dataclass
class ProjMeta:
    key: str
    d_out: int
    d_in: int
    n1: int
    n2: int
    n3: int
    k_sparse: int
    tile_mode: str  # '2d_16x16', '1d_row', or 'tcq33_l12'


class CompressedLayerHandle:
    """Holds one compressed MoE layer on host (`uint8` codes + `fp16` scales + `int8` signs) and materializes
    GPU views ONLY while its 4-layer window is active. Writes updated FP16 scales and codebooks in place."""

    def __init__(
        self,
        layer_id: int,
        fmt: str,
        projs: Dict[str, ProjMeta],
        signs_cpu: Dict[str, torch.Tensor],
        fields_cpu: Dict[str, Dict[str, torch.Tensor]],
        codebook_init_fp16: torch.Tensor,
        row_of_cpu: Optional[torch.Tensor] = None,
        missing_ids: Optional[List[int]] = None,
        tcq_npz_meta: Optional[dict] = None,
    ):
        self.layer_id = int(layer_id)
        self.fmt = fmt  # 'qtip_hyb' or 'tcq33'
        self.projs = projs
        self.signs_cpu = signs_cpu
        self.fields_cpu = fields_cpu
        self.n_kept = int(fields_cpu["g"]["map"].shape[0])
        self.row_of_cpu = row_of_cpu
        self.missing_ids = sorted(missing_ids or [])
        self.tcq_npz_meta = tcq_npz_meta

        # Differentiable codebook parameter:
        #   - qtip_hyb: [3, 2, 512, 2] FP32 master (g/u/d x k=1,2 x 512 x 2 = 6,144 floats = 12 KiB FP16 total!)
        #   - tcq33   : [3, 4096] FP32 master (k=1,2,3 x 4096 = 12,288 floats = 24 KiB FP16 total!)
        cb_f32 = codebook_init_fp16.detach().float().clone()
        self.cb_init_cpu = cb_f32.cpu().clone()
        self.cb_param = nn.Parameter(cb_f32)

        self.scale_opt = HostPagedScaleAdamW(
            init_scales={p: fields_cpu[p]["scale"] for p in ("g", "u", "d")}
        )
        self._gpu_cache: Optional[dict] = None

    @classmethod
    def from_qtip_pt(cls, layer_id: int, path: str) -> "CompressedLayerHandle":
        blob = torch.load(path, map_location="cpu")
        tile_mode = str(blob.get("tile_mode", "2d_16x16"))
        projs = {}
        fields_cpu = {}
        for p, default_dout, default_din, default_ks in (
            ("g", 2048, 4096, 64),
            ("u", 2048, 4096, 64),
            ("d", 4096, 2048, 256),
        ):
            f = blob["fields"][p]
            d_out = int(f["map"].shape[1])
            d_in = int(blob["signs"][p].numel())
            n1 = int(f["c1"].shape[1] * (16 if tile_mode == "2d_16x16" else 1))
            n2 = d_out - n1
            ks = int(f["sp"].shape[1]) if "sp" in f else default_ks
            projs[p] = ProjMeta(p, d_out, d_in, n1, n2, 0, ks, tile_mode)
            fields_cpu[p] = {k: v.contiguous() for k, v in f.items() if torch.is_tensor(v)}

        # Pack [3, 2, 512, 2] FP16 HYB LUTs across ('g', 'u', 'd') and k in (1, 2)
        lut_list = []
        for p in ("g", "u", "d"):
            lp = blob["hyb_luts"][p]
            lut_list.append(torch.stack([lp[1].to(torch.float16), lp[2].to(torch.float16)], dim=0))
        cb_fp16 = torch.stack(lut_list, dim=0)  # [3, 2, 512, 2]

        perm = blob.get("perm", None)
        n_tot = int(blob.get("n_experts_total", 288))
        row_of = None
        missing: List[int] = []
        if perm is not None:
            perm_t = torch.as_tensor(perm, dtype=torch.long)
            row_of = torch.full((n_tot,), -1, dtype=torch.long)
            row_of[perm_t] = torch.arange(perm_t.numel(), dtype=torch.long)
            missing = torch.nonzero(row_of < 0).flatten().tolist()

        signs_cpu = {p: blob["signs"][p].to(torch.int8).contiguous() for p in ("g", "u", "d")}
        return cls(
            layer_id=layer_id,
            fmt="qtip_hyb",
            projs=projs,
            signs_cpu=signs_cpu,
            fields_cpu=fields_cpu,
            codebook_init_fp16=cb_fp16,
            row_of_cpu=row_of,
            missing_ids=missing,
            tcq_npz_meta={
                "src_path": path,
                "format": str(blob.get("format", "qtip_hyb_r39")),
                "rule": str(blob.get("rule", "A")),
                "meta": blob.get("meta", {}),
                "perm": perm,
                "n_experts_total": n_tot,
            },
        )

    @classmethod
    def from_tcq33_npz(cls, layer_id: int, path: str, TF_mod, TR_mod) -> "CompressedLayerHandle":
        arch = TR_mod.TcqArchive(path, mmap_slab=False)
        preset = arch.preset
        slab = arch.expert_slab_u8
        projs = {}
        fields_cpu = {}
        for p, sp in (("g", preset.gate), ("u", preset.up), ("d", preset.down)):
            projs[p] = ProjMeta(p, sp.d_out, sp.d_in, sp.n1, sp.n2, sp.n3, sp.k_sparse, "tcq33_l12")
            uf = TF_mod.unpack_fields(preset, slab, p)
            fields_cpu[p] = {k: v.clone().contiguous() for k, v in uf.items()}
        cb_fp16 = arch.tables.to(torch.float16).contiguous()  # [3, 4096]
        missing = arch.missing.tolist() if arch.missing is not None else []
        return cls(
            layer_id=layer_id,
            fmt="tcq33",
            projs=projs,
            signs_cpu=arch.signs,
            fields_cpu=fields_cpu,
            codebook_init_fp16=cb_fp16,
            row_of_cpu=arch.row_of,
            missing_ids=missing,
            tcq_npz_meta={
                "src_path": path,
                "rule": arch.rule,
                "preset": preset,
                "meta": arch.meta,
                "perm": arch.perm,
                "slab_u8": slab,
            },
        )

    def to_gpu(self, device: torch.device) -> None:
        if self._gpu_cache is not None:
            return
        g_fields = {}
        for p in ("g", "u", "d"):
            fc = self.fields_cpu[p]
            g_fields[p] = {
                k: v.to(device, non_blocking=True)
                for k, v in fc.items()
                if k in ("map", "c1", "c2", "c3", "sp")
            }
        self._gpu_cache = {
            "signs": {p: self.signs_cpu[p].to(device=device, dtype=torch.float32) for p in ("g", "u", "d")},
            "fields": g_fields,
            "row_of": self.row_of_cpu.to(device) if self.row_of_cpu is not None else None,
        }

    def offload_cpu(self) -> None:
        self._gpu_cache = None

    @torch.no_grad()
    def commit_scales_in_place(self, updated_scales_fp16: Dict[str, torch.Tensor]) -> None:
        """Writes updated FP16 row scales directly into `fields_cpu[p]['scale']` AND (for TCQ33) directly into
        the exact `off_scale` byte slice of `expert_slab_u8` (`0 B` size change, byte-lock preserved!)."""
        for p in ("g", "u", "d"):
            sc = updated_scales_fp16[p].to(torch.float16).contiguous()
            assert sc.shape == self.fields_cpu[p]["scale"].shape
            self.fields_cpu[p]["scale"].copy_(sc)

        if self.fmt == "tcq33" and self.tcq_npz_meta is not None and "slab_u8" in self.tcq_npz_meta:
            preset = self.tcq_npz_meta["preset"]
            lay, row_bytes = preset.layout()
            slab = self.tcq_npz_meta["slab_u8"]
            E = slab.shape[0]
            for p in ("g", "u", "d"):
                off, nb = lay[f"{p}_scale"]
                raw_bytes = self.fields_cpu[p]["scale"].contiguous().view(torch.uint8).reshape(E, nb)
                slab[:, off : off + nb] = raw_bytes
            assert slab.shape[1] == row_bytes, "Byte lock violated during off_scale writeback!"

    @torch.no_grad()
    def decode_all_dense_bf16(self, cb_param: torch.Tensor, device: torch.device, group: int = 16) -> Dict[str, torch.Tensor]:
        """[R3.9 addition, 2026-11-16] Every kept expert of the layer as bf16 [E_kept, d_out, d_in] per projection
        (about 12.6 GB for 250 experts), decoded with the SAME group routine, live LUT and live scales as
        `CompressedMoERoutedFunction.forward`, which casts each decoded fp32 matrix to bf16 at its matmul anyway.
        Lets the no-grad passes decode a layer once instead of once per sequence."""
        sc_live = {p: self.scale_opt.current_scale_fp32(p, device) for p in ("g", "u", "d")}
        out: Dict[str, torch.Tensor] = {}
        for p in ("g", "u", "d"):
            pm = self.projs[p]
            W = torch.empty((self.n_kept, pm.d_out, pm.d_in), dtype=torch.bfloat16, device=device)
            for g0 in range(0, self.n_kept, group):
                sl = torch.arange(g0, min(self.n_kept, g0 + group), device=device, dtype=torch.long)
                Wf, _, _, _ = self.decode_expert_group_unit_and_full(p, sl, cb_param, sc_live[p], device)
                W[g0 : g0 + sl.numel()] = Wf.bfloat16()
                del Wf
            out[p] = W
        torch.cuda.empty_cache()  # [R3.9 fix 10] drop the cached fp32 group temporaries before the forward uses the GPU
        return out

    @torch.no_grad()
    def decode_expert_group_unit_and_full(
        self,
        p: str,
        slot_ids: torch.Tensor,  # [G] int64 indices into 0..E_kept-1
        cb_live_f32: torch.Tensor,
        sc_live_f32: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[int, torch.Tensor]]:
        """Decodes a small batch of `G <= 16` experts into:
          - W_full [G, d_out, d_in] fp32 (original domain, dense + sparse, rounded via FP16 codebook!)
          - U_unit [G, d_out, d_in] fp32 (stored row order, BEFORE multiplying by row scale!)
          - rmap   [G, d_out] int64
          - unpacked_states: dict k -> int64 state tensor (reused in backward for exact LUT bucket scatter!)."""
        assert self._gpu_cache is not None
        pm = self.projs[p]
        gf = self._gpu_cache["fields"][p]
        signs = self._gpu_cache["signs"][p]
        G = int(slot_ids.numel())
        d_out, d_in = pm.d_out, pm.d_in

        rmap = (gf["map"].index_select(0, slot_ids).long()) & 0xFFFF
        sc = sc_live_f32.index_select(0, slot_ids)  # [G, d_out] fp32
        U_unit = torch.empty((G, d_out, d_in), dtype=torch.float32, device=device)
        unpacked_states: Dict[int, torch.Tensor] = {}

        # Round live FP32 codebook through FP16 via straight-through rounding so forward == exported FP16 LUT
        cb_q = cb_live_f32.to(torch.float16).float()

        if self.fmt == "qtip_hyb":
            p_idx = {"g": 0, "u": 1, "d": 2}[p]
            idx_all, sx_all, sy_all = _hyb_hash_tables(device)
            for k, a, b in ((1, 0, pm.n1), (2, pm.n1, d_out)):
                if b <= a:
                    continue
                lut_k = cb_q[p_idx, k - 1]  # [512, 2]
                base = lut_k[idx_all]
                pair_tab = torch.stack([base[:, 0] * sx_all, base[:, 1] * sy_all], dim=-1)  # [65536, 2]
                if pm.tile_mode == "2d_16x16":
                    nt, nb = (b - a) // 16, d_in // 16
                    packed = gf[f"c{k}"].index_select(0, slot_ids).reshape(G * nt * nb, 32 * k)
                    st = unpack_tailbite_l16(packed, M=128, step_bits=2 * k).long()
                    unpacked_states[k] = st
                    tiles = pair_tab[st].reshape(G, nt, nb, 16, 16).permute(0, 1, 3, 2, 4).reshape(G, b - a, d_in)
                    U_unit[:, a:b, :] = tiles
                else:
                    packed = gf[f"c{k}"].index_select(0, slot_ids).reshape(G * (b - a), (d_in * k) // 8)
                    st = unpack_tailbite_l16(packed, M=d_in // 2, step_bits=2 * k).long()
                    unpacked_states[k] = st
                    U_unit[:, a:b, :] = pair_tab[st].reshape(G, b - a, d_in)
        else:
            # TCQ33 L=12 explicit 4096-entry LUT per k in {1, 2, 3}
            a = 0
            for k, n_rows in ((1, pm.n1), (2, pm.n2), (3, pm.n3)):
                if n_rows == 0:
                    continue
                packed = gf[f"c{k}"].index_select(0, slot_ids).reshape(G * n_rows, -1)
                st = unpack_tcq33_l12(packed, T=d_in, k=k).long()
                unpacked_states[k] = st
                U_unit[:, a : a + n_rows, :] = cb_q[k - 1][st].view(G, n_rows, d_in)
                a += n_rows

        Us = U_unit * sc.unsqueeze(-1)
        U = torch.zeros_like(Us)
        U.scatter_(1, rmap.unsqueeze(-1).expand(G, d_out, d_in), Us)
        W_full = fwht_full(U) * signs.unsqueeze(0)

        if pm.k_sparse > 0:
            spb = gf["sp"].index_select(0, slot_ids)
            rows = spb[..., 0:2].contiguous().view(torch.int16).long().view(G, -1) & 0xFFFF
            cols = spb[..., 2:4].contiguous().view(torch.int16).long().view(G, -1) & 0xFFFF
            vals = spb[..., 4:6].contiguous().view(torch.float16).float().view(G, -1)
            W_full.view(G, -1).scatter_add_(1, rows * d_in + cols, vals)

        return W_full, U_unit, rmap, unpacked_states


# ================================================================================================================
# 4. Exact Bilinear Custom Autograd Function (`CompressedMoERoutedFunction`) + `AnchoredRouterModule`
# ================================================================================================================
class CompressedMoERoutedFunction(torch.autograd.Function):
    """Exact forward + exact analytical backward through compressed SwiGLU MoE experts in 16-expert Tensor-Core
    chunks (`~256 MiB` temp VRAM), propagating exact gradients to:
      1) `x_flat` (upstream residual stream for global 45-layer teacher KL backprop),
      2) `top_w_st` (`ST-Top8` router weights -> `AnchoredRouterLoRA` + `delta_gate_b`),
      3) `cb_param` (`2 KiB` `HYB` `[512, 2]` codebooks or `TCQ33` `[3, 4096]` tables), and
      4) `grad_scale_accum` (in-place FP16 row scales via `HostPagedScaleAdamW`)."""

    @staticmethod
    def forward(
        ctx,
        x_flat: torch.Tensor,               # [N, D] fp32
        top_w_st: torch.Tensor,             # [N, K] fp32
        top_idx: torch.Tensor,              # [N, K] int64
        cb_param: torch.Tensor,             # Layer codebook parameter
        handle: CompressedLayerHandle,
        grad_scale_accum: Dict[str, torch.Tensor],
        swiglu_limit: float = 10.0,
        expert_group: int = 16,
    ) -> torch.Tensor:
        dev = x_flat.device
        N, D = x_flat.shape
        K = top_idx.shape[1]

        uniq, inv = torch.unique(top_idx, sorted=True, return_inverse=True)
        inv_f = inv.reshape(-1)
        order = torch.argsort(inv_f, stable=True)
        counts = torch.bincount(inv_f, minlength=int(uniq.numel())).tolist()
        tok_all = order // K
        w_all = top_w_st.reshape(-1)[order]

        starts = [0]
        for c in counts:
            starts.append(starts[-1] + c)

        assert handle._gpu_cache is not None
        row_of = handle._gpu_cache["row_of"]
        slot_of_u = row_of[uniq] if row_of is not None else uniq
        if bool((slot_of_u < 0).any()):
            bad = uniq[slot_of_u < 0].cpu().tolist()
            raise RuntimeError(f"Layer {handle.layer_id}: routed to pruned experts {bad} without slab rows!")

        sc_live = {p: handle.scale_opt.current_scale_fp32(p, dev) for p in ("g", "u", "d")}
        y_out = torch.zeros((N, D), dtype=torch.float32, device=dev)

        U_cnt = int(uniq.numel())
        for g0 in range(0, U_cnt, expert_group):
            g1 = min(U_cnt, g0 + expert_group)
            sl = slot_of_u[g0:g1]
            Wg, _, _, _ = handle.decode_expert_group_unit_and_full("g", sl, cb_param, sc_live["g"], dev)
            Wu, _, _, _ = handle.decode_expert_group_unit_and_full("u", sl, cb_param, sc_live["u"], dev)
            Wd, _, _, _ = handle.decode_expert_group_unit_and_full("d", sl, cb_param, sc_live["d"], dev)

            for j in range(g1 - g0):
                a, b = starts[g0 + j], starts[g0 + j + 1]
                if b == a:
                    continue
                t = tok_all[a:b]
                xt = x_flat.index_select(0, t)
                g_pre = (xt.bfloat16() @ Wg[j].bfloat16().T).float()
                u_pre = (xt.bfloat16() @ Wu[j].bfloat16().T).float()
                z = F.silu(g_pre.clamp(max=swiglu_limit)) * u_pre.clamp(-swiglu_limit, swiglu_limit)
                d_vec = (z.bfloat16() @ Wd[j].bfloat16().T).float()
                y_out.index_add_(0, t, d_vec * w_all[a:b].unsqueeze(-1))

            del Wg, Wu, Wd

        ctx.save_for_backward(x_flat, top_w_st, top_idx, cb_param)
        ctx.handle = handle
        ctx.grad_scale_accum = grad_scale_accum
        ctx.swiglu_limit = float(swiglu_limit)
        ctx.expert_group = int(expert_group)
        return y_out

    @staticmethod
    @torch.no_grad()
    def backward(ctx, grad_y: torch.Tensor):
        x_flat, top_w_st, top_idx, cb_param = ctx.saved_tensors
        handle: CompressedLayerHandle = ctx.handle
        grad_scale_accum: Dict[str, torch.Tensor] = ctx.grad_scale_accum
        lim = ctx.swiglu_limit
        expert_group = ctx.expert_group

        dev = x_flat.device
        N, D = x_flat.shape
        K = top_idx.shape[1]
        gy = grad_y.float()

        uniq, inv = torch.unique(top_idx, sorted=True, return_inverse=True)
        inv_f = inv.reshape(-1)
        order = torch.argsort(inv_f, stable=True)
        counts = torch.bincount(inv_f, minlength=int(uniq.numel())).tolist()
        tok_all = order // K
        pair_pos_all = order  # index into flattened [N * K]
        w_all = top_w_st.reshape(-1)[order]

        starts = [0]
        for c in counts:
            starts.append(starts[-1] + c)

        assert handle._gpu_cache is not None
        row_of = handle._gpu_cache["row_of"]
        slot_of_u = row_of[uniq] if row_of is not None else uniq
        signs_gpu = handle._gpu_cache["signs"]

        sc_live = {p: handle.scale_opt.current_scale_fp32(p, dev) for p in ("g", "u", "d")}
        grad_x = torch.zeros_like(x_flat, dtype=torch.float32)
        grad_w_flat = torch.zeros((N * K,), dtype=torch.float32, device=dev)
        grad_cb = torch.zeros_like(cb_param, dtype=torch.float32)

        idx_all, sx_all, sy_all = _hyb_hash_tables(device=dev) if handle.fmt == "qtip_hyb" else (None, None, None)

        def _accum_proj_grads(
            p_key: str,
            j_idx: int,
            slot_id: int,
            grad_W: torch.Tensor,         # [d_out, d_in] fp32
            U_unit_grp: torch.Tensor,     # [G, d_out, d_in] fp32
            rmap_grp: torch.Tensor,       # [G, d_out] int64
            st_dict: Dict[int, torch.Tensor],
            G_len: int,
        ) -> None:
            pm = handle.projs[p_key]
            d_out, d_in = pm.d_out, pm.d_in
            rmap_j = rmap_grp[j_idx]
            U_unit_j = U_unit_grp[j_idx]
            sc_j = sc_live[p_key][slot_id]  # [d_out]

            # Orthonormal self-adjoint FWHT maps original-domain weight grad into rotated stored row order
            grad_U = fwht_full(grad_W * signs_gpu[p_key].unsqueeze(0))
            grad_Us = grad_U.index_select(0, rmap_j)  # [d_out, d_in]

            # 1) Exact per-row scale gradient: dL/ds_r = sum_c grad_Us[r, c] * U_unit[r, c]
            grad_scale_accum[p_key][slot_id].add_((grad_Us * U_unit_j).sum(dim=-1))

            # 2) Exact codebook gradient: dL/dU_unit = grad_Us * sc_j[:, None]
            grad_U_unit = grad_Us * sc_j.unsqueeze(-1)
            if handle.fmt == "qtip_hyb":
                p_idx = {"g": 0, "u": 1, "d": 2}[p_key]
                for k, a, b in ((1, 0, pm.n1), (2, pm.n1, d_out)):
                    if b <= a:
                        continue
                    if pm.tile_mode == "2d_16x16":
                        nt, nb = (b - a) // 16, d_in // 16
                        st_grp = st_dict[k].view(G_len, nt * nb * 128)
                        st_j = st_grp[j_idx]
                        g_tiles = (
                            grad_U_unit[a:b, :]
                            .reshape(nt, 16, nb, 16)
                            .permute(0, 2, 1, 3)
                            .reshape(-1, 2)
                        )
                    else:
                        st_grp = st_dict[k].view(G_len, (b - a) * (d_in // 2))
                        st_j = st_grp[j_idx]
                        g_tiles = grad_U_unit[a:b, :].reshape(-1, 2)

                    b_idx = idx_all[st_j]
                    grad_cb[p_idx, k - 1, :, 0].index_add_(0, b_idx, g_tiles[:, 0] * sx_all[st_j])
                    grad_cb[p_idx, k - 1, :, 1].index_add_(0, b_idx, g_tiles[:, 1] * sy_all[st_j])
            else:
                a = 0
                for k, n_rows in ((1, pm.n1), (2, pm.n2), (3, pm.n3)):
                    if n_rows == 0:
                        continue
                    st_grp = st_dict[k].view(G_len, n_rows * d_in)
                    st_j = st_grp[j_idx]
                    grad_cb[k - 1].index_add_(0, st_j, grad_U_unit[a : a + n_rows, :].reshape(-1))
                    a += n_rows

        U_cnt = int(uniq.numel())
        for g0 in range(0, U_cnt, expert_group):
            g1 = min(U_cnt, g0 + expert_group)
            G_len = g1 - g0
            sl = slot_of_u[g0:g1]

            Wg, Ug_u, rmap_g, st_g = handle.decode_expert_group_unit_and_full("g", sl, cb_param, sc_live["g"], dev)
            Wu, Uu_u, rmap_u, st_u = handle.decode_expert_group_unit_and_full("u", sl, cb_param, sc_live["u"], dev)
            Wd, Ud_u, rmap_d, st_d = handle.decode_expert_group_unit_and_full("d", sl, cb_param, sc_live["d"], dev)

            for j in range(G_len):
                a, b = starts[g0 + j], starts[g0 + j + 1]
                if b == a:
                    continue
                slot_id = int(sl[j].item())
                t = tok_all[a:b]
                xt = x_flat.index_select(0, t)
                gyt = gy.index_select(0, t)
                wj = w_all[a:b]

                # Recompute local expert activations (exact match to forward)
                g_pre = (xt.bfloat16() @ Wg[j].bfloat16().T).float()
                u_pre = (xt.bfloat16() @ Wu[j].bfloat16().T).float()
                g_c = g_pre.clamp(max=lim)
                u_c = u_pre.clamp(-lim, lim)
                silu_g = F.silu(g_c)
                z = silu_g * u_c
                d_vec = (z.bfloat16() @ Wd[j].bfloat16().T).float()

                # 1) Router ST-Top8 weight gradient
                grad_w_flat.index_copy_(0, pair_pos_all[a:b], (gyt * d_vec).sum(dim=-1))

                # 2) Down-proj weight & activation gradients
                grad_d_vec = gyt * wj.unsqueeze(-1)
                grad_Wd = (grad_d_vec.bfloat16().T @ z.bfloat16()).float()
                grad_z = (grad_d_vec.bfloat16() @ Wd[j].bfloat16()).float()
                _accum_proj_grads("d", j, slot_id, grad_Wd, Ud_u, rmap_d, st_d, G_len)

                # 3) Clamped SwiGLU backward -> Gate & Up weight & input gradients
                grad_g_c = silu_backward(g_c, grad_z * u_c)
                grad_g_pre = torch.where(g_pre <= lim, grad_g_c, torch.zeros_like(grad_g_c))
                grad_u_pre = torch.where((u_pre >= -lim) & (u_pre <= lim), grad_z * silu_g, torch.zeros_like(grad_z))

                grad_Wg = (grad_g_pre.bfloat16().T @ xt.bfloat16()).float()
                grad_Wu = (grad_u_pre.bfloat16().T @ xt.bfloat16()).float()
                _accum_proj_grads("g", j, slot_id, grad_Wg, Ug_u, rmap_g, st_g, G_len)
                _accum_proj_grads("u", j, slot_id, grad_Wu, Uu_u, rmap_u, st_u, G_len)

                dx = (grad_g_pre.bfloat16() @ Wg[j].bfloat16() + grad_u_pre.bfloat16() @ Wu[j].bfloat16()).float()
                grad_x.index_add_(0, t, dx)

            del Wg, Wu, Wd, Ug_u, Uu_u, Ud_u

        return grad_x, grad_w_flat.view(N, K), None, grad_cb, None, None, None, None


@torch.no_grad()
def moe_forward_dense_nograd(
    x_flat: torch.Tensor, top_w: torch.Tensor, top_idx: torch.Tensor, dense: Dict[str, torch.Tensor],
    row_of: Optional[torch.Tensor], layer_id: int, swiglu_limit: float = 10.0,
) -> torch.Tensor:
    return moe_forward_dense(x_flat, top_w, top_idx, dense, row_of, layer_id, swiglu_limit)


def moe_forward_dense(
    x_flat: torch.Tensor, top_w: torch.Tensor, top_idx: torch.Tensor, dense: Dict[str, torch.Tensor],
    row_of: Optional[torch.Tensor], layer_id: int, swiglu_limit: float = 10.0,
    log_scale: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """[R3.9 addition, 2026-11-16] `CompressedMoERoutedFunction.forward` with the layer's experts already decoded
    (`CompressedLayerHandle.decode_all_dense_bf16`). Same token grouping, same bf16 matmuls, same clamp.
    [R3.9 fix 12, 2026-11-17] Plain torch ops, so autograd gives dL/dx and dL/d(top_w) when called with gradients
    enabled (the decoded experts are constants). `log_scale` = {p: zeros [E_kept, d_out] requiring grad}: each
    projection's output rows are multiplied by exp(log_scale[p][slot]) (= 1), so `log_scale[p].grad` is exactly
    dL/d(log row scale) of the live scales."""
    dev = x_flat.device
    N, D = x_flat.shape
    K = top_idx.shape[1]
    uniq, inv = torch.unique(top_idx, sorted=True, return_inverse=True)
    inv_f = inv.reshape(-1)
    order = torch.argsort(inv_f, stable=True)
    counts = torch.bincount(inv_f, minlength=int(uniq.numel())).tolist()
    tok_all = order // K
    w_all = top_w.reshape(-1)[order]
    slot_of_u = row_of[uniq] if row_of is not None else uniq
    if bool((slot_of_u < 0).any()):
        bad = uniq[slot_of_u < 0].cpu().tolist()
        raise RuntimeError(f"Layer {layer_id}: routed to pruned experts {bad} without slab rows!")
    slots = slot_of_u.tolist()
    y_out = torch.zeros((N, D), dtype=torch.float32, device=dev)
    xb = x_flat.bfloat16()
    a = 0
    for j, c in enumerate(counts):
        b = a + c
        if c:
            t = tok_all[a:b]
            xt = xb.index_select(0, t)
            sl = slots[j]
            g_pre = (xt @ dense["g"][sl].T).float()
            u_pre = (xt @ dense["u"][sl].T).float()
            if log_scale is not None:
                g_pre = g_pre * torch.exp(log_scale["g"][sl])
                u_pre = u_pre * torch.exp(log_scale["u"][sl])
            z = F.silu(g_pre.clamp(max=swiglu_limit)) * u_pre.clamp(-swiglu_limit, swiglu_limit)
            d_vec = (z.bfloat16() @ dense["d"][sl].T).float()
            if log_scale is not None:
                d_vec = d_vec * torch.exp(log_scale["d"][sl])
            y_out.index_add_(0, t, d_vec * w_all[a:b].unsqueeze(-1))
        a = b
    return y_out


class AnchoredRouterLoRAModule(nn.Module):
    """Exact GLM-5.3-Flash Sigmoid + Bias Top-8 Router augmented with zero-initialized LoRA + trainable alive-expert
    `e_score_correction_bias` offset + Straight-Through selection surrogate (`ST-Top8`).
    Merges with 0.0 ULP forward change into (`gate_w`, `gate_b`) at export (`0 B` serving overhead)."""

    def __init__(
        self,
        gate_w0: torch.Tensor,          # [288, 4096] fp32
        gate_b0: torch.Tensor,          # [288] fp32 (unmasked base bias)
        pruned_ids: List[int],
        top_k: int = 8,
        routed_scaling_factor: float = 2.5,
        rank: int = 16,
        alpha: float = 32.0,
        st_tau: float = 0.25,
        full_delta: bool = False,
    ):
        super().__init__()
        n_exp, d_model = gate_w0.shape
        self.n_experts = int(n_exp)
        self.d_model = int(d_model)
        self.top_k = int(top_k)
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.rank = int(rank)
        self.scale = float(alpha) / max(int(rank), 1)
        self.st_tau = float(st_tau)

        self.register_buffer("gate_w0", gate_w0.detach().float().clone())
        self.register_buffer("gate_b0", gate_b0.detach().float().clone())

        alive_mask = torch.ones(n_exp, dtype=torch.bool, device=gate_w0.device)
        if pruned_ids:
            alive_mask[torch.tensor(sorted(set(pruned_ids)), dtype=torch.long, device=gate_w0.device)] = False
        self.register_buffer("alive_mask", alive_mask)

        self.lora_A = nn.Parameter(torch.empty((rank, d_model), dtype=torch.float32, device=gate_w0.device))
        self.lora_B = nn.Parameter(torch.zeros((n_exp, rank), dtype=torch.float32, device=gate_w0.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.delta_gate_b = nn.Parameter(torch.zeros((n_exp,), dtype=torch.float32, device=gate_w0.device))
        # [R3.9 fix 12, 2026-11-17] full_delta: train the dense gate matrix itself (zero-initialised additive delta),
        # as the published router-only distillation recipes do (0xSero GLM-5.2 Router-KD, arXiv 2603.02217), instead of
        # the rank-16 LoRA. The LoRA parameters stay at their initial values (B = 0) and are not given to the optimizer.
        self.full_delta = bool(full_delta)
        self.delta_w = (
            nn.Parameter(torch.zeros((n_exp, d_model), dtype=torch.float32, device=gate_w0.device)) if self.full_delta else None
        )

    def _delta(self) -> torch.Tensor:
        dw = self.scale * (self.lora_B @ self.lora_A)
        return dw + self.delta_w if self.delta_w is not None else dw

    def effective_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        w_eff = self.gate_w0 + self._delta()
        b_raw = self.gate_b0 + self.delta_gate_b * self.alive_mask.float()
        b_eff = torch.where(self.alive_mask, b_raw, torch.full_like(b_raw, float("-inf")))
        return w_eff, b_eff

    def forward(self, x_flat: torch.Tensor, straight_through: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        xf = x_flat.float()
        if self.delta_w is not None:
            logits = F.linear(xf, self.gate_w0 + self.delta_w)
        else:
            logits = F.linear(xf, self.gate_w0) + self.scale * F.linear(F.linear(xf, self.lora_A), self.lora_B)
        scores = torch.sigmoid(logits)
        b_raw = self.gate_b0 + self.delta_gate_b * self.alive_mask.float()
        b_masked = torch.where(self.alive_mask, b_raw, torch.full_like(b_raw, float("-inf")))
        choices = scores + b_masked.unsqueeze(0)

        sel_choices, top_idx = torch.topk(choices, k=self.top_k, dim=-1, sorted=False)
        top_scores = torch.gather(scores, 1, top_idx)
        top_w = (top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-20)) * self.routed_scaling_factor

        if straight_through and self.training and torch.is_grad_enabled():
            p_st = torch.softmax(sel_choices / self.st_tau, dim=-1) * self.routed_scaling_factor
            top_w = top_w + (p_st - p_st.detach())
        return top_w, top_idx

    def anchor_penalty(self) -> torch.Tensor:
        dw = self._delta()
        db = self.delta_gate_b * self.alive_mask.float()
        return dw.pow(2).mean() + db.pow(2).mean()


@contextlib.contextmanager
def _differentiable_backbone_scope(bb, FB_mod=None) -> Iterator[None]:
    """Disables raw non-autograd NVRTC decode shortcuts (`mhc_prepare`, `mhc_merge`, `fp8_gemv`) inside
    `torch.enable_grad()` windows so PyTorch autograd flows cleanly through every mHC Sinkhorn / KDA / DSA op."""
    prev_uk = getattr(bb, "use_kernels", False)
    prev_fm = FB_mod._FUSED_MERGE[0] if (FB_mod is not None and hasattr(FB_mod, "_FUSED_MERGE")) else None
    bb.use_kernels = False
    if prev_fm is not None:
        FB_mod._FUSED_MERGE[0] = False
    try:
        yield
    finally:
        bb.use_kernels = prev_uk
        if prev_fm is not None:
            FB_mod._FUSED_MERGE[0] = prev_fm


# ================================================================================================================
# 5. Exact Top-K + Residual-Tail Teacher KL Divergence (48 MiB Host Cache for 65,536 Tokens, < 0.4 GiB Head VRAM)
# ================================================================================================================
@dataclass
class CompactTeacherTargets:
    """Stores top-K (`K = 128`) exact teacher token ids + log-probabilities + residual tail mass `p_tail` on CPU."""

    top_ids: torch.Tensor      # [N_seq, S, K] int32 (CPU)
    top_logp: torch.Tensor     # [N_seq, S, K] float16 (CPU)
    tail_prob: torch.Tensor    # [N_seq, S] float32 (CPU)


@torch.no_grad()
def build_compact_teacher_targets_from_logits_fn(
    logits_fn,
    n_seqs: int,
    seq_len: int,
    top_k: int = 128,
    tok_chunk: int = 256,
) -> CompactTeacherTargets:
    top_ids = torch.empty((n_seqs, seq_len, top_k), dtype=torch.int32, device="cpu")
    top_logp = torch.empty((n_seqs, seq_len, top_k), dtype=torch.float16, device="cpu")
    tail_prob = torch.empty((n_seqs, seq_len), dtype=torch.float32, device="cpu")

    for i in range(n_seqs):
        logits_seq = logits_fn(i)  # [S, V] on GPU
        for s0 in range(0, seq_len, tok_chunk):
            s1 = min(s0 + tok_chunk, seq_len)
            lg = logits_seq[s0:s1].float()
            logp = F.log_softmax(lg, dim=-1)
            vals, idx = torch.topk(logp, k=min(top_k, logp.shape[-1]), dim=-1)
            p_top = vals.exp().sum(dim=-1)
            p_tail = (1.0 - p_top).clamp_min(0.0)
            top_ids[i, s0:s1] = idx.to(torch.int32).cpu()
            top_logp[i, s0:s1] = vals.to(torch.float16).cpu()
            tail_prob[i, s0:s1] = p_tail.cpu()
        del logits_seq
    return CompactTeacherTargets(top_ids=top_ids, top_logp=top_logp, tail_prob=tail_prob)


def compute_chunked_head_kl_and_grad_h(
    bb,
    h_final_batch: torch.Tensor,       # [B, S, hc, D] fp32 on GPU (detached; grad computed wrt this tensor!)
    targets: CompactTeacherTargets,
    seq_indices: List[int],
    min_pos: int = 10,
    total_scored_tokens: int = 1,
    tok_chunk: int = 256,
    need_grad: bool = True,
    valid_lens: Optional[List[int]] = None,
    acc_valid: Optional[dict] = None,
) -> Tuple[float, int, torch.Tensor]:
    """Streams `final_logits` in `tok_chunk`-token slices so `[B * S, V]` (`151,552` vocab) never materializes at once
    in VRAM (`< 0.35 GiB` peak!), returning `(kl_sum_nats, top1_matches, grad_h_final [B, S, hc, D])`."""
    dev = h_final_batch.device
    B, S, hc, D = h_final_batch.shape
    grad_h = torch.zeros_like(h_final_batch, dtype=torch.float32)
    kl_sum = 0.0
    top1_hits = 0

    for b_loc, seq_id in enumerate(seq_indices):
        if S <= min_pos:
            continue
        t_ids_cpu = targets.top_ids[seq_id]
        t_logp_cpu = targets.top_logp[seq_id]
        t_tail_cpu = targets.tail_prob[seq_id]

        for s0 in range(min_pos, S, tok_chunk):
            s1 = min(s0 + tok_chunk, S)
            with torch.set_grad_enabled(need_grad):
                h_slice = h_final_batch[b_loc : b_loc + 1, s0:s1].detach().clone().requires_grad_(need_grad)
                logits, _ = bb.final_logits(h_slice, last_only=False)
                if hasattr(bb, "expand_logits") and getattr(bb, "vocab_ids", None) is not None:
                    logits = bb.expand_logits(logits)
                log_q = F.log_softmax(logits[0].float(), dim=-1)  # [C, V]

                idx_d = t_ids_cpu[s0:s1].to(device=dev, dtype=torch.long)
                log_p = t_logp_cpu[s0:s1].to(device=dev, dtype=torch.float32)
                p_top = log_p.exp()
                p_tail = t_tail_cpu[s0:s1].to(device=dev, dtype=torch.float32)

                log_q_top = torch.gather(log_q, 1, idx_d)
                kl_top = (p_top * (log_p - log_q_top)).sum(dim=-1)

                q_top_sum = log_q_top.exp().sum(dim=-1)
                q_tail = (1.0 - q_top_sum).clamp_min(1e-12)
                kl_tail = torch.where(
                    p_tail > 1e-9,
                    p_tail * (torch.log(p_tail.clamp_min(1e-30)) - torch.log(q_tail)),
                    torch.zeros_like(p_tail),
                )
                kl_tok = kl_top + kl_tail
                loss_scaled = kl_tok.sum() / float(max(total_scored_tokens, 1))
                if need_grad:
                    loss_scaled.backward()

            if need_grad:
                grad_h[b_loc : b_loc + 1, s0:s1].copy_(h_slice.grad.detach())
            kl_sum += float(kl_tok.detach().sum().item())
            pred_top1 = torch.argmax(log_q.detach(), dim=-1)
            hit = pred_top1 == idx_d[:, 0]
            top1_hits += int(hit.sum().item())
            if valid_lens is not None and acc_valid is not None:
                n_ok = max(0, min(int(valid_lens[seq_id]), s1) - s0)
                if n_ok > 0:
                    acc_valid["kl"] += float(kl_tok.detach()[:n_ok].sum().item())
                    acc_valid["top1"] += int(hit[:n_ok].sum().item())
                    acc_valid["n"] += n_ok

    return kl_sum, top1_hits, grad_h


# ================================================================================================================
# 6. Layer-Major Windowed Global-KL Trainer (11 Windows x 4 Layers, Fits 32 GiB VRAM + 96 GB DDR5 RAM)
# ================================================================================================================
class Stage2BGlobalKLTrainer:
    """Executes Layer-Major Global-KL co-adaptation across all MoE layers on a single 32 GiB GPU."""

    def __init__(
        self,
        bb,
        handles: Dict[int, CompressedLayerHandle],
        routers: Dict[int, AnchoredRouterLoRAModule],
        n_layers: int,
        window_size: int = 4,
        FB_mod=None,
        lr_router_lora: float = 2e-4,
        lr_router_bias: float = 6e-5,
        lr_codebook: float = 1.2e-4,
        lr_scale: float = 4e-4,
        router_anchor_wd: float = 1e-3,
        cb_anchor_wd: float = 5e-4,
        scale_anchor_wd: float = 2e-3,
        min_pos: int = 10,
        device: torch.device = torch.device("cuda"),
        adam_beta2: float = 0.995,
    ):
        self.bb = bb
        self.handles = handles
        self.routers = nn.ModuleDict({str(l): r.to(device) for l, r in routers.items()})
        self.n_layers = int(n_layers)
        self.window_size = int(window_size)
        self.FB_mod = FB_mod
        self.router_anchor_wd = float(router_anchor_wd)
        self.cb_anchor_wd = float(cb_anchor_wd)
        self.min_pos = int(min_pos)
        self.device = device
        self.fast_forward = True  # [R3.9 addition] decode each layer once in the no-grad passes
        # [R3.9 fix 12] dense reverse path (see `_reverse_moe_layer_dense`): experts are constants there, so the LUTs
        # get no gradient; the row scales are trained only if `train_scales`.
        self.dense_reverse = False
        self.train_scales = True
        self.capture_grads: Optional[dict] = None  # test hook: fill with the accumulated gradients and do not step

        for h in self.handles.values():
            h.cb_param.data = h.cb_param.data.to(device)
            h.scale_opt.lr = float(lr_scale)
            h.scale_opt.anchor_wd = float(scale_anchor_wd)

        # Build window partitions: e.g. [0..3], [4..7], ..., [40..44]
        self.windows: List[Tuple[int, int]] = []
        l0 = 0
        while l0 < self.n_layers:
            l1 = min(l0 + self.window_size, self.n_layers)
            if self.n_layers - l1 == 1:
                l1 = self.n_layers  # fold single trailing layer 44 into final window [40..44]
            self.windows.append((l0, l1))
            l0 = l1

        # Build lightweight per-window AdamW optimizers for GPU-resident Router LoRA + Bias + 2 KiB HYB LUTs
        self.win_opts: List[torch.optim.AdamW] = []
        for l_start, l_end in self.windows:
            groups = []
            lora_p, bias_p, cb_p = [], [], []
            for l in range(l_start, l_end):
                if str(l) in self.routers:
                    rmod: AnchoredRouterLoRAModule = self.routers[str(l)]  # type: ignore
                    if rmod.delta_w is not None:
                        lora_p.append(rmod.delta_w)  # full gate matrix instead of the LoRA factors
                    else:
                        lora_p.extend([rmod.lora_A, rmod.lora_B])
                    bias_p.append(rmod.delta_gate_b)
                if l in self.handles:
                    cb_p.append(self.handles[l].cb_param)
            if lora_p:
                groups.append({"params": lora_p, "lr": lr_router_lora, "weight_decay": 0.0})
            if bias_p:
                groups.append({"params": bias_p, "lr": lr_router_bias, "weight_decay": 0.0})
            if cb_p:
                groups.append({"params": cb_p, "lr": lr_codebook, "weight_decay": 0.0})
            self.win_opts.append(torch.optim.AdamW(groups, betas=(0.9, float(adam_beta2))) if groups else None)  # type: ignore
        for h in self.handles.values():
            h.scale_opt.beta2 = float(adam_beta2)

    def _forward_single_layer(
        self,
        l: int,
        h: torch.Tensor,
        grad_scale_accum: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
        straight_through: bool = True,
        dense: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        h_att = self.bb.attn_block(l, h, None)
        post, comb, x_ffn = self.bb.ffn_prepare(l, h_att)
        B, S, D = x_ffn.shape
        if not self.bb.cfg.is_moe_layer(l):
            w_l = self.bb.layer(l)
            y = self.bb.swiglu_mlp(x_ffn, w_l["w1"], w_l["w3"], w_l["w2"])
        else:
            x_flat = x_ffn.reshape(-1, D)
            rmod: AnchoredRouterLoRAModule = self.routers[str(l)]  # type: ignore
            handle = self.handles[l]
            top_w, top_idx = rmod(x_flat, straight_through=straight_through)
            gs_l = grad_scale_accum[l] if grad_scale_accum is not None else None
            if dense is not None:  # [R3.9 addition] no-grad passes with the layer decoded once
                assert handle._gpu_cache is not None
                y_routed = moe_forward_dense_nograd(x_flat, top_w, top_idx, dense, handle._gpu_cache["row_of"], l, LIM)
            else:
                y_routed = CompressedMoERoutedFunction.apply(
                    x_flat, top_w, top_idx, handle.cb_param, handle, gs_l, LIM, 16
                )
            y = (y_routed + self.bb.shared_expert(l, x_flat)).view(B, S, D)
        return self.bb.mhc_merge(post, comb, y, h_att)

    @torch.no_grad()
    def evaluate_kl(
        self,
        token_ids: torch.Tensor,          # [N_eval, S] int64 CPU
        targets: CompactTeacherTargets,
        micro_bs: int = 2,
        valid_lens: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        self.routers.eval()
        n_seqs, S = token_ids.shape
        total_toks = n_seqs * max(S - self.min_pos, 0)
        t_eval0 = time.perf_counter()
        gc.collect()
        torch.cuda.empty_cache()

        h_cur: List[torch.Tensor] = []
        for i in range(0, n_seqs, micro_bs):
            ids_b = token_ids[i : i + micro_bs].to(self.device)
            h_cur.append(self.bb.embed_streams(ids_b).to(torch.bfloat16).cpu())

        for l in range(self.n_layers):
            self.bb.layer(l)
            dense = None
            if l in self.handles:
                self.handles[l].to_gpu(self.device)
                if self.fast_forward:
                    dense = self.handles[l].decode_all_dense_bf16(self.handles[l].cb_param, self.device)
            for b_idx in range(len(h_cur)):
                hb = h_cur[b_idx].to(self.device).float()
                hb_out = self._forward_single_layer(l, hb, grad_scale_accum=None, straight_through=False, dense=dense)
                h_cur[b_idx] = hb_out.to(torch.bfloat16).cpu()
            del dense
            # [R3.9 fix 10, 2026-11-17] The three dense blocks have a different size in every layer (E_kept differs), so
            # the allocator could not reuse the previous layer's cached blocks and reserved about 27 GiB for 20.5 GiB in
            # use. After a training step that was enough to spill the evaluation into shared GPU memory (measured: VRAM
            # 31.7 GiB, RAM free 21.9 -> 1.2 GiB, about 10x slower). Release the cache per layer.
            torch.cuda.empty_cache()
            if getattr(self, "verbose_mem", False) and l % 11 == 3:
                print(f"    [eval L{l} at {time.perf_counter() - t_eval0:.0f}s] {_mem_line()}", flush=True)
            if l in self.handles:
                self.handles[l].offload_cpu()
            self.bb.unload_layer(l)

        kl_sum = 0.0
        top1_hits = 0
        seq_cursor = 0
        acc_valid = {"kl": 0.0, "top1": 0, "n": 0}
        for b_idx in range(len(h_cur)):
            hb = h_cur[b_idx].to(self.device).float()
            bsz = hb.shape[0]
            s_ids = list(range(seq_cursor, seq_cursor + bsz))
            seq_cursor += bsz
            ks, th, _ = compute_chunked_head_kl_and_grad_h(
                self.bb, hb, targets, s_ids, min_pos=self.min_pos, total_scored_tokens=total_toks,
                need_grad=False, valid_lens=valid_lens, acc_valid=acc_valid,
            )
            kl_sum += ks
            top1_hits += th

        out = {
            "kl_nats": round(kl_sum / max(total_toks, 1), 6),
            "top1_agree": round(top1_hits / max(total_toks, 1), 5),
            "scored_tokens": total_toks,
            "seconds": round(time.perf_counter() - t_eval0, 1),
        }
        if valid_lens is not None:
            # [R3.9 fix 7] the same metric over real positions only (rows shorter than S are padded with their last
            # token and the last target is copied forward: 32.5 % of the qbench positions)
            out["kl_nats_valid"] = round(acc_valid["kl"] / max(acc_valid["n"], 1), 6)
            out["top1_agree_valid"] = round(acc_valid["top1"] / max(acc_valid["n"], 1), 5)
            out["scored_tokens_valid"] = acc_valid["n"]
        return out

    def _reverse_moe_layer_dense(
        self,
        l: int,
        ckpts: List[torch.Tensor],
        grad_h_host: List[torch.Tensor],
        gs_all: Dict[int, Dict[str, torch.Tensor]],
    ) -> None:
        """[R3.9 fix 12, 2026-11-17] Reverse pass of ONE MoE layer for all sequences of the chunk, with the layer's
        experts decoded once (dense bf16, constants) instead of once per sequence inside `CompressedMoERoutedFunction`.

        The layer is h_att = attn(h_in); (post, comb, x) = ffn_prepare(h_att); y = MoE(x) + shared(x);
        h_out = mhc_merge(post, comb, y, h_att). The attention graph of one 2,048-token sequence takes about 26 GiB
        and the decoded layer about 12.6 GiB, so the two are never on the GPU together:
          1. no grad, per sequence: h_att, post, comb, x (values only);
          2. with grad, all sequences at once: y = MoE(x) + shared(x) on the dense experts (router inside);
          3. with grad, per sequence: h_out = mhc_merge(leaves) -> dL/dy, dL/dpost, dL/dcomb, dL/dh_att (direct);
          4. backward of (2) with dL/dy -> router / bias (/ log-scale) gradients and dL/dx; dense experts freed;
          5. with grad, per sequence: attention + ffn_prepare again, backward from (h_att, post, comb, x) with the
             gradients of (3) and (4) -> dL/dh_in.
        Same mathematics as the one-layer window of the compressed path; the LUTs receive no gradient here.
        Replaces grad_h_host[b] by dL/dh_in in place; log-scale gradients are added to gs_all[l] (as dL/d log s)."""
        dev = self.device
        handle = self.handles[l]
        rmod: AnchoredRouterLoRAModule = self.routers[str(l)]  # type: ignore
        n = len(ckpts)
        with _differentiable_backbone_scope(self.bb, self.FB_mod):
            pre = []
            with torch.no_grad():
                for b in range(n):
                    h_in = ckpts[b].to(dev).float()
                    h_att = self.bb.attn_block(l, h_in, None)
                    post, comb, x_ffn = self.bb.ffn_prepare(l, h_att)
                    pre.append((h_att, post, comb, x_ffn))
                    del h_in
            torch.cuda.empty_cache()
            D = pre[0][3].shape[-1]
            offs = [0]
            for t in pre:
                offs.append(offs[-1] + t[3].shape[0] * t[3].shape[1])
            shapes = [tuple(t[3].shape) for t in pre]
            dense = handle.decode_all_dense_bf16(handle.cb_param, dev)
            assert handle._gpu_cache is not None
            row_of = handle._gpu_cache["row_of"]
            ls = None
            if self.train_scales:
                ls = {
                    p: torch.zeros((handle.n_kept, handle.projs[p].d_out), dtype=torch.float32, device=dev, requires_grad=True)
                    for p in ("g", "u", "d")
                }
            side = []
            with torch.enable_grad():
                x_all = torch.cat([t[3].reshape(-1, D) for t in pre], dim=0).detach().requires_grad_(True)
                top_w, top_idx = rmod(x_all, straight_through=True)
                y_all = moe_forward_dense(x_all, top_w, top_idx, dense, row_of, l, LIM, log_scale=ls)
                y_all = y_all + self.bb.shared_expert(l, x_all)
                gy = torch.empty((offs[-1], D), dtype=torch.float32, device=dev)
                for b in range(n):
                    h_att, post, comb, _ = pre[b]
                    y_leaf = y_all[offs[b] : offs[b + 1]].detach().view(*shapes[b]).requires_grad_(True)
                    ha = h_att.detach().requires_grad_(True)
                    po = post.detach().requires_grad_(True)
                    co = comb.detach().requires_grad_(True)
                    h_out = self.bb.mhc_merge(po, co, y_leaf, ha)
                    gh_up = grad_h_host[b].to(dev).float()
                    g_y, g_ha, g_po, g_co = torch.autograd.grad(h_out, [y_leaf, ha, po, co], grad_outputs=gh_up)
                    gy[offs[b] : offs[b + 1]] = g_y.reshape(-1, D)
                    side.append((g_ha, g_po, g_co))
                    del h_out, gh_up, y_leaf, ha, po, co, g_y
                del pre
                torch.autograd.backward(y_all, grad_tensors=gy)
            g_x = x_all.grad.detach()
            if ls is not None:
                acc = gs_all.setdefault(l, {})
                for p in ("g", "u", "d"):
                    g_cpu = ls[p].grad.detach().cpu()
                    if p in acc:
                        acc[p].add_(g_cpu)
                    else:
                        acc[p] = g_cpu
            del dense, y_all, gy, top_w, top_idx, x_all, ls
            torch.cuda.empty_cache()

            for b in range(n):
                h_in = ckpts[b].to(dev).float().detach().requires_grad_(True)
                g_ha, g_po, g_co = side[b]
                with torch.enable_grad():
                    h_att = self.bb.attn_block(l, h_in, None)
                    post, comb, x_ffn = self.bb.ffn_prepare(l, h_att)
                    torch.autograd.backward(
                        [h_att, post, comb, x_ffn],
                        grad_tensors=[g_ha, g_po, g_co, g_x[offs[b] : offs[b + 1]].view(*shapes[b])],
                    )
                grad_h_host[b] = h_in.grad.detach().to(torch.bfloat16).cpu()
                side[b] = None  # type: ignore
                del h_in, h_att, post, comb, x_ffn, g_ha, g_po, g_co
            del g_x, side

    def _grad_windows(self) -> List[Tuple[int, int]]:
        """[R3.9 fix 9] Layer windows of the reverse (autograd) pass. 0 = the optimizer windows (shipped behaviour)."""
        gw = int(getattr(self, "grad_window", 0) or 0)
        if gw <= 0:
            return list(self.windows)
        return [(l0, min(l0 + gw, self.n_layers)) for l0 in range(0, self.n_layers, gw)]

    def run_macro_step(
        self,
        token_ids: torch.Tensor,          # [N_macro, S] int64 CPU
        targets: CompactTeacherTargets,
        micro_bs: int = 1,
    ) -> Dict[str, float]:
        """Executes ONE complete Layer-Major Global-KL macro-step.

        [R3.9 fix 9, 2026-11-17] Measured on the real model (RTX 5090 32 GiB, 125.7 GiB RAM): the reverse pass of ONE
        layer at S=2048 needs about 20-26 GiB of VRAM, so the shipped 4-layer window spills tens of GiB into shared GPU
        memory and never finishes its first sequence; and the host checkpoints of 32 sequences at 1-layer windows would
        need 88 GiB. The step is therefore split: `grad_window` sets the layers per autograd window (memory only),
        `chunk_seqs` sets how many sequences go through forward / head / reverse at a time, and the gradients of all
        chunks are ACCUMULATED and the optimizers stepped once at the end, in the shipped optimizer windows
        (`self.windows`, same parameter groups, same per-window clip). In the shipped code every window was stepped
        right after its own backward, and nothing computed later in the macro-step read the stepped parameters, so
        accumulate-then-step gives the same update. With grad_window = 0 and chunk_seqs = 0 the pass order is the
        shipped one."""
        self.routers.train()
        n_seqs, S = token_ids.shape
        total_toks = n_seqs * max(S - self.min_pos, 0)
        t0 = time.perf_counter()
        gwins = self._grad_windows()
        chunk = int(getattr(self, "chunk_seqs", 0) or 0) or n_seqs
        verbose = bool(getattr(self, "verbose_mem", False))

        for opt in self.win_opts:
            if opt is not None:
                opt.zero_grad(set_to_none=True)
        gs_all: Dict[int, Dict[str, torch.Tensor]] = {}   # layer -> p -> CPU fp32 dL/ds summed over the chunks
        kl_sum = 0.0
        top1_hits = 0
        t_a_tot = t_b_tot = t_c_tot = 0.0
        peak_a = peak_c = 0.0
        win_starts = {w0: idx for idx, (w0, _) in enumerate(gwins)}

        for c0 in range(0, n_seqs, chunk):
            c1 = min(c0 + chunk, n_seqs)
            tc0 = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(self.device)

            # ------------------------------------------------------------------------------------
            # Phase A: layer-by-layer forward of this chunk -> host BF16 checkpoints at window starts
            # ------------------------------------------------------------------------------------
            ckpt_host: Dict[int, List[torch.Tensor]] = {}
            h_stream: List[torch.Tensor] = []
            with torch.no_grad():
                for i in range(c0, c1, micro_bs):
                    ids_b = token_ids[i : min(i + micro_bs, c1)].to(self.device)
                    h_stream.append(self.bb.embed_streams(ids_b).to(torch.bfloat16).cpu())

                for l in range(self.n_layers):
                    if l in win_starts:
                        ckpt_host[win_starts[l]] = [t.clone() for t in h_stream]

                    self.bb.layer(l)
                    dense = None
                    if l in self.handles:
                        self.handles[l].to_gpu(self.device)
                        if self.fast_forward:
                            dense = self.handles[l].decode_all_dense_bf16(self.handles[l].cb_param, self.device)
                    for b_idx in range(len(h_stream)):
                        hb = h_stream[b_idx].to(self.device).float()
                        hb = self._forward_single_layer(l, hb, grad_scale_accum=None, straight_through=False, dense=dense)
                        h_stream[b_idx] = hb.to(torch.bfloat16).cpu()
                    del dense
                    torch.cuda.empty_cache()  # [R3.9 fix 10] see evaluate_kl
                    if l in self.handles:
                        self.handles[l].offload_cpu()
                    self.bb.unload_layer(l)
            t_a = time.perf_counter()
            peak_a = max(peak_a, torch.cuda.max_memory_allocated(self.device) / (1024.0 ** 3))
            if verbose and c0 == 0:
                ck_gib = sum(t.numel() * t.element_size() for lst in ckpt_host.values() for t in lst) / 2**30
                print(f"    [A done {t_a - tc0:.0f}s] {len(ckpt_host)} window checkpoints {ck_gib:.1f} GiB, h {tuple(h_stream[0].shape)} "
                      f"{h_stream[0].dtype} | {_mem_line()}", flush=True)

            # ------------------------------------------------------------------------------------
            # Phase B: exact teacher-KL at the head and upstream gradient dL / dh_{n_layers}
            # ------------------------------------------------------------------------------------
            grad_h_host: List[torch.Tensor] = []
            seq_cursor = c0
            for b_idx in range(len(h_stream)):
                hb = h_stream[b_idx].to(self.device).float()
                bsz = hb.shape[0]
                s_ids = list(range(seq_cursor, seq_cursor + bsz))
                seq_cursor += bsz
                ks, th, gh = compute_chunked_head_kl_and_grad_h(
                    self.bb, hb, targets, s_ids, min_pos=self.min_pos, total_scored_tokens=total_toks
                )
                kl_sum += ks
                top1_hits += th
                grad_h_host.append(gh.to(torch.bfloat16).cpu())
                del hb, gh
            del h_stream
            t_b = time.perf_counter()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

            # ------------------------------------------------------------------------------------
            # Phase C: reverse sweep over the autograd windows; gradients accumulate, nothing is stepped
            # ------------------------------------------------------------------------------------
            for w_idx in range(len(gwins) - 1, -1, -1):
                l_start, l_end = gwins[w_idx]
                if self.dense_reverse and (l_end - l_start) == 1 and l_start in self.handles:
                    # [R3.9 fix 12] one MoE layer, all sequences of the chunk, experts decoded once
                    self.bb.layer(l_start)
                    self.handles[l_start].to_gpu(self.device)
                    self._reverse_moe_layer_dense(l_start, ckpt_host[w_idx], grad_h_host, gs_all)
                    self.handles[l_start].offload_cpu()
                    self.bb.unload_layer(l_start)
                    pk_w = torch.cuda.max_memory_allocated(self.device) / (1024.0 ** 3)
                    peak_c = max(peak_c, pk_w)
                    del ckpt_host[w_idx]
                    torch.cuda.empty_cache()
                    if verbose and c0 == 0:
                        print(f"    [C dense L{l_start} done at {time.perf_counter() - t_b:.0f}s, peak {pk_w:.1f} GiB] "
                              f"{_mem_line()}", flush=True)
                    torch.cuda.reset_peak_memory_stats(self.device)
                    continue
                grad_scale_win: Dict[int, Dict[str, torch.Tensor]] = {}
                for l in range(l_start, l_end):
                    self.bb.layer(l)
                    if l in self.handles:
                        h_obj = self.handles[l]
                        h_obj.to_gpu(self.device)
                        grad_scale_win[l] = {
                            p: torch.zeros((h_obj.n_kept, h_obj.projs[p].d_out), dtype=torch.float32, device=self.device)
                            for p in ("g", "u", "d")
                        }

                with _differentiable_backbone_scope(self.bb, self.FB_mod):
                    for b_idx in range(len(grad_h_host)):
                        h_in = ckpt_host[w_idx][b_idx].to(self.device).float().detach().requires_grad_(True)
                        gh_up = grad_h_host[b_idx].to(self.device).float()

                        with torch.enable_grad():
                            h_curr = h_in
                            for l in range(l_start, l_end):
                                h_curr = self._forward_single_layer(
                                    l, h_curr, grad_scale_accum=grad_scale_win, straight_through=True
                                )
                            torch.autograd.backward(h_curr, grad_tensors=gh_up)

                        grad_h_host[b_idx] = h_in.grad.detach().to(torch.bfloat16).cpu()
                        del h_in, gh_up, h_curr

                for l in range(l_start, l_end):
                    if l in self.handles:
                        acc = gs_all.setdefault(l, {})
                        for p in ("g", "u", "d"):
                            g_cpu = grad_scale_win[l][p].cpu()
                            if p in acc:
                                acc[p].add_(g_cpu)
                            else:
                                acc[p] = g_cpu
                        self.handles[l].offload_cpu()
                    self.bb.unload_layer(l)

                pk_w = torch.cuda.max_memory_allocated(self.device) / (1024.0 ** 3)
                peak_c = max(peak_c, pk_w)
                del ckpt_host[w_idx], grad_scale_win
                torch.cuda.empty_cache()
                if verbose and c0 == 0:
                    print(f"    [C window {w_idx} L{l_start}..{l_end - 1} done at {time.perf_counter() - t_b:.0f}s, peak {pk_w:.1f} GiB] "
                          f"{_mem_line()}", flush=True)
                torch.cuda.reset_peak_memory_stats(self.device)

            del grad_h_host, ckpt_host
            t_c = time.perf_counter()
            t_a_tot += t_a - tc0
            t_b_tot += t_b - t_a
            t_c_tot += t_c - t_b
            if chunk < n_seqs:
                print(f"    [chunk {c0}:{c1}] A {t_a - tc0:.0f}s B {t_b - t_a:.0f}s C {t_c - t_b:.0f}s | running train KL "
                      f"{kl_sum / max((c1) * max(S - self.min_pos, 0), 1):.5f} | {_mem_line()}", flush=True)

        if self.capture_grads is not None:  # [R3.9 fix 12] test hook: hand out the accumulated gradients, no step
            cap = self.capture_grads
            cap["router"] = {n: p.grad.detach().clone() for n, p in self.routers.named_parameters() if p.grad is not None}
            cap["scale_log"] = {}
            for l, d in gs_all.items():
                so = self.handles[l].scale_opt
                cap["scale_log"][l] = {
                    p: (d[p].clone() if self.dense_reverse else d[p] * (so.s0[p].float() * torch.exp(so.la[p].float())))
                    for p in ("g", "u", "d")
                }
            cap["train_kl_nats"] = kl_sum / max(total_toks, 1)
            for opt in self.win_opts:
                if opt is not None:
                    opt.zero_grad(set_to_none=True)
            return {"train_kl_nats": round(kl_sum / max(total_toks, 1), 6), "seconds": round(time.perf_counter() - t0, 2)}

        # ----------------------------------------------------------------------------------------
        # One optimizer step per shipped optimizer window (anchor penalties, per-window clip), then the row scales
        # ----------------------------------------------------------------------------------------
        for w_idx in range(len(self.windows) - 1, -1, -1):
            l_start, l_end = self.windows[w_idx]
            opt = self.win_opts[w_idx]
            if opt is not None:
                with torch.enable_grad():
                    reg = torch.tensor(0.0, device=self.device)
                    for l in range(l_start, l_end):
                        if str(l) in self.routers and self.router_anchor_wd > 0:
                            rmod: AnchoredRouterLoRAModule = self.routers[str(l)]  # type: ignore
                            reg = reg + 0.5 * self.router_anchor_wd * rmod.anchor_penalty()
                        if l in self.handles and self.cb_anchor_wd > 0:
                            ho = self.handles[l]
                            cb0 = ho.cb_init_cpu.to(self.device)
                            reg = reg + 0.5 * self.cb_anchor_wd * (ho.cb_param - cb0).pow(2).mean()
                    if reg.requires_grad:
                        reg.backward()

                win_params = [p for pg in opt.param_groups for p in pg["params"]]
                torch.nn.utils.clip_grad_norm_(win_params, max_norm=1.0)
                opt.step()

            for l in range(l_start, l_end):
                if l in self.handles and l in gs_all:
                    ho = self.handles[l]
                    # [R3.9 fix 6, 2026-11-16] The shipped code clamped the HYB LUT to >= 1e-4 here ("magnitudes are
                    # non-negative"). The builder's LUTs are k-means centroids and DO contain negative entries (25 of
                    # 1,024 at k=2 in layer_03 of the real archive, minimum -0.153): the clamp rewrote those weights at
                    # the first step, outside the loss and the anchor. Removed.
                    updated_fp16 = ho.scale_opt.step_and_pack_fp16(gs_all[l], self.device, grad_is_log=self.dense_reverse)
                    ho.commit_scales_in_place(updated_fp16)
        del gs_all

        print(
            f"  [PHASES] A forward {t_a_tot:.0f}s (peak {peak_a:.1f} GiB) | B head {t_b_tot:.0f}s | "
            f"C reverse windows {t_c_tot:.0f}s (peak {peak_c:.1f} GiB)",
            flush=True,
        )
        return {
            "train_kl_nats": round(kl_sum / max(total_toks, 1), 6),
            "train_top1_agree": round(top1_hits / max(total_toks, 1), 5),
            "peak_vram_gib": round(max(peak_a, peak_c), 3),
            "seconds": round(time.perf_counter() - t0, 2),
        }

    @torch.no_grad()
    def diagnostics(self) -> Dict[str, float]:
        r_rel, cb_rel, sc_rel = [], [], []
        for l_str, rmod in self.routers.items():
            w_eff, _ = rmod.effective_weights()  # type: ignore
            dw = (w_eff - rmod.gate_w0).norm() / rmod.gate_w0.norm().clamp_min(1e-12)
            r_rel.append(float(dw.item()))
        for l, ho in self.handles.items():
            cb0 = ho.cb_init_cpu.to(ho.cb_param.device)
            dcb = (ho.cb_param.detach() - cb0).norm() / cb0.norm().clamp_min(1e-12)
            cb_rel.append(float(dcb.item()))
            for p in ("g", "u", "d"):
                sc_rel.append(float(ho.scale_opt.la[p].abs().mean().item()))
        return {
            "router_rel_delta": round(float(np.mean(r_rel)) if r_rel else 0.0, 6),
            "codebook_rel_delta": round(float(np.mean(cb_rel)) if cb_rel else 0.0, 6),
            "mean_abs_log_scale": round(float(np.mean(sc_rel)) if sc_rel else 0.0, 6),
        }

    @torch.no_grad()
    def save_adapter_state(self, path: str, meta: dict) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "meta": meta,
            "routers": self.routers.state_dict(),
            "codebooks": {l: ho.cb_param.detach().to(torch.float16).cpu() for l, ho in self.handles.items()},
            "scales_opt": {l: ho.scale_opt.state_dict() for l, ho in self.handles.items()},
        }
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)

    @torch.no_grad()
    def load_adapter_state(self, path: str) -> dict:
        """[R3.9 addition, 2026-11-17] Inverse of `save_adapter_state` (the shipped script could save a state but not
        read one back, so a tuned result could only be exported by the process that trained it). Restores routers,
        LUTs (stored fp16) and the log-scales, and rewrites the row scales as fp16(s0 * exp(la)). Optimizer moments are
        not in the state: training continued from here restarts its moments."""
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.routers.load_state_dict(payload["routers"])
        for l, ho in self.handles.items():
            ho.cb_param.data.copy_(payload["codebooks"][l].to(device=ho.cb_param.device, dtype=ho.cb_param.dtype))
            ho.scale_opt.load_state_dict(payload["scales_opt"][l])
            ho.commit_scales_in_place(
                {p: (ho.scale_opt.s0[p].float() * torch.exp(ho.scale_opt.la[p].float())).to(torch.float16) for p in ("g", "u", "d")}
            )
        return dict(payload.get("meta") or {})

    @torch.no_grad()
    def export_merged_checkpoint(self, export_dir: str, TR_mod=None) -> Dict[str, int]:
        """Exports deploy-ready zero-overhead artifacts into `export_dir`:
          1) Merged BF16/FP32 router weights (`gate_w` & `e_score_correction_bias` with `-inf` on pruned ids),
          2) Updated `2 KiB` `HYB` FP16 LUTs (or `TCQ33` FP16 tables), and
          3) Byte-locked expert slabs with updated `off_scale` FP16 row scales written in place."""
        os.makedirs(export_dir, exist_ok=True)
        merged_routers = {}
        for l_str, rmod in self.routers.items():
            l = int(l_str)
            w_eff, b_eff = rmod.effective_weights()  # type: ignore
            merged_routers[f"model.language_model.layers.{l}.mlp.gate.weight"] = w_eff.detach().to(torch.bfloat16).cpu()
            merged_routers[f"model.language_model.layers.{l}.mlp.gate.e_score_correction_bias"] = b_eff.detach().float().cpu()
        router_out_path = os.path.join(export_dir, "merged_routers_bf16.pt")
        torch.save(merged_routers, router_out_path)

        total_archive_bytes = 0
        for l, ho in self.handles.items():
            if ho.fmt == "qtip_hyb":
                cb16 = ho.cb_param.detach().to(torch.float16).cpu()  # [3, 2, 512, 2]
                hyb_luts = {
                    p: {1: cb16[i, 0].contiguous(), 2: cb16[i, 1].contiguous()}
                    for i, p in enumerate(("g", "u", "d"))
                }
                out_p = os.path.join(export_dir, f"layer_{l:02d}_qtip_hyb.pt")
                meta = ho.tcq_npz_meta or {}
                blob = {
                    "format": meta.get("format", "qtip_hyb_r39"),
                    "rule": meta.get("rule", "A"),
                    "layer_id": l,
                    "tile_mode": ho.projs["g"].tile_mode,
                    "signs": ho.signs_cpu,
                    "hyb_luts": hyb_luts,
                    "fields": ho.fields_cpu,
                    "perm": meta.get("perm"),
                    "n_experts_total": meta.get("n_experts_total", 288),
                    "meta": dict(meta.get("meta") or {}, stage2b_finetuned=True),
                }
                tmp_p = out_p + ".tmp"
                with open(tmp_p, "wb") as fh_out:
                    torch.save(blob, fh_out)
                    fh_out.flush()
                    os.fsync(fh_out.fileno())
                os.replace(tmp_p, out_p)
                total_archive_bytes += os.path.getsize(out_p)
            else:
                assert TR_mod is not None and ho.tcq_npz_meta is not None
                out_p = os.path.join(export_dir, f"layer_{l:02d}_apex_fold.npz")
                total_archive_bytes += TR_mod.save_layer(
                    path=out_p,
                    rule=ho.tcq_npz_meta["rule"],
                    layer_id=l,
                    signs=ho.signs_cpu,
                    tables=ho.cb_param.detach().to(torch.float16).cpu(),
                    slab=ho.tcq_npz_meta["slab_u8"],
                    meta=ho.tcq_npz_meta["meta"],
                    perm=ho.tcq_npz_meta["perm"],
                )
        # Copy/hardlink companion serving files (MTP Layer 45, fold_manifest.json, prefix caches, super_experts)
        any_meta = next((ho.tcq_npz_meta for ho in self.handles.values() if ho.tcq_npz_meta and "src_path" in ho.tcq_npz_meta), None)
        src_arch_dir = os.path.dirname(os.path.abspath(any_meta["src_path"])) if any_meta else ""
        if src_arch_dir and os.path.isdir(src_arch_dir) and os.path.normcase(src_arch_dir) != os.path.normcase(os.path.abspath(export_dir)):
            for companion in ("layer_45_qtip_hyb.pt", "layer_45_apex_fold.npz", "exact_prefix_cache_prefix2.pt", "exact_prefix_cache_prefix10.pt"):
                sp_c = os.path.join(src_arch_dir, companion)
                dp_c = os.path.join(export_dir, companion)
                if os.path.exists(sp_c) and not os.path.exists(dp_c):
                    try:
                        os.link(sp_c, dp_c)
                    except OSError:
                        shutil.copy2(sp_c, dp_c)
            se_src = os.path.join(src_arch_dir, "super_experts")
            se_dst = os.path.join(export_dir, "super_experts")
            if os.path.isdir(se_src) and not os.path.exists(se_dst):
                os.makedirs(se_dst, exist_ok=True)
                for fn_se in os.listdir(se_src):
                    s_f, d_f = os.path.join(se_src, fn_se), os.path.join(se_dst, fn_se)
                    if os.path.isfile(s_f):
                        try:
                            os.link(s_f, d_f)
                        except OSError:
                            shutil.copy2(s_f, d_f)
            man_src = os.path.join(src_arch_dir, "fold_manifest.json")
            if os.path.exists(man_src):
                man_obj = json.load(open(man_src, "r", encoding="utf-8"))
                man_obj["stage2b_finetuned"] = True
                man_obj["merged_routers"] = "merged_routers_bf16.pt"
                with open(os.path.join(export_dir, "fold_manifest.json"), "w", encoding="utf-8") as fh_m:
                    json.dump(man_obj, fh_m, indent=2)
        return {"layers_exported": len(self.handles), "total_archive_bytes": total_archive_bytes}


# ================================================================================================================
# 7. Standalone GPU Self-Test (--selftest: Verifies Full 6-Layer mHC + ST-TopK + HYB + off_scale Loop in ~8s)
# ================================================================================================================
class _MiniGLM53Backbone(nn.Module):
    """Architecture-faithful miniature GLM-5.3-Flash backbone (4-stream Sinkhorn mHC + dense L0-L1 + MoE L2-L5)
    allowing full end-to-end verification of `Stage2BGlobalKLTrainer` on any CUDA GPU without 90 GB shards."""

    def __init__(self, vocab: int = 512, D: int = 64, Di: int = 64, n_layers: int = 6, n_experts: int = 8, top_k: int = 2, dev="cuda"):
        super().__init__()
        self.device = torch.device(dev)
        self.use_kernels = False
        self.hc = 4
        self.D = D
        self.eps = 1e-6
        self.cfg = argparse.Namespace(
            num_hidden_layers=n_layers,
            hidden_size=D,
            n_routed_experts=n_experts,
            num_experts_per_tok=top_k,
            routed_scaling_factor=2.0,
            swiglu_limit=LIM,
            hc_eps=1e-4,
            hc_sinkhorn_iters=3,
            is_moe_layer=lambda l: l >= 2,
        )
        gen = torch.Generator(device=dev).manual_seed(2026)
        self.embed_w = torch.randn((vocab, D), generator=gen, device=dev) / math.sqrt(D)
        self.head_w = torch.randn((vocab, D), generator=gen, device=dev) / math.sqrt(D)
        self.final_norm = torch.ones((D,), device=dev)
        self._store: Dict[int, dict] = {}
        for l in range(n_layers):
            d = {
                "in_norm": torch.ones(D, device=dev),
                "post_norm": torch.ones(D, device=dev),
                "hc_attn_fn": torch.randn((24, 4 * D), generator=gen, device=dev) * 0.05,
                "hc_attn_base": torch.zeros(24, device=dev),
                "hc_attn_scale": torch.ones(3, device=dev) * 0.5,
                "hc_ffn_fn": torch.randn((24, 4 * D), generator=gen, device=dev) * 0.05,
                "hc_ffn_base": torch.zeros(24, device=dev),
                "hc_ffn_scale": torch.ones(3, device=dev) * 0.5,
                "wq": torch.randn((D, D), generator=gen, device=dev) / math.sqrt(D),
                "wk": torch.randn((D, D), generator=gen, device=dev) / math.sqrt(D),
                "wv": torch.randn((D, D), generator=gen, device=dev) / math.sqrt(D),
                "wo": torch.randn((D, D), generator=gen, device=dev) / math.sqrt(D),
            }
            if l < 2:
                d["w1"] = torch.randn((Di, D), generator=gen, device=dev) / math.sqrt(D)
                d["w3"] = torch.randn((Di, D), generator=gen, device=dev) / math.sqrt(D)
                d["w2"] = torch.randn((D, Di), generator=gen, device=dev) / math.sqrt(Di)
            else:
                d["gate_w"] = torch.randn((n_experts, D), generator=gen, device=dev) / math.sqrt(D)
                d["gate_b"] = torch.zeros(n_experts, device=dev)
                d["sh_w1"] = torch.randn((Di, D), generator=gen, device=dev) * (0.3 / math.sqrt(D))
                d["sh_w3"] = torch.randn((Di, D), generator=gen, device=dev) * (0.3 / math.sqrt(D))
                d["sh_w2"] = torch.randn((D, Di), generator=gen, device=dev) * (0.3 / math.sqrt(Di))
            self._store[l] = d

    def layer(self, l: int) -> dict:
        return self._store[l]

    def unload_layer(self, l: int) -> None:
        pass

    def rmsnorm(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps))

    def mhc_prepare(self, h: torch.Tensor, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor):
        hc = 4
        eps_hc = 1e-4
        flat = h.flatten(2).float()
        flat = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.eps)
        mix = F.linear(flat, fn)
        pre_w, post_w, comb_w = mix.split([hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = base.split([hc, hc, hc * hc])
        pre = torch.sigmoid(pre_w * scale[0] + pre_b) + eps_hc
        post = 2.0 * torch.sigmoid(post_w * scale[1] + post_b)
        comb = torch.softmax(comb_w.view(*comb_w.shape[:-1], hc, hc) * scale[2] + comb_b.view(hc, hc), dim=-1) + eps_hc
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps_hc)
        for _ in range(2):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + eps_hc)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + eps_hc)
        x_in = (pre.unsqueeze(-1) * h.float()).sum(dim=2)
        return post, comb, x_in

    @staticmethod
    def mhc_merge(post: torch.Tensor, comb: torch.Tensor, y: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
        return post.unsqueeze(-1) * y.float().unsqueeze(-2) + torch.matmul(comb.transpose(-1, -2), res.float())

    def attn_block(self, l: int, h: torch.Tensor, cache=None) -> torch.Tensor:
        w = self._store[l]
        post, comb, x_in = self.mhc_prepare(h, w["hc_attn_fn"], w["hc_attn_base"], w["hc_attn_scale"])
        xn = self.rmsnorm(x_in, w["in_norm"])
        q, k, v = F.linear(xn, w["wq"]), F.linear(xn, w["wk"]), F.linear(xn, w["wv"])
        S = xn.shape[1]
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.D)
        causal = torch.tril(torch.ones((S, S), dtype=torch.bool, device=xn.device))
        scores = scores.masked_fill(~causal.unsqueeze(0), -1e9)
        y = F.linear(torch.matmul(torch.softmax(scores, dim=-1), v), w["wo"])
        return self.mhc_merge(post, comb, y, h)

    def ffn_prepare(self, l: int, h: torch.Tensor):
        w = self._store[l]
        post, comb, xf = self.mhc_prepare(h, w["hc_ffn_fn"], w["hc_ffn_base"], w["hc_ffn_scale"])
        return post, comb, self.rmsnorm(xf, w["post_norm"])

    def swiglu_mlp(self, x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
        return F.linear(F.silu(F.linear(x, w1).clamp(max=LIM)) * F.linear(x, w3).clamp(-LIM, LIM), w2)

    def shared_expert(self, l: int, x_flat: torch.Tensor) -> torch.Tensor:
        w = self._store[l]
        return self.swiglu_mlp(x_flat, w["sh_w1"], w["sh_w3"], w["sh_w2"])

    def embed_streams(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embed_w[ids].unsqueeze(2).expand(-1, -1, 4, -1).contiguous()

    def final_logits(self, h: torch.Tensor, last_only: bool = False):
        coll = h.mean(dim=2)
        if last_only:
            coll = coll[:, -1:, :]
        return F.linear(self.rmsnorm(coll, self.final_norm), self.head_w), coll


def run_selftest() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("--selftest requires a CUDA GPU.")
    dev = torch.device("cuda")
    print(f"[SELFTEST] Running end-to-end Stage-2B Layer-Major Global-KL verification on {torch.cuda.get_device_name(0)}...", flush=True)

    mini_bb = _MiniGLM53Backbone(vocab=512, D=64, Di=64, n_layers=6, n_experts=8, top_k=2, dev="cuda")
    gen_cpu = torch.Generator(device="cpu").manual_seed(101)
    gen_gpu = torch.Generator(device=dev).manual_seed(202)

    # Create exact FP32 teacher experts for MoE layers 2..5, then prune 2/8 experts (25% prune!) and quantize to QTIP HYB 2d_16x16
    teacher_experts = {}
    handles: Dict[int, CompressedLayerHandle] = {}
    routers: Dict[int, AnchoredRouterLoRAModule] = {}

    for l in range(2, 6):
        Wg = torch.randn((8, 64, 64), generator=gen_gpu, device=dev) / 8.0
        Wu = torch.randn((8, 64, 64), generator=gen_gpu, device=dev) / 8.0
        Wd = torch.randn((8, 64, 64), generator=gen_gpu, device=dev) / 8.0
        teacher_experts[l] = (Wg, Wu, Wd)

        # Prune experts [1, 5] -> keep 6 experts in compact slab
        kept_ids = torch.tensor([0, 2, 3, 4, 6, 7], dtype=torch.long)
        pruned_ids = [1, 5]
        row_of = torch.full((8,), -1, dtype=torch.long)
        row_of[kept_ids] = torch.arange(6, dtype=torch.long)

        cb_init = (torch.rand((3, 2, 512, 2), generator=gen_cpu, dtype=torch.float32) * 1.2 + 0.1).to(torch.float16)
        signs = {p: (torch.randint(0, 2, (64,), generator=gen_cpu, dtype=torch.int8) * 2 - 1) for p in ("g", "u", "d")}
        projs = {p: ProjMeta(p, 64, 64, 16, 48, 0, 4, "2d_16x16") for p in ("g", "u", "d")}
        fields = {}
        for p_idx, (p, W_ex) in enumerate((("g", Wg), ("u", Wu), ("d", Wd))):
            Wk = W_ex[kept_ids.to(dev)]
            rmap = torch.arange(64, dtype=torch.int16).unsqueeze(0).expand(6, 64).contiguous()
            sc = Wk.pow(2).mean(-1).sqrt().cpu().to(torch.float16).contiguous()
            c1 = torch.randint(0, 256, (6, 1, 4, 32), generator=gen_cpu, dtype=torch.uint8)
            c2 = torch.randint(0, 256, (6, 3, 4, 64), generator=gen_cpu, dtype=torch.uint8)
            sp = torch.zeros((6, 4, 6), dtype=torch.uint8)
            fields[p] = {"map": rmap, "scale": sc, "c1": c1, "c2": c2, "sp": sp}

        handles[l] = CompressedLayerHandle(
            layer_id=l,
            fmt="qtip_hyb",
            projs=projs,
            signs_cpu=signs,
            fields_cpu=fields,
            codebook_init_fp16=cb_init,
            row_of_cpu=row_of,
            missing_ids=pruned_ids,
            tcq_npz_meta={"perm": kept_ids, "n_experts_total": 8},
        )
        wl = mini_bb.layer(l)
        routers[l] = AnchoredRouterLoRAModule(
            gate_w0=wl["gate_w"],
            gate_b0=wl["gate_b"],
            pruned_ids=pruned_ids,
            top_k=2,
            routed_scaling_factor=2.0,
            rank=8,
            alpha=16.0,
            full_delta=os.environ.get("R39_S2B_TEST_FULL", "0") == "1",
        )

    # Generate exact unpruned teacher targets on train (8 seqs x 48 tok) and held-out qbench (4 seqs x 48 tok)
    train_ids = torch.randint(0, 512, (8, 48), generator=gen_cpu, dtype=torch.long)
    eval_ids = torch.randint(0, 512, (4, 48), generator=gen_cpu, dtype=torch.long)

    @torch.no_grad()
    def exact_teacher_logits(ids_1d: torch.Tensor) -> torch.Tensor:
        h = mini_bb.embed_streams(ids_1d.unsqueeze(0).to(dev))
        for l in range(6):
            h_att = mini_bb.attn_block(l, h)
            post, comb, xf = mini_bb.ffn_prepare(l, h_att)
            if l < 2:
                wl = mini_bb.layer(l)
                y = mini_bb.swiglu_mlp(xf, wl["w1"], wl["w3"], wl["w2"])
            else:
                wl = mini_bb.layer(l)
                x_fl = xf.reshape(-1, 64)
                scores = torch.sigmoid(F.linear(x_fl, wl["gate_w"]))
                _, tidx = torch.topk(scores + wl["gate_b"].unsqueeze(0), k=2, dim=-1)
                tw = torch.gather(scores, 1, tidx)
                tw = (tw / (tw.sum(-1, keepdim=True) + 1e-20)) * 2.0
                Wg, Wu, Wd = teacher_experts[l]
                yr = torch.zeros_like(x_fl)
                for e in range(8):
                    tk, sl = torch.where(tidx == e)
                    if tk.numel():
                        z = F.silu((x_fl[tk] @ Wg[e].T).clamp(max=LIM)) * (x_fl[tk] @ Wu[e].T).clamp(-LIM, LIM)
                        yr.index_add_(0, tk, (z @ Wd[e].T) * tw[tk, sl].unsqueeze(-1))
                y = (yr + mini_bb.shared_expert(l, x_fl)).view_as(xf)
            h = mini_bb.mhc_merge(post, comb, y, h_att)
        lg, _ = mini_bb.final_logits(h, last_only=False)
        return lg[0]

    train_targets = build_compact_teacher_targets_from_logits_fn(
        lambda i: exact_teacher_logits(train_ids[i]), n_seqs=8, seq_len=48, top_k=64
    )
    eval_targets = build_compact_teacher_targets_from_logits_fn(
        lambda i: exact_teacher_logits(eval_ids[i]), n_seqs=4, seq_len=48, top_k=64
    )

    trainer = Stage2BGlobalKLTrainer(
        bb=mini_bb,
        handles=handles,
        routers=routers,
        n_layers=6,
        window_size=2,  # 3 windows ([0..1], [2..3], [4..5]) exercising full cross-window gradient handoff!
        lr_router_lora=3e-3,
        lr_router_bias=1e-3,
        lr_codebook=3e-3,
        lr_scale=4e-3,
        min_pos=4,
        device=dev,
    )

    # Measure initial export byte size before any updates
    tmp_dir_0 = "/tmp/r39_selftest_before"
    tmp_dir_1 = "/tmp/r39_selftest_after"
    shutil.rmtree(tmp_dir_0, ignore_errors=True)
    shutil.rmtree(tmp_dir_1, ignore_errors=True)
    b0_info = trainer.export_merged_checkpoint(tmp_dir_0)

    ev0 = trainer.evaluate_kl(train_ids, train_targets, micro_bs=4)
    print(f"  [INIT] Pre-FT Global Teacher KL = {ev0['kl_nats']:.5f} nats | top1 = {ev0['top1_agree']:.4f}", flush=True)

    # [R3.9 fix 9] test hooks for the chunked / 1-layer-window path (defaults keep the shipped pass order)
    trainer.grad_window = int(os.environ.get("R39_S2B_TEST_GW", "0"))
    trainer.chunk_seqs = int(os.environ.get("R39_S2B_TEST_CHUNK", "0"))
    if trainer.grad_window or trainer.chunk_seqs:
        print(f"  [TEST] grad_window={trainer.grad_window} chunk_seqs={trainer.chunk_seqs} (train seqs {train_ids.shape[0]})", flush=True)

    # [R3.9 fix 12] gradient comparison: compressed one-layer windows vs the dense reverse path, same parameters
    if os.environ.get("R39_S2B_TEST_GRADCMP", "0") == "1":
        caps = {}
        for mode in (False, True):
            trainer.dense_reverse = mode
            trainer.grad_window = 1
            trainer.chunk_seqs = 4
            trainer.capture_grads = {}
            trainer.run_macro_step(train_ids, train_targets, micro_bs=2)
            caps[mode] = trainer.capture_grads
        trainer.capture_grads = None

        def _cmp(a: torch.Tensor, b: torch.Tensor) -> str:
            a, b = a.double().flatten().cpu(), b.double().flatten().cpu()
            na, nb_ = float(a.norm()), float(b.norm())
            if na == 0.0 and nb_ == 0.0:
                return "both zero"
            cos = float((a @ b) / max(na * nb_, 1e-300))
            return f"|a|={na:.4e} |b|={nb_:.4e} rel_err={float((a - b).norm()) / max(na, 1e-300):.3e} cos={cos:.6f}"

        print(f"  [GRADCMP] train KL compressed {caps[False]['train_kl_nats']:.6f} | dense {caps[True]['train_kl_nats']:.6f}", flush=True)
        worst = 1.0
        for name in sorted(caps[False]["router"]):
            if name not in caps[True]["router"]:
                print(f"  [GRADCMP] router {name}: missing in the dense path", flush=True)
                worst = 0.0
                continue
            line = _cmp(caps[False]["router"][name], caps[True]["router"][name])
            print(f"  [GRADCMP] router {name}: {line}", flush=True)
            if "cos=" in line:
                worst = min(worst, float(line.split("cos=")[1]))
        for l in sorted(caps[False]["scale_log"]):
            for p in ("g", "u", "d"):
                line = _cmp(caps[False]["scale_log"][l][p], caps[True]["scale_log"][l][p])
                print(f"  [GRADCMP] log-scale L{l} {p}: {line}", flush=True)
                if "cos=" in line:
                    worst = min(worst, float(line.split("cos=")[1]))
        print(f"[GRADCMP {'PASSED' if worst > 0.999 else 'FAILED'}] worst cosine {worst:.6f}", flush=True)
        return

    if os.environ.get("R39_S2B_TEST_DENSE", "0") == "1":
        trainer.dense_reverse = True
        trainer.grad_window = 1
        trainer.chunk_seqs = trainer.chunk_seqs or 4
        trainer.train_scales = os.environ.get("R39_S2B_TEST_FREEZE_SCALES", "0") != "1"
        print(f"  [TEST] dense reverse path, train_scales={trainer.train_scales}", flush=True)

    for sw in range(1, 5):
        st = trainer.run_macro_step(train_ids, train_targets, micro_bs=2)
        diag = trainer.diagnostics()
        print(
            f"  [SWEEP {sw}] KL = {st['train_kl_nats']:.5f} nats | peak_VRAM = {st['peak_vram_gib']:.3f} GiB | "
            f"d_router={diag['router_rel_delta']:.5f} d_hyb={diag['codebook_rel_delta']:.5f} d_scale={diag['mean_abs_log_scale']:.5f}",
            flush=True,
        )

    ev1 = trainer.evaluate_kl(train_ids, train_targets, micro_bs=4)
    ev_hold = trainer.evaluate_kl(eval_ids, eval_targets, micro_bs=4)
    diag_final = trainer.diagnostics()
    b1_info = trainer.export_merged_checkpoint(tmp_dir_1)
    shutil.rmtree(tmp_dir_0, ignore_errors=True)
    shutil.rmtree(tmp_dir_1, ignore_errors=True)

    assert ev1["kl_nats"] < ev0["kl_nats"], f"Expected KL drop: {ev0['kl_nats']} -> {ev1['kl_nats']}"
    assert diag_final["router_rel_delta"] > 0.0, "Router LoRA did not receive gradient!"
    if not trainer.dense_reverse:
        assert diag_final["codebook_rel_delta"] > 0.0, "HYB 2 KiB LUT did not receive gradient!"
    if trainer.train_scales or not trainer.dense_reverse:
        assert diag_final["mean_abs_log_scale"] > 0.0, "In-place FP16 row scales did not receive gradient!"
    assert b1_info["total_archive_bytes"] == b0_info["total_archive_bytes"], (
        f"Byte-lock violated: {b0_info['total_archive_bytes']} vs {b1_info['total_archive_bytes']}"
    )
    print(
        f"[SELFTEST PASSED] Global Teacher KL dropped {ev0['kl_nats']:.5f} -> {ev1['kl_nats']:.5f} nats "
        f"(held-out KL={ev_hold['kl_nats']:.5f}); exact 0-byte archive invariance verified ({b1_info['total_archive_bytes']} B)!",
        flush=True,
    )


# ================================================================================================================
# 8. Real GLM-5.3-Flash Corpus Packing, Archive Loader & CLI Entrypoint
# ================================================================================================================
def load_real_sequences(tok, ct_path: str, c4_path: str, qb_path: str, n_train: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Packs mixed `cal_trace` + `C4` training sequences and `qbench` held-out sequences (`[N, seq_len]`)."""
    train_seqs: List[List[int]] = []
    n_ct_goal = (2 * n_train) // 3
    if os.path.exists(ct_path):
        rows = json.load(open(ct_path, encoding="utf-8"))["rows"]
        buf: List[int] = []
        for r in rows:
            buf.extend(r["input_ids"] + r["response_ids"])
            while len(buf) >= seq_len and len(train_seqs) < n_ct_goal:
                train_seqs.append(buf[:seq_len])
                buf = buf[seq_len:]
            if len(train_seqs) >= n_ct_goal:
                break

    if len(train_seqs) < n_train and os.path.exists(c4_path):
        gm = tok.gmask_sop() if hasattr(tok, "gmask_sop") else []
        buf = []
        with gzip.open(c4_path, "rt", encoding="utf-8") as fh:
            for line in fh:
                buf.extend(tok.encode_raw(json.loads(line)["text"] + "\n\n"))
                need = seq_len - len(gm)
                if len(buf) >= need:
                    train_seqs.append(gm + buf[:need])
                    buf = []
                    if len(train_seqs) >= n_train:
                        break

    qb_rows = json.load(open(qb_path, encoding="utf-8"))["rows"]
    qb_seqs: List[List[int]] = []
    for r in qb_rows:
        ids = (r["input_ids"] + r["response_ids"])[:seq_len]
        if len(ids) < seq_len:
            ids = ids + [ids[-1]] * (seq_len - len(ids))
        qb_seqs.append(ids)

    return torch.tensor(train_seqs[:n_train], dtype=torch.long), torch.tensor(qb_seqs, dtype=torch.long)


def main() -> None:
    ap = argparse.ArgumentParser(description="Round 3.9 Stage-2B Layer-Major Global-KL Trainer (1x RTX 5090)")
    ap.add_argument("--selftest", action="store_true", help="Run standalone GPU end-to-end verification and exit")
    ap.add_argument("--pkg-dir", default=DEFAULT_PKG, help="Path to claude_r32_apex_fold directory")
    ap.add_argument("--backbone-dir", default=DEFAULT_BB, help="Path to GLM53-backbone-r34h (or GLM53-backbone)")
    ap.add_argument("--archive-dir", default=DEFAULT_ARCH, help="Path to QTIP HYB or TCQ33 compressed expert directory")
    ap.add_argument("--archive-format", default="auto", choices=["auto", "qtip_hyb", "tcq33"], help="Archive format")
    ap.add_argument("--prune-spec", default="", help="Optional Round 3.8/3.9 JSON prune mask (if archive is unpruned)")
    ap.add_argument("--teacher-cache", required=False, default="", help="Path to cached CompactTeacherTargets .pt (created if missing)")
    ap.add_argument("--teacher-archive-dir", default="", help="Unpruned reference archive used to build --teacher-cache if absent")
    ap.add_argument("--qbench-teacher-evx", default="", help="Optional path to _r33_ckpt/evx.pt exact qbench hidden states")
    ap.add_argument("--ct-path", default=DEFAULT_CT, help="Calibration chat trace JSON path")
    ap.add_argument("--c4-path", default=DEFAULT_C4, help="C4 validation json.gz path")
    ap.add_argument("--qb-path", default=DEFAULT_QB, help="qbench_prompts.json path")
    ap.add_argument("--export-dir", default="", help="Output directory for merged routers + updated HYB/off_scale archives")
    ap.add_argument("--save-state", default=os.path.join(HERE, "out_r32", "r39_stage2b_state.pt"), help="Adapter checkpoint path")
    ap.add_argument("--sweeps", type=int, default=6, help="Number of full training sweeps")
    ap.add_argument("--train-seqs", type=int, default=128, help="Total training calibration sequences")
    ap.add_argument("--macro-seqs", type=int, default=32, help="Sequences per Layer-Major Global-KL macro-step (32 -> 22 GiB RAM)")
    ap.add_argument("--seq-len", type=int, default=2048, help="Sequence length")
    ap.add_argument("--micro-bs", type=int, default=1, help="Micro-batch sequences on GPU inside 4-layer window")
    ap.add_argument("--window-size", type=int, default=4, help="Layers per resident VRAM window (default: 4 -> 11 windows)")
    ap.add_argument("--lora-rank", type=int, default=16, help="AnchoredRouterLoRA rank")
    ap.add_argument("--lora-alpha", type=float, default=32.0, help="AnchoredRouterLoRA alpha")
    ap.add_argument("--lr-router", type=float, default=2e-4, help="Learning rate for AnchoredRouterLoRA")
    ap.add_argument("--lr-bias", type=float, default=6e-5, help="Learning rate for e_score_correction_bias")
    ap.add_argument("--lr-codebook", type=float, default=1.2e-4, help="Learning rate for 2 KiB HYB LUTs / TCQ tables")
    ap.add_argument("--lr-scale", type=float, default=4e-4, help="Learning rate for in-place FP16 row scales")
    # [R3.9 additions, 2026-11-16]
    ap.add_argument("--eval-every-macro", type=int, default=0, help="Also run the qbench eval after every N macro-steps (0 = only per sweep)")
    ap.add_argument("--max-macro-steps", type=int, default=0, help="Stop after this many macro-steps in total (0 = all sweeps)")
    ap.add_argument("--grad-window", type=int, default=0,
                    help="Layers per autograd window of the reverse pass (0 = --window-size, shipped). 1 is the only value that "
                         "fits 32 GiB at S=2048; --window-size still sets the optimizer groups")
    ap.add_argument("--chunk-seqs", type=int, default=0,
                    help="Sequences per forward/reverse pass inside a macro-step; gradients accumulate over the chunks "
                         "(0 = all --macro-seqs at once, shipped). Host RAM for the checkpoints is 64 MiB x chunk x windows")
    ap.add_argument("--load-state", default="", help="Adapter state written by --save-state to start from")
    ap.add_argument("--export-only", action="store_true", help="With --load-state and --export-dir: evaluate, export, exit (no training)")
    ap.add_argument("--no-fast-forward", action="store_true", help="Shipped behaviour: decode every expert once per sequence in the no-grad passes")
    # [R3.9 fix 12, 2026-11-17] published router-distillation setup + dense reverse path
    ap.add_argument("--router-full", action="store_true",
                    help="Train the dense gate matrix (zero-initialised delta) instead of the LoRA; --lr-router is its rate")
    ap.add_argument("--dense-reverse", action="store_true",
                    help="Reverse pass with each MoE layer decoded once per chunk (experts constant: no LUT gradient); "
                         "forces --grad-window 1")
    ap.add_argument("--freeze-scales", action="store_true", help="With --dense-reverse: do not train the row scales")
    ap.add_argument("--adam-beta2", type=float, default=0.995, help="AdamW beta2 (shipped 0.995; published recipes 0.999)")
    ap.add_argument("--skip-macro", type=int, default=0, help="Skip the first N macro-steps of the schedule (resume)")
    ap.add_argument("--patience", type=int, default=0,
                    help="Stop after this many evaluations in a row without a new best (0 = never)")
    ap.add_argument("--probe-scales", type=int, default=0,
                    help="Diagnostic: accumulate the log-scale gradients on 2 x N training sequences, report the "
                         "gradient of one shared scale offset per (layer, projection) and per expert, line-search "
                         "the shared offsets on qbench, keep the best, save the state and exit")
    args = ap.parse_args()

    if args.selftest:
        run_selftest()
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for Stage-2B Global-KL training.")
    dev = torch.device("cuda")

    if os.path.isdir(args.pkg_dir) and args.pkg_dir not in sys.path:
        sys.path.insert(0, args.pkg_dir)

    import glm53_fold_backbone as FB  # type: ignore
    import glm53_fold_tcq_format as TF  # type: ignore
    import glm53_fold_tcq_runtime as TR  # type: ignore
    from glm53_fold_config import GLM53FlashModelConfig  # type: ignore
    from glm53_fold_generate import RealTokenizer, load_prune_spec  # type: ignore

    cfg = GLM53FlashModelConfig()
    tok = RealTokenizer(args.backbone_dir)
    ck_bb = FB.RealGLM53Checkpoint(args.backbone_dir)
    bb = FB.RealGLM53Backbone(ck_bb, cfg=cfg, device="cuda", backbone_mode="fp8")
    bb.load_global()

    extra_prune = load_prune_spec(args.prune_spec, cfg) if args.prune_spec else {}

    # Detect archive format (`layer_03_qtip_hyb.pt` vs `layer_03_apex_fold.npz`)
    fmt = args.archive_format
    if fmt == "auto":
        qtip_probe = os.path.join(args.archive_dir, "layer_03_qtip_hyb.pt")
        fmt = "qtip_hyb" if os.path.exists(qtip_probe) else "tcq33"
    print(f"[STAGE-2B] Archive format='{fmt}' from {args.archive_dir} | Backbone={args.backbone_dir}", flush=True)

    handles: Dict[int, CompressedLayerHandle] = {}
    routers: Dict[int, AnchoredRouterLoRAModule] = {}
    for l in range(cfg.num_hidden_layers):
        if not cfg.is_moe_layer(l):
            continue
        if fmt == "qtip_hyb":
            lp = os.path.join(args.archive_dir, f"layer_{l:02d}_qtip_hyb.pt")
            ho = CompressedLayerHandle.from_qtip_pt(l, lp)
        else:
            lp = os.path.join(args.archive_dir, f"layer_{l:02d}_apex_fold.npz")
            ho = CompressedLayerHandle.from_tcq33_npz(l, lp, TF, TR)
        handles[l] = ho

        wl = bb.layer(l)
        pruned_l = sorted(set(ho.missing_ids) | set(extra_prune.get(l, [])))
        gb0 = bb._gate_b0[l] if l in bb._gate_b0 else wl["gate_b"].clone()
        routers[l] = AnchoredRouterLoRAModule(
            gate_w0=wl["gate_w"],
            gate_b0=gb0,
            pruned_ids=pruned_l,
            top_k=cfg.num_experts_per_tok,
            routed_scaling_factor=cfg.routed_scaling_factor,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            full_delta=args.router_full,
        )
        bb.unload_layer(l)

    train_ids, qb_ids = load_real_sequences(tok, args.ct_path, args.c4_path, args.qb_path, args.train_seqs, args.seq_len)
    print(f"[DATA] Packed {train_ids.shape[0]} train seqs + {qb_ids.shape[0]} qbench seqs (S={args.seq_len})", flush=True)

    # Load or build CompactTeacherTargets (`top_k = 128` + exact tail probability)
    def _to_ctt(obj) -> CompactTeacherTargets:
        if isinstance(obj, CompactTeacherTargets):
            return obj
        return CompactTeacherTargets(top_ids=obj["top_ids"], top_logp=obj["top_logp"], tail_prob=obj["tail_prob"])

    if args.teacher_cache and os.path.exists(args.teacher_cache):
        tc_blob = torch.load(args.teacher_cache, map_location="cpu", weights_only=False)
        train_targets: CompactTeacherTargets = _to_ctt(tc_blob["train"])
        qb_targets: CompactTeacherTargets = _to_ctt(tc_blob["qbench"])
        # [R3.9 fix 7, 2026-11-16] The targets belong to the ids stored in the cache. The shipped code took them only
        # when the shape happened to match and never took the qbench ids; the trainer's own C4 rows lack the
        # [gMASK]<sop> prefix the extractor adds, so any mismatch meant training on ids that do not fit the targets.
        if "train_ids" in tc_blob:
            train_ids = tc_blob["train_ids"].long()[: args.train_seqs]
            assert train_ids.shape[0] <= train_targets.top_ids.shape[0] and train_ids.shape[1] == train_targets.top_ids.shape[1]
        if "qb_ids" in tc_blob:
            assert tc_blob["qb_ids"].shape == qb_ids.shape and bool((tc_blob["qb_ids"].long() == qb_ids).all()), "qbench ids differ from the teacher cache"
        print(f"[TEACHER] Loaded cached Top-128+Tail teacher targets from {args.teacher_cache}", flush=True)
    else:
        t_arch_dir = args.teacher_archive_dir or args.archive_dir
        print(f"[TEACHER] Building Top-128+Tail teacher cache from unpruned reference {t_arch_dir}...", flush=True)

        @torch.no_grad()
        def _stream_ref_hidden(ids_all: torch.Tensor) -> List[torch.Tensor]:
            hs = [bb.embed_streams(ids_all[i : i + 1].to(dev)).to(torch.bfloat16).cpu() for i in range(ids_all.shape[0])]
            for l in range(cfg.num_hidden_layers):
                bb.layer(l)
                rt = None
                if cfg.is_moe_layer(l):
                    npz_p = os.path.join(t_arch_dir, f"layer_{l:02d}_apex_fold.npz")
                    rt = TR.TcqLayerRuntime(TR.TcqArchive(npz_p, mmap_slab=False), dev, slab_in_vram=True)
                for i in range(len(hs)):
                    hi = hs[i].to(dev).float()
                    moe_fn = None if rt is None else (lambda _l, xf: rt.forward_routed_experts(xf, *bb.route(_l, xf), swiglu_limit=LIM).float())
                    hs[i] = bb.layer_forward(l, hi, None, moe_fn=moe_fn).to(torch.bfloat16).cpu()
                del rt
                bb.unload_layer(l)
            return hs

        tr_h = _stream_ref_hidden(train_ids)
        train_targets = build_compact_teacher_targets_from_logits_fn(
            lambda i: bb.final_logits(tr_h[i].to(dev).float(), last_only=False)[0][0],
            n_seqs=train_ids.shape[0],
            seq_len=args.seq_len,
        )
        del tr_h

        r39_qb_t128 = os.path.join(args.archive_dir, "_r39_ckpt", "qbench_teacher_top128.pt")
        if os.path.exists(r39_qb_t128) and not args.qbench_teacher_evx:
            qb_raw = torch.load(r39_qb_t128, map_location="cpu")
            # Pad each variable-length qbench row's top128 cache to args.seq_len
            t_ids_l, t_lp_l, t_pr_l = [], [], []
            for tid_i, tlp_i, tpr_i in zip(qb_raw["top_ids"], qb_raw["top_logp"], qb_raw["tail_prob"]):
                s_i = tid_i.shape[0]
                if s_i < args.seq_len:
                    pad_n = args.seq_len - s_i
                    tid_i = torch.cat([tid_i, tid_i[-1:].expand(pad_n, -1)], dim=0)
                    tlp_i = torch.cat([tlp_i, tlp_i[-1:].expand(pad_n, -1)], dim=0)
                    tpr_i = torch.cat([tpr_i, tpr_i[-1:].expand(pad_n)], dim=0)
                t_ids_l.append(tid_i[: args.seq_len])
                t_lp_l.append(tlp_i[: args.seq_len])
                t_pr_l.append(tpr_i[: args.seq_len])
            qb_targets = CompactTeacherTargets(
                top_ids=torch.stack(t_ids_l, dim=0),
                top_logp=torch.stack(t_lp_l, dim=0),
                tail_prob=torch.stack(t_pr_l, dim=0),
            )
            print(f"[TEACHER] Loaded exact FP8 qbench Top-128 targets from {r39_qb_t128}", flush=True)
        else:
            if args.qbench_teacher_evx and os.path.exists(args.qbench_teacher_evx):
                evx_h = torch.load(args.qbench_teacher_evx, map_location="cpu")
                qb_h = evx_h["h"] if isinstance(evx_h, dict) and "h" in evx_h else evx_h
            else:
                qb_h = _stream_ref_hidden(qb_ids)
            qb_targets = build_compact_teacher_targets_from_logits_fn(
                lambda i: bb.final_logits(qb_h[i][:, : args.seq_len].to(dev).float(), last_only=False)[0][0],
                n_seqs=qb_ids.shape[0],
                seq_len=args.seq_len,
            )
            del qb_h

        if args.teacher_cache:
            os.makedirs(os.path.dirname(os.path.abspath(args.teacher_cache)), exist_ok=True)
            torch.save({"format": "r39_compact_teacher_top128_v1", "train_ids": train_ids, "qb_ids": qb_ids, "train": {"top_ids": train_targets.top_ids, "top_logp": train_targets.top_logp, "tail_prob": train_targets.tail_prob}, "qbench": {"top_ids": qb_targets.top_ids, "top_logp": qb_targets.top_logp, "tail_prob": qb_targets.tail_prob}}, args.teacher_cache)

    trainer = Stage2BGlobalKLTrainer(
        bb=bb,
        handles=handles,
        routers=routers,
        n_layers=cfg.num_hidden_layers,
        window_size=args.window_size,
        FB_mod=FB,
        lr_router_lora=args.lr_router,
        lr_router_bias=args.lr_bias,
        lr_codebook=args.lr_codebook,
        lr_scale=args.lr_scale,
        device=dev,
        adam_beta2=args.adam_beta2,
    )

    trainer.fast_forward = not args.no_fast_forward
    trainer.grad_window = int(args.grad_window)
    trainer.chunk_seqs = int(args.chunk_seqs)
    trainer.verbose_mem = True
    trainer.dense_reverse = bool(args.dense_reverse)
    trainer.train_scales = not args.freeze_scales
    if trainer.dense_reverse and trainer.grad_window != 1:
        print("[CFG] --dense-reverse: grad window set to 1 layer", flush=True)
        trainer.grad_window = 1
    print(f"[CFG] router={'full gate matrix' if args.router_full else 'LoRA r' + str(args.lora_rank)} lr {args.lr_router} | bias lr {args.lr_bias} | "
          f"LUT {'frozen (dense reverse)' if trainer.dense_reverse else 'lr ' + str(args.lr_codebook)} | "
          f"scales {'frozen' if (trainer.dense_reverse and not trainer.train_scales) else 'lr ' + str(args.lr_scale)} | "
          f"beta2 {args.adam_beta2} | reverse {'dense' if trainer.dense_reverse else 'compressed'}", flush=True)
    if args.load_state:
        st_meta = trainer.load_adapter_state(args.load_state)
        print(f"[STATE] Loaded {args.load_state}: sweep {st_meta.get('sweep')}, macro-steps {st_meta.get('macro_steps')}, "
              f"saved qbench KL {(st_meta.get('qbench') or {}).get('kl_nats')}", flush=True)
    # real lengths of the qbench rows (shorter rows are padded to --seq-len with their last token)
    qb_lens = [min(len(r["input_ids"] + r["response_ids"]), args.seq_len) for r in json.load(open(args.qb_path, encoding="utf-8"))["rows"]]
    assert len(qb_lens) == qb_ids.shape[0]

    def _eval_line(tag: str, q: dict, extra: str = "") -> None:
        print(
            f"[EVAL {tag}] qbench KL = {q['kl_nats']:.5f} nats | top1 = {q['top1_agree']:.4f} | real positions only: "
            f"KL = {q['kl_nats_valid']:.5f}, top1 = {q['top1_agree_valid']:.4f} ({q['scored_tokens_valid']} tok) | {q['seconds']:.0f}s{extra}",
            flush=True,
        )

    qb0 = trainer.evaluate_kl(qb_ids, qb_targets, micro_bs=args.micro_bs, valid_lens=qb_lens)
    # [R3.9 fix 8, 2026-11-17] Select the best state on REAL positions. On the real archive the padded metric is
    # 3.837 nats while the real positions give 0.2906: the 19,231 padded positions (32.5 %) average about 11 nats and
    # would decide every "new best".
    def _sel(q: dict) -> float:
        return float(q.get("kl_nats_valid", q["kl_nats"]))

    best_qb_kl = _sel(qb0)
    print(f"[EVAL 0] Initial qbench KL = {qb0['kl_nats']:.5f} nats | top1 = {qb0['top1_agree']:.4f}", flush=True)
    _eval_line("0", qb0)
    if args.export_only:
        assert args.load_state and args.export_dir, "--export-only needs --load-state and --export-dir"
        exp_info = trainer.export_merged_checkpoint(args.export_dir, TR_mod=TR)
        print(f"[EXPORT] {args.export_dir}: {exp_info}", flush=True)
        return
    if args.probe_scales > 0:
        # [R3.9 fix 13, 2026-11-17] Is there a SYSTEMATIC scale error? The per-row scale gradients are noisy, but the
        # gradient of one shared log-offset per (layer, projection) is their sum over all rows of the layer, and per
        # expert the sum over that expert's rows. Two disjoint halves of data tell signal from noise (correlation of
        # the two estimates); a line search on qbench then says what a shared correction is worth.
        n_p = int(args.probe_scales)
        trainer.dense_reverse, trainer.grad_window, trainer.train_scales = True, 1, True
        halves = []
        for hf in range(2):
            idx = torch.arange(hf * n_p, (hf + 1) * n_p)
            sub_t = CompactTeacherTargets(top_ids=train_targets.top_ids[idx], top_logp=train_targets.top_logp[idx],
                                          tail_prob=train_targets.tail_prob[idx])
            trainer.capture_grads = {}
            st_p = trainer.run_macro_step(train_ids[idx], sub_t, micro_bs=args.micro_bs)
            halves.append(trainer.capture_grads["scale_log"])
            print(f"[PROBE] half {hf}: {n_p} sequences, train KL {st_p['train_kl_nats']:.5f}, {st_p['seconds']:.0f}s", flush=True)
        trainer.capture_grads = None
        torch.save({"halves": halves, "n_per_half": n_p}, args.save_state + ".scaleprobe.pt")
        keys = [(l, p) for l in sorted(halves[0]) for p in ("g", "u", "d")]
        lay = [torch.tensor([float(h[l][p].sum()) for (l, p) in keys], dtype=torch.float64) for h in halves]
        exp = [torch.cat([h[l][p].sum(dim=1).double() for (l, p) in keys]) for h in halves]
        row = [torch.cat([h[l][p].flatten().double() for (l, p) in keys[:9]]) for h in halves]   # first 3 layers only

        def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
            return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-300))

        print(f"[PROBE] cosine between the two halves: per (layer, proj) {_corr(lay[0], lay[1]):.4f} | per expert "
              f"{_corr(exp[0], exp[1]):.4f} | per row (first 3 layers) {_corr(row[0], row[1]):.4f}", flush=True)
        g_lay = 0.5 * (lay[0] + lay[1])
        g_exp = 0.5 * (exp[0] + exp[1])
        for pi, p in enumerate(("g", "u", "d")):
            vals = g_lay[pi::3]
            print(f"[PROBE] dKL/d(shared log-scale) proj {p}: mean {float(vals.mean()):+.5f}, min {float(vals.min()):+.5f}, "
                  f"max {float(vals.max()):+.5f}, negative in {int((vals < 0).sum())} of {vals.numel()} layers", flush=True)
        print("[PROBE] per layer (g, u, d): " + " | ".join(
            f"L{l}: " + ",".join(f"{float(g_lay[3 * i + j]):+.4f}" for j in range(3)) for i, l in enumerate(sorted(halves[0]))), flush=True)

        def _apply(level: str, step: float, sign: float) -> None:
            """la += sign * (-step * g / max|g|) for the shared offsets of `level`; scales recommitted."""
            off_e = 0
            for ki, (l, p) in enumerate(keys):
                ho = trainer.handles[l]
                n_e = ho.scale_opt.la[p].shape[0]
                if level == "layer":
                    d = -step * float(g_lay[ki] / g_lay.abs().max())
                    ho.scale_opt.la[p].add_(sign * d)
                else:
                    d_e = (-step * g_exp[off_e : off_e + n_e] / g_exp.abs().max()).float()
                    ho.scale_opt.la[p].add_(sign * d_e.unsqueeze(1))
                off_e += n_e
            for l in sorted(halves[0]):
                ho = trainer.handles[l]
                ho.commit_scales_in_place({p: (ho.scale_opt.s0[p].float() * torch.exp(ho.scale_opt.la[p].float())).to(torch.float16)
                                           for p in ("g", "u", "d")})

        best_kl, best_cfg = _sel(qb0), None
        for level in ("layer", "expert"):
            lvl_best = None
            for step in (0.01, 0.03, 0.1):
                _apply(level, step, +1.0)
                q = trainer.evaluate_kl(qb_ids, qb_targets, micro_bs=args.micro_bs, valid_lens=qb_lens)
                _apply(level, step, -1.0)
                better = _sel(q) < best_kl
                _eval_line(f"probe {level} max-offset {step}", q, " [BETTER]" if better else "")
                if better:
                    best_kl, lvl_best = _sel(q), step
                elif lvl_best is not None:
                    break  # past the minimum of this level
            if lvl_best is not None:
                _apply(level, lvl_best, +1.0)   # keep it; the next level is searched on top
                best_cfg = (best_cfg or []) + [(level, lvl_best)]
        print(f"[PROBE] kept offsets: {best_cfg} | qbench real-position KL {_sel(qb0):.5f} -> {best_kl:.5f}", flush=True)
        if best_cfg:
            trainer.save_adapter_state(args.save_state, meta={"probe": best_cfg, "qbench_kl_valid": best_kl, "diag": trainer.diagnostics()})
            print(f"[PROBE] state saved to {args.save_state}", flush=True)
        return

    n_macro_done = 0
    stop_now = False
    no_improve = 0   # [R3.9 fix 13] evaluations in a row without a new best (--patience)

    for sw in range(1, args.sweeps + 1):
        perm = torch.randperm(train_ids.shape[0], generator=torch.Generator().manual_seed(1000 + sw))
        for m0 in range(0, train_ids.shape[0], args.macro_seqs):
            m1 = min(m0 + args.macro_seqs, train_ids.shape[0])
            if n_macro_done < args.skip_macro:  # [R3.9 fix 12] resume: keep the schedule position, do not train
                n_macro_done += 1
                continue
            sub_idx = perm[m0:m1]
            sub_targets = CompactTeacherTargets(
                top_ids=train_targets.top_ids[sub_idx],
                top_logp=train_targets.top_logp[sub_idx],
                tail_prob=train_targets.tail_prob[sub_idx],
            )
            step_stat = trainer.run_macro_step(train_ids[sub_idx], sub_targets, micro_bs=args.micro_bs)
            diag = trainer.diagnostics()
            print(
                f"[SWEEP {sw} | {m0}:{m1}] train_KL={step_stat['train_kl_nats']:.5f} "
                f"VRAM={step_stat['peak_vram_gib']:.2f}GiB ({step_stat['seconds']:.1f}s) | {json.dumps(diag)}",
                flush=True,
            )
            n_macro_done += 1
            stop_now = bool(args.max_macro_steps and n_macro_done >= args.max_macro_steps)
            last_of_sweep = m1 >= train_ids.shape[0]
            if args.eval_every_macro and n_macro_done % args.eval_every_macro == 0 and not last_of_sweep and not stop_now:
                qb_mid = trainer.evaluate_kl(qb_ids, qb_targets, micro_bs=args.micro_bs, valid_lens=qb_lens)
                mid_best = _sel(qb_mid) < best_qb_kl
                if mid_best:
                    best_qb_kl = _sel(qb_mid)
                    trainer.save_adapter_state(args.save_state, meta={"sweep": sw, "macro_steps": n_macro_done, "qbench": qb_mid, "diag": trainer.diagnostics()})
                # [R3.9 fix 12] the latest state as well, so a stopped run can continue from where it was
                trainer.save_adapter_state(args.save_state + ".last", meta={"sweep": sw, "macro_steps": n_macro_done, "qbench": qb_mid, "diag": trainer.diagnostics()})
                _eval_line(f"{sw}.{n_macro_done}", qb_mid, " [NEW BEST]" if mid_best else "")
                no_improve = 0 if mid_best else no_improve + 1
                if args.patience and no_improve >= args.patience:
                    print(f"[STOP] --patience {args.patience}: {no_improve} evaluations without a new best", flush=True)
                    stop_now = True
            if stop_now:
                break
        if stop_now and args.patience and no_improve >= args.patience:
            break

        qb_cur = trainer.evaluate_kl(qb_ids, qb_targets, micro_bs=args.micro_bs, valid_lens=qb_lens)
        _eval_line(f"{sw} (after {n_macro_done} macro-steps)", qb_cur)
        improved = _sel(qb_cur) < best_qb_kl
        trainer.save_adapter_state(args.save_state + ".last", meta={"sweep": sw, "macro_steps": n_macro_done, "qbench": qb_cur, "diag": trainer.diagnostics()})
        no_improve = 0 if improved else no_improve + 1
        if args.patience and no_improve >= args.patience:
            stop_now = True
        if improved:
            best_qb_kl = _sel(qb_cur)
            trainer.save_adapter_state(args.save_state, meta={"sweep": sw, "macro_steps": n_macro_done, "qbench": qb_cur, "diag": trainer.diagnostics()})
            if args.export_dir:
                exp_info = trainer.export_merged_checkpoint(args.export_dir, TR_mod=TR)
                print(f"  [EXPORT] Saved improved merged checkpoint to {args.export_dir}: {exp_info}", flush=True)
        print(
            f"[EVAL {sw}] qbench KL = {qb_cur['kl_nats']:.5f} nats (best real-position KL={best_qb_kl:.5f}) | "
            f"top1 = {qb_cur['top1_agree']:.4f} {'[NEW BEST]' if improved else ''}",
            flush=True,
        )
        if stop_now:
            print(f"[STOP] after {n_macro_done} macro-steps (--max-macro-steps {args.max_macro_steps}, --patience {args.patience}, "
                  f"{no_improve} evaluations without a new best)", flush=True)
            break


if __name__ == "__main__":
    main()
