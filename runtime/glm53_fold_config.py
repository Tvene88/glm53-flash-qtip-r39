#!/usr/bin/env python3
"""
glm53_fold_config.py
====================
Round-3.1 APEX-FOLD-FP16 Configuration & Hard Byte-Locked Presets for
GLM-5.3-Flash-Abliterated (`320B-A18B`, `Glm5NextForConditionalGeneration`).

Includes all `PATCHES.md` fixes (`R3-01a` through `R3-01e`) plus the Round-3.1
Orthogonal-Expert Full-Width Hadamard Upgrade:
----------------------------------------------------------------------
1. 100% CHANNEL COVERAGE (`c2 = d_in`) & ZERO WASTED SHARED SUBSPACE (`r_shared = 0`):
   Real Layer-3 testing on `dealignai/GLM-5.3-Flash-UNCENSORED-FP8` showed that the 288
   routed experts are mutually orthogonal (`B_shared` at `r_shared = 80` captured `< 3%`
   of energy while consuming `30.1%` of the byte budget) and that truncating `c2 < d_in`
   (`1792 / 4096`) left `2,304` input channels (`55%` of real activation energy) uncoded,
   capping weight SNR at `2.03 dB` and shrinking routed output magnitude to `6.6%`.
   In Round-3.1:
   - `r_shared = 0` across all presets (reclaiming `288-427 MiB/layer` of dead subspace bytes).
   - `SCALE_GROUP = 256` (since Full-Dimension 4096/2048-point FWHT equalizes intra-row
     group variance to within `0.08 dB`, cutting `S_blk` overhead from `0.25 bpw` to
     `0.0625 bpw` and freeing `141.56 MiB/layer` for codebook indices).
   - `c2 = d_in` (`4096` on `gate`/`up`, `2048` on `down`) across `heterogeneous_96gb`,
     `balanced_ram`, `high_fidelity`, and `ultra_96gb` so **100% of input channels are
     coded on every expert**, while matching the **exact `expert_row_bytes`** of the
     original presets down to the single byte!

2. VERIFIED CHECKPOINT GEOMETRY (`PATCH R3-01a..e`):
   - `dense_intermediate_size = 12288`
   - `swiglu_limit = 10.0`
   - `hc_sinkhorn_iters = 20`, `hc_eps = 1e-6`
   - `eos_token_ids = (154820, 154827, 154829)`
   - `qk_nope_head_dim = 256`, `qk_rope_head_dim = 0`, `v_head_dim = 256`, `index_head_dim = 128`
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


def _align16(n: int) -> int:
    return (n + 15) & ~15


@dataclass(frozen=True)
class GLM53FlashModelConfig:
    """Verified Hugging Face `Glm5NextForConditionalGeneration` (`glm5_next`) geometry."""
    model_type: str = "glm5_next"
    architectures: Tuple[str, ...] = ("Glm5NextForConditionalGeneration",)
    hf_repo_fp8: str = "dealignai/GLM-5.3-Flash-UNCENSORED-FP8"
    hf_repo_bf16: str = "zai-org/GLM-5.3-Flash"

    vocab_size: int = 154880
    hidden_size: int = 4096
    moe_intermediate_size: int = 2048
    dense_intermediate_size: int = 12288  # [PATCH R3-01a] config.json text_config.intermediate_size

    num_hidden_layers: int = 45          # Layers 0..44
    num_nextn_predict_layers: int = 1    # Layer 45 (MTP)
    total_checkpointed_layers: int = 46  # Layers 0..45
    first_k_dense_replace: int = 3       # Layers 0, 1, 2 are dense SwiGLU

    n_routed_experts: int = 288
    num_experts_per_tok: int = 8
    n_shared_experts: int = 1
    routed_scaling_factor: float = 2.5
    swiglu_limit: float = 10.0           # [PATCH R3-01b] config.json text_config.swiglu_limit = 10.0 (was 7.0)
    rms_norm_eps: float = 1e-5
    hc_mult: int = 4                     # 4-stream mHC on Layers 0..44 (absent on Layer 45)
    hc_sinkhorn_iters: int = 20          # [PATCH R3-01c] config.json hc_sinkhorn_iters
    hc_eps: float = 1e-6                 # [PATCH R3-01c] config.json hc_eps
    eos_token_ids: Tuple[int, ...] = (154820, 154827, 154829)  # [PATCH R3-01c] generation_config.json

    # KDA Linear Attention (34 layers)
    kda_num_heads: int = 64
    kda_head_dim: int = 128              # 64 * 128 = 8192
    kda_short_conv_kernel_size: int = 4
    kda_chunk_size: int = 32

    # DSA Multi-Head Latent Attention + K-Pool Indexer (11 main + Layer 45 = 12 layers)
    num_attention_heads: int = 64
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 256          # [PATCH R3-01d] config.json (pure NoPE MLA)
    qk_rope_head_dim: int = 0            # [PATCH R3-01d] config.json; qk_head_dim = 256
    v_head_dim: int = 256                # [PATCH R3-01d] config.json
    index_n_heads: int = 32
    index_head_dim: int = 128            # [PATCH R3-01e] config.json (indexer wk [128,4096])
    index_kpool: int = 4
    index_topk: int = 2048
    index_kpool_always_select_tail: bool = True

    @property
    def dsa_layers(self) -> Tuple[int, ...]:
        return (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 45)

    def is_kda_layer(self, layer_id: int) -> bool:
        return layer_id not in self.dsa_layers

    def is_moe_layer(self, layer_id: int) -> bool:
        return layer_id >= self.first_k_dense_replace

    @property
    def moe_layer_ids(self) -> List[int]:
        return list(range(self.first_k_dense_replace, self.num_hidden_layers + self.num_nextn_predict_layers))

    @property
    def num_moe_layers(self) -> int:
        return len(self.moe_layer_ids)  # 43 layers (3..45)

    @property
    def weights_per_expert(self) -> int:
        return 3 * self.hidden_size * self.moe_intermediate_size  # 25,165,824

    @property
    def routed_weights_per_layer(self) -> int:
        return self.n_routed_experts * self.weights_per_expert    # 7,247,757,312

    @property
    def total_routed_weights(self) -> int:
        return self.num_moe_layers * self.routed_weights_per_layer  # 311,653,564,416


VEC_DIM: int = 8
FWHT_BLK: int = 32
SCALE_GROUP: int = 256  # 256 channels = 32 vectors of 8D per FP16 scale (0.0625 bpw after Full-Dimension FWHT)
CODEBOOK_SIZE: int = 256
MAX_STAGES: int = 4


@dataclass(frozen=True)
class ProjectionFoldSpec:
    """
    Hard byte-locked specification for a single projection (`gate`, `up`, or `down`).
    `c2` is a multiple of `scale_group` (`256` on full model, `64` on scaled unit tests),
    and `c0, c1` are multiples of `VEC_DIM = 8` (and `32`).
    """
    name: str
    d_out: int
    d_in: int
    r_shared: int
    c0: int          # Tier-0 end channel (stages_t0 additive FP16 codebook lookups per 8D vector)
    c1: int          # Tier-1 end channel (2 additive FP16 codebook lookups per 8D vector)
    c2: int          # Tier-2 end channel (1 FP16 codebook lookup per 8D vector)
    stages_t0: int = 3
    scale_group: int = SCALE_GROUP

    def __post_init__(self) -> None:
        assert 0 <= self.c0 <= self.c1 <= self.c2 <= self.d_in, f"Invalid cuts {(self.c0, self.c1, self.c2)}"
        assert self.c0 % VEC_DIM == 0 and self.c1 % VEC_DIM == 0 and self.c2 % VEC_DIM == 0
        assert self.c2 % self.scale_group == 0 and self.d_in % self.scale_group == 0
        assert self.stages_t0 in (3, 4)

    @property
    def f_s0(self) -> int:
        """Stage 0 active 8D blocks (covers Tiers 0, 1, 2: channels 0..c2)."""
        return self.c2 // VEC_DIM

    @property
    def f_s1(self) -> int:
        """Stage 1 active 8D blocks (covers Tiers 0, 1: channels 0..c1)."""
        return self.c1 // VEC_DIM

    @property
    def f_s2(self) -> int:
        """Stage 2 active 8D blocks (covers Tier 0: channels 0..c0)."""
        return self.c0 // VEC_DIM

    @property
    def f_s3(self) -> int:
        """Stage 3 active 8D blocks (covers Tier 0 when stages_t0 == 4)."""
        return (self.c0 // VEC_DIM) if self.stages_t0 >= 4 else 0

    @property
    def stage_f_widths(self) -> Tuple[int, ...]:
        if self.stages_t0 == 4:
            return (self.f_s0, self.f_s1, self.f_s2, self.f_s3)
        return (self.f_s0, self.f_s1, self.f_s2)

    @property
    def num_scale_groups(self) -> int:
        return self.c2 // self.scale_group

    @property
    def f_per_group(self) -> int:
        return self.scale_group // VEC_DIM

    @property
    def shared_header_bytes(self) -> int:
        """Layer-shared FP16 tensors for this projection (resident in VRAM for all layers)."""
        b_trans = self.d_in * 2 + self.d_in * 1 + self.d_in * 2  # perm(u16) + signs(i8) + w_pre(fp16)
        b_b_sh = self.r_shared * self.d_in * 2                   # B_shared: float16[r_shared, d_in]
        b_cb = MAX_STAGES * CODEBOOK_SIZE * VEC_DIM * 2          # codebooks: float16[4, 256, 8] = 16,384 B
        return b_trans + b_b_sh + b_cb

    @property
    def per_expert_field_specs(self) -> List[Tuple[str, Tuple[int, ...], str, int]]:
        """
        Ordered list of (field_suffix, shape, dtype_str, elem_bytes) for ONE expert in `expert_slab_u8`.
        All FP16 fields (`A_e`, `S_blk`) are placed first so 2-byte/16-byte alignment is guaranteed.
        Stage-aligned index arrays (`idx_s0`, `idx_s1`, `idx_s2`, `[idx_s3]`) allow zero-copy
        contiguous stage gathers without `torch.cat`!
        """
        specs: List[Tuple[str, Tuple[int, ...], str, int]] = []
        if self.r_shared > 0:
            specs.append(("A_e", (self.d_out, self.r_shared), "float16", 2))
        if self.c2 > 0:
            specs.append(("S_blk", (self.d_out, self.num_scale_groups), "float16", 2))
        for s_idx, f_w in enumerate(self.stage_f_widths):
            if f_w > 0:
                specs.append((f"idx_s{s_idx}", (self.d_out, f_w), "uint8", 1))
        return specs

    @property
    def per_expert_slab_bytes(self) -> int:
        total = 0
        for _, shape, _, eb in self.per_expert_field_specs:
            n = 1
            for s in shape:
                n *= s
            total = _align16(total + n * eb)
        return total

    def exact_bytes(self, n_experts: int = 288) -> int:
        return self.shared_header_bytes + n_experts * self.per_expert_slab_bytes

    def bpw(self, n_experts: int = 288) -> float:
        return (self.exact_bytes(n_experts) * 8.0) / (n_experts * self.d_out * self.d_in)


@dataclass(frozen=True)
class LayerFoldPreset:
    """Complete layer-level folding preset with exact byte-locked arithmetic."""
    name: str
    description: str
    gate: ProjectionFoldSpec
    up: ProjectionFoldSpec
    down: ProjectionFoldSpec
    default_backbone_dtype: str = "fp8"  # Keeps non-expert 2D weights in FP8_E4M3 (~8.65 GiB VRAM)

    def slab_layout(self) -> Tuple[Dict[str, Tuple[int, int, Tuple[int, ...], str]], int]:
        """
        Computes the unified 16-byte-aligned byte layout for a single expert row in
        `expert_slab_u8[E, row_bytes]` combining `gate`, `up`, and `down`.
        All `float16` fields across `gate`, `up`, `down` are placed before `uint8` fields!
        """
        layout: Dict[str, Tuple[int, int, Tuple[int, ...], str]] = {}
        layout = {}
        cursor = 0
        # Pass 1: All float16 fields (A_e, S_blk) across gate, up, down
        for proj_prefix, p_spec in (("g", self.gate), ("u", self.up), ("d", self.down)):
            for suffix, shape, dtype_str, eb in p_spec.per_expert_field_specs:
                if dtype_str != "float16":
                    continue
                n_bytes = eb
                for s in shape:
                    n_bytes *= s
                layout[f"{proj_prefix}_{suffix}"] = (cursor, cursor + n_bytes, shape, dtype_str)
                cursor = _align16(cursor + n_bytes)

        # Pass 2: All uint8 index fields (idx_s0, idx_s1, idx_s2, [idx_s3]) across gate, up, down
        for proj_prefix, p_spec in (("g", self.gate), ("u", self.up), ("d", self.down)):
            for suffix, shape, dtype_str, eb in p_spec.per_expert_field_specs:
                if dtype_str == "float16":
                    continue
                n_bytes = eb
                for s in shape:
                    n_bytes *= s
                layout[f"{proj_prefix}_{suffix}"] = (cursor, cursor + n_bytes, shape, dtype_str)
                cursor = _align16(cursor + n_bytes)

        return layout, _align16(cursor)

    @property
    def expert_row_bytes(self) -> int:
        _, row_bytes = self.slab_layout()
        return row_bytes

    def shared_basis_bytes_per_layer(self) -> int:
        return self.gate.shared_header_bytes + self.up.shared_header_bytes + self.down.shared_header_bytes

    def slab_bytes_per_layer(self, n_experts: int = 288) -> int:
        return n_experts * self.expert_row_bytes

    def layer_bytes(self, n_experts: int = 288) -> int:
        return self.shared_basis_bytes_per_layer() + self.slab_bytes_per_layer(n_experts)

    def layer_mib(self, n_experts: int = 288) -> float:
        return self.layer_bytes(n_experts) / (1024.0 ** 2)

    def total_gib(self, n_layers: int = 43, n_experts: int = 288) -> float:
        return (n_layers * self.layer_bytes(n_experts)) / (1024.0 ** 3)

    def bpw(self, n_experts: int = 288) -> float:
        total_w = n_experts * (
            self.gate.d_out * self.gate.d_in
            + self.up.d_out * self.up.d_in
            + self.down.d_out * self.down.d_in
        )
        return (self.layer_bytes(n_experts) * 8.0) / total_w


APEX_FOLD_PRESETS: Dict[str, LayerFoldPreset] = {
    "vram_100pct_32gb": LayerFoldPreset(
        name="vram_100pct_32gb",
        description="100% RTX 5090 32 GB VRAM Resident (All 43 MoE layers + FP8 backbone in VRAM, 0 PCIe traffic; r_shared=0, 2.67x wider Hadamard coverage)",
        gate=ProjectionFoldSpec("gate", 2048, 4096, r_shared=0, c0=0, c1=64, c2=2048, stages_t0=3),
        up=ProjectionFoldSpec("up", 2048, 4096, r_shared=0, c0=0, c1=64, c2=2048, stages_t0=3),
        down=ProjectionFoldSpec("down", 4096, 2048, r_shared=0, c0=32, c1=128, c2=1024, stages_t0=3),
        default_backbone_dtype="fp8",
    ),
    "heterogeneous_96gb": LayerFoldPreset(
        name="heterogeneous_96gb",
        description="Compact 1.10 bpw Full-Coverage Preset (100% channels coded c2=4096/2048, r_shared=0; 21 MoE layers in VRAM + 22 in DDR5, ~5.35 dB orthogonal Weight SNR)",
        gate=ProjectionFoldSpec("gate", 2048, 4096, r_shared=0, c0=64, c1=128, c2=4096, stages_t0=3),
        up=ProjectionFoldSpec("up", 2048, 4096, r_shared=0, c0=64, c1=128, c2=4096, stages_t0=3),
        down=ProjectionFoldSpec("down", 4096, 2048, r_shared=0, c0=0, c1=64, c2=2048, stages_t0=3),
        default_backbone_dtype="fp8",
    ),
    "balanced_ram": LayerFoldPreset(
        name="balanced_ram",
        description="Balanced 1.62 bpw Full-Coverage Preset (100% channels coded c2=4096/2048, >50% 2-stage; 14 MoE layers in VRAM + 29 in DDR5, ~7.6 dB orthogonal Weight SNR)",
        gate=ProjectionFoldSpec("gate", 2048, 4096, r_shared=0, c0=256, c1=2080, c2=4096, stages_t0=3),
        up=ProjectionFoldSpec("up", 2048, 4096, r_shared=0, c0=256, c1=2080, c2=4096, stages_t0=3),
        down=ProjectionFoldSpec("down", 4096, 2048, r_shared=0, c0=128, c1=928, c2=2048, stages_t0=3),
        default_backbone_dtype="fp8",
    ),
    "high_fidelity": LayerFoldPreset(
        name="high_fidelity",
        description="Recommended 96 GB Workstation Preset (~2.19 bpw, 79.38 GiB: 100% 2-stage + Tier-0 3-stage across all 4096/2048 channels; 10 layers in VRAM + 33 in 60.8 GiB DDR5, ~11.2 dB Weight SNR)",
        gate=ProjectionFoldSpec("gate", 2048, 4096, r_shared=0, c0=768, c1=4096, c2=4096, stages_t0=3),
        up=ProjectionFoldSpec("up", 2048, 4096, r_shared=0, c0=768, c1=4096, c2=4096, stages_t0=3),
        down=ProjectionFoldSpec("down", 4096, 2048, r_shared=0, c0=0, c1=2048, c2=2048, stages_t0=3),
        default_backbone_dtype="fp8",
    ),
    "ultra_96gb": LayerFoldPreset(
        name="ultra_96gb",
        description="Maximum-Fidelity 3-Stage Workstation Preset (~2.69 bpw, 97.51 GiB with zero-copy mmap_slab: 75% 3-stage + 100% 2-stage across all channels; ~13.9 dB Weight SNR, ~9.9 dB Routed SNR)",
        gate=ProjectionFoldSpec("gate", 2048, 4096, r_shared=0, c0=3072, c1=4096, c2=4096, stages_t0=3),
        up=ProjectionFoldSpec("up", 2048, 4096, r_shared=0, c0=3072, c1=4096, c2=4096, stages_t0=3),
        down=ProjectionFoldSpec("down", 4096, 2048, r_shared=0, c0=1536, c1=2048, c2=2048, stages_t0=3),
        default_backbone_dtype="fp8",
    ),
}


def compute_exact_hardware_budget(
    preset_name: str = "heterogeneous_96gb",
    vram_budget_gib: float = 30.6,
    backbone_dtype: Optional[str] = None,
    kv_cache_gib: float = 1.20,
    cfg: Optional[GLM53FlashModelConfig] = None,
) -> Dict[str, object]:
    """
    Single source of truth for both `--doctor` and `ApexFoldPlacementManager`.
    Accounts for every single byte:
      1. Backbone VRAM (`8.65 GiB` in native `fp8`, or `17.13 GiB` in `bf16`)
      2. Shared FP16 headers + codebooks for ALL 43 layers (`~0.11 MiB/layer`, `< 0.01 GiB` total!)
      3. PinnedDMA staging buffers (`~64 MiB`)
      4. Exact number of MoE expert slabs that fit in remaining VRAM vs Host DDR5 RAM / mmap.
    """
    cfg = cfg or GLM53FlashModelConfig()
    preset = APEX_FOLD_PRESETS[preset_name]
    bb_dtype = backbone_dtype or preset.default_backbone_dtype
    backbone_vram_gib = 8.65 if bb_dtype == "fp8" else 17.13

    n_layers = cfg.num_moe_layers
    n_exp = cfg.n_routed_experts

    shared_per_layer_bytes = preset.shared_basis_bytes_per_layer()
    slab_per_layer_bytes = preset.slab_bytes_per_layer(n_exp)
    layer_total_bytes = preset.layer_bytes(n_exp)

    all_shared_vram_gib = (n_layers * shared_per_layer_bytes) / (1024.0 ** 3)
    dma_staging_gib = (32 * preset.expert_row_bytes * 2) / (1024.0 ** 3)

    fixed_vram_gib = backbone_vram_gib + kv_cache_gib + all_shared_vram_gib + dma_staging_gib
    avail_for_slabs_gib = max(0.0, vram_budget_gib - fixed_vram_gib)
    slab_gib = slab_per_layer_bytes / (1024.0 ** 3)

    vram_layers_count = min(n_layers, int(avail_for_slabs_gib // slab_gib))
    host_layers_count = n_layers - vram_layers_count

    priority_order: List[int] = []
    for l in (3, 4, 5, 6, 45):
        if l in cfg.moe_layer_ids and l not in priority_order:
            priority_order.append(l)
    for l in cfg.dsa_layers:
        if l in cfg.moe_layer_ids and l not in priority_order:
            priority_order.append(l)
    for l in cfg.moe_layer_ids:
        if l not in priority_order:
            priority_order.append(l)

    vram_layer_ids = sorted(priority_order[:vram_layers_count])
    host_layer_ids = sorted(priority_order[vram_layers_count:])

    vram_slabs_gib = vram_layers_count * slab_gib
    host_slabs_gib = host_layers_count * slab_gib
    total_vram_used_gib = fixed_vram_gib + vram_slabs_gib
    total_archive_gib = (n_layers * layer_total_bytes) / (1024.0 ** 3)

    return {
        "preset": preset_name,
        "description": preset.description,
        "backbone_dtype": bb_dtype,
        "bpw": round(preset.bpw(n_exp), 4),
        "layer_mib": round(preset.layer_mib(n_exp), 2),
        "shared_header_mib_per_layer": round(shared_per_layer_bytes / (1024.0 ** 2), 3),
        "expert_row_kib": round(preset.expert_row_bytes / 1024.0, 2),
        "total_43_layer_archive_gib": round(total_archive_gib, 2),
        "backbone_vram_gib": round(backbone_vram_gib, 2),
        "all_43_shared_headers_vram_gib": round(all_shared_vram_gib, 3),
        "kv_and_dma_vram_gib": round(kv_cache_gib + dma_staging_gib, 2),
        "vram_layers_count": vram_layers_count,
        "host_layers_count": host_layers_count,
        "vram_layer_ids": vram_layer_ids,
        "host_layer_ids": host_layer_ids,
        "vram_slabs_gib": round(vram_slabs_gib, 2),
        "total_vram_used_gib": round(total_vram_used_gib, 2),
        "host_ram_slabs_gib": round(host_slabs_gib, 2),
        "cuts_gate": (preset.gate.c0, preset.gate.c1, preset.gate.c2),
        "cuts_up": (preset.up.c0, preset.up.c1, preset.up.c2),
        "cuts_down": (preset.down.c0, preset.down.c1, preset.down.c2),
        "r_shared": (preset.gate.r_shared, preset.up.r_shared, preset.down.r_shared),
    }


if __name__ == "__main__":
    for name in APEX_FOLD_PRESETS:
        b = compute_exact_hardware_budget(name)
        print(
            f"[{name:18s}] {b['bpw']:.4f} bpw | Layer: {b['layer_mib']:7.2f} MiB "
            f"(Shared hdr: {b['shared_header_mib_per_layer']:.2f} MiB) | "
            f"43-Layer Archive: {b['total_43_layer_archive_gib']:5.2f} GiB | "
            f"VRAM: {b['total_vram_used_gib']:5.2f} GiB ({b['vram_layers_count']}/43 layers in VRAM) | "
            f"Host RAM: {b['host_ram_slabs_gib']:5.2f} GiB ({b['host_layers_count']}/43 layers)"
        )
