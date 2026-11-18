#!/usr/bin/env python3
"""
glm53_fold_backbone.py
======================
Architecture-Faithful GLM-5.3-Flash (`Glm5NextForConditionalGeneration` / `glm5_next`)
Backbone Engine & Real Safetensors Checkpoint Loader for Round-3.1 `APEX-FOLD-FP16`.

Verified Against `transformers==5.16.1` (`modeling_glm5_next.py`) & `C:\\models\\GLM53-backbone`:
------------------------------------------------------------------------------------------------
1. EXACT REAL CHECKPOINT TENSOR KEYS & DYNAMIC PER-TOKEN mHC (`PATCH R3-02a..e`):
   - `LM = "model.language_model."` prefix on all layers `0..45` and `embed_tokens`/`norm`,
     with top-level `"lm_head.weight"`.
   - Handles bare tensor names without `.weight`: `dt_bias`, `A_log`, `hc_{attn,ffn}_{fn,base,scale}` [PATCH R3-02],
     `mlp.gate.e_score_correction_bias`, `index_kpool_compress_gate`, `index_kpool_compress_ape`.
   - Dynamic per-token mHC (`Glm5NextTextHyperConnection.forward` port, 0.0 bit-exact diff vs HF).
2. IN-PLACE NATIVE `FP8_E4M3` DEQUANTIZATION (`backbone_dtype="fp8"`):
   - Performs block-128 scale multiplication in-place (`.mul_()`), cutting temporary GDDR7 BF16
     allocation and memory traffic in half during `lin()`.
3. VECTORIZED CHUNK-PARALLEL WY KDA PREFILL (`S > 2`) & IN-PLACE KDA DECODE (`S <= 2`).
"""

from __future__ import annotations
import glob
import json
import os
import struct
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from glm53_torch_compat import torch, nn, F

from glm53_fold_config import GLM53FlashModelConfig

import glm53_fold_kernels as KR  # [R3.2] NVRTC fused kernels
_FUSED_MERGE = [True]  # [R3.2-1.4] module switch for the (static) mhc_merge

LM = "model.language_model."


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0 or unit == "TiB":
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} TiB"


@dataclass
class FP8LinearWeight:
    """
    Holds a 1-byte `torch.float8_e4m3fn` weight matrix `[Out, In]` and its `128x128` block
    inverse scale `[ceil(Out/128), ceil(In/128)]` directly in VRAM without 2x BF16 expansion!
    """
    qweight: torch.Tensor    # [Out, In] float8_e4m3fn (or int8/uint8 view)
    scale_inv: torch.Tensor  # [so, si] bfloat16/float32
    block_size: int = 128
    # [R3.4-V] True: scale_inv is [Out, In/bs] - one scale row per OUTPUT row (a row-sliced block-128 head: every kept row
    # keeps its original block's scale row, so its products are bit-identical to the unsliced weight's)
    row_scales: bool = False

    @property
    def shape(self) -> torch.Size:
        return self.qweight.shape

    def bytes(self) -> int:
        return (
            self.qweight.numel() * self.qweight.element_size()
            + self.scale_inv.numel() * self.scale_inv.element_size()
        )

    def dequantize(self, target_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        O, I = self.qweight.shape
        s = self.scale_inv.to(target_dtype)
        w_out = self.qweight.to(target_dtype)
        if self.row_scales:                      # [R3.4-V] per-row scales [O, I/bs]
            bs = self.block_size
            return w_out.view(O, I // bs, bs).mul_(s[:, :, None]).reshape(O, I)
        if s.ndim == 0 or s.numel() == 1:
            return w_out.mul_(s)
        if s.ndim == 1:
            return w_out.mul_(s.view(-1, 1))
        so, si = s.shape
        bs = self.block_size
        if O == so * bs and I == si * bs:
            return w_out.view(so, bs, si, bs).mul_(s[:, None, :, None]).reshape(O, I)
        s_full = s.repeat_interleave(bs, dim=0)[:O].repeat_interleave(bs, dim=1)[:, :I]
        return w_out.mul_(s_full)


WeightTensor = Union[torch.Tensor, FP8LinearWeight]


class _MmapSafetensorsShard:
    """Zero-copy read-only memory-mapped `.safetensors` reader supporting `F8_E4M3`, `BF16`, `F16`, `F32`."""

    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "rb")
        hdr_len = struct.unpack("<Q", self._f.read(8))[0]
        self.header = json.loads(self._f.read(hdr_len).decode("utf-8"))
        self.data_offset = 8 + hdr_len
        self._mmap = np.memmap(path, dtype=np.uint8, mode="r", offset=self.data_offset)

    def keys(self):
        return [k for k in self.header.keys() if k != "__metadata__"]

    def get_raw_slice(self, name: str) -> Tuple[np.ndarray, str, Tuple[int, ...]]:
        info = self.header[name]
        s, e = info["data_offsets"]
        return self._mmap[s:e], info["dtype"], tuple(info["shape"])

    def close(self) -> None:
        self._mmap = None
        if self._f is not None:
            self._f.close()
            self._f = None


class RealGLM53Checkpoint:
    """
    Unified loader for `dealignai/GLM-5.3-Flash-UNCENSORED-FP8` and `C:\\models\\GLM53-backbone`.
    Supports both FP32/BF16 numpy dequantization (for encoder consumption) and direct
    `FP8LinearWeight` GPU loading (keeping the non-expert backbone at `~8.65 GiB` VRAM!).
    """

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.weight_map: Dict[str, str] = {}
        self._shards: Dict[str, _MmapSafetensorsShard] = {}
        idx_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(idx_path):
            with open(idx_path, "r", encoding="utf-8") as f:
                self.weight_map = json.load(f).get("weight_map", {})
        else:
            for p in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
                sh = _MmapSafetensorsShard(p)
                bn = os.path.basename(p)
                for k in sh.keys():
                    self.weight_map[k] = bn
                self._shards[bn] = sh

    def _shard(self, rel_path: str) -> _MmapSafetensorsShard:
        if rel_path not in self._shards:
            full = os.path.join(self.model_dir, rel_path)
            self._shards[rel_path] = _MmapSafetensorsShard(full)
        return self._shards[rel_path]

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def close(self) -> None:
        for sh in list(self._shards.values()):
            sh.close()
        self._shards.clear()

    @staticmethod
    def _fp8_e4m3_bytes_to_f32_numpy(raw_u8: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        t_u8 = torch.from_numpy(np.array(raw_u8, copy=True)).reshape(shape)
        return t_u8.view(torch.float8_e4m3fn).to(torch.float32).numpy()

    def get_numpy_f32(self, name: str) -> np.ndarray:
        """Returns any checkpoint tensor as an `np.float32` array, applying `.weight_scale_inv` if FP8."""
        sh = self._shard(self.weight_map[name])
        raw, dt, shape = sh.get_raw_slice(name)
        if dt == "F32":
            arr = np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
        elif dt == "F16":
            arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
        elif dt == "BF16":
            u16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
            arr = (u16.astype(np.uint32) << 16).view(np.float32).copy()
        elif dt in ("F8_E4M3", "F8_E4M3FN"):
            arr = self._fp8_e4m3_bytes_to_f32_numpy(raw, shape)
            scale_key = name + "_scale_inv"
            if scale_key in self.weight_map:
                sc = self.get_numpy_f32(scale_key)
                if sc.ndim == 2 and arr.ndim == 2:
                    O, I = arr.shape
                    so, si = sc.shape
                    bs_o = (O + so - 1) // so
                    bs_i = (I + si - 1) // si
                    if O == so * bs_o and I == si * bs_i:
                        arr = (
                            arr.reshape(so, bs_o, si, bs_i) * sc[:, None, :, None]
                        ).reshape(O, I)
                    else:
                        sc_full = np.repeat(np.repeat(sc, bs_o, axis=0)[:O], bs_i, axis=1)[:, :I]
                        arr = arr * sc_full
                elif sc.ndim == 1 and arr.ndim == 2:
                    arr = arr * sc[:, None]
                else:
                    arr = arr * sc
        else:
            raise ValueError(f"Unsupported safetensors dtype {dt} for {name}")
        return arr

    def get_torch_weight(
        self,
        name: str,
        device: torch.device,
        mode: str = "fp8",
    ) -> WeightTensor:
        sh = self._shard(self.weight_map[name])
        raw, dt, shape = sh.get_raw_slice(name)
        scale_key = name + "_scale_inv"
        if mode == "fp8" and dt in ("F8_E4M3", "F8_E4M3FN") and scale_key in self.weight_map:
            t_u8 = torch.from_numpy(np.array(raw, copy=True)).reshape(shape).to(device)
            qweight = t_u8.view(torch.float8_e4m3fn)
            sc_np = self.get_numpy_f32(scale_key)
            # [R3.4-2b] block scales stay bf16-ROUNDED (bit-exact with R3.2/R3.3) but are kept as fp32, so fp8_gemv
            # no longer converts them on every call (bf16 -> fp32 is exact)
            sc_t = torch.from_numpy(sc_np).to(device=device, dtype=torch.bfloat16).float()
            return FP8LinearWeight(qweight=qweight, scale_inv=sc_t, block_size=128)

        arr = self.get_numpy_f32(name)
        target_dt = torch.float32 if mode == "fp32" else torch.bfloat16
        return torch.from_numpy(arr).to(device=device, dtype=target_dt)

    def expert(self, layer_id: int, expert_id: int, proj: str) -> np.ndarray:
        key = f"{LM}layers.{layer_id}.mlp.experts.{expert_id}.{proj}.weight"
        return self.get_numpy_f32(key)

    def load_all_experts_torch(
        self,
        layer_id: int,
        n_experts: int = 288,
        dtype: torch.dtype = torch.bfloat16,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g0 = self.expert(layer_id, 0, "gate_proj")
        d_inter, d_model = g0.shape
        W_gate = torch.empty((n_experts, d_inter, d_model), dtype=dtype, device="cpu")
        W_up = torch.empty((n_experts, d_inter, d_model), dtype=dtype, device="cpu")
        W_down = torch.empty((n_experts, d_model, d_inter), dtype=dtype, device="cpu")

        W_gate[0] = torch.from_numpy(g0).to(dtype)
        W_up[0] = torch.from_numpy(self.expert(layer_id, 0, "up_proj")).to(dtype)
        W_down[0] = torch.from_numpy(self.expert(layer_id, 0, "down_proj")).to(dtype)

        for e in range(1, n_experts):
            W_gate[e] = torch.from_numpy(self.expert(layer_id, e, "gate_proj")).to(dtype)
            W_up[e] = torch.from_numpy(self.expert(layer_id, e, "up_proj")).to(dtype)
            W_down[e] = torch.from_numpy(self.expert(layer_id, e, "down_proj")).to(dtype)

        return W_gate, W_up, W_down


class RealCache:
    """Recurrent KDA state + conv1d cache and DSA latent KV/indexer cache with MTP snapshot/rollback."""

    def __init__(self):
        self.pos: int = 0
        self.kda_state: Dict[int, torch.Tensor] = {}
        self.kda_conv: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.dsa_kv: Dict[int, torch.Tensor] = {}
        self.dsa_ki: Dict[int, torch.Tensor] = {}
        self.dsa_gs: Dict[int, torch.Tensor] = {}

    def snapshot(self) -> Dict[str, object]:
        return {
            "pos": self.pos,
            "kda_state": {l: s.clone() for l, s in self.kda_state.items()},
            "kda_conv": {l: (c[0].clone(), c[1].clone(), c[2].clone()) for l, c in self.kda_conv.items()},
            "dsa_len": {l: v.shape[1] for l, v in self.dsa_kv.items()},
        }

    def restore(self, snap: Dict[str, object]) -> None:
        self.pos = int(snap["pos"])
        for l, s in snap["kda_state"].items():
            self.kda_state[l] = s.clone()
        for l, c in snap["kda_conv"].items():
            self.kda_conv[l] = (c[0].clone(), c[1].clone(), c[2].clone())
        for l, length in snap["dsa_len"].items():
            self.dsa_kv[l] = self.dsa_kv[l][:, :length]
            self.dsa_ki[l] = self.dsa_ki[l][:, :length]
            self.dsa_gs[l] = self.dsa_gs[l][:, :length]
        for l in list(self.dsa_kv.keys()):
            if l not in snap["dsa_len"]:
                del self.dsa_kv[l], self.dsa_ki[l], self.dsa_gs[l]

    def bytes(self) -> int:
        n = 0
        for s in self.kda_state.values():
            n += s.numel() * s.element_size()
        for cs in self.kda_conv.values():
            n += sum(c.numel() * c.element_size() for c in cs)
        for d in (self.dsa_kv, self.dsa_ki, self.dsa_gs):
            for t in d.values():
                n += t.numel() * t.element_size()
        return n


class RealGLM53Backbone:
    """
    Architecture-faithful non-expert backbone of GLM-5.3-Flash (`Glm5NextForConditionalGeneration`)
    with native `FP8_E4M3` weight retention (`backbone_mode="fp8"`, `~8.65 GiB` VRAM) and
    vectorized chunk-parallel KDA prefill (`max_abs_diff < 1e-6` vs token-by-token recurrence).
    """

    def __init__(
        self,
        ckpt: RealGLM53Checkpoint,
        cfg: Optional[GLM53FlashModelConfig] = None,
        device: str = "cuda",
        backbone_mode: str = "fp8",
    ):
        self.ckpt = ckpt
        self.cfg = cfg or GLM53FlashModelConfig()
        self.device = torch.device(device)
        self.backbone_mode = backbone_mode
        self.act_dtype = torch.float32 if backbone_mode == "fp32" else torch.bfloat16

        self.hc = self.cfg.hc_mult
        self.D = self.cfg.hidden_size
        self.eps = float(self.cfg.rms_norm_eps)
        self.H_kda = self.cfg.kda_num_heads
        self.hd_kda = self.cfg.kda_head_dim
        self.lower_bound = -5.0

        self.H = self.cfg.num_attention_heads
        self.qk = self.cfg.qk_nope_head_dim + self.cfg.qk_rope_head_dim  # 256
        self.vd = self.cfg.v_head_dim                                    # 256

        self.embed: Optional[WeightTensor] = None
        self.head: Optional[WeightTensor] = None
        self.final_norm: Optional[torch.Tensor] = None
        self._layers: Dict[int, Dict[str, WeightTensor]] = {}
        self.indexer_checks = {"indexer_calls": 0, "indexer_equals_causal": 0}
        # [R3.2] fast-path switches (all mathematically equivalent to the Round-3.1 path)
        self.embed_on_host = True          # 1.2: embedding rows gathered in host RAM
        self.dsa_absorbed = True           # 1.3: MLA absorption (no per-step kv_b re-projection of the cache)
        self.dsa_skip_indexer_short = True # 1.3: for T <= index_topk the indexer selects every visible pool (exact)
        self.indexer_diagnostics = False   # 1.2: the equality check forced a GPU->CPU sync on every DSA call
        self._dsa_wkv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.use_kernels = self.device.type == "cuda"  # 1.4: NVRTC fused mHC + FP8 GEMV for <= 4 tokens
        self.fp8_diet = False  # 1.7: BF16 KDA q/k/v/o -> block-128 e4m3 at load (opt-in, measured)
        # [R3.4-S1] expert pruning masks: layer -> sorted masked expert ids, written IN PLACE into the cached fp32 router
        # bias gate_b (-inf) - route(), route_ids() (next-layer prefetch prediction), MTP and the captured CUDA graphs all
        # read that same tensor. _gate_b0 keeps the unmasked bias for restore / re-masking.
        self.prune_spec: Dict[int, List[int]] = {}
        self._gate_b0: Dict[int, torch.Tensor] = {}
        # [R3.4-V] trimmed OUTPUT head (set_vocab_keep): vocab_ids [K] = original token id of head row k (device long),
        # vocab_pos [V] = head row of token id or -1 (dropped); None = full head (default)
        self.vocab_ids: Optional[torch.Tensor] = None
        self.vocab_pos: Optional[torch.Tensor] = None
        self.vocab_rows_full: Optional[int] = None
        self._head_rows: Optional[int] = None  # K (an FP8 head is padded to a multiple of 128 rows; logits are cut to K)
        self.graphs_captured = False        # [R3.4 review] set by GraphDecoder.capture(): the head may no longer change
        # [R3.5-T H7] env R35_DSA_QBLOCK_MIB (MiB, default 0 = off): the eager absorbed DSA attention materialises fp32
        # scores [B, H, S, T] (about 3 such tensors live at once); when ONE of them would exceed this budget, the queries
        # are processed in blocks of rows so that a block's scores fit it. Every query row still softmaxes over all its
        # visible keys; below the budget the base statements run unchanged (bit-identical), above it only the GEMM
        # tiling may differ (gate: prefill logits top-1 100 %, KL <= 1e-4 vs unblocked). For long one-shot prompts.
        self.dsa_qblock_mib = float(os.environ.get("R35_DSA_QBLOCK_MIB", "0") or 0.0)
        if os.environ.get('R36_LONGCTX','0')=='1' and self.dsa_qblock_mib==0:
            self.dsa_qblock_mib=128.0
        if self.dsa_qblock_mib < 0:
            raise ValueError(f"R35_DSA_QBLOCK_MIB must be >= 0, not {self.dsa_qblock_mib}")
        self.dsa_qblock_stats = {"calls": 0, "blocks": 0}
        # [R3.7 Stage 7] three opt-in switches for long prompt passes (all unset = unchanged behaviour):
        # R37_VRAM_CAP_MIB: upper bound of the memory PyTorch may reserve on the GPU. Under Windows a full card does
        #   not raise an error, the driver moves memory to shared (system) memory and the request crawls; with the
        #   bound PyTorch first returns its cached free blocks and only then fails with an out-of-memory error.
        # R37_EMPTY_CACHE=1: return cached free blocks to the driver after every layer of a pass over more than 4
        #   positions (a prompt pass allocates many block sizes once; they stay reserved otherwise).
        # R37_MEMTRACE=1: print the peak of allocated memory per layer part for such passes (diagnostic).
        self._r37_empty_cache = os.environ.get("R37_EMPTY_CACHE", "0") == "1"
        self._r37_memtrace = os.environ.get("R37_MEMTRACE", "0") in ("1", "2")
        self._mt = None
        cap = float(os.environ.get("R37_VRAM_CAP_MIB", "0") or 0.0)
        if cap > 0 and torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            frac = min(1.0, cap * 2**20 / total)
            torch.cuda.set_per_process_memory_fraction(frac, 0)
            print(f"[load] [R3.7] PyTorch GPU memory bound: {cap:.0f} MiB of {total / 2**20:.0f} "
                  f"(R37_VRAM_CAP_MIB, fraction {frac:.4f})", flush=True)

    def _t(self, name: str) -> torch.Tensor:
        return torch.from_numpy(self.ckpt.get_numpy_f32(name)).to(self.device, dtype=torch.float32)

    def _m(self, name: str) -> WeightTensor:
        return self.ckpt.get_torch_weight(name, self.device, mode=self.backbone_mode)

    def load_global(self) -> None:
        if self.embed is None:
            if self.embed_on_host and self.device.type == "cuda":
                # [R3.2-1.2] 1.18 GiB BF16 table kept in host RAM; embed_streams gathers rows on the CPU
                self.embed = self.ckpt.get_torch_weight(f"{LM}embed_tokens.weight", torch.device("cpu"), mode="bf16")
            else:
                self.embed = self._m(f"{LM}embed_tokens.weight")
            self.head = self._m("lm_head.weight")
            self.final_norm = self._t(f"{LM}norm.weight")

    def is_kda(self, l: int) -> bool:
        p = f"{LM}layers.{l}.self_attn."
        if self.ckpt.has(p + "q_conv1d.weight"):
            return True
        if self.ckpt.has(p + "q_a_proj.weight"):
            return False
        return self.cfg.is_kda_layer(l)

    def layer(self, l: int) -> Dict[str, WeightTensor]:
        if l in self._layers:
            return self._layers[l]
        p = f"{LM}layers.{l}."
        t, m = self._t, self._m
        w: Dict[str, WeightTensor] = {
            "in_norm": t(p + "input_layernorm.weight"),
            "post_norm": t(p + "post_attention_layernorm.weight"),
        }
        if l < self.cfg.num_hidden_layers:
            for site in ("attn", "ffn"):  # [PATCH R3-02] real mHC parameters: fn [24,16384], base [24], scale [3]
                w[f"hc_{site}_fn"] = t(p + f"hc_{site}_fn")
                w[f"hc_{site}_base"] = t(p + f"hc_{site}_base")
                w[f"hc_{site}_scale"] = t(p + f"hc_{site}_scale")

        a = p + "self_attn."
        if self.is_kda(l):
            w["q"] = self._diet(m(a + "q_proj.weight"))
            w["k"] = self._diet(m(a + "k_proj.weight"))
            w["v"] = self._diet(m(a + "v_proj.weight"))
            w["qconv"] = t(a + "q_conv1d.weight").reshape(-1, self.cfg.kda_short_conv_kernel_size)
            w["kconv"] = t(a + "k_conv1d.weight").reshape(-1, self.cfg.kda_short_conv_kernel_size)
            w["vconv"] = t(a + "v_conv1d.weight").reshape(-1, self.cfg.kda_short_conv_kernel_size)
            w["fa"] = m(a + "f_a_proj.weight")
            w["fb"] = m(a + "f_b_proj.weight")
            w["dt_bias"] = t(a + "dt_bias")
            w["A_log"] = t(a + "A_log")
            w["b"] = m(a + "b_proj.weight")
            w["ga"] = m(a + "g_a_proj.weight")
            w["gb"] = m(a + "g_b_proj.weight")
            w["o_norm"] = t(a + "o_norm.weight")
            w["o"] = self._diet(m(a + "o_proj.weight"))
        else:
            w["qa"] = m(a + "q_a_proj.weight")
            w["qa_norm"] = t(a + "q_a_layernorm.weight")
            w["qb"] = m(a + "q_b_proj.weight")
            w["kva"] = m(a + "kv_a_proj_with_mqa.weight")
            w["kva_norm"] = t(a + "kva_layernorm.weight" if self.ckpt.has(a + "kva_layernorm.weight") else a + "kv_a_layernorm.weight")
            w["kvb"] = m(a + "kv_b_proj.weight")
            w["o"] = m(a + "o_proj.weight")
            i = a + "indexer."
            w["i_wqb"] = m(i + "wq_b.weight")
            w["i_wk"] = m(i + "wk.weight")
            w["i_kn_w"] = t(i + "k_norm.weight")
            w["i_kn_b"] = t(i + "k_norm.bias")
            w["i_wp"] = m(i + "weights_proj.weight")
            w["i_gate"] = m(i + "index_kpool_compress_gate")
            w["i_ape"] = t(i + "index_kpool_compress_ape")

        if not self.cfg.is_moe_layer(l):
            w["w1"] = m(p + "mlp.gate_proj.weight")
            w["w3"] = m(p + "mlp.up_proj.weight")
            w["w2"] = m(p + "mlp.down_proj.weight")
        else:
            w["gate_w"] = t(p + "mlp.gate.weight")
            w["gate_b"] = t(p + "mlp.gate.e_score_correction_bias")
            if l in self.prune_spec:                # [R3.4-S1] (re)loaded layer: re-apply its mask to the fresh bias
                self._gate_b0.pop(l, None)
                self._mask_gate(l, w)
            w["sh_w1"] = m(p + "mlp.shared_experts.gate_proj.weight")
            w["sh_w3"] = m(p + "mlp.shared_experts.up_proj.weight")
            w["sh_w2"] = m(p + "mlp.shared_experts.down_proj.weight")

        if l == self.cfg.num_hidden_layers:
            w["eh_proj"] = m(p + "eh_proj.weight")
            w["enorm"] = t(p + "enorm.weight")
            w["hnorm"] = t(p + "hnorm.weight")
            w["shared_head_norm"] = t(p + "shared_head.norm.weight")

        self._layers[l] = w
        return w

    def _diet(self, wt: WeightTensor) -> WeightTensor:
        """[R3.2-1.7] BF16 [O, I] (O, I multiples of 128) -> FP8LinearWeight with amax/448 block scales."""
        if not self.fp8_diet or isinstance(wt, FP8LinearWeight) or wt.ndim != 2:
            return wt
        O, I = wt.shape
        bs = 128
        if O % bs or I % bs:
            return wt
        wf = wt.float().view(O // bs, bs, I // bs, bs)
        amax = wf.abs().amax(dim=(1, 3)).clamp_min(1e-12)
        s = amax / 448.0
        q = (wf / s[:, None, :, None]).clamp(-448.0, 448.0).view(O, I).to(torch.float8_e4m3fn)
        del wf
        return FP8LinearWeight(q.contiguous(), s.contiguous(), block_size=bs)

    def unload_layer(self, l: int) -> None:
        if l in self._layers:
            del self._layers[l]

    # ---- [R3.4-S1] expert pruning by router mask ----
    def set_pruned(self, l: int, ids) -> None:
        """Mask layer l's experts `ids` out of routing (an empty list restores the original bias). The mask is the router
        bias gate_b = -inf, written IN PLACE (never rebound: captured CUDA graphs hold this tensor's address), so it
        changes only WHICH experts are selected: the routing weights stay the sigmoid scores of the 8 selected (kept)
        experts, renormalised and scaled by routed_scaling_factor - exactly 8 experts per token, graph shapes unchanged.
        Safe to call before or after graph capture (e.g. between requests); prefetch prediction and MTP see it too.
        [R3.4 review] `ids` REPLACES the layer's whole mask. At run time use the runner's set_prune(spec), which always
        adds the experts a pruned archive lacks (unmasking one would make the expert pool raise mid-request)."""
        ids = sorted({int(e) for e in ids})
        n_exp, k = int(self.cfg.n_routed_experts), int(self.cfg.num_experts_per_tok)
        if len(ids) > n_exp - k - 1:            # [R3.4 review] explicit raises (an assert vanishes under python -O)
            raise ValueError(f"layer {l}: {len(ids)} masked experts leave fewer than {k + 1} of {n_exp}")
        if not all(0 <= e < n_exp for e in ids):
            raise ValueError(f"layer {l}: expert ids outside [0, {n_exp})")
        if ids:
            self.prune_spec[l] = ids
        else:
            self.prune_spec.pop(l, None)
        if l in self._layers and "gate_b" in self._layers[l]:
            self._mask_gate(l, self._layers[l])

    def _mask_gate(self, l: int, w: Dict[str, WeightTensor]) -> None:
        gb = w["gate_b"]                        # fp32; read by route(), route_ids() and the captured graphs
        if l not in self._gate_b0:
            self._gate_b0[l] = gb.clone()
        gb.copy_(self._gate_b0[l])              # in place - never rebind (graphs hold this address)
        ids = self.prune_spec.get(l)
        if ids:
            gb[torch.tensor(ids, dtype=torch.long, device=gb.device)] = float("-inf")

    # ---- [R3.4-V] trimmed output head ----
    def set_vocab_keep(self, keep_ids) -> Dict[str, object]:
        """Keep only the lm_head rows of `keep_ids` (sorted unique original token ids). Logits then have K = len(keep_ids)
        entries in that order; token_ids() / argmax_token() map a head row back to its token id, expand_logits() rebuilds
        a full-vocabulary tensor (dropped rows -inf). The embedding table and the tokenizer stay complete. A BF16/FP32 head
        is sliced to [K, D]; a block-128 FP8 head keeps its e4m3 rows with one scale row per output row (row_scales, the
        row's original block scale -> per-row products identical to the full head) and is padded with zero rows to a
        multiple of 128 rows (final_logits / mtp_forward cut the logits back to K). Call once, before any CUDA-graph
        capture (the graphs read the head and vocab_ids storage)."""
        self.load_global()
        if self.vocab_ids is not None:
            raise RuntimeError("set_vocab_keep: the output head is already trimmed")
        if self.graphs_captured:                # [R3.4 review] captured graphs read the old head's storage
            raise RuntimeError("set_vocab_keep: CUDA graphs are already captured (they read the full head) - trim first")
        head = self.head
        V = int(head.qweight.shape[0] if isinstance(head, FP8LinearWeight) else head.shape[0])
        ids = torch.as_tensor(list(keep_ids) if not isinstance(keep_ids, torch.Tensor) else keep_ids,
                              dtype=torch.long).flatten().cpu()
        K = int(ids.numel())
        if K <= 0:                              # [R3.4 review] explicit raises (an assert vanishes under python -O)
            raise ValueError("empty keep list")
        if not bool((ids[1:] > ids[:-1]).all()):
            raise ValueError("keep ids must be sorted and unique")
        if not (int(ids[0]) >= 0 and int(ids[-1]) < V):
            raise ValueError(f"keep ids outside [0, {V})")
        missing_eos = [int(e) for e in getattr(self.cfg, "eos_token_ids", ()) if int(e) < V and not bool((ids == int(e)).any())]
        if missing_eos:
            raise ValueError(f"keep list drops EOS ids {missing_eos}")
        dev = head.qweight.device if isinstance(head, FP8LinearWeight) else head.device
        dids = ids.to(dev)
        bytes0 = self._w_bytes(head)
        if isinstance(head, FP8LinearWeight):
            bs = int(head.block_size)
            I = int(head.qweight.shape[1])
            Kp = ((K + 127) // 128) * 128
            q = torch.zeros((Kp, I), dtype=torch.uint8, device=dev)
            q[:K] = head.qweight.view(torch.uint8).index_select(0, dids)
            src_rows = dids if head.row_scales else torch.div(dids, bs, rounding_mode="floor")
            srow = head.scale_inv.float().index_select(0, src_rows)
            s = torch.zeros((Kp, int(srow.shape[1])), dtype=torch.float32, device=dev)
            s[:K] = srow
            new_head: WeightTensor = FP8LinearWeight(q.view(torch.float8_e4m3fn), s, block_size=bs, row_scales=True)
        else:
            new_head = head.index_select(0, dids).contiguous()
            Kp = K
        self.head = new_head
        del head
        self._head_rows = K
        self.vocab_rows_full = V
        self.vocab_ids = dids.contiguous()
        pos = torch.full((V,), -1, dtype=torch.long)
        pos[ids] = torch.arange(K, dtype=torch.long)
        self.vocab_pos = pos.to(dev)
        return {"vocab_rows": V, "kept": K, "head_rows": Kp, "head_bytes_before": int(bytes0),
                "head_bytes_after": int(self._w_bytes(self.head)), "fp8": isinstance(self.head, FP8LinearWeight)}

    def token_ids(self, idx: torch.Tensor) -> torch.Tensor:
        """[R3.4-V] head-row indices (e.g. an argmax over the logits) -> original token ids (identity on a full head)."""
        return idx if self.vocab_ids is None else self.vocab_ids[idx]

    def argmax_token(self, logits_row: torch.Tensor) -> int:
        """Greedy token id of one logits row (the argmax over the - possibly trimmed - head rows, mapped back)."""
        if self.vocab_ids is None:
            return int(torch.argmax(logits_row).item())
        return int(self.vocab_ids[torch.argmax(logits_row)].item())

    def expand_logits(self, logits: torch.Tensor, fill: float = float("-inf")) -> torch.Tensor:
        """[R3.4-V] trimmed logits [..., K] -> full-vocabulary [..., V] with `fill` in the dropped rows (identity on a
        full head)."""
        if self.vocab_ids is None:
            return logits
        out = logits.new_full((*logits.shape[:-1], int(self.vocab_rows_full)), fill)
        out[..., self.vocab_ids.to(logits.device)] = logits
        return out

    @staticmethod
    def _w_bytes(v: WeightTensor) -> int:
        if isinstance(v, FP8LinearWeight):
            return v.bytes()
        return v.numel() * v.element_size()

    def layer_param_bytes(self, l: int) -> int:
        return sum(self._w_bytes(v) for v in self.layer(l).values())

    def resident_bytes(self) -> int:
        n = sum(self.layer_param_bytes(l) for l in self._layers)
        for t in (self.embed, self.head, self.final_norm):
            if t is not None:
                n += self._w_bytes(t)
        return n

    def lin(self, x: torch.Tensor, w: WeightTensor) -> torch.Tensor:
        """Linear projection supporting both `FP8LinearWeight` (in-place dequant) and `torch.Tensor`."""
        if isinstance(w, FP8LinearWeight):
            O, I = w.qweight.shape
            n_tok = x.numel() // I
            if w.row_scales:                    # [R3.4-V] row-sliced FP8 head: per-row scale kernel (same arithmetic)
                if self.use_kernels and n_tok <= 4 and x.is_cuda and I % w.block_size == 0:
                    y = KR.fp8_gemv_rows(w.qweight, w.scale_inv, x.reshape(n_tok, I), w.block_size)
                    return y.view(*x.shape[:-1], O)
                w_mat = w.dequantize(self.act_dtype)
                return F.linear(x.to(self.act_dtype), w_mat).float()
            if (self.use_kernels and n_tok <= 4 and x.is_cuda and w.scale_inv.ndim == 2
                    and O % w.block_size == 0 and I % w.block_size == 0):
                # [R3.2-1.4] FP8 weights read once (1 B/weight) and dequantised in registers
                y = KR.fp8_gemv(w.qweight, w.scale_inv, x.reshape(n_tok, I), w.block_size)
                return y.view(*x.shape[:-1], O)
            w_mat = w.dequantize(self.act_dtype)
            return F.linear(x.to(self.act_dtype), w_mat).float()
        return F.linear(x.to(w.dtype), w).float()

    # ---- [R3.5-M1] row-locality predicates for the N=2 verify graphs (glm53_fold_graphs.GraphDecoder, pair set) ----
    # They mirror the dispatch conditions of lin() / mhc_prepare() / mhc_merge() (keep them in sync). Not used by any
    # existing path.
    def lin_row_local(self, w: WeightTensor, n_tok: int, is_cuda: bool) -> bool:
        """True when lin(x, w) on n_tok rows runs the FP8 GEMV kernel (fp8_gemv / fp8_gemv_rows). Every output element
        is then computed by one warp with the same lane split, FMA chain and warp reduction whatever the number of
        rows, so batching rows reads the weights once and leaves every row bit-identical to a 1-row call."""
        if not isinstance(w, FP8LinearWeight) or not (self.use_kernels and is_cuda and 1 <= int(n_tok) <= 4):
            return False
        O, I = w.qweight.shape
        if w.row_scales:
            return I % w.block_size == 0
        return w.scale_inv.ndim == 2 and O % w.block_size == 0 and I % w.block_size == 0

    def mhc_prepare_row_local(self, h: torch.Tensor) -> bool:
        """True when mhc_prepare(h) runs the fused kernel (one thread block per token: rows are independent)."""
        return bool(self.use_kernels and h.is_cuda and h.shape[2] == 4)

    @staticmethod
    def mhc_merge_row_local(residual_h: torch.Tensor) -> bool:
        """True when mhc_merge(..., residual_h) runs the fused kernel (one thread per output element)."""
        return bool(_FUSED_MERGE[0] and residual_h.is_cuda and residual_h.shape[-2] == 4)

    def rmsnorm(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps))

    @staticmethod
    def causal_conv_silu(
        x: torch.Tensor,
        w_conv: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x.shape
        K = w_conv.shape[-1]
        if conv_state is None:
            conv_state = x.new_zeros((B, K - 1, D), dtype=torch.float32)
        xp = torch.cat([conv_state, x.float()], dim=1)
        new_state = xp[:, -(K - 1):, :].contiguous()
        if S > 1:
            # [R3.7 Stage 7] with B = 1 the slice above is already contiguous, so .contiguous() returns a VIEW and the
            # stored state kept all of xp alive: S x D fp32 per conv, three convs per KDA layer, 35 layers (measured
            # 2026-11-14: 94 MiB per layer for a 960-token prompt, 3.3 GiB in all, for the whole request). A copy of
            # the K - 1 rows has the same values and frees the rest. S = 1 (decode) is left as it was.
            new_state = new_state.clone()
        if S == 1:
            out = (xp * w_conv.T.unsqueeze(0)).sum(dim=1, keepdim=True)
            return F.silu(out), new_state
        xt = xp.transpose(1, 2)
        out = F.conv1d(xt, w_conv.unsqueeze(1), groups=D).transpose(1, 2)
        return F.silu(out), new_state

    # ---- mHC Manifold-Constrained Hyper-Connection (dynamic, per token) ----
    # [PATCH R3-02] Port of HF `Glm5NextTextHyperConnection.forward` (transformers 5.16.1, validated in
    # Round 1/2 via glm53_real.py): pre/post/comb are computed PER TOKEN from the RMS-normalised flattened
    # streams through `fn`, plus `base`, with three separate scales; comb = softmax + Sinkhorn(hc_eps).
    def mhc_prepare(
        self,
        h: torch.Tensor,
        fn: torch.Tensor,
        base: torch.Tensor,
        scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hc = h.shape[2]
        eps_hc = float(self.cfg.hc_eps)
        if self.use_kernels and h.is_cuda and hc == 4:
            # [R3.2-1.4] one fused launch instead of ~130 (RMS, 24-way mix, sigmoid/softmax, Sinkhorn, collapse)
            return KR.mhc_prepare(h, fn, base, scale, self.eps, eps_hc, int(self.cfg.hc_sinkhorn_iters))
        flat = h.flatten(2).float()
        flat = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.eps)
        mix = F.linear(flat, fn.float())
        pre_w, post_w, comb_w = mix.split([hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = base.float().view(-1).split([hc, hc, hc * hc])
        sc = scale.float().view(-1)
        pre = torch.sigmoid(pre_w * sc[0] + pre_b) + eps_hc
        post = 2.0 * torch.sigmoid(post_w * sc[1] + post_b)
        comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * sc[2] + comb_b.view(hc, hc)
        comb = torch.softmax(comb_logits, dim=-1) + eps_hc
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps_hc)
        for _ in range(int(self.cfg.hc_sinkhorn_iters) - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + eps_hc)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + eps_hc)
        x_in = (pre.unsqueeze(-1) * h.float()).sum(dim=2)
        return post, comb, x_in

    @staticmethod
    def mhc_merge(
        post_w: torch.Tensor,
        comb_M: torch.Tensor,
        branch_y: torch.Tensor,
        residual_h: torch.Tensor,
    ) -> torch.Tensor:
        # [PATCH R3-02] HF: post (x) y + comb^T @ residual, per token
        if _FUSED_MERGE[0] and residual_h.is_cuda and residual_h.shape[-2] == 4:
            return KR.mhc_merge(post_w, comb_M, branch_y, residual_h)  # [R3.2-1.4] one launch
        return post_w.unsqueeze(-1) * branch_y.float().unsqueeze(-2) + torch.matmul(
            comb_M.transpose(-1, -2), residual_h.float()
        )

    # ---- KDA Linear Attention (Recurrent Decode + Vectorized WY Chunk-Parallel Prefill) ----
    def kda(self, l: int, x: torch.Tensor, cache: Optional[RealCache] = None) -> torch.Tensor:
        if os.environ.get('R36_FUSE_KDA','0')=='1' and self.use_kernels and x.is_cuda and x.shape[0]==1 and x.shape[1]<=2:
            from glm53_fold_fused_r36 import kda_eager
            return kda_eager(self,l,x,cache)
        w = self.layer(l)
        B, S, _ = x.shape
        H, hd = self.H_kda, self.hd_kda
        cs = cache.kda_conv.get(l) if cache is not None else None

        q, cq = self.causal_conv_silu(self.lin(x, w["q"]), w["qconv"], cs[0] if cs else None)
        k, ck = self.causal_conv_silu(self.lin(x, w["k"]), w["kconv"], cs[1] if cs else None)
        v, cv = self.causal_conv_silu(self.lin(x, w["v"]), w["vconv"], cs[2] if cs else None)

        q = q.view(B, S, H, hd)
        k = k.view(B, S, H, hd)
        v = v.view(B, S, H, hd)
        q = (q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)) * (hd ** -0.5)
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)

        g = (self.lin(self.lin(x, w["fa"]), w["fb"]) + w["dt_bias"]).view(B, S, H, hd)
        g = self.lower_bound * torch.sigmoid(torch.exp(w["A_log"]).view(1, 1, H, 1) * g)
        beta = torch.sigmoid(self.lin(x, w["b"]))  # [B, S, H]

        state = cache.kda_state.get(l) if cache is not None else None
        if state is None:
            state = x.new_zeros((B, H, hd, hd), dtype=torch.float32)

        if S <= 2:
            out = x.new_empty((B, S, H, hd), dtype=torch.float32)
            for t in range(S):
                state = state * g[:, t].exp().unsqueeze(-1)
                kv_mem = torch.matmul(k[:, t].unsqueeze(-2), state).squeeze(-2)
                delta = (v[:, t] - kv_mem) * beta[:, t].unsqueeze(-1)
                state = state + k[:, t].unsqueeze(-1) * delta.unsqueeze(-2)
                out[:, t] = torch.matmul(q[:, t].unsqueeze(-2), state).squeeze(-2)
        else:
            out = x.new_empty((B, S, H, hd), dtype=torch.float32)
            chunk_sz = self.cfg.kda_chunk_size
            for c_start in range(0, S, chunk_sz):
                c_end = min(c_start + chunk_sz, S)
                L = c_end - c_start
                g_c = g[:, c_start:c_end]
                k_c = k[:, c_start:c_end]
                q_c = q[:, c_start:c_end]
                v_c = v[:, c_start:c_end]
                b_c = beta[:, c_start:c_end]

                csum = torch.cumsum(g_c, dim=1)
                diff = csum[:, :, None, :, :] - csum[:, None, :, :, :]
                idx = torch.arange(L, device=x.device)
                strict_mask = (idx[:, None] > idx[None, :])[None, :, :, None, None]
                causal_mask = (idx[:, None] >= idx[None, :])[None, :, :, None, None]

                exp_strict = torch.exp(torch.where(strict_mask, diff, torch.full_like(diff, -1e9)))
                exp_strict = torch.where(strict_mask, exp_strict, torch.zeros_like(exp_strict))

                K_kk = torch.einsum("bihd,bijhd,bjhd->bhij", k_c, exp_strict, k_c)
                b_bh = b_c.permute(0, 2, 1).unsqueeze(-1)
                eye_L = torch.eye(L, dtype=torch.float32, device=x.device).view(1, 1, L, L)
                A_mat = eye_L + b_bh * K_kk

                k_init = k_c * torch.exp(csum)
                kv_init = torch.einsum("bhkv,blhk->bhlv", state, k_init)
                rhs = b_bh * (v_c.permute(0, 2, 1, 3) - kv_init)

                deltas = torch.linalg.solve_triangular(A_mat, rhs, upper=False, unitriangular=True)

                q_init = q_c * torch.exp(csum)
                out_init = torch.einsum("bhkv,blhk->bhlv", state, q_init)
                exp_causal = torch.exp(torch.where(causal_mask, diff, torch.full_like(diff, -1e9)))
                exp_causal = torch.where(causal_mask, exp_causal, torch.zeros_like(exp_causal))
                K_qk = torch.einsum("bihd,bijhd,bjhd->bhij", q_c, exp_causal, k_c)

                out_chunk = out_init + torch.matmul(K_qk, deltas)
                out[:, c_start:c_end] = out_chunk.permute(0, 2, 1, 3)

                state = state * torch.exp(csum[:, -1]).unsqueeze(-1)
                k_end = (k_c * torch.exp(csum[:, -1:] - csum)).permute(0, 2, 3, 1)
                state = state + torch.matmul(k_end, deltas)

        if cache is not None:
            cache.kda_state[l] = state
            cache.kda_conv[l] = (cq, ck, cv)

        gate = self.lin(self.lin(x, w["ga"]), w["gb"]).view(B, S, H, hd)
        o = self.rmsnorm(out, w["o_norm"]) * torch.sigmoid(gate)
        return self.lin(o.reshape(B, S, H * hd), w["o"])

    # ---- DSA (Multi-Head Latent Attention + K-Pool Indexer) ----
    def _indexer_mask(
        self,
        l: int,
        x: torch.Tensor,
        q_resid: torch.Tensor,
        ki_all: torch.Tensor,
        gs_all: torch.Tensor,
        q_pos: torch.Tensor,
        pool_keys: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if os.environ.get('R36_LONGCTX','0')=='1' and x.shape[1]>128:
            return torch.cat([self._indexer_mask(l,x[:,s:s+128],q_resid[:,s:s+128],ki_all,gs_all,q_pos[s:s+128],pool_keys=pool_keys)
                              for s in range(0,x.shape[1],128)],dim=1)
        # [R3.7 Stage 7] R37_INDEXER_QBLOCK=N (0 / unset = off): the same query blocking without R36_LONGCTX. The
        # indexer scores are [B, S, 32, T/4] fp32 = 32 x S x T bytes (2.69 GiB for one pass over 9,500 positions).
        iq = int(os.environ.get('R37_INDEXER_QBLOCK', '0') or 0)
        if iq > 0 and x.shape[1] > iq:
            return torch.cat([self._indexer_mask(l, x[:, s:s + iq], q_resid[:, s:s + iq], ki_all, gs_all, q_pos[s:s + iq], pool_keys=pool_keys)
                              for s in range(0, x.shape[1], iq)], dim=1)
        w = self.layer(l)
        cfg = self.cfg
        B, S, _ = x.shape
        T = ki_all.shape[1]
        P = cfg.index_kpool
        nh, hdi = cfg.index_n_heads, cfg.index_head_dim
        qi = self.lin(q_resid, w["i_wqb"]).view(B, S, nh, hdi)
        n_pools = (T + P - 1) // P
        pool_idx = torch.arange(n_pools * P, device=x.device).view(n_pools, P)
        valid_tok = pool_idx < T
        safe = pool_idx.clamp(0, T - 1)
        # [R3.7 Stage 7] pool_keys given (glm53_fold_longctx_r37: a static buffer [B, n_pools, hdi] that already holds
        # the key of every complete pool): the two gathers and the softmax over the whole length are skipped. With
        # pool_keys None the statements and their order are the unchanged ones.
        if pool_keys is None:
            gk = ki_all[:, safe]
            gg = gs_all[:, safe]
        pool_valid = valid_tok.all(-1)
        if pool_keys is None:
            logits = gg + w["i_ape"].view(1, 1, P, hdi)
            logits = logits.masked_fill(~valid_tok.view(1, n_pools, P, 1), float("-inf"))
            probs = torch.nan_to_num(torch.softmax(logits, dim=2))
            pool_keys = (probs * gk).sum(dim=2)
        scores = torch.relu(torch.matmul(qi, pool_keys.transpose(-1, -2).unsqueeze(1)) * (hdi ** -0.5))
        weights = self.lin(x, w["i_wp"]) * (nh ** -0.5)
        index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)
        pool_end = pool_idx[:, -1].clamp(0, T - 1)
        pool_visible = pool_end.view(1, 1, n_pools) <= q_pos.view(1, S, 1)
        valid_cand = pool_visible & pool_valid.view(1, 1, n_pools)
        index_scores = index_scores.masked_fill(~valid_cand, torch.finfo(index_scores.dtype).min)
        select_k = min(cfg.index_topk // P, n_pools)
        selected = index_scores.topk(select_k, dim=-1).indices
        sel_valid = valid_cand.expand(B, S, n_pools).gather(-1, selected)
        sel_tokens = pool_idx[selected]
        sel_tokens = sel_tokens.masked_fill(~sel_valid.unsqueeze(-1), -1).flatten(-2)
        mask = torch.zeros((B, S, T + 1), dtype=torch.bool, device=x.device)
        mask.scatter_(-1, sel_tokens.masked_fill(sel_tokens < 0, T), True)
        if cfg.index_kpool_always_select_tail:
            visible_count = q_pos + 1
            tail_count = visible_count.remainder(P)
            tail_start = visible_count - tail_count
            offs = torch.arange(P - 1, device=x.device)
            tail_idx = tail_start.unsqueeze(-1) + offs.unsqueeze(0)
            tail_valid = offs.unsqueeze(0) < tail_count.unsqueeze(-1)
            tail_idx = tail_idx.masked_fill(~tail_valid, T)
            mask.scatter_(-1, tail_idx.view(1, S, P - 1).expand(B, S, P - 1), True)
        mask = mask[..., :T]
        self.indexer_checks["indexer_calls"] += 1
        if self.indexer_diagnostics and T <= cfg.index_topk:
            causal = torch.arange(T, device=x.device).view(1, 1, T) <= q_pos.view(1, S, 1)
            if bool((mask == causal).all()):
                self.indexer_checks["indexer_equals_causal"] += 1
        return mask

    def dsa(self, l: int, x: torch.Tensor, cache: Optional[RealCache] = None) -> torch.Tensor:
        w = self.layer(l)
        B, S, _ = x.shape
        H, qk, vd = self.H, self.qk, self.vd
        q_resid = self.rmsnorm(self.lin(x, w["qa"]), w["qa_norm"])
        q = self.lin(q_resid, w["qb"]).view(B, S, H, qk)
        kv_new = self.rmsnorm(self.lin(x, w["kva"]), w["kva_norm"])
        ki_new = F.layer_norm(self.lin(x, w["i_wk"]), (self.cfg.index_head_dim,), w["i_kn_w"], w["i_kn_b"], eps=1e-6)
        gs_new = self.lin(x, w["i_gate"])
        hist = getattr(cache, "r38", None) if cache is not None else None
        if hist is not None and l in hist.dsa:
            # [R3.8] coded history (glm53_fold_qstate_r38): the rows go to the decoder's buffers, every query attends
            # over the rows the indexer selects for it. cache.pos is the first position of this pass (forward()
            # advances it after the last layer).
            from glm53_fold_qstate_r38 import prefill_rows, prefill_attend
            if B != 1 or not self.dsa_absorbed:
                raise RuntimeError("the coded history needs batch 1 and the absorbed DSA path")
            t0 = int(cache.pos)
            prefill_rows(self, l, hist, t0, kv_new, ki_new, gs_new)
            W_k, W_v = self._dsa_kv_split(l, w["kvb"])
            q_abs = torch.einsum("bshd,hdc->bshc", q.float(), W_k)
            o_lat = prefill_attend(self, l, hist, t0, x, q_resid, q_abs, qk)
            out = torch.einsum("bshc,hdc->bshd", o_lat, W_v)
            return self.lin(out.reshape(B, S, H * vd), w["o"])
        if cache is not None and l in cache.dsa_kv:
            kv_all = torch.cat([cache.dsa_kv[l], kv_new], dim=1)
            ki_all = torch.cat([cache.dsa_ki[l], ki_new], dim=1)
            gs_all = torch.cat([cache.dsa_gs[l], gs_new], dim=1)
        else:
            kv_all, ki_all, gs_all = kv_new, ki_new, gs_new
        if cache is not None:
            cache.dsa_kv[l] = kv_all
            cache.dsa_ki[l] = ki_all
            cache.dsa_gs[l] = gs_all
        T = kv_all.shape[1]
        q_pos = torch.arange(T - S, T, device=x.device)
        if self.dsa_skip_indexer_short and T <= self.cfg.index_topk:
            # [R3.2-1.3] with T <= index_topk the K-pool indexer selects every visible pool plus the tail,
            # i.e. exactly the causal mask (the Round-3.1 diagnostic `indexer_equals_causal` confirmed it).
            mask = None if S == 1 else (
                torch.arange(T, device=x.device).view(1, 1, T) <= q_pos.view(1, S, 1)
            ).expand(B, S, T)
        else:
            mask = self._indexer_mask(l, x, q_resid, ki_all, gs_all, q_pos)
        tap = getattr(self, "_r37_mask_tap", None)
        if tap is not None and S == 1:       # [R3.7 Stage 7] gates only (longctx_gate2_r37.py): the positions this
            #                                   single query may attend, to compare with the graph step's selection
            tap[l] = torch.arange(T, device=x.device) if mask is None else mask[0, 0].nonzero().flatten()
            if tap.get("inputs") is not None:    # the query's inputs too (the selection on identical inputs)
                tap["inputs"][l] = (x.clone(), q_resid.clone(), ki_all.clone(), gs_all.clone())
        if self.dsa_absorbed:
            # [R3.2-1.3] MLA absorption (qk_rope_head_dim == 0, so no decoupled RoPE part):
            #   q.k_t = (W_k^T q) . c_t   and   sum_t p_t v_t = W_v (sum_t p_t c_t)
            W_k, W_v = self._dsa_kv_split(l, w["kvb"])                      # [H, qk, C], [H, vd, C]
            q_abs = torch.einsum("bshd,hdc->bshc", q.float(), W_k)         # [B, S, H, C]
            kv_f = kv_all.float()
            qb = self._dsa_qblock(B, S, T) if getattr(self, "dsa_qblock_mib", 0.0) > 0 else 0
            if qb:                                       # [R3.5-T H7] query blocks (budget exceeded, see __init__)
                outs = []
                for s0 in range(0, S, qb):
                    s1 = min(S, s0 + qb)
                    sc = torch.einsum("bshc,btc->bhst", q_abs[:, s0:s1], kv_f) * (qk ** -0.5)
                    if mask is not None:
                        sc = sc.masked_fill(~mask[:, s0:s1].unsqueeze(1), torch.finfo(sc.dtype).min)
                    outs.append(torch.einsum("bhst,btc->bshc", torch.softmax(sc, dim=-1), kv_f))
                    del sc
                o_lat = torch.cat(outs, dim=1)                              # [B, S, H, C]
                del outs
            else:
                scores = torch.einsum("bshc,btc->bhst", q_abs, kv_f) * (qk ** -0.5)
                if mask is not None:
                    scores = scores.masked_fill(~mask.unsqueeze(1), torch.finfo(scores.dtype).min)
                probs = torch.softmax(scores, dim=-1)
                o_lat = torch.einsum("bhst,btc->bshc", probs, kv_f)        # [B, S, H, C]
            out = torch.einsum("bshc,hdc->bshd", o_lat, W_v)               # [B, S, H, vd]
            return self.lin(out.reshape(B, S, H * vd), w["o"])
        kvb = self.lin(kv_all, w["kvb"]).view(B, T, H, qk + vd)
        k, v = kvb[..., :qk], kvb[..., qk:]
        scores = torch.einsum("bshd,bthd->bhst", q, k) * (qk ** -0.5)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhst,bthd->bshd", probs, v)
        return self.lin(out.reshape(B, S, H * vd), w["o"])

    def _dsa_qblock(self, B: int, S: int, T: int) -> int:
        """[R3.5-T H7] query rows per block when ONE fp32 score tensor [B, H, S, T] exceeds R35_DSA_QBLOCK_MIB (a block's
        scores then fit it; at least 1 row), else 0 (= the unblocked base statements)."""
        budget = int(float(self.dsa_qblock_mib) * 2**20)
        per_row = int(B) * int(self.H) * int(T) * 4
        if S <= 1 or budget <= 0 or int(S) * per_row <= budget:
            return 0
        qb = max(1, budget // per_row)
        st = self.dsa_qblock_stats
        st["calls"] += 1
        st["blocks"] += -(-int(S) // qb)
        return qb

    def _dsa_kv_split(self, l: int, kvb_w: WeightTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """[R3.2-1.3] kv_b [H*(qk+vd), C] split once into FP32 W_k [H, qk, C] and W_v [H, vd, C]."""
        if l not in self._dsa_wkv:
            Wf = kvb_w.dequantize(torch.float32) if isinstance(kvb_w, FP8LinearWeight) else kvb_w.float()
            Wf = Wf.view(self.H, self.qk + self.vd, -1)
            self._dsa_wkv[l] = (Wf[:, : self.qk, :].contiguous(), Wf[:, self.qk :, :].contiguous())
        return self._dsa_wkv[l]

    # ---- FFN & Routing ----
    def swiglu_mlp(self, x: torch.Tensor, w1: WeightTensor, w3: WeightTensor, w2: WeightTensor) -> torch.Tensor:
        lim = self.cfg.swiglu_limit
        gate = self.lin(x, w1).clamp(max=lim)
        up = self.lin(x, w3).clamp(-lim, lim)
        return self.lin(F.silu(gate) * up, w2)

    def shared_expert(self, l: int, x_flat: torch.Tensor) -> torch.Tensor:
        w = self.layer(l)
        return self.swiglu_mlp(x_flat, w["sh_w1"], w["sh_w3"], w["sh_w2"])

    def route(self, l: int, x_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if getattr(self,'r36_router',None) is not None:
            return self.r36_router.route(self,l,x_flat)
        w = self.layer(l)
        logits = F.linear(x_flat.float(), w["gate_w"].float())
        scores = torch.sigmoid(logits)
        choices = scores + w["gate_b"].float().unsqueeze(0)
        _, top_idx = torch.topk(choices, k=self.cfg.num_experts_per_tok, dim=-1, sorted=False)
        top_w = torch.gather(scores, 1, top_idx)
        top_w = (top_w / (top_w.sum(dim=-1, keepdim=True) + 1e-20)) * self.cfg.routed_scaling_factor
        return top_w, top_idx

    def route_ids(self, l: int, x_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """[R3.4-2c] next-layer prediction: route()'s top-k ids (the same ops up to the top-k) and their selection
        scores (sigmoid + bias), without the gather / normalisation that only the routing weights need."""
        if getattr(self,'r36_router',None) is not None:
            weights,ids=self.r36_router.route(self,l,x_flat)
            return ids,weights
        w = self.layer(l)
        logits = F.linear(x_flat.float(), w["gate_w"].float())
        scores = torch.sigmoid(logits)
        choices = scores + w["gate_b"].float().unsqueeze(0)
        sel, top_idx = torch.topk(choices, k=self.cfg.num_experts_per_tok, dim=-1, sorted=False)
        return top_idx, sel

    def attn_block(self, l: int, h: torch.Tensor, cache: Optional[RealCache] = None) -> torch.Tensor:
        w = self.layer(l)
        post, comb, x_in = self.mhc_prepare(h, w["hc_attn_fn"], w["hc_attn_base"], w["hc_attn_scale"])  # [PATCH R3-02]
        xn = self.rmsnorm(x_in, w["in_norm"])
        y = self.kda(l, xn, cache) if self.is_kda(l) else self.dsa(l, xn, cache)
        return self.mhc_merge(post, comb, y, h)

    def ffn_prepare(self, l: int, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w = self.layer(l)
        post, comb, xf = self.mhc_prepare(h, w["hc_ffn_fn"], w["hc_ffn_base"], w["hc_ffn_scale"])  # [PATCH R3-02]
        return post, comb, self.rmsnorm(xf, w["post_norm"])

    def layer_forward(
        self,
        l: int,
        h: torch.Tensor,
        cache: Optional[RealCache] = None,
        moe_fn: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
        moe_fn_includes_shared: bool = False,
    ) -> torch.Tensor:
        h = self.attn_block(l, h, cache)
        if getattr(self, "_mt", None) is not None:   # [R3.7] R37_MEMTRACE: peak of the attention part, then of the rest
            self._mt["attn"] = torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        w = self.layer(l)
        residual = h
        post, comb, x_ffn = self.ffn_prepare(l, h)
        B, S, D = x_ffn.shape
        if not self.cfg.is_moe_layer(l):
            y = self.swiglu_mlp(x_ffn, w["w1"], w["w3"], w["w2"])
        else:
            x_flat = x_ffn.reshape(-1, D)
            assert moe_fn is not None, "moe_fn required for sparse MoE layers"
            y_r = moe_fn(l, x_flat)
            y = y_r if moe_fn_includes_shared else (y_r + self.shared_expert(l, x_flat))
            y = y.view(B, S, D)
        return self.mhc_merge(post, comb, y, residual)

    def embed_streams(self, ids: torch.Tensor) -> torch.Tensor:
        self.load_global()
        if isinstance(self.embed, FP8LinearWeight):
            emb_w = self.embed.dequantize(torch.float32)
            e = emb_w[ids]
        elif self.embed.device != ids.device:
            # [R3.2-1.2] host-resident table: gather the rows on the CPU, ship only [B, S, D]
            e = self.embed[ids.cpu()].to(ids.device, non_blocking=False).float()
        else:
            e = self.embed[ids].float()
        return e.unsqueeze(2).expand(-1, -1, self.hc, -1).contiguous()

    def final_logits(self, h: torch.Tensor, last_only: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        self.load_global()
        collapsed = h.mean(dim=2)
        if last_only:
            collapsed_out = collapsed[:, -1:, :]
        else:
            collapsed_out = collapsed
        hidden = self.rmsnorm(collapsed_out, self.final_norm)
        if self._head_rows is not None:          # [R3.4-V] trimmed head: K logits (padded FP8 rows cut off)
            return self.lin(hidden, self.head)[..., : self._head_rows], collapsed
        return self.lin(hidden, self.head), collapsed

    def forward(
        self,
        ids: torch.Tensor,
        cache: Optional[RealCache] = None,
        moe_fn: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
        last_only: bool = True,
        moe_fn_includes_shared: bool = False,
        max_layers: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # [R3.7 Stage 7] R37_PREFILL_CHUNK=N (0 / unset = off): a pass over more than N positions with a cache is run as
        # consecutive passes of at most N positions (the cache carries the KDA state and the DSA history between
        # them). Every pass streams the experts it needs once, so N should be as large as the GPU memory left by the
        # expert pool allows (measured 2026-11-14, 16 GiB pool: 2,292 positions peak +1.5 GiB, 4,740 +2.9 GiB).
        # It takes precedence over the 128..512-position pieces of R36_LONGCTX.
        r37_chunk = int(os.environ.get('R37_PREFILL_CHUNK', '0') or 0)
        if r37_chunk > 0 and cache is not None and ids.shape[1] > r37_chunk:
            hidden, logits_all, lg = [], [], None
            for s0 in range(0, ids.shape[1], r37_chunk):
                lg, hc_ = self.forward(ids[:, s0:s0 + r37_chunk], cache, moe_fn, last_only, moe_fn_includes_shared,
                                       max_layers)
                hidden.append(hc_ if not last_only else hc_[:, -1:, :].clone())
                if not last_only:
                    logits_all.append(lg)
            if last_only:
                return lg, hidden[-1]
            return torch.cat(logits_all, 1), torch.cat(hidden, 1)
        if os.environ.get('R36_LONGCTX','0')=='1' and cache is not None and r37_chunk <= 0:
            from glm53_fold_longctx_r36 import prefill_chunk,forward_chunked
            if ids.shape[1]>prefill_chunk():
                return forward_chunked(self,ids,cache,moe_fn,last_only,moe_fn_includes_shared,max_layers)
        h = self.embed_streams(ids)
        n_run = max_layers if max_layers is not None else self.cfg.num_hidden_layers
        long_pass = h.is_cuda and ids.shape[1] > 4
        trace = long_pass and getattr(self, "_r37_memtrace", False)
        if trace:                                # [R3.7] R37_MEMTRACE
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            rows = []
            self._mt = {}
        for l in range(n_run):
            if trace:
                torch.cuda.reset_peak_memory_stats()
            h = self.layer_forward(l, h, cache, moe_fn, moe_fn_includes_shared)
            if trace:
                rows.append((l, self._mt.get("attn", 0) - base, torch.cuda.max_memory_allocated() - base,
                             torch.cuda.memory_allocated() - base, torch.cuda.memory_reserved()))
                if os.environ.get("R37_MEMTRACE", "0") == "2":      # one line per layer, as it happens
                    r = rows[-1]
                    print(f"[memtrace] layer {l} ({'kda' if self.is_kda(l) else 'dsa'}"
                          f"{', moe' if self.cfg.is_moe_layer(l) else ''}): attention peak +{r[1] / 2**20:.0f} MiB, "
                          f"rest peak +{r[2] / 2**20:.0f}, kept +{r[3] / 2**20:.0f}, reserved {r[4] / 2**20:.0f}",
                          flush=True)
            if long_pass and getattr(self, "_r37_empty_cache", False):   # [R3.7] R37_EMPTY_CACHE
                torch.cuda.empty_cache()
        if trace:
            self._mt = None
            mib = 2**20
            top_a = sorted(rows, key=lambda r: -r[1])[:3]
            top_f = sorted(rows, key=lambda r: -r[2])[:3]
            print(f"[memtrace] pass of {ids.shape[1]} positions: allocated before {base / mib:.0f} MiB; "
                  f"attention peak +{max(r[1] for r in rows) / mib:.0f} MiB (layers "
                  f"{[(r[0], round(r[1] / mib)) for r in top_a]}); rest-of-layer peak +{max(r[2] for r in rows) / mib:.0f} "
                  f"MiB (layers {[(r[0], round(r[2] / mib)) for r in top_f]}); kept after the last layer "
                  f"+{rows[-1][3] / mib:.0f} MiB; reserved at most {max(r[4] for r in rows) / mib:.0f} MiB, "
                  f"at the end {rows[-1][4] / mib:.0f} MiB", flush=True)
        if cache is not None:
            cache.pos += ids.shape[1]
        return self.final_logits(h, last_only=last_only)

    def mtp_forward(
        self,
        next_ids: torch.Tensor,
        h_last: torch.Tensor,
        cache: Optional[RealCache] = None,
        moe_fn: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
        moe_fn_includes_shared: bool = False,
    ) -> torch.Tensor:
        l = self.cfg.num_hidden_layers
        w = self.layer(l)
        self.load_global()
        if isinstance(self.embed, FP8LinearWeight):
            emb_next = self.embed.dequantize(torch.float32)[next_ids]
        elif self.embed.device != next_ids.device:
            # [R3.2-1.2] host-resident embedding table (as in embed_streams): gather on the CPU, ship the rows
            emb_next = self.embed[next_ids.cpu()].to(next_ids.device).float()
        else:
            emb_next = self.embed[next_ids].float()
        e = self.rmsnorm(emb_next, w["enorm"])
        hn = self.rmsnorm(h_last, w["hnorm"])
        x = self.lin(torch.cat([e, hn], dim=-1), w["eh_proj"])
        a = self.dsa(l, self.rmsnorm(x, w["in_norm"]), cache)
        x = x + a
        xf = self.rmsnorm(x, w["post_norm"])
        B, S, D = xf.shape
        x_flat = xf.reshape(-1, D)
        assert moe_fn is not None
        y_r = moe_fn(l, x_flat)
        y = y_r if moe_fn_includes_shared else (y_r + self.shared_expert(l, x_flat))
        x = x + y.view(B, S, D)
        if self._head_rows is not None:          # [R3.4-V] MTP drafts with the same trimmed head
            return self.lin(self.rmsnorm(x, w["shared_head_norm"]), self.head)[..., : self._head_rows]
        return self.lin(self.rmsnorm(x, w["shared_head_norm"]), self.head)
