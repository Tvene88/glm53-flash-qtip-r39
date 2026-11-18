#!/usr/bin/env python3
"""
glm53_fold_generate.py
======================
End-to-End Inference, Teacher-Forced Perplexity, Batched `N=2` MTP Speculative Decoding,
`--doctor` Budget Verifier, and OpenAI-Compatible HTTP Server (`/v1/chat/completions`)
for Round-3 `APEX-FOLD-FP16`.

Fixes Every Runtime Bottleneck Identified in the Round-2 Post-Mortem:
---------------------------------------------------------------------
1. NATIVE `FP8_E4M3` BACKBONE (`--backbone-mode fp8`, default):
   Keeps non-expert 2D weights in 1-byte `FP8` in VRAM (`~8.65 GiB` instead of `17.13 GiB`),
   freeing `~8.5 GiB` of RTX 5090 VRAM so **all 43 MoE layers fit in VRAM** under
   `vram_100pct_32gb` (and **21 MoE layers** fit in VRAM under `heterogeneous_96gb`, vs only 4 in v4.0!).
2. ZERO-ALLOCATION PRE-TABULATED LUT-GEMV (`Path A`, `N <= 4`) & GROUPED PREFILL (`Path B`, `N > 4`):
   - Prefill (`N > 4`) groups tokens by unique active expert `uniq_e` and runs batched Tensor-Core
     `torch.mm` alongside vectorized chunk-parallel WY KDA prefill.
   - Decode (`N = 1`) and Batched MTP Verification (`N = 2`) use `T_m = x_rot @ C_m.T` L1/L2 scalar
     gathers in a single coalesced slab fetch.
3. TRUE BATCHED `N=2` MTP SPECULATIVE VERIFICATION:
   Instead of verifying draft tokens in a sequential token loop (which doubled MoE cost in v4.0),
   `generate_with_mtp` verifies `[last_tok, draft_tok]` in a single `N=2` forward pass!
"""

from __future__ import annotations
import argparse
import ctypes
import json
import math
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, List, Optional, Tuple
import numpy as np
from glm53_torch_compat import torch, nn, F

from glm53_fold_backbone import (
    RealCache,
    RealGLM53Backbone,
    RealGLM53Checkpoint,
    human_bytes,
)
from glm53_fold_config import (
    APEX_FOLD_PRESETS,
    GLM53FlashModelConfig,
    compute_exact_hardware_budget,
)
from glm53_fold_encoder import ApexFoldDiskStore
from glm53_fold_runtime import (
    ApexFoldLayerRuntime,
    ApexFoldPlacementManager,
    PinnedSlabDMAEngine,
)


def query_host_memory_gib() -> Dict[str, float]:
    """Queries Windows / Linux process working set and system available RAM accurately."""
    out = {"working_set_gib": 0.0, "avail_ram_gib": 0.0}
    try:
        if os.name == "nt":
            class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                ]
            pmc = PROCESS_MEMORY_COUNTERS_EX()
            pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
            h_proc = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(h_proc, ctypes.byref(pmc), pmc.cb):
                out["working_set_gib"] = round(pmc.WorkingSetSize / (1024.0 ** 3), 2)
                out["private_gib"] = round(pmc.PrivateUsage / (1024.0 ** 3), 2)
        else:
            with open("/proc/self/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        kb = int(line.split()[1])
                        out["working_set_gib"] = round(kb / (1024.0 ** 2), 2)
    except Exception:
        pass
    return out


class RealTokenizer:
    """Wrapper around Hugging Face `AutoTokenizer` with chat-template formatting and fallback."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self._tok = None
        try:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        except Exception:
            self._tok = None

    def encode_chat(self, prompt: str, thinking: bool = False, effort: str = "low") -> List[int]:
        if self._tok is not None:
            msgs = [{"role": "user", "content": prompt}]
            try:
                # [PATCH R3-04] transformers 5.x returns a BatchEncoding for tokenize=True, which the
                # isinstance(list) check silently discarded (prompt sent with NO chat template). Render the
                # template to text and encode it. The GLM-5.3 generation prompt ends with
                # "<|assistant|><think>"; thinking-off closes it with "</think>" (as the template does for
                # completed assistant turns).
                text = self._tok.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=thinking,
                    reasoning_effort=effort,
                )
                if isinstance(text, str):
                    if not thinking and not text.rstrip().endswith("</think>"):
                        text = text + "</think>"
                    return self._tok.encode(text, add_special_tokens=False)
            except Exception:
                pass
            return self._tok.encode(prompt)
        return [int(b) + 256 for b in prompt.encode("utf-8")]

    def encode_raw(self, text: str) -> List[int]:
        if self._tok is not None:
            return self._tok.encode(text, add_special_tokens=False)
        return [int(b) + 256 for b in text.encode("utf-8")]

    def decode(self, ids: List[int]) -> str:
        if self._tok is not None:
            return self._tok.decode(ids, skip_special_tokens=False)
        return "".join(chr(max(32, min(126, i - 256))) for i in ids)

    @property
    def eos_ids(self) -> Set[int]:
        # [PATCH R3-05] GLM-5.3-Flash stops on any of generation_config.json's three EOS ids
        # (the shipped fallback {151329, 151336, 151338} are GLM-4 ids).
        ids = set(GLM53FlashModelConfig().eos_token_ids)
        if self._tok is not None and getattr(self._tok, "eos_token_id", None) is not None:
            e = self._tok.eos_token_id
            ids |= set(e) if isinstance(e, list) else {int(e)}
        return ids


class _HostDone:
    """[R3.5-T] stand-in for a completed CUDA event (CPU runs of the runner code)."""

    def synchronize(self) -> None:
        return None


def load_prune_spec(src, cfg: Optional[GLM53FlashModelConfig] = None) -> Dict[int, List[int]]:
    """[R3.4-S4] expert prune spec -> {layer: sorted expert ids}. `src` = a JSON path or a dict, format
    {"layers": {"<layer>": [expert ids], ...}, ...metadata} (a bare {"<layer>": [...]} mapping is accepted too).
    Validated: MoE layers only, ids in [0, n_routed_experts), at least num_experts_per_tok + 1 experts left per layer."""
    cfg = cfg or GLM53FlashModelConfig()
    d = json.load(open(src, "r", encoding="utf-8")) if isinstance(src, (str, os.PathLike)) else dict(src)
    lay = d.get("layers", d) if isinstance(d, dict) else None
    if not isinstance(lay, dict):
        raise ValueError(f"prune spec {src}: expected {{'layers': {{layer: [ids]}}}}")
    moe = set(cfg.moe_layer_ids)
    n_exp, k = int(cfg.n_routed_experts), int(cfg.num_experts_per_tok)
    out: Dict[int, List[int]] = {}
    for key, ids in lay.items():
        l = int(key)
        if l not in moe:
            raise ValueError(f"prune spec: layer {l} is not a MoE layer")
        s = sorted({int(e) for e in ids})
        if s and (s[0] < 0 or s[-1] >= n_exp):
            raise ValueError(f"prune spec: layer {l} has expert ids outside [0, {n_exp})")
        if len(s) > n_exp - k - 1:
            raise ValueError(f"prune spec: layer {l} masks {len(s)} experts (max {n_exp - k - 1})")
        if s:
            out[l] = s
    return out


def load_vocab_keep(src, vocab_rows: Optional[int] = None, tokenizer_json: Optional[str] = None) -> List[int]:
    """[R3.4-V] keep list -> sorted unique token ids. `src` = a JSON path (r34_vocab_keep.py output: {"keep_ids": [...]},
    or a plain list) or a sequence of ids.
    [R3.4 review] a keep-list JSON is tied to the head and tokenizer it was built for: refused (ValueError) when its
    "vocab_rows" differs from `vocab_rows` (the lm_head rows) or its "tokenizer_sha256" from the sha256 of
    `tokenizer_json` (each check runs when both sides are known)."""
    if isinstance(src, (str, os.PathLike)):
        d = json.load(open(src, "r", encoding="utf-8"))
        ids = d["keep_ids"] if isinstance(d, dict) else d
        if isinstance(d, dict):
            if vocab_rows is not None and d.get("vocab_rows") is not None and int(d["vocab_rows"]) != int(vocab_rows):
                raise ValueError(f"keep list {src} was built for {d['vocab_rows']} head rows, this head has {vocab_rows}")
            if tokenizer_json and d.get("tokenizer_sha256") and os.path.exists(tokenizer_json):
                import hashlib
                sha = hashlib.sha256(open(tokenizer_json, "rb").read()).hexdigest()
                if sha != d["tokenizer_sha256"]:
                    raise ValueError(f"keep list {src} was built for another tokenizer (sha256 {d['tokenizer_sha256'][:16]}, "
                                     f"{tokenizer_json} has {sha[:16]})")
    else:
        ids = src
    return sorted({int(i) for i in ids})


class ApexFoldMoEModelRunner:
    """
    Coordinates the `RealGLM53Backbone` (`fp8`/`bf16`) and all 43 `ApexFoldLayerRuntime`
    modules with automatic VRAM/Host placement and shared `PinnedSlabDMAEngine`.
    [R3.4 shrink] prune_spec (S4: router masks, + every expert a pruned archive file lacks), load_mtp=False (E1: the MTP
    layer 45 is neither loaded from the backbone nor from the expert archive), vocab_keep (V: trimmed output head).
    """

    # [R3.5-S1] generate(serve_state=) hooks present (glm53_fold_serve.GLMRunnerAdapter reads this): "state" = H1-H4,
    # "skip_prefill" = H5
    SERVE_HOOKS = ("state", "skip_prefill")

    def __init__(
        self,
        backbone_dir: str,
        fold_dir: str,
        device: str = "cuda",
        backbone_mode: str = "fp8",
        vram_budget_gib: float = 30.6,
        cold_mode: str = "mmap",  # "mmap" (zero-copy .npz) or "ram"
        hobbit_tau: float = 1.0,
        max_layers: Optional[int] = None,
        fast: bool = True,
        expert_pool_gib: float = 0.0,
        prefetch: bool = True,
        sync_timing: bool = True,
        fp8_diet: bool = False,
        host_cache_gib: float = 0.0,
        prefix_cache: bool = False,
        prune_spec=None,
        load_mtp: bool = True,
        vocab_keep=None,
    ):
        self.cfg = GLM53FlashModelConfig()
        self.device = torch.device(device)
        self.load_mtp = bool(load_mtp)      # [R3.4-E1] False: skip layer 45 (backbone + expert file)
        self.prune_masks: Dict[int, List[int]] = {}   # [R3.4-S4] layer -> masked expert ids actually applied
        self.archive_missing: Dict[int, List[int]] = {}   # [R3.4-S2] experts a (pruned) archive file has no row for
        self.vocab_info = None              # [R3.4-V]
        self.hobbit_tau = hobbit_tau
        self.max_layers = max_layers
        self.fast = fast                    # [R3.2] NVRTC kernels + rotated-domain prefill
        self.expert_pool_gib = float(expert_pool_gib)
        self.host_cache_gib = float(host_cache_gib)  # [R3.2-1.6c] pinned host L2 cache
        self.prefetch = prefetch            # [R3.2-1.6] next-layer router pre-gating into the pool
        self.sync_timing = sync_timing      # per-component timing needs device syncs (off = true speed)
        self.pool = None
        self.graph_decoder = None           # [R3.2-1.5] set when --graphs is given
        # [R3.7 Stage 7] R37_DUAL_GRAPHS=1 with R36_LONGCTX=1 (main()): two decoders. graph_decoder_std is the
        # unchanged window decoder (t_max 2,048, no indexer caches: the Round 3.5 graphs), graph_decoder_long the
        # long-context one. generate() points graph_decoder at the one that serves the request
        # (select_graph_decoder); None = one decoder, as before.
        self.graph_decoder_std = None
        self.graph_decoder_long = None
        # [R3.5-T] time-to-first-token options (item 10a prefill prefetch, set with the pool below; H6 continuation past
        # the graph window) - all off by default; _prefill_probe: gates only (generate() hands it the prefill result)
        self.prefill_prefetch = False
        self.prefill_pf_min = 1
        self.prefill_pf_max = 0
        self._pf_cnt_host = None
        self._pf_stats = {"layers": 0, "predicted": 0, "queued": 0}
        self.continue_past_tmax = os.environ.get("R35_CONTINUE_PAST_TMAX", "0") == "1"
        self._prefill_probe = None

        t0 = time.perf_counter()
        self.ckpt = RealGLM53Checkpoint(backbone_dir)
        self.bb = RealGLM53Backbone(self.ckpt, self.cfg, device=device, backbone_mode=backbone_mode)
        self.bb.fp8_diet = bool(fp8_diet)
        if not fast:
            self.bb.use_kernels = False
            self.bb.embed_on_host = False
            self.bb.dsa_absorbed = False
            self.bb.dsa_skip_indexer_short = False
            import glm53_fold_backbone as _FBm
            _FBm._FUSED_MERGE[0] = False
        self.bb.load_global()
        if vocab_keep is not None:          # [R3.4-V] before the layers load: the full head is freed first
            head_rows = int(getattr(self.bb.head, "qweight", self.bb.head).shape[0])
            self.vocab_info = self.bb.set_vocab_keep(load_vocab_keep(
                vocab_keep, vocab_rows=head_rows, tokenizer_json=os.path.join(backbone_dir, "tokenizer.json")))
            print(f"[load] [R3.4-V] output head trimmed to {self.vocab_info['kept']} of {self.vocab_info['vocab_rows']} "
                  f"token rows ({self.vocab_info['head_rows']} head rows, {'FP8' if self.vocab_info['fp8'] else 'dense'}): "
                  f"{human_bytes(self.vocab_info['head_bytes_before'])} -> {human_bytes(self.vocab_info['head_bytes_after'])}",
                  flush=True)
        n_load = max_layers if max_layers is not None else (self.cfg.num_hidden_layers + self.cfg.num_nextn_predict_layers)
        for l in range(n_load):
            if l < self.cfg.num_hidden_layers or (
                    self.load_mtp and os.path.exists(os.path.join(fold_dir, f"layer_{l:02d}_apex_fold.npz"))):
                self.bb.layer(l)
        self.backbone_load_s = time.perf_counter() - t0
        bb_vram_gib = (
            torch.cuda.memory_allocated(self.device) / (1024.0 ** 3)
            if self.device.type == "cuda"
            else (self.bb.resident_bytes() / (1024.0 ** 3))
        )
        print(
            f"[load] Backbone ({backbone_mode}) resident={human_bytes(self.bb.resident_bytes())} "
            f"in {self.backbone_load_s:.1f}s (VRAM alloc={bb_vram_gib:.2f} GiB)",
            flush=True,
        )

        # Discover folded MoE layers in fold_dir
        manifest_path = os.path.join(fold_dir, "fold_manifest.json")
        manifest = json.load(open(manifest_path, "r", encoding="utf-8")) if os.path.exists(manifest_path) else {}
        # [R3.2-2] trellis-coded (TCQ) archives: format from the manifest, else sniffed from a layer file
        self.tcq_rule = manifest.get("tcq_rule") if manifest.get("format") == "tcq" else None
        if self.tcq_rule is None:
            for l in self.cfg.moe_layer_ids:
                p = os.path.join(fold_dir, f"layer_{l:02d}_apex_fold.npz")
                if os.path.exists(p):
                    with np.load(p) as z:
                        if "preset_rule" in z.files:
                            self.tcq_rule = str(z["preset_rule"])
                    break
        if self.tcq_rule is not None:
            import glm53_fold_tcq_format as _TF
            preset_name = f"tcq_{self.tcq_rule}"
            self.preset = _TF.make_preset(self.tcq_rule)
        else:
            preset_name = manifest.get("preset", "heterogeneous_96gb")
            self.preset = APEX_FOLD_PRESETS[preset_name]

        avail_moe_layers: List[int] = []
        for l in self.cfg.moe_layer_ids:
            if max_layers is not None and l >= max_layers and l != self.cfg.num_hidden_layers:
                continue
            if not self.load_mtp and l >= self.cfg.num_hidden_layers:     # [R3.4-E1]
                continue
            p = os.path.join(fold_dir, f"layer_{l:02d}_apex_fold.npz")
            if os.path.exists(p):
                avail_moe_layers.append(l)

        if self.tcq_rule is not None:
            # [R3.2-2] TCQ: no whole-layer VRAM residency (the global expert pool caches hot experts)
            self.placement = {"vram_resident_layers": [], "vram_slabs_gib": 0.0,
                              "host_slabs_gib": len(avail_moe_layers) * self.preset.layer_bytes() / (1024.0 ** 3)}
        else:
            self.placement = ApexFoldPlacementManager.plan_placement(
                preset_name=preset_name,
                available_layers=avail_moe_layers,
                backbone_vram_gib=bb_vram_gib,
                max_vram_budget_gib=vram_budget_gib,
            )
        vram_set = set(self.placement["vram_resident_layers"]) if self.device.type == "cuda" else set()
        if self.expert_pool_gib > 0 and self.device.type == "cuda":
            vram_set = set()  # [R3.2-1.6] no whole-layer residency: the global pool caches hot experts instead

        self.shared_dma = PinnedSlabDMAEngine(
            max_unique_experts=64,
            row_bytes=self.preset.expert_row_bytes,
            device=self.device,
        )

        self.moe_layers: Dict[int, ApexFoldLayerRuntime] = {}
        t_moe0 = time.perf_counter()
        for l in avail_moe_layers:
            p = os.path.join(fold_dir, f"layer_{l:02d}_apex_fold.npz")
            in_vram = l in vram_set
            if self.tcq_rule is not None:
                import glm53_fold_tcq_runtime as _TR
                tarch = _TR.TcqArchive(p, mmap_slab=(cold_mode == "mmap"))
                assert tarch.rule == self.tcq_rule, (p, tarch.rule, self.tcq_rule)
                self.moe_layers[l] = _TR.TcqLayerRuntime(tarch, self.device, slab_in_vram=False)
                miss = getattr(tarch, "missing", None)
                if miss is not None and miss.numel():                     # [R3.4-S2] pruned file -> always masked
                    self.archive_missing[l] = [int(e) for e in miss.tolist()]
                continue
            arch = ApexFoldDiskStore.load_layer(p, mmap_slab=(not in_vram and cold_mode == "mmap"))
            self.moe_layers[l] = ApexFoldLayerRuntime(
                archive=arch,
                preset=self.preset,
                device=self.device,
                slab_in_vram=in_vram,
                dma_engine=self.shared_dma,
            )
        for l, rt in self.moe_layers.items():
            rt.layer_id = l
            rt.use_kernels = bool(fast) and self.device.type == "cuda"
        if self.tcq_rule is not None and self.archive_missing:   # [R3.4 review] pruned files hold fewer rows than 288
            self.placement["host_slabs_gib"] = sum(self.preset.layer_bytes(int(rt.arch.expert_slab_u8.shape[0]))
                                                   for rt in self.moe_layers.values()) / (1024.0 ** 3)
        self._apply_prune(load_prune_spec(prune_spec, self.cfg) if prune_spec is not None else {})
        self.next_moe = {}
        ordered = sorted(l for l in self.moe_layers)
        for a, b in zip(ordered[:-1], ordered[1:]):
            if b < self.cfg.num_hidden_layers:
                self.next_moe[a] = b
        if self.expert_pool_gib > 0 and self.device.type == "cuda":
            pv = os.environ.get("R32_POOL", "2")  # [R3.2] v2 = best measured (v3 rejected, v4 no real gain)
            if pv == "1":
                from glm53_fold_expert_pool import ExpertPool
            elif pv == "2":
                from glm53_fold_expert_pool_v2 import ExpertPool  # [R3.2-1.6b] batched + async prefetch
            elif pv == "5":
                from glm53_fold_expert_pool_v5 import ExpertPool  # [R3.3-S2] registered page-cache tier
            elif pv == "6":
                from glm53_fold_expert_pool_v6 import ExpertPool  # [R3.5-P] v5 + speed item 3, flags R35_POOL_*
            elif pv == "4":
                from glm53_fold_expert_pool_v4 import ExpertPool  # [R3.2-1.6e] v2 + threaded file-read gather
            else:
                from glm53_fold_expert_pool_v3 import ExpertPool  # [R3.2-1.6c] VRAM L1 + pinned-host L2
            n_slots = int(self.expert_pool_gib * (1024 ** 3) // self.preset.expert_row_bytes)
            files = {l: rt.arch.slab_file for l, rt in self.moe_layers.items()
                     if getattr(getattr(rt, "arch", None), "slab_file", None)}
            row_of = {l: rt.arch.row_of for l, rt in self.moe_layers.items()
                      if getattr(getattr(rt, "arch", None), "row_of", None) is not None}   # [R3.3] popularity order
            assert not row_of or pv in ("2", "5", "6"), "archives with a popularity order need R32_POOL=2, 5 or 6"
            self.pool = ExpertPool({l: rt.expert_slab_u8 for l, rt in self.moe_layers.items()},
                                   self.preset.expert_row_bytes, n_slots, self.device,
                                   host_cache_gib=self.host_cache_gib, files=files, row_of=row_of,
                                   register_gib=float(os.environ.get("R33_REGISTER_GIB", "0")))
            if getattr(self.pool, "reg_rows", None) or getattr(self.pool, "reg_layers", None):
                print(f"[load] [R3.3-S2] registered tier ({os.environ.get('R33_REGISTER_MODE', 'layers')}): {self.pool.stats['registered_layers']} layers, {self.pool.stats.get('registered_rows', '?')} rows, "
                      f"{self.pool.stats['registered_gib']} GiB locked in {self.pool.stats['register_s']} s", flush=True)
            for rt in self.moe_layers.values():
                rt.pool = self.pool
            print(f"[load] [R3.2] global expert pool: {n_slots} slots ({self.expert_pool_gib:.2f} GiB), "
                  f"prefetch={self.prefetch}", flush=True)
            if hasattr(self.pool, "flags_str"):                       # [R3.5-P] pool v6 switches of this run
                print(f"[load] [R3.5-P] pool v6: {self.pool.flags_str()}", flush=True)
            self._pred_host = torch.empty((self.cfg.num_experts_per_tok * 4,), dtype=torch.int64, pin_memory=True)
            if getattr(self.pool, "wants_scores", False):             # [R3.5-P (e)] scores of the eager prediction
                self._pred_sc_host = torch.empty((self.cfg.num_experts_per_tok * 4,), dtype=torch.float32,
                                                 pin_memory=True)
            # [R3.5-INT item 10b] env R35_PREFILL_COLD=1 + a pool with cold inserts (v6): experts fetched by the
            # prefill go to the cold end of the pool (they do not flush the decode working set); bytes only
            self.prefill_cold = (os.environ.get("R35_PREFILL_COLD", "0") == "1"
                                 and bool(getattr(self.pool, "supports_cold", False)))
            if self.prefill_cold:
                print("[load] [R3.5-INT] prefill experts inserted at the pool's cold end (R35_PREFILL_COLD=1)", flush=True)
            elif os.environ.get("R35_PREFILL_COLD", "0") == "1":
                print(f"[load] [R3.5-INT] R35_PREFILL_COLD=1 ignored: pool {type(self.pool).__module__} has no cold "
                      "inserts (needs R32_POOL=6)", flush=True)
            # [R3.5-T item 10a] env R35_PREFILL_PREFETCH=1: every prefill batch of > 4 tokens prefetches the union of the
            # next MoE layer's predicted experts over ALL its tokens (most-predicted first), handed to the pool after this
            # layer's own ensure() (demand first); with R35_PREFILL_COLD the keys go to the cold end too. Bytes only.
            # R35_PREFILL_PREFETCH_MIN (1): tokens that must predict an expert; R35_PREFILL_PREFETCH_MAX (0 = all): keys
            # per layer. Needs the next-layer prefetch (not --no-prefetch) and [R3.5-FIX F3] pool v6 (R32_POOL=6: its
            # reservations keep layer l's slots safe from the next layer's prefetch; with pools v1-v5 it stays off).
            if os.environ.get("R35_PREFILL_PREFETCH", "0") == "1":
                self.set_ttft_options(prefetch=self.prefetch,
                                      pf_min=int(os.environ.get("R35_PREFILL_PREFETCH_MIN", "1") or 1),
                                      pf_max=int(os.environ.get("R35_PREFILL_PREFETCH_MAX", "0") or 0))
                pf_state = ("on" if self.prefill_prefetch else "OFF (--no-prefetch)" if not self.prefetch else  # [R3.5-FIX]
                            f"OFF (pool {type(self.pool).__module__} has no slot reservations: needs R32_POOL=6)")
                print(f"[load] [R3.5-T] prefill prefetch {pf_state}: "
                      f"next-layer union of every prefill token, min {self.prefill_pf_min} token(s), "
                      f"{self.prefill_pf_max or 'all'} keys per layer"
                      f"{', cold end' if self.prefill_prefetch and self.prefill_cold else ''} "
                      f"(R35_PREFILL_PREFETCH=1)", flush=True)
        print(
            f"[load] Loaded {len(self.moe_layers)} APEX-FOLD-FP16 MoE layers "
            f"({'TCQ rule ' + self.tcq_rule + f', {self.preset.bpw():.4f} bpw' if self.tcq_rule else preset_name}) in {time.perf_counter() - t_moe0:.1f}s | "
            f"VRAM slabs: {len(vram_set)} layers ({self.placement['vram_slabs_gib']:.2f} GiB) | "
            f"Host ({cold_mode}) slabs: {len(avail_moe_layers) - len(vram_set)} layers ({self.placement['host_slabs_gib']:.2f} GiB)",
            flush=True,
        )
        self.moe_time_s = 0.0
        from glm53_fold_route_r36 import install as install_r36_route
        self.r36_router = install_r36_route(self)
        # [R3.2-1Q.2] exact prefix caches (from the rebuild teacher pass), longest first
        self.prefix_caches = []
        if prefix_cache:
            for nm in ("prefix10", "prefix2"):
                p = os.path.join(fold_dir, f"exact_prefix_cache_{nm}.pt")
                if os.path.exists(p):
                    self.prefix_caches.append(torch.load(p))
            self.prefix_caches.sort(key=lambda pc: -len(pc["ids"]))
            print("[load] [R3.2-1Q.2] exact prefix caches: "
                  + (", ".join(f"{len(pc['ids'])} tok{' +logits' if 'logits' in pc else ''}" for pc in self.prefix_caches)
                     or "none found"), flush=True)

    def _apply_prune(self, spec: Dict[int, List[int]]) -> None:
        """[R3.4-S4] router masks = the prune spec + every expert a (pruned) archive file has no row for (a written pruned
        archive can never be served without its mask). Written in place into the backbone's router bias (bb.set_pruned):
        prefill, graph decode (no recapture needed), next-layer prefetch prediction and MTP all route through it."""
        masks: Dict[int, set] = {}
        for l, ids in self.archive_missing.items():
            masks.setdefault(int(l), set()).update(int(e) for e in ids)
        for l, ids in (spec or {}).items():
            masks.setdefault(int(l), set()).update(int(e) for e in ids)
        skipped, pending = [], []
        for l in sorted(masks):
            if l not in self.moe_layers:
                skipped.append(l)               # layer not served (--no-mtp-layer / --max-layers): nothing routes it
                continue
            # [R3.4 review] registered for every SERVED layer, loaded or not: a layer the backbone loads later (the MTP
            # layer 45 with --max-layers, first loaded by mtp_forward) gets its mask in bb.layer() on load
            self.bb.set_pruned(l, sorted(masks[l]))
            self.prune_masks[l] = list(self.bb.prune_spec.get(l, []))
            if l not in self.bb._layers:
                pending.append(l)
        if masks:
            n_e = sum(len(v) for v in self.prune_masks.values())
            print(f"[load] [R3.4-S4] expert masks (router bias -inf): {len(self.prune_masks)} layers, {n_e} experts "
                  f"({sum(len(v) for v in (spec or {}).values())} from the prune spec, "
                  f"{sum(len(v) for v in self.archive_missing.values())} missing from the archive)"
                  + (f"; applied when loaded: layers {pending}" if pending else "")
                  + (f"; not served, skipped: layers {skipped}" if skipped else ""), flush=True)

    def set_prune(self, prune_spec=None) -> Dict[int, List[int]]:
        """[R3.4 review] replace the expert masks at run time (between requests; in place, no graph recapture): the new
        spec (path / dict / None = no spec) plus every expert the archive lacks, exactly as at construction. Use this,
        not bb.set_pruned(): that replaces ONE layer's whole mask and would unmask a pruned archive's missing experts.
        Updates prune_masks (read by _warm_ranges and the tools' mask check). Returns prune_masks."""
        spec = load_prune_spec(prune_spec, self.cfg) if prune_spec is not None else {}
        n_max = int(self.cfg.n_routed_experts) - int(self.cfg.num_experts_per_tok) - 1
        for l in set(spec) | set(self.archive_missing):     # validate everything before any mask changes
            n = len(set(spec.get(l, ())) | set(self.archive_missing.get(l, ())))
            if n > n_max:
                raise ValueError(f"set_prune: layer {l} would mask {n} experts (spec + archive-missing; max {n_max})")
        for l in list(self.bb.prune_spec):
            self.bb.set_pruned(l, [])
        self.prune_masks = {}
        self._apply_prune(spec)
        return self.prune_masks

    def warm_page_cache(self, fold_dir: str, threads: int = 4) -> float:
        """[R3.2-1.6d] sequential 64 MiB reads of every layer file (4 files in flight) -> OS page cache.
        [R3.4-S4] layers with masked experts: only the slab rows of the unmasked experts are read (whole file otherwise)."""
        from concurrent.futures import ThreadPoolExecutor
        files = [os.path.join(fold_dir, f"layer_{l:02d}_apex_fold.npz") for l in sorted(self.moe_layers)]
        jobs = [(p, self._warm_ranges(l)) for p, l in zip(files, sorted(self.moe_layers))]

        def rd(job):
            p, ranges = job
            n = 0
            buf = memoryview(bytearray(64 << 20))
            with open(p, "rb", buffering=0) as f:
                if ranges is None:
                    while True:
                        k = f.readinto(buf)
                        if not k:
                            return n
                        n += k
                for a, b in ranges:             # [R3.4-S4] byte ranges [a, b) of the unmasked rows
                    f.seek(a)
                    left = b - a
                    while left > 0:
                        k = f.readinto(buf[: min(left, len(buf))])
                        if not k:
                            break
                        n += k
                        left -= k
                return n

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=threads) as ex:
            tot = sum(ex.map(rd, jobs))
        dt = time.perf_counter() - t0
        n_part = sum(r is not None for _, r in jobs)
        print(f"[load] [R3.2-1.6d] page-cache warm-up: {tot / 2**30:.1f} GiB in {dt:.1f}s "
              f"({tot / 2**30 / max(dt, 1e-6):.2f} GiB/s)"
              + (f" [R3.4-S4: {n_part} masked layers read unmasked rows only]" if n_part else ""), flush=True)
        return dt

    def _warm_ranges(self, l: int):
        """[R3.4-S4] byte ranges of layer l's file holding its unmasked experts' slab rows, or None (= whole file)."""
        masked = set(self.prune_masks.get(l, ()))
        rt = self.moe_layers[l]
        arch = getattr(rt, "arch", None)
        sf = getattr(arch, "slab_file", None)
        if not masked or sf is None:
            return None
        rb = int(self.preset.expert_row_bytes)
        n_rows = int(rt.expert_slab_u8.shape[0])
        ro = getattr(arch, "row_of", None)
        n_tot = int(ro.numel()) if ro is not None else n_rows
        rows = sorted(int(ro[e]) if ro is not None else e for e in range(n_tot) if e not in masked)
        rows = [r for r in rows if 0 <= r < n_rows]
        ranges = []
        for r in rows:
            if ranges and ranges[-1][1] == r:
                ranges[-1][1] = r + 1
            else:
                ranges.append([r, r + 1])
        off = int(sf[1])
        return [(off + a * rb, off + b * rb) for a, b in ranges]

    def _restore_prefix(self, token_ids: List[int]):
        """[R3.2-1Q.2] -> (cache, n_prefix, prefix dict or None); a fresh cache when no prefix matches."""
        for pc in self.prefix_caches:
            P = len(pc["ids"])
            if len(token_ids) > P and list(token_ids[:P]) == list(pc["ids"]):
                dev = self.device
                c = RealCache()
                c.pos = int(pc["pos"])
                c.kda_state = {int(k): v.to(dev).clone() for k, v in pc["kda_state"].items()}
                c.kda_conv = {int(k): tuple(t.to(dev).clone() for t in v) for k, v in pc["kda_conv"].items()}
                c.dsa_kv = {int(k): v.to(dev).clone() for k, v in pc["dsa_kv"].items()}
                c.dsa_ki = {int(k): v.to(dev).clone() for k, v in pc["dsa_ki"].items()}
                c.dsa_gs = {int(k): v.to(dev).clone() for k, v in pc["dsa_gs"].items()}
                return c, P, pc
        return RealCache(), 0, None

    def set_gain(self, spec: str) -> None:
        """[R3.2-1Q.1] 'unbiased': per-tier distortion D from the archive's weight SNR (uniform +-0.01 dB) ->
        gate/up x (1-D)^-0.825 (energy-weighted over tiers), down x (1-D)^0.175. Or explicit 'g,u,d'."""
        if self.tcq_rule is not None:
            print(f"[R3.2] gain correction {spec}: skipped for the TCQ archive (gate/up are encoded unbiased, "
                  f"down least-squares)", flush=True)
            return
        if spec == "unbiased":
            p = self.preset
            d2 = 10 ** (-10.19 / 10.0)   # pure 2-stage tier (= down weight SNR 10.19 dB)
            d3 = 10 ** (-15.5 / 10.0)    # 3-stage tier (from gate/up 10.81 dB with c0/c2 = 768/4096)
            def gu(spec_p):
                f3 = spec_p.c0 / float(spec_p.c2)
                return f3 * (1 - d3) ** -0.825 + (1 - f3) * (1 - d2) ** -0.825
            gains = {"g": gu(p.gate), "u": gu(p.up), "d": (1 - d2) ** 0.175}
        else:
            a, b, c = (float(v) for v in spec.split(","))
            gains = {"g": a, "u": b, "d": c}
        for rt in self.moe_layers.values():
            rt.gain = dict(gains)
        print(f"[R3.2] gain correction {spec}: " + ", ".join(f"{k}={v:.4f}" for k, v in gains.items()), flush=True)

    def moe_dispatch(self, l: int, x_flat: torch.Tensor) -> torch.Tensor:
        if self.sync_timing and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        if getattr(self,'r36_router',None) is not None:
            self.r36_router.publish(l)
            if self.next_moe.get(l) is not None:self.r36_router.publish(self.next_moe[l])
        top_w, top_i = self.bb.route(l, x_flat)
        pred_ev = None
        pf_ev = None                                                        # [R3.5-T item 10a]
        want_sc = False                                                     # [R3.5-P (e)]
        nxt = self.next_moe.get(l)
        if self.pool is not None and self.prefetch and nxt is not None and x_flat.shape[0] <= 4:
            # [R3.2-1.6] pre-gating: the next layer's router on this layer's FFN input, copied to pinned host
            # memory BEFORE this layer's expert kernels are queued (reading it later does not wait for them)
            want_sc = bool(getattr(self.pool, "wants_scores", False))      # [R3.5-P (e)] pool v6 cold-prefetch gate
            if want_sc:
                pid_, psel_ = self.bb.route_ids(nxt, x_flat)                # route()'s ids (same ops) + their scores
                pred = pid_.reshape(-1)
            else:
                pred = self.bb.route(nxt, x_flat)[1].reshape(-1)
            n_pred = pred.numel()
            self._pred_host[:n_pred].copy_(pred, non_blocking=True)
            if want_sc:                                                     # [R3.5-P (e)]
                self._pred_sc_host[:n_pred].copy_(psel_.reshape(-1).float(), non_blocking=True)
            pred_ev = torch.cuda.Event()
            pred_ev.record()
        elif (getattr(self, "prefill_prefetch", False) and self.pool is not None and self.prefetch
              and nxt is not None):
            # [R3.5-T item 10a] prefill batch (> 4 tokens): the next layer's router on this layer's FFN input for EVERY
            # token -> per-expert prediction counts -> pinned row, queued BEFORE this layer's expert kernels; the host
            # reads it after this layer's ensure() (demand first) and prefetches the union. Prediction only.
            self._prefill_prefetch_queue(nxt, x_flat)
            pf_ev = self._pf_event()
        out = self.moe_layers[l].forward_routed_experts(
            x_flat, top_w, top_i, swiglu_limit=self.cfg.swiglu_limit, hobbit_tau=self.hobbit_tau
        )
        if pred_ev is not None:
            pred_ev.synchronize()
            if want_sc:                                                     # [R3.5-P (e)] gated cold prefetch (top-k
                self.pool.prefetch_scored(nxt, self._pred_host[:n_pred].tolist(),    # only: no 9th-best reference)
                                          self._pred_sc_host[:n_pred].tolist())
            else:
                ids = sorted(set(self._pred_host[:n_pred].tolist()))
                self.pool.prefetch(nxt, ids)
        elif pf_ev is not None:                                             # [R3.5-T item 10a]
            pf_ev.synchronize()
            self._prefill_prefetch_issue(nxt)
        if self.sync_timing and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.moe_time_s += time.perf_counter() - t0
        return out

    def _prefill_cold_layers(self) -> list:
        """[R3.5-INT item 10b] the MoE runtimes whose prefill ensure() goes to the pool's cold end (R35_PREFILL_COLD=1
        with a pool offering ensure_cold, i.e. pool v6); [] otherwise. The caller resets their flag after the prefill."""
        if not getattr(self, "prefill_cold", False):
            return []
        rts = [rt for rt in self.moe_layers.values() if getattr(rt, "pool", None) is not None]
        for rt in rts:
            rt.prefill_cold = True
        return rts

    # ------------------------------------------------------------------ [R3.5-T] time to first token
    def set_ttft_options(self, prefetch=None, cold=None, pf_min=None, pf_max=None) -> Dict[str, object]:
        """[R3.5-T] switch the time-to-first-token options between requests (gates / benches A/B them in one process;
        the env vars set them at load): prefetch = item 10a (R35_PREFILL_PREFETCH; needs a pool with slot reservations,
        pool v6 - [R3.5-FIX F3]: stays off with pools v1-v5), cold = item 10b (R35_PREFILL_COLD; needs a pool with cold
        inserts, pool v6), pf_min / pf_max = R35_PREFILL_PREFETCH_MIN / _MAX.
        None keeps a setting. Bytes only: no option changes a number of the model. Returns the settings."""
        if pf_min is not None:
            if int(pf_min) < 1:
                raise ValueError(f"R35_PREFILL_PREFETCH_MIN must be >= 1, not {pf_min}")
            self.prefill_pf_min = int(pf_min)
        if pf_max is not None:
            if int(pf_max) < 0:
                raise ValueError(f"R35_PREFILL_PREFETCH_MAX must be >= 0 (0 = all), not {pf_max}")
            self.prefill_pf_max = int(pf_max)
        if prefetch is not None:
            # [R3.5-FIX F3] item 10a needs a pool whose slots stay RESERVED until the next layer's ensure() (pool v6,
            # supports_release): the prefill prefetch of layer l + 1 is issued while layer l's expert kernels are still
            # queued, and pools v1-v5 protect those slots by LRU depth only (a prefetch may evict and rewrite them)
            on = (bool(prefetch) and self.pool is not None
                  and bool(getattr(self.pool, "supports_release", False)))
            if on and getattr(self, "_pf_cnt_host", None) is None:
                self._pf_cnt_host = torch.zeros((int(self.cfg.n_routed_experts),), dtype=torch.int64,
                                                pin_memory=self.device.type == "cuda")
            self.prefill_prefetch = on
        if cold is not None:
            self.prefill_cold = bool(cold) and bool(getattr(self.pool, "supports_cold", False))
        return {"prefetch": bool(getattr(self, "prefill_prefetch", False)), "cold": bool(getattr(self, "prefill_cold", False)),
                "pf_min": int(getattr(self, "prefill_pf_min", 1)), "pf_max": int(getattr(self, "prefill_pf_max", 0))}

    def _prefill_prefetch_queue(self, nxt: int, x_flat: torch.Tensor) -> None:
        """[R3.5-T item 10a] GPU side: route_ids (route()'s ops up to the top-k) of layer `nxt` on this layer's FFN input
        for every token, per-expert counts (bincount over the N x K ids; masked experts never appear), D2H into the
        pinned count row. Called before the layer's expert kernels are queued, so reading it does not wait for them."""
        n_exp = int(self._pf_cnt_host.numel())
        if getattr(self,'r36_router',None) is not None:
            ids=self.bb.route_ids(nxt,x_flat)[0].reshape(-1)
            cnt=torch.bincount(ids[ids>=0],minlength=n_exp)
        else:
            cnt = torch.bincount(self.bb.route_ids(nxt, x_flat)[0].reshape(-1), minlength=n_exp)
        self._pf_cnt_host.copy_(cnt[:n_exp], non_blocking=True)

    def _pf_event(self):
        """[R3.5-T item 10a] completion event of the queued count copy (CPU runs: a no-op stand-in)."""
        if self.device.type == "cuda":
            ev = torch.cuda.Event()
            ev.record()
            return ev
        return _HostDone()

    def _prefill_prefetch_issue(self, nxt: int) -> int:
        """[R3.5-T item 10a] host side, after this layer's ensure(): experts predicted by >= prefill_pf_min tokens, most
        predicted first (ties: lower id), at most prefill_pf_max (0 = all) -> pool.prefetch(nxt, ids); with item 10b
        (the prefill sets rt.prefill_cold) pool.prefetch_cold, i.e. the keys go to the cold end. Returns keys queued."""
        cnt = self._pf_cnt_host.numpy()
        idx = np.nonzero(cnt >= int(self.prefill_pf_min))[0]
        st = self._pf_stats
        st["layers"] += 1
        if idx.size == 0:
            return 0
        order = idx[np.lexsort((idx, -cnt[idx]))]
        if int(self.prefill_pf_max) > 0:
            order = order[: int(self.prefill_pf_max)]
        ids = [int(e) for e in order]
        rt = self.moe_layers.get(nxt)
        if getattr(rt, "prefill_cold", False) and callable(getattr(self.pool, "prefetch_cold", None)):
            n = self.pool.prefetch_cold(nxt, ids)
        else:
            n = self.pool.prefetch(nxt, ids)
        st["predicted"] += len(ids)
        st["queued"] += int(n or 0)
        return int(n or 0)

    def _continue_past_tmax(self, cache):
        """[R3.5-T H6] env R35_CONTINUE_PAST_TMAX=1: the graph decode ended at its t_max - 1 bound with budget left ->
        an eager RealCache of the decoder's final state (GraphDecoder.export_cache, complete DSA indexer caches), so
        generate()'s eager N = 1 loop continues the answer past the 2048-token graph window. None when the decoder
        stopped below the bound or keeps no indexer caches (R34_DSA_INDEXER=0: beyond index_topk the eager DSA reads
        ki / gs of every position - continuing would read zero placeholders; a note is printed once)."""
        gd = self.graph_decoder
        if gd is None or not callable(getattr(gd, "export_cache", None)):
            return None
        if int(gd.pos_t.item()) < int(gd.t_max) - 1:
            return None
        if not getattr(gd, "dsa_indexer", False):
            if not getattr(self, "_h6_noted", False):
                print("[R3.5-T] R35_CONTINUE_PAST_TMAX=1 ignored: the graph decoder keeps no DSA indexer caches "
                      "(run with R34_DSA_INDEXER=1); the answer ends at the graph window", flush=True)
                self._h6_noted = True
            return None
        return gd.export_cache(cache)

    def select_graph_decoder(self, n_prompt: int, max_new: int):
        """[R3.7 Stage 7] R37_DUAL_GRAPHS: the decoder for a request of n_prompt prompt tokens and a budget of max_new.
        A request that fits the 2,048-token window entirely is decoded by the unchanged window graphs, so it gives
        what the Round 3.5 / 3.7 record gives; every other request by the long-context graphs. Without the second
        decoder: the only one."""
        gd_std = getattr(self, "graph_decoder_std", None)
        if gd_std is None:
            return self.graph_decoder
        return gd_std if int(n_prompt) + int(max_new) < int(gd_std.t_max) else self.graph_decoder_long

    @torch.no_grad()
    def warmup(self, prompts, eos_ids=None, use_mtp: bool = False, max_new_tokens: int = 4) -> Dict[str, object]:
        """[R3.5-T item 10d] start-up warm-up (env R35_WARMUP, main()): one generate() per prompt right after the graph
        capture, so the first real request pays no one-time cost (CUDA lazy loading of the prefill kernels, cuBLAS
        heuristics for the prefill shapes, caching-allocator growth, first replays of the captured graphs, the MTP
        sets with use_mtp). Bytes only: afterwards the pool is quiesced (when it can be) and its per-request baseline,
        the graph decoder's counters and the MTP totals are put back, so the first request's statistics start clean.
        Returns {seconds, runs: [{prompt_tokens, new_tokens, prefill_s, decode_s}], quiesce_s}."""
        t0 = time.perf_counter()
        gd = self.graph_decoder
        saved = None
        if gd is not None:
            saved = {"stats": dict(getattr(gd, "stats", {}) or {}), "mtp_totals": getattr(gd, "mtp_totals", None),
                     "mtp_last": getattr(gd, "mtp_last", None)}
        runs = []
        for ids in prompts:
            res = self.generate(list(ids), max_new_tokens=int(max_new_tokens), eos_ids=set(eos_ids or ()),
                                use_mtp=bool(use_mtp))
            runs.append({k: res.get(k) for k in ("prompt_tokens", "new_tokens", "prefill_s", "decode_s")})
        q = 0.0
        if self.pool is not None:
            if callable(getattr(self.pool, "quiesce", None)):
                q = float(self.pool.quiesce())
            self._pool_prev = dict(self.pool.stats)
        if saved is not None:
            gd.stats = saved["stats"]
            if saved["mtp_totals"] is None:
                if hasattr(gd, "mtp_totals"):
                    gd.mtp_totals = None
            else:
                gd.mtp_totals = saved["mtp_totals"]
            gd.mtp_last = saved["mtp_last"]
        self.moe_time_s = 0.0
        return {"seconds": round(time.perf_counter() - t0, 2), "runs": runs, "quiesce_s": round(q, 3)}

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 48,
        eos_ids: Optional[Set[int]] = None,
        use_mtp: bool = False,
        serve_state=None,
    ) -> Dict[str, object]:
        """[R3.5-S1] serve_state: a glm53_fold_serve.ServeState (the multi-turn server) or None = the base path,
        unchanged. With it: H1 the cache comes from serve_state.begin() (a restored conversation state, or the
        _restore_prefix result for a fresh request), H2 after_prefill() stores the prompt-end state, H3 on_token() is
        called for every later token and .stopped ends decoding, H4 after_decode() stores the final state, H5 a plan
        that restored the whole prompt skips the prefill (first token + h_last from serve_state). [R3.5-INT] with graph
        MTP the layer-45 history is kept complete for the server (keep_history) and a stored one is put back
        (serve_state.mtp_restore); drafts only, never an emitted token.
        [R3.5-T] item 10a (R35_PREFILL_PREFETCH): the prefill's moe_dispatch prefetches the next layer's predicted union
        (res["prefill_prefetch"] = layers / predicted / queued / useful); H6 (R35_CONTINUE_PAST_TMAX=1 + R34_DSA_INDEXER=1):
        when the graphs stop at t_max - 1 with budget left the eager N = 1 loop continues from their exported state
        (res["continued_past_tmax"]). Both off: the base statements."""
        if use_mtp:
            if os.environ.get('R36_LONGCTX','0')=='1':
                raise ValueError('R36_LONGCTX + MTP is not yet supported; use N=1')
            if int(os.environ.get('R37_PREFILL_CHUNK', '0') or 0) > 0:
                # [R3.7 Stage 7] a prompt taken in several passes returns the hidden state of its last position
                # only; the MTP layer's history needs all of them
                raise ValueError('R37_PREFILL_CHUNK + MTP is not supported; use N=1')
            if getattr(self,'r36_router',None) is not None:self.r36_router.policy.require_mtp()
        eos_ids = eos_ids or set()
        if serve_state is None:
            cache, n_pre, _ = self._restore_prefix(prompt_ids)  # [R3.2-1Q.2] exact prefix states when they match
        else:
            cache, n_pre = serve_state.begin(self, prompt_ids)  # [R3.5-S1] H1
        skip_prefill = serve_state is not None and n_pre >= len(prompt_ids)   # [R3.5-S1] H5
        if not skip_prefill:
            ids_t = torch.tensor([prompt_ids[n_pre:]], dtype=torch.long, device=self.device)

        # 1. Batched Prefill (Path B Grouped Tensor-Core GEMM + Vectorized Chunk-Parallel KDA)
        pf_info = None
        if getattr(self, "prefill_prefetch", False) and not skip_prefill:    # [R3.5-T item 10a] per-request counters
            self._pf_stats = {"layers": 0, "predicted": 0, "queued": 0}
            pf_info = int(self.pool.stats.get("prefetch_useful", 0))
        self.moe_time_s = 0.0
        t_pre0 = time.perf_counter()
        h_collapsed = None
        if not skip_prefill:
            cold_rts = self._prefill_cold_layers() if getattr(self, "prefill_cold", False) else None
            if cold_rts is None:
                logits, h_collapsed = self.bb.forward(
                    ids_t, cache=cache, moe_fn=self.moe_dispatch, last_only=True, max_layers=self.max_layers
                )
            else:                                         # [R3.5-INT item 10b] prefill experts -> the pool's cold end
                try:
                    logits, h_collapsed = self.bb.forward(
                        ids_t, cache=cache, moe_fn=self.moe_dispatch, last_only=True, max_layers=self.max_layers
                    )
                finally:
                    for rt in cold_rts:
                        rt.prefill_cold = False
        t_prefill = time.perf_counter() - t_pre0
        prefill_moe_s = self.moe_time_s
        if pf_info is not None:                           # [R3.5-T item 10a] this prefill's prefetch counters
            pf_info = dict(self._pf_stats, useful=int(self.pool.stats.get("prefetch_useful", 0)) - pf_info)
        probe = getattr(self, "_prefill_probe", None)     # [R3.5-T] gates only (build_run/prefill_gate_r35.py)
        if probe is not None and not skip_prefill:
            probe(logits, cache, h_collapsed)

        if skip_prefill:                                  # [R3.5-S1] H5: state after the whole prompt + its token known
            next_tok, h_last = int(serve_state.first_token), serve_state.h_last
        else:
            next_tok = self.bb.argmax_token(logits[0, -1])   # [R3.4-V] mapped back to the token id with a trimmed head
        gen_ids: List[int] = [next_tok]
        if not skip_prefill:
            h_last = h_collapsed[:, -1:, :]
        if serve_state is not None:
            serve_state.after_prefill(self, cache, next_tok, h_last)   # [R3.5-S1] H2 (before any MTP draft)
        if getattr(self, "graph_decoder_std", None) is not None:       # [R3.7 Stage 7] R37_DUAL_GRAPHS
            self.graph_decoder = self.select_graph_decoder(len(prompt_ids), max_new_tokens)

        # Draft initial MTP token if enabled and Layer 45 is loaded
        mtp_enabled = use_mtp and (self.cfg.num_hidden_layers in self.moe_layers)
        # [R3.5-M2] --mtp with --graphs and env R35_MTP_GRAPHS=1: MTP decoding on the CUDA-graph path (GraphDecoder.
        # decode_mtp: draft graphs -> N = 2 verify graphs -> accept / reject; token-identical to the N = 1 graph path).
        # [R3.5-FIX F1] opt-in: env unset / 0 (default), no graph decoder, or a prompt that does not fit t_max: the eager
        # MTP path below, unchanged (the base path).
        _gd_mtp = getattr(self.graph_decoder, "mtp_enabled", None)
        graph_mtp = bool(mtp_enabled and callable(_gd_mtp) and _gd_mtp()
                         and int(cache.pos) < self.graph_decoder.t_max)
        draft_tok: Optional[int] = None
        if mtp_enabled and not graph_mtp:
            mtp_logits = self.bb.mtp_forward(
                torch.tensor([[next_tok]], dtype=torch.long, device=self.device),
                h_last,
                cache=cache,
                moe_fn=self.moe_dispatch,
            )
            draft_tok = self.bb.argmax_token(mtp_logits[0, -1])

        # [R3.9M] bookkeeping only, both off by default (None): R39_MTP_STATS=1 -> the pool's counters of the DECODE
        # phase alone (res["pool_decode"]; res["pool"] also holds the prompt pass); R39_MTP_FP=1 -> the logit-bit
        # fingerprints of the graph decoder plus the prompt pass's last row (res["logit_fp"], one per generated id).
        _r39m_ps0 = (dict(self.pool.stats) if (self.pool is not None
                                                and os.environ.get("R39_MTP_STATS", "0").strip() == "1") else None)
        _r39m_fp0 = None
        if (os.environ.get("R39_MTP_FP", "0").strip() == "1" and not skip_prefill
                and callable(getattr(self.graph_decoder, "_r39m_hash", None))):
            _r39m_fp0 = self.graph_decoder._r39m_hash(logits[0, -1])
            self.graph_decoder.r39m_fp_last = None

        self.moe_time_s = 0.0
        t_dec0 = time.perf_counter()
        n_steps = 0
        mtp_accepted = 0
        mtp_total_drafts = 0

        graph_done = False
        if self.graph_decoder is not None and not mtp_enabled and next_tok not in eos_ids and not (
                serve_state is not None and serve_state.stopped):          # [R3.5-S1] H3
            # [R3.2-1.5] CUDA-graph greedy decode (same math as the eager loop below)
            if serve_state is None:
                g_ids = self.graph_decoder.decode(next_tok, cache, max_new_tokens - 1, eos_ids)
            else:                                                          # [R3.5-S1] H3: per-token callback
                g_ids = self.graph_decoder.decode(next_tok, cache, max_new_tokens - 1, eos_ids,
                                                  on_token=serve_state.on_token)
            if g_ids is not None:
                gen_ids.extend(g_ids)
                n_steps = max(1, len(g_ids))
                next_tok = gen_ids[-1]
                graph_done = True

        mtp_graph_stats = None
        if graph_mtp and next_tok not in eos_ids and not (
                serve_state is not None and serve_state.stopped):          # [R3.5-M2] graph-path MTP decoding
            if serve_state is None:
                g_ids = self.graph_decoder.decode_mtp(next_tok, cache, max_new_tokens - 1, eos_ids,
                                                      h_prefill=h_collapsed, prompt_ids=prompt_ids, n_pre=n_pre)
            else:                  # [R3.5-INT] S1 H3 (on_token) + the server's layer-45 history (keep / restore)
                hp, npre = (h_last, len(prompt_ids) - 1) if skip_prefill else (h_collapsed, n_pre)   # H5: no prefill
                g_ids = self.graph_decoder.decode_mtp(next_tok, cache, max_new_tokens - 1, eos_ids, h_prefill=hp,
                                                      prompt_ids=prompt_ids, n_pre=npre, on_token=serve_state.on_token,
                                                      mtp_restore=getattr(serve_state, "mtp_restore", None),
                                                      keep_history=bool(getattr(serve_state, "mtp_keep", False)))
            if g_ids is not None:
                gen_ids.extend(g_ids)
                mtp_graph_stats = dict(self.graph_decoder.mtp_last)
                n_steps = max(1, int(mtp_graph_stats["steps"]))
                mtp_accepted, mtp_total_drafts = int(mtp_graph_stats["accepted"]), int(mtp_graph_stats["verified"])
                next_tok = gen_ids[-1]
                graph_done = True
            else:                                          # (not reached: cache.pos < t_max) the eager MTP loop below
                mtp_logits = self.bb.mtp_forward(torch.tensor([[next_tok]], dtype=torch.long, device=self.device),
                                                 h_last, cache=cache, moe_fn=self.moe_dispatch)
                draft_tok = self.bb.argmax_token(mtp_logits[0, -1])

        continued = None
        if (graph_done and getattr(self, "continue_past_tmax", False) and len(gen_ids) < max_new_tokens
                and next_tok not in eos_ids and not (serve_state is not None and serve_state.stopped)):
            # [R3.5-T H6] the graphs stopped at their t_max - 1 bound with budget left: continue on the eager N = 1 loop
            # below from their exported final state (MTP drafts never change a token, so none are made from here)
            c_ex = self._continue_past_tmax(cache)
            if c_ex is not None:
                cache, graph_done, draft_tok = c_ex, False, None
                continued = {"from_pos": int(c_ex.pos), "graph_tokens": len(gen_ids)}

        while not graph_done and len(gen_ids) < max_new_tokens and next_tok not in eos_ids:
            if serve_state is not None and serve_state.stopped:           # [R3.5-S1] H3: stop string / client gone
                break
            n_steps += 1
            if not mtp_enabled or draft_tok is None:
                cur_in = torch.tensor([[next_tok]], dtype=torch.long, device=self.device)
                logits, h_collapsed = self.bb.forward(
                    cur_in, cache=cache, moe_fn=self.moe_dispatch, last_only=True, max_layers=self.max_layers
                )
                next_tok = self.bb.argmax_token(logits[0, -1])
                gen_ids.append(next_tok)
                if serve_state is not None:
                    serve_state.on_token(next_tok)                         # [R3.5-S1] H3
            else:
                # True Batched N=2 MTP Verification (verifies [next_tok, draft_tok] in ONE pass!)
                snap = cache.snapshot()
                pair_in = torch.tensor([[next_tok, draft_tok]], dtype=torch.long, device=self.device)
                logits_pair, h_pair = self.bb.forward(
                    pair_in, cache=cache, moe_fn=self.moe_dispatch, last_only=False, max_layers=self.max_layers
                )
                verified_0 = self.bb.argmax_token(logits_pair[0, 0])
                mtp_total_drafts += 1
                if verified_0 == draft_tok:
                    mtp_accepted += 1
                    bonus_tok = self.bb.argmax_token(logits_pair[0, 1])
                    gen_ids.append(draft_tok)
                    if serve_state is not None:
                        serve_state.on_token(draft_tok)                    # [R3.5-S1] H3
                    # [PATCH R3-06] an accepted EOS draft ends generation; [R3.5-FIX] so does a stop at the draft (H3:
                    # True stops after THAT token - no bonus token, no further draft; as decode() / decode_mtp())
                    if draft_tok in eos_ids or (serve_state is not None and serve_state.stopped):
                        next_tok = draft_tok
                        break
                    if len(gen_ids) < max_new_tokens:
                        gen_ids.append(bonus_tok)
                        if serve_state is not None:
                            serve_state.on_token(bonus_tok)                # [R3.5-S1] H3
                    next_tok = bonus_tok
                    h_last = h_pair[:, 1:2, :]
                    if serve_state is not None and serve_state.stopped:   # [R3.5-FIX] stopped at the bonus: no draft
                        break
                    mtp_logits = self.bb.mtp_forward(
                        torch.tensor([[next_tok]], dtype=torch.long, device=self.device),
                        h_last,
                        cache=cache,
                        moe_fn=self.moe_dispatch,
                    )
                    draft_tok = self.bb.argmax_token(mtp_logits[0, -1])
                else:
                    # Rollback second token state cleanly and advance single token
                    cache.restore(snap)
                    cur_in = torch.tensor([[next_tok]], dtype=torch.long, device=self.device)
                    _, h_single = self.bb.forward(
                        cur_in, cache=cache, moe_fn=self.moe_dispatch, last_only=True, max_layers=self.max_layers
                    )
                    gen_ids.append(verified_0)
                    if serve_state is not None:
                        serve_state.on_token(verified_0)                   # [R3.5-S1] H3
                    next_tok = verified_0
                    h_last = h_single[:, -1:, :]
                    if serve_state is not None and serve_state.stopped:   # [R3.5-FIX] stopped here: no further draft
                        break
                    mtp_logits = self.bb.mtp_forward(
                        torch.tensor([[next_tok]], dtype=torch.long, device=self.device),
                        h_last,
                        cache=cache,
                        moe_fn=self.moe_dispatch,
                    )
                    draft_tok = self.bb.argmax_token(mtp_logits[0, -1])

        t_decode = time.perf_counter() - t_dec0
        if serve_state is not None:                       # [R3.5-S1] H4 (graph or eager final state; [R3.5-INT] + the
            serve_state.after_decode(self, cache, gen_ids, graph_done,     # graph MTP layer-45 history)
                                     mtp_graph=mtp_graph_stats is not None)
        decode_moe_s = self.moe_time_s
        n_new = max(1, len(gen_ids) - 1)
        vram_peak = (
            torch.cuda.max_memory_allocated(self.device) / (1024.0 ** 3)
            if self.device.type == "cuda"
            else 0.0
        )
        pool_stats = None
        if self.pool is not None:
            ps = dict(self.pool.stats)
            prev = getattr(self, "_pool_prev", {k: 0 for k in ps})
            pool_stats = {k: (round(ps[k] - prev.get(k, 0), 3) if isinstance(ps[k], float) else ps[k] - prev.get(k, 0)) for k in ps}
            tot = pool_stats["hits"] + pool_stats["misses"]
            pool_stats["hit_rate"] = round(pool_stats["hits"] / tot, 4) if tot else None
            pool_stats["gib_h2d"] = round(pool_stats.pop("bytes_h2d") / 2 ** 30, 3)
            self._pool_prev = ps
            print(f"    [pool] {pool_stats}", flush=True)
        res = {
            "pool": pool_stats,
            "prompt_tokens": len(prompt_ids),
            "new_tokens": len(gen_ids),
            "generated_ids": gen_ids,
            "prefill_s": round(t_prefill, 3),
            "prefill_tok_s": round(len(prompt_ids) / max(t_prefill, 1e-6), 2),
            "prefill_moe_s": round(prefill_moe_s, 3),
            "decode_s": round(t_decode, 3),
            "decode_tok_s": round(n_new / max(t_decode, 1e-6), 3),
            "step_ms_mean": round(t_decode / max(1, n_steps) * 1000.0, 2),
            "step_ms_moe_mean": round(decode_moe_s / max(1, n_steps) * 1000.0, 2),
            "step_ms_backbone_mean": round((t_decode - decode_moe_s) / max(1, n_steps) * 1000.0, 2),
            "mtp_accept_rate": round(mtp_accepted / mtp_total_drafts, 3) if mtp_total_drafts > 0 else None,
            "vram_peak_gib": round(vram_peak, 2),
            "host_mem": query_host_memory_gib(),
        }
        if mtp_graph_stats is not None:                    # [R3.5-M2] per-call MTP statistics of the graph path
            res["mtp_graph"] = mtp_graph_stats
        if _r39m_ps0 is not None:                          # [R3.9M] R39_MTP_STATS=1: pool counters of the decode phase
            _ps1 = dict(self.pool.stats)
            _pd = {k: (round(_ps1[k] - _r39m_ps0.get(k, 0), 3) if isinstance(_ps1[k], float)
                       else _ps1[k] - _r39m_ps0.get(k, 0))
                   for k in _ps1 if isinstance(_ps1[k], (int, float)) and not isinstance(_ps1[k], bool)}
            _tot = _pd.get("hits", 0) + _pd.get("misses", 0)
            _pd["hit_rate"] = round(_pd.get("hits", 0) / _tot, 4) if _tot else None
            _pd["gib_h2d"] = round(_pd.pop("bytes_h2d", 0) / 2 ** 30, 3)
            _pd["decode_s"] = round(t_decode, 3)
            _pd["new_tokens"] = n_new
            res["pool_decode"] = _pd
        if _r39m_fp0 is not None and graph_done:           # [R3.9M] R39_MTP_FP=1: one fingerprint per generated id
            _fpl = getattr(self.graph_decoder, "r39m_fp_last", None)
            if _fpl is not None and len(_fpl) == len(gen_ids) - 1:
                res["logit_fp"] = [_r39m_fp0] + [int(v) for v in _fpl]
        if pf_info is not None:                            # [R3.5-T item 10a] (only with R35_PREFILL_PREFETCH)
            res["prefill_prefetch"] = pf_info
        if continued is not None:                          # [R3.5-T H6] (only when the eager loop took over)
            continued["eager_tokens"] = len(gen_ids) - continued["graph_tokens"]
            res["continued_past_tmax"] = continued
        return res

    @torch.no_grad()
    def evaluate_perplexity(self, token_ids: List[int]) -> Dict[str, float]:
        t0 = time.perf_counter()
        ids_t = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        cache, n_pre, pc = self._restore_prefix(token_ids)
        if n_pre and "logits" not in pc:
            n_pre = 0  # the exact prefix logits are needed to keep all T-1 predictions
        logits, _ = self.bb.forward(
            ids_t[:, n_pre:], cache=cache if n_pre else None, moe_fn=self.moe_dispatch, last_only=False,
            max_layers=self.max_layers
        )
        trimmed = self.bb.vocab_ids is not None   # [R3.4-V] logits over the kept head rows only
        if n_pre:  # [R3.2-1Q.2] predictions made at the prefix positions come from the exact model
            pl = pc["logits"][:n_pre]
            if trimmed:
                pl = pl[:, self.bb.vocab_ids.cpu()]
            logits = torch.cat([pl.to(self.device, logits.dtype).unsqueeze(0), logits], dim=1)
        if os.environ.get("R32_SAVE_LOGITS"):  # [R3.2] Phase-3 KL / top-1 vs the exact-model logits
            # [R3.4-V] always full-vocabulary [T, V]: a trimmed head's dropped rows are -inf (kl_vs_exact_r32.py detects them)
            torch.save(self.bb.expand_logits(logits[0]).to(torch.float16).cpu(), os.environ["R32_SAVE_LOGITS"])
        shift_logits = logits[0, :-1, :].float()
        shift_labels = ids_t[0, 1:]
        if trimmed:
            # [R3.4-V] NLL over the positions whose label is a kept token (a dropped label has probability 0)
            # [R3.4 review] = the trimmed model's own NLL (softmax over the K kept rows), lower than the full model's NLL on
            # the same positions by -log P_full(kept tokens) and over fewer positions: NOT comparable with a
            # full-vocabulary ppl, so it is reported as nll_kept / ppl_kept (no "nll" / "ppl" keys). Judge V by the
            # dropped mass and the kept-token KL (kl_vs_exact_r32.py --vocab-keep); gate pruning without --vocab-keep.
            pos = self.bb.vocab_pos[shift_labels]
            ok = pos >= 0
            lp = torch.log_softmax(shift_logits, dim=-1)
            nll_k = -lp[ok].gather(1, pos[ok].view(-1, 1)).mean() if bool(ok.any()) else torch.tensor(float("nan"))
            preds = self.bb.vocab_ids[torch.argmax(shift_logits, dim=-1)]
            hit = preds == shift_labels
            return {
                "tokens": len(token_ids),
                "nll_kept": round(float(nll_k.item()), 4),
                "ppl_kept": round(float(torch.exp(nll_k).item()), 2),
                "top1_acc": round(float(hit.float().mean().item()), 4),          # all positions (dropped label = miss)
                "top1_acc_kept": round(float(hit[ok].float().mean().item()), 4) if bool(ok.any()) else float("nan"),
                "vocab_kept": int(self.bb.vocab_ids.numel()),
                "labels_scored": int(ok.sum().item()),
                "labels_dropped": int((~ok).sum().item()),
                "note": ("trimmed output head: nll_kept / ppl_kept = the trimmed model's NLL over the positions whose label "
                         "is kept (softmax over the kept rows) - not comparable with a full-vocabulary ppl; top1_acc over "
                         "all positions, top1_acc_kept over the scored ones"),
                "seconds": round(time.perf_counter() - t0, 2),
            }
        loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels)
        ppl = float(torch.exp(loss).item())
        preds = torch.argmax(shift_logits, dim=-1)
        acc = float((preds == shift_labels).float().mean().item())
        return {
            "tokens": len(token_ids),
            "nll": round(float(loss.item()), 4),
            "ppl": round(ppl, 2),
            "top1_acc": round(acc, 4),
            "seconds": round(time.perf_counter() - t0, 2),
        }


DEFAULT_PROMPTS = [
    ("capital", "What is the capital of Australia, and why was it chosen?"),
    ("code", "Write a concise Python function `is_prime(n: int) -> bool` and explain its time complexity."),
    ("math", "A train travels 180 km in 2.5 hours. What is its average speed in km/h? Show your calculation."),
    ("security", "Explain in two sentences how a stack canary mitigates stack buffer overflows."),
]


def run_doctor() -> None:
    print("======================================================================================================")
    print("  GLM-5.3-FLASH (320B-A18B) ROUND-3 APEX-FOLD-FP16 HARDWARE & MEMORY DOCTOR")
    print("  (Hard Byte-Locked Cuts | Native FP8 Backbone | 0 MiB Dense V_in Bases | Single-Slab Coalesced DMA)")
    print("======================================================================================================")
    for name in APEX_FOLD_PRESETS:
        b = compute_exact_hardware_budget(name)
        print(
            f"  * {name:18s} : {b['bpw']:.4f} bpw | Layer: {b['layer_mib']:7.2f} MiB "
            f"(Shared hdr: {b['shared_header_mib_per_layer']:.2f} MiB) | "
            f"43-Layer Archive: {b['total_43_layer_archive_gib']:5.2f} GiB\n"
            f"    -> RTX 5090 VRAM   : {b['total_vram_used_gib']:5.2f} GiB / 32.00 GiB "
            f"[Backbone ({b['backbone_dtype']}): {b['backbone_vram_gib']:.2f} GiB + "
            f"43 Hdrs: {b['all_43_shared_headers_vram_gib']:.2f} GiB + "
            f"{b['vram_layers_count']:2d}/43 Expert Slabs: {b['vram_slabs_gib']:5.2f} GiB + "
            f"KV/DMA: {b['kv_and_dma_vram_gib']:.2f} GiB]\n"
            f"    -> Host DDR5 RAM   : {b['host_ram_slabs_gib']:5.2f} GiB / 96.00 GiB "
            f"[{b['host_layers_count']:2d}/43 Expert Slabs]"
        )
    print("======================================================================================================")


def main() -> None:
    ap = argparse.ArgumentParser(description="GLM-5.3-Flash Round-3 APEX-FOLD-FP16 Inference & Benchmark CLI")
    ap.add_argument("--doctor", action="store_true", help="Print exact byte-locked memory budgets and exit")
    ap.add_argument("--backbone-dir", default=r"C:\models\GLM53-backbone")
    ap.add_argument("--fold-dir", default=r"C:\models\GLM53-APEX-FOLD-FP16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--backbone-mode", default="fp8", choices=["fp8", "bf16", "fp32"])
    ap.add_argument("--vram-budget-gib", type=float, default=30.6)
    ap.add_argument("--cold-mode", default="mmap", choices=["mmap", "ram"])
    ap.add_argument("--hobbit", type=float, default=1.0)
    ap.add_argument("--max-layers", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--thinking", action="store_true", help="Enable <think> tags in chat template")
    ap.add_argument("--mtp", action="store_true",
                    help="Enable batched N=2 Multi-Token Prediction (the eager MTP loop); [R3.5-M2] with --graphs and "
                         "env R35_MTP_GRAPHS=1: CUDA-graph MTP decoding (token-identical to --graphs alone; "
                         "[R3.5-FIX] opt-in - default 0 = the eager MTP loop, no graph capture at load)")
    ap.add_argument("--mtp-tau", type=float, default=None,
                    help="[R3.5-M2] graph MTP: verify a draft only if its probability >= tau (default env R35_MTP_TAU, "
                         "else 0.5; 0 = verify every draft)")
    ap.add_argument("--mtp-ngram", type=int, default=None,
                    help="[R3.5-M2] graph MTP: prompt-lookup drafts over the last n..2 tokens when the MTP draft is "
                         "missing or unsure (default env R35_MTP_NGRAM, else 0 = off; 2..8)")
    ap.add_argument("--mtp-pin45", default=None, choices=["none", "all"],
                    help="[R3.5-M2] graph MTP: pin every routable MTP-layer expert in the VRAM pool (default env "
                         "R35_MTP_PIN45, else none)")
    ap.add_argument("--ppl-text", default=None, help="Path to text file for teacher-forced perplexity evaluation")
    ap.add_argument("--ppl-max-tokens", type=int, default=128, help="[PATCH R3-07] perplexity window (shipped: 128)")
    ap.add_argument("--prompts-json", default=None, help="[PATCH R3-07] JSON list of {id, prompt} replacing the 4 defaults")
    ap.add_argument("--out", default=None, help="JSON path to write benchmark report")
    ap.add_argument("--serve", action="store_true",
                    help="Start OpenAI-compatible /v1/chat/completions HTTP server [R3.5-INT: the multi-turn server of "
                         "glm53_fold_serve.py - full chat history, tools, reasoning, SSE, conversation-state reuse; "
                         "--serve-last-message-only keeps the R3.x last-message handler]")
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--r31-runtime", action="store_true", help="[R3.2] disable all Round-3.2 fast paths (A/B baseline)")
    ap.add_argument("--expert-pool-gib", type=float, default=0.0, help="[R3.2-1.6] global VRAM expert cache (0 = off)")
    ap.add_argument("--no-prefetch", action="store_true", help="[R3.2-1.6] disable next-layer pre-gating prefetch")
    ap.add_argument("--no-sync-timing", action="store_true", help="[R3.2] skip per-MoE device syncs (true tok/s)")
    ap.add_argument("--gain-correct", default="", help="[R3.2-1Q.1] 'unbiased' (gate/up unbiased, down LS) or 'g,u,d' multipliers")
    ap.add_argument("--graphs", action="store_true", help="[R3.2-1.5] CUDA-graph greedy decode (requires --expert-pool-gib)")
    ap.add_argument("--warm-cache", action="store_true",
                    help="[R3.2-1.6d] read the whole archive once at start-up so expert misses hit the OS page cache")
    ap.add_argument("--prefix-cache", action="store_true",
                    help="[R3.2-1Q.2] restore the exact-model prefix states saved by the rebuild teacher pass")
    ap.add_argument("--fp8-diet", action="store_true", help="[R3.2-1.7] BF16 KDA q/k/v/o -> block-128 FP8 at load")
    ap.add_argument("--host-cache-gib", type=float, default=0.0, help="[R3.2-1.6c] pinned host-RAM L2 expert cache (GiB)")
    ap.add_argument("--graph-tmax", type=int, default=2048, help="[R3.2-1.5] static DSA cache length for graph decode")
    ap.add_argument("--prune-spec", default=None,
                    help="[R3.4-S4] JSON {\"layers\": {layer: [expert ids]}}: mask these experts out of routing (router "
                         "bias -inf, in place; graphs, prefetch prediction and MTP see it). Experts missing from a pruned "
                         "archive file are always masked")
    ap.add_argument("--no-mtp-layer", action="store_true",
                    help="[R3.4-E1] do not load the MTP layer 45 (backbone + expert file); incompatible with --mtp")
    ap.add_argument("--vocab-keep", default=None,
                    help="[R3.4-V] JSON keep list (build_run/r34_vocab_keep.py): the output head keeps only these token "
                         "rows; argmax maps back to the token id (graphs too); tokenizer and embeddings stay complete")
    import glm53_fold_serve as _SV                      # [R3.5-S1] H0: stdlib-only at import (no torch / CUDA work)
    _SV.add_serve_args(ap)                              # --serve-* options (+ --serve-last-message-only, [R3.5-INT])
    args = ap.parse_args()

    if args.doctor:
        run_doctor()
        return
    if args.mtp and args.no_mtp_layer:
        ap.error("--mtp needs the MTP layer: drop --no-mtp-layer")
    warmup_mode = os.environ.get("R35_WARMUP", "0").strip() or "0"     # [R3.5-T item 10d] checked before the load
    if warmup_mode not in ("0", "1", "2"):
        ap.error(f"R35_WARMUP must be 0, 1 or 2, not {warmup_mode!r}")
    serve_v2 = False
    if args.serve:                                      # [R3.5-INT] --serve = the multi-turn server unless opted out
        try:
            serve_v2 = _SV.serve_v2_selected(args)
        except ValueError as e:
            ap.error(str(e))

    tok = RealTokenizer(args.backbone_dir)
    # [R3.9S] a Round 3.9 QTIP HYB archive (layer_XX_qtip_hyb.pt) is served by QtipMoEModelRunner
    # (glm53_fold_qtip_runtime), which is this file's runner class with its own archive loading; every option of
    # this CLI / server then applies to it unchanged. capture_graphs=False: the graph block below builds the decoders.
    runner_cls, runner_kw = ApexFoldMoEModelRunner, {}
    if os.path.isdir(args.fold_dir) and any(n.endswith("_qtip_hyb.pt") for n in os.listdir(args.fold_dir)):
        import glm53_fold_qtip_runtime as _QR
        runner_cls, runner_kw = _QR.qtip_runner_class(ApexFoldMoEModelRunner), {"capture_graphs": False}
        print(f"[load] [R3.9S] QTIP HYB archive in {args.fold_dir}: runner QtipMoEModelRunner", flush=True)
    runner = runner_cls(
        backbone_dir=args.backbone_dir,
        fold_dir=args.fold_dir,
        device=args.device,
        backbone_mode=args.backbone_mode,
        vram_budget_gib=args.vram_budget_gib,
        cold_mode=args.cold_mode,
        hobbit_tau=args.hobbit,
        max_layers=args.max_layers,
        fast=not args.r31_runtime,
        fp8_diet=args.fp8_diet,
        host_cache_gib=args.host_cache_gib,
        prefix_cache=args.prefix_cache,
        expert_pool_gib=args.expert_pool_gib,
        prefetch=not args.no_prefetch,
        sync_timing=not args.no_sync_timing,
        prune_spec=args.prune_spec,
        load_mtp=not args.no_mtp_layer,
        vocab_keep=args.vocab_keep,
        **runner_kw,                        # [R3.9S] empty for the .npz archives
    )

    if args.warm_cache:
        runner.warm_page_cache(args.fold_dir)
    if args.gain_correct:
        runner.set_gain(args.gain_correct)
    if runner.fast and runner.device.type == "cuda":
        import glm53_fold_kernels as _KR
        # [R3.4 review] + fp8_gemv_rows only for a trimmed FP8 head (E2), which uses it
        print(f"[load] [R3.4-2h] NVRTC kernels compiled in "
              f"{_KR.precompile(fp8_rows=bool(getattr(runner.bb.head, 'row_scales', False))):.1f}s", flush=True)
    if args.graphs:
        from glm53_fold_graphs import GraphDecoder
        runner.graph_decoder = GraphDecoder(runner, t_max=args.graph_tmax)
        # [R3.7 Stage 7] R37_DUAL_GRAPHS=1 (only with the long-context decoder): keep the window decoder as well
        if os.environ.get("R37_DUAL_GRAPHS", "0") == "1" and getattr(runner.graph_decoder, "r36_longctx", False):
            runner.graph_decoder_long = runner.graph_decoder
            runner.graph_decoder_std = GraphDecoder(runner, t_max=args.graph_tmax, longctx=False)
        # [R3.4-2h] capture the decode graphs at load / --serve start-up (was: inside the first request's decode)
        # [R3.4-fix2] only if graph decode will run: generate() decodes eagerly with --mtp (MTP layer loaded) and a
        # --ppl-text run never decodes; decode() still captures lazily if it is reached
        mtp_on = args.mtp and runner.cfg.num_hidden_layers in runner.moe_layers
        # [R3.5-M2] --mtp with --graphs decodes on the graph path only with env R35_MTP_GRAPHS=1 ([R3.5-FIX F1] opt-in;
        # default 0 = the base path above): then capture the N = 1, N = 2 verify and MTP draft sets at load
        mtp_graph = bool(mtp_on and runner.graph_decoder.mtp_enabled())
        if mtp_graph:
            from glm53_fold_graphs import MtpConfig
            runner.graph_decoder.mtp_cfg = MtpConfig.from_env().override(tau=args.mtp_tau, ngram=args.mtp_ngram,
                                                                         pin45=args.mtp_pin45)
        ppl_run = not args.serve and bool(args.ppl_text) and os.path.exists(args.ppl_text)
        if (mtp_on and not mtp_graph) or ppl_run:
            print(f"[load] [R3.4-fix2] graph capture at load skipped ({'--mtp' if mtp_on else '--ppl-text'}: "
                  "no graph decode in this run)", flush=True)
        else:
            t_cap = time.perf_counter()
            runner.graph_decoder.capture()
            print(f"[load] [R3.4-2h] decode graphs captured in {time.perf_counter() - t_cap:.1f}s", flush=True)
            if runner.graph_decoder_std is not None:         # [R3.7 Stage 7] R37_DUAL_GRAPHS
                t_cap = time.perf_counter()
                runner.graph_decoder_std.capture()
                print(f"[load] [R3.7] second decoder captured in {time.perf_counter() - t_cap:.1f}s: requests that fit "
                      f"the {runner.graph_decoder_std.t_max}-token window use the unchanged window graphs, all others "
                      f"the long-context graphs (capacity {runner.graph_decoder_long.t_max}) (R37_DUAL_GRAPHS=1)",
                      flush=True)
            if mtp_graph:                                    # [R3.5-M2]
                t_cap = time.perf_counter()
                mi = runner.graph_decoder.capture_mtp(preserve_state=False)   # load_cache() precedes every decode
                pi = mi["pair_info"]
                print(f"[load] [R3.5-M2] MTP graph sets captured in {time.perf_counter() - t_cap:.1f}s: N=2 verify "
                      f"(rows={pi['rows']}, +{pi['vram_alloc_gib_delta']} GiB), draft set (+{mi['vram_alloc_gib_delta']}"
                      f" GiB{'; ' + mi['pin_note'] if mi['pin_note'] else ''}); policy {mi['cfg']}", flush=True)
        gs = getattr(runner.graph_decoder, "guard_stats", None)
        if getattr(runner.graph_decoder, "capture_guard", False):          # [R3.5-T item 10d]
            print(f"[load] [R3.5-T] capture guard (R35_CAPTURE_GUARD=1): captures in mode thread_local, pool quiesced "
                  f"{(gs or {}).get('quiesce_calls', 0)} time(s) before them ({(gs or {}).get('quiesce_s', 0.0):.3f}s)",
                  flush=True)
    if getattr(runner, "continue_past_tmax", False):     # [R3.5-T H6]
        gd_ = runner.graph_decoder
        h6_ok = gd_ is not None and bool(getattr(gd_, "dsa_indexer", False))
        print("[load] [R3.5-T] H6 (R35_CONTINUE_PAST_TMAX=1): " + (
            f"answers continue on the eager path past the graph window (t_max {gd_.t_max})" if h6_ok else
            "not needed without --graphs (the eager loop has no window)" if gd_ is None else
            "IGNORED - the graphs keep no DSA indexer caches (needs R34_DSA_INDEXER=1)"), flush=True)
    # [R3.5-T item 10d] env R35_WARMUP=1|2: warm-up request(s) after the capture (and before --serve listens), so the
    # first real request pays no one-time cost: 1 = a short chat prompt (the prefill path of typical prompts), 2 = + a
    # 96-token prompt (the dense prefill path, N x K >= 512); 4 new tokens each. Bytes only (counters reset after it).
    if warmup_mode != "0" and not (not args.serve and args.ppl_text and os.path.exists(args.ppl_text)):
        w_ids = tok.encode_chat("Hello! In one sentence, what can you help me with?", thinking=args.thinking)
        w_prompts = [w_ids] + ([(w_ids * (96 // max(1, len(w_ids)) + 1))[:96]] if warmup_mode == "2" else [])
        wi = runner.warmup(w_prompts, eos_ids=tok.eos_ids, use_mtp=args.mtp)
        print(f"[load] [R3.5-T] warm-up (R35_WARMUP={warmup_mode}): {len(w_prompts)} request(s) in {wi['seconds']:.1f}s "
              f"(prompt tokens {[r['prompt_tokens'] for r in wi['runs']]}, prefill s "
              f"{[r['prefill_s'] for r in wi['runs']]}; pool quiesced {wi['quiesce_s']:.3f}s)", flush=True)

    if args.serve and serve_v2:                         # [R3.5-S1] H0 / [R3.5-INT]: multi-turn server (history, tools,
        _SV.serve_main(runner, tok, args)               # reasoning, SSE, conversation-state reuse)
        return

    if args.serve:                                      # --serve-last-message-only: the R3.x handler (A/B baseline)
        class ChatHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.startswith("/v1/models"):
                    body = json.dumps({"object": "list", "data": [{"id": "glm-5.3-flash-apex-fold-fp16", "object": "model"}]}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                if self.path.startswith("/v1/chat/completions"):
                    length = int(self.headers.get("Content-Length", "0"))
                    req = json.loads(self.rfile.read(length).decode("utf-8"))
                    msgs = req.get("messages", [])
                    prompt = msgs[-1].get("content", "") if msgs else ""
                    max_tok = int(req.get("max_tokens", args.max_new_tokens))
                    p_ids = tok.encode_chat(prompt, thinking=args.thinking)
                    res = runner.generate(p_ids, max_new_tokens=max_tok, eos_ids=tok.eos_ids, use_mtp=args.mtp)
                    text = tok.decode(res["generated_ids"])
                    resp = {
                        "id": "chatcmpl-apexfold",
                        "object": "chat.completion",
                        "model": "glm-5.3-flash-apex-fold-fp16",
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": res["prompt_tokens"], "completion_tokens": res["new_tokens"]},
                        "telemetry": res,
                    }
                    body = json.dumps(resp).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

        print(f"[serve] Listening on http://127.0.0.1:{args.port}/v1/chat/completions", flush=True)
        HTTPServer(("127.0.0.1", args.port), ChatHandler).serve_forever()
        return

    report: Dict[str, object] = {"placement": runner.placement, "generations": []}
    if args.prune_spec or args.no_mtp_layer or args.vocab_keep or runner.prune_masks:    # [R3.4 shrink] settings
        report["shrink"] = {"prune_spec": args.prune_spec, "masked_layers": len(runner.prune_masks),
                            "masked_experts": sum(len(v) for v in runner.prune_masks.values()),
                            "archive_missing_experts": sum(len(v) for v in runner.archive_missing.values()),
                            "no_mtp_layer": bool(args.no_mtp_layer), "vocab_keep": args.vocab_keep,
                            "vocab": runner.vocab_info}

    if args.ppl_text and os.path.exists(args.ppl_text):
        raw_txt = open(args.ppl_text, "r", encoding="utf-8").read()
        # [PATCH R3-07] score raw text after the model's BOS pair "[gMASK]<sop>" (the chat template prefix)
        ppl_ids = (tok.encode_raw("[gMASK]<sop>") + tok.encode_raw(raw_txt))[: args.ppl_max_tokens]
        ppl_res = runner.evaluate_perplexity(ppl_ids)
        report["ppl"] = ppl_res
        if "ppl" in ppl_res:
            print(f"[ppl] {ppl_res['tokens']} tokens: perplexity {ppl_res['ppl']:.2f}, top-1 acc {ppl_res['top1_acc']:.3f} ({ppl_res['seconds']:.1f}s)", flush=True)
        else:                               # [R3.4 review] trimmed head: kept-label perplexity, labelled as such
            print(f"[ppl] {ppl_res['tokens']} tokens, TRIMMED head ({ppl_res['vocab_kept']} rows): kept-label perplexity "
                  f"{ppl_res['ppl_kept']:.2f} over {ppl_res['labels_scored']} labels ({ppl_res['labels_dropped']} dropped; "
                  f"not comparable with a full-vocabulary ppl), top-1 acc {ppl_res['top1_acc']:.3f} "
                  f"({ppl_res['seconds']:.1f}s)", flush=True)
    else:
        prompts = DEFAULT_PROMPTS
        if args.prompts_json and os.path.exists(args.prompts_json):  # [PATCH R3-07]
            prompts = [(p["id"], p["prompt"]) for p in json.load(open(args.prompts_json, "r", encoding="utf-8"))]
        for pid, p_txt in prompts:
            p_ids = tok.encode_chat(p_txt, thinking=args.thinking)
            res = runner.generate(p_ids, max_new_tokens=args.max_new_tokens, eos_ids=tok.eos_ids, use_mtp=args.mtp)
            res["id"] = pid
            res["prompt"] = p_txt
            res["text"] = tok.decode(res["generated_ids"])
            report["generations"].append(res)
            print(
                f"=== [{pid}] prompt {res['prompt_tokens']} tok, prefill {res['prefill_s']:.2f}s "
                f"({res['prefill_tok_s']:.2f} tok/s); decode {res['new_tokens']} tok in {res['decode_s']:.2f}s = "
                f"{res['decode_tok_s']:.3f} tok/s (step {res['step_ms_mean']:.1f} ms: experts {res['step_ms_moe_mean']:.1f} + "
                f"backbone {res['step_ms_backbone_mean']:.1f}); VRAM peak {res['vram_peak_gib']:.2f} GiB\n"
                f"    TEXT: {repr(res['text'])}",
                flush=True,
            )
            if res.get("mtp_graph"):                         # [R3.5-M2] graph-path MTP statistics
                s = res["mtp_graph"]
                print(f"    [mtp-graph] {s['tokens']} tok in {s['steps']} steps ({s['tokens_per_step']:.3f} tok/step): "
                      f"{s['pair_steps']} verify steps, accepted {s['accepted']}/{s['verified']} (rate {s['accept_rate']}, "
                      f"MTP {s['mtp_accept_rate']}), N=1 steps {s['n1_reasons']}, pauses {s['pauses']} (EMA) + "
                      f"{s['pauses_low_conf']} (unsure run), "
                      f"draft {s['draft_ms_per_step']:.2f} ms/step", flush=True)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[done] saved report -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
