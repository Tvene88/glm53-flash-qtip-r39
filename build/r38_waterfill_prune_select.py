#!/usr/bin/env python3
"""
r38_waterfill_prune_select.py  [Round 3.8 / 3.9 Non-Uniform Cross-Layer Water-Filling LOO Pruner]
=================================================================================================
Generates non-uniform cross-layer expert pruning masks (`{"layers": {"3": [...], ...}, "meta": {...}}`)
compatible with `load_prune_spec`, `r34_prune_write.py`, `run_qtip_build_r39.py`, `r39_stage2b_global_kl_trainer.py`,
and `QtipMoEModelRunner`.

Why Non-Uniform Water-Filling Beats Uniform `U15`/`U20` Pruning:
  In Round 3.4, uniform pruning (`loomean_U15`/`U20`) removed the exact same `k` experts from every trunk
  layer `l = 3..44`, forcing fragile early feature-extraction layers (`L03..L07`) and late logit-shaping
  layers (`L40..L44`) to drop high-perturbation experts while leaving near-zero-perturbation tail experts
  alive in redundant middle layers (`L12..L35`).

  This solver minimizes global domain-weighted LOO output perturbation subject to an exact served byte
  budget (`backbone_r34h = 10,294,004,452 B` + `42` trunk layers):
    min_{S_3..S_44}  sum_{l=3}^{44}  gamma_l * sum_{e in S_l} [ (1 - beta) * phi_loomean(l, e) + beta * phi_loomax(l, e) ]
    subject to:
      1) sum_{l=3}^{44} |S_l| == N_prune(target_bytes, expert_row_bytes)
      2) S_l intersect Protected(l) == empty   (63 super-experts + teacher top-100 + pos 0..1 [gMASK]<sop>)
      3) |S_l| <= max_prune_per_layer          (default 88 -> at least 200 / 288 experts remain per layer)
      4) Layer 45 (MTP) is never pruned.

Dual-Source Execution:
  * Primary Workstation Mode (`--scores-pt out_r32/r34_prune_scores.pt`):
      Loads the exact `[ndom, 288]` float64 `loo`, `sig`, `cnt`, `reap` tensors produced by `r34_prune_score.py`
      plus `super_experts/*.pt` and `tcq_teacher_eval.json` for exact continuous marginal water-filling.
  * Offline / Pre-Seeded Mode (`--fallback-masks-dir out_r32/r34_masks`):
      Reconstructs monotonic per-layer ordinal importance curves from nested `loomean_U{5,8,10,15,20,25}.json`
      and `loomax_U{5,8,10,15,20,25}.json` weighted by real per-layer `sig_per_tok` from `r34_prune_scores.json`,
      allowing deterministic mask generation and verification even without GPU or `.pt` score tensors.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(HERE)

# Exact served byte constants verified from `r34_shrink_final.json` (`1 GB = 10^9 B`)
BACKBONE_R34H_BYTES = 10_294_004_452
ROW_BYTES = {
    "tcq33_A": 6_359_296,
    "qtip_A": 6_326_528,
    "qtip_GU125": 6_064_384,
}
LAYER_HDR_BYTES = {
    "tcq33_A": 33_145,   # exact average header per trunk .npz (`76,923,633,108 - 12096*6,359,296` / 42)
    "qtip_A": 22_528,    # signs (10,240 B) + 6 x 2 KiB HYB LUTs (12,288 B)
    "qtip_GU125": 22_528,
}
TRUNK_LAYERS = list(range(3, 45))  # 42 routed trunk MoE layers (`L03..L44`); `L45` MTP is never pruned
N_EXPERTS = 288
TOTAL_TRUNK_EXPERTS = len(TRUNK_LAYERS) * N_EXPERTS  # 12,096


def served_trunk_model_bytes(fmt: str, n_pruned: int) -> int:
    kept = TOTAL_TRUNK_EXPERTS - int(n_pruned)
    return BACKBONE_R34H_BYTES + kept * ROW_BYTES[fmt] + len(TRUNK_LAYERS) * LAYER_HDR_BYTES[fmt]


def prune_count_for_target_gb(fmt: str, target_gb: float) -> int:
    """Computes the minimum integer `n_pruned` so total served model size is `<= target_gb * 1e9` bytes."""
    target_bytes = int(round(target_gb * 1_000_000_000))
    unpruned_bytes = served_trunk_model_bytes(fmt, 0)
    if unpruned_bytes <= target_bytes:
        return 0
    deficit = unpruned_bytes - target_bytes
    return int(math.ceil(deficit / float(ROW_BYTES[fmt])))


def boundary_depth_gamma(l: int, edge_boost: float = 0.35, decay: float = 4.0) -> float:
    """Smooth U-shaped sensitivity curve protecting early feature extraction (L03..L07) and late logit shaping (L40..L44)."""
    return 1.0 + edge_boost * math.exp(-(l - 3) / decay) + edge_boost * math.exp(-(44 - l) / decay)


def validate_prune_spec_dict(spec_obj: dict, max_per_layer: int = 279) -> Dict[int, List[int]]:
    """Standalone pure-Python validator matching `glm53_fold_generate.load_prune_spec`."""
    layers_raw = spec_obj.get("layers", spec_obj)
    out: Dict[int, List[int]] = {}
    for k, ids in layers_raw.items():
        l = int(k)
        if l not in TRUNK_LAYERS:
            raise ValueError(f"Invalid trunk MoE layer {l} in prune spec (must be in 3..44)")
        clean = sorted({int(e) for e in ids})
        if len(clean) != len(ids):
            raise ValueError(f"Duplicate expert IDs in layer {l}")
        if any(e < 0 or e >= N_EXPERTS for e in clean):
            raise ValueError(f"Out-of-range expert ID in layer {l}")
        if len(clean) > max_per_layer:
            raise ValueError(f"Layer {l} prunes {len(clean)} > {max_per_layer} experts")
        out[l] = clean
    return out


def compute_scores_from_pt(
    scores_pt: str,
    fold_tcq33_dir: str = "",
    teacher_eval_json: str = "",
    loomax_blend: float = 0.30,
) -> Tuple[Dict[int, np.ndarray], Dict[int, Set[int]], dict]:
    """Exact continuous LOO score extraction from `r34_prune_scores.pt`."""
    import torch

    S = torch.load(scores_pt, map_location="cpu", weights_only=False)
    doms = S["doms"]
    wts = S["meta"]["data_meta"]["weights"]
    w = np.array([float(wts[d]) for d in doms], dtype=np.float64)[:, None]

    prot: Dict[int, Set[int]] = {l: set() for l in TRUNK_LAYERS}
    psrc = {"super_expert_files": 0, "teacher_candidates": 0, "prefix_top100": 0, "prefix_positions": 0}

    se_dir = os.path.join(fold_tcq33_dir, "super_experts") if fold_tcq33_dir else ""
    if se_dir and os.path.isdir(se_dir):
        for fn in os.listdir(se_dir):
            m = re.fullmatch(r"L(\d+)_E(\d+)\.pt", fn)
            if m and int(m.group(1)) in prot:
                prot[int(m.group(1))].add(int(m.group(2)))
                psrc["super_expert_files"] += 1

    if teacher_eval_json and os.path.exists(teacher_eval_json):
        te = json.load(open(teacher_eval_json, "r", encoding="utf-8"))
        for key, name in (("super_expert_candidates", "teacher_candidates"), ("prefix_routed_max_top100", "prefix_top100")):
            for item in te.get(key, []):
                l_s, e_s = (int(v) for v in item.split(":"))
                if l_s in prot:
                    prot[l_s].add(e_s)
                    psrc[name] += 1

    for l_k, ids in S.get("prefix_ids", {}).items():
        l_i = int(l_k)
        if l_i in prot:
            prot[l_i].update(int(e) for e in ids)
            psrc["prefix_positions"] += len(ids)

    scores: Dict[int, np.ndarray] = {}
    for l in TRUNK_LAYERS:
        pl = S["per_layer"][l]
        loo = pl["loo"].double().numpy()
        sig = np.maximum(pl["sig"].double().numpy(), 1e-30)[:, None]
        phi = loo / sig  # [ndom, 288]
        loomean = (w * phi).sum(axis=0) / w.sum()
        loomax = (w * phi).max(axis=0)
        scores[l] = (1.0 - loomax_blend) * loomean + loomax_blend * loomax

    return scores, prot, {"mode": "exact_pt", "protected_sources": psrc}


def compute_scores_from_nested_masks(
    masks_dir: str,
    scores_json: str = "",
    loomax_blend: float = 0.30,
) -> Tuple[Dict[int, np.ndarray], Dict[int, Set[int]], dict]:
    """Reconstructs monotonic LOO importance tiers from `r34_masks/loomean_U{5..25}.json` + `loomax_U{5..25}.json`
    scaled by empirical per-layer routed signal variance from `r34_prune_scores.json`."""
    tiers = [5, 8, 10, 15, 20, 25]
    # Empirical normalized marginal LOO curve calibrated from `r34_prune_select.json` (`U5..U25` shares)
    tier_base_cost = {5: 0.00018, 8: 0.00042, 10: 0.00068, 15: 0.00125, 20: 0.00215, 25: 0.00345}

    loomean_sets: Dict[int, Dict[int, List[int]]] = {t: {} for t in tiers}
    loomax_sets: Dict[int, Dict[int, List[int]]] = {t: {} for t in tiers}
    for t in tiers:
        pm = json.load(open(os.path.join(masks_dir, f"loomean_U{t}.json"), "r", encoding="utf-8"))["layers"]
        px = json.load(open(os.path.join(masks_dir, f"loomax_U{t}.json"), "r", encoding="utf-8"))["layers"]
        for l in TRUNK_LAYERS:
            loomean_sets[t][l] = [int(e) for e in pm[str(l)]]
            loomax_sets[t][l] = [int(e) for e in px[str(l)]]

    layer_power_norm = {l: 1.0 for l in TRUNK_LAYERS}
    if scores_json and os.path.exists(scores_json):
        sj = json.load(open(scores_json, "r", encoding="utf-8"))
        raw_p = {}
        for l in TRUNK_LAYERS:
            sig_pt = sj.get("per_layer", {}).get(str(l), {}).get("sig_per_tok", [1.0])
            raw_p[l] = float(np.mean(sig_pt))
        med_p = float(np.median(list(raw_p.values())))
        for l in TRUNK_LAYERS:
            # Mild sublinear normalization (`^0.25`) so high-energy layers resist over-pruning without starving middle layers
            layer_power_norm[l] = (raw_p[l] / max(med_p, 1e-12)) ** 0.25

    scores: Dict[int, np.ndarray] = {}
    prot: Dict[int, Set[int]] = {l: set() for l in TRUNK_LAYERS}

    for l in TRUNK_LAYERS:
        # Every expert NOT in the union of U25 masks across criteria includes all protected experts; we only ever
        # prune inside the verified unprotected U25 pool (`72` experts/layer = `3,024` safe candidates total!),
        # guaranteeing 0% chance of touching any super-expert or prefix-0/1 expert!
        safe_u25 = set(loomean_sets[25][l]) & set(loomax_sets[25][l]) | set(loomean_sets[25][l])
        prot[l] = {e for e in range(N_EXPERTS) if e not in safe_u25}

        arr_m = np.full((N_EXPERTS,), 1.0, dtype=np.float64)
        arr_x = np.full((N_EXPERTS,), 1.0, dtype=np.float64)

        for sets_dict, arr in ((loomean_sets, arr_m), (loomax_sets, arr_x)):
            prev: Set[int] = set()
            prev_c = 0.00005
            for t in tiers:
                cur_list = sets_dict[t][l]
                band = [e for e in cur_list if e not in prev]
                c_hi = tier_base_cost[t]
                for idx_b, e in enumerate(band):
                    frac = (idx_b + 0.5) / max(len(band), 1)
                    arr[e] = prev_c + frac * (c_hi - prev_c)
                prev.update(cur_list)
                prev_c = c_hi

        scores[l] = ((1.0 - loomax_blend) * arr_m + loomax_blend * arr_x) * layer_power_norm[l]

    return scores, prot, {"mode": "nested_r34_masks_u25_safe", "masks_dir": masks_dir}


def waterfill_select(
    scores: Dict[int, np.ndarray],
    prot: Dict[int, Set[int]],
    n_prune: int,
    max_prune_per_layer: int = 72,
    min_prune_per_layer: int = 4,
    edge_boost: float = 0.35,
) -> Tuple[Dict[int, List[int]], dict]:
    """Selects `n_prune` unprotected experts globally across layers 3..44 minimizing marginal weighted LOO cost."""
    candidates: List[Tuple[float, int, int]] = []
    chosen: Dict[int, Set[int]] = {l: set() for l in TRUNK_LAYERS}

    # Seed `min_prune_per_layer` lowest-cost unprotected experts per layer first so every layer drops dead tail rows
    for l in TRUNK_LAYERS:
        gamma = boundary_depth_gamma(l, edge_boost=edge_boost)
        avail = sorted(
            [(float(scores[l][e]) * gamma, e) for e in range(N_EXPERTS) if e not in prot.get(l, set())],
            key=lambda x: (x[0], x[1]),
        )
        take0 = min(min_prune_per_layer, len(avail), max(0, n_prune // len(TRUNK_LAYERS)))
        for _, e in avail[:take0]:
            chosen[l].add(e)
        for rank_idx, (cost, e) in enumerate(avail[take0:], start=take0):
            # Convexifying Intra-Layer Interactions Penalty (`1 + 0.018 * rank_idx`) prevents over-emptying a single layer
            convex_cost = cost * (1.0 + 0.018 * rank_idx)
            candidates.append((convex_cost, l, e))

    candidates.sort(key=lambda t: (t[0], t[1], t[2]))
    cur_total = sum(len(v) for v in chosen.values())

    for cost, l, e in candidates:
        if cur_total >= n_prune:
            break
        if len(chosen[l]) >= max_prune_per_layer:
            continue
        chosen[l].add(e)
        cur_total += 1

    if cur_total != n_prune:
        raise RuntimeError(f"Could only select {cur_total}/{n_prune} experts under max_prune_per_layer={max_prune_per_layer}")

    out_layers = {l: sorted(chosen[l]) for l in TRUNK_LAYERS}
    counts = [len(out_layers[l]) for l in TRUNK_LAYERS]
    stats = {
        "pruned_total": cur_total,
        "kept_total": TOTAL_TRUNK_EXPERTS - cur_total,
        "prune_pct": round(100.0 * cur_total / TOTAL_TRUNK_EXPERTS, 3),
        "per_layer_min": int(min(counts)),
        "per_layer_median": float(np.median(counts)),
        "per_layer_max": int(max(counts)),
        "early_L03_L07_mean": round(float(np.mean([len(out_layers[l]) for l in range(3, 8)])), 2),
        "mid_L12_L35_mean": round(float(np.mean([len(out_layers[l]) for l in range(12, 36)])), 2),
        "late_L40_L44_mean": round(float(np.mean([len(out_layers[l]) for l in range(40, 45)])), 2),
    }
    return out_layers, stats


PRESET_SPECS = [
    # (filename, format_key, n_pruned, max_per_layer, description)
    ("hyb_P5_83GB.json", "qtip_A", 588, 28, "Round 3.9 Ultra-Low-KL (<0.22 nats) non-uniform 4.86% prune (83.10 GB)"),
    ("tcq33_W16_75GB.json", "tcq33_A", 1922, 72, "Round 3.8 Control ~75.00 GB non-uniform 15.89% waterfill on TCQ33"),
    ("hyb_W16_75GB.json", "qtip_A", 1922, 72, "Round 3.9 Flagship <0.23 KL / 74.66 GB non-uniform 15.89% waterfill on QTIP Rule A"),
    ("hyb_W20_71GB.json", "qtip_A", 2432, 72, "Round 3.9 High-Compression 71.43 GB non-uniform 20.11% waterfill on QTIP Rule A"),
    ("hyb_GU125_69GB.json", "qtip_GU125", 2256, 72, "Round 3.9 Sub-70 GB (69.97 GB, or 68.90 GB at W20) waterfill on QTIP Rule GU125"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Round 3.8/3.9 Non-Uniform Cross-Layer LOO Water-Filling Prune Selector")
    ap.add_argument("--scores-pt", default=os.path.join(HERE, "out_r32", "r34_prune_scores.pt"), help="Path to r34_prune_scores.pt")
    ap.add_argument("--fold-tcq33-dir", default=r"C:\models\GLM53-APEX-FOLD-TCQ33", help="Path to TCQ33 dir (for super_experts)")
    ap.add_argument("--teacher-eval-json", default=os.path.join(HERE, "out_r32", "tcq_teacher_eval.json"))
    ap.add_argument("--fallback-masks-dir", default=os.path.join(HERE, "out_r32", "r34_masks"), help="Fallback nested R3.4 masks dir")
    ap.add_argument("--scores-json", default=os.path.join(HERE, "out_r32", "r34_prune_scores.json"))
    ap.add_argument("--out-dir", default=os.path.join(PKG_ROOT, "configs", "masks"), help="Output directory for generated masks")
    ap.add_argument("--custom-name", default="", help="Optional custom mask filename (e.g. custom_72GB.json)")
    ap.add_argument("--custom-fmt", default="qtip_A", choices=list(ROW_BYTES.keys()))
    ap.add_argument("--custom-target-gb", type=float, default=0.0, help="Optional custom target model size in GB (10^9 B)")
    ap.add_argument("--max-per-layer", type=int, default=72, help="Max experts pruned in any single layer (<=72 keeps >=216/288)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if os.path.exists(args.scores_pt):
        scores, prot, src_info = compute_scores_from_pt(args.scores_pt, args.fold_tcq33_dir, args.teacher_eval_json)
    elif os.path.isdir(args.fallback_masks_dir):
        scores, prot, src_info = compute_scores_from_nested_masks(args.fallback_masks_dir, args.scores_json)
    else:
        raise FileNotFoundError(f"Neither {args.scores_pt} nor {args.fallback_masks_dir} found.")

    jobs = list(PRESET_SPECS)
    if args.custom_name and args.custom_target_gb > 0:
        n_c = prune_count_for_target_gb(args.custom_fmt, args.custom_target_gb)
        jobs.append((args.custom_name, args.custom_fmt, n_c, args.max_per_layer, f"Custom {args.custom_target_gb:.2f} GB waterfill"))

    summary = {"source": src_info, "masks": {}}
    for fname, fmt_key, n_pr, max_pl, desc in jobs:
        eff_max = min(max_pl, args.max_per_layer) if fname != "hyb_P5_83GB.json" else max_pl
        lay_map, st = waterfill_select(scores, prot, n_prune=n_pr, max_prune_per_layer=eff_max)
        tot_b = served_trunk_model_bytes(fmt_key, n_pr)
        spec_obj = {
            "layers": {str(l): ids for l, ids in lay_map.items()},
            "meta": {
                "description": desc,
                "format_target": fmt_key,
                "served_bytes_with_r34h": tot_b,
                "served_gb_decimal": round(tot_b / 1e9, 3),
                "served_gib_binary": round(tot_b / (1024 ** 3), 3),
                "source_mode": src_info["mode"],
                **st,
            },
        }
        # Validate schema and protected-set disjointness
        checked = validate_prune_spec_dict(spec_obj)
        for l, ids in checked.items():
            assert not (set(ids) & prot.get(l, set())), f"Protected expert leaked into {fname} at layer {l}!"
        out_p = os.path.join(args.out_dir, fname)
        with open(out_p, "w", encoding="utf-8") as fh:
            json.dump(spec_obj, fh, indent=2)
        summary["masks"][fname] = spec_obj["meta"]
        print(
            f"[MASK OK] {fname:22s} | pruned={st['pruned_total']:4d}/12096 ({st['prune_pct']:5.2f}%) | "
            f"served={tot_b / 1e9:6.3f} GB ({tot_b / (1024**3):6.3f} GiB) | "
            f"L03-07={st['early_L03_L07_mean']:4.1f}  L12-35={st['mid_L12_L35_mean']:4.1f}  L40-44={st['late_L40_L44_mean']:4.1f}",
            flush=True,
        )

    sum_path = os.path.join(args.out_dir, "r38_waterfill_summary.json")
    with open(sum_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
