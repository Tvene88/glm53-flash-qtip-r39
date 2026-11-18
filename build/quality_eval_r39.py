#!/usr/bin/env python3
"""
quality_eval_r39.py  [Round 3.9 Full-Vocabulary Quality & Alternating Speed Gate Evaluator]
===========================================================================================
Upgrades `quality_eval_r37.py` to evaluate Round 3.9 `QtipMoEModelRunner` (`layer_XX_qtip_hyb.pt` +
`merged_routers_bf16.pt`) AND Round 3.7/3.8 `ApexFoldMoEModelRunner` (`layer_XX_apex_fold.npz` +
optional `merged_routers_bf16.pt`) on the exact declared 111-row quality suite (`29 qbench`, `40 devt`,
`42 devx`) plus an optional alternating 3-round throughput & forced-decode parity gate (`--speed-gate`).

Enforces all Round 3.7 review invariants (`INT-01`, `INT-02`):
  1. Single-runtime module provenance check (`glm53_*` modules must all resolve inside `--runtime-dir`).
  2. True `chunk=1` `GraphDecoder` (`N=1` captured CUDA graphs after 2-token eager prefix) required for
     passing decode-quality budgets (`<= 0.260 nats` for Sub-70 GB; `<= 0.230 nats` for 74.66 GB R3.9).
  3. Full FP8 `154,880`-vocab teacher head used for `qbench` KL (`pos >= 10`) and `dev` KL (`pos >= 2`).
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
DEFAULT_RUNTIME = Path(os.environ.get("R39_RUNTIME", os.environ.get("R34_RUNTIME", str(PKG_ROOT / "runtime"))))

QB_DEFAULT = r"C:\models\calib_text\exl3\qbench_prompts.json"
SEQ = 2048
QB_DOMAIN = {
    0: "math", 7: "math", 9: "math",
    1: "code", 8: "code",
    2: "science", 4: "science",
    3: "tool", 6: "tool", 10: "tool", 13: "tool",
    5: "strategy", 11: "strategy",
    12: "creative",
}
GRAPH_T_MAX_DEFAULT = 2048

SPEED_PROMPTS = [
    ("code", "Write a compact Python class implementing an LRU cache using collections.OrderedDict with get and put in O(1) time."),
    ("capital", "List the capitals of France, Japan, Brazil, Canada, and Australia and state one river flowing near each."),
    ("math", "Derive the closed-form sum of k^2 for k = 1..n by induction and verify n = 10."),
    ("systems", "Explain why PCIe 5.0 pinned host memory DMA requires page-locked physical frames under Windows WDDM."),
]


def load_rows(qbench_path: str, data_path: str, smoke: bool = False) -> List[dict]:
    with open(qbench_path, encoding="utf-8") as fh:
        qb = json.load(fh)["rows"]
    rows = []
    for i, r in enumerate(qb):
        ids = (r["input_ids"] + r["response_ids"])[:SEQ]
        rows.append({
            "id": f"qb{i}",
            "group": "qb",
            "split": "qbench",
            "domain": QB_DOMAIN[r["conversation"]],
            "source": "qbench",
            "conv": r["conversation"],
            "turn": r["turn"],
            "tools": bool(r["tools"]),
            "n_in": len(r["input_ids"]),
            "ids": ids,
            "resp": [[len(r["input_ids"]), len(ids)]],
        })
    if data_path and os.path.exists(data_path):
        with open(data_path, encoding="utf-8") as fh:
            data = json.load(fh)["rows"]
        for r in data:
            if r["split"] == "dev" and r["source"] == "cal_trace":
                g = "devt"
            elif r["split"] in ("dev", "nontarget", "ppl"):
                g = "devx"
            else:
                continue
            rows.append({k: v for k, v in r.items() if k != "weight"} | {"group": g})
    if smoke:
        per: Counter = Counter()
        keep = []
        for r in rows:
            key = r["group"] + ("ppl" if r["split"] == "ppl" else "")
            if per[key] < 2:
                keep.append(r)
                per[key] += 1
        rows = keep
    return rows


def row_coverage(rows: List[dict]) -> dict:
    counts = dict(Counter(r["group"] for r in rows))
    unique = len({r["id"] for r in rows}) == len(rows)
    positions = {
        group: sum(max(0, len(r["ids"]) - 1 - (10 if group == "qb" else 2)) for r in rows if r["group"] == group)
        for group in counts
    }
    complete = counts == {"qb": 29, "devt": 40, "devx": 42} and unique and all(v > 0 for v in positions.values())
    return {"groups": counts, "unique_ids": unique, "positions": positions, "complete": complete}


def check_capacity(rows: List[dict], t_max: int) -> None:
    bad = [(r["id"], len(r["ids"])) for r in rows if len(r["ids"]) > t_max]
    if bad:
        raise SystemExit(
            f"{len(bad)} quality row(s) exceed graph capacity t_max={t_max}: "
            + ", ".join(f"{i} ({n} ids)" for i, n in bad)
        )


def _norm(path: str | Path) -> str:
    return os.path.normcase(os.path.normpath(str(Path(path).resolve())))


def _inside(path: str, root: str) -> bool:
    if os.path.splitdrive(path)[0].lower() != os.path.splitdrive(root)[0].lower():
        return False
    try:
        return os.path.commonpath([root, path]) == root
    except ValueError:
        return False


def check_runtime_modules(runtime_dir: Path, where: str) -> None:
    root = _norm(runtime_dir)
    bad = sorted(
        (name, str(getattr(m, "__file__", None)))
        for name, m in list(sys.modules.items())
        if name.startswith("glm53_") and getattr(m, "__file__", None) and not _inside(_norm(getattr(m, "__file__")), root)
    )
    if bad:
        raise SystemExit(
            f"Runtime module provenance check failed {where}: foreign glm53_* modules outside {runtime_dir}: "
            + "; ".join(f"{n} -> {p}" for n, p in bad)
        )


def run_speed_gate(runner, tok, rounds: int = 3, max_new: int = 128) -> dict:
    """Runs 3 warmup prompts + `rounds` alternating timed decode passes across `SPEED_PROMPTS`."""
    encoded = [(name, tok.encode_chat(p, thinking=False)) for name, p in SPEED_PROMPTS]
    runner.warmup([ids for _, ids in encoded[:2]], eos_ids=tok.eos_ids, max_new_tokens=4)
    per_prompt: Dict[str, List[float]] = {name: [] for name, _ in encoded}
    records = []
    for r_idx in range(rounds):
        for name, ids in encoded:
            res = runner.generate(ids, max_new_tokens=max_new, eos_ids=tok.eos_ids)
            tps = float(res.get("decode_tok_s", res.get("tok_per_s", 0.0)))
            per_prompt[name].append(tps)
            records.append({"round": r_idx + 1, "prompt": name, "tok_s": tps, "hit_rate": res.get("pool_hit_rate")})
    means = {k: round(sum(v) / max(len(v), 1), 3) for k, v in per_prompt.items()}
    overall = round(sum(means.values()) / max(len(means), 1), 3)
    return {
        "rounds": rounds,
        "max_new_tokens": max_new,
        "per_prompt_mean_tok_s": means,
        "overall_mean_tok_s": overall,
        "pass_18_tok_s_gate": bool(overall >= 18.0 and means.get("code", 0.0) >= 18.0 and means.get("capital", 0.0) >= 18.0),
        "runs": records,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Round 3.9 Full-Vocabulary Quality & Speed Gate Evaluator")
    ap.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME)
    ap.add_argument("--runner", default="auto", choices=["auto", "qtip", "tcq"])
    ap.add_argument("--backbone", default=r"C:\models\GLM53-backbone-r34h")
    ap.add_argument("--fold", default=r"C:\models\GLM53-APEX-FOLD-R39-HYB-FT")
    ap.add_argument("--merged-routers", default="", help="Optional merged_routers_bf16.pt (auto-detected in --fold)")
    ap.add_argument("--prune-spec", default="", help="Optional extra prune spec JSON")
    ap.add_argument("--exact-backbone", default=r"C:\models\GLM53-backbone")
    ap.add_argument("--evx", default=r"C:\models\GLM53-APEX-FOLD-TCQ33\_r33_ckpt\evx.pt")
    ap.add_argument("--baseline-dir", type=Path, default=None, help="Optional R3.7 hidden-state baseline dir for dev KL")
    ap.add_argument("--reference", action="store_true", help="Write baseline hidden states into --baseline-dir")
    ap.add_argument("--data", default=str(HERE / "out_r32" / "r34_prune_data.json"))
    ap.add_argument("--qbench", default=QB_DEFAULT)
    ap.add_argument("--qbench-only", action="store_true", help="Evaluate only the 29 exact qbench rows (fast ~4 min gate)")
    ap.add_argument("--chunk", type=int, default=1, help="1 = real N=1 GraphDecoder (required for official gate); >1 = prefill diagnostic")
    ap.add_argument("--pool-gib", type=float, default=18.0)
    ap.add_argument("--register-gib", type=float, default=42.0)
    ap.add_argument("--kl-budget", type=float, default=0.230, help="Target qbench KL gate threshold in nats")
    ap.add_argument("--speed-gate", action="store_true", help="Also run 3-round alternating decode speed proof (>=18 tok/s)")
    ap.add_argument("--smoke", action="store_true")
    # [R3.9 additions, 2026-11-16]
    ap.add_argument("--speed-only", action="store_true", help="Skip the quality rows; load the runner and run only the speed gate")
    ap.add_argument("--prefix-cache", action="store_true",
                    help="QTIP runner: use the exact prefix caches (shipped default was ON for QTIP and OFF for the TCQ runner; "
                         "now OFF for both unless this flag is given, so the two runners are measured alike)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if os.environ.get("R37_GPU_WINDOW", "1") != "1" and os.environ.get("R36_GPU_HANDOVER", "1") != "1":
        raise SystemExit("GPU handover blocked by env guard (set R37_GPU_WINDOW=1)")

    runtime_dir = args.runtime_dir.resolve()
    sys.path.insert(0, str(runtime_dir))

    rows = load_rows(args.qbench, "" if args.qbench_only else args.data, smoke=args.smoke)
    coverage = row_coverage(rows)
    if not args.smoke and not args.qbench_only and not coverage["complete"]:
        raise SystemExit(f"Quality row coverage incomplete: {coverage}")
    if args.chunk == 1:
        check_capacity(rows, GRAPH_T_MAX_DEFAULT)

    import torch
    import torch.nn.functional as F
    import glm53_fold_backbone as B  # type: ignore
    import glm53_fold_generate as G  # type: ignore
    import glm53_fold_qtip_runtime as QR  # type: ignore
    from glm53_fold_graphs import GraphDecoder  # type: ignore

    check_runtime_modules(runtime_dir, "after import")

    mode = args.runner
    if mode == "auto":
        mode = "qtip" if os.path.exists(os.path.join(args.fold, "layer_03_qtip_hyb.pt")) else "tcq"

    t0 = time.perf_counter()
    if mode == "qtip":
        runner = QR.QtipMoEModelRunner(
            backbone_dir=args.backbone,
            fold_dir=args.fold,
            merged_routers_path=args.merged_routers,
            device="cuda",
            backbone_mode="fp8",
            expert_pool_gib=args.pool_gib,
            register_gib=args.register_gib,
            prefetch=True,
            capture_graphs=(args.chunk == 1),
            t_max=GRAPH_T_MAX_DEFAULT,
            load_mtp=False,
            vocab_keep=None,
            prune_spec=args.prune_spec or None,
            pkg_dir=str(runtime_dir),
            prefix_cache=bool(args.prefix_cache),
        )
        dec = runner.graph_decoder
    else:
        runner = G.ApexFoldMoEModelRunner(
            backbone_dir=args.backbone,
            fold_dir=args.fold,
            device="cuda",
            backbone_mode="fp8",
            cold_mode="mmap",
            expert_pool_gib=args.pool_gib,
            prefetch=True,
            sync_timing=False,
            fp8_diet=True,
            load_mtp=False,
            vocab_keep=None,
            prune_spec=args.prune_spec or None,
        )
        auto_r = args.merged_routers or os.path.join(args.fold, "merged_routers_bf16.pt")
        if os.path.exists(auto_r):
            QR.apply_merged_routers(runner.bb, auto_r, runner.prune_masks)
        dec = None
        if args.chunk == 1:
            dec = GraphDecoder(runner, t_max=GRAPH_T_MAX_DEFAULT)
            runner.graph_decoder = dec
            with torch.inference_mode():
                dec.capture()

    check_runtime_modules(runtime_dir, "after graph capture")
    last_key = (dec._seq[-1][3] or dec._seq[-1][1]) if dec is not None else None

    free_b, tot_b = torch.cuda.mem_get_info()
    print(f"[eval-r39] runner loaded in {time.perf_counter() - t0:.0f}s; VRAM free {free_b / 2**30:.2f} of {tot_b / 2**30:.2f} GiB", flush=True)
    if args.speed_only:
        rows = []
    teacher = B.RealGLM53Backbone(B.RealGLM53Checkpoint(args.exact_backbone), runner.cfg, device="cuda", backbone_mode="fp8")
    if not args.speed_only:
        teacher.load_global()

    # Support either `_r33_ckpt/evx.pt` or `_r39_ckpt/evx.pt`
    evx_path = args.evx
    if not os.path.exists(evx_path):
        alt_evx = os.path.join(args.fold, "_r39_ckpt", "evx.pt")
        if os.path.exists(alt_evx):
            evx_path = alt_evx
    evx = None if args.speed_only else torch.load(evx_path, map_location="cpu", weights_only=True, mmap=True)["h"]

    row_hash = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    report: dict = {
        "runner_mode": mode,
        "fold_dir": str(args.fold),
        "backbone_dir": str(args.backbone),
        "rows_sha256": row_hash,
        "chunk": args.chunk,
        "execution": "graph_n1" if dec is not None else "prefill_diagnostic",
        "coverage": coverage,
        "kl_budget": args.kl_budget,
        "rows": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.baseline_dir is not None:
        args.baseline_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for row in rows:
            ids = row["ids"]
            cache = B.RealCache()
            hidden = []
            sums = {"kl_exact": 0.0, "agree_exact": 0, "n_exact": 0, "kl_baseline": 0.0, "n_baseline": 0}
            base_h = None
            if args.baseline_dir is not None and not args.reference:
                bp = args.baseline_dir / f"{row['id']}.pt"
                if bp.exists():
                    base_h = torch.load(bp, map_location="cpu", weights_only=True, mmap=True)

            for start in range(0, len(ids) - 1, args.chunk):
                stop = min(len(ids) - 1, start + args.chunk)
                if dec is not None and start >= 2:
                    if start == 2:
                        assert dec.load_cache(cache)
                    dec.step(ids[start])
                    logits = dec.logits
                    h = dec.out[last_key].mean(dim=2) if args.reference else None
                else:
                    logits, h = runner.bb.forward(
                        torch.tensor([ids[start:stop]], device="cuda"),
                        cache=cache,
                        moe_fn=runner.moe_dispatch,
                        last_only=False,
                        max_layers=runner.max_layers,
                    )
                logq = F.log_softmax(logits[0].float(), -1)
                if args.reference and h is not None:
                    hidden.append(h.cpu())

                lo = max(10 if row["group"] == "qb" else 2, start) - start
                if lo >= stop - start:
                    continue

                if base_h is not None:
                    bh = base_h[:, start:stop].to("cuda")
                    base_logits = runner.bb.lin(runner.bb.rmsnorm(bh, runner.bb.final_norm), runner.bb.head)[0]
                    lp_b = F.log_softmax(base_logits.float(), -1)
                    sums["kl_baseline"] += float((lp_b.exp() * (lp_b - logq))[lo:].sum(dtype=torch.float64))
                    sums["n_baseline"] += stop - start - lo

                ref = None
                if row["group"] == "qb":
                    th = evx[int(row["id"][2:])][:, start:stop].to("cuda").float()
                    ref = teacher.final_logits(th, last_only=False)[0][0]
                elif row["split"] == "ppl" and "exact_logits" in row and os.path.exists(row["exact_logits"]):
                    ref = torch.load(row["exact_logits"], map_location="cpu", weights_only=True, mmap=True)[start:stop].to("cuda")

                if ref is not None:
                    lp = F.log_softmax(ref.float(), -1)
                    sums["kl_exact"] += float((lp.exp() * (lp - logq))[lo:].sum(dtype=torch.float64))
                    sums["agree_exact"] += int((lp.argmax(-1) == logq.argmax(-1))[lo:].sum())
                    sums["n_exact"] += stop - start - lo

            if args.reference and args.baseline_dir is not None and hidden:
                torch.save(torch.cat(hidden, dim=1), args.baseline_dir / f"{row['id']}.pt")

            report["rows"].append({"id": row["id"], "group": row["group"], "domain": row["domain"], **sums})
            args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"[eval-r39] row {row['id']:6s} | n_exact={sums['n_exact']:4d} | kl={sums['kl_exact']/max(sums['n_exact'],1):.5f}", flush=True)

    qb_rows = [r for r in report["rows"] if r["group"] == "qb"]
    n_qb = sum(r["n_exact"] for r in qb_rows)
    qb_kl = sum(r["kl_exact"] for r in qb_rows) / max(n_qb, 1)
    qb_top1 = sum(r["agree_exact"] for r in qb_rows) / max(n_qb, 1)
    report["qbench"] = {
        "kl": round(qb_kl, 6),
        "top1": round(qb_top1, 5),
        "positions": n_qb,
        "rows": len(qb_rows),
        "pass_kl_budget": bool(dec is not None and not args.smoke and qb_kl <= args.kl_budget),
    }

    report["domains"] = {}
    for domain in sorted({r["domain"] for r in report["rows"]}):
        rr = [r for r in report["rows"] if r["domain"] == domain and r["group"] != "qb"]
        nd = sum(r["n_baseline"] for r in rr)
        report["domains"][domain] = {"kl_baseline": round(sum(r["kl_baseline"] for r in rr) / nd, 6) if nd else None, "positions": nd}

    if args.speed_gate or args.speed_only:
        # [R3.9 fix] the teacher's 1.2 GiB head stayed on the GPU during the timed decode
        del teacher, evx
        torch.cuda.empty_cache()
        free_b, tot_b = torch.cuda.mem_get_info()
        print(f"[eval-r39] speed gate: VRAM free {free_b / 2**30:.2f} of {tot_b / 2**30:.2f} GiB", flush=True)
        tok = G.RealTokenizer(args.backbone)
        if mode == "tcq":
            # [R3.9 fix, 2026-11-17] the TCQ runner's graphs are captured above under torch.inference_mode(), so
            # their state tensors are inference tensors; runner.generate (torch.no_grad) then fails on the first
            # in-place state update ("Inplace update to inference tensor outside InferenceMode"). Same mode here.
            with torch.inference_mode():
                report["speed_gate"] = run_speed_gate(runner, tok)
        else:
            report["speed_gate"] = run_speed_gate(runner, tok)
        report["speed_gate"]["vram_free_gib_before"] = round(free_b / 2**30, 2)
        report["speed_gate"]["env"] = {k: v for k, v in os.environ.items() if k[:4] in ("R32_", "R33_", "R34_", "R35_", "R36_", "R37_", "R38_", "R39_")}

    report["elapsed_min"] = round((time.perf_counter() - t0) / 60.0, 2)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[EVAL SUMMARY] {json.dumps(report['qbench'])}", flush=True)
    if "speed_gate" in report:
        print(f"[SPEED SUMMARY] {json.dumps({k: v for k, v in report['speed_gate'].items() if k != 'runs'})}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
