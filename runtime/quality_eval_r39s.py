#!/usr/bin/env python3
"""
quality_eval_r39s.py  [Round 3.9 Full-Vocabulary Quality & Alternating Speed Gate Evaluator]
============================================================================================
[R3.9S] Port of claude_r39_qtip/build_r39/quality_eval_r39.py (state of 2026-11-17, with its fixes) to the flat
Round 3.8 runtime folder claude_r39_speed. Same rows, same metrics, same speed gate, same command line; every changed
region carries the tag [R3.9S]:
  * the runtime is this file's own folder (was ../runtime); --data defaults to the package's build_run/out_r32 copy;
  * --register-gib defaults to R33_REGISTER_GIB when that is set (the production variable), else the shipped 42.0;
  * the QTIP runner's pool follows R39_POOL / R32_POOL (see glm53_fold_qtip_runtime); the report records which pool
    served the run and with which flags;
  * with R37_DUAL_GRAPHS=1 the quality rows are decoded by the window decoder (the gate's rows fit 2,048 tokens).
The TCQ speed gate still runs inside torch.inference_mode() (fix of 2026-11-17, see the end of main()).

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
# [R3.9S] flat runtime folder: the glm53_* modules lie next to this file (shipped: PKG_ROOT / "runtime")
DEFAULT_RUNTIME = Path(os.environ.get("R39_RUNTIME", os.environ.get("R34_RUNTIME", str(HERE))))
# [R3.9S] the 111-row data file is not in this folder; the package keeps it in build_run\out_r32
_DATA_CANDIDATES = [HERE / "out_r32" / "r34_prune_data.json", PKG_ROOT / "build_run" / "out_r32" / "r34_prune_data.json"]
DEFAULT_DATA = next((p for p in _DATA_CANDIDATES if p.exists()), _DATA_CANDIDATES[0])

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


def _r39m_summarize(runs: List[dict]) -> dict:
    """[R3.9M] R39_MTP_STATS=1: totals over the speed gate's timed runs. Decode phase only (the prompt pass is not in
    it): tokens, seconds, expert lookups / demand misses / prefetches per generated token, GiB copied host -> GPU per
    token, host ms per token inside the pool; for MTP runs the mean ms of an accepted / rejected verify step, an
    N = 1 step and a draft, the same pool numbers per step kind, and tokens per step."""
    tot = {"tokens": 0, "decode_s": 0.0, "lookups": 0, "misses": 0, "pf_issued": 0, "pf_useful": 0,
           "pf_skipped_reg": 0, "pf_skipped_cold": 0, "gib_h2d": 0.0, "t_ensure": 0.0, "t_prefetch": 0.0}
    kinds: Dict[str, dict] = {}
    steps = pair_steps = accepted = verified = 0
    for r in runs:
        pd = r.get("pool_decode") or {}
        tot["tokens"] += int(pd.get("new_tokens") or 0)
        tot["decode_s"] += float(pd.get("decode_s") or 0.0)
        tot["lookups"] += int(pd.get("hits", 0)) + int(pd.get("misses", 0))
        tot["misses"] += int(pd.get("misses", 0))
        tot["pf_issued"] += int(pd.get("prefetch_issued", 0))
        tot["pf_useful"] += int(pd.get("prefetch_useful", 0))
        tot["pf_skipped_reg"] += int(pd.get("prefetch_skipped_reg", 0))
        tot["pf_skipped_cold"] += int(pd.get("prefetch_skipped_cold", 0))
        tot["gib_h2d"] += float(pd.get("gib_h2d") or 0.0)
        tot["t_ensure"] += float(pd.get("t_ensure") or 0.0)
        tot["t_prefetch"] += float(pd.get("t_prefetch") or 0.0)
        m = r.get("mtp") or {}
        steps += int(m.get("steps") or 0)
        pair_steps += int(m.get("pair_steps") or 0)
        accepted += int(m.get("accepted") or 0)
        verified += int(m.get("verified") or 0)
        for kind, b in (m.get("r39m") or {}).items():
            k = kinds.setdefault(kind, {"n": 0, "ms": 0.0, "misses": 0.0, "lookups": 0.0, "pf_issued": 0.0,
                                        "ensure_ms": 0.0, "prefetch_ms": 0.0, "gib_h2d": 0.0})
            n = int(b.get("n") or 0)
            k["n"] += n
            k["ms"] += n * float(b.get("ms_mean") or 0.0)
            for f in ("misses", "lookups", "pf_issued", "ensure_ms", "prefetch_ms", "gib_h2d"):
                k[f] += n * float(b.get(f) or 0.0)
    nt = max(1, tot["tokens"])
    out = {"tokens": tot["tokens"], "decode_s": round(tot["decode_s"], 3),
           "ms_per_token": round(1000.0 * tot["decode_s"] / nt, 3),
           "lookups_per_token": round(tot["lookups"] / nt, 2), "misses_per_token": round(tot["misses"] / nt, 2),
           "pf_issued_per_token": round(tot["pf_issued"] / nt, 2), "pf_useful_per_token": round(tot["pf_useful"] / nt, 2),
           "pf_skipped_reg_per_token": round(tot["pf_skipped_reg"] / nt, 2),
           "pf_skipped_cold_per_token": round(tot["pf_skipped_cold"] / nt, 2),
           "gib_h2d_per_token": round(tot["gib_h2d"] / nt, 4),
           "pool_host_ms_per_token": round(1000.0 * (tot["t_ensure"] + tot["t_prefetch"]) / nt, 3)}
    if steps:
        out["mtp"] = {"steps": steps, "pair_steps": pair_steps, "accepted": accepted, "verified": verified,
                      "accept_rate": round(accepted / verified, 4) if verified else None,
                      "tokens_per_step": round(tot["tokens"] / steps, 4),
                      "kinds": {kind: {"n": k["n"], **{("ms_mean" if f == "ms" else f): round(k[f] / max(1, k["n"]), 3)
                                                        for f in ("ms", "misses", "lookups", "pf_issued", "ensure_ms",
                                                                  "prefetch_ms", "gib_h2d")}}
                                for kind, k in kinds.items()}}
    return out


def _r39m_strict_compare(ref_sg: dict, cur_sg: dict) -> dict:
    """[R3.9M] the stricter lossless verdict of a speed-gate report against a reference report: per prompt, the ids
    of EVERY round against the reference's first round, and - when both reports carry fingerprints (R39_MTP_FP=1) -
    the bits of the logits behind every generated token. logits_all is None when one side has no fingerprints."""
    other = ref_sg.get("generated_ids") or {}
    ref_fp = ref_sg.get("logit_fp") or {}
    cur_fp = cur_sg.get("logit_fp") or {}
    strict = {}
    for name, runs in (cur_sg.get("generated_ids") or {}).items():
        ref0 = (other.get(name) or [[]])[0]
        ids_all = bool(runs) and bool(ref0) and all(r == ref0 for r in runs)
        rf0 = (ref_fp.get(name) or [None])[0]
        fps = cur_fp.get(name) or []
        lg_all, first = None, None
        if rf0 is not None and fps and all(f is not None for f in fps):
            lg_all = all(f == rf0 for f in fps)
            if not lg_all:
                first = min(next((i for i, (a, b) in enumerate(zip(rf0, f)) if a != b), min(len(rf0), len(f)))
                            for f in fps if f != rf0)
        strict[name] = {"ids_identical_all_rounds": ids_all, "logits_identical_all_rounds": lg_all,
                        "first_logit_difference_at": first, "rounds": len(runs)}
    lgs = [v["logits_identical_all_rounds"] for v in strict.values()]
    return {"ids_all": bool(strict) and all(v["ids_identical_all_rounds"] for v in strict.values()),
            "logits_all": (None if (not lgs or any(x is None for x in lgs)) else all(lgs)),
            "prompts": strict}


def run_speed_gate(runner, tok, rounds: int = 3, max_new: int = 128, use_mtp: bool = False) -> dict:
    """Runs 3 warmup prompts + `rounds` alternating timed decode passes across `SPEED_PROMPTS`.
    [R3.9S, 2026-11-17] use_mtp: speculative decoding with the model's MTP layer (needs load_mtp and, for the graph
    path, env R35_MTP_GRAPHS=1). The generated ids of every run are recorded so a drafting run can be compared token
    for token with a plain run."""
    encoded = [(name, tok.encode_chat(p, thinking=False)) for name, p in SPEED_PROMPTS]
    runner.warmup([ids for _, ids in encoded[:2]], eos_ids=tok.eos_ids, max_new_tokens=4, use_mtp=use_mtp)
    per_prompt: Dict[str, List[float]] = {name: [] for name, _ in encoded}
    records = []
    gen_ids: Dict[str, List[List[int]]] = {name: [] for name, _ in encoded}
    mtp_recs = []
    # [R3.9M] bookkeeping only (run_mtp_tests.ps1 sets both; unset = the report as before): R39_MTP_STATS=1 adds the
    # decode-phase pool counters of every run and the per-step statistics of decode_mtp; R39_MTP_FP=1 records the
    # logit-bit fingerprints of every generated token, so --compare-ids can also say "bit-identical logits".
    r39m_stats = os.environ.get("R39_MTP_STATS", "0").strip() == "1"
    r39m_fp = os.environ.get("R39_MTP_FP", "0").strip() == "1"
    fp_ids: Dict[str, List] = {name: [] for name, _ in encoded}
    r39m_runs = []
    # [R3.9T] R39T_PROF=1 (bookkeeping): the decoder's step-time split, per prompt, without the warm-up
    # [R3.9T] R39T_AB="<fast_demand>,<preevict>" (bookkeeping / measurement only): a PAIRED comparison inside this one
    # process. The timed rounds are cut to 1 (ids and fingerprints for --compare-ids), then every prompt is generated
    # R39T_AB_ROUNDS (4) times with the pool's R39T_FAST_DEMAND / R39T_PREEVICT off (arm A) and at the given values (arm
    # B), A and B alternating and the order swapped every round. Reason: two separate runs differ by 1 % on a quiet
    # machine and by up to 10 % when another job loads it; the paired arms see the same load.
    r39t_ab = os.environ.get("R39T_AB", "").strip()
    if r39t_ab and callable(getattr(runner.pool, "r39t_set", None)):
        rounds = 1
    else:
        r39t_ab = ""
    r39t_gd = getattr(runner, "graph_decoder", None) if os.environ.get("R39T_PROF", "0").strip() == "1" else None
    if r39t_gd is not None and callable(getattr(r39t_gd, "r39t_reset", None)) and r39t_gd._r39t_enabled():
        r39t_gd.r39t_reset()
    else:
        r39t_gd = None
    for r_idx in range(rounds):
        for name, ids in encoded:
            if r39t_gd is not None:
                r39t_gd.r39t_tag = name
            if callable(getattr(runner.pool, "r39t_mark", None)):    # [R3.9T] call-log mark (no-op without R39T_TRACE)
                runner.pool.r39t_mark(f"request {r_idx + 1} {name}")
            res = runner.generate(ids, max_new_tokens=max_new, eos_ids=tok.eos_ids, use_mtp=use_mtp)
            tps = float(res.get("decode_tok_s", res.get("tok_per_s", 0.0)))
            per_prompt[name].append(tps)
            gen_ids[name].append([int(t) for t in (res.get("generated_ids") or [])])
            if r39m_fp:                                  # [R3.9M]
                fp_ids[name].append(res.get("logit_fp"))
            if r39m_stats:                               # [R3.9M]
                mg_ = res.get("mtp_graph") or {}
                r39m_runs.append({"round": r_idx + 1, "prompt": name, "tok_s": tps,
                                  "decode_s": res.get("decode_s"), "pool_decode": res.get("pool_decode"),
                                  "mtp": {k: mg_.get(k) for k in (
                                      "tokens", "steps", "pair_steps", "n1_steps", "n1_reasons", "drafts", "verified",
                                      "accepted", "accept_rate", "tokens_per_step", "draft_ms_mean", "low_conf",
                                      "pauses", "pauses_low_conf", "ngram_verified", "ngram_accepted", "calib", "cfg",
                                      "r39m") if k in mg_}})
            if use_mtp:
                mg = res.get("mtp_graph") or {}
                mtp_recs.append({"round": r_idx + 1, "prompt": name, "mtp_accept_rate": res.get("mtp_accept_rate"),
                                 "graph": {k: mg.get(k) for k in ("pair_steps", "accepted", "verified", "accept_rate",
                                                                  "mtp_accept_rate", "n1_reasons", "pauses") if k in mg}})
            # [R3.9S] bookkeeping only (tok_s and the gate are unchanged): the Round 3.8 generate() reports the pool's
            # per-request statistics under "pool"; "pool_hit_rate" (cumulative) and "used_graphs" exist for the QTIP
            # runner only. new_tokens shows a run that stopped early (EOS) and so timed fewer than max_new steps.
            pool_d = res.get("pool") or {}
            records.append({"round": r_idx + 1, "prompt": name, "tok_s": tps, "hit_rate": res.get("pool_hit_rate"),
                            "hit_rate_request": pool_d.get("hit_rate"), "misses_request": pool_d.get("misses"),
                            "new_tokens": res.get("new_tokens"), "used_graphs": res.get("used_graphs")})
    means = {k: round(sum(v) / max(len(v), 1), 3) for k, v in per_prompt.items()}
    overall = round(sum(means.values()) / max(len(means), 1), 3)
    r39m_extra = {}
    if r39m_fp:                                          # [R3.9M] keys present only with the flags set
        r39m_extra["logit_fp"] = fp_ids
        r39m_extra["logit_fp_stable_across_rounds"] = {
            k: (None if (not v or any(x is None for x in v)) else all(v[0] == x for x in v[1:]))
            for k, v in fp_ids.items()}
    if r39m_stats:
        r39m_extra["r39m_runs"] = r39m_runs
        r39m_extra["r39m_summary"] = _r39m_summarize(r39m_runs)
    if r39t_gd is not None:                              # [R3.9T] after every timed run: split + the no-miss floors
        r39m_extra["r39t_prof"] = r39t_gd.r39t_summary()
        r39m_extra["r39t_floor"] = r39t_gd.r39t_floor()
        import torch as _t39
        import gc as _gc39
        seen39, big39 = set(), []                        # census of the CUDA tensors alive now, by storage
        for o in _gc39.get_objects():
            try:
                if isinstance(o, _t39.Tensor) and o.is_cuda:
                    st_ = o.untyped_storage()
                    if st_.data_ptr() not in seen39:
                        seen39.add(st_.data_ptr())
                        big39.append((st_.nbytes(), str(o.dtype).replace("torch.", ""), list(o.shape)))
            except Exception:
                pass
        big39.sort(reverse=True)
        r39m_extra["r39t_census"] = {"storages": len(big39), "gib": round(sum(b[0] for b in big39) / 2**30, 3),
                                     "top": [[round(b[0] / 2**30, 4), b[1], b[2]] for b in big39[:60]]}
        r39m_extra["r39t_mem"] = {"torch_allocated_gib": round(_t39.cuda.memory_allocated() / 2**30, 3),
                                  "torch_reserved_gib": round(_t39.cuda.memory_reserved() / 2**30, 3),
                                  "free_gib": round(_t39.cuda.mem_get_info()[0] / 2**30, 3),
                                  "total_gib": round(_t39.cuda.mem_get_info()[1] / 2**30, 3)}
    if callable(getattr(runner.pool, "r39t_trace_save", None)):      # [R3.9T] no-op without R39T_TRACE
        runner.pool.r39t_trace_save()
    if r39t_ab:                                          # [R3.9T] the paired A/B (see above)
        fd_b, pe_b = (int(x) for x in (r39t_ab.split(",") + ["0"])[:2])
        fd_a = int(os.environ.get("R39T_FAST_DEMAND", "0").strip() or 0)
        pe_a = int(os.environ.get("R39T_PREEVICT", "0").strip() or 0)
        n_ab = int(os.environ.get("R39T_AB_ROUNDS", "4"))
        arms = {"A": {name: [] for name, _ in encoded}, "B": {name: [] for name, _ in encoded}}
        ids_equal = True
        for r_idx in range(n_ab):
            for name, ids in encoded:
                for arm in (("A", "B") if r_idx % 2 == 0 else ("B", "A")):
                    runner.pool.r39t_set(*((fd_a, pe_a) if arm == "A" else (fd_b, pe_b)))
                    res = runner.generate(ids, max_new_tokens=max_new, eos_ids=tok.eos_ids, use_mtp=use_mtp)
                    arms[arm][name].append(float(res.get("decode_tok_s", 0.0)))
                    ids_equal &= [int(t) for t in (res.get("generated_ids") or [])] == gen_ids[name][0]
        runner.pool.r39t_set(fd_a, pe_a)
        mean = lambda v: sum(v) / max(1, len(v))        # noqa: E731
        ab = {"A": {"fast_demand": fd_a, "preevict": pe_a}, "B": {"fast_demand": fd_b, "preevict": pe_b},
              "rounds": n_ab, "ids_equal_to_round_1": bool(ids_equal), "per_prompt": {}}
        ratios = []
        for name, _ in encoded:
            a_, b_ = arms["A"][name], arms["B"][name]
            pr = [b / a for a, b in zip(a_, b_)]         # same round, same prompt, back to back
            ratios += pr
            ab["per_prompt"][name] = {"A_tok_s": [round(x, 3) for x in a_], "B_tok_s": [round(x, 3) for x in b_],
                                      "A_mean": round(mean(a_), 3), "B_mean": round(mean(b_), 3),
                                      "B_over_A": round(mean(pr), 4)}
        ab["A_mean_tok_s"] = round(mean([v["A_mean"] for v in ab["per_prompt"].values()]), 3)
        ab["B_mean_tok_s"] = round(mean([v["B_mean"] for v in ab["per_prompt"].values()]), 3)
        ab["B_over_A_mean_of_pairs"] = round(mean(ratios), 4)
        srt = sorted(ratios)
        ab["B_over_A_median_of_pairs"] = round(srt[len(srt) // 2] if len(srt) % 2 else (srt[len(srt) // 2 - 1] + srt[len(srt) // 2]) / 2, 4)
        ab["pairs_B_faster"] = sum(1 for x in ratios if x > 1.0)
        ab["pairs"] = len(ratios)
        r39m_extra["r39t_ab"] = ab
        print(f"[R3.9T AB] {json.dumps({k: v for k, v in ab.items() if k != 'per_prompt'})}", flush=True)
    return {
        "rounds": rounds,
        "max_new_tokens": max_new,
        "per_prompt_mean_tok_s": means,
        "overall_mean_tok_s": overall,
        "pass_18_tok_s_gate": bool(overall >= 18.0 and means.get("code", 0.0) >= 18.0 and means.get("capital", 0.0) >= 18.0),
        "runs": records,
        "use_mtp": bool(use_mtp),
        "mtp": mtp_recs,
        "generated_ids": gen_ids,
        "ids_stable_across_rounds": {k: all(v[0] == x for x in v[1:]) for k, v in gen_ids.items() if v},
        **r39m_extra,                                    # [R3.9M] empty unless R39_MTP_STATS / R39_MTP_FP are set
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
    ap.add_argument("--data", default=str(DEFAULT_DATA))   # [R3.9S] see DEFAULT_DATA
    ap.add_argument("--qbench", default=QB_DEFAULT)
    ap.add_argument("--qbench-only", action="store_true", help="Evaluate only the 29 exact qbench rows (fast ~4 min gate)")
    ap.add_argument("--chunk", type=int, default=1, help="1 = real N=1 GraphDecoder (required for official gate); >1 = prefill diagnostic")
    ap.add_argument("--pool-gib", type=float, default=18.0)
    # [R3.9S] QTIP runner only (the TCQ runner reads R33_REGISTER_GIB itself, as before). Default None =
    # R33_REGISTER_GIB when set, else the shipped 42.0, so one environment drives both runners.
    ap.add_argument("--register-gib", type=float, default=None)
    ap.add_argument("--kl-budget", type=float, default=0.230, help="Target qbench KL gate threshold in nats")
    ap.add_argument("--speed-gate", action="store_true", help="Also run 3-round alternating decode speed proof (>=18 tok/s)")
    ap.add_argument("--smoke", action="store_true")
    # [R3.9 additions, 2026-11-16]
    ap.add_argument("--speed-only", action="store_true", help="Skip the quality rows; load the runner and run only the speed gate")
    ap.add_argument("--prefix-cache", action="store_true",
                    help="QTIP runner: use the exact prefix caches (shipped default was ON for QTIP and OFF for the TCQ runner; "
                         "now OFF for both unless this flag is given, so the two runners are measured alike)")
    ap.add_argument("--mtp", action="store_true",
                    help="[R3.9S] speed gate with speculative decoding: load the MTP layer 45 and draft with it "
                         "(graph path needs env R35_MTP_GRAPHS=1). Speed gate only; the quality rows are unaffected")
    ap.add_argument("--compare-ids", type=Path, default=None,
                    help="[R3.9S] a previous report of this script: compare the speed gate's generated ids with it")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if os.environ.get("R37_GPU_WINDOW", "1") != "1" and os.environ.get("R36_GPU_HANDOVER", "1") != "1":
        raise SystemExit("GPU handover blocked by env guard (set R37_GPU_WINDOW=1)")

    runtime_dir = args.runtime_dir.resolve()
    sys.path.insert(0, str(runtime_dir))

    # [R3.9S, 2026-11-17] --speed-only needs no quality rows, no test-prompt file and no exact backbone, so the speed
    # gate can be run from the public release alone (the quality rows are not part of it).
    rows = [] if args.speed_only else load_rows(args.qbench, "" if args.qbench_only else args.data, smoke=args.smoke)
    coverage = row_coverage(rows)
    if not args.smoke and not args.qbench_only and not args.speed_only and not coverage["complete"]:
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
            load_mtp=bool(args.mtp),
            vocab_keep=None,
            prune_spec=args.prune_spec or None,
            pkg_dir=str(runtime_dir),
            prefix_cache=bool(args.prefix_cache),
        )
        # [R3.9S] R37_DUAL_GRAPHS=1 (serving environment): the rows of this gate fit the 2,048-token window, so they
        # are decoded by the window decoder, as generate() would decode them. One decoder: that one, as shipped.
        dec = getattr(runner, "graph_decoder_std", None) or runner.graph_decoder
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
            load_mtp=bool(args.mtp),
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
                if args.mtp and dec.mtp_enabled():   # [R3.9S] graph MTP sets of the TCQ runner, as main() captures them
                    dec.capture_mtp(preserve_state=False)

    check_runtime_modules(runtime_dir, "after graph capture")
    if dec is not None and getattr(dec, "r36_longctx", False):
        # [R3.9S] the Round 3.8 GraphDecoder follows the environment; the gate's reference numbers are window-decoder
        # numbers (the long-context step is a different float path: Round 3.7 measured mean KL 0.002 against it)
        print("[eval-r39] WARNING: the quality rows will be decoded by the LONG-CONTEXT decoder (R36_LONGCTX=1 and no "
              "window decoder). Unset R36_LONGCTX (or set R37_DUAL_GRAPHS=1 for the QTIP runner) for the gate.", flush=True)
    last_key = (dec._seq[-1][3] or dec._seq[-1][1]) if dec is not None else None

    free_b, tot_b = torch.cuda.mem_get_info()
    print(f"[eval-r39] runner loaded in {time.perf_counter() - t0:.0f}s; VRAM free {free_b / 2**30:.2f} of {tot_b / 2**30:.2f} GiB", flush=True)
    if args.speed_only:
        rows = []
    teacher = None
    if not args.speed_only:
        teacher = B.RealGLM53Backbone(B.RealGLM53Checkpoint(args.exact_backbone), runner.cfg, device="cuda", backbone_mode="fp8")
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
        # [R3.9S] which runtime folder, pool and decoder produced the numbers below
        "runtime": {
            "runtime_dir": str(runtime_dir),
            "pool_module": type(runner.pool).__module__,
            "pool_class": type(runner.pool).__name__,
            "pool_flags": runner.pool.flags_str() if callable(getattr(runner.pool, "flags_str", None)) else None,
            "pool_slots": int(getattr(runner.pool, "n_slots", 0) or 0),
            "registered_gib": runner.pool.stats.get("registered_gib"),
            "registered_rows": runner.pool.stats.get("registered_rows"),
            "decoder_longctx": bool(getattr(dec, "r36_longctx", False)) if dec is not None else None,
            "decoder_t_max": int(dec.t_max) if dec is not None else None,
        },
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
                report["speed_gate"] = run_speed_gate(runner, tok, use_mtp=bool(args.mtp))
        else:
            report["speed_gate"] = run_speed_gate(runner, tok, use_mtp=bool(args.mtp))
        if args.compare_ids is not None and args.compare_ids.exists():   # [R3.9S] token-for-token against another run
            other = (json.loads(args.compare_ids.read_text(encoding="utf-8")).get("speed_gate") or {}).get("generated_ids") or {}
            cmp_ids = {}
            for name, runs in report["speed_gate"]["generated_ids"].items():
                ref = (other.get(name) or [[]])[0]
                cur = runs[0] if runs else []
                n_eq = 0
                for a, b in zip(ref, cur):
                    if a != b:
                        break
                    n_eq += 1
                cmp_ids[name] = {"identical": bool(ref == cur), "first_difference_at": None if ref == cur else n_eq,
                                 "len_ref": len(ref), "len_cur": len(cur)}
            report["speed_gate"]["ids_vs_reference"] = {"reference": str(args.compare_ids), "prompts": cmp_ids}
            print(f"[IDS VS REFERENCE] {json.dumps(cmp_ids)}", flush=True)
            if (os.environ.get("R39_MTP_FP", "0").strip() == "1"
                    or os.environ.get("R39_MTP_STATS", "0").strip() == "1"):
                # [R3.9M] the stricter verdict (only with the flags set; the block above is unchanged): EVERY round's
                # ids against the reference's first round, and - when both reports carry fingerprints - the bits of
                # the logits behind every generated token. logits_all None = no fingerprints on one side.
                ref_sg = json.loads(args.compare_ids.read_text(encoding="utf-8")).get("speed_gate") or {}
                verdict = _r39m_strict_compare(ref_sg, report["speed_gate"])
                report["speed_gate"]["ids_vs_reference"]["r39m"] = verdict
                print(f"[R3.9M LOSSLESS] ids_all={verdict['ids_all']} logits_all={verdict['logits_all']} "
                      f"{json.dumps(verdict['prompts'])}", flush=True)
        if os.environ.get("R39_MTP_STATS", "0").strip() == "1":
            # [R3.9M] bookkeeping (flag only): how the verify / draft graph sets were captured (rows mode, batched op
            # classes, ops per step, VRAM) and whether the mHC kernels are row-local (then R35_MTP_BATCH=mhc is a no-op)
            mi = getattr(dec, "mtp_info", None) if dec is not None else None
            if mi is not None:
                report["speed_gate"]["mtp_info"] = json.loads(json.dumps(mi, default=str))   # plain JSON types only
                try:
                    report["speed_gate"]["mhc_row_local"] = bool(
                        runner.bb.use_kernels and int(runner.bb.hc) == 4 and B._FUSED_MERGE[0])
                except Exception:
                    report["speed_gate"]["mhc_row_local"] = None
            free_a, _ = torch.cuda.mem_get_info()
            report["speed_gate"]["vram_free_gib_after"] = round(free_a / 2**30, 2)
            report["speed_gate"]["vram_peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        report["speed_gate"]["vram_free_gib_before"] = round(free_b / 2**30, 2)
        report["speed_gate"]["env"] = {k: v for k, v in os.environ.items() if k[:4] in ("R32_", "R33_", "R34_", "R35_", "R36_", "R37_", "R38_", "R39_")
                                       or k.startswith("R39T_")}   # [R3.9T] the traffic flags are named R39T_*

    report["elapsed_min"] = round((time.perf_counter() - t0) / 60.0, 2)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[EVAL SUMMARY] {json.dumps(report['qbench'])}", flush=True)
    if "speed_gate" in report:
        # [R3.9M] the two long flag-only lists are left out of the log line (they are in the report file)
        print(f"[SPEED SUMMARY] {json.dumps({k: v for k, v in report['speed_gate'].items() if k not in ('runs', 'logit_fp', 'r39m_runs')})}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
