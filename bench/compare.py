#!/usr/bin/env python3
"""
compare.py - table of several bench_openai.py runs of the SAME task, with paired comparisons.

  python compare.py new=runs/new/humaneval shipped=runs/shipped/humaneval unsloth=runs/unsloth/humaneval

Each argument is [label=]path; path is a run directory (with items.jsonl), an items.jsonl or a summary.json next to
one. Printed: per run the score with its Wilson 95 % interval, truncated answers, mean tokens and tok/s; then for
every pair of runs, on the items both answered: both right / only A / only B / neither, and the exact two-sided
McNemar p-value (binomial test on the discordant items). The pair table also says whether the two servers counted
the same number of prompt tokens for every common item, which they must if the prompts were built identically.
Standard library only.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple


def wilson(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: b, c = the two discordant counts; binomial(b + c, 1/2)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def load_run(spec: str) -> Dict[str, Any]:
    label, sep, path = spec.partition("=")
    if not sep:
        label, path = "", spec
    if os.path.isdir(path):
        d = path
    else:
        d = os.path.dirname(os.path.abspath(path))
    items_path = path if path.endswith(".jsonl") and os.path.isfile(path) else os.path.join(d, "items.jsonl")
    if not os.path.isfile(items_path):
        raise SystemExit(f"{spec}: no items.jsonl at {items_path}")
    items = {}
    with open(items_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                items[r["id"]] = r
    summary = {}
    sp = os.path.join(d, "summary.json")
    if os.path.isfile(sp):
        with open(sp, "r", encoding="utf-8") as f:
            summary = json.load(f)
    if not label:
        label = os.path.basename(os.path.normpath(d)) or d
    return {"label": label, "items": items, "summary": summary}


def mean(vals) -> Optional[float]:
    v = [float(x) for x in vals if isinstance(x, (int, float))]
    return sum(v) / len(v) if v else None


def fmt(x: Optional[float], spec: str = ".1f") -> str:
    return "-" if x is None else format(x, spec)


def table(rows: List[List[str]]) -> str:
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    out = []
    for n, r in enumerate(rows):
        out.append("  ".join(c.ljust(w[i]) if i == 0 else c.rjust(w[i]) for i, c in enumerate(r)))
        if n == 0:
            out.append("  ".join("-" * x for x in w))
    return "\n".join(out)


def stats(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(items)
    k = sum(1 for r in items if r["correct"])
    lo, hi = wilson(k, n)
    return {"n": n, "k": k, "score": k / n if n else None, "lo": lo, "hi": hi}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="[label=]run directory | items.jsonl | summary.json")
    ap.add_argument("--json", default=None, help="also write the table data to this file")
    args = ap.parse_args(argv)
    runs = [load_run(s) for s in args.runs]
    tasks = {r["summary"].get("task") for r in runs if r["summary"].get("task")}
    if len(tasks) > 1:
        raise SystemExit(f"the runs are of different tasks: {sorted(tasks)}")
    common = set.intersection(*[set(r["items"]) for r in runs]) if runs else set()
    print(f"task: {next(iter(tasks)) if tasks else '?'}; runs: {len(runs)}; items answered by all runs: {len(common)}\n")

    head = ["run", "items", "correct", "score %", "Wilson 95 %", "truncated", "prompt tok", "compl tok",
            "tok/s wall", "tok/s reported", "score % (common items)"]
    rows, data = [head], {"runs": [], "pairs": [], "n_common": len(common)}
    for r in runs:
        its = list(r["items"].values())
        s = stats(its)
        sc = stats([r["items"][i] for i in common])
        d = {"label": r["label"], "n": s["n"], "n_correct": s["k"], "score": s["score"], "wilson95": [s["lo"], s["hi"]],
             "n_truncated": sum(1 for x in its if x.get("truncated")),
             "mean_prompt_tokens": mean(x.get("prompt_tokens") for x in its),
             "mean_completion_tokens": mean(x.get("completion_tokens") for x in its),
             "mean_tok_s_wall": mean(x.get("tok_s_wall") for x in its),
             "mean_tok_s_reported": mean(x.get("tok_s_reported") for x in its),
             "score_common": sc["score"], "complete": r["summary"].get("complete")}
        data["runs"].append(d)
        flag = "" if d["complete"] in (True, None) else " (incomplete)"
        rows.append([r["label"] + flag, str(s["n"]), str(s["k"]), fmt(None if s["score"] is None else 100 * s["score"]),
                     f"{100 * s['lo']:.1f}-{100 * s['hi']:.1f}", str(d["n_truncated"]), fmt(d["mean_prompt_tokens"]),
                     fmt(d["mean_completion_tokens"]), fmt(d["mean_tok_s_wall"], ".2f"), fmt(d["mean_tok_s_reported"], ".2f"),
                     fmt(None if sc["score"] is None else 100 * sc["score"])])
    print(table(rows))
    print("\ntok/s wall = completion tokens / request time (prompt processing included), mean over items;\n"
          "tok/s reported = the server's own decode rate, when it reports one. Truncated answers count as wrong.\n")

    if len(runs) > 1:
        prow = [["A vs B", "common", "both right", "only A", "only B", "neither", "A-B points", "McNemar exact p",
                 "same prompt tokens"]]
        for a, b in itertools.combinations(runs, 2):
            ids = sorted(set(a["items"]) & set(b["items"]))
            both = sum(1 for i in ids if a["items"][i]["correct"] and b["items"][i]["correct"])
            oa = sum(1 for i in ids if a["items"][i]["correct"] and not b["items"][i]["correct"])
            ob = sum(1 for i in ids if not a["items"][i]["correct"] and b["items"][i]["correct"])
            nei = len(ids) - both - oa - ob
            p = mcnemar_exact(oa, ob)
            pt = [(a["items"][i].get("prompt_tokens"), b["items"][i].get("prompt_tokens")) for i in ids]
            known = [(x, y) for x, y in pt if isinstance(x, int) and isinstance(y, int)]
            same = sum(1 for x, y in known if x == y)
            parity = f"{same}/{len(known)}" if known else "-"
            diff = 100.0 * (oa - ob) / len(ids) if ids else None
            prow.append([f"{a['label']} vs {b['label']}", str(len(ids)), str(both), str(oa), str(ob), str(nei),
                         "-" if diff is None else f"{diff:+.1f}", f"{p:.4f}", parity])
            data["pairs"].append({"a": a["label"], "b": b["label"], "n_common": len(ids), "both_right": both, "only_a": oa,
                                  "only_b": ob, "neither": nei, "mcnemar_exact_p": p,
                                  "same_prompt_tokens": same, "prompt_tokens_known": len(known)})
        print(table(prow))
        print("\nsame prompt tokens: items for which both servers report the same prompt token count. Anything below "
              "all of them\nmeans the two servers did not build the same prompt (chat template, thinking switch, "
              "reasoning effort).")
    if args.json:
        with open(args.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
