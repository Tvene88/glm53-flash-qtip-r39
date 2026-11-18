#!/usr/bin/env python3
"""
port_exact_r39s.py  [R3.9S] forced-decode record / compare for the QTIP port (new file of claude_r39_speed)
============================================================================================================
Answers "is the port exact?" at the level of the whole model, the way the Round 3.7 exactness gates did for the
kernel changes (3 prompts x 128 forced graph steps, logits compared bitwise):

  record   load one runtime folder (this one, or the shipped Round 3.9 runtime claude_r39_qtip\\runtime), prefill three
           fixed prompts eagerly, then run --steps N = 1 CUDA-graph steps per prompt and keep every step's full
           logits. Without --force the next input is the step's own argmax (greedy); with --force REC the inputs are
           the ones of REC, so two records stay comparable step by step even if a logit differs.
           --eager-steps K also runs the first K steps of every prompt on the eager path (bb.forward, one token) and
           prints graph-against-eager KL / top-1. INFORMATION ONLY: in this code base the graph step and the eager
           step are different float paths (Round 3.7 measured mean KL 0.0038 for the TCQ archive inside the window),
           so that comparison is not expected to be bit-exact. The bit-exact graph == eager statement exists at the
           level of the routed-expert core: glm53_fold_qtip_runtime.py --selftest / --selftest-port.
  compare  two records -> per prompt: steps bitwise identical, max |diff|, mean / max KL, top-1 agreement. Exit code 0
           only when every step is bitwise identical. CPU only.

Both runners are supported (--runner auto | qtip | tcq), with the same construction arguments quality_eval_r39s.py
uses, so a record of the shipped TCQ33P5 archive is the control. Needs the GPU for `record`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent

# the first three prompts of quality_eval_r39s.SPEED_PROMPTS
PROMPTS = [
    ("code", "Write a compact Python class implementing an LRU cache using collections.OrderedDict with get and put in O(1) time."),
    ("capital", "List the capitals of France, Japan, Brazil, Canada, and Australia and state one river flowing near each."),
    ("math", "Derive the closed-form sum of k^2 for k = 1..n by induction and verify n = 10."),
]
ENV_PREFIXES = ("R32_", "R33_", "R34_", "R35_", "R36_", "R37_", "R38_", "R39_")


def _norm(path) -> str:
    return os.path.normcase(os.path.normpath(str(Path(path).resolve())))


def check_runtime_modules(runtime_dir: Path, where: str) -> None:
    root = _norm(runtime_dir)
    bad = []
    for name, m in list(sys.modules.items()):
        f = getattr(m, "__file__", None)
        if name.startswith("glm53_") and f:
            p = _norm(f)
            try:
                inside = os.path.commonpath([root, p]) == root
            except ValueError:
                inside = False
            if not inside:
                bad.append(f"{name} -> {f}")
    if bad:
        raise SystemExit(f"module provenance check failed {where}: glm53_* modules outside {runtime_dir}: " + "; ".join(sorted(bad)))


def record(args) -> int:
    if os.environ.get("R37_GPU_WINDOW", "1") != "1" and os.environ.get("R36_GPU_HANDOVER", "1") != "1":
        raise SystemExit("GPU handover blocked by env guard (set R37_GPU_WINDOW=1)")
    runtime_dir = Path(args.runtime_dir).resolve()
    if _norm(runtime_dir) != _norm(HERE):          # another runtime folder: this file's folder must not shadow it
        sys.path[:] = [p for p in sys.path if p and _norm(p) != _norm(HERE)]
    sys.path.insert(0, str(runtime_dir))

    import torch
    import glm53_fold_backbone as B  # type: ignore
    import glm53_fold_generate as G  # type: ignore
    from glm53_fold_graphs import GraphDecoder  # type: ignore

    mode = args.runner
    if mode == "auto":
        mode = "qtip" if any(n.endswith("_qtip_hyb.pt") for n in os.listdir(args.fold)) else "tcq"
    t0 = time.perf_counter()
    if mode == "qtip":
        import glm53_fold_qtip_runtime as QR  # type: ignore
        runner = QR.QtipMoEModelRunner(
            backbone_dir=args.backbone, fold_dir=args.fold, merged_routers_path="", device="cuda", backbone_mode="fp8",
            expert_pool_gib=args.pool_gib, register_gib=float(args.register_gib), prefetch=True, capture_graphs=True,
            t_max=2048, load_mtp=False, vocab_keep=None, prune_spec=None, pkg_dir=str(runtime_dir), prefix_cache=False,
        )
        dec = getattr(runner, "graph_decoder_std", None) or runner.graph_decoder
    else:
        runner = G.ApexFoldMoEModelRunner(
            backbone_dir=args.backbone, fold_dir=args.fold, device="cuda", backbone_mode="fp8", cold_mode="mmap",
            expert_pool_gib=args.pool_gib, prefetch=True, sync_timing=False, fp8_diet=True, load_mtp=False, vocab_keep=None,
        )
        dec = GraphDecoder(runner, t_max=2048)
        runner.graph_decoder = dec
        dec.capture()
    check_runtime_modules(runtime_dir, "after load")
    print(f"[port-exact] {mode} runner of {runtime_dir} loaded in {time.perf_counter() - t0:.0f}s; pool "
          f"{type(runner.pool).__module__}.{type(runner.pool).__name__}", flush=True)

    tok = G.RealTokenizer(args.backbone)
    forced = torch.load(args.force, map_location="cpu")["prompts"] if args.force else None
    out = {
        "meta": {
            "runtime_dir": str(runtime_dir), "runner": mode, "fold": str(args.fold), "backbone": str(args.backbone),
            "steps": int(args.steps), "forced_by": str(args.force or ""), "pool": type(runner.pool).__module__,
            "pool_flags": runner.pool.flags_str() if callable(getattr(runner.pool, "flags_str", None)) else None,
            "env": {k: v for k, v in os.environ.items() if k[:4] in ENV_PREFIXES},
        },
        "prompts": [],
    }

    def prefill(ids):
        cache = B.RealCache()
        logits, _ = runner.bb.forward(torch.tensor([ids], device="cuda"), cache=cache, moe_fn=runner.moe_dispatch,
                                      last_only=True, max_layers=runner.max_layers)
        return cache, logits[0, -1].float()

    with torch.no_grad():
        for pi, (name, text) in enumerate(PROMPTS):
            ids = [int(i) for i in tok.encode_chat(text, thinking=False)]
            inputs = None
            if forced is not None:
                if forced[pi]["prompt_ids"] != ids:
                    raise SystemExit(f"--force record was made with another prompt / tokenizer (prompt {name})")
                inputs = [int(t) for t in forced[pi]["inputs"]]
                if len(inputs) < args.steps:
                    raise SystemExit(f"--force record has {len(inputs)} steps, --steps asks for {args.steps}")
            cache, lg_pre = prefill(ids)
            first = int(runner.bb.argmax_token(lg_pre))
            if not dec.load_cache(cache):
                raise SystemExit(f"prompt {name}: {len(ids)} tokens do not fit the graph window")
            tok_in = inputs[0] if inputs is not None else first
            g_logits, g_inputs, g_next = [], [], []
            for t in range(args.steps):
                dec.step(int(tok_in))
                nt = int(dec.next_tok.item())                      # syncs: the step is complete
                g_logits.append(dec.logits.reshape(-1).float().cpu())
                g_inputs.append(int(tok_in))
                g_next.append(nt)
                tok_in = inputs[t + 1] if (inputs is not None and t + 1 < len(inputs)) else nt
            torch.cuda.synchronize()
            rec = {"name": name, "prompt_ids": ids, "first_token": first, "prefill_logits": lg_pre.cpu(),
                   "inputs": g_inputs, "next": g_next, "logits": torch.stack(g_logits, 0)}

            if args.eager_steps > 0:                               # information only, see the module docstring
                import torch.nn.functional as F
                cache_e, _ = prefill(ids)
                n_e = min(int(args.eager_steps), int(args.steps))
                kl, mx, agree, same = 0.0, 0.0, 0, 0
                for t in range(n_e):
                    lg_e, _ = runner.bb.forward(torch.tensor([[g_inputs[t]]], device="cuda"), cache=cache_e,
                                                moe_fn=runner.moe_dispatch, last_only=True, max_layers=runner.max_layers)
                    le = lg_e[0, -1].float()
                    lgr = g_logits[t].to("cuda")
                    lp, lq = F.log_softmax(le, -1), F.log_softmax(lgr, -1)
                    kl += float((lp.exp() * (lp - lq)).sum(dtype=torch.float64))
                    mx = max(mx, float((le - lgr).abs().max()))
                    agree += int(le.argmax() == lgr.argmax())
                    same += int(torch.equal(le, lgr))
                rec["graph_vs_eager"] = {"steps": n_e, "bitwise_identical": same, "top1_agree": agree,
                                         "mean_kl_eager_to_graph": kl / max(n_e, 1), "max_abs_logit_diff": mx}
                print(f"[port-exact] {name}: graph against eager over {n_e} steps (information): "
                      f"{json.dumps(rec['graph_vs_eager'])}", flush=True)
            out["prompts"].append(rec)
            print(f"[port-exact] {name}: {len(ids)} prompt tokens, {args.steps} graph steps recorded "
                  f"({'forced' if inputs is not None else 'greedy'})", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"[port-exact] record written: {args.out}", flush=True)
    return 0


def compare(args) -> int:
    import torch
    import torch.nn.functional as F

    a = torch.load(args.a, map_location="cpu")
    b = torch.load(args.b, map_location="cpu")
    print(f"A: {a['meta']['runtime_dir']} ({a['meta']['runner']}, pool {a['meta']['pool']}, flags {a['meta'].get('pool_flags')})")
    print(f"B: {b['meta']['runtime_dir']} ({b['meta']['runner']}, pool {b['meta']['pool']}, flags {b['meta'].get('pool_flags')})")
    ea, eb = a["meta"].get("env", {}), b["meta"].get("env", {})
    diff_env = {k: (ea.get(k), eb.get(k)) for k in sorted(set(ea) | set(eb)) if ea.get(k) != eb.get(k)}
    print(f"environment differences (A, B): {json.dumps(diff_env)}")
    tot = ident = 0
    report = {"prompts": [], "env_diff": diff_env}
    for pa, pb in zip(a["prompts"], b["prompts"]):
        if pa["prompt_ids"] != pb["prompt_ids"]:
            raise SystemExit(f"prompt {pa['name']}: the two records used different prompt ids")
        n = min(pa["logits"].shape[0], pb["logits"].shape[0])
        if pa["inputs"][:n] != pb["inputs"][:n]:
            print(f"  NOTE {pa['name']}: the step inputs differ (record B was not made with --force A); "
                  "steps after the first difference are not comparable")
        la, lb = pa["logits"][:n].float(), pb["logits"][:n].float()
        same = (la.contiguous().view(torch.int32) == lb.contiguous().view(torch.int32)).all(dim=1)
        lp, lq = F.log_softmax(la, -1), F.log_softmax(lb, -1)
        kl = (lp.exp() * (lp - lq)).sum(-1, dtype=torch.float64)
        row = {
            "name": pa["name"], "steps": int(n), "bitwise_identical": int(same.sum()),
            "first_different_step": (int((~same).nonzero()[0]) if not bool(same.all()) else None),
            "max_abs_diff": float((la - lb).abs().max()), "mean_kl": float(kl.mean()), "max_kl": float(kl.max()),
            "top1_agree": int((la.argmax(-1) == lb.argmax(-1)).sum()),
            "prefill_logits_identical": bool(torch.equal(pa["prefill_logits"], pb["prefill_logits"])),
            "first_token_equal": bool(pa["first_token"] == pb["first_token"]),
        }
        report["prompts"].append(row)
        tot += n
        ident += row["bitwise_identical"]
        print(f"  {json.dumps(row)}")
    report["total_steps"], report["bitwise_identical"] = tot, ident
    verdict = "BIT-EXACT" if ident == tot and tot > 0 else "DIFFERENT"
    print(f"[port-exact] {ident} of {tot} forced graph steps bitwise identical: {verdict}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if verdict == "BIT-EXACT" else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="[R3.9S] forced-decode record / compare for the QTIP port")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="GPU: record the logits of forced / greedy graph steps")
    r.add_argument("--runtime-dir", default=str(HERE))
    r.add_argument("--runner", default="auto", choices=["auto", "qtip", "tcq"])
    r.add_argument("--backbone", default=r"C:\models\GLM53-backbone-r34h")
    r.add_argument("--fold", required=True)
    r.add_argument("--pool-gib", type=float, default=16.0)
    r.add_argument("--register-gib", type=float, default=42.0,
                   help="QTIP runner only (explicit, so both runtime folders get the same value); the TCQ runner "
                        "reads R33_REGISTER_GIB")
    r.add_argument("--steps", type=int, default=128)
    r.add_argument("--force", default="", help="record whose step inputs are replayed (teacher forcing)")
    r.add_argument("--eager-steps", type=int, default=0, help="also compare the first K steps with the eager path")
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare", help="CPU: compare two records")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--json", default="")
    args = ap.parse_args()
    return record(args) if args.cmd == "record" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
