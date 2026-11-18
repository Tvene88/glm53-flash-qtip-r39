#!/usr/bin/env python3
"""
test_pool_r39t.py  [R3.9T]  GPU torture test of the new expert-pool flags (short: about 20 s per configuration)
================================================================================================================
Reason: the speed gate shows that a flag changes no token and no logit bit on four prompts; this test checks the
property that makes that so - every slot ensure() returns holds exactly the archive row of the requested expert when
its reader runs - under the pool's torture load (build_run/test_pool_v6_r35.py, part T: synthetic archives, a
300-slot pool, random worker delays, copy-stream sleeps, lagging readers, reused slots filled with 0xFF before they
are rewritten, every returned slot hashed on the GPU and compared with the archive). Nothing of the model is loaded.

The flags of this round are environment variables read by the pool's constructor; each configuration below sets them
around one torture run on top of the production pool options (signal, demand, demand stream, gdsf, the two rank
gates). `rows` registers 35 % of the rows (the cold path runs too, so R39T_LAZY_STAGING really allocates late);
`layers` registers whole layers (the registered fast path of R39T_FAST_DEMAND serves whole calls).

  python test_pool_r39t.py [--ops 30000] [--modes rows,layers] [--configs base,fast1,...] [--json out_r39s\\pool_torture_r39t.json]
Exit code 0 = every configuration passed (0 mismatching entries, clean final scan, invariants hold, no worker error).
Needs a free GPU (one CUDA process at a time); about 2 GiB of RAM and under 1 GiB of VRAM.
"""
import argparse
import gc
import json
import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD_RUN = os.path.join(os.path.dirname(HERE), "build_run")
os.environ["R34_RUNTIME"] = HERE                       # the torture module imports the pool from this directory
sys.path.insert(0, HERE)
sys.path.insert(1, BUILD_RUN)

PROD = {"signal": True, "demand": True, "demand_stream": True, "gather": "readfile", "evict": "gdsf",
        "cold_topk": 6, "cold_margin": 0.01, "reg_topk": 6}
CONFIGS = {  # name: (extra pool options, environment flags of this round)
    "base": ({}, {}),
    "fast1": ({}, {"R39T_FAST_DEMAND": "1"}),
    "fast2": ({}, {"R39T_FAST_DEMAND": "2"}),
    "lazy": ({}, {"R39T_LAZY_STAGING": "1"}),
    "preevict": ({}, {"R39T_PREEVICT": "4"}),
    "pfad": ({}, {"R39T_PF_AFTER_DEMAND": "1"}),
    "paced1": ({"drop": "all"}, {"R39T_PF_PACED": "1"}),
    "paced2": ({"drop": "all"}, {"R39T_PF_PACED": "2"}),
    "all": ({"drop": "all"}, {"R39T_FAST_DEMAND": "2", "R39T_LAZY_STAGING": "1", "R39T_PREEVICT": "4",
                              "R39T_PF_AFTER_DEMAND": "1", "R39T_PF_PACED": "1"}),
    "recommended": ({}, {"R39T_FAST_DEMAND": "1", "R39T_LAZY_STAGING": "1"}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", type=int, default=30000)
    ap.add_argument("--modes", default="rows,layers")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--slots", type=int, default=300)
    ap.add_argument("--seed", type=int, default=39)
    ap.add_argument("--json", default=os.path.join(HERE, "out_r39s", "pool_torture_r39t.json"))
    a = ap.parse_args()
    for k in [k for k in os.environ if k.startswith("R35_POOL_") or k.startswith("R37_POOL_") or k.startswith("R39T_")]:
        os.environ.pop(k)                               # options come from this test, not the shell
    os.environ.pop("R33_REGISTER_PROFILE", None)
    import torch
    import test_pool_v6_r35 as T                        # build_run's torture test (read only)
    assert os.path.normcase(os.path.dirname(T.P6.__file__)) == os.path.normcase(HERE), T.P6.__file__
    T.watchdog(120.0)
    rep = {"runtime": HERE, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "gpu": torch.cuda.get_device_name(0),
           "args": vars(a), "runs": {}}
    ok = True
    tmp = tempfile.mkdtemp(prefix="r39t_pool_gpu_")
    try:
        layers = [3, 4, 5, 6, 7, 8, 9, 10]
        arch = T.Arch(tmp, layers, 64, 98352, a.seed, prune={5: 9, 8: 21, 10: 3})
        seed = a.seed
        for mode in [m for m in a.modes.split(",") if m]:
            for nm in [c for c in a.configs.split(",") if c]:
                opts, env = CONFIGS[nm]
                seed += 1
                for k, v in env.items():
                    os.environ[k] = v
                try:
                    r = T.torture(arch, mode, nm, dict(PROD, **opts), a.ops, seed, a.slots, faults=False)
                finally:
                    for k in env:
                        os.environ.pop(k, None)
                rep["runs"][f"{mode}/{nm}"] = r
                ok &= bool(r["PASS"])
                st = r["stats"]
                extra = " ".join(f"{k[5:]}={st[k]}" for k in sorted(st) if k.startswith("r39t_"))
                print(f"[r39t torture] {mode}/{nm}: {'PASS' if r['PASS'] else 'FAIL'} ops {r['ops']} entries {r['entries']} "
                      f"mismatches {r['mismatch_entries']} scan {r['scan_bad']} inv {r['invariants']} hits {st['hits']} "
                      f"misses {st['misses']} prefetched {st['prefetch_issued']} claimed {st['prefetch_claimed']} dropped "
                      f"{st['prefetch_dropped']} skipped_reg {st['prefetch_skipped_reg']} errors {st['worker_errors']} "
                      f"{extra} ({r['seconds']} s)", flush=True)
    finally:
        rep["PASS"] = bool(ok)
        os.makedirs(os.path.dirname(a.json), exist_ok=True)
        with open(a.json, "w") as fh:
            json.dump(rep, fh, indent=1, default=str)
        arch = None
        gc.collect()
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"[r39t torture] {'PASS' if ok else 'FAIL'} -> {a.json}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
