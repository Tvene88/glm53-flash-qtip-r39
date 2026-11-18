#!/usr/bin/env python3
"""prefetch_shards_r33.py - [R3.3] prefetch_shards_r31_par.py with disk accounting: a download starts only if free disk minus 5 GiB per in-flight shard, minus this shard, minus --reserve-gib for the converter (layer file + checkpoint transients) stays above --floor-gib.
(per-connection throughput from the Hub is ~30 MiB/s unauthenticated; parallel streams add up). Download-only; never
deletes. Targets = shards of the next --ahead pending layers (current layer = first pending, fetched by the converter).
New downloads only start while free disk >= --floor-gib."""
import argparse
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

LM = "model.language_model."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--repo", default="dealignai/GLM-5.3-Flash-UNCENSORED-FP8")
    ap.add_argument("--order", required=True)
    ap.add_argument("--ahead", type=int, default=3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--floor-gib", type=float, default=25.0)
    ap.add_argument("--reserve-gib", type=float, default=6.0)
    args = ap.parse_args()
    from huggingface_hub import hf_hub_download

    order = [int(x) for x in args.order.split(",") if x.strip()]
    wm = json.load(open(os.path.join(args.src, "model.safetensors.index.json"), encoding="utf-8"))["weight_map"]
    need = {l: sorted({sh for k, sh in wm.items() if k.startswith(f"{LM}layers.{l}.mlp.experts.")}) for l in order}
    prog_path = os.path.join(args.dst, "convert_progress.json")
    lock = threading.Lock()
    inflight, attempted = set(), set()

    last_done = [set()]

    def done_layers():
        # The converter rewrites convert_progress.json in place (truncate + write); a read that lands mid-write fails
        # to parse. Keep the last good value instead of treating every layer as pending (which re-fetched shard 3).
        try:
            d = set(int(k) for k in json.load(open(prog_path, encoding="utf-8"))["layers"].keys())
            if d >= last_done[0]:
                last_done[0] = d
        except Exception:
            pass
        return set(last_done[0])

    def fetch(sh, for_layers, free_gib):
        t0 = time.perf_counter()
        print(f"[prefetch] {time.strftime('%H:%M:%S')} fetching {sh} for layer(s) {for_layers} (free {free_gib:.1f} GiB, in flight {len(inflight)})", flush=True)
        ok = False
        for attempt in range(5):
            try:
                hf_hub_download(repo_id=args.repo, filename=sh, local_dir=args.src)
                ok = True
                break
            except Exception as ex:
                print(f"[prefetch] {sh} attempt {attempt + 1} failed: {ex!r}", flush=True)
                time.sleep(15 * (attempt + 1))
        p = os.path.join(args.src, sh)
        dt = time.perf_counter() - t0
        if ok and os.path.exists(p):
            print(f"[prefetch] {time.strftime('%H:%M:%S')} {sh} {os.path.getsize(p) / 2**30:.2f} GiB in {dt:.0f}s ({os.path.getsize(p) / 2**20 / max(dt, 1e-3):.0f} MiB/s)", flush=True)
        else:
            print(f"[prefetch] giving up on {sh}; converter will fetch it itself", flush=True)
        with lock:
            inflight.discard(sh)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while True:
            done = done_layers()
            pending = [l for l in order if l not in done]
            if not pending:
                print(f"[prefetch] {time.strftime('%H:%M:%S')} all layers converted; exiting", flush=True)
                return 0
            nxt = pending[1: 1 + args.ahead]
            targets = []
            for l in nxt:
                for sh in need[l]:
                    if sh not in targets:
                        targets.append(sh)
            with lock:
                todo = [sh for sh in targets if sh not in inflight and sh not in attempted
                        and not os.path.exists(os.path.join(args.src, sh))]
                slots = args.workers - len(inflight)
            free_gib = shutil.disk_usage(args.src).free / 2**30
            if todo and slots > 0 and free_gib - 5.0 * (args.workers - slots) - args.reserve_gib - 5.0 >= args.floor_gib:  # [R3.3] room for in-flight + this shard + driver writes
                sh = todo[0]
                with lock:
                    inflight.add(sh)
                    attempted.add(sh)
                pool.submit(fetch, sh, [l for l in nxt if sh in need[l]], free_gib)
                time.sleep(1)
                continue
            time.sleep(5)


if __name__ == "__main__":
    sys.exit(main())
