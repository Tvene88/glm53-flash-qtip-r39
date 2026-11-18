#!/usr/bin/env python3
"""bench_guard_r34.py - benchmark hygiene for timed runs on this workstation (machine diagnosis 2026-11-10).

Back-to-back runs that page-lock 48 GiB start cold (registration 24-26 s instead of ~10 s, +6-8 ms/token of SSD
re-reads), and other applications or heavy disk work cause one-off stalls. Use:

    python bench_guard_r34.py pre  [--fold DIR] [--min-avail-gib 80] [--cooldown 60]   # before a timed run
    python bench_guard_r34.py post --log RUN_LOG [--max-register-s 12]                  # after it: accept / reject

`pre`  waits until no other python / GPU compute process runs, the last guarded run ended >= cooldown s ago and
       available RAM >= the floor (says why every 10 min), then reads the archive's layer files once sequentially so the
       non-locked part starts warm. Exit 0 = go; with --stop-file PATH: exit 5 as soon as PATH exists (nothing started).
`post` parses "registered tier ... locked in X s" from the run log; exit 0 if X <= the limit (run valid), 3 if the run
       started cold (re-run it), 4 if no registration line was found."""
import argparse
import ctypes
import glob
import json
import os
import re
import subprocess
import sys
import time
from ctypes import wintypes

STAMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_r32", "bench_guard_last_end.json")
LIMITS = os.path.join(os.path.dirname(STAMP), "bench_guard_limits.json")


def override(name, value):
    """Operator override, read at every call from out_r32/bench_guard_limits.json ({"max_register_s": 22.0,
    "min_avail_gib": 73}): corrects a limit between the runs of a chain that is already running, whose command lines
    are fixed. Returns (limit in force, note for the guard line); without the file or the key: (value, "")."""
    try:
        v = float(json.load(open(LIMITS, encoding="utf-8"))[name])
    except (OSError, ValueError, TypeError, KeyError):
        return value, ""
    return v, f" [{name} {v:g} from bench_guard_limits.json, command line {value:g}]"


def avail_gib():
    class M(ctypes.Structure):
        _fields_ = [("l", wintypes.DWORD), ("load", wintypes.DWORD), ("tot", ctypes.c_ulonglong),
                    ("av", ctypes.c_ulonglong), ("a", ctypes.c_ulonglong), ("b", ctypes.c_ulonglong),
                    ("c", ctypes.c_ulonglong), ("d", ctypes.c_ulonglong), ("e", ctypes.c_ulonglong)]
    m = M()
    m.l = ctypes.sizeof(M)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.av / 2**30


def other_python():
    me = os.getpid()
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ForEach-Object { $_.ProcessId }"],
                         capture_output=True, text=True).stdout.split()
    pids = {int(p) for p in out if p.isdigit()} - {me, os.getppid()}
    return sorted(pids)


def gpu_apps():
    """[R3.4 shrink Phase A] entries whose used_memory is "[N/A]" are skipped: on this WDDM driver nvidia-smi lists every
    desktop graphics client (dwm, explorer, Chrome, ...) as a compute app with "[N/A]" memory, so the unfiltered check
    never let a run start (2026-11-10 05:05). GPU jobs here are python processes, which other_python() still catches."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return [l for l in out.splitlines() if l.strip() and not l.strip().endswith("[N/A]")]
    except Exception:  # noqa: BLE001
        return []


def pre(args):
    t0 = t_said = time.time()
    stop = getattr(args, "stop_file", None)  # other callers (guard_r36.py) may pass their own namespace
    while True:
        if stop and os.path.exists(stop):  # operator stop request (2026-11-10): start nothing
            print(f"[bench_guard] stop file {stop} exists: not starting", flush=True)
            return 5
        reasons = []
        py = other_python()
        if py:
            reasons.append(f"other python processes {py}")
        g = gpu_apps()
        if g:
            reasons.append(f"GPU compute apps {g}")
        if os.path.exists(STAMP):
            dt = time.time() - json.load(open(STAMP))["t"]
            if dt < args.cooldown:
                reasons.append(f"cooldown {args.cooldown - dt:.0f} s left")
        av = avail_gib()
        floor, _ = override("min_avail_gib", args.min_avail_gib)
        if av < floor:
            reasons.append(f"available RAM {av:.1f} GiB < {floor:g}")
        if not reasons:
            break
        if time.time() - t_said > 600:  # every 10 min: what still blocks the start
            print(f"[bench_guard] still waiting after {(time.time() - t0) / 60:.0f} min: " + "; ".join(reasons), flush=True)
            t_said = time.time()
        time.sleep(10)
    if args.fold:
        t1 = time.time()
        n = 0
        for p in sorted(glob.glob(os.path.join(args.fold, "layer_*_apex_fold.npz"))):
            with open(p, "rb", buffering=0) as f:
                while True:
                    b = f.read(64 << 20)
                    if not b:
                        break
                    n += len(b)
        print(f"[bench_guard] warmed {n / 2**30:.1f} GiB in {time.time() - t1:.0f} s", flush=True)
    print(f"[bench_guard] go (available {avail_gib():.1f} GiB)", flush=True)
    return 0


def post(args):
    os.makedirs(os.path.dirname(STAMP), exist_ok=True)
    json.dump({"t": time.time()}, open(STAMP, "w"))
    txt = open(args.log, encoding="utf-8", errors="replace").read()
    m = re.findall(r"registered tier.*?locked in ([0-9.]+) s", txt)
    if not m:
        print("[bench_guard] no registration line found", flush=True)
        return 4
    s = float(m[-1])
    lim, note = override("max_register_s", args.max_register_s)
    ok = s <= lim
    print(f"[bench_guard] registration {s:.1f} s -> {'VALID' if ok else 'COLD START, re-run'}{note}", flush=True)
    return 0 if ok else 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pre", "post"])
    ap.add_argument("--fold", default=None)
    ap.add_argument("--min-avail-gib", type=float, default=80.0)
    ap.add_argument("--cooldown", type=float, default=60.0)
    ap.add_argument("--log", default=None)
    ap.add_argument("--max-register-s", type=float, default=12.0)
    ap.add_argument("--stop-file", default=None)  # pre: exit 5 while this file exists
    args = ap.parse_args()
    return pre(args) if args.mode == "pre" else post(args)


if __name__ == "__main__":
    sys.exit(main())
