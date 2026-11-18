#!/usr/bin/env python3
"""
run_r39_master_pipeline.py  [Round 3.9 Idempotent Gate-Enforced Master Orchestrator]
====================================================================================
End-to-end automated pipeline driver for terminal coding agents (Claude Code / Codex) on 1x RTX 5090
(Windows 11 PowerShell or WSL2 Ubuntu). Executes stages sequentially or individually, enforces
pre-flight GPU/RAM/disk safety checks, records an auditable TSV ledger (`out_r39/r39_ledger.tsv`),
and supports crash-safe `--resume` across every long-running phase.

Ordered Stages:
  0. `preflight`     : Offline package integrity (`verify_r39_package.py`), disk space, RAM & GPU idle guard
  1. `selftests`     : Runs `--selftest` (~35s total) on all 4 GPU modules (L27 lab, encoder, trainer, runtime)
  2. `masks`         : Builds/refreshes non-uniform cross-layer LOO prune masks (`r38_waterfill_prune_select.py`)
  3. `r38_control`   : (Optional) Fast Round 3.8 control (`r34_prune_write.py` + Stage-2B on TCQ33PW16 -> 75.00 GB)
  4. `l27_gate`      : Single-layer Layer 27 7-rung SNR validation (`r39_l27_qtip_2d_ldlq_lab.py`, requires >= +0.35 dB)
  5. `build_r39`     : Full 42-layer + MTP Layer-45 streaming QTIP HYB encoder (`run_qtip_build_r39.py`, resumable)
  6. `teacher_cache` : Builds/verifies `train_cal_top128.pt` (`r39_extract_teacher_top128.py`)
  7. `stage2b_train` : Layer-Major Global-KL co-adaptation (`r39_stage2b_global_kl_trainer.py`, exports `-FT` dir)
  8. `eval_r39`      : Official `chunk=1` `GraphDecoder` quality + 3-round speed gate (`quality_eval_r39.py`)
  9. `serve_r39`     : Launches live OpenAI-compatible multi-turn HTTP server (`glm53_fold_qtip_runtime.py --serve`)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Dict, List

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
OUT_DIR = HERE / "out_r39"
PROFILES_JSON = PKG_ROOT / "configs" / "profiles_r39.json"

ALL_STAGES = (
    "preflight",
    "selftests",
    "masks",
    "r38_control",
    "l27_gate",
    "build_r39",
    "teacher_cache",
    "stage2b_train",
    "eval_r39",
    "serve_r39",
)
DEFAULT_PIPELINE = (
    "preflight",
    "selftests",
    "masks",
    "l27_gate",
    "build_r39",
    "teacher_cache",
    "stage2b_train",
    "eval_r39",
)


def adapt_path(p: str, models_root: str) -> str:
    """Translates default `C:/models/...` profile paths to `--models-root` (e.g. `/mnt/c/models` on WSL2)."""
    norm = p.replace("\\", "/")
    if norm.lower().startswith("c:/models/"):
        suffix = norm[len("c:/models/") :]
        return str(Path(models_root) / suffix)
    if norm.lower() == "c:/models":
        return str(Path(models_root))
    return str(PKG_ROOT / norm) if not os.path.isabs(norm) else norm


def check_gpu_idle(max_used_mib: int = 4096) -> Dict[str, object]:
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return {"ok": False, "reason": "nvidia-smi not found in PATH"}
    proc = subprocess.run(
        [nvsmi, "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return {"ok": False, "reason": f"nvidia-smi failed: {proc.stderr.strip()}"}
    first = [x.strip() for x in proc.stdout.strip().splitlines()[0].split(",")]
    name, used, total = first[0], int(first[1]), int(first[2])
    return {
        "ok": used <= max_used_mib,
        "gpu": name,
        "used_mib": used,
        "total_mib": total,
        "reason": "ok" if used <= max_used_mib else f"VRAM in use {used} MiB > {max_used_mib} MiB",
    }


def append_ledger(stage: str, status: str, elapsed_s: float, detail: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ledger = OUT_DIR / "r39_ledger.tsv"
    if not ledger.exists():
        ledger.write_text("timestamp\tstage\tstatus\telapsed_s\tdetail\n", encoding="utf-8")
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(ledger, "a", encoding="utf-8") as fh:
        fh.write(f"{ts}\t{stage}\t{status}\t{elapsed_s:.1f}\t{detail}\n")


def run_cmd(stage: str, cmd: List[str], env: Dict[str, str], dry_run: bool) -> int:
    cmd_str = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    print(f"\n{'=' * 92}\n[STAGE: {stage}] {cmd_str}\n{'=' * 92}", flush=True)
    if dry_run:
        append_ledger(stage, "DRY_RUN", 0.0, cmd_str)
        return 0
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, cwd=str(PKG_ROOT), check=False)
    dt = time.perf_counter() - t0
    status = "PASS" if proc.returncode == 0 else ("STOP_CKPT" if proc.returncode == 3 else f"FAIL({proc.returncode})")
    append_ledger(stage, status, dt, cmd_str)
    return int(proc.returncode)


def main() -> int:
    ap = argparse.ArgumentParser(description="Round 3.9 Master Build, Co-Adaptation & Evaluation Orchestrator")
    ap.add_argument("--profile", default="r39_flagship_75gb_sub023", help="Profile key in configs/profiles_r39.json")
    ap.add_argument("--stages", default="default", help="'default', 'all', or comma-separated subset of stages")
    ap.add_argument("--models-root", default=("C:/models" if os.name == "nt" else "/mnt/c/models"), help="Root models directory")
    ap.add_argument("--backbone-r34h", default="", help="Override slim FP8 backbone directory")
    ap.add_argument("--exact-backbone", default="", help="Override full-head FP8 backbone directory")
    ap.add_argument("--fp8-src", default="", help="Override 328 GB FP8 safetensor shards directory")
    ap.add_argument("--tcq33-dir", default="", help="Override existing TCQ33 archive directory")
    ap.add_argument("--teacher-cache", default="", help="Override train_cal_top128.pt path")
    ap.add_argument("--sweeps", type=int, default=6, help="Stage-2B Global-KL sweeps")
    ap.add_argument("--resume", action="store_true", help="Resume interrupted build_r39 stage")
    ap.add_argument("--dry-run", action="store_true", help="Print planned commands without executing GPU jobs")
    ap.add_argument("--skip-vram-guard", action="store_true", help="Bypass <=4096 MiB idle VRAM precheck")
    args = ap.parse_args()

    profiles = json.loads(PROFILES_JSON.read_text(encoding="utf-8"))["profiles"]
    if args.profile not in profiles:
        raise SystemExit(f"Unknown profile {args.profile!r}. Available: {', '.join(profiles)}")
    prof = profiles[args.profile]

    stages = list(DEFAULT_PIPELINE) if args.stages == "default" else (
        list(ALL_STAGES) if args.stages == "all" else [s.strip() for s in args.stages.split(",") if s.strip()]
    )
    bad_s = [s for s in stages if s not in ALL_STAGES]
    if bad_s:
        raise SystemExit(f"Invalid stages {bad_s}. Valid: {ALL_STAGES}")

    mroot = args.models_root
    bb_r34h = args.backbone_r34h or str(Path(mroot) / "GLM53-backbone-r34h")
    bb_exact = args.exact_backbone or str(Path(mroot) / "GLM53-backbone")
    fp8_src = args.fp8_src or str(Path(mroot) / "GLM-5.3-Flash-UNCENSORED-FP8")
    tcq33_dir = args.tcq33_dir or str(Path(mroot) / "GLM53-APEX-FOLD-TCQ33")
    tcache = args.teacher_cache or str(Path(mroot) / "r39_teacher_top128" / "train_cal_top128.pt")
    raw_arch = adapt_path(prof.get("raw_archive_dir", str(Path(mroot) / "GLM53-APEX-FOLD-R39-HYB")), mroot)
    exp_arch = adapt_path(prof.get("export_dir", str(Path(mroot) / "GLM53-APEX-FOLD-R39-HYB-FT")), mroot)
    mask_path = adapt_path(prof["prune_mask"], mroot)
    rule = prof["rule"]
    kl_budget = float(prof.get("kl_budget_nats", 0.230))

    # Clean environment preventing stale R32..R37 debug flags from leaking
    env = {k: v for k, v in os.environ.items() if not k.startswith(("R32_", "R33_", "R34_", "R35_", "R36_", "R37_"))}
    env.update({
        "PYTHONUTF8": "1",
        "R37_GPU_WINDOW": "1",
        "R36_GPU_HANDOVER": "1",
        "R39_RUNTIME": str(PKG_ROOT / "runtime"),
        "R34_RUNTIME": str(PKG_ROOT / "runtime"),
        "R36_FUSE_MHC": "1",
        "R36_FP8_GEMV": "v2",
        "R37_POOL_REG_TOPK": "6",
    })
    py = sys.executable
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for st in stages:
        if st not in ("preflight", "masks") and not args.dry_run and not args.skip_vram_guard:
            gchk = check_gpu_idle()
            if not gchk["ok"]:
                print(f"[ABORT] GPU precheck failed before stage '{st}': {gchk['reason']}", flush=True)
                return 2

        if st == "preflight":
            rc = run_cmd(st, [py, "-u", str(PKG_ROOT / "tests" / "verify_r39_package.py")], env, args.dry_run)
            if rc != 0:
                return rc

        elif st == "selftests":
            for script in (
                PKG_ROOT / "build" / "r39_l27_qtip_2d_ldlq_lab.py",
                PKG_ROOT / "build" / "r39_extract_teacher_top128.py",
                PKG_ROOT / "build" / "run_qtip_build_r39.py",
                PKG_ROOT / "build" / "r39_stage2b_global_kl_trainer.py",
                PKG_ROOT / "runtime" / "glm53_fold_qtip_runtime.py",
            ):
                rc = run_cmd(f"selftest:{script.stem}", [py, "-u", str(script), "--selftest"], env, args.dry_run)
                if rc != 0:
                    return rc

        elif st == "masks":
            scores_pt = HERE / "out_r32" / "r34_prune_scores.pt"
            if scores_pt.exists() or args.dry_run:
                rc = run_cmd(
                    st,
                    [py, "-u", str(HERE / "r38_waterfill_prune_select.py"), "--fold-tcq33-dir", tcq33_dir],
                    env,
                    args.dry_run,
                )
                if rc != 0:
                    return rc
            else:
                print(f"[masks] Using pre-verified shipped mask {mask_path} (no local r34_prune_scores.pt override).", flush=True)

        elif st == "r38_control":
            r38_dst = str(Path(mroot) / "GLM53-APEX-FOLD-TCQ33PW16")
            r38_ft = str(Path(mroot) / "GLM53-APEX-FOLD-TCQ33PW16-FT")
            r38_mask = str(PKG_ROOT / "configs" / "masks" / "tcq33_W16_75GB.json")
            rc = run_cmd(
                "r38_prune_write",
                [py, "-u", str(HERE / "r34_prune_write.py"), "--src", tcq33_dir, "--dst", r38_dst, "--spec", r38_mask, "--tag", "W16"],
                env,
                args.dry_run,
            )
            if rc != 0:
                return rc
            rc = run_cmd(
                "r38_stage2b_ft",
                [
                    py, "-u", str(HERE / "r39_stage2b_global_kl_trainer.py"),
                    "--backbone-dir", bb_r34h,
                    "--archive-dir", r38_dst,
                    "--archive-format", "tcq33",
                    "--teacher-cache", tcache,
                    "--teacher-archive-dir", tcq33_dir,
                    "--export-dir", r38_ft,
                    "--sweeps", str(args.sweeps),
                ],
                env,
                args.dry_run,
            )
            if rc != 0:
                return rc

        elif st == "l27_gate":
            l27_json = OUT_DIR / "r39_l27_qtip_lab.json"
            rc = run_cmd(
                st,
                [
                    py, "-u", str(HERE / "r39_l27_qtip_2d_ldlq_lab.py"),
                    "--layer", "27", "--score-split", "qb", "--cal-split", "ct",
                    "--out", str(l27_json),
                ],
                env,
                args.dry_run,
            )
            if rc != 0:
                return rc
            if not args.dry_run and l27_json.exists():
                rep = json.loads(l27_json.read_text(encoding="utf-8"))
                # Verify positive SNR jump over TCQ33_A_best before committing to 42-layer build
                sub = rep.get("L27_qb_calct_E288") or next(iter(rep.values()), {})
                delta_db = sub.get("QTIP_L16_HYB_2D16_QZ", {}).get("delta_vs_tcq33_db", None)
                if delta_db is not None and float(delta_db) < 0.30:
                    print(f"[GATE FAIL] L27 SNR jump {float(delta_db):+.3f} dB < +0.30 dB minimum gate!", flush=True)
                    return 4

        elif st == "build_r39":
            qtip_rule = "GU125" if rule == "GU125" else "A"
            cmd = [
                py, "-u", str(HERE / "run_qtip_build_r39.py"),
                "--dst", raw_arch,
                "--old", tcq33_dir,
                "--src", fp8_src,
                "--bb", bb_exact,
                "--rule", qtip_rule,
                "--prune-spec", mask_path,
            ]
            if args.resume or (Path(raw_arch) / "_r39_ckpt" / "meta.json").exists():
                cmd.append("--resume")
            rc = run_cmd(st, cmd, env, args.dry_run)
            if rc != 0:
                return rc

        elif st == "teacher_cache":
            if not args.dry_run and os.path.exists(tcache):
                print(f"[teacher_cache] Reusing existing {tcache}", flush=True)
            else:
                rc = run_cmd(
                    st,
                    [
                        py, "-u", str(HERE / "r39_extract_teacher_top128.py"),
                        "--backbone-dir", bb_exact,
                        "--fp8-src-dir", fp8_src,
                        "--tcq33-dir", tcq33_dir,
                        "--out", tcache,
                    ],
                    env,
                    args.dry_run,
                )
                if rc != 0:
                    return rc

        elif st == "stage2b_train":
            rc = run_cmd(
                st,
                [
                    py, "-u", str(HERE / "r39_stage2b_global_kl_trainer.py"),
                    "--backbone-dir", bb_exact,
                    "--archive-dir", raw_arch,
                    "--teacher-cache", tcache,
                    "--export-dir", exp_arch,
                    "--save-state", str(OUT_DIR / f"{args.profile}_stage2b_adapter.pt"),
                    "--sweeps", str(args.sweeps),
                ],
                env,
                args.dry_run,
            )
            if rc != 0:
                return rc

        elif st == "eval_r39":
            eval_out = OUT_DIR / f"quality_eval_{args.profile}.json"
            rc = run_cmd(
                st,
                [
                    py, "-u", str(HERE / "quality_eval_r39.py"),
                    "--backbone", bb_r34h,
                    "--fold", exp_arch,
                    "--exact-backbone", bb_exact,
                    "--kl-budget", str(kl_budget),
                    "--speed-gate",
                    "--out", str(eval_out),
                ],
                env,
                args.dry_run,
            )
            if rc != 0:
                return rc

        elif st == "serve_r39":
            s_cli = prof.get("serve_cli", {})
            cmd = [
                py, "-u", str(PKG_ROOT / "runtime" / "glm53_fold_qtip_runtime.py"),
                "--backbone-dir", bb_r34h,
                "--fold-dir", exp_arch,
                "--expert-pool-gib", str(s_cli.get("expert_pool_gib", 14.0)),
                "--register-gib", str(s_cli.get("register_gib", 48.0)),
                "--graphs",
                "--serve",
                "--port", str(s_cli.get("port", 8088)),
                "--serve-max-context", str(s_cli.get("serve_max_context", 65536)),
                "--serve-cache-entries", str(s_cli.get("serve_cache_entries", 4)),
                "--serve-cache-points", str(s_cli.get("serve_cache_points", 3)),
                "--serve-state-gib", str(s_cli.get("serve_state_gib", 1.5)),
            ]
            rc = run_cmd(st, cmd, env, args.dry_run)
            if rc != 0:
                return rc

    print(f"\n[PIPELINE COMPLETE] Profile '{args.profile}' finished all requested stages ({stages}). Ledger: {OUT_DIR / 'r39_ledger.tsv'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
