#!/usr/bin/env bash
# run_r39_wsl2.sh — WSL2 Ubuntu Launcher for APEX-FOLD Round 3.9 (1x RTX 5090 + 96 GB DDR5)
set -euo pipefail

PROFILE="${1:-r39_flagship_75gb_sub023}"
STAGES="${2:-default}"
MODELS_ROOT="${MODELS_ROOT:-/mnt/c/models}"
PYTHON_EXE="${PYTHON_EXE:-python3}"
EXTRA_ARGS=("${@:3}")

export PYTHONUTF8=1
export R37_GPU_WINDOW=1
export R36_GPU_HANDOVER=1
export R36_FUSE_MHC=1
export R36_FP8_GEMV=v2
export R37_POOL_REG_TOPK=6

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$PYTHON_EXE" -u "$SCRIPT_DIR/run_r39_master_pipeline.py" \
    --profile "$PROFILE" \
    --stages "$STAGES" \
    --models-root "$MODELS_ROOT" \
    "${EXTRA_ARGS[@]}"
