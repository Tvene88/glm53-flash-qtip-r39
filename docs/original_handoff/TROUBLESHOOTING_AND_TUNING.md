> **ORIGINAL, UNVERIFIED DOCUMENT.** This file is part of the original handoff package (written by an AI model,
> Gemini 4 Carbon). Its numbers are forecasts, several of which were refuted by measurement. The measured results
> are in [`RESULTS.md`](../../RESULTS.md). Apart from this banner the text is unchanged.

# Troubleshooting, Diagnostics & Hyperparameter Tuning Guide (`1x RTX 5090 + 96 GB DDR5`)

## 1. Windows 11 WDDM vs. WSL2 Host Memory & `cudaHostRegister` Diagnostics

### Symptom A: `registered_gib` is lower than `--register-gib 48.0` or registration takes `> 18s`
* **Root Cause**: On Windows 11 WDDM (`96 GB` DDR5), `cudaHostRegister` pins physical pages inside the process working set (`SetProcessWorkingSetSizeEx`). Browser tabs, Docker Desktop VM balloons, or stale Python processes reduce contiguous free RAM below `register_floor_gib` (`10.0 GiB`).
* **Fix**:
  1. Close unused WSL2/Docker containers (`wsl --shutdown` if running natively on Windows, or cap `.wslconfig` `memory=84GB` if running inside WSL2).
  2. Run `python build_r39/bench_guard_r34.py` to confirm `>= 75 GiB` available host RAM before timed benchmarks.
  3. On first boot after Stage-2B export, `.slab` sidecars are written once (~45s NVMe sequential write); run one warm-up boot before timing startup registration.

### Symptom B: CUDA OOM (`cudaErrorMemoryAllocation`) during 32k–65k Long-Context Serving
* **Root Cause**: Benchmark profile `expert_pool_gib = 18.0–18.5 GiB` leaves `~1.8 GiB` free VRAM on a 32 GiB RTX 5090, which fits `<= 4k` prompts comfortably but crowds one-shot eager prefills above `16k` tokens.
* **Fix**: Use the `r39s_agentic_serve_65k` profile (`--expert-pool-gib 14.0 --register-gib 50.0 --serve-prefill-chunk 2048 --serve-state-device cpu`), which keeps `> 5.8 GiB` free GDDR7 VRAM for long-context absorbed-MLA prefill chunks while serving missed experts from locked DDR5 RAM over PCIe 5.0.

## 2. Stage-2B Global-KL Training Diagnostics (`r39_stage2b_global_kl_trainer.py`)

| Observation in Logs | Cause | Recommended Adjustment |
| :--- | :--- | :--- |
| `peak_vram_gib > 24.0 GiB` | `--micro-bs` or `--window-size` set too high | Keep `--micro-bs 1 --window-size 4` (`~11.8 GiB` peak VRAM). Only raise `--micro-bs 2` if VRAM is confirmed idle. |
| Host RAM pressure during macro-step | `32` seqs x `2048` tok `bf16` checkpoints use `22.0 GiB` DDR5 | Lower `--macro-seqs 16` (`11.0 GiB` DDR5) while keeping `--train-seqs 128` (runs 8 half-size macro-steps per sweep with identical math). |
| `train_KL` drops rapidly (`< 0.16`) while `qbench KL` plateaus after sweep 4 | Mild router LoRA overfitting on 128 calibration sequences | Increase `--train-seqs 192` (re-extract `train_cal_top128.pt` with `--train-seqs 192`), or lower `--lr-router 1e-4 --lora-rank 8`. Note: `r39_stage2b_global_kl_trainer.py` only overwrites `--export-dir` on strict held-out `qbench` KL improvement. |
| `codebook_rel_delta == 0.0` | Wrong `--archive-format` or empty routing | Verify `[STAGE-2B] Archive format='qtip_hyb'` is logged at startup. |
| Router routes to pruned expert ID | External `--prune-spec` mismatched with archive `perm` | Omit `--prune-spec` when `--archive-dir` was already built with `--prune-spec` (the trainer and runtime read `ho.missing_ids` / `qarch.missing` directly from `perm`). |

## 3. Layer 27 Lab & 42-Layer Streaming Encoder Diagnostics

1. **Early Abort (`exit code 2`) at Layer 12 in `run_qtip_build_r39.py`**:
   * Note that the early-abort SNR gate (`--abort-min-gain-db 0.25`) is automatically disabled when `--prune-spec` is active (`not prune_map`), because comparing a 16%-pruned `evn` stream directly against an unpruned 288-expert `evo` `TCQ33` stream conflates pruning perturbation (healed later in Stage-2B) with quantizer fidelity. Unpruned `Rule A` builds enforce the `+0.25 dB` gate automatically.
2. **NVRTC Compilation Failure (`torch.cuda._compile_kernel`)**:
   * Ensure PyTorch 2.6+ built with CUDA 12.8 (`cu128` for Blackwell `sm_120`) is active in your venv (`python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"` -> `(12, 0)`).
   * On Windows, ensure the CUDA Toolkit `nvrtc64_120_0.dll` directory is on `PATH` (standard in PyTorch `cu128` wheels under `site-packages/torch/lib`).
3. **Tuning Kernel Occupancy on Blackwell `sm_120`**:
   * `glm53_fold_qtip_runtime.py` exposes `R39_QTIP_WARPS` (default `8` = `256` threads/CTA -> `4` CTAs/SM at `20 KiB` shared memory) and `R39_QTIP_GX` (default `64`). You can benchmark `R39_QTIP_WARPS=16 R39_QTIP_GX=32` (matching R3.7 `e_sh16`) via `quality_eval_r39.py --qbench-only --speed-gate`; both configurations are IEEE-754 deterministic and bit-exact (`0.0 ULP` delta).
