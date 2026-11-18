> **ORIGINAL, UNVERIFIED DOCUMENT.** This file is part of the original handoff package (written by an AI model,
> Gemini 4 Carbon). Its numbers are forecasts, several of which were refuted by measurement. The measured results
> are in [`RESULTS.md`](../../RESULTS.md). Apart from this banner the text is unchanged.

# Step-by-Step Coding-Agent Handoff Runbook (Round 3.8 & Round 3.9)

This runbook provides exact, sequential instructions for an autonomous coding agent (Claude Code, Codex, or OpenHands) to verify, build, co-adapt, gate-evaluate, and serve Round 3.8 and Round 3.9 `APEX-FOLD` checkpoints for `GLM-5.3-Flash` (`320.8B`) on a single **NVIDIA GeForce RTX 5090 (`32 GiB` GDDR7 VRAM) + `96 GB` DDR5 RAM** workstation.

All commands assume you have unzipped `GLM53_APEX_FOLD_R39_Complete_Agent_Handoff.zip` into a working folder such as `C:\models\GLM53_APEX_FOLD_R39` (Windows PowerShell) or `/mnt/c/models/GLM53_APEX_FOLD_R39` (WSL2 Ubuntu) and `cd`-ed into its root directory.

## Pre-Requisite Host Paths Checklist

Verify the following existing host directories before starting (pass CLI overrides to `run_r39_master_pipeline.py` if your paths differ):

| Asset | Default Windows Path | Default WSL2 Path | Required By |
| :--- | :--- | :--- | :--- |
| Full-head FP8 backbone (`154,880` vocab) | `C:\models\GLM53-backbone` | `/mnt/c/models/GLM53-backbone` | Encoder, Teacher Cache, Stage-2B, Quality Eval |
| Slim `r34h` FP8 backbone (`10.294 GB`) | `C:\models\GLM53-backbone-r34h` | `/mnt/c/models/GLM53-backbone-r34h` | Serving, R3.8 Control, Quality Eval |
| Source 288-expert FP8 shards (`62` files) | `C:\models\GLM-5.3-Flash-UNCENSORED-FP8` | `/mnt/c/models/GLM-5.3-Flash-UNCENSORED-FP8` | `run_qtip_build_r39.py` (or `--fetch shards`), Exact Teacher |
| Existing Round 3.3/3.7 `TCQ33` archive | `C:\models\GLM53-APEX-FOLD-TCQ33` | `/mnt/c/models/GLM53-APEX-FOLD-TCQ33` | Side-by-side `evo` SNR delta, R3.8 Control, `_r33_ckpt/evx.pt` |
| Layer-27 capture & Range Experts (optional) | `C:\models\r33_capture`, `C:\models\GLM53-RANGE-EXPERTS` | `/mnt/c/models/r33_capture` | Phase 2 (`r39_l27_qtip_2d_ldlq_lab.py`) |
| Calibration & `qbench` corpora | `C:\models\calib_text\{exl3,en}` | `/mnt/c/models/calib_text/{exl3,en}` | Calibration, Stage-2B Global-KL, `qbench` Eval |

## Phase 0 — Offline Package Integrity & Mathematical Pre-Flight (`~60s`, No GPU)

Run the 9-gate offline verifier first. This checks AST/bytecode compilation across all 39 Python files, exact slab byte-locks (`6,326,528 B` and `6,064,384 B`), [QTIP (arXiv:2406.11235v4)](https://arxiv.org/html/2406.11235v4) `L=16` collar tail-biting bijection, `512/512` `HYB-mul1` bucket coverage, analytical vs. finite-difference Stage-2B gradients (`< 1e-8`), all 5 non-uniform prune masks, orchestrator `--dry-run`, and both multi-turn HTTP server + `stable_messages` suites:

```bash
python -u tests/verify_r39_package.py
```

* **Pass Criterion**: `[ALL 9 GATES PASSED]` with exit code `0`. Stop immediately if any gate fails.

## Phase 1 — GPU NVRTC Kernel, Encoder, Trainer & CUDAGraph Self-Tests (`~35s` on RTX 5090)

Confirm exclusive GPU access (`nvidia-smi` memory used `<= 4,096 MiB`) and execute all five standalone GPU `--selftest` suites:

```bash
python -u build_r39/run_r39_master_pipeline.py --stages selftests
```

This runs:
1. `python -u build_r39/r39_l27_qtip_2d_ldlq_lab.py --selftest` — compiles NVRTC `L=16` Viterbi + `16x16` GEMV kernels on `sm_120`, verifies `cos > 0.99999`, and proves positive SNR gain from `2D-BlockLDLQ` + Cross-Covariance `H_down`.
2. `python -u build_r39/r39_extract_teacher_top128.py --selftest` — verifies `weights_only=True` Top-128 + residual-tail probability conservation (`max_err < 2e-3`).
3. `python -u build_r39/run_qtip_build_r39.py --selftest` — exercises the 4-stream (`cal`, `evx`, `evn`, `evo`) loop, atomic `.tmp` + `fsync` checkpointing, `--stop-after` + `--resume`, and archive round-trip.
4. `python -u build_r39/r39_stage2b_global_kl_trainer.py --selftest` — runs 4 sweeps of Layer-Major Global-KL across 3 multi-layer windows on a miniature 4-stream Sinkhorn `mHC` MoE model with 25% pruned experts, proving monotonic KL drop, non-zero gradients on all 3 parameter families, and exact `0 B` archive size invariance.
5. `python -u runtime/glm53_fold_qtip_runtime.py --selftest` — verifies `QtipArchive` `.slab` memmap, `cudaHostRegister` DDR5 DMA, prefill `dense_core` vs. `decode_core` parity (`cos > 0.9999`), and **0.0-ULP bit-identical** `torch.cuda.CUDAGraph` replay vs. eager decode.

## Phase 2 — Refresh Non-Uniform LOO Water-Filling Prune Masks (`~2s`, CPU)

Five verified non-uniform prune masks are already pre-built in `configs/masks/`. If `r34_prune_scores.pt` from Round 3.4 is available on your workstation, copy or symlink it to `build_r39/out_r32/r34_prune_scores.pt` and run:

```bash
python -u build_r39/r38_waterfill_prune_select.py \
    --scores-pt build_r39/out_r32/r34_prune_scores.pt \
    --fold-tcq33-dir C:/models/GLM53-APEX-FOLD-TCQ33 \
    --out-dir configs/masks
```

* **Pass Criterion**: `[MASK OK]` printed for all 5 presets (`hyb_P5_83GB.json`, `tcq33_W16_75GB.json`, `hyb_W16_75GB.json`, `hyb_W20_71GB.json`, `hyb_GU125_69GB.json`), zero protected experts touched, early (`L03..L07`) and late (`L40..L44`) layers pruned less heavily than middle layers (`L12..L35`).

## Phase 3 — Build Reusable Compact Teacher Cache (`train_cal_top128.pt`, `~6–18 min`)

Build the reusable `~92 MiB` Top-128 + Residual-Tail teacher target cache once; it is shared across BOTH Round 3.8 and Round 3.9 Stage-2B Global-KL runs:

```bash
python -u build_r39/r39_extract_teacher_top128.py \
    --backbone-dir C:/models/GLM53-backbone \
    --fp8-src-dir C:/models/GLM-5.3-Flash-UNCENSORED-FP8 \
    --tcq33-dir C:/models/GLM53-APEX-FOLD-TCQ33 \
    --train-seqs 128 --seq-len 2048 \
    --out C:/models/r39_teacher_top128/train_cal_top128.pt
```

* **Note**: If raw 328 GB FP8 shards are not locally staged, pass `--teacher-mode unpruned_tcq33` (`~6 min`), which uses unpruned `GLM53-APEX-FOLD-TCQ33` for `cal` sequences and exact FP8 `_r33_ckpt/evx.pt` for `qbench`.
* **Resource Footprint**: Peak VRAM `~9.4 GiB`, host DDR5 `~8.5 GiB` (`bf16` hidden stream buffer).

## Phase 4 — Optional Fast Round 3.8 Control (`75.00 GB` Frozen-`TCQ33` + Stage-2B, `~45 min`)

Before encoding all 42 layers in QTIP HYB, you can materialize and validate the Round 3.8 control (`74.995 GB` at `<= 0.260 nats` KL) without re-quantizing any expert bitstreams:

1. **Compact `TCQ33` slabs via zero-GPU sequential NVMe copy (`~8 min`)**:
   ```bash
   python -u build_r39/r34_prune_write.py \
       --src C:/models/GLM53-APEX-FOLD-TCQ33 \
       --dst C:/models/GLM53-APEX-FOLD-TCQ33PW16 \
       --spec configs/masks/tcq33_W16_75GB.json \
       --tag W16
   ```
2. **Run Stage-2B Global-KL co-adaptation on `TCQ33PW16` (`~35 min`, Peak VRAM `~11.8 GiB`)**:
   ```bash
   python -u build_r39/r39_stage2b_global_kl_trainer.py \
       --backbone-dir C:/models/GLM53-backbone \
       --archive-dir C:/models/GLM53-APEX-FOLD-TCQ33PW16 \
       --archive-format tcq33 \
       --teacher-cache C:/models/r39_teacher_top128/train_cal_top128.pt \
       --export-dir C:/models/GLM53-APEX-FOLD-TCQ33PW16-FT \
       --sweeps 6 --macro-seqs 32 --micro-bs 1
   ```

## Phase 5 — Single-Layer L27 7-Rung SNR Gate (`~12 min` for 288 experts, or `~45s` for 32-expert smoke)

Verify the [QTIP (arXiv:2406.11235v4)](https://arxiv.org/html/2406.11235v4) `L=16` `HYB-mul1` + `16x16` `2D-BlockLDLQ` + Cross-Covariance `H_down` SNR jump on Layer 27 against the recorded Round 3.3 `TCQ33` baseline (`9.187 dB` on `27_qb_calct`):

```bash
# Optional 32-expert rapid smoke check (~45s):
python -u build_r39/r39_l27_qtip_2d_ldlq_lab.py --layer 27 --score-split qb --cal-split ct --max-experts 32

# Full 288-expert Layer 27 gate (~12 min):
python -u build_r39/r39_l27_qtip_2d_ldlq_lab.py \
    --layer 27 --score-split qb --cal-split ct \
    --out build_r39/out_r39/r39_l27_qtip_lab.json
```

* **Mandatory Gate Thresholds**:
  * `TCQ33_A_best.content_snr_db` reproduces `~9.187 dB` (`+-0.02 dB`).
  * `QTIP_L16_HYB_2D16_QZ.delta_vs_tcq33_db >= +0.30 dB` (expected `+0.45` to `+0.75 dB`, with `down_proj` weight/output SNR strictly beating `QTIP_L16_HYB_2D16`, confirming Cross-Covariance `W_d*` fixes the Round 3.3 `qz` regression).
  * `QTIP_L16_HYB_2D16_QZ.nvrtc_gemv_check.cos >= 0.99999`.
  * For Sub-70 GB builds, `QTIP_Sub70_GU125_2D16_QZ_Tune.content_snr_db` beats Round 3.3 `L190_best` (`7.977 dB`) by `>= +0.60 dB`.

## Phase 6 — Full 42-Layer + MTP Sequential Streaming QTIP HYB Build (`~2.6–3.1 hours`)

Choose your target operating point (`Flagship 74.66 GB / <=0.23 KL` OR `Sub-70 GB 69.97 GB / <=0.253 KL`) and launch the sequential streaming builder:

* **Option A — Flagship `74.66 GB` (`<= 0.228 KL` target, `Rule A` + `hyb_W16_75GB.json`)**:
  ```bash
  python -u build_r39/run_qtip_build_r39.py \
      --dst C:/models/GLM53-APEX-FOLD-R39-HYB-W16 \
      --old C:/models/GLM53-APEX-FOLD-TCQ33 \
      --src C:/models/GLM-5.3-Flash-UNCENSORED-FP8 \
      --bb  C:/models/GLM53-backbone \
      --rule A \
      --prune-spec configs/masks/hyb_W16_75GB.json \
      --keep-src
  ```
* **Option B — Sub-70 GB `69.97 GB` (`<= 0.253 KL` target, `Rule GU125` + `hyb_GU125_69GB.json`; or pass `hyb_W20_71GB.json` with `--rule GU125` for `68.90 GB`)**:
  ```bash
  python -u build_r39/run_qtip_build_r39.py \
      --dst C:/models/GLM53-APEX-FOLD-R39-SUB70 \
      --old C:/models/GLM53-APEX-FOLD-TCQ33 \
      --src C:/models/GLM-5.3-Flash-UNCENSORED-FP8 \
      --bb  C:/models/GLM53-backbone \
      --rule GU125 \
      --prune-spec configs/masks/hyb_GU125_69GB.json \
      --keep-src
  ```

* **Crash / Window Pause & Resume**:
  * To pause cleanly after a specific layer (e.g., Layer 24) and release the GPU (`exit code 3`), pass `--stop-after 24`.
  * To resume after any pause or power interruption, re-run the exact same command with `--resume` appended.
* **Built-In Quality Gates**:
  * Every layer asserts `nvrtc_gemv_cos > 0.99999` on real calibration activations before advancing `cal` and `evn`.
  * At completion, writes `build_r39/out_r32/r39_qtip_build_eval.json` (reporting pre-Stage-2B `kl_evn_vs_exact`, `top1_evn_vs_exact`, and Layer 45 `mtp` acceptance rate) plus `_r39_ckpt/qbench_teacher_top128.pt`.

## Phase 7 — Stage-2B Layer-Major Global-KL Co-Adaptation (`~45–65 min` for 6 Sweeps)

Jointly optimize `AnchoredRouterLoRA` (`rank=16`), `e_score_correction_bias`, layer-shared `2 KiB` `HYB` FP16 LUTs, and in-place per-expert FP16 row scales (`off_scale`) against full-model teacher KL ([Tetra, arXiv:2609.35465](https://arxiv.org/pdf/2609.35465)):

```bash
python -u build_r39/r39_stage2b_global_kl_trainer.py \
    --backbone-dir C:/models/GLM53-backbone \
    --archive-dir C:/models/GLM53-APEX-FOLD-R39-HYB-W16 \
    --teacher-cache C:/models/r39_teacher_top128/train_cal_top128.pt \
    --export-dir C:/models/GLM53-APEX-FOLD-R39-HYB-W16-FT \
    --save-state build_r39/out_r39/r39_w16_stage2b_state.pt \
    --sweeps 6 --train-seqs 128 --macro-seqs 32 --micro-bs 1 --window-size 4 --lora-rank 16
```

* **Memory Profile on 1x RTX 5090 + 96 GB DDR5**:
  * Host DDR5: `22.0 GiB` for 11 boundary `bf16` checkpoints (`32 x 2048 x 4 x 4096`) + `2.0 GiB` `grad_h` + `~0.5 GiB` `HostPagedScaleAdamW` state (`< 32 GiB` total host footprint, leaving `> 60 GB` DDR5 free).
  * GPU GDDR7 VRAM: `~6.4 GiB` active 4-layer compressed slabs + `~0.9 GiB` active backbone layers + `~3.8 GiB` `micro_bs=1` autograd activations = **`~11.8 GiB / 32.0 GiB` peak VRAM**.
* **Pass Criterion**:
  * Held-out `[EVAL sw] qbench KL` decreases monotonically across early sweeps and exports improved self-contained checkpoints (`layer_03..45_qtip_hyb.pt`, `merged_routers_bf16.pt`, `fold_manifest.json`, `exact_prefix_cache_prefix*.pt`, `super_experts/`) into `--export-dir` only when held-out `qbench` KL improves (`[NEW BEST]`).
  * Exported `total_archive_bytes` matches input byte-for-byte (`0 B` overhead).

## Phase 8 — Official `chunk=1` `GraphDecoder` Quality Gate, Speed Proof & Live Serving

1. **Fast `qbench`-Only Gate (`29` rows via captured `N=1` CUDA graphs, `~4 min`)**:
   ```bash
   R37_GPU_WINDOW=1 R36_FUSE_MHC=1 R36_FP8_GEMV=v2 R37_POOL_REG_TOPK=6 \
   python -u build_r39/quality_eval_r39.py \
       --backbone C:/models/GLM53-backbone-r34h \
       --fold C:/models/GLM53-APEX-FOLD-R39-HYB-W16-FT \
       --exact-backbone C:/models/GLM53-backbone \
       --qbench-only --speed-gate --kl-budget 0.230 \
       --out build_r39/out_r39/eval_qbench_w16_ft.json
   ```
2. **Full 111-Row Quality & 3-Round Alternating Speed Proof (`29 qbench + 40 devt + 42 devx`)**:
   ```bash
   R37_GPU_WINDOW=1 R36_FUSE_MHC=1 R36_FP8_GEMV=v2 R37_POOL_REG_TOPK=6 \
   python -u build_r39/quality_eval_r39.py \
       --backbone C:/models/GLM53-backbone-r34h \
       --fold C:/models/GLM53-APEX-FOLD-R39-HYB-W16-FT \
       --exact-backbone C:/models/GLM53-backbone \
       --speed-gate --kl-budget 0.230 \
       --out build_r39/out_r39/eval_full111_w16_ft.json
   ```
   * **Official Acceptance Gates**:
     * Flagship `74.66 GB` (`W16`): `qbench.kl <= 0.230 nats`, `qbench.top1 >= 0.846`, `speed_gate.pass_18_tok_s_gate == true` (`>= 18.0 tok/s` across `code`, `capital`, and overall mean; expected `21.5–23.5 tok/s`).
     * Sub-70 GB `69.97 GB` (`GU125`): `qbench.kl <= 0.260 nats`, `speed_gate.pass_18_tok_s_gate == true`.
3. **Launch Multi-Turn OpenAI HTTP Server (`localhost:8088`, KV State Reuse + `stable_messages` + 65k Context)**:
   ```bash
   R36_FUSE_MHC=1 R36_FP8_GEMV=v2 R37_POOL_REG_TOPK=6 \
   python -u runtime/glm53_fold_qtip_runtime.py \
       --backbone-dir C:/models/GLM53-backbone-r34h \
       --fold-dir C:/models/GLM53-APEX-FOLD-R39-HYB-W16-FT \
       --expert-pool-gib 14.0 --register-gib 50.0 \
       --graphs --serve --port 8088 \
       --serve-max-context 65536 --serve-cache-entries 4 --serve-cache-points 3 --serve-state-gib 1.5
   ```
   * Note: On first load of a newly exported directory, `QtipArchive` materializes raw `.slab` sidecar files alongside `layer_XX_qtip_hyb.pt`; every subsequent server boot memory-maps all 42 layers in `< 0.3s` and locks up to `50 GiB` in physical DDR5 RAM via `cudaHostRegister` for direct PCIe DMA.
