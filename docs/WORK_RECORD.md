# Round 3.9 (QTIP HYB) — working record

> **Release copy.** This is the working record of the project, edited for publication. Machine-specific paths are
> replaced by placeholders (`<PKG_ROOT>`, `<PYTHON>`). Passages about the administration of the owner's machine
> (other local services, container and disk clean-up, one interrupted session) were removed, and the owner's
> messages are paraphrased instead of quoted. No measurement, fix or result was changed. "I" / "me" is the
> engineering agent (Claude) that did the work. Folder names are those of the working tree:
> `claude_r39_qtip\build_r39` is `build/` in this repository, `claude_r39_speed` is `runtime/`,
> `claude_r39_qtip\runtime` is the ORIGINAL package's runtime (not in this repository), and the files under
> `out_r39\`, `out_r32\` and `out_r39s\` that are published are under `results/` (see `results/README.md`).

Working tree: this folder, a copy of `..\GLM53_APEX_FOLD_R39_Complete_Agent_Handoff` (the handoff package is kept
unmodified; its MANIFEST.sha256 verified clean on 2026-11-16). The owner's request (2026-11-16): implement the
package, follow its instructions, run tests to confirm or refute the claimed improvement.

Python: `<PYTHON>`
Logs and ledger: `build_r39\out_r39\`.

## Timeline

### Session 1 (2026-11-16 20:04-20:19, interrupted)
- Phase 0 (offline, 9 gates): reported passed.
- Phase 1 (5 GPU self-tests): FAILED as shipped. Both Viterbi encoder kernels launch 1,024 threads per block but
  compiled to 128 and 222 registers per thread (limit 64): "too many resources requested for launch".
  **Fix 1**: `--maxrregcount=64` in `_compile_nvrtc` of `build_r39\r39_l27_qtip_2d_ldlq_lab.py` and
  `build_r39\run_qtip_build_r39.py` (env `R39_MAXRREG`). After it: 5 of 5 pass (`out_r39\phase1_selftests_run2.log`).
- Layer-27 expert weights fetched by range requests: `C:\models\GLM53-RANGE-EXPERTS\layer_27_experts.safetensors`
  (6.75 GiB, 5.2 min).
- Phase 5 smoke (32 experts) started 20:16. Reported before the session ended (`out_r39\phase5_l27_smoke32.log`):
  `TCQ33_A_best` 8.839 dB; `QTIP_L16_mul1_1D` 7.927 dB (-0.912 dB against TCQ33).
- The session ended here, in the lab's next arm; the smoke gate was repeated in session 2.

### Session 2 (2026-11-16 from 20:27)
- Start state: GPU 62 MiB in use, RAM 111.8 GiB free.
- Phase 0 re-run on this tree (`out_r39\phase0_verify_s2.log`): gates 1-8 pass; gate 9 (manifest) fails only on the
  files that carry the fixes below. The unmodified handoff package hashes clean against its manifest.
- GPU power cap set to 450 W for the encoder runs (`nvidia-smi -pl 450`; default 600; resets at reboot, or
  `nvidia-smi -pl 600`). Telemetry: `build_r39\gpu_telemetry_r39.py` -> `out_r39\gpu_telemetry_s2.csv`.
- Smoke gate, 32 experts, second attempt 20:33-20:36 (`out_r39\phase5_l27_smoke32_s2.log`): it passed the arm that
  was running when session 1 was interrupted (HYB_1D) and encoded the first 2D arm; GPU drew up to 462 W under the
  cap, 67 C.
  `TCQ33_A_best` 8.839 dB, `QTIP_L16_mul1_1D` 7.927 (-0.912), `QTIP_L16_HYB_1D` 8.019 (-0.820).
  Then it failed on a package defect. **Fix 2**: `X_ev[mask_ev]` indexed a CPU tensor with a CUDA mask in
  `r39_l27_qtip_2d_ldlq_lab.py` (the NVRTC GEMV check of every 2D arm); now `mask_ev.cpu()`.
- Smoke gate third attempt (2D arms only), 20:37-20:42, clean (`out_r39\phase5_l27_smoke32_s2b.log`), 32 experts,
  baseline 8.839 dB: `HYB_2D16` 9.259 (+0.420), `HYB_2D16_QZ` 9.244 (+0.405), `HYB_2D16_QZ_Tune` 9.261 (+0.422),
  `Sub70_GU125_2D16_QZ_Tune` 8.522 (-0.317, 1.9278 bpw). NVRTC GEMV cos 1.0 in all. The cross-covariance arm (QZ) is
  0.015 dB BELOW the plain 2D arm in routed output (down weight SNR +0.023, down H-weighted SNR -0.085).
- Full gate, 288 experts, first attempt 20:43-21:08 (`out_r39\phase5_l27_full288.log`): `TCQ33_A_best` 9.188 dB
  (record 9.187: reproduced). Then killed by me: the lab decodes all 288 experts of a projection on the GPU at once
  (60-70 GiB of temporaries; six such decodes in the tune step). Windows spilled 89.7 GiB into shared GPU memory, the
  process committed 140 GiB and free RAM fell to 0.7 GiB. On the 96 GB the package names it cannot run at all.
  **Fix 3** (`r39_l27_qtip_2d_ldlq_lab.py`): decode in chunks of 16 experts to a CPU fp32 tensor (`R39_DECODE_CHUNK`),
  tune step decodes one expert at a time and moves only its 64 reference experts to the GPU, `--baseline-snr-db`.
  Arithmetic unchanged: the 32-expert smoke run reproduces 9.259 / 9.261 dB and every sub-figure exactly
  (`out_r39\phase5_l27_smoke32_fix3check.log`), GPU peak 22.9 GiB, RAM free >= 100 GiB.
  The builder `run_qtip_build_r39.py` has the same all-experts decode (`decode_projection_qtip_2d`, about 240 kept
  experts per layer, results held on the GPU): it needs the same fix before any 42-layer build. NOT yet applied.
- Full gate second attempt (2D arms, baseline 9.188 passed in): `out_r39\phase5_l27_full288_b.log`.
- **Full Layer-27 gate, 288 experts, 21:12-21:37** (`out_r39\phase5_l27_full288_b.log`, `out_r39\r39_l27_qtip_lab.json`),
  baseline `TCQ33_A_best` 9.188 dB, GPU peak 32.1 GiB, RAM free >= 60.7 GiB, 450 W cap:
  | variant | content SNR dB | vs TCQ33 | weight SNR g/u/d | H-weighted SNR g/u/d |
  |---|---|---|---|---|
  | QTIP_L16_HYB_2D16 | 9.616 | +0.428 | 10.597 / 10.568 / 11.396 | 14.335 / 13.665 / 12.145 |
  | QTIP_L16_HYB_2D16_QZ | 9.581 | +0.393 | 10.597 / 10.568 / 11.419 | 14.335 / 13.665 / 12.000 |
  | QTIP_L16_HYB_2D16_QZ_Tune | 9.597 | +0.409 | (215 experts tuned, val rel err 0.07592 -> 0.07311) | |
  | QTIP_Sub70_GU125_2D16_QZ_Tune (1.9278 bpw) | 8.882 | -0.306 | 9.139 / 9.069 / 11.419 | |
  Gate 3 thresholds: QZ >= +0.30 dB: MET (+0.393; forecast was +0.45 to +0.75: not reached). NVRTC GEMV cos 1.0: MET.
  Sub-70 >= +0.60 over R3.3 `L190_best` 7.977: MET (+0.905). The claim that the cross-covariance target makes
  `down_proj` output strictly better than the plain 2D arm is REFUTED at Layer 27: -0.035 dB routed output, -0.145 dB
  H-weighted down SNR, norm ratio 0.982 -> 0.968 (only the unweighted down weight SNR rises, +0.023 dB).
  The 1D ablation arms were only run at 32 experts (-0.82 / -0.91 dB); not gates.
- Phase 1 re-run on the patched scripts: 5 of 5 pass (`out_r39\phase1_selftests_run3_after_fix3.log`).
- **Builder addition**: `--teacher-out` (+ `--teacher-train-seqs`, `--teacher-group`). The FP8 shards exist on disk only
  while their layer is built, so an exact stream `trx` over the trainer's 128 train sequences is advanced inside the
  build (`advance_eval_grouped`) and written at the end as `train_cal_top128.pt` (`teacher_mode: exact_fp8_inbuild`);
  checkpointed as `_r39_ckpt\trx.pt` with the other streams. Replaces Phase 3, which needs all 328 GB locally.
- **Phase 6 build started 21:50**: flagship, Rule A, `hyb_W16_75GB.json`, `--fetch shards` (no `--keep-src`: shards are
  streamed and evicted as in Round 3.3), `--old C:\models\GLM53-APEX-FOLD-TCQ33P5` (the TCQ33 layers are gone; this gives
  a same-harness per-layer and final-KL comparison against the shipped archive; `--replace-old` is NEVER passed),
  `--dst C:\models\GLM53-APEX-FOLD-R39-HYB-W16`. First leg `--stop-after 4` to validate two real layers
  (`out_r39\phase6_build_w16_part1.log`, builder log `build_r39\out_r32\r39_qtip_build.log`). Continue with the same
  command plus `--resume` (drop `--stop-after`).
- Build leg 1 (21:41-22:06, layers 3 and 4, stopped cleanly with exit 3): no defect. Per layer: encode 284 / 327 s,
  streams 37-38 s (teacher stream 17-19 s), NVRTC GEMV cos 1.0, GPU peak 25.1 GiB, 458 W max, RAM free min 17.2 GiB.
  Routed-output SNR against the exact unpruned teacher, new (QTIP, W16 mask) vs old (shipped TCQ33P5):
  L03 15.517 vs 16.092 dB (261 experts kept), L04 8.187 vs 9.684 dB (238 kept). Before tuning the new archive is
  below the shipped one per layer; it prunes 27 and 50 experts where P5 prunes about 14.
  Checkpoint took 440 s. **Fix 4**: stream checkpoints written with the legacy torch container (27 -> 249 MB/s for
  the same object; measured with `ckpt_speed.py`, zip 317 s vs legacy 34.5 s for the 8.6 GB teacher stream).
- Disk for the build: dst 66 GB + stream checkpoints 12.5 GB (+ a temporary copy while writing) + shards kept for
  later layers (evicted under pressure). Freed: `C:\models\GLM53-RANGE-EXPERTS\layer_27_experts.safetensors`
  (re-downloadable with `build_run\fetch_layer_experts_r33.py --layers 27`).
- Build leg 2 (`--resume`) started 22:17 (`out_r39\phase6_build_w16_part2.log`); checkpoint now 25 s. Per layer about
  5-7 min. L05 7.263 vs 8.592 dB, L06 8.139 vs 9.342 dB (new vs shipped P5, before tuning).
- Build leg 2 ran 22:17-23:21 through L14 (12 trunk layers done). New-minus-shipped routed SNR per layer, before
  tuning: -0.575, -1.497, -1.329, -1.203, -0.753, -1.155, -0.995, -0.811, -0.667, -0.896, -0.803, -0.830 dB
  (L03..L14, mean -0.96 dB): every layer is below the shipped TCQ33P5 archive.
  Host memory: the builder process sits at about 106 GiB private between layers (peak 124 GiB with the GPU share),
  5-12 GiB RAM free, stable over L11-L14 (`out_r39\builder_mem.csv`); I can account for about 35 GiB of it and did not
  find the rest. On 96 GB the build could not run. Disk dips by 12.5 GiB during each checkpoint (22.4 GiB free at
  L14). Stopped by me right after the L14 checkpoint and resumed 23:22 as leg 3 with `--shard-margin-gib 28
  --floor-gib 14` (`out_r39\phase6_build_w16_part3.log`) so a checkpoint cannot meet a nearly full disk.
- `quality_eval_r39.py` (shipped copy kept as `out_r39\quality_eval_r39.py.shipped`): `--speed-only`, prefix cache OFF
  for both runners unless `--prefix-cache`, teacher head freed before the timed decode, VRAM-free and env recorded.
- **Stage-2B trainer reviewed before its first real run** (independent reviewer, read-only; shipped file kept as
  `out_r39\r39_stage2b_global_kl_trainer.py.shipped`). Confirmed in the code and fixed:
  **Fix 5**: `HostPagedScaleAdamW.s0` aliased the archive's scale tensor, so every step compounded the scales.
  **Fix 6**: the HYB LUT was clamped to >= 1e-4 after every step although the real LUTs hold negative centroids
  (25 of 1,024 at k=2 in layer 03): the first step would have rewritten those weights. Clamp removed.
  **Fix 7**: train/qbench ids are taken from / checked against the teacher cache; the qbench metric is also reported
  over real positions only (17 of 29 rows are shorter than 2,048 and were padded: 32.5 % of scored positions).
  Additions: `decode_all_dense_bf16` + `moe_forward_dense_nograd` (the no-grad passes decode a layer once instead of
  once per sequence; same decode routine and bf16 matmuls; `--no-fast-forward` restores the shipped path),
  `--eval-every-macro`, `--max-macro-steps`, per-phase time and VRAM line `[PHASES]`.
  Reviewer's estimates, NOT yet measured: reverse-window pass needs 40-55 GiB VRAM at `--window-size 4`; the shipped
  run would take 35-40 h for 6 sweeps (documented: 45-65 min); the export writes all 66 GB at every new best and
  nothing can resume from the saved state; host RAM about 94 GiB (the whole archive stays in RAM).
- **Evaluation script + QTIP runtime reviewed before their first real run** (second independent reviewer, read-only).
  Findings to act on when Phase 8 runs (none applied yet):
  1. The R3.9 package's copies of the shared runtime modules (`glm53_fold_backbone/generate/graphs/kernels/
     tcq_runtime/expert_pool_v6.py`) are OLDER than the production Round 3.7/3.8 tree: no fused mHC, no FP8 GEMV v2,
     no `e_sh16` TCQ kernel, no long-context / 8-bit-history modules (so no 524K window). `R36_FUSE_MHC`,
     `R36_FP8_GEMV` in the runbook's command are read by nothing. A QTIP archive can only be served by this older
     runtime until the QTIP layer runtime is ported into `claude_r38_ctx`.
  2. `--runner tcq` in this tree runs the baseline with pool v2 and nothing locked (R3.5-level speed, about 13.7 tok/s)
     unless `R32_POOL=6 R33_REGISTER_GIB=.. R33_REGISTER_MODE=rows` are set; production speed (20-21.6 tok/s) is only
     measurable in the R3.8 tree.
  3. `R37_POOL_REG_TOPK=6` (in the runbook's command) makes the QTIP pool drop 2 of 8 predicted experts per layer by
     expert id: unset it for QTIP runs.
  4. Disk: `QtipArchive` writes a `.slab` sidecar per layer (64-66 GB per archive) and still needs the `.pt` at every
     start (it `torch.load`s all 64 GB each time; the documented "< 0.3 s" boot cannot hold). An archive therefore
     occupies about 130 GB on disk to be served.
  5. VRAM at `--pool-gib 18` with the teacher head loaded is probably over the edge (use 16 for the quality pass).
  6. A qbench-only run is 40,132 graph steps: 35-50 min, not the runbook's "~4 min".
  7. The KL metric of `quality_eval_r39.py` is sound and comparable between the two runners; it is not the same number
     as the build's `kl_evn_vs_exact` (different head precision and positions).
  8. Speed-gate rule is stricter than Round 3.7's (code and capital each >= 18); prefix cache on for QTIP only.
- Fix 3 also applied to the builder: `decode_projection_qtip_2d_chunked` used for `Gh_k` / `Uh_k` in
  `run_qtip_build_r39.py` (compiles; not yet run on a real layer).

### Session 3 (2026-11-17, after a context reset; same machine state)
- **Phase 6 build COMPLETE 02:32** (leg 3 23:22-02:32, 184.6 min; all three legs about 4.6 h wall including the
  shard downloads; documented 2.6-3.1 h). `complete: true`, 43 layer files (L03..L45), NVRTC GEMV cos 1.0 on every
  layer, exit 0. Report `build_r39\out_r32\r39_qtip_build_eval.json`.
  | quantity (qbench, 39,871 positions, vs exact FP8 teacher) | new QTIP W16, untuned | shipped TCQ33P5 |
  |---|---|---|
  | KL nats | **0.29438** | **0.26038** (recorded 0.2613) |
  | top-1 agreement | 0.8380 | 0.8442 |
  | MTP L45 draft acceptance (exact 0.7238) | 0.7158 | 0.7228 |
  | expert layer files on disk | 66.19 GB | 75.02 GB |
  Size claim (8.8 GB smaller than P5) CONFIRMED. Before tuning the new archive is WORSE than the shipped one
  (+0.034 nats KL, -0.6 pt top-1, -0.7 pt MTP acceptance); the target KL <= 0.228 needs Stage 2B to recover 0.066.
  New-minus-shipped routed SNR per layer L03..L44: mean -0.548 dB; L03-L24 between -0.21 and -1.50, L25-L37 between
  -0.12 and -0.42, L38-L44 positive (+0.03 to +0.44) (the mask prunes most in the early layers).
- Teacher cache written by the build: `C:\models\r39_teacher_top128\train_cal_top128.pt` (239 MiB, exact FP8).
  Stream checkpoints still in `--dst\_r39_ckpt` (11.7 GiB; `evx.pt` is needed by `quality_eval_r39.py`).
- Patched trainer `--selftest` passes (`out_r39\phase7_trainer_selftest_patched.log`): 0.18197 -> 0.16520, same
  initial KL as the shipped script.
- **Stage 2B started 02:34** on the untuned archive, before the Phase 8 evaluations (the claim depends on it; the
  build already gives the untuned comparison in one harness): shipped settings (`--sweeps 6 --train-seqs 128
  --macro-seqs 32 --micro-bs 1 --window-size 4 --lora-rank 16`) plus `--eval-every-macro 1`, NO `--export-dir`,
  state to `out_r39\r39_w16_stage2b_state.pt`, log `out_r39\phase7_stage2b_probe.log`, memory `out_r39\trainer_mem.csv`.
  Code check before launch: `CompressedMoERoutedFunction` decodes 16 experts at a time in forward AND backward and
  keeps no dense weights, so the reviewer's 40-55 GiB VRAM estimate for the reverse pass may be too high; time per
  sequence in that pass is the open figure. Watch the first `[PHASES]` line.

- First Stage 2B start (02:34-02:41, `out_r39\phase7_stage2b_probe.log`): archive loaded in about 90 s (64 GiB
  private), starting evaluation 171 s, 26.8 GiB VRAM, about 100 GiB private. **Starting score: shipped metric 3.837
  nats / top-1 0.597; real positions only 0.29058 nats / top-1 0.8371 over 39,871 tokens** (build: 0.29438 / 0.8380,
  so the trainer's forward agrees with the build). The shipped metric is dominated by the 19,231 padded positions
  (about 11 nats each) and the shipped script picks its best state on it. **Fix 8**: best state chosen on the
  real-position KL (`_sel`). Stopped by me and restarted 02:43 with fix 8 (`out_r39\phase7_stage2b_probe_b.log`).
- Trainer additions (compile; not yet run): `--load-state` + `--export-only` (`load_adapter_state`), so a saved state
  can be exported by a later process. To check before any export is served: the export writes only the tensor
  entries of `fields` and a fixed key set; compare against an original layer file's keys.

- **Stage 2B as shipped cannot run on this machine (measured 02:39-03:32)**, logs `out_r39\phase7_stage2b_probe_b.log`,
  `phase7_stage2b_diag4.log`, `phase7_stage2b_diag4_w1.log`, memory `out_r39\trainer_mem.csv`, `gpu_telemetry_s2.csv`:
  - Shipped settings (`--window-size 4 --macro-seqs 32`): the first step drove RAM free to 2.7 GiB with the GPU at
    32.1 GiB; stopped by me.
  - 4 sequences, `--window-size 4`: forward pass fine (65 s, 20.5 GiB), the reverse pass of the first 4-layer window
    did not finish one sequence in 5 min (VRAM 32.1 GiB + about 50 GiB spilled to shared memory); watchdog killed it at
    1.3 GiB RAM free. The reviewer's 40-55 GiB estimate holds.
  - 4 sequences, `--window-size 1`: works. Reverse pass 477 s for 4 sequences x 45 layers = **2.65 s per sequence
    per layer**, reserved VRAM about 26 GiB for a 1-layer window (the folded 2-layer tail window L43..44: 30.7 GiB
    allocated, 33 reserved, 97 s instead of about 20). Checkpoints 64 MiB per sequence per window (h is
    [1, 2048, 4, 4096] bf16): 32 sequences at 1-layer windows = 88 GiB of host RAM on top of the 82 GiB base
    (archive 64 GiB + backbone). After the step the evaluation spilled (VRAM 31.7 GiB, RAM free 21.9 -> 1.2 GiB,
    killed): allocator cache, see fix 10.
  - Projection from these figures: one 32-sequence step = 32 x 45 x 2.65 s = 64 min reverse + about 6 min forward;
    24 steps (6 sweeps) = about 28 h plus evaluations. Documented: 45-65 min for the whole stage, on 96 GB.
- **Fix 9** (`run_macro_step` rewritten; copy before it `out_r39\r39_stage2b_global_kl_trainer.py.before_fix9`):
  `--grad-window N` = layers per autograd window (memory only; 1 is the only value that fits 32 GiB), `--chunk-seqs N`
  = sequences per forward/head/reverse pass; gradients of all chunks accumulate and the optimizers step once at the
  end in the shipped optimizer windows (`--window-size`, same groups, same per-window clip). In the shipped code no
  later computation of the macro-step read a stepped parameter, so this is the same update. Defaults (0, 0) keep the
  shipped pass order. Self-test in 5 variants, all pass (`out_r39\phase7_trainer_selftest_fix9.log`):
  (gw 0, chunk 0) 0.18197 -> 0.16504; (1, 0) 0.16492; (1, 2) 0.16453; (0, 2) 0.16525; (1, 1) 0.16514
  (before fix 9: 0.16520; spread is run-to-run level, not separately measured).
- **Fix 10**: `torch.cuda.empty_cache()` after each layer's dense decode / forward in the no-grad passes and at the
  start of an evaluation (dense blocks differ in size per layer, the allocator reserved 27 GiB for 20.5 GiB in use).
- Added `_mem_line` (host + GPU memory) on phase, window, chunk and evaluation lines.

- Diagnostic with fixes 9 + 10 (2 sequences, chunk 1, grad window 1; `out_r39\phase7_stage2b_diag2_chunk.log`):
  runs clean, RAM free 32 GiB throughout, evaluation after the step 161 s with reserved VRAM about 5 GiB between
  layers. Reverse pass 122-125 s per sequence (45 layers). VRAM peak per 1-layer window: 26.0 GiB on three of every
  four MoE layers, 14.5 GiB on L3, 7, 11, ... 43, 18.4 GiB on the dense layers L0-2 (so the attention graph, not the
  experts, holds the memory; the time is in the expert decode: L0-2 take about 0.5 s, MoE layers about 2.9 s).
  One step on 2 sequences: real-position KL 0.29058 -> 0.29053, top-1 0.8371 -> 0.8379.
- **Stage 2B run 1 started 03:50** (`out_r39\phase7_stage2b_run1.log`, watcher `out_r39\watch_stage2b.ps1`):
  `--sweeps 6 --train-seqs 128 --macro-seqs 32 --chunk-seqs 8 --grad-window 1 --micro-bs 1 --window-size 4
  --lora-rank 16 --eval-every-macro 1`, shipped learning rates, no export, state `out_r39\r39_w16_stage2b_state.pt`
  (written at every new best on the real-position KL). Expected about 75 min per step (24 steps = about 30 h).
  To stop: kill the python process; the state file can be exported later with
  `--load-state <state> --export-only --export-dir <new dir>` (never the source archive dir).

- Run 1 (chunks of 8) killed by the watchdog at 04:03 in the reverse pass of its first chunk: RSS 107.6 GiB after
  the forward pass (base 82 + 22.5 GiB checkpoints) drifting to 112.8 GiB over 28 windows, RAM free 6.9 -> 2.0 GiB.
  The drift is not explained (the 2-sequence diagnostic was flat at 82 GiB). **Run 2 started 04:05 with
  `--chunk-seqs 4`** (11 GiB of checkpoints), otherwise identical: `out_r39\phase7_stage2b_run2.log`.

- Run 2, step 1 of 24 (04:10-05:15): 3,863 s (forward 576 s, reverse 3,285 s = 103 s per sequence), evaluation
  162 s; RSS 92.9 GiB flat, RAM free 20-22 GiB, VRAM peak 26.2 GiB, no spill. Train KL of the batch 0.27669.
  **qbench real positions: KL 0.29058 -> 0.29044 (-0.00014), top-1 0.8371 -> 0.8383.** Parameter movement after one
  step: router 0.044 %, codebooks 0.009 %, mean |log scale| 0.00038. The target 0.228 needs -0.0626; the shipped
  archive is at 0.260.

- Router export check on the step-1 state (CPU only, helper script `router_bf16.py`, not in this repository): the base gate weights are exactly
  bf16; the export casts `w0 + LoRA` to bf16. After step 1 only 9.7 % of the router weights change value in bf16, and
  the change that survives has cosine 0.45 with the trained change (norm ratio 0.60): the exported router carries
  about a quarter of the trained change plus rounding noise of about twice that. The trainer's evaluation uses the
  fp32 router, so its KL is NOT the KL of the exported archive while the change is this small. Re-measure on the
  final state; an exported archive must be scored by `quality_eval_r39.py`, not by the trainer's number.

- Run 2, step 2 of 24 (05:18-06:24): 3,949 s (forward 593 s, reverse 3,354 s), evaluation 165 s, RSS 93.1 GiB,
  RAM free 19.3-19.8 GiB, VRAM peak 26.3 GiB. Train KL of the batch (sequences 32..64) 0.23649.
  **qbench real positions: KL 0.29120, top-1 0.8375: WORSE than the untuned start (0.29058)**; not a new best.
  Parameter movement after two steps: router 0.069 %, codebooks 0.015 %, mean |log scale| 0.00041.
  | state | KL real positions | top-1 |
  |---|---|---|
  | untuned (EVAL 0) | 0.29058 | 0.8371 |
  | after step 1 | 0.29044 | 0.8383 |
  | after step 2 | 0.29120 | 0.8375 |
- **Run 2 stopped by me at 06:25** (in the first chunk of step 3). Reasons: (a) the two steps moved the score by
  -0.00014 then +0.00076 (net +0.00062), i.e. no trend against a needed -0.0626; (b) the shipped learning rates (router 2e-4, bias
  6e-5, LUT 1.2e-4, scales 4e-4) with Adam bound the total movement of 24 steps to about 24 x lr per parameter
  (under 1 % of a row scale, 0.3 % of a LUT entry); the trainer's own self-test uses rates 10-25 times larger;
  (c) each step costs 66 min + evaluation, so the full 24 steps are about 27 h (documented: 45-65 min for the stage);
  (d) speed and the official quality gate had not been measured at all and need the GPU. Best state (step 1) is in
  `out_r39\r39_w16_stage2b_state.pt` (544 MB, 05:15); `--load-state` restores parameters, not the Adam moments.
  NOT done: steps 3-24 at the shipped rates; a run at larger learning rates (a deviation from the package; would
  show whether the method can recover anything on this archive in practical time).
- Phase 8 started 06:27 on the UNTUNED archive (`out_r39\phase8_speed_w16_untuned.log`, guard `out_r39\watch_eval.ps1`,
  memory `out_r39\eval_mem.csv`). `--speed-only` needs `--qbench-only` as well (the 111-row data file
  `out_r32\r34_prune_data.json` is not in this tree, so the full-coverage check fails without it; a copy exists in
  `..\build_run\out_r32\`, usable through `--data` for the full 111-row run).

- **Phase 8 speed, QTIP W16 untuned (06:27-06:44, `out_r39\eval_speed_w16_untuned.json`)**: this package's runtime,
  pool 18 GiB (3,054 slots), 41.8 GiB registered, graphs on, no MTP, no `POOL_REG_TOPK`, prefix cache off.
  First load 875 s (it wrote the 42 trunk `.slab` sidecars, 66 GB; C: free 86 -> 24 GiB). VRAM free before the
  timed part 1.63 GiB. **Overall 16.46 tok/s** (code 14.18, capital 18.05, math 15.64, systems 17.96; rounds 2-3
  alone: 15.18 / 18.55 / 15.89 / 18.01); pool hit rate 0.63 -> 0.79 over the run. Gate (>= 18 on code, capital and
  overall) FAILS; forecast 21.5-23.5 tok/s not reached. Not yet measured: the shipped TCQ33P5 archive in this same
  runtime (needed before "slower" can be attributed to the format instead of the older runtime), and the runbook's
  exact environment (`R37_POOL_REG_TOPK=6`).
- Phase 8 official qbench gate + speed, QTIP W16 untuned, pool 16 GiB, started 06:46
  (`out_r39\phase8_qbench_w16_untuned.log`, `eval_qbench_w16_untuned.json`), evx = `GLM53-APEX-FOLD-TCQ33\_r33_ckpt\evx.pt`.
- Mask facts (`configs\masks\hyb_W16_75GB.json`): 1,922 of 12,096 trunk experts pruned = 15.89 %, 27-54 per layer
  (kept 234-261). Bytes per expert: QTIP 6,326,528, TCQ33P5 6,359,296; the size gain is from the pruning, not from
  the code (shipped P5 prunes about 4.9 %).
- The owner asked (2026-11-17, during Phase 8) whether a published GLM-5.3-Flash release with about 15 % of the
  experts pruned and a retrained router already exists and could be reused. Searched Hugging Face + web: NO
  GLM-5.3-Flash release with about 15 % pruned and a retrained router. Nearest: autotrust E256 (11 %, NAS,
  NVFP4, card silent on router training), autotrust E224 (22 %), OpenMOSE REAP-250B (20.8 %, no retraining),
  lapadatbogdan RCO-152E (47 %, "healed", GGUF), 0xSero GLM-5.2-504B (34 %, router-only distillation, but GLM-5.2).
  All are on the zai-org base with their own masks; ours is the dealignai uncensored base with the W16 mask, so
  none would drop in.
- Disk: deleted the finished build's resume checkpoints `_r39_ckpt\{trx,evn,evo}.pt` (10.4 GiB; `complete: true`
  checked first; `evx.pt`, `qbench_teacher_top128.pt` kept). The `.slab` sidecars (66 GB, regenerated automatically
  at the next QTIP load, about 15 min) are to be deleted when the QTIP evaluations are done.

- **Phase 8 official qbench gate, QTIP W16 untuned (06:45-07:30, `out_r39\eval_qbench_w16_untuned.json`)**:
  **KL 0.295316, top-1 0.83658** over 39,842 positions, 29 rows, graph N=1 execution. Gate (KL <= 0.230, top-1 >=
  0.846) FAILS. Agrees with the build's 0.29438 / 0.8380. Speed at pool 16 GiB after the quality rows: 16.58 tok/s
  (code 14.91, capital 18.25, math 15.54, systems 17.64). Second load took 66 s with the sidecars present
  (documented: < 0.3 s).
- QTIP speed variants at pool 18 (`out_r39\phase8_speed_w16_variants.log`): runbook environment
  (`R36_FUSE_MHC=1 R36_FP8_GEMV=v2 R37_POOL_REG_TOPK=6`) 16.07 tok/s (code 13.85, capital 17.69, math 15.43,
  systems 17.31); `R39_QTIP_WARPS=16 R39_QTIP_GX=32` 16.61 tok/s (14.23 / 18.24 / 15.86 / 18.11). Four QTIP speed
  measurements in all: 16.1-16.6 tok/s, none passes the gate.
- 07:39 deleted the 42 `.slab` sidecars (59.9 GiB; the 43 `.pt` layer files, 66.19 GB, untouched). C: free 94.6 GiB.
- Baseline started 07:39: shipped TCQ33P5 through this same runtime (`--runner tcq`, `R32_POOL=6
  R33_REGISTER_GIB=42 R33_REGISTER_MODE=rows`), speed at pool 18 then qbench + speed at pool 16
  (`out_r39\phase8_baseline_tcq33p5.log`). The TCQ loader only memory-maps the layer files; the log records the
  P5 folder's file count and byte total before and after.

- **Fix 11** (`quality_eval_r39.py`): `--runner tcq` with `--speed-gate` / `--speed-only` crashed in the warmup
  ("Inplace update to inference tensor outside InferenceMode"): the script captures the TCQ runner's graphs under
  `torch.inference_mode()` and then calls `runner.generate` (a `torch.no_grad` function) outside it. The speed gate
  of the TCQ runner now runs inside `torch.inference_mode()`. The QTIP path is unchanged. First attempt 07:38-07:39
  failed on this; P5 folder unchanged (110 files, 78,493,454,705 bytes before and after).
- **Baseline speed, shipped TCQ33P5 in this same runtime (07:40-07:43, `out_r39\eval_baseline_p5_speed_pool18.json`)**:
  `--runner tcq`, `R32_POOL=6 R33_REGISTER_GIB=42 R33_REGISTER_MODE=rows` (pool v6 with every option off, which the
  Round 3.5 A/B recorded as its best setting: `build_run\out_r32\r35rb_pool_ab_sum.json`), pool 18 GiB (3,039
  slots), 41.79 GiB registered, 588 experts masked (4.86 %). **Overall 12.70 tok/s** (code 10.81, capital 13.99,
  math 12.09, systems 13.93), pool hit rate 0.74-0.83.
  **Like for like in this runtime the QTIP archive is about 30 % FASTER than the shipped one (16.5 vs 12.7 tok/s).**
  The shipped archive's production figure (20-21.6 tok/s) comes from the Round 3.7/3.8 runtime, which the QTIP layer
  has not been ported to; whether QTIP would reach the forecast 21.5-23.5 there is untested.

- **Baseline official qbench gate, shipped TCQ33P5 in this same runtime (07:43-08:39,
  `out_r39\eval_baseline_p5_qbench_pool16.json`)**: **KL 0.260544, top-1 0.84348** over 39,842 positions; speed at
  pool 16 GiB 12.73 tok/s (code 11.28, capital 14.10, math 11.90, systems 13.65). P5 folder unchanged.
  | same runtime, same harness, same rows | new QTIP W16 untuned | shipped TCQ33P5 |
  |---|---|---|
  | qbench KL (gate <= 0.230) | 0.2953 | 0.2605 |
  | top-1 (gate >= 0.846) | 0.8366 | 0.8435 |
  | decode tok/s, pool 18 / pool 16 | 16.46 / 16.58 | 12.70 / 12.73 |
  | expert files | 66.19 GB | 75.02 GB |
  Neither archive passes the package's quality gate in this harness; the new one is 0.035 nats worse, 30 % faster
  and 8.8 GB smaller.
- The owner asked (08:20) for additional training or an adjusted training stack, with web research to pin down the
  method as documented by others who have carried it out. Research agent started (Router-KD cards by 0xSero,
  Tetra's row-scale retraining, RCO-152E healing, MaskCoFT, MoE-Pruner, QTIP fine-tuning).

- **Research result (agent, primary sources quoted)**: no source documents a router- or scale-only recipe at our
  scale. Router-only distillation: 0xSero GLM-5.2-504B REPORT.md ("train only the 75 router-gate matrices",
  "KL-match the unpruned teacher's top-20 next-token distribution", "AdamW (lr 5e-5)"; recovers routing but
  plateaus; 6x more data was not better) and arXiv 2603.02217 (router parameters only, token KL, T=1, lr 5e-5,
  batch 2 x accum 4, 512 tokens, 3,000 samples = about 375 steps of about 4k tokens, 2 h on an A100; small gain on
  REAP-pruned models). Both train the FULL gate matrices, not a LoRA. Tetra (arXiv 2609.35465 sec. 4.2): one scale
  per row, codes fixed, KL on output probabilities, 19.5M tokens for a 4B model, +3.15/3.29/1.67 MMLU points;
  optimizer and rate not in the paper. RCO-152E healing (same model family): LoRA r16 on attention + shared
  expert, NOT the router, AdamW lr 2e-5, 96 steps x 8 windows, cosine. Ours as shipped: LoRA r16 at 2e-4, 24 steps
  of 65k tokens. Our throughput caps us at about 60-100k tokens per hour.
- **Fix 12** (trainer; the shipped behaviour is unchanged without the new flags):
  `--router-full` (dense gate matrix delta instead of the LoRA), `--dense-reverse` (`_reverse_moe_layer_dense`: per
  MoE layer the experts are decoded once per chunk and the layer's backward is split so the attention graph and
  the decoded layer are never on the GPU together; experts are constants there, so no LUT gradient; row-scale
  gradients come from `moe_forward_dense(..., log_scale=)`), `--freeze-scales`, `--adam-beta2`, `--skip-macro`,
  latest state saved as `<save-state>.last` at every evaluation, `capture_grads` test hook.
  Self-tests (`out_r39\phase7_trainer_selftest_fix12.log`): shipped order 0.18197 -> 0.16463 (pass); gradient
  comparison compressed vs dense path, LoRA and full gate: worst cosine 0.999996, relative error <= 3e-3 (bf16
  matmuls); dense training LoRA + scales 0.17429, full gate + frozen scales 0.17388 (both pass).
- Dense reverse path on the real model (first step of run 1): reverse pass 111 s for 4 sequences (28 s per
  sequence; the compressed path took 103 s), forward 73 s, step of 4 sequences 185 s; peak VRAM 23.7 GiB, RSS
  92.6 GiB, RAM free 20.3 GiB. About 46 s per sequence in all against 121 s before (2.6x).
- **Router-KD run 1 started 08:52** (`out_r39\phase7_routerkd_run1.log`, state `out_r39\r39_w16_routerkd_state.pt`
  and `.last`): `--router-full --dense-reverse --freeze-scales --lr-router 5e-5 --lr-bias 5e-5 --adam-beta2 0.999
  --macro-seqs 4 --chunk-seqs 4 --eval-every-macro 8 --max-macro-steps 16`. Plan: router-only first (documented
  recipe), then add the row scales (Tetra) from the `.last` state.
- The owner asked (09:1x) for the speed work as well: bring this archive to almost 30 tokens per second by applying
  the optimisation methods of the production runtime. Port agent started (code only, no GPU): new tree
  `..\claude_r39_speed\` = copy of
  `claude_r38_ctx` + the QTIP runtime ported to the Round 3.8 modules + `quality_eval_r39s.py`. GPU tests of the
  port (self-test, graph vs eager, KL must stay 0.295316, speed) are mine to run between training runs.
  `claude_r38_ctx` (live server code) is not to be modified.

- **Port agent finished (code only; NOTHING run on CUDA yet)**: `..\claude_r39_speed\` = 41 files copied from
  `claude_r38_ctx` + `glm53_fold_qtip_runtime.py` (runner rebuilt as a mixin on the Round 3.8
  `ApexFoldMoEModelRunner`; kernels, slab layout, LUT, scales, signs, clamp, casts unchanged),
  `quality_eval_r39s.py`, `port_exact_r39s.py` (forced-decode logits, bitwise compare), and `[R3.9S]` regions in
  `glm53_fold_generate.py` (QTIP branch in `main()`), `glm53_fold_route_r36.py`, `glm53_fold_overlap_r36.py`.
  Pool: `R39_POOL=v6|qtip` (v6 when `R32_POOL=6`). Agent's own additions beyond porting: both pools start
  zero-filled; optional `R39_SLAB_DIR`; the `R37_POOL_REG_TOPK` cut in the QTIP pool now ranks by score instead of
  by id. Not carried over: the TCQ `e_sh16` kernel and its flags (TCQ bitstream only). Its GPU test plan is in
  `claude_r39_speed\run_port_tests.ps1` (`-Step selftest | exact | speed | quality | control | speedab`; logs in
  `claude_r39_speed\out_r39s\`). Expected if exact: 384 of 384 forced-decode logits bit-identical, KL 0.295316.
  Known unknowns from the agent: MTP with QTIP untested; `R33_REGISTER_GIB=68` locks about 60 GiB of RAM (check
  `registered_gib` in each report); production's 20-21.6 tok/s used a trimmed vocabulary head and pool 18, so the
  control in this harness may read lower. One slip reported by the agent: one CPU check ran with CUDA visible and
  exited at an assert before creating a context.

- **Router-KD run 1 result (08:52-09:49, 16 steps of 4 sequences = 64 sequences, 183-185 s per step)**:
  | state | qbench KL, real positions | top-1 | router rel. delta |
  |---|---|---|---|
  | untuned | 0.29058 | 0.8371 | 0 |
  | 8 steps | 0.29002 | 0.8379 | 0.52 % |
  | 16 steps | **0.28524** | 0.8379 | 0.73 % |
  Router only (full gate matrices + bias, AdamW 5e-5, beta2 0.999), LUT and scales frozen. -0.0053 nats in 50 min;
  the second 8 steps gave nine times the first 8. State: `out_r39\r39_w16_routerkd_state.pt` (16 steps, best);
  `.last` is the 8-step state. Trainer numbers use the fp32 router; an export rounds it to bf16 (re-check).
- **Fix 13** (trainer): `--probe-scales N`: log-scale gradients on 2 x N sequences, cosine between the halves at
  the (layer, projection), expert and row level, line search of shared offsets on qbench (kept only if better).
  Compiles; not yet run.
- 09:50 port GPU tests started (`claude_r39_speed\run_port_tests.ps1 -Step selftest`).

- **Port GPU tests (09:50-10:09, `claude_r39_speed\out_r39s\port_*.log`)**:
  - Self-tests: shipped QTIP self-test PASSED; `--selftest-port` P1-P5 all 0.0, with flags off and with the
    production flags.
  - Bit-exactness: forced-decode logits of 3 prompts x 128 graph steps, old runtime vs the port: **384 of 384
    bitwise identical** with flags off AND with the production flags (pool v6, fused mHC, FP8 GEMV v2,
    `R37_POOL_REG_TOPK=6`, 59.95 GiB registered = all 10,174 rows). So the port's quality is the old runtime's
    (KL 0.295316); the 45-min quality run was not repeated. Graph vs eager inside the port: top-1 16 of 16, mean
    KL 2.6e-7 (not bit-exact, as in Round 3.7).
  - **Speed, QTIP W16 untuned in the production runtime, pool 18 GiB: 22.06 tok/s overall (code 20.65, capital
    23.74, math 21.20, systems 22.65); the gate (>= 18 on code, capital, overall) PASSES and the package's
    forecast 21.5-23.5 is met.** Old package runtime: 16.46. Load 75 s with the sidecars present.
  Sidecars were rebuilt by the old-runtime record (59.9 GiB; C: free 32.4 GiB).

- **Control and A/B in the production runtime (10:09-10:21, speed only, pool 18 GiB, same script and prompts)**:
  | archive / setting | overall tok/s | code | capital | math | systems |
  |---|---|---|---|---|---|
  | shipped TCQ33P5, production flags (`e_sh16` kernel), 67.91 GiB registered | **23.20** | 20.74 | 25.84 | 22.03 | 24.19 |
  | QTIP W16, production flags | 22.06 | 20.65 | 23.74 | 21.20 | 22.65 |
  | QTIP W16, `R37_POOL_REG_TOPK` unset | 22.17 | 20.48 | 24.15 | 21.12 | 22.95 |
  | QTIP W16, `R39_QTIP_WARPS=16 R39_QTIP_GX=32` | 21.83 | 20.45 | 23.50 | 20.98 | 22.40 |
  | QTIP W16, shipped QTIP pool (`R39_POOL=qtip`) | 21.33 | 19.65 | 23.12 | 20.35 | 22.19 |
  **In the production runtime the shipped archive is about 5 % FASTER than the QTIP archive** (23.2 vs 22.1); the
  30 % QTIP advantage exists only in the old runtime, where the TCQ kernel is the unoptimised one. The QTIP expert
  kernel has not had the Round 3.7 kernel work (`e_sh16`); that is the open speed item.
- `quality_eval_r39s.py`: `--mtp` (loads layer 45, speed gate with `use_mtp`; graph path with `R35_MTP_GRAPHS=1`),
  `--compare-ids <report>`; the speed gate now records the generated ids. MTP run started 10:22
  (`out_r39s\port_mtp.log`, pool 16 GiB).
- The owner asked (10:1x) whether this is the most efficient / highest fidelity GLM-5.3-Flash compression on record.
  Answered: not supportable (no common benchmark; SAGE REXL3 2.48 bpw reports KL 0.030 at 96 GiB; RCO-152E and
  exlr8 K2 are smaller or comparable); the defensible angle is one 32 GB consumer GPU at 22 tok/s, to be backed by
  a public-text KL, one competitor measured on this machine, and task benchmarks (not yet scheduled).

- **Speculative decoding (MTP layer 45, graph path, `R35_MTP_GRAPHS=1`, production flags, pool 16 GiB;
  `out_r39s\eval_qtip_speed_mtp*_pool16.json`)**: works on the QTIP archive in the port.
  | mode | overall tok/s | tokens vs a plain run (4 prompts x 128) | draft acceptance |
  |---|---|---|---|
  | plain (pool 18) | 22.06 | - | - |
  | MTP, rows exact | 22.81 | identical on all 4 | code 0.969, capital 0.887, math 0.953, systems 0.855 |
  | MTP, `R35_MTP_ROWS=batched` | 23.06 | DIFFERENT on 3 of 4 (first at token 26 / 58 / 117) | - |
  Lossless drafting gives +3 %: a two-position verify step costs almost two steps (each position runs its own 8
  experts x 42 layers). Not worth shipping as a speed feature; batched mode is not lossless. Per request the pool
  misses about 12,400 expert rows per 128 tokens (hit rate 0.75), i.e. roughly 97 x 6.3 MB of host-to-GPU copies
  per token.
- Kernel agent started 10:30 (code only): `R39_QTIP_KERNEL=v2` in `claude_r39_speed\glm53_fold_qtip_runtime.py`,
  bit-identical to v1 by requirement, with `--selftest-kernel` (bitwise check + micro-benchmark) for me to run.
  Do not run port tests from `claude_r39_speed` while that agent is editing the file.
- 10:30 scale probe started on top of the 16-step router state (`out_r39\phase7_scaleprobe.log`, state
  `r39_w16_routerkd_scaleprobe_state.pt` if anything is kept).

- **Order of work set by the owner (10:3x)**: finish the training and the kernel work, then rerun the drafting / MTP
  tests to see whether that changes the result. So: (1) scale probe, long router run (evaluation every 16
  steps, stop after two evaluations without improvement or 6 passes); (2) kernel v2 GPU checks in pauses at
  training checkpoints; (3) export the best state, score the export with `quality_eval_r39s.py`, then RERUN the
  MTP tests (exact and batched, `--mtp --compare-ids`) on the trained archive with the final kernel; (4) release
  preparation (public GitHub + Hugging Face; recorded agent sessions left out; MIT; the file list is shown
  before anything is pushed).

- **Scale probe result (10:30-11:06, `out_r39\phase7_scaleprobe.log`, on the 16-step router state, KL 0.28524)**:
  log-scale gradients from two disjoint halves of 8 sequences each. Cosine between the halves: per (layer,
  projection) 0.28, per expert -0.07, per row -0.01, i.e. the per-row and per-expert scale gradients at this data
  size are noise. Shared-offset line search on qbench: every tried offset made KL WORSE (per layer 0.01 / 0.03 /
  0.1 of the largest gradient: 0.28713 / 0.28729 / 0.29512; per expert 0.01 / 0.03: 0.28745 / 0.28821). Nothing
  kept. **No systematic scale error; row-scale training is dropped** (and the shipped Stage 2B's Adam steps on 83M
  row scales were noise injection, consistent with its step 2 getting worse).
- **Kernel agent finished (code only)**: `R39_QTIP_KERNEL=v2` in `claude_r39_speed\glm53_fold_qtip_runtime.py`
  (`[R3.9K]`): one word load per lane and tile + neighbour by shuffle instead of 12 byte loads, separate k=1 / k=2
  loops, load-ahead (`R39_QTIP_PF`, default 4), gate+up in one GEMV launch and one sparse-add launch (4 launches per
  layer instead of 6), `int row_bytes`, `__launch_bounds__(512)`; float sequence copied from v1. CPU evidence only
  (NVRTC compile to PTX, PTX read, the agent's own numpy model). Its reading: the TCQ `e_sh16` gain was mostly
  occupancy and does not transfer; QTIP is latency-bound (serial tile chains); estimate of the QTIP expert stage
  9-10 ms of a 45 ms step, expert traffic 13-19 ms, backbone about 20 ms (a model, off by 1-2 ms on its control).
  The idea that would attack the latency (splitting a row's tiles over several warps) changes the summation order
  and was left out. GPU tests: `claude_r39_speed\run_kernel_tests.ps1 -Step selftest | sweep | exact | speed`.
  Kernel self-test started 11:07.

- **Kernel v2 GPU results (11:03-11:10, `claude_r39_speed\out_r39s\kernel_*.log`)**:
  - `--selftest-kernel` (synthetic, rule A, GU125, pool stride, REAL rows of layer 24): 440 eager comparisons and
    every replayed graph output v2 == v1 bitwise. Timing, graph replay, N=1, per layer: decode_core 298.3 -> 126.2
    us (x2.36), the three GEMVs 259.4 -> 89.3 us (x2.9); x42 layers = 12.5 -> 5.3 ms per token. N=2: 333.8 ->
    177.5 us (x1.88).
  - In the model, production flags: forced-decode logits kernel v1 vs v2: **384 of 384 bitwise identical**.
  - **Speed, QTIP W16 untuned, production flags + `R39_QTIP_KERNEL=v2`, pool 18 GiB: 25.29 tok/s overall (code
    23.13, capital 27.95, math 23.97, systems 26.10)**; generated tokens identical to the reference run on all 4
    prompts. Kernel v1: 22.06. Shipped TCQ33P5 in the same runtime and script: 23.20. So with v2 the QTIP archive
    is about 9 % faster than the shipped one in the production runtime. Not yet done: `-Step sweep` (grid of v2
    settings), MTP with v2.
- **Fix 13b** (trainer): `--patience N`; `.last` state also written at the end of a sweep.
- **Router-KD run 2 started 11:11** (`out_r39\phase7_routerkd_run2.log`): `--load-state` the 16-step state,
  `--skip-macro 16`, `--eval-every-macro 16 --sweeps 6 --patience 2`, otherwise as run 1. Best state keeps being
  written to `out_r39\r39_w16_routerkd_state.pt`; a copy of the 16-step state is `..._state_step16.pt`.
  AdamW moments restart from zero (not in the state).

- **Kernel v2 sweep (11:13-11:15, real rows of layer 24, 25 settings, all bitwise equal to v1;
  `out_r39s\kernel_sweep.log`)**: decode_core per layer, graph replay: v1 298.0 us; v2 default (warps 8, PF 4)
  125.9; best (warps 16, PF 8, shared) 111.6 = 4.69 ms per token. Load-ahead depth is the lever (PF 0 / 2 / 4 / 8:
  160.6 / 148.5 / 125.9 / 113.6 us at warps 8); activations from global memory are slower everywhere; separate
  gate/up launches (`fuse_gu=0`) 172.2 us.
- **Full-model speed with the swept settings (production flags, pool 18 GiB, tokens identical to the reference on
  all 4 prompts)**:
  | kernel setting | overall tok/s | code | capital | math | systems |
  |---|---|---|---|---|---|
  | v2 default (warps 8, PF 4) | 25.29 | 23.13 | 27.95 | 23.97 | 26.10 |
  | v2 warps 16, PF 8 | 25.32 | 23.07 | 28.13 | 23.94 | 26.15 |
  | **v2 warps 16, PF 16** | **25.90** | 23.66 | 28.69 | 24.54 | 26.70 |
  Best so far: `R39_QTIP_KERNEL=v2 R39_QTIP_WARPS=16 R39_QTIP_PF=16` = 25.9 tok/s (shipped TCQ33P5: 23.2; QTIP
  kernel v1: 22.1). PF 16 is the maximum the kernel accepts.
- The 11:10 start of run 2 was stopped by me before its first step (the owner asked for the kernel settings sweep as well);
  **run 2 restarted 11:22** with the same command. Remaining schedule: at most 176 steps x about 3.1 min + 11
  evaluations = about 9.6 h; `--patience 2` may end it earlier.
- The owner asked (11:2x) to begin organising the release materials now, with a headline that the measurements can
  back up. Release-assembly agent started (local staging only, no publishing): `..\release_r39\github\` and
  `..\release_r39\huggingface\`, mandatory scrub, unfinished numbers carry a marker token.

- **Order confirmed by the owner (11:3x)**: release only after the drafting / MTP optimisation has been tried for
  additional efficiency and speed, then the benchmarks, then the release. So after training:
  (1) export the best state and score it; (2) MTP / drafting optimisation and its GPU tests (agent started 11:3x,
  code only: lossless gains behind new flags tagged `[R3.9M]`, test script `claude_r39_speed\run_mtp_tests.ps1`);
  (3) benchmarks the owner agreed to: HumanEval and a GSM8K sample on the final archive, plus one public 2-bit
  build measured on this machine for a speed comparison; (4) release (show the file list first). The release
  staging tree must be re-synchronised with `claude_r39_speed` at the end (the MTP agent edits runtime files after
  the release agent copied them).

- Correction to the per-layer figures of the build (found while a paper draft was prepared): the build report has
  L25 -0.611 and L26 -0.610, so the per-layer bands are L03-L26 (between -0.21 and -1.50) and L27-L37 (between
  -0.12 and -0.42); the earlier line that put L25 and L26 into the second band was wrong.

- **MTP agent finished (code only, nothing run on CUDA)**; GPU plan `claude_r39_speed\run_mtp_tests.ps1`, steps in
  order: `reference` (plain, pool 16) -> `exact` -> `variants` -> `policy` -> `batchsweep` (optional) -> `kernel2`
  (optional) -> `one -Tag best -Extra '...' -PoolGib 17.5`; `-Step table` reprints. New default-off flags
  (`[R3.9M]`): `R39_MTP_PF_ROWS=1` (one prefetch call per row of a pair), `R39_QTIP_N2_WARPS/_PF/_GX`,
  `R39_MTP_FP=1` (fingerprint of the logit bits per token; `--compare-ids` now checks every round and the logits),
  `R39_MTP_STATS=1`. Its findings from today's reports (fits within 0.7 %):
  plain kernel v2 = 18.95 ms x tokens + 0.267 ms x demand misses, i.e. about 19 ms compute + 19.7 ms of
  host-to-GPU expert copies per token (38.6 ms); MTP makes about 35 % more misses per token than plain because the
  pool's rank gates (`R37_POOL_REG_TOPK=6`, `R35_POOL_COLD_TOPK=6`) rank one merged call for both rows (6 of up to
  16 predictions pass instead of 6 per row) and the pool is 16 instead of 18 GiB. Its expectation: MTP with kernel
  v2 as it stands is level with or slower than plain (25.90); +5 to +10 % at best with the fixes; +20 % not
  plausible. Lossless `R35_MTP_BATCH` classes: `norm`, `conv` provable; `rms`, `final` not; `core`, `attn`,
  `route`, `lin` need a run. **Half of every token is expert copy traffic: that, not the kernel, is now the
  largest speed item.**

- **Release-staging agent finished (local only)**: `..\release_r39\github\` (code repository tree),
  `..\release_r39\huggingface\` (model card, upload manifest, a cleaned manifest copy) and staging tools. Open items it
  listed: re-sync `runtime\` after the MTP work and re-run the bit-exactness checks on the final code; the public
  cannot run the quality gate, the build or the training (their inputs derive from private recorded sessions; even
  `--speed-only` read the private prompt file and opened `--exact-backbone`); the archive's `fold_manifest.json` stores
  an absolute mask path (upload a cleaned copy, and upload by explicit file list); the slim backbone has no
  vision-tower tensors (text only). Record vs logs: graph-vs-eager KL is 1.8e-5 (code), 1.6e-7 (capital), 2.6e-7
  (math), not one value; scale probe per-expert offset 0.1 gave 0.28582 (also worse); the header comment of the
  QTIP runtime still says kernel v2 was never run on a GPU (stale).


- **Scheduled with the owner (12:1x)**: expert-copy traffic work. During the MTP tests read the host-time share of
  a step from `R39_MTP_STATS=1`. Then: if host overhead is more than about 2-3 ms per token, a native decode loop
  (Rust or C++, zero-copy ring buffer) is worth doing; otherwise the effort goes into overlapping expert copies
  with compute (a ring of staging slots filled ahead of the decoder) and into the pool hit rate (larger effective
  pool, router trained to prefer resident experts).

- The owner agreed (12:2x) to patch the speed test so that it needs no private file and to make public benchmarks
  the public quality evidence. Done now: `quality_eval_r39s.py --speed-only` no longer reads the qbench file or
  opens `--exact-backbone` (compiles; to be exercised at the next GPU speed run; the row hash in the report changes
  for speed-only reports).

- **Owner decisions (12:1x-12:2x)**: (a) the comparison build is `unsloth/GLM-5.3-Flash-GGUF` `UD-IQ2_XXS` (101.8 GB,
  4 files; llama.cpp, experts split GPU / RAM); this REPLACES a public-text KL (no 328 GB download). Compare speed,
  HumanEval, GSM8K sample, size and memory on this machine; if the public build is slow, the same fixed subset for
  both. (b) The copy-traffic work is done BEFORE the release: measure host time vs GPU wait, then implement it
  (native loop and/or overlap of copies with compute, hit rate), verify bit-exactness, re-measure speed; only then
  benchmarks and release.

- **Run 2, evaluation after 32 steps (end of pass 1, 12:20)**: qbench real positions KL **0.28931**, top-1 0.8369;
  router rel. delta 1.04 %. WORSE than the 16-step state (0.28524), better than untuned (0.29058); not a new
  best, `--patience` count 1 of 2. Series so far: 0.29058 (0) / 0.29002 (8) / 0.28524 (16) / 0.28931 (32).
  Possible cause (not tested): the restart reset the AdamW moments, and Adam's first bias-corrected steps move
  every weight by about the full learning rate whatever its gradient; run 1's first 8 steps were also its weakest.
  If the next evaluation (48 steps) is not a new best the run stops by itself; then: add optimizer-state saving or
  a warm-up before any further restart, and consider a lower rate.

- **Run 2 ended 13:11 by `--patience 2`**: evaluation after 48 steps KL **0.28746**, top-1 0.8366 (not a new best).
  Router-KD series: 0.29058 (0) / 0.29002 (8) / 0.28524 (16) / 0.28931 (32) / 0.28746 (48). Best state remains
  the 16-step one (`out_r39\r39_w16_routerkd_state.pt` = `..._step16.pt`); `.last` is the 48-step state. Reading:
  the gain from router-only distillation at these settings is about 0.002-0.005 nats and fluctuates by about
  0.004 between evaluations; the gap to the shipped archive is 0.025. Not tried: lower rate, larger batch,
  warm-up, optimizer state across restarts, weight averaging (EMA) for evaluation.
- 13:13 MTP tests started on the free GPU (`run_mtp_tests.ps1`: reference, exact, variants).

- **MTP / drafting tests with kernel v2 (13:13-13:33, `claude_r39_speed\out_r39s\mtp_*.log`; production flags,
  `R39_QTIP_KERNEL=v2 WARPS=16 PF=16`; every MTP run: ids AND logit bits equal to the plain reference)**:
  | setting | tok/s | misses per token (decode) | free VRAM GiB |
  |---|---|---|---|
  | plain, pool 18 (11:21) | 25.90 | - | 1.72 |
  | plain, pool 16 (reference) | 24.19 | 52.5 | 3.24 |
  | MTP exact, pool 16 | 24.33 | 73.4 | 2.78 |
  | MTP + `R39_MTP_PF_ROWS=1`, pool 16 | 24.89 | 56.5 | 2.78 |
  | same, `REG_TOPK=8` / `4` | 24.00 / 24.61 | 47.0 / 70.2 | 2.78 |
  | MTP + PF_ROWS, pool 17.0 | 25.92 | 53.8 | 1.78 |
  | MTP + PF_ROWS, pool 17.5 | **25.99** | 52.4 | 1.28 |
  Draft acceptance 0.92, 1.83 tokens per verify step, verify step + draft = 1.77 plain steps at pool 17.5.
  **Lossless drafting at its best equals plain decode (25.99 vs 25.90): no gain; not worth shipping.** Side
  finding: plain decode gains 7 % from 2 GiB more pool (24.19 -> 25.90), so resident experts / VRAM budget is a
  real lever for the copy-traffic work. `policy`, `batchsweep`, `kernel2` steps not run (no upside left).

- 13:35 copy-traffic agent started (GPU allowed for short runs, `[R3.9T]` flags, `run_traffic_tests.ps1`,
  reports `out_r39s\traffic_*`). Its reference run: 25.41 tok/s plain pool 18 (with fingerprints on).
- Agreed with the owner (13:4x): benchmark both archives (new and shipped TCQ33P5) on HumanEval and the GSM8K sample,
  plus the Unsloth UD-IQ2_XXS build; decide between accepting the quality and a rebuild with lighter pruning from
  that result (if within noise: release on speed and size; if clearly behind: rebuild).

- 14:0x: download of `unsloth/GLM-5.3-Flash-GGUF` `UD-IQ2_XXS/*` started
  into `C:\models\competitors\unsloth-GLM-5.3-Flash-UD-IQ2_XXS` (101.8 GB). Benchmark-harness agent started (code
  only): `..\bench_r39\` (`bench_openai.py` for HumanEval + a fixed GSM8K sample of 250 over an OpenAI-compatible
  endpoint, `compare.py`, datasets, a CPU self-check, how to serve the three models incl. which llama.cpp build on
  this machine supports the architecture).

- **Benchmark-harness agent finished (nothing served, no GPU)**: `..\bench_r39\` (`bench_openai.py`, `compare.py`,
  `selfcheck.py` 90/90 ok, `README.md` with serve and run commands, `prep_data.py`, `data\` with pinned HumanEval
  164 and GSM8K test + `gsm8k_sample_250.json` seed 3901 + `humaneval_order.json`). Key points for the runs:
  - Our server: `python glm53_fold_generate.py --serve --port N --backbone-dir .. --fold-dir .. --expert-pool-gib ..
    --graphs`; archive type auto-detected; greedy only; thinking off = `</think>` appended after the template;
    effort low/high/max. QTIP over HTTP has NEVER been run (first start is a smoke test). `glm53_fold_generate.py`
    has no `--merged-routers`: a router file must sit in the archive folder as `merged_routers_bf16.pt`, or serve
    via `glm53_fold_qtip_runtime.py --serve --merged-routers`. Proposed pool 14 GiB for serving (the 18 GiB pool
    ran out of VRAM on a 180-token prompt in the recorded profile); speed at pool 14 unmeasured.
  - Fair settings: greedy, thinking off, `--reasoning-effort low` sent explicitly, `--max-tokens 1024`,
    `cache_prompt: false`, concurrency 1; pre-flight that llama.cpp renders the same prompt (`/apply-template`).
  - Unsloth GGUF declares architecture `glm5-next`; none of the three llama.cpp source builds on disk support it; a
    prebuilt llama.cpp server build (CUDA 12) that supports the glm5-next architecture ships a real
    `llama-server.exe` and runs standalone (untested at this point). Proposed:
    `-c 8192 -np 1 -t 8 -ngl 99 --n-cpu-moe 36 --jinja --reasoning-format deepseek --no-context-shift
    --cache-ram 0`, port 8188; about 27 GiB on the GPU, 68 GiB of experts in RAM; nothing else may be loaded.
  - Time per model at 25 tok/s: HumanEval about 0.5 h, GSM8K-250 about 0.8 h; at 3 tok/s 3.7 h and 5.7 h: for a
    slow model use `--ids-file data\humaneval_order.json --limit 80` and GSM8K `--limit 100`.

- Unsloth download complete 15:53 (25 min): 4 files, sizes equal to the Hub listing (9,429,984 / 49,896,298,080 /
  49,248,507,840 / 2,690,716,000 bytes) in `C:\models\competitors\unsloth-GLM-5.3-Flash-UD-IQ2_XXS\UD-IQ2_XXS\`.
  C: free 62.5 GiB. The copy-traffic speed runs between 15:28 and 15:53 overlapped with it (treat as noisy).

- **Copy-traffic agent finished (13:35-15:58; GPU runs, all lossless: ids and logit fingerprints identical)**.
  Measured split of a decode step, pool 18 GiB, ms per token (`out_r39s\traffic_proftrace.json`):
  | prompt | wall | GPU in graphs | stalled behind copies | GPU idle behind host | demand misses | prefetched (used) |
  |---|---|---|---|---|---|---|
  | code | 44.05 | 27.22 | 16.77 | 0.03 | 58.2 | 74.5 (51.9) |
  | capital | 36.06 | 25.92 | 10.04 | 0.06 | 35.2 | 42.5 (32.8) |
  | math | 42.50 | 26.99 | 15.44 | 0.03 | 53.5 | 68.2 (46.8) |
  | systems | 39.31 | 26.60 | 12.52 | 0.15 | 42.1 | 58.6 (47.9) |
  | no-miss floor | 24.39 | 24.23 | 0 | 0.13 | 0 | 0 |
  The no-miss floor is 24.4 ms = 41 tok/s; attention + router graphs are 14.2 ms of it, the expert kernels 4.7 ms.
  **The GPU waits on Python only 0.03-0.15 ms per token: a native (Rust) decode loop is NOT justified; not
  built** (toolchain present: rustc / cargo 1.98.1, MSVC). The stall is copy time of rows that could not be
  predicted (47 demand rows x 0.245 ms + 1.8 ms of host time before the first copy is issued). Fit: ms per token =
  21.65 + 0.227 x demand misses + 0.118 x prefetched rows.
  Changes (`glm53_fold_expert_pool_v6.py`, `[R3.9T]`, default off): `R39T_FAST_DEMAND=2` (+1.34 % paired),
  `R39T_PREEVICT=4` (with it +1.61 % paired), `R39T_LAZY_STAGING=1` + `--pool-gib 18.47` (80 more slots).
  **Best setting: those three, pool 18.47: +2.7 % against the reference in alternating runs (25.29 / 25.23 ->
  26.09 / 25.78).** Torture test `test_pool_r39t.py` 20 of 20, 0 mismatches. Did not help: looser prefetch gate,
  hit-first scheduling (-12 %), paced prefetch, other eviction policies (simulation). Remaining ceilings: optimal
  eviction would need 38.6 instead of 108 fetched rows per token (about 33 tok/s by the fit; no usable signal
  found); +1.2 % per 0.5 GiB of pool; backbone floor 41 tok/s. Re-run: `run_traffic_tests.ps1` (`-Step ab`,
  `-Step table`). Only plain N=1 decode measured with the new flags (not MTP, long context, server).
- **Hardware finding**: `nvidia-smi` reports PCIe link generation max 4 (GPU max 5, host max 4), width 16;
  motherboard ASUS PRIME B650M-A AX6 II (B650: the x16 slot is Gen4). Expert copies run at about 22-25 GiB/s; a
  Gen5 x16 slot (X670E / B650E class board) would roughly halve the copy time: about 32 tok/s by the fit.
  Not a software item; tell the owner.
- 16:0x quiet-machine A/B of reference vs best traffic setting started (`run_traffic_tests.ps1 -Step ab`).

- **Quiet-machine A/B (16:01-16:11, fingerprint bookkeeping on, pool 18 / 18.47)**: reference 25.07, 25.42;
  best traffic setting (`R39T_FAST_DEMAND=2 R39T_PREEVICT=4 R39T_LAZY_STAGING=1`) 25.38, 26.45: mean 25.25 ->
  25.92 (+2.7 %), ids and logit bits identical. Without the bookkeeping the best setting ran 26.09 / 25.78.
  Headline speed: about 26 tok/s.
- **Benchmarks started 16:1x** with `..\bench_r39\run_apex.ps1 -Kind new|shipped` (my driver: serves per README
  3.1 at pool 14, smoke test of 3 items, then HumanEval 164 + GSM8K 250, stops the server; results in
  `bench_r39\runs\`). Decision: the "new" model is benchmarked UNTUNED (archive as built, no router file): the
  router gain is within evaluation noise, and nothing is written into `C:\models`. The traffic flags are not used
  for serving (never run with the server; they do not change outputs). First: `-Kind new -SmokeOnly`.

- Serving smoke test 16:14: the QTIP archive serves over HTTP (first time); 3 of 3 items; decode 20.8 tok/s at
  pool 14 GiB (16.5 tok/s wall incl. prompt).
- **HumanEval, new archive (untuned, greedy, thinking off, effort low, max 1,024 tokens, served at pool 14):
  154 of 164 = 93.9 % (Wilson 95 % 89.1-96.7), 0 truncated, mean 177 completion tokens, decode 20.97 tok/s
  reported, 16.27 tok/s wall** (`bench_r39\runs\new\humaneval\summary.json`, 16:47). GSM8K-250 running.
- The owner decided (16:5x) to compare the Unsloth build directly with the new archive and to skip the comparison
  with the shipped archive for now. Driver for the Unsloth build written (`bench_r39\run_unsloth.ps1`: a
  prebuilt llama.cpp `llama-server.exe` that supports glm5-next; prompt pre-flight, smoke test, full sets or the
  fixed 80 / 100 subsets if under 8 tok/s). Consequence: the new-vs-shipped task comparison that was to decide the
  lighter-pruning rebuild is not being measured.

- **GSM8K sample, new archive (same settings): 241 of 250 = 96.4 % (Wilson 95 % 93.3-98.1), 0 truncated, 0 without
  an answer line, mean 47 completion tokens, decode 23.8 tok/s reported** (`runs\new\gsm8k\summary.json`, 17:04).
- The owner (17:0x): the shipped-archive benchmark is back on the list as the LAST item (`run_apex.ps1 -Kind
  shipped`), after the release work, not blocking it.
- 17:05 Unsloth smoke test started (`run_unsloth.ps1 -SmokeOnly`).

- Unsloth smoke tests 17:05 and 17:08: the prebuilt `llama-server.exe` (CUDA 12 build) runs standalone on the
  UD-IQ2_XXS file; VRAM 27.2 GiB, RAM free 76.9 GiB; 3 of 3; decode 12.3 tok/s reported by the server
  (8.0 tok/s wall incl. prompt) against 20.8 / 16.5 for the QTIP archive on the same 3 items. Prompt check: with
  `enable_thinking: false` this build renders `...<|assistant|><think>` (no `</think>`), with `--reasoning-budget
  -1` and with `0` alike; its prompt is 1 token shorter than ours and the model emits the closing tag itself as
  its first token (0 thinking characters on all 3 items). Accepted as equivalent; to be stated in the results.
- 17:1x full Unsloth run started (`run_unsloth.ps1`: full sets, then `compare.py` against `runs\new`).

- **Head-to-head on this machine, same harness and settings (greedy, thinking off, effort low, max 1,024 tokens,
  one request at a time, no prompt caching; `bench_r39\runs\`; finished 19:00)**:
  | | new QTIP W16 (untuned, 66.2 GB experts, pool 14 GiB) | Unsloth UD-IQ2_XXS (101.8 GB, llama.cpp, 27 GiB VRAM) |
  |---|---|---|
  | HumanEval (164) | 154 = 93.9 % (89.1-96.7) | 158 = 96.3 % (92.2-98.3) |
  | GSM8K sample (250) | 241 = 96.4 % (93.3-98.1) | 239 = 95.6 % (92.3-97.5) |
  | decode tok/s reported, HumanEval / GSM8K | 20.97 / 23.81 | 12.26 / 12.94 |
  | tok/s wall incl. prompt, HumanEval / GSM8K | 15.81 / 11.13 | 7.84 / 5.92 |
  | mean completion tokens, HumanEval / GSM8K | 177 / 47 | 180 / 75 |
  Paired: HumanEval both right 152, only new 2, only Unsloth 6, neither 4 (exact McNemar p = 0.29); GSM8K both
  right 233, only new 8, only Unsloth 6, neither 3 (p = 0.79). Neither difference is statistically significant.
  No truncated answers on either side. Prompt token counts differ by exactly 1 on every item (the unclosed
  `<think>` of the llama.cpp build; see above). So: quality statistically level with a file 35 GB larger; decode
  1.7-1.8x faster, end to end about 2x.

- The owner agreed (19:0x) to DROP the lighter-pruning rebuild and release the archive as built (benchmarks level
  with the Unsloth build). Proposed headline (backed by the table above): GLM-5.3-Flash (320B) on one RTX 5090 at
  about twice the speed of the most-downloaded 2-bit build, two-thirds the size, benchmark scores statistically
  level. Remaining order: served-speed test at larger pools (`run_apex.ps1 -Kind new -PoolGib 17|18.47 -Traffic
  -Tag ... -LimitHe 40 -LimitGsm 40`) -> final numbers into the documents, re-sync of the release tree, rerun of
  the exactness checks on the final code -> file list -> publication -> shipped-archive benchmark.

- The owner asked (19:2x) for the trained router to be benchmarked as well. Router export (CPU, helper script
  `export_router.py`, not in this repository): `out_r39\merged_routers_step16_bf16.pt` from the 16-step state, 42
  layers; 80.9 % of the router weights change after bf16 rounding, cosine with the trained change 0.976, norm ratio
  1.027 (the rounding concern from the 1-step state no longer applies). Plan: copy it into the archive folder as
  `merged_routers_bf16.pt` (the runner auto-loads that name), run `run_apex.ps1 -Kind new -Tag router16` (pool 14,
  same settings as the untrained run), remove the file again, `compare.py untrained vs router16`. Rule agreed:
  better or equal -> the router can ship; worse -> it stays out.

- **Served speed at a larger pool** (`runs\new_p17t`, pool 17 GiB + the three copy-traffic flags, first 40 items of
  each task, no out-of-memory): HumanEval 38 of 40, decode 23.19 tok/s reported (17.5 wall); GSM8K 39 of 40, decode
  25.74 (12.3 wall). Against pool 14 without the flags on all items: 20.97 and 23.81. The pool 18.47 run did NOT
  happen: my driver cleared its own `R39_PYTHON` variable after the first run (fixed) and its health check
  answered from the router run's server, which had started meanwhile; that server was not touched.
- Router benchmark running since 23:2x: the server log shows `patched_routers=42` (router file loaded).
  `compare.py` at the end of that job will fail for the same variable reason: run it by hand.

- The first router benchmark attempt (23:24) produced NOTHING: the failed pool-18.47 run, started in the same
  minute, deleted the shared `runs\smoke_new` folder while the router run's smoke test was writing to it; the
  router run stopped at its smoke test and removed the router file as designed. My error: two benchmark drivers
  were allowed to overlap. Restarted cleanly at 00:1x (one job, GPU free, archive folder as built before copy).

- **Trained-router benchmark (16-step router loaded, `patched_routers=42`, pool 14, same settings; finished
  00:15 on 2026-11-18; `runs\new_router16`)**:
  | | untrained | router16 | Unsloth |
  |---|---|---|---|
  | HumanEval (164) | 154 = 93.9 % | 156 = 95.1 % | 158 = 96.3 % |
  | GSM8K (250) | 241 = 96.4 % | 240 = 96.0 % | 239 = 95.6 % |
  | decode tok/s, HumanEval / GSM8K | 20.97 / 23.81 | 20.90 / 23.82 | 12.26 / 12.94 |
  Paired untrained vs router16: HumanEval only-untrained 1, only-router 3 (p = 0.63); GSM8K 2 vs 1 (p = 1.0).
  Net +1 item over 414: equal within noise; no speed change. By the agreed rule (better or equal -> may ship)
  the router MAY ship; router16 vs Unsloth: HumanEval 2 vs 4 (p = 0.69), GSM8K 7 vs 6 (p = 1.0). The router file
  was removed from the archive folder after the run. Owner to choose: ship the archive with or without
  `merged_routers_bf16.pt`.

- **Owner's decision (00:2x, 2026-11-18): ship the trained router.** The release archive = the 43 layer files + the
  cleaned `fold_manifest.json` + `merged_routers_bf16.pt` (= `out_r39\merged_routers_step16_bf16.pt`, 99,168,981
  bytes).
- Finalisation started 00:3x: (a) GPU job: `run_port_tests.ps1 -Step exact`, `run_kernel_tests.ps1 -Step exact
  -Tag k2final` on the final code, then the official qbench gate + speed of archive + router (production flags,
  kernel v2 w16 pf16, traffic flags, pool 16; `out_r39s\final_gate_router16.{log,json}`); (b) release agent
  resumed to fill final numbers, add `bench\`, re-sync, re-scrub and produce the upload file lists; (c) whitepaper
  agent resumed to fill Section 9 and rebuild. The results of (a) go below under this heading:



### Final gate with the trained router
- **Bit-exactness on the final code (00:1x-00:28, 2026-11-18)**: forced-decode logits, 3 prompts x 128 graph steps:
  old package runtime vs `claude_r39_speed` with flags off: **384 of 384 bitwise identical**; with the production
  flags: **384 of 384**; kernel v1 record vs kernel v2 record (both production flags): **384 of 384**
  (`out_r39s\port_exact.log`, `out_r39s\kernel_exact.log`). The `[R3.9M]` and `[R3.9T]` changes did not alter
  the decode path at default settings.
- **Official qbench gate of the release archive (QTIP W16 + `merged_routers_bf16.pt`, `patched_routers=42`,
  production runtime, production flags, kernel v2 w16 pf16, R39T flags, pool 16 GiB; 00:28-00:59,
  `out_r39s\final_gate_router16.{log,json}`): KL 0.290724, top-1 0.83505 over 39,842 positions, 29 rows.**
  Untuned in the old package runtime (same metric, pool 16): KL 0.295316, top-1 0.83658. Router: KL -0.0046,
  top-1 -0.0015. Shipped TCQ33P5 in the same harness: 0.260544 / 0.84348. Package gate (<= 0.230, >= 0.846):
  not met. Speed gate at pool 16 with the R39T flags, after the quality rows: **24.62 tok/s** (code 22.21, capital
  27.62, math 23.38, systems 25.25; gate passes); not directly comparable with the pool-18 figures (25.90 /
  25.92).
- The release-staging agent stopped at 00:2x before the model card and the upload manifest (about 212 files
  updated); continued with a new agent.

## Blockers known before the 42-layer build
- Disk: the flagship archive alone is about 66 GB, the tuned copy another 66 GB, its `.slab` sidecars another 66 GB.
- The 328 GB of FP8 source shards are not on disk (`--fetch shards` re-downloads, about 5 min per layer).
- `C:\models\GLM53-APEX-FOLD-TCQ33` has no layer files any more (deleted 2026-11-14; only `_r33_ckpt`): the Round 3.8
  control (Phase 4) and `--teacher-mode unpruned_tcq33` (Phase 3) as written need them.
