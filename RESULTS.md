# Measured results, Round 3.9 (QTIP HYB W16)

Every number here is copied from the working record (`docs/WORK_RECORD.md`) or from the report and log files under
`results/` (benchmark runs: `results/benchmarks/`). Nothing is rounded beyond what the source shows.

Common conditions, unless a table says otherwise:

- Machine: one RTX 5090 (32 GiB), 128 GB RAM, Windows 11, Python 3.14.0, PyTorch 2.11.0.dev20260124+cu128.
- Archive: `GLM53-APEX-FOLD-R39-HYB-W16`, QTIP HYB rule A (2.0112 bpw, 6,326,528 B per expert), mask
  `configs/masks/hyb_W16_75GB.json`, as built, WITHOUT the trained router file unless a row says "router".
- "Original runtime" = the runtime that came with the original package (not in this repository; evaluator
  `build/quality_eval_r39.py`). "Production runtime" = `runtime/` of this repository (evaluator
  `runtime/quality_eval_r39s.py`).
- "Production flags" = `R32_POOL=6 R33_REGISTER_GIB=68 R33_REGISTER_MODE=rows R35_POOL_SIGNAL=1 R35_POOL_DEMAND=1
  R35_POOL_DEMAND_STREAM=1 R35_POOL_GATHER=readfile R35_POOL_EVICT=gdsf R35_POOL_COLD_TOPK=6 R35_POOL_COLD_MARGIN=0.01
  R36_FUSE_MHC=1 R36_FP8_GEMV=v2 R37_POOL_REG_TOPK=6 R37_GPU_WINDOW=1` (the TCQ control adds
  `R37_TCQ_KERNEL=e_sh16 R37_TCQ_WARPS=16 R37_TCQ_GX=32`, which the QTIP path ignores).
- "Speed gate" = 4 prompts (code, capital, math, systems) x 3 alternating rounds x 128 new tokens, greedy, single
  stream, CUDA-graph decode, no speculative decoding, prefix cache off; the figure is the mean decode tok/s. The
  package's pass rule: code >= 18, capital >= 18 and overall >= 18.
- "qbench" = 29 held-out chat rows; KL (nats) and top-1 agreement of the compressed model against the exact FP8 model
  with the full 154,880-token head, per position. The official gate decodes with captured N=1 graphs after a
  2-token eager prefix (39,842 scored positions). The builder and the trainer score the same rows with their own
  forward pass (39,871 positions): 0.29438 (builder) and 0.29058 (trainer) for the untuned archive, against the
  gate's 0.295316. They are close but not the same metric. The test rows are private and not part of this release.

## 1. Offline gates and self-tests

| check | result | source |
|---|---|---|
| Phase 0, offline verifier (9 gates), patched tree | gates 1 to 8 pass; gate 9 (SHA-256 manifest) fails only on the files that carry the fixes. The unmodified original package hashes clean against its manifest | `results/selftests/phase0_verify_s2.log` |
| Phase 1, five GPU self-tests, as shipped | FAILED: both Viterbi encoder kernels compiled to 128 and 222 registers per thread at 1,024 threads per block (limit 64), "too many resources requested for launch" | `results/selftests/phase1_selftests.log` |
| Phase 1 after fix 1 (`--maxrregcount=64`) | 5 of 5 pass | `results/selftests/phase1_selftests_run2.log` |
| Phase 1 after fix 3 (chunked decode) | 5 of 5 pass | `results/selftests/phase1_selftests_run3_after_fix3.log` |
| Trainer self-test, patched (fixes 5 to 8) | passes: KL 0.18197 -> 0.16520 (same initial KL as the shipped script) | `results/selftests/phase7_trainer_selftest_patched.log` |
| Trainer self-test after fix 9, five variants (grad window, chunk) | all pass: (0, 0) 0.18197 -> 0.16504; (1, 0) 0.16492; (1, 2) 0.16453; (0, 2) 0.16525; (1, 1) 0.16514 | `results/selftests/phase7_trainer_selftest_fix9.log` |
| Trainer self-test after fix 12 | shipped order 0.18197 -> 0.16463 (pass); gradients of the compressed path vs the dense path, LoRA and full gate: worst cosine 0.999996, relative error <= 3e-3; dense training LoRA + scales 0.17429, full gate + frozen scales 0.17388 (both pass) | `results/selftests/phase7_trainer_selftest_fix12.log` |

## 2. Layer-27 codec gate

Content-token routed-output SNR of layer 27, scored on the qbench split, calibrated on the chat-trace split, against
the Round 3.3 codec `TCQ33_A_best`. GPU power capped at 450 W.

Smoke gate, 32 experts (baseline `TCQ33_A_best` 8.839 dB):

| variant | content SNR dB | vs TCQ33 |
|---|---|---|
| QTIP_L16_mul1_1D | 7.927 | -0.912 |
| QTIP_L16_HYB_1D | 8.019 | -0.820 |
| QTIP_L16_HYB_2D16 | 9.259 | +0.420 |
| QTIP_L16_HYB_2D16_QZ | 9.244 | +0.405 |
| QTIP_L16_HYB_2D16_QZ_Tune | 9.261 | +0.422 |
| QTIP_Sub70_GU125_2D16_QZ_Tune (1.9278 bpw) | 8.522 | -0.317 |

Source: `results/layer27/phase5_l27_smoke32.log`, `phase5_l27_smoke32_s2.log`, `phase5_l27_smoke32_s2b.log`,
`r39_l27_qtip_lab_smoke32.json`. The 1D arms were only run at 32 experts; they are not gates. After fix 3 the smoke run
reproduces 9.259 / 9.261 dB and every sub-figure exactly (`phase5_l27_smoke32_fix3check.log`), GPU peak 22.9 GiB.

Full gate, 288 experts (baseline `TCQ33_A_best` 9.188 dB; the recorded Round 3.3 value is 9.187), GPU peak 32.1 GiB,
RAM free >= 60.7 GiB:

| variant | content SNR dB | vs TCQ33 | weight SNR g / u / d | H-weighted SNR g / u / d |
|---|---|---|---|---|
| QTIP_L16_HYB_2D16 | 9.616 | +0.428 | 10.597 / 10.568 / 11.396 | 14.335 / 13.665 / 12.145 |
| QTIP_L16_HYB_2D16_QZ | 9.581 | +0.393 | 10.597 / 10.568 / 11.419 | 14.335 / 13.665 / 12.000 |
| QTIP_L16_HYB_2D16_QZ_Tune | 9.597 | +0.409 | 215 experts tuned, validation relative error 0.07592 -> 0.07311 | |
| QTIP_Sub70_GU125_2D16_QZ_Tune (1.9278 bpw) | 8.882 | -0.306 | 9.139 / 9.069 / 11.419 | |

Source: `results/layer27/phase5_l27_full288.log` (baseline), `phase5_l27_full288_b.log`, `r39_l27_qtip_lab.json`.

Gate outcome: QZ >= +0.30 dB: MET (+0.393; the forecast was +0.45 to +0.75 and was not reached). NVRTC GEMV cosine
1.0: MET. Sub-70 >= +0.60 dB over the Round 3.3 `L190_best` (7.977 dB): MET (+0.905).

Negative result: the cross-covariance target ("QZ") does NOT make the `down_proj` output better than the plain 2D
arm at layer 27: -0.035 dB routed output, -0.145 dB H-weighted down SNR, norm ratio 0.982 -> 0.968; only the
unweighted down weight SNR rises (+0.023 dB).

Resources: as shipped the full gate could not complete (the lab decoded all 288 experts of a projection on the GPU at
once: 60 to 70 GiB of temporaries, 89.7 GiB spilled into shared GPU memory, 140 GiB committed, free RAM 0.7 GiB; it
was stopped). With fix 3 (decode in chunks of 16 experts) the four 2D arms ran 21:12 to 21:37.

## 3. Archive build

| quantity | value | source |
|---|---|---|
| Build legs | leg 1 21:41-22:06 (L03, L04); leg 2 22:17-23:21 (through L14); leg 3 23:22-02:32 (184.6 min) | `results/build/phase6_build_w16_part1.log`, `_part2.log`, `_part3.log`, `r39_qtip_build.log` |
| Wall time, all three legs, including the shard downloads | about 4.6 h (documented by the package: 2.6 to 3.1 h) | work record |
| Per layer (leg 1) | encode 284 / 327 s, streams 37 to 38 s (teacher stream 17 to 19 s), GPU peak 25.1 GiB, 458 W max | `results/build/r39_qtip_build.log` |
| NVRTC GEMV check | cosine 1.0 on every layer | `results/build/r39_qtip_build.log` |
| Builder process memory | about 106 GiB private between layers, peak 124 GiB with the GPU share, 5 to 12 GiB RAM free | `results/build/builder_mem.csv` |
| Result | `complete: true`, 43 layer files (L03..L45), exit 0 | `results/build/r39_qtip_build_eval.json` |
| Expert layer files on disk | 66.19 GB (shipped TCQ33P5: 75.02 GB) | work record; directory listing: 66,189,441,160 bytes in 43 files |
| Bytes per expert | QTIP 6,326,528; TCQ33P5 6,359,296 | work record |
| Mask `hyb_W16_75GB.json` | 1,922 of 12,096 trunk experts pruned = 15.89 %, 27 to 54 per layer (kept 234 to 261). TCQ33P5 prunes 588 (4.86 %) | `configs/masks/hyb_W16_75GB.json` |

The size gain comes from the pruning, not from the code: the two formats differ by 32,768 bytes per expert.

End-of-build evaluation (builder's own pass, qbench, 39,871 positions, against the exact FP8 teacher):

| quantity | new QTIP W16, untuned | shipped TCQ33P5 |
|---|---|---|
| KL nats | 0.29438 | 0.26038 (recorded in Round 3.7: 0.2613) |
| top-1 agreement | 0.8380 | 0.8442 |
| MTP layer 45 draft acceptance (exact model: 0.7238) | 0.7158 | 0.7228 |
| expert layer files | 66.19 GB | 75.02 GB |

Source: `results/build/r39_qtip_build_eval.json`, `results/build/r39_qtip_build.log`.

Routed-output SNR per layer against the exact unpruned teacher, new minus shipped (dB), L03..L44, before any tuning:
-0.575, -1.497, -1.329, -1.203, -0.753, -1.155, -0.995, -0.811, -0.667, -0.896, -0.803, -0.830, -0.907, -1.057,
-1.135, -0.208, -0.810, -1.167, -1.043, -0.919, -0.486, -1.011, -0.611, -0.610, -0.363, -0.420, -0.258, -0.205,
-0.363, -0.215, -0.298, -0.301, -0.121, -0.246, -0.223, +0.107, +0.028, +0.137, +0.188, +0.242, +0.357, +0.435.
Mean -0.548 dB. Source: `results/build/r39_qtip_build_eval.json`.

## 4. Quality and speed in the original package runtime (same harness for both archives)

Runner `build/quality_eval_r39.py`, original runtime, graphs on, no MTP, prefix cache off.

| same runtime, same harness, same rows | new QTIP W16, untuned | shipped TCQ33P5 | source |
|---|---|---|---|
| qbench KL (package gate <= 0.230), 39,842 positions, pool 16 GiB | 0.295316 (no router) | 0.260544 | `results/original_runtime_eval/eval_qbench_w16_untuned.json`, `eval_baseline_p5_qbench_pool16.json` |
| top-1 (package gate >= 0.846) | 0.83658 (no router) | 0.84348 | same |
| speed gate, pool 18 GiB | 16.46 tok/s (code 14.18, capital 18.05, math 15.64, systems 17.96) | 12.70 tok/s (10.81 / 13.99 / 12.09 / 13.93) | `eval_speed_w16_untuned.json`, `eval_baseline_p5_speed_pool18.json` |
| speed gate, pool 16 GiB (after the quality rows) | 16.58 tok/s (14.91 / 18.25 / 15.54 / 17.64) | 12.73 tok/s (11.28 / 14.10 / 11.90 / 13.65) | `eval_qbench_w16_untuned.json`, `eval_baseline_p5_qbench_pool16.json` |
| host memory registered | 41.8 GiB | 41.79 GiB | logs `phase8_*.log` |

Conditions of the two columns: QTIP: this package's pool, 3,054 slots at pool 18, no `R37_POOL_REG_TOPK`. TCQ33P5:
`--runner tcq`, `R32_POOL=6 R33_REGISTER_GIB=42 R33_REGISTER_MODE=rows` (pool v6 with every option off), 3,039 slots,
588 experts masked.

Neither archive passes the package's quality gate in this harness. The package's speed gate fails for both.

Further QTIP speed measurements in the original runtime, pool 18 GiB:

| setting | overall tok/s | code | capital | math | systems | source |
|---|---|---|---|---|---|---|
| default | 16.46 | 14.18 | 18.05 | 15.64 | 17.96 | `eval_speed_w16_untuned.json` |
| the runbook's environment (`R36_FUSE_MHC=1 R36_FP8_GEMV=v2 R37_POOL_REG_TOPK=6`) | 16.07 | 13.85 | 17.69 | 15.43 | 17.31 | `eval_speed_w16_runbook_env.json` |
| `R39_QTIP_WARPS=16 R39_QTIP_GX=32` | 16.61 | 14.23 | 18.24 | 15.86 | 18.11 | `eval_speed_w16_warps16_gx32.json` |

Load time: first load 875 s (it wrote the 42 trunk `.slab` sidecars, 66 GB); second load 66 s with the sidecars
present (documented by the package: < 0.3 s). A qbench-only run took 45 minutes (documented: about 4 minutes).

## 5. Port into the production runtime: bit-exactness

| check | result | source |
|---|---|---|
| Original QTIP self-test in the port | passed | `results/production_runtime/port_selftest.log` |
| Port self-test P1 to P5, flags off and with the production flags | all 0.0 | `port_selftest.log` |
| Forced-decode logits, 3 prompts x 128 graph steps, original runtime vs port, flags off | 384 of 384 bitwise identical | `port_exact.log` |
| Same, port under the production flags (pool v6, fused mHC, FP8 GEMV v2, `R37_POOL_REG_TOPK=6`, 59.95 GiB registered = all 10,174 rows) | 384 of 384 bitwise identical | `port_exact.log` |
| Graph step vs eager step inside the port, 16 steps per prompt (information; not expected to be bit-exact) | top-1 16 of 16 on each prompt; mean KL 1.8e-5 (code), 1.6e-7 (capital), 2.6e-7 (math) | `port_exact.log` |

Because the port's logits are the original runtime's bit for bit, its quality is the original runtime's
(KL 0.295316); the 45-minute quality run was not repeated in the port.

## 6. Speed in the production runtime

Speed gate, pool 18 GiB, same script and prompts for every row.

| archive / setting | overall tok/s | code | capital | math | systems | source (`results/production_runtime/`) |
|---|---|---|---|---|---|---|
| shipped TCQ33P5, production flags (`e_sh16` kernel), 67.91 GiB registered | 23.20 | 20.74 | 25.84 | 22.03 | 24.19 | `eval_p5_speed_pool18.json` |
| QTIP W16, kernel v1, production flags, 59.95 GiB registered | 22.06 | 20.65 | 23.74 | 21.20 | 22.65 | `eval_qtip_w16_speed_pool18.json` |
| QTIP W16, kernel v1, `R37_POOL_REG_TOPK` unset | 22.17 | 20.48 | 24.15 | 21.12 | 22.95 | `eval_qtip_speed_noregtopk.json` |
| QTIP W16, kernel v1, `R39_QTIP_WARPS=16 R39_QTIP_GX=32` | 21.83 | 20.45 | 23.50 | 20.98 | 22.40 | `eval_qtip_speed_w16gx32.json` |
| QTIP W16, kernel v1, the package's own pool (`R39_POOL=qtip`) | 21.33 | 19.65 | 23.12 | 20.35 | 22.19 | `eval_qtip_speed_qtippool.json` |
| QTIP W16, kernel v2 default (warps 8, load-ahead 4) | 25.29 | 23.13 | 27.95 | 23.97 | 26.10 | `eval_qtip_w16_speed_pool18_k2.json` |
| QTIP W16, kernel v2, `R39_QTIP_WARPS=16 R39_QTIP_PF=8` | 25.32 | 23.07 | 28.13 | 23.94 | 26.15 | `eval_qtip_w16_speed_pool18_k2_w16pf8.json` |
| QTIP W16, kernel v2, `R39_QTIP_WARPS=16 R39_QTIP_PF=16` | 25.90 | 23.66 | 28.69 | 24.54 | 26.70 | `eval_qtip_w16_speed_pool18_k2_w16pf16.json` |

All QTIP rows pass the package's speed gate (>= 18 on code, capital and overall). In the three kernel v2 runs the
generated tokens are identical to the reference run on all 4 prompts. Load 75 s with the sidecars present.
The copy-traffic flags add about 2.7 % on top of the last row (section 14).

Reading: with kernel v1 the QTIP archive is about 5 % slower than the shipped TCQ33P5 archive in the production
runtime (22.1 against 23.2); with kernel v2 it is about 9 % faster at the default setting (25.29 against 23.20). The
30 % advantage of section 4 exists only in the original runtime, where the TCQ kernel is the unoptimised one.

Kernel v2 micro-benchmark (`--selftest-kernel`, real rows of layer 24, CUDA-graph replay, per layer, cold; does not
include the backbone or the pool traffic):

| measure | kernel v1 | kernel v2 | ratio | source |
|---|---|---|---|---|
| decode_core, N=1 | 298.3 us | 126.2 us | x2.36 | `kernel_selftest.log` |
| the three GEMVs, N=1 | 259.4 us | 89.3 us | x2.9 | `kernel_selftest.log` |
| decode_core x 42 layers | 12.5 ms per token | 5.3 ms per token | | `kernel_selftest.log` |
| decode_core, N=2 | 333.8 us | 177.5 us | x1.88 | `kernel_selftest.log` |

Bit-exactness of kernel v2: 440 eager comparisons and every replayed graph output v2 == v1 bitwise (synthetic slabs,
rule A, rule GU125, pool row stride, real rows of layer 24); in the model, forced-decode logits kernel v1 vs v2 under
the production flags: 384 of 384 bitwise identical (`kernel_exact.log`).

Kernel v2 sweep (25 settings on real rows of layer 24, all bitwise equal to v1; `kernel_sweep.log`), decode_core per
layer in a graph: v1 298.0 us; v2 default (warps 8, load-ahead 4) 125.9 us; best (warps 16, load-ahead 8, shared)
111.6 us = 4.69 ms per token. Load-ahead 0 / 2 / 4 / 8 at warps 8: 160.6 / 148.5 / 125.9 / 113.6 us. Activations
read from global memory are slower in every setting; separate gate/up launches (`fuse_gu=0`): 172.2 us. Load-ahead 16
is the maximum the kernel accepts; it was measured in the full model only (table above).

## 7. Speculative decoding (MTP layer 45)

Graph path (`R35_MTP_GRAPHS=1`), production flags, kernel v1, pool 16 GiB, 61.64 GiB registered (43 layers).

| mode | overall tok/s | tokens vs a plain run (4 prompts x 128) | draft acceptance | source (`results/production_runtime/`) |
|---|---|---|---|---|
| plain (pool 18 GiB) | 22.06 | - | - | `eval_qtip_w16_speed_pool18.json` |
| MTP, rows exact | 22.81 | identical on all 4 | code 0.969, capital 0.887, math 0.953, systems 0.855 | `eval_qtip_speed_mtp_pool16.json`, `port_mtp.log` |
| MTP, `R35_MTP_ROWS=batched` | 23.06 | DIFFERENT on 3 of 4 (first difference at token 26 / 58 / 117) | code 0.969, capital 0.875, math 0.968, systems 0.823 | `eval_qtip_speed_mtp_batched_pool16.json`, `port_mtp.log` |

Lossless drafting gives +3 %: a two-position verify step costs almost two steps (each position runs its own 8
experts x 42 layers). The batched mode is not lossless. Note the plain row was measured at pool 18 GiB and the MTP
rows at pool 16 GiB. The same test with kernel v2 and matched pools is in section 13: no gain.

## 8. Stage 2B as shipped: could not run

The original package's tuning stage ("Stage 2B": router LoRA rank 16, HYB tables and row scales, 6 sweeps of 4
steps of 32 sequences, `--window-size 4`), documented as 45 to 65 minutes with about 11.8 GiB peak VRAM on a 96 GB
machine.

| attempt | measurement | source (`results/training/`) |
|---|---|---|
| Archive load and starting evaluation | load about 90 s (64 GiB private); evaluation 171 s, 26.8 GiB VRAM, about 100 GiB private | `phase7_stage2b_probe.log` |
| Starting score | shipped metric 3.837 nats / top-1 0.597; real positions only 0.29058 nats / top-1 0.8371 over 39,871 tokens. The shipped metric is dominated by 19,231 padded positions and the shipped script picked its best state on it (fix 8) | `phase7_stage2b_probe.log`, `phase7_stage2b_probe_b.log` |
| Shipped settings (`--window-size 4 --macro-seqs 32`) | the first step drove RAM free to 2.7 GiB with the GPU at 32.1 GiB; stopped | `phase7_stage2b_probe_b.log`, `trainer_mem.csv` |
| 4 sequences, `--window-size 4` | forward pass 65 s, 20.5 GiB; the reverse pass of the first 4-layer window did not finish one sequence in 5 min (32.1 GiB VRAM plus about 50 GiB spilled to shared memory); stopped at 1.3 GiB RAM free | `phase7_stage2b_diag4.log` |
| 4 sequences, `--window-size 1` | works: reverse pass 477 s for 4 sequences x 45 layers = 2.65 s per sequence per layer, about 26 GiB reserved VRAM | `phase7_stage2b_diag4_w1.log` |
| With fixes 9 and 10, 2 sequences, chunk 1, grad window 1 | clean; RAM free 32 GiB; reverse pass 122 to 125 s per sequence; VRAM peak 26.0 GiB; one step: KL 0.29058 -> 0.29053, top-1 0.8371 -> 0.8379 | `phase7_stage2b_diag2_chunk.log` |
| Run 1, chunks of 8 sequences | stopped by the watchdog in the reverse pass of its first chunk (RSS 107.6 -> 112.8 GiB, RAM free 6.9 -> 2.0 GiB) | `phase7_stage2b_run1.log` |
| Run 2, chunks of 4, step 1 of 24 | 3,863 s (forward 576 s, reverse 3,285 s), evaluation 162 s, RSS 92.9 GiB, VRAM peak 26.2 GiB; KL 0.29058 -> 0.29044, top-1 0.8371 -> 0.8383 | `phase7_stage2b_run2.log` |
| Run 2, step 2 of 24 | 3,949 s, evaluation 165 s; KL 0.29120, top-1 0.8375: WORSE than the untuned start | `phase7_stage2b_run2.log` |

Run 2 was stopped in step 3. At 66 minutes per step plus evaluation the full 24 steps are about 27 hours. Parameter
movement after two steps: router 0.069 %, tables 0.015 %, mean |log scale| 0.00041. The package's target needs
-0.0626 nats; the two steps gave -0.00014 and +0.00076.

Defects found in the shipped trainer before its first real run and fixed (the shipped behaviour is otherwise
unchanged): fix 5 (the scale optimizer aliased the archive's scale tensor, so every step compounded the scales);
fix 6 (the HYB table was clamped to >= 1e-4 after every step although the real tables hold negative entries, 25 of
1,024 at k=2 in layer 03); fix 7 (train and qbench ids taken from, and checked against, the teacher cache; 17 of 29
qbench rows are shorter than 2,048 and were padded: 32.5 % of the scored positions); fix 8 (best state chosen on
the real-position KL).

## 9. Router distillation

Trainer flags: `--router-full --dense-reverse --freeze-scales --lr-router 5e-5 --lr-bias 5e-5 --adam-beta2 0.999
--macro-seqs 4 --chunk-seqs 4`: the full router gate matrices and the bias are trained with AdamW, tables and scales
frozen, 4 sequences of 2,048 tokens per step. Metric: the trainer's qbench evaluation on real positions (39,871
tokens), with the router in fp32. This is an internal measurement on the private rows.

| state | qbench KL, real positions | top-1 | router relative change | run |
|---|---|---|---|---|
| untuned | 0.29058 | 0.8371 | 0 | |
| 8 steps | 0.29002 | 0.8379 | 0.52 % | run 1 |
| 16 steps (64 sequences): the state that is shipped | 0.28524 | 0.8379 | 0.73 % | run 1 |
| 32 steps | 0.28931 | 0.8369 | 1.04 % | run 2 |
| 48 steps | 0.28746 | 0.8366 | | run 2 |

Source: `results/training/phase7_routerkd_run1.log` (run 1) and the work record (run 2).

Run 1: 183 to 185 s per step (forward 73 s, reverse 111 s for 4 sequences), peak VRAM 23.7 GiB, process 92.6 GiB.
Run 2 continued from the 16-step state (the AdamW moments restart from zero) and ended by itself after two
evaluations without a new best. Reading of the record: the gain from router-only distillation at these settings is
about 0.002 to 0.005 nats and fluctuates by about 0.004 between evaluations; the gap to the previously shipped
archive on this metric is 0.025. Not tried: lower rate, larger batch, warm-up, optimizer state across restarts,
weight averaging.

Export of the 16-step state to `merged_routers_bf16.pt` (99,168,981 bytes, 42 layers): 80.9 % of the router weights
change value after rounding to bf16, the cosine of the rounded change with the trained change is 0.976, norm ratio
1.027. (On the 1-step state of the earlier Stage 2B run only 9.7 % of the weights changed and the cosine was 0.45;
that concern does not apply to this state.)

Verdict from the public benchmarks (section 11): with the router 156 / 240, without 154 / 241: a net change of 1 item
in 414, equal within noise, no speed change. By the rule agreed beforehand (better or equal: may ship) the router is
shipped. The official private-set gate of archive + router is in section 15.

## 10. Scale probe (negative result)

On top of the 16-step router state (KL 0.28524): log-scale gradients from two disjoint halves of 8 sequences each.

| quantity | value |
|---|---|
| cosine between the halves, per (layer, projection) | 0.28 |
| cosine between the halves, per expert | -0.07 |
| cosine between the halves, per row (first 3 layers) | -0.01 |
| shared offsets per layer, largest offset 0.01 / 0.03 / 0.1 | KL 0.28713 / 0.28729 / 0.29512 |
| shared offsets per expert, largest offset 0.01 / 0.03 / 0.1 | KL 0.28745 / 0.28821 / 0.28582 |
| offsets kept | none (every tried offset made the KL worse than 0.28524) |

Source: `results/training/phase7_scaleprobe.log`. Conclusion of the record: no systematic scale error; row-scale
training was dropped.

## 11. Public benchmarks, head to head

Harness `bench/` (`bench_openai.py`, `compare.py`): HumanEval (164 problems) and a fixed, seeded sample of 250 GSM8K
test questions over an OpenAI-compatible endpoint. Identical for every model: greedy decoding (temperature 0),
thinking off, reasoning effort low, at most 1,024 new tokens, one request at a time, no prompt caching, the model's
own chat template. A truncated answer would count as wrong (there were none).

- This archive: served by `runtime/glm53_fold_generate.py --serve` with the production flags, kernel v2
  (`R39_QTIP_WARPS=16 R39_QTIP_PF=16`), expert pool 14 GiB, 59.95 GiB of RAM locked, MTP layer not loaded, and the
  output head restricted by `configs/r34_vocab_keep.json` (111,189 of 154,880 rows kept: Latin-script, Greek,
  mathematical, symbol, emoji and special tokens; CJK, Cyrillic, Arabic and other scripts dropped). "Router" = the
  16-step router file present in the archive folder (`patched_routers=42` in the server log).
- Comparison: `unsloth/GLM-5.3-Flash-GGUF`, `UD-IQ2_XXS` (4 files, 101,844,951,904 bytes), served by a prebuilt
  llama.cpp server build (`llama-server.exe`, CUDA 12) that supports the glm5-next architecture, started standalone:
  `-c 8192 -np 1 -t 8 -ngl 99 --n-cpu-moe 36 --jinja --reasoning-format deepseek --reasoning-budget -1
  --no-context-shift --cache-ram 0`; 27.2 GiB of VRAM in use, the experts of 36 blocks in RAM. It keeps all experts.
- Prompt difference: with thinking off the llama.cpp build renders `...<|assistant|><think>` without the closing
  tag, so its prompt is exactly 1 token shorter on every item, and the model emits the closing tag as its first
  token (0 thinking characters observed). Accepted as equivalent.

| | this archive, no router | this archive + router | Unsloth `UD-IQ2_XXS` |
|---|---|---|---|
| HumanEval (164) | 154 = 93.9 % (89.1-96.7) | 156 = 95.1 % (90.7-97.5) | 158 = 96.3 % (92.2-98.3) |
| GSM8K sample (250) | 241 = 96.4 % (93.3-98.1) | 240 = 96.0 % (92.8-97.8) | 239 = 95.6 % (92.3-97.5) |
| decode tok/s reported by the server, HumanEval / GSM8K | 20.97 / 23.81 | 20.90 / 23.82 | 12.26 / 12.94 |
| tok/s wall including the prompt (mean per item), HumanEval / GSM8K | 15.81 / 11.13 | 15.72 / 11.23 | 7.84 / 5.92 |
| mean completion tokens, HumanEval / GSM8K | 177 / 47 | 177 / 47 | 180 / 75 |
| truncated answers | 0 / 0 | 0 / 0 | 0 / 0 |
| size | 66.2 GB of expert files (76.5 GB with the backbone) | plus a 0.1 GB router file | 101.8 GB |

In brackets: Wilson 95 % intervals. Source: `results/benchmarks/new/`, `results/benchmarks/new_router16/`, `results/benchmarks/unsloth/`
(`summary.json` and `items.jsonl` per task).

Paired comparison (exact two-sided McNemar test):

| pair | task | both right | only first | only second | neither | p |
|---|---|---|---|---|---|---|
| no router vs Unsloth | HumanEval | 152 | 2 | 6 | 4 | 0.29 |
| no router vs Unsloth | GSM8K | 233 | 8 | 6 | 3 | 0.79 |
| router vs Unsloth | HumanEval | | 2 | 4 | | 0.69 |
| router vs Unsloth | GSM8K | | 7 | 6 | | 1.0 |
| no router vs router | HumanEval | | 1 | 3 | | 0.63 |
| no router vs router | GSM8K | | 2 | 1 | | 1.0 |

No difference is statistically significant. Empty cells are not in the record; `compare.py` prints them from the item
files. Ratios: decode 1.7 to 1.8 times the comparison build, wall 1.9 to 2.0 times.

A benchmark of the previously shipped TCQ33P5 archive with the same harness is to be added; it had not been run when
this file was written.

## 12. Served speed (HTTP server, the benchmark requests)

| configuration | items | HumanEval decode / wall tok/s | GSM8K decode / wall tok/s | source |
|---|---|---|---|---|
| pool 14 GiB, no router | 164 / 250 | 20.97 / 15.81 | 23.81 / 11.13 | `results/benchmarks/new/` |
| pool 14 GiB, router | 164 / 250 | 20.90 / 15.72 | 23.82 / 11.23 | `results/benchmarks/new_router16/` |
| pool 17 GiB + `R39T_FAST_DEMAND=2 R39T_PREEVICT=4 R39T_LAZY_STAGING=1`, no router | first 40 / 40 | 23.19 / 17.5 | 25.74 / 12.3 | `results/benchmarks/new_p17t/` |

Scores of the 40-item run: 38 of 40 and 39 of 40; no out-of-memory. A run at pool 18.47 GiB did not take place.
The first HTTP start of this archive (3 items) decoded at 20.8 tok/s at pool 14 GiB.

## 13. Speculative decoding with kernel v2 (no gain)

Production flags, kernel v2 (`R39_QTIP_WARPS=16 R39_QTIP_PF=16`), speed gate. In every MTP run the generated ids
AND the logit bits behind every generated token equal the plain reference.

| setting | tok/s | demand misses per token (decode) | free VRAM GiB |
|---|---|---|---|
| plain, pool 18 | 25.90 | - | 1.72 |
| plain, pool 16 (reference) | 24.19 | 52.5 | 3.24 |
| MTP exact, pool 16 | 24.33 | 73.4 | 2.78 |
| MTP + `R39_MTP_PF_ROWS=1`, pool 16 | 24.89 | 56.5 | 2.78 |
| same, `R37_POOL_REG_TOPK=8` / `4` | 24.00 / 24.61 | 47.0 / 70.2 | 2.78 |
| MTP + `R39_MTP_PF_ROWS=1`, pool 17.0 | 25.92 | 53.8 | 1.78 |
| MTP + `R39_MTP_PF_ROWS=1`, pool 17.5 | 25.99 | 52.4 | 1.28 |

Source: `results/mtp_kernel_v2/` (`mtp_reference.json`, `mtp_exact.json`, `mtp_pfrows*.json`, logs).

Draft acceptance 0.92, 1.83 tokens per verify step, and a verify step plus its draft costs 1.77 plain steps at pool
17.5. Lossless drafting at its best equals plain decoding (25.99 against 25.90), so it is not offered as a speed
feature. Side finding: plain decoding gains 7 % from 2 GiB more expert pool (24.19 -> 25.90). The further steps of
the test plan (`policy`, `batchsweep`, `kernel2`) were not run.

## 14. Where a decode step goes, and the copy-traffic flags

Split of a decode step, plain decoding, pool 18 GiB, ms per token (`results/copy_traffic/traffic_proftrace.json`):

| prompt | wall | GPU in graphs | stalled behind copies | GPU idle behind host code | demand misses | prefetched rows (used) |
|---|---|---|---|---|---|---|
| code | 44.05 | 27.22 | 16.77 | 0.03 | 58.2 | 74.5 (51.9) |
| capital | 36.06 | 25.92 | 10.04 | 0.06 | 35.2 | 42.5 (32.8) |
| math | 42.50 | 26.99 | 15.44 | 0.03 | 53.5 | 68.2 (46.8) |
| systems | 39.31 | 26.60 | 12.52 | 0.15 | 42.1 | 58.6 (47.9) |
| no-miss floor | 24.39 | 24.23 | 0 | 0.13 | 0 | 0 |

The no-miss floor is 24.4 ms = 41 tok/s; attention and router graphs are 14.2 ms of it, the expert kernels 4.7 ms.
The GPU waits on Python for 0.03 to 0.15 ms per token, so a native decode loop was not built. The stall is the copy
time of expert rows that could not be predicted. Fit of the record: ms per token = 21.65 + 0.227 x demand misses +
0.118 x prefetched rows.

Lossless changes in `runtime/glm53_fold_expert_pool_v6.py` (default off; ids and logit fingerprints identical in
every run):

| setting | result | source |
|---|---|---|
| `R39T_FAST_DEMAND=2` | +1.34 % (paired, inside one process) | `results/copy_traffic/traffic_pair_*.json` |
| `R39T_FAST_DEMAND=2 R39T_PREEVICT=4` | +1.61 % (paired) | same |
| those two + `R39T_LAZY_STAGING=1`, pool 18.47 GiB (80 more slots), against the reference at pool 18, alternating, quiet machine, fingerprint bookkeeping on | reference 25.07, 25.42; with flags 25.38, 26.45: mean 25.25 -> 25.92 = +2.7 % | `traffic_ab_best_A1/A2/B1/B2.json`, `traffic_ab.log` |
| the same without the bookkeeping | 26.09 and 25.78 (reference 25.29 and 25.23; these runs overlapped a large download) | `traffic_nofp_best_*.json`, `traffic_nofp_ref_*.json` |
| pool torture test with every new flag | 20 of 20 pass, 0 mismatches | `pool_torture_r39t.json`, `traffic_torture.log` |

Did not help: a looser prefetch gate, hit-first scheduling (-12 %), paced prefetch, other eviction policies (in
simulation, `traffic_sim_pool18.txt`). Ceilings by the fit: +1.2 % per 0.5 GiB of pool; an optimal eviction would
need 38.6 instead of 108 fetched rows per token (about 33 tok/s; no usable signal for it was found); backbone
floor 41 tok/s. The flags were measured for plain single-token decoding only, plus the one served run of section 12.

Hardware finding: on this machine the GPU's PCIe link is generation 4, width 16 (the GPU supports 5, the board's
slot 4). Expert copies run at about 22 to 25 GiB/s. By the fit a generation-5 slot would roughly halve the copy time:
about 32 tok/s. This is an estimate, not a measurement.

## 15. Final gate with the trained router

Archive + `merged_routers_bf16.pt`, production runtime, final code and settings.

| check | result |
|---|---|
| official private-set gate (29 rows, 39,842 positions), `patched_routers=42`, production flags, kernel v2 warps 16 load-ahead 16, copy-traffic flags, pool 16 GiB: KL / top-1 | 0.290724 / 0.83505 (`results/final_gate/final_gate_router16.json`, `.log`). Untuned, original runtime, same metric, pool 16: 0.295316 / 0.83658; router: KL -0.0046, top-1 -0.0015. Shipped TCQ33P5, same harness: 0.260544 / 0.84348. Package gate (<= 0.230, >= 0.846): not met |
| speed gate with the router, same settings, after the quality rows | 24.62 tok/s (code 22.21, capital 27.62, math 23.38, systems 25.25); the package's speed gate passes. Not directly comparable with the pool-18 figures (25.90 / 25.92) |
| bit-exactness on the final code (forced-decode logits, 3 prompts x 128 graph steps): original runtime vs this runtime with its flags off | 384 of 384 bitwise identical (`results/production_runtime/port_exact.log`) |
| the same, this runtime with the production flags | 384 of 384 (`port_exact.log`) |
| kernel v1 record vs kernel v2 record, both with the production flags | 384 of 384 (`results/production_runtime/kernel_exact.log`) |

The speculative-decoding (`[R3.9M]`) and copy-traffic (`[R3.9T]`) changes did not alter the decode path at default
settings.

## 16. Claims of the original package vs measurement

"Claim" = what the original package's README, runbook or specification states or forecasts (see
`docs/original_handoff/`).

| # | claim of the original package | status | measurement |
|---|---|---|---|
| 1 | The offline verifier passes all 9 gates | confirmed | reported passed before the fixes; on the patched tree gate 9 (file hashes) fails only on the patched files (section 1) |
| 2 | Five GPU self-tests pass as shipped | refuted | failed as shipped (register limit); pass after fix 1 (section 1) |
| 3 | Layer 27: `TCQ33_A_best` reproduces about 9.187 dB | confirmed | 9.188 dB |
| 4 | Layer 27: QZ arm at least +0.30 dB over TCQ33 (gate) | confirmed | +0.393 dB |
| 5 | Layer 27: QZ arm expected +0.45 to +0.75 dB | refuted | +0.393 dB |
| 6 | The cross-covariance target makes the `down_proj` output strictly better than the plain 2D arm | refuted | -0.035 dB routed output, -0.145 dB H-weighted down SNR at layer 27 |
| 7 | NVRTC GEMV cosine >= 0.99999 | confirmed | 1.0 at layer 27 and on every layer of the build |
| 8 | Layer 27: Sub-70 arm at least +0.60 dB over `L190_best` | confirmed | +0.905 dB (single layer only) |
| 9 | The Layer-27 gate takes about 12 minutes and the pipeline runs on 96 GB RAM | refuted | as shipped it committed 140 GiB and was stopped; after fix 3 the 2D arms alone took 25 minutes |
| 10 | The 42-layer build takes 2.6 to 3.1 hours | refuted | about 4.6 hours wall including shard downloads; the builder process holds about 106 GiB (peak 124 GiB), so it does not run on 96 GB |
| 11 | Rule A slab row is 6,326,528 bytes per expert | confirmed | every load reports 6326528 B/row |
| 12 | The flagship archive is 8.8 GB smaller than TCQ33P5 | confirmed | 66.19 GB against 75.02 GB of expert files |
| 13 | Flagship qbench KL <= 0.228 nats after Stage 2B (gate 0.230), top-1 >= 0.846 | refuted | without router 0.295316 / 0.83658; Stage 2B as shipped cannot run; router distillation reached 0.28524 at best on the trainer's metric (the target needs about 0.228); official gate with the router: 0.290724 / 0.83505 (section 15) |
| 14 | Stage 2B takes 45 to 65 minutes for 6 sweeps, about 11.8 GiB peak VRAM, under 32 GiB of host RAM, on 96 GB | refuted | with the shipped settings the first step drove the GPU to 32.1 GiB and free RAM to 2.7 GiB and was stopped; after the fixes one step takes 66 minutes (about 27 hours for 24 steps), VRAM peak 26.2 GiB, process 93 GiB |
| 15 | Stage 2B: the held-out KL decreases monotonically across early sweeps | refuted | step 1 -0.00014, step 2 +0.00076 (worse than untuned) |
| 16 | Decode speed 21.5 to 23.5 tok/s and the speed gate passes (in the package as delivered) | refuted | 16.1 to 16.6 tok/s in the package's runtime, gate fails |
| 17 | Decode speed 21.5 to 23.5 tok/s (after the port into the production runtime) | confirmed | 22.06 tok/s with kernel v1; 25.29 to 25.90 with kernel v2 |
| 18 | The runbook's flags `R36_FUSE_MHC`, `R36_FP8_GEMV` act in the package's runtime | refuted | the package's runtime modules are older and read neither flag; `R37_POOL_REG_TOPK=6` made the package's pool drop 2 of 8 predicted experts per layer by expert id |
| 19 | The qbench-only gate takes about 4 minutes | refuted | 45 minutes |
| 20 | After the first load an archive boots in under 0.3 s | refuted | 66 to 75 s with the sidecars present; first load 875 s |
| 21 | CUDA-graph replay is bit-identical to eager decode | confirmed at the level tested | the routed-expert self-test passes (0.0); at whole-model level graph and eager steps are different float paths (top-1 16 of 16, mean KL 1.6e-7 to 1.8e-5) |
| 22 | `R39_QTIP_WARPS=16 R39_QTIP_GX=32` is a bit-exact alternative worth benchmarking | speed: no gain; exactness of this setting not checked separately | 16.61 against 16.46 tok/s (original runtime); 21.83 against 22.06 (production runtime, kernel v1) |
| 23 | Sub-70 GB profile (69.97 GB, KL <= 0.253, 22 to 24 tok/s) | not tested | not built |
| 24 | Max-fidelity 83 GB profile (KL about 0.215 to 0.220) | not tested | not built |
| 25 | Round 3.8 control (TCQ33 with the W16 mask plus Stage 2B, KL <= 0.258) | not tested | its source layer files were not on disk |
| 26 | Teacher cache by the separate extraction script (6 to 18 minutes, about 92 MiB) | not tested as written | replaced by an exact teacher stream inside the build (`--teacher-out`, 239 MiB) |
| 27 | HTTP serving with 65k context on the QTIP archive | partly tested | the archive serves over HTTP (the benchmarks of section 11 ran that way, prompts up to a few hundred tokens); long contexts were not tested |
| 28 | Speculative decoding as a speed feature | refuted | lossless, and equal to plain decoding at best (section 13) |

## 17. Not measured

- A benchmark of the previously shipped TCQ33P5 archive (to be added).
- Quality on other public benchmarks or public text; thinking-on runs; sampling (the runtime decodes greedily).
- The `tests/` folder and the `build/` scripts were run against the original package's runtime modules. In this
  repository they resolve to `runtime/`; that combination has only been syntax-checked.
- The copy-traffic flags with speculative decoding or long contexts; long-context serving; any GPU other than the
  RTX 5090; a PCIe generation-5 slot.
