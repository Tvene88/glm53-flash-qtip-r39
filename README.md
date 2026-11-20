# GLM-5.3-Flash APEX-FOLD, Round 3.9 (QTIP HYB W16)

**GLM-5.3-Flash (320B) on one RTX 5090: about 1.7-1.8x the decode speed of the most-downloaded 2-bit GGUF build,
with expert weights about two-thirds its size, and statistically level HumanEval and GSM8K scores.**

Conditions and caveats of that sentence:

- **The comparison.** `unsloth/GLM-5.3-Flash-GGUF`, variant `UD-IQ2_XXS` (101.8 GB), on the same machine through the
  same benchmark harness (`bench/`). That build keeps all experts and ran under llama.cpp with about 27 GiB on the
  GPU (27.2 GiB measured) and the rest in RAM.
- **Speed.** Decode 20.90 / 23.82 tok/s (HumanEval / GSM8K answers, with the trained router, expert pool 14 GiB)
  against 12.26 / 12.94: 1.7 to 1.8 times. Including the prompt: 15.72 / 11.23 against 7.84 / 5.92 tok/s. One request
  at a time. With a larger serving pool (17 GiB plus the three copy-traffic flags, no router) decode reached 23.19 /
  25.74 tok/s on the first 40 items of each task, about 1.9 to 2.0 times the comparison build's full-set rates; that
  run covers 40 + 40 items only.
- **Size.** 66.2 GB is our expert files only, against the comparison's 101.8 GB (65 %). The runtime also needs a
  backbone directory of about 10 GB (10.29 GB) and the 0.1 GB router file, so the whole served set is 76.6 GB, closer
  to three-quarters (75 %).
- **What was given up.** 84 % of the routed experts are kept (1,922 of 12,096 pruned, 15.89 %), coded at about 2 bits
  per weight (2.0112 bpw).
- **Scores.** HumanEval (164): 156 against 158. GSM8K sample (250): 240 against 239. HumanEval leans to the comparison
  build by 2 problems (with the router; exact McNemar p = 0.69) and GSM8K to ours by 1 (p = 1.0); neither difference
  is significant. Greedy decoding, thinking off, reasoning effort low, at most 1,024 new tokens.
- **Prompt difference.** The llama.cpp build renders a prompt one token shorter on every item (it leaves the thinking
  block open; the model closes it with its first token). This was accepted as equivalent.
- **What it needs.** 128 GB of RAM (the runtime page-locks about 60 GiB: 59.95 GiB measured), one RTX 5090 class GPU
  with 32 GiB, Windows, and the custom runtime in this repository. The files are not GGUF and no other program reads
  them.
- **The base.** The weights are built from `dealignai/GLM-5.3-Flash-UNCENSORED-FP8`, a refusal-removed
  ("uncensored") variant of `zai-org/GLM-5.3-Flash`, not from the original model.
- **Benchmark server detail.** Our server ran with an output head restricted to Latin-script, Greek, mathematical,
  symbol and emoji tokens (`configs/r34_vocab_keep.json`: 111,189 of 154,880 rows; CJK, Cyrillic, Arabic and other
  scripts cannot be produced in that mode). The speed-test figures below use the full head.

Every number in this repository is copied from the working record (`docs/WORK_RECORD.md`) with its conditions. All
tables, with the report or log file behind each number, are in [`RESULTS.md`](RESULTS.md). Commands are in
[`REPRODUCE.md`](REPRODUCE.md).

## What this is

A single-GPU inference runtime and compression pipeline for the GLM-5.3-Flash mixture-of-experts language model
(320B parameters, 42 routed MoE layers plus one MTP layer, 288 routed experts per layer, 8 used per token).
Round 3.9 adds:

- an expert archive at about 2 bits per weight (QTIP trellis code "HYB", rule A, 2.0112 bpw, 6,326,528 bytes per
  expert) with 1,922 of the 12,096 trunk experts pruned (mask "W16"): 43 files, 66.19 GB;
- a router file trained by router-only distillation (16 steps), shipped with the archive;
- a port of the archive format into the fast production runtime of the earlier rounds, bit-identical to the original
  package's runtime (384 of 384 forced-decode logits);
- an optimised expert kernel ("kernel v2"), bit-identical to v1 (384 of 384), and three lossless changes to the
  host-to-GPU expert traffic.

## Head to head (same machine, same harness, same settings)

| | this archive + trained router | this archive, no router | Unsloth `UD-IQ2_XXS` |
|---|---|---|---|
| HumanEval, 164 problems | 156 = 95.1 % (90.7-97.5) | 154 = 93.9 % (89.1-96.7) | 158 = 96.3 % (92.2-98.3) |
| GSM8K, fixed sample of 250 | 240 = 96.0 % (92.8-97.8) | 241 = 96.4 % (93.3-98.1) | 239 = 95.6 % (92.3-97.5) |
| decode tok/s, HumanEval / GSM8K | 20.90 / 23.82 | 20.97 / 23.81 | 12.26 / 12.94 |
| tok/s including the prompt, HumanEval / GSM8K | 15.72 / 11.23 | 15.81 / 11.13 | 7.84 / 5.92 |
| size | 66.2 GB experts + 0.1 GB router + 10.3 GB backbone | 66.2 GB experts + 10.3 GB backbone | 101.8 GB |
| serving | this runtime, expert pool 14 GiB, 59.95 GiB of RAM locked | same | llama.cpp, about 27 GiB on the GPU |

In brackets: Wilson 95 % intervals. Paired results: with router against Unsloth, HumanEval 2 problems only ours
right, 4 only theirs (p = 0.69), GSM8K 7 against 6 (p = 1.0); without router against Unsloth, HumanEval 2 against 6
(p = 0.29), GSM8K 8 against 6 (p = 0.79); with against without router, HumanEval 3 against 1 (p = 0.63), GSM8K 1
against 2 (p = 1.0). No truncated answers in any run. The prompt of the llama.cpp build is one token shorter on
every item (it leaves the thinking block open and the model closes it with its first token); this was accepted as
equivalent. A benchmark of the previously shipped TCQ33P5 archive is to be added.

## Speed test (4 fixed prompts, 3 rounds, 128 new tokens each, greedy, single stream, full output head)

| configuration | tok/s overall | code | capital | math | systems |
|---|---|---|---|---|---|
| this archive (no router), kernel v2, `R39_QTIP_WARPS=16 R39_QTIP_PF=16`, pool 18 GiB | 25.90 | 23.66 | 28.69 | 24.54 | 26.70 |
| same with the three copy-traffic flags, pool 18.47 GiB | +2.7 % in an alternating A/B (25.25 -> 25.92 with bookkeeping on; 26.09 and 25.78 without) | | | | |
| this archive, kernel v2 default | 25.29 | 23.13 | 27.95 | 23.97 | 26.10 |
| this archive, kernel v1 | 22.06 | 20.65 | 23.74 | 21.20 | 22.65 |
| control: previously shipped TCQ33P5 archive (75.02 GB), same runtime and script | 23.20 | 20.74 | 25.84 | 22.03 | 24.19 |
| this archive + trained router, final settings: the copy-traffic flags, pool 16 GiB, measured after the quality rows | 24.62 | 22.21 | 27.62 | 23.38 | 25.25 |

Production flags, 59.95 GiB of RAM locked. The last row ran at pool 16 GiB and is not directly comparable with the
pool-18 figures above it (25.90 / 25.92). Served over HTTP the decode rate is lower because the server keeps a
smaller expert pool: 20.97 / 23.81 tok/s at pool 14 GiB, and 23.19 / 25.74 at pool 17 GiB with the copy-traffic
flags (first 40 items of each task).

## Internal quality measurement (private test set)

KL divergence to the exact FP8 model on 29 private chat rows, token by token (39,842 positions). The rows are not
published, so nobody else can check these numbers; they are given for completeness.

| archive | KL nats | top-1 agreement |
|---|---|---|
| this archive, no router | 0.295316 | 0.83658 |
| this archive + trained router (the release; production runtime, final settings, pool 16 GiB) | 0.290724 | 0.83505 |
| previously shipped TCQ33P5 archive, same harness | 0.260544 | 0.84348 |

The first and third rows were measured in the original package runtime, the second in this repository's runtime
(bit-identical to it on this archive). The original package's gate (KL <= 0.230, top-1 >= 0.846) is not met. On this
internal measure the new archive is worse than the previously shipped one; the router lowers the KL by 0.0046 and
the top-1 agreement by 0.0015. Router distillation moved the
trainer's own version of the metric from 0.29058 to 0.28524 (16 steps), then to 0.28931 and 0.28746 (32 and 48
steps); the 16-step state is the one shipped. In the benchmarks the router changes 1 item net out of 414, which is
within noise.

## What did not work

Round 3.9 started from a design package whose documents forecast its results. Several forecasts failed:

- The package's quality target (KL <= 0.228 on the private set after its "Stage 2B" tuning) was not reached.
  Stage 2B as shipped could not run on the target machine (its first step filled the GPU to 32.1 GiB and left
  2.7 GiB of host RAM free before it was stopped); after it was made to run, two steps of about 66 minutes each
  moved the score by -0.00014 and then +0.00076 nats.
- The package's speed forecast (21.5 to 23.5 tok/s) failed in the package's own runtime (16.1 to 16.6 tok/s). It was
  met only after the format was ported into the production runtime (22.06 tok/s).
- Speculative decoding with the model's MTP layer is lossless on this archive and gives no gain: at its best 25.99
  tok/s against 25.90 for plain decoding. It is not offered as a speed feature.
- Training the per-row scales was dropped: a probe found the scale gradients at this data size to be noise.
- A native (Rust or C++) decode loop was considered and not built: the GPU waits on host code for only 0.03 to 0.15
  ms per token. The stall is the copy time of experts that could not be predicted (10 to 17 ms per token).
- Build time, training time, load time and memory forecasts were all too low; see "Claims of the original package vs
  measurement" in `RESULTS.md`.

One hardware finding: on the test machine the GPU's PCIe link runs at generation 4 (the board's slot), not 5. By a
fit to the measurements a generation-5 slot would give about 32 tok/s. That is an estimate, not a measurement.

## Hardware and software

- One NVIDIA RTX 5090 class GPU with 32 GiB (all work was done on an RTX 5090; nothing smaller was tried).
- 128 GB RAM. With the production flags (`R33_REGISTER_GIB=68`) the runtime page-locks 59.95 GiB of host memory
  for this archive; the build needed about 106 GiB of process memory (peak 124 GiB) and the trainer about 93 GiB.
- Windows 11. The original package also lists WSL2; that path was not used in this work.
- Disk: 66.19 GB for the layer files plus the same again for the `.slab` sidecar files that the runtime writes next
  to them at the first load, plus 10.29 GB backbone (about 143 GB to serve).
- Python 3.14.0, PyTorch 2.11.0.dev20260124+cu128 (CUDA 12.8), NumPy 2.4.1, SciPy 1.16.2, tokenizers 0.23.3,
  as installed in the environment the measurements were made in; GPU driver 621.24 as recorded in the project's
  Round 3.7 history. The CUDA kernels are compiled at run time with NVRTC through `torch.cuda`.
  Other imports in the code: `jinja2` (chat template, HTTP server), `psutil`, `huggingface_hub` (shard download in the
  builder), `transformers` (a tokenizer path and one test; 5.16.1 installed).

## Repository layout

| folder | contents |
|---|---|
| `runtime/` | the runtime: `glm53_fold_*.py`, the evaluator `quality_eval_r39s.py`, the bit-exactness tool `port_exact_r39s.py`, CPU-only checks, and the GPU test scripts `run_port_tests.ps1`, `run_kernel_tests.ps1`, `run_mtp_tests.ps1`, `run_traffic_tests.ps1` |
| `bench/` | the HumanEval / GSM8K harness, the pinned data and the two serving drivers (the results of the four benchmark runs are in `results/benchmarks/`) |
| `build/` | the builder, the Layer-27 codec lab, the teacher extraction, the trainer (router distillation is a mode of it), pipeline scripts |
| `configs/` | prune masks (`masks/hyb_W16_75GB.json` is the one used), profiles, the output-head keep list of the benchmark server |
| `tests/` | the original package's offline checks |
| `results/` | the reports and logs behind every number (see `results/README.md`) |
| `docs/WORK_RECORD.md` | the full working record: every step, fix, measurement and negative result |
| `docs/original_handoff/` | the original design documents, unverified, for reference |
| `docs/SAMPLER.md` | the token sampler and thinking budget added on 2026-11-19: why, what changed, flags, tests, measurements |
| `docs/history/` | a note of an earlier round that the code refers to |

## Quick start

Model files are not in this repository. They are published as a Hugging Face model repository
(`https://huggingface.co/Tvene88/GLM-5.3-Flash-QTIP-W16-Uncensored`): the 43 `layer_XX_qtip_hyb.pt` files, `fold_manifest.json`,
`merged_routers_bf16.pt` and the backbone directory. Put the first three into one folder (the runtime loads the
router file by its name from the archive folder), for example:

```
C:\models\GLM53-APEX-FOLD-R39-HYB-W16\   layer files, fold_manifest.json, merged_routers_bf16.pt
C:\models\GLM53-backbone-r34h\           backbone
```

```powershell
# 1. GPU self-tests that need no model files (seconds to a minute each)
python runtime\glm53_fold_qtip_runtime.py --selftest
python runtime\glm53_fold_qtip_runtime.py --selftest-port
python runtime\glm53_fold_qtip_runtime.py --selftest-kernel

# 2. production environment
$env:PYTHONUTF8 = '1'; $env:R37_GPU_WINDOW = '1'
$env:R32_POOL = '6'; $env:R35_POOL_SIGNAL = '1'; $env:R35_POOL_DEMAND = '1'; $env:R35_POOL_DEMAND_STREAM = '1'
$env:R35_POOL_GATHER = 'readfile'; $env:R35_POOL_EVICT = 'gdsf'; $env:R35_POOL_COLD_TOPK = '6'; $env:R35_POOL_COLD_MARGIN = '0.01'
$env:R33_REGISTER_GIB = '68'; $env:R33_REGISTER_MODE = 'rows'
$env:R36_FUSE_MHC = '1'; $env:R36_FP8_GEMV = 'v2'; $env:R37_POOL_REG_TOPK = '6'
$env:R39_QTIP_KERNEL = 'v2'; $env:R39_QTIP_WARPS = '16'; $env:R39_QTIP_PF = '16'
$env:R39T_FAST_DEMAND = '2'; $env:R39T_PREEVICT = '4'; $env:R39T_LAZY_STAGING = '1'

# 3. speed test: needs only the two folders above
python runtime\quality_eval_r39s.py --runner qtip --fold C:\models\GLM53-APEX-FOLD-R39-HYB-W16 `
    --backbone C:\models\GLM53-backbone-r34h --speed-only --pool-gib 18 --out out\speed.json

# 4. serve an OpenAI-compatible endpoint with the recommended settings (REPRODUCE.md, section 9; the full benchmark
#    runs used -PoolGib 14 without -Traffic, section 8)
.\bench\run_apex.ps1 -Kind new -Python python -Models C:\models -PoolGib 17 -Traffic -SmokeOnly
```

The first load of an archive writes the `.slab` sidecars (about 15 minutes); later loads took 66 to 77 seconds.
Decoding is greedy by default. Since 2026-11-19 the server also samples (`--serve-temperature`, `--serve-top-p`,
`--serve-top-k`, per-request `temperature` / `top_p` / `top_k` / `seed`) and can cap the thinking block
(`--serve-thinking-budget`); see [`docs/SAMPLER.md`](docs/SAMPLER.md). For agent work with thinking on, use
`--serve-temperature 1.0 --serve-top-p 0.95 --serve-thinking-budget 6144`: greedy decoding there can fail to end
its thinking or fall into repetition. Every benchmark and speed figure here was measured greedy.

## Limitations

- Custom runtime on Windows. The archive format is not GGUF, not safetensors, and is read by nothing but `runtime/`.
- One request at a time. The MTP (speculative) paths decode greedily only. The sampler was added after the
  benchmarks, which were all run greedy (`docs/SAMPLER.md`).
- The two benchmarks are small (164 and 250 items) and both models score near the ceiling on them; "statistically
  level" means the differences are inside the noise of these tests, not that the models are equal.
- On the project's private KL measurement this archive is below the previously shipped TCQ33P5 archive
  (0.2953 against 0.2605 nats, no router). A benchmark of that archive is to be added.
- The private test rows and the calibration text of the build (recorded agent sessions) are not part of this release.
  The build, the training and the private KL cannot be reproduced by others; the speed test, the benchmarks and the
  bit-exactness checks can (see `REPRODUCE.md`).
- The copy-traffic flags were measured in the speed test and in one served run of 40 + 40 items, not with
  speculative decoding or long contexts. Long-context serving was not tested with this archive.
- Text only: the published backbone holds no vision tower.
- One machine, one GPU model, one operating system were measured.
- The base weights are the refusal-removed `dealignai` variant; the archive inherits that behaviour.

## Credits

- The Round 3.9 design and the original package (`docs/original_handoff/`) were written by an AI model, Gemini 4 Carbon,
  for the owner of this repository.
- The fixes to that package, the port into the production runtime, kernel v2, the copy-traffic work, the
  router-distillation path, the benchmark harness and all measurements were done in this project with Claude
  (Anthropic).
- Base model: `zai-org/GLM-5.3-Flash` (MIT). Weights used: `dealignai/GLM-5.3-Flash-UNCENSORED-FP8` (MIT).
- Comparison build: `unsloth/GLM-5.3-Flash-GGUF`. Benchmarks: HumanEval (`openai/openai_humaneval`) and GSM8K
  (`openai/gsm8k`), both MIT; revisions and checksums in `bench/data/SOURCES.md`.
- Methods this builds on are cited in `docs/original_handoff/ARCHITECTURE_AND_MATH_SPEC.md` and
  `docs/WORK_RECORD.md` (QTIP, QuIP#, GPTQ, router-only distillation reports, Tetra).

## Licence

MIT, see [`LICENSE`](LICENSE).
