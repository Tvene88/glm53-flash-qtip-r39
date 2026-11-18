# How to reproduce

Commands are PowerShell, on Windows 11, one RTX 5090 (32 GiB), 128 GB RAM. They come from the working record
(`docs/WORK_RECORD.md`) and from the two GPU test scripts in `runtime/`. Where the record lists the flags of a run but
not its full command line, the command below is put together from those flags and says so.

Read this first:

- **Private data.** The build was calibrated on 64 C4 sequences and 64 sequences of recorded agent sessions
  (`cal_trace.json`), and the quality gate is scored on 29 held-out chat rows (`qbench_prompts.json`). The recorded
  sessions and the 29 rows are NOT part of this release, and neither are the files derived from them (the teacher
  cache `train_cal_top128.pt`, the exact teacher streams `evx.pt`, `qbench_teacher_top128.pt`). Without them, steps
  2 to 5 (build, training, export, private-set KL) can be run only with your own text in the same file formats, and
  will not reproduce the published numbers exactly.
- **The public path.** Everything else needs only the published model files: the speed test (step 6), the
  bit-exactness checks (step 7) and the HumanEval / GSM8K benchmarks with the head-to-head comparison (step 8).
  Step 9 is the recommended serving configuration.
- **Backbone directories.** The runtime reads the non-expert weights (attention, shared experts, routers, embeddings,
  head, tokenizer) from a "backbone" directory. Two exist: the slim serving backbone `GLM53-backbone-r34h`
  (10.29 GB, published with the model files) and a full-head backbone `GLM53-backbone` (used by the builder, the
  trainer and as the exact teacher of the quality gate). Both were made in earlier rounds by tools that are not in
  this repository.
- **One CUDA process at a time.** Every GPU step assumes an otherwise idle GPU (the scripts check for <= 4,096 MiB in
  use).
- **What was run against what.** The scripts in `build/` and `tests/` were run against the ORIGINAL package's runtime
  modules. In this repository they find `runtime/` instead (the production runtime with the port). That combination
  has been syntax-checked only. The paths inside `build/run_r39_master_pipeline.py` and `tests/verify_r39_package.py`
  were changed from `build_r39` to `build` for this layout; apart from that only machine paths in the `.ps1`
  scripts, folder names in the two benchmark drivers and a few comments were changed for the release.

## Variables

```powershell
$REPO   = (Get-Location).Path                      # root of this repository
$PY     = 'python'                                 # Python 3.14.0 with PyTorch 2.11.0.dev20260124+cu128 in the record
$MODELS = 'C:\models'
$FP8    = "$MODELS\GLM-5.3-Flash-UNCENSORED-FP8"   # source shards (62 safetensors files, 328 GB if all are on disk)
$BB     = "$MODELS\GLM53-backbone"                 # full-head backbone
$BBS    = "$MODELS\GLM53-backbone-r34h"            # slim serving backbone
$ARCH   = "$MODELS\GLM53-APEX-FOLD-R39-HYB-W16"    # the archive (output of step 2)
$ARCHX  = "$MODELS\GLM53-APEX-FOLD-R39-HYB-W16-RKD"   # export of the trained state (output of step 4; any new folder)
$P5     = "$MODELS\GLM53-APEX-FOLD-TCQ33P5"        # previously shipped archive, used as the control
$TEACH  = "$MODELS\r39_teacher_top128\train_cal_top128.pt"
$EVX    = "$MODELS\GLM53-APEX-FOLD-TCQ33\_r33_ckpt\evx.pt"   # exact teacher states of the 29 rows (private)
$CT     = "$MODELS\calib_text\exl3\cal_trace.json"            # calibration chat traces (private)
$C4     = "$MODELS\calib_text\en\c4-validation.00000-of-00008.json.gz"   # public C4 validation shard
$QB     = "$MODELS\calib_text\exl3\qbench_prompts.json"       # the 29 test rows (private)
$OUT    = "$REPO\out"
$STATE  = "$OUT\r39_w16_routerkd_state.pt"
New-Item -ItemType Directory -Force $OUT | Out-Null
$env:PYTHONUTF8 = '1'; $env:R37_GPU_WINDOW = '1'

function Clear-R { Get-ChildItem Env: | Where-Object { $_.Name -match '^R3[2-9]_' } | ForEach-Object { Remove-Item "Env:$($_.Name)" }
                   $env:PYTHONUTF8 = '1'; $env:R37_GPU_WINDOW = '1' }
function Set-Prod {   # the production flags of every production-runtime number in RESULTS.md
  Clear-R
  $env:R32_POOL = '6'; $env:R35_POOL_SIGNAL = '1'; $env:R35_POOL_DEMAND = '1'; $env:R35_POOL_DEMAND_STREAM = '1'
  $env:R35_POOL_GATHER = 'readfile'; $env:R35_POOL_EVICT = 'gdsf'; $env:R35_POOL_COLD_TOPK = '6'; $env:R35_POOL_COLD_MARGIN = '0.01'
  $env:R33_REGISTER_GIB = '68'; $env:R33_REGISTER_MODE = 'rows'
  $env:R36_FUSE_MHC = '1'; $env:R36_FP8_GEMV = 'v2'; $env:R37_POOL_REG_TOPK = '6'
}
```

Formats of the two private files, for anyone who substitutes their own text (read from the code, not tested with
substitutes): `qbench_prompts.json` is `{"rows": [{"input_ids": [...], "response_ids": [...], "conversation": 0..13,
"turn": int, "tools": bool}, ...]}`; the evaluator scores the response part of each row. The chat-trace file is read
by `build/run_qtip_build_r39.py` (`--ct-path`).

## 1. Get the base weights

- Source weights: `dealignai/GLM-5.3-Flash-UNCENSORED-FP8` on the Hugging Face Hub (MIT), the refusal-removed variant
  of `zai-org/GLM-5.3-Flash`. The build in the record did not keep the 328 GB on disk: `--fetch shards` downloads
  each shard when its layer is needed (about 5 minutes per layer) and evicts it afterwards. A complete download into
  `$FP8` plus `--keep-src` is the alternative the original runbook describes; it was not used.
- Backbone directories `$BB` and `$BBS`: see "Backbone directories" above. `$BBS` is published with the model files.
- Optional, for the Layer-27 gate only: the layer-27 expert weights (6.75 GiB, fetched by range requests in 5.2
  minutes in the record, with a tool that is not in this repository) and a layer-27 activation capture.

## 2. Build the archive

Checks before the long build, in the order the original package demands:

```powershell
$env:CUDA_VISIBLE_DEVICES = ''                      # offline verifier: no GPU
& $PY -u "$REPO\tests\verify_r39_package.py"
Remove-Item Env:CUDA_VISIBLE_DEVICES
& $PY -u "$REPO\build\run_r39_master_pipeline.py" --stages selftests     # five GPU self-tests

nvidia-smi -pl 450                                   # the record capped the GPU at 450 W for the encoder runs
# Layer-27 gate: 32-expert smoke run, then all 288 experts
& $PY -u "$REPO\build\r39_l27_qtip_2d_ldlq_lab.py" --layer 27 --score-split qb --cal-split ct --max-experts 32
& $PY -u "$REPO\build\r39_l27_qtip_2d_ldlq_lab.py" --layer 27 --score-split qb --cal-split ct `
    --out "$OUT\r39_l27_qtip_lab.json"
```

The verifier's manifest gate (gate 9) reports "skipped" here: this repository carries no `MANIFEST.sha256`. In the
record's second full Layer-27 run `--baseline-snr-db 9.188` was passed (the TCQ33 arm's value from the first run), so
the baseline arm was not repeated.

The build itself. The record ran it in three legs. The command is put together from the flags the record lists
(flagship, rule A, mask `hyb_W16_75GB.json`, `--fetch shards`, no `--keep-src`, `--old` = the shipped TCQ33P5 archive
for a same-harness comparison, `--teacher-out`; `--replace-old` is never passed) and agrees with the arguments the
builder stored for its first leg (`convert_progress.json` in the archive folder; every other option at its default:
64 C4 and 64 chat-trace calibration sequences of 2,048 tokens, `--n0 8192 --damp 0.10 --qz-damp 0.15`):

```powershell
$build = @("$REPO\build\run_qtip_build_r39.py", '--dst', $ARCH, '--old', $P5, '--src', $FP8, '--bb', $BB,
           '--rule', 'A', '--prune-spec', "$REPO\configs\masks\hyb_W16_75GB.json", '--fetch', 'shards',
           '--ct-path', $CT, '--c4-path', $C4, '--qb-path', $QB, '--teacher-out', $TEACH)
& $PY -u @build --stop-after 4                                          # leg 1: two layers, exits with code 3
& $PY -u @build --resume --shard-margin-gib 28 --floor-gib 14           # continue to the end
```

Measured: about 4.6 hours wall for all legs including the shard downloads; the builder process holds about 106 GiB
(peak 124 GiB); disk: 66 GB for the archive, 12.5 GB of stream checkpoints in `$ARCH\_r39_ckpt` (plus a temporary
copy while one is written), and the shards kept for later layers. Outputs: 43 `layer_XX_qtip_hyb.pt` files,
`fold_manifest.json`, `build\out_r32\r39_qtip_build_eval.json` (end-of-build KL and top-1), the teacher cache
`$TEACH`. Never delete `$ARCH` after an interruption; continue with `--resume`.

## 3. Router distillation

The trainer is `build/r39_stage2b_global_kl_trainer.py`. The original "Stage 2B" settings cannot run on this hardware
(RESULTS.md section 8). The mode that works trains the full router gate matrices and the bias only. Flags as listed in
the record; the path arguments are put together from the record's file names:

```powershell
$kd = @("$REPO\build\r39_stage2b_global_kl_trainer.py", '--backbone-dir', $BB, '--archive-dir', $ARCH,
        '--teacher-cache', $TEACH, '--save-state', $STATE,
        '--router-full', '--dense-reverse', '--freeze-scales', '--lr-router', '5e-5', '--lr-bias', '5e-5',
        '--adam-beta2', '0.999', '--macro-seqs', '4', '--chunk-seqs', '4')
# run 1: 16 steps of 4 sequences, evaluation every 8 steps (57 minutes in the record)
& $PY -u @kd --eval-every-macro 8 --max-macro-steps 16
# run 2: continue from the 16-step state, evaluation every 16 steps, up to 6 sweeps, stop after 2 evaluations
# without improvement (the AdamW moments restart from zero; they are not in the state file)
& $PY -u @kd --load-state $STATE --skip-macro 16 --eval-every-macro 16 --sweeps 6 --patience 2
```

No `--export-dir` is given during training. The best state (on the real-position qbench KL) is written to `$STATE`,
the latest one to `$STATE.last`. Measured: 183 to 185 s per step, peak VRAM 23.7 GiB, process 92.6 GiB. The record
kept a watchdog beside the trainer that stops it when free RAM runs out.

Not part of the recipe: training the row scales (dropped after the probe, RESULTS.md section 10; the probe is
`--probe-scales N`), and the original Stage 2B (`--sweeps 6 --train-seqs 128 --macro-seqs 32 --micro-bs 1
--window-size 4 --lora-rank 16`, which needs `--grad-window 1 --chunk-seqs 4` to run at all and then takes about 66
minutes per step).

## 4. Export

The published router file `merged_routers_bf16.pt` is the 16-step state of step 3 with the router weights
(base + trained change) rounded to bf16 and the bias in fp32, one pair of tensors per MoE layer (schema:
`docs/original_handoff/CONTRACTS_AND_SCHEMAS.md`, section 2.2). In the record it was written by a small CPU helper
script that is not in this repository. The trainer's own export does the same and more:

```powershell
& $PY -u @kd --load-state $STATE --export-only --export-dir $ARCHX
```

That command was not executed in the record. `$ARCHX` must be a new folder, never the source archive; the export
writes `merged_routers_bf16.pt` and a copy of every layer file (unchanged content for router-only training) and
links or copies the companion files of the source archive.

The runtime picks up `merged_routers_bf16.pt` automatically when it lies in the archive folder (`--fold-dir` /
`--fold`); `glm53_fold_qtip_runtime.py` and `quality_eval_r39s.py` also take `--merged-routers <file>`. A router
file changes the model: record whether it was present. The server log line `patched_routers=42` confirms it.

## 5. Quality gate

The official gate: 29 qbench rows, token by token through captured N=1 graphs, against the exact model with the full
head. About 45 minutes per archive. In the production runtime (this is `run_port_tests.ps1 -Step quality`):

```powershell
Set-Prod
& $PY -u "$REPO\runtime\quality_eval_r39s.py" --runner qtip --fold $ARCH `
    --backbone $BBS --exact-backbone $BB --evx $EVX --qbench $QB `
    --qbench-only --speed-gate --pool-gib 16 --out "$OUT\eval_qtip_qbench_pool16.json"
```

The package's thresholds are KL <= 0.230 and top-1 >= 0.846 (`--kl-budget 0.230` is the default). The router file
is used when it lies in `$ARCH`. `--runner tcq --fold $P5` scores the control (`run_port_tests.ps1 -Step control`).

The published gate numbers (0.295316 untuned, 0.260544 control) were measured with the ORIGINAL runtime and
`build/quality_eval_r39.py`, pool 16 GiB, 42 GiB registered, no production flags; the port is bit-identical to that
runtime for the QTIP archive (step 7), so the quality run was not repeated in the port. The full 111-row suite of the
evaluator (without `--qbench-only`) needs a further private data file and was not run.

## 6. Speed gate with the production flags and kernel v2

As run for RESULTS.md section 6 (the scripts write logs and reports into `runtime\out_r39s`):

```powershell
$env:R39_PYTHON = $PY                                # optional; default is python on PATH
& "$REPO\runtime\run_port_tests.ps1" -Step speed      # kernel v1, pool 18 GiB                      -> 22.06 tok/s
& "$REPO\runtime\run_port_tests.ps1" -Step speedab    # three A/B runs; the last one also writes the reference token ids
& "$REPO\runtime\run_kernel_tests.ps1" -Step speed    # kernel v2 default                           -> 25.29 tok/s
& "$REPO\runtime\run_kernel_tests.ps1" -Step speed -Extra 'R39_QTIP_WARPS=16,R39_QTIP_PF=8'  -Tag k2_w16pf8    # -> 25.32
& "$REPO\runtime\run_kernel_tests.ps1" -Step speed -Extra 'R39_QTIP_WARPS=16,R39_QTIP_PF=16' -Tag k2_w16pf16   # -> 25.90
```

The same as one direct command, which needs only the published files (archive folder and slim backbone): with
`--speed-only` the evaluator reads no test rows and opens no other backbone.

```powershell
Set-Prod; $env:R39_QTIP_KERNEL = 'v2'; $env:R39_QTIP_WARPS = '16'; $env:R39_QTIP_PF = '16'
& $PY -u "$REPO\runtime\quality_eval_r39s.py" --runner qtip --fold $ARCH --backbone $BBS `
    --speed-only --pool-gib 18 --out "$OUT\eval_qtip_speed_pool18_k2_w16pf16.json"

# with the three copy-traffic flags and 80 more pool slots (+2.7 % in the alternating A/B of RESULTS.md, section 14)
$env:R39T_FAST_DEMAND = '2'; $env:R39T_PREEVICT = '4'; $env:R39T_LAZY_STAGING = '1'
& $PY -u "$REPO\runtime\quality_eval_r39s.py" --runner qtip --fold $ARCH --backbone $BBS `
    --speed-only --pool-gib 18.47 --out "$OUT\eval_qtip_speed_pool1847_traffic.json"
```

For a speed claim use the alternating A/B of `run_traffic_tests.ps1 -Step ab` (single runs differ by about 1 % on a
quiet machine and by 10 % on a busy one); `-Step reference`, `-Step prof` and `-Step torture` reproduce the other
measurements of section 14, and `run_mtp_tests.ps1` (`reference`, `exact`, `variants`) those of section 13.

Notes:

- The scripts use fixed model paths under `C:\models` (top of each script); edit them if yours differ. The older
  scripts (`run_port_tests.ps1`, `run_kernel_tests.ps1`) still pass the private-file arguments, which a speed-only
  run ignores. `run_kernel_tests.ps1 -Step speed` passes `--compare-ids` with the report of the last `speedab` run,
  so run `speedab` first.
- A router file in the archive folder is loaded automatically. The published speed-test figures are for the archive
  without it, except where a row says otherwise.
- Control: `run_port_tests.ps1 -Step control` runs the TCQ33P5 archive with quality and speed at pool 16 GiB. The
  23.20 tok/s control of RESULTS.md was a speed-only run at pool 18 GiB with the same flags
  (`results/production_runtime/port_control_speed.log`).
- Speculative decoding: add `--mtp` and `$env:R35_MTP_GRAPHS = '1'` (pool 16 GiB in the record);
  `$env:R35_MTP_ROWS = 'batched'` selects the faster mode that is not lossless.
- The first load of an archive writes the `.slab` sidecars (66 GB, about 15 minutes); time the second load.

## 7. Bit-exactness checks

Self-tests that need no model files (GPU, seconds):

```powershell
Clear-R
& $PY -u "$REPO\runtime\glm53_fold_qtip_runtime.py" --selftest          # the original QTIP self-test
& $PY -u "$REPO\runtime\glm53_fold_qtip_runtime.py" --selftest-port     # port checks P1 to P5
& $PY -u "$REPO\runtime\glm53_fold_qtip_runtime.py" --selftest-kernel   # kernel v2 == v1 bitwise, synthetic slabs
```

These are `run_port_tests.ps1 -Step selftest` and `run_kernel_tests.ps1 -Step selftest`; the second also repeats the
kernel test on real rows (`--selftest-kernel --fold $ARCH`), and `-Step sweep` adds `--kbench-sweep`.

CPU-only pre-flight of the kernel sources (no GPU; it refuses to run unless CUDA is hidden):

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'; & $PY -I "$REPO\runtime\kernel_cpu_check_r39k.py"; Remove-Item Env:CUDA_VISIBLE_DEVICES
```

Kernel v1 against kernel v2 in the whole model (3 prompts x 128 forced graph steps, logits compared bitwise). The
record did this with `run_kernel_tests.ps1 -Step exact`, which reuses records made by `run_port_tests.ps1 -Step
exact`. The same check with this repository alone:

```powershell
Set-Prod                                             # kernel v1: record greedy
& $PY -u "$REPO\runtime\port_exact_r39s.py" record --fold $ARCH --backbone $BBS --pool-gib 16 --register-gib 68 `
    --out "$OUT\pe_v1.pt"
Set-Prod; $env:R39_QTIP_KERNEL = 'v2'                # kernel v2: replay the same inputs
& $PY -u "$REPO\runtime\port_exact_r39s.py" record --fold $ARCH --backbone $BBS --pool-gib 16 --register-gib 68 `
    --force "$OUT\pe_v1.pt" --out "$OUT\pe_v2.pt"
Clear-R; $env:CUDA_VISIBLE_DEVICES = '-1'            # compare on the CPU; exit code 0 only if all 384 steps are identical
& $PY -u "$REPO\runtime\port_exact_r39s.py" compare "$OUT\pe_v1.pt" "$OUT\pe_v2.pt"
Remove-Item Env:CUDA_VISIBLE_DEVICES
```

(The record forced both kernel records from the original runtime's record; forcing from the kernel v1 record, as
above, is the same comparison without the original runtime.)

Original runtime against the port (`run_port_tests.ps1 -Step exact`): this needs the ORIGINAL package's runtime
folder, which is not in this repository. With that folder at hand:

```powershell
$env:R39_OLD_RUNTIME = '<folder of the original package runtime>'
& "$REPO\runtime\run_port_tests.ps1" -Step exact
```

Result in the record: 384 of 384 bitwise identical, with the port's flags off and with the production flags
(`results/production_runtime/port_exact.log`).

## 8. Public benchmarks and the head-to-head comparison

Harness: `bench/` (standard library only; details and scoring rules in `bench/README.md`). HumanEval runs
model-written code on your machine without a sandbox. One server at a time on port 8188; nothing else may hold the
GPU.

**1. Serve and run.** Each driver starts one server, runs a 3-item smoke test, then HumanEval (164) and the GSM8K
sample (250), and stops the server. New runs are written to `bench\runs\<name>\<task>\`; the published runs are in
`results\benchmarks\`.

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'; & $PY "$REPO\bench\selfcheck.py"; Remove-Item Env:CUDA_VISIBLE_DEVICES   # harness self-test, CPU

# this archive (-Models = folder that holds GLM53-APEX-FOLD-R39-HYB-W16 and GLM53-backbone-r34h; default C:\models)
& "$REPO\bench\run_apex.ps1" -Kind new -Python $PY -Models $MODELS -Tag router16   # pool 14 GiB -> bench\runs\new_router16
# the router file is used when merged_routers_bf16.pt lies in the archive folder (server log: patched_routers=42);
# the published "new" run is the same command with the router file moved out of the folder and without -Tag

# comparison build: unsloth/GLM-5.3-Flash-GGUF, variant UD-IQ2_XXS (4 files, 101.8 GB)
$env:LLAMA_SERVER_DIR = '<folder of llama-server.exe>'   # a prebuilt llama.cpp server build that supports glm5-next
& "$REPO\bench\run_unsloth.ps1" -Python $PY -Gguf '<folder>\GLM-5.3-Flash-UD-IQ2_XXS-00001-of-00004.gguf'   # -> bench\runs\unsloth
```

The comparison server, as one generic command (what `run_unsloth.ps1` starts; `--reasoning-budget -1` is the
default of the published run):

```powershell
llama-server -m <folder>\GLM-5.3-Flash-UD-IQ2_XXS-00001-of-00004.gguf --host 127.0.0.1 --port 8188 `
    --alias unsloth-ud-iq2_xxs -c 8192 -np 1 -t 8 -ngl 99 --n-cpu-moe 36 --jinja --reasoning-format deepseek `
    --reasoning-budget -1 --no-context-shift --cache-ram 0
```

Notes on the llama.cpp server: the GGUF declares the architecture `glm5-next`, which older llama.cpp builds do not
know; the published run used a prebuilt llama.cpp server build (CUDA 12, Windows x64) that supports it, started
standalone. If the build needs extra runtime DLLs in another folder, pass `-LlamaVendor <folder>` (or set
`LLAMA_VENDOR_DIR`). With these settings it held about 27 GiB on the GPU and the experts of 36 blocks in RAM; nothing
else may be loaded. With thinking off this build renders the prompt without the closing `</think>` (1 token shorter on
every item); the driver prints the rendered prompt (`/apply-template`) so this can be checked.

**2. Compare.** `compare.py` runs on the published item files as they are (no GPU): it prints the score table, the
paired counts and the exact McNemar p of RESULTS.md, section 11.

```powershell
$R = "$REPO\results\benchmarks"
& $PY "$REPO\bench\compare.py" new="$R\new\humaneval" router16="$R\new_router16\humaneval" unsloth="$R\unsloth\humaneval"
& $PY "$REPO\bench\compare.py" new="$R\new\gsm8k"     router16="$R\new_router16\gsm8k"     unsloth="$R\unsloth\gsm8k"
```

Conditions that must be kept for a like-for-like run: greedy, thinking off, reasoning effort low, `--max-tokens
1024`, `cache_prompt: false`, one request at a time (the drivers set them). `run_apex.ps1` starts the server with
`--vocab-keep configs\r34_vocab_keep.json`, the restricted output head the published runs used. Do not run two
drivers at once: they share the port and the smoke-test folder.

Time in the record: this archive about 0.5 h for HumanEval and 0.3 h for GSM8K; the comparison build about 1 h and
0.8 h.

## 9. Recommended serving configuration

The served configuration with the highest measured decode rate (RESULTS.md, section 12): expert pool 17 GiB, the production
flags, `R39_QTIP_KERNEL=v2 R39_QTIP_WARPS=16 R39_QTIP_PF=16`, and the three copy-traffic flags
`R39T_FAST_DEMAND=2 R39T_PREEVICT=4 R39T_LAZY_STAGING=1`, with `merged_routers_bf16.pt` in the archive folder.

```powershell
& "$REPO\bench\run_apex.ps1" -Kind new -Python $PY -Models $MODELS -PoolGib 17 -Traffic -SmokeOnly   # starts, checks 3 items, stops
```

Speed test of the archive (no server; needs only the archive folder and the slim backbone: with `--speed-only`
`quality_eval_r39s.py` reads no test-row file and opens no exact backbone):

```powershell
Set-Prod; $env:R39_QTIP_KERNEL = 'v2'; $env:R39_QTIP_WARPS = '16'; $env:R39_QTIP_PF = '16'
$env:R39T_FAST_DEMAND = '2'; $env:R39T_PREEVICT = '4'; $env:R39T_LAZY_STAGING = '1'
& $PY -u "$REPO\runtime\quality_eval_r39s.py" --runner qtip --fold $ARCH --backbone $BBS `
    --speed-only --pool-gib 18.47 --out "$OUT\speed_final.json"
```

To keep a server running, start it by hand with the same environment and arguments (they are the lines of
`bench\run_apex.ps1`):

```powershell
Set-Prod
$env:R35_DSA_QBLOCK_MIB = '64'; $env:R37_DENSE_MIN_PAIRS = '1000000000'; $env:R37_EMPTY_CACHE = '1'
$env:R37_VRAM_CAP_MIB = '31400'; $env:R37_PREFILL_CHUNK = '2048'
$env:R36_LONGCTX = '1'; $env:R36_LONGCTX_MAX = '65536'; $env:R37_LONGCTX_FAST = '1'; $env:R37_DUAL_GRAPHS = '1'
$env:R39_QTIP_KERNEL = 'v2'; $env:R39_QTIP_WARPS = '16'; $env:R39_QTIP_PF = '16'
$env:R39T_FAST_DEMAND = '2'; $env:R39T_PREEVICT = '4'; $env:R39T_LAZY_STAGING = '1'
Set-Location "$REPO\runtime"
& $PY -u -W ignore "$REPO\runtime\glm53_fold_generate.py" `
    --backbone-dir $BBS --fold-dir $ARCH --backbone-mode fp8 --cold-mode mmap --vram-budget-gib 28 --expert-pool-gib 17 `
    --graphs --no-sync-timing --warm-cache --no-mtp-layer `
    --serve --port 8188 --serve-model-name apex-fold-r39-qtip --serve-early-token `
    --serve-max-context 65536 --serve-max-tokens 4096 --serve-state-gib 5 --serve-cache-entries 1 --serve-cache-points 2
```

Endpoint: `http://127.0.0.1:8188/v1/chat/completions` (OpenAI-compatible, greedy; `temperature`, `top_p` and `seed`
are accepted and ignored). Add `--vocab-keep "$REPO\configs\r34_vocab_keep.json"` for the restricted output head of
the benchmark runs; without it the full head is used (a served run with the full head is not in the record).

What was and was not measured for this configuration: 40 HumanEval and 40 GSM8K requests at pool 17 GiB with the
flags, no out-of-memory, 23.19 / 25.74 tok/s decode; the full benchmark sets ran at pool 14 GiB without the flags
(20.97 / 23.81). Prompts in these runs had up to a few hundred tokens; an earlier serving profile ran out of GPU
memory at pool 18 GiB on a 180-token prompt, which is why the server uses a smaller pool than the speed test. Long
contexts and speculative decoding were not tested with these flags.
