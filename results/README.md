# Result files

The reports and logs behind the numbers in [`../RESULTS.md`](../RESULTS.md). They are the original files with these
changes only:

- machine paths are replaced by placeholders: `<PKG_ROOT>` is the working tree the project lived in, `<PYTHON>` the
  Python interpreter of its environment. Inside `<PKG_ROOT>`, `claude_r39_speed` is this repository's `runtime/`,
  `claude_r39_qtip\build_r39` is `build/`, and `claude_r39_qtip\runtime` is the original package's runtime (not in
  this repository);
- one library warning about unauthenticated Hub downloads was dropped from `build/r39_build_prefetch.log`.

Paths under `C:\models` are the model folders of the machine the measurements were made on. Speed reports contain token
ids of generated text for the four public speed prompts, never text; quality reports contain per-row metrics only. The
only generated text in this folder is in `benchmarks/*/*/responses.jsonl`: the answers to the public HumanEval and
GSM8K prompts.

| folder | what | RESULTS.md section |
|---|---|---|
| `selftests/` | offline verifier, the five GPU self-tests (as shipped and after the fixes), trainer self-tests | 1 |
| `layer27/` | Layer-27 codec lab: smoke runs (32 experts), full gate (288 experts), JSON reports, the range download log | 2 |
| `build/` | the three build legs, the builder's own log, the shard prefetch log, the end-of-build report, builder memory | 3 |
| `original_runtime_eval/` | quality gate and speed gate in the original package runtime, both archives (`eval_*.json`, `phase8_*.log`), memory during the runs | 4 |
| `production_runtime/` | port self-tests and bit-exactness (`port_selftest.log`, `port_exact.log`), speed runs (`port_speed.log`, `port_control_speed.log`, `port_speedab.log`, `eval_*.json`), speculative decoding (`port_mtp.log`, `eval_qtip_speed_mtp*.json`), kernel v2 (`kernel_selftest.log`, `kernel_exact.log`, `kernel_speed.log`, `kernel_sweep.log`, `eval_qtip_w16_speed_pool18_k2*.json`) | 5, 6, 7 |
| `training/` | Stage 2B as shipped and with the fixes (`phase7_stage2b_*.log`), router distillation run 1 (`phase7_routerkd_run1.log`), scale probe, trainer memory | 8, 9, 10 |
| `mtp_kernel_v2/` | speculative decoding with kernel v2: plain reference, exact drafting and its variants (`mtp_*.json`, logs) | 13 |
| `copy_traffic/` | decode-step split (`traffic_proftrace.json`), reference run, paired and alternating A/B runs of the copy-traffic flags, pool torture test, eviction simulation table | 14 |
| `final_gate/` | the final private-set gate and speed gate of the archive with its router (`final_gate_router16.log`, `.json`); the bit-exactness re-run on the final code is `production_runtime/port_exact.log` and `kernel_exact.log` | 15 |
| `benchmarks/` | HumanEval and GSM8K runs: per run (`new` = no router, `new_router16` = with the router, `new_p17t` = 40-item speed run at pool 17 GiB, `unsloth` = comparison build) and task: `summary.json`, `items.jsonl` (per-item scores) and `responses.jsonl` (the servers' raw JSON answers to the public benchmark prompts); `server_new_router16.log` and `server_unsloth.log` are the server logs of the router run and of the full comparison run | 11, 12 |

The benchmark files are written by `bench/bench_openai.py`; `bench/compare.py` reads them, for example
`python bench\compare.py new=results\benchmarks\new\humaneval router16=results\benchmarks\new_router16\humaneval unsloth=results\benchmarks\unsloth\humaneval`.

Not included: tensors and checkpoints (`*.pt`), the per-second GPU telemetry (3.6 MB), the pipeline ledger, helper
scripts of the session, the log of router-distillation run 2, the pool call trace (11 MB) and the exploratory
copy-traffic runs that did not help.

Reading the speed reports: `speed_gate.per_prompt_mean_tok_s` and `overall_mean_tok_s` are the published figures;
`speed_gate.env` is the `R3x_*` environment of the run; `runtime.registered_gib` is the page-locked host memory;
`runtime.pool_slots` the number of expert slots in VRAM. In the `original_runtime_eval` reports the `rows` list holds
one entry per qbench row (`kl_exact` is the sum over its `n_exact` positions).
