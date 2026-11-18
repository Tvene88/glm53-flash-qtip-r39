# bench: HumanEval and GSM8K over an OpenAI-compatible endpoint

A small harness that scores any model served at `/v1/chat/completions` on HumanEval (164 problems) and on a fixed,
seeded sample of GSM8K test questions (250), and compares several models on the same items. It produced the
head-to-head table of `RESULTS.md`, section 11.

| file | purpose |
|---|---|
| `bench_openai.py` | runs one task against one endpoint; writes `responses.jsonl`, `items.jsonl`, `summary.json` |
| `compare.py` | table of several runs, paired comparison, exact McNemar test |
| `selfcheck.py` | CPU-only test of extraction, scoring, resume and comparison (90 checks, fake server) |
| `prep_data.py` | downloads the two datasets again (needs `huggingface_hub`, `pyarrow`); not needed for a run |
| `data/` | the datasets as plain files, the fixed item orders, `SOURCES.md` with revisions and sha256 |
| `extra_body.json` | request fields added to every request (`cache_prompt: false`) |
| `preflight_body.json` | one request body for the prompt check |
| `run_apex.ps1` | serves this repository's archive, runs a 3-item smoke test and both tasks, stops the server |
| `run_unsloth.ps1` | the same for `unsloth/GLM-5.3-Flash-GGUF` `UD-IQ2_XXS` with a `llama-server` |
| `runs/` | written by new runs (not in the repository); the published runs are in `../results/benchmarks/` |

`bench_openai.py`, `compare.py` and `selfcheck.py` use the Python standard library only (Python 3.10 or later).
**The HumanEval task runs model-written code on your machine without a sandbox.**

## Published runs (`../results/benchmarks/<folder>/<task>/`)

| folder | model | served with |
|---|---|---|
| `new` | this archive, no router file | this runtime, expert pool 14 GiB |
| `new_router16` | this archive + `merged_routers_bf16.pt` (16-step router) | same |
| `new_p17t` | this archive, no router; first 40 items of each task | pool 17 GiB + the three copy-traffic flags (a speed run) |
| `unsloth` | `unsloth/GLM-5.3-Flash-GGUF`, `UD-IQ2_XXS` | a prebuilt llama.cpp `llama-server` build (CUDA 12) that supports glm5-next, `-ngl 99 --n-cpu-moe 36` |

The item files hold, per item: id, correctness, finish reason, token counts, timings, the kind of answer extraction
and a hash of the extracted code; `responses.jsonl` holds the raw JSON answer of the server for every item. In `summary.json`, `model` is the request's
model label (`new` for all three runs of this archive).

```powershell
$R = '..\results\benchmarks'
python compare.py new=$R\new\humaneval router16=$R\new_router16\humaneval unsloth=$R\unsloth\humaneval
python compare.py new=$R\new\gsm8k     router16=$R\new_router16\gsm8k     unsloth=$R\unsloth\gsm8k
```

## Running it

The two drivers do everything for one model: `run_apex.ps1 -Kind new -Python <python.exe>` and
`run_unsloth.ps1 -Python <python.exe> -LlamaDir <folder of llama-server.exe> -Gguf <first .gguf file>` (see `REPRODUCE.md`, section 8, for the conditions and the paths they
expect). By hand, against any endpoint:

```powershell
$Common = @('--base-url', 'http://127.0.0.1:8188/v1', '--max-tokens', '1024', '--temperature', '0',
            '--thinking', 'off', '--reasoning-effort', 'low', '--extra-body', '@extra_body.json', '--concurrency', '1')
python bench_openai.py --task humaneval --ids-file data\humaneval_order.json --model NAME --out runs\NAME\humaneval @Common
python bench_openai.py --task gsm8k --model NAME --out runs\NAME\gsm8k @Common
```

An interrupted run continues with the same command plus `--resume`. `--rescore` scores stored answers again without
sending anything. `--limit N` takes the first N ids of the fixed random order (seed 3901), so a subset is contained
in every full run.

## What is identical for all models, and what is not

Identical (the harness records them in `summary.json` and refuses `--resume` when they change): the prompt text (one
user message, no system message), the item set and order, `--max-tokens 1024`, temperature 0, thinking off
(`chat_template_kwargs.enable_thinking: false`), reasoning effort `low` (sent explicitly; the template prints it
into the system line), `cache_prompt: false`, one request at a time, each model's own GLM-5.3 chat template.

With thinking off the intended prompt is `[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>PROMPT<|assistant|><think></think>`.
The llama.cpp build used for the published run renders it without the closing `</think>` (one token shorter on every
item) and the model emits the closing tag as its first token; this was accepted as equivalent.

Model-specific: the serving command. This repository's server always decodes greedily and ignores sampling fields;
in the published runs its output head was restricted by `configs/r34_vocab_keep.json`. "tok/s reported" is each
server's own decode rate (`apex_fold.timing.decode_tok_s`; llama.cpp `timings.predicted_per_second`); "tok/s wall"
includes prompt processing. No server is given a seed.

## Scoring rules

- The final answer is `message.content` after any thinking: `reasoning_content` is ignored, text up to the last
  `</think>` in the content is removed, a `<think>` that never closes leaves an empty answer.
- HumanEval: fenced blocks tagged `python` / `py` / `python3` are candidates (untagged blocks only when there is
  no tagged one). The last candidate that defines the requested function is taken; if none does, the last
  candidate. A complete function is run after the imports and helper functions of the original prompt; a bare
  body is put under the original header. Then the canonical `check()` runs in a subprocess (`python -I`, new
  temporary directory, 30 s). Pass = exit code 0.
- GSM8K: the last `ANSWER: <number>` of the final answer (case-insensitive; `$`, thousands separators, `**` and
  a trailing unit are tolerated) must equal the reference number exactly as a decimal (`18` = `18.0`).
- `finish_reason: length` = wrong, counted as truncated.
- A request that fails is not scored: the run ends with exit code 3 and `complete: false`; repeat with `--resume`.
- Score interval: Wilson, 95 %. Pairs: both right / only A / only B / neither, exact two-sided McNemar p.
