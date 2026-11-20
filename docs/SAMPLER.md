# Token sampler and thinking budget (R3.9SMP, 2026-11-19)

Until this update the runtime decoded greedily only. The server accepted `temperature`, `top_p`, `top_k` and `seed`
and ignored them; the next token was `torch.argmax` inside the final CUDA graph and in the eager paths. This update
adds a real sampler and an optional cap on the length of the thinking block. With the default settings (temperature
0, no budget) the runtime behaves exactly as before, and every benchmark and speed figure in this repository was
measured that way.

## Why

Driven by a coding agent at reasoning effort high with thinking on, greedy decoding had two failure modes:

- **Thinking that cannot end.** The end-of-thinking token is often likely but rarely the single most likely token, so
  greedy decoding keeps choosing a continuation word. In one fresh run the first turn was a single thinking block of
  about 12,700 tokens (40,425 characters) that announced "let me write the files now" seven times and drafted the
  whole program twice before writing any file.
- **Repetition loops.** Once a passage has been repeated, repeating it again tends to become the argmax.

The vendor's generation defaults for GLM-5.3-Flash (`zai-org/GLM-5.3-Flash`, `generation_config.json`) are
temperature 1.0 and top_p 0.95.

## What changed (every change is tagged `[R3.9SMP]` in the code)

| file | change |
|---|---|
| `runtime/glm53_fold_sampler_r39.py` (new) | `Sampler.pick(row)`: branch-free and capturable in a CUDA graph. It computes the greedy token (the unmodified `argmax`, the old expression) **and** a sampled token, and a device flag selects between them; with the flag off the result is bit-identical to the old path. Sampling: the top 64 candidates by logit, temperature, nucleus (top_p) cut, optional smaller top_k, then one Gumbel-max draw. The random numbers are refilled by one small kernel before each replay of the final graph (`FinalReplay`), from a private generator, so a per-request seed works and no RNG op is captured. `R39_SAMPLER=0` captures exactly as before. |
| `runtime/glm53_fold_graphs.py` | the final graph segment uses the sampler; `capture()` creates it before the final graph and wraps that graph in `FinalReplay`. `GraphDecoder.decode(..., rewrite=None)`: an optional hook that may replace a decoded token before it is committed (used by the thinking budget). |
| `runtime/glm53_fold_backbone.py` | `pick_token()`: the sampler when it is on, the old `argmax_token` otherwise. |
| `runtime/glm53_fold_generate.py` | the first token after the prompt and the eager single-token loop use `pick_token` and the rewrite hook. The MTP (speculative) paths stay greedy. |
| `runtime/glm53_fold_serve.py` | `temperature`, `top_p`, `top_k`, `seed` parsed and validated (HTTP 400 on bad values) and passed to the runner per request; server defaults; thinking budget; the settings used are reported in the response (`apex_fold.sampling`, `apex_fold.thinking_budget`) and in `/health`. Internal one-token generations stay greedy; with `--mtp` the server forces greedy. |

## How to use

Server flags (all optional; the defaults keep the old behaviour):

| flag | default | meaning |
|---|---|---|
| `--serve-temperature T` | 0 | default temperature; 0 = greedy |
| `--serve-top-p P` | 1.0 | default nucleus cut |
| `--serve-top-k K` | 0 | default top-k (0 = off; at most 64 candidates are considered in any case) |
| `--serve-thinking-budget N` | 0 | after N reasoning tokens, the next token that contains a line break is replaced by `</think>` (at the latest 256 tokens later); 0 = off |

Per request: `temperature` (0 to 5), `top_p` (0 to 1, above 0), `top_k` (0 or more), `seed` (integer) override the
defaults; `"apex_fold": {"thinking_budget": N}` overrides the budget (0 = off for this request). The budget only
applies when thinking is on.

Settings used by the author for agent work: `--serve-temperature 1.0 --serve-top-p 0.95 --serve-thinking-budget 6144`.
Add them to the server command in `REPRODUCE.md`, section 9. Leave the defaults for benchmark runs.

## Tests

CPU only, no model files needed (run from `runtime/`):

| test | result |
|---|---|
| `python test_sampler_r39.py` (add `--cuda` on a GPU) | 18 of 18 with `--cuda`: greedy equals `argmax` on 600 rows in 3 dtypes with ties; sampled frequencies at T = 1 and T = 0.5 within 0.5 % of the target softmax; top_p and top_k cuts; same seed gives the same draws; capture in a CUDA graph, greedy replays equal `argmax`, sampled replays vary. The pick inside the graph costs 102 µs per token at the 154,880-token vocabulary. |
| `python test_serve_sampling_r39.py` | 12 of 12: defaults, overrides, invalid bodies answered with 400, hand-off to the runner, `/health`, default engine greedy |
| `python test_serve_budget_r39.py` | 13 of 13: `</think>` forced at a line break, the rest becomes the answer, per-request override, thinking off means no budget, a negative budget is rejected |

`tests/test_serve_r35.py` was updated: two of its checks asserted that `temperature` is ignored. Its check
`test_template_python_vs_jinja2` fails with and without this update (it needs a chat-template file that is not in the
repository at the path it looks for).

## Measured on the live model (this archive with its router, one RTX 5090)

- **Temperature 0 is unchanged.** On 4 fixed prompts, including one that produced 4,000 tokens, the new runtime at
  temperature 0 gave output identical token for token to the old runtime. Decode rate 19.62 against 19.73 tok/s
  (a serving profile with a 524,288-token context window, not the speed-test settings of `RESULTS.md`).
- **Default sampling** (temperature 1.0, top_p 0.95, two seeds): answers coherent and correct.
- **Thinking budget.** On a prompt where all 4,000 tokens went to thinking (greedy and sampled alike), a budget of
  1,500 ended the thinking at token 1,529 and an answer of about 11,300 characters followed.
- **Agent runs.** Fresh coding tasks driven by an agent harness (reasoning effort high, thinking on); the tests the
  model wrote were re-run independently afterwards:

| run | task | settings | its tests | wall time | first-turn thinking |
|---|---|---|---|---|---|
| 1 | key-value store | greedy (old runtime) | 13 of 13 pass | 23 min | 40,425 characters, 2 full drafts |
| 2 | key-value store | sampling | 20 of 20 | 33 min | 17,475 characters |
| 3 | job queue | sampling | 48 of 48 | 55 min | 40,182 characters, 2 full drafts |
| 4 | job queue | sampling + budget 6,144 | 24 of 24 | 30 min | 25,718 characters (budget reached at 6,159 tokens) |

Runs 2 to 4 also used fixes in the agent harness itself (a detector for long repeated passages, a planning-state
bug) that are not part of this repository. No run showed a
repeated identical tool call. Four runs are a small sample: they show that sampling alone does not stop over-long
planning at high effort (run 3) and that the budget does (run 4); they do not measure the rate of repetition loops.

## Limits

- The MTP / speculative decoding paths remain greedy (`--mtp` forces temperature 0).
- The budget also cuts reasoning that genuinely needs more than N tokens; raise or disable it per request.
- Sampled output is reproducible only with a fixed `seed` and the same settings.
