# Original handoff documents: UNVERIFIED FORECASTS

> **Read this first.** The four documents in this folder are the ORIGINAL design documents of Round 3.9. They were
> written by an AI model (Gemini 4 Carbon) before anything was built or measured. Their numbers (quality targets,
> speeds, run times, memory budgets) are FORECASTS. Several of them were refuted when the work was done.
>
> **The measurements are in [`RESULTS.md`](../../RESULTS.md)**, including a table "Claims of the original package vs
> measurement" that lists each forecast as confirmed, refuted or not tested. Where these documents and `RESULTS.md`
> disagree, `RESULTS.md` is right.

They are kept because the design (the QTIP "HYB" trellis code, the 16x16 block LDLQ, the slab layout, the file
schemas) is described here and the code follows it, and because the record of what was changed
([`WORK_RECORD.md`](../WORK_RECORD.md)) refers to them.

| file | content | what to distrust |
|---|---|---|
| `AGENT_HANDOFF_RUNBOOK.md` | phase-by-phase commands of the original pipeline | every duration, every memory figure, the quality and speed expectations; folder names (`build_r39/` is `build/` here; its `runtime/` is not this repository's `runtime/`); several commands needed fixes or different flags (see `REPRODUCE.md`) |
| `ARCHITECTURE_AND_MATH_SPEC.md` | the trellis code, the Viterbi encoder, block LDLQ, the cross-covariance target, the Stage 2B trainer | the statement that the cross-covariance target "strictly improves" the `down_proj` output (refuted at layer 27); the Stage 2B memory and schedule figures |
| `CONTRACTS_AND_SCHEMAS.md` | slab byte layout, file schemas, runtime interface contracts | the "< 0.3 s" boot (measured: 66 to 75 s) |
| `TROUBLESHOOTING_AND_TUNING.md` | diagnostics and tuning hints | the 96 GB RAM assumption, Stage 2B figures |

Examples of forecasts that failed: qbench KL <= 0.228 after tuning (measured, untuned: 0.295316; the tuning stage as
written could not run); 21.5 to 23.5 tok/s in the package's own runtime (measured: 16.1 to 16.6); a 2.6 to 3.1 hour
build on 96 GB RAM (measured: about 4.6 hours, about 106 GiB of process memory); a 45 to 65 minute tuning stage
(measured after fixes: about 66 minutes per step, 24 steps planned).

Each file carries a one-paragraph banner at its top; apart from that banner the text is the original. The package's
PDF whitepaper, its README and its agent instruction file are not included.
