# History of earlier rounds

The runtime in `runtime/` is the result of Rounds 3.2 to 3.8; its code comments carry tags such as `[R3.5-P]` or
`[R3.7]` and its behaviour is switched by environment variables named after the round that introduced them
(`R32_*` to `R39_*`).

Only one of the earlier rounds' notes is published here:

- `INTEGRATION_P.md`: how expert pool v6 (Round 3.5, `R32_POOL=6` and the `R35_POOL_*` flags) is wired into the
  generator and the graph decoder. The production flags used in `RESULTS.md` select this pool.

The change logs and gating notes of the other rounds are not part of this release. The flags that matter for
Round 3.9 are listed in `RESULTS.md` ("production flags") and in the header of
`runtime/glm53_fold_qtip_runtime.py`.
