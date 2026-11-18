> [R3.5-INT 2026-11-10] STATUS: APPLIED to claude_r35_speed by the integration step - sections 1a/1b/1c (R32_POOL=6), 2a-2e (scores, incl. the N = 2 set), 3a/3b (eager scores), 4a-4e (R35_PREGATE=1) and 5 (item 10b, flag R35_PREFILL_COLD=1) as written here, with M1's N = 2 set in place. Details, deviations and the gating: CHANGES_R35.md / GATING_R35.md (INT sections).

# INTEGRATION_P - wiring expert pool v6 (package P, speed item 3) into the runtime

Package P owns only `glm53_fold_expert_pool_v6.py` (+ build_run tests / bench). Every edit another file needs is listed
here as an exact, self-contained instruction for the integration step: file, function, anchor text, old -> new.
Anchors are quoted from the R3.5 base snapshot (`claude_r35_speed_base/`, 2026-11-10 04:43); package M1 has since added
an N = 2 set to `glm53_fold_graphs.py` - locate code by function name and the quoted lines, not by line number.

Ground rules kept by every edit below:
- Nothing changes unless it is switched on: pool v6 only with `R32_POOL=6`; each v6 feature only with its own
  `R35_POOL_*` flag (all default off = pool v5 behaviour); the layer-3 pre-gate only with `R35_PREGATE=1`; the score
  plumbing for (e) only when `pool.wants_scores` is True (i.e. `R35_POOL_COLD_TOPK` or `R35_POOL_COLD_MARGIN` set).
- Pools v1-v5 keep working: new pool features are reached through `getattr(pool, "<feature>", False)` checks.
- Pool changes move bytes only, so model outputs stay bit-identical (gate: forced replay v5 vs v6, P_GATING.md G1).

---------------------------------------------------------------------------------------------------------------------
## 1. generate.py - select pool v6 (REQUIRED for any v6 test)

### 1a. `ApexFoldMoEModelRunner.__init__`, pool selector
Anchor (base):
```python
            elif pv == "5":
                from glm53_fold_expert_pool_v5 import ExpertPool  # [R3.3-S2] registered page-cache tier
```
New (insert directly after those two lines, before `elif pv == "4":`):
```python
            elif pv == "6":
                from glm53_fold_expert_pool_v6 import ExpertPool  # [R3.5-P] v5 + speed item 3, flags R35_POOL_*
```
Why: today `R32_POOL=6` falls into the final `else:` and silently loads pool **v3**.

### 1b. same function, popularity-order assertion
Old:
```python
            assert not row_of or pv in ("2", "5"), "archives with a popularity order need R32_POOL=2 or 5"
```
New:
```python
            assert not row_of or pv in ("2", "5", "6"), "archives with a popularity order need R32_POOL=2, 5 or 6"
```

### 1c. same function, load log (optional, recommended for the gating logs)
Anchor:
```python
            print(f"[load] [R3.2] global expert pool: {n_slots} slots ({self.expert_pool_gib:.2f} GiB), "
                  f"prefetch={self.prefetch}", flush=True)
```
Insert right after it:
```python
            if hasattr(self.pool, "flags_str"):                       # [R3.5-P] pool v6 switches of this run
                print(f"[load] [R3.5-P] pool v6: {self.pool.flags_str()}", flush=True)
```
No constructor change is needed: the existing call (`files=`, `row_of=`, `register_gib=`, `host_cache_gib=`) works
unchanged; v6 reads its flags from the environment (new constructor args `flags=`, `debug=` are optional).

---------------------------------------------------------------------------------------------------------------------
## 2. graphs.py - (e) confidence scores for the cold-prefetch gate (needed only to A/B flag (e))

The N = 1 prediction graph p{l} must also copy the predicted experts' router selection scores (sigmoid + bias, the
values the top-k ranks) and the 9th-best score (margin reference) to pinned memory; the host step passes them to the
pool. Prediction only: routing, outputs and exactness are untouched.

### 2a. `GraphDecoder._alloc_state` - new pinned mailboxes
Anchor:
```python
        self.ids_np, self.pred_np = self.ids_pin.numpy(), self.pred_pin.numpy()
```
Insert right after it:
```python
        # [R3.5-P (e)] prediction scores + 9th-best score (margin reference), only when the pool gates on them
        self.pred_scores = bool(getattr(self.pool, "wants_scores", False))
        if self.pred_scores:
            self.pred_sc_pin = torch.zeros((self.n_layers, Kx), dtype=torch.float32, pin_memory=True)
            self.pred_ref_pin = torch.zeros((self.n_layers,), dtype=torch.float32, pin_memory=True)
            self.pred_sc_np, self.pred_ref_np = self.pred_sc_pin.numpy(), self.pred_ref_pin.numpy()
```

### 2b. new method in `GraphDecoder` (put it next to `_seg_p`)
```python
    def _route_ids9(self, l, x_flat):
        """[R3.5-P (e)] next-layer prediction: bb.route_ids' ops with a top-9 (sorted): the 8 predicted ids, their
        selection scores and the 9th-best score (margin reference for pool v6's cold-prefetch gate)."""
        w = self.bb.layer(l)
        logits = F.linear(x_flat.float(), w["gate_w"].float())
        choices = torch.sigmoid(logits) + w["gate_b"].float().unsqueeze(0)
        sel, idx = torch.topk(choices, k=self.cfg.num_experts_per_tok + 1, dim=-1, sorted=True)
        return idx[:, :-1], sel[:, :-1], sel[:, -1]
```
(Masked experts have gate_b = -inf, so they never enter the 8 ids; a -inf 9th score only makes margins infinite.)

### 2c. `GraphDecoder._seg_p` - copy scores when wanted
Old:
```python
        if nxt is not None:
            pred = bb.route_ids(nxt, a["x"])[0]                               # [R3.4-2c] ids only
            self.pred_pin[l].copy_(pred.view(-1), non_blocking=True)
```
New:
```python
        if nxt is not None and self.pred_scores:                          # [R3.5-P (e)] ids + scores + 9th best
            pred, sel, ref = self._route_ids9(nxt, a["x"])
            self.pred_pin[l].copy_(pred.view(-1), non_blocking=True)
            self.pred_sc_pin[l].copy_(sel.view(-1), non_blocking=True)
            self.pred_ref_pin[l:l + 1].copy_(ref.view(-1), non_blocking=True)
        elif nxt is not None:
            pred = bb.route_ids(nxt, a["x"])[0]                               # [R3.4-2c] ids only
            self.pred_pin[l].copy_(pred.view(-1), non_blocking=True)
```

### 2d. `GraphDecoder.step` - hand the scores to the pool
Old:
```python
            if nl is not None and pf:
                ev_p.synchronize()                # p{l} done -> predicted ids in the pinned mailbox
                pool.prefetch(nl, sorted(set(self.pred_np[l].tolist() if frozen is None else frozen[nl])))
```
New:
```python
            if nl is not None and pf:
                ev_p.synchronize()                # p{l} done -> predicted ids in the pinned mailbox
                if self.pred_scores and frozen is None:                   # [R3.5-P (e)] scored, confidence order
                    pool.prefetch_scored(nl, self.pred_np[l].tolist(), self.pred_sc_np[l].tolist(),
                                         float(self.pred_ref_np[l]))
                else:
                    pool.prefetch(nl, sorted(set(self.pred_np[l].tolist() if frozen is None else frozen[nl])))
```
Use `prefetch_scored()` (not `prefetch(..., scores=)`): build_run/r34_shrink_opts.install_mask_check wraps
`pool.prefetch` with a function that forwards only `(l, ids, protect)`; prefetch_scored() hands the scores past it.

### 2e. (optional) the N = 2 set of package M1 (`step2`, `_seg_p2`)
Same pattern: in `_seg_p2` compute `_route_ids9` per row (row-exact path) and copy the 16 scores + 2 references into
pinned rows; in `step2` replace the prefetch by
`pool.prefetch_scored(nl, pred16, sc16, [ref_row0] * 8 + [ref_row1] * 8, protect=[(l, int(e)) for e in set(ids)])`
with `pred16` = the two rows' 8 ids each, NOT deduplicated (the pool keeps the best margin per expert and queues the
distinct experts in confidence order). Only when `self.pred_scores`; otherwise keep M1's code.

---------------------------------------------------------------------------------------------------------------------
## 3. generate.py - (e) scores on the eager path (optional; the graph path is the production decode)

### 3a. `ApexFoldMoEModelRunner.__init__`
Anchor:
```python
            self._pred_host = torch.empty((self.cfg.num_experts_per_tok * 4,), dtype=torch.int64, pin_memory=True)
```
Insert after it:
```python
            self._pred_sc_host = torch.empty((self.cfg.num_experts_per_tok * 4,), dtype=torch.float32,
                                             pin_memory=True)                     # [R3.5-P (e)]
```
### 3b. `moe_dispatch` - whole new body (the lines marked [R3.5-P (e)] are the only changes)
```python
    def moe_dispatch(self, l: int, x_flat: torch.Tensor) -> torch.Tensor:
        if self.sync_timing and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        top_w, top_i = self.bb.route(l, x_flat)
        pred_ev = None
        want_sc = False                                                     # [R3.5-P (e)]
        nxt = self.next_moe.get(l)
        if self.pool is not None and self.prefetch and nxt is not None and x_flat.shape[0] <= 4:
            # [R3.2-1.6] pre-gating: the next layer's router on this layer's FFN input, copied to pinned host
            # memory BEFORE this layer's expert kernels are queued (reading it later does not wait for them)
            want_sc = bool(getattr(self.pool, "wants_scores", False))      # [R3.5-P (e)]
            if want_sc:
                pid_, psel_ = self.bb.route_ids(nxt, x_flat)                # [R3.5-P (e)] route()'s ids + scores
                pred = pid_.reshape(-1)
            else:
                pred = self.bb.route(nxt, x_flat)[1].reshape(-1)
            n_pred = pred.numel()
            self._pred_host[:n_pred].copy_(pred, non_blocking=True)
            if want_sc:                                                     # [R3.5-P (e)]
                self._pred_sc_host[:n_pred].copy_(psel_.reshape(-1).float(), non_blocking=True)
            pred_ev = torch.cuda.Event()
            pred_ev.record()
        out = self.moe_layers[l].forward_routed_experts(
            x_flat, top_w, top_i, swiglu_limit=self.cfg.swiglu_limit, hobbit_tau=self.hobbit_tau
        )
        if pred_ev is not None:
            pred_ev.synchronize()
            if want_sc:                                                     # [R3.5-P (e)] gated cold prefetch
                self.pool.prefetch_scored(nxt, self._pred_host[:n_pred].tolist(),
                                          self._pred_sc_host[:n_pred].tolist())
            else:
                ids = sorted(set(self._pred_host[:n_pred].tolist()))
                self.pool.prefetch(nxt, ids)
        if self.sync_timing and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.moe_time_s += time.perf_counter() - t0
        return out
```
(No 9th-best reference on the eager path: only R35_POOL_COLD_TOPK applies there; the margin gate needs the reference.
route() and route_ids() run the same ops up to the top-k, so the predicted ids are the same as today.)

---------------------------------------------------------------------------------------------------------------------
## 4. graphs.py - (f) layer-3 pre-gate (R35_PREGATE=1; measure recall first: P_GATING.md G5)

The first MoE layer (3) has no prefetch source. With the flag, the last dense layer before it (2) is captured as TWO
graphs: `dA2` = attention + FFN prepare + **route_ids(3, FFN input of layer 2)** + D2H of the 8 ids, and `d2` = the
dense MLP + merge (same kernels as today's d2, so outputs stay bit-identical; `d2` keeps its name and its output entry,
so exact_gate's `--perturb-layer 2` and replay_bench's M1 / M1EV passes keep working). step() records an event after
dA2 and, once a3 and p3 are queued, waits for it and prefetches layer 3 - the worker gets d2 + a3 + p3 of lead time.

### 4a. `GraphDecoder.__init__` - flag (append after the R34_BLOCKING_SYNC line)
```python
        self.pregate = os.environ.get("R35_PREGATE", "0") == "1"   # [R3.5-P (f)] layer-3 pre-gate from dense layer 2
        self._pg = None                                           # (source dense layer, first MoE layer) once captured
        self._pg_graph = None
```
### 4b. `GraphDecoder._alloc_state` - mailbox + event (append at the end of the method)
```python
        if self.pregate:                                                       # [R3.5-P (f)]
            self.pg_pin = torch.zeros((self.cfg.num_experts_per_tok,), dtype=torch.long, pin_memory=True)
            self.pg_np = self.pg_pin.numpy()
            self.ev_pg = torch.cuda.Event(blocking=self.blocking_sync)
```
### 4c. new segment methods (next to `_seg_dense`)
```python
    def _seg_dense_a(self, l, h, pro, first):
        """[R3.5-P (f)] first half of dense layer l: attention + FFN prepare, plus the pre-gate of MoE layer `first`
        (its router on this FFN input) and the D2H of the 8 predicted ids."""
        bb = self.bb
        if pro:
            self._embed_prologue()
        h = self._attn(l, h)
        post, comb, x_ffn = bb.ffn_prepare(l, h)
        pred = bb.route_ids(first, x_ffn.reshape(1, -1))[0]
        self.pg_pin.copy_(pred.view(-1), non_blocking=True)
        return {"h": h, "post": post, "comb": comb, "x": x_ffn}

    def _seg_dense_b(self, l, d):
        """[R3.5-P (f)] second half of dense layer l: the dense MLP + merge (the rest of _seg_dense, same kernels)."""
        bb = self.bb
        w = bb.layer(l)
        y = bb.swiglu_mlp(d["x"], w["w1"], w["w3"], w["w2"])
        return bb.mhc_merge(d["post"], d["comb"], y, d["h"])
```
### 4d. `GraphDecoder.capture` - capture dA/d for the source layer
Anchor (inside the `for l in range(self.n_layers):` loop):
```python
            if l not in moe:
                h_in = h
                h = self._capture(f"d{l}", lambda l=l, h_in=h_in, pro=pro: self._seg_dense(l, h_in, pro))
                self.out[f"d{l}"] = h
```
New (also add the 4 `_pg` lines just BEFORE the `for` loop, after `h = self.h0`):
```python
        first_moe = min(moe) if moe else None                                  # [R3.5-P (f)]
        if self.pregate and first_moe is not None and first_moe >= 1 and (first_moe - 1) not in moe:
            self._pg = (first_moe - 1, first_moe)
            self._pg_graph = f"dA{first_moe - 1}"
```
```python
            if l not in moe:
                h_in = h
                if self._pg is not None and l == self._pg[0]:                  # [R3.5-P (f)] dA{l} + d{l}
                    dd = self._capture(f"dA{l}", lambda l=l, h_in=h_in, pro=pro: self._seg_dense_a(l, h_in, pro,
                                                                                                   self._pg[1]))
                    self.out[f"dA{l}"] = dd
                    h = self._capture(f"d{l}", lambda l=l, dd=dd: self._seg_dense_b(l, dd))
                else:
                    h = self._capture(f"d{l}", lambda l=l, h_in=h_in, pro=pro: self._seg_dense(l, h_in, pro))
                self.out[f"d{l}"] = h
```
and the `_seq` list (end of capture). Old:
```python
        self._seq = [(l, f"a{l}", f"p{l}", f"b{l}", self.r.next_moe.get(l)) if l in moe else (l, f"d{l}", None, None, None)
                     for l in range(self.n_layers)]
```
New:
```python
        self._seq = []
        for l in range(self.n_layers):
            if l in moe:
                self._seq.append((l, f"a{l}", f"p{l}", f"b{l}", self.r.next_moe.get(l)))
            else:
                if self._pg is not None and l == self._pg[0]:                  # [R3.5-P (f)] dA{l} first
                    self._seq.append((l, f"dA{l}", None, None, None))
                self._seq.append((l, f"d{l}", None, None, None))
```
(Tuples keep their layout - dense entries have n_p None - so replay_bench's M1 / M1EV loops replay dA2 like any dense
graph.)
### 4e. `GraphDecoder.step` - the pre-gate prefetch
Old (start of the per-layer loop):
```python
        for l, n_a, n_p, n_b, nl in self._seq:
            g[n_a].replay()
            if n_p is None:
                continue
            ev_a, ev_p = self.ev_a[l], self.ev_p[l]
            ev_a.record(stream)
            g[n_p].replay()                       # prediction + shared expert: queued before the host waits
            ev_p.record(stream)
            ev_a.synchronize()                    # a{l} done -> its routed ids are in the pinned mailbox
```
New:
```python
        pg_pending = False
        for l, n_a, n_p, n_b, nl in self._seq:
            g[n_a].replay()
            if n_p is None:
                if n_a == self._pg_graph and pf:                              # [R3.5-P (f)]
                    self.ev_pg.record(stream)
                    pg_pending = True
                continue
            ev_a, ev_p = self.ev_a[l], self.ev_p[l]
            ev_a.record(stream)
            g[n_p].replay()                       # prediction + shared expert: queued before the host waits
            ev_p.record(stream)
            if pg_pending:                        # [R3.5-P (f)] dA finished while d / a / p were queued: prefetch now
                self.ev_pg.synchronize()
                pool.prefetch(l, sorted(set(self.pg_np.tolist() if frozen is None else frozen[l])))
                pg_pending = False
            ev_a.synchronize()                    # a{l} done -> its routed ids are in the pinned mailbox
```
(`pf = self.r.prefetch` is already defined before the loop. The pinned pg row is rewritten by the next token's dA2
only after this read.) The eager generate path has no pre-gate (dense layers do not go through moe_fn).
With (e) on as well, the pre-gate can be gated like the other predictions: in `_seg_dense_a` use
`pred, sel, ref = self._route_ids9(first, x_ffn.reshape(1, -1))` and copy `sel` / `ref` into two more pinned rows
(`pg_sc_pin` [8] fp32, `pg_ref_pin` [1] fp32, allocated in 4b when `self.pred_scores`), then call
`pool.prefetch_scored(l, self.pg_np.tolist(), self.pg_sc_np.tolist(), float(self.pg_ref_np[0]))` in 4e. Optional.
N = 2 set (M1): the same split can be applied to its dense layer 2 graph; not required for item 3.

---------------------------------------------------------------------------------------------------------------------
## 5. tcq_runtime.py / generate.py - item 10(b) cold insertion of prefill experts (for the item-10 package)
Pool v6 offers `ensure(..., cold=True)` (`pool.supports_cold`): misses go to the cold end (LRU front / GDSF H = L),
hits are not refreshed. Suggested wiring (flag R35_PREFILL_COLD=1, owner: the item-10 package):
- `generate.py generate()`: around the prefill `self.bb.forward(ids_t, ...)` set `rt.prefill_cold = True` on every
  `self.moe_layers` runtime when the flag is on and `getattr(self.pool, "supports_cold", False)`; reset to False after
  (also in a `finally:`). Not for the eager N = 2 MTP verify (that is decode traffic).
- `glm53_fold_tcq_runtime.py TcqLayerRuntime.forward_routed_experts`, Old:
```python
            slot_of_u = self.pool.ensure(self.layer_id, uniq.cpu().tolist())
```
  New:
```python
            if getattr(self, "prefill_cold", False):                   # [R3.5 item 10b] prefill-only experts: cold end
                slot_of_u = self.pool.ensure_cold(self.layer_id, uniq.cpu().tolist())
            else:
                slot_of_u = self.pool.ensure(self.layer_id, uniq.cpu().tolist())
```
  (`ensure_cold()` rather than `ensure(..., cold=True)`: install_mask_check's ensure wrapper forwards only
  `(l, experts, out)`; ensure_cold() hands the flag past it. Set `prefill_cold` only when
  `getattr(pool, "supports_cold", False)`.)

---------------------------------------------------------------------------------------------------------------------
## 6. Contract notes for M1 / M2 (N = 2 verify, MTP draft) - nothing to edit, rules to keep

- `ensure(l, ids16, out=row16)`: one call per layer per verify step, 16 entries (row 0's 8 then row 1's 8),
  duplicates allowed, one slot per entry (duplicates get the same slot; v5 and v6 both count a repeat as a hit).
  `out` = pinned int32 row of >= 16 entries; the call writes and returns it. Works with v5 and v6.
- LIFETIME RULE (module docstring of v6): slots stay valid until the next ensure() on ANOTHER layer, or release().
  Replay the graph that reads them (b{l}2, the draft's m_b) before that next ensure(). Same-layer calls accumulate.
  v5 gives the same in practice through LRU depth (fine with >= 1,000 slots).
- `prefetch(..., protect=)` is accepted by v6 (the current layer is already reserved, so protect is redundant).
- Layer 45: `pool.pin([(45, e) ...])` works on v5 and v6 (v6 removes pinned keys from the GDSF heap). Pinning the whole
  layer costs 288 pool slots (1.71 GiB of the pool); registering its rows (R33 tier) costs host RAM instead.
- `pool.release()` (v6 only, `getattr(pool, "supports_release", False)`): ends all reservations now (fence on the
  current stream) - e.g. after a request; never needed for correctness.
- Never call ensure() / release() while a CUDA graph is being captured (they record events on the current stream).

---------------------------------------------------------------------------------------------------------------------
## 7. Tools (already written, no integration needed)
- `build_run/test_pool_v6_policy_r35.py` - CPU-only tests (simulated GPU); `build_run/test_pool_v6_r35.py` - GPU
  torture + equivalence vs v5; `build_run/pool_v6_bench_r35.py` - replay bench with every pool counter (incl. the
  main / worker gather split and the pre-gate recall) and an eager `--mode pregate-recall` that needs no integration.
- replay_bench_r32.py and exact_gate_r34.py run pool v6 unchanged once 1a/1b are in (env R32_POOL=6, R34_RUNTIME).
- Commands, pass criteria and run times: `notes/P_GATING.md`.
