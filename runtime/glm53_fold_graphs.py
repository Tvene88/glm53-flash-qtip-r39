#!/usr/bin/env python3
"""
glm53_fold_graphs.py  [Round 3.2, Claude]
=========================================
CUDA-graph greedy decoder for the Round-3.2 fast path (N = 1 token per step). Mathematically the same computation as
`RealGLM53Backbone.forward` + `ApexFoldLayerRuntime._forward_kernel_path`, re-expressed with static buffers:

  * KDA layers   : recurrent state and short-conv states updated IN PLACE (state *= exp(g); state += k (x) delta).
  * DSA layers   : latent KV / indexer caches preallocated to `t_max`; the new token is written at a device-side
                   position and attention runs over [0, pos] with a mask (exact while pos < index_topk, where the
                   K-pool indexer selects every visible pool). Absorbed MLA as in the eager fast path.
  * MoE layers   : graph a{l} (attention + FFN prepare + router + D2H of the routed ids into a pinned mailbox), graph
                   p{l} (next-layer pre-gating ids + their D2H, shared expert; queued before the host waits, so it runs
                   during the host step) -> host step (waits for a{l} only; expert-pool lookup / miss copies, writes 8
                   slot ids; prefetch from p{l}'s ids) -> graph b{l} (routed-expert LUT-GEMVs from the pool + mHC merge).
                   [R3.4-2f: a{l} used to hold the pre-gating and the shared expert too.]
  * final graph  : stream collapse, final RMSNorm, lm_head, argmax (+ [R3.4-V] head row -> token id with a trimmed head).
  [R3.4-S1] expert masks (bb.set_pruned) live in the router bias tensors the graphs read, so they apply without recapture.

Requires the global expert pool (runner.pool). Falls back (returns None) when the context would exceed `t_max`.

[R3.5-M1] N = 2 VERIFY GRAPH SET (MTP step 1; opt-in: nothing below runs unless capture_pair() is called; the N = 1 set
above is unchanged and stays the default). A second, separate set of graphs - its own CUDA-graph memory pool, pinned
mailboxes and events - runs ONE verify step for the two tokens [t_last, t_draft] at positions [p, p + 1] through all
layers, on the SAME static decode state (KDA recurrent + conv states, DSA latent caches, pos_t) as the N = 1 set:
  * KDA   : the two recurrent updates in sequence, in place; the state after t_last is saved (kda_s0) between them. Conv:
            window [s0 s1 s2 x0 x1], conv0 <- the window after x0 (rollback point), state <- the window after x1.
  * DSA   : both latents written at pos2 = [p, p + 1]; per-row causal mask (row 1 sees row 0's latent, row 0 does not
            see it). A rejected row's latent at p + 1 needs no undo: every later step writes p + 1 before reading it.
  * MoE   : router + prediction per row; ONE host step per layer for both rows (pool.ensure of the 16 routed ids,
            duplicates allowed); routed experts in decode_core pair mode (16 pairs in one launch per projection).
  * final : collapse, norm and head on both rows, per-row argmax (+ head row -> token id), accept flag
            = (argmax of row 0 == t_draft) [or a forced value, benches only], pos_t += 1 + accepted, D2H of
            [tok0, tok1, accepted], and the in-graph rollback: a conditional copy (one kernel; nothing copied when
            accepted) puts back the KDA / conv states saved after t_last when the draft is rejected.
  Exactness: R35_MTP_ROWS=exact (default) keeps every ROW bit-identical to an N = 1 step - only kernels whose rows are
  computed independently are batched (FP8 GEMV: weights read once; the mHC kernels; the TCQ pair kernels); reductions
  and cuBLAS GEMMs run per row with the N = 1 shapes; elementwise ops run on both rows at once. R35_MTP_ROWS=batched
  batches those too (fewer launches; equal only up to float rounding). R35_MTP_BATCH=<classes> batches only the listed
  op classes inside exact mode (PAIR_OP_CLASSES; a bisection knob for the GPU test). Accepted step == two N = 1 steps, rejected
  step == one N = 1 step (state and token). Use while pair_ok(pos) (both positions inside the N = 1 bound), else step().
  API (for the MTP decoder of item 5): capture_pair(), pair_ok(), step2(), result2(), pair_tokens(), require_mtp(),
  state_snapshot() / state_restore(), graph_sequence2(), release_pair(); outputs next2 [2], logits2 [1, 2, V],
  collapsed2 [1, 2, D] (the MTP draft input), acc2_t [1].

[R3.5-M2] MTP DECODING ON THE GRAPH PATH (speed item 5; opt-in: runs only through capture_mtp() / decode_mtp(), which
generate() calls for --mtp together with --graphs when env R35_MTP_GRAPHS=1 - [R3.5-FIX F1] default 0 = the eager MTP
loop of the base code; decode(), step() and the N = 2 set are unchanged). A third graph set (own CUDA-graph pool, dict
graphs3) runs the MTP layer 45 (enorm / hnorm / eh_proj, its DSA
attention, its MoE through the expert pool, shared_head_norm + the shared, possibly trimmed head) to draft the next token;
the N = 2 verify set checks it; greedy output is token-identical to decode() (rows=exact: every verify row == an N = 1
step). One position counter: pos = tokens processed by the main model (= pos_t); layer-45 entry i = (h_i, emb(t_{i+1}))
with h_i the main model's collapsed hidden state of position i, kept in dsa45["kv"] [1, t_max, 512] (not in self.dsa:
load_cache / state_snapshot iterate the main layers only). The draft for the token after t_last (the input of the next
step, at position pos) comes from row pos - 1 = (h_{pos-1}, emb(t_last)), which attends to the entries
[m_start, pos - 1] (device mask). Writers: the prefill fills [n_pre, T - 2] eagerly from its h_collapsed
(mtp_load_history) and graph mp writes T - 1 (first draft); after an N = 1 step at p graph m1 (entry p + draft) or f1
(entry only, drafting paused); after a verify step at [p, p + 1] graph m2 writes p (row 0: always valid) and p + 1 (row 1:
valid when accepted; a rejected row's entry is masked for row 0 and rewritten by the next step before any read) and
selects the drafting row on the device (acc2_t). Graph mb then runs the selected row's routed experts (8 pool slots from a
pinned row: ONE host step = pool.ensure(45, ids), mb replayed right after it - pool v6 lifetime rule), shared expert,
head, argmax (+ head row -> token id) and the draft's softmax probability; one sync reads both. Restored prefix
positions (--prefix-cache, S1 state reuse) have no layer-45 entries: the history starts at n_pre (m_start), lossless
either way. Adaptive drafting (MtpDraftPolicy, host only): verify only drafts with probability >= tau (R35_MTP_TAU,
default 0.5), pause drafting R35_MTP_PAUSE steps when the acceptance EMA (R35_MTP_EMA) falls below tau or after
R35_MTP_SKIPRUN consecutive low-confidence drafts, optional prompt-lookup drafts (R35_MTP_NGRAM); toggling never desyncs (positions come from the device, every processed position
gets its entry). Pairs run only when two tokens fit max_new and pair_ok(); an accepted EOS draft ends generation
(bonus dropped). Optional: R35_MTP_PIN45=all pins the routable layer-45 experts in the pool; R35_MTP_CHECK=1 checks the
host position against pos_t every step; R35_MTP_TRACE=1 keeps a per-step trace in mtp_last. API: capture_mtp(),
release_mtp(), mtp_load_history(), decode_mtp() (stats in mtp_last / mtp_totals), mtp_enabled(); module level:
mtp_graphs_enabled(), MtpConfig, NgramIndex, MtpDraftPolicy.

[R3.5-INT] INTEGRATION of pool v6 (package P) and the multi-turn server (package S1); every item is opt-in:
  * (f) layer-3 pre-gate, env R35_PREGATE=1: the dense layer before the first MoE layer is captured as two graphs, dA{l}
    (attention + FFN prepare + the first MoE layer's router on this FFN input + D2H of its 8 ids into pg_pin) and d{l}
    (dense MLP + merge, the same kernels as the single d{l}, which keeps its name and output entry); step() prefetches
    the first MoE layer from pg_pin once a{l} / p{l} of that layer are queued. Prediction only (bytes, not math).
  * (e) confidence scores for pool v6's cold-prefetch gate, only when pool.wants_scores (R35_POOL_COLD_TOPK /
    R35_POOL_COLD_MARGIN): p{l} (and p{l} of the N = 2 set) also copy the predicted experts' selection scores and the
    9th-best score (_route_ids9) to pinned rows; step() / step2() call pool.prefetch_scored(). Prediction only.
  * S1 hook H3b: decode(..., on_token=None) - on_token(tok) -> True stops after that token (stop strings / client gone).
  * Serve-side layer-45 history (speed plan item 10c "keep the layer-45 latents between turns"): decode_mtp(...,
    keep_history=True) also writes the history entries of steps that do not draft (end of a generation; f1 / m2 without
    graph mb, plus graph mp without mb for the prompt's last position), so that entries [m_start, pos - 1) are complete
    when it returns; mtp_history_export() hands them out with h_{pos-1}; decode_mtp(..., mtp_restore=(kv, start, h_prev))
    / mtp_load_history(..., restore=) put a stored history back below the restored prefix (entries [start, n_pre - 1)
    from the store, entry n_pre - 1 recomputed from h_prev = h_{n_pre-1} and the new prompt's token at n_pre), so a
    restored request drafts from the same history a full prefill would give. Drafts only: lossless either way.

[R3.5-T] TIME TO FIRST TOKEN + the serve-adapter items INT left open; every item is opt-in:
  * item 10d capture guard, env R35_CAPTURE_GUARD=1: capture(), capture_pair() and capture_mtp() first quiesce the expert
    pool's prefetch worker (quiesce_pool: pool v6 quiesce(), v2 / v5 by polling) and record in CUDA's thread_local
    capture mode (_graph_ctx). A LAZY capture (decode() / decode_mtp() / capture_pair() reached inside a request, right
    after a prefill that queued prefetches - item 10a makes that the normal case) can then not be invalidated by the
    worker's CUDA calls. The recorded graphs are the same; off = the base torch.cuda.graph(g, pool=...) calls.
  * H4b export_cache(cache, copy=None): the static decode state as an eager RealCache (inverse of load_cache). Used by the
    runner's H6 continuation past t_max (R35_CONTINUE_PAST_TMAX) and by glm53_fold_serve's export_graph (one store copy).

[R3.9M] MTP SPEED WORK (2026-11-17, written without a GPU: NOT YET RUN ON ONE). Three flags, all default off = the code
above unchanged; every changed region carries the tag [R3.9M]:
  * R39_MTP_PF_ROWS=1: step2() hands each row's next-layer predictions to the pool in its own prefetch call, so the
    pool's rank gates (R35_POOL_COLD_TOPK / R37_POOL_REG_TOPK) apply per row as in two N = 1 steps, not to the union
    of both rows. Bytes only (which experts are copied early); no slot content and no arithmetic changes.
  * R39_MTP_STATS=1: decode_mtp() adds mtp_last["r39m"]: per step kind (pair_acc, pair_rej, n1, draft, fill,
    draft_first) the count, mean wall ms and the pool's lookups / demand misses / prefetches / host ms per step.
  * R39_MTP_FP=1: decode() and decode_mtp() keep a 64-bit fingerprint of the logit bits behind every emitted token
    (r39m_fp_last, one entry per generated id). Equal lists = every emitted row bit-identical. One extra host sync and
    three small eager kernels per emitted token (measured on no GPU yet; expected well below 1 % of a step).

[R3.9T] DECODE-STEP TIME SPLIT (2026-11-17; bookkeeping only, default off): R39T_PROF=1 makes decode() run _r39t_step,
which issues exactly step()'s calls plus timing events on the compute stream and host stamps (r39t_summary(): GPU ms per
graph kind, GPU ms stalled behind expert copies, GPU ms idle behind the host, host ms by activity, per prompt tag);
r39t_floor() measures the no-miss floors (the loop with every expert resident; the graphs of a token queued with no host
step). decode() also marks its start / end in the pool's call log (R39T_TRACE). Used by run_traffic_tests.ps1 -Step prof.
"""

from __future__ import annotations

import inspect
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

import glm53_fold_kernels as KR
from glm53_fold_math import transform_activations_fwht


def quiesce_pool(pool, dev=None, timeout: float = 120.0) -> float:
    """[R3.5-T item 10d] wait until the expert pool's prefetch worker is idle (every item it was handed is finished)
    and its copies have landed, so it makes no CUDA call during a graph capture that follows (prefetch() - the only
    producer of worker items - runs on the capturing thread). Pool v6: pool.quiesce(timeout). Pools without it (v2 /
    v5): poll until the worker queue and the pending-key map are empty, then synchronize the device (the worker's last
    event wait returns at once) and give it 2 ms to block in queue.get() again. Returns the seconds waited."""
    f = getattr(pool, "quiesce", None)
    if callable(f):
        return float(f(timeout))
    t0 = time.perf_counter()
    q, pend = getattr(pool, "q", None), getattr(pool, "pending", None)
    while (q is not None and not q.empty()) or pend:
        if time.perf_counter() - t0 > timeout:
            raise RuntimeError(f"quiesce_pool: {type(pool).__module__} worker not idle after {timeout:.0f} s "
                               f"({0 if q is None else q.qsize()} queued items, {len(pend or ())} pending keys)")
        time.sleep(0.001)
    if dev is not None and getattr(dev, "type", "") == "cuda":
        torch.cuda.synchronize(dev)
    time.sleep(0.002)
    return time.perf_counter() - t0


class GraphDecoder:
    def __init__(self, runner, t_max: int = 2048, longctx: Optional[bool] = None):
        self.r = runner
        self.bb = runner.bb
        self.cfg = runner.cfg
        self.dev = runner.device
        self.t_max = int(min(t_max, self.cfg.index_topk))
        # [R3.7 Stage 7] longctx: None = as the environment says (R36_LONGCTX=1, unless R37_GRAPH_STD=1 asks for the
        # window decoder in a process that has R36_LONGCTX set: used to gate that decoder alone); False / True =
        # explicit (R37_DUAL_GRAPHS builds one decoder of each kind, see glm53_fold_generate).
        if longctx is None:
            longctx = (os.environ.get('R36_LONGCTX', '0') == '1' and os.environ.get('R37_GRAPH_STD', '0') != '1')
        self.r36_longctx = bool(longctx)
        if self.r36_longctx:
            from glm53_fold_longctx_r36 import capacity
            self.t_max = capacity(self.cfg.index_topk)
            if not self.cfg.index_kpool_always_select_tail:
                raise ValueError('R36_LONGCTX currently requires index_kpool_always_select_tail')
        assert runner.pool is not None, "GraphDecoder requires the global expert pool (--expert-pool-gib)"
        self.pool = runner.pool
        self.r36_overlap = os.environ.get('R36_OVERLAP', '0') == '1'
        # [R3.4-fix1] step() passes ensure() its pinned slot row (out=, R3.4-2g): a pool without it fails here, not
        # in the first decode step
        assert "out" in inspect.signature(self.pool.ensure).parameters, (
            f"GraphDecoder needs pool.ensure(l, ids, out=) [R3.4-2g]; {type(self.pool).__module__} lacks it")
        self.n_layers = self.cfg.num_hidden_layers
        self.captured = False
        self.graphs: Dict[str, torch.cuda.CUDAGraph] = {}
        self.mempool = torch.cuda.graph_pool_handle()
        self.stats = {"tokens": 0, "seconds": 0.0}
        # [R3.4-2d] the DSA indexer caches (ki, gs) are never read while t_max <= index_topk (the K-pool indexer
        # then selects every visible pool = the causal mask used here); R34_DSA_INDEXER=1 keeps computing them
        # (needed only by a future > index_topk context path)
        self.dsa_indexer = os.environ.get("R34_DSA_INDEXER", "0") == "1"
        if self.r36_longctx:
            self.dsa_indexer = True
        # [R3.7 Stage 7] R37_LONGCTX_FAST=1 (only with R36_LONGCTX=1): the long-context step keeps the pool keys in a
        # static buffer and orders the selected positions by a prefix sum (glm53_fold_longctx_r37). Default off.
        self.r37_fast = self.r36_longctx and os.environ.get('R37_LONGCTX_FAST', '0') == '1'
        # [R3.8] R38_KV_BITS = 4 | 8 | 32 (only with the fast long-context step): the attention history is kept in the
        # coded layout of glm53_fold_qstate_r38 (latent codes + scales, pool keys, the open pool's indexer rows).
        # Unset = the Round 3.7 buffers. 32 = fp32 values in the new layout (the reference arm of the gate).
        from glm53_fold_qstate_r38 import kv_bits_from_env
        self.r38_bits = kv_bits_from_env(os.environ) if self.r37_fast else 0
        if self.r36_longctx and not self.r37_fast and kv_bits_from_env(os.environ):
            raise ValueError('R38_KV_BITS needs R37_LONGCTX_FAST=1 (the coded history exists for the fast long-context step only)')
        self.kvcodec = None
        self.r38_shape = None
        # [R3.4-2i] A/B switches (default off): R34_WDDM_FLUSH=1 -> stream.query() after every b{l} replay (makes
        # WDDM submit the queued work at once); R34_BLOCKING_SYNC=1 -> blocking (yielding) event waits, not spinning
        self.wddm_flush = os.environ.get("R34_WDDM_FLUSH", "0") == "1"
        self.blocking_sync = os.environ.get("R34_BLOCKING_SYNC", "0") == "1"
        self.pregate = os.environ.get("R35_PREGATE", "0") == "1"   # [R3.5-P (f)] layer-3 pre-gate from dense layer 2
        self._pg = None                                           # (source dense layer, first MoE layer) once captured
        self._pg_graph = None
        self.captured2 = False                        # [R3.5-M1] N = 2 verify set (capture_pair); off by default
        self.mtp_captured = False                     # [R3.5-M2] MTP draft set (capture_mtp); off by default
        self.mtp_cfg = None                           # [R3.5-M2] MtpConfig (None: from the env at first use)
        self.mtp_last = None                          # [R3.5-M2] stats of the last decode_mtp() call
        # [R3.5-T item 10d] env R35_CAPTURE_GUARD=1: every capture (at load, or lazily inside a request right after a
        # prefill that queued prefetches) first quiesces the expert pool's prefetch worker and runs in CUDA's
        # thread_local capture mode, so a CUDA call of another thread can no longer invalidate it. 0 = the base captures.
        self.capture_guard = os.environ.get("R35_CAPTURE_GUARD", "0") == "1"
        self._cap_mode = "thread_local" if self.capture_guard else None
        self.guard_stats = {"quiesce_calls": 0, "quiesce_s": 0.0}
        self.r39m_fp_last = None                      # [R3.9M] R39_MTP_FP=1: logit fingerprints of the last decode call

    # ------------------------------------------------------------------ [R3.9M] bookkeeping helpers (flags, default off)
    # [R3.9M] Reason: the GPU tests of the MTP path need (a) a check that is stronger than "same tokens" - the bits of
    # the logits behind every emitted token - and (b) where a verify step's time goes (per step kind: wall time and the
    # expert pool's traffic). Both are host-side reads after a step has finished; no graph, kernel or pool call changes.
    # R39_MTP_FP=1 / R39_MTP_STATS=1 switch them on; unset = none of the code below runs.
    _R39M_KEYS = ("hits", "misses", "prefetch_issued", "prefetch_useful", "prefetch_skipped_reg",
                  "prefetch_skipped_cold", "prefetch_claimed", "t_ensure", "t_prefetch", "bytes_h2d")

    def _r39m_fp_begin(self):
        """[R3.9M] R39_MTP_FP=1: a new fingerprint list for this decode call (kept in r39m_fp_last); else None."""
        if os.environ.get("R39_MTP_FP", "0").strip() != "1":
            self.r39m_fp_last = None
            return None
        self.r39m_fp_last = []
        return self.r39m_fp_last

    def _r39m_hash(self, lg: torch.Tensor) -> int:
        """[R3.9M] 64-bit fingerprint of the BITS of a logit row (fp32): sum of bit pattern x (index + 1), wrapping.
        Two rows have the same value only if they are bit-identical (up to a 2^-64 collision). Eager, one host sync."""
        v = lg.reshape(-1).float().contiguous().view(torch.int32).to(torch.int64)
        w = getattr(self, "_r39m_w", None)
        if w is None or w.numel() < v.numel() or w.device != v.device:
            w = self._r39m_w = torch.arange(1, v.numel() + 1, dtype=torch.int64, device=v.device)
        return int((v * w[: v.numel()]).sum().item())

    def _r39m_mark(self):
        """[R3.9M] R39_MTP_STATS=1: (time, pool counters) before a step."""
        st = getattr(getattr(self, "pool", None), "stats", None) or {}
        return time.perf_counter(), [st.get(k, 0) for k in self._R39M_KEYS]

    def _r39m_add(self, prof: Dict[str, dict], kind: str, mark) -> None:
        """[R3.9M] R39_MTP_STATS=1: book the wall time and the pool-counter deltas since `mark` under `kind`."""
        t1 = time.perf_counter()
        st = getattr(getattr(self, "pool", None), "stats", None) or {}
        b = prof.get(kind)
        if b is None:
            b = prof[kind] = {"n": 0, "s": 0.0, **{k: 0 for k in self._R39M_KEYS}}
        b["n"] += 1
        b["s"] += t1 - mark[0]
        for k, v0 in zip(self._R39M_KEYS, mark[1]):
            b[k] += st.get(k, 0) - v0

    @staticmethod
    def _r39m_summary(prof: Dict[str, dict]) -> Dict[str, dict]:
        """[R3.9M] per step kind (pair_acc / pair_rej / n1 / draft / fill): count, mean wall ms, and per step the pool's
        lookups, demand misses, prefetches, host ms inside ensure() / prefetch() and GiB copied host -> GPU."""
        out = {}
        for kind, b in prof.items():
            n = max(1, int(b["n"]))
            out[kind] = {"n": int(b["n"]), "ms_mean": round(1000.0 * b["s"] / n, 3),
                         "lookups": round((b["hits"] + b["misses"]) / n, 2), "misses": round(b["misses"] / n, 2),
                         "pf_issued": round(b["prefetch_issued"] / n, 2), "pf_useful": round(b["prefetch_useful"] / n, 2),
                         "pf_skipped_reg": round(b["prefetch_skipped_reg"] / n, 2),
                         "pf_skipped_cold": round(b["prefetch_skipped_cold"] / n, 2),
                         "pf_claimed": round(b["prefetch_claimed"] / n, 2),
                         "ensure_ms": round(1000.0 * b["t_ensure"] / n, 3),
                         "prefetch_ms": round(1000.0 * b["t_prefetch"] / n, 3),
                         "gib_h2d": round(b["bytes_h2d"] / n / 2**30, 4)}
        return out

    # ------------------------------------------------------------------ static state
    def _alloc_state(self):
        bb, cfg = self.bb, self.cfg
        D = cfg.hidden_size
        self.h0 = torch.zeros((1, 1, bb.hc, D), dtype=torch.float32, device=self.dev)
        self.pos_t = torch.zeros((1,), dtype=torch.long, device=self.dev)
        self.ar_t = torch.arange(self.t_max, device=self.dev)
        self.kda: Dict[int, Dict[str, torch.Tensor]] = {}
        self.dsa: Dict[int, Dict[str, torch.Tensor]] = {}
        for l in range(self.n_layers):
            if bb.is_kda(l):
                H, hd = bb.H_kda, bb.hd_kda
                Kc = cfg.kda_short_conv_kernel_size
                self.kda[l] = {"state": torch.zeros((1, H, hd, hd), dtype=torch.float32, device=self.dev),
                               "cq": torch.zeros((1, Kc - 1, H * hd), dtype=torch.float32, device=self.dev),
                               "ck": torch.zeros((1, Kc - 1, H * hd), dtype=torch.float32, device=self.dev),
                               "cv": torch.zeros((1, Kc - 1, H * hd), dtype=torch.float32, device=self.dev)}
            elif getattr(self, 'r38_bits', 0):                                # [R3.8] coded history
                from glm53_fold_qstate_r38 import KvCodec, alloc_layer_state, capacity_view
                if self.kvcodec is None:
                    self.kvcodec = KvCodec(self.r38_bits, cfg.kv_lora_rank, device=self.dev)
                    self.r38_shape = capacity_view(self.t_max, self.dev)
                self.dsa[l] = alloc_layer_state(self.kvcodec, self.t_max, cfg.index_kpool, cfg.index_head_dim, self.dev)
            else:
                self.dsa[l] = {"kv": torch.zeros((1, self.t_max, cfg.kv_lora_rank), dtype=torch.float32, device=self.dev)}
                if self.dsa_indexer:                                          # [R3.4-2d]
                    self.dsa[l]["ki"] = torch.zeros((1, self.t_max, cfg.index_head_dim), dtype=torch.float32, device=self.dev)
                    self.dsa[l]["gs"] = torch.zeros((1, self.t_max, cfg.index_head_dim), dtype=torch.float32, device=self.dev)
        if getattr(self, 'r37_fast', False):                                   # [R3.7 Stage 7] see __init__
            Pk = cfg.index_kpool
            assert self.t_max % Pk == 0, "R37_LONGCTX_FAST needs a capacity that is a multiple of index_kpool"
            n_sel = min(self.t_max, cfg.index_topk + Pk - 1)
            self.lpk = {l: torch.zeros((1, self.t_max // Pk, cfg.index_head_dim), dtype=torch.float32, device=self.dev)
                        for l in self.dsa}
            self.lbuf = {l: torch.zeros((n_sel + 1,), dtype=torch.long, device=self.dev) for l in self.dsa}
            self.ar_p = torch.arange(Pk, device=self.dev)
        # [R3.4-2e] exp(A_log) once per KDA layer (the same torch.exp kernel as inside the graph -> bit-identical)
        self.A_exp = {l: torch.exp(bb.layer(l)["A_log"]).view(1, 1, bb.H_kda, 1) for l in self.kda}
        self.slots = {l: torch.zeros((self.cfg.num_experts_per_tok,), dtype=torch.int32, device=self.dev)
                      for l in self.r.moe_layers if l < self.n_layers}
        # [R3.4-2f] pinned host mailboxes, one row per layer, filled by D2H copies captured INSIDE the graphs: the routed
        # ids (end of a{l}) and the next MoE layer's predicted ids (p{l}); one completion event per graph and layer
        Kx = self.cfg.num_experts_per_tok
        self.ids_pin = torch.zeros((self.n_layers, Kx), dtype=torch.long, pin_memory=True)
        self.pred_pin = torch.zeros((self.n_layers, Kx), dtype=torch.long, pin_memory=True)
        self.ids_np, self.pred_np = self.ids_pin.numpy(), self.pred_pin.numpy()
        # [R3.5-P (e)] prediction scores + 9th-best score (margin reference), only when the pool gates on them
        self.pred_scores = bool(getattr(self.pool, "wants_scores", False))
        if self.pred_scores:
            self.pred_sc_pin = torch.zeros((self.n_layers, Kx), dtype=torch.float32, pin_memory=True)
            self.pred_ref_pin = torch.zeros((self.n_layers,), dtype=torch.float32, pin_memory=True)
            self.pred_sc_np, self.pred_ref_np = self.pred_sc_pin.numpy(), self.pred_ref_pin.numpy()
        self.ev_a = {l: torch.cuda.Event(blocking=self.blocking_sync) for l in self.slots}   # [R3.4-2i] A/B
        self.ev_p = {l: torch.cuda.Event(blocking=self.blocking_sync) for l in self.slots}
        # [R3.4-2g] pinned slot mailbox: pool.ensure writes a layer's 8 slots here, b{l} starts with their H2D copy
        self.slots_pin = torch.zeros((self.n_layers, Kx), dtype=torch.int32, pin_memory=True)
        self.slots_np = self.slots_pin.numpy()
        # [R3.4-2g] pinned embedding row (host-resident table): the host writes the token's row, the first layer's graph
        # copies it to the GPU and expands it into the hc streams (bf16 -> fp32 is exact, as in the eager copy)
        emb = bb.embed
        self.e_pin = None
        if isinstance(emb, torch.Tensor) and emb.device.type == "cpu" and emb.element_size() in (2, 4):
            iv = torch.int16 if emb.element_size() == 2 else torch.int32
            self.e_pin = torch.zeros((D,), dtype=emb.dtype, pin_memory=True)
            self.e_dev = torch.zeros((D,), dtype=emb.dtype, device=self.dev)
            self._emb_np, self._e_np = emb.view(iv).numpy(), self.e_pin.view(iv).numpy()
        self.final_adds_pos = True                    # [R3.4-2g] pos_t += 1 is the last node of the final graph
        if getattr(self, "pregate", False):                                    # [R3.5-P (f)]
            self.pg_pin = torch.zeros((Kx,), dtype=torch.long, pin_memory=True)
            self.pg_np = self.pg_pin.numpy()
            self.ev_pg = torch.cuda.Event(blocking=self.blocking_sync)
            if self.pred_scores:                                               # [R3.5-P (e) + (f)] scored pre-gate
                self.pg_sc_pin = torch.zeros((Kx,), dtype=torch.float32, pin_memory=True)
                self.pg_ref_pin = torch.zeros((1,), dtype=torch.float32, pin_memory=True)
                self.pg_sc_np, self.pg_ref_np = self.pg_sc_pin.numpy(), self.pg_ref_pin.numpy()

    def r38_history(self, clear: bool = True):
        """[R3.8] the handle a prompt pass writes through (set it as cache.r38 of a fresh RealCache): this decoder's own
        history buffers. clear=True starts a new conversation."""
        if not getattr(self, 'r38_bits', 0):
            raise RuntimeError("r38_history needs R38_KV_BITS")
        from glm53_fold_qstate_r38 import CodedHistory
        hist = CodedHistory(self.kvcodec, self.dsa, self.lpk, self.cfg.index_kpool, self.t_max,
                            qblock=int(os.environ.get('R38_PREFILL_QBLOCK', '16') or 16))
        if clear:
            hist.clear()
        return hist

    def load_cache(self, cache) -> bool:
        """Copy an eager RealCache (after prefill) into the static buffers. False if the context is too long."""
        T = int(cache.pos)
        if T >= self.t_max:
            return False
        hist = getattr(cache, 'r38', None)
        if hist is not None:                                                 # [R3.8] the history is already in place
            if not getattr(self, 'r38_bits', 0) or hist.dsa is not self.dsa:
                raise RuntimeError("load_cache: this cache writes into the history of another decoder")
            for l, st in self.kda.items():
                st["state"].copy_(cache.kda_state[l])
                cq, ck, cv = cache.kda_conv[l]
                st["cq"].copy_(cq); st["ck"].copy_(ck); st["cv"].copy_(cv)
            self.pos_t.fill_(T)
            return True
        if getattr(self,'r36_longctx',False):
            from glm53_fold_longctx_r36 import validate_cache
            validate_cache(cache,self.dsa)
        for l, st in self.kda.items():
            st["state"].copy_(cache.kda_state[l])
            cq, ck, cv = cache.kda_conv[l]
            st["cq"].copy_(cq); st["ck"].copy_(ck); st["cv"].copy_(cv)
        for l, st in self.dsa.items():
            if getattr(self, 'r38_bits', 0):                                 # [R3.8] codes, pool keys, open pool
                from glm53_fold_qstate_r38 import load_layer_state
                load_layer_state(self.kvcodec, st, self.lpk[l], self.bb, l, cache.dsa_kv[l][:, :T], cache.dsa_ki[l][:, :T],
                                 cache.dsa_gs[l][:, :T])
                continue
            st["kv"].zero_()
            st["kv"][:, :T].copy_(cache.dsa_kv[l])
            if "ki" in st:                                                   # [R3.4-2d]
                st["ki"].zero_(); st["gs"].zero_()
                st["ki"][:, :T].copy_(cache.dsa_ki[l]); st["gs"][:, :T].copy_(cache.dsa_gs[l])
                if getattr(self, 'r37_fast', False):                         # [R3.7 Stage 7] keys of the whole prefix
                    from glm53_fold_longctx_r37 import pool_keys_all
                    self.lpk[l].copy_(pool_keys_all(self.bb, l, st["ki"], st["gs"]))
        self.pos_t.fill_(T)
        return True

    def export_cache(self, cache, copy=None):
        """[R3.5-T H4b] the static decode state as an eager RealCache - the inverse of load_cache(): pos = pos_t, KDA
        recurrent + conv states, DSA latents [:, :pos] in the eager cache's dtypes, and the indexer caches ki / gs for
        every position (the graphs' own with R34_DSA_INDEXER=1; else the prompt's from `cache` - the eager cache decode
        started from - plus zero placeholders for the decoded positions). Attributes indexer_complete / indexer_gap (None
        or the first placeholder position). Main layers only (the MTP layer's history lives in dsa45). `copy` maps each
        tensor to its stored copy (default clone(); the server passes its host / device store copy - one copy each).
        Used by generate()'s H6 continuation (R35_CONTINUE_PAST_TMAX) and by glm53_fold_serve's export_graph."""
        from glm53_fold_backbone import RealCache
        cp = copy if copy is not None else (lambda t: t.clone())
        if getattr(self, 'r38_bits', 0):
            raise RuntimeError("export_cache: the coded history (R38_KV_BITS) has no eager form; the server stores it in "
                               "its own format (Round 3.8 store), and R35_CONTINUE_PAST_TMAX is not available with it")
        T, P0 = int(self.pos_t.item()), int(cache.pos)
        if not (P0 <= T <= int(self.t_max)):
            raise RuntimeError(f"export_cache: graph position {T} outside [{P0}, {int(self.t_max)}]")
        c = RealCache()
        c.pos = T
        for l, st in self.kda.items():
            c.kda_state[l] = cp(st["state"])
            c.kda_conv[l] = (cp(st["cq"]), cp(st["ck"]), cp(st["cv"]))
        complete = True
        for l, st in self.dsa.items():
            c.dsa_kv[l] = cp(st["kv"][:, :T].to(cache.dsa_kv[l].dtype))
            for key, src, dst in (("ki", cache.dsa_ki, c.dsa_ki), ("gs", cache.dsa_gs, c.dsa_gs)):
                if key in st:
                    dst[l] = cp(st[key][:, :T].to(src[l].dtype))
                else:
                    e = src[l][:, :P0]
                    dst[l] = cp(torch.cat([e, e.new_zeros((e.shape[0], T - P0, e.shape[2]))], dim=1))
                    complete = complete and T == P0
        c.indexer_complete = bool(complete)
        c.indexer_gap = None if complete else P0
        return c

    # ------------------------------------------------------------------ graph-safe layer pieces
    def _conv_step(self, xp_new, w_conv, conv_state):
        xp = torch.cat([conv_state, xp_new.float()], dim=1)
        out = (xp * w_conv.T.unsqueeze(0)).sum(dim=1, keepdim=True)
        conv_state.copy_(xp[:, 1:])
        return F.silu(out)

    def _kda_step(self, l, x):
        # [R3.7, review K2-03] the fused dispatch honours the backbone's use_kernels switch, as backbone.py does
        if os.environ.get('R36_FUSE_KDA', '0') == '1' and getattr(self.bb, 'use_kernels', True):
            from glm53_fold_fused_r36 import kda_graph
            return kda_graph(self, l, x)
        bb = self.bb
        w = bb.layer(l)
        st = self.kda[l]
        H, hd = bb.H_kda, bb.hd_kda
        q = self._conv_step(bb.lin(x, w["q"]), w["qconv"], st["cq"]).view(1, 1, H, hd)
        k = self._conv_step(bb.lin(x, w["k"]), w["kconv"], st["ck"]).view(1, 1, H, hd)
        v = self._conv_step(bb.lin(x, w["v"]), w["vconv"], st["cv"]).view(1, 1, H, hd)
        q = (q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)) * (hd ** -0.5)
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        g = (bb.lin(bb.lin(x, w["fa"]), w["fb"]) + w["dt_bias"]).view(1, 1, H, hd)
        g = bb.lower_bound * torch.sigmoid(self.A_exp[l] * g)                 # [R3.4-2e]
        beta = torch.sigmoid(bb.lin(x, w["b"]))
        state = st["state"]
        state.mul_(g[:, 0].exp().unsqueeze(-1))
        kv_mem = torch.matmul(k[:, 0].unsqueeze(-2), state).squeeze(-2)
        delta = (v[:, 0] - kv_mem) * beta[:, 0].unsqueeze(-1)
        state.add_(k[:, 0].unsqueeze(-1) * delta.unsqueeze(-2))
        out = torch.matmul(q[:, 0].unsqueeze(-2), state).squeeze(-2).unsqueeze(1)
        gate = bb.lin(bb.lin(x, w["ga"]), w["gb"]).view(1, 1, H, hd)
        o = bb.rmsnorm(out, w["o_norm"]) * torch.sigmoid(gate)
        return bb.lin(o.reshape(1, 1, H * hd), w["o"])

    def _dsa_step(self, l, x):
        bb, cfg = self.bb, self.cfg
        w = bb.layer(l)
        st = self.dsa[l]
        H, qk, vd = bb.H, bb.qk, bb.vd
        q_resid = bb.rmsnorm(bb.lin(x, w["qa"]), w["qa_norm"])
        q = bb.lin(q_resid, w["qb"]).view(1, 1, H, qk)
        kv_new = bb.rmsnorm(bb.lin(x, w["kva"]), w["kva_norm"])
        if getattr(self, 'r38_bits', 0):                                  # [R3.8] the Round 3.7 fast step on the coded history
            from glm53_fold_qstate_r38 import step_write, step_pool_key as pool_key_r38, selected_latents as selected_r38
            ki_new = F.layer_norm(bb.lin(x, w["i_wk"]), (cfg.index_head_dim,), w["i_kn_w"], w["i_kn_b"], eps=1e-6)
            gs_new = bb.lin(x, w["i_gate"])
            step_write(self.kvcodec, st, self.pos_t, cfg.index_kpool, kv_new, ki_new, gs_new)
            W_k, W_v = bb._dsa_kv_split(l, w["kvb"])
            q_abs = torch.einsum("bshd,hdc->bshc", q.float(), W_k)
            pool_key_r38(bb, l, st, self.lpk[l], self.pos_t)
            kv, valid = selected_r38(bb, l, x, q_resid, st, self.lpk[l], self.pos_t, self.ar_t, self.lbuf[l],
                                     self.kvcodec, self.r38_shape)
            scores = torch.einsum("bshc,btc->bhst", q_abs, kv) * (qk ** -0.5)
            scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
            probs = torch.softmax(scores, dim=-1)
            o_lat = torch.einsum("bhst,btc->bshc", probs, kv)
            out = torch.einsum("bshc,hdc->bshd", o_lat, W_v)
            return bb.lin(out.reshape(1, 1, H * vd), w["o"])
        st["kv"].index_copy_(1, self.pos_t, kv_new.float())
        if self.dsa_indexer:                                              # [R3.4-2d] dead work otherwise
            ki_new = F.layer_norm(bb.lin(x, w["i_wk"]), (cfg.index_head_dim,), w["i_kn_w"], w["i_kn_b"], eps=1e-6)
            gs_new = bb.lin(x, w["i_gate"])
            st["ki"].index_copy_(1, self.pos_t, ki_new.float())
            st["gs"].index_copy_(1, self.pos_t, gs_new.float())
        W_k, W_v = bb._dsa_kv_split(l, w["kvb"])
        q_abs = torch.einsum("bshd,hdc->bshc", q.float(), W_k)
        kv=st['kv']
        if getattr(self,'r36_longctx',False):
            if getattr(self, 'r37_fast', False):                             # [R3.7 Stage 7] same selection, less work
                from glm53_fold_longctx_r37 import step_pool_key, selected_latents_fast
                step_pool_key(bb, l, st, self.lpk[l], self.pos_t, self.ar_p)
                kv, valid = selected_latents_fast(bb, l, x, q_resid, st, self.lpk[l], self.pos_t, self.ar_t,
                                                  self.lbuf[l])
            else:
                from glm53_fold_longctx_r36 import selected_latents
                kv,valid=selected_latents(bb,l,x,q_resid,st,self.pos_t,self.ar_t)
            scores = torch.einsum("bshc,btc->bhst", q_abs, kv) * (qk ** -0.5)
        else:
            # Keep the R35 launch order with the R36 flag off.
            scores = torch.einsum("bshc,btc->bhst", q_abs, st['kv']) * (qk ** -0.5)
            valid = (self.ar_t <= self.pos_t).view(1, 1, 1, self.t_max)
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        o_lat = torch.einsum("bhst,btc->bshc", probs, kv)
        out = torch.einsum("bshc,hdc->bshd", o_lat, W_v)
        return bb.lin(out.reshape(1, 1, H * vd), w["o"])

    def _attn(self, l, h):
        bb = self.bb
        w = bb.layer(l)
        post, comb, x_in = bb.mhc_prepare(h, w["hc_attn_fn"], w["hc_attn_base"], w["hc_attn_scale"])
        xn = bb.rmsnorm(x_in, w["in_norm"])
        y = self._kda_step(l, xn) if bb.is_kda(l) else self._dsa_step(l, xn)
        return bb.mhc_merge(post, comb, y, h)

    # ------------------------------------------------------------------ segments (captured)
    def _embed_prologue(self):
        """[R3.4-2g] first nodes of the first layer's graph: pinned embedding row -> GPU -> h0 (hc copies)."""
        self.e_dev.copy_(self.e_pin, non_blocking=True)
        self.h0.copy_(self.e_dev.view(1, 1, 1, -1).expand(1, 1, self.bb.hc, -1))

    def _seg_dense(self, l, h, pro=False):
        bb = self.bb
        if pro:
            self._embed_prologue()
        h = self._attn(l, h)
        w = bb.layer(l)
        post, comb, x_ffn = bb.ffn_prepare(l, h)
        y = bb.swiglu_mlp(x_ffn, w["w1"], w["w3"], w["w2"])
        return bb.mhc_merge(post, comb, y, h)

    def _seg_dense_a(self, l, h, pro, first):
        """[R3.5-P (f)] first half of dense layer l: attention + FFN prepare, plus the pre-gate of MoE layer `first`
        (its router on this FFN input) and the D2H of the 8 predicted ids (+ scores with (e)). Prediction only."""
        bb = self.bb
        if pro:
            self._embed_prologue()
        h = self._attn(l, h)
        post, comb, x_ffn = bb.ffn_prepare(l, h)
        if getattr(self, "pred_scores", False):                          # [R3.5-P (e)] ids + scores + 9th best
            pred, sel, ref = self._route_ids9(first, x_ffn.reshape(1, -1))
            self.pg_sc_pin.copy_(sel.view(-1), non_blocking=True)
            self.pg_ref_pin.copy_(ref.view(-1), non_blocking=True)
        else:
            pred = bb.route_ids(first, x_ffn.reshape(1, -1))[0]
        self.pg_pin.copy_(pred.view(-1), non_blocking=True)
        return {"h": h, "post": post, "comb": comb, "x": x_ffn}

    def _seg_dense_b(self, l, d):
        """[R3.5-P (f)] second half of dense layer l: the dense MLP + merge (the rest of _seg_dense, same kernels)."""
        bb = self.bb
        w = bb.layer(l)
        y = bb.swiglu_mlp(d["x"], w["w1"], w["w3"], w["w2"])
        return bb.mhc_merge(d["post"], d["comb"], y, d["h"])

    def _route_ids9(self, l, x_flat):
        """[R3.5-P (e)] next-layer prediction: bb.route_ids' ops with a top-9 (sorted): the 8 predicted ids, their
        selection scores (sigmoid + bias) and the 9th-best score (margin reference for pool v6's cold-prefetch gate).
        Masked experts have gate_b = -inf, so they never enter the 8 ids (a -inf 9th score only makes margins infinite)."""
        if getattr(getattr(self,'bb',None),'r36_router',None) is not None:
            weights,idx=self.bb.route(l,x_flat)
            return idx,weights,weights.new_zeros((weights.shape[0],))
        w = self.bb.layer(l)
        logits = F.linear(x_flat.float(), w["gate_w"].float())
        choices = torch.sigmoid(logits) + w["gate_b"].float().unsqueeze(0)
        sel, idx = torch.topk(choices, k=self.cfg.num_experts_per_tok + 1, dim=-1, sorted=True)
        return idx[:, :-1], sel[:, :-1], sel[:, -1]

    def _seg_a(self, l, h, pro=False):
        """[R3.4-2f] a{l} = attention + FFN prepare + router; ends with a D2H copy of the routed ids into the pinned
        mailbox (the host waits for this graph only)."""
        bb = self.bb
        if pro:
            self._embed_prologue()
        h = self._attn(l, h)
        post, comb, x_ffn = bb.ffn_prepare(l, h)
        x_flat = x_ffn.reshape(1, -1)
        top_w, top_i = bb.route(l, x_flat)
        self.ids_pin[l].copy_(top_i.view(-1), non_blocking=True)
        return {"h": h, "post": post, "comb": comb, "x": x_flat, "w": top_w, "i": top_i}

    def _seg_p(self, l, a):
        """[R3.4-2f] p{l} = the rest of the old a{l}: next-layer prediction (+ its D2H copy) and the shared expert. It is
        replayed BEFORE the host waits for a{l}, so the GPU runs it during the host step (pool lookup / misses)."""
        bb = self.bb
        nxt = self.r.next_moe.get(l)
        if nxt is not None and getattr(self, "pred_scores", False):       # [R3.5-P (e)] ids + scores + 9th best
            pred, sel, ref = self._route_ids9(nxt, a["x"])
            self.pred_pin[l].copy_(pred.view(-1), non_blocking=True)
            self.pred_sc_pin[l].copy_(sel.view(-1), non_blocking=True)
            self.pred_ref_pin[l:l + 1].copy_(ref.view(-1), non_blocking=True)
        elif nxt is not None:
            pred = bb.route_ids(nxt, a["x"])[0]                               # [R3.4-2c] ids only
            self.pred_pin[l].copy_(pred.view(-1), non_blocking=True)
        else:
            pred = a["i"]
        y_sh = bb.shared_expert(l, a["x"])
        return {"pred": pred, "y_sh": y_sh}

    def _seg_b(self, l, a):
        if getattr(self, 'r36_overlap', False):
            y_r = self.r36_parts[('n1', l)].finish(a['x'], a['w'])
            return self.bb.mhc_merge(a['post'], a['comb'], (y_r+a['y_sh']).view(1,1,-1), a['h'])
        self.slots[l].copy_(self.slots_pin[l], non_blocking=True)   # [R3.4-2g] pinned slot mailbox -> GPU (graph node)
        rt = self.r.moe_layers[l]
        # [R3.2] format-agnostic routed core (RVQ or TCQ archive) reading the pool slots written by the host step
        y_r = rt.decode_core(a["x"].float(), self.pool.pool, self.slots[l], a["w"].reshape(-1).float(), 1,
                             self.cfg.swiglu_limit)
        y = (y_r + a["y_sh"]).view(1, 1, -1)
        return self.bb.mhc_merge(a["post"], a["comb"], y, a["h"])

    def _seg_final(self, h):
        logits, _ = self.bb.final_logits(h, last_only=True)
        nt = torch.argmax(logits[0, -1]).view(1)
        if self.bb.vocab_ids is not None:             # [R3.4-V] trimmed head: head row -> original token id (graph node)
            nt = self.bb.vocab_ids.index_select(0, nt)
        self.pos_t.add_(1)                            # [R3.4-2g] position advance inside the graph (was eager)
        return nt, logits

    # ------------------------------------------------------------------ capture
    def _graph_ctx(self, g, pool):
        """[R3.5-T item 10d] the capture context of every set: the base torch.cuda.graph(g, pool=pool), or - with the
        capture guard (R35_CAPTURE_GUARD=1) - the same call in thread_local capture mode (the same graph is recorded;
        only CUDA's checking of OTHER threads' calls during the capture is relaxed)."""
        mode = getattr(self, "_cap_mode", None)
        if mode is None:
            return torch.cuda.graph(g, pool=pool)
        return torch.cuda.graph(g, pool=pool, capture_error_mode=mode)

    def _guard_capture(self) -> None:
        """[R3.5-T item 10d] R35_CAPTURE_GUARD=1: quiesce the expert pool's prefetch worker before a capture (it may
        still copy prefetches of the prefill that preceded a LAZY capture); no-op with the guard off."""
        if not getattr(self, "capture_guard", False):
            return
        dt = quiesce_pool(self.pool, self.dev)
        gs = getattr(self, "guard_stats", None)
        if gs is None:
            gs = self.guard_stats = {"quiesce_calls": 0, "quiesce_s": 0.0}
        gs["quiesce_calls"] += 1
        gs["quiesce_s"] += dt

    def _capture(self, name, fn):
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream(device=self.dev)
        s.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(s):
            fn()  # warm-up (compiles NVRTC kernels, cuBLAS handles) - side effects are overwritten by load_cache
        torch.cuda.current_stream(self.dev).wait_stream(s)
        torch.cuda.synchronize(self.dev)
        with self._graph_ctx(g, self.mempool):        # [R3.5-T] = torch.cuda.graph(g, pool=self.mempool) unless guarded
            out = fn()
        self.graphs[name] = g
        return out

    @torch.no_grad()                                  # [R3.4-fix2] called at load now (2h), not only in decode()
    def capture(self):
        self._guard_capture()                         # [R3.5-T item 10d] no-op unless R35_CAPTURE_GUARD=1
        self.bb.graphs_captured = True                # [R3.4 review] bb.set_vocab_keep() refuses from now on
        self._alloc_state()
        self.out: Dict[str, object] = {}
        moe = set(l for l in self.r.moe_layers if l < self.n_layers)
        h = self.h0
        first_moe = min(moe) if moe else None                                  # [R3.5-P (f)]
        self._pg, self._pg_graph = None, None
        if getattr(self, "pregate", False) and first_moe is not None and first_moe >= 1 and (first_moe - 1) not in moe:
            self._pg = (first_moe - 1, first_moe)
            self._pg_graph = f"dA{first_moe - 1}"
        for l in range(self.n_layers):
            pro = l == 0 and self.e_pin is not None      # [R3.4-2g] the first graph also stages the embedding row
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
            else:
                h_in = h
                a = self._capture(f"a{l}", lambda l=l, h_in=h_in, pro=pro: self._seg_a(l, h_in, pro))
                a.update(self._capture(f"p{l}", lambda l=l, a=a: self._seg_p(l, a)))   # [R3.4-2f] pred, y_sh
                self.out[f"a{l}"] = a
                if getattr(self, 'r36_overlap', False):
                    from glm53_fold_overlap_r36 import capture_hit
                    capture_hit(self, l, a['x'], a['w'], 1, 'n1', self._capture)
                h = self._capture(f"b{l}", lambda l=l, a=a: self._seg_b(l, a))
                self.out[f"b{l}"] = h
        nt, logits = self._capture("final", lambda h=h: self._seg_final(h))
        self.next_tok, self.logits = nt, logits
        # [R3.4-2f] per-token replay order: (layer, first graph, p graph | None, b graph | None, next MoE layer | None)
        self._seq = [(l, f"a{l}", f"p{l}", f"b{l}", self.r.next_moe.get(l)) if l in moe else (l, f"d{l}", None, None, None)
                     for l in range(self.n_layers)]
        if self._pg is not None:                     # [R3.5-P (f)] dA{l} first (dense entries keep their tuple layout)
            j = next(i for i, e in enumerate(self._seq) if e[0] == self._pg[0])
            self._seq.insert(j, (self._pg[0], self._pg_graph, None, None, None))
        self.captured = True

    def graph_sequence(self):
        """[R3.4-2f] the per-token replay order (see capture); benches use it for their diagnostic passes."""
        return self._seq

    # ------------------------------------------------------------------ decode loop
    @torch.no_grad()
    def step(self, tok: int, hook=None, frozen=None) -> None:
        """[R3.4-2f] ONE greedy decode step for input token `tok` - the production per-token loop (decode() and the
        benches call it): embedding -> every layer's graphs, with the host step per MoE layer -> final graph (argmax in
        self.next_tok, logits in self.logits). Graphs are looked up in self.graphs at every replay (harness wrappers
        keep working). hook(l, ids): called before pool.ensure (benches only). frozen {layer: ids}: replaces the routed
        and predicted ids (replay_bench M2 diagnostic only).
        [R3.4-2g] host -> GPU data goes through pinned rows read by graph nodes (the embedding row by the first graph,
        the slots by b{l}); the final graph advances pos_t. A pinned row is rewritten only after the graph that reads
        it has completed (the host waits on a later event of the same token, or on next_tok, first)."""
        bb, g, pool = self.bb, self.graphs, self.pool
        if self.e_pin is not None:                    # [R3.4-2g] the first graph copies this pinned row to the GPU
            self._e_np[:] = self._emb_np[int(tok)]
        else:
            e = bb.embed[torch.tensor([tok], device=bb.embed.device)].float().to(self.dev)
            self.h0.copy_(e.view(1, 1, 1, -1).expand(1, 1, bb.hc, -1), non_blocking=True)
        stream = torch.cuda.current_stream(self.dev)
        pf = self.r.prefetch
        pg_graph = getattr(self, "_pg_graph", None) if pf else None   # [R3.5-P (f)] None unless R35_PREGATE=1
        pg_pending = False
        scored = getattr(self, "pred_scores", False) and frozen is None    # [R3.5-P (e)]
        for l, n_a, n_p, n_b, nl in self._seq:
            if getattr(getattr(self,'bb',None),'r36_router',None) is not None:
                self.bb.r36_router.publish(l)
                if nl is not None:self.bb.r36_router.publish(nl)
            g[n_a].replay()
            if n_p is None:
                if pg_graph is not None and n_a == pg_graph:                    # [R3.5-P (f)]
                    self.ev_pg.record(stream)
                    pg_pending = True
                continue
            ev_a, ev_p = self.ev_a[l], self.ev_p[l]
            ev_a.record(stream)
            g[n_p].replay()                       # prediction + shared expert: queued before the host waits
            ev_p.record(stream)
            if pg_pending:                        # [R3.5-P (f)] dA finished while d / a / p were queued: prefetch now
                self.ev_pg.synchronize()
                if scored:
                    pool.prefetch_scored(l, self.pg_np.tolist(), self.pg_sc_np.tolist(), float(self.pg_ref_np[0]))
                else:
                    pool.prefetch(l, sorted(set(self.pg_np.tolist() if frozen is None else frozen[l])))
                pg_pending = False
            ev_a.synchronize()                    # a{l} done -> its routed ids are in the pinned mailbox
            ids = self.ids_np[l].tolist()
            if frozen is not None:
                ids = frozen[l]
            if hook is not None:
                hook(l, ids)
            if getattr(self, 'r36_overlap', False):
                self.r36_parts[('n1',l)].replay(ids,g['hitn1'+str(l)],g[n_b])
            else:
                if getattr(getattr(self,'bb',None),'r36_router',None) is not None:
                    from glm53_fold_route_r36 import ensure_active
                    ensure_active(pool,l,ids,self.slots_np[l])
                else:
                    pool.ensure(l, ids, out=self.slots_np[l])     # [R3.4-2g] slots -> pinned mailbox; b{l} copies them
                g[n_b].replay()
            if self.wddm_flush:                   # [R3.4-2i] A/B: submit b{l} now
                stream.query()
            if nl is not None and pf:
                ev_p.synchronize()                # p{l} done -> predicted ids in the pinned mailbox
                if scored:                        # [R3.5-P (e)] scored: confidence order + pool v6's cold gate
                    pool.prefetch_scored(nl, self.pred_np[l].tolist(), self.pred_sc_np[l].tolist(),
                                         float(self.pred_ref_np[l]))
                else:
                    pool.prefetch(nl, sorted(set(self.pred_np[l].tolist() if frozen is None else frozen[nl])))
        g["final"].replay()                           # [R3.4-2g] includes pos_t += 1

    @torch.no_grad()
    def decode(self, first_tok: int, cache, max_new: int, eos_ids, on_token=None) -> Optional[List[int]]:
        """Greedy decode after an eager prefill; returns generated ids (excluding first_tok) or None if not usable.
        [R3.5-S1] on_token(tok) -> bool: called for every generated token (after it is appended); True stops decoding
        (stop strings / client gone in the multi-turn server). None = the base loop, unchanged."""
        if not self.captured:
            self.capture()
        if not self.load_cache(cache):
            return None
        out_ids: List[int] = []
        tok = int(first_tok)
        pos = int(cache.pos)                          # [R3.4-2g] host copy of pos_t (no per-token .item() of it)
        fp = self._r39m_fp_begin()                    # [R3.9M] None unless R39_MTP_FP=1 (reference fingerprints)
        prof = self._r39t_enabled()                   # [R3.9T] False unless R39T_PROF=1 (step-time split, bookkeeping)
        mark = getattr(self.pool, "r39t_mark", None)  # [R3.9T] call-log marks (no-op unless R39T_TRACE is set)
        if mark is not None:
            mark("decode_begin")
        t0 = time.perf_counter()
        for _ in range(max_new):
            if pos >= self.t_max - 1:
                break
            if prof:                                  # [R3.9T] the same calls as step() + timing events / host stamps
                self._r39t_step(tok)
            else:
                self.step(tok)
            pos += 1
            tok = int(self.next_tok.item())
            if prof:
                self._r39t_collect()
            out_ids.append(tok)
            if fp is not None:                        # [R3.9M] bits of the logits this token came from
                fp.append(self._r39m_hash(self.logits))
            if on_token is not None and on_token(tok):    # [R3.5-S1] H3b
                break
            if tok in eos_ids:
                break
        torch.cuda.synchronize(self.dev)
        self.stats["tokens"] += len(out_ids)
        self.stats["seconds"] += time.perf_counter() - t0
        if mark is not None:
            mark("decode_end")
        return out_ids

    # ================================================================== [R3.9T] decode-step time split (R39T_PROF=1)
    # [R3.9T] Reason: before choosing between a native host loop, more copy / compute overlap and fewer misses, the
    # time of one decode step has to be split into GPU time in the graphs, GPU time stalled behind expert copies, GPU
    # time idle behind the host, and host time by activity. Bookkeeping only: _r39t_step issues exactly the calls of
    # step() (same graphs, same pool calls, same order) plus timing events on the compute stream and perf_counter
    # stamps. Unset (default) = none of this runs. Not for R36_OVERLAP / R36 routing policies / R35_PREGATE /
    # R34_WDDM_FLUSH (then the flag is ignored and step() runs).
    _R39T_HOST = ("embed", "replay", "sync_a", "ensure", "sync_p", "prefetch", "item")
    _R39T_GPU = ("a", "p", "b", "dense", "final", "gap_hit", "gap_stall")

    def _r39t_enabled(self) -> bool:
        if os.environ.get("R39T_PROF", "0").strip() != "1":
            return False
        if (getattr(self, "r36_overlap", False) or getattr(self.bb, "r36_router", None) is not None
                or getattr(self, "_pg_graph", None) is not None or self.wddm_flush):
            return False
        if getattr(self, "_r39t_all", None) is None:
            self._r39t_all = {}
            self._r39t_ev = []
            self.r39t_tag = "all"
        return True

    def _r39t_bucket(self) -> dict:
        b = self._r39t_all.get(self.r39t_tag)
        if b is None:
            b = self._r39t_all[self.r39t_tag] = {
                "steps": 0, "wall": 0.0, "host": {k: 0.0 for k in self._R39T_HOST},
                "gpu_ms": {k: 0.0 for k in self._R39T_GPU}, "n_gap_stall": 0, "n_gap_hit": 0,
                "host_ensure_stall": 0.0, "host_ensure_hit": 0.0, "host_b2a": 0.0, "pool0": None, "pool1": None}
        return b

    def r39t_reset(self) -> None:
        """[R3.9T] drop everything collected so far (the speed gate calls it after its warm-up)."""
        if getattr(self, "_r39t_all", None) is not None:
            self._r39t_all = {}

    def _r39t_event(self):
        i = self._r39t_i
        ev = self._r39t_ev
        if i == len(ev):
            ev.append(torch.cuda.Event(enable_timing=True))
        self._r39t_i = i + 1
        return ev[i]

    @torch.no_grad()
    def _r39t_step(self, tok: int, frozen=None) -> None:
        """[R3.9T] step() with stamps (see the section comment). frozen {layer: ids}: as in step() - with the same ids
        every step every expert is resident, which gives the no-miss floor of the loop."""
        pc = time.perf_counter
        bb, g, pool = self.bb, self.graphs, self.pool
        B = self._r39t_bucket()
        h = B["host"]
        st = pool.stats
        self._r39t_ps = {k: v for k, v in st.items() if isinstance(v, (int, float))}
        t0 = pc()
        if self.e_pin is not None:
            self._e_np[:] = self._emb_np[int(tok)]
        else:
            e = bb.embed[torch.tensor([tok], device=bb.embed.device)].float().to(self.dev)
            self.h0.copy_(e.view(1, 1, 1, -1).expand(1, 1, bb.hc, -1), non_blocking=True)
        stream = torch.cuda.current_stream(self.dev)
        pf = self.r.prefetch
        scored = getattr(self, "pred_scores", False) and frozen is None
        self._r39t_i = 0
        E = self._r39t_event
        moe, dense = [], []
        ev_prev = E()
        ev_prev.record(stream)
        tl = pc()
        h["embed"] += tl - t0
        t_b = None
        for l, n_a, n_p, n_b, nl in self._seq:
            g[n_a].replay()
            if n_p is None:
                ed = E()
                ed.record(stream)
                dense.append((ev_prev, ed))
                ev_prev = ed
                t = pc()
                h["replay"] += t - tl
                tl = t
                continue
            ev_a, ev_p = self.ev_a[l], self.ev_p[l]
            ev_a.record(stream)
            ea = E()
            ea.record(stream)
            g[n_p].replay()
            ev_p.record(stream)
            ep = E()
            ep.record(stream)
            t = pc()
            h["replay"] += t - tl
            if t_b is not None:
                B["host_b2a"] += t - t_b                 # host time from "b{l} queued" to "a{next} + p{next} queued"
            ev_a.synchronize()
            t_a = pc()
            h["sync_a"] += t_a - t
            ids = self.ids_np[l].tolist()
            if frozen is not None:
                ids = frozen[l]
            m0 = st["misses"] + st.get("prefetch_claimed", 0) + st.get("r39t_cuda_waits", 0) + st.get("r39t_host_waits", 0)
            pool.ensure(l, ids, out=self.slots_np[l])
            t = pc()
            stalled = (st["misses"] + st.get("prefetch_claimed", 0) + st.get("r39t_cuda_waits", 0)
                       + st.get("r39t_host_waits", 0)) > m0
            h["ensure"] += t - t_a
            B["host_ensure_stall" if stalled else "host_ensure_hit"] += t - t_a
            ew = E()
            ew.record(stream)
            g[n_b].replay()
            eb = E()
            eb.record(stream)
            t_b = tl = pc()
            h["replay"] += tl - t
            moe.append((ev_prev, ea, ep, ew, eb, stalled))
            ev_prev = eb
            if nl is not None and pf:
                ev_p.synchronize()
                t = pc()
                h["sync_p"] += t - tl
                if scored:
                    pool.prefetch_scored(nl, self.pred_np[l].tolist(), self.pred_sc_np[l].tolist(),
                                         float(self.pred_ref_np[l]))
                else:
                    pool.prefetch(nl, sorted(set(self.pred_np[l].tolist() if frozen is None else frozen[nl])))
                tl = pc()
                h["prefetch"] += tl - t
        g["final"].replay()
        ef = E()
        ef.record(stream)
        t = pc()
        h["replay"] += t - tl
        self._r39t_pend = (B, moe, dense, ev_prev, ef, t0, t)

    def _r39t_collect(self) -> None:
        """[R3.9T] after the host has read next_tok (the step's graphs have completed): book the step."""
        B, moe, dense, ev_last, ef, t0, t_end = self._r39t_pend
        t = time.perf_counter()
        B["host"]["item"] += t - t_end
        B["wall"] += t - t0
        B["steps"] += 1
        G = B["gpu_ms"]
        for e0, ea, ep, ew, eb, stalled in moe:
            G["a"] += e0.elapsed_time(ea)
            G["p"] += ea.elapsed_time(ep)
            gap = ep.elapsed_time(ew)
            if stalled:
                G["gap_stall"] += gap
                B["n_gap_stall"] += 1
            else:
                G["gap_hit"] += gap
                B["n_gap_hit"] += 1
            G["b"] += ew.elapsed_time(eb)
        for e0, e1 in dense:
            G["dense"] += e0.elapsed_time(e1)
        G["final"] += ev_last.elapsed_time(ef)
        acc = B["pool1"]
        if acc is None:
            acc = B["pool1"] = {}
        ps = self._r39t_ps
        for k, v in self.pool.stats.items():             # the pool's counters of this step only
            if isinstance(v, (int, float)):
                acc[k] = acc.get(k, 0) + v - ps.get(k, 0)

    def r39t_summary(self) -> dict:
        """[R3.9T] per tag (prompt) and per token: wall ms; GPU ms by graph kind (a = attention + router, includes the
        idle time while the host queues it; p = prediction + shared expert; b = routed experts + merge; dense; final),
        gap_stall = compute stream between p{l} and b{l} in layers whose ensure() had to copy / wait (copy stall +
        the host time of that ensure), gap_hit = the same gap in all-hit layers (GPU idle behind the host); host ms by
        activity (replay = graph launches + event records, sync_a / sync_p = waiting for the GPU, ensure / prefetch =
        the pool calls, item = waiting for the final graph); pool counters per token."""
        out = {}
        for tag, B in (getattr(self, "_r39t_all", None) or {}).items():
            n = max(1, B["steps"])
            d = {"steps": B["steps"], "wall_ms": round(1000.0 * B["wall"] / n, 3),
                 "gpu_ms": {k: round(v / n, 3) for k, v in B["gpu_ms"].items()},
                 "host_ms": {k: round(1000.0 * v / n, 3) for k, v in B["host"].items()},
                 "layers_stalled": round(B["n_gap_stall"] / n, 2), "layers_hit": round(B["n_gap_hit"] / n, 2),
                 "host_ensure_stall_ms": round(1000.0 * B["host_ensure_stall"] / n, 3),
                 "host_ensure_hit_ms": round(1000.0 * B["host_ensure_hit"] / n, 3),
                 "host_b2a_ms": round(1000.0 * B["host_b2a"] / n, 3)}
            d["gpu_graphs_ms"] = round(sum(d["gpu_ms"][k] for k in ("a", "p", "b", "dense", "final")), 3)
            d["host_busy_ms"] = round(sum(d["host_ms"][k] for k in ("embed", "replay", "ensure", "prefetch")), 3)
            d["host_wait_ms"] = round(sum(d["host_ms"][k] for k in ("sync_a", "sync_p", "item")), 3)
            pd = {}
            for k, dv in (B["pool1"] or {}).items():
                if k.startswith("t_"):
                    pd[k + "_ms"] = round(1000.0 * dv / n, 3)
                elif k.startswith("bytes_"):
                    pd[k.replace("bytes_", "gib_")] = round(dv / n / 2**30, 4)
                elif k not in ("registered_layers", "registered_rows", "registered_gib", "register_s"):
                    pd[k] = round(dv / n, 3)
            d["pool_per_token"] = pd
            out[tag] = d
        return out

    @torch.no_grad()
    def r39t_floor(self, n: int = 64) -> dict:
        """[R3.9T] no-miss floors, measured on the decode state left by the last decode call (which this destroys: the
        next decode call reloads its state as always). (1) loop floor: n steps of the production loop with the routed
        and predicted ids frozen to the last step's (every expert resident, nothing copied), split as r39t_summary.
        (2) GPU floor: the graphs of n tokens queued back to back with no host step in between (b{l} reads the slots of
        the last step) and one synchronize: `gpu_queue_ms` per token; `submit_ms` = the host time to launch them.
        (3) the same per token with a synchronize after every token."""
        if not self._r39t_enabled() or not self.captured:
            return {}
        pos = int(self.pos_t.item())
        n = max(0, min(int(n), (self.t_max - 2 - pos) // 3))
        if n < 8:
            return {"skipped": f"position {pos} too close to t_max {self.t_max}"}
        frozen = {l: self.ids_np[l].tolist() for l in self.slots}
        tok = int(self.next_tok.item())
        keep = self.r39t_tag
        self.r39t_tag = "_warm"
        for _ in range(3):
            self._r39t_step(tok, frozen=frozen)
            int(self.next_tok.item())
            self._r39t_collect()
        self._r39t_all.pop("_warm", None)
        self.r39t_tag = "_floor"
        for _ in range(n):
            self._r39t_step(tok, frozen=frozen)
            int(self.next_tok.item())
            self._r39t_collect()
        loop = self.r39t_summary().get("_floor")
        self._r39t_all.pop("_floor", None)
        self.r39t_tag = keep
        g = self.graphs
        torch.cuda.synchronize(self.dev)
        pc = time.perf_counter
        t0 = pc()
        for _ in range(n):
            for l, n_a, n_p, n_b, nl in self._seq:
                g[n_a].replay()
                if n_p is not None:
                    g[n_p].replay()
                    g[n_b].replay()
            g["final"].replay()
        t1 = pc()
        torch.cuda.synchronize(self.dev)
        t2 = pc()
        for _ in range(n):
            for l, n_a, n_p, n_b, nl in self._seq:
                g[n_a].replay()
                if n_p is not None:
                    g[n_p].replay()
                    g[n_b].replay()
            g["final"].replay()
            torch.cuda.synchronize(self.dev)
        t3 = pc()
        launches = sum(1 if e[2] is None else 3 for e in self._seq) + 1
        return {"n": n, "loop_no_miss": loop, "graph_launches_per_token": launches,
                "gpu_queue_ms": round(1000.0 * (t2 - t0) / n, 3), "submit_ms": round(1000.0 * (t1 - t0) / n, 3),
                "gpu_per_token_sync_ms": round(1000.0 * (t3 - t2) / n, 3)}

    # ================================================================== [R3.5-M1] N = 2 verify graph set (MTP step 1)
    # See the module docstring. Every name below is new; the N = 1 code above is not touched by it. Rows of a two-token
    # activation live in dim 1 ([1, 2, ...]) or dim 0 for the flat MoE input ([2, D]); pair order everywhere = row 0's 8
    # experts, then row 1's 8.

    # ------------------------------------------------------------------ host-side API
    def require_mtp(self) -> None:
        """[R3.5-M1] MTP decoding needs the MTP layer 45 (draft head). Raises a clear error when it is not served
        (--no-mtp-layer / load_mtp=False, or no layer_45 file); the N = 2 verify set itself does not need it."""
        if getattr(self,'r36_longctx',False):
            raise ValueError('R36_LONGCTX + MTP graphs is not yet supported; use N=1')
        if getattr(getattr(self,'bb',None),'r36_router',None) is not None:self.bb.r36_router.policy.require_mtp()
        L = int(self.cfg.num_hidden_layers)
        if not getattr(self.r, "load_mtp", True) or L not in self.r.moe_layers:
            raise RuntimeError(f"MTP decoding needs the MTP layer {L} (backbone + expert file), which is not loaded "
                               "(--no-mtp-layer / load_mtp=False or no layer file): run without --no-mtp-layer or decode "
                               "with N = 1")

    def pair_ok(self, pos: int) -> bool:
        """[R3.5-M1] True when an N = 2 step may run at host position `pos` (the length of the decoded context): it writes
        the DSA caches at pos and pos + 1, and both must satisfy the N = 1 bound of decode() (a step at q needs
        q < t_max - 1), so the causal mask stays the indexer's selection (t_max <= index_topk). Otherwise use step()."""
        return bool(self.captured2) and int(pos) + 1 < self.t_max - 1

    @staticmethod
    def pair_tokens(t_draft: int, res: Tuple[int, int, bool], eos_ids=()) -> Tuple[List[int], bool]:
        """[R3.5-M1] EOS hook for the MTP decoder: tokens one verify step emits and whether generation ends there.
        res = result2() = (tok0, tok1, accepted). Accepted: [t_draft, tok1] (t_draft == tok0 by the accept rule); an EOS
        draft ends generation and drops the bonus token tok1. Rejected: [tok0] (the corrected token)."""
        tok0, tok1, acc = res
        eos = set(int(e) for e in eos_ids)
        if acc:
            if int(t_draft) in eos:
                return [int(t_draft)], True
            return [int(t_draft), int(tok1)], int(tok1) in eos
        return [int(tok0)], int(tok0) in eos

    def graph_sequence2(self):
        """[R3.5-M1] per-step replay order of the N = 2 set (same tuple layout as graph_sequence(); names index graphs2)."""
        return self._seq2

    def state_snapshot(self, into: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        """[R3.5-M1] device copy of the mutable decode state shared by both sets: pos_t, KDA recurrent + conv states,
        DSA latent (+ indexer) caches. `into` = an earlier snapshot whose buffers are overwritten (no allocation)."""
        if into is None:
            return {"pos": self.pos_t.clone(),
                    "kda": {l: {k: v.clone() for k, v in st.items()} for l, st in self.kda.items()},
                    "dsa": {l: {k: v.clone() for k, v in st.items()} for l, st in self.dsa.items()}}
        into["pos"].copy_(self.pos_t)
        for grp in ("kda", "dsa"):
            for l, st in getattr(self, grp).items():
                for k, v in st.items():
                    into[grp][l][k].copy_(v)
        return into

    def state_restore(self, snap: Dict[str, object]) -> None:
        """[R3.5-M1] write a state_snapshot() back into the static buffers (in place: the graphs hold their addresses)."""
        self.pos_t.copy_(snap["pos"])
        for grp in ("kda", "dsa"):
            for l, st in getattr(self, grp).items():
                for k, v in st.items():
                    v.copy_(snap[grp][l][k])

    # ------------------------------------------------------------------ static state of the N = 2 set
    def _alloc_state2(self):
        bb, cfg = self.bb, self.cfg
        D, Kx, dev = cfg.hidden_size, cfg.num_experts_per_tok, self.dev
        cuda = dev.type == "cuda"                    # (CPU only in the logic unit test: no pinned memory, no events)
        self.h0_2 = torch.zeros((1, 2, bb.hc, D), dtype=torch.float32, device=dev)
        self.pos2_t = torch.zeros((2,), dtype=torch.long, device=dev)
        self.ar2_t = torch.arange(2, device=dev)
        self.acc2_t = torch.zeros((1,), dtype=torch.long, device=dev)
        # host -> GPU: [t_last, t_draft, force] (force -1 = the accept rule; 0 / 1 = forced reject / accept, benches only)
        self.tok2_pin = torch.full((3,), -1, dtype=torch.long, pin_memory=cuda)
        self.tok2_dev = torch.full((3,), -1, dtype=torch.long, device=dev)
        # GPU -> host: [token of row 0, token of row 1, accepted], written by the final graph
        self.res2_pin = torch.zeros((3,), dtype=torch.long, pin_memory=cuda)
        self._tok2_np, self.res2_np = self.tok2_pin.numpy(), self.res2_pin.numpy()
        # rollback points (the state after t_last): KDA recurrent state (4 MiB / layer) + the 3 conv windows
        self.kda_s0 = {l: torch.zeros_like(st["state"]) for l, st in self.kda.items()}
        self.kda_c0 = {l: (torch.zeros_like(st["cq"]), torch.zeros_like(st["ck"]), torch.zeros_like(st["cv"]))
                       for l, st in self.kda.items()}
        dst, src = [], []
        for l, st in self.kda.items():
            dst += [st["state"], st["cq"], st["ck"], st["cv"]]
            src += [self.kda_s0[l], *self.kda_c0[l]]
        self._rb_dst, self._rb_src = dst, src
        self._rb_tab = KR.cond_copy_table(dst, src) if cuda else None
        moe = list(getattr(self, "slots", {}))
        self.slots2 = {l: torch.zeros((2 * Kx,), dtype=torch.int32, device=dev) for l in moe}
        self.ids2_pin = torch.zeros((self.n_layers, 2 * Kx), dtype=torch.long, pin_memory=cuda)
        self.pred2_pin = torch.zeros((self.n_layers, 2 * Kx), dtype=torch.long, pin_memory=cuda)
        self.slots2_pin = torch.zeros((self.n_layers, 2 * Kx), dtype=torch.int32, pin_memory=cuda)
        self.ids2_np, self.pred2_np, self.slots2_np = self.ids2_pin.numpy(), self.pred2_pin.numpy(), self.slots2_pin.numpy()
        if getattr(self, "pred_scores", False):       # [R3.5-P (e)] both rows' prediction scores + 9th-best per row
            self.pred2_sc_pin = torch.zeros((self.n_layers, 2 * Kx), dtype=torch.float32, pin_memory=cuda)
            self.pred2_ref_pin = torch.zeros((self.n_layers, 2), dtype=torch.float32, pin_memory=cuda)
            self.pred2_sc_np, self.pred2_ref_np = self.pred2_sc_pin.numpy(), self.pred2_ref_pin.numpy()
        if cuda:
            self.ev_a2 = {l: torch.cuda.Event(blocking=self.blocking_sync) for l in moe}
            self.ev_p2 = {l: torch.cuda.Event(blocking=self.blocking_sync) for l in moe}
            self.ev_f2 = torch.cuda.Event(blocking=self.blocking_sync)
        self.e_pin2 = None
        if getattr(self, "e_pin", None) is not None:  # host-resident bf16 / fp32 table: two pinned rows, as N = 1
            emb = bb.embed
            iv = torch.int16 if emb.element_size() == 2 else torch.int32
            self.e_pin2 = torch.zeros((2, D), dtype=emb.dtype, pin_memory=cuda)
            self.e_dev2 = torch.zeros((2, D), dtype=emb.dtype, device=dev)
            self._e2_np = self.e_pin2.view(iv).numpy()
        self._pair_inflight = False

    # ------------------------------------------------------------------ row helpers (exact vs batched)
    @staticmethod
    def _row(t: torch.Tensor, i: int, dim: int = 1) -> torch.Tensor:
        """Row i of a rows-in-`dim` tensor as a size-1 slice with STANDARD strides: view() recomputes them, so the slice
        is laid out exactly like the tensor an N = 1 step holds (no copy)."""
        s = t.narrow(dim, i, 1)
        return s.view(s.shape)

    def _count(self, key: str) -> None:
        c = getattr(self, "_cap_counts", None)     # only while capture_pair() runs (ops per step for pair_info)
        if c is not None:
            c[key] = c.get(key, 0) + 1

    # op classes that exact mode runs per row; R35_MTP_BATCH (capture_pair batch_ops) may batch some of them anyway
    # (a GPU test then tells whether they stay bit-exact: a bisection knob for the fastest exact set)
    PAIR_OP_CLASSES = ("rms", "norm", "conv", "lin", "head", "route", "attn", "final", "core", "mhc")

    def _per_row(self, cls: str) -> bool:
        """True when op class `cls` runs per row (exact mode, not listed in R35_MTP_BATCH)."""
        return bool(self.pair_exact) and cls not in getattr(self, "_batch_ops", ())

    def _rows_red(self, t: torch.Tensor, fn, dim: int = 1, cls: str = "rms") -> torch.Tensor:
        """A row-local reduction: per row (the N = 1 shape), concatenated, or fn(t) at once (see _per_row)."""
        if not self._per_row(cls):
            self._count(f"{cls}_batched")
            return fn(t)
        self._count(f"{cls}_per_row")
        return torch.cat([fn(self._row(t, i, dim)) for i in range(t.shape[dim])], dim=dim)

    def _rms2(self, x: torch.Tensor, w: torch.Tensor, cls: str = "rms") -> torch.Tensor:
        """bb.rmsnorm (same op sequence) with the mean per row in exact mode; rows in dim 1."""
        xf = x.float()
        ms = self._rows_red(xf.pow(2), lambda u: u.mean(-1, keepdim=True), cls=cls)
        return w * (xf * torch.rsqrt(ms + self.bb.eps))

    def _lin2(self, x: torch.Tensor, w, cls: str = "lin") -> torch.Tensor:
        """bb.lin on rows in dim -2: one call when the FP8 GEMV kernel runs (weights read once, rows bit-identical) or in
        batched mode; else (exact mode: cuBLAS GEMMs pick kernels by the row count) one N = 1 call per row."""
        bb = self.bb
        R = int(x.shape[-2])
        if bb.lin_row_local(w, R, x.is_cuda):
            self._count("lin_fp8_both_rows")
            return bb.lin(x, w)
        if not self._per_row(cls):
            self._count(f"{cls}_batched")
            return bb.lin(x, w)
        self._count(f"{cls}_per_row")
        d = x.dim() - 2
        return torch.cat([bb.lin(self._row(x, i, d), w) for i in range(R)], dim=d)

    def _mhc_prep2(self, h, fn, base, scale):
        bb = self.bb
        if not self._per_row("mhc") or bb.mhc_prepare_row_local(h):
            return bb.mhc_prepare(h, fn, base, scale)
        parts = [bb.mhc_prepare(self._row(h, i), fn, base, scale) for i in range(h.shape[1])]
        return tuple(torch.cat([p[j] for p in parts], dim=1) for j in range(3))

    def _mhc_merge2(self, post, comb, y, h):
        bb = self.bb
        if not self._per_row("mhc") or bb.mhc_merge_row_local(h):
            return bb.mhc_merge(post, comb, y, h)
        return torch.cat([bb.mhc_merge(self._row(post, i), self._row(comb, i), self._row(y, i), self._row(h, i))
                          for i in range(h.shape[1])], dim=1)

    def _swiglu2(self, x, w1, w3, w2):
        """bb.swiglu_mlp on rows (same op sequence)."""
        lim = self.cfg.swiglu_limit
        gate = self._lin2(x, w1).clamp(max=lim)
        up = self._lin2(x, w3).clamp(-lim, lim)
        return self._lin2(F.silu(gate) * up, w2)

    def _route2(self, l, x_flat):
        """bb.route for x_flat [2, D]; exact mode: one N = 1 call per row (GEMM, top-k order and weight sum as N = 1).
        The router bias holds the expert masks (bb.set_pruned), so they apply to both rows."""
        bb = self.bb
        if not self._per_row("route"):
            self._count("route_batched")
            return bb.route(l, x_flat)
        self._count("route_per_row")
        parts = [bb.route(l, self._row(x_flat, i, 0)) for i in range(x_flat.shape[0])]
        return torch.cat([p[0] for p in parts], 0), torch.cat([p[1] for p in parts], 0)

    def _route_ids2(self, l, x_flat):
        bb = self.bb
        if not self._per_row("route"):
            return bb.route_ids(l, x_flat)[0]
        return torch.cat([bb.route_ids(l, self._row(x_flat, i, 0))[0] for i in range(x_flat.shape[0])], 0)

    # ------------------------------------------------------------------ graph-safe layer pieces (two tokens)
    def _conv_step2(self, xp_new, w_conv, conv_state, conv0):
        """Short conv for the two new inputs [x0, x1]: window [s0 s1 s2 x0 x1]; out_t = sum_k win[t + k] w[k] (each row
        with the N = 1 [1, K, C] product + sum in exact mode); conv0 <- window after x0 (rollback), state <- after x1."""
        Kc = conv_state.shape[1] + 1
        xp = torch.cat([conv_state, xp_new.float()], dim=1)              # [1, Kc + 1, C]
        wt = w_conv.T.unsqueeze(0)
        if self._per_row("conv"):
            out = torch.cat([(xp[:, t:t + Kc] * wt).sum(dim=1, keepdim=True) for t in range(2)], dim=1)
        else:
            win = torch.stack([xp[:, 0:Kc], xp[:, 1:Kc + 1]], dim=1)      # [1, 2, Kc, C]
            out = (win * wt.unsqueeze(1)).sum(dim=2)
        conv0.copy_(xp[:, 1:Kc])
        conv_state.copy_(xp[:, 2:Kc + 1])
        return F.silu(out)

    def _kda_step2(self, l, x):
        """_kda_step for two tokens: projections for both rows at once, the two recurrent updates in sequence (in place),
        kda_s0 <- the state after t_last between them."""
        if os.environ.get('R36_FUSE_KDA', '0') == '1' and getattr(self.bb, 'use_kernels', True):   # [R3.7, K2-03]
            from glm53_fold_fused_r36 import kda_graph
            return kda_graph(self, l, x, pair=True)
        bb = self.bb
        w = bb.layer(l)
        st = self.kda[l]
        c0q, c0k, c0v = self.kda_c0[l]
        H, hd = bb.H_kda, bb.hd_kda
        q = self._conv_step2(self._lin2(x, w["q"]), w["qconv"], st["cq"], c0q).view(1, 2, H, hd)
        k = self._conv_step2(self._lin2(x, w["k"]), w["kconv"], st["ck"], c0k).view(1, 2, H, hd)
        v = self._conv_step2(self._lin2(x, w["v"]), w["vconv"], st["cv"], c0v).view(1, 2, H, hd)
        qs = self._rows_red(q * q, lambda u: u.sum(-1, keepdim=True), cls="norm")
        q = (q / torch.sqrt(qs + 1e-6)) * (hd ** -0.5)
        k = k / torch.sqrt(self._rows_red(k * k, lambda u: u.sum(-1, keepdim=True), cls="norm") + 1e-6)
        g = (self._lin2(self._lin2(x, w["fa"]), w["fb"]) + w["dt_bias"]).view(1, 2, H, hd)
        g = bb.lower_bound * torch.sigmoid(self.A_exp[l] * g)
        beta = torch.sigmoid(self._lin2(x, w["b"]))                      # [1, 2, H]
        state = st["state"]
        outs = []
        for t in range(2):                                               # the N = 1 update, token by token
            state.mul_(g[:, t].exp().unsqueeze(-1))
            kv_mem = torch.matmul(k[:, t].unsqueeze(-2), state).squeeze(-2)
            delta = (v[:, t] - kv_mem) * beta[:, t].unsqueeze(-1)
            state.add_(k[:, t].unsqueeze(-1) * delta.unsqueeze(-2))
            outs.append(torch.matmul(q[:, t].unsqueeze(-2), state).squeeze(-2).unsqueeze(1))
            if t == 0:
                self.kda_s0[l].copy_(state)                              # rollback point
        out = torch.cat(outs, dim=1)                                     # [1, 2, H, hd]
        gate = self._lin2(self._lin2(x, w["ga"]), w["gb"]).view(1, 2, H, hd)
        o = self._rms2(out, w["o_norm"], cls="norm") * torch.sigmoid(gate)      # per-head norm over hd, like q / k
        return self._lin2(o.reshape(1, 2, H * hd), w["o"])

    def _dsa_step2(self, l, x):
        """_dsa_step for two tokens: both latents written at pos2 = [p, p + 1], absorbed MLA with a per-row causal mask
        (row 0 sees [0, p], row 1 sees [0, p + 1]); exact mode runs the attention per row with the N = 1 shapes."""
        bb, cfg = self.bb, self.cfg
        w = bb.layer(l)
        st = self.dsa[l]
        H, qk, vd = bb.H, bb.qk, bb.vd
        q_resid = self._rms2(self._lin2(x, w["qa"]), w["qa_norm"])
        q = self._lin2(q_resid, w["qb"]).view(1, 2, H, qk)
        kv_new = self._rms2(self._lin2(x, w["kva"]), w["kva_norm"])
        st["kv"].index_copy_(1, self.pos2_t, kv_new.float())
        if self.dsa_indexer:                                             # [R3.4-2d] (R34_DSA_INDEXER=1 only)
            ki_new = self._rows_red(self._lin2(x, w["i_wk"]), lambda u: F.layer_norm(
                u, (cfg.index_head_dim,), w["i_kn_w"], w["i_kn_b"], eps=1e-6), cls="norm")
            gs_new = self._lin2(x, w["i_gate"])
            st["ki"].index_copy_(1, self.pos2_t, ki_new.float())
            st["gs"].index_copy_(1, self.pos2_t, gs_new.float())
        W_k, W_v = bb._dsa_kv_split(l, w["kvb"])
        kv = st["kv"]
        if self._per_row("attn"):
            self._count("attn_per_row")
            outs = []
            for i in range(2):
                q_abs = torch.einsum("bshd,hdc->bshc", self._row(q, i).float(), W_k)
                scores = torch.einsum("bshc,btc->bhst", q_abs, kv) * (qk ** -0.5)
                valid = (self.ar_t <= self.pos2_t[i:i + 1]).view(1, 1, 1, self.t_max)
                scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
                probs = torch.softmax(scores, dim=-1)
                o_lat = torch.einsum("bhst,btc->bshc", probs, kv)
                outs.append(torch.einsum("bshc,hdc->bshd", o_lat, W_v))
            out = torch.cat(outs, dim=1)                                 # [1, 2, H, vd]
        else:
            self._count("attn_batched")
            q_abs = torch.einsum("bshd,hdc->bshc", q.float(), W_k)       # [1, 2, H, C]
            scores = torch.einsum("bshc,btc->bhst", q_abs, kv) * (qk ** -0.5)   # [1, H, 2, T]
            valid = (self.ar_t.view(1, -1) <= self.pos2_t.view(-1, 1)).view(1, 1, 2, self.t_max)
            scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
            probs = torch.softmax(scores, dim=-1)
            o_lat = torch.einsum("bhst,btc->bshc", probs, kv)
            out = torch.einsum("bshc,hdc->bshd", o_lat, W_v)
        return self._lin2(out.reshape(1, 2, H * vd), w["o"])

    def _attn2(self, l, h):
        bb = self.bb
        w = bb.layer(l)
        post, comb, x_in = self._mhc_prep2(h, w["hc_attn_fn"], w["hc_attn_base"], w["hc_attn_scale"])
        xn = self._rms2(x_in, w["in_norm"])
        y = self._kda_step2(l, xn) if bb.is_kda(l) else self._dsa_step2(l, xn)
        return self._mhc_merge2(post, comb, y, h)

    def _ffn_prep2(self, l, h):
        w = self.bb.layer(l)
        post, comb, xf = self._mhc_prep2(h, w["hc_ffn_fn"], w["hc_ffn_base"], w["hc_ffn_scale"])
        return post, comb, self._rms2(xf, w["post_norm"])

    # ------------------------------------------------------------------ segments of the N = 2 set (captured)
    def _prologue2(self):
        """First nodes of the first graph: the two pinned embedding rows -> h0_2 (hc copies), the pinned
        [t_last, t_draft, force] row -> tok2_dev, and pos2 = [p, p + 1]."""
        if self.e_pin2 is not None:
            self.e_dev2.copy_(self.e_pin2, non_blocking=True)
            self.h0_2.copy_(self.e_dev2.view(1, 2, 1, -1).expand(1, 2, self.bb.hc, -1))
        self.tok2_dev.copy_(self.tok2_pin, non_blocking=True)
        torch.add(self.pos_t, self.ar2_t, out=self.pos2_t)

    def _seg_dense2(self, l, h, pro=False):
        if pro:
            self._prologue2()
        h = self._attn2(l, h)
        w = self.bb.layer(l)
        post, comb, x_ffn = self._ffn_prep2(l, h)
        y = self._swiglu2(x_ffn, w["w1"], w["w3"], w["w2"])
        return self._mhc_merge2(post, comb, y, h)

    def _seg_a2(self, l, h, pro=False):
        """a{l} of the N = 2 set: attention + FFN prepare + router for both rows; ends with the D2H copy of the 16
        routed ids (row 0 then row 1) into the pinned mailbox."""
        if pro:
            self._prologue2()
        h = self._attn2(l, h)
        post, comb, x_ffn = self._ffn_prep2(l, h)
        x_flat = x_ffn.reshape(2, -1)
        top_w, top_i = self._route2(l, x_flat)
        self.ids2_pin[l].copy_(top_i.view(-1), non_blocking=True)
        return {"h": h, "post": post, "comb": comb, "x": x_flat, "w": top_w, "i": top_i}

    def _seg_p2(self, l, a):
        """p{l} of the N = 2 set: the next MoE layer's predicted ids for both rows (+ D2H) and the shared expert."""
        w = self.bb.layer(l)
        nxt = self.r.next_moe.get(l)
        if nxt is not None and getattr(self, "pred_scores", False):      # [R3.5-P (e)] per row: ids + scores + 9th
            parts = [self._route_ids9(nxt, self._row(a["x"], i, 0)) for i in range(2)]
            pred = torch.cat([p[0] for p in parts], 0)
            self.pred2_pin[l].copy_(pred.view(-1), non_blocking=True)
            self.pred2_sc_pin[l].copy_(torch.cat([p[1] for p in parts], 0).view(-1), non_blocking=True)
            self.pred2_ref_pin[l].copy_(torch.cat([p[2] for p in parts], 0).view(-1), non_blocking=True)
        elif nxt is not None:
            pred = self._route_ids2(nxt, a["x"])
            self.pred2_pin[l].copy_(pred.view(-1), non_blocking=True)
        else:
            pred = a["i"]
        y_sh = self._swiglu2(a["x"], w["sh_w1"], w["sh_w3"], w["sh_w2"])
        return {"pred": pred, "y_sh": y_sh}

    def _seg_b2(self, l, a):
        """b{l} of the N = 2 set: 16 slots from the pinned mailbox, routed experts of both rows in pair mode, merge."""
        if getattr(self, 'r36_overlap', False):
            y_r = self.r36_parts[('n2',l)].finish(a['x'],a['w'])
            return self._mhc_merge2(a['post'],a['comb'],(y_r+a['y_sh']).view(1,2,-1),a['h'])
        self.slots2[l].copy_(self.slots2_pin[l], non_blocking=True)
        rt = self.r.moe_layers[l]
        lim = self.cfg.swiglu_limit
        x, wp = a["x"].float(), a["w"].reshape(-1).float()
        if not self._per_row("core"):
            y_r = rt.decode_core(x, self.pool.pool, self.slots2[l], wp, 2, lim)
        elif self._core_row_exact.get(l, False):
            y_r = rt.decode_core(x, self.pool.pool, self.slots2[l], wp, 2, lim, row_exact=True)
        else:                       # a runtime without row_exact (RVQ archive): one exact n_tok = 1 call per token
            K = int(self.cfg.num_experts_per_tok)
            y_r = torch.cat([rt.decode_core(self._row(x, i, 0), self.pool.pool, self.slots2[l][i * K:(i + 1) * K],
                                            wp[i * K:(i + 1) * K], 1, lim) for i in range(2)], 0)
        y = (y_r + a["y_sh"]).view(1, 2, -1)
        return self._mhc_merge2(a["post"], a["comb"], y, a["h"])

    def _seg_final2(self, h):
        """Final graph of the N = 2 set: final_logits on both rows (collapsed kept: the MTP draft input), per-row argmax
        (+ head row -> token id), accept flag, pos_t += 1 + accepted, D2H of [tok0, tok1, accepted], rollback."""
        bb = self.bb
        coll = self._rows_red(h, lambda u: u.mean(dim=2), cls="final")  # [1, 2, D] (= final_logits' collapsed)
        hidden = self._rms2(coll, bb.final_norm, cls="final")
        logits = self._lin2(hidden, bb.head, cls="head")
        if bb._head_rows is not None:                # [R3.4-V] trimmed head: K logits (padded FP8 rows cut off)
            logits = logits[..., : bb._head_rows]
        nt = torch.argmax(logits[0], dim=-1)          # [2] head rows (first maximum, as torch.argmax at N = 1)
        if bb.vocab_ids is not None:                  # [R3.4-V] head row -> original token id
            nt = bb.vocab_ids.index_select(0, nt)
        acc_rule = (nt[0:1] == self.tok2_dev[1:2]).to(torch.long)      # accept rule: t_draft == argmax of row 0
        force = self.tok2_dev[2:3]
        acc = torch.where(force >= 0, force, acc_rule)
        self.acc2_t.copy_(acc)
        self.pos_t.add_(acc + 1)                      # device-side position: + 2 on accept, + 1 on reject
        self.res2_pin.copy_(torch.cat([nt, acc]), non_blocking=True)
        self._rollback2()
        return nt, logits, coll

    def _rollback2(self):
        """Rejected draft (acc2_t == 0): KDA states and conv windows <- their copies taken after t_last. Nothing to undo
        for DSA (the stale p + 1 latent is masked, then overwritten by the next step) or the pool (bytes only)."""
        if self._rb_tab is not None:
            KR.cond_copy(self._rb_tab, self.acc2_t, when=0)
        elif self._rb_dst:
            KR.cond_copy_torch(self._rb_dst, self._rb_src, self.acc2_t, when=0)

    # ------------------------------------------------------------------ capture
    def _capture2(self, name, fn):
        """As _capture, into the N = 2 set's OWN graph memory pool (N = 1 and N = 2 replays interleave freely)."""
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream(device=self.dev)
        s.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(s):
            fn()        # warm-up; its side effects on the live state are undone by capture_pair()
        torch.cuda.current_stream(self.dev).wait_stream(s)
        torch.cuda.synchronize(self.dev)
        with self._graph_ctx(g, self.mempool2):       # [R3.5-T] = torch.cuda.graph(g, pool=self.mempool2) unless guarded
            out = fn()
        self.graphs2[name] = g
        return out

    @torch.no_grad()
    def capture_pair(self, rows: Optional[str] = None, require_mtp_layer: bool = False,
                     preserve_state: bool = True, batch_ops: Optional[str] = None) -> Dict[str, object]:
        """[R3.5-M1] capture the N = 2 verify set (after the N = 1 set; captures that first if needed). rows: "exact"
        (default, env R35_MTP_ROWS) or "batched" (see the module docstring). batch_ops (env R35_MTP_BATCH, default ""):
        comma list of PAIR_OP_CLASSES that exact mode batches anyway (rms, norm, conv, lin, head, route, attn, final,
        core, mhc) - a bisection knob: a GPU test shows whether they stay bit-exact. require_mtp_layer=True (the MTP
        decoder) refuses without the MTP layer. preserve_state=True (default): the decode state is saved before and
        restored after the warm-up runs (which write it), so this may be called mid-generation (transient ~0.2 GiB
        copy); False: skip that copy - only when a load_cache() follows before the next decode step (e.g. at load, like
        the N = 1 capture). Returns (and keeps in pair_info) the VRAM growth, the configuration and the ops per step."""
        if getattr(self,'r36_longctx',False):
            raise ValueError('R36_LONGCTX currently supports N=1 graphs only; pair capture is refused')
        if getattr(getattr(self,'bb',None),'r36_router',None) is not None:self.bb.r36_router.policy.require_mtp()
        if require_mtp_layer:
            self.require_mtp()
        mode = (rows or os.environ.get("R35_MTP_ROWS", "exact")).strip().lower()
        if mode not in ("exact", "batched"):
            raise ValueError(f"R35_MTP_ROWS / rows must be 'exact' or 'batched', not {mode!r}")
        bops = os.environ.get("R35_MTP_BATCH", "") if batch_ops is None else batch_ops
        bops = sorted({b.strip().lower() for b in str(bops).split(",") if b.strip()})
        bad = [b for b in bops if b not in self.PAIR_OP_CLASSES]
        if bad:
            raise ValueError(f"R35_MTP_BATCH / batch_ops: unknown op classes {bad} (known: {self.PAIR_OP_CLASSES})")
        if self.captured2:
            if mode != self.pair_info.get("rows") or bops != self.pair_info.get("batch_ops"):
                raise RuntimeError(f"the N = 2 set is already captured with rows={self.pair_info.get('rows')} "
                                   f"batch_ops={self.pair_info.get('batch_ops')}: release_pair() first")
            return self.pair_info
        if self.t_max < 4:
            raise RuntimeError(f"t_max {self.t_max} is too short for N = 2 steps")
        if not self.captured:
            self.capture()
        self._guard_capture()                        # [R3.5-T item 10d] no-op unless R35_CAPTURE_GUARD=1
        dev = self.dev
        torch.cuda.synchronize(dev)
        m0, r0 = torch.cuda.memory_allocated(dev), torch.cuda.memory_reserved(dev)
        t0 = time.perf_counter()
        self.pair_exact = mode == "exact"
        self._batch_ops = tuple(bops)
        self._alloc_state2()
        self._pf_protect = "protect" in inspect.signature(self.pool.prefetch).parameters
        # [R3.9M] R39_MTP_PF_ROWS=1 (default 0 = the one merged prefetch call of step2, unchanged): the two rows'
        # predictions are handed to the pool's rank gates separately, as two N = 1 steps hand them (see step2). Bytes only.
        self._r39m_pf_rows = os.environ.get("R39_MTP_PF_ROWS", "0").strip() == "1"
        self._core_row_exact = {l: "row_exact" in inspect.signature(rt.decode_core).parameters
                                for l, rt in self.r.moe_layers.items() if l < self.n_layers}
        m1 = torch.cuda.memory_allocated(dev)
        self.mempool2 = torch.cuda.graph_pool_handle()
        self.graphs2: Dict[str, torch.cuda.CUDAGraph] = {}
        self.out2: Dict[str, object] = {}
        # the warm-up runs write the live state: put it back afterwards (unless a load_cache() follows anyway)
        snap = self.state_snapshot() if preserve_state else None
        self._cap_counts = {}                        # ops per step (each fn runs twice: warm-up + capture)
        try:
            self.pos_t.zero_()                       # warm-up / capture positions 0 and 1 are inside every cache
            moe = set(l for l in self.r.moe_layers if l < self.n_layers)
            h = self.h0_2
            for l in range(self.n_layers):
                pro = l == 0
                h_in = h
                if l not in moe:
                    h = self._capture2(f"d{l}", lambda l=l, h_in=h_in, pro=pro: self._seg_dense2(l, h_in, pro))
                    self.out2[f"d{l}"] = h
                else:
                    a = self._capture2(f"a{l}", lambda l=l, h_in=h_in, pro=pro: self._seg_a2(l, h_in, pro))
                    a.update(self._capture2(f"p{l}", lambda l=l, a=a: self._seg_p2(l, a)))
                    self.out2[f"a{l}"] = a
                    if getattr(self, 'r36_overlap', False):
                        from glm53_fold_overlap_r36 import capture_hit
                        if not self._per_row('core'):
                            raise ValueError('R36_OVERLAP requires exact row mode for the MTP core')
                        capture_hit(self,l,a['x'],a['w'],2,'n2',self._capture2)
                    h = self._capture2(f"b{l}", lambda l=l, a=a: self._seg_b2(l, a))
                    self.out2[f"b{l}"] = h
            nt, logits, coll = self._capture2("final", lambda h=h: self._seg_final2(h))
            self.next2, self.logits2, self.collapsed2 = nt, logits, coll
        finally:
            torch.cuda.synchronize(dev)
            if snap is not None:
                self.state_restore(snap)
                del snap
            else:
                self.pos_t.zero_()
            counts, self._cap_counts = self._cap_counts, None
            torch.cuda.synchronize(dev)
        self._seq2 = [(l, f"a{l}", f"p{l}", f"b{l}", self.r.next_moe.get(l)) if l in moe else (l, f"d{l}", None, None, None)
                      for l in range(self.n_layers)]
        self.captured2 = True
        m2, r2 = torch.cuda.memory_allocated(dev), torch.cuda.memory_reserved(dev)
        exact_moe = all(self._core_row_exact.get(l, False) for l in moe)
        core_rows = self._per_row("core")
        self.pair_info = {"rows": mode, "batch_ops": bops, "graphs": len(self.graphs2),
                          "capture_s": round(time.perf_counter() - t0, 2),
                          "rollback": "kernel" if self._rb_tab is not None else "torch",
                          "moe_core": ("pair" if (not core_rows or exact_moe) else "per-token")
                                      + ("+row_exact" if (core_rows and exact_moe) else ""),
                          "prefetch_protect": bool(self._pf_protect), "state_preserved": bool(preserve_state),
                          "ops_per_step": {k: v // 2 for k, v in sorted(counts.items())},
                          "static_gib": round((m1 - m0) / 2**30, 3), "vram_alloc_gib_delta": round((m2 - m0) / 2**30, 3),
                          "vram_reserved_gib_delta": round((r2 - r0) / 2**30, 3)}
        if self._r39m_pf_rows:                       # [R3.9M] recorded only when the flag is on (default dict unchanged)
            self.pair_info["prefetch_per_row"] = True
        return self.pair_info

    def release_pair(self) -> None:
        """[R3.5-M1] drop the N = 2 set (graphs, pool, static buffers); call torch.cuda.empty_cache() after it to return
        the memory. The N = 1 set and the decode state are untouched. [R3.5-M2] The MTP draft set reads collapsed2 /
        pos2_t / acc2_t, so it is released first."""
        if getattr(self, "mtp_captured", False):     # [R3.5-M2]
            self.release_mtp()
        if getattr(self,'r36_overlap',False):
            from glm53_fold_overlap_r36 import release_parts
            release_parts(self,'n2')
        for k in ("graphs2", "out2", "next2", "logits2", "collapsed2", "mempool2", "kda_s0", "kda_c0", "_rb_tab",
                  "_rb_dst", "_rb_src", "slots2", "h0_2", "e_pin2", "e_dev2"):
            if hasattr(self, k):
                setattr(self, k, None)
        self.captured2 = False

    # ------------------------------------------------------------------ verify step
    def stage2(self, t_last: int, t_draft: int, force: int = -1) -> None:
        """[R3.5-M1] host writes of one N = 2 step: the two embedding rows (pinned, read by the first graph) and
        [t_last, t_draft, force]. Waits first for a previous step2 whose graphs may still read the pinned rows."""
        if not self.captured2:
            raise RuntimeError("N = 2 verify set not captured: call GraphDecoder.capture_pair() first")
        if self._pair_inflight:
            self.ev_f2.synchronize()
            self._pair_inflight = False
        if self.e_pin2 is not None:
            self._e2_np[0] = self._emb_np[int(t_last)]
            self._e2_np[1] = self._emb_np[int(t_draft)]
        else:
            bb = self.bb
            e = bb.embed[torch.tensor([int(t_last), int(t_draft)], device=bb.embed.device)].float().to(self.dev)
            self.h0_2.copy_(e.view(1, 2, 1, -1).expand(1, 2, bb.hc, -1), non_blocking=True)
        self._tok2_np[0], self._tok2_np[1], self._tok2_np[2] = int(t_last), int(t_draft), int(force)

    @torch.no_grad()
    def step2(self, t_last: int, t_draft: int, hook=None, frozen=None, force: int = -1) -> None:
        """[R3.5-M1] ONE N = 2 verify step for [t_last, t_draft] at positions [p, p + 1] (p = the current position; the
        caller checks pair_ok(p) first). Same host protocol as step(): per MoE layer a2 -> event -> p2 -> event -> wait
        for a2 -> ONE pool.ensure of the 16 routed ids (row 0's 8 then row 1's 8, duplicates allowed, one slot each,
        written into the pinned slot mailbox) -> b2 -> prefetch of the distinct predicted ids of both rows (the current
        layer's keys protected). The final graph decides acceptance on the device (force -1; 0 / 1 force a reject /
        accept for benches), advances pos_t by 1 + accepted and rolls the KDA / conv states back on a reject. Read the
        outcome with result2(). hook(l, ids16): called before ensure (benches). frozen {layer: ids16}: replaces the routed
        and predicted ids (replay M2 diagnostic only)."""
        g, pool = self.graphs2, self.pool
        self.stage2(t_last, t_draft, force)
        stream = torch.cuda.current_stream(self.dev)
        pf = self.r.prefetch
        scored = getattr(self, "pred_scores", False) and frozen is None    # [R3.5-P (e)]
        for l, n_a, n_p, n_b, nl in self._seq2:
            g[n_a].replay()
            if n_p is None:
                continue
            ev_a, ev_p = self.ev_a2[l], self.ev_p2[l]
            ev_a.record(stream)
            g[n_p].replay()                       # predictions + shared expert of both rows: queued before the wait
            ev_p.record(stream)
            ev_a.synchronize()                    # a2 done -> the 16 routed ids are in the pinned mailbox
            ids = self.ids2_np[l].tolist() if frozen is None else frozen[l]
            if hook is not None:
                hook(l, ids)
            if getattr(self, 'r36_overlap', False):
                self.r36_parts[('n2',l)].replay(ids,g['hitn2'+str(l)],g[n_b])
            else:
                if getattr(getattr(self,'bb',None),'r36_router',None) is not None:
                    from glm53_fold_route_r36 import ensure_active
                    ensure_active(pool,l,ids,self.slots2_np[l])
                else:
                    pool.ensure(l, ids, out=self.slots2_np[l])     # 16 slots -> pinned mailbox; b2 copies them
                g[n_b].replay()
            if self.wddm_flush:                   # [R3.4-2i] A/B: submit b2 now
                stream.query()
            if nl is not None and pf:
                ev_p.synchronize()                # p2 done -> predicted ids of both rows in the pinned mailbox
                if scored and getattr(self, "_r39m_pf_rows", False):
                    # [R3.9M] R39_MTP_PF_ROWS=1. Reason: the pool's rank gates (R35_POOL_COLD_TOPK, R37_POOL_REG_TOPK)
                    # count ranks over the keys of ONE prefetch call. The merged call below ranks the union of both
                    # rows (up to 16 experts), so with a top-6 gate a verify step prefetches 6 experts where two
                    # N = 1 steps prefetch 6 each; the rest become demand misses. Here each row is ranked on its own
                    # (its 8 ids, its scores, its 9th-best score), exactly the call an N = 1 step makes. Each call
                    # also protects the other row's predicted keys, as the merged call does by listing them together.
                    # A prefetch moves bytes only: the slots b2 reads always hold the requested rows.
                    rf = self.pred2_ref_np[l].tolist()
                    pn, ps = self.pred2_np[l].tolist(), self.pred2_sc_np[l].tolist()
                    K = len(pn) // 2
                    prot = [(l, int(e)) for e in set(ids)]
                    for r_ in (0, 1):
                        o_ = pn[(1 - r_) * K:(2 - r_) * K]
                        pool.prefetch_scored(nl, pn[r_ * K:(r_ + 1) * K], ps[r_ * K:(r_ + 1) * K], float(rf[r_]),
                                             protect=prot + [(nl, int(e)) for e in o_])
                elif scored:                      # [R3.5-P (e)] both rows' 8 ids each, NOT deduplicated (the pool
                    rf = self.pred2_ref_np[l].tolist()      # keeps each expert's best margin, queues in confidence order)
                    K = len(self.pred2_np[l]) // 2
                    pool.prefetch_scored(nl, self.pred2_np[l].tolist(), self.pred2_sc_np[l].tolist(),
                                         [float(rf[0])] * K + [float(rf[1])] * K,
                                         protect=[(l, int(e)) for e in set(ids)])
                else:
                    pred = sorted(set(self.pred2_np[l].tolist() if frozen is None else frozen[nl]))
                    if self._pf_protect:
                        pool.prefetch(nl, pred, protect=[(l, int(e)) for e in set(ids)])
                    else:
                        pool.prefetch(nl, pred)
        g["final"].replay()                           # accept flag, pos_t += 1 + accepted, D2H, rollback
        self.ev_f2.record(stream)
        self._pair_inflight = True

    def result2(self) -> Tuple[int, int, bool]:
        """[R3.5-M1] wait for the last step2 and return (token of row 0, token of row 1, accepted). Accepted: emit t_draft
        (== row 0's token) and row 1's token (bonus); next t_last = row 1's token. Rejected: emit row 0's token; the
        state is that of one N = 1 step. See pair_tokens() for EOS."""
        self.ev_f2.synchronize()
        self._pair_inflight = False
        a, b, acc = self.res2_np.tolist()
        return int(a), int(b), bool(acc)

    # ================================================================== [R3.5-M2] MTP draft set + graph-path MTP decoding
    # See the module docstring. Every name below is new; decode(), step() and the N = 2 set never call it. Rows of the
    # draft graphs: [1, R, ...] with R = 2 (m2: the two verify rows) or R = 1 (m1 / mp / f1); flat FFN input [R, D].

    # ------------------------------------------------------------------ host-side API
    def mtp_enabled(self) -> bool:
        """[R3.5-M2] generate(): --mtp with --graphs decodes on the graph path when env R35_MTP_GRAPHS=1 ([R3.5-FIX F1]
        opt-in; default 0 = the eager MTP loop)."""
        return mtp_graphs_enabled()

    def _mtp_conf(self) -> "MtpConfig":
        if getattr(self, "mtp_cfg", None) is None:
            self.mtp_cfg = MtpConfig.from_env()
        return self.mtp_cfg

    def _sync(self) -> None:
        if self.dev.type == "cuda":
            torch.cuda.synchronize(self.dev)

    # ------------------------------------------------------------------ static state of the draft set
    def _alloc_state_mtp(self):
        bb, cfg, dev = self.bb, self.cfg, self.dev
        cuda = dev.type == "cuda"                    # (CPU only in the logic tests: no pinned memory, no events)
        D, K = cfg.hidden_size, cfg.num_experts_per_tok
        self.m_l = int(cfg.num_hidden_layers)        # the MTP layer (45)
        # layer-45 latent history: entry i = (h_i, emb(t_{i+1})); entries below m_start are masked (restored prefix)
        self.dsa45 = {"kv": torch.zeros((1, self.t_max, cfg.kv_lora_rank), dtype=torch.float32, device=dev)}
        self.m_start = torch.zeros((1,), dtype=torch.long, device=dev)
        self.m_pos1 = torch.zeros((1,), dtype=torch.long, device=dev)       # pos_t - 1 (position of a one-row graph)
        self.m_h1 = torch.zeros((1, 1, D), dtype=torch.float32, device=dev)  # hidden state of a one-row graph
        self.m_e_f = torch.zeros((2, D), dtype=torch.float32, device=dev)    # fp32 embedding rows of its token(s)
        self._m_pinned_emb = bool(cuda and getattr(self, "e_pin", None) is not None)
        if self._m_pinned_emb:                       # host-resident table: two pinned rows, copied by the graph prologue
            emb = bb.embed
            iv = torch.int16 if emb.element_size() == 2 else torch.int32
            self.m_e_pin = torch.zeros((2, D), dtype=emb.dtype, pin_memory=True)
            self.m_e_dev = torch.zeros((2, D), dtype=emb.dtype, device=dev)
            self._me_np = self.m_e_pin.view(iv).numpy()
        self.m_ids_pin = torch.zeros((2 * K,), dtype=torch.long, pin_memory=cuda)    # layer-45 routed ids, rows 0 / 1
        self.m_slots_pin = torch.zeros((K,), dtype=torch.int32, pin_memory=cuda)    # the selected row's 8 pool slots
        self.m_slots = torch.zeros((K,), dtype=torch.int32, device=dev)
        self.m_tok_pin = torch.zeros((1,), dtype=torch.long, pin_memory=cuda)       # draft token id
        self.m_prob_pin = torch.zeros((1,), dtype=torch.float32, pin_memory=cuda)   # its softmax probability
        self.m_ids_np, self.m_slots_np = self.m_ids_pin.numpy(), self.m_slots_pin.numpy()
        self.m_tok_np, self.m_prob_np = self.m_tok_pin.numpy(), self.m_prob_pin.numpy()
        self.m_x = torch.zeros((1, 1, D), dtype=torch.float32, device=dev)   # selected row: residual after attention
        self.m_xf = torch.zeros((1, D), dtype=torch.float32, device=dev)     # its FFN input (post_norm)
        self.m_w = torch.zeros((K,), dtype=torch.float32, device=dev)        # its routing weights
        if cuda:
            self.ev_m = torch.cuda.Event(blocking=self.blocking_sync)
            self.ev_mb = torch.cuda.Event(blocking=self.blocking_sync)
        self._m_pend = False                         # a draft a-graph may still read the pinned embedding rows
        last = self.n_layers - 1                     # the N = 1 set's last-layer output = h of the step's position
        self._m_last_key = f"b{last}" if f"b{last}" in (getattr(self, "out", None) or {}) else f"d{last}"

    # ------------------------------------------------------------------ host side of the draft graphs
    def _mtp_embed(self, ids) -> torch.Tensor:
        """fp32 embedding rows [n, D] on the decode device, gathered as mtp_forward does (host table: on the CPU)."""
        idx = torch.tensor([int(t) for t in ids], dtype=torch.long)
        emb = self.bb.embed
        if getattr(emb, "qweight", None) is not None:   # FP8 table (not on the fast path); a torch.Tensor has dequantize()
            return emb.dequantize(torch.float32)[idx.to(emb.qweight.device)].to(self.dev)
        if emb.device.type == "cpu":
            return emb[idx].to(self.dev).float()
        return emb[idx.to(emb.device)].float().to(self.dev)

    def _mtp_stage(self, toks) -> None:
        """Host writes of a draft graph's token rows: the pinned rows its prologue copies (host table), else straight into
        m_e_f (stream-ordered). Waits first for a draft graph that may still read the pinned rows."""
        if self._m_pend:
            self.ev_m.synchronize()
            self._m_pend = False
        if self._m_pinned_emb:
            for i, t in enumerate(toks):
                self._me_np[i] = self._emb_np[int(t)]
        else:
            self.m_e_f[: len(toks)].copy_(self._mtp_embed(list(toks)))

    def _mtp_run_a(self, name: str) -> None:
        self.graphs3[name].replay()
        self.ev_m.record(torch.cuda.current_stream(self.dev))
        self._m_pend = True

    def _mtp_finish(self, row: int, kind: str, tok: int, hook):
        """Host step of one draft: ONE pool.ensure of the selected row's 8 layer-45 ids into the pinned slot row, graph mb
        replayed right after it (pool v6 lifetime rule), ONE sync reads (draft token, probability)."""
        self.ev_m.synchronize()
        self._m_pend = False
        K = int(self.cfg.num_experts_per_tok)
        ids = self.m_ids_np[row * K:(row + 1) * K].tolist()
        if getattr(self,'r36_overlap',False):
            self.r36_parts[('mtp',self.m_l)].replay(ids,self.graphs3['hitmtp'+str(self.m_l)],self.graphs3['mb'])
        else:
            if getattr(getattr(self,'bb',None),'r36_router',None) is not None:
                from glm53_fold_route_r36 import ensure_active
                ensure_active(self.pool,self.m_l,ids,self.m_slots_np)
            else:
                self.pool.ensure(self.m_l, ids, out=self.m_slots_np)
            self.graphs3["mb"].replay()
        self.ev_mb.record(torch.cuda.current_stream(self.dev))
        self.ev_mb.synchronize()
        p = float(self.m_prob_np[0])
        d = (int(self.m_tok_np[0]), p if p == p else 0.0)     # NaN -> 0 (treated as unsure)
        if hook is not None:
            hook(kind=kind, row=row, tok=int(tok), draft=d)
        return d

    def _mtp_draft_host(self, tok: int, hook=None):
        """First draft after the prefill: row (h_{T-1} = m_h1, emb(first token)) at entry T - 1 (graph mp)."""
        self._mtp_stage([tok])
        self._mtp_run_a("mp")
        return self._mtp_finish(0, "mp", tok, hook)

    def _mtp_after_n1(self, tok: int, want: bool, hook=None):
        """After an N = 1 step at p (pos_t = p + 1): entry p = (h_p, emb(tok)); with `want` also the next draft (m1),
        else the entry only (f1, drafting paused)."""
        self._mtp_stage([tok])
        if not want:
            self._mtp_run_a("f1")
            return None
        self._mtp_run_a("m1")
        return self._mtp_finish(0, "m1", tok, hook)

    def _mtp_after_pair(self, tok0: int, tok1: int, acc: bool, want: bool, hook=None):
        """After a verify step at [p, p + 1]: entries p = (h_p, emb(tok0)) and p + 1 = (h_{p+1}, emb(tok1)) (valid when
        accepted); with `want` the draft of the selected row (1 accepted, 0 rejected; the device selects by acc2_t)."""
        self._mtp_stage([tok0, tok1])
        self._mtp_run_a("m2")
        if not want:
            return None
        row = 1 if acc else 0
        return self._mtp_finish(row, "m2", tok1 if acc else tok0, hook)

    @torch.no_grad()
    def mtp_load_history(self, h_prefill, prompt_ids, n_pre: int, T: int, restore=None) -> int:
        """[R3.5-M2] layer-45 history after an eager prefill of positions [n_pre, T): entry i = (h_i, emb(t_{i+1})) for i in
        [n_pre, T - 2] (h_prefill [1, T - n_pre, D] = the prefill's h_collapsed; prompt_ids = the T prompt ids), and
        m_h1 <- h_{T-1} (the first draft, graph mp, writes entry T - 1). Entries below the returned start are masked
        (restored prefix). Returns n_pre; T - 1 when only h_{T-1} is usable (ids / shape mismatch); T without h_prefill
        (no first draft: the history then starts with the first decode step).
        [R3.5-INT] restore = (kv [1, k, C], s, h_prev): a STORED history (entries [s, s + k) of the same token prefix, from
        the multi-turn server) put back below that start when it joins without a gap: s + k == start (e.g. skip-prefill,
        start = T - 1), or s + k == start - 1 with h_prev = h_{start-1} ([1, 1, D]) and the prompt's token at `start`
        (entry start - 1 = (h_prev, emb(t_start)) is recomputed with the ops above). Returns s then; otherwise the stored
        history is ignored (the start rules above)."""
        T, n_pre = int(T), int(n_pre)
        S = T - n_pre
        self.dsa45["kv"].zero_()                     # (hygiene: stale entries of an earlier request are masked anyway)
        if h_prefill is None or int(h_prefill.shape[1]) < 1:
            start = T
        elif prompt_ids is None or len(prompt_ids) != T or not (1 <= S == int(h_prefill.shape[1])):
            start = T - 1
            self.m_h1.copy_(h_prefill[:, -1:].float())
        else:
            start = n_pre
            self.m_h1.copy_(h_prefill[:, -1:].float())
            if S >= 2:
                bb = self.bb
                w = bb.layer(self.m_l)
                e = self._mtp_embed(prompt_ids[n_pre + 1:T]).view(1, S - 1, -1)
                hn = bb.rmsnorm(h_prefill[:, :S - 1].float(), w["hnorm"])
                x = bb.lin(torch.cat([bb.rmsnorm(e, w["enorm"]), hn], dim=-1), w["eh_proj"])
                self.dsa45["kv"][:, n_pre:T - 1].copy_(self._mtp_latent(bb.rmsnorm(x, w["in_norm"])).float())
        if restore is not None and start <= T - 1:   # [R3.5-INT] stored history below `start` (server state reuse)
            start = self._mtp_join_history(restore, start, prompt_ids, T)
        self.m_start.fill_(start)
        return start

    def _mtp_join_history(self, restore, start: int, prompt_ids, T: int) -> int:
        """[R3.5-INT] see mtp_load_history(restore=). Returns the new history start."""
        kv_r, s_r, h_prev = restore
        s_r = int(s_r)
        k = int(kv_r.shape[1]) if kv_r is not None else 0
        e_r = s_r + k
        if s_r < 0 or e_r > start:
            return start
        bridge = e_r == start - 1
        if bridge and (h_prev is None or prompt_ids is None or len(prompt_ids) != T or start >= T):
            return start
        if not bridge and e_r != start:
            return start
        if k:
            self.dsa45["kv"][:, s_r:e_r].copy_(kv_r.to(self.dev, torch.float32))
        if bridge:                                   # entry start - 1 = (h_{start-1}, emb(t_start)), the ops above
            bb = self.bb
            w = bb.layer(self.m_l)
            e = self._mtp_embed([prompt_ids[start]]).view(1, 1, -1)
            hn = bb.rmsnorm(h_prev.to(self.dev, torch.float32).reshape(1, 1, -1), w["hnorm"])
            x = bb.lin(torch.cat([bb.rmsnorm(e, w["enorm"]), hn], dim=-1), w["eh_proj"])
            self.dsa45["kv"][:, start - 1:start].copy_(self._mtp_latent(bb.rmsnorm(x, w["in_norm"])).float())
        return s_r

    def _mtp_fill_host(self, tok: int) -> None:
        """[R3.5-INT] keep_history without a first draft: entry T - 1 = (h_{T-1} = m_h1, emb(first token)) by graph mp
        (its router + D2H run as well; graph mb, the pool step and the sync do not)."""
        self._mtp_stage([tok])
        self._mtp_run_a("mp")

    def _mtp_hist_fill(self, pair: bool, tok0, tok1, acc, tok) -> None:
        """[R3.5-INT] keep_history: the history entries of a step that does not draft - m2 without mb after a verify
        step (entries p and p + 1), f1 after an N = 1 step (entry p)."""
        if pair:
            self._mtp_after_pair(tok0, tok1, acc, False)
        else:
            self._mtp_after_n1(tok, False)

    def mtp_history_export(self):
        """[R3.5-INT] the layer-45 history the last decode_mtp(keep_history=True) left, for the multi-turn server: dict
        kv [1, k, C] (a clone of the valid entries [start, start + k)), start, h_last ([1, 1, D] clone of h_{n-1}, the
        collapsed hidden state of the last processed position; None if unknown), n (= pos_t). Only entries whose next
        token lies inside the processed sequence are returned (start + k <= n - 1); the entry n - 1 is recomputed from
        h_last when a later request restores this state. None when the last decode_mtp() kept no history or the static
        state has moved on since."""
        mh = getattr(self, "m_hist", None)
        if not mh or not getattr(self, "mtp_captured", False):
            return None
        n = int(self.pos_t.item())
        if n != int(mh["pos"]):
            return None
        s = int(mh["start"])
        e = max(s, min(int(mh["end"]), n - 1))
        kind, acc = mh["last"]
        if kind == "n1":
            h = self.out[self._m_last_key].mean(dim=2)          # = the m1 / f1 graphs' h of the step's position
        elif kind == "pair":
            h = self.collapsed2[:, int(bool(acc)):int(bool(acc)) + 1]
        else:                                                   # no step ran: h_{T-1} (set by mtp_load_history)
            h = self.m_h1 if s <= n - 1 else None
        return {"kv": self.dsa45["kv"][:, s:e].clone(), "start": s, "n": n,
                "h_last": None if h is None else h.reshape(1, 1, -1).clone()}

    # ------------------------------------------------------------------ graph-safe pieces (mtp_forward's ops) + segments
    def _mtp_prologue(self) -> None:
        if self._m_pinned_emb:                       # pinned bf16 rows -> GPU -> fp32 (exact, as the eager .float())
            self.m_e_dev.copy_(self.m_e_pin, non_blocking=True)
            self.m_e_f.copy_(self.m_e_dev)

    def _mtp_pre(self, hidden, R: int):
        """enorm(emb) | hnorm(h) -> eh_proj, then in_norm (mtp_forward's first ops) for R rows."""
        bb = self.bb
        w = bb.layer(self.m_l)
        e = bb.rmsnorm(self.m_e_f[:R].view(1, R, -1), w["enorm"])
        x = bb.lin(torch.cat([e, bb.rmsnorm(hidden, w["hnorm"])], dim=-1), w["eh_proj"])
        return x, bb.rmsnorm(x, w["in_norm"])

    def _mtp_latent(self, xn):
        """The DSA latent of layer 45 (kva -> kva_norm), as RealGLM53Backbone.dsa computes kv_new."""
        bb = self.bb
        w = bb.layer(self.m_l)
        return bb.rmsnorm(bb.lin(xn, w["kva"]), w["kva_norm"])

    def _mtp_attn(self, xn, pos_rows, R: int):
        """DSA(45) for R rows at the device positions pos_rows [R]: latents written first, then absorbed MLA over the
        history with the per-row mask m_start <= t <= pos (t_max <= index_topk: the indexer selects exactly that)."""
        bb = self.bb
        w = bb.layer(self.m_l)
        H, qk, vd = bb.H, bb.qk, bb.vd
        q = bb.lin(bb.rmsnorm(bb.lin(xn, w["qa"]), w["qa_norm"]), w["qb"]).view(1, R, H, qk)
        kv = self.dsa45["kv"]
        kv.index_copy_(1, pos_rows, self._mtp_latent(xn).float())
        W_k, W_v = bb._dsa_kv_split(self.m_l, w["kvb"])
        q_abs = torch.einsum("bshd,hdc->bshc", q.float(), W_k)
        scores = torch.einsum("bshc,btc->bhst", q_abs, kv) * (qk ** -0.5)          # [1, H, R, t_max]
        ar = self.ar_t.view(1, -1)
        valid = (ar <= pos_rows.view(-1, 1)) & (ar >= self.m_start.view(1, 1))     # [R, t_max]
        scores = scores.masked_fill(~valid.view(1, 1, R, self.t_max), torch.finfo(scores.dtype).min)
        o_lat = torch.einsum("bhst,btc->bshc", torch.softmax(scores, dim=-1), kv)
        out = torch.einsum("bshc,hdc->bshd", o_lat, W_v)
        return bb.lin(out.reshape(1, R, H * vd), w["o"])

    def _seg_mtp_a(self, R: int, src: str):
        """Draft a-graphs (m2: src "pair", R = 2; m1: "n1"; mp: "host"): layer 45 up to its router, D2H of the R x 8 routed
        ids, and the drafting row's residual / FFN input / routing weights -> m_x / m_xf / m_w (read by mb). m2 selects
        that row on the device: row acc2_t (1 accepted, 0 rejected)."""
        bb, K = self.bb, int(self.cfg.num_experts_per_tok)
        w = bb.layer(self.m_l)
        self._mtp_prologue()
        if src == "pair":
            hidden, pos_rows = self.collapsed2, self.pos2_t
        else:
            if src == "n1":                          # = final_logits' collapsed of the N = 1 step (the same op)
                self.m_h1.copy_(self.out[self._m_last_key].mean(dim=2))
            torch.sub(self.pos_t, 1, out=self.m_pos1)
            hidden, pos_rows = self.m_h1, self.m_pos1
        x, xn = self._mtp_pre(hidden, R)
        x = x + self._mtp_attn(xn, pos_rows, R)
        xf = bb.rmsnorm(x, w["post_norm"]).reshape(R, -1)
        top_w, top_i = bb.route(self.m_l, xf)         # the layer-45 router bias holds its masks (prune / archive)
        self.m_ids_pin[: R * K].copy_(top_i.reshape(-1), non_blocking=True)
        if R == 1:
            self.m_x.copy_(x)
            self.m_xf.copy_(xf)
            self.m_w.copy_(top_w.reshape(-1))
        else:
            sel = self.acc2_t.clamp(0, R - 1)
            self.m_x.copy_(x.index_select(1, sel))
            self.m_xf.copy_(xf.index_select(0, sel))
            self.m_w.copy_(top_w.index_select(0, sel).reshape(-1))

    def _seg_mtp_fill(self):
        """f1: only the history entry of an N = 1 step at pos_t - 1 (drafting paused); the same ops as m1's entry."""
        self._mtp_prologue()
        self.m_h1.copy_(self.out[self._m_last_key].mean(dim=2))
        torch.sub(self.pos_t, 1, out=self.m_pos1)
        _, xn = self._mtp_pre(self.m_h1, 1)
        self.dsa45["kv"].index_copy_(1, self.m_pos1, self._mtp_latent(xn).float())

    def _seg_mtp_b(self):
        """mb: routed experts of the drafting row (8 slots from the pinned row, decode_core n_tok = 1) + shared expert,
        residual, shared_head_norm, the shared (possibly trimmed) head, argmax (-> token id), the argmax's softmax
        probability; D2H of both."""
        bb, L = self.bb, self.m_l
        w = bb.layer(L)
        if getattr(self,'r36_overlap',False):
            y_r=self.r36_parts[('mtp',L)].finish(self.m_xf,self.m_w)
        else:
            self.m_slots.copy_(self.m_slots_pin, non_blocking=True)
            y_r = self.r.moe_layers[L].decode_core(self.m_xf, self.pool.pool, self.m_slots, self.m_w, 1,
                                                    self.cfg.swiglu_limit)
        x = self.m_x + (y_r + bb.shared_expert(L, self.m_xf)).view(1, 1, -1)
        logits = bb.lin(bb.rmsnorm(x, w["shared_head_norm"]), bb.head)
        if bb._head_rows is not None:                # [R3.4-V] trimmed head: K logits (padded FP8 rows cut off)
            logits = logits[..., : bb._head_rows]
        lg = logits[0, -1]
        idx = torch.argmax(lg).view(1)
        tok = bb.vocab_ids.index_select(0, idx) if bb.vocab_ids is not None else idx
        prob = torch.exp(lg - lg.index_select(0, idx)).sum().reciprocal().view(1)
        self.m_tok_pin.copy_(tok, non_blocking=True)
        self.m_prob_pin.copy_(prob, non_blocking=True)
        return logits

    # ------------------------------------------------------------------ capture / release / pin
    def _capture3(self, name, fn):
        """As _capture2, into the draft set's OWN graph memory pool (all inter-graph data live in static buffers, so the
        N = 1, N = 2 and draft graphs replay in any order)."""
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream(device=self.dev)
        s.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(s):
            fn()                                     # warm-up (writes only the draft set's own buffers)
        torch.cuda.current_stream(self.dev).wait_stream(s)
        torch.cuda.synchronize(self.dev)
        with self._graph_ctx(g, self.mempool3):      # [R3.5-T] = torch.cuda.graph(g, pool=self.mempool3) unless guarded
            out = fn()
        self.graphs3[name] = g
        return out

    @torch.no_grad()
    def capture_mtp(self, preserve_state: bool = True) -> Dict[str, object]:
        """[R3.5-M2] capture the MTP draft set (m2, m1, mp, f1, mb) after the N = 1 and N = 2 sets (captured first if needed:
        capture_pair(require_mtp_layer=True, preserve_state)). Refuses without the MTP layer (require_mtp: --no-mtp-layer).
        Own graph pool; the warm-ups write only the draft buffers (pos_t / pos2_t are set to valid positions and restored).
        R35_MTP_PIN45=all (MtpConfig.pin45) pins the routable layer-45 experts afterwards. Returns mtp_info (VRAM growth,
        graphs, pin note, configuration, the N = 2 set's pair_info)."""
        self.require_mtp()
        if getattr(self, "mtp_captured", False):
            return self.mtp_info
        cfg = self._mtp_conf().validate()
        if not self.captured2:
            self.capture_pair(require_mtp_layer=True, preserve_state=preserve_state)
        self._guard_capture()                        # [R3.5-T item 10d] no-op unless R35_CAPTURE_GUARD=1
        dev = self.dev
        torch.cuda.synchronize(dev)
        m0, r0 = torch.cuda.memory_allocated(dev), torch.cuda.memory_reserved(dev)
        t0 = time.perf_counter()
        self._alloc_state_mtp()
        m1 = torch.cuda.memory_allocated(dev)
        self.mempool3 = torch.cuda.graph_pool_handle()
        self.graphs3: Dict[str, torch.cuda.CUDAGraph] = {}
        keep = (self.pos_t.clone(), self.pos2_t.clone())
        try:
            self.pos_t.fill_(1)                      # warm-up / capture: pos_t - 1 = 0 and pos2 = [0, 1] are in range
            self.pos2_t.copy_(self.ar2_t)
            self._capture3("m2", lambda: self._seg_mtp_a(2, "pair"))
            self._capture3("m1", lambda: self._seg_mtp_a(1, "n1"))
            self._capture3("mp", lambda: self._seg_mtp_a(1, "host"))
            self._capture3("f1", self._seg_mtp_fill)
            if getattr(self,'r36_overlap',False):
                from glm53_fold_overlap_r36 import capture_hit
                capture_hit(self,self.m_l,self.m_xf,self.m_w,1,'mtp',self._capture3)
            self.m_logits = self._capture3("mb", self._seg_mtp_b)
        finally:
            torch.cuda.synchronize(dev)
            self.pos_t.copy_(keep[0])
            self.pos2_t.copy_(keep[1])
            torch.cuda.synchronize(dev)
        self.mtp_captured = True
        self._m_pin_note = None
        pinned = self._mtp_pin() if cfg.pin45 == "all" else 0
        torch.cuda.synchronize(dev)
        m2, r2 = torch.cuda.memory_allocated(dev), torch.cuda.memory_reserved(dev)
        self.mtp_info = {"graphs": sorted(self.graphs3), "capture_s": round(time.perf_counter() - t0, 2),
                         "static_mib": round((m1 - m0) / 2**20, 2), "vram_alloc_gib_delta": round((m2 - m0) / 2**30, 3),
                         "vram_reserved_gib_delta": round((r2 - r0) / 2**30, 3), "pinned_layer45": pinned,
                         "pin_note": self._m_pin_note, "pair_info": dict(self.pair_info), "cfg": cfg.as_dict()}
        return self.mtp_info

    def release_mtp(self) -> None:
        """[R3.5-M2] drop the draft set (graphs, graph pool, buffers; torch.cuda.empty_cache() then returns the memory).
        Pinned layer-45 experts stay pinned. The N = 1 / N = 2 sets and the decode state are untouched."""
        if getattr(self,'r36_overlap',False):
            from glm53_fold_overlap_r36 import release_parts
            release_parts(self,'mtp')
        for k in ("graphs3", "mempool3", "m_logits", "dsa45", "m_start", "m_pos1", "m_h1", "m_e_f", "m_e_pin",
                  "m_e_dev", "_me_np", "m_ids_pin", "m_slots_pin", "m_slots", "m_tok_pin", "m_prob_pin", "m_ids_np",
                  "m_slots_np", "m_tok_np", "m_prob_np", "m_x", "m_xf", "m_w"):
            if hasattr(self, k):
                setattr(self, k, None)
        self.mtp_captured = False

    def _mtp_pin(self) -> int:
        """[R3.5-M2] R35_MTP_PIN45=all: pin every routable layer-45 expert in the pool (outside any capture: pin() fetches
        and records events). Not done (note in mtp_info) when the pool has no pin() or fewer than experts + 1024 slots."""
        L = self.m_l
        if not callable(getattr(self.pool, "pin", None)):
            self._m_pin_note = f"not pinned: {type(self.pool).__module__} has no pin()"
            return 0
        masked = {int(e) for e in (getattr(self.r, "prune_masks", None) or {}).get(L, ())}
        keys = [(L, e) for e in range(int(self.cfg.n_routed_experts)) if e not in masked]
        n_slots = int(getattr(self.pool, "n_slots", 0) or 0)
        if n_slots < len(keys) + 1024:
            self._m_pin_note = f"not pinned: pool of {n_slots} slots < {len(keys)} + 1024"
            return 0
        self.pool.pin(keys)
        self._m_pin_note = f"{len(keys)} layer-{L} experts pinned ({n_slots} slots)"
        return len(keys)

    def _mtp_check(self, pos: int) -> None:
        dp = int(self.pos_t.item())
        if dp != int(pos):
            raise RuntimeError(f"[R3.5-M2] position desync: host {pos}, device pos_t {dp}")

    # ------------------------------------------------------------------ decode loop
    @torch.no_grad()
    def decode_mtp(self, first_tok: int, cache, max_new: int, eos_ids, h_prefill=None, prompt_ids=None,
                   n_pre: int = 0, on_token=None, draft_hook=None, mtp_restore=None,
                   keep_history: bool = False) -> Optional[List[int]]:
        """[R3.5-M2] greedy MTP decoding on the graph path after an eager prefill. Returns the generated ids excluding
        first_tok, like decode(), or None when the context does not fit t_max. Token-identical to decode() with the N = 2
        set in rows=exact (every verify row == an N = 1 step). Per step: a pair [t_last, draft] through the N = 2 set
        when two tokens still fit max_new, pair_ok(pos) holds and the policy takes the draft; else one N = 1 step. An
        accepted EOS draft ends generation (bonus dropped); the t_max - 1 bound is decode()'s. h_prefill / prompt_ids /
        n_pre: the layer-45 history (mtp_load_history; None = no first draft). on_token(t) is called for every emitted
        token and True stops after it (S1 hook H3). draft_hook(kind=, row=, tok=, draft=) runs after every draft (tests).
        Stats: self.mtp_last (this call), self.mtp_totals (cumulative).
        [R3.5-INT] keep_history=True (the multi-turn server): steps that do not draft (end of the generation) still write
        their history entries (f1 / m2 without mb; graph mp without mb for the prompt's last position), so entries
        [m_start, pos - 1) are complete on return (mtp_history_export()). mtp_restore = (kv, start, h_prev): a stored
        history below the restored prefix (mtp_load_history(restore=)). Neither changes any emitted token."""
        if not getattr(self, "mtp_captured", False):
            self.capture_mtp(preserve_state=False)   # a load_cache() follows: no state copy needed
        self.m_hist = None                           # [R3.5-INT] (mtp_history_export) set at the end when tracked
        if not self.load_cache(cache):
            return None
        cfg = self._mtp_conf()
        T, max_new = int(cache.pos), int(max_new)
        eos = {int(e) for e in eos_ids}
        pol = MtpDraftPolicy(cfg, (list(prompt_ids) if prompt_ids is not None else []) + [int(first_tok)])
        why = {"pair": 0, "low_conf": 0, "paused": 0, "no_draft": 0, "budget": 0, "boundary": 0}
        trace = [] if cfg.trace else None
        out_ids: List[int] = []
        tok, pos = int(first_tok), T
        n_drafts, t_draft = 0, 0.0
        t0 = time.perf_counter()
        live = max_new >= 2 and self.pair_ok(pos)    # some N = 2 step can still run (monotone: then drafts are useful)
        start, draft, paused = T, None, False
        tok0 = tok1 = None
        keep = bool(keep_history)                    # [R3.5-INT] serve: complete history for mtp_history_export()
        hist_end = None                              # [R3.5-INT] entries [start, hist_end) are valid
        last = ("prefill", False)                    # [R3.5-INT] which set processed position pos - 1
        # [R3.9M] bookkeeping only, both None by default: R39_MTP_STATS=1 -> per step kind wall time + pool traffic
        # (st["r39m"]); R39_MTP_FP=1 -> the bits of the logits behind every emitted token (r39m_fp_last), so a run can
        # be compared with a plain decode() run bit for bit and not only token for token.
        prof = {} if os.environ.get("R39_MTP_STATS", "0").strip() == "1" else None
        fp = self._r39m_fp_begin()
        if fp is not None and (getattr(self, "logits2", None) is None or getattr(self, "logits", None) is None):
            fp = self.r39m_fp_last = None            # (a decoder without captured logit buffers: nothing to fingerprint)
        if live or keep:
            td = time.perf_counter()
            pm = self._r39m_mark() if prof is not None else None                    # [R3.9M]
            if mtp_restore is None:
                start = self.mtp_load_history(h_prefill, prompt_ids, n_pre, T)
            else:
                start = self.mtp_load_history(h_prefill, prompt_ids, n_pre, T, restore=mtp_restore)
            if start <= T - 1:
                if live:
                    draft = self._mtp_draft_host(tok, draft_hook)
                    n_drafts += 1
                else:                                # [R3.5-INT] keep_history: entry T - 1 without a draft
                    self._mtp_fill_host(tok)
            hist_end = T
            t_draft += time.perf_counter() - td
            if prof is not None:                     # [R3.9M] (includes the history load of the prompt)
                self._r39m_add(prof, "draft_first", pm)
        while len(out_ids) < max_new:
            if pos >= self.t_max - 1:                # decode()'s N = 1 bound
                break
            d, src = None, None
            if max_new - len(out_ids) < 2:
                reason = "budget"
            elif not self.pair_ok(pos):
                reason = "boundary"
            else:
                d, src = pol.choose(draft)
                reason = "pair" if d is not None else (
                    "low_conf" if draft is not None else ("paused" if paused else "no_draft"))
            why[reason] += 1
            pm = self._r39m_mark() if prof is not None else None                    # [R3.9M]
            if d is not None:
                self.step2(tok, d)
                tok0, tok1, acc = self.result2()
                true_next = int(tok0)
                emitted = [int(d), int(tok1)] if acc else [true_next]
                pos += 2 if acc else 1
            else:
                self.step(tok)
                true_next = int(self.next_tok.item())
                acc = False
                emitted = [true_next]
                pos += 1
            if prof is not None:                     # [R3.9M] the step has finished (result2 / item waited for it)
                self._r39m_add(prof, ("pair_acc" if acc else "pair_rej") if d is not None else "n1", pm)
            fps = None
            if fp is not None:                       # [R3.9M] verify row 0 -> first emitted token, row 1 -> the bonus
                if d is not None:
                    fps = [self._r39m_hash(self.logits2[0, 0])]
                    if acc:
                        fps.append(self._r39m_hash(self.logits2[0, 1]))
                else:
                    fps = [self._r39m_hash(self.logits)]
            n_out0 = len(out_ids)                    # [R3.9M]
            last = ("pair", bool(acc)) if d is not None else ("n1", False)      # [R3.5-INT]
            if draft is not None:
                pol.record(draft, true_next == draft[0])
            pol.update(src, bool(acc), emitted)
            if trace is not None:
                trace.append([2 if d is not None else 1, src or "", None if draft is None else round(draft[1], 4),
                              int(bool(acc)), reason])
            ended = False
            for t in emitted:
                out_ids.append(t)
                if (on_token is not None and on_token(t)) or t in eos:
                    ended = True
                    break
            if fps is not None:                      # [R3.9M] one fingerprint per token actually emitted
                fp.extend(fps[:len(out_ids) - n_out0])
            if cfg.check:
                self._mtp_check(pos)
            if ended or len(out_ids) >= max_new:
                if keep and hist_end is not None:    # [R3.5-INT] the last step's entries too (no draft)
                    td = time.perf_counter()
                    self._mtp_hist_fill(d is not None, tok0, tok1, acc, emitted[-1])
                    hist_end = pos
                    t_draft += time.perf_counter() - td
                break
            tok = emitted[-1]
            live = live and max_new - len(out_ids) >= 2 and self.pair_ok(pos)
            paused = not pol.drafting()
            draft = None
            if live:
                td = time.perf_counter()
                pm = self._r39m_mark() if prof is not None else None                # [R3.9M]
                if d is not None:
                    draft = self._mtp_after_pair(tok0, tok1, acc, not paused, draft_hook)
                else:
                    draft = self._mtp_after_n1(tok, not paused, draft_hook)
                n_drafts += int(draft is not None)
                t_draft += time.perf_counter() - td
                if prof is not None:                 # [R3.9M] draft = a-graph + ensure(45) + mb + sync; fill = entry only
                    self._r39m_add(prof, "draft" if draft is not None else "fill", pm)
                hist_end = pos
            elif keep and hist_end is not None:      # [R3.5-INT] no draft can run any more: history entries only
                td = time.perf_counter()
                self._mtp_hist_fill(d is not None, tok0, tok1, acc, tok)
                hist_end = pos
                t_draft += time.perf_counter() - td
        self._sync()
        dt = time.perf_counter() - t0
        c = pol.c
        steps = sum(why.values())
        ver, accd = c["mtp_verified"] + c["ngram_verified"], c["mtp_accepted"] + c["ngram_accepted"]
        st = {"tokens": len(out_ids), "steps": steps, "pair_steps": why["pair"], "n1_steps": steps - why["pair"],
              "n1_reasons": {k: v for k, v in why.items() if k != "pair"}, "drafts": n_drafts, "verified": ver,
              "accepted": accd, "accept_rate": round(accd / ver, 4) if ver else None,
              "mtp_accept_rate": round(c["mtp_accepted"] / c["mtp_verified"], 4) if c["mtp_verified"] else None,
              "tokens_per_step": round(len(out_ids) / max(1, steps), 4),
              "draft_ms_per_step": round(1000.0 * t_draft / max(1, steps), 3),
              "draft_ms_mean": round(1000.0 * t_draft / max(1, n_drafts), 3), "seconds": round(dt, 3),
              "tok_s": round(len(out_ids) / max(dt, 1e-9), 3), "history_start": int(start), "prompt_len": T,
              "calib": pol.calib, "cfg": cfg.as_dict(), **c}
        if trace is not None:
            st["trace"] = trace
        if prof is not None:                         # [R3.9M] R39_MTP_STATS=1 only (the default dict is unchanged)
            st["r39m"] = self._r39m_summary(prof)
        if hist_end is not None:                     # [R3.5-INT] valid layer-45 entries [start, hist_end)
            st["history_end"] = int(hist_end)
            if keep:
                self.m_hist = {"start": int(start), "end": int(hist_end), "pos": int(pos), "last": last}
        if mtp_restore is not None:
            st["history_restored"] = bool(int(start) < int(n_pre))
        self.mtp_last = st
        tot = getattr(self, "mtp_totals", None) or {"calls": 0, "tokens": 0, "steps": 0, "verified": 0,
                                                      "accepted": 0, "seconds": 0.0}
        for k, v in (("calls", 1), ("tokens", len(out_ids)), ("steps", steps), ("verified", ver), ("accepted", accd),
                     ("seconds", dt)):
            tot[k] += v
        self.mtp_totals = tot
        return out_ids


# ====================================================================== [R3.5-M2] MTP configuration and drafting policy
def mtp_graphs_enabled() -> bool:
    """[R3.5-M2] env R35_MTP_GRAPHS: 1 = --mtp with --graphs decodes on the graph path. [R3.5-FIX F1] default 0 (opt-in
    until its GPU gates - M1 Gate 0, M2 G3 / G7 - pass): unset / 0 = the eager MTP loop, i.e. the base code path."""
    return os.environ.get("R35_MTP_GRAPHS", "0").strip() == "1"


class MtpConfig:
    """[R3.5-M2] decode_mtp() settings. Env defaults (glm53_fold_generate.py --mtp-tau / --mtp-ngram / --mtp-pin45
    override them): R35_MTP_TAU 0.5 (verify a draft only if its probability >= tau; 0 = always) | R35_MTP_PAUSE 16 (steps
    without drafting after the acceptance EMA fell below tau; 0 = never pause) | R35_MTP_EMA 0.125 (EMA weight) |
    R35_MTP_SKIPRUN 8 (also pause after this many CONSECUTIVE low-confidence MTP drafts: a draft that is never verified
    costs its draft time and gains nothing; 0 = off) | R35_MTP_NGRAM 0 (off; 2..8 = prompt-lookup drafts over the last
    n..2 tokens) | R35_MTP_PIN45 none|all | R35_MTP_CHECK 0|1 (host position == pos_t after every step; one sync per step)
    | R35_MTP_TRACE 0|1 (per-step trace). All of them change only WHICH steps draft / verify, never the output."""
    FIELDS = ("tau", "pause", "ema_w", "ngram", "pin45", "check", "trace", "skip_run")

    def __init__(self, tau=0.5, pause=16, ema_w=0.125, ngram=0, pin45="none", check=False, trace=False, skip_run=8):
        self.tau, self.pause, self.ema_w = float(tau), int(pause), float(ema_w)
        self.ngram, self.pin45 = int(ngram), str(pin45).strip().lower()
        self.check, self.trace = bool(check), bool(trace)
        self.skip_run = int(skip_run)
        self.validate()

    def validate(self) -> "MtpConfig":
        if not 0.0 <= self.tau <= 1.0:
            raise ValueError(f"R35_MTP_TAU / --mtp-tau must be in [0, 1], not {self.tau}")
        if self.pause < 0:
            raise ValueError(f"R35_MTP_PAUSE must be >= 0, not {self.pause}")
        if not 0.0 < self.ema_w <= 1.0:
            raise ValueError(f"R35_MTP_EMA must be in (0, 1], not {self.ema_w}")
        if self.ngram and not 2 <= self.ngram <= 8:
            raise ValueError(f"R35_MTP_NGRAM / --mtp-ngram must be 0 or 2..8, not {self.ngram}")
        if self.pin45 not in ("none", "all"):
            raise ValueError(f"R35_MTP_PIN45 / --mtp-pin45 must be none or all, not {self.pin45!r}")
        if self.skip_run < 0:
            raise ValueError(f"R35_MTP_SKIPRUN must be >= 0, not {self.skip_run}")
        return self

    @classmethod
    def from_env(cls) -> "MtpConfig":
        e = os.environ.get
        return cls(tau=e("R35_MTP_TAU", "0.5"), pause=e("R35_MTP_PAUSE", "16"), ema_w=e("R35_MTP_EMA", "0.125"),
                   ngram=e("R35_MTP_NGRAM", "0"), pin45=e("R35_MTP_PIN45", "none"),
                   check=e("R35_MTP_CHECK", "0").strip() == "1", trace=e("R35_MTP_TRACE", "0").strip() == "1",
                   skip_run=e("R35_MTP_SKIPRUN", "8"))

    def override(self, **kw) -> "MtpConfig":
        """Replace the fields given (None = keep); validates."""
        d = self.as_dict()
        d.update({k: v for k, v in kw.items() if v is not None})
        self.__init__(**d)
        return self

    def as_dict(self) -> Dict[str, object]:
        return {k: getattr(self, k) for k in self.FIELDS}


class NgramIndex:
    """[R3.5-M2] prompt-lookup drafts: the token that followed the most recent EARLIER occurrence of the context's last n
    tokens (n = n_max .. n_min, longest first); O(n_max) per appended token."""

    def __init__(self, n_max: int = 3, n_min: int = 2):
        self.n_max = int(n_max)
        self.n_min = max(1, min(int(n_min), self.n_max))
        self.ctx: List[int] = []
        self.maps = {k: {} for k in range(self.n_min, self.n_max + 1)}

    def extend(self, toks) -> None:
        for t in toks:
            self.append(t)

    def append(self, t) -> None:
        c, i = self.ctx, len(self.ctx)
        for k, m in self.maps.items():
            if i >= k:
                m[tuple(c[i - k:i])] = i             # the k tokens before index i are followed by the token at i
        c.append(int(t))

    def lookup(self) -> Optional[int]:
        c = self.ctx
        for k in range(self.n_max, self.n_min - 1, -1):
            if len(c) >= k:
                j = self.maps[k].get(tuple(c[-k:]))
                if j is not None:
                    return c[j]
        return None


class MtpDraftPolicy:
    """[R3.5-M2] adaptive drafting (plan item 5e), host only, one instance per decode_mtp() call:
    choose(draft) -> (token, "mtp" | "ngram") or (None, None): an MTP draft is verified when its probability >= tau;
    otherwise (or when there is none) a prompt-lookup draft while the n-gram acceptance EMA >= tau (R35_MTP_NGRAM > 0).
    update(): the EMA of verified MTP drafts (start 1.0); when it falls below tau, drafting pauses for `pause` steps and the
    EMA restarts at (1 + tau) / 2; `skip_run` consecutive low-confidence MTP drafts also pause drafting (counted in
    pauses_low_conf). record(): draft-probability deciles -> [drafts, correct] (skipped drafts included, checked against
    the token the step produced): the calibration table for choosing tau."""

    def __init__(self, cfg: MtpConfig, ctx=()):
        self.tau, self.pause, self.w = float(cfg.tau), int(cfg.pause), float(cfg.ema_w)
        self.skip_run = int(getattr(cfg, "skip_run", 0))
        self.reset = 0.5 * (1.0 + self.tau)
        self.ema = self.ema_ng = 1.0
        self.pause_left = 0
        self.n_skip, self._unsure = 0, False        # consecutive low-confidence MTP drafts; this step's draft was one
        self.ng = NgramIndex(int(cfg.ngram), 2) if int(cfg.ngram) else None
        if self.ng is not None:
            self.ng.extend(ctx)
        self.c = {"mtp_verified": 0, "mtp_accepted": 0, "low_conf": 0, "ngram_verified": 0, "ngram_accepted": 0,
                  "pauses": 0, "pauses_low_conf": 0}
        self.calib = [[0, 0] for _ in range(10)]    # draft-probability decile -> [drafts, correct]

    def drafting(self) -> bool:
        return self.pause_left == 0

    def choose(self, draft):
        self._unsure = False
        if draft is not None:
            if draft[1] >= self.tau:
                return int(draft[0]), "mtp"
            self.c["low_conf"] += 1
            self._unsure = True
        if self.ng is not None and self.ema_ng >= self.tau:
            t = self.ng.lookup()
            if t is not None:
                return int(t), "ngram"
        return None, None

    def record(self, draft, correct: bool) -> None:
        b = min(9, max(0, int(float(draft[1]) * 10.0)))
        self.calib[b][0] += 1
        self.calib[b][1] += int(bool(correct))

    def update(self, src, acc: bool, emitted) -> None:
        trig = False
        if src == "mtp":
            self.n_skip = 0
            self.c["mtp_verified"] += 1
            self.c["mtp_accepted"] += int(acc)
            self.ema += self.w * (float(acc) - self.ema)
            if self.pause > 0 and self.ema < self.tau:
                self.pause_left, self.ema, trig = self.pause, self.reset, True
                self.c["pauses"] += 1
        elif src == "ngram":
            self.c["ngram_verified"] += 1
            self.c["ngram_accepted"] += int(acc)
            self.ema_ng += self.w * (float(acc) - self.ema_ng)
        if self._unsure:                             # this step's MTP draft was skipped as unsure
            self._unsure = False
            self.n_skip += 1
            if (not trig and self.skip_run > 0 and self.pause > 0 and self.pause_left == 0
                    and self.n_skip >= self.skip_run):
                self.pause_left, self.n_skip, trig = self.pause, 0, True
                self.c["pauses_low_conf"] += 1
        if not trig and self.pause_left > 0:
            self.pause_left -= 1
        if self.ng is not None:
            self.ng.extend(emitted)
