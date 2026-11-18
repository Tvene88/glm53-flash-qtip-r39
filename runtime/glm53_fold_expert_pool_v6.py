#!/usr/bin/env python3
"""
glm53_fold_expert_pool_v6.py  [Round 3.5, Claude - package P = speed item 3 of build_run/out_r32/r34_speed_plan.json]
===================================================================================================================
Pool v5 (global VRAM expert cache + registered page-cache tier, R33_REGISTER_MODE layers | rows) plus the item-3
changes. Selected with R32_POOL=6 (generate.py wiring: see INTEGRATION_P.md). Pool changes move bytes only: the slots
returned for a request always hold exactly the archive rows of the requested experts, so model outputs are
bit-identical to pool v5 (kernels read per pair; slot numbers do not enter the math).

FLAGS (env vars read once at construction; ExpertPool(..., flags={name: value}) overrides them). All off = v5:
  (a) R35_POOL_SIGNAL=1        rows-mode signalling fix in the prefetch worker: registered rows are copied first and
                               each key is signalled right after its own DMA (own CUDA event); cold rows get their own
                               event; the worker's pinned/staging blocks are double-buffered (R35_POOL_WORKER_ROWS rows
                               each, default 16), so it gathers the next cold batch while the previous one is copied.
  (b) R35_POOL_DEMAND=1        demand first: ensure() claims queued, not yet started prefetch keys and copies them
                               itself; the worker takes cold rows in claimable batches of R35_POOL_COLD_BATCH (4) and
                               drops queued cold prefetches (R35_POOL_DROP=all: also registered ones) of layers that
                               ensure() has already passed.
      R35_POOL_DEMAND_STREAM=1 (b2) ensure()'s own copies run on a dedicated high-priority stream (not queued behind
                               p{l} on the compute stream); the compute stream waits for them before ensure() returns.
      Timers (always): t_gather = t_gather_main (ensure) + t_gather_worker (prefetch worker), kept separately.
  (c) R35_POOL_GATHER=readfile cold-row gather by pool v4's multi-threaded ReadFile into pinned memory
                               (R35_POOL_GATHER_THREADS=8 per thread, R35_POOL_GATHER_SEG_MIB=2 pieces; one pool for
                               ensure, one for the worker) instead of index_select on the memory map (default "mmap").
                               Byte position = slab offset + row_of[e] * row_bytes (pruned archives with partial
                               permutations, any slab offset); a per-layer self-check at load falls back to mmap.
  (d) R35_POOL_EVICT=gdsf      GreedyDual-Size-Frequency eviction (H = L + F * C, all rows equal size) with per-tier
                               running-mean fetch costs C (registered: sampled CUDA-event DMA time, cold: gather time per
                               row + the DMA part; initial R35_POOL_COST_REG_MS=0.30 / R35_POOL_COST_COLD_MS=1.45, EMA
                               weight R35_POOL_COST_ALPHA=0.05; optional frequency halving every R35_POOL_GDSF_AGE
                               accesses, default 0 = never). Default "lru" = v5's OrderedDict LRU.
                               [R3.5-FIX2] registered sample (with (a)): the timing pair holds only the start record, the
                               bare copy and the end record (addresses, stream handle and events are made before); with
                               R35_POOL_COST_HOLD_GIL=1 (default) the start record and the copy go through a GIL-holding
                               cudart handle (no GIL hand-off inside the pair). A sample is never below the DMA time, so
                               each one is clamped at 3x the current cost and the EMA is fed the R35_POOL_COST_Q quantile
                               (default 0.2) of the last 32 samples instead of their mean.
  (e) R35_POOL_COLD_TOPK=k     confidence-gated cold prefetch: with scores (prefetch(..., scores=) / prefetch_scored()),
      R35_POOL_COLD_MARGIN=m   a cold (non-registered, non-resident) prediction is prefetched only if its confidence
                               rank is < k and / or (with score_ref, e.g. the 9th-best router score) its margin
                               score - score_ref >= m. Keys are queued in confidence order. Without scores, or with both
                               off (0): today's behaviour (scores ignored).
  (f) layer-3 pre-gate         pool side: prefetch() accepts any served layer at any time, including the first MoE layer
                               before the token's first ensure(); the drop rule of (b) uses per-layer ensure()
                               generations, so a pre-gated prefetch counts as current until that layer's ensure().
  Watchdog: R35_POOL_WATCHDOG_S (300) - an ensure() waiting longer for the worker raises instead of hanging.
  Debug (torture tests only; ExpertPool(..., debug={...}) or R35_POOL_DEBUG_*): poison (fill every reused slot with
  0xFF on the writer's stream right before its new row is written), delay_ms (random worker sleeps), gpu_sleep
  (cycles of torch.cuda._sleep on the copy stream per batch), fail_every (inject a worker exception every N points).

[R3.9T] EXPERT TRAFFIC (2026-11-17; flags R39T_*, read from the environment at construction, all default off = the
code above unchanged; each is explained where it is read in __init__; measured in run_traffic_tests.ps1, torture-tested
in test_pool_r39t.py). Every one moves the same bytes into the slots it returns; they differ in WHEN a copy is issued
and WHICH experts are resident, never in what a slot holds for a key.
  R39T_FAST_DEMAND=1|2   ensure()'s copies of registered rows through a short path (1); + hits refreshed afterwards (2)
  R39T_PREEVICT=n        prefetch() leaves n free slots for the next ensure()
  R39T_LAZY_STAGING=1    the cold-path pinned / VRAM staging blocks are allocated at first use (never, when every row
                         is registered): 0.47 GiB of VRAM for 80 more slots
  R39T_PF_AFTER_DEMAND=1 the copy stream starts a prefetch batch after ensure()'s running demand copies (no gain measured)
  R39T_PF_PACED=n        at most n prefetch copies in flight; queued ones can be claimed / dropped (no gain measured)
  R39T_PROF=1, R39T_TRACE=<file>   bookkeeping: counters for the decode-step split; the call log for traffic_sim_r39t.py
  r39t_set(fast_demand, preevict)  test hook: switch the first two on a live pool (paired A/B of quality_eval_r39s)

API (v5's, unchanged semantics; new arguments optional):
  ensure(l, experts, out=None, cold=False) -> slots, one per listed entry, in order; duplicates allowed (same slot; v5
      counts the repeat as a hit); any length (decode 8, N=2 union 16, prefill <= 288). out = host int32 array
      (pinned mailbox row) of at least len(experts) entries, written and returned; else a new int32 device tensor.
      cold=True (item 10b, prefill-only experts): misses are inserted at the cold end (LRU front / GDSF H = L) and hits
      are not refreshed, so a prefill does not flush the decode working set. ensure_cold(l, experts, out=None) = the
      same through wrappers that forward only (l, experts, out) (install_mask_check).
  prefetch(l, experts, protect=(), scores=None, score_ref=None, cold=False) -> number of keys queued.
      [R3.5-T item 10a+b] cold=True: the queued keys are placed at the cold end like ensure(..., cold=True) misses
      (the prefill prefetch while R35_PREFILL_COLD is on); prefetch_cold(l, experts, protect=()) = the same through
      wrappers. A prefetch never hands over reservations, cold or not.
  prefetch_scored(l, experts, scores, score_ref=None, protect=()) - same, passing the scores around wrappers that only
      forward (l, experts, protect) (build_run/r34_shrink_opts.install_mask_check).
  wants_scores - True when (e) is on (callers compute / copy scores only then).
  pin(keys), hit_rate(), stats (all values numeric: generate.py subtracts them), release(), close(), info().
  [R3.5-T item 10d] quiesce(timeout=None): wait until the worker finished every item handed to it and the copy stream
      ran dry (GraphDecoder calls it before a graph capture with R35_CAPTURE_GUARD=1); flush(): benches only - quiesce,
      release, synchronize, drop every unpinned resident key (same starting state for time-to-first-token runs).

LIFETIME RULE (the contract the N=2 / MTP paths build on):
  The slots ensure(l, ...) returns RESERVE their keys: a reserved key is never evicted and its slot never rewritten.
  The reservation ends at the next ensure() on a layer != l, or at release(); ensure() calls on the SAME layer
  accumulate (all their slots stay reserved together). At that hand-over the pool records a CUDA event (the reader
  fence) on the stream that is current at the call, and every later write into those slots is ordered after it:
  compute-stream copies by stream order, copies on the worker's copy stream and on the demand stream (b2) by
  cudaStreamWaitEvent(fence). So the caller's only duty: issue (or join with wait_event) every GPU read of the slots on
  the current stream BEFORE the next ensure() on another layer / release(). GraphDecoder replays b{l} right after
  ensure(l); the eager path queues its expert kernels inside forward_routed_experts; an N=2 verify or MTP draft graph
  must be replayed before the next layer's ensure(). pin()ned keys never move.
  One exception keeps small pools usable: a same-layer ensure() that finds no free or evictable slot first hands over
  the EARLIER same-layer reservations (fence recorded now, holds_overflow stat). Callers that need several same-layer
  results valid together without queuing reads in between need n_slots >= 2 * experts_per_layer + pinned + 64.
  Pool v5 offers the same in practice through LRU depth only (ensure() moves its keys to the MRU end, prefetch() evicts
  from the LRU end): fine for pools of thousands of slots, not guaranteed for tiny ones.

CONCURRENCY DESIGN (why the races named by the plan cannot happen)
  Threads: the caller (ensure / prefetch / pin / release; one thread, as in v5), one worker (prefetch copies, FIFO), and
  with (c) two ReadFile thread pools that only read files into pinned blocks. ONE lock (self.lock) guards slot_of, the
  eviction policy, free, pending, ready_cuda, _kstate, _kfence, _wfence, _slot_fence, _held, _ensure_gen, _prefetched,
  pinned and the stats both threads write. Nothing blocks while it is held (no event waits, host syncs or file I/O;
  only non-blocking CUDA calls: event create / record / query).
  Key states, key = (layer, expert):
    absent --prefetch()--> QUEUED (slot allocated; in pending + _kstate) --worker _take()--> IN FLIGHT --worker, after
    issuing its copy--> RESIDENT (ready event in ready_cuda until ensure() consumes it).
    QUEUED --ensure() (b)--> CLAIMED: ensure() copies it itself, on its own stream.
    QUEUED --worker (b), layer already passed--> CANCELLED: slot freed (no copy was ever issued).
    QUEUED / IN FLIGHT --worker exception--> FAILED: slot freed after the copy stream drained (or quarantined), pending
    event set; an ensure() waiting on it fetches the key itself.
    absent --ensure() miss--> RESIDENT (copied on the compute / demand stream before ensure() returns).
  1. Claim vs worker start: both transitions leave QUEUED under the same lock; the worker copies only keys it moved to
     IN FLIGHT and ensure() only keys it moved to CLAIMED, so exactly one of them writes the slot.
  2. Cancel vs DMA in flight: only QUEUED keys are cancelled (under the lock); a key's first copy is issued only after
     the worker moved it to IN FLIGHT, so a cancelled key never had a copy and an IN FLIGHT key is never cancelled.
  3. Eviction vs a queued b{l}: reserved keys (the current layer's) are not evictable; at the hand-over the reader
     fence lands after b{l} in stream order and every later writer of the slot waits for it on the GPU. Eviction order
     (LRU depth or GDSF priority) plays no part in this - it only decides WHICH unreserved key goes.
  4. Two writers on one slot: QUEUED / IN FLIGHT keys and keys whose worker copy has not completed (ready event not
     done) are not evictable, so a slot gets a new writer only after the old copy completed (or on the same stream).
  5. Exceptions: ensure() rolls back what it allocated or claimed (slots freed behind a fence) and re-raises; the worker
     catches everything per item, fails the undelivered keys and keeps serving; waits have a watchdog timeout.
     [R3.5-FIX] a hit's ready event leaves ready_cuda only after ensure() issued its stream wait (an ensure() that raises
     earlier leaves it there, so the slot stays unevictable until that copy completed); a failed ReadFile gather waits
     for all of its reads before it raises (no read of a failed gather lands in a later gather's pinned block).
  Deadlock freedom: one lock, never held while blocking; the worker never waits for the caller; every GPU wait refers
  to an event recorded earlier in host time on the other stream (copy stream -> fences recorded before the copy was
  queued; compute stream -> copy events recorded before the wait), so two streams can never wait on each other.
"""

from __future__ import annotations

import atexit
import concurrent.futures as _cf
import heapq
import json
import os
import queue
import random
import sys
import threading
import time
import traceback
from collections import OrderedDict, deque
from contextlib import nullcontext
from typing import Dict, Iterable, List, Optional, Tuple

import torch

_QUEUED = 1        # [R3.5-P] handed to the worker, no copy issued yet (claimable / cancellable)
_FLIGHT = 2        # [R3.5-P] the worker has taken it and is issuing its copy


class _RegisteredViews:
    """[S2] read-only Win32 file views of layer files, registered as CUDA host memory and locked."""

    def __init__(self):
        import ctypes
        from ctypes import wintypes
        self.ct = ctypes
        dll = os.path.join(os.path.dirname(torch.__file__), "lib", "cudart64_12.dll")
        self.cud = ctypes.WinDLL(dll)
        self.cud.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        self.cud.cudaHostUnregister.argtypes = [ctypes.c_void_p]
        self.cud.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.cud.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                                             ctypes.c_void_p]
        # [R3.5-FIX2 GIL-GAP] the same DLL through a GIL-holding ctypes handle (PyDLL calls do not release the GIL), used
        # only by the (d) cost-sampled registered DMA: its start-event record and its copy (both non-blocking calls) then
        # leave no GIL hand-off inside the timing pair. WinDLL calls (and torch's Event.record) release the GIL and must
        # take it back, which waits up to the 5 ms switch interval while the main thread holds it. None = not loaded (the
        # sample then records with torch and copies through the WinDLL handle).
        try:
            g = ctypes.PyDLL(dll)
            g.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                                          ctypes.c_void_p]
            g.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            self.cud_g = g
        except Exception:
            self.cud_g = None
        k = ctypes.windll.kernel32
        k.CreateFileW.restype = wintypes.HANDLE
        k.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
                                  wintypes.DWORD, wintypes.HANDLE]
        k.CreateFileMappingW.restype = wintypes.HANDLE
        k.CreateFileMappingW.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                                         wintypes.DWORD, wintypes.LPCWSTR]
        k.MapViewOfFile.restype = ctypes.c_void_p
        k.MapViewOfFile.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.c_size_t]
        k.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
        self.k32 = k
        self.views = {}

    @staticmethod
    def avail_gib():
        import ctypes
        from ctypes import wintypes

        class M(ctypes.Structure):
            _fields_ = [("l", wintypes.DWORD), ("load", wintypes.DWORD), ("tot", ctypes.c_ulonglong),
                        ("av", ctypes.c_ulonglong), ("a", ctypes.c_ulonglong), ("b", ctypes.c_ulonglong),
                        ("c", ctypes.c_ulonglong), ("d", ctypes.c_ulonglong), ("e", ctypes.c_ulonglong)]
        m = M()
        m.l = ctypes.sizeof(M)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.av / 2**30

    def add(self, l, path, slab_off, scratch, nbytes=None):
        size = os.path.getsize(path)
        if nbytes is not None:                              # [R3.3-rows] lock only the first slab rows
            size = min(size, ((int(slab_off) + int(nbytes) + 65535) // 65536) * 65536)
        hf = self.k32.CreateFileW(path, 0x80000000, 7, None, 3, 0, None)
        hm = self.k32.CreateFileMappingW(hf, None, 2, 0, 0, None)
        v = self.k32.MapViewOfFile(hm, 4, 0, 0, 0)
        if not v:
            self.k32.CloseHandle(hm); self.k32.CloseHandle(hf)
            return False
        rc = self.cud.cudaHostRegister(v, size, 8)
        if rc != 0:
            self.k32.UnmapViewOfFile(v); self.k32.CloseHandle(hm); self.k32.CloseHandle(hf)
            return False
        ch = scratch.numel()
        for off in range(0, size, ch):                      # first-touch lock: one sequential DMA pass
            self.cud.cudaMemcpy(scratch.data_ptr(), v + off, min(ch, size - off), 1)
        self.views[l] = (v, size, hm, hf, int(slab_off))
        return True

    def copy_row(self, l, e, row_bytes, dst_ptr, stream_handle):
        v, size, hm, hf, off = self.views[l]
        rc = self.cud.cudaMemcpyAsync(dst_ptr, v + off + int(e) * row_bytes, row_bytes, 1, stream_handle)
        if rc != 0:
            raise RuntimeError(f"cudaMemcpyAsync from registered view failed rc={rc} (layer {l}, expert {e})")

    def row_src(self, l, e, row_bytes):
        """[R3.5-FIX2] host address of slab row e of layer l in its locked view (copy_row's source address)."""
        v, size, hm, hf, off = self.views[l]
        return v + off + int(e) * row_bytes

    def memcpy_row(self, dst_ptr, src_ptr, row_bytes, stream_handle, l, e, hold_gil=False):
        """[R3.5-FIX2] copy_row's cudaMemcpyAsync and rc check for addresses computed beforehand (row_src); hold_gil:
        through the GIL-holding handle (if loaded)."""
        cud = self.cud_g if hold_gil and self.cud_g is not None else self.cud
        rc = cud.cudaMemcpyAsync(dst_ptr, src_ptr, row_bytes, 1, stream_handle)
        if rc != 0:
            raise RuntimeError(f"cudaMemcpyAsync from registered view failed rc={rc} (layer {l}, expert {e})")

    def record_raw(self, ev_handle, stream_handle):
        """[R3.5-FIX2] cudaEventRecord of an already created event through the GIL-holding handle."""
        rc = self.cud_g.cudaEventRecord(ev_handle, stream_handle)
        if rc != 0:
            raise RuntimeError(f"cudaEventRecord (registered cost sample) failed rc={rc}")

    def close(self):
        for v, size, hm, hf, off in self.views.values():
            self.cud.cudaHostUnregister(v)
            self.k32.UnmapViewOfFile(v); self.k32.CloseHandle(hm); self.k32.CloseHandle(hf)
        self.views.clear()


# ---------------------------------------------------------------------------------------------- [R3.5-P] flags
_FLAG_SPEC = {  # name: (env var, type, default)
    "signal": ("R35_POOL_SIGNAL", bool, False),                 # (a)
    "worker_rows": ("R35_POOL_WORKER_ROWS", int, 16),           # (a) rows per double-buffered worker block
    "demand": ("R35_POOL_DEMAND", bool, False),                 # (b)
    "cold_batch": ("R35_POOL_COLD_BATCH", int, 4),              # (b) claimable batch size of the worker
    "drop": ("R35_POOL_DROP", str, "cold"),                     # (b) cold | all
    "demand_stream": ("R35_POOL_DEMAND_STREAM", bool, False),   # (b2)
    "gather": ("R35_POOL_GATHER", str, "mmap"),                 # (c) mmap | readfile
    "gather_threads": ("R35_POOL_GATHER_THREADS", int, 8),      # (c)
    "gather_seg_mib": ("R35_POOL_GATHER_SEG_MIB", float, 2.0),  # (c)
    "evict": ("R35_POOL_EVICT", str, "lru"),                    # (d) lru | gdsf
    "cost_reg_ms": ("R35_POOL_COST_REG_MS", float, 0.30),       # (d)
    "cost_cold_ms": ("R35_POOL_COST_COLD_MS", float, 1.45),     # (d)
    "cost_alpha": ("R35_POOL_COST_ALPHA", float, 0.05),         # (d)
    "gdsf_age": ("R35_POOL_GDSF_AGE", int, 0),                  # (d)
    "cost_hold_gil": ("R35_POOL_COST_HOLD_GIL", bool, True),    # (d)+(a) [R3.5-FIX2] sampled DMA keeps the GIL
    "cost_q": ("R35_POOL_COST_Q", float, 0.2),                  # (d)+(a) [R3.5-FIX2] folded quantile of the samples
    "cold_topk": ("R35_POOL_COLD_TOPK", int, 0),                # (e)
    "cold_margin": ("R35_POOL_COLD_MARGIN", float, 0.0),        # (e)
    "reg_topk": ("R37_POOL_REG_TOPK", int, 0),                  # (r) [R3.7] as (e), for REGISTERED predictions; 0 = off
    "reg_margin": ("R37_POOL_REG_MARGIN", float, 0.0),          # (r) [R3.7]
    "watchdog_s": ("R35_POOL_WATCHDOG_S", float, 300.0),
}
_DEBUG_SPEC = {
    "poison": ("R35_POOL_DEBUG_POISON", bool, False),
    "delay_ms": ("R35_POOL_DEBUG_DELAY_MS", float, 0.0),
    "gpu_sleep": ("R35_POOL_DEBUG_GPU_SLEEP", int, 0),
    "fail_every": ("R35_POOL_DEBUG_FAIL_EVERY", int, 0),
    "seed": ("R35_POOL_DEBUG_SEED", int, 0),
}


def _parse(typ, v):
    if typ is bool:
        return v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on")
    return typ(v)


def _read_spec(spec, overrides):
    out = {}
    ov = dict(overrides or {})
    bad = sorted(set(ov) - set(spec))
    if bad:
        raise ValueError(f"ExpertPool v6: unknown option(s) {bad}")
    for name, (env, typ, dflt) in spec.items():
        if name in ov:
            out[name] = _parse(typ, ov[name])
        elif os.environ.get(env, "") != "":
            out[name] = _parse(typ, os.environ[env])
        else:
            out[name] = dflt
    return out


class _Opts:
    def __init__(self, d):
        self.__dict__.update(d)

    def as_dict(self):
        return dict(self.__dict__)


def read_flags(overrides=None) -> _Opts:
    """[R3.5-P] the item-3 switches (env R35_POOL_*, overridden by `overrides`), validated."""
    f = _Opts(_read_spec(_FLAG_SPEC, overrides))
    if f.gather not in ("mmap", "readfile"):
        raise ValueError(f"R35_POOL_GATHER must be mmap or readfile, not {f.gather!r}")
    if f.evict not in ("lru", "gdsf"):
        raise ValueError(f"R35_POOL_EVICT must be lru or gdsf, not {f.evict!r}")
    if f.drop not in ("cold", "all"):
        raise ValueError(f"R35_POOL_DROP must be cold or all, not {f.drop!r}")
    if f.cold_batch < 1 or f.worker_rows < 1 or f.gather_threads < 1 or f.cold_topk < 0:
        raise ValueError("R35_POOL_COLD_BATCH / WORKER_ROWS / GATHER_THREADS must be >= 1, COLD_TOPK >= 0")
    if not 0.0 <= f.cost_q < 1.0:
        raise ValueError(f"R35_POOL_COST_Q must be in [0, 1), not {f.cost_q!r}")
    if f.reg_topk < 0:
        raise ValueError("R37_POOL_REG_TOPK must be >= 0")
    return f


# ---------------------------------------------------------------------------------------------- [R3.5-P] GDSF
class _GDSF:
    """GreedyDual-Size-Frequency (Cherkasova 1998) with equal sizes: H(k) = L + F(k) * C(k); the smallest H goes;
    L = max(L, H(victim)) at every eviction (aging by inflation). F counts accesses and survives eviction ("perfect"
    frequency: the variant the CPU trace sims measured, policy_sim.py / evict_sim.py, not in this repository), optionally halved
    every `age_every` accesses. Lazy-deletion heap: an entry is current iff its sequence number is ver[k]."""

    def __init__(self, age_every: int = 0):
        self.L = 0.0
        self.H: Dict[Tuple[int, int], float] = {}
        self.ver: Dict[Tuple[int, int], int] = {}
        self.heap: List[Tuple[float, int, Tuple[int, int]]] = []
        self.freq: Dict[Tuple[int, int], int] = {}
        self.seq = 0
        self.n_acc = 0
        self.age_every = int(age_every)

    def __contains__(self, k):
        return k in self.H

    def __len__(self):
        return len(self.H)

    def _push(self, k, h):
        self.seq += 1
        self.H[k] = h
        self.ver[k] = self.seq
        heapq.heappush(self.heap, (h, self.seq, k))
        if len(self.heap) > 4 * len(self.H) + 4096:          # drop stale entries
            self.heap = [(hh, self.ver[kk], kk) for kk, hh in self.H.items()]
            heapq.heapify(self.heap)

    def _bump(self, k):
        f = self.freq.get(k, 0) + 1
        self.freq[k] = f
        self.n_acc += 1
        if self.age_every and self.n_acc % self.age_every == 0:
            for kk in list(self.freq):
                v = self.freq[kk] >> 1
                if v:
                    self.freq[kk] = v
                else:
                    del self.freq[kk]
            f = self.freq.get(k, 1)
        return f

    def insert(self, k, cost: float, kind: str) -> None:
        """kind: 'miss' (an access: F += 1), 'prefetch' (not an access: historical F), 'cold' (H = L)."""
        if kind == "cold":
            h = self.L
        elif kind == "prefetch":
            h = self.L + self.freq.get(k, 0) * cost
        else:
            h = self.L + self._bump(k) * cost
        self._push(k, h)

    def touch(self, k, cost: float, cold: bool = False) -> None:
        if k not in self.H or cold:
            return
        self._push(k, self.L + self._bump(k) * cost)

    def remove(self, k) -> None:
        self.H.pop(k, None)
        self.ver.pop(k, None)

    def pop_victim(self, evictable):
        """Smallest-H key with evictable(key) True (removed, L updated), or None. Skipped keys keep their entries."""
        skipped = []
        victim = None
        heap = self.heap
        while heap:
            h, s, k = heapq.heappop(heap)
            if self.ver.get(k) != s:
                continue
            if evictable(k):
                victim = k
                self.L = max(self.L, h)      # monotone (a key skipped while in flight may sit below L)
                del self.H[k]
                del self.ver[k]
                break
            skipped.append((h, s, k))
        for e in skipped:
            heapq.heappush(heap, e)
        return victim

    def order(self):
        """Current keys in eviction order (tests / diagnostics)."""
        return [k for _, _, k in sorted((h, self.ver[k], k) for k, h in self.H.items())]


# ---------------------------------------------------------------------------------------------- [R3.5-P] device ops
class _CudaDev:
    """Every CUDA call of the pool, exactly pool v5's torch / cudart calls. The CPU tests (build_run/
    test_pool_v6_policy_r35.py) replace this object with a simulated GPU (streams = executor threads)."""

    def __init__(self, device):
        self.device = device

    def alloc_pool(self, n, rb):
        return torch.empty((n, rb), dtype=torch.uint8, device=self.device)

    def alloc_pinned(self, n, rb):
        return torch.empty((n, rb), dtype=torch.uint8, pin_memory=True)

    def alloc_staging(self, n, rb):
        return torch.empty((n, rb), dtype=torch.uint8, device=self.device)

    def alloc_scratch(self, nbytes):
        return torch.empty((nbytes,), dtype=torch.uint8, device=self.device)

    def new_stream(self, priority=0):
        return torch.cuda.Stream(device=self.device, priority=priority)

    def worker_init(self):
        torch.cuda.set_device(self.device.index if self.device.index is not None else torch.cuda.current_device())

    def current_stream(self):
        return torch.cuda.current_stream(self.device)

    def stream_ctx(self, s):
        return torch.cuda.stream(s)

    def new_event(self, timing=False):
        return torch.cuda.Event(enable_timing=timing)

    def record(self, ev, stream):
        ev.record(stream)

    def query(self, ev):
        return ev.query()

    def sync_event(self, ev):
        ev.synchronize()

    def wait(self, stream, ev):
        stream.wait_event(ev)

    def elapsed_ms(self, a, b):
        return a.elapsed_time(b)

    def synchronize(self):
        torch.cuda.synchronize(self.device)

    def slots_upload(self, sl):
        return torch.tensor(sl, dtype=torch.long, pin_memory=True).to(self.device, non_blocking=True)

    def h2d(self, stg, pin, m):
        stg[:m].copy_(pin[:m], non_blocking=True)

    def scatter(self, pool, slot_t, stg, m):
        pool.index_copy_(0, slot_t, stg[:m])

    def d2d_scatter(self, pool, slot_t, src, rows):
        pool.index_copy_(0, slot_t, src.index_select(0, torch.tensor(rows, dtype=torch.long, device=self.device)))

    def dma_registered(self, reg, l, row, rb, pool, slot, stream):
        reg.copy_row(l, row, rb, pool[slot].data_ptr(), stream.cuda_stream)

    def dma_registered_prep(self, reg, l, row, rb, pool, slot, stream, hold_gil=False):
        """[R3.5-FIX2 GIL-GAP] (d) cost sample, part 1, BEFORE the timing pair: everything dma_registered does on the host
        (source / destination address, stream handle) and both timing events. raw (hold_gil and the GIL-holding handle
        loaded): the start event gets one torch record here - torch creates an event's CUDA event lazily at its first
        record - so that dma_registered_timed can record it again through cudart (the latest record is the one timed).
        Returns (t0, t1, prep)."""
        t0 = self.new_event(timing=True)
        t1 = self.new_event(timing=True)
        raw = bool(hold_gil and getattr(reg, "cud_g", None) is not None)
        if raw:
            t0.record(stream)
        prep = (reg.row_src(l, row, rb), pool[slot].data_ptr(), rb, stream.cuda_stream, l, row, raw,
                t0.cuda_event if raw else None)
        return t0, t1, prep

    def dma_registered_timed(self, reg, prep, t0, t1, stream):
        """[R3.5-FIX2 GIL-GAP] part 2 = the timing pair: start record, the bare copy (copy_row's cudaMemcpyAsync with the
        prepared addresses, same bytes), end record, back to back. raw: the start record and the copy go through the
        GIL-holding handle, so the worker keeps the GIL from the start record to the copy's submission (torch's
        Event.record releases it around cudaEventRecord, a WinDLL call around the copy); the end record may release it -
        the end event is queued behind the copy by then."""
        src, dst, rb, sh, l, row, raw, h0 = prep
        if raw:
            reg.record_raw(h0, sh)
        else:
            t0.record(stream)
        reg.memcpy_row(dst, src, rb, sh, l, row, raw)
        t1.record(stream)

    def slots_tensor(self, slots):
        return torch.tensor(slots, dtype=torch.int32).to(self.device, non_blocking=True)

    def poison(self, pool, slots):                       # debug: on the current stream
        for s in slots:
            pool[int(s)].fill_(255)

    def gpu_sleep(self, cycles):                         # debug: on the current stream
        torch.cuda._sleep(int(cycles))


class _LayerKeys:
    """[R3.9T] R39T_PREEVICT: stands in for the `locked` set of _evictable(): every key of one layer."""

    def __init__(self, layer: int):
        self.layer = layer

    def __contains__(self, key) -> bool:
        return key[0] == self.layer


class ExpertPool:
    # [R3.5-P] feature detection for callers that must also run with pools v1-v5 (getattr(pool, name, False))
    POOL_VERSION = 6
    supports_cold = True            # ensure(..., cold=True)
    supports_release = True         # release()
    supports_scores = True          # prefetch(..., scores=, score_ref=) / prefetch_scored()
    supports_cold_prefetch = True   # [R3.5-T item 10a+b] prefetch(..., cold=True) / prefetch_cold()
    supports_quiesce = True         # [R3.5-T item 10d] quiesce() (GraphDecoder capture guard), flush() (benches)

    def __init__(self, sources: Dict[int, torch.Tensor], row_bytes: int, n_slots: int, device: torch.device,
                 staging_rows: int = 48, ring_slots: int = 0, threads: int = 0,
                 files: Optional[Dict[int, Tuple[str, int]]] = None, register_gib: float = 0.0,
                 register_floor_gib: float = 10.0, flags: Optional[dict] = None, debug: Optional[dict] = None,
                 _dev=None, **_ignored):
        self.f = read_flags(flags)                          # [R3.5-P]
        self._dbg = _Opts(_read_spec(_DEBUG_SPEC, debug))
        self._dv = _dev if _dev is not None else _CudaDev(device)
        self.device = device
        self.row_bytes = int(row_bytes)
        self.n_slots = int(n_slots)
        self.pool = self._dv.alloc_pool(self.n_slots, self.row_bytes)
        self.sources = sources
        self.files = dict(files or {})
        self.row_of = {l: [int(r) for r in v.tolist()] for l, v in (_ignored.get("row_of") or {}).items()}  # [R3.3]
        # [R3.4-S3] pruned files: experts without a slab row (row_of = -1) are refused by ensure() / prefetch() (a -1 row
        # would copy from 1 row before the mapped view); only layers that have such experts are checked
        self._missing = {l: {e for e, r in enumerate(ro) if r < 0} for l, ro in self.row_of.items()}
        self._missing = {l: s for l, s in self._missing.items() if s}
        self.slot_of: Dict[Tuple[int, int], int] = {}
        self.lru: "OrderedDict[Tuple[int, int], int]" = OrderedDict()
        self._gd = _GDSF(self.f.gdsf_age) if self.f.evict == "gdsf" else None   # [R3.5-P (d)]
        self.pinned: set = set()
        self.free: List[int] = list(range(self.n_slots))
        self.lock = threading.Lock()
        self.R = int(staging_rows)
        # [R3.9T] R39T_LAZY_STAGING=1: the pinned + VRAM staging blocks of the cold-row path (48 + 2 x 16 rows = 0.47 GiB
        # of VRAM and as much pinned RAM with the production flags) are allocated at their first use, not here. Reason:
        # with every row in the registered tier (R33_REGISTER_GIB covers the archive) no row ever takes the cold path, so
        # the blocks are never touched and their VRAM can hold 80 more experts (--pool-gib + 0.47). Same copies, same
        # bytes. Default (unset) = allocated here, as before.
        self._r39t_lazy = os.environ.get("R39T_LAZY_STAGING", "0").strip() == "1"
        # [R3.9T] R39T_FAST_DEMAND=1: ensure()'s own copies of REGISTERED rows are issued by _sync_fetch_fast (below):
        # the same cudaMemcpyAsync calls (source, destination, size, stream) in the same order, without entering the
        # demand stream's torch context and without building a tensor view per row. Reason: the compute stream is
        # stalled from "a{l} done" until these copies are issued, so every host microsecond before them is step time.
        # Only with the real device object and a registered tier; otherwise _sync_fetch as before. (The torture test's
        # poison option is honoured: the slots are filled with 0xFF on the copy's stream before the copies.)
        # R39T_FAST_DEMAND=2: the same, and ensure() refreshes the eviction priority of its HITS after its copies are
        # issued instead of before (the hits of the call are locked against eviction either way; only the order in
        # which the eviction policy sees this call's hits and misses changes, i.e. which experts are resident later).
        _fd = os.environ.get("R39T_FAST_DEMAND", "0").strip()
        self._r39t_fast = _fd in ("1", "2") and isinstance(self._dv, _CudaDev)
        self._r39t_defer = _fd == "2"
        self._pool_ptr = int(self.pool.data_ptr()) if self._r39t_fast else 0
        # [R3.9T] R39T_PF_AFTER_DEMAND=1 (needs R35_POOL_DEMAND_STREAM=1 and R35_POOL_SIGNAL=1): the copy stream waits
        # for ensure()'s latest demand copies before it starts a batch of registered prefetch copies. Reason: the
        # compute stream is stalled behind the demand copies of the CURRENT layer, while a prefetch is for the NEXT
        # layer and has that layer's attention time to arrive; issued together they share the PCIe link and the stalled
        # copies take longer. Order of copies only: every slot gets the same bytes, every ready event is recorded after
        # its copy as before. No wait cycle: the demand event is recorded (host time) before the worker reads it, so
        # before the prefetch's ready event exists, so before the compute stream can wait for that ready event; the
        # reader fences the demand copies wait for were recorded on the compute stream earlier still.
        self._r39t_pfad = os.environ.get("R39T_PF_AFTER_DEMAND", "0").strip() == "1" and self.f.signal
        self._r39t_last_demand = None
        # [R3.9T] R39T_PF_PACED=n (n >= 1; needs R35_POOL_SIGNAL=1, R35_POOL_DEMAND=1 and R35_POOL_DEMAND_STREAM=1; use
        # it with R35_POOL_DROP=all): the worker keeps at most n registered prefetch copies in flight and starts none
        # while ensure()'s demand copies run (host waits in the worker thread). Reason: a copy that has been handed to
        # CUDA can no longer be withdrawn, and today the worker hands over a whole prediction at once; a key that is
        # still QUEUED when its layer's ensure() arrives is either claimed (routed: copied at once on the demand
        # stream, as today) or - with R35_POOL_DROP=all - dropped without ever being copied (not routed), and it does
        # not share the link with the demand copies the compute stream is waiting for. Which rows are copied early
        # changes; the bytes a slot holds for a key do not.
        try:
            self._r39t_paced = max(0, int(os.environ.get("R39T_PF_PACED", "0").strip() or 0))
        except ValueError:
            self._r39t_paced = 0
        if not (self.f.signal and self.f.demand and self.f.demand_stream):
            self._r39t_paced = 0
        self._r39t_flight: "deque" = deque()             # ready events of the worker's last n prefetch copies
        # [R3.9T] R39T_PREEVICT=n: prefetch(l, ...) ends by evicting until n slots are free (victims by the eviction
        # policy as always, but never an expert of layer l, whose ensure() comes next). Reason: ensure() then takes a
        # free slot for a miss instead of searching the eviction heap while the compute stream is already stalled
        # behind that miss. The victims leave a few layers earlier than they would; no byte of any slot changes.
        try:
            self._r39t_preevict = max(0, int(os.environ.get("R39T_PREEVICT", "0").strip() or 0))
        except ValueError:
            self._r39t_preevict = 0
        self._r39t_staging_allocs = 0
        # synchronous path buffers (compute stream, or the demand stream with b2)
        self.pin_a = None if self._r39t_lazy else self._dv.alloc_pinned(self.R, self.row_bytes)
        self.stg_a = None if self._r39t_lazy else self._dv.alloc_staging(self.R, self.row_bytes)
        self.ev_a = None
        # prefetch worker buffers (copy stream): v5's single block, or [R3.5-P (a)] two smaller blocks used alternately
        if self.f.signal:
            self.Rw = max(1, min(self.R, int(self.f.worker_rows)))
            if self._r39t_lazy:
                self.pin_b2, self.stg_b2 = [None, None], [None, None]
            else:
                self.pin_b2 = [self._dv.alloc_pinned(self.Rw, self.row_bytes) for _ in range(2)]
                self.stg_b2 = [self._dv.alloc_staging(self.Rw, self.row_bytes) for _ in range(2)]
            self._ev_b2 = [None, None]
            self._wb = 1
        else:
            self.Rw = self.R
            self.pin_b = None if self._r39t_lazy else self._dv.alloc_pinned(self.R, self.row_bytes)
            self.stg_b = None if self._r39t_lazy else self._dv.alloc_staging(self.R, self.row_bytes)
        self.copy_stream = self._dv.new_stream()
        self.demand_stream = self._dv.new_stream(priority=-1) if self.f.demand_stream else None   # [R3.5-P (b2)]
        self.pending: Dict[Tuple[int, int], threading.Event] = {}
        self.ready_cuda: Dict[Tuple[int, int], torch.cuda.Event] = {}
        self.q: "queue.Queue" = queue.Queue()
        self.stats = {"hits": 0, "misses": 0, "prefetch_issued": 0, "prefetch_useful": 0, "bytes_h2d": 0,
                      "t_ensure": 0.0, "t_prefetch": 0.0, "t_gather": 0.0, "t_wait_prefetch": 0.0}
        # [R3.5-P] new counters (numeric only: generate.py prints per-request deltas of every stats entry)
        self.stats.update({"t_gather_main": 0.0, "t_gather_worker": 0.0, "prefetch_claimed": 0, "prefetch_dropped": 0,
                           "prefetch_skipped_cold": 0, "prefetch_noslot": 0, "evictions": 0, "fence_waits": 0,
                           "holds_overflow": 0, "worker_errors": 0, "worker_refetch": 0})
        # [R3.5-T] cold-end prefetches (item 10a+b), quiesce() calls / seconds (item 10d), flush() calls (benches)
        self.stats.update({"prefetch_cold": 0, "quiesce_calls": 0, "t_quiesce": 0.0, "flushes": 0})
        self.stats["prefetch_skipped_reg"] = 0             # [R3.7 (r)] registered predictions the gate did not prefetch
        self._cold_pf_stash = None                         # [R3.5-T] prefetch_cold() -> prefetch() hand-over
        # [R3.9T] R39T_PROF=1 (bookkeeping for the decode-step time split; unset = no counter, no stamp): stream waits
        # ensure() issued for prefetch copies still running, host waits for the worker, host seconds in ensure()'s own
        # copies, and the worker's latency (prefetch() hand-over -> item started / all its copies issued)
        if self._r39t_pfad:                                # [R3.9T] counters of the two copy-order flags (flag only)
            self.stats["r39t_pf_waits"] = 0
        if self._r39t_preevict:
            self.stats["r39t_preevicted"] = 0
        if self._r39t_paced:
            self.stats.update({"r39t_paced_link_waits": 0, "r39t_paced_demand_waits": 0})
        self._r39t = os.environ.get("R39T_PROF", "0").strip() == "1"
        if self._r39t:
            self.stats.update({"r39t_cuda_waits": 0, "r39t_host_waits": 0, "r39t_fetch_calls": 0, "t_r39t_fetch": 0.0,
                               "r39t_wk_items": 0, "t_r39t_wk_start": 0.0, "t_r39t_wk_done": 0.0,
                               "t_r39t_pre_issue": 0.0})   # ensure() entry -> its first copy handed to CUDA
            self._r39t_te = 0.0
            self._r39t_q: "deque[float]" = deque()
        # [R3.9T] R39T_TRACE=<file.json> (bookkeeping): the list of every ensure() / prefetch() call (layer, ids, cold;
        # prefetch scores) and the caller's marks, for the offline pool simulations of traffic_sim_r39t.py. Unset = off.
        self._r39t_trace = [] if os.environ.get("R39T_TRACE", "").strip() else None
        self._wq_cv = threading.Condition()                # [R3.5-T item 10d] worker items handed over / finished
        self._wq_put = 0
        self._wq_done = 0
        self._prefetched: set = set()
        # [R3.5-P] bookkeeping of the lifetime rule and of the worker states (see the module docstring)
        self._kstate: Dict[Tuple[int, int], int] = {}      # key -> _QUEUED | _FLIGHT (keys handed to the worker)
        self._kfence: Dict[Tuple[int, int], object] = {}   # QUEUED key -> reader fence of the slot it will write
        self._wfence: Dict[Tuple[int, int], object] = {}   # IN FLIGHT key -> the same fence, while its copy is issued
        self._slot_fence: Dict[int, object] = {}           # slot -> reader fence (event) of its last readers
        self._held: Dict[Tuple[int, int], None] = {}       # reserved keys (ordered set)
        self._held_layer = None
        self._ensure_gen: Dict[int, int] = {}              # layer -> number of ensure() calls (drop rule of (b))
        self._stash = None                                 # prefetch_scored() -> prefetch() hand-over
        self._cold_stash = None                            # ensure_cold() -> ensure() hand-over
        self._worker_error = None
        self._closed = False
        self._dbg_rng = random.Random(self._dbg.seed)
        self._dbg_n = 0
        self._reg_tick = 0
        self._reg_tq: List[tuple] = []
        self._reg_ring: "deque[float]" = deque(maxlen=32)   # [R3.5-FIX2] last per-key registered DMA samples (ms)
        self._cost_reg = float(self.f.cost_reg_ms)
        self._cost_cold_g = max(float(self.f.cost_cold_ms) - self._cost_reg, 0.0)
        # [S2] registered tier: whole layers, in layer order, within the RAM budget and above the floor
        self._init_registration(files, register_gib, register_floor_gib)
        # [R3.5-P (c)] ReadFile gather (pool v4's code with the row map), one thread pool per caller thread
        self._rf = None
        self._rf_files: Dict[int, Tuple[str, int]] = {}
        if self.f.gather == "readfile" and self.files:
            from concurrent.futures import ThreadPoolExecutor
            self._tls = threading.local()
            self._np_views: Dict[int, object] = {}
            self.f.gather_seg = max(1 << 16, int(self.f.gather_seg_mib * 2**20))
            for l, (path, off) in sorted(self.files.items()):
                src = self.sources.get(l)
                if src is not None and not src.is_cuda and self._rf_check(l, path, int(off)):
                    self._rf_files[l] = (path, int(off))
            if self._rf_files:
                self._rf = {"main": ThreadPoolExecutor(max_workers=self.f.gather_threads, thread_name_prefix="p6rd-m"),
                            "worker": ThreadPoolExecutor(max_workers=self.f.gather_threads, thread_name_prefix="p6rd-w")}
        self.worker = threading.Thread(target=self._worker_loop, daemon=True, name="pool6-worker")
        self.worker.start()
        atexit.register(self.close)

    # ------------------------------------------------------------------ registration (v5)
    def _init_registration(self, files, register_gib, register_floor_gib):
        self.reg = None
        self.reg_layers = set()
        self.reg_rows: Dict[int, int] = {}                  # [R3.3-rows] layer -> rows 0..K-1 registered
        self.stats["bytes_dma_registered"] = 0
        if files and register_gib > 0:
            import time as _t
            t0 = _t.perf_counter()
            self.reg = _RegisteredViews()
            scratch = self._dv.alloc_scratch(64 << 20)
            budget = float(register_gib)
            mode = os.environ.get("R33_REGISTER_MODE", "layers")
            if mode == "rows":
                # [R3.3-rows] hottest rows of EVERY layer (popularity-ordered archives); per-layer counts from
                # R33_REGISTER_PROFILE (JSON {layer: weight}) or uniform
                n_rows = {l: int(self.sources[l].shape[0]) for l in files}
                left = int(budget * 2**30 // self.row_bytes)
                prof = os.environ.get("R33_REGISTER_PROFILE", "")
                wts0 = {int(k): float(v) for k, v in json.load(open(prof)).items()} if prof else {}
                wts = {l: max(wts0.get(l, 1.0), 1e-6) for l in files}
                K = {l: 0 for l in files}
                open_l = set(files)
                while left > 0 and open_l:
                    ws = sum(wts[l] for l in open_l)
                    give = {l: int(left * wts[l] / ws) for l in open_l}
                    if not any(give.values()):
                        break
                    for l in list(open_l):
                        g = min(give[l], n_rows[l] - K[l])
                        K[l] += g
                        left -= g
                        if K[l] >= n_rows[l]:
                            open_l.discard(l)
                for l in sorted(files):
                    if K[l] <= 0:
                        continue
                    path, off = files[l]
                    gib = K[l] * self.row_bytes / 2**30
                    if self.reg.avail_gib() - gib < register_floor_gib:
                        break
                    if self.reg.add(l, path, off, scratch, nbytes=K[l] * self.row_bytes):
                        self.reg_rows[l] = K[l]
                        if K[l] >= n_rows[l]:
                            self.reg_layers.add(l)
                        budget -= gib
            else:
                for l in sorted(files):
                    path, off = files[l]
                    gib = os.path.getsize(path) / 2**30
                    if gib > budget or self.reg.avail_gib() - gib < register_floor_gib:
                        break
                    if self.reg.add(l, path, off, scratch):
                        self.reg_layers.add(l)
                        self.reg_rows[l] = int(self.sources[l].shape[0])
                        budget -= gib
            del scratch
            self._dv.synchronize()
            self.stats["registered_layers"] = len(self.reg_rows)
            self.stats["registered_rows"] = int(sum(self.reg_rows.values()))
            self.stats["registered_gib"] = round(float(register_gib) - budget, 2)
            self.stats["register_s"] = round(_t.perf_counter() - t0, 1)

    # ------------------------------------------------------------------ [R3.5-P] small helpers
    def _row(self, l: int, e: int) -> int:
        ro = self.row_of.get(l)
        return ro[e] if ro is not None else e

    def _is_cold(self, key) -> bool:
        """Not in the registered tier (its miss goes through the CPU gather)."""
        l, e = key
        return self._row(l, e) >= self.reg_rows.get(l, 0)

    def _cost(self, key) -> float:
        return self._cost_reg + (self._cost_cold_g if self._is_cold(key) else 0.0)

    def costs(self) -> dict:
        """[R3.5-P (d)] current per-tier running-mean fetch costs (ms per row)."""
        return {"registered_ms": round(self._cost_reg, 4), "cold_ms": round(self._cost_reg + self._cost_cold_g, 4)}

    def info(self) -> dict:
        d = {"version": 6, "flags": self.f.as_dict(), "n_slots": self.n_slots, "row_bytes": self.row_bytes,
             "readfile_layers": len(self._rf_files), "worker_error": self._worker_error}
        if self._r39t_lazy:                              # [R3.9T] 0 = the cold path was never taken
            d["r39t_lazy_staging_allocs"] = self._r39t_staging_allocs
        if self._gd is not None:
            d["costs"] = self.costs()
        return d

    def flags_str(self) -> str:
        f = self.f
        parts = [f"a_signal={int(f.signal)}", f"b_demand={int(f.demand)}", f"b2_stream={int(f.demand_stream)}",
                 f"c_gather={f.gather}", f"d_evict={f.evict}", f"e_topk={f.cold_topk}", f"e_margin={f.cold_margin:g}"]
        if f.reg_topk or f.reg_margin:                     # [R3.7 (r)] shown only when set: the default line is unchanged
            parts += [f"r_topk={f.reg_topk}", f"r_margin={f.reg_margin:g}"]
        return " ".join(parts)

    @property
    def wants_scores(self) -> bool:
        """[R3.5-P (e)] True when prefetch() uses scores (callers compute / copy them only then).
        [R3.7 (r)] also for the registered-tier gate."""
        return bool(self.f.cold_topk or self.f.cold_margin or self.f.reg_topk or self.f.reg_margin)

    # ------------------------------------------------------------------ slots (call with self.lock held)
    def _evictable(self, key, locked) -> bool:
        if key in self.pinned or key in locked or key in self.pending or key in self._held:
            return False
        ev = self.ready_cuda.get(key)
        return ev is None or self._dv.query(ev)     # a worker copy may still be writing this slot

    def _evict_one(self, locked: set):
        if self._gd is None:                         # v5 LRU
            for key in self.lru:
                if not self._evictable(key, locked):
                    continue
                slot = self.lru.pop(key)
                del self.slot_of[key]
                self._prefetched.discard(key)
                self.ready_cuda.pop(key, None)
                self.stats["evictions"] += 1
                return slot
            return None
        key = self._gd.pop_victim(lambda k: self._evictable(k, locked))   # [R3.5-P (d)]
        if key is None:
            return None
        slot = self.slot_of.pop(key)
        self._prefetched.discard(key)
        self.ready_cuda.pop(key, None)
        self.stats["evictions"] += 1
        return slot

    def _alloc(self, key, locked: set, kind: str, cur=None, may_fail: bool = False):
        """-> (slot, reader fence of the slot or None). kind: miss | cold | prefetch | prefetch_cold ([R3.5-T] a
        prefetch placed like a cold insert; like every prefetch it never hands over reservations)."""
        cost = self._cost(key) if self._gd is not None else 0.0     # may raise (bad id): before any state change
        if self.free:
            slot = self.free.pop()
        else:
            slot = self._evict_one(locked)
            if slot is None and kind not in ("prefetch", "prefetch_cold") and self._held:
                # same-layer accumulation overflow (see LIFETIME RULE): hand over the earlier same-layer reservations
                self._release_holds_locked(None, cur, force=True)
                self.stats["holds_overflow"] += 1
                slot = self._evict_one(locked)
            if slot is None:
                if may_fail:
                    return None, None
                raise RuntimeError("ExpertPool: no evictable slot")
        self.slot_of[key] = slot
        if self._gd is None:
            self.lru[key] = slot
            if kind in ("cold", "prefetch_cold"):
                self.lru.move_to_end(key, last=False)            # [R3.5-P item 10b] cold end ([R3.5-T] + prefetches)
        elif key not in self.pinned:
            self._gd.insert(key, cost, "cold" if kind == "prefetch_cold" else kind)
        return slot, self._slot_fence.pop(slot, None)

    def _touch(self, key, cold: bool) -> None:
        if self._gd is None:
            if not cold:
                self.lru.move_to_end(key)
        elif key not in self.pinned:
            self._gd.touch(key, self._cost(key), cold)

    def _policy_remove(self, key) -> None:
        if self._gd is None:
            self.lru.pop(key, None)
        else:
            self._gd.remove(key)

    def _release_holds_locked(self, l_new, cur, force: bool = False) -> None:
        """LIFETIME RULE hand-over: a reader fence on `cur` for every reserved slot, then the reservations end."""
        if self._held and (force or l_new != self._held_layer):
            st = cur if cur is not None else self._dv.current_stream()
            ev = self._dv.new_event()
            self._dv.record(ev, st)
            fence = (ev, st)                         # a fence = (event, the stream it was recorded on)
            for key in self._held:
                s = self.slot_of.get(key)
                if s is not None:
                    self._slot_fence[s] = fence
            self._held.clear()
        if not force:
            self._held_layer = l_new

    def _signal_locked(self, k) -> None:
        self._kstate.pop(k, None)
        self._wfence.pop(k, None)
        e = self.pending.pop(k, None)
        if e is not None:
            e.set()

    def _cancel_locked(self, k) -> None:
        """(b) drop a QUEUED prefetch: no copy was issued; its slot goes back with its previous readers' fence."""
        s = self.slot_of.pop(k)
        self._policy_remove(k)
        self._prefetched.discard(k)
        f_ = self._kfence.pop(k, None)
        if f_ is not None:
            self._slot_fence[s] = f_
        self.free.append(s)
        self._signal_locked(k)
        self.stats["prefetch_dropped"] += 1

    def _wait_fences(self, stream, fences) -> int:
        """Order `stream` after every reader fence (event, recorded-on stream) that is not done yet. A fence recorded on
        `stream` itself needs nothing: the writer is queued after it on the same stream. Returns the waits issued."""
        n = 0
        seen = set()
        for f_ in fences:
            if f_ is None:
                continue
            ev, fs = f_
            if id(ev) in seen:
                continue
            seen.add(id(ev))
            if fs is stream or fs == stream:
                continue
            if not self._dv.query(ev):
                self._dv.wait(stream, ev)
                n += 1
        return n

    # ------------------------------------------------------------------ transfers
    def _r39t_staging(self, which: str, j: int = 0) -> None:
        """[R3.9T] R39T_LAZY_STAGING=1: allocate a cold-path block at its first use (never, when every row is
        registered). which: a (ensure), b (v5 worker chunk), b2 (worker block j)."""
        if which == "a":
            if self.pin_a is None:
                self.pin_a = self._dv.alloc_pinned(self.R, self.row_bytes)
                self.stg_a = self._dv.alloc_staging(self.R, self.row_bytes)
                self._r39t_staging_allocs += 1
        elif which == "b":
            if self.pin_b is None:
                self.pin_b = self._dv.alloc_pinned(self.R, self.row_bytes)
                self.stg_b = self._dv.alloc_staging(self.R, self.row_bytes)
                self._r39t_staging_allocs += 1
        elif self.pin_b2[j] is None:
            self.pin_b2[j] = self._dv.alloc_pinned(self.Rw, self.row_bytes)
            self.stg_b2[j] = self._dv.alloc_staging(self.Rw, self.row_bytes)
            self._r39t_staging_allocs += 1

    def _gather(self, l: int, experts: List[int], pin: torch.Tensor, who: str) -> float:
        """Cold rows of layer l into the pinned block (CPU). Returns the seconds spent (the caller books them)."""
        m = len(experts)
        if m > int(pin.shape[0]):           # [R3.5-FIX COLD-BATCH-UNBOUNDED] never more rows than the pinned block holds
            raise IndexError(f"ExpertPool: {m} rows for a pinned block of {int(pin.shape[0])} rows (layer {l})")
        src = self.sources[l]
        t0 = time.perf_counter()
        ro = self.row_of.get(l)
        rows_idx = [ro[e] for e in experts] if ro is not None else experts       # [R3.3] expert -> slab row
        if self._rf is not None and l in self._rf_files:
            self._gather_readfile(l, rows_idx, pin, who)                         # [R3.5-P (c)]
        else:
            torch.index_select(src, 0, torch.tensor(rows_idx, dtype=torch.long), out=pin[:m])
        dt = time.perf_counter() - t0
        if self._gd is not None and m:                                           # [R3.5-P (d)] cold cost (gather part)
            self._cost_cold_g += self.f.cost_alpha * (1000.0 * dt / m - self._cost_cold_g)
        return dt

    def _read_seg(self, path: str, pos: int, dst) -> None:
        """[R3.5-P (c)] pool v4's segment read (thread-local unbuffered handles)."""
        fh = getattr(self._tls, "fh", None)
        if fh is None:
            fh = self._tls.fh = {}
        f = fh.get(path)
        if f is None:
            f = fh[path] = open(path, "rb", buffering=0)
        f.seek(pos)
        mv = memoryview(dst)
        got = 0
        while got < len(mv):
            n = f.readinto(mv[got:])
            if not n:
                raise IOError(f"short read at {pos + got} in {path}")
            got += n

    def _gather_readfile(self, l: int, rows: List[int], pin: torch.Tensor, who: str) -> None:
        """[R3.5-P (c)] rows -> pin[:len(rows)] by threaded file reads at slab offset + row * row_bytes.
        [R3.5-FIX RF-STALE-FUTURES] exception safe: the row count and every row are checked before the first read is
        submitted, and no submitted read outlives the call - on any error (a failed read, a failed submit) the reads not
        started yet are cancelled and the running ones waited for, then the first error in submission order is raised.
        So a leftover read of a failed gather can never land in a pinned block a later gather (or its H2D) uses."""
        path, off = self._rf_files[l]
        key = pin.data_ptr()
        pv = self._np_views.get(key)
        if pv is None:
            pv = self._np_views[key] = pin.numpy()
        rb, seg = self.row_bytes, self.f.gather_seg
        n_rows = int(self.sources[l].shape[0])
        rows = [int(r) for r in rows]
        if len(rows) > int(pv.shape[0]):
            raise IndexError(f"ExpertPool: {len(rows)} rows for a pinned block of {int(pv.shape[0])} rows (layer {l})")
        for r in rows:
            if not 0 <= r < n_rows:
                raise IndexError(f"ExpertPool: layer {l} row {r} outside the slab [0, {n_rows})")
        tp = self._rf[who]
        futs = []
        try:
            for j, r in enumerate(rows):
                base = off + r * rb
                for s0 in range(0, rb, seg):
                    futs.append(tp.submit(self._read_seg, path, base + s0, pv[j, s0:min(rb, s0 + seg)]))
            _cf.wait(futs, return_when=_cf.FIRST_EXCEPTION)
        finally:
            for fu in futs:                      # no submitted read may outlive this call: drop the ones not started
                fu.cancel()                      # (a no-op for running / finished reads), wait for the running ones
            _cf.wait(futs)
        for fu in futs:
            if not fu.cancelled():
                fu.result()                      # re-raises the first failed read (submission order)

    def _rf_check(self, l: int, path: str, off: int) -> bool:
        """[R3.5-P (c)] the file bytes at off + r * row_bytes are the source rows (head and tail of the first and the
        last row, and the file size) - else this layer keeps the mmap gather."""
        src = self.sources[l]
        n, rb = int(src.shape[0]), self.row_bytes
        try:
            if n == 0 or os.path.getsize(path) < off + n * rb:
                return False
            k = min(4096, rb)
            with open(path, "rb", buffering=0) as fh:
                for r in sorted({0, n - 1}):
                    for a in (0, rb - k):
                        fh.seek(off + r * rb + a)
                        if fh.read(k) != src[r, a:a + k].numpy().tobytes():
                            return False
            return True
        except Exception:
            return False

    def _sync_fetch_fast(self, l: int, experts: List[int], slots: List[int], fences, cur) -> bool:
        """[R3.9T] R39T_FAST_DEMAND=1: _sync_fetch for calls whose rows are ALL registered (False = not such a call,
        nothing was issued, the caller runs _sync_fetch). Same order of CUDA calls on the same streams: reader-fence
        waits, one cudaMemcpyAsync per row (registered view -> pool slot), the demand stream's event and the compute
        stream's wait for it; same counters."""
        kreg = self.reg_rows.get(l, 0)
        if not kreg or self.reg is None or self.ev_a is not None:
            return False
        ro = self.row_of.get(l)
        rows = [ro[e] for e in experts] if ro is not None else experts
        for r_ in rows:
            if not 0 <= r_ < kreg:
                return False
        D = self.demand_stream
        st = D if D is not None else cur
        rb = self.row_bytes
        nfw = self._wait_fences(st, fences)
        if self._dbg.poison:                             # debug only (torture test): as _sync_fetch, on the copy's stream
            with self._dv.stream_ctx(st):
                self._dv.poison(self.pool, slots)
        v, _size, _hm, _hf, off = self.reg.views[l]
        base = v + off
        dst0 = self._pool_ptr
        sh = st.cuda_stream
        memcpy = self.reg.cud.cudaMemcpyAsync
        if self._r39t:                                   # [R3.9T] R39T_PROF bookkeeping
            self.stats["t_r39t_pre_issue"] += time.perf_counter() - self._r39t_te
        for r_, s_ in zip(rows, slots):
            rc = memcpy(dst0 + s_ * rb, base + r_ * rb, rb, 1, sh)
            if rc != 0:
                raise RuntimeError(f"cudaMemcpyAsync from registered view failed rc={rc} (layer {l}, row {r_})")
        if D is not None:
            evd = self._dv.new_event()
            self._dv.record(evd, D)
            if self._r39t_pfad or self._r39t_paced:
                self._r39t_last_demand = evd
            self._dv.wait(cur, evd)
        nb = len(rows) * rb
        with self.lock:                                  # counters the worker writes too
            self.stats["bytes_dma_registered"] += nb
            self.stats["bytes_h2d"] += nb
            self.stats["fence_waits"] += nfw
        return True

    def _sync_fetch(self, l: int, experts: List[int], slots: List[int], fences, cur) -> None:
        """Current-layer misses (and (b) claimed prefetches), ordered before the expert kernels: on the compute
        stream, or [R3.5-P (b2)] on the demand stream followed by a compute-stream wait."""
        if self._r39t_fast and self._sync_fetch_fast(l, experts, slots, fences, cur):   # [R3.9T]
            return
        D = self.demand_stream
        st = D if D is not None else cur
        rb = self.row_bytes
        reg_b = h2d_b = 0
        gdt = 0.0
        nfw = 0
        with (self._dv.stream_ctx(D) if D is not None else nullcontext()):
            nfw = self._wait_fences(st, fences)      # [R3.5-P] previous readers of the reused slots
            for c0 in range(0, len(experts), self.R):
                ex, sl = experts[c0:c0 + self.R], slots[c0:c0 + self.R]
                m = len(ex)
                if self.ev_a is not None:
                    self._dv.sync_event(self.ev_a)  # pinned block reuse
                src = self.sources[l]
                if self._dbg.poison:
                    self._dv.poison(self.pool, sl)
                kreg = self.reg_rows.get(l, 0)
                if self._r39t and c0 == 0:                       # [R3.9T] R39T_PROF bookkeeping
                    self.stats["t_r39t_pre_issue"] += time.perf_counter() - self._r39t_te
                if kreg:                                         # [S2] direct DMA from the locked view
                    ro = self.row_of.get(l)
                    rest_e, rest_s = [], []
                    for e, s_ in zip(ex, sl):
                        r_ = ro[e] if ro is not None else e
                        if r_ < kreg:
                            self._dv.dma_registered(self.reg, l, r_, rb, self.pool, s_, st)
                        else:
                            rest_e.append(e)
                            rest_s.append(s_)
                    nd = m - len(rest_e)
                    reg_b += nd * rb
                    h2d_b += nd * rb
                    if not rest_e:
                        continue
                    ex, sl, m = rest_e, rest_s, len(rest_e)
                slot_t = self._dv.slots_upload(sl)
                if src.is_cuda:
                    ro = self.row_of.get(l)
                    exr = [ro[e] for e in ex] if ro is not None else ex
                    self._dv.d2d_scatter(self.pool, slot_t, src, exr)
                else:
                    if self._r39t_lazy:
                        self._r39t_staging("a")          # [R3.9T] first cold row of ensure()
                    gdt += self._gather(l, ex, self.pin_a, "main")
                    self._dv.h2d(self.stg_a, self.pin_a, m)
                    self._dv.scatter(self.pool, slot_t, self.stg_a, m)
                self.ev_a = self._dv.new_event()
                self._dv.record(self.ev_a, st)
                h2d_b += m * rb
            if D is not None:
                evd = self._dv.new_event()
                self._dv.record(evd, D)
                if self._r39t_pfad or self._r39t_paced:  # [R3.9T] the worker orders its next prefetch copies after it
                    self._r39t_last_demand = evd
        if D is not None:
            self._dv.wait(cur, evd)
        self.stats["t_gather_main"] += gdt
        with self.lock:                                  # counters the worker writes too
            self.stats["bytes_dma_registered"] += reg_b
            self.stats["bytes_h2d"] += h2d_b
            self.stats["t_gather"] += gdt
            self.stats["fence_waits"] += nfw

    # ------------------------------------------------------------------ prefetch worker
    def _worker_loop(self):
        self._dv.worker_init()
        while True:
            item = self.q.get()
            if item is None:
                return
            t_put = None
            if self._r39t:                               # [R3.9T] FIFO queue, one worker: stamps arrive in item order
                t_put = self._r39t_q.popleft() if self._r39t_q else None
                t_ws = time.perf_counter()
            try:
                self._process_item(item)
            except BaseException as exc:                 # [R3.5-P] never die: fail the undelivered keys, go on
                self._fail_item(item, exc)
            finally:
                if t_put is not None:                    # [R3.9T]
                    with self.lock:
                        self.stats["r39t_wk_items"] += 1
                        self.stats["t_r39t_wk_start"] += t_ws - t_put
                        self.stats["t_r39t_wk_done"] += time.perf_counter() - t_put                                     # [R3.5-T item 10d] quiesce(): this item is finished (its
                with self._wq_cv:                        # last CUDA call returned; next: a blocking q.get())
                    self._wq_done += 1
                    self._wq_cv.notify_all()

    def _dbg_point(self, where: str) -> None:
        if self._dbg.delay_ms:
            time.sleep(self._dbg_rng.random() * self._dbg.delay_ms / 1000.0)
        if self._dbg.fail_every:
            self._dbg_n += 1
            if self._dbg_n % self._dbg.fail_every == 0:
                raise RuntimeError(f"[R3.5-P debug] injected worker fault at {where}")

    def _process_item(self, item):
        l, keys, gen = item
        self._dbg_point("item")
        if self._gd is not None and self._reg_tq:
            self._poll_reg_timing()
        if not self.f.signal:
            # v5 chunks (claimable sub-chunks of cold_batch keys with (b)); [R3.5-FIX COLD-BATCH-UNBOUNDED] at most R
            # (staging_rows) keys: _chunk_v5 gathers into pin_b / stg_b, which hold R rows (as (a) caps B by Rw below)
            C = min(self.f.cold_batch, self.R) if self.f.demand else self.R
            for c0 in range(0, len(keys), C):
                take = self._take(l, keys[c0:c0 + C], gen)
                if take:
                    self._chunk_v5(l, take)
            return
        # [R3.5-P (a)] registered rows first, each signalled after its own DMA; cold rows in their own batches
        kreg = self.reg_rows.get(l, 0)
        reg, cold = [], []
        for k in keys:
            (reg if self._row(l, k[1]) < kreg else cold).append(k)
        if reg and self._r39t_paced:                    # [R3.9T] R39T_PF_PACED: one key at a time (see __init__)
            self._issue_registered_paced(l, reg, gen)
        elif reg:
            take = self._take(l, reg, gen)
            if take:
                self._issue_registered(l, take)
        B = min(self.f.cold_batch, self.Rw) if self.f.demand else self.Rw
        for c0 in range(0, len(cold), B):
            take = self._take(l, cold[c0:c0 + B], gen)
            if take:
                self._issue_cold(l, take)

    def _take(self, l: int, ks, gen: int):
        """Under the lock: QUEUED -> IN FLIGHT for the keys still queued (claimed / cancelled / failed ones are
        skipped); (b) drops the cold (or all) ones of a layer ensure() has passed since the prefetch was queued."""
        out = []
        with self.lock:
            passed = self.f.demand and self._ensure_gen.get(l, 0) > gen
            for k in ks:
                if self._kstate.get(k) != _QUEUED:
                    continue
                if passed and (self.f.drop == "all" or self._is_cold(k)):
                    self._cancel_locked(k)
                    continue
                self._kstate[k] = _FLIGHT
                f_ = self._kfence.pop(k, None)
                if f_ is not None:
                    self._wfence[k] = f_
                out.append((k, self.slot_of[k], f_))
        return out

    def _chunk_v5(self, l: int, take) -> None:
        """Pool v5's worker chunk: registered DMAs + one cold gather, ONE event for the chunk, then a host wait before
        the pinned / staging block is reused. [R3.5-P] + the reader-fence waits."""
        ks = [k for k, _, _ in take]
        ex = [k[1] for k in ks]
        sl = [s for _, s, _ in take]
        m = len(ex)
        src = self.sources[l]
        cs = self.copy_stream
        reg_b = 0
        gdt = 0.0
        self._dbg_point("chunk")
        with self._dv.stream_ctx(cs):
            nfw = self._wait_fences(cs, [f_ for _, _, f_ in take])
            if self._dbg.gpu_sleep:
                self._dv.gpu_sleep(self._dbg.gpu_sleep)
            if self._dbg.poison:
                self._dv.poison(self.pool, sl)
            kreg = self.reg_rows.get(l, 0)
            if kreg:                                     # [S2] direct DMA from the locked view
                ro = self.row_of.get(l)
                rest_e, rest_s = [], []
                for e, s_ in zip(ex, sl):
                    r_ = ro[e] if ro is not None else e
                    if r_ < kreg:
                        self._dv.dma_registered(self.reg, l, r_, self.row_bytes, self.pool, s_, cs)
                    else:
                        rest_e.append(e)
                        rest_s.append(s_)
                reg_b = (m - len(rest_e)) * self.row_bytes
                ex, sl = rest_e, rest_s
            mm = len(ex)
            slot_t = self._dv.slots_upload(sl)
            if mm == 0:
                pass
            elif src.is_cuda:
                ro_ = self.row_of.get(l)
                exr = [ro_[e] for e in ex] if ro_ is not None else ex
                self._dv.d2d_scatter(self.pool, slot_t, src, exr)
            else:
                if self._r39t_lazy:
                    self._r39t_staging("b")              # [R3.9T] first cold row of the v5 worker chunk
                gdt = self._gather(l, ex, self.pin_b, "worker")
                self._dv.h2d(self.stg_b, self.pin_b, mm)
                self._dv.scatter(self.pool, slot_t, self.stg_b, mm)
            ev = self._dv.new_event()
            self._dv.record(ev, cs)
        with self.lock:
            for k in ks:
                self.ready_cuda[k] = ev
                self._signal_locked(k)
            self.stats["bytes_dma_registered"] += reg_b
            self.stats["bytes_h2d"] += m * self.row_bytes
            self.stats["t_gather"] += gdt
            self.stats["t_gather_worker"] += gdt
            self.stats["fence_waits"] += nfw
        self._dv.sync_event(ev)  # before reusing pin_b / stg_b

    def _issue_registered_paced(self, l: int, keys, gen: int) -> None:
        """[R3.9T] R39T_PF_PACED=n: registered prefetch keys one by one, in the confidence order they were queued in.
        Before a key's copy is issued the worker waits (host) until ensure()'s latest demand copies have finished and
        fewer than n of its own copies are in flight; a key that left QUEUED meanwhile (claimed by ensure()) or whose
        layer ensure() has passed is not waited for: _take() skips it or (R35_POOL_DROP=all) drops it."""
        fl = self._r39t_flight
        n = self._r39t_paced
        nw_l = nw_d = 0
        for k in keys:
            with self.lock:
                live = (self._kstate.get(k) == _QUEUED
                        and not (self.f.demand and self._ensure_gen.get(l, 0) > gen))
            if live:
                dv = self._r39t_last_demand
                if dv is not None and not self._dv.query(dv):
                    self._dv.sync_event(dv)
                    nw_d += 1
                while len(fl) >= n:
                    ev0 = fl.popleft()
                    if not self._dv.query(ev0):
                        self._dv.sync_event(ev0)
                        nw_l += 1
            take = self._take(l, [k], gen)
            if take:
                self._issue_registered(l, take)
                ev1 = self.ready_cuda.get(k)             # recorded by _issue_registered right after the copy
                if ev1 is not None:
                    fl.append(ev1)
        if nw_l or nw_d:
            with self.lock:
                self.stats["r39t_paced_link_waits"] += nw_l
                self.stats["r39t_paced_demand_waits"] += nw_d

    def _issue_registered(self, l: int, take) -> None:
        """[R3.5-P (a)] one DMA per registered row; each key gets its own event and is signalled right away.
        [R3.5-FIX GDSF-REGCOST] (d) cost sample (every 8th batch): a timing-event pair right around each key's DMA, after
        that key's reader-fence wait and poison, so neither fence waits nor host gaps between the keys are measured.
        [R3.5-FIX2 GIL-GAP] and inside the pair only the start record, the bare copy and the end record: addresses, stream
        handle and both events come first (dma_registered_prep); with R35_POOL_COST_HOLD_GIL=1 (default) the start record
        and the copy keep the GIL (dma_registered_timed). Same bytes, same stream order as dma_registered."""
        cs = self.copy_stream
        rb = self.row_bytes
        timing = self._gd is not None and self._reg_tick % 8 == 0     # (d) sampled DMA-cost measurement
        self._reg_tick += 1
        pairs = [] if timing else None                                # [R3.5-FIX] per-key (start, end) events
        nfw = 0
        with self._dv.stream_ctx(cs):
            if self._dbg.gpu_sleep:
                self._dv.gpu_sleep(self._dbg.gpu_sleep)
            if self._r39t_pfad:                          # [R3.9T] demand copies first (see __init__)
                dev_ = self._r39t_last_demand
                if dev_ is not None:
                    if not self._dv.query(dev_):         # still copying: this batch starts after it
                        self._dv.wait(cs, dev_)
                        with self.lock:
                            self.stats["r39t_pf_waits"] += 1
            for i, (k, s, f_) in enumerate(take):
                if i:
                    self._dbg_point("registered")
                nfw += self._wait_fences(cs, (f_,))
                if self._dbg.poison:
                    self._dv.poison(self.pool, [s])
                if pairs is not None:                                # [R3.5-FIX2] host work first, then the pair
                    t0, t1, prep = self._dv.dma_registered_prep(self.reg, l, self._row(l, k[1]), rb, self.pool, s, cs,
                                                                self.f.cost_hold_gil)
                    self._dv.dma_registered_timed(self.reg, prep, t0, t1, cs)
                    pairs.append((t0, t1))
                else:
                    self._dv.dma_registered(self.reg, l, self._row(l, k[1]), rb, self.pool, s, cs)
                ev = self._dv.new_event()
                self._dv.record(ev, cs)
                with self.lock:
                    self.ready_cuda[k] = ev
                    self._signal_locked(k)
                    self.stats["bytes_dma_registered"] += rb
                    self.stats["bytes_h2d"] += rb
            if pairs:
                self._reg_tq.append(pairs)
        if nfw:
            with self.lock:
                self.stats["fence_waits"] += nfw

    def _issue_cold(self, l: int, take) -> None:
        """[R3.5-P (a)] one cold batch: gather into the free pinned block (the other one may still be copying), H2D,
        reader-fence waits, scatter into the slots, one event for the batch."""
        ks = [k for k, _, _ in take]
        ex = [k[1] for k in ks]
        sl = [s for _, s, _ in take]
        m = len(ex)
        src = self.sources[l]
        cs = self.copy_stream
        self._wb ^= 1
        j = self._wb
        if self._ev_b2[j] is not None:
            self._dv.sync_event(self._ev_b2[j])         # pin_b2[j]: its H2D (two batches back) has finished
        if self._r39t_lazy and not self.sources[l].is_cuda:
            self._r39t_staging("b2", j)                 # [R3.9T] first cold batch of this worker block
        pin, stg = self.pin_b2[j], self.stg_b2[j]
        gdt = 0.0
        self._dbg_point("cold")
        with self._dv.stream_ctx(cs):
            slot_t = self._dv.slots_upload(sl)
            if src.is_cuda:
                nfw = self._wait_fences(cs, [f_ for _, _, f_ in take])
                ro_ = self.row_of.get(l)
                self._dv.d2d_scatter(self.pool, slot_t, src, [ro_[e] for e in ex] if ro_ is not None else ex)
            else:
                gdt = self._gather(l, ex, pin, "worker")
                if self._dbg.gpu_sleep:
                    self._dv.gpu_sleep(self._dbg.gpu_sleep)
                self._dv.h2d(stg, pin, m)
                nfw = self._wait_fences(cs, [f_ for _, _, f_ in take])
                if self._dbg.poison:
                    self._dv.poison(self.pool, sl)
                self._dv.scatter(self.pool, slot_t, stg, m)
            ev = self._dv.new_event()
            self._dv.record(ev, cs)
        self._ev_b2[j] = ev
        with self.lock:
            for k in ks:
                self.ready_cuda[k] = ev
                self._signal_locked(k)
            self.stats["bytes_h2d"] += m * self.row_bytes
            self.stats["t_gather"] += gdt
            self.stats["t_gather_worker"] += gdt
            self.stats["fence_waits"] += nfw

    def _poll_reg_timing(self) -> None:
        """[R3.5-P (d)] fold completed DMA timing samples into the registered-tier cost (worker thread only).
        [R3.5-FIX GDSF-REGCOST] a sample = one batch's per-key (start, end) pairs; folded once its last end event has
        completed (one stream: every earlier event has then completed too).
        [R3.5-FIX2 GIL-GAP] one-sided robust fold instead of the mean: elapsed(start, end) >= the DMA time always (one
        stream), so host time in the pair (GIL waits, WDDM submission batching) only inflates a sample. Every per-key
        sample is clamped at 3x the current cost (an outlier cannot dominate; a truly higher cost still gets in, at up to
        +2 * alpha per batch, where dropping such samples would freeze the estimate) and kept in a ring of the last 32;
        once per completed batch the ring's R35_POOL_COST_Q quantile (default 0.2: the 7th smallest of 32, the minimum
        while it holds < 5 samples) is folded into the EMA (weight R35_POOL_COST_ALPHA)."""
        keep = []
        ring = self._reg_ring
        for pairs in self._reg_tq:
            if self._dv.query(pairs[-1][1]):
                cap = 3.0 * self._cost_reg if self._cost_reg > 0 else float("inf")
                n = 0
                for t0, t1 in pairs:
                    ms = self._dv.elapsed_ms(t0, t1)
                    if ms > 0:
                        ring.append(min(ms, cap))
                        n += 1
                if n:
                    q = sorted(ring)[min(int(self.f.cost_q * len(ring)), len(ring) - 1)]
                    self._cost_reg += self.f.cost_alpha * (q - self._cost_reg)
            else:
                keep.append(pairs)
        self._reg_tq = keep[-64:]

    def _fail_item(self, item, exc) -> None:
        """[R3.5-P] worker exception: drain the copy stream (every copy the item issued has landed), then free the
        slots of its undelivered keys and set their pending events (a waiting ensure() refetches them)."""
        l, keys, gen = item
        self._worker_error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        injected = "[R3.5-P debug]" in str(exc)
        if not injected:
            print(f"[pool v6] WARNING: prefetch worker error (layer {l}); the keys fall back to ensure():\n"
                  f"{self._worker_error}", file=sys.stderr, flush=True)
        with self.lock:
            self.stats["worker_errors"] += 1
            fl = [self._wfence.get(k) for k in keys if self._kstate.get(k) == _FLIGHT]
        drained = True
        try:
            with self._dv.stream_ctx(self.copy_stream):
                self._wait_fences(self.copy_stream, fl)
                fev = self._dv.new_event()
                self._dv.record(fev, self.copy_stream)
            self._dv.sync_event(fev)
        except BaseException:
            drained = False                              # CUDA itself failed: in-flight slots are quarantined
        with self.lock:
            for k in keys:
                st = self._kstate.get(k)
                if st is None:
                    continue                             # delivered, claimed or cancelled
                s = self.slot_of.pop(k, None)
                self._policy_remove(k)
                self._prefetched.discard(k)
                self.ready_cuda.pop(k, None)
                f_ = self._kfence.pop(k, None)
                self._wfence.pop(k, None)
                if s is not None:
                    if st == _QUEUED:
                        if f_ is not None:
                            self._slot_fence[s] = f_     # nothing issued: the previous readers' fence stays
                        self.free.append(s)
                    elif drained:
                        self.free.append(s)              # its copies (if any) and its fence have completed
                self._signal_locked(k)

    # ------------------------------------------------------------------ public API
    def _check_rows(self, l: int, keys) -> None:
        """[R3.4-S3] raise for experts that have no row in layer l's (pruned) file."""
        ms = self._missing.get(l)
        if ms:
            bad = sorted({e for _, e in keys if e in ms})
            if bad:
                raise RuntimeError(f"ExpertPool: layer {l} experts {bad} have no row in the archive (pruned, row_of = -1)"
                                   f" - the router mask for them is missing")

    def _undo_locked(self, fetch, fences_by_slot, ev=None) -> None:
        """[R3.5-P] ensure() failed: drop the keys it allocated / claimed (their slots go back to the free list behind
        `ev` when copies may have been queued, else with their previous readers' fence)."""
        for key, s in fetch:
            if self.slot_of.get(key) != s:
                continue
            del self.slot_of[key]
            self._policy_remove(key)
            self._held.pop(key, None)
            self._prefetched.discard(key)
            if ev is not None:
                self._slot_fence[s] = ev
            elif fences_by_slot.get(s) is not None:
                self._slot_fence[s] = fences_by_slot[s]
            self.free.append(s)

    def ensure(self, l: int, experts: Iterable[int], out=None, cold: bool = False) -> torch.Tensor:
        """Slots of layer l's experts (missing rows are fetched on the compute stream; in-flight prefetches are waited
        for). Returns a new int32 device tensor - or, [R3.4-2g], writes them into `out` (a host int32 array, e.g. the
        numpy view of a pinned mailbox row that a graph copies to the GPU) and returns `out`.
        [R3.5-P] the slots stay reserved until the next ensure() on another layer / release() (LIFETIME RULE)."""
        t_e = time.perf_counter()
        if self._r39t:
            self._r39t_te = t_e
        if not cold and self._cold_stash is not None and self._cold_stash == int(l):
            cold = True                                  # ensure_cold() through a wrapper (see there)
        keys = [(l, int(e)) for e in experts]
        if self._r39t_trace is not None:                 # [R3.9T] call log (R39T_TRACE)
            self._r39t_trace.append(["e", int(l), [k[1] for k in keys], int(bool(cold))])
        if self._missing:
            self._check_rows(l, keys)
        locked = set(keys)
        cur = self._dv.current_stream()
        fetch: List[Tuple[Tuple[int, int], int]] = []    # misses + (b) claimed prefetches: copied here
        fences: Dict[int, object] = {}                   # slot -> reader fence of its previous readers
        waits: List[Tuple[Tuple[int, int], threading.Event]] = []
        # [R3.5-FIX ENSURE-LOST-READY-EVENTS] (key, ready event) of hits whose worker copy may still run: the event stays
        # in ready_cuda until the stream wait below is issued, so on every exception path _evictable keeps querying it
        cuda_waits: List[Tuple[Tuple[int, int], object]] = []
        slots: List[int] = []
        seen: Dict[Tuple[int, int], int] = {}
        demand = self.f.demand
        defer = [] if (self._r39t_defer and not cold) else None    # [R3.9T] R39T_FAST_DEMAND=2 (None = as before)
        with self.lock:
            self._release_holds_locked(l, cur)
            try:
                for key in keys:
                    s = seen.get(key)
                    if s is not None:                    # repeated entry: same slot (v5 counts it as a hit)
                        self.stats["hits"] += 1
                        if defer is None:
                            self._touch(key, cold)
                        else:
                            defer.append(key)            # [R3.9T] refreshed after the copies are issued
                        slots.append(s)
                        continue
                    s = self.slot_of.get(key)
                    if s is not None:
                        self.stats["hits"] += 1
                        if key in self._prefetched:
                            self.stats["prefetch_useful"] += 1
                            self._prefetched.discard(key)
                        if defer is None:
                            self._touch(key, cold)
                        else:
                            defer.append(key)            # [R3.9T]
                        pe = self.pending.get(key)
                        if pe is not None:
                            if demand and self._kstate.get(key) == _QUEUED:   # [R3.5-P (b)] claim it
                                f_ = self._kfence.pop(key, None)
                                self._signal_locked(key)
                                if f_ is not None:
                                    fences[s] = f_
                                fetch.append((key, s))
                                self.stats["prefetch_claimed"] += 1
                            else:
                                waits.append((key, pe))
                        else:
                            rc = self.ready_cuda.get(key)        # [R3.5-FIX] removed only after its wait (below)
                            if rc is not None:
                                cuda_waits.append((key, rc))
                    else:
                        self.stats["misses"] += 1
                        s, f_ = self._alloc(key, locked, "cold" if cold else "miss", cur)
                        if f_ is not None:
                            fences[s] = f_
                        fetch.append((key, s))
                    seen[key] = s
                    slots.append(s)
            except BaseException:
                self._undo_locked(fetch, fences)
                raise
            for key in seen:
                self._held[key] = None
            self._ensure_gen[l] = self._ensure_gen.get(l, 0) + 1
        if fetch:
            if self._r39t:                               # [R3.9T] host seconds of ensure()'s own copies
                t_f = time.perf_counter()
                self._fetch_or_undo(l, fetch, fences, cur)
                self.stats["r39t_fetch_calls"] += 1
                self.stats["t_r39t_fetch"] += time.perf_counter() - t_f
            else:
                self._fetch_or_undo(l, fetch, fences, cur)
        if defer:                                        # [R3.9T] the hits' priority refresh, after the copies
            with self.lock:
                for key in defer:
                    if key in self.slot_of:              # (a failed prefetch may have left; nothing to refresh)
                        self._touch(key, cold)
        if waits:
            if self._r39t:
                self.stats["r39t_host_waits"] += len(waits)
            t_w = time.perf_counter()
            for key, w in waits:
                if not w.wait(self.f.watchdog_s):
                    raise RuntimeError(f"ExpertPool v6: prefetch of {key} not delivered within {self.f.watchdog_s} s "
                                       f"(worker error: {self._worker_error})")
            self.stats["t_wait_prefetch"] += time.perf_counter() - t_w
            refetch: List[Tuple[Tuple[int, int], int]] = []
            ref_f: Dict[int, object] = {}
            with self.lock:
                try:
                    for key, _ in waits:
                        ev = self.ready_cuda.get(key)            # [R3.5-FIX] removed only after its wait (below)
                        if ev is not None:
                            cuda_waits.append((key, ev))
                        elif self.slot_of.get(key) != seen[key]:     # [R3.5-P] the worker failed it: fetch it here
                            s, f_ = self._alloc(key, locked, "miss", cur)
                            if f_ is not None:
                                ref_f[s] = f_
                            refetch.append((key, s))
                            self.stats["worker_refetch"] += 1
                except BaseException:                    # [R3.5-FIX] no refetch slot stays mapped without its copy,
                    self._undo_locked(refetch, ref_f)    # and no reservation of a failed key stays without a slot
                    for k in seen:
                        if k not in self.slot_of:
                            self._held.pop(k, None)
                    raise
            if refetch:
                self._fetch_or_undo(l, refetch, ref_f, cur)
                new = dict(refetch)
                slots = [new.get(k, s) for k, s in zip(keys, slots)]
        for _, ev in cuda_waits:
            if not self._dv.query(ev):                   # [R3.4-2g] a completed copy needs no stream wait
                self._dv.wait(cur, ev)
                if self._r39t:
                    self.stats["r39t_cuda_waits"] += 1
        if cuda_waits:                                   # [R3.5-FIX] the copies are ordered before this stream's
            with self.lock:                              # readers now: the events have done their job
                for key, ev in cuda_waits:
                    if self.ready_cuda.get(key) is ev:
                        del self.ready_cuda[key]
        if out is not None:                              # [R3.4-2g] no allocation, no pageable upload
            out[:len(slots)] = slots
        else:
            out = self._dv.slots_tensor(slots)
        self.stats["t_ensure"] += time.perf_counter() - t_e
        return out

    def _fetch_or_undo(self, l, fetch, fences, cur) -> None:
        try:
            self._sync_fetch(l, [k[1] for k, _ in fetch], [s for _, s in fetch], list(fences.values()), cur)
        except BaseException:
            ev = None
            try:                                         # behind everything this call may have queued
                st = self.demand_stream if self.demand_stream is not None else cur
                self._wait_fences(st, list(fences.values()))
                e_ = self._dv.new_event()
                self._dv.record(e_, st)
                ev = (e_, st)
            except BaseException:
                ev = None
            with self.lock:
                if ev is None:                           # CUDA failed: quarantine (do not reuse) these slots
                    for key, s in fetch:
                        if self.slot_of.get(key) == s:
                            del self.slot_of[key]
                            self._policy_remove(key)
                            self._held.pop(key, None)
                            self._prefetched.discard(key)
                else:
                    self._undo_locked(fetch, {}, ev)
            raise

    def prefetch(self, l: int, experts: Iterable[int], protect: Iterable[Tuple[int, int]] = (),
                 scores=None, score_ref=None, cold: bool = False) -> int:
        """Queue layer l's predicted experts for the worker. [R3.5-P (e)] scores (aligned with experts, larger = more
        confident) and score_ref (scalar or aligned list, e.g. the 9th-best router score) gate the cold ones.
        [R3.5-T item 10a+b] cold=True (or prefetch_cold()): the queued keys are placed at the cold end (LRU front /
        GDSF H = L), as ensure(..., cold=True) places prefill misses; resident keys are skipped as always."""
        t_p = time.perf_counter()
        sv = self._stash
        if scores is None and sv is not None and sv[0] == int(l):
            scores, score_ref = sv[1], sv[2]
        cs = getattr(self, "_cold_pf_stash", None)
        if not cold and cs is not None and cs == int(l):
            cold = True                                  # [R3.5-T] prefetch_cold() through a wrapper (see there)
        keys = [(l, int(e)) for e in experts]
        if self._r39t_trace is not None:                 # [R3.9T] call log (R39T_TRACE)
            self._r39t_trace.append(["p", int(l), [k[1] for k in keys], int(bool(cold)),
                                     None if scores is None else [round(float(x), 6) for x in scores],
                                     None if score_ref is None else (round(float(score_ref), 6) if isinstance(
                                         score_ref, (int, float)) else [round(float(x), 6) for x in score_ref])])
        if self._missing:
            self._check_rows(l, keys)
        locked = set(keys) | set(protect)
        allow = None
        allow_reg = None                                 # [R3.7 (r)] None = the registered-tier gate is off
        if scores is not None and self.wants_scores:
            keys, allow, allow_reg = self._rank_all(keys, scores, score_ref)
        ks = []
        kind = "prefetch_cold" if cold else "prefetch"   # [R3.5-T item 10a+b]
        with self.lock:
            gen = self._ensure_gen.get(l, 0)
            try:
                for key in keys:
                    if key in self.slot_of:
                        continue
                    if allow is not None and key not in allow and self._is_cold(key):
                        self.stats["prefetch_skipped_cold"] += 1
                        continue
                    if allow_reg is not None and key not in allow_reg and not self._is_cold(key):
                        self.stats["prefetch_skipped_reg"] += 1      # [R3.7 (r)] a miss, if routed, is served on demand
                        continue
                    s, f_ = self._alloc(key, locked, kind, None, may_fail=True)
                    if s is None:                        # [R3.5-P] nothing evictable: skip the rest (v5 raised)
                        self.stats["prefetch_noslot"] += 1
                        break
                    self.pending[key] = threading.Event()
                    self._kstate[key] = _QUEUED
                    if f_ is not None:
                        self._kfence[key] = f_
                    self._prefetched.add(key)
                    ks.append(key)
            except BaseException:                        # [R3.5-P] nothing was queued yet: give the slots back
                for key in ks:
                    s = self.slot_of.pop(key)
                    self._policy_remove(key)
                    self._prefetched.discard(key)
                    f_ = self._kfence.pop(key, None)
                    if f_ is not None:
                        self._slot_fence[s] = f_
                    self.free.append(s)
                    self._signal_locked(key)
                raise
        if ks:
            with self._wq_cv:                            # [R3.5-T item 10d] counted before the hand-over
                self._wq_put += 1
            if self._r39t:
                self._r39t_q.append(time.perf_counter())
            self.q.put((l, ks, gen))
            self.stats["prefetch_issued"] += len(ks)
            if cold:
                self.stats["prefetch_cold"] += len(ks)
        if self._r39t_preevict and not cold:             # [R3.9T] free slots for the next ensure() (see __init__)
            with self.lock:
                need = self._r39t_preevict - len(self.free)
                if need > 0:
                    skip = _LayerKeys(int(l))
                    while need > 0:
                        s_ = self._evict_one(skip)
                        if s_ is None:
                            break
                        self.free.append(s_)             # its reader fence stays in _slot_fence until the slot is reused
                        self.stats["r39t_preevicted"] += 1
                        need -= 1
        self.stats["t_prefetch"] += time.perf_counter() - t_p
        return len(ks)

    def _rank(self, keys, scores, score_ref):
        """[R3.5-P (e)] distinct keys in confidence order (margin over score_ref if given) + the set allowed cold."""
        order, allow, _ = self._rank_all(keys, scores, score_ref)
        return order, allow

    def _rank_all(self, keys, scores, score_ref):
        """_rank plus [R3.7 (r)] the set of keys that may be prefetched when they are REGISTERED: rank below
        reg_topk and, when score_ref is given, margin of at least reg_margin. None when that gate is off, so the
        default behaviour (every registered prediction is prefetched) costs nothing extra."""
        sc = [float(x) for x in scores]
        if len(sc) != len(keys):
            raise ValueError(f"prefetch: {len(sc)} scores for {len(keys)} experts")
        if score_ref is None:
            refs = None
        elif isinstance(score_ref, (int, float)):
            refs = [float(score_ref)] * len(sc)
        else:
            refs = [float(x) for x in score_ref]
            if len(refs) != len(sc):
                raise ValueError("prefetch: score_ref must be a scalar or aligned with experts")
        best: Dict[Tuple[int, int], float] = {}
        for i, k in enumerate(keys):
            v = sc[i] - refs[i] if refs is not None else sc[i]
            if k not in best or v > best[k]:
                best[k] = v
        order = sorted(best, key=lambda k: -best[k])
        allow = set()
        for rank, k in enumerate(order):
            if self.f.cold_topk and rank >= self.f.cold_topk:
                continue
            if self.f.cold_margin and refs is not None and best[k] < self.f.cold_margin:
                continue
            allow.add(k)
        allow_reg = None
        if self.f.reg_topk or self.f.reg_margin:         # [R3.7 (r)] the same two tests with their own limits
            allow_reg = set()
            for rank, k in enumerate(order):
                if self.f.reg_topk and rank >= self.f.reg_topk:
                    continue
                if self.f.reg_margin and refs is not None and best[k] < self.f.reg_margin:
                    continue
                allow_reg.add(k)
        return order, allow, allow_reg

    def ensure_cold(self, l: int, experts: Iterable[int], out=None):
        """[R3.5-P item 10b] ensure(..., cold=True) that survives wrappers forwarding only (l, experts, out)
        (build_run/r34_shrink_opts.install_mask_check): the flag is handed over to the wrapped ensure()."""
        self._cold_stash = int(l)
        try:
            return self.ensure(l, experts, out=out)
        finally:
            self._cold_stash = None

    def prefetch_scored(self, l: int, experts: Iterable[int], scores, score_ref=None,
                        protect: Iterable[Tuple[int, int]] = ()) -> int:
        """[R3.5-P (e)] prefetch() with scores that survive wrappers forwarding only (l, experts, protect)."""
        self._stash = (int(l), list(scores), score_ref)
        try:
            return self.prefetch(l, experts, protect)
        finally:
            self._stash = None

    def prefetch_cold(self, l: int, experts: Iterable[int], protect: Iterable[Tuple[int, int]] = ()) -> int:
        """[R3.5-T item 10a+b] prefetch(..., cold=True) that survives wrappers forwarding only (l, experts, protect)
        (build_run/r34_shrink_opts.install_mask_check): the flag is handed over to the wrapped prefetch(). Used by the
        prefill prefetch while R35_PREFILL_COLD is on, so prefill-only predictions do not push the decode working set out."""
        self._cold_pf_stash = int(l)
        try:
            return self.prefetch(l, experts, protect)
        finally:
            self._cold_pf_stash = None

    def r39t_set(self, fast_demand: int = 0, preevict: int = 0) -> None:
        """[R3.9T] test hook (the paired A/B of quality_eval_r39s, R39T_AB): switch R39T_FAST_DEMAND (0 | 1 | 2) and
        R39T_PREEVICT (n) of a live pool between two requests. Call it between requests only (no copy in flight is
        affected either way: both settings issue the same copies)."""
        self._r39t_fast = int(fast_demand) in (1, 2) and isinstance(self._dv, _CudaDev)
        self._r39t_defer = int(fast_demand) == 2
        if self._r39t_fast and not self._pool_ptr:
            self._pool_ptr = int(self.pool.data_ptr())
        self._r39t_preevict = max(0, int(preevict))
        self.stats.setdefault("r39t_preevicted", 0)

    def r39t_mark(self, what: str) -> None:
        """[R3.9T] R39T_TRACE: a caller's mark in the call log (decode start / end, prompt name)."""
        if self._r39t_trace is not None:
            self._r39t_trace.append(["m", str(what)])

    def r39t_trace_save(self) -> Optional[str]:
        """[R3.9T] R39T_TRACE: write the call log (+ what a simulation needs to know about this pool)."""
        if self._r39t_trace is None:
            return None
        path = os.environ["R39T_TRACE"].strip()
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"n_slots": self.n_slots, "row_bytes": self.row_bytes, "flags": self.f.as_dict(),
                       "rows": {str(l): int(v.shape[0]) for l, v in self.sources.items()},
                       "reg_rows": {str(l): int(v) for l, v in self.reg_rows.items()},
                       "costs": self.costs(), "stats": {k: v for k, v in self.stats.items()},
                       "calls": self._r39t_trace}, fh, separators=(",", ":"))
        return path

    def quiesce(self, timeout: Optional[float] = None) -> float:
        """[R3.5-T item 10d] wait until the prefetch worker has finished every item handed to it so far (queue empty,
        nothing in progress: it then blocks in queue.get() and makes no CUDA call until the next prefetch()), then until
        the copy stream (and the demand stream, b2) executed what was queued on it. GraphDecoder calls this before a
        CUDA-graph capture (env R35_CAPTURE_GUARD=1): during a capture another thread's synchronous CUDA call (event
        synchronize, pinned / device allocation) is an error in CUDA's global capture mode, and prefetch() - the only
        producer of worker items - runs on the capturing thread, so the worker stays idle for the whole capture. Moves no
        byte and changes no slot. Returns the seconds waited; RuntimeError after `timeout` s (default R35_POOL_WATCHDOG_S)
        or when the worker thread is gone with items left."""
        t0 = time.perf_counter()
        lim = float(self.f.watchdog_s if timeout is None else timeout)
        with self._wq_cv:
            while self._wq_done < self._wq_put:
                left = lim - (time.perf_counter() - t0)
                if left <= 0 or not self.worker.is_alive():
                    raise RuntimeError(f"ExpertPool v6: quiesce: {self._wq_put - self._wq_done} prefetch item(s) not "
                                       f"finished after {time.perf_counter() - t0:.1f} s (worker alive: "
                                       f"{self.worker.is_alive()}, worker error: {self._worker_error})")
                self._wq_cv.wait(min(0.05, left))
        for st in (self.copy_stream, self.demand_stream):
            if st is not None:
                ev = self._dv.new_event()
                self._dv.record(ev, st)
                self._dv.sync_event(ev)
        dt = time.perf_counter() - t0
        with self.lock:
            self.stats["quiesce_calls"] += 1
            self.stats["t_quiesce"] += dt
        return dt

    def flush(self, timeout: Optional[float] = None) -> int:
        """[R3.5-T] BENCHES ONLY (time-to-first-token runs that must start from the same pool state): quiesce(), end
        every reservation, wait for the GPU (every queued reader of a slot has run), then drop every resident key that is
        not pinned - its slot goes back to the free list (no byte is copied; the next requests fetch again). GDSF keeps
        its access frequencies (they survive eviction by design). Returns the number of keys dropped."""
        self.quiesce(timeout)
        self.release()
        self._dv.synchronize()
        n = 0
        with self.lock:
            for key in list(self.slot_of):
                if key in self.pinned or key in self.pending:
                    continue
                s = self.slot_of.pop(key)
                self._policy_remove(key)
                self._prefetched.discard(key)
                self.ready_cuda.pop(key, None)
                self._held.pop(key, None)
                self._slot_fence.pop(s, None)
                self.free.append(s)
                n += 1
            self.stats["flushes"] += 1
        return n

    def pin(self, keys: Iterable[Tuple[int, int]]) -> None:
        for key in keys:
            l, e = key
            self.ensure(l, [e])
            with self.lock:
                self.pinned.add(key)
                if self._gd is not None:
                    self._gd.remove(key)                 # [R3.5-P (d)] never a victim: out of the heap

    def release(self) -> None:
        """[R3.5-P] end every reservation now (reader fence on the current stream); see LIFETIME RULE."""
        cur = self._dv.current_stream()
        with self.lock:
            self._release_holds_locked(None, cur, force=True)

    def hit_rate(self) -> float:
        t = self.stats["hits"] + self.stats["misses"]
        return self.stats["hits"] / t if t else 0.0

    def close(self) -> None:
        """[R3.5-P] stop the worker, the ReadFile pools and (once the worker is gone) the registered views."""
        if self._closed:
            return
        self._closed = True
        try:
            atexit.unregister(self.close)                # drop the atexit reference (lets a closed pool be freed)
        except Exception:
            pass
        alive = False
        try:
            self.q.put(None)
            self.worker.join(timeout=5.0)
            alive = self.worker.is_alive()
        except Exception:
            alive = True
        if self._rf is not None:
            for tp in self._rf.values():
                tp.shutdown(wait=not alive)
        if self.reg is not None and not alive:           # never unmap a view a running DMA may still read
            try:
                self.reg.close()
            except Exception:
                pass
