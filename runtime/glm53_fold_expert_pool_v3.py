#!/usr/bin/env python3
"""
glm53_fold_expert_pool_v3.py  [Round 3.2, Claude]
=================================================
Two-level expert cache over ALL MoE layers:
  L1 = VRAM slot pool (LRU), L2 = page-locked host-RAM row cache (LRU).

Transfers are performed by ONE worker thread that owns L2 entirely (no cross-thread races on host rows):
  * high-priority queue: current-layer misses (the compute thread waits for them),
  * low-priority queue : next-layer prefetch predicted by the router pre-gating trick.
For each requested row: L2 hit -> async DMA pinned row -> VRAM slot (PCIe 4.0 x16 measured 28.7 GB/s, no CPU memcpy);
L2 miss -> one memcpy from the memory-mapped archive into a free L2 row, then the same DMA. All DMAs go on the copy
stream; the compute stream waits on the per-batch CUDA event before the expert kernels that read those slots.
L2 rows are reused only after the CUDA event of their last DMA has completed.
Public API identical to glm53_fold_expert_pool.ExpertPool: ensure(l, ids) -> int32 CUDA slots; prefetch(l, ids); pin.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from typing import Dict, Iterable, List, Optional, Tuple

import torch


class ExpertPool:
    def __init__(self, sources: Dict[int, torch.Tensor], row_bytes: int, n_slots: int, device: torch.device,
                 host_cache_gib: float = 0.0, chunk_gib: float = 4.0, **_ignored):
        self.device = device
        self.row_bytes = int(row_bytes)
        self.n_slots = int(n_slots)
        self.pool = torch.empty((self.n_slots, self.row_bytes), dtype=torch.uint8, device=device)
        self.sources = sources
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        # L1 bookkeeping (compute thread, under lock)
        self.slot_of: Dict[Tuple[int, int], int] = {}
        self.lru: "OrderedDict[Tuple[int, int], int]" = OrderedDict()
        self.free: List[int] = list(range(self.n_slots))
        self.pinned: set = set()
        self.pending: Dict[Tuple[int, int], threading.Event] = {}   # key -> set when its DMA is enqueued
        self.ready_cuda: Dict[Tuple[int, int], torch.cuda.Event] = {}
        self._prefetched: set = set()
        # L2 (worker thread only)
        n_h = max(0, int(host_cache_gib * 2 ** 30 // self.row_bytes))
        if n_h == 0:
            n_h = 32  # minimal staging ring when the host cache is disabled
        rows_per_chunk = max(1, int(chunk_gib * 2 ** 30 // self.row_bytes))
        self.h_chunks: List[torch.Tensor] = []
        left = n_h
        while left > 0:
            r = min(rows_per_chunk, left)
            self.h_chunks.append(torch.empty((r, self.row_bytes), dtype=torch.uint8, pin_memory=True))
            left -= r
        self.rows_per_chunk = rows_per_chunk
        self.H = n_h
        self.h_slot_of: Dict[Tuple[int, int], int] = {}
        self.h_lru: "OrderedDict[Tuple[int, int], int]" = OrderedDict()
        self.h_free: List[int] = list(range(n_h))
        self.h_ev: Dict[int, torch.cuda.Event] = {}
        self.copy_stream = torch.cuda.Stream(device=device)
        self.hi: deque = deque()
        self.lo: deque = deque()
        self.stats = {"hits": 0, "misses": 0, "prefetch_issued": 0, "prefetch_useful": 0, "bytes_h2d": 0,
                      "l2_hits": 0, "l2_fills": 0, "t_ensure": 0.0, "t_prefetch": 0.0, "t_gather": 0.0,
                      "t_wait_miss": 0.0, "t_wait_prefetch": 0.0}
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    # ------------------------------------------------------------------ L1 (call with lock held)
    def _evict_l1(self, locked: set) -> int:
        for key in self.lru:
            if key in self.pinned or key in locked or key in self.pending:
                continue
            ev = self.ready_cuda.get(key)
            if ev is not None and not ev.query():
                continue
            slot = self.lru.pop(key)
            del self.slot_of[key]
            self._prefetched.discard(key)
            self.ready_cuda.pop(key, None)
            return slot
        raise RuntimeError("ExpertPool: no evictable VRAM slot")

    def _alloc_l1(self, key, locked: set) -> int:
        slot = self.free.pop() if self.free else self._evict_l1(locked)
        self.slot_of[key] = slot
        self.lru[key] = slot
        return slot

    # ------------------------------------------------------------------ L2 (worker thread only)
    def _hrow(self, hs: int) -> torch.Tensor:
        return self.h_chunks[hs // self.rows_per_chunk][hs % self.rows_per_chunk]

    def _l2_row(self, key) -> int:
        hs = self.h_slot_of.get(key)
        if hs is not None:
            self.h_lru.move_to_end(key)
            self.stats["l2_hits"] += 1
            return hs
        if self.h_free:
            hs = self.h_free.pop()
        else:
            hs = -1
            for k2, h2 in self.h_lru.items():
                ev = self.h_ev.get(h2)
                if ev is None or ev.query():
                    hs = h2
                    break
            if hs < 0:  # every row is under an in-flight DMA: wait for the oldest
                k2, h2 = next(iter(self.h_lru.items()))
                self.h_ev[h2].synchronize()
                hs = h2
            del self.h_lru[k2]
            del self.h_slot_of[k2]
            self.h_ev.pop(hs, None)
        l, e = key
        t0 = time.perf_counter()
        self._hrow(hs).copy_(self.sources[l][e])
        self.stats["t_gather"] += time.perf_counter() - t0
        self.stats["l2_fills"] += 1
        self.h_slot_of[key] = hs
        self.h_lru[key] = hs
        return hs

    def _serve(self, keys, slots) -> torch.cuda.Event:
        hrows = []
        for key in keys:
            src = self.sources[key[0]]
            hrows.append(None if src.is_cuda else self._l2_row(key))
        with torch.cuda.stream(self.copy_stream):
            for key, s, hs in zip(keys, slots, hrows):
                if hs is None:
                    self.pool[s].copy_(self.sources[key[0]][key[1]], non_blocking=True)
                else:
                    self.pool[s].copy_(self._hrow(hs), non_blocking=True)
                self.stats["bytes_h2d"] += self.row_bytes
            ev = torch.cuda.Event()
            ev.record(self.copy_stream)
        for hs in hrows:
            if hs is not None:
                self.h_ev[hs] = ev
        return ev

    def _worker_loop(self):
        torch.cuda.set_device(self.device.index if self.device.index is not None else torch.cuda.current_device())
        while True:
            with self.cv:
                while not self.hi and not self.lo:
                    self.cv.wait()
                item = self.hi.popleft() if self.hi else self.lo.popleft()
            keys, slots, done = item
            # serve prefetch items one row at a time so a current-layer miss never waits long behind them
            if done is None and len(keys) > 1:
                with self.cv:
                    for k, s in zip(keys[1:][::-1], slots[1:][::-1]):
                        self.lo.appendleft(([k], [s], None))
                keys, slots = keys[:1], slots[:1]
            ev = self._serve(keys, slots)
            with self.lock:
                for k in keys:
                    self.ready_cuda[k] = ev
                    pe = self.pending.pop(k, None)
                    if pe is not None:
                        pe.set()
            if done is not None:
                done.set()

    # ------------------------------------------------------------------ public API
    def ensure(self, l: int, experts: Iterable[int], out=None) -> torch.Tensor:
        t_e = time.perf_counter()
        keys = [(l, int(e)) for e in experts]
        locked = set(keys)
        miss_k, miss_s, waits, slots = [], [], [], []
        with self.lock:
            for key in keys:
                if key in self.slot_of:
                    self.stats["hits"] += 1
                    if key in self._prefetched:
                        self.stats["prefetch_useful"] += 1
                        self._prefetched.discard(key)
                    self.lru.move_to_end(key)
                    if key in self.pending:
                        waits.append(self.pending[key])
                    slots.append(self.slot_of[key])
                else:
                    self.stats["misses"] += 1
                    s = self._alloc_l1(key, locked)
                    self.pending[key] = threading.Event()
                    miss_k.append(key)
                    miss_s.append(s)
                    slots.append(s)
        done = None
        if miss_k:
            done = threading.Event()
            with self.cv:
                self.hi.append((miss_k, miss_s, done))
                self.cv.notify()
        if done is not None:
            t_w = time.perf_counter()
            done.wait()
            self.stats["t_wait_miss"] += time.perf_counter() - t_w
        if waits:
            t_w = time.perf_counter()
            for w in waits:
                w.wait()
            self.stats["t_wait_prefetch"] += time.perf_counter() - t_w
        cur = torch.cuda.current_stream(self.device)
        with self.lock:
            evs = {id(ev): ev for ev in (self.ready_cuda.pop(k, None) for k in keys) if ev is not None}
        for ev in evs.values():
            cur.wait_event(ev)
        if out is not None:                       # [R3.4-fix1] host int32 row (GraphDecoder's pinned mailbox), as v5
            out[:len(slots)] = slots
        else:
            out = torch.tensor(slots, dtype=torch.int32).to(self.device, non_blocking=True)
        self.stats["t_ensure"] += time.perf_counter() - t_e
        return out

    def prefetch(self, l: int, experts: Iterable[int], protect: Iterable[Tuple[int, int]] = ()) -> int:
        t_p = time.perf_counter()
        keys = [(l, int(e)) for e in experts]
        locked = set(keys) | set(protect)
        ks, sl = [], []
        with self.lock:
            for key in keys:
                if key in self.slot_of:
                    continue
                s = self._alloc_l1(key, locked)
                self.pending[key] = threading.Event()
                self._prefetched.add(key)
                ks.append(key)
                sl.append(s)
        if ks:
            with self.cv:
                self.lo.append((ks, sl, None))
                self.cv.notify()
            self.stats["prefetch_issued"] += len(ks)
        self.stats["t_prefetch"] += time.perf_counter() - t_p
        return len(ks)

    def pin(self, keys: Iterable[Tuple[int, int]]) -> None:
        for key in keys:
            self.ensure(key[0], [key[1]])
            self.pinned.add(key)

    def hit_rate(self) -> float:
        t = self.stats["hits"] + self.stats["misses"]
        return self.stats["hits"] / t if t else 0.0
