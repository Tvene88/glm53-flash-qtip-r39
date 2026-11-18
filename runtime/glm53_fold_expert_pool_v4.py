#!/usr/bin/env python3
"""
glm53_fold_expert_pool_v4.py  [Round 3.2, Claude]
=================================================
[1.6e] v2 + threaded file-read gather into pinned memory (no mmap soft faults) + double-buffered sync path.

Global VRAM expert cache (same contract as glm53_fold_expert_pool.ExpertPool) with batched, asynchronous transfers:

  * synchronous misses of the current layer: ONE torch.index_select of all missing rows from the memory-mapped slab
    into a pinned block (torch parallelises the row copies), ONE H2D copy into a GPU staging block, ONE index_copy_
    scatter into the pool slots - all on the compute stream, so the expert kernels queued next see the data.
  * prefetch (next layer, predicted by its router): handed to a background worker thread that runs the same batched
    gather on its own CUDA stream; per-key readiness (threading.Event + CUDA event) lets ensure() wait only for the
    rows it actually needs.
  * LRU eviction; keys in flight or pinned are never evicted.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional, Tuple

import torch


class ExpertPool:
    def __init__(self, sources: Dict[int, torch.Tensor], row_bytes: int, n_slots: int, device: torch.device,
                 staging_rows: int = 48, ring_slots: int = 0, threads: int = 0,
                 files: Optional[Dict[int, Tuple[str, int]]] = None, gather_threads: int = 8,
                 seg_bytes: int = 2 << 20, **_ignored):
        self.device = device
        self.row_bytes = int(row_bytes)
        self.n_slots = int(n_slots)
        self.pool = torch.empty((self.n_slots, self.row_bytes), dtype=torch.uint8, device=device)
        self.sources = sources
        self.slot_of: Dict[Tuple[int, int], int] = {}
        self.lru: "OrderedDict[Tuple[int, int], int]" = OrderedDict()
        self.pinned: set = set()
        self.free: List[int] = list(range(self.n_slots))
        self.lock = threading.Lock()
        self.R = int(staging_rows)
        # synchronous path buffers (compute stream)
        # [1.6e] two pinned/staging pairs used alternately, so chunk k+1 is read while chunk k is copied
        self.pin_a2 = [torch.empty((self.R, self.row_bytes), dtype=torch.uint8, pin_memory=True) for _ in range(2)]
        self.stg_a2 = [torch.empty((self.R, self.row_bytes), dtype=torch.uint8, device=device) for _ in range(2)]
        self.ev_a2 = [None, None]
        self.pin_a, self.stg_a, self.ev_a = self.pin_a2[0], self.stg_a2[0], None
        # [1.6e] file-read gather: layer -> (path, byte offset of the slab inside the .npz)
        self.files = dict(files or {})
        self.seg = max(1 << 16, int(seg_bytes))
        self._tp = ThreadPoolExecutor(max_workers=int(gather_threads)) if self.files else None
        self._tls = threading.local()
        self._np_views: Dict[int, object] = {}
        # prefetch worker buffers (copy stream)
        self.pin_b = torch.empty((self.R, self.row_bytes), dtype=torch.uint8, pin_memory=True)
        self.stg_b = torch.empty((self.R, self.row_bytes), dtype=torch.uint8, device=device)
        self.copy_stream = torch.cuda.Stream(device=device)
        self.pending: Dict[Tuple[int, int], threading.Event] = {}
        self.ready_cuda: Dict[Tuple[int, int], torch.cuda.Event] = {}
        self.q: "queue.Queue" = queue.Queue()
        self.stats = {"hits": 0, "misses": 0, "prefetch_issued": 0, "prefetch_useful": 0, "bytes_h2d": 0,
                      "t_ensure": 0.0, "t_prefetch": 0.0, "t_gather": 0.0, "t_wait_prefetch": 0.0}
        self._prefetched: set = set()
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    # ------------------------------------------------------------------ slots (call with self.lock held)
    def _evict_one(self, locked: set) -> int:
        for key in self.lru:
            if key in self.pinned or key in locked or key in self.pending:
                continue
            ev = self.ready_cuda.get(key)
            if ev is not None and not ev.query():
                continue  # its prefetch copy may still be writing this slot
            slot = self.lru.pop(key)
            del self.slot_of[key]
            self._prefetched.discard(key)
            self.ready_cuda.pop(key, None)
            return slot
        raise RuntimeError("ExpertPool: no evictable slot")

    def _alloc(self, key, locked: set) -> int:
        slot = self.free.pop() if self.free else self._evict_one(locked)
        self.slot_of[key] = slot
        self.lru[key] = slot
        return slot

    # ------------------------------------------------------------------ transfers
    def _read_seg(self, path: str, pos: int, dst) -> None:
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

    def _gather(self, l: int, experts: List[int], pin: torch.Tensor) -> int:
        m = len(experts)
        src = self.sources[l]
        t0 = time.perf_counter()
        if src.is_cuda:
            return -m  # caller copies device->device
        if l in self.files and self._tp is not None:
            path, off = self.files[l]
            key = pin.data_ptr()
            pv = self._np_views.get(key)
            if pv is None:
                pv = self._np_views[key] = pin.numpy()
            rb, seg = self.row_bytes, self.seg
            futs = []
            for j, e in enumerate(experts):
                base = off + int(e) * rb
                for s0 in range(0, rb, seg):
                    s1 = min(rb, s0 + seg)
                    futs.append(self._tp.submit(self._read_seg, path, base + s0, pv[j, s0:s1]))
            for fu in futs:
                fu.result()
        else:
            torch.index_select(src, 0, torch.tensor(experts, dtype=torch.long), out=pin[:m])
        self.stats["t_gather"] += time.perf_counter() - t0
        return m

    def _sync_fetch(self, l: int, experts: List[int], slots: List[int]) -> None:
        """Current-layer misses, on the compute stream (ordered before the expert kernels)."""
        for k, c0 in enumerate(range(0, len(experts), self.R)):
            ex, sl = experts[c0:c0 + self.R], slots[c0:c0 + self.R]
            m = len(ex)
            b = k & 1
            if self.ev_a2[b] is not None:
                self.ev_a2[b].synchronize()  # pinned block reuse (the other block may still be copying)
            src = self.sources[l]
            slot_t = torch.tensor(sl, dtype=torch.long, device=self.device)
            if src.is_cuda:
                self.pool.index_copy_(0, slot_t, src.index_select(0, torch.tensor(ex, dtype=torch.long, device=self.device)))
            else:
                self._gather(l, ex, self.pin_a2[b])
                self.stg_a2[b][:m].copy_(self.pin_a2[b][:m], non_blocking=True)
                self.pool.index_copy_(0, slot_t, self.stg_a2[b][:m])
            ev = torch.cuda.Event()
            ev.record()
            self.ev_a2[b] = ev
            self.stats["bytes_h2d"] += m * self.row_bytes

    def _worker_loop(self):
        torch.cuda.set_device(self.device.index if self.device.index is not None else torch.cuda.current_device())
        while True:
            item = self.q.get()
            if item is None:
                return
            l, experts, slots, keys = item
            for c0 in range(0, len(experts), self.R):
                ex, sl, ks = experts[c0:c0 + self.R], slots[c0:c0 + self.R], keys[c0:c0 + self.R]
                m = len(ex)
                src = self.sources[l]
                with torch.cuda.stream(self.copy_stream):
                    slot_t = torch.tensor(sl, dtype=torch.long, device=self.device)
                    if src.is_cuda:
                        self.pool.index_copy_(0, slot_t, src.index_select(0, torch.tensor(ex, dtype=torch.long, device=self.device)))
                    else:
                        self._gather(l, ex, self.pin_b)
                        self.stg_b[:m].copy_(self.pin_b[:m], non_blocking=True)
                        self.pool.index_copy_(0, slot_t, self.stg_b[:m])
                    ev = torch.cuda.Event()
                    ev.record(self.copy_stream)
                with self.lock:
                    for k in ks:
                        self.ready_cuda[k] = ev
                        e = self.pending.pop(k, None)
                        if e is not None:
                            e.set()
                self.stats["bytes_h2d"] += m * self.row_bytes
                ev.synchronize()  # before reusing pin_b / stg_b

    # ------------------------------------------------------------------ public API
    def ensure(self, l: int, experts: Iterable[int], out=None) -> torch.Tensor:
        t_e = time.perf_counter()
        keys = [(l, int(e)) for e in experts]
        locked = set(keys)
        miss_e: List[int] = []
        miss_s: List[int] = []
        waits: List[threading.Event] = []
        cuda_waits = []
        slots: List[int] = []
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
                    elif key in self.ready_cuda:
                        cuda_waits.append(self.ready_cuda.pop(key))
                    slots.append(self.slot_of[key])
                else:
                    self.stats["misses"] += 1
                    s = self._alloc(key, locked)
                    miss_e.append(key[1])
                    miss_s.append(s)
                    slots.append(s)
        if miss_e:
            self._sync_fetch(l, miss_e, miss_s)
        if waits:
            t_w = time.perf_counter()
            for w in waits:
                w.wait()
            self.stats["t_wait_prefetch"] += time.perf_counter() - t_w
            with self.lock:
                for key in keys:
                    ev = self.ready_cuda.pop(key, None)
                    if ev is not None:
                        cuda_waits.append(ev)
        cur = torch.cuda.current_stream(self.device)
        for ev in cuda_waits:
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
        ex, sl, ks = [], [], []
        with self.lock:
            for key in keys:
                if key in self.slot_of:
                    continue
                s = self._alloc(key, locked)
                self.pending[key] = threading.Event()
                self._prefetched.add(key)
                ex.append(key[1]); sl.append(s); ks.append(key)
        if ex:
            self.q.put((l, ex, sl, ks))
            self.stats["prefetch_issued"] += len(ex)
        self.stats["t_prefetch"] += time.perf_counter() - t_p
        return len(ex)

    def pin(self, keys: Iterable[Tuple[int, int]]) -> None:
        for key in keys:
            l, e = key
            self.ensure(l, [e])
            self.pinned.add(key)

    def hit_rate(self) -> float:
        t = self.stats["hits"] + self.stats["misses"]
        return self.stats["hits"] / t if t else 0.0
