#!/usr/bin/env python3
"""
glm53_fold_expert_pool.py  [Round 3.2, Claude]
==============================================
Global VRAM expert cache shared by ALL MoE layers (replaces "N whole layers resident + the rest streamed").

  * pool        : uint8 [n_slots, row_bytes] on the GPU; one slot = one expert's full single-slab row (6.88 MB at
                  high_fidelity), i.e. the exact bytes of the archive -> decoding is bit-identical to the slab path.
  * policy      : LRU with optional pinned keys (super experts / hot experts are never evicted).
  * misses      : host row (memory-mapped archive) -> pinned ring slot (multi-threaded CPU copy) -> async H2D copy on a
                  dedicated CUDA stream -> the compute stream waits on a per-request event.
  * prefetch    : `prefetch(l, ids)` starts the same pipeline without waiting (used with the next layer's router
                  applied to the current layer's FFN input; HOBBIT/ProMoE-style pre-gating, no fine-tuning).

Safety: slots used by the current request are locked; a ring slot is reused only after its copy event completed.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


class ExpertPool:
    def __init__(self, sources: Dict[int, torch.Tensor], row_bytes: int, n_slots: int, device: torch.device,
                 ring_slots: int = 64, threads: int = 6, **_ignored):
        self.device = device
        self.row_bytes = int(row_bytes)
        self.n_slots = int(n_slots)
        self.pool = torch.empty((self.n_slots, self.row_bytes), dtype=torch.uint8, device=device)
        self.sources = sources  # layer -> uint8 [288, row_bytes] CPU tensor (mmap) or CUDA tensor (resident)
        self.slot_of: Dict[Tuple[int, int], int] = {}
        self.lru: "OrderedDict[Tuple[int, int], int]" = OrderedDict()
        self.pinned: set = set()
        self.free: List[int] = list(range(self.n_slots))
        self.ring = torch.empty((ring_slots, self.row_bytes), dtype=torch.uint8, pin_memory=True)
        self.ring_events: List[Optional[torch.cuda.Event]] = [None] * ring_slots
        self.ring_next = 0
        self.copy_stream = torch.cuda.Stream(device=device)
        self.inflight: Dict[Tuple[int, int], torch.cuda.Event] = {}
        self.exec = ThreadPoolExecutor(max_workers=threads)
        self.lock = threading.Lock()
        self.stats = {"hits": 0, "misses": 0, "prefetch_issued": 0, "prefetch_useful": 0, "bytes_h2d": 0,
                      "t_ensure": 0.0, "t_prefetch": 0.0, "t_gather": 0.0}
        self._prefetched: set = set()

    # ------------------------------------------------------------------ slot management
    def _evict_one(self, locked: set) -> int:
        for key in self.lru:
            if key in self.pinned or key in locked or key in self.inflight:
                continue
            slot = self.lru.pop(key)
            del self.slot_of[key]
            self._prefetched.discard(key)
            return slot
        raise RuntimeError("ExpertPool: no evictable slot (pool too small for one request)")

    def _alloc(self, key, locked: set) -> int:
        slot = self.free.pop() if self.free else self._evict_one(locked)
        self.slot_of[key] = slot
        self.lru[key] = slot
        return slot

    def _ring_slot(self) -> int:
        j = self.ring_next
        self.ring_next = (j + 1) % self.ring.shape[0]
        ev = self.ring_events[j]
        if ev is not None:
            ev.synchronize()
        return j

    def _issue_copies(self, items: List[Tuple[Tuple[int, int], int]]) -> torch.cuda.Event:
        """items: [(key, slot)] -> host rows gathered into pinned ring slots by worker threads (torch copies release
        the GIL), then one async H2D copy per row on `copy_stream`; returns the event recorded after the copies."""
        jobs, futs = [], []
        for key, slot in items:
            l, e = key
            src = self.sources[l]
            if src.is_cuda:
                jobs.append((key, slot, None, src[e]))
            else:
                j = self._ring_slot()
                jobs.append((key, slot, j, None))
                futs.append(self.exec.submit(self.ring[j].copy_, src[e]))
        t_g = time.perf_counter()
        for f in futs:
            f.result()
        self.stats["t_gather"] += time.perf_counter() - t_g
        with torch.cuda.stream(self.copy_stream):
            for key, slot, j, gsrc in jobs:
                self.pool[slot].copy_(gsrc if gsrc is not None else self.ring[j], non_blocking=True)
                self.stats["bytes_h2d"] += self.row_bytes
            ev = torch.cuda.Event()
            ev.record(self.copy_stream)
        for key, slot, j, _ in jobs:
            if j is not None:
                self.ring_events[j] = ev
            self.inflight[key] = ev
        return ev

    # ------------------------------------------------------------------ public API
    def ensure(self, l: int, experts: Iterable[int], out=None) -> torch.Tensor:
        """Make experts of layer l resident; returns int32 slot ids (CUDA). The current stream waits for any copy."""
        t_e = time.perf_counter()
        keys = [(l, int(e)) for e in experts]
        locked = set(keys)
        need: List[Tuple[Tuple[int, int], int]] = []
        slots: List[int] = []
        wait_events = set()
        for key in keys:
            if key in self.slot_of:
                self.stats["hits"] += 1
                if key in self._prefetched:
                    self.stats["prefetch_useful"] += 1
                    self._prefetched.discard(key)
                self.lru.move_to_end(key)
                if key in self.inflight:
                    wait_events.add(self.inflight[key])
                slots.append(self.slot_of[key])
            else:
                self.stats["misses"] += 1
                s = self._alloc(key, locked)
                need.append((key, s))
                slots.append(s)
        if need:
            wait_events.add(self._issue_copies(need))
        cur = torch.cuda.current_stream(self.device)
        for ev in wait_events:
            cur.wait_event(ev)
        self._retire()
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
        need = []
        for key in keys:
            if key not in self.slot_of:
                need.append((key, self._alloc(key, locked)))
                self._prefetched.add(key)
        if need:
            self._issue_copies(need)
            self.stats["prefetch_issued"] += len(need)
        self.stats["t_prefetch"] += time.perf_counter() - t_p
        return len(need)

    def pin(self, keys: Iterable[Tuple[int, int]]) -> None:
        for key in keys:
            l, e = key
            if key not in self.slot_of:
                self._alloc(key, set())
                self._issue_copies([(key, self.slot_of[key])])
            self.pinned.add(key)

    def _retire(self):
        done = [k for k, ev in self.inflight.items() if ev.query()]
        for k in done:
            del self.inflight[k]

    def hit_rate(self) -> float:
        t = self.stats["hits"] + self.stats["misses"]
        return self.stats["hits"] / t if t else 0.0
