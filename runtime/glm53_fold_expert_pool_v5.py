#!/usr/bin/env python3
"""
glm53_fold_expert_pool_v5.py  [Round 3.3, Claude]
=================================================
[S2] v2 + registered page-cache tier: misses on registered layers are direct DMA from locked read-only file views.

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

import atexit
import json
import os
import queue
import threading
import time
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple

import torch


class _RegisteredViews:
    """[S2] read-only Win32 file views of layer files, registered as CUDA host memory and locked."""

    def __init__(self):
        import ctypes
        from ctypes import wintypes
        self.ct = ctypes
        self.cud = ctypes.WinDLL(os.path.join(os.path.dirname(torch.__file__), "lib", "cudart64_12.dll"))
        self.cud.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        self.cud.cudaHostUnregister.argtypes = [ctypes.c_void_p]
        self.cud.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.cud.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                                             ctypes.c_void_p]
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

    def close(self):
        for v, size, hm, hf, off in self.views.values():
            self.cud.cudaHostUnregister(v)
            self.k32.UnmapViewOfFile(v); self.k32.CloseHandle(hm); self.k32.CloseHandle(hf)
        self.views.clear()


class ExpertPool:
    def __init__(self, sources: Dict[int, torch.Tensor], row_bytes: int, n_slots: int, device: torch.device,
                 staging_rows: int = 48, ring_slots: int = 0, threads: int = 0,
                 files: Optional[Dict[int, Tuple[str, int]]] = None, register_gib: float = 0.0,
                 register_floor_gib: float = 10.0, **_ignored):
        self.device = device
        self.row_bytes = int(row_bytes)
        self.n_slots = int(n_slots)
        self.pool = torch.empty((self.n_slots, self.row_bytes), dtype=torch.uint8, device=device)
        self.sources = sources
        self.row_of = {l: [int(r) for r in v.tolist()] for l, v in (_ignored.get("row_of") or {}).items()}  # [R3.3]
        # [R3.4-S3] pruned files: experts without a slab row (row_of = -1) are refused by ensure() / prefetch() (a -1 row
        # would copy from 1 row before the mapped view); only layers that have such experts are checked
        self._missing = {l: {e for e, r in enumerate(ro) if r < 0} for l, ro in self.row_of.items()}
        self._missing = {l: s for l, s in self._missing.items() if s}
        self.slot_of: Dict[Tuple[int, int], int] = {}
        self.lru: "OrderedDict[Tuple[int, int], int]" = OrderedDict()
        self.pinned: set = set()
        self.free: List[int] = list(range(self.n_slots))
        self.lock = threading.Lock()
        self.R = int(staging_rows)
        # synchronous path buffers (compute stream)
        self.pin_a = torch.empty((self.R, self.row_bytes), dtype=torch.uint8, pin_memory=True)
        self.stg_a = torch.empty((self.R, self.row_bytes), dtype=torch.uint8, device=device)
        self.ev_a = None
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
        # [S2] registered tier: whole layers, in layer order, within the RAM budget and above the floor
        self.reg = None
        self.reg_layers = set()
        self.reg_rows: Dict[int, int] = {}                  # [R3.3-rows] layer -> rows 0..K-1 registered
        self.stats["bytes_dma_registered"] = 0
        if files and register_gib > 0:
            import time as _t
            t0 = _t.perf_counter()
            self.reg = _RegisteredViews()
            scratch = torch.empty((64 << 20,), dtype=torch.uint8, device=device)
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
            torch.cuda.synchronize(device)
            self.stats["registered_layers"] = len(self.reg_rows)
            self.stats["registered_rows"] = int(sum(self.reg_rows.values()))
            self.stats["registered_gib"] = round(float(register_gib) - budget, 2)
            self.stats["register_s"] = round(_t.perf_counter() - t0, 1)
            atexit.register(self.reg.close)
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
    def _gather(self, l: int, experts: List[int], pin: torch.Tensor) -> int:
        m = len(experts)
        src = self.sources[l]
        t0 = time.perf_counter()
        if src.is_cuda:
            return -m  # caller copies device->device
        ro = self.row_of.get(l)
        rows_idx = [ro[e] for e in experts] if ro is not None else experts       # [R3.3] expert -> slab row
        torch.index_select(src, 0, torch.tensor(rows_idx, dtype=torch.long), out=pin[:m])
        self.stats["t_gather"] += time.perf_counter() - t0
        return m

    def _sync_fetch(self, l: int, experts: List[int], slots: List[int]) -> None:
        """Current-layer misses, on the compute stream (ordered before the expert kernels)."""
        for c0 in range(0, len(experts), self.R):
            ex, sl = experts[c0:c0 + self.R], slots[c0:c0 + self.R]
            m = len(ex)
            if self.ev_a is not None:
                self.ev_a.synchronize()  # pinned block reuse
            src = self.sources[l]
            kreg = self.reg_rows.get(l, 0)
            if kreg:                                         # [S2] direct DMA from the locked view
                h = torch.cuda.current_stream(self.device).cuda_stream
                ro = self.row_of.get(l)
                rest_e, rest_s = [], []
                for e, s_ in zip(ex, sl):
                    r_ = ro[e] if ro is not None else e
                    if r_ < kreg:
                        self.reg.copy_row(l, r_, self.row_bytes, self.pool[s_].data_ptr(), h)
                    else:
                        rest_e.append(e)
                        rest_s.append(s_)
                nd = m - len(rest_e)
                self.stats["bytes_dma_registered"] += nd * self.row_bytes
                self.stats["bytes_h2d"] += nd * self.row_bytes
                if not rest_e:
                    continue
                ex, sl, m = rest_e, rest_s, len(rest_e)
            slot_t = torch.tensor(sl, dtype=torch.long, pin_memory=True).to(self.device, non_blocking=True)
            if src.is_cuda:
                ro = self.row_of.get(l)
                exr = [ro[e] for e in ex] if ro is not None else ex
                self.pool.index_copy_(0, slot_t, src.index_select(0, torch.tensor(exr, dtype=torch.long, device=self.device)))
            else:
                self._gather(l, ex, self.pin_a)
                self.stg_a[:m].copy_(self.pin_a[:m], non_blocking=True)
                self.pool.index_copy_(0, slot_t, self.stg_a[:m])
            self.ev_a = torch.cuda.Event()
            self.ev_a.record()
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
                    kreg = self.reg_rows.get(l, 0)
                    if kreg:                                 # [S2] direct DMA from the locked view
                        h = self.copy_stream.cuda_stream
                        ro = self.row_of.get(l)
                        rest_e, rest_s = [], []
                        for e, s_ in zip(ex, sl):
                            r_ = ro[e] if ro is not None else e
                            if r_ < kreg:
                                self.reg.copy_row(l, r_, self.row_bytes, self.pool[s_].data_ptr(), h)
                            else:
                                rest_e.append(e)
                                rest_s.append(s_)
                        self.stats["bytes_dma_registered"] += (m - len(rest_e)) * self.row_bytes
                        ex, sl = rest_e, rest_s
                    mm = len(ex)
                    slot_t = torch.tensor(sl, dtype=torch.long, pin_memory=True).to(self.device, non_blocking=True)
                    if mm == 0:
                        pass
                    elif src.is_cuda:
                        ro_ = self.row_of.get(l)
                        exr = [ro_[e] for e in ex] if ro_ is not None else ex
                        self.pool.index_copy_(0, slot_t, src.index_select(0, torch.tensor(exr, dtype=torch.long, device=self.device)))
                    else:
                        self._gather(l, ex, self.pin_b)
                        self.stg_b[:mm].copy_(self.pin_b[:mm], non_blocking=True)
                        self.pool.index_copy_(0, slot_t, self.stg_b[:mm])
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
    def _check_rows(self, l: int, keys) -> None:
        """[R3.4-S3] raise for experts that have no row in layer l's (pruned) file."""
        ms = self._missing.get(l)
        if ms:
            bad = sorted({e for _, e in keys if e in ms})
            if bad:
                raise RuntimeError(f"ExpertPool: layer {l} experts {bad} have no row in the archive (pruned, row_of = -1)"
                                   f" - the router mask for them is missing")

    def ensure(self, l: int, experts: Iterable[int], out=None) -> torch.Tensor:
        """Slots of layer l's experts (missing rows are fetched on the compute stream; in-flight prefetches are waited
        for). Returns a new int32 device tensor - or, [R3.4-2g], writes them into `out` (a host int32 array, e.g. the
        numpy view of a pinned mailbox row that a graph copies to the GPU) and returns `out`."""
        t_e = time.perf_counter()
        keys = [(l, int(e)) for e in experts]
        if self._missing:
            self._check_rows(l, keys)
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
            if not ev.query():                    # [R3.4-2g] a completed copy needs no stream wait
                cur.wait_event(ev)
        if out is not None:                       # [R3.4-2g] no allocation, no pageable upload
            out[:len(slots)] = slots
        else:
            out = torch.tensor(slots, dtype=torch.int32).to(self.device, non_blocking=True)
        self.stats["t_ensure"] += time.perf_counter() - t_e
        return out

    def prefetch(self, l: int, experts: Iterable[int], protect: Iterable[Tuple[int, int]] = ()) -> int:
        t_p = time.perf_counter()
        keys = [(l, int(e)) for e in experts]
        if self._missing:
            self._check_rows(l, keys)
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
