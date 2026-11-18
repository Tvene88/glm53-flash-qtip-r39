#!/usr/bin/env python3
"""
traffic_sim_r39t.py  [R3.9T]  CPU-only replay of an expert-pool call log (R39T_TRACE=<file> of a speed-gate run)
==================================================================================================================
Reason: which experts are resident decides how many rows are copied per token, and a GPU run costs 3.5 minutes per
setting. The call log holds every ensure() / prefetch() call of a run (prefill and decode, with the prediction scores),
so the pool's bookkeeping (pool v6: GDSF or LRU eviction, reservations of the current layer, the two rank gates) can be
replayed here for other settings in seconds. It models WHICH rows are fetched, not time: a prefetched row counts as
resident from its prefetch() call on. The first variant printed is the logged run's own setting; its counters must
match the pool's (`stats` in the log) - that is the check of this model.

  python traffic_sim_r39t.py out_r39s\\trace_r39t_pool18.json            # the standard table
  python traffic_sim_r39t.py <trace> --slots 3054 --reg-topk 8 --ring 80   # one variant

Columns (per generated token, decode calls only): demand = rows ensure() had to copy itself (each stalls the step for
one row copy), pf = rows prefetched, used = prefetched rows that were then routed, rows = demand + pf (bytes copied).
"""
from __future__ import annotations

import argparse
import heapq
import json
import sys
from collections import OrderedDict


class Gdsf:
    """pool v6's _GDSF with equal costs: H = L + F * C; F survives eviction."""

    def __init__(self, cost=0.3, freq_keep=True):
        self.L = 0.0
        self.H = {}
        self.ver = {}
        self.heap = []
        self.freq = {}
        self.seq = 0
        self.C = cost
        self.freq_keep = freq_keep

    def _push(self, k, h):
        self.seq += 1
        self.H[k] = h
        self.ver[k] = self.seq
        heapq.heappush(self.heap, (h, self.seq, k))
        if len(self.heap) > 4 * len(self.H) + 4096:
            self.heap = [(hh, self.ver[kk], kk) for kk, hh in self.H.items()]
            heapq.heapify(self.heap)

    def _bump(self, k):
        f = self.freq.get(k, 0) + 1
        self.freq[k] = f
        return f

    def insert(self, k, kind):
        if kind == "cold":
            h = self.L
        elif kind == "prefetch":
            h = self.L + self.freq.get(k, 0) * self.C
        else:
            h = self.L + self._bump(k) * self.C
        self._push(k, h)

    def touch(self, k, cold=False):
        if k not in self.H or cold:
            return
        self._push(k, self.L + self._bump(k) * self.C)

    def remove(self, k):
        self.H.pop(k, None)
        self.ver.pop(k, None)

    def pop_victim(self, evictable):
        skipped = []
        victim = None
        heap = self.heap
        while heap:
            h, s, k = heapq.heappop(heap)
            if self.ver.get(k) != s:
                continue
            if evictable(k):
                victim = k
                self.L = max(self.L, h)
                del self.H[k]
                del self.ver[k]
                if not self.freq_keep:
                    self.freq.pop(k, None)
                break
            skipped.append((h, s, k))
        for e in skipped:
            heapq.heappush(heap, e)
        return victim

    def __len__(self):
        return len(self.H)


class Decay:
    """Exponentially decayed access count (LRFU): score = sum over accesses of 2^(-age / hl), age in layer calls / 42
    (= tokens in decode). hl = inf is plain LFU, a small hl approaches LRU. The score survives eviction, as GDSF's
    frequency does. Stored as P = log2(score) + t / hl, which orders keys the same way at any later time."""

    def __init__(self, hl, keep=True):
        self.hl = float(hl)
        self.P = {}          # history of every key ever accessed
        self.H = {}          # residents: key -> priority in the heap
        self.ver = {}
        self.heap = []
        self.seq = 0
        self.t = 0.0
        self.keep = keep

    def _push(self, k, h):
        self.seq += 1
        self.H[k] = h
        self.ver[k] = self.seq
        heapq.heappush(self.heap, (h, self.seq, k))
        if len(self.heap) > 4 * len(self.H) + 4096:
            self.heap = [(hh, self.ver[kk], kk) for kk, hh in self.H.items()]
            heapq.heapify(self.heap)

    def _bump(self, k):
        import math
        tt = self.t / self.hl if self.hl != float("inf") else 0.0
        p0 = self.P.get(k)
        s_old = 2.0 ** (p0 - tt) if p0 is not None else 0.0
        p1 = tt + math.log2(s_old + 1.0)
        self.P[k] = p1
        return p1

    def insert(self, k, kind):
        if kind == "cold":
            h = -1e30
        elif kind == "prefetch":
            h = self.P.get(k, -1e30)
        else:
            h = self._bump(k)
        self._push(k, h)

    def touch(self, k, cold=False):
        if k not in self.H or cold:
            return
        self._push(k, self._bump(k))

    def remove(self, k):
        self.H.pop(k, None)
        self.ver.pop(k, None)

    def pop_victim(self, evictable):
        skipped = []
        victim = None
        heap = self.heap
        while heap:
            h, s_, k = heapq.heappop(heap)
            if self.ver.get(k) != s_:
                continue
            if evictable(k):
                victim = k
                del self.H[k]
                del self.ver[k]
                if not self.keep:
                    self.P.pop(k, None)
                break
            skipped.append((h, s_, k))
        for e in skipped:
            heapq.heappush(heap, e)
        return victim

    def __len__(self):
        return len(self.H)


class LruK(Decay):
    """LRU-2: the victim is the key whose second-last access is oldest (keys seen once go first, oldest first). The
    access history survives eviction."""

    def __init__(self, keep=True):
        Decay.__init__(self, 1.0, keep)

    def _bump(self, k):
        t1 = self.P.get(k)
        last = t1[1] if t1 is not None else None
        self.P[k] = (last, self.t)
        return self._prio(k)

    def _prio(self, k):
        t2, t1 = self.P[k]
        return t2 if t2 is not None else -1e9 + t1

    def insert(self, k, kind):
        if kind == "cold":
            h = -1e30
        elif kind == "prefetch":
            h = self._prio(k) if k in self.P else -1e30
        else:
            h = self._bump(k)
        self._push(k, h)


class Lru:
    def __init__(self):
        self.od = OrderedDict()

    def insert(self, k, kind):
        self.od[k] = None
        if kind == "cold":
            self.od.move_to_end(k, last=False)

    def touch(self, k, cold=False):
        if k in self.od and not cold:
            self.od.move_to_end(k)

    def remove(self, k):
        self.od.pop(k, None)

    def pop_victim(self, evictable):
        for k in self.od:
            if evictable(k):
                del self.od[k]
                return k
        return None

    def __len__(self):
        return len(self.od)


class Sim:
    """Pool v6's resident-set bookkeeping. ring > 0 = [R3.9T] the probation ring (R39T_PF_RING): prefetched rows live
    in `ring` slots of their own (FIFO among the not yet routed ones) and join the main set when they are routed."""

    def __init__(self, n_slots, evict="gdsf", reg_topk=6, reg_margin=0.0, ring=0, prefill_cold=False, cost=0.3,
                 freq_keep=True, pf_bump=False, pf_touch=False):
        self.ring_n = int(ring)
        self.n_main = int(n_slots) - self.ring_n
        if evict == "gdsf":
            self.pol = Gdsf(cost, freq_keep)
        elif evict.startswith("decay:"):                 # decay:<half-life in tokens | inf>
            self.pol = Decay(float(evict.split(":")[1]), freq_keep)
        elif evict == "lru2":
            self.pol = LruK(freq_keep)
        else:
            self.pol = Lru()
        self.n_calls = 0
        self.res = set()                 # main residents
        self.ring = OrderedDict()        # probation residents (prefetched, not routed yet), oldest first
        self.held = set()
        self.held_layer = None
        self.prefetched = set()
        self.reg_topk, self.reg_margin = int(reg_topk), float(reg_margin)
        self.prefill_cold = prefill_cold
        self.pf_bump = pf_bump
        self.pf_touch = pf_touch         # a predicted expert that is resident is refreshed like a hit
        self.c = {"hits": 0, "misses": 0, "pf": 0, "used": 0, "skipped": 0, "noslot": 0, "evictions": 0,
                  "ring_evicted_unused": 0}

    def _evict_main(self, locked):
        k = self.pol.pop_victim(lambda q: q not in locked and q not in self.held)
        if k is None:
            return False
        self.res.discard(k)
        self.prefetched.discard(k)
        self.c["evictions"] += 1
        return True

    def _add_main(self, k, locked, kind):
        if len(self.res) >= self.n_main and not self._evict_main(locked):
            return False
        self.res.add(k)
        self.pol.insert(k, kind)
        return True

    def ensure(self, l, ids, cold):
        keys = [(l, e) for e in ids]
        locked = set(keys)
        self.n_calls += 1
        if hasattr(self.pol, "t"):
            self.pol.t = self.n_calls / 42.0
        if l != self.held_layer:
            self.held.clear()
            self.held_layer = l
        c = self.c
        seen = set()
        for k in keys:
            if k in seen:
                c["hits"] += 1
                self.pol.touch(k, cold)
                continue
            seen.add(k)
            if k in self.res:
                c["hits"] += 1
                if k in self.prefetched:
                    c["used"] += 1
                    self.prefetched.discard(k)
                self.pol.touch(k, cold)
            elif k in self.ring:                         # probation hit: joins the main set (one victim leaves it)
                c["hits"] += 1
                c["used"] += 1
                del self.ring[k]
                if not self._add_main(k, locked, "cold" if cold else "miss"):
                    raise RuntimeError("no evictable slot")
            else:
                c["misses"] += 1
                if not self._add_main(k, locked, "cold" if cold else "miss"):
                    self.held.clear()
                    if not self._add_main(k, locked, "cold" if cold else "miss"):
                        raise RuntimeError("no evictable slot")
        self.held |= seen

    def prefetch(self, l, ids, scores, ref):
        keys = [(l, e) for e in ids]
        locked = set(keys)
        allow_reg = None
        if scores is not None:
            best = {}
            for i, k in enumerate(keys):
                r = ref if not isinstance(ref, list) else ref[i]
                v = scores[i] - r if r is not None else scores[i]
                if k not in best or v > best[k]:
                    best[k] = v
            order = sorted(best, key=lambda k: -best[k])
            if self.reg_topk or self.reg_margin:
                allow_reg = set()
                for rank, k in enumerate(order):
                    if self.reg_topk and rank >= self.reg_topk:
                        continue
                    if self.reg_margin and ref is not None and best[k] < self.reg_margin:
                        continue
                    allow_reg.add(k)
            keys = order
        c = self.c
        for k in keys:
            if k in self.res or k in self.ring:
                if self.pf_touch and k in self.res and k not in self.prefetched:
                    self.pol.touch(k)
                continue
            if allow_reg is not None and k not in allow_reg:
                c["skipped"] += 1
                continue
            if self.ring_n:
                if len(self.ring) >= self.ring_n:
                    old = next((q for q in self.ring if q not in locked), None)
                    if old is None:
                        c["noslot"] += 1
                        break
                    del self.ring[old]
                    c["ring_evicted_unused"] += 1
                self.ring[k] = None
            else:
                if not self._add_main(k, locked, "prefetch"):
                    c["noslot"] += 1
                    break
                self.prefetched.add(k)
            c["pf"] += 1


def belady(calls, n_slots):
    """Fewest rows any eviction rule could fetch for this lookup sequence (prefetch calls ignored; the keys of the
    current call are never evicted). Returns the rows fetched inside decode per decode token."""
    seq = []
    dec = False
    tokens = 0
    first = None
    for c in calls:
        if c[0] == "m":
            if c[1] == "decode_begin":
                dec, first = True, None
            elif c[1] == "decode_end":
                dec = False
        elif c[0] == "e":
            if dec:
                if first is None:
                    first = c[1]
                if c[1] == first:
                    tokens += 1
            seq.append((c[1], c[2], dec))
    nxt = {}
    nu = [None] * len(seq)
    for i in range(len(seq) - 1, -1, -1):
        l, ids, _ = seq[i]
        nu[i] = [nxt.get((l, e), 1 << 60) for e in ids]
        for e in ids:
            nxt[(l, e)] = i
    res = {}
    heap = []
    fetch_dec = 0
    for i, (l, ids, d) in enumerate(seq):
        cur = set((l, e) for e in ids)
        for j, e in enumerate(ids):
            k = (l, e)
            if k not in res:
                if d:
                    fetch_dec += 1
                keep = []
                while len(res) >= n_slots:
                    nx, kk = heapq.heappop(heap)
                    if res.get(kk) != -nx:
                        continue
                    if kk in cur:
                        keep.append((nx, kk))            # a key of the current call stays
                        continue
                    del res[kk]
                for it in keep:
                    heapq.heappush(heap, it)
            res[k] = nu[i][j]
            heapq.heappush(heap, (-nu[i][j], k))
    return fetch_dec / max(1, tokens), tokens


def run(trace, **kw):
    sim = Sim(**kw)
    dec = False
    first = None
    tokens = 0
    tot = None
    dsum = {k: 0 for k in sim.c}
    for c in trace["calls"]:
        t = c[0]
        if t == "m":
            if c[1] == "decode_begin":
                dec, first = True, None
                tot = dict(sim.c)
            elif c[1] == "decode_end":
                dec = False
                for k in dsum:
                    dsum[k] += sim.c[k] - tot[k]
        elif t == "e":
            if dec:
                if first is None:
                    first = c[1]
                if c[1] == first:
                    tokens += 1
            sim.ensure(c[1], c[2], bool(c[3]) or (sim.prefill_cold and not dec))
        else:
            sim.prefetch(c[1], c[2], c[4], c[5])
    n = max(1, tokens)
    return {"tokens": tokens, "demand": dsum["misses"] / n, "pf": dsum["pf"] / n, "used": dsum["used"] / n,
            "skipped": dsum["skipped"] / n, "rows": (dsum["misses"] + dsum["pf"]) / n, "all": dict(sim.c)}


def est_ms(r, base=21.65, c_d=0.227, c_p=0.118):
    """Rough time model, a least-squares-free fit through three GPU runs of 2026-11-17 (pool 18 with the prefetch gate
    at 6 and at 8, pool 16 with the gate at 6): ms per token = 21.65 + 0.227 per demand miss + 0.118 per prefetched
    row. A prefetched row costs half a demand row: the link is busy two thirds of the time, so a prefetch that is not
    routed is not free. Good to about 1 % on the gate-7 run; use it to rank variants, not to predict tok/s."""
    return base + c_d * r["demand"] + c_p * r["pf"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--slots", type=int, default=0)
    ap.add_argument("--evict", default="")
    ap.add_argument("--reg-topk", type=int, default=-1)
    ap.add_argument("--ring", type=int, default=0)
    ap.add_argument("--prefill-cold", action="store_true")
    ap.add_argument("--pf-touch", action="store_true")
    ap.add_argument("--reg-margin", type=float, default=-1.0)
    a = ap.parse_args()
    tr = json.load(open(a.trace, encoding="utf-8"))
    f = tr["flags"]
    n0 = int(tr["n_slots"])
    base = dict(n_slots=n0, evict=f["evict"], reg_topk=int(f["reg_topk"]), reg_margin=float(f["reg_margin"]))
    if a.slots or a.evict or a.reg_topk >= 0 or a.ring or a.prefill_cold or a.pf_touch or a.reg_margin >= 0:
        kw = dict(base)
        if a.slots:
            kw["n_slots"] = a.slots
        if a.evict:
            kw["evict"] = a.evict
        if a.reg_topk >= 0:
            kw["reg_topk"] = a.reg_topk
        kw["ring"] = a.ring
        kw["prefill_cold"] = a.prefill_cold
        kw["pf_touch"] = a.pf_touch
        if a.reg_margin >= 0:
            kw["reg_margin"] = a.reg_margin
        variants = [("logged setting", base), ("requested", kw)]
    else:
        variants = [("logged setting", base)]
        for k in (4, 5, 7, 8):
            variants.append((f"reg_topk {k}", dict(base, reg_topk=k)))
        variants.append(("no prefetch gate (0)", dict(base, reg_topk=0)))
        for m in (0.02, 0.03, 0.04, 0.05, 0.065, 0.08):
            variants.append((f"margin gate {m} (no rank gate)", dict(base, reg_topk=0, reg_margin=m)))
        for ring in (40, 80, 160):
            for k in (6, 8):
                variants.append((f"ring {ring} inside the pool, reg_topk {k}", dict(base, ring=ring, reg_topk=k)))
        for k in (6, 8):
            variants.append((f"ring 80 added (+80 slots), reg_topk {k}", dict(base, n_slots=n0 + 80, ring=80, reg_topk=k)))
        variants.append(("+80 slots (no ring)", dict(base, n_slots=n0 + 80)))
        for g in (18.5, 19.0, 19.5, 20.0):
            variants.append((f"pool {g} GiB", dict(base, n_slots=int(g * 2**30 // tr["row_bytes"]))))
        variants.append(("lru", dict(base, evict="lru")))
        for hl in ("8", "32", "128", "512", "2048", "8192", "inf"):
            variants.append((f"decayed frequency, half-life {hl} tokens", dict(base, evict=f"decay:{hl}")))
        variants.append(("gdsf, frequency reset at eviction", dict(base, freq_keep=False)))
        variants.append(("prefill inserts cold", dict(base, prefill_cold=True)))
    print(f"trace: {len(tr['calls'])} calls, pool {n0} slots, flags evict={f['evict']} reg_topk={f['reg_topk']}")
    st = tr.get("stats") or {}
    print(f"logged pool totals (whole run, prefill included): hits {st.get('hits')} misses {st.get('misses')} "
          f"prefetch_issued {st.get('prefetch_issued')} prefetch_useful {st.get('prefetch_useful')}")
    print(f"{'variant':46s} {'slots':>6s} {'demand':>7s} {'pf':>7s} {'used':>7s} {'rows':>7s} {'est ms':>7s} {'est tok/s':>9s}")
    for name, kw in variants:
        r = run(tr, **kw)
        ms = est_ms(r)
        print(f"{name:46s} {kw['n_slots']:6d} {r['demand']:7.2f} {r['pf']:7.2f} {r['used']:7.2f} {r['rows']:7.2f} "
              f"{ms:7.2f} {1000.0 / ms:9.2f}")
        if name == "logged setting":
            print(f"    simulated totals (whole run): hits {r['all']['hits']} misses {r['all']['misses']} "
                  f"prefetch_issued {r['all']['pf']} prefetch_useful {r['all']['used']}  [check against the logged totals]")
    for n in sorted({n0, int(19.0 * 2**30 // tr["row_bytes"])}):
        b, tk = belady(tr["calls"], n)
        print(f"{'optimal eviction (Belady), no prefetch, ' + str(n):46s} {n:6d} {'':7s} {'':7s} {'':7s} {b:7.2f}   "
              f"(lower bound of rows fetched per token)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
