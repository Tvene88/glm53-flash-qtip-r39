"""[R3.7 Stage 7] cheaper per-token work for the long-context graph step (R36_LONGCTX=1 + R37_LONGCTX_FAST=1).

The Round 3.6 step (glm53_fold_longctx_r36.selected_latents) calls the eager K-pool indexer on the full capacity for
every token. Two parts of that cost grow with the capacity and not with what the token needs:

1. Pool keys. RealGLM53Backbone._indexer_mask rebuilds the key of EVERY pool on every token: two gathers of
   capacity x 128 fp32 and a softmax, per DSA layer (67 MB read per layer at capacity 65,536, eleven layers).
   A pool's key depends only on that pool's own P rows. Here the keys live in a static buffer
   [1, capacity / P, 128]: filled once for the whole prefix when the cache is loaded (pool_keys_all), and on every
   step only the key of the pool that holds the current position is recomputed from its P rows (step_pool_key).
   Keys of incomplete or future pools are never read: the indexer masks every pool whose last row is not visible.
2. Order. selected_latents takes the 2,051 smallest of `capacity` candidates with a sorted topk, i.e. it sorts the
   capacity. The selected positions are written in increasing order by a prefix sum and one scatter instead
   (selected_latents_fast). The result is the same list of positions, padded with the same sentinel.

Selection and attention are otherwise the Round 3.6 statements. ROUNDING class against eager, as Round 3.6: the
gate is build_run/longctx_gate_r37.py with R37_LONGCTX_FAST=1.
"""
from __future__ import annotations

import torch


def pool_keys_all(bb, l, ki_all, gs_all):
    """Key of every pool of ki_all / gs_all [B, T, hdi] -> [B, T / P, hdi]. The statements of _indexer_mask."""
    w = bb.layer(l)
    cfg = bb.cfg
    P, hdi = cfg.index_kpool, cfg.index_head_dim
    T = ki_all.shape[1]
    if T % P:
        raise ValueError("R37_LONGCTX_FAST needs a capacity that is a multiple of index_kpool")
    n_pools = T // P
    pool_idx = torch.arange(n_pools * P, device=ki_all.device).view(n_pools, P)
    gk = ki_all[:, pool_idx]
    gg = gs_all[:, pool_idx]
    logits = gg + w["i_ape"].view(1, 1, P, hdi)
    probs = torch.nan_to_num(torch.softmax(logits, dim=2))
    return (probs * gk).sum(dim=2)


def step_pool_key(bb, l, st, pk, pos, ar_p):
    """Recompute the key of the pool that holds position `pos` ([1] long) from its P rows of st['ki'] / st['gs'] and
    store it in pk [1, T / P, hdi]. Graph-capturable (no host read). Call it after this position's rows are written."""
    w = bb.layer(l)
    cfg = bb.cfg
    P, hdi = cfg.index_kpool, cfg.index_head_dim
    p = torch.div(pos, P, rounding_mode="floor")
    rows = p * P + ar_p
    gk = st["ki"].index_select(1, rows).view(1, 1, P, hdi)
    gg = st["gs"].index_select(1, rows).view(1, 1, P, hdi)
    logits = gg + w["i_ape"].view(1, 1, P, hdi)
    probs = torch.nan_to_num(torch.softmax(logits, dim=2))
    pk.index_copy_(1, p, (probs * gk).sum(dim=2))


def select_ids_fast(bb, l, x, q_resid, ki, gs, pk, pos, arange, buf):
    """The positions the query at `pos` attends, in increasing order, padded with the sentinel T (= ki.shape[1]):
    long [n], n = min(T, index_topk + P - 1). buf: static long [n + 1]; its last element is a dump slot."""
    cfg = bb.cfg
    mask = bb._indexer_mask(l, x, q_resid, ki, gs, pos, pool_keys=pk)
    causal = (arange <= pos).view(1, 1, -1)
    if bb.dsa_skip_indexer_short:
        mask = torch.where((pos + 1 <= cfg.index_topk).view(1, 1, 1), causal, mask)
    T = ki.shape[1]
    n = min(T, cfg.index_topk + cfg.index_kpool - 1)
    m = mask[0, 0]
    cnt = torch.cumsum(m.to(torch.int32), dim=0)
    dst = torch.where(m, cnt - 1, torch.full_like(cnt, n)).long()     # unselected positions all go to the dump slot
    buf.fill_(T)                                                       # sentinel T = "no position", as Round 3.6
    buf.scatter_(0, dst, arange)
    return buf[:n]


def selected_latents_fast(bb, l, x, q_resid, st, pk, pos, arange, buf):
    """selected_latents of Round 3.6 with the pool keys taken from pk and the selected positions ordered by a prefix
    sum (select_ids_fast)."""
    ids = select_ids_fast(bb, l, x, q_resid, st["ki"], st["gs"], pk, pos, arange, buf)
    T = st["kv"].shape[1]
    return st["kv"].index_select(1, ids.clamp_max(T - 1)), (ids < T).view(1, 1, 1, ids.shape[0])
