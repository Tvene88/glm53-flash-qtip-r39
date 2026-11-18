"""[R3.8] compressed attention history for a 262,144-token window.

The per-token state of the runtime is, per DSA layer, the attention latent `kv` (512 fp32) and the two indexer rows
`ki` / `gs` (128 fp32 each): 33,792 bytes per token over the eleven DSA layers, 8.25 GiB at 262,144 tokens. This
module holds the storage format of Round 3.8:

* `kv` is stored as a code per value (8 bits or 4 bits) and one fp32 scale per row. Only rows that attention has
  selected are ever decoded (2,051 per layer and token in the decode step).
* `ki` / `gs` are not stored per token: the indexer reads pool keys (one fp32 row of 128 per `index_kpool` tokens),
  which are computed from the unrounded rows when a pool is written. Selection therefore stays exactly what it is
  with an fp32 state; rounding touches the attention values only.

Bytes per token and layer: 8 bits 516 + 128 = 644, 4 bits 260 + 128 = 388 (fp32: 3,072 + 128).

Codes:
  8 bits  uniform, symmetric, scale = largest magnitude of the row / 127, int8.
  4 bits  16 levels, the minimum-mean-square-error levels of a unit normal value (Lloyd-Max), scale = root mean
          square of the row; two codes per byte (high nibble = even index).
An optional per-channel divisor (`chan`, fp32 [C], from calibration) is applied before coding and after decoding.
A row of zeros (a position not yet written) has scale 0 and decodes to zeros.

Every function is plain torch and has no data-dependent control flow, so the decode of gathered rows can be captured
in a CUDA graph. CPU tests: build_run/test_qstate_r38.py.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

# Lloyd-Max levels of N(0, 1) for 16 levels (positive half); distortion 0.009497 (signal-to-noise 20.2 dB).
_LM16_POS = (0.1284, 0.3881, 0.6568, 0.9424, 1.2562, 1.6181, 2.0690, 2.7326)


def lm16_levels(device=None) -> torch.Tensor:
    pos = torch.tensor(_LM16_POS, dtype=torch.float32, device=device)
    return torch.cat([-pos.flip(0), pos])                                  # [16], increasing


def lm16_bounds(device=None) -> torch.Tensor:
    lv = lm16_levels(device)
    return (lv[1:] + lv[:-1]) * 0.5                                        # [15] decision thresholds


class KvCodec:
    """Storage format of the attention latent. bits: 8 or 4 (32 = stored as fp32, for the reference arm of a gate)."""

    def __init__(self, bits: int, width: int, device=None, chan: Optional[torch.Tensor] = None):
        if bits not in (4, 8, 32):
            raise ValueError("R38_KV_BITS must be 4, 8 or 32")
        if bits == 4 and width % 2:
            raise ValueError("4-bit codes need an even row width")
        self.bits, self.width, self.device = bits, width, device
        self.chan = None if chan is None else chan.to(device=device, dtype=torch.float32).view(width)
        if self.chan is not None and not bool((self.chan > 0).all()):
            raise ValueError("per-channel divisors must be positive")
        self.levels = lm16_levels(device) if bits == 4 else None
        self.bounds = lm16_bounds(device) if bits == 4 else None

    # ---- layout -----------------------------------------------------------------------------------------------
    @property
    def code_width(self) -> int:
        return {4: self.width // 2, 8: self.width, 32: self.width}[self.bits]

    @property
    def code_dtype(self):
        return {4: torch.uint8, 8: torch.int8, 32: torch.float32}[self.bits]

    def row_bytes(self) -> int:
        return {4: self.width // 2 + 4, 8: self.width + 4, 32: self.width * 4}[self.bits]

    def alloc(self, rows: int, device=None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Zeroed storage for `rows` rows: (codes [1, rows, code_width], scale [1, rows, 1]; [1, 1, 1] unused for fp32)."""
        dev = self.device if device is None else device
        codes = torch.zeros((1, rows, self.code_width), dtype=self.code_dtype, device=dev)
        # fp32 arm: no scales; a one-element tensor keeps the state dictionaries free of None (snapshots copy every entry)
        scale = torch.zeros((1, 1 if self.bits == 32 else rows, 1), dtype=torch.float32, device=dev)
        return codes, scale

    # ---- coding -----------------------------------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """x [..., width] float -> (codes [..., code_width], scale [..., 1] fp32 or None)."""
        x = x.float()
        if self.bits == 32:
            return x, None
        if self.chan is not None:
            x = x / self.chan
        if self.bits == 8:
            scale = x.abs().amax(dim=-1, keepdim=True) / 127.0
            q = torch.round(x / scale.clamp_min(1e-30)).clamp_(-127, 127)
            return q.to(torch.int8), scale
        top = x.abs().amax(dim=-1, keepdim=True)                           # squares of x / top cannot overflow
        scale = top * (x / top.clamp_min(1e-30)).pow(2).mean(dim=-1, keepdim=True).sqrt()
        idx = torch.bucketize(x / scale.clamp_min(1e-30), self.bounds).to(torch.uint8)       # 0..15
        idx = idx.view(*idx.shape[:-1], self.width // 2, 2)
        return (idx[..., 0] << 4) | idx[..., 1], scale

    def decode(self, codes: torch.Tensor, scale: Optional[torch.Tensor]) -> torch.Tensor:
        """Inverse of encode: fp32 [..., width]."""
        if self.bits == 32:
            return codes
        if self.bits == 8:
            x = codes.float() * scale
        else:
            idx = torch.stack([codes >> 4, codes & 15], dim=-1).view(*codes.shape[:-1], self.width).long()
            x = self.levels[idx] * scale
        return x if self.chan is None else x * self.chan

    # ---- static buffers ---------------------------------------------------------------------------------------
    def write(self, codes: torch.Tensor, scale: Optional[torch.Tensor], pos: torch.Tensor, x: torch.Tensor) -> None:
        """Store rows x [1, n, width] at positions pos (long [n]) of the static buffers (in place)."""
        c, s = self.encode(x)
        codes.index_copy_(1, pos, c)
        if s is not None:
            scale.index_copy_(1, pos, s)

    def gather(self, codes: torch.Tensor, scale: Optional[torch.Tensor], ids: torch.Tensor) -> torch.Tensor:
        """Rows ids (long [n], already clamped to the buffer) decoded to fp32 [1, n, width]."""
        return self.decode(codes.index_select(1, ids), None if self.bits == 32 else scale.index_select(1, ids))


def state_bytes(tokens: int, dsa_layers: int, kv_rank: int, index_dim: int, kpool: int, bits: int) -> int:
    """Bytes of the per-token state for `tokens` positions in the Round 3.8 format (pool keys fp32, latent coded)."""
    row = {4: kv_rank // 2 + 4, 8: kv_rank + 4, 32: kv_rank * 4}[bits]
    return tokens * dsa_layers * row + (tokens // kpool) * dsa_layers * index_dim * 4


def snr_db(x: torch.Tensor, y: torch.Tensor) -> float:
    err = (x.double() - y.double()).pow(2).sum()
    return float(10 * torch.log10(x.double().pow(2).sum() / err.clamp_min(1e-300)))


# ---- decode step on the coded state (GraphDecoder, R38_KV_BITS set) ---------------------------------------------
# State of one DSA layer:  kvc / kvs  codes and scales of the latent [1, capacity, ...]
#                          ki / gs    the P rows of the OPEN pool only [1, P, hdi] (slot = position mod P)
# Pool keys live in the decoder's lpk buffer [1, capacity / P, hdi] as in Round 3.7.

def kv_bits_from_env(env) -> int:
    """R38_KV_BITS: unset / '' / '0' = Round 3.7 state (0); 4, 8 or 32 = the coded layout."""
    raw = str(env.get("R38_KV_BITS", "0") or "0").strip()
    if raw not in ("0", "4", "8", "32"):
        raise ValueError("R38_KV_BITS must be 4, 8 or 32 (unset = the Round 3.7 state)")
    return int(raw)


def alloc_layer_state(codec: KvCodec, capacity: int, kpool: int, index_dim: int, device=None) -> dict:
    codes, scale = codec.alloc(capacity, device)
    return {"kvc": codes, "kvs": scale,
            "ki": torch.zeros((1, kpool, index_dim), dtype=torch.float32, device=device),
            "gs": torch.zeros((1, kpool, index_dim), dtype=torch.float32, device=device)}


def capacity_view(capacity: int, device=None) -> torch.Tensor:
    """[1, capacity, 1] without storage: the indexer reads only the capacity from the rows it is handed."""
    return torch.zeros((1, 1, 1), dtype=torch.float32, device=device).expand(1, capacity, 1)


def step_write(codec: KvCodec, st: dict, pos: torch.Tensor, kpool: int, kv_new, ki_new, gs_new) -> None:
    """Store the rows of position pos ([1] long): the latent coded, the indexer rows in the open pool's slot."""
    codec.write(st["kvc"], st["kvs"], pos, kv_new)
    slot = pos.remainder(kpool)
    st["ki"].index_copy_(1, slot, ki_new.float())
    st["gs"].index_copy_(1, slot, gs_new.float())


def step_pool_key(bb, l, st: dict, pk: torch.Tensor, pos: torch.Tensor) -> None:
    """The statements of glm53_fold_longctx_r37.step_pool_key on the open pool's rows. Slots above pos mod P hold rows
    of the previous pool; the key of an incomplete pool is never read (the indexer masks it), and when the pool
    completes all P slots are its own, so every key that is read has the bits of Round 3.7."""
    w = bb.layer(l)
    cfg = bb.cfg
    P, hdi = cfg.index_kpool, cfg.index_head_dim
    p = torch.div(pos, P, rounding_mode="floor")
    gk = st["ki"].view(1, 1, P, hdi)
    gg = st["gs"].view(1, 1, P, hdi)
    logits = gg + w["i_ape"].view(1, 1, P, hdi)
    probs = torch.nan_to_num(torch.softmax(logits, dim=2))
    pk.index_copy_(1, p, (probs * gk).sum(dim=2))


def selected_latents(bb, l, x, q_resid, st: dict, pk, pos, arange, buf, codec: KvCodec, shape: torch.Tensor):
    """selected_latents_fast of Round 3.7 on the coded latent: the same positions, their rows decoded to fp32."""
    from glm53_fold_longctx_r37 import select_ids_fast
    ids = select_ids_fast(bb, l, x, q_resid, shape, shape, pk, pos, arange, buf)        # shape: capacity_view()
    T = st["kvc"].shape[1]
    return codec.gather(st["kvc"], st["kvs"], ids.clamp_max(T - 1)), (ids < T).view(1, 1, 1, ids.shape[0])


def load_layer_state(codec: KvCodec, st: dict, pk: torch.Tensor, bb, l, kv, ki, gs, chunk: int = 8192) -> None:
    """Fill one layer's coded state from fp32 rows kv / ki / gs [1, T, ...] (an eager cache). Pool keys of the complete
    pools are those of Round 3.7 (pool_keys_all on their rows); the rows of the open pool go to the slots."""
    from glm53_fold_longctx_r37 import pool_keys_all
    P = bb.cfg.index_kpool
    T = int(kv.shape[1])
    st["kvc"].zero_()
    st["kvs"].zero_()
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        c, sc = codec.encode(kv[:, s:e])
        st["kvc"][:, s:e].copy_(c)
        if sc is not None:
            st["kvs"][:, s:e].copy_(sc)
    full = (T // P) * P
    pk.zero_()
    if full:
        pk[:, :full // P].copy_(pool_keys_all(bb, l, ki[:, :full].float(), gs[:, :full].float()))
    st["ki"].zero_()
    st["gs"].zero_()
    if T > full:
        st["ki"][:, :T - full].copy_(ki[:, full:T])
        st["gs"][:, :T - full].copy_(gs[:, full:T])


# ---- prompt passes on the coded state (eager path, RealGLM53Backbone.dsa with cache.r38 set) ----------------------
# The eager cache carries no DSA history of its own: `cache.r38` is a CodedHistory, a handle on the decoder's static
# buffers. A pass of S positions writes its rows there and every query attends over the rows the indexer selects for
# it, gathered and decoded exactly as the decode step does (the mask of the eager path and this gather are the same
# selection). Nothing is read or allocated in proportion to the whole history except the indexer's scores over the
# visible pools, in blocks of R38_PREFILL_QBLOCK queries (32 x queries x positions bytes per block).

class CodedHistory:
    def __init__(self, codec: KvCodec, dsa: dict, lpk: dict, kpool: int, capacity: int, qblock: int = 16):
        if not 1 <= int(qblock) <= 128:
            raise ValueError("R38_PREFILL_QBLOCK must be 1..128")
        self.codec, self.dsa, self.lpk = codec, dsa, lpk
        self.kpool, self.capacity, self.qblock = int(kpool), int(capacity), int(qblock)
        self.tap = None                         # tests / gates: a list that receives (layer, first position, ids [s, n])

    def clear(self) -> None:
        for st in self.dsa.values():
            for t in st.values():
                t.zero_()
        for pk in self.lpk.values():
            pk.zero_()


def prefill_rows(bb, l, hist: CodedHistory, t0: int, kv_new, ki_new, gs_new) -> None:
    """Store the rows of positions t0 .. t0 + S - 1: latent codes, the keys of every pool these rows complete (the
    statements of pool_keys_all, on unrounded rows), and the rows of the pool left open."""
    from glm53_fold_longctx_r37 import pool_keys_all
    st, pk, P = hist.dsa[l], hist.lpk[l], hist.kpool
    S = int(kv_new.shape[1])
    if t0 + S > hist.capacity:
        raise ValueError(f"the prompt pass ends at {t0 + S}, beyond the capacity {hist.capacity}")
    hist.codec.write(st["kvc"], st["kvs"], torch.arange(t0, t0 + S, device=kv_new.device), kv_new)
    r = t0 % P                                                    # rows of the open pool that are already stored
    ki_all = torch.cat([st["ki"][:, :r], ki_new.float()], dim=1)
    gs_all = torch.cat([st["gs"][:, :r], gs_new.float()], dim=1)
    full = ((r + S) // P) * P
    if full:
        first = (t0 - r) // P
        pk[:, first:first + full // P].copy_(pool_keys_all(bb, l, ki_all[:, :full], gs_all[:, :full]))
    rem = r + S - full
    st["ki"].zero_()
    st["gs"].zero_()
    if rem:
        st["ki"][:, :rem].copy_(ki_all[:, full:])
        st["gs"][:, :rem].copy_(gs_all[:, full:])


def prefill_attend(bb, l, hist: CodedHistory, t0: int, x, q_resid, q_abs, qk: int) -> torch.Tensor:
    """Attention output in the latent space, [1, S, H, C], for the queries at t0 .. t0 + S - 1 (their rows are stored).
    q_abs [1, S, H, C] is the query with W_k absorbed, as in RealGLM53Backbone.dsa."""
    cfg = bb.cfg
    st, codec, P = hist.dsa[l], hist.codec, hist.kpool
    S = int(x.shape[1])
    t_vis = min(hist.capacity, ((t0 + S + P - 1) // P) * P)     # whole pools up to the last query
    n = min(t_vis, cfg.index_topk + P - 1)
    dev = x.device
    shape = capacity_view(t_vis, dev)
    pk = hist.lpk[l][:, :t_vis // P]
    ar = torch.arange(t_vis, device=dev)
    kvc, kvs = st["kvc"][0], st["kvs"][0]
    outs = []
    for s0 in range(0, S, hist.qblock):
        s1 = min(S, s0 + hist.qblock)
        s = s1 - s0
        q_pos = torch.arange(t0 + s0, t0 + s1, device=dev)
        causal = ar.view(1, 1, t_vis) <= q_pos.view(1, s, 1)
        if bb.dsa_skip_indexer_short and t0 + s1 <= cfg.index_topk:
            mask = causal
        else:
            mask = bb._indexer_mask(l, x[:, s0:s1], q_resid[:, s0:s1], shape, shape, q_pos, pool_keys=pk)
            if bb.dsa_skip_indexer_short:                       # per query, as the decode step decides it
                mask = torch.where((q_pos + 1 <= cfg.index_topk).view(1, s, 1), causal, mask)
        m = mask[0]
        cnt = torch.cumsum(m.to(torch.int32), dim=1)
        dst = torch.where(m, cnt - 1, torch.full_like(cnt, n)).long()
        buf = torch.full((s, n + 1), t_vis, dtype=torch.long, device=dev)
        buf.scatter_(1, dst, ar.view(1, t_vis).expand(s, t_vis))
        ids = buf[:, :n]                                          # increasing positions, padded with the sentinel
        if hist.tap is not None:
            hist.tap.append((l, t0 + s0, ids.clone()))
        valid = ids < t_vis
        safe = ids.clamp_max(t_vis - 1)
        rows = codec.decode(kvc[safe], None if codec.bits == 32 else kvs[safe])        # [s, n, C]
        sc = torch.einsum("shc,stc->hst", q_abs[0, s0:s1], rows) * (qk ** -0.5)
        sc = sc.masked_fill(~valid.view(1, s, n), torch.finfo(sc.dtype).min)
        outs.append(torch.einsum("hst,stc->shc", torch.softmax(sc, dim=-1), rows))
        del mask, m, cnt, dst, buf, rows, sc
    return torch.cat(outs, dim=0).unsqueeze(0)
