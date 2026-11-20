"""[R3.9SMP, 2026-11-19] Token sampler for the APEX-FOLD runtime (temperature / top-k / top-p).

Why: the runtime decoded greedily only (argmax in the final CUDA graph and in the eager paths); temperature, top_p,
top_k were accepted by the server and ignored. With long high-effort thinking that made GLM-5.3-Flash unable to leave
its thinking block (the end-of-thinking token is likely but rarely the single top token) and prone to repetition loops
(once a passage repeats, repeating it again becomes the argmax). Measured in a fresh agent-harness run on 2026-11-19: one
12,700-token thinking block that announced "let me write the files now" seven times and drafted the whole program
twice before writing anything.

Design:
  * pick(row) is branch-free and capturable: it runs inside the final CUDA graph. It computes BOTH the greedy token
    (torch.argmax of the unmodified logits row, exactly the old expression) and a sampled token, and selects with a
    device flag. Flag off (temperature 0, the default) -> the chosen token is bit-identical to the old greedy path.
  * Sampling: top-K (K = 64) candidates by logit, temperature, nucleus (top_p) cut on the tempered softmax, an optional
    smaller top_k, then the Gumbel-max trick (argmax of tempered logits + Gumbel noise) = one draw from the truncated,
    renormalised distribution.
  * Random numbers live in a static buffer `u` that is refilled by ONE small kernel BEFORE each final-graph replay from
    a private torch.Generator (FinalReplay wrapper below), so no RNG op is captured into a graph and a per-request seed
    is possible. Eager callers call refresh() themselves (pick_token does).
  * Settings are device scalars written before a request (set()); a captured graph reads them at replay.
Env: R39_SAMPLER=0 disables the sampler entirely (the final graph is captured exactly as before).
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch

K_MAX = 64                     # candidates kept before top_p / top_k (a top_p of 0.95 rarely needs more than a few dozen)
U_EPS = 1e-7


def sampler_enabled() -> bool:
    return os.environ.get("R39_SAMPLER", "1").strip().lower() not in ("0", "off", "false", "no")


class Sampler:
    def __init__(self, device, k_max: int = K_MAX):
        self.dev = torch.device(device)
        self.k = int(k_max)
        self.on = torch.zeros(1, dtype=torch.int32, device=self.dev)
        self.inv_temp = torch.ones(1, dtype=torch.float32, device=self.dev)
        self.top_p = torch.ones(1, dtype=torch.float32, device=self.dev)
        self.top_k = torch.full((1,), self.k, dtype=torch.int32, device=self.dev)
        self.u = torch.full((self.k,), 0.5, dtype=torch.float32, device=self.dev)
        self.ar = torch.arange(self.k, dtype=torch.int32, device=self.dev)
        self.gen = torch.Generator(device=self.dev)
        self.gen.seed()
        self.active = False                      # host mirror of `on`
        self.settings: Dict[str, Any] = {"mode": "greedy"}

    # ------------------------------------------------------------------ settings (host, before a request)
    @torch.no_grad()
    def set(self, temperature: Optional[float] = None, top_p: Optional[float] = None, top_k: Optional[int] = None,
            seed: Optional[int] = None) -> Dict[str, Any]:
        t = 0.0 if temperature is None else float(temperature)
        if not (t > 1e-6):
            self.on.fill_(0)
            self.active = False
            self.settings = {"mode": "greedy"}
            return self.settings
        p = 1.0 if top_p is None else min(max(float(top_p), 1e-6), 1.0)
        k = self.k if (top_k is None or int(top_k) <= 0) else min(int(top_k), self.k)
        self.inv_temp.fill_(1.0 / t)
        self.top_p.fill_(p)
        self.top_k.fill_(k)
        self.on.fill_(1)
        if seed is not None:
            self.gen.manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)
        self.active = True
        self.settings = {"mode": "sample", "temperature": t, "top_p": p, "top_k": k,
                         **({"seed": int(seed)} if seed is not None else {})}
        return self.settings

    @torch.no_grad()
    def refresh(self) -> None:
        """New uniforms for the next pick (one kernel, outside any graph). No-op while greedy."""
        if self.active:
            self.u.uniform_(U_EPS, 1.0 - U_EPS, generator=self.gen)

    # ------------------------------------------------------------------ the pick (capturable, branch-free)
    def pick(self, row: torch.Tensor) -> torch.Tensor:
        """row: [V] logits of one position (head rows). Returns [1] int64 head-row index."""
        greedy = torch.argmax(row).view(1)                       # identical to the old greedy expression
        x = row.float()
        v, i = torch.topk(x, self.k, dim=-1, largest=True, sorted=True)
        z = v * self.inv_temp
        p = torch.softmax(z, dim=-1)
        c = torch.cumsum(p, dim=-1)
        keep = ((c - p) < self.top_p) & (self.ar < self.top_k)  # the top candidate is always kept
        z = torch.where(keep, z, torch.full_like(z, float("-inf")))
        g = -torch.log(-torch.log(self.u))
        s = i.index_select(0, torch.argmax(z + g).view(1))
        return torch.where(self.on > 0, s, greedy)


class FinalReplay:
    """Stands in for the captured "final" CUDAGraph in GraphDecoder.graphs: refreshes the sampler's uniforms, then
    replays the graph. Every replay site looks the graph up by name, so all of them get fresh noise."""

    def __init__(self, graph, sampler: Sampler):
        self.graph = graph
        self.sampler = sampler

    def replay(self) -> None:
        self.sampler.refresh()
        self.graph.replay()

    def __getattr__(self, name):                                # anything else (reset, pool, ...) -> the real graph
        return getattr(self.graph, name)


def ensure_sampler(bb, device) -> Optional[Sampler]:
    """The backbone's sampler, created on first use (None with R39_SAMPLER=0)."""
    if not sampler_enabled():
        return None
    s = getattr(bb, "sampler", None)
    if s is None:
        s = Sampler(device)
        bb.sampler = s
    return s


def parse_request_sampling(body: dict, defaults: Dict[str, Any]) -> Dict[str, Any]:
    """OpenAI-style request fields -> sampler settings. Raises ValueError with the offending field name."""
    out = dict(defaults)
    for key, typ in (("temperature", float), ("top_p", float), ("top_k", int), ("seed", int)):
        v = body.get(key)
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(key)
        if typ is int and float(v) != int(v):
            raise ValueError(key)
        out[key] = typ(v)
    t = out.get("temperature")
    if t is not None and (t < 0 or t > 5):
        raise ValueError("temperature")
    tp = out.get("top_p")
    if tp is not None and not (0 < tp <= 1):
        raise ValueError("top_p")
    tk = out.get("top_k")
    if tk is not None and tk < 0:
        raise ValueError("top_k")
    return out
