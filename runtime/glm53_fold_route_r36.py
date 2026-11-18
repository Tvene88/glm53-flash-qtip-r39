"""R36 Q: deliberate, opt-in routing approximations. No quality pass implied.

Dropped pairs have id=-1, weight=0, and are never passed to an archive fetch.
Static top-k/min-weight and registered-tier bias are row-local and MTP-safe in
principle (GPU equality gate pending). Live VRAM policy is refused with MTP:
pair/draft scheduling changes residency, so claiming lossless verification would
be wrong without a common frozen residency trajectory. Cold-skip disables all
speculative prefetch; predicting a future weight cannot guarantee a skip.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import os
from contextlib import nullcontext
import numpy as np


@dataclass(frozen=True)
class Policy:
    tau: float=0.
    renorm: bool=True
    bias: float=0.
    tier: str='vram'
    topk: int=8
    minimum: float=0.

    @classmethod
    def from_env(cls,env=None):
        e=os.environ if env is None else env
        p=cls(float(e.get('R36_SKIP_COLD_TAU',0)),e.get('R36_SKIP_RENORM','1')=='1',
              float(e.get('R36_CACHE_BIAS',0)),e.get('R36_CACHE_BIAS_TIER','vram'),
              int(e.get('R36_TOPK',8)),float(e.get('R36_MIN_WEIGHT',0)))
        if e.get('R36_SKIP_RENORM','1') not in ('0','1') or p.tier not in ('vram','registered') or p.topk not in (6,7,8):
            raise ValueError('invalid R36 route policy enum')
        if any(not math.isfinite(v) or v<0 for v in (p.tau,p.bias,p.minimum)) or max(p.tau,p.minimum)>1:
            raise ValueError('R36 thresholds must be finite, nonnegative, and weights <=1')
        return p

    @property
    def active(self):return bool(self.tau or self.bias or self.topk!=8 or self.minimum)
    @property
    def live_residency(self):return bool(self.tau or (self.bias and self.tier=='vram'))
    def require_mtp(self):
        if self.live_residency:
            raise ValueError('R36 live-residency routing + MTP is unsupported: a shared residency trajectory is required')


def route_reference(scores,selection_bias,policy,resident=None,registered=None,scale=1.,k=8):
    scores=np.atleast_2d(np.asarray(scores,np.float32));E=scores.shape[-1]
    resident=np.zeros(E,bool) if resident is None else np.asarray(resident,bool)
    registered=np.zeros(E,bool) if registered is None else np.asarray(registered,bool)
    choices=scores+np.asarray(selection_bias,np.float32)
    choices=choices+policy.bias*(resident if policy.tier=='vram' else registered)
    ids=np.argsort(-choices,axis=-1,kind='stable')[:,:k]
    raw=np.take_along_axis(scores,ids,axis=-1)
    normal=raw/(raw.sum(-1,keepdims=True)+1e-20)
    valid=np.isfinite(np.take_along_axis(choices,ids,axis=-1))
    keep=valid & (normal>=policy.minimum) & (resident[ids] | (normal>=policy.tau))
    keep[:,policy.topk:]=False
    for r in range(len(keep)):
        if not keep[r].any():
            if not valid[r].any():raise ValueError('all experts masked')
            keep[r,np.argmax(np.where(valid[r],normal[r],-1))]=True
    mass=np.where(keep,normal,0)
    if policy.renorm:mass=mass/(mass.sum(-1,keepdims=True)+1e-20)
    return mass.astype(np.float32)*scale,np.where(keep,ids,-1),np.where(keep,0,normal).sum(-1)


def ensure_active(pool,layer,ids,out):
    active=[j for j,e in enumerate(ids) if e>=0]
    out[:len(ids)]=-1
    if active:
        tmp=np.empty(len(active),np.int32)
        pool.ensure(layer,[ids[j] for j in active],out=tmp)
        out[active]=tmp
    return out


def fill_resident(a,layer,slot_of,pending,ready_cuda):
    """R3.7 (Q-5): mark a[e] for every resident, settled expert e of `layer`.

    Probes only the candidate keys (layer, 0..len(a)-1) instead of iterating the
    whole all-layer pool map; the result is the same set the R3.6 scan produced.
    Caller holds the pool lock.
    """
    for e in range(len(a)):
        key=(layer,e)
        if key in slot_of and key not in pending:
            # A pending-ready CUDA event is a miss until actually complete.
            event=ready_cuda.get(key)
            if event is None or event.query():a[e]=True
    return a


class Router:
    def __init__(self,runner,policy):
        import torch
        from glm53_fold_tcq_kernels_v3 import rule_a
        self.runner,self.policy=runner,policy
        if runner.pool is None:raise ValueError('R36 Q currently requires an expert pool')
        for rt in runner.moe_layers.values():
            # [R3.9S] a layer runtime that declares masked_pairs (QtipLayerRuntime: slot -1 -> zero pair, no slab
            # read) satisfies what the TCQ runtime needs kernel v3 + rule A for; TCQ layers: the unchanged test.
            if getattr(rt,'masked_pairs',False):
                rt.r36_route=True
                continue
            if not getattr(rt,'r36_tcq',False) or not all(rule_a(lay) for lay in rt.lay.values()):
                raise ValueError('R36 Q requires rule-A R36_TCQ_KERNEL=v3 for masked pairs')
            rt.r36_route=True
        if policy.live_residency and not hasattr(runner.pool,'slot_of'):
            raise ValueError('R36 live residency requires a pool with a published slot map')
        self.vram={};self.reg={};self.pin={};self.ev={};self.pending=set()
        # Prefer the actual router's output dimension; model configurations differ.
        for l in runner.moe_layers:
            E=runner.bb.layer(l)['gate_w'].shape[0]
            self.vram[l]=torch.zeros(E,dtype=torch.bool,device=runner.device)
            self.pin[l]=torch.zeros(E,dtype=torch.bool,pin_memory=runner.device.type=='cuda')
            pool=runner.pool;ro=getattr(pool,'row_of',{}).get(l)
            rows=np.arange(E) if ro is None else np.asarray(ro)
            count=getattr(pool,'reg_rows',{}).get(l,0)
            self.reg[l]=torch.from_numpy((rows>=0)&(rows<count)).to(runner.device)
            if runner.device.type=='cuda':self.ev[l]=torch.cuda.Event()
        if policy.tau:
            runner.prefetch=False
            runner.prefill_prefetch=False
            runner.bb.r36_no_prefetch=True
        self._wrap_prefetch(runner.pool)

    def _wrap_prefetch(self,pool):
        import inspect
        original=pool.prefetch
        params=inspect.signature(original).parameters
        def prefetch(layer,experts,protect=(),scores=None,score_ref=None,**kw):
            if self.policy.tau:return 0
            ids=list(experts);take=[j for j,e in enumerate(ids) if e>=0]
            filtered=[ids[j] for j in take]
            if not filtered:return 0
            # R35 prefetch_scored carries scores through wrappers in _stash.
            # Filter that channel together with IDs, including when an outer
            # compatibility wrapper omits the score keyword arguments.
            stash=getattr(pool,'_stash',None)
            if scores is None and stash is not None and stash[0]==int(layer):
                scores,score_ref=stash[1],stash[2]
            if scores is not None:scores=[scores[j] for j in take]
            if score_ref is not None and not np.isscalar(score_ref):score_ref=[score_ref[j] for j in take]
            args={}
            if 'protect' in params:args['protect']=protect
            if 'scores' in params and scores is not None:args['scores']=scores
            if 'score_ref' in params and score_ref is not None:args['score_ref']=score_ref
            args.update(kw)
            restash=hasattr(pool,'_stash') and scores is not None
            if restash:pool._stash=(int(layer),scores,score_ref)
            try:return original(layer,filtered,**args)
            finally:
                if restash:pool._stash=stash
        pool.prefetch=prefetch

    def publish(self,layer):
        if not self.policy.live_residency or layer not in self.vram:return
        pool=self.runner.pool;pin=self.pin[layer]
        if layer in self.pending:self.ev[layer].synchronize()
        a=pin.numpy();a.fill(False)
        with getattr(pool,'lock',nullcontext()):
            fill_resident(a,layer,pool.slot_of,getattr(pool,'pending',{}),getattr(pool,'ready_cuda',{}))
        self.vram[layer].copy_(pin,non_blocking=True)
        if layer in self.ev:
            self.ev[layer].record();self.pending.add(layer)

    def route(self,bb,layer,x):
        import torch
        import torch.nn.functional as F
        p=self.policy;w=bb.layer(layer)
        scores=torch.sigmoid(F.linear(x.float(),w['gate_w'].float()))
        choices=scores+w['gate_b'].float().unsqueeze(0)
        if p.bias:choices=choices+p.bias*(self.vram[layer] if p.tier=='vram' else self.reg[layer]).float()
        K=int(bb.cfg.num_experts_per_tok)
        selected,idx=torch.topk(choices,K,dim=-1,sorted=False)
        raw=scores.gather(1,idx);normal=raw/(raw.sum(-1,keepdim=True)+1e-20)
        valid=torch.isfinite(selected)
        keep=valid & (normal>=p.minimum)
        if p.tau:keep=keep & (self.vram[layer][idx] | (normal>=p.tau))
        if p.topk<K:
            ranks=torch.topk(selected,p.topk,dim=-1,sorted=False).indices
            rank_keep=torch.zeros_like(keep).scatter_(1,ranks,True);keep=keep & rank_keep
        best=normal.masked_fill(~valid,-1).argmax(-1,keepdim=True)
        rescue=torch.zeros_like(keep).scatter_(1,best,True)
        keep=torch.where(keep.any(-1,keepdim=True),keep,rescue & valid)
        mass=torch.where(keep,normal,0.)
        if p.renorm:mass=mass/(mass.sum(-1,keepdim=True)+1e-20)
        return mass*bb.cfg.routed_scaling_factor,idx.masked_fill(~keep,-1)


def install(runner):
    policy=Policy.from_env()
    if policy.active:
        router=Router(runner,policy)
        runner.bb.r36_router=router
        return router
    return None
