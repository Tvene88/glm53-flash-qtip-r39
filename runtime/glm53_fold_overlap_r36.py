"""R3.6 K3 hit-first scheduling, pool-v6 lifetime protocol, fixed pair merge.

This conservative first stage still consumes the routed-ID host mailbox. It
publishes per-pair GPU slot tables AFTER ensure has reserved keys; it does not
speculatively read an unreserved global table. The latter can race eviction.
Hit kernels are enqueued before the miss ensure/gather/DMA. Both publish their
results at original pair indices; torch.where selects, then the SAME R35 weighted
sum runs once. No summation depends on arrival order. No atomics.

Use pool v6 only. Other pools retain R35 behavior with flag off. The 2nd ensure
may hand over same-layer holds under capacity pressure, but the hit graph is
already queued on the current stream, satisfying the v6 reader-fence rule.
"""
from __future__ import annotations
import numpy as np


def partition(ids, resident):
    """Order/multiplicity preserved, including duplicate experts across N2 rows."""
    hit=[i for i,e in enumerate(ids) if e>=0 and e in resident]
    miss=[i for i,e in enumerate(ids) if e>=0 and e not in resident]
    return hit,miss


def ready_experts(pool,layer,ids):
    # Snapshot is just a hint. ensure revalidates under lock and waits for bytes.
    # R3.7 (K3-02): probe only the routed ids (at most K*N dict lookups) instead of
    # scanning every resident key of every layer under the pool lock.
    with pool.lock:
        so,pe,rc=pool.slot_of,pool.pending,pool.ready_cuda
        out=set()
        for e in set(int(e) for e in ids):
            if e<0:continue
            key=(layer,e)
            if key in so and key not in pe and key not in rc:out.add(e)
        return out


def record_stats(stats,result):
    """R3.7 (K3-05): accumulate one schedule() result into a counter dict."""
    h,m=int(result['hit_pairs']),int(result['miss_pairs'])
    stats['calls']=stats.get('calls',0)+1
    stats['hit_pairs']=stats.get('hit_pairs',0)+h
    stats['miss_pairs']=stats.get('miss_pairs',0)+m
    stats['all_hit_calls']=stats.get('all_hit_calls',0)+int(h>0 and m==0)
    stats['all_miss_calls']=stats.get('all_miss_calls',0)+int(m>0 and h==0)
    return stats


def schedule(pool,layer,ids,hit_slots,miss_slots,replay_hit,replay_miss):
    """CPU-testable state machine; callbacks enqueue graph reads on current stream."""
    hit,miss=partition(ids,ready_experts(pool,layer,ids))
    hit_slots.fill(-1);miss_slots.fill(-1)
    if hit:
        tmp=np.empty(len(hit),np.int32)
        pool.ensure(layer,[ids[i] for i in hit],out=tmp)
        hit_slots[hit]=tmp
    # Always replay, even for zero hits, to overwrite stale per-pair outputs.
    replay_hit()
    if miss:
        tmp=np.empty(len(miss),np.int32)
        pool.ensure(layer,[ids[i] for i in miss],out=tmp)
        miss_slots[miss]=tmp
    replay_miss()
    return {'hit_pairs':len(hit),'miss_pairs':len(miss)}


class PartitionGraphs:
    def __init__(self,dec,layer,n_tokens,tag):
        import torch
        from glm53_fold_tcq_kernels_v3 import rule_a
        rt=dec.r.moe_layers[layer]
        # [R3.9S] QtipLayerRuntime declares masked_pairs (slot -1 -> zero pair) and has decode_core(pair_output=):
        # it needs no TCQ kernel v3. TCQ layers: the unchanged test.
        if not getattr(rt,'masked_pairs',False) and (
                not getattr(rt,'r36_tcq',False) or not all(rule_a(lay) for lay in rt.lay.values())):
            raise ValueError('R36_OVERLAP requires R36_TCQ_KERNEL=v3 with rule-A TCQ')
        if type(dec.pool).__module__ != 'glm53_fold_expert_pool_v6':
            raise ValueError('R36_OVERLAP requires pool v6 and its reader-fence contract')
        self.dec,self.layer,self.N,self.tag=dec,layer,n_tokens,tag
        P=n_tokens*int(dec.cfg.num_experts_per_tok)
        self.pin=torch.full((2,P),-1,dtype=torch.int32,pin_memory=True)
        self.table=torch.full((2,P),-1,dtype=torch.int32,device=dec.dev)
        self.np=self.pin.numpy()
        self.hit_result=None
        self.done=torch.cuda.Event()
        self.pending=False
        self.stats={}

    def hit(self,x,weights):
        self.table[0].copy_(self.pin[0],non_blocking=True)
        rt=self.dec.r.moe_layers[self.layer]
        return rt.decode_core(x.float(),self.dec.pool.pool,self.table[0],weights.reshape(-1).float(),
                              self.N,self.dec.cfg.swiglu_limit,row_exact=self.N==2,pair_output=True)

    def finish(self,x,weights):
        import torch
        self.table[1].copy_(self.pin[1],non_blocking=True)
        rt=self.dec.r.moe_layers[self.layer]
        miss=rt.decode_core(x.float(),self.dec.pool.pool,self.table[1],weights.reshape(-1).float(),
                            self.N,self.dec.cfg.swiglu_limit,row_exact=self.N==2,pair_output=True)
        pairs=torch.where((self.table[0]>=0).view(-1,1),self.hit_result,miss)
        # All-dropped rows (Q) are zero from masked K1; multiplication order R35.
        dw=pairs*weights.reshape(-1,1).float()
        K=dw.shape[0]//self.N
        return torch.cat([dw[t*K:(t+1)*K].view(1,K,-1).sum(1) for t in range(self.N)],0)

    def replay(self,ids,hit_graph,miss_graph):
        import torch
        # Explicit lifetime protection, including revisits after MTP rollback.
        if self.pending:self.done.synchronize()
        try:
            result=schedule(self.dec.pool,self.layer,ids,self.np[0],self.np[1],hit_graph.replay,miss_graph.replay)
        finally:
            # R3.7 (K3-06): a queued hit graph may still read pin[0] even if the miss
            # ensure raised, so the completion event must always describe this step.
            self.done.record(torch.cuda.current_stream(self.dec.dev));self.pending=True
        # R3.7 (K3-05): per-part and per-decoder hit/miss counters (read by tools via
        # dec.r36_overlap_stats; graphs.py callers still ignore the return value).
        record_stats(self.stats,result)
        if not hasattr(self.dec,'r36_overlap_stats'):self.dec.r36_overlap_stats={}
        record_stats(self.dec.r36_overlap_stats.setdefault(self.tag,{}),result)
        return result


def capture_hit(dec,layer,x,weights,n_tokens,tag,capture):
    if not hasattr(dec,'r36_parts'):dec.r36_parts={}
    part=PartitionGraphs(dec,layer,n_tokens,tag)
    dec.r36_parts[(tag,layer)]=part
    part.hit_result=capture('hit'+tag+str(layer),lambda:part.hit(x,weights))
    return part


def release_parts(dec,tag):
    """Release only this graph set after its pinned-table readers complete."""
    parts=getattr(dec,'r36_parts',{})
    for key in [key for key in parts if key[0]==tag]:
        part=parts[key]
        if part.pending:part.done.synchronize()
        del parts[key]


def merge_reference(hit,miss,hit_slots,weights,n_tokens):
    pairs=np.where(np.asarray(hit_slots)[:,None]>=0,hit,miss)
    # CPU oracle for selection/order, not PyTorch's implementation-specific sum tree.
    return (pairs*np.asarray(weights)[:,None]).reshape(n_tokens,-1,pairs.shape[-1]).sum(1)
