"""Fixed-capacity DSA caches with the existing K-pool indexer inside the graph.

ROUNDING vs eager: selected positions are gathered in increasing token order;
the softmax/readout see <= index_topk + P - 1 slots, with masked padding. KDA
state remains recurrent. MTP long-context is explicitly refused until gated.
"""
from __future__ import annotations
import os
import numpy as np


def capacity(index_topk):
    maximum=int(os.environ.get('R36_LONGCTX_MAX','16384'))
    # [R3.8] 524,288 with the coded history (R38_KV_BITS); the fp32 buffers of Round 3.7 stay limited to 131,072
    top = 524288 if os.environ.get('R38_KV_BITS', '0') not in ('', '0') else 131072
    if maximum < index_topk or maximum > top:
        raise ValueError(f'R36_LONGCTX_MAX must be between index_topk and {top}')
    return maximum


def cache_bytes(tokens,dsa_layers,kv_rank,index_dim):
    return int(tokens)*int(dsa_layers)*(int(kv_rank)+2*int(index_dim))*4


def prefill_chunk():
    size=int(os.environ.get('R36_LONGCTX_PREFILL_CHUNK','128'))
    if size<1 or size>512:raise ValueError('R36_LONGCTX_PREFILL_CHUNK must be 1..512')
    return size


def forward_chunked(bb,ids,cache,moe_fn,last_only,includes_shared,max_layers):
    import torch
    hidden=[];logits=[];size=prefill_chunk()
    for start in range(0,ids.shape[1],size):
        lg,h=bb.forward(ids[:,start:start+size],cache,moe_fn,last_only,includes_shared,max_layers)
        hidden.append(h)
        if not last_only:logits.append(lg)
    return (lg if last_only else torch.cat(logits,1)),torch.cat(hidden,1)


def validate_cache(cache,layers):
    if getattr(cache,'indexer_complete',True) is False:
        raise ValueError('R36 long context cannot import incomplete indexer keys; prefill the full history again')
    for l in layers:
        for name in ('dsa_kv','dsa_ki','dsa_gs'):
            value=getattr(cache,name,{}).get(l)
            if value is None or value.shape[1] < int(cache.pos):
                raise ValueError(f'R36 long context needs complete {name} for layer {l}')


def selection_reference(pool_scores,valid_tokens,pool_size,topk,always_tail=True):
    """Independent CPU oracle for complete-pool top-k and unpooled causal tail.

    Ties use lower pool index for reproducible CPU fixtures; production uses
    torch.topk exactly as the eager indexer does. Tie equivalence is a GPU gate.
    """
    if valid_tokens < 1 or pool_size < 1:
        raise ValueError('positive prefix and pool size required')
    if valid_tokens <= topk:
        return np.arange(valid_tokens,dtype=np.int64)
    complete=valid_tokens//pool_size
    if len(pool_scores)<complete:
        raise ValueError('missing complete-pool scores')
    order=np.argsort(-np.asarray(pool_scores)[:complete],kind='stable')[:topk//pool_size]
    tokens=(order[:,None]*pool_size+np.arange(pool_size)).reshape(-1)
    if always_tail:tokens=np.concatenate([tokens,np.arange(complete*pool_size,valid_tokens)])
    return np.sort(tokens)


def selected_latents(bb,l,x,q_resid,st,pos,arange):
    import torch
    cfg=bb.cfg
    # This is the owning eager indexer, not a second interpretation of K-pools.
    mask=bb._indexer_mask(l,x,q_resid,st['ki'],st['gs'],pos)
    causal=(arange<=pos).view(1,1,-1)
    if bb.dsa_skip_indexer_short:
        mask=torch.where((pos+1<=cfg.index_topk).view(1,1,1),causal,mask)
    T=st['kv'].shape[1]
    n=min(T,cfg.index_topk+cfg.index_kpool-1)
    # Sentinel T gathers the last cache row, masked out. This avoids copying the
    # entire latent cache to append a zero row. Finite-min masks make probability
    # zero when at least one real key is selected (always_select_tail required).
    candidates=torch.where(mask[0,0],arange,torch.full_like(arange,T))
    ids=torch.topk(candidates,n,largest=False,sorted=True).values
    return st['kv'].index_select(1,ids.clamp_max(T-1)), (ids<T).view(1,1,1,n)
