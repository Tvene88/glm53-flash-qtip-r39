"""R3.6 opt-in backbone kernels. Importing this module does not import torch.

FP8 conversion preserves e4m3fn values; split-K is separately selectable and
ROUNDING. mHC parallelises the 24 dots over SMs, then uses the original Sinkhorn
and collapse. KDA is ROUNDING: row-local, sequential recurrent state updates.
All CUDA entry points are explicit; CPU references use synthetic NumPy arrays.
"""
from __future__ import annotations
import os
import numpy as np

FP8_SRC = r'''
__device__ __forceinline__ float e4m3(unsigned char b) {
    unsigned int m = b & 7, e = (b >> 3) & 15, s = (unsigned int)(b & 128) << 24;
    if (e == 15 && m == 7) return 0.f; // original LUT nan_to_num(0)
    if (e == 0) return __uint_as_float(s) + (b & 128 ? -1.f : 1.f) * (float)m * 0.001953125f;
    return __uint_as_float(s | ((e + 120) << 23) | (m << 20));
}
extern "C" __global__ void fp8_v2_r36_kernel(
    const unsigned char* W, const float* sinv, const float* x, float* y,
    int O, int I, int N, int sc_cols, int bs, int row_scales, int splits)
{
    int lane=threadIdx.x&31, warp=threadIdx.x>>5;
    int o=blockIdx.x*(blockDim.x>>5)+warp, part=blockIdx.y;
    if(o>=O) return;
    const unsigned char* row=W+(size_t)o*I;
    const float* sr=sinv+(size_t)(row_scales ? o : o/bs)*sc_cols;
    float a0=0.f, a1=0.f, a2=0.f, a3=0.f;
    for(int c=part*512+lane*16;c<I;c+=512*splits) {
        uint4 pack=*(const uint4*)(row+c);
        const unsigned char* b=(const unsigned char*)(&pack);
        float p0=0.f,p1=0.f,p2=0.f,p3=0.f;
        #pragma unroll
        for(int j=0;j<16;++j) {
            float w=e4m3(b[j]);
            p0+=w*x[c+j];
            if(N>1) p1+=w*x[I+c+j];
            if(N>2) p2+=w*x[2*I+c+j];
            if(N>3) p3+=w*x[3*I+c+j];
        }
        float s=sr[c/bs];
        a0+=p0*s; a1+=p1*s; a2+=p2*s; a3+=p3*s;
    }
    #pragma unroll
    for(int d=16;d>0;d>>=1) {
        a0+=__shfl_down_sync(0xffffffff,a0,d); a1+=__shfl_down_sync(0xffffffff,a1,d);
        a2+=__shfl_down_sync(0xffffffff,a2,d); a3+=__shfl_down_sync(0xffffffff,a3,d);
    }
    if(lane==0) {
        y[(size_t)part*N*O+o]=a0;
        if(N>1) y[(size_t)part*N*O+O+o]=a1;
        if(N>2) y[(size_t)part*N*O+2*O+o]=a2;
        if(N>3) y[(size_t)part*N*O+3*O+o]=a3;
    }
}
'''

MHC_DOT_SRC = r'''
extern "C" __global__ void mhc_dot_r36_kernel(const float* h, const float* fn, float* dots, int n) {
    int j=blockIdx.x, t=blockIdx.y, lane=threadIdx.x&31, warp=threadIdx.x>>5;
    __shared__ float partial[16];
    float a=0.f;
    // Same 512-thread striding and reduction as R3.5, independently for each dot.
    for(int i=threadIdx.x;i<n;i+=512) {
        float v=h[(size_t)t*n+i];
        a += (j==24 ? v : fn[(size_t)j*n+i])*v;
    }
    #pragma unroll
    for(int d=16;d>0;d>>=1) a+=__shfl_down_sync(0xffffffff,a,d);
    if(lane==0) partial[warp]=a;
    __syncthreads();
    if(threadIdx.x==0) {float total=0.f;for(int w=0;w<16;++w)total+=partial[w];dots[t*25+j]=total;}
}
'''


def mhc_finish_source(original):
    old = 'const float* __restrict__ scale, float* __restrict__ post_out'
    new = 'const float* __restrict__ scale, const float* __restrict__ dots, float* __restrict__ post_out'
    if original.count(old) != 1:
        raise RuntimeError('mHC signature changed')
    source = original.replace('mhc_prepare_kernel(', 'mhc_finish_r36_kernel(').replace(old, new)
    start = source.index('    float acc[NMIX];')
    stop = source.index('    if (threadIdx.x == 0) {', start)
    return source[:start] + '''    if(threadIdx.x < NMIX) s_mix[threadIdx.x] = dots[t*25+threadIdx.x];
    if(threadIdx.x == NMIX) s_rms = rsqrtf(dots[t*25+NMIX]/(float)n+eps);
    __syncthreads();
''' + source[stop:]


CONV_SRC = r'''
extern "C" __global__ void kda_conv_r36_kernel(const float* x, const float* w, float* state,
    float* rollback, float* out, int C, int K, int N) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=C)return;
    for(int t=0;t<N;++t){
        float v=0.f;
        for(int k=0;k<K-1;++k)v=__fadd_rn(v,__fmul_rn(state[k*C+c],w[c*K+k]));
        v=__fadd_rn(v,__fmul_rn(x[t*C+c],w[c*K+K-1]));
        out[t*C+c]=v/(1.f+expf(-v));
        for(int k=0;k<K-2;++k)state[k*C+c]=state[(k+1)*C+c];
        state[(K-2)*C+c]=x[t*C+c];
        if(N==2 && t==0)for(int k=0;k<K-1;++k)rollback[k*C+c]=state[k*C+c];
    }
}
'''

KDA_PRE_SRC = r'''
extern "C" __global__ void kda_pre_r36_kernel(
 const float* q,const float* k,const float* rawg,const float* rawb,const float* A,
 const float* dt,float* qn,float* kn,float* decay,float* beta,int H,int D,double lower_d) {
    const float lower=(float)lower_d; // Python scalar ABI is double
    int h=blockIdx.x,t=blockIdx.y,d=threadIdx.x;
    __shared__ float sq[256],sk[256];
    size_t i=((size_t)t*H+h)*D+d;
    float vq=d<D?q[i]:0.f,vk=d<D?k[i]:0.f;
    sq[d]=vq*vq;sk[d]=vk*vk;__syncthreads();
    for(int off=blockDim.x/2;off>0;off>>=1){if(d<off){sq[d]+=sq[d+off];sk[d]+=sk[d+off];}__syncthreads();}
    if(d<D){
        qn[i]=(vq/sqrtf(sq[0]+1.e-6f))*rsqrtf((float)D);
        kn[i]=vk/sqrtf(sk[0]+1.e-6f);
        float g=__fadd_rn(rawg[i],dt[h*D+d]);
        decay[i]=expf(lower/(1.f+expf(-A[h]*g)));
    }
    if(d==0)beta[t*H+h]=1.f/(1.f+expf(-rawb[t*H+h]));
}
'''

KDA_REC_SRC = r'''
extern "C" __global__ void kda_recur_r36_kernel(
 const float* q,const float* k,const float* v,const float* decay,const float* beta,
 float* state,float* rollback,float* out,int H,int D,int N) {
    int h=blockIdx.x,j=blockIdx.y*32+threadIdx.x;
    if(j>=D)return;
    float col[256];
    for(int t=0;t<N;++t){
        size_t a=((size_t)t*H+h)*D;
        float mem=0.f;
        for(int i=0;i<D;++i){
            col[i]=__fmul_rn(state[((size_t)h*D+i)*D+j],decay[a+i]);
            mem+=k[a+i]*col[i];
        }
        float delta=__fmul_rn(__fsub_rn(v[a+j],mem),beta[t*H+h]);
        float result=0.f;
        for(int i=0;i<D;++i){
            float s=__fadd_rn(col[i],__fmul_rn(k[a+i],delta));
            state[((size_t)h*D+i)*D+j]=s;
            if(N==2 && t==0)rollback[((size_t)h*D+i)*D+j]=s;
            result+=q[a+i]*s;
        }
        out[a+j]=result;
    }
}
'''

KDA_OUT_SRC = r'''
extern "C" __global__ void kda_out_r36_kernel(const float* x,const float* gate,const float* weight,
 float* y,int H,int D,double eps_d){
 const float eps=(float)eps_d;
 int h=blockIdx.x,t=blockIdx.y,d=threadIdx.x;
 __shared__ float sq[256];size_t i=((size_t)t*H+h)*D+d;
 float v=d<D?x[i]:0.f;sq[d]=v*v;__syncthreads();
 for(int off=blockDim.x/2;off>0;off>>=1){if(d<off)sq[d]+=sq[d+off];__syncthreads();}
 if(d<D){float n=__fmul_rn(v,rsqrtf(sq[0]/(float)D+eps));
 y[i]=__fmul_rn(__fmul_rn(n,weight[d]),1.f/(1.f+expf(-gate[i])));}
}
'''


def compile_sources(literal):
    return [('fp8_v2', FP8_SRC), ('mhc_dot', MHC_DOT_SRC),
            ('mhc_finish', mhc_finish_source(literal('_MHC_PREPARE_SRC'))),
            ('kda_conv', CONV_SRC), ('kda_pre', KDA_PRE_SRC),
            ('kda_recur', KDA_REC_SRC), ('kda_out', KDA_OUT_SRC)]


def fp8_gemv(weight, scales, x, block=128, *, row_scales=False):
    import torch
    import glm53_fold_kernels as KR
    O, I = weight.shape
    N = x.shape[0]
    if N not in (1, 2, 3, 4) or I % 128 or block != 128:
        raise ValueError('R36 FP8 needs N<=4, block=128 and aligned input width')
    split = int(os.environ.get('R36_FP8_SPLIT_K', '1')) if O < 256 else 1
    if split not in (1, 2, 4, 8):
        raise ValueError('R36_FP8_SPLIT_K must be 1,2,4,8')
    out = torch.empty((split, N, O), device=x.device, dtype=torch.float32)
    kernel = KR._compile('fp8_v2_r36_kernel', FP8_SRC)
    kernel(grid=((O+3)//4, split, 1), block=(128, 1, 1), args=[
        weight.view(torch.uint8), scales.float().contiguous(), x.float().contiguous(), out,
        O, I, N, scales.shape[1], block, int(row_scales), split])
    return out[0] if split == 1 else out.sum(0)


def mhc_prepare(h, fn, base, scale, eps, hc_eps, iters):
    import torch
    import glm53_fold_kernels as KR
    lead, D = h.shape[:-2], h.shape[-1]
    hf = h.reshape(-1, 4, D).float().contiguous()
    T = hf.shape[0]
    dots = torch.empty((T, 25), device=h.device, dtype=torch.float32)
    post, comb, xin = [torch.empty(s, device=h.device, dtype=torch.float32) for s in ((T,4),(T,4,4),(T,D))]
    dot = KR._compile('mhc_dot_r36_kernel', MHC_DOT_SRC)
    dot(grid=(25,T,1), block=(512,1,1), args=[hf, fn.float().contiguous(), dots, 4*D])
    finish = KR._compile('mhc_finish_r36_kernel', mhc_finish_source(KR._MHC_PREPARE_SRC))
    finish(grid=(T,1,1), block=(512,1,1), args=[hf,fn.float().contiguous(),base.float().contiguous(),
        scale.float().contiguous(),dots,post,comb,xin,D,float(eps),float(hc_eps),int(iters)])
    return post.view(*lead,4), comb.view(*lead,4,4), xin.view(*lead,D)


def conv(x, weight, state, rollback=None):
    import torch
    import glm53_fold_kernels as KR
    N,C=x.shape[1:]
    K=state.shape[1]+1
    if x.shape[0]!=1 or N not in (1,2) or K<2:
        raise ValueError('R36 conv supports B=1, N=1/2 and K>=2')
    if N==2 and rollback is None:
        rollback=torch.empty_like(state)
    out=torch.empty_like(x,dtype=torch.float32)
    kernel=KR._compile('kda_conv_r36_kernel',CONV_SRC)
    kernel(grid=((C+255)//256,1,1),block=(256,1,1),args=[x.float().contiguous(),weight.float().contiguous(),
        state,rollback if rollback is not None else state,out,C,K,N])
    return out


def kda_recurrent(q,k,v,rawg,rawb,A,dt,state,lower,rollback=None):
    import torch
    import glm53_fold_kernels as KR
    B,N,H,D=q.shape
    if B!=1 or N not in (1,2) or D>256 or D & (D-1):
        raise ValueError('R36 recurrent requires B=1,N=1/2,power-of-two hd<=256')
    qn,kn,decay,out=[torch.empty_like(q) for _ in range(4)]
    beta=torch.empty((N,H),device=q.device,dtype=torch.float32)
    pre=KR._compile('kda_pre_r36_kernel',KDA_PRE_SRC)
    pre(grid=(H,N,1),block=(D,1,1),args=[q,k,rawg.float().contiguous(),rawb.float().contiguous(),A.float().contiguous(),dt.float().contiguous(),
        qn,kn,decay,beta,H,D,float(lower)])
    if N==2 and rollback is None: rollback=torch.empty_like(state)
    rec=KR._compile('kda_recur_r36_kernel',KDA_REC_SRC)
    rec(grid=(H,(D+31)//32,1),block=(32,1,1),args=[qn,kn,v,decay,beta,state,
        rollback if rollback is not None else state,out,H,D,N])
    return out


def kda_output(x,gate,weight,eps):
    import torch
    import glm53_fold_kernels as KR
    _,N,H,D=x.shape
    y=torch.empty_like(x)
    kernel=KR._compile('kda_out_r36_kernel',KDA_OUT_SRC)
    kernel(grid=(H,N,1),block=(D,1,1),args=[x,gate.contiguous(),weight.float().contiguous(),y,H,D,float(eps)])
    return y


def kda_graph(dec,l,x,pair=False):
    """N1 and N2 share identical row-local conv/norm/recurrence kernels.

    N2 saves state and convolution histories after row0 for R35 rollback.
    Projection batching keeps the existing R35 _lin2 row-local policy.
    """
    bb,w,st=dec.bb,dec.bb.layer(l),dec.kda[l]
    lin=dec._lin2 if pair else bb.lin
    N,H,D=(2 if pair else 1),bb.H_kda,bb.hd_kda
    oldconv=dec.kda_c0[l] if pair else (None,None,None)
    q,k,v=[conv(lin(x,w[name]),w[name+'conv'],st['c'+name],oldconv[j]).view(1,N,H,D)
           for j,name in enumerate(('q','k','v'))]
    rawg=lin(lin(x,w['fa']),w['fb']).view(1,N,H,D)
    out=kda_recurrent(q,k,v,rawg,lin(x,w['b']),dec.A_exp[l],w['dt_bias'],st['state'],bb.lower_bound,
                      dec.kda_s0[l] if pair else None)
    gate=lin(lin(x,w['ga']),w['gb']).view(1,N,H,D)
    o=kda_output(out,gate,w['o_norm'],bb.eps)
    return lin(o.reshape(1,N,H*D),w['o'])


def kda_eager(bb,l,x,cache):
    """Same fused recurrent pieces for short eager calls; preserve cache ownership."""
    import torch
    w=bb.layer(l);_,N,_=x.shape;H,D=bb.H_kda,bb.hd_kda
    cs=cache.kda_conv.get(l) if cache is not None else None
    states=[s.clone() for s in cs] if cs is not None else [
        torch.zeros((1,w[name+'conv'].shape[-1]-1,H*D),device=x.device) for name in ('q','k','v')]
    q,k,v=[conv(bb.lin(x,w[name]),w[name+'conv'],states[j]).view(1,N,H,D) for j,name in enumerate(('q','k','v'))]
    state=cache.kda_state.get(l) if cache is not None else None
    state=state.clone() if state is not None else torch.zeros((1,H,D,D),device=x.device)
    out=kda_recurrent(q,k,v,bb.lin(bb.lin(x,w['fa']),w['fb']).view(1,N,H,D),bb.lin(x,w['b']),
                      torch.exp(w['A_log']).contiguous(),w['dt_bias'],state,bb.lower_bound)
    if cache is not None:cache.kda_state[l]=state;cache.kda_conv[l]=tuple(states)
    gate=bb.lin(bb.lin(x,w['ga']),w['gb']).view(1,N,H,D)
    return bb.lin(kda_output(out,gate,w['o_norm'],bb.eps).reshape(1,N,H*D),w['o'])


def e4m3_reference(b):
    b=np.asarray(b,np.uint8)
    e=(b>>3)&15;m=b&7
    mag=np.where(e==0,m.astype(np.float32)*np.float32(2**-9),
                 (1+m.astype(np.float32)/8)*np.exp2(e.astype(np.float32)-7))
    return np.where((e==15)&(m==7),0,np.copysign(mag,np.where(b&128,-1,1))).astype(np.float32)


def kda_reference(q,k,v,decay,beta,state):
    """[N,H,D] normalised q/k, decay exp(g); pure tensor-equation oracle."""
    state=np.asarray(state,np.float32).copy();outputs=[];snapshots=[]
    for t in range(len(q)):
        state=state*decay[t,...,None]
        mem=np.einsum('hi,hij->hj',k[t],state)
        delta=(v[t]-mem)*beta[t,:,None]
        state=state+k[t,...,None]*delta[:,None,:]
        outputs.append(np.einsum('hi,hij->hj',q[t],state));snapshots.append(state.copy())
    return np.stack(outputs),state,snapshots


def kda_column_reference(q,k,v,decay,beta,state):
    state=np.asarray(state,np.float32).copy();N,H,D=q.shape
    out=np.empty_like(q);snapshots=[]
    for t in range(N):
        for h in range(H):
            for j in range(D):
                col=state[h,:,j]*decay[t,h]
                mem=np.float32(0)
                for i in range(D):mem=np.float32(mem+np.float32(k[t,h,i]*col[i]))
                delta=np.float32((v[t,h,j]-mem)*beta[t,h])
                col=col+k[t,h]*delta;state[h,:,j]=col
                acc=np.float32(0)
                for i in range(D):acc=np.float32(acc+np.float32(q[t,h,i]*col[i]))
                out[t,h,j]=acc
        snapshots.append(state.copy())
    return out,state,snapshots
