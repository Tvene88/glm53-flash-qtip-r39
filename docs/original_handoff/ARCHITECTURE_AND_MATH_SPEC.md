> **ORIGINAL, UNVERIFIED DOCUMENT.** This file is part of the original handoff package (written by an AI model,
> Gemini 4 Carbon). Its numbers are forecasts, several of which were refuted by measurement. The measured results
> are in [`RESULTS.md`](../../RESULTS.md). Apart from this banner the text is unchanged.

# Round 3.8 & Round 3.9 Mathematical & CUDA Kernel Specification

This specification documents the mathematical foundations and CUDA kernel implementations of the Round 3.8 and Round 3.9 upgrades over Round 3.7 `TCQ33P5` on [dealignai/GLM-5.3-Flash-UNCENSORED-FP8](https://huggingface.co/dealignai/GLM-5.3-Flash-UNCENSORED-FP8) ([GLM-5 Technical Report, arXiv:2602.15763](https://arxiv.org/abs/2602.15763)).

## 1. Bijective `HYB-mul1` `L=16, V=2, Q=9` Trellis (`2 KiB` Differentiable Codebooks)

Round 3.3/3.7 `TCQ33` used an `L=12` (`4,096`-state) 1D trellis (`V=1`) with an explicit `24 KiB` (`3 x 4,096` `fp16`) lookup table and free 12-bit start states (`+4 B`/row overhead). Following [QTIP (Tseng et al., Cornell RelaxML, arXiv:2406.11235v4)](https://arxiv.org/html/2406.11235v4) ([Cornell-RelaxML/qtip](https://github.com/Cornell-RelaxML/qtip)) and modular congruential hashing in [cuda-exl3](https://github.com/Zeuss5/cuda-exl3) / [ExLlamaV3](https://github.com/turboderp-org/exllamav3), Round 3.9 upgrades to a `65,536`-state (`L=16`) 2D pair trellis (`V=2`, `128` pairs per `256`-weight `16x16` tile) driven by an odd-multiplier linear congruential bijection over $\mathbb{Z}_{2^{32}}$:

$$h(s) = (34038481 \cdot s + 76625530) \bmod 2^{32}, \qquad s \in \{0, \dots, 65535\}$$

Because $\gcd(34038481, 2^{32}) = 1$, $h(s)$ is strictly bijective across all $65,536$ states (avoiding the ~50% quadratic-residue bucket collisions of $s^2 + s$). Each state maps to a `2 KiB` shared-memory `half2[512]` magnitude table `LUT` ($Q=9$ index bits) plus two independent sign flips extracted from bit 31 of $h(s)$ and bit 15 of the packed `__half2` word via a single bitwise XOR (`0x80008000u`):

```cuda
__device__ __forceinline__ float2 qtip_hyb_mul1_pair(unsigned int s16, const unsigned int* __restrict__ lut_u32) {
    unsigned int h = 34038481u * s16 + 76625530u;
    unsigned int raw = lut_u32[(h >> 17) & 511u];
    unsigned int signed_bits = raw ^ (h & 0x80008000u);
    return __half22float2(*reinterpret_cast<const __half2*>(&signed_bits));
}
```

Wait-free `2 KiB` `half2` LUTs (`4 KiB` total per projection across `k=1` and `k=2`) + `16 KiB` rotated activation `xs` occupy **`20 KiB` shared memory per CTA**, allowing **4 concurrent CTAs (`32` warps) per SM** on Blackwell `sm_120` while remaining **fully differentiable** during Stage-2B Global-KL co-adaptation.

## 2. Register-Resident 2-Pass Collar Tail-Biting Viterbi (`0` Start-State Bits)

For tile length $M = 128$ pairs (`256` weights) and rate $k \in \{1, 2\}$ bpw (`2k` step bits per pair), a circular tail-biting trellis enforces $s_0 = s_M$, eliminating the `16`-bit initial-state header completely:
* `Rule A` (`k=2` everywhere): every `16x16` tile is `64 B` (`1,024 B` per `4,096`-col row vs. `1,028 B` in `TCQ33`), saving **`32,768 B` per expert (`6,326,528 B` vs. `6,359,296 B`)**.
* `Rule GU125` (`n_1 = 256` `k=1` lowest-`hnorm` rows on `gate_proj` and `up_proj`; `down_proj` 100% `k=2`): **`6,064,384 B` per expert** (`1.9173 bpw`).

In `run_qtip_build_r39.py` and `r39_l27_qtip_2d_ldlq_lab.py`, the `65,536`-state Viterbi dynamic program holds path metrics directly in **`32x __half2` hardware registers** across `1,024` threads/CTA (`0 B` global `cost_buf` traffic), subtracting `sm_best` every 8 steps to prevent FP16 overflow and executing a 2-pass `collar = 12` warm-up + pinned-seam pass (`states[:, -1] == seam` verified bit-exact).

## 3. `16x16` `2D-BlockLDLQ` (`g = 16` Cholesky Column Feedback)

Following [QuIP# (arXiv:2402.04396)](https://arxiv.org/abs/2402.04396), [GPTQ (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323), and Section 4.3 of [QTIP (arXiv:2406.11235v4)](https://arxiv.org/html/2406.11235v4), weights are randomized via Rademacher sign flips $S \in \{\pm 1\}^{d_{\text{in}}}$ and orthonormal Kronecker FWHT $H_{d_{\text{in}}}$:

$$\tilde{W} = (W \operatorname{diag}(s)) H_{d_{\text{in}}}, \qquad \tilde{H} = H_{d_{\text{in}}} (\operatorname{diag}(s) H_x \operatorname{diag}(s)) H_{d_{\text{in}}}$$

Whereas Round 3.3 quantized full `4,096`-element 1D rows with block feedback $g = 256$, Round 3.9 factors $\tilde{H} + \lambda \mu I = U_f D_{\text{blk}} U_f^\top$ with block size **$g = 16$** (`16x` tighter column feedback). Output rows are sorted by `hnorm` (placing lowest-sensitivity rows into `n_1` when `GU125` is active) and grouped by row RMS into homogeneous `16`-row bands. Each `[16, 16]` tile is quantized jointly by the `L=16` tail-biting Viterbi kernel weighted by normalized per-row $\text{rms}_r^2$, propagating exact error feedback $(W_{\text{blk}} - \hat{W}_{\text{blk}})(U_{f,\text{col}} - I)$ into unprocessed column blocks.

## 4. Cross-Covariance Quantized-`gate/up` `H_down` Compensation ($W_d^\star$)

Let $z = \operatorname{silu}(\min(X W_g^\top, 10)) \odot \operatorname{clamp}(X W_u^\top, -10, 10)$ denote the exact teacher SwiGLU intermediate activation and $\hat{z}$ denote the realized activation produced by the newly quantized $(\hat{W}_g, \hat{W}_u)$ on layer $l$. Minimizing expected squared expert output error $\mathbb{E}\|z W_d^\top - \hat{z} \hat{W}_d^\top\|_2^2$ yields the cross-covariance target matrix:

$$W_d^\star = W_d + W_d \left(H_{z\hat{z}} - H_{\hat{z}\hat{z}}\right)\left(H_{\hat{z}\hat{z}} + \lambda_{\text{qz}} \mu I\right)^{-1}$$

where $H_{z\hat{z}} = \frac{1}{N}\sum_t w_t z_t^\top \hat{z}_t$ and $H_{\hat{z}\hat{z}} = \frac{1}{N}\sum_t w_t \hat{z}_t^\top \hat{z}_t$. Round 3.3's experimental `down_from_quantized` updated $H_{\hat{z}\hat{z}}$ without applying the cross-covariance drift correction $W_d(H_{z\hat{z}} - H_{\hat{z}\hat{z}})(H_{\hat{z}\hat{z}} + \lambda_{\text{qz}}\mu I)^{-1}$, leaving first-order `gate/up` quantization noise uncompensated. Encoding $W_d^\star$ under $H_{\hat{z}\hat{z}}$ via `2D-BlockLDLQ` strictly improves `down_proj` output SNR.

## 5. Stage-2B Layer-Major Global-KL Co-Adaptation (`1x RTX 5090`)

Inspired by zero-overhead scale retraining in [Tetra (arXiv:2609.35465)](https://arxiv.org/pdf/2609.35465), `r39_stage2b_global_kl_trainer.py` minimizes exact Top-128 + Residual-Tail full-model teacher KL divergence across all 45 trunk layers without unpacking FP16 expert matrices into VRAM and without adding a single byte to deployed checkpoints:

$$\mathcal{L}_{\text{KL}}(p \| q) = \sum_{j=1}^{128} p_{v_j}\left(\log p_{v_j} - \log q_{v_j}\right) + p_{\text{tail}}\left(\log p_{\text{tail}} - \log\!\left(1 - \sum_{j=1}^{128} q_{v_j}\right)\right)$$

1. **Anchored Straight-Through Sigmoid Router (`AnchoredRouterLoRA`)**:
   * $W_{\text{eff}} = W_0 + \frac{\alpha}{r} B A$, $b_{\text{eff}} = b_0 + \Delta b$ (with $-\infty$ pinned on pruned expert IDs).
   * Forward selects exact discrete `top-8` experts (`scores + b_eff`), while straight-through gradients flow through normalized sigmoid gate weights $w_{t,k}$ into $(A, B, \Delta b)$ with Frobenius anchor decay toward $(W_0, b_0)$. At export, $W_{\text{eff}}$ and $b_{\text{eff}}$ overwrite `merged_routers_bf16.pt` (`0 B` runtime cost).
2. **Differentiable `2 KiB` `HYB` Codebooks**:
   * Exact analytical bucket scatter (`scatter_add_`) accumulates $\partial \mathcal{L} / \partial \text{LUT}[b]$ across all active tiles per `(layer, proj, k)`, rounded through `fp16` via STE on forward.
3. **Host-Paged Multiplicative Row Scales (`HostPagedScaleAdamW`)**:
   * Parameterizes $s = \operatorname{fp16}(s_0 \exp(\lambda))$ with `bf16` AdamW moments paged in `~0.5 GiB` DDR5 RAM (`0 B` idle VRAM) and commits updated `fp16` scales straight into `off_scale` in place.
4. **11-Window Layer-Major Schedule (`W = 4` Layers/Window)**:
   * Streams `32 x 2048` macro-batches forward (`no_grad`, `use_kernels=True`), stores 11 `bf16` `[32, 2048, 4, 4096]` boundary checkpoints in `22.0 GiB` DDR5 RAM, computes `grad_h` in 256-token head slices (`< 0.35 GiB` VRAM), and sweeps windows `w = 10 \dots 0` in reverse under `_DifferentiableBackboneScope` (`Peak VRAM ~11.8 GiB`).
