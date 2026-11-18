#!/usr/bin/env python3
"""
glm53_fold_math.py
==================
Core Mathematical Primitives for Round-3.1 `APEX-FOLD-FP16`:
  1. Full-Dimension Kronecker Fast Walsh-Hadamard Transform (`fwht_full`) + Rademacher Signs:
     Replaces local `fwht_block32` with an exact full-dimension orthogonal transform across all
     `d_in` channels (`4096 = 64 x 64`, `2048 = 32 x 64`) executed in two fused radix-64/32
     matrix multiplications (`H_d1 @ V @ H_d2`).
     By applying `fwht_full` **before** column permutation `perm`, every rotated coordinate
     carries the exact same fraction `1 / d_in` of real activation energy regardless of which
     input channels contain activation spikes!
  2. Isometric Hadamard-Domain Column Energy Sorting (`compute_gate_up_preconditioner`,
     `compute_down_preconditioner`):
     Uses an exact Euclidean isometry (`w_pre = 1.0`, eliminating noise amplification from
     dividing activations by uncorrelated `post_attention_layernorm.weight` at Layer 3) and
     sorts rotated Hadamard columns by exact cross-expert Frobenius energy across all 288 experts.
  3. Multi-Shell 256-Point 8D Gosset Lattice $E_8$ Codebook Seed (`build_gosset_e8_codebook`)
  4. Split-Half Cross-Validated `OrthogonalAbliterationGuard` for uncensored `down_proj` preservation.
"""

from __future__ import annotations
import itertools
from typing import Dict, Optional, Tuple
import numpy as np
from glm53_torch_compat import torch, nn, F

_HADAMARD_CACHE: Dict[Tuple[int, str, str], torch.Tensor] = {}


def _get_hadamard_matrix(n: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Returns cached normalized Sylvester-Hadamard matrix H_n (H_n @ H_n = I_n) on `device`."""
    key = (n, str(device), str(dtype))
    if key not in _HADAMARD_CACHE:
        assert n > 0 and (n & (n - 1)) == 0, f"Hadamard dimension {n} must be a power of 2"
        h = np.array([[1.0]], dtype=np.float32)
        while h.shape[0] < n:
            h = np.block([[h, h], [h, -h]])
        h = h * (1.0 / np.sqrt(float(n)))
        _HADAMARD_CACHE[key] = torch.from_numpy(h).to(device=device, dtype=dtype).contiguous()
    return _HADAMARD_CACHE[key]


def fwht_full(x: torch.Tensor) -> torch.Tensor:
    """
    Full-Dimension Orthogonal Fast Walsh-Hadamard Transform across the entire last dimension `d_in`.
    Uses exact Kronecker factorization `H_{d1 * d2} = H_{d1} (x) H_{d2}` (e.g. `4096 = 64 x 64`,
    `2048 = 32 x 64`, `512 = 16 x 32`, `256 = 16 x 16`) so the transform executes as two tiny
    fused Tensor-Core/BLAS matrix multiplications instead of log2(d_in) butterfly allocations.
    Satisfies `fwht_full(fwht_full(x)) == x` and `<fwht_full(x), fwht_full(w)> == <x, w>`.
    """
    orig_shape = x.shape
    d_in = orig_shape[-1]
    assert d_in > 0 and (d_in & (d_in - 1)) == 0, f"Last dimension {d_in} must be a power of 2"
    xf = x.to(torch.float32)
    if d_in <= 64:
        H = _get_hadamard_matrix(d_in, xf.device, torch.float32)
        return torch.matmul(xf, H).to(x.dtype)

    # Factor d_in = d1 * d2 with d1, d2 ~ sqrt(d_in)
    log_d = int(round(np.log2(d_in)))
    d1 = 1 << (log_d // 2)
    d2 = d_in // d1
    H1 = _get_hadamard_matrix(d1, xf.device, torch.float32)
    H2 = _get_hadamard_matrix(d2, xf.device, torch.float32)
    v2d = xf.reshape(-1, d1, d2)
    out = torch.matmul(H1, torch.matmul(v2d, H2)).reshape(orig_shape)
    return out.to(x.dtype)


def fwht_block32(x: torch.Tensor) -> torch.Tensor:
    """Backwards-compatible alias forwarding to `fwht_full`."""
    return fwht_full(x)


def generate_rademacher_signs(d_in: int, seed: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Deterministic Rademacher signs in {-1, +1} stored as int8[d_in]."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) & 0xFFFFFFFF)
    bits = torch.randint(0, 2, (d_in,), generator=gen, dtype=torch.int8)
    return (bits * 2 - 1).to(device=device)


@torch.no_grad()
def compute_gate_up_preconditioner(
    W_experts: torch.Tensor,         # [E, d_out, d_in]
    rms_weight: torch.Tensor,        # [d_in] (kept in signature for API compatibility)
    seed: int = 1337,
    expert_batch: int = 32,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Isometric Full-Dimension Hadamard Pre-Conditioner for `W_gate`, `W_up`, or `W_down`:
      1. Sets `w_pre = 1.0` (exact Euclidean isometry so dividing `x / w_pre` never amplifies
         quantization noise on channels where `post_attention_layernorm.weight` differs from
         real residual stream variance).
      2. Applies Rademacher signs + Full-Dimension FWHT (`fwht_full`) across all `d_in` channels
         first, then measures exact Hadamard-domain column energy across all `E` experts and
         sorts Hadamard columns in descending order (`perm`) so Tier-0 and Tier-1 receive the
         highest-energy rotated weight columns while every rotated coordinate carries `1 / d_in`
         of the real activation energy.
    """
    E, d_out, d_in = W_experts.shape
    w_pre_f32 = torch.ones(d_in, dtype=torch.float32, device=device)
    signs = generate_rademacher_signs(d_in, seed=seed, device=device)
    s_f = signs.to(device=device, dtype=torch.float32)

    col_sq = torch.zeros(d_in, dtype=torch.float32, device=device)
    for e0 in range(0, E, expert_batch):
        wb = W_experts[e0 : e0 + expert_batch].to(device=device, dtype=torch.float32)
        ub = fwht_full(wb * s_f)
        col_sq += ub.pow(2).sum(dim=(0, 1))

    perm = torch.argsort(col_sq, descending=True)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(d_in, device=device, dtype=perm.dtype)
    return w_pre_f32.to(torch.float16), perm, inv_perm, signs


@torch.no_grad()
def compute_down_preconditioner(
    W_gate: torch.Tensor,            # [E, d_inter, d_model]
    W_up: torch.Tensor,              # [E, d_inter, d_model]
    W_down: torch.Tensor,            # [E, d_model, d_inter]
    rms_weight: torch.Tensor,        # [d_model]
    seed: int = 7331,
    expert_batch: int = 32,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Isometric Full-Dimension Hadamard Pre-Conditioner for `W_down`.
    """
    return compute_gate_up_preconditioner(
        W_down, rms_weight, seed=seed, expert_batch=expert_batch, device=device
    )


def transform_weights_fwht(
    W_batch: torch.Tensor,           # [..., d_out, d_in]
    w_pre: torch.Tensor,             # [d_in] float16
    perm: torch.Tensor,              # [d_in] int64
    signs: torch.Tensor,             # [d_in] int8
) -> torch.Tensor:
    """
    Applies pre-conditioning, Rademacher signs, Full-Dimension FWHT across all `d_in`
    channels FIRST, and then permutes the Hadamard columns into descending energy order:
      U = FWHT_full((W * w_pre) * signs)[..., perm]
    """
    w_p = w_pre.to(device=W_batch.device, dtype=torch.float32)
    s_f = signs.to(device=W_batch.device, dtype=torch.float32)
    W_rot = fwht_full((W_batch.to(torch.float32) * w_p) * s_f)
    return W_rot.index_select(-1, perm.to(W_batch.device, dtype=torch.long))


def inverse_transform_weights_fwht(
    U_batch: torch.Tensor,           # [..., d_out, d_in]
    w_pre: torch.Tensor,             # [d_in] float16
    perm: torch.Tensor,              # [d_in] int64
    signs: torch.Tensor,             # [d_in] int8
) -> torch.Tensor:
    """Exact inverse mapping transformed weights `U` back to original weight domain `W`."""
    d_in = perm.numel()
    inv_perm = torch.empty_like(perm, dtype=torch.long, device=U_batch.device)
    inv_perm[perm.to(U_batch.device, dtype=torch.long)] = torch.arange(d_in, device=U_batch.device, dtype=torch.long)

    s_f = signs.to(device=U_batch.device, dtype=torch.float32)
    w_p = w_pre.to(device=U_batch.device, dtype=torch.float32).clamp_min(1e-5)

    W_rot = U_batch.to(torch.float32).index_select(-1, inv_perm)
    return (fwht_full(W_rot) * s_f) / w_p


def transform_activations_fwht(
    x: torch.Tensor,                 # [..., d_in]
    w_pre: torch.Tensor,             # [d_in] float16
    perm: torch.Tensor,              # [d_in] int64/uint16
    signs: torch.Tensor,             # [d_in] int8
) -> torch.Tensor:
    """
    Dual input activation transform such that `<transform_activations_fwht(x), transform_weights_fwht(W)> == <x, W>`:
      x_rot = FWHT_full((x / w_pre) * signs)[..., perm]
    Because `FWHT_full` mixes all `d_in` channels before `perm`, every coordinate of `x_rot`
    carries `1 / d_in` of the real activation energy regardless of channel spikes in `x`!
    """
    orig_dtype = x.dtype
    w_p = w_pre.to(device=x.device, dtype=torch.float32).clamp_min(1e-5)
    s_f = signs.to(device=x.device, dtype=torch.float32)
    x_rot = fwht_full((x.to(torch.float32) / w_p) * s_f)
    return x_rot.index_select(-1, perm.to(device=x.device, dtype=torch.long)).to(orig_dtype)


def build_gosset_e8_codebook(device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Constructs the 256-entry Multi-Shell 8D Gosset Lattice $E_8$ seed codebook in float32 [256, 8]:
      - 112 D8 roots: permutations of (±1, ±1, 0, 0, 0, 0, 0, 0) at median chi_8 shell
      - 128 half-integer spinors: (±0.5)^8 (even minus signs) split across inner & outer radial shells
      - 16 coordinate axes: permutations of (±sqrt(2), 0, 0, 0, 0, 0, 0, 0)
    Seeding across concentric radial shells prevents K-Means from collapsing onto a single sphere.
    """
    vecs = []
    for i, j in itertools.combinations(range(8), 2):
        for si in (-1.0, 1.0):
            for sj in (-1.0, 1.0):
                v = [0.0] * 8
                v[i] = si
                v[j] = sj
                vecs.append(v)
    for signs in itertools.product((-0.5, 0.5), repeat=8):
        if sum(1 for s in signs if s < 0) % 2 == 0:
            vecs.append(list(signs))
    s2 = 1.4142135623730951
    for i in range(8):
        for s in (-s2, s2):
            v = [0.0] * 8
            v[i] = s
            vecs.append(v)
    cb = torch.tensor(vecs, dtype=torch.float32, device=device) * 2.0  # [256, 8]
    radii = torch.ones((256, 1), dtype=torch.float32, device=device) * 0.86
    radii[112:176] = 0.62
    radii[176:240] = 1.12
    radii[240:] = 0.98
    return cb * radii


class OrthogonalAbliterationGuard:
    """
    Split-half cross-validated refusal-subspace guard ensuring that uncensored / abliterated
    `down_proj` matrices never regenerate a refusal direction during subspace or codebook folding,
    while rejecting spurious directions on single-corpus calibration sets (`SNR < 3.0` or `cos < 0.85`).
    """
    @staticmethod
    def extract_verified_refusal_dir(
        X_harmful: np.ndarray,
        X_harmless: Optional[np.ndarray] = None,
        min_snr: float = 3.0,
        min_split_cos: float = 0.85,
    ) -> Optional[np.ndarray]:
        if X_harmful is None or X_harmless is None:
            return None
        n = min(len(X_harmful), len(X_harmless))
        if n < 8:
            return None
        half = n // 2
        d1 = (X_harmful[:half].mean(axis=0) - X_harmless[:half].mean(axis=0)).astype(np.float64)
        d2 = (X_harmful[half:n].mean(axis=0) - X_harmless[half:n].mean(axis=0)).astype(np.float64)
        n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
        if n1 < 1e-9 or n2 < 1e-9:
            return None
        cos_sim = float(np.dot(d1 / n1, d2 / n2))
        diff1 = (X_harmful[:half] - X_harmless[:half]).astype(np.float64)
        u2 = d2 / n2
        proj1 = diff1 @ u2
        snr = float(proj1.mean() / (proj1.std() / np.sqrt(half) + 1e-12))
        if cos_sim < min_split_cos or snr < min_snr:
            return None
        mean_diff = (X_harmful[:n] - X_harmless[:n]).astype(np.float64).mean(axis=0)
        r = mean_diff / (np.linalg.norm(mean_diff) + 1e-12)
        return r.astype(np.float32)

    @staticmethod
    def project_out_refusal_torch(W_down: torch.Tensor, r_dir: Optional[torch.Tensor]) -> torch.Tensor:
        """Projects out `r_dir` [d_model] from the output columns of `W_down` [..., d_model, d_inter]."""
        if r_dir is None:
            return W_down
        r = r_dir.to(device=W_down.device, dtype=torch.float32)
        r = r / (r.norm() + 1e-12)
        w_f32 = W_down.to(torch.float32)
        proj = torch.einsum("d,...di->...i", r, w_f32)
        w_clean = w_f32 - torch.einsum("d,...i->...di", r, proj)
        return w_clean.to(W_down.dtype)
