> **ORIGINAL, UNVERIFIED DOCUMENT.** This file is part of the original handoff package (written by an AI model,
> Gemini 4 Carbon). Its numbers are forecasts, several of which were refuted by measurement. The measured results
> are in [`RESULTS.md`](../../RESULTS.md). Apart from this banner the text is unchanged.

# Binary Slab Layouts, File Schemas & Runtime API Contracts

All Round 3.8 and Round 3.9 tools obey strict binary and dictionary schemas verified by `tests/verify_r39_package.py`.

## 1. Byte-Locked Expert Slab Layouts (`expert_slab_u8` `[E_kept, row_bytes]`)

Every kept expert $r \in \{0, \dots, E_{\text{kept}}-1\}$ (stored in descending calibration popularity order `perm[r]`) occupies one contiguous, **16-byte-aligned** `uint8` row packing `(gate, up, down)` fields `(map, scale, c1, c2, sp)`:

### 1.1 `Rule A` (`qtip_hyb_A_2d16`: `2.0112 bpw`, `6,326,528 B` / expert)

| Field | Projection | Shape & Dtype | Byte Offset (`off`) | Byte Length (`nb`) |
| :--- | :--- | :--- | ---: | ---: |
| `g_map` | `gate_proj` (`2048 x 4096`) | `int16[2048]` | `0` | `4,096` |
| `g_scale` | `gate_proj` | `float16[2048]` | `4,096` | `4,096` |
| `g_c1` | `gate_proj` (`n1=0`) | `uint8[0, 256, 32]` | `8,192` | `0` |
| `g_c2` | `gate_proj` (`n2=2048`, `128x256` tiles) | `uint8[128, 256, 64]` | `8,192` | `2,097,152` |
| `g_sp` | `gate_proj` (`k_sparse=64`) | `uint8[64, 6]` (`r:i16, c:i16, v:f16`) | `2,105,344` | `384` |
| `u_map` | `up_proj` (`2048 x 4096`) | `int16[2048]` | `2,105,728` | `4,096` |
| `u_scale` | `up_proj` | `float16[2048]` | `2,109,824` | `4,096` |
| `u_c1` | `up_proj` (`n1=0`) | `uint8[0, 256, 32]` | `2,113,920` | `0` |
| `u_c2` | `up_proj` (`n2=2048`, `128x256` tiles) | `uint8[128, 256, 64]` | `2,113,920` | `2,097,152` |
| `u_sp` | `up_proj` (`k_sparse=64`) | `uint8[64, 6]` | `4,211,072` | `384` |
| `d_map` | `down_proj` (`4096 x 2048`) | `int16[4096]` | `4,211,456` | `8,192` |
| `d_scale` | `down_proj` | `float16[4096]` | `4,219,648` | `8,192` |
| `d_c1` | `down_proj` (`n1=0`) | `uint8[0, 128, 32]` | `4,227,840` | `0` |
| `d_c2` | `down_proj` (`n2=4096`, `256x128` tiles) | `uint8[256, 128, 64]` | `4,227,840` | `2,097,152` |
| `d_sp` | `down_proj` (`k_sparse=256`) | `uint8[256, 6]` | `6,324,992` | `1,536` |
| **Total** | **`6,326,528 % 16 == 0`** | — | — | **`6,326,528 B`** |

### 1.2 `Rule GU125` (`qtip_hyb_GU125_2d16`: `1.9173 bpw`, `6,064,384 B` / expert)

* Sets `n1 = 256` (`16` row-tiles at `k=1`, `32 B`/tile -> `131,072 B`) and `n2 = 1792` (`112` row-tiles at `k=2`, `64 B`/tile -> `1,835,008 B`) on `gate_proj` and `up_proj`, saving `131,072 B` per projection (`262,144 B`/expert total), while keeping `down_proj` 100% at `k=2` (`n1 = 0, n2 = 4096`).
* Total row size = `6,326,528 - 262,144` = **`6,064,384 B`** (`6,064,384 % 16 == 0`).
* MTP Layer 45 always uses `Rule A` (`6,326,528 B`). `QtipPinnedSlotPool` sizes VRAM slots to `max_row_bytes` and passes `rows.stride(0)` into all NVRTC kernels so `GU125` trunk layers and `Rule A` Layer 45 share one pool seamlessly.

## 2. File & Checkpoint Schemas

### 2.1 `layer_XX_qtip_hyb.pt` (+ Automatic `.slab` Sidecar)
```python
{
    "format": "qtip_hyb_r39",
    "rule": "A" | "GU125",
    "layer_id": int,                     # 3..44 trunk MoE, 45 MTP
    "tile_mode": "2d_16x16",
    "signs": {"g": int8[4096], "u": int8[4096], "d": int8[2048]},
    "hyb_luts": {
        "g": {1: float16[512, 2], 2: float16[512, 2]},
        "u": {1: float16[512, 2], 2: float16[512, 2]},
        "d": {1: float16[512, 2], 2: float16[512, 2]},
    },
    "fields": {
        p: {
            "map": int16[E_kept, d_out],
            "scale": float16[E_kept, d_out],
            "c1": uint8[E_kept, n1 // 16, d_in // 16, 32],
            "c2": uint8[E_kept, n2 // 16, d_in // 16, 64],
            "sp": uint8[E_kept, k_sparse, 6],
        } for p in ("g", "u", "d")
    },
    "perm": int16[E_kept],               # global expert ID stored at slab row r (`0..287`)
    "n_experts_total": 288,
    "meta": dict,
}
```
When `QtipArchive(pt_path, mmap_slab=True)` opens `layer_XX_qtip_hyb.pt`, it materializes an uncompressed `layer_XX_qtip_hyb.pt.slab` sidecar (`E_kept * row_bytes` bytes, regenerated automatically if `.pt` mtime is newer) so subsequent boots `mmap` all 42 slabs in `< 0.3s`.

### 2.2 `merged_routers_bf16.pt`
* Keys:
  * `"model.language_model.layers.{l}.mlp.gate.weight"`: `bfloat16[288, 4096]` (`W_0 + (alpha/r) B @ A`).
  * `"model.language_model.layers.{l}.mlp.gate.e_score_correction_bias"`: `float32[288]` (`b_0 + delta_b` on kept experts, `-inf` on pruned experts).
* Loaded automatically from `os.path.join(fold_dir, "merged_routers_bf16.pt")` by `QtipMoEModelRunner` and `quality_eval_r39.py` BEFORE `GraphDecoder.capture()`.

### 2.3 `train_cal_top128.pt` (`weights_only=True` Pure Tensor Dict)
```python
{
    "format": "r39_compact_teacher_top128_v1",
    "top_k": 128,
    "seq_len": 2048,
    "train_ids": int64[N_train, 2048],
    "qb_ids": int64[29, 2048],
    "train": {"top_ids": int32[N_train, 2048, 128], "top_logp": float16[N_train, 2048, 128], "tail_prob": float32[N_train, 2048]},
    "qbench": {"top_ids": int32[29, 2048, 128], "top_logp": float16[29, 2048, 128], "tail_prob": float32[29, 2048]},
}
```

### 2.4 Prune Mask JSON (`configs/masks/*.json`)
```json
{
  "layers": {
    "3": [12, 41, ...],
    "...": "...",
    "44": [7, 19, ...]
  },
  "meta": {
    "format_target": "qtip_A",
    "served_bytes_with_r34h": 74661048116,
    "served_gb_decimal": 74.661,
    "pruned_total": 1922,
    "kept_total": 10174
  }
}
```

## 3. Runtime Interface Contracts (`GraphDecoder` & `ServeEngine`)

1. **`QtipPinnedSlotPool.ensure(l, experts, out=None)`**:
   * Signature includes `out=None` (verified via `inspect.signature` in `GraphDecoder.__init__`).
   * Writes resolved `int32` slot IDs directly into `out[:len(slots)]` (`GraphDecoder.slots_np[l]`) and issues `cur_stream.wait_event(ev)` on any in-flight `copy_stream` DMA events.
2. **`QtipLayerRuntime.decode_core(x, rows, pair_slots, w_pairs, n_tok, lim, row_exact=False)`**:
   * `n_tok == 1`: captured inside `GraphDecoder` `b{l}` and MTP `mb`.
   * `n_tok == 2, row_exact=True`: captured inside `GraphDecoder` `_seg_b2` (`N=2` MTP verification), running per-row FWHT so row 0 is bit-identical to an `N=1` decode step.
3. **`QtipMoEModelRunner.SERVE_HOOKS = ("state", "skip_prefill")`**:
   * Implements `H1` (`serve_state.begin`), `H2` (`serve_state.after_prefill`), `H3` (`serve_state.on_token`), `H4` (`serve_state.after_decode`), and `H5` (`skip_prefill` when `n_pre >= len(prompt_ids)`), enabling full KV state reuse and `apex_fold.stable_messages` prefix anchoring in `glm53_fold_serve.py`.
