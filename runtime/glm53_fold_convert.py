#!/usr/bin/env python3
"""
glm53_fold_convert.py
=====================
Streaming, Resumable, Data-Free GPU Weight-Folding Converter for GLM-5.3-Flash
(`APEX-FOLD-FP16` v3.1, Layers 3..45).

Includes:
1. Full-Dimension Kronecker FWHT (`fwht_full`) + 100% Channel Coverage (`c2 = d_in`, `r_shared = 0`)
2. Post-Quantization Variance-Restituted Least-Squares Scale Refit (`S_blk`)
3. Full 288-Expert Weight SNR Audit (batched in chunks of 32 experts) + SwiGLU Routed Output SNR & NormRatio
"""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import time
from typing import Dict, List, Set, Tuple
import numpy as np
from glm53_torch_compat import torch, nn, F

from glm53_fold_backbone import (
    LM,
    RealGLM53Checkpoint,
    human_bytes,
)
from glm53_fold_config import (
    APEX_FOLD_PRESETS,
    GLM53FlashModelConfig,
    compute_exact_hardware_budget,
)
from glm53_fold_encoder import ApexFoldDiskStore, ApexFoldEncoder
from glm53_fold_runtime import ApexFoldLayerRuntime


def snr_db(ref: torch.Tensor, est: torch.Tensor) -> float:
    r = ref.to(torch.float32)
    e = est.to(torch.float32)
    sig = float(torch.sum(r * r).item())
    err = float(torch.sum((r - e) ** 2).item())
    return float(10.0 * np.log10((sig + 1e-30) / (err + 1e-30)))


def get_layer_expert_shards(weight_map: Dict[str, str], layer_id: int) -> List[str]:
    prefix = f"{LM}layers.{layer_id}.mlp.experts."
    shards: Set[str] = set()
    for k, sh in weight_map.items():
        if k.startswith(prefix):
            shards.add(sh)
    return sorted(shards)


def download_shard_if_missing(repo_id: str, shard_name: str, dst_dir: str) -> str:
    dst_path = os.path.join(dst_dir, shard_name)
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 1024 * 1024:
        return dst_path
    os.makedirs(dst_dir, exist_ok=True)
    url = f"https://huggingface.co/{repo_id}/resolve/main/{shard_name}"
    print(f"[dl] fetching {shard_name} from {repo_id} ...", flush=True)
    t0 = time.perf_counter()
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id=repo_id, filename=shard_name, local_dir=dst_dir)
    except Exception:
        subprocess.run(["curl", "-sL", "--retry", "5", "-C", "-", "-o", dst_path, url], check=True)
    dt = time.perf_counter() - t0
    sz = os.path.getsize(dst_path)
    print(f"[dl] {shard_name} ({human_bytes(sz)}) in {dt:.1f}s ({sz / max(dt, 1e-3) / 2**20:.1f} MiB/s)", flush=True)
    return dst_path


def parse_layers_arg(arg: str, cfg: GLM53FlashModelConfig) -> List[int]:
    if arg.strip().lower() in ("all", "*"):
        return cfg.moe_layer_ids
    out: List[int] = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            for l in range(int(a), int(b) + 1):
                if cfg.is_moe_layer(l):
                    out.append(l)
        else:
            l = int(part)
            if cfg.is_moe_layer(l):
                out.append(l)
    return sorted(set(out))


def evaluate_layer_on_activations(
    layer_id: int,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    gate_w: torch.Tensor,
    gate_b: torch.Tensor,
    sh_w1: torch.Tensor,
    sh_w3: torch.Tensor,
    sh_w2: torch.Tensor,
    rms_w: torch.Tensor,
    arch,
    preset,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluates the folded layer against the exact FP32/BF16 experts across all 288 experts
    (in batches of 32) and on two independent activation batches (`X_a`, `X_b`) to verify:
      1. `W_gate`, `W_up`, `W_down` weight SNR across all 288 experts
      2. Routed-expert output SNR & NormRatio (`||y_est|| / ||y_ref||`)
      3. Full MoE layer output SNR (including shared expert)
      4. Path-A Zero-Transpose Pre-Tabulated LUT-GEMV single-token decode latency (`ms/layer`)
    """
    cfg = GLM53FlashModelConfig()
    d_model = cfg.hidden_size
    lim = cfg.swiglu_limit

    rt = ApexFoldLayerRuntime(arch, preset, device=device, slab_in_vram=True)
    E = arch.n_experts
    g_sig, g_err = 0.0, 0.0
    u_sig, u_err = 0.0, 0.0
    d_sig, d_err = 0.0, 0.0

    for e0 in range(0, E, 32):
        e1 = min(E, e0 + 32)
        eb_idx = torch.arange(e0, e1, device=device, dtype=torch.long)
        views = rt._fetch_and_unpack(eb_idx)
        Wg_rec = ApexFoldEncoder.reconstruct_projection_weights(rt.gate_hdr, views, "g", device=device)
        Wu_rec = ApexFoldEncoder.reconstruct_projection_weights(rt.up_hdr, views, "u", device=device)
        Wd_rec = ApexFoldEncoder.reconstruct_projection_weights(rt.down_hdr, views, "d", device=device)

        wg_ref = W_gate[e0:e1].to(device, torch.float32)
        wu_ref = W_up[e0:e1].to(device, torch.float32)
        wd_ref = W_down[e0:e1].to(device, torch.float32)

        g_sig += float(torch.sum(wg_ref * wg_ref).item())
        g_err += float(torch.sum((wg_ref - Wg_rec) ** 2).item())
        u_sig += float(torch.sum(wu_ref * wu_ref).item())
        u_err += float(torch.sum((wu_ref - Wu_rec) ** 2).item())
        d_sig += float(torch.sum(wd_ref * wd_ref).item())
        d_err += float(torch.sum((wd_ref - Wd_rec) ** 2).item())

    wg_snr = float(10.0 * np.log10((g_sig + 1e-30) / (g_err + 1e-30)))
    wu_snr = float(10.0 * np.log10((u_sig + 1e-30) / (u_err + 1e-30)))
    wd_snr = float(10.0 * np.log10((d_sig + 1e-30) / (d_err + 1e-30)))

    gen = torch.Generator(device=device)
    gen.manual_seed(9000 + layer_id)
    raw_a = torch.randn((128, d_model), generator=gen, device=device, dtype=torch.float32)
    raw_b = torch.randn((128, d_model), generator=gen, device=device, dtype=torch.float32)
    w_rms = rms_w.to(device=device, dtype=torch.float32)
    X_a = raw_a * torch.rsqrt(raw_a.pow(2).mean(-1, keepdim=True) + 1e-5) * w_rms
    X_b = raw_b * torch.rsqrt(raw_b.pow(2).mean(-1, keepdim=True) + 1e-5) * w_rms

    def _route_and_ref(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = torch.matmul(X, gate_w.to(device, torch.float32).T)
        scores = torch.sigmoid(logits)
        choices = scores + gate_b.to(device, torch.float32).unsqueeze(0)
        _, top_i = torch.topk(choices, k=cfg.num_experts_per_tok, dim=-1, sorted=False)
        top_w = torch.gather(scores, 1, top_i)
        top_w = (top_w / (top_w.sum(dim=-1, keepdim=True) + 1e-20)) * cfg.routed_scaling_factor

        y_ref = torch.zeros_like(X)
        for e in torch.unique(top_i).tolist():
            tok_idx, slot_idx = torch.where(top_i == e)
            xe = X[tok_idx]
            we = top_w[tok_idx, slot_idx].unsqueeze(-1)
            wg = W_gate[e].to(device, torch.float32)
            wu = W_up[e].to(device, torch.float32)
            wd = W_down[e].to(device, torch.float32)
            ge = torch.matmul(xe, wg.T).clamp(max=lim)
            ue = torch.matmul(xe, wu.T).clamp(min=-lim, max=lim)
            ze = torch.nn.functional.silu(ge) * ue
            de = torch.matmul(ze, wd.T)
            y_ref.index_add_(0, tok_idx, de * we)

        sg = torch.matmul(X, sh_w1.to(device, torch.float32).T).clamp(max=lim)
        su = torch.matmul(X, sh_w3.to(device, torch.float32).T).clamp(min=-lim, max=lim)
        y_sh = torch.matmul(torch.nn.functional.silu(sg) * su, sh_w2.to(device, torch.float32).T)
        return top_w, top_i, y_ref, y_sh

    w_a, i_a, y_ref_a, y_sh_a = _route_and_ref(X_a)
    w_b, i_b, y_ref_b, y_sh_b = _route_and_ref(X_b)

    y_est_a = rt.forward_routed_experts(X_a, w_a, i_a, swiglu_limit=lim)
    y_est_b = rt.forward_routed_experts(X_b, w_b, i_b, swiglu_limit=lim)

    snr_routed_a = snr_db(y_ref_a, y_est_a)
    snr_routed_b = snr_db(y_ref_b, y_est_b)
    snr_full_b = snr_db(y_ref_b + y_sh_b, y_est_b + y_sh_b)
    norm_ratio = float((torch.linalg.norm(y_est_b) / (torch.linalg.norm(y_ref_b) + 1e-12)).item())

    x1 = X_b[:1]
    w1, i1 = w_b[:1], i_b[:1]
    for _ in range(3):
        _ = rt.forward_routed_experts(x1, w1, i1, swiglu_limit=lim)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_rep = 20
    for _ in range(n_rep):
        _ = rt.forward_routed_experts(x1, w1, i1, swiglu_limit=lim)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms_decode = (time.perf_counter() - t0) / n_rep * 1000.0

    return {
        "wg_snr_db": round(wg_snr, 2),
        "wu_snr_db": round(wu_snr, 2),
        "wd_snr_db": round(wd_snr, 2),
        "routed_snr_setA_db": round(snr_routed_a, 2),
        "routed_snr_heldout_db": round(snr_routed_b, 2),
        "overfit_gap_db": round(snr_routed_a - snr_routed_b, 2),
        "full_layer_snr_db": round(snr_full_b, 2),
        "norm_ratio": round(norm_ratio, 4),
        "decode_ms_vram": round(ms_decode, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Round-3.1 APEX-FOLD-FP16 Streaming GPU Converter")
    ap.add_argument("--src", default=r"C:\models\GLM-5.3-Flash-UNCENSORED-FP8", help="Directory for raw FP8 expert shards")
    ap.add_argument("--backbone-dir", default=r"C:\models\GLM53-backbone", help="Extracted non-expert backbone directory")
    ap.add_argument("--dst", default=r"C:\models\GLM53-APEX-FOLD-FP16", help="Output directory for folded .npz archives")
    ap.add_argument("--preset", default="high_fidelity", choices=list(APEX_FOLD_PRESETS.keys()))
    ap.add_argument("--layers", default="all", help="'all', single layer '3', or range '3-10'")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--expert-batch", type=int, default=32)
    ap.add_argument("--beam-width", type=int, default=4)
    ap.add_argument("--delete-used-shards", action="store_true", help="Delete expert shards once no remaining layer needs them")
    ap.add_argument("--keep-shards", default="00001-of-00062.safetensors,00002-of-00062.safetensors,00062-of-00062.safetensors")
    args = ap.parse_args()

    cfg = GLM53FlashModelConfig()
    preset = APEX_FOLD_PRESETS[args.preset]
    dev = torch.device(args.device)
    os.makedirs(args.dst, exist_ok=True)

    budget = compute_exact_hardware_budget(args.preset)
    print("======================================================================================")
    print(f"  GLM-5.3-FLASH ROUND-3.1 APEX-FOLD-FP16 CONVERTER ({args.preset})")
    print(f"  Target Bitrate : {budget['bpw']:.4f} bpw (HARD BYTE-LOCKED: {budget['layer_mib']:.2f} MiB / layer)")
    print(f"  43-Layer Pool  : {budget['total_43_layer_archive_gib']:.2f} GiB | VRAM Layers: {budget['vram_layers_count']}/43")
    print("======================================================================================", flush=True)

    idx_path = os.path.join(args.src, "model.safetensors.index.json")
    if not os.path.exists(idx_path):
        os.makedirs(args.src, exist_ok=True)
        download_shard_if_missing(cfg.hf_repo_fp8, "model.safetensors.index.json", args.src)
    with open(idx_path, "r", encoding="utf-8") as f:
        master_weight_map: Dict[str, str] = json.load(f)["weight_map"]

    bb_dir = args.backbone_dir if os.path.exists(args.backbone_dir) else args.src
    bb_ckpt = RealGLM53Checkpoint(bb_dir)

    layers_todo = parse_layers_arg(args.layers, cfg)
    progress_path = os.path.join(args.dst, "convert_progress.json")
    manifest_path = os.path.join(args.dst, "fold_manifest.json")

    progress: Dict[str, object] = (
        json.load(open(progress_path, "r", encoding="utf-8"))
        if os.path.exists(progress_path)
        else {"preset": args.preset, "layers": {}}
    )

    keep_set = {f"model-{s.strip()}" if not s.strip().startswith("model-") else s.strip() for s in args.keep_shards.split(",") if s.strip()}

    for idx_l, l in enumerate(layers_todo):
        out_npz = os.path.join(args.dst, f"layer_{l:02d}_apex_fold.npz")
        if os.path.exists(out_npz) and str(l) in progress["layers"]:
            print(f"[skip] Layer {l:02d} already folded at {out_npz}", flush=True)
            continue

        t_layer0 = time.perf_counter()
        needed_shards = get_layer_expert_shards(master_weight_map, l)
        for sh_name in needed_shards:
            download_shard_if_missing(cfg.hf_repo_fp8, sh_name, args.src)

        src_ckpt = RealGLM53Checkpoint(args.src)
        p_prefix = f"{LM}layers.{l}."
        rms_source = bb_ckpt if bb_ckpt.has(p_prefix + "post_attention_layernorm.weight") else src_ckpt
        rms_w = torch.from_numpy(rms_source.get_numpy_f32(p_prefix + "post_attention_layernorm.weight"))

        t_deq0 = time.perf_counter()
        W_gate, W_up, W_down = src_ckpt.load_all_experts_torch(l, n_experts=cfg.n_routed_experts, dtype=torch.bfloat16)
        t_deq = time.perf_counter() - t_deq0

        t_enc0 = time.perf_counter()
        arch = ApexFoldEncoder.encode_moe_layer(
            layer_id=l,
            W_gate=W_gate,
            W_up=W_up,
            W_down=W_down,
            rms_weight=rms_w,
            preset=preset,
            expert_batch=args.expert_batch,
            beam_width=args.beam_width,
            device=dev,
        )
        t_enc = time.perf_counter() - t_enc0

        t_sav0 = time.perf_counter()
        file_bytes = ApexFoldDiskStore.save_layer(out_npz, arch)
        t_sav = time.perf_counter() - t_sav0

        gate_w = torch.from_numpy(rms_source.get_numpy_f32(p_prefix + "mlp.gate.weight"))
        gate_b = torch.from_numpy(rms_source.get_numpy_f32(p_prefix + "mlp.gate.e_score_correction_bias"))
        sh_w1 = torch.from_numpy(rms_source.get_numpy_f32(p_prefix + "mlp.shared_experts.gate_proj.weight"))
        sh_w3 = torch.from_numpy(rms_source.get_numpy_f32(p_prefix + "mlp.shared_experts.up_proj.weight"))
        sh_w2 = torch.from_numpy(rms_source.get_numpy_f32(p_prefix + "mlp.shared_experts.down_proj.weight"))

        metrics = evaluate_layer_on_activations(
            l, W_gate, W_up, W_down, gate_w, gate_b, sh_w1, sh_w3, sh_w2, rms_w, arch, preset, dev
        )

        del W_gate, W_up, W_down
        src_ckpt.close()
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        if args.delete_used_shards:
            future_shards: Set[str] = set()
            for fut_l in layers_todo[idx_l + 1 :]:
                future_shards.update(get_layer_expert_shards(master_weight_map, fut_l))
            for sh_name in needed_shards:
                if sh_name not in future_shards and sh_name not in keep_set:
                    sh_path = os.path.join(args.src, sh_name)
                    if os.path.exists(sh_path):
                        os.remove(sh_path)
                        print(f"[rm] removed completed shard {sh_name}", flush=True)

        rec = {
            "layer": l,
            "preset": args.preset,
            "archive_path": out_npz,
            "archive_bytes": file_bytes,
            "exact_memory_bytes": arch.total_bytes,
            "bpw": round(preset.bpw(cfg.n_routed_experts), 4),
            "dequant_s": round(t_deq, 2),
            "encode_s": round(t_enc, 2),
            "save_s": round(t_sav, 2),
            "total_s": round(time.perf_counter() - t_layer0, 2),
            **metrics,
        }
        progress["layers"][str(l)] = rec
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)

        manifest = {
            "format": "APEX-FOLD-FP16-v3.1",
            "preset": args.preset,
            "budget": budget,
            "layers": progress["layers"],
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(
            f"[L{l:02d}] {rec['bpw']:.3f} bpw ({human_bytes(arch.total_bytes)}) | "
            f"Wg {metrics['wg_snr_db']:.2f} dB, Wu {metrics['wu_snr_db']:.2f} dB, Wd {metrics['wd_snr_db']:.2f} dB | "
            f"Routed SNR: {metrics['routed_snr_heldout_db']:.2f} dB (NormRatio: {metrics['norm_ratio']:.3f}) | "
            f"Full SNR: {metrics['full_layer_snr_db']:.2f} dB | Decode: {metrics['decode_ms_vram']:.2f} ms | "
            f"Encode: {t_enc:.1f}s",
            flush=True,
        )

    bb_ckpt.close()
    print(f"[done] Converted {len(progress['layers'])} layers into {args.dst}", flush=True)


if __name__ == "__main__":
    main()
