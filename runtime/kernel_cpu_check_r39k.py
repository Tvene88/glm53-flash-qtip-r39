#!/usr/bin/env python3
"""
kernel_cpu_check_r39k.py  [R3.9K] CPU-only pre-flight of the QTIP expert kernels v2 (new file of claude_r39_speed)
==================================================================================================================
No GPU, no CUDA context, no driver call. Refuses to run unless CUDA is hidden (CUDA_VISIBLE_DEVICES=-1).

  --nvrtc    Compile every build of the v2 GEMV (load-ahead 0 / 1 / 2 / 4 / 8 / 16 x activations shared / global), the
             fused sparse add, and - as a control - the three v1 kernels to PTX with NVRTC alone, with the options
             torch.cuda._compile_kernel passes (sm_120, torch's include paths, torch's -D flags). Fails on any error
             or diagnostic. Shows: the source compiles. Does not show: that the driver accepts the PTX, that the code
             computes the right values, or how fast it is.

  --emulate  Run glm53_fold_qtip_runtime.run_selftest_kernel() with numpy models in place of the four kernels (each
             model written from that kernel's C source and its own argument list) and with stubs for the
             torch.cuda calls the self-test makes (a "graph replay" does nothing). Shows: every Python path of the
             self-test and of decode_core runs, the wrappers pass their arguments in the order and number the kernel
             signatures have, and the v2 algorithm (one code word per lane + neighbour lanes, gate / up in one
             launch, fused sparse add, masked pairs, a row stride larger than the row, both k = 1 and k = 2 row
             tiles) gives v1's bits IN THE MODEL. Does not show: that the CUDA code behaves like the model on a GPU.
             The timings it prints are meaningless.

  (no flag)  both.

  PowerShell:  $env:CUDA_VISIBLE_DEVICES='-1'; python -I kernel_cpu_check_r39k.py
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes as C
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise SystemExit("kernel_cpu_check_r39k.py is a CPU-only check: set CUDA_VISIBLE_DEVICES to -1 first "
                     "(PowerShell: $env:CUDA_VISIBLE_DEVICES='-1')")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

if torch.cuda.is_available():
    raise SystemExit("CUDA is visible; refusing to run")

import glm53_fold_qtip_runtime as Q  # noqa: E402


# ----------------------------------------------------------------------------------------------------------------
# 1. NVRTC compile only
# ----------------------------------------------------------------------------------------------------------------
def nvrtc_compile(source: str, name: str, arch: str = "sm_120"):
    """(return code, log, PTX bytes, options). NVRTC only: the PTX is never loaded."""
    from torch.utils.cpp_extension import COMMON_NVCC_FLAGS, include_paths

    dll_dir = Path(torch.__file__).resolve().parent / "lib"
    cands = [p for p in dll_dir.glob("nvrtc64_*.dll") if ".alt." not in p.name] or list(dll_dir.glob("libnvrtc.so*"))
    if not cands:
        raise RuntimeError(f"no NVRTC library in {dll_dir}")
    dll = cands[0]
    handle = os.add_dll_directory(str(dll.parent)) if hasattr(os, "add_dll_directory") else None
    _builtins = [C.CDLL(str(p)) for p in dll.parent.glob("nvrtc-builtins*.dll")]   # NVRTC loads it by name later
    lib = C.CDLL(str(dll))
    lib.nvrtcCreateProgram.argtypes = [C.POINTER(C.c_void_p), C.c_char_p, C.c_char_p, C.c_int, C.c_void_p, C.c_void_p]
    lib.nvrtcCompileProgram.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_char_p)]
    lib.nvrtcGetProgramLogSize.argtypes = [C.c_void_p, C.POINTER(C.c_size_t)]
    lib.nvrtcGetProgramLog.argtypes = [C.c_void_p, C.c_void_p]
    lib.nvrtcGetPTXSize.argtypes = [C.c_void_p, C.POINTER(C.c_size_t)]
    lib.nvrtcGetPTX.argtypes = [C.c_void_p, C.c_void_p]
    lib.nvrtcAddNameExpression.argtypes = [C.c_void_p, C.c_char_p]
    lib.nvrtcDestroyProgram.argtypes = [C.POINTER(C.c_void_p)]
    program = C.c_void_p()
    rc = lib.nvrtcCreateProgram(C.byref(program), source.encode(), f"{name}.cu".encode(), 0, None, None)
    if rc:
        raise RuntimeError(f"nvrtcCreateProgram={rc}")
    lib.nvrtcAddNameExpression(program, name.encode())
    opts = [f"--gpu-architecture={arch}"] + [f"-I{p}" for p in include_paths("cuda")]
    opts += [f for f in COMMON_NVCC_FLAGS if f != "--expt-relaxed-constexpr"]
    raw = (C.c_char_p * len(opts))(*(o.encode() for o in opts))
    try:
        rc = lib.nvrtcCompileProgram(program, len(opts), raw)
        size = C.c_size_t()
        lib.nvrtcGetProgramLogSize(program, C.byref(size))
        log = C.create_string_buffer(size.value or 1)
        lib.nvrtcGetProgramLog(program, log)
        ptx = b""
        if rc == 0:
            psz = C.c_size_t()
            lib.nvrtcGetPTXSize(program, C.byref(psz))
            buf = C.create_string_buffer(psz.value)
            lib.nvrtcGetPTX(program, buf)
            ptx = buf.value
        return rc, log.value.decode(errors="replace").strip(), ptx, opts
    finally:
        lib.nvrtcDestroyProgram(C.byref(program))
        if handle is not None:
            handle.close()


def nvrtc_check(keep: str = "") -> int:
    jobs = [Q.qtip_gemv_v2_source(pf, xsh) for pf in (0, 1, 2, 4, 8, 16) for xsh in (True, False)]
    jobs.append(("qtip_r39k_sparse_add2_kernel", Q._R39K_SPARSE2_SRC))
    for n in ("qtip_tile16x16_hyb_slab_gemv_kernel", "qtip_tile16x16_dense_bf16_kernel", "qtip_sparse_add_kernel"):
        jobs.append((n, Q._QTIP_SERVE_CUDA_SRC))                 # control: the v1 kernels through the same path
    bad = 0
    opts = []
    for name, src in jobs:
        rc, log, ptx, opts = nvrtc_compile(src, name)
        text = ptx.decode(errors="replace")
        entry = f".entry {name}(" in text
        note = ""
        if name.startswith("qtip_r39k_gemv_"):
            # what the wrapper relies on: 23 parameters with a 32-bit row stride (the wrapper passes a C int), no
            # stack frame (the load-ahead words stay in registers), 512 threads at most. The fmaf count is a
            # multiple of 8 per tile; the compiler may unroll, so the count itself is only printed.
            n_par = len(set(re.findall(rf"\.param \.u(?:32|64) {name}_param_(\d+)", text)))
            n_fma = text.count("fma.rn.f32")
            ok = (n_par == 23 and f".param .u32 {name}_param_1," in text and "__local_depot" not in text
                  and ".maxntid 512" in text and n_fma >= 16 and n_fma % 8 == 0)
            note = f" params={n_par} fma.rn.f32={n_fma} checks={'ok' if ok else 'FAILED'}"
            if not ok:
                bad += 1
        if keep and ptx:
            Path(keep).mkdir(parents=True, exist_ok=True)
            (Path(keep) / f"{name}.cu").write_text(src, encoding="utf-8")
            (Path(keep) / f"{name}.ptx").write_bytes(ptx)
        print(f"  {name:46s} rc={rc} ptx={len(ptx):7d} B entry={entry}{note} {('LOG: ' + log[:1200]) if log else ''}")
        if rc or log or not entry:
            bad += 1
    print("  options:", " ".join(opts))
    print(f"[NVRTC] {len(jobs)} kernels: " + (f"{bad} PROBLEM(S)" if bad else "all compile without diagnostics"))
    return bad


# ----------------------------------------------------------------------------------------------------------------
# 2. numpy models of the kernels (vectorised over pairs and lanes; one model per kernel, each with that kernel's
#    own argument list) and the emulated self-test
# ----------------------------------------------------------------------------------------------------------------
CALLS = {"v1_gemv": 0, "v2_gemv": 0, "v2_gemv_fused": 0, "v1_sparse": 0, "v2_sparse2": 0}
LANES = np.arange(32)
R_LOCAL, HALF_COL = LANES >> 1, LANES & 1


def flat(t, dt):
    assert t.is_contiguous()
    return t.numpy().reshape(-1).view(dt)


def fma(a, b, c):
    """fmaf modelled in double precision (the same model on both sides of every comparison)"""
    return (a.astype(np.float64) * b.astype(np.float64) + c.astype(np.float64)).astype(np.float32)


def hyb_pair(s16, lut):
    """qtip_hyb_mul1_pair / r39k_hyb_pair: s16 uint32 array -> (x, y) float32 arrays; lut uint32 [512]"""
    h = ((np.uint64(34038481) * s16.astype(np.uint64) + np.uint64(76625530)) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    raw = lut[(h >> np.uint32(17)) & np.uint32(511)]
    sb = raw ^ (h & np.uint32(0x80008000))
    lo = (sb & np.uint32(0xFFFF)).astype(np.uint16).view(np.float16).astype(np.float32)
    hi = (sb >> np.uint32(16)).astype(np.uint16).view(np.float16).astype(np.float32)
    return lo, hi


def acc8(acc, w, xv):
    """acc [n, 32]; w: 4 (x, y) pairs [n, 32]; xv [n, 32, 8]: the eight fmaf of one tile, in the kernels' order"""
    k = 0
    for wx, wy in w:
        acc = fma(wx, xv[:, :, k], acc)
        acc = fma(wy, xv[:, :, k + 1], acc)
        k += 2
    return acc


def gather(R, addr):
    return R[addr].astype(np.uint32)


def epilogue(acc, rt, bases, off_map, off_scale, R, Y2, live):
    """acc += shfl_xor(acc, 1); lanes with half_col == 0 store acc * scale at y[pr, rmap[stored_r]]"""
    tot = (acc + acc[:, LANES ^ 1]).astype(np.float32)[:, 0::2]            # [n, 16], r_local = 0..15
    stored = (rt << 4) + np.arange(16)
    am = bases[:, None] + off_map + 2 * stored[None, :]
    rmap = gather(R, am) | (gather(R, am + 1) << 8)
    asc = bases[:, None] + off_scale + 2 * stored[None, :]
    sc = (gather(R, asc) | (gather(R, asc + 1) << 8)).astype(np.uint16).view(np.float16).astype(np.float32)
    Y2[live[:, None], rmap] = (tot * sc).astype(np.float32)


def emu_gemv_v1(grid, block, args, shared_mem=0):
    """qtip_tile16x16_hyb_slab_gemv_kernel: read_tailbite_state16 as written (three bytes per state, byte index
    modulo the tile size)"""
    rows, row_bytes, slots, off_map, off_scale, off_c1, off_c2, n1, n2, lut, x, xpp, y, d_out, T = args
    assert len(args) == 15
    CALLS["v1_gemv"] += 1
    assert shared_mem == (1024 + T) * 4 and block[0] % 32 == 0
    R, S, L = flat(rows, np.uint8), flat(slots, np.int32), flat(lut, np.uint32)
    X2, Y2 = flat(x, np.float32).reshape(-1, T), flat(y, np.float32).reshape(-1, d_out)
    P = grid[1]
    assert Y2.shape[0] == P
    Y2[np.nonzero(S[:P] < 0)[0]] = 0.0
    live = np.nonzero(S[:P] >= 0)[0]
    if live.size == 0:
        return
    bases = S[live].astype(np.int64) * row_bytes
    XG = X2[live if xpp else np.zeros_like(live)]
    nb_col = T >> 4
    for rt in range(d_out >> 4):
        if rt < (n1 >> 4):
            k, tb = 1, 32
            rtb = bases + off_c1 + rt * nb_col * 32
        else:
            k, tb = 2, 64
            rtb = bases + off_c2 + (rt - (n1 >> 4)) * nb_col * 64
        tk = L[(k - 1) * 512:(k - 1) * 512 + 512]
        step_bits = k << 1
        m_base = R_LOCAL * 8 + HALF_COL * 4
        acc = np.zeros((live.size, 32), np.float32)
        for ct in range(nb_col):
            tp = gather(R, rtb[:, None] + ct * tb + np.arange(tb)[None, :])
            xv = XG.reshape(-1, nb_col, 2, 8)[:, ct][:, HALF_COL]
            w = []
            for j in range(4):
                bit_pos = (m_base + j) * step_bits
                b0 = (bit_pos >> 3) & (tb - 1)
                b1 = (b0 + 1) & (tb - 1)
                b2 = (b0 + 2) & (tb - 1)
                off = (bit_pos & 7).astype(np.uint32)
                w24 = (tp[:, b0] << 16) | (tp[:, b1] << 8) | tp[:, b2]
                w.append(hyb_pair((w24 >> (np.uint32(8) - off)[None, :]) & np.uint32(0xFFFF), tk))
            acc = acc8(acc, w, xv)
        epilogue(acc, rt, bases, off_map, off_scale, R, Y2, live)


def byte_rev(v):
    """__byte_perm(v, 0, 0x0123)"""
    return ((v & 0xFF) << 24) | (((v >> 8) & 0xFF) << 16) | (((v >> 16) & 0xFF) << 8) | ((v >> 24) & 0xFF)


def emu_gemv_v2(grid, block, args, shared_mem=0, x_shared=True):
    """qtip_r39k_gemv_*_kernel: one code word per lane and tile + the neighbour lanes' words (__shfl_sync)"""
    (rows, row_bytes, slots, om_a, os_a, oc1_a, oc2_a, n1_a, om_b, os_b, oc1_b, oc2_b, n1_b,
     lut_a, lut_b, x_a, x_b, xpp, y_a, y_b, d_out, T, P) = args
    assert len(args) == 23
    assert shared_mem == ((1024 + T) * 4 if x_shared else 4096) and block[0] % 32 == 0 and block[0] <= 512
    assert grid[1] in (P, 2 * P)
    CALLS["v2_gemv_fused" if grid[1] == 2 * P else "v2_gemv"] += 1
    R, S = flat(rows, np.uint8), flat(slots, np.int32)
    nx1, nx2 = (LANES + 1) & 31, (LANES + 2) & 31
    nb_col = T >> 4
    for second in range(grid[1] // P):                                   # blockIdx.y >= P: the second projection
        Y2 = flat(y_b if second else y_a, np.float32).reshape(-1, d_out)
        assert Y2.shape[0] == P
        off_map, off_scale, off_c1, off_c2, n1 = ((om_b, os_b, oc1_b, oc2_b, n1_b) if second
                                                  else (om_a, os_a, oc1_a, oc2_a, n1_a))
        L = flat(lut_b if second else lut_a, np.uint32)
        X2 = flat(x_b if second else x_a, np.float32).reshape(-1, T)
        Y2[np.nonzero(S[:P] < 0)[0]] = 0.0
        live = np.nonzero(S[:P] >= 0)[0]
        if live.size == 0:
            continue
        bases = S[live].astype(np.int64) * row_bytes
        XG = X2[live if xpp else np.zeros_like(live)]
        n1_tiles = n1 >> 4
        for rt in range(d_out >> 4):
            acc = np.zeros((live.size, 32), np.float32)
            for ct in range(nb_col):
                xv = XG.reshape(-1, nb_col, 2, 8)[:, ct][:, HALF_COL]
                if rt < n1_tiles:
                    rp = bases + off_c1 + rt * nb_col * 32
                    cw = gather(R, rp[:, None] + ct * 32 + LANES[None, :])              # one byte per lane
                    bw = (cw << 16) | (cw[:, nx1] << 8) | cw[:, nx2]
                    st = [bw >> 8, (bw >> 6) & 0xFFFF, (bw >> 4) & 0xFFFF, (bw >> 2) & 0xFFFF]
                    tk = L[:512]
                else:
                    rp = bases + off_c2 + (rt - n1_tiles) * nb_col * 64
                    assert not (rp % 2).any()
                    a16 = rp[:, None] + ct * 64 + 2 * LANES[None, :]                    # 16-bit little-endian word
                    cw = gather(R, a16) | (gather(R, a16 + 1) << 8)
                    bw = byte_rev(cw | (cw[:, nx1] << 16))
                    st = [bw >> 16, (bw >> 12) & 0xFFFF, (bw >> 8) & 0xFFFF, (bw >> 4) & 0xFFFF]
                    tk = L[512:1024]
                acc = acc8(acc, [hyb_pair(s.astype(np.uint32), tk) for s in st], xv)
            epilogue(acc, rt, bases, off_map, off_scale, R, Y2, live)


def sparse_pair(R, base_sp, K, xp, op):
    """one pair of qtip_sparse_add_kernel: products, then per entry the ordered sum written by its first owner"""
    sp = R[base_sp: base_sp + 6 * K].view(np.uint16)
    s_row = [int(sp[3 * k]) for k in range(K)]
    s_prod = [np.float32(sp[3 * k + 2:3 * k + 3].view(np.float16)[0].astype(np.float32)) * np.float32(xp[int(sp[3 * k + 1])])
              for k in range(K)]
    snap = op.copy()                      # every thread reads op[r] before the single writer of r can matter
    for k in range(K):
        r = s_row[k]
        acc = np.float32(snap[r])
        first = True
        for j in range(K):
            if s_row[j] == r:
                if j < k:
                    first = False
                else:
                    acc = np.float32(acc + s_prod[j])
        if first:
            op[r] = acc


def emu_sparse_v1(grid, block, args, shared_mem=0):
    """qtip_sparse_add_kernel"""
    rows, row_bytes, slots, off_sp, K, x, x_stride, out, out_stride, P = args
    assert len(args) == 10
    CALLS["v1_sparse"] += 1
    assert shared_mem == 8 * K and grid[0] == P
    R, S, X, O = flat(rows, np.uint8), flat(slots, np.int32), flat(x, np.float32), flat(out, np.float32)
    for pr in range(grid[0]):
        sl = int(S[pr])
        if sl < 0:
            continue
        sparse_pair(R, sl * row_bytes + off_sp, K, X[pr * x_stride:], O[pr * out_stride:])


def emu_sparse_v2(grid, block, args, shared_mem=0):
    """qtip_r39k_sparse_add2_kernel"""
    rows, row_bytes, slots, off_a, off_b, K, x, x_stride, out_a, out_b, out_stride, P = args
    assert len(args) == 12
    CALLS["v2_sparse2"] += 1
    assert shared_mem == 8 * K and grid[0] == 2 * P
    R, S, X = flat(rows, np.uint8), flat(slots, np.int32), flat(x, np.float32)
    for bx in range(grid[0]):
        second = 1 if bx >= P else 0
        pr = bx - second * P
        sl = int(S[pr])
        if sl < 0:
            continue
        O = flat(out_b if second else out_a, np.float32)
        sparse_pair(R, sl * row_bytes + (off_b if second else off_a), K, X[pr * x_stride:], O[pr * out_stride:])


def fake_compile(name, src, shared_bytes=0):
    if name == "qtip_tile16x16_hyb_slab_gemv_kernel":
        return emu_gemv_v1
    if name == "qtip_sparse_add_kernel":
        return emu_sparse_v1
    if name == "qtip_r39k_sparse_add2_kernel":
        assert name in src
        return emu_sparse_v2
    if name.startswith("qtip_r39k_gemv_"):
        assert name in src
        xs = name.endswith("_xs_kernel")
        return lambda grid, block, args, shared_mem=0: emu_gemv_v2(grid, block, args, shared_mem, x_shared=xs)
    raise KeyError(name)


class _FakeStream:
    def __init__(self, device=None):
        pass

    def wait_stream(self, other):
        pass


class _FakeGraph:
    def replay(self):
        pass


def _mini_archive(folder: str, layer: int, n_rows: int) -> None:
    """a small layer file in the archive format (16 rows, D = 64, two k = 1 and two k = 2 row tiles, no pruning)"""
    gen = torch.Generator(device="cpu").manual_seed(5)
    fields = {}
    for p in ("g", "u", "d"):
        ent = torch.empty((n_rows, 8, 3), dtype=torch.int16)
        ent[:, :, 0] = torch.randint(0, 64, (n_rows, 8), generator=gen).to(torch.int16)
        ent[:, :, 1] = torch.randint(0, 64, (n_rows, 8), generator=gen).to(torch.int16)
        ent[:, :, 2] = ((torch.rand((n_rows, 8), generator=gen) - 0.5) * 0.3).to(torch.float16).view(torch.int16)
        fields[p] = {
            "map": torch.stack([torch.randperm(64, generator=gen).to(torch.int16) for _ in range(n_rows)]),
            "scale": (torch.rand((n_rows, 64), generator=gen) * 0.4 + 0.1).to(torch.float16),
            "c1": torch.randint(0, 256, (n_rows, 2, 4, 32), generator=gen, dtype=torch.uint8),
            "c2": torch.randint(0, 256, (n_rows, 2, 4, 64), generator=gen, dtype=torch.uint8),
            "sp": ent.view(torch.uint8).reshape(n_rows, 8, 6).clone(),
        }
    blob = {
        "format": "qtip_hyb_r39", "rule": "GU125", "layer_id": layer, "tile_mode": "2d_16x16",
        "signs": {p: (torch.randint(0, 2, (64,), generator=gen, dtype=torch.int8) * 2 - 1) for p in ("g", "u", "d")},
        "hyb_luts": {p: {k: (torch.rand((512, 2), generator=gen) - 0.5).to(torch.float16) for k in (1, 2)}
                     for p in ("g", "u", "d")},
        "fields": fields, "n_experts_total": n_rows,
    }
    torch.save(blob, os.path.join(folder, f"layer_{layer:02d}_qtip_hyb.pt"))


def emulate() -> int:
    saved_env = {k: v for k, v in os.environ.items() if k.startswith("R39_")}
    for k in saved_env:
        del os.environ[k]
    Q._compile_nvrtc = fake_compile
    torch.cuda.synchronize = lambda *a, **k: None
    torch.cuda.Stream = _FakeStream
    torch.cuda.current_stream = lambda *a, **k: _FakeStream()
    torch.cuda.stream = lambda s: contextlib.nullcontext()
    torch.cuda.CUDAGraph = _FakeGraph
    torch.cuda.graph = lambda g, **k: contextlib.nullcontext()
    torch.cuda.empty_cache = lambda: None

    # small stand-ins for the production presets, same structure: rule A all k = 2, GU125 with k = 1 row tiles in
    # gate / up (the real layouts are checked without kernels at the end)
    orig_preset = Q.make_qtip_preset

    def mini_preset(rule, custom_specs=None):
        if custom_specs is not None:
            return orig_preset(rule, custom_specs=custom_specs)
        n1 = 0 if rule == "A" else 16
        g = Q.QtipProjSpec("gate", 32, 64, n1=n1, k_sparse=8, unbiased=True)
        u = Q.QtipProjSpec("up", 32, 64, n1=n1, k_sparse=8, unbiased=True)
        d = Q.QtipProjSpec("down", 64, 32, n1=0, k_sparse=12, unbiased=False)
        return Q.QtipPreset(f"mini_{rule}", g, u, d)

    Q.make_qtip_preset = mini_preset
    cpu = torch.device("cpu")
    fold = tempfile.mkdtemp(prefix="r39k_emu_fold_")
    try:
        _mini_archive(fold, 7, 16)
        print("[EMULATE] run 1: default v2 settings, a small archive folder as --fold, with the sweep", flush=True)
        Q.run_selftest_kernel(fold=fold, calls=3, bench_rows=16, layer=-1, sweep=True, _dev=cpu)
        print("  kernel model calls:", CALLS, flush=True)
        assert all(v > 0 for v in CALLS.values()), CALLS

        print("[EMULATE] run 2: R39_QTIP_WARPS=4 R39_QTIP_PF=8 R39_QTIP_XSRC=g R39_QTIP_FUSE=0", flush=True)
        os.environ.update({"R39_QTIP_KERNEL": "v2", "R39_QTIP_WARPS": "4", "R39_QTIP_PF": "8", "R39_QTIP_XSRC": "g",
                           "R39_QTIP_FUSE": "0"})
        Q.run_selftest_kernel(fold="", calls=2, bench_rows=16, _dev=cpu)

        # the flag as the runtime reads it
        fa = Q._r39k_fake_arch("GU125", 16, 1)
        rt = Q.QtipLayerRuntime(fa, cpu, slab_in_vram=False)
        assert rt.r39k is not None and rt.r39k.describe() == "warps=4 gx=64 pf=8 x=global fuse_gu=0", rt.r39k.describe()
        os.environ["R39_QTIP_KERNEL"] = "v1"
        assert Q.QtipLayerRuntime(fa, cpu, slab_in_vram=False).r39k is None
        del os.environ["R39_QTIP_KERNEL"]
        assert Q.QtipLayerRuntime(fa, cpu, slab_in_vram=False).r39k is None and Q.qtip_kernel_cfg() is None
        os.environ["R39_QTIP_KERNEL"] = "v3"
        try:
            Q.qtip_kernel_cfg()
            raise SystemExit("R39_QTIP_KERNEL=v3 was accepted")
        except ValueError:
            pass
        print("[EMULATE] R39_QTIP_KERNEL: unset -> v1, v1 -> v1, v2 -> v2 with its knobs, v3 -> refused", flush=True)

        # the production layouts through the v2 layout checks and the random-slab builder (no kernels)
        Q.make_qtip_preset = orig_preset
        for k in [k for k in os.environ if k.startswith("R39_")]:
            del os.environ[k]
        os.environ["R39_QTIP_KERNEL"] = "v2"
        for rule, want in (("A", 6326528), ("GU125", 6064384)):
            fa = Q._r39k_fake_arch(rule, 2, 7)
            rb = fa.preset.expert_row_bytes
            assert rb == want == int(fa.expert_slab_u8.shape[1]), (rule, rb)
            rt = Q.QtipLayerRuntime(fa, cpu, slab_in_vram=False)           # runs check_layout on gate / up / down
            for p, sp in (("g", fa.preset.gate), ("u", fa.preset.up), ("d", fa.preset.down)):
                off = rt.lay[p].off
                m = fa.expert_slab_u8[:, off["map"]:off["map"] + 2 * sp.d_out].contiguous().view(torch.int16)
                assert all(sorted(r.tolist()) == list(range(sp.d_out)) for r in m), "row map is not a permutation"
                assert all(o % 2 == 0 for o in off.values())
            print(f"[EMULATE] production layout rule {rule}: {rb:,} B/row, passes the v2 layout checks "
                  f"({rt.r39k.describe()})", flush=True)
    finally:
        Q.make_qtip_preset = orig_preset
        shutil.rmtree(fold, ignore_errors=True)
        for k in [k for k in os.environ if k.startswith("R39_")]:
            del os.environ[k]
        os.environ.update(saved_env)
    print("[EMULATE] complete: v2 == v1 bitwise in the numpy model on every path of --selftest-kernel", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="[R3.9K] CPU-only pre-flight of the QTIP expert kernels v2")
    ap.add_argument("--nvrtc", action="store_true", help="NVRTC compile only")
    ap.add_argument("--emulate", action="store_true", help="the kernel self-test against numpy models of the kernels")
    ap.add_argument("--keep-ptx", default="", help="with --nvrtc: folder for the generated .cu / .ptx files")
    a = ap.parse_args()
    both = not (a.nvrtc or a.emulate)
    bad = 0
    if a.nvrtc or both:
        bad += nvrtc_check(a.keep_ptx)
    if a.emulate or both:
        bad += emulate()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
