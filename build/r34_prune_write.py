#!/usr/bin/env python3
r"""r34_prune_write.py - Round 3.4 shrink, design step B8: write a PRUNED expert archive into a NEW directory (source rows of
the kept experts only), then verify it row by row against the source. CPU + disk only, no GPU. The source directory is
only read (never modified, never deleted).

USAGE (PY = the Python of the environment; PYTHONUTF8=1)
  PY -u r34_prune_write.py --src C:\models\GLM53-APEX-FOLD-TCQ33 --dst C:\models\GLM53-APEX-FOLD-TCQ33P5 \
        --spec out_r32\r34_masks\loomean_U5.json --tag P5            # write + immediate per-layer check
  PY -u r34_prune_write.py ... --resume                               # continue after a crash (same src / spec)
  PY -u r34_prune_write.py ... --verify-only                          # full re-verification from the files (exit 0 = PASS)

PRUNED LAYER FILE = the build's own writer glm53_fold_tcq_runtime.save_layer (np.savez: every member ZIP_STORED, same member
order and dtypes as the source): preset_rule, layer_id, perm int16 [R] (slab row r holds expert perm[r]), signs_g/u/d and
tables byte-copied, meta = source meta + {"n_experts_total": 288, "pruned": {...}, "source": {...}, "row_order": ...},
expert_slab_u8 [R, 6,359,296] = the source rows of the kept experts in the SOURCE ROW ORDER (popularity order) with the pruned
rows removed (stable). The runtime (R3.4-S2..S4) maps experts through the partial perm (row_of = -1 for missing experts) and
masks the missing experts in the router by itself, so the archive is served without a --prune-spec.
Trunk layers without pruned experts and the MTP layer 45 (never pruned) are HARD LINKS to the source files, as are the
exact prefix caches and the super_experts\*.pt files (new directory entries on the same volume, no new disk space).
fold_manifest.json = the source manifest + "pruned" + "files" (rows / bytes / sha256 or hardlink_of), written last.

REFUSED before anything is written: a spec that load_prune_spec (the runtime's validator) rejects; spec layers outside the
trunk (the MTP layer 45 is never pruned); a spec expert that has a super_experts\L<l>_E<e>.pt file; dst not a new directory
named GLM53-APEX-FOLD-TCQ33P* directly under C:\models (or under --test-root, unit test only); dst == src; an existing dst
without --resume (a --resume needs the writer's own state file with the same src and spec sha256); free disk below the new
bytes + --min-free-gb (re-checked before every layer). A source file that is not a full 288-row file is refused.
CHECK of every written file (immediately after writing it, and again by --verify-only): zip members and order equal to the
source, every member ZIP_STORED; small members array- and dtype-equal to the source; perm == source perm minus the spec
(stable); missing experts == the spec; meta n_experts_total / pruned record; TcqArchive (the runtime loader, byte lock and
perm checks) on the mmap; EVERY kept row (6,359,296 B) byte-equal to its source row; sha256 of the file. --verify-only also
checks the hard links (same file), the manifest (files, sizes, sha256) and loads one file fully into RAM (mmap_slab=False).
Reports: out_r32\r34_prune_write_<tag>.{json,log}, out_r32\r34_prune_verify_<tag>.json.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import time
import zipfile

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out_r32")
RUNTIME = os.environ.get("R34_RUNTIME") or os.path.join(os.path.dirname(HERE), "runtime")
sys.path.insert(0, RUNTIME)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import glm53_fold_tcq_runtime as TR  # noqa: E402
from glm53_fold_config import GLM53FlashModelConfig  # noqa: E402
from glm53_fold_encoder import ApexFoldDiskStore  # noqa: E402
from glm53_fold_generate import load_prune_spec  # noqa: E402

MODELS = r"C:\models"
NAME_RE = re.compile(r"GLM53-APEX-FOLD-TCQ33P\w+$")
STATE = "_prune_write_state.json"
CHUNK_ROWS = 32
LOG_FH = [None]


def log(msg):
    line = f"[r34-pw {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG_FH[0] is not None:
        LOG_FH[0].write(line + "\n")
        LOG_FH[0].flush()


def sha256_file(path, block=64 << 20):
    h = hashlib.sha256()
    buf = memoryview(bytearray(block))
    with open(path, "rb", buffering=0) as f:
        while True:
            k = f.readinto(buf)
            if not k:
                break
            h.update(buf[:k])
    return h.hexdigest()


def layer_name(l):
    return f"layer_{l:02d}_apex_fold.npz"


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def runs(rows):
    """consecutive source-row runs -> [(a, b, i)]: source rows [a, b) go to destination rows [i, i + b - a)."""
    out, i = [], 0
    for r in rows:
        if out and out[-1][1] == r:
            out[-1][1] = r + 1
        else:
            out.append([r, r + 1, i])
        i += 1
    return [tuple(x) for x in out]


def super_experts(src):
    s = set()
    for p in glob.glob(os.path.join(src, "super_experts", "L*_E*.pt")):
        m = re.match(r"L(\d+)_E(\d+)\.pt$", os.path.basename(p))
        if m:
            s.add((int(m.group(1)), int(m.group(2))))
    return s


# ------------------------------------------------------------------------------------------------- checks
def check_dst_name(args):
    src, dst = os.path.abspath(args.src), os.path.abspath(args.dst)
    if os.path.normcase(src) == os.path.normcase(dst):
        raise SystemExit("REFUSED: dst == src")
    if args.test_root:
        root = os.path.abspath(args.test_root)
        if os.path.normcase(os.path.dirname(dst)) != os.path.normcase(root):
            raise SystemExit(f"REFUSED: dst must be directly under the test root {root}")
    else:
        if os.path.normcase(os.path.dirname(dst)) != os.path.normcase(MODELS) or not NAME_RE.match(os.path.basename(dst)):
            raise SystemExit(rf"REFUSED: dst must be a directory GLM53-APEX-FOLD-TCQ33P* directly under {MODELS}")


def load_spec(args, cfg):
    spec = load_prune_spec(args.spec, cfg)          # the runtime's validator (MoE layers, id range, <= 279 per layer)
    bad = sorted(l for l in spec if l >= cfg.num_hidden_layers)
    if bad:
        raise SystemExit(f"REFUSED: the spec prunes layers {bad}; only trunk layers 3..{cfg.num_hidden_layers - 1} may be "
                         f"pruned (the MTP layer is copied as a hard link)")
    sup = super_experts(args.src)
    hit = sorted((l, e) for l, ids in spec.items() for e in ids if (l, e) in sup)
    if hit:
        raise SystemExit(f"REFUSED: the spec prunes super experts (layer, expert) {hit}")
    raw = open(args.spec, "rb").read()
    d = json.loads(raw.decode("utf-8"))
    return spec, hashlib.sha256(raw).hexdigest(), (d.get("meta") or {})


def source_layers(src, cfg):
    man = json.load(open(os.path.join(src, "fold_manifest.json"), encoding="utf-8"))
    if man.get("format") != "tcq":
        raise SystemExit(f"REFUSED: {src} is not a TCQ archive (fold_manifest format {man.get('format')!r})")
    layers = sorted(int(l) for l in man.get("layers", []))
    for l in layers:
        if not os.path.exists(os.path.join(src, layer_name(l))):
            raise SystemExit(f"REFUSED: {src} lacks {layer_name(l)}")
    return man, layers


def precheck_sources(args, spec, trunk, cfg):
    """before anything is written: every pruned layer's source file is a full permutation file that holds the spec's
    experts, and at least num_experts_per_tok + 1 experts stay. Only the small members are read. -> {layer: rows}"""
    n_rows = {}
    for l in trunk:
        if l not in spec:
            continue
        with np.load(os.path.join(args.src, layer_name(l)), allow_pickle=False) as raw:
            if "perm" not in raw.files:
                raise SystemExit(f"REFUSED: {layer_name(l)} has no perm member")
            perm = raw["perm"].astype(np.int64).tolist()
            n_tot = int(json.loads(str(raw["meta"])).get("n_experts_total", 288))
        if sorted(perm) != list(range(n_tot)):
            raise SystemExit(f"REFUSED: {layer_name(l)} is not a full {n_tot}-row file")
        absent = sorted(set(spec[l]) - set(perm))
        if absent:
            raise SystemExit(f"REFUSED: layer {l}: spec experts {absent} are not in the source file")
        if n_tot - len(spec[l]) < int(cfg.num_experts_per_tok) + 1:
            raise SystemExit(f"REFUSED: layer {l} would keep {n_tot - len(spec[l])} experts")
        n_rows[l] = n_tot
    missing_layers = sorted(set(spec) - set(trunk))
    if missing_layers:
        raise SystemExit(f"REFUSED: spec layers {missing_layers} are not trunk layers of the source archive")
    return n_rows


def read_source(src_p):
    raw = np.load(src_p, allow_pickle=False)
    small = {k: raw[k] for k in raw.files if k != "expert_slab_u8"}
    order = list(raw.files)
    raw.close()
    meta = json.loads(str(small["meta"]))
    n_tot = int(meta.get("n_experts_total", 288))
    perm = small["perm"].astype(np.int64)
    if sorted(perm.tolist()) != list(range(n_tot)):
        raise SystemExit(f"REFUSED: {src_p} is not a full {n_tot}-row file (perm is not a permutation)")
    mm = ApexFoldDiskStore._npz_memmaps(src_p)["expert_slab_u8"]
    return small, order, meta, n_tot, perm, mm


def check_layer(src_p, dst_p, l, spec_l, full_record=None, hash_file=True):
    """complete check of one written pruned file against its source -> dict (raises AssertionError on any failure)."""
    t0 = time.perf_counter()
    small, order, smeta, n_tot, perm, smm = read_source(src_p)
    keep_rows = [r for r in range(perm.size) if int(perm[r]) not in spec_l]
    kept = perm[keep_rows]
    with zipfile.ZipFile(dst_p) as zf:
        infos = zf.infolist()
        assert all(zi.compress_type == zipfile.ZIP_STORED for zi in infos), "a member is compressed"
        names = [zi.filename[:-4] for zi in infos]
    assert names == order, f"member order {names} != source {order}"
    raw = np.load(dst_p, allow_pickle=False)
    for k in ("preset_rule", "layer_id", "signs_g", "signs_u", "signs_d", "tables"):
        a, b = raw[k], small[k]
        assert a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b), f"member {k} differs from the source"
    dperm = raw["perm"]
    assert dperm.dtype == np.int16 and np.array_equal(dperm.astype(np.int64), kept), "perm != source perm minus the spec"
    meta = json.loads(str(raw["meta"]))
    raw.close()
    for k, v in smeta.items():
        assert meta.get(k) == v, f"source meta key {k} changed"
    assert int(meta["n_experts_total"]) == n_tot, "n_experts_total"
    assert sorted(meta["pruned"]["experts"]) == sorted(spec_l), "meta pruned experts != spec"
    arch = TR.TcqArchive(dst_p, mmap_slab=True)          # the runtime loader: byte lock, perm range / repeats / length
    assert arch.n_experts_total == n_tot and arch.layer_id == l
    assert sorted(arch.missing.tolist()) == sorted(spec_l), "TcqArchive.missing != spec"
    assert int(arch.expert_slab_u8.shape[0]) == len(keep_rows)
    for i, r in enumerate(keep_rows):                    # row_of maps every kept expert to its new row
        assert int(arch.row_of[int(perm[r])]) == i
    dmm = arch.expert_slab_u8.numpy()
    rb = int(dmm.shape[1])
    assert rb == int(smm.shape[1])
    n_eq = 0
    for i0 in range(0, len(keep_rows), CHUNK_ROWS):
        i1 = min(len(keep_rows), i0 + CHUNK_ROWS)
        sr = keep_rows[i0:i1]
        if sr[-1] - sr[0] == i1 - i0 - 1:
            s_chunk = smm[sr[0]: sr[-1] + 1]
        else:
            s_chunk = smm[np.asarray(sr)]
        d_chunk = dmm[i0:i1]
        if not np.array_equal(d_chunk, s_chunk):
            bad = [i0 + j for j in range(i1 - i0) if not np.array_equal(d_chunk[j], s_chunk[j])]
            raise AssertionError(f"layer {l}: destination rows {bad} differ from their source rows")
        n_eq += i1 - i0
    del dmm, arch, smm
    out = {"rows": len(keep_rows), "rows_equal": n_eq, "bytes": os.path.getsize(dst_p), "missing": sorted(spec_l),
           "check_s": None}
    if hash_file:
        out["sha256"] = sha256_file(dst_p)
    out["check_s"] = round(time.perf_counter() - t0, 1)
    if full_record is not None:
        assert out["bytes"] == full_record["bytes"], "size differs from the write record"
        if hash_file and full_record.get("sha256"):
            assert out["sha256"] == full_record["sha256"], "sha256 differs from the write record"
    return out


# ------------------------------------------------------------------------------------------------- write
def write_layer(args, src_p, dst_p, l, spec_l, spec_sha, spec_meta):
    t0 = time.perf_counter()
    small, order, smeta, n_tot, perm, mm = read_source(src_p)
    missing_in_file = sorted(set(spec_l) - set(perm.tolist()))
    if missing_in_file:
        raise SystemExit(f"REFUSED: layer {l}: spec experts {missing_in_file} are not in the source file")
    keep_rows = [r for r in range(perm.size) if int(perm[r]) not in spec_l]
    kept = perm[keep_rows]
    rb = int(mm.shape[1])
    slab = np.empty((len(keep_rows), rb), dtype=np.uint8)
    for a, b, i in runs(keep_rows):
        slab[i: i + b - a] = mm[a:b]
    t_read = time.perf_counter() - t0
    del mm
    meta = dict(smeta)
    meta.update({
        "n_experts_total": n_tot,
        "pruned": {"experts": sorted(int(e) for e in spec_l), "n": len(spec_l), "criterion": spec_meta.get("criterion"),
                   "level_pct": spec_meta.get("level_pct"), "spec": os.path.abspath(args.spec), "spec_sha256": spec_sha},
        "source": {"dir": os.path.abspath(args.src), "file": os.path.basename(src_p), "bytes": os.path.getsize(src_p),
                   "rows": int(perm.size)},
        "row_order": "source popularity order with the pruned rows removed (stable)",
        "tool": "build_run\\r34_prune_write.py",
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    t1 = time.perf_counter()
    nbytes = TR.save_layer(dst_p, str(small["preset_rule"]), small["layer_id"],
                           {p: torch.from_numpy(small[f"signs_{p}"]) for p in ("g", "u", "d")},
                           torch.from_numpy(small["tables"]), torch.from_numpy(slab), meta,
                           perm=torch.from_numpy(kept.astype(np.int16)))
    t_write = time.perf_counter() - t1
    del slab
    chk = check_layer(src_p, dst_p, l, spec_l)
    rec = {"rows": len(keep_rows), "bytes": nbytes, "sha256": chk["sha256"], "verified": True,
           "read_s": round(t_read, 1), "write_s": round(t_write, 1), "check_s": chk["check_s"],
           "write_mb_s": round(nbytes / 1e6 / max(t_write, 1e-6), 1)}
    log(f"L{l:02d}: {len(keep_rows)} rows ({len(spec_l)} pruned), {nbytes / 1e9:.3f} GB; read {t_read:.1f} s, write "
        f"{t_write:.1f} s ({rec['write_mb_s']} MB/s), check {chk['check_s']} s (rows equal {chk['rows_equal']}/"
        f"{len(keep_rows)}, sha256 {chk['sha256'][:16]}) | {time.perf_counter() - t0:.0f} s")
    return rec


def link(src_f, dst_f):
    if os.path.exists(dst_f):
        if os.path.samefile(src_f, dst_f):
            return "existing link"
        raise SystemExit(f"STOP: {dst_f} exists and is not a link to {src_f}")
    os.link(src_f, dst_f)
    assert os.path.samefile(src_f, dst_f)
    return "linked"


def other_files(src):
    """(relative path) of the non-layer files to hard-link: prefix caches and super experts."""
    rel = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(src, "exact_prefix_cache_*.pt")))]
    rel += [os.path.join("super_experts", os.path.basename(p))
            for p in sorted(glob.glob(os.path.join(src, "super_experts", "*.pt")))]
    return rel


def do_write(args):
    cfg = GLM53FlashModelConfig()
    check_dst_name(args)
    spec, spec_sha, spec_meta = load_spec(args, cfg)
    man, layers = source_layers(args.src, cfg)
    trunk = [l for l in layers if l < cfg.num_hidden_layers]
    n_rows = precheck_sources(args, spec, trunk, cfg)
    rb = int(man.get("expert_row_bytes", 6359296))
    st_path = os.path.join(args.dst, STATE)
    if os.path.exists(args.dst):
        if not args.resume:
            raise SystemExit(f"REFUSED: {args.dst} exists (use --resume to continue the writer's own partial directory)")
        if not os.path.exists(st_path):
            raise SystemExit(f"REFUSED: {args.dst} exists but has no {STATE} (not this writer's directory)")
        state = json.load(open(st_path, encoding="utf-8"))
        if (os.path.normcase(state["src"]) != os.path.normcase(os.path.abspath(args.src))
                or state["spec_sha256"] != spec_sha):
            raise SystemExit("REFUSED: --resume with another src / spec than the partial directory was written with")
        for p in glob.glob(os.path.join(args.dst, "*.tmp.npz")):
            os.remove(p)
            log(f"resume: removed the partial file {os.path.basename(p)}")
    else:
        state = {"src": os.path.abspath(args.src), "dst": os.path.abspath(args.dst), "spec": os.path.abspath(args.spec),
                 "spec_sha256": spec_sha, "tag": args.tag, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "layers": {},
                 "complete": False}
    todo = [l for l in trunk if l in spec and not (str(l) in state["layers"] and state["layers"][str(l)].get("verified")
                                                   and os.path.exists(os.path.join(args.dst, layer_name(l))))]
    need = sum((n_rows[l] - len(spec[l])) * rb + (1 << 20) for l in todo)
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(args.dst))).free
    log(f"src {args.src} ({len(layers)} layer files), dst {args.dst}, spec {args.spec} (sha256 {spec_sha[:16]}, "
        f"{sum(len(v) for v in spec.values())} experts in {len(spec)} layers); to write {len(todo)} layers = "
        f"{need / 1e9:.2f} GB; free {free / 1e9:.1f} GB; min free {args.min_free_gb} GB")
    if free < need + args.min_free_gb * 1e9:
        raise SystemExit(f"REFUSED: free disk {free / 1e9:.1f} GB < {need / 1e9:.1f} GB + {args.min_free_gb} GB")
    os.makedirs(args.dst, exist_ok=True)
    save_json(st_path, state)
    crash_after = int(os.environ.get("R34_PW_CRASH_AFTER_LAYER", "-1"))      # unit test only
    t_all = time.perf_counter()
    for l in trunk:
        dst_p = os.path.join(args.dst, layer_name(l))
        src_p = os.path.join(args.src, layer_name(l))
        if l not in spec:
            state["layers"][str(l)] = {"hardlink_of": src_p, "rows": 288, "bytes": os.path.getsize(src_p),
                                       "how": link(src_p, dst_p)}
            save_json(st_path, state)
            continue
        if l not in todo:
            log(f"L{l:02d}: already written and verified (resume)")
            continue
        need = sum((n_rows[x] - len(spec[x])) * rb + (1 << 20) for x in todo if x >= l)
        free = shutil.disk_usage(args.dst).free
        if free < need + args.min_free_gb * 1e9:
            raise SystemExit(f"STOP before layer {l}: free disk {free / 1e9:.1f} GB < {need / 1e9:.1f} GB + "
                             f"{args.min_free_gb} GB (resume later with --resume)")
        state["layers"][str(l)] = write_layer(args, src_p, dst_p, l, set(spec[l]), spec_sha, spec_meta)
        save_json(st_path, state)
        if l == crash_after:
            log(f"simulated crash after layer {l} (R34_PW_CRASH_AFTER_LAYER)")
            raise SystemExit(9)
    for l in layers:                                   # the MTP layer (and any other non-trunk layer): hard link
        if l in trunk:
            continue
        src_p, dst_p = os.path.join(args.src, layer_name(l)), os.path.join(args.dst, layer_name(l))
        state["layers"][str(l)] = {"hardlink_of": src_p, "rows": 288, "bytes": os.path.getsize(src_p),
                                   "how": link(src_p, dst_p)}
    links = {}
    for rel in other_files(args.src):
        os.makedirs(os.path.dirname(os.path.join(args.dst, rel)), exist_ok=True)
        links[rel] = link(os.path.join(args.src, rel), os.path.join(args.dst, rel))
    state["links"] = {rel: os.path.join(os.path.abspath(args.src), rel) for rel in links}
    files = {}
    for l in layers:
        r = state["layers"][str(l)]
        files[layer_name(l)] = ({"rows": r["rows"], "bytes": r["bytes"], "sha256": r["sha256"]} if "sha256" in r
                                else {"rows": r["rows"], "bytes": r["bytes"], "hardlink_of": r["hardlink_of"]})
    for rel, tgt in state["links"].items():
        files[rel.replace(os.sep, "/")] = {"bytes": os.path.getsize(tgt), "hardlink_of": tgt}
    k_per = {str(l): len(spec[l]) for l in sorted(spec)}
    man2 = dict(man)
    man2["pruned"] = {"source_dir": os.path.abspath(args.src), "spec": os.path.abspath(args.spec), "spec_sha256": spec_sha,
                      "criterion": spec_meta.get("criterion"), "level_pct": spec_meta.get("level_pct"),
                      "k_per_layer": k_per, "experts_total": int(sum(len(v) for v in spec.values())),
                      "n_experts_total": 288,
                      "row_order": "source popularity order with the pruned rows removed (stable)",
                      "layer_45": "unpruned MTP layer, hard link to the source file",
                      "router_mask": "the runtime masks every expert a file has no row for (TcqArchive.missing)",
                      "tool": "build_run\\r34_prune_write.py", "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    man2["files"] = files
    man2["served_bytes_trunk"] = int(sum(state["layers"][str(l)]["bytes"] for l in trunk))
    save_json(os.path.join(args.dst, "fold_manifest.json"), man2)
    state["complete"] = True
    state["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    state["write_total_s"] = round(time.perf_counter() - t_all, 1)
    save_json(st_path, state)
    rep = {"tag": args.tag, "state": state, "manifest_pruned": man2["pruned"],
           "served_bytes_trunk": man2["served_bytes_trunk"],
           "dst_bytes_new": int(sum(r["bytes"] for r in state["layers"].values() if "sha256" in r)),
           "free_after_gb": round(shutil.disk_usage(args.dst).free / 1e9, 2)}
    save_json(os.path.join(args.out_dir, f"r34_prune_write_{args.tag}.json"), rep)
    log(f"done: {sum('sha256' in r for r in state['layers'].values())} pruned layer files, "
        f"{sum('hardlink_of' in r for r in state['layers'].values())} layer links, {len(links)} other links; trunk "
        f"{man2['served_bytes_trunk'] / 1e9:.3f} GB; {rep['dst_bytes_new'] / 1e9:.2f} GB new; free "
        f"{rep['free_after_gb']} GB; {state['write_total_s']:.0f} s")
    return 0


# ------------------------------------------------------------------------------------------------- verify
def do_verify(args):
    cfg = GLM53FlashModelConfig()
    t0 = time.perf_counter()
    spec, spec_sha, _ = load_spec(args, cfg)
    man, layers = source_layers(args.src, cfg)
    trunk = [l for l in layers if l < cfg.num_hidden_layers]
    res = {"tag": args.tag, "src": args.src, "dst": args.dst, "spec": args.spec, "spec_sha256": spec_sha, "layers": {},
           "links": {}, "failures": []}

    def fail(msg):
        res["failures"].append(msg)
        log(f"FAIL: {msg}")

    st_path = os.path.join(args.dst, STATE)
    state = json.load(open(st_path, encoding="utf-8")) if os.path.exists(st_path) else {"layers": {}}
    if not state.get("complete"):
        fail("the writer's state file is missing or not complete")
    if state.get("spec_sha256") not in (None, spec_sha):
        fail("the state file was written with another spec")
    man2 = json.load(open(os.path.join(args.dst, "fold_manifest.json"), encoding="utf-8")) \
        if os.path.exists(os.path.join(args.dst, "fold_manifest.json")) else None
    if man2 is None:
        fail("fold_manifest.json missing")
    else:
        for k in ("format", "tcq_rule", "preset", "layers", "expert_row_bytes"):
            if man2.get(k) != man.get(k):
                fail(f"manifest key {k} differs from the source")
        if (man2.get("pruned") or {}).get("spec_sha256") != spec_sha:
            fail("manifest pruned.spec_sha256 != spec")
    for l in layers:
        src_p, dst_p = os.path.join(args.src, layer_name(l)), os.path.join(args.dst, layer_name(l))
        if not os.path.exists(dst_p):
            fail(f"{layer_name(l)} missing")
            continue
        if l in trunk and l in spec:
            try:
                rec = (man2 or {}).get("files", {}).get(layer_name(l))
                r = check_layer(src_p, dst_p, l, set(spec[l]), full_record=rec)
                res["layers"][str(l)] = r
                log(f"L{l:02d}: PASS ({r['rows_equal']}/{r['rows']} rows equal, {r['bytes']} B, sha256 {r['sha256'][:16]} "
                    f"= manifest, {r['check_s']} s)")
            except AssertionError as e:
                fail(f"layer {l}: {e}")
        else:
            same = os.path.samefile(src_p, dst_p)
            res["layers"][str(l)] = {"hardlink": same}
            if not same:
                fail(f"{layer_name(l)} is not a hard link to the source")
    for rel in other_files(args.src):
        p = os.path.join(args.dst, rel)
        ok = os.path.exists(p) and os.path.samefile(os.path.join(args.src, rel), p)
        res["links"][rel] = ok
        if not ok:
            fail(f"{rel} is not a hard link to the source")
    # one full RAM load through the runtime loader (mmap_slab=False) of the first pruned layer
    lp = [l for l in trunk if l in spec]
    if lp:
        a = TR.TcqArchive(os.path.join(args.dst, layer_name(lp[0])), mmap_slab=False)
        b = TR.TcqArchive(os.path.join(args.dst, layer_name(lp[0])), mmap_slab=True)
        eq = bool(torch.equal(a.expert_slab_u8, b.expert_slab_u8)) and bool(torch.equal(a.perm, b.perm))
        res["ram_load_equal_mmap"] = {"layer": lp[0], "equal": eq}
        if not eq:
            fail("RAM load != mmap load")
        del a, b
    res["served_bytes_trunk"] = int(sum(os.path.getsize(os.path.join(args.dst, layer_name(l))) for l in trunk))
    res["verify_s"] = round(time.perf_counter() - t0, 1)
    res["verdict"] = "PASS" if not res["failures"] else "FAIL"
    save_json(os.path.join(args.out_dir, f"r34_prune_verify_{args.tag}.json"), res)
    log(f"verify {res['verdict']}: {sum(1 for v in res['layers'].values() if 'rows_equal' in v)} pruned files row-checked, "
        f"{sum(1 for v in res['layers'].values() if v.get('hardlink'))} layer links, "
        f"{sum(res['links'].values())}/{len(res['links'])} other links; trunk {res['served_bytes_trunk'] / 1e9:.3f} GB; "
        f"{res['verify_s']:.0f} s")
    return 0 if not res["failures"] else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=5.0)
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--test-root", default="", help="unit test only: allow a dst directly under this directory")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    LOG_FH[0] = open(os.path.join(args.out_dir, f"r34_prune_write_{args.tag}.log"), "a", encoding="utf-8")
    log(f"start {'verify-only' if args.verify_only else 'write'} {vars(args)}")
    if args.verify_only:
        check_dst_name(args)
        return do_verify(args)
    return do_write(args)


if __name__ == "__main__":
    sys.exit(main())
