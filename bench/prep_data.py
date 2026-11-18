#!/usr/bin/env python3
"""
prep_data.py - one-time download of the two public benchmark datasets into ./data as plain files.

Needs the network and the `huggingface_hub` + `pyarrow` packages (only this script does; bench_openai.py, compare.py
and selfcheck.py use the standard library and read the JSONL files written here).

  openai/openai_humaneval   openai_humaneval/test-00000-of-00001.parquet   -> data/humaneval.jsonl        (164 rows)
  openai/gsm8k              main/test-00000-of-00001.parquet               -> data/gsm8k_test.jsonl       (1319 rows)

It also writes the fixed item orders (seed 3901):
  data/gsm8k_sample_250.json   250 test questions, in the order they were drawn. `--limit N` on this file therefore
                               gives a fixed random subset that is nested in the 250.
  data/humaneval_order.json    all 164 problems in a seeded random order: `--limit N` on it gives a fixed random subset.
and data/SOURCES.md with the repository revisions and the sha256 of every file.

The dataset revisions are pinned below, so a second run gives byte-identical files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import tempfile

SEED = 3901
GSM8K_N = 250
SOURCES = [
    # repo id, revision, file in the repo, local parquet name
    ("openai/openai_humaneval", "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
     "openai_humaneval/test-00000-of-00001.parquet", "humaneval_test.parquet"),
    ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866",
     "main/test-00000-of-00001.parquet", "gsm8k_main_test.parquet"),
]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: str, rows) -> None:
    with open(path, "wb") as f:                      # binary: "\n" line ends on every platform
        for r in rows:
            f.write((json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))


def write_json(path: str, obj) -> None:
    with open(path, "wb") as f:
        f.write((json.dumps(obj, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    args = ap.parse_args()
    from huggingface_hub import hf_hub_download     # imported here: only this script needs it
    import pyarrow.parquet as pq

    os.makedirs(args.data_dir, exist_ok=True)
    parquet = {}
    with tempfile.TemporaryDirectory() as cache:
        for repo, rev, fname, local in SOURCES:
            p = hf_hub_download(repo, fname, repo_type="dataset", revision=rev, cache_dir=cache, token=False)
            dst = os.path.join(args.data_dir, local)
            shutil.copyfile(p, dst)
            parquet[repo] = dst

    he = pq.read_table(parquet["openai/openai_humaneval"]).to_pylist()
    he_rows = [{k: r[k] for k in ("task_id", "prompt", "canonical_solution", "test", "entry_point")} for r in he]
    assert len(he_rows) == 164, len(he_rows)
    write_jsonl(os.path.join(args.data_dir, "humaneval.jsonl"), he_rows)

    gs = pq.read_table(parquet["openai/gsm8k"]).to_pylist()
    gs_rows = [{"id": f"gsm8k/test/{i}", "question": r["question"], "answer": r["answer"]} for i, r in enumerate(gs)]
    assert len(gs_rows) == 1319, len(gs_rows)
    write_jsonl(os.path.join(args.data_dir, "gsm8k_test.jsonl"), gs_rows)

    drawn = random.Random(SEED).sample(range(len(gs_rows)), GSM8K_N)
    write_json(os.path.join(args.data_dir, "gsm8k_sample_250.json"), {
        "task": "gsm8k", "seed": SEED, "n": GSM8K_N,
        "method": f"random.Random({SEED}).sample(range({len(gs_rows)}), {GSM8K_N}); ids in the order drawn "
                  "(row index of openai/gsm8k, config main, split test)",
        "ids": [f"gsm8k/test/{i}" for i in drawn]})
    order = random.Random(SEED).sample(range(len(he_rows)), len(he_rows))
    write_json(os.path.join(args.data_dir, "humaneval_order.json"), {
        "task": "humaneval", "seed": SEED, "n": len(he_rows),
        "method": f"random.Random({SEED}).sample(range({len(he_rows)}), {len(he_rows)}): all problems in a seeded "
                  "random order, so that --limit N is a fixed random subset",
        "ids": [he_rows[i]["task_id"] for i in order]})

    files = ["humaneval_test.parquet", "humaneval.jsonl", "gsm8k_main_test.parquet", "gsm8k_test.jsonl",
             "gsm8k_sample_250.json", "humaneval_order.json"]
    lines = ["# Data sources", "",
             "Written by `prep_data.py`. Both datasets are public on the Hugging Face Hub (MIT licence).", "",
             "| Dataset | Config / split | Revision | File in the repository |", "|---|---|---|---|",
             f"| `openai/openai_humaneval` | `openai_humaneval` / `test` (164 rows) | `{SOURCES[0][1]}` | `{SOURCES[0][2]}` |",
             f"| `openai/gsm8k` | `main` / `test` (1,319 rows) | `{SOURCES[1][1]}` | `{SOURCES[1][2]}` |", "",
             "The `.parquet` files are the downloads, unchanged. The `.jsonl` files hold the same rows as JSON lines "
             "(UTF-8, `\\n` line ends, keys sorted); `gsm8k_test.jsonl` adds the id `gsm8k/test/<row index>`.", "",
             f"`gsm8k_sample_250.json`: `random.Random({SEED}).sample(range(1319), {GSM8K_N})`, in the order drawn. "
             f"`humaneval_order.json`: `random.Random({SEED}).sample(range(164), 164)`. "
             "With `--limit N` either file gives a fixed random subset.", "",
             "| File | Bytes | sha256 |", "|---|---:|---|"]
    for fn in files:
        p = os.path.join(args.data_dir, fn)
        lines.append(f"| `{fn}` | {os.path.getsize(p):,} | `{sha256(p)}` |")
        print(f"{fn:28s} {os.path.getsize(p):>9,d}  {sha256(p)}")
    with open(os.path.join(args.data_dir, "SOURCES.md"), "wb") as f:
        f.write(("\n".join(lines) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
