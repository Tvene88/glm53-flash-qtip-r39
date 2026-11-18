# Data sources

Written by `prep_data.py`. Both datasets are public on the Hugging Face Hub (MIT licence).

| Dataset | Config / split | Revision | File in the repository |
|---|---|---|---|
| `openai/openai_humaneval` | `openai_humaneval` / `test` (164 rows) | `7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544` | `openai_humaneval/test-00000-of-00001.parquet` |
| `openai/gsm8k` | `main` / `test` (1,319 rows) | `740312add88f781978c0658806c59bc2815b9866` | `main/test-00000-of-00001.parquet` |

The `.parquet` files are the downloads, unchanged. The `.jsonl` files hold the same rows as JSON lines (UTF-8, `\n` line ends, keys sorted); `gsm8k_test.jsonl` adds the id `gsm8k/test/<row index>`.

`gsm8k_sample_250.json`: `random.Random(3901).sample(range(1319), 250)`, in the order drawn. `humaneval_order.json`: `random.Random(3901).sample(range(164), 164)`. With `--limit N` either file gives a fixed random subset.

| File | Bytes | sha256 |
|---|---:|---|
| `humaneval_test.parquet` | 83,920 | `2f2871a15fbc95b6c683043359f4ed8e144c5a1c4f24f25f66bc51f598dfcfb6` |
| `humaneval.jsonl` | 214,315 | `b19293e42932e70278b8e5ed83f8c82879cf34e0d3a44cc580f060b7cdaba4f8` |
| `gsm8k_main_test.parquet` | 419,088 | `ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59` |
| `gsm8k_test.jsonl` | 780,798 | `f99a71f3d1f0c0155e6d4a331ff0090260dbca5126b30b3ae6f4823657c0167a` |
| `gsm8k_sample_250.json` | 5,235 | `39ac05a2ba644a7be520492119bc5176143b1fda12b971207e514598ebfe2a67` |
| `humaneval_order.json` | 3,211 | `98af726fbdb8d4d2cedcc8d2b52c0a3b86fe3a3ee732d1356bdbc5549af55835` |
