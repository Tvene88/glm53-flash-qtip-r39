#!/usr/bin/env python3
"""
bench_openai.py - score a model served over an OpenAI-compatible /v1/chat/completions endpoint on

  humaneval   164 problems (openai/openai_humaneval). The model gets the function header + docstring and is asked for
              the complete function in one python code block. The last python block of the FINAL answer (the text
              after any thinking section) that defines the function is run against the canonical tests in a
              subprocess (time limit, temporary working directory).
  gsm8k       a fixed, seeded sample of test questions (openai/gsm8k, config main; default: the 250 ids of
              data/gsm8k_sample_250.json). The final answer must end with a line "ANSWER: <number>"; the last such
              number is compared exactly (as a decimal number) with the reference.

A truncated answer (finish_reason "length") counts as wrong. One request per item, one user message, no system
message, no few-shot examples. Standard library only. The harness itself opens no network connection except to
--base-url. NOTE: the humaneval task executes model-written code on this machine, without a sandbox.

Output directory (--out):
  responses.jsonl   append-only: one line per request with the raw response (the --resume state)
  items.jsonl       one scored line per item (rewritten at the end of every run from responses.jsonl)
  summary.json      score, Wilson 95 % interval, truncation counts, token and speed means, the settings used

Examples
  python bench_openai.py --base-url http://127.0.0.1:8088/v1 --model any --task humaneval --out runs/new/humaneval
  python bench_openai.py --base-url http://127.0.0.1:8088/v1 --model any --task gsm8k --limit 100 --resume \
         --out runs/new/gsm8k
  python bench_openai.py --task humaneval --out runs/new/humaneval --rescore        # no requests: score again
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

HARNESS_VERSION = "bench_r39-1.0"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "data")
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"

# ------------------------------------------------------------------------------------------------------------ prompts
HUMANEVAL_TEMPLATE = (
    "Complete the following Python function.\n"
    "Return the complete function (signature and body), together with any imports it needs, in one ```python code "
    "block. Do not add tests or example calls.\n\n"
    "```python\n{prompt}\n```"
)
GSM8K_TEMPLATE = (
    "Solve the following math problem. Explain your reasoning briefly, then finish with one last line of the form\n"
    "ANSWER: <number>\n"
    "where <number> is the final numeric answer alone (digits, an optional minus sign and decimal point; no units, "
    "no thousands separators, no other text).\n\n"
    "Problem: {question}"
)


def prompt_for(task: str, item: Dict[str, Any]) -> str:
    if task == "humaneval":
        return HUMANEVAL_TEMPLATE.format(prompt=item["prompt"].rstrip("\n"))
    return GSM8K_TEMPLATE.format(question=item["question"].strip())


# --------------------------------------------------------------------------------------------------------------- data
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_ids_file(path: str) -> List[str]:
    """A JSON list of ids, a JSON object with an "ids" list, or a text file with one id per line."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    ids = obj["ids"] if isinstance(obj, dict) else obj
    if not isinstance(ids, list):
        raise SystemExit(f"{path}: expected a list of ids or an object with an 'ids' list")
    return [str(i) for i in ids]


def load_task(task: str, data_dir: str, ids_file: Optional[str], limit: Optional[int]
              ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Returns the items to run, in run order (each with an "id"), and a description of the data used."""
    if task == "humaneval":
        data_path = os.path.join(data_dir, "humaneval.jsonl")
        rows = read_jsonl(data_path)
        for r in rows:
            r["id"] = r["task_id"]
        default_ids = None
    elif task == "gsm8k":
        data_path = os.path.join(data_dir, "gsm8k_test.jsonl")
        rows = read_jsonl(data_path)
        default_ids = os.path.join(data_dir, "gsm8k_sample_250.json")
    else:
        raise SystemExit(f"unknown task {task!r}")
    by_id = {r["id"]: r for r in rows}
    ids_path = ids_file or default_ids
    if ids_path:
        ids = load_ids_file(ids_path)
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise SystemExit(f"{ids_path}: {len(missing)} ids are not in {data_path} (first: {missing[0]!r}); "
                             "is this the ids file of the other task?")
        if len(set(ids)) != len(ids):
            raise SystemExit(f"{ids_path}: duplicate ids")
    else:
        ids = [r["id"] for r in rows]
    if limit is not None:
        ids = ids[: max(0, int(limit))]
    info = {"data_file": os.path.basename(data_path), "data_sha256": sha256_file(data_path),
            "ids_file": os.path.basename(ids_path) if ids_path else None,
            "ids_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(), "n_items": len(ids)}
    return [by_id[i] for i in ids], info


# ---------------------------------------------------------------------------------------------- answer text handling
def split_thinking(message: Dict[str, Any]) -> Tuple[str, str]:
    """(final answer text, thinking text) of an assistant message. Thinking is whatever the server put into
    reasoning_content / reasoning, plus anything inside <think> ... </think> left in the content. A thinking section
    that never closes leaves an empty final answer."""
    content = message.get("content")
    if isinstance(content, list):                    # content parts
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    content = content or ""
    thinking = message.get("reasoning_content") or message.get("reasoning") or ""
    if not isinstance(thinking, str):
        thinking = json.dumps(thinking, ensure_ascii=False)
    if THINK_CLOSE in content:                       # also covers a template that opened <think> in the prompt
        head, _, tail = content.rpartition(THINK_CLOSE)
        thinking = (thinking + "\n" + head.replace(THINK_OPEN, "")).strip()
        content = tail
    elif THINK_OPEN in content:                      # opened, never closed: no final answer
        head, _, tail = content.partition(THINK_OPEN)
        thinking = (thinking + "\n" + tail).strip()
        content = head
    return content.strip(), thinking


def extract_code_blocks(text: str) -> List[Dict[str, Any]]:
    """Fenced blocks of `text`, in order: {"lang", "code", "closed"}. A fence is a line that starts with ```."""
    blocks: List[Dict[str, Any]] = []
    cur: Optional[List[str]] = None
    lang = ""
    for line in text.splitlines():
        s = line.strip()
        if cur is None:
            if s.startswith("```"):
                lang = s[3:].strip().lower()
                cur = []
        elif s.startswith("```"):
            blocks.append({"lang": lang, "code": "\n".join(cur), "closed": True})
            cur = None
        else:
            cur.append(line)
    if cur is not None:
        blocks.append({"lang": lang, "code": "\n".join(cur), "closed": False})
    return blocks


def _defines(code: str, name: str) -> bool:
    return re.search(r"^[ \t]*(?:async[ \t]+)?def[ \t]+" + re.escape(name) + r"[ \t]*\(", code, re.M) is not None


def pick_python_block(text: str, entry_point: str) -> Tuple[Optional[str], str]:
    """The code to test and how it was chosen. Candidates are the blocks tagged python / py / python3, or, when
    there is none, the untagged blocks. The LAST candidate that defines `entry_point` is taken; when no candidate
    defines it, the last candidate."""
    blocks = extract_code_blocks(text)
    tagged = [b for b in blocks if b["lang"] in ("python", "py", "python3")]
    cands = tagged or [b for b in blocks if b["lang"] == ""]
    if not cands:
        return None, "no_code_block"
    for b in reversed(cands):
        if _defines(b["code"], entry_point):
            return b["code"], "block_with_function"
    return cands[-1]["code"], "last_block"


def humaneval_preamble(problem: Dict[str, Any]) -> str:
    """The part of the prompt before the function to write: imports and helper functions the tests may use."""
    lines = problem["prompt"].split("\n")
    pat = re.compile(r"^[ \t]*def[ \t]+" + re.escape(problem["entry_point"]) + r"[ \t]*\(")
    for i, ln in enumerate(lines):
        if pat.match(ln):
            return "\n".join(lines[:i])
    return ""


def build_humaneval_program(problem: Dict[str, Any], code: str) -> Tuple[str, str]:
    entry = problem["entry_point"]
    if _defines(code, entry):
        body, mode = humaneval_preamble(problem) + "\n\n" + code, "function"
    else:                                            # only a function body was returned: put it under the header
        lines = code.split("\n")
        first = next((ln for ln in lines if ln.strip()), "")
        if first and not first.startswith((" ", "\t")):
            lines = [("    " + ln) if ln.strip() else ln for ln in lines]
        body, mode = problem["prompt"].rstrip("\n") + "\n" + "\n".join(lines), "body"
    return body + "\n\n\n" + problem["test"] + "\n\ncheck(" + entry + ")\n", mode


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


def run_python(program: str, timeout_s: float) -> Dict[str, Any]:
    """Runs `program` with this interpreter in a new temporary directory. passed = exit code 0 inside the limit."""
    with tempfile.TemporaryDirectory(prefix="bench_exec_", ignore_cleanup_errors=True) as tmp:
        path = os.path.join(tmp, "prog.py")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(program)
        env = {k: v for k, v in os.environ.items()
               if k.upper() in ("SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "PATH", "PATHEXT", "COMSPEC", "LANG", "LC_ALL")}
        env.update({"TEMP": tmp, "TMP": tmp, "TMPDIR": tmp, "HOME": tmp, "USERPROFILE": tmp})
        kw: Dict[str, Any] = {}
        if os.name == "nt":
            kw["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kw["start_new_session"] = True
        t0 = time.perf_counter()
        proc = subprocess.Popen([sys.executable, "-I", "-B", "-X", "utf8", path], cwd=tmp, env=env,
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kw)
        try:
            _, err = proc.communicate(timeout=timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            try:
                _, err = proc.communicate(timeout=10)
            except Exception:
                err = b""
            timed_out = True
        dt = time.perf_counter() - t0
        tail = err.decode("utf-8", "replace").strip().splitlines()[-1:] if err else []
        return {"passed": (not timed_out) and proc.returncode == 0, "timed_out": timed_out,
                "exit_code": proc.returncode, "exec_s": round(dt, 3), "error": (tail[0][:300] if tail else "")}


_NUM = r"-?[ \t]*(?:\d[\d,]*(?:\.\d+)?|\.\d+)"
_ANSWER_RE = re.compile(r"ANSWER[ \t]*:[ \t]*(?:\\boxed\{|\\\(|\\\[|[*_`$ \t])*(" + _NUM + ")", re.I)


def to_decimal(text: str) -> Optional[Decimal]:
    try:
        return Decimal(re.sub(r"[,\s$]", "", text))
    except (InvalidOperation, ValueError):
        return None


def extract_gsm8k_answer(text: str) -> Optional[Decimal]:
    """The number of the LAST "ANSWER: <number>" in the final answer text (None when there is none)."""
    found = _ANSWER_RE.findall(text)
    return to_decimal(found[-1]) if found else None


def gsm8k_gold(item: Dict[str, Any]) -> Optional[Decimal]:
    return to_decimal(item["answer"].rsplit("####", 1)[-1].strip())


# ------------------------------------------------------------------------------------------------------------ scoring
def reported_tok_s(resp: Dict[str, Any]) -> Optional[float]:
    """Decode speed as the server reports it: llama.cpp `timings.predicted_per_second`, the APEX-FOLD server
    `apex_fold.timing.decode_tok_s`. None when the server reports neither."""
    for path in (("timings", "predicted_per_second"), ("apex_fold", "timing", "decode_tok_s")):
        cur: Any = resp
        for k in path:
            cur = cur.get(k) if isinstance(cur, dict) else None
        if isinstance(cur, (int, float)) and cur > 0:
            return float(cur)
    return None


def score_item(task: str, item: Dict[str, Any], rec: Dict[str, Any], exec_timeout: float) -> Dict[str, Any]:
    """One scored line from one raw response record."""
    resp = rec.get("response") or {}
    choice = (resp.get("choices") or [{}])[0] or {}
    message = choice.get("message") or {}
    usage = resp.get("usage") or {}
    finish = choice.get("finish_reason")
    answer, thinking = split_thinking(message)
    ptoks, ctoks = usage.get("prompt_tokens"), usage.get("completion_tokens")
    wall = rec.get("wall_s")
    out: Dict[str, Any] = {
        "id": item["id"], "task": task, "finish_reason": finish, "truncated": finish == "length",
        "prompt_tokens": ptoks, "completion_tokens": ctoks, "wall_s": wall,
        "tok_s_wall": round(ctoks / wall, 3) if isinstance(ctoks, int) and wall else None,
        "tok_s_reported": reported_tok_s(resp),
        "thinking_chars": len(thinking), "answer_chars": len(answer),
    }
    if task == "humaneval":
        code, how = pick_python_block(answer, item["entry_point"])
        out["extraction"] = how
        if code is None:
            ok, detail = False, "no python code block in the final answer"
        else:
            program, mode = build_humaneval_program(item, code)
            res = run_python(program, exec_timeout)
            ok = bool(res["passed"])
            detail = "" if ok else ("timeout" if res["timed_out"] else (res["error"] or f"exit code {res['exit_code']}"))
            out.update({"program_mode": mode, "exec_s": res["exec_s"], "code_sha256": hashlib.sha256(code.encode()).hexdigest()})
    else:
        got, gold = extract_gsm8k_answer(answer), gsm8k_gold(item)
        out.update({"predicted": None if got is None else str(got), "reference": None if gold is None else str(gold)})
        if got is None:
            ok, detail = False, "no 'ANSWER: <number>' line in the final answer"
        else:
            ok = gold is not None and got == gold
            detail = "" if ok else "wrong number"
    out["correct_ignoring_truncation"] = ok          # information only
    if out["truncated"]:
        ok, detail = False, "truncated (finish_reason length)" + (": " + detail if detail else "")
    out["correct"] = ok
    out["detail"] = detail
    return out


def wilson(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _mean(vals: List[Optional[float]]) -> Optional[float]:
    v = [float(x) for x in vals if isinstance(x, (int, float))]
    return round(sum(v) / len(v), 3) if v else None


def summarize(scored: List[Dict[str, Any]], n_selected: int) -> Dict[str, Any]:
    n = len(scored)
    k = sum(1 for s in scored if s["correct"])
    lo, hi = wilson(k, n)
    tot_c = sum(s["completion_tokens"] for s in scored if isinstance(s["completion_tokens"], int))
    tot_w = sum(s["wall_s"] for s in scored if isinstance(s["wall_s"], (int, float)))
    reasons: Dict[str, int] = {}
    for s in scored:
        reasons[str(s["finish_reason"])] = reasons.get(str(s["finish_reason"]), 0) + 1
    return {
        "n_selected": n_selected, "n_scored": n, "n_missing": n_selected - n, "complete": n == n_selected,
        "n_correct": k, "score": round(k / n, 5) if n else None,
        "wilson95": [round(lo, 5), round(hi, 5)],
        "n_truncated": sum(1 for s in scored if s["truncated"]),
        "n_truncated_but_otherwise_correct": sum(1 for s in scored if s["truncated"] and s["correct_ignoring_truncation"]),
        "n_no_extractable_answer": sum(1 for s in scored if s["detail"].endswith(("no python code block in the final answer",
                                                                                "no 'ANSWER: <number>' line in the final answer"))),
        "finish_reasons": reasons,
        "mean_prompt_tokens": _mean([s["prompt_tokens"] for s in scored]),
        "mean_completion_tokens": _mean([s["completion_tokens"] for s in scored]),
        "mean_wall_s": _mean([s["wall_s"] for s in scored]),
        "mean_tok_s_wall": _mean([s["tok_s_wall"] for s in scored]),
        "mean_tok_s_reported": _mean([s["tok_s_reported"] for s in scored]),
        "total_completion_tokens": tot_c, "total_wall_s": round(tot_w, 2),
        "aggregate_tok_s_wall": round(tot_c / tot_w, 3) if tot_w else None,
    }


# --------------------------------------------------------------------------------------------------------------- HTTP
def deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def request_settings(args) -> Dict[str, Any]:
    """Everything of the request body except the messages; the same for every item of a run."""
    body: Dict[str, Any] = {"model": args.model, "max_tokens": int(args.max_tokens),
                            "temperature": float(args.temperature), "stream": False}
    kwargs: Dict[str, Any] = {}
    if args.reasoning_effort:
        if args.reasoning_effort_field in ("template", "both"):
            kwargs["reasoning_effort"] = args.reasoning_effort
        if args.reasoning_effort_field in ("top", "both"):
            body["reasoning_effort"] = args.reasoning_effort
    if args.thinking in ("on", "off"):
        kwargs["enable_thinking"] = args.thinking == "on"
    if kwargs:
        body["chat_template_kwargs"] = kwargs
    if args.extra_body:
        text = args.extra_body
        if text.startswith("@"):
            with open(text[1:], "r", encoding="utf-8") as f:
                text = f.read()
        extra = json.loads(text)
        if not isinstance(extra, dict):
            raise SystemExit("--extra-body must be a JSON object")
        deep_merge(body, extra)
    return body


def post_chat(base_url: str, body: Dict[str, Any], api_key: str, timeout_s: float) -> Tuple[int, Any, str]:
    """(HTTP status, parsed JSON or None, error text). Status 0 = no HTTP answer (connection problem, timeout)."""
    url = base_url.rstrip("/") + "/chat/completions"
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))       # never through a proxy
    try:
        with opener.open(req, timeout=timeout_s) as r:
            raw = r.read().decode("utf-8", "replace")
            status = r.status
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        return e.code, None, raw[:2000]
    except Exception as e:                                                       # URLError, timeout, reset ...
        return 0, None, f"{type(e).__name__}: {e}"
    try:
        return status, json.loads(raw), ""
    except json.JSONDecodeError:
        return status, None, "response is not JSON: " + raw[:500]


def response_problem(resp: Any) -> str:
    if not isinstance(resp, dict):
        return "response is not a JSON object"
    if resp.get("error"):
        return "error object: " + json.dumps(resp["error"], ensure_ascii=False)[:500]
    ch = resp.get("choices")
    if not isinstance(ch, list) or not ch or not isinstance(ch[0], dict) or not isinstance(ch[0].get("message"), dict):
        return "no choices[0].message"
    return ""


# ---------------------------------------------------------------------------------------------------------------- run
class RawLog:
    """responses.jsonl: line 1 = {"type": "meta", ...}; then {"type": "response", ...} lines, appended and flushed
    one by one. For an id with several lines, the last line with ok = true counts."""

    def __init__(self, path: str):
        self.path = path
        self.lock = threading.Lock()

    def load(self) -> Tuple[Optional[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        meta, ok = None, {}
        if not os.path.exists(self.path):
            return meta, ok
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue                           # a line cut off by an interruption
                if rec.get("type") == "meta" and meta is None:
                    meta = rec
                elif rec.get("type") == "response" and rec.get("ok"):
                    ok[rec["id"]] = rec
        return meta, ok

    def append(self, rec: Dict[str, Any]) -> None:
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        with self.lock:
            with open(self.path, "a", encoding="utf-8", newline="\n") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())


def write_atomic(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["humaneval", "gsm8k"])
    ap.add_argument("--out", required=True, help="output directory of this run (one model, one task)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8088/v1", help="endpoint root that has /chat/completions")
    ap.add_argument("--model", default="default", help="model name sent in the request")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""), help="bearer token (default: env OPENAI_API_KEY)")
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--ids-file", default=None, help="ids to run, in this order (JSON list, JSON {\"ids\": [...]} or one id per "
                                                     "line). Default: humaneval = all 164; gsm8k = data/gsm8k_sample_250.json")
    ap.add_argument("--limit", type=int, default=None, help="only the first N ids of the selection")
    ap.add_argument("--max-tokens", type=int, default=2048, help="max_tokens of every request (thinking included)")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (default)")
    ap.add_argument("--reasoning-effort", default=None, help="e.g. low | high | max; not sent when omitted")
    ap.add_argument("--reasoning-effort-field", default="template", choices=["template", "top", "both"],
                    help="where --reasoning-effort goes: chat_template_kwargs.reasoning_effort (template, default: the "
                         "chat template reads it on every server), the top-level field, or both")
    ap.add_argument("--thinking", default="default", choices=["default", "on", "off"],
                    help="on / off sends chat_template_kwargs.enable_thinking; default sends nothing")
    ap.add_argument("--extra-body", default=None, help="JSON object merged into every request body (or @file.json)")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--request-timeout", type=float, default=3600.0, help="seconds per request")
    ap.add_argument("--retries", type=int, default=2, help="more attempts after a failed request")
    ap.add_argument("--exec-timeout", type=float, default=30.0, help="humaneval: seconds per test program")
    ap.add_argument("--resume", action="store_true", help="continue a run: keep the answers already in <out>/responses.jsonl")
    ap.add_argument("--allow-settings-change", action="store_true", help="with --resume: continue although the settings differ")
    ap.add_argument("--rescore", action="store_true", help="send nothing; score the stored responses again")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1")
    items, data_info = load_task(args.task, args.data_dir, args.ids_file, args.limit)
    settings = request_settings(args)
    template = HUMANEVAL_TEMPLATE if args.task == "humaneval" else GSM8K_TEMPLATE
    fingerprint = {"harness": HARNESS_VERSION, "task": args.task, "request": settings,
                   "prompt_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
                   "data_sha256": data_info["data_sha256"]}
    os.makedirs(args.out, exist_ok=True)
    log = RawLog(os.path.join(args.out, "responses.jsonl"))
    meta, done = log.load()
    if meta is None and done:
        raise SystemExit(f"{log.path} has answers but no settings line; move it away or use another --out")
    if meta is not None:
        if not (args.resume or args.rescore):
            raise SystemExit(f"{log.path} exists. Use --resume to continue that run, --rescore to score it again, or "
                             "another --out.")
        same = {k: meta.get(k) for k in fingerprint} == fingerprint
        if not same and not (args.allow_settings_change or args.rescore):
            diff = [k for k in fingerprint if meta.get(k) != fingerprint[k]]
            raise SystemExit(f"--resume: the settings differ from those of the stored run ({', '.join(diff)}): "
                             f"stored {json.dumps({k: meta.get(k) for k in diff})[:600]}. Use the same settings, another "
                             "--out, or --allow-settings-change.")
    elif args.rescore:
        raise SystemExit(f"--rescore: no stored run in {args.out}")
    else:
        log.append(dict(fingerprint, type="meta", base_url=args.base_url,
                        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))

    by_id = {it["id"]: it for it in items}
    pending = [] if args.rescore else [it for it in items if it["id"] not in done]
    print(f"[bench] task {args.task}: {len(items)} items selected, {len(items) - len(pending)} already answered, "
          f"{len(pending)} to request; endpoint {args.base_url}; settings {json.dumps(settings)}", flush=True)

    scored: Dict[str, Dict[str, Any]] = {}
    counter = {"n": len(items) - len(pending), "ok": 0, "failed": 0}
    stop = threading.Event()
    plock = threading.Lock()

    def work(item: Dict[str, Any]) -> None:
        if stop.is_set():
            return
        body = dict(settings, messages=[{"role": "user", "content": prompt_for(args.task, item)}])
        rec: Dict[str, Any] = {}
        for attempt in range(args.retries + 1):
            if stop.is_set():
                return
            t0 = time.perf_counter()
            started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            status, resp, err = post_chat(args.base_url, body, args.api_key, args.request_timeout)
            wall = time.perf_counter() - t0
            problem = err or (response_problem(resp) if status == 200 else f"HTTP {status}")
            rec = {"type": "response", "id": item["id"], "ok": not problem, "http_status": status, "error": problem,
                   "attempt": attempt + 1, "started_utc": started, "wall_s": round(wall, 4), "response": resp}
            log.append(rec)
            if not problem or 400 <= status < 500:        # a 4xx answer will not change on a retry
                break
            time.sleep(min(30.0, 2.0 * (attempt + 1)))
        with plock:
            counter["n"] += 1
            pos = counter["n"]
        if rec.get("ok"):
            s = score_item(args.task, item, rec, args.exec_timeout)
            with plock:
                scored[item["id"]] = s
                counter["ok"] += 1
                right = sum(1 for x in scored.values() if x["correct"])
            print(f"[{pos}/{len(items)}] {item['id']}: {'PASS' if s['correct'] else 'FAIL'} "
                  f"({s['finish_reason']}, {s['completion_tokens']} tok, {s['wall_s']:.1f} s, "
                  f"{s['tok_s_wall'] if s['tok_s_wall'] is not None else '?'} tok/s"
                  f"{'; ' + s['detail'] if s['detail'] else ''}) | this session {right}/{len(scored)}", flush=True)
        else:
            with plock:
                counter["failed"] += 1
            print(f"[{pos}/{len(items)}] {item['id']}: REQUEST FAILED ({rec.get('error', '')[:200]})", flush=True)

    interrupted = False
    if pending:
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency)
        futs = [ex.submit(work, it) for it in pending]
        try:
            for f in concurrent.futures.as_completed(futs):
                f.result()
        except KeyboardInterrupt:
            interrupted = True
            stop.set()
            for f in futs:
                f.cancel()
            print("[bench] interrupted: scoring what is stored; continue later with --resume", flush=True)
        ex.shutdown(wait=not interrupted, cancel_futures=True)

    _, done = log.load()
    rows = []
    for it in items:
        rec = done.get(it["id"])
        if rec is None:
            continue
        rows.append(scored.get(it["id"]) or score_item(args.task, by_id[it["id"]], rec, args.exec_timeout))
    write_atomic(os.path.join(args.out, "items.jsonl"),
                 "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    summary = {"harness": HARNESS_VERSION, "task": args.task, "model": args.model, "base_url": args.base_url,
               "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    summary.update(summarize(rows, len(items)))
    summary.update({"request": settings, "exec_timeout_s": args.exec_timeout if args.task == "humaneval" else None,
                    "data": data_info, "prompt_template_sha256": fingerprint["prompt_template_sha256"],
                    "truncated_counts_as": "wrong"})
    write_atomic(os.path.join(args.out, "summary.json"), json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    sc = summary["score"]
    print(f"[bench] {args.task}: {summary['n_correct']}/{summary['n_scored']} correct"
          f"{'' if sc is None else f' = {100 * sc:.1f} %'} (Wilson 95 % {100 * summary['wilson95'][0]:.1f}-"
          f"{100 * summary['wilson95'][1]:.1f} %); truncated {summary['n_truncated']}; missing {summary['n_missing']}; "
          f"mean completion tokens {summary['mean_completion_tokens']}; mean tok/s wall {summary['mean_tok_s_wall']}, "
          f"reported {summary['mean_tok_s_reported']}", flush=True)
    print(f"[bench] wrote {os.path.join(args.out, 'summary.json')}", flush=True)
    if interrupted:
        sys.stdout.flush()
        os._exit(130)                                    # requests still in flight are abandoned
    return 0 if summary["complete"] else 3


if __name__ == "__main__":
    sys.exit(main())
