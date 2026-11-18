#!/usr/bin/env python3
"""
selfcheck.py - CPU-only test of bench_openai.py and compare.py. No GPU, no model, no network beyond 127.0.0.1.

  1. unit checks of the statistics and of the answer extraction;
  2. the test executor: the canonical solution of all 164 HumanEval problems must pass through the same path a model
     answer takes (prompt -> answer text -> code block -> program -> subprocess);
  3. end to end against a fake OpenAI server in this process that returns canned answers (right, wrong, truncated,
     with a thinking section before the code, without a code block, an endless loop, ...), including an interrupted
     run that is continued with --resume, and compare.py on two fake models.

Exit code 0 = every check passed.
"""
from __future__ import annotations

import concurrent.futures
import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_openai as B          # noqa: E402
import compare as C               # noqa: E402

CHECKS = {"n": 0, "failed": []}


def check(name: str, cond: bool, info: str = "") -> None:
    CHECKS["n"] += 1
    if not cond:
        CHECKS["failed"].append(name)
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{' -- ' + info if (info and not cond) else ''}", flush=True)


def close(a: float, b: float, tol: float = 5e-4) -> bool:
    return abs(a - b) <= tol


# ------------------------------------------------------------------------------------------------------- 1. unit checks
def unit_checks() -> None:
    print("[1] statistics and extraction")
    lo, hi = B.wilson(8, 10)
    check("wilson(8, 10) = 0.4902 .. 0.9433", close(lo, 0.4902) and close(hi, 0.9433), f"{lo:.4f} {hi:.4f}")
    lo, hi = B.wilson(0, 20)
    check("wilson(0, 20) = 0 .. 0.1611", close(lo, 0.0) and close(hi, 0.1611), f"{lo:.4f} {hi:.4f}")
    lo, hi = B.wilson(164, 164)
    check("wilson(164, 164) upper bound 1", close(hi, 1.0) and close(lo, 0.9771), f"{lo:.4f} {hi:.4f}")
    check("compare.wilson equals bench.wilson", C.wilson(37, 90) == B.wilson(37, 90))
    check("mcnemar_exact(1, 9) = 22/1024", close(C.mcnemar_exact(1, 9), 22 / 1024, 1e-12))
    check("mcnemar_exact(9, 1) symmetric", close(C.mcnemar_exact(9, 1), 22 / 1024, 1e-12))
    check("mcnemar_exact(0, 0) = 1", C.mcnemar_exact(0, 0) == 1.0)
    check("mcnemar_exact(5, 5) = 1", C.mcnemar_exact(5, 5) == 1.0)
    check("mcnemar_exact(0, 6) = 2/64", close(C.mcnemar_exact(0, 6), 2 / 64, 1e-12))

    ans, th = B.split_thinking({"content": "<think>plan</think>\nfinal"})
    check("inline <think> block removed", ans == "final" and th == "plan", repr((ans, th)))
    ans, th = B.split_thinking({"content": "plan without opening tag</think>final"})
    check("closing tag alone (tag opened by the template)", ans == "final" and "plan" in th, repr((ans, th)))
    ans, th = B.split_thinking({"content": "<think>never ends ```python\nx=1\n```"})
    check("unclosed thinking leaves no final answer", ans == "" and "never ends" in th, repr((ans, th)))
    ans, th = B.split_thinking({"content": "final", "reasoning_content": "hidden"})
    check("reasoning_content kept apart", ans == "final" and th == "hidden")
    ans, th = B.split_thinking({"content": None, "reasoning_content": "only thinking"})
    check("null content", ans == "" and th == "only thinking")
    ans, th = B.split_thinking({"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]})
    check("content parts joined", ans == "ab")

    txt = "x\n```python\ndef f():\n    return 1\n```\ny\n```python\nprint(f())\n```\n"
    code, how = B.pick_python_block(txt, "f")
    check("last python block that defines the function", how == "block_with_function" and code.startswith("def f"), how)
    code, how = B.pick_python_block("```python\ndef g():\n    pass\n```\n```python\nreturn 2\n```", "f")
    check("no block defines it: the last python block", how == "last_block" and code == "return 2", repr(code))
    code, how = B.pick_python_block("```\ndef f():\n    return 1\n```\n```text\nnotes\n```", "f")
    check("untagged block used when no python block exists", how == "block_with_function", how)
    code, how = B.pick_python_block("```text\ndef f(): pass\n```", "f")
    check("blocks of other languages ignored", code is None and how == "no_code_block")
    code, how = B.pick_python_block("```python\ndef f():\n    return 1", "f")
    check("unclosed final block accepted", code is not None and "return 1" in code)
    code, how = B.pick_python_block("def f(): return 1", "f")
    check("no fence, no code", code is None)

    D = B.Decimal
    cases = [("so\nANSWER: 18", D(18)), ("ANSWER: 18.0", D(18)), ("ANSWER: $1,234.50", D("1234.50")),
             ("**ANSWER: 42**", D(42)), ("ANSWER: -7", D(-7)), ("ANSWER: 5\nlater\nANSWER: 6", D(6)),
             ("Answer: 9", D(9)), ("ANSWER: \\boxed{12}", D(12)), ("ANSWER: 18 dollars.", D(18)),
             ("ANSWER: .5", D("0.5")), ("The answer is 18.", None), ("ANSWER: eighteen", None), ("", None)]
    for text, want in cases:
        got = B.extract_gsm8k_answer(text)
        check(f"gsm8k answer of {text!r:.40} -> {want}", got == want, repr(got))
    check("gsm8k reference '#### 1,234'", B.gsm8k_gold({"answer": "work\n#### 1,234"}) == D(1234))
    check("18 == 18.0 as decimals", D("18") == D("18.0"))


# ---------------------------------------------------------------------------------------- 2. canonical solutions pass
def answer_with(code: str, before: str = "Here is the function.\n") -> str:
    return f"{before}```python\n{code.rstrip()}\n```\n"


def canonical_checks(problems) -> None:
    print("[2] test executor: canonical solutions through the model-answer path")
    bad_pre = []
    for p in problems:
        try:
            compile(B.humaneval_preamble(p), "<preamble>", "exec")
        except SyntaxError:
            bad_pre.append(p["task_id"])
    check("the part of every prompt before the function compiles on its own", not bad_pre, str(bad_pre))

    def one(p, body_only):
        code = p["canonical_solution"] if body_only else p["prompt"] + p["canonical_solution"]
        rec = {"wall_s": 1.0, "response": {"choices": [{"message": {"content": answer_with(code)}, "finish_reason": "stop"}],
                                           "usage": {"prompt_tokens": 1, "completion_tokens": 1}}}
        return B.score_item("humaneval", p, rec, 60.0)

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 2)) as ex:
        full = list(ex.map(lambda p: one(p, False), problems))
        body = list(ex.map(lambda p: one(p, True), problems[:40]))
    bad = [(s["id"], s["detail"]) for s in full if not s["correct"]]
    check(f"complete function: {sum(s['correct'] for s in full)}/164 canonical solutions pass", not bad, str(bad[:5]))
    check("complete function: all run as mode 'function'", all(s["program_mode"] == "function" for s in full))
    bad = [(s["id"], s["detail"]) for s in body if not s["correct"]]
    check(f"body only: {sum(s['correct'] for s in body)}/40 canonical bodies pass", not bad, str(bad[:5]))
    check("body only: all run as mode 'body'", all(s["program_mode"] == "body" for s in body))
    wrong = dict(problems[0])
    rec = {"wall_s": 1.0, "response": {"choices": [{"message": {"content": answer_with(
        problems[0]["prompt"] + "    return None\n")}, "finish_reason": "stop"}], "usage": {}}}
    s = B.score_item("humaneval", wrong, rec, 60.0)
    check("a wrong body fails the tests", not s["correct"] and s["detail"] != "", s["detail"])
    print(f"      ({time.perf_counter() - t0:.1f} s for {len(full) + len(body) + 1} test programs)")


# ------------------------------------------------------------------------------------------------- 3. fake server runs
class Fake:
    """Canned answers keyed by the exact prompt text. behaviours[prompt][model] = dict(content, finish, reasoning,
    fail_first) ; counts requests per (model, prompt)."""

    def __init__(self):
        self.table = {}
        self.requests = {}
        self.fail_once = set()
        self.bodies = []
        self.lock = threading.Lock()

    def handler(self):
        fake = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                data = json.dumps(obj).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
                if not self.path.endswith("/chat/completions"):
                    return self._send(404, {"error": {"message": "not found"}})
                prompt = body["messages"][-1]["content"]
                model = body.get("model")
                with fake.lock:
                    fake.bodies.append(body)
                    key = (model, prompt)
                    fake.requests[key] = fake.requests.get(key, 0) + 1
                    fail = key in fake.fail_once
                    fake.fail_once.discard(key)
                if fail:
                    return self._send(500, {"error": {"message": "canned failure", "type": "server_error"}})
                beh = fake.table.get(prompt, {}).get(model)
                if beh is None:
                    return self._send(400, {"error": {"message": "unknown prompt", "type": "invalid_request_error"}})
                time.sleep(0.02)
                msg = {"role": "assistant", "content": beh["content"]}
                if beh.get("reasoning") is not None:
                    msg["reasoning_content"] = beh["reasoning"]
                resp = {"id": "fake", "object": "chat.completion", "model": model,
                        "choices": [{"index": 0, "message": msg, "finish_reason": beh.get("finish", "stop")}],
                        "usage": {"prompt_tokens": 120, "completion_tokens": 60, "total_tokens": 180}}
                if model == "fake-a":
                    resp["timings"] = {"predicted_per_second": 25.0, "prompt_per_second": 100.0}   # llama.cpp style
                else:
                    resp["apex_fold"] = {"timing": {"decode_tok_s": 20.0}}                          # APEX-FOLD style
                return self._send(200, resp)

        return H


def run_bench(argv) -> tuple:
    buf = io.StringIO()
    code = None
    with contextlib.redirect_stdout(buf):
        try:
            code = B.main(argv)
        except SystemExit as e:
            code = e.code
    return code, buf.getvalue()


def fake_server_checks(problems, gsm_rows, data_dir) -> None:
    print("[3] end to end against the fake server")
    fake = Fake()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), fake.handler())
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    tmp = tempfile.mkdtemp(prefix="bench_selfcheck_")
    try:
        # ---------------- humaneval: (name, content, finish, reasoning, right for model a, right for model b)
        P = problems[:12]
        ok = [p["prompt"] + p["canonical_solution"] for p in P]
        bad = [p["prompt"] + "    return None\n" for p in P]
        he = [
            ("right answer", answer_with(ok[0]), "stop", None),
            ("wrong answer", answer_with(bad[1]), "stop", None),
            ("right code but truncated", answer_with(ok[2]), "length", None),
            ("thinking section with wrong code, then the right code",
             "<think>First idea:\n" + answer_with(bad[3], "") + "No, that returns nothing.</think>\n" + answer_with(ok[3]), "stop", None),
            ("wrong code in reasoning_content, right code in the answer", answer_with(ok[4]), "stop", answer_with(bad[4])),
            ("no code block", "The function should compare the numbers pairwise.", "stop", None),
            ("endless loop", answer_with(P[6]["prompt"] + "    while True:\n        pass\n"), "stop", None),
            ("function block followed by an example block",
             answer_with(ok[7]) + "Example:\n```python\nprint('demo')\n```\n", "stop", None),
            ("thinking never closed", "<think>" + answer_with(ok[8]), "stop", None),
            ("function body only", answer_with(P[9]["canonical_solution"]), "stop", None),
            ("closing think tag only, then the right code", "Reasoning text ...</think>\n" + answer_with(ok[10]), "stop", None),
            ("right answer, request fails once", answer_with(ok[11]), "stop", None),
        ]
        he_expect_a = [True, False, False, True, True, False, False, True, False, True, True, True]
        he_ids = [p["task_id"] for p in P]
        for i, (p, (name, content, finish, reasoning)) in enumerate(zip(P, he)):
            beh_a = {"content": content, "finish": finish, "reasoning": reasoning}
            # model b: items 0 and 1 swapped (0 wrong, 1 right), item 4 wrong; otherwise the same answers
            if i == 0:
                beh_b = {"content": answer_with(bad[0]), "finish": "stop"}
            elif i == 1:
                beh_b = {"content": answer_with(ok[1]), "finish": "stop"}
            elif i == 4:
                beh_b = {"content": answer_with(bad[4]), "finish": "stop"}
            else:
                beh_b = beh_a
            fake.table[B.prompt_for("humaneval", dict(p, id=p["task_id"]))] = {"fake-a": beh_a, "fake-b": beh_b}
        he_expect_b = list(he_expect_a)
        he_expect_b[0], he_expect_b[1], he_expect_b[4] = False, True, False
        ids_path = os.path.join(tmp, "he_ids.json")
        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump({"ids": he_ids}, f)
        fail_prompt = B.prompt_for("humaneval", dict(P[11], id=P[11]["task_id"]))
        fake.fail_once.add(("fake-a", fail_prompt))

        out_a = os.path.join(tmp, "a_humaneval")
        common = ["--base-url", base, "--task", "humaneval", "--data-dir", data_dir, "--ids-file", ids_path,
                  "--max-tokens", "512", "--exec-timeout", "4", "--retries", "0", "--reasoning-effort", "low",
                  "--thinking", "off", "--extra-body", '{"seed": 1, "chat_template_kwargs": {"clear_thinking": true}}']
        code, text = run_bench(common + ["--model", "fake-a", "--out", out_a])
        summ = json.load(open(os.path.join(out_a, "summary.json"), encoding="utf-8"))
        check("run 1 (one request fails): exit code 3, 11 scored, 1 missing",
              code == 3 and summ["n_scored"] == 11 and summ["n_missing"] == 1 and summ["complete"] is False,
              f"code {code} {summ['n_scored']} {summ['n_missing']}")
        b0 = fake.bodies[0]
        check("request body: one user message, max_tokens, temperature 0, stream false",
              [m["role"] for m in b0["messages"]] == ["user"] and b0["max_tokens"] == 512 and b0["temperature"] == 0
              and b0["stream"] is False, json.dumps({k: v for k, v in b0.items() if k != "messages"}))
        check("request body: chat_template_kwargs = effort low, thinking off, merged extra field; seed from --extra-body",
              b0.get("chat_template_kwargs") == {"reasoning_effort": "low", "enable_thinking": False, "clear_thinking": True}
              and b0.get("seed") == 1 and "reasoning_effort" not in b0, json.dumps(b0.get("chat_template_kwargs")))
        check("request body: the prompt holds the function header in a python block",
              "```python\n" + P[0]["prompt"].rstrip("\n") + "\n```" in fake.bodies[0]["messages"][0]["content"]
              or any("```python\n" + P[0]["prompt"].rstrip("\n") + "\n```" in b["messages"][0]["content"] for b in fake.bodies))
        code, text = run_bench(common + ["--model", "fake-a", "--out", out_a])
        check("a second run without --resume is refused", code not in (0, 3) and "--resume" in str(code), str(code)[:80])
        code, text = run_bench(common[:10] + ["--max-tokens", "999"] + common[12:] + ["--model", "fake-a", "--out", out_a, "--resume"])
        check("--resume with other settings is refused", code not in (0, 3) and "settings differ" in str(code), str(code)[:80])
        before = dict(fake.requests)
        code, text = run_bench(common + ["--model", "fake-a", "--out", out_a, "--resume"])
        new = {k: v - before.get(k, 0) for k, v in fake.requests.items() if v - before.get(k, 0)}
        check("--resume requests only the missing item", code == 0 and new == {("fake-a", fail_prompt): 1}, f"code {code} {len(new)}")
        summ = json.load(open(os.path.join(out_a, "summary.json"), encoding="utf-8"))
        items = {r["id"]: r for r in B.read_jsonl(os.path.join(out_a, "items.jsonl"))}
        for i, (name, *_rest) in enumerate(he):
            got = items[he_ids[i]]["correct"]
            check(f"humaneval: {name} -> {'right' if he_expect_a[i] else 'wrong'}", got == he_expect_a[i], items[he_ids[i]]["detail"])
        check("humaneval: truncated item is flagged and would have passed otherwise",
              items[he_ids[2]]["truncated"] and items[he_ids[2]]["correct_ignoring_truncation"]
              and items[he_ids[2]]["detail"].startswith("truncated"))
        check("humaneval: endless loop reported as timeout", items[he_ids[6]]["detail"] == "timeout", items[he_ids[6]]["detail"])
        check("humaneval: missing code block reported", items[he_ids[5]]["extraction"] == "no_code_block")
        check("humaneval: body-only answer ran under the prompt header", items[he_ids[9]]["program_mode"] == "body")
        k = sum(he_expect_a)
        lo, hi = B.wilson(k, 12)
        check(f"summary: {k}/12 correct, score, Wilson interval",
              summ["n_correct"] == k and close(summ["score"], k / 12, 1e-4) and close(summ["wilson95"][0], lo, 1e-4)
              and close(summ["wilson95"][1], hi, 1e-4) and summ["complete"] is True, json.dumps(summ["wilson95"]))
        check("summary: 1 truncated, 1 of them otherwise correct, 2 without an extractable answer",
              summ["n_truncated"] == 1 and summ["n_truncated_but_otherwise_correct"] == 1 and summ["n_no_extractable_answer"] == 2,
              f"{summ['n_truncated']} {summ['n_truncated_but_otherwise_correct']} {summ['n_no_extractable_answer']}")
        check("summary: token means and reported tok/s (llama.cpp timings field)",
              summ["mean_prompt_tokens"] == 120 and summ["mean_completion_tokens"] == 60 and summ["mean_tok_s_reported"] == 25.0
              and summ["mean_tok_s_wall"] is not None and summ["mean_tok_s_wall"] > 0 and summ["total_completion_tokens"] == 720)
        raw_lines = sum(1 for _ in open(os.path.join(out_a, "responses.jsonl"), encoding="utf-8"))
        check("responses.jsonl: 1 settings line + 13 response lines (12 answers, 1 failure), nothing rewritten", raw_lines == 14, str(raw_lines))
        before = dict(fake.requests)
        code, text = run_bench(["--task", "humaneval", "--data-dir", data_dir, "--ids-file", ids_path, "--out", out_a, "--rescore",
                                "--exec-timeout", "4"])
        summ2 = json.load(open(os.path.join(out_a, "summary.json"), encoding="utf-8"))
        check("--rescore sends nothing and gives the same score", code == 0 and fake.requests == before and summ2["n_correct"] == k)

        out_b = os.path.join(tmp, "b_humaneval")
        code, text = run_bench(common + ["--model", "fake-b", "--out", out_b, "--concurrency", "4"])
        summ_b = json.load(open(os.path.join(out_b, "summary.json"), encoding="utf-8"))
        check(f"model b, concurrency 4: {sum(he_expect_b)}/12 correct, reported tok/s from the apex_fold field",
              code == 0 and summ_b["n_correct"] == sum(he_expect_b) and summ_b["mean_tok_s_reported"] == 20.0,
              f"code {code} {summ_b['n_correct']}")

        # ---------------- gsm8k
        sample = B.load_ids_file(os.path.join(data_dir, "gsm8k_sample_250.json"))
        by_id = {r["id"]: r for r in gsm_rows}
        G = [by_id[i] for i in sample[:9]]
        gold = [int(B.gsm8k_gold(g)) for g in G]
        gs = [
            ("right answer", f"Step by step ...\nANSWER: {gold[0]}", "stop", None, True),
            ("wrong number", f"Step by step ...\nANSWER: {gold[1] + 1}", "stop", None, False),
            ("right number but truncated", f"ANSWER: {gold[2]}", "length", None, False),
            ("wrong ANSWER inside the thinking section, right one after it",
             f"<think>try\nANSWER: {gold[3] + 7}\nno</think>Recomputed.\nANSWER: {gold[3]}", "stop", None, True),
            ("no ANSWER line", f"The answer is {gold[4]}.", "stop", None, False),
            ("formatted number ($, thousands separator, .00, bold)", f"**ANSWER: ${gold[5]:,}.00**", "stop", None, True),
            ("two ANSWER lines, the last one right", f"ANSWER: {gold[6] + 3}\nCorrection.\nANSWER: {gold[6]}", "stop", None, True),
            ("right number only in reasoning_content", "I am done.", "stop", f"ANSWER: {gold[7]}", False),
            ("right number with a unit after it", f"ANSWER: {gold[8]} dollars", "stop", None, True),
        ]
        for g, (name, content, finish, reasoning, _exp) in zip(G, gs):
            beh = {"content": content, "finish": finish, "reasoning": reasoning}
            fake.table[B.prompt_for("gsm8k", g)] = {"fake-a": beh, "fake-b": beh}
        out_g = os.path.join(tmp, "a_gsm8k")
        code, text = run_bench(["--base-url", base, "--task", "gsm8k", "--data-dir", data_dir, "--limit", "9", "--model", "fake-a",
                                "--out", out_g, "--retries", "0"])
        summ = json.load(open(os.path.join(out_g, "summary.json"), encoding="utf-8"))
        items = {r["id"]: r for r in B.read_jsonl(os.path.join(out_g, "items.jsonl"))}
        check("gsm8k default ids = first 9 of the fixed sample (--limit 9)", code == 0 and list(items) == sample[:9]
              and summ["data"]["ids_file"] == "gsm8k_sample_250.json", f"code {code}")
        for g, (name, _c, _f, _r, exp) in zip(G, gs):
            check(f"gsm8k: {name} -> {'right' if exp else 'wrong'}", items[g["id"]]["correct"] == exp, items[g["id"]]["detail"])
        k = sum(x[4] for x in gs)
        check(f"gsm8k summary: {k}/9 correct, 1 truncated, 2 without an ANSWER line",
              summ["n_correct"] == k and summ["n_truncated"] == 1 and summ["n_no_extractable_answer"] == 2,
              f"{summ['n_correct']} {summ['n_truncated']} {summ['n_no_extractable_answer']}")
        nb = [b for b in fake.bodies if b["messages"][0]["content"].startswith("Solve the following math problem")]
        check("gsm8k request without thinking / effort options sends no chat_template_kwargs",
              bool(nb) and all("chat_template_kwargs" not in b and "reasoning_effort" not in b for b in nb))
        code, text = run_bench(["--base-url", base, "--task", "gsm8k", "--data-dir", data_dir, "--ids-file", ids_path, "--model", "fake-a",
                                "--out", os.path.join(tmp, "bad_ids")])
        check("an ids file of the other task is refused", code not in (0, 3) and "not in" in str(code), str(code)[:80])

        # ---------------- compare.py
        buf = io.StringIO()
        cj = os.path.join(tmp, "compare.json")
        with contextlib.redirect_stdout(buf):
            C.main([f"a={out_a}", f"b={os.path.join(out_b, 'items.jsonl')}", "--json", cj])
        cmp_text = buf.getvalue()
        data = json.load(open(cj, encoding="utf-8"))
        pair = data["pairs"][0]
        both = sum(1 for x, y in zip(he_expect_a, he_expect_b) if x and y)
        oa = sum(1 for x, y in zip(he_expect_a, he_expect_b) if x and not y)
        ob = sum(1 for x, y in zip(he_expect_a, he_expect_b) if y and not x)
        check(f"compare: both right {both}, only a {oa}, only b {ob}, neither {12 - both - oa - ob}",
              (pair["both_right"], pair["only_a"], pair["only_b"], pair["neither"]) == (both, oa, ob, 12 - both - oa - ob), json.dumps(pair))
        check("compare: exact McNemar p of the discordant pairs", close(pair["mcnemar_exact_p"], C.mcnemar_exact(oa, ob), 1e-12)
              and pair["mcnemar_exact_p"] == 1.0)
        check("compare: same prompt token count on all common items", pair["same_prompt_tokens"] == 12 and pair["prompt_tokens_known"] == 12)
        check("compare: scores of both runs", data["runs"][0]["n_correct"] == sum(he_expect_a) and data["runs"][1]["n_correct"] == sum(he_expect_b))
        print("\n--- compare.py output on the two fake models " + "-" * 50)
        print(cmp_text.rstrip())
        print("-" * 96)
    finally:
        srv.shutdown()
        srv.server_close()
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    data_dir = B.DEFAULT_DATA
    print(f"selfcheck of {B.HARNESS_VERSION}; python {sys.version.split()[0]}; data files in ./data: "
          f"{'present' if os.path.exists(os.path.join(data_dir, 'humaneval.jsonl')) else 'MISSING (run prep_data.py)'}")
    problems = B.read_jsonl(os.path.join(data_dir, "humaneval.jsonl"))
    for p in problems:
        p["id"] = p["task_id"]
    gsm_rows = B.read_jsonl(os.path.join(data_dir, "gsm8k_test.jsonl"))
    check("data: 164 HumanEval problems, 1319 GSM8K test questions, 250 sample ids (unique)",
          len(problems) == 164 and len(gsm_rows) == 1319
          and len(set(B.load_ids_file(os.path.join(data_dir, "gsm8k_sample_250.json")))) == 250)
    check("data: every GSM8K reference is a number", all(B.gsm8k_gold(g) is not None for g in gsm_rows))
    unit_checks()
    canonical_checks(problems)
    fake_server_checks(problems, gsm_rows, data_dir)
    n, failed = CHECKS["n"], CHECKS["failed"]
    print(f"\nSELFCHECK {'PASSED' if not failed else 'FAILED'}: {n - len(failed)}/{n} checks ok"
          + ("" if not failed else "; failed: " + "; ".join(failed)))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
