"""[R3.9SMP] Thinking budget with the FakeRunner (no model, no GPU). Run: python test_serve_budget_r39.py"""
import sys

import glm53_fold_serve as S

FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


REASONING = "".join(f"step {i}: thinking about the design again\n" for i in range(60))


def script(prompt_text, prompt, h):
    return REASONING + "</think>The answer is 42."      # a model that thinks for a long time, then answers


tpl = S.ChatTemplate.from_file("C:/models/GLM53-backbone-r34h/chat_template.jinja")
tok = S.FakeTokenizer()
close = tok.encode("</think>")
check("fake tokenizer encodes </think> as one id", len(close) == 1, str(close))

msgs = [{"role": "user", "content": "design something"}]


def run(budget_default=0, ext=None, thinking=True):
    eng = S.ServeEngine(S.FakeRunner(tokenizer=tok, script=script), tpl, default_thinking=thinking, max_context=100000,
                        default_max_tokens=5000, default_thinking_budget=budget_default)
    body = {"messages": msgs}
    if ext is not None:
        body["apex_fold"] = ext
    return eng.execute(eng.prepare(body)), eng


res, eng = run(0)
m = res["choices"][0]["message"]
check("no budget: full reasoning kept", m.get("reasoning_content", "").count("step ") == 60, f"({m.get('reasoning_content','').count('step ')} steps)")
check("no budget: answer intact", (m.get("content") or "").strip() == "The answer is 42.")
check("no budget: reported as not forced", res["apex_fold"]["thinking_budget"]["forced_at"] is None)

res, eng = run(20)
m = res["choices"][0]["message"]
tbi = res["apex_fold"]["thinking_budget"]
steps = m.get("reasoning_content", "").count("step ")
check("budget 20: </think> forced", tbi["forced_at"] is not None and tbi["forced_at"] >= 20, str(tbi))
check("budget 20: reasoning cut early (far fewer than 60 steps)", 0 < steps < 15, f"({steps} steps)")
check("budget 20: cut at a line break", (m.get("reasoning_content") or "").rstrip(" ").endswith("again"), repr((m.get("reasoning_content") or "")[-30:]))
check("budget 20: the rest became the answer", "The answer is 42." in (m.get("content") or ""), repr((m.get("content") or "")[:80]))
check("budget 20: counted in server counters", eng.counters.get("thinking_budget_forced") == 1)

res, _ = run(20, ext={"thinking_budget": 0})
check("per-request apex_fold.thinking_budget=0 turns it off", res["apex_fold"]["thinking_budget"]["forced_at"] is None)
res, _ = run(0, ext={"thinking_budget": 25})
check("per-request budget turns it on", res["apex_fold"]["thinking_budget"]["forced_at"] is not None)
res, _ = run(20, thinking=False)
check("thinking off: no budget applied", res["apex_fold"]["thinking_budget"]["budget"] == 0)
try:
    run(0, ext={"thinking_budget": -1})
    check("negative budget rejected", False)
except S.RequestError:
    check("negative budget rejected", True)

print("RESULT:", "ALL PASS" if not FAILS else f"FAILED {FAILS}")
sys.exit(1 if FAILS else 0)
