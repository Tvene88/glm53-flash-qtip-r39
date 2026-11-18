#!/usr/bin/env python3
"""test_serve_stable_r37.py - CPU tests of the stable-prefix anchor of claude_r37_speed/glm53_fold_serve.py [R3.7-STABLE].

No GPU, no model weights, no torch: the FakeRunner's state is a non-rewindable hash that is checked on every restore.

  python -I build_run/test_serve_stable_r37.py

What an agent client with a request proxy does: its proxy appends per-request text (retrieved context, reminders) to the
END of the newest message and removes it again from that message in the next request. The client therefore sends the
conversation twice: `messages` (with the tail) and `apex_fold.stable_messages` (without it).

Tests
  agent_loop    a tool loop of four steps with a different tail in every request: answers equal those of an engine
                without reuse; from the second request on the stable step restores the previous stable point and the
                real request restores the new one; an engine with reuse but without the field restores less.
  no_field      without the field the engine makes no stable step (one runner call per request).
  bad_field     a field the engine cannot use (not a list; a different conversation) never fails the request.
  points        three points per entry keep final, prompt-end and stable; with two the stable point wins over an older
                anchor (the order of ConversationCache._protected).
Runtime dir: env R34_RUNTIME, else ../claude_r37_speed. Exit 0 = all passed.
"""
import os
import sys
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("R34_RUNTIME", os.path.join(os.path.dirname(HERE), "runtime"))
os.environ.setdefault("R34_BACKBONE", "C:/models/GLM53-backbone-r34h")
sys.path.insert(0, HERE)
import test_serve_r35 as T  # noqa: E402  (helpers: check, _chat, _strip_call_ids, _assistant_echo, _agent_script, WEATHER)

S = T.S
check = T.check
S.STABLE_MIN = 8                         # the fake tokenizer makes short prompts; the rule itself is what is tested

FILL = "The survey team walked along the river and measured the depth of the water at every bend. " * 6


def _tail(i):
    return f"\n\n<reference-context>retrieved for request {i}: note {i * 7919 % 101}</reference-context>" \
           f"\n\n<system-reminder>emit the tool call in this response ({i})</system-reminder>"


def _with_tail(hist, i):
    full = [dict(m) for m in hist]
    last = full[-1]
    check(last["role"] in ("user", "tool"), "the newest message is a user or tool message")
    last["content"] = last["content"] + _tail(i)
    return full


def _engines():
    tpl = S.ChatTemplate(T.TPL_SRC)
    ra, rb, rc = (S.FakeRunner(script=T._agent_script) for _ in range(3))
    ea = S.ServeEngine(ra, tpl, default_thinking=False, min_reuse=8, cache_points=3, log=lambda s: None)
    eb = S.ServeEngine(rb, tpl, default_thinking=False, reuse=False, log=lambda s: None)
    ec = S.ServeEngine(rc, tpl, default_thinking=False, min_reuse=8, cache_points=3, log=lambda s: None)
    return (ra, ea), (rb, eb), (rc, ec)


def test_agent_loop():
    (ra, ea), (rb, eb), (rc, ec) = _engines()
    hist = [{"role": "system", "content": "You are an agent. " + FILL},
            {"role": "user", "content": "Check the weather in three cities. " + FILL}]
    stable_ns, rows = [], []
    for step in range(6):
        full = _with_tail(hist, step)
        a = T._chat(ea, {"messages": full, "tools": [T.WEATHER], "apex_fold": {"stable_messages": hist}})
        b = T._chat(eb, {"messages": full, "tools": [T.WEATHER]})
        c = T._chat(ec, {"messages": full, "tools": [T.WEATHER]})
        ma, mb = T._strip_call_ids(a["choices"][0]["message"]), T._strip_call_ids(b["choices"][0]["message"])
        check(ma == mb, f"step {step}: the stable step changed the answer")
        check(T._strip_call_ids(c["choices"][0]["message"]) == mb, f"step {step}: plain reuse changed the answer")
        st = a["apex_fold"]["reuse"].get("stable")
        check(isinstance(st, dict) and st["n"] > 0, f"step {step}: no stable step reported ({a['apex_fold']['reuse']})")
        sub, real = ra.calls[-2], ra.calls[-1]
        check(sub["prompt"] == st["n"] and sub["gen"] == 1, f"step {step}: the stable step is a one-token generation at K {sub}")
        check(real["keep"] == st["n"], f"step {step}: the request restores the new stable point ({real}, K {st['n']})")
        if stable_ns:
            check(sub["keep"] == stable_ns[-1], f"step {step}: the stable step restores the previous stable point "
                                                f"(keep {sub['keep']}, previous K {stable_ns[-1]})")
            check(a["usage"]["prompt_tokens_details"]["cached_tokens"] == stable_ns[-1],
                  f"step {step}: usage reports the tokens the request really restored")
            check(rc.calls[-1]["keep"] < stable_ns[-1], f"step {step}: without the field less is restored "
                                                        f"({rc.calls[-1]['keep']} vs {stable_ns[-1]})")
        P = a["usage"]["prompt_tokens"]
        check(a["apex_fold"]["reuse"]["prefilled_tokens"] + a["usage"]["prompt_tokens_details"]["cached_tokens"] == P,
              f"step {step}: restored + prefilled = prompt tokens")
        stable_ns.append(st["n"])
        rows.append((P, st["cached"], rc.calls[-1]["keep"]))
        msg = T._assistant_echo(a)
        hist.append(msg)
        if a["choices"][0]["finish_reason"] != "tool_calls":
            break
        for tcall in msg["tool_calls"]:
            hist.append({"role": "tool", "tool_call_id": tcall["id"], "content": f"sunny, {20 + step} degrees. " + FILL})
    check(len(stable_ns) == 4 and a["choices"][0]["message"]["content"] == "All cities checked.", f"loop of 4 requests {stable_ns}")
    check(all(y > x for x, y in zip(stable_ns, stable_ns[1:])), f"stable points grow {stable_ns}")
    return "4 requests; (prompt, restored with the field, restored without): " + ", ".join(str(r) for r in rows)


def test_no_field():
    (ra, ea), _, _ = _engines()
    hist = [{"role": "user", "content": "Check the weather in three cities. " + FILL}]
    a = T._chat(ea, {"messages": _with_tail(hist, 0), "tools": [T.WEATHER]})
    check(len(ra.calls) == 1 and a["apex_fold"]["reuse"].get("stable") is None, f"one runner call, no stable step {ra.calls}")
    return "one runner call"


def test_bad_field():
    _, (_rb, eb), _ = _engines()
    hist = [{"role": "user", "content": "Check the weather in three cities. " + FILL}]
    full = _with_tail(hist, 0)
    ref = T._strip_call_ids(T._chat(eb, {"messages": full, "tools": [T.WEATHER]})["choices"][0]["message"])
    for name, field in (("not a list", "x"), ("empty list", []), ("bad role", [{"role": "nobody", "content": "x"}])):
        (ra, ea), _, _ = _engines()                               # a fresh engine: nothing cached
        a = T._chat(ea, {"messages": full, "tools": [T.WEATHER], "apex_fold": {"stable_messages": field}})
        check(T._strip_call_ids(a["choices"][0]["message"]) == ref, f"{name}: answer changed")
        check(a["apex_fold"]["reuse"].get("stable") is None, f"{name}: a stable step ran")
        check(len(ra.calls) == 1, f"{name}: one runner call")
        check(any("stable_messages" in s for s in a["apex_fold"]["notes"]), f"{name}: a note says why ({a['apex_fold']['notes']})")
    # Another conversation shares only the template head and the tool block with the prompt: that prefix is a valid
    # (small) stable point, so a stable step may run there; the answer and the token accounting must still be right.
    (ra, ea), _, _ = _engines()
    a = T._chat(ea, {"messages": full, "tools": [T.WEATHER], "apex_fold": {"stable_messages": [{"role": "user", "content": "Unrelated."}]}})
    check(T._strip_call_ids(a["choices"][0]["message"]) == ref, "another conversation: answer changed")
    st, P = a["apex_fold"]["reuse"].get("stable"), a["usage"]["prompt_tokens"]
    check(st is None or 0 < st["n"] < P, f"another conversation: stable point inside the prompt ({st}, prompt {P})")
    check(a["apex_fold"]["reuse"]["prefilled_tokens"] + a["usage"]["prompt_tokens_details"]["cached_tokens"] == P,
          "another conversation: restored + prefilled = prompt tokens")
    # An identical second request restores everything; no stable step is needed then.
    n0 = len(ra.calls)
    a2 = T._chat(ea, {"messages": full, "tools": [T.WEATHER], "apex_fold": {"stable_messages": hist}})
    check(T._strip_call_ids(a2["choices"][0]["message"]) == ref and len(ra.calls) == n0 + 1 and
          a2["apex_fold"]["reuse"].get("stable") is None, f"identical request: one runner call, no stable step {ra.calls[n0:]}")
    return "3 unusable fields answered normally with a note; a foreign prefix and an identical re-request handled"


def test_points():
    mk = lambda n, kind="", anchor=False, stable=False: S.RestorePoint(n_tokens=n, kind=kind, anchor=anchor, stable=stable)
    pts = [mk(900, "final"), mk(880, "prompt-end"), mk(700, "prompt-end", stable=True), mk(400, "prompt-end", anchor=True),
           mk(300, "prompt-end", stable=True)]
    k3 = [p.n_tokens for p in S.ConversationCache._select_points(pts, 3)]
    k2 = [p.n_tokens for p in S.ConversationCache._select_points(pts, 2)]
    k4 = [p.n_tokens for p in S.ConversationCache._select_points(pts, 4)]
    check(k3 == [900, 880, 700], f"three points: final, prompt-end, newest stable {k3}")
    check(k4 == [900, 880, 700, 400], f"four points add the anchor {k4}")
    check(k2 == [900, 880], f"two points: final and prompt-end {k2}")
    old = [mk(900, "final"), mk(880, "prompt-end"), mk(400, "prompt-end", anchor=True), mk(300, "")]
    check([p.n_tokens for p in S.ConversationCache._select_points(old, 3)] == [900, 880, 400], "without stable points: as before")
    return "selection order final, prompt-end, stable, anchor"


def main():
    failed = 0
    for fn in (test_agent_loop, test_no_field, test_bad_field, test_points):
        try:
            print(f"PASS {fn.__name__}: {fn()}")
        except Exception as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"test_serve_stable_r37: {4 - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
