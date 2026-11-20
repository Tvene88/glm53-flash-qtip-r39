#!/usr/bin/env python3
"""test_serve_r35.py - CPU tests of claude_r35_speed/glm53_fold_serve.py [R3.5 package S1]. No GPU, no model weights.

  python -I build_run/test_serve_r35.py [--no-torch] [--no-tokenizer] [--with-transformers] [-k NAME]

Groups
  template    pure-Python port vs jinja2 (the transformers environment) on 17 conversations x {gen prompt on/off} x
              efforts (tools, multi-turn tool loops, parallel calls with out-of-order results, ids missing / duplicated
              / int, legacy functions, clear_thinking, content parts, unicode); fixed expected strings; the engine's
              prompt = render + the encode_chat thinking-off rule.
  normalize   OpenAI message / tool normalization and its 400 errors.
  parse       tool-call parsing (schema typing, malformed blocks), reasoning split, round trip parse -> echo -> render.
  stream      OutputAssembler streaming == one-shot parse for random chunkings, stop strings, holdback; incremental
              detokenizer on split UTF-8.
  reuse       ConversationCache decisions (LCP / restore points / skip-prefill / min_reuse / indexer gap / limits) and
              the engine oracle: FakeRunner's state is a non-rewindable hash checked on every restore; multi-turn
              chats (thinking on / off, tool loops, EOS / max_tokens / stop endings, MTP-like states, regenerate,
              edited history) give identical answers with and without reuse while prefilling less.
  http        localhost round trip with the FakeRunner: /health, /v1/models, multi-turn tool use, SSE (role,
              reasoning / content deltas, tool_calls, finish, usage, [DONE]), errors (400 / 401 / 404 / 411), auth,
              two concurrent clients serialized.
  tokenizer   (needs `tokenizers` + C:/models/GLM53-backbone/tokenizer.json) real BPE: added tokens split the text,
              so the next turn's prompt ids extend prompt + answer ids exactly (thinking off) / up to the prompt end
              (thinking on).
  torch       (needs torch; CUDA hidden) GLMRunnerAdapter state copies on CPU tensors: snapshot / eager export /
              graph export (indexer gap) / restore, no aliasing.
  transformers (opt-in) jinja2 render == AutoTokenizer.apply_chat_template, encode_chat-style ids == engine ids.
Runtime dir: env R34_RUNTIME, else ../claude_r35_speed.
"""
import argparse
import json
import os
import random
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # never touch the GPU (torch imported by the optional groups)
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("RAYON_NUM_THREADS", "2")

HERE = os.path.dirname(os.path.abspath(__file__))
RUNTIME = os.environ.get("R34_RUNTIME") or os.path.join(os.path.dirname(HERE), "runtime")
sys.path.insert(0, RUNTIME)
import glm53_fold_serve as S  # noqa: E402

BACKBONE = os.environ.get("R34_BACKBONE") or "C:/models/GLM53-backbone"
TEMPLATE_PATH = os.path.join(BACKBONE, "chat_template.jinja")
TPL_SRC = open(TEMPLATE_PATH, "r", encoding="utf-8").read() if os.path.exists(TEMPLATE_PATH) else S.BUILTIN_GLM53_SENTINEL

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"},
                                                    "units": {"type": "string", "enum": ["c", "f"]}},
                   "required": ["city"]}}}
CALC = {"type": "function", "function": {
    "name": "calc", "description": "Evaluate arithmetic",
    "parameters": {"type": "object", "properties": {"expr": {"type": "string"}, "precision": {"type": "number"},
                                                    "exact": {"type": "boolean"}, "vars": {"type": "object"},
                                                    "list": {"type": "array"},
                                                    "maybe": {"anyOf": [{"type": "integer"}, {"type": "string"}]}}}}}
SEARCH_FLAT = {"type": "function", "name": "search", "description": "flat tool format",
               "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}
DEFER = {"type": "function", "function": {"name": "hidden", "defer_loading": True, "parameters": {}}}
STRICT = {"type": "function", "function": {"name": "strict_tool", "strict": True,
                                           "parameters": {"type": "object", "properties": {"a": {"type": "string"}}}}}


def tc(i, name, args):
    return {"id": i, "type": "function", "function": {"name": name, "arguments": args}}


# 17 conversations (raw OpenAI form; normalized before rendering)
CONVS = {
    "single": dict(messages=[{"role": "user", "content": "Hello"}]),
    "system": dict(messages=[{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]),
    "developer": dict(messages=[{"role": "developer", "content": "Dev rules"}, {"role": "user", "content": "Hi"}]),
    "multiturn": dict(messages=[
        {"role": "system", "content": "You are a math tutor."}, {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4", "reasoning_content": "add"}, {"role": "user", "content": "times 3?"},
        {"role": "assistant", "content": "  12 \n", "reasoning": "multiply"}, {"role": "user", "content": "thanks"}]),
    "think_in_content": dict(messages=[
        {"role": "user", "content": "q"}, {"role": "assistant", "content": "x<think>inner</think>  final  "},
        {"role": "user", "content": "q2"}, {"role": "assistant", "content": "pre<think>kept</think>post"}]),
    "tools_question": dict(messages=[{"role": "user", "content": "Weather in Paris?"}], tools=[WEATHER, CALC]),
    "tool_loop": dict(messages=[
        {"role": "user", "content": "Weather in Paris for 2 days?"},
        {"role": "assistant", "content": None, "reasoning_content": "need the tool",
         "tool_calls": [tc("c1", "get_weather", json.dumps({"city": "Paris", "days": 2}))]},
        {"role": "tool", "tool_call_id": "c1", "content": "{\"temp\": 20, \"sky\": \"clear\"}"}], tools=[WEATHER]),
    "parallel_out_of_order": dict(messages=[
        {"role": "user", "content": "Paris and Rome?"},
        {"role": "assistant", "content": "Checking both.", "tool_calls": [
            tc("a1", "get_weather", {"city": "Paris"}), tc("a2", "get_weather", {"city": "Rome", "units": "c"})]},
        {"role": "tool", "tool_call_id": "a2", "content": "Rome 25"},
        {"role": "tool", "tool_call_id": "a1", "content": "Paris 20"},
        {"role": "assistant", "content": "Paris 20, Rome 25."}, {"role": "user", "content": "and 2*21?"},
        {"role": "assistant", "content": "", "tool_calls": [tc("b1", "calc", {"expr": "2*21", "precision": 0.5,
                                                                              "exact": True, "vars": {"x": [1, None]},
                                                                              "list": [], "maybe": "s"})]},
        {"role": "tool", "tool_call_id": "b1", "content": "42"}], tools=[WEATHER, CALC]),
    "ids_missing_dup": dict(messages=[
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [tc("d", "calc", {"expr": "1"}), tc("d", "calc", {"expr": "2"})]},
        {"role": "tool", "tool_call_id": "d", "content": "1"}, {"role": "tool", "content": "2"},
        {"role": "user", "content": "next"}, {"role": "tool", "tool_call_id": "zz", "content": "orphan"}], tools=[CALC]),
    "legacy_functions": dict(messages=[
        {"role": "user", "content": "calc 3+4"},
        {"role": "assistant", "content": None, "function_call": {"name": "calc", "arguments": "{\"expr\": \"3+4\"}"}},
        {"role": "function", "name": "calc", "content": "7"}], functions=[CALC["function"]]),
    "clear_thinking_false": dict(messages=[
        {"role": "user", "content": "a"}, {"role": "assistant", "content": "b", "reasoning_content": "R-old"},
        {"role": "user", "content": "c"}], kwargs={"clear_thinking": False}),
    "unicode_effort_high": dict(messages=[
        {"role": "user", "content": "Ünïcödé ∑ π ≈ 3.14159 — emoji 🚀 and 中文 text"},
        {"role": "assistant", "content": "ok ✓"}, {"role": "user", "content": "\"quotes\" \\ backslash <tag>"}],
        kwargs={"reasoning_effort": "high"}),
    "defer_strict_flat": dict(messages=[{"role": "user", "content": "x"}], tools=[DEFER, STRICT, SEARCH_FLAT]),
    "content_parts": dict(messages=[
        {"role": "system", "content": [{"type": "text", "text": "S1 "}, {"type": "text", "text": "S2"}]},
        {"role": "user", "content": [{"type": "text", "text": "a"}, "b", {"type": "input_text", "text": "c"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "ans"}],
         "tool_calls": [tc("p1", "calc", "{\"expr\": \"1+1\"}")]},
        {"role": "tool", "tool_call_id": "p1", "content": [{"type": "text", "text": "2"}]}], tools=[CALC]),
    "agent_multi_step": dict(messages=[
        {"role": "system", "content": "agent"}, {"role": "user", "content": "plan"},
        {"role": "assistant", "content": "step1", "reasoning_content": "r1", "tool_calls": [tc(7, "calc", {"expr": "1"})]},
        {"role": "tool", "tool_call_id": 7, "content": "1"},
        {"role": "assistant", "content": "step2", "reasoning_content": "r2", "tool_calls": [tc("8", "calc", {"expr": "2"})]},
        {"role": "tool", "tool_call_id": "8", "content": "2"},
        {"role": "system", "content": "mid-conversation system note"},
        {"role": "assistant", "content": "done", "reasoning_content": "r3"}], tools=[CALC], kwargs={"reasoning_effort": "max"}),
    "empty_tools": dict(messages=[{"role": "user", "content": "plain"}], tools=[]),
    "empty_strings": dict(messages=[
        {"role": "system", "content": ""}, {"role": "user", "content": ""}, {"role": "assistant", "content": ""},
        {"role": "user", "content": " "}], kwargs={"reasoning_effort": "low"}),
}

PASS, FAIL, SKIP = [], [], []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def norm_conv(c):
    msgs = S.normalize_messages(c["messages"])
    tools = S.normalize_tools(c.get("tools"), c.get("functions"))
    return msgs, tools, dict(c.get("kwargs", {}))


# ============================================================================================ template
def test_template_python_vs_jinja2():
    if TPL_SRC == S.BUILTIN_GLM53_SENTINEL:
        return "SKIP chat_template.jinja not mounted on offline host (verified by test_template_expected_strings)"
    jt = S.ChatTemplate(TPL_SRC, engine="jinja2", origin=TEMPLATE_PATH)
    pt = S.ChatTemplate(TPL_SRC, engine="python", origin=TEMPLATE_PATH)
    check(jt.engine == "jinja2" and pt.engine == "python" and jt.is_glm53, "engines / template hash")
    n = 0
    for name, c in CONVS.items():
        msgs, tools, kw = norm_conv(c)
        for gen in (True, False):
            for eff in ([kw.get("reasoning_effort")] if "reasoning_effort" in kw else [None, "low", "high", "max", "LOW"]):
                kk = {k: v for k, v in kw.items() if k != "reasoning_effort"}
                if eff is not None:
                    kk["reasoning_effort"] = eff
                a = jt.render(msgs, tools, add_generation_prompt=gen, **kk)
                b = pt.render(msgs, tools, add_generation_prompt=gen, **kk)
                if a != b:
                    i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
                    raise AssertionError(f"{name} gen={gen} eff={eff}: python port differs at {i}: "
                                         f"jinja2 {a[max(0, i - 40):i + 40]!r} vs python {b[max(0, i - 40):i + 40]!r}")
                n += 1
    check(len(CONVS) >= 10 and n >= 60, f"coverage {len(CONVS)} conversations / {n} renders")
    return f"{len(CONVS)} conversations, {n} renders identical"


def test_template_expected_strings():
    jt = S.ChatTemplate(TPL_SRC, engine="jinja2")
    r = lambda c, **k: jt.render(*norm_conv(CONVS[c])[:2], add_generation_prompt=True, **k)  # noqa: E731
    check(r("single", reasoning_effort="low") ==
          "[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>Hello<|assistant|><think>", "single")
    check(r("single") == "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Hello<|assistant|><think>", "default Max")
    m = r("multiturn", reasoning_effort="low")
    check("<|user|>2+2?<|assistant|><think></think>4<|user|>times 3?<|assistant|><think></think>12<|user|>thanks"
          "<|assistant|><think>" in m, "multi-turn: reasoning cleared before the last user, content stripped")
    t = r("tool_loop", reasoning_effort="low")
    check(t.endswith("<|user|>Weather in Paris for 2 days?<|assistant|><think>need the tool</think><tool_call>get_weather"
                     "<arg_key>city</arg_key><arg_value>Paris</arg_value><arg_key>days</arg_key><arg_value>2</arg_value>"
                     "</tool_call><|observation|><tool_response>{\"temp\": 20, \"sky\": \"clear\"}</tool_response>"
                     "<|assistant|><think>"), "tool loop: reasoning kept after the last user, JSON-string args decoded, "
                                             "null content not printed as None")
    p = r("parallel_out_of_order", reasoning_effort="low")
    check("<|observation|><tool_response>Paris 20</tool_response><tool_response>Rome 25</tool_response>" in p,
          "tool results sorted by the assistant's call order")
    check("<arg_key>vars</arg_key><arg_value>{\"x\": [1, null]}</arg_value>" in p and
          "<arg_key>exact</arg_key><arg_value>true</arg_value>" in p, "non-string args as JSON")
    d = r("defer_strict_flat", reasoning_effort="low")
    check("hidden" not in d and "\"strict\"" not in d and "{\"name\": \"search\"" in d, "defer_loading / strict / flat tools")
    e = S.ServeEngine(S.FakeRunner(), jt)
    prep_off = e.prepare({"messages": CONVS["single"]["messages"], "enable_thinking": False})
    prep_on = e.prepare({"messages": CONVS["single"]["messages"], "chat_template_kwargs": {"enable_thinking": True}})
    check(prep_off.text == r("single", reasoning_effort="low") + "</think>", "thinking off = encode_chat rule (+</think>)")
    check(prep_on.text == r("single", reasoning_effort="low"), "thinking on ends with <think>")
    check(e.prepare({"messages": [{"role": "user", "content": "x"}], "thinking": {"type": "enabled"},
                     "reasoning_effort": "medium"}).text.startswith("[gMASK]<sop><|system|>Reasoning Effort: High"),
          "thinking: {type: enabled}, medium -> high")
    check(e.prepare({"messages": [{"role": "user", "content": "x"}], "tools": [WEATHER], "tool_choice": "none"}).text
          .find("# Tools") < 0, "tool_choice none drops the tools block")
    return "fixed strings + encode_chat thinking rule"


# ============================================================================================ normalize
def test_normalize_and_errors():
    m = S.normalize_messages([{"role": "assistant", "content": None, "tool_calls": [
        {"id": "x", "type": "function", "function": {"name": "f", "arguments": ""}}]}])
    check(m[0]["content"] == "" and m[0]["tool_calls"][0]["function"]["arguments"] == {}, "null content / empty args")
    bad = [
        ([], "messages"), ([{"role": "robot", "content": "x"}], "messages[0].role"),
        ([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "u"}}]}], "messages[0].content[0]"),
        ([{"role": "user", "content": {"a": 1}}], "messages[0].content"),
        ([{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f", "arguments": "{bad"}}]}],
         "messages[0].tool_calls[0].function.arguments"),
        ([{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f", "arguments": "[1]"}}]}],
         "messages[0].tool_calls[0].function.arguments"),
        ([{"role": "assistant", "content": "", "tool_calls": [{"function": {"arguments": "{}"}}]}],
         "messages[0].tool_calls[0].function.name"),
    ]
    for raw, param in bad:
        try:
            S.normalize_messages(raw)
            raise AssertionError(f"no error for {raw}")
        except S.RequestError as e:
            check(e.status == 400 and e.param == param, f"{raw}: status {e.status} param {e.param} (want {param})")
    for tools, ok in (([{"type": "web_search"}], False), ([{"type": "function"}], False), ([WEATHER, SEARCH_FLAT], True)):
        try:
            S.normalize_tools(tools)
            check(ok, f"tools {tools} accepted")
        except S.RequestError:
            check(not ok, f"tools {tools} refused")
    eng = S.ServeEngine(S.FakeRunner(), S.ChatTemplate(TPL_SRC), max_context=200)
    for body, param in (({"messages": [{"role": "user", "content": "x"}], "n": 2}, "n"),
                        ({"messages": [{"role": "user", "content": "x"}], "logprobs": True}, "logprobs"),
                        ({"messages": [{"role": "user", "content": "x"}], "max_tokens": 0}, "max_tokens"),
                        ({"messages": [{"role": "user", "content": "x"}], "stop": 5}, "stop"),
                        ({"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "huge"}, "reasoning_effort"),
                        ({"messages": [{"role": "user", "content": "y" * 400}]}, "messages")):
        try:
            eng.prepare(body)
            raise AssertionError(f"no error for {param}")
        except S.RequestError as e:
            check(e.param == param, f"{param}: got {e.param}")
    eng2 = S.ServeEngine(S.FakeRunner(), S.ChatTemplate(TPL_SRC), max_context=4000)
    p = eng2.prepare({"messages": [{"role": "user", "content": "x"}], "max_tokens": 10 ** 6, "temperature": 0.7,
                      "top_p": 0.9, "seed": 1, "response_format": {"type": "json_object"}, "tool_choice": "required",
                      "tools": [WEATHER]})
    check(p.max_tokens == 4000 - len(p.ids) and any("capped" in n for n in p.notes), "max_tokens capped with a note")
    check(set(p.ignored) >= {"response_format", "tool_choice"}, f"ignored {p.ignored}")
    # [R3.9SMP] temperature / top_p / seed are no longer ignored: they are parsed into the request's sampling settings
    check(not set(p.ignored) & {"temperature", "top_p", "seed"} and p.sampling.get("temperature") == 0.7
          and p.sampling.get("top_p") == 0.9, f"sampling {p.sampling} ignored {p.ignored}")
    return "normalization + 13 error cases"


# ============================================================================================ parse
def test_tool_call_parsing():
    schemas = {t["function"]["name"]: t["function"] for t in (WEATHER, CALC)}
    out = ("Plan: call tools.</think>\n\nI'll check.<tool_call>get_weather<arg_key>city</arg_key><arg_value> New York\n"
           "</arg_value><arg_key>days</arg_key><arg_value>3</arg_value></tool_call>\n<tool_call>calc\n<arg_key>expr"
           "</arg_key>\n<arg_value>2*(3+4)</arg_value><arg_key>precision</arg_key><arg_value>2</arg_value><arg_key>exact"
           "</arg_key><arg_value>false</arg_value><arg_key>vars</arg_key><arg_value>{\"a\": 1}</arg_value><arg_key>list"
           "</arg_key><arg_value>[1, 2]</arg_value><arg_key>maybe</arg_key><arg_value>12</arg_value></tool_call>")
    p = S.parse_generation(out, thinking=True, schemas=schemas)
    check(p.reasoning == "Plan: call tools." and p.content == "I'll check.", f"split {p.reasoning!r} / {p.content!r}")
    check(len(p.tool_calls) == 2, "two calls")
    a0, a1 = p.tool_calls[0]["arguments"], p.tool_calls[1]["arguments"]
    check(a0 == {"city": " New York\n", "days": 3}, f"string raw verbatim / integer typed: {a0}")
    check(a1 == {"expr": "2*(3+4)", "precision": 2, "exact": False, "vars": {"a": 1}, "list": [1, 2], "maybe": 12},
          f"typed args {a1}")
    p2 = S.parse_generation("<tool_call>get_weather<arg_key>days</arg_key><arg_value>soon</arg_value></tool_call>",
                            thinking=False, schemas=schemas)
    check(p2.tool_calls[0]["arguments"] == {"days": "soon"} and p2.content == "", "invalid integer kept raw")
    p3 = S.parse_generation("x<tool_call>unknown<arg_key>n</arg_key><arg_value>5</arg_value><arg_key>s</arg_key>"
                            "<arg_value>hi</arg_value></tool_call>", thinking=False, schemas=schemas)
    check(p3.tool_calls[0] == {"name": "unknown", "arguments": {"n": 5, "s": "hi"}} and p3.content == "x",
          "no schema: JSON when it parses")
    p4 = S.parse_generation("text <tool_call></tool_call> mid <tool_call>calc<arg_key>expr</arg_key>"
                            "<arg_value>1</arg_value>", thinking=False, schemas=schemas)
    check(p4.tool_calls == [] and p4.content == "text <tool_call></tool_call> mid <tool_call>calc<arg_key>expr</arg_key>"
          "<arg_value>1</arg_value>", "empty-name block and unterminated block stay in content")
    p5 = S.parse_generation("all reasoning, cut by max_tokens", thinking=True, schemas=schemas)
    check(p5.reasoning == "all reasoning, cut by max_tokens" and p5.content == "", "unclosed reasoning")
    p6 = S.parse_generation("<think>x</think>y", thinking=False, schemas=schemas, parse_tools=False)
    check(p6.reasoning is None and p6.content == "<think>x</think>y", "thinking off: no split")
    p7 = S.parse_generation("<tool_call>calc</tool_call>", thinking=False, schemas=schemas)
    check(p7.tool_calls == [{"name": "calc", "arguments": {}}], "call without arguments")
    # round trip: the model's canonical markup -> OpenAI message -> client echo -> render == the generated text
    jt = S.ChatTemplate(TPL_SRC, engine="jinja2")
    gen = ("look it up</think>Sure.<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>"
           "<arg_key>days</arg_key><arg_value>2</arg_value></tool_call>")
    q = S.parse_generation(gen, thinking=True, schemas=schemas)
    echo = {"role": "assistant", "content": q.content, "reasoning_content": q.reasoning,
            "tool_calls": [{"id": "c9", "type": "function", "function": {
                "name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}} for c in q.tool_calls]}
    base = [{"role": "user", "content": "w?"}]
    pre = jt.render(S.normalize_messages(base), [WEATHER], add_generation_prompt=True, reasoning_effort="low")
    full = jt.render(S.normalize_messages(base + [echo, {"role": "tool", "tool_call_id": "c9", "content": "20C"}]),
                     [WEATHER], add_generation_prompt=True, reasoning_effort="low")
    check(full.startswith(pre + gen + "<|observation|>"), "echoed tool-call turn re-renders the generated text exactly")
    return "schema typing, malformed blocks, reasoning split, exact echo round trip"


# ============================================================================================ stream
def _stream_all(text, thinking, stops, parse_tools, chunks):
    asm = S.OutputAssembler(thinking=thinking, stops=stops, parse_tools=parse_tools)
    r, c = [], []
    pos = 0
    for k in chunks:
        for kind, t in asm.feed(text[pos:pos + k]):
            (r if kind == "reasoning" else c).append(t)
        pos += k
        if asm.stopped:
            break
    if not asm.stopped and pos < len(text):
        for kind, t in asm.feed(text[pos:]):
            (r if kind == "reasoning" else c).append(t)
    ev, parsed = asm.finish({t["function"]["name"]: t["function"] for t in (WEATHER, CALC)})
    for kind, t in ev:
        (r if kind == "reasoning" else c).append(t)
    return "".join(r), "".join(c), parsed


def test_stream_assembler_equivalence():
    texts = [
        ("Reasoning with </thi partial tags </think>\n  Answer </think> text <tool_c x", True),
        ("abc</think>Hello <tool_call>calc<arg_key>expr</arg_key><arg_value>1</arg_value></tool_call> tail", True),
        ("  \n Leading ws then STOP here and more", False),
        ("x" * 50 + "</think>" + "y" * 30 + "<tool_call>get_weather</tool_call><tool_call>calc</tool_call>end", True),
        ("no markers at all, just text with < and </ and <tool and </think", False),
        ("ünïcödé 🚀 reasoning</think>中文 answer ✓<tool_call>calc<arg_key>expr</arg_key><arg_value>π</arg_value>"
         "</tool_call>", True),
        ("A <tool_call>calc</tool_call> B", False),
        ("stop spans <tool_call>calc</tool_call>", False),
    ]
    stops_sets = [[], ["STOP"], ["here", "an"], ["</tool_call>"], [" <tool_call>c"], ["\n\n"]]
    rng = random.Random(1234)
    n = 0
    for text, thinking in texts:
        for stops in stops_sets:
            for parse_tools in (True, False):
                ref = S.parse_generation(text, thinking=thinking, stops=stops, parse_tools=parse_tools,
                                         schemas={t["function"]["name"]: t["function"] for t in (WEATHER, CALC)})
                for trial in range(12):
                    if trial == 0:
                        chunks = [1] * len(text)
                    else:
                        chunks, left = [], len(text)
                        while left > 0:
                            k = rng.randint(1, 9)
                            chunks.append(k)
                            left -= k
                    r, c, parsed = _stream_all(text, thinking, stops, parse_tools, chunks)
                    check(parsed.content == ref.content and parsed.reasoning == ref.reasoning and
                          parsed.tool_calls == ref.tool_calls and parsed.stop_string == ref.stop_string,
                          f"chunked parse != one-shot for {text!r} stops={stops}")
                    check(r == (ref.reasoning or ""), f"streamed reasoning {r!r} != {ref.reasoning!r} ({text!r}, {stops})")
                    check(c.rstrip() == ref.content, f"streamed content {c!r} != {ref.content!r} ({text!r}, {stops}, "
                                                     f"tools={parse_tools}, chunks={chunks})")
                    n += 1
    p = S.parse_generation("Answer: 42STOP ignored", thinking=False, stops=["STOP"])
    check(p.content == "Answer: 42" and p.stop_string == "STOP", "stop string removed")
    p = S.parse_generation("think STOP inside</think>out", thinking=True, stops=["STOP"])
    check(p.reasoning == "think STOP inside" and p.content == "out" and p.stop_string is None,
          "stop strings do not apply inside the reasoning")
    tok = S.FakeTokenizer()
    for s in ("héllo wörld 🚀 中文 ∑", "<think>a</think>b<tool_call>c", "plain"):
        ids = tok.encode(s)
        d = S.IncrementalDetokenizer(tok.decode, skip_ids=S.GLM53_EOS_IDS)
        parts = [d.push(i) for i in ids + [154827]]
        check(not any("\ufffd" in x for x in parts), f"no partial characters released for {s!r}")
        check("".join(parts) + d.flush() == s, f"incremental detokenizer for {s!r}")
    return f"{n} chunkings identical to the one-shot parse; detokenizer OK"


# ============================================================================================ reuse decisions
def _out(n_state, gen, points, gap=None, nbytes=10):
    return S.GenerateOutcome(generated_ids=gen, n_state=n_state, store={"n": n_state}, store_nbytes=nbytes,
                             points=points, indexer_gap=gap)


def test_cache_decisions():
    rel = []
    cc = S.ConversationCache(max_entries=2, max_points=3, max_bytes=10 ** 9, min_reuse=4, release=rel.append)
    P1 = list(range(100, 140))                                       # 40-token prompt
    G1 = [1, 2, 3, 4, 5, 154827]                                     # answer + EOS
    pl = cc.plan(P1)
    check(pl.point is None and pl.keep == 0, "empty cache -> fresh")
    pe = S.RestorePoint(40, handle="h40", next_token=1, nbytes=5, kind="prompt-end")
    fi = S.RestorePoint(45, handle="h45", nbytes=5, kind="final")      # EOS not processed: 40 + 6 - 1
    cc.commit(pl, P1, _out(45, G1, [pe, fi]))
    P2 = P1 + G1[:-1] + [154827, 7, 8, 9]                             # thinking-off follow-up: extends the sequence
    pl = cc.plan(P2)
    check(pl.keep == 45 and pl.point is fi and not pl.skip_prefill, f"follow-up restores the final state: {pl}")
    P2b = P1 + [999, 7, 8]                                            # thinking-on follow-up: diverges at the prompt end
    pl = cc.plan(P2b)
    check(pl.keep == 40 and pl.point is pe, "diverging answer -> prompt-end point")
    pl = cc.plan(P1)
    check(pl.keep == 0, "identical prompt without skip-prefill -> no point (needs n <= len-1)")
    pl = cc.plan(P1, allow_skip_prefill=True)
    check(pl.keep == 40 and pl.skip_prefill, "identical prompt with skip-prefill -> prompt-end point, no prefill")
    pl = cc.plan(P1[:3] + [5] * 50)
    check(pl.keep == 0 and pl.lcp == 3, "short common prefix below min_reuse -> fresh")
    pl = cc.plan(P1[:39] + [0])
    check(pl.keep == 0, "edit inside the first prompt -> no point at or below the common prefix")
    # commit of the follow-up inherits points <= keep, releases the superseded store and non-inherited points
    pl = cc.plan(P2)
    G2 = [11, 12, 154827]
    pe2 = S.RestorePoint(len(P2), handle="hP2", next_token=11, nbytes=5, kind="prompt-end")
    fi2 = S.RestorePoint(len(P2) + 2, handle="hF2", nbytes=5, kind="final")
    e = cc.commit(pl, P2, _out(len(P2) + 2, G2, [pe2, fi2]))
    check([p.n_tokens for p in e.points] == [len(P2) + 2, len(P2), 45], f"points {[p.n_tokens for p in e.points]}")
    check(len(cc.entries) == 1 and {"n": 45} in rel and "h40" in rel, f"superseded store + dropped point released {rel}")
    # max_points trims the smallest; LRU entries beyond max_entries are evicted
    cc2 = S.ConversationCache(max_entries=2, max_points=2, max_bytes=10 ** 9, min_reuse=1, release=rel.append)
    for k in range(3):
        P = [500 + k] * 10
        cc2.commit(cc2.plan(P), P, _out(12, [1, 2, 154827], [S.RestorePoint(10, f"a{k}", 1, 1),
                                                           S.RestorePoint(12, f"b{k}", None, 1)]))
    check(len(cc2.entries) == 2 and cc2.entries[0].tokens[0] == 502 and "a0" in rel and "b0" in rel, "LRU eviction")
    # byte budget: older entries first, then the smallest points of the newest entry, then the entry
    cc3 = S.ConversationCache(max_entries=4, max_points=4, max_bytes=100, min_reuse=1)
    cc3.commit(cc3.plan([1] * 5), [1] * 5, _out(6, [2, 3], [S.RestorePoint(5, "x", 2, 30), S.RestorePoint(6, "y", None, 30)], nbytes=10))
    cc3.commit(cc3.plan([9] * 5), [9] * 5, _out(6, [2, 3], [S.RestorePoint(5, "p", 2, 30), S.RestorePoint(6, "q", None, 30)], nbytes=10))
    check(len(cc3.entries) == 1 and cc3.entries[0].tokens[0] == 9 and cc3.total_bytes() <= 100, "budget evicts LRU entry")
    cc4 = S.ConversationCache(max_entries=4, max_points=4, max_bytes=50, min_reuse=1)
    cc4.commit(cc4.plan([1] * 5), [1] * 5, _out(6, [2, 3], [S.RestorePoint(5, "x", 2, 30), S.RestorePoint(6, "y", None, 30)], nbytes=10))
    check(len(cc4.entries) == 1 and [p.n_tokens for p in cc4.entries[0].points] == [6], "budget drops the smallest point")
    # indexer gap: points above the gap only when the runner says the request cannot read the placeholders
    cc5 = S.ConversationCache(max_entries=2, max_points=3, min_reuse=1)
    PA = list(range(10, 60))
    cc5.commit(cc5.plan(PA), PA, _out(55, [1] * 5 + [154827], [S.RestorePoint(50, "pe", 1, 1), S.RestorePoint(55, "fi", None, 1)], gap=50))
    PB = PA + [1] * 5 + [154827, 3, 4]
    check(cc5.plan(PB, allow_gap_points=True).keep == 55, "gap: safe request may use the point above the gap")
    check(cc5.plan(PB, allow_gap_points=False).keep == 50, "gap: unsafe request -> the point below the gap")
    fr = S.FakeRunner(index_topk=100)
    check(fr.indexer_gap_safe(58, 10) and not fr.indexer_gap_safe(58, 100) and
          not S.FakeRunner(index_topk=100, indexer_short_ok=False).indexer_gap_safe(58, 10), "FakeRunner gap rule")
    # the gap is inherited by the next entry while it lies inside the restored prefix
    pl = cc5.plan(PB, allow_gap_points=True)
    e = cc5.commit(pl, PB, _out(len(PB) + 1, [5, 6, 154827], [S.RestorePoint(len(PB), "pe2", 5, 1),
                                                              S.RestorePoint(len(PB) + 1, "fi2", None, 1)]))
    check(e.indexer_gap == 50, f"inherited gap {e.indexer_gap}")
    check(S.common_prefix_len(list(range(1000)), list(range(700)) + [0]) == 700, "common_prefix_len")
    # retention: the newest final, the newest prompt-end and the newest anchor survive trimming (agentic loops)
    pts = [S.RestorePoint(10, "a", 1, 1, kind="prompt-end", anchor=True), S.RestorePoint(20, "b", None, 1, kind="final"),
           S.RestorePoint(25, "c", 2, 1, kind="prompt-end"), S.RestorePoint(30, "d", None, 1, kind="final"),
           S.RestorePoint(35, "e", 3, 1, kind="prompt-end")]
    check([p.n_tokens for p in S.ConversationCache._select_points(pts, 3)] == [35, 30, 10], "anchor kept")
    check([p.n_tokens for p in S.ConversationCache._select_points(pts, 4)] == [35, 30, 25, 10], "then largest n")
    cc7 = S.ConversationCache(max_entries=1, max_points=5, max_bytes=3, min_reuse=1)
    cc7.commit(cc7.plan([1] * 35), [1] * 35, _out(36, [2, 3], pts, nbytes=0))
    check([p.n_tokens for p in cc7.entries[0].points] == [35, 30, 10], "budget drops unprotected points first")
    return "LCP / points / skip-prefill / min_reuse / inheritance / limits / indexer gap / anchors"


def _strip_call_ids(msg):
    m = dict(msg)
    if m.get("tool_calls"):
        m["tool_calls"] = [{k: v for k, v in t.items() if k != "id"} for t in m["tool_calls"]]
    return m


def _agent_script(prompt_text, ids, h):
    tail = prompt_text.rsplit("<|user|>", 1)[-1]
    n_res = tail.count("<tool_response>")
    pre = f"step {n_res} reasoning {h % 13}</think>" if prompt_text.endswith("<think>") else ""
    if "summarize" in tail.lower():
        return pre + "Summary done."
    if n_res < 3:
        return (pre + f"Calling.<tool_call>get_weather<arg_key>city</arg_key><arg_value>City{n_res}</arg_value>"
                "</tool_call><|observation|>")
    return pre + "All cities checked."


def test_engine_agentic_anchor():
    """Thinking-on tool loop (reasoning echoed back, as this server returns it): every step restores the previous
    final state; the next USER turn clears all earlier reasoning, so its prompt diverges right after the loop's first
    prompt - the protected anchor point must still be there (3 points per entry, 4 loop steps)."""
    tpl = S.ChatTemplate(TPL_SRC)
    ra, rb = S.FakeRunner(script=_agent_script), S.FakeRunner(script=_agent_script)
    ea = S.ServeEngine(ra, tpl, default_thinking=True, min_reuse=8, log=lambda s: None)
    eb = S.ServeEngine(rb, tpl, default_thinking=True, reuse=False, log=lambda s: None)
    hist = [{"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "Check the weather in three cities."}]
    for step in range(6):
        a, b = _chat(ea, {"messages": hist, "tools": [WEATHER]}), _chat(eb, {"messages": hist, "tools": [WEATHER]})
        check(_strip_call_ids(a["choices"][0]["message"]) == _strip_call_ids(b["choices"][0]["message"]),
              f"step {step}: reuse changed the answer")
        msg = _assistant_echo(a, with_reasoning=True)
        hist.append(msg)
        if a["choices"][0]["finish_reason"] != "tool_calls":
            break
        for tcall in msg["tool_calls"]:
            hist.append({"role": "tool", "tool_call_id": tcall["id"], "content": "sunny"})
    check(len(ra.calls) == 4 and a["choices"][0]["message"]["content"] == "All cities checked.", f"loop {ra.calls}")
    check(all(c["keep"] > ra.calls[i - 1]["prompt"] for i, c in enumerate(ra.calls) if i > 0),
          f"tool steps restore the previous final state {ra.calls}")
    first_prompt = ra.calls[0]["prompt"]
    hist.append({"role": "user", "content": "Now summarize."})
    a, b = _chat(ea, {"messages": hist, "tools": [WEATHER]}), _chat(eb, {"messages": hist, "tools": [WEATHER]})
    check(_strip_call_ids(a["choices"][0]["message"]) == _strip_call_ids(b["choices"][0]["message"]) and
          a["choices"][0]["message"]["content"] == "Summary done.", "summary turn")
    check(ra.calls[-1]["keep"] == first_prompt, f"new user turn restores the anchor (keep {ra.calls[-1]['keep']}, "
                                                f"first prompt {first_prompt})")
    return f"4 tool steps restored the previous final state; the user turn restored the anchor @{first_prompt}"


def _hash_script(prompt_text, ids, h):
    if prompt_text.endswith("<think>"):
        return f"reason-{h % 97}</think>answer-{h % 89}"
    return f"answer-{h % 89}"


def _chat(engine, body):
    prep = engine.prepare(body)
    return engine.execute(prep)


def _assistant_echo(resp, with_reasoning=False):
    m = dict(resp["choices"][0]["message"])
    if not with_reasoning:
        m.pop("reasoning_content", None)
    return m


def test_engine_reuse_oracle():
    tpl = S.ChatTemplate(TPL_SRC)
    results = []
    for variant in ("off", "on", "on_echo", "mtp_like", "no_skip"):
        thinking = variant != "off"
        caps = None if variant != "no_skip" else ("reuse", "on_token", "stream")
        ra = S.FakeRunner(script=_hash_script, process_last_token=(variant == "mtp_like"), capabilities=caps)
        rb = S.FakeRunner(script=_hash_script, process_last_token=(variant == "mtp_like"), capabilities=caps)
        ea = S.ServeEngine(ra, tpl, default_thinking=thinking, min_reuse=8, log=lambda s: None)
        eb = S.ServeEngine(rb, tpl, default_thinking=thinking, reuse=False, log=lambda s: None)
        hist = [{"role": "system", "content": "You are terse."}]
        for turn, (umsg, mt) in enumerate((("first question", None), ("second, longer question?", 4),
                                           ("third", None), ("fourth", None))):
            hist.append({"role": "user", "content": umsg})
            body = {"messages": hist, "max_tokens": mt}
            a, b = _chat(ea, body), _chat(eb, body)
            check(a["choices"][0]["message"] == b["choices"][0]["message"] and
                  a["choices"][0]["finish_reason"] == b["choices"][0]["finish_reason"],
                  f"{variant} turn {turn}: reuse changed the answer {a['choices'][0]} vs {b['choices'][0]}")
            hist.append(_assistant_echo(a, with_reasoning=(variant == "on_echo")))
        # regenerate the last turn (identical prompt) and edit an early message
        regen = {"messages": hist[:-1]}
        a, b = _chat(ea, regen), _chat(eb, regen)
        check(a["choices"][0]["message"] == b["choices"][0]["message"], f"{variant}: regenerate")
        edited = [dict(m) for m in hist[:-1]]
        edited[1]["content"] = "first question (edited)"
        a, b = _chat(ea, {"messages": edited}), _chat(eb, {"messages": edited})
        check(a["choices"][0]["message"] == b["choices"][0]["message"], f"{variant}: edited history")
        pa = sum(c["prefilled"] for c in ra.calls)
        pb = sum(c["prefilled"] for c in rb.calls)
        reused = [c["keep"] for c in ra.calls]
        check(pa < pb and sum(1 for k in reused if k > 0) >= 3, f"{variant}: reuse ineffective {ra.calls}")
        if variant in ("off", "on", "mtp_like"):
            check(ra.calls[-2]["skip"], f"{variant}: regenerate should skip the prefill ({ra.calls[-2]})")
        if variant == "no_skip":
            check(not any(c["skip"] for c in ra.calls), "no skip without the capability")
        if variant == "off":
            check(ra.calls[2]["keep"] > ra.calls[1]["prompt"], "thinking off: next turn restores the FINAL state")
        if variant == "on":
            check(all(c["keep"] in (0, ra.calls[i - 1]["prompt"]) for i, c in enumerate(ra.calls) if 0 < i < 4),
                  f"thinking on (reasoning cleared): restores the previous PROMPT-END state {ra.calls}")
        results.append(f"{variant}: prefilled {pa} vs {pb}")
    # stop string ending: the state covers what was generated; the next turn still matches the no-reuse answer
    ra, rb = S.FakeRunner(script=lambda t, i, h: f"alpha beta STOP gamma {h % 7}"), S.FakeRunner(
        script=lambda t, i, h: f"alpha beta STOP gamma {h % 7}")
    ea = S.ServeEngine(ra, tpl, min_reuse=8, log=lambda s: None)
    eb = S.ServeEngine(rb, tpl, reuse=False, log=lambda s: None)
    hist = [{"role": "user", "content": "go"}]
    for k in range(3):
        a, b = _chat(ea, {"messages": hist, "stop": ["STOP"]}), _chat(eb, {"messages": hist, "stop": "STOP"})
        check(a["choices"][0]["message"]["content"] == "alpha beta" == b["choices"][0]["message"]["content"] and
              a["choices"][0]["finish_reason"] == "stop", f"stop turn {k}: {a['choices'][0]}")
        hist += [_assistant_echo(a), {"role": "user", "content": f"more {k}"}]
    check(sum(1 for c in ra.calls if c["keep"] > 0) == 2, f"stop-string turns reuse the prompt-end state {ra.calls}")
    # indexer-gap runner: requests that may cross index_topk restore below the gap only
    rg = S.FakeRunner(script=_hash_script, indexer_gap_after_prompt=True, index_topk=150)
    eg = S.ServeEngine(rg, tpl, min_reuse=8, default_max_tokens=8, log=lambda s: None)
    hist = [{"role": "user", "content": "q1"}]
    a = _chat(eg, {"messages": hist})
    hist += [_assistant_echo(a), {"role": "user", "content": "q2"}]
    _chat(eg, {"messages": hist, "max_tokens": 8})
    k_short = rg.calls[-1]["keep"]
    hist2 = hist[:-1] + [{"role": "user", "content": "q2 alt"}]
    _chat(eg, {"messages": hist2, "max_tokens": 120})
    k_long = rg.calls[-1]["keep"]
    check(k_short > rg.calls[0]["prompt"] and k_long <= rg.calls[0]["prompt"],
          f"indexer gap rule: short {k_short}, long {k_long}, first prompt {rg.calls[0]['prompt']}")
    results.append("stop strings, indexer gap")
    return "; ".join(results)


# ============================================================================================ http
def _tool_script(prompt_text, ids, h):
    think = prompt_text.endswith("<think>")
    pre = "I need data.</think>" if think else ""
    if "</tool_response><|assistant|><think>" in prompt_text[-60:]:     # answer after a tool result
        return pre + "Paris: 20 C, clear."
    if "weather" in prompt_text.lower().rsplit("<|user|>", 1)[-1]:
        return (pre + "Let me check.<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>"
                "<arg_key>days</arg_key><arg_value>2</arg_value></tool_call><|observation|>")
    return pre + "Hello there, friend."


class _Server:
    def __init__(self, engine, **kw):
        self.srv = S.make_server(engine, "127.0.0.1", 0, **kw)
        self.port = self.srv.server_address[1]
        self.th = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.th.start()

    def url(self, p):
        return f"http://127.0.0.1:{self.port}{p}"

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def _req(url, body=None, headers=None, raw=None, method=None):
    data = raw if raw is not None else (json.dumps(body).encode("utf-8") if body is not None else None)
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    rq = urllib.request.Request(url, data=data, headers=h, method=method or ("POST" if data is not None else "GET"))
    try:
        with urllib.request.urlopen(rq, timeout=30) as r:
            return r.status, r.read().decode("utf-8"), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8"), dict(e.headers)


def _sse(url, body):
    rq = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
    events = []
    with urllib.request.urlopen(rq, timeout=30) as r:
        check(r.headers.get("Content-Type", "").startswith("text/event-stream"), "SSE content type")
        for line in r:
            line = line.decode("utf-8").rstrip("\n")
            if line.startswith("data: "):
                events.append(line[6:])
    return events


def test_http_round_trip():
    runner = S.FakeRunner(script=_tool_script, delay_s=0.05)
    eng = S.ServeEngine(runner, S.ChatTemplate(TPL_SRC), default_thinking=True, min_reuse=8, log=lambda s: None)
    srv = _Server(eng)
    try:
        st, body, hdr = _req(srv.url("/health"))
        check(st == 200 and json.loads(body)["status"] == "ok", "health")
        st, body, _ = _req(srv.url("/v1/models"))
        check(st == 200 and json.loads(body)["data"][0]["id"] == eng.model_id, "models")
        u = srv.url("/v1/chat/completions")
        msgs = [{"role": "user", "content": "What's the weather in Paris?"}]
        st, body, hdr = _req(u, {"model": "x", "messages": msgs, "tools": [WEATHER], "temperature": 0.7})
        r1 = json.loads(body)
        check(st == 200 and hdr.get("X-Apex-Sampling") == "greedy", f"turn 1 status {st}")
        c1 = r1["choices"][0]
        check(c1["finish_reason"] == "tool_calls" and c1["message"]["reasoning_content"] == "I need data." and
              c1["message"]["content"] == "Let me check.", f"turn 1 message {c1}")
        call = c1["message"]["tool_calls"][0]
        check(call["type"] == "function" and call["function"]["name"] == "get_weather" and
              json.loads(call["function"]["arguments"]) == {"city": "Paris", "days": 2}, f"tool call {call}")
        # [R3.9SMP] the fake runner reports no sampler, so "greedy" is still stated; temperature is honoured, not ignored
        check(r1["apex_fold"]["sampling"] == "greedy" and "temperature" not in r1["apex_fold"]["ignored_params"],
              "greedy stated, temperature not ignored")
        msgs += [c1["message"], {"role": "tool", "tool_call_id": call["id"], "content": "{\"temp\": 20}"}]
        st, body, _ = _req(u, {"messages": msgs, "tools": [WEATHER]})
        r2 = json.loads(body)
        check(st == 200 and r2["choices"][0]["message"]["content"] == "Paris: 20 C, clear." and
              r2["choices"][0]["finish_reason"] == "stop", f"turn 2 {r2['choices'][0]}")
        check(r2["usage"]["prompt_tokens_details"]["cached_tokens"] > 0 and r2["apex_fold"]["reuse"]["cached_tokens"] > 0,
              f"turn 2 reused state: {r2['apex_fold']['reuse']}")
        msgs += [r2["choices"][0]["message"], {"role": "user", "content": "And tomorrow's weather?"}]
        ev = _sse(u, {"messages": msgs, "tools": [WEATHER], "stream": True, "stream_options": {"include_usage": True}})
        check(ev[-1] == "[DONE]", "SSE ends with [DONE]")
        objs = [json.loads(e) for e in ev[:-1]]
        check(objs[0]["choices"][0]["delta"].get("role") == "assistant", "first chunk carries the role")
        reas = "".join(o["choices"][0]["delta"].get("reasoning_content", "") for o in objs if o.get("choices"))
        cont = "".join(o["choices"][0]["delta"].get("content", "") for o in objs if o.get("choices"))
        tcs = [t for o in objs if o.get("choices") for t in o["choices"][0]["delta"].get("tool_calls", [])]
        fins = [o["choices"][0]["finish_reason"] for o in objs if o.get("choices") and o["choices"][0]["finish_reason"]]
        check(reas == "I need data." and cont == "Let me check." and len(tcs) == 1 and tcs[0]["index"] == 0 and
              json.loads(tcs[0]["function"]["arguments"])["city"] == "Paris" and fins == ["tool_calls"],
              f"SSE content {reas!r} {cont!r} {tcs} {fins}")
        check(objs[-1]["choices"] == [] and objs[-1]["usage"]["completion_tokens"] > 0, "usage chunk")
        # thinking off per request + stop + non-stream errors
        st, body, _ = _req(u, {"messages": [{"role": "user", "content": "hi"}], "enable_thinking": False,
                               "stop": ["friend"]})
        m = json.loads(body)["choices"][0]
        check(m["message"]["content"] == "Hello there," and "reasoning_content" not in m["message"] and
              m["finish_reason"] == "stop", f"thinking off + stop {m}")
        for raw, code in ((b"{not json", 400), (json.dumps({"messages": []}).encode(), 400)):
            st, body, _ = _req(u, raw=raw)
            check(st == code and "error" in json.loads(body), f"error {code}: {st} {body}")
        st, body, _ = _req(srv.url("/v1/nothing"), {"a": 1})
        check(st == 404 and json.loads(body)["error"]["code"] == "not_found", "404")
        st, body, _ = _req(srv.url("/v1/serve/stats"))
        check(st == 200 and json.loads(body)["cache"]["entries"] >= 1, "stats")
        # concurrency: two clients at once are serialized (FakeRunner raises on concurrent generate)
        res = []

        def worker(k):
            res.append(_req(u, {"messages": [{"role": "user", "content": f"hello {k}"}]})[0])

        ths = [threading.Thread(target=worker, args=(k,)) for k in range(3)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        check(res == [200, 200, 200], f"concurrent requests {res}")
    finally:
        srv.close()
    # auth + 411
    srv = _Server(eng, api_key="sekret")
    try:
        u = srv.url("/v1/chat/completions")
        st, body, _ = _req(u, {"messages": [{"role": "user", "content": "x"}]})
        check(st == 401, f"no key -> 401 ({st})")
        st, body, _ = _req(u, {"messages": [{"role": "user", "content": "x"}]}, headers={"Authorization": "Bearer sekret"})
        check(st == 200, f"key -> 200 ({st})")
        st, _, _ = _req(srv.url("/health"))
        check(st == 200, "health without key")
        import http.client
        conn = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=10)
        conn.putrequest("POST", "/v1/chat/completions")
        conn.putheader("Authorization", "Bearer sekret")
        conn.endheaders()
        r = conn.getresponse()
        check(r.status == 411, f"missing Content-Length -> 411 ({r.status})")
        conn.close()
    finally:
        srv.close()
    return "health/models, 3-turn tool chat with reuse, SSE, thinking off + stop, errors, stats, concurrency, auth, 411"


# ============================================================================================ real tokenizer
def test_real_tokenizer_prefix():
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return "SKIP: tokenizers not installed"
    path = os.path.join(BACKBONE, "tokenizer.json")
    if not os.path.exists(path):
        return f"SKIP: {path} missing"
    tk = Tokenizer.from_file(path)

    class TokWrap:
        def encode(self, text):
            return tk.encode(text, add_special_tokens=False).ids

        def decode(self, ids):
            return tk.decode(list(ids), skip_special_tokens=False)

    tw = TokWrap()
    for t in S.MARKUP_TOKENS:
        check(tw.encode(t) == [S.GLM53_ADDED_TOKENS[t]], f"{t} is one added token")
    jt = S.ChatTemplate(TPL_SRC)
    answers = ["The answer is 42.", "Sure! Here's code:\n```python\nprint('hi')\n```", "Ünïcödé ∑ 🚀 中文 ok",
               "Two\n\nparagraphs here."]
    n = 0
    for ans in answers:
        for thinking in (False, True):
            hist = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q1?"}]
            p1 = jt.render(S.normalize_messages(hist), None, add_generation_prompt=True, reasoning_effort="low")
            if not thinking:
                p1 += "</think>"
            ids1 = tw.encode(p1)
            gen_text = ("some reasoning</think>" if thinking else "") + ans
            gen = tw.encode(gen_text) + [154827]
            hist2 = hist + [{"role": "assistant", "content": ans}, {"role": "user", "content": "q2?"}]
            p2 = jt.render(S.normalize_messages(hist2), None, add_generation_prompt=True, reasoning_effort="low")
            if not thinking:
                p2 += "</think>"
            ids2 = tw.encode(p2)
            lcp = S.common_prefix_len(ids2, ids1 + gen)
            check(ids2[:len(ids1)] == ids1, "turn-2 prompt extends the turn-1 prompt")
            if thinking:
                check(lcp == len(ids1), f"thinking on: common prefix ends at the prompt end ({lcp} vs {len(ids1)})")
            else:
                check(lcp == len(ids1) + len(gen), f"thinking off: turn-2 prompt extends prompt + answer + <|user|> "
                                                   f"({lcp} vs {len(ids1) + len(gen)}) for {ans!r}")
            check(tw.decode(gen[:-1]) == gen_text, "decode round trip")
            n += 1
    return f"{n} real-BPE turn boundaries behave as the reuse logic assumes"


# ============================================================================================ torch (CPU)
def test_torch_state_store():
    try:
        import torch
    except ImportError:
        return "SKIP: torch not installed"
    torch.manual_seed(0)                                             # CPU tensors only; no torch.cuda call anywhere

    class FakeCache:
        def __init__(self):
            self.pos = 0
            self.kda_state, self.kda_conv, self.dsa_kv, self.dsa_ki, self.dsa_gs = {}, {}, {}, {}, {}

    class Cfg:
        num_hidden_layers = 6
        index_topk = 2048

    class BB:
        dsa_skip_indexer_short = True

    class Runner:
        cfg, bb, device = Cfg(), BB(), torch.device("cpu")
        moe_layers = {3: 0, 4: 0, 5: 0, 6: 0}
        SERVE_HOOKS = ("state",)

        def generate(self, prompt_ids, max_new_tokens=48, eos_ids=None, use_mtp=False, serve_state=None):
            raise AssertionError("not used")

    class Tok:
        eos_ids = {154820, 154827, 154829}

        def encode_raw(self, t):
            return [S.GLM53_ADDED_TOKENS.get(t, 0)]

        def decode(self, ids):
            return ""

    ad = S.GLMRunnerAdapter(Runner(), Tok(), use_mtp=True, cache_factory=FakeCache, log=lambda s: None)
    check(ad.hooked and {"reuse", "on_token"} <= ad.capabilities and ad.use_mtp, f"capabilities {ad.capabilities}")
    kda_layers, dsa_layers = [0, 1, 2, 4], [3, 5]
    T = 9

    def make_cache(T):
        c = FakeCache()
        c.pos = T
        for l in kda_layers:
            c.kda_state[l] = torch.randn(1, 2, 4, 4)
            c.kda_conv[l] = tuple(torch.randn(1, 3, 8) for _ in range(3))
        for l in dsa_layers + [6]:                                 # 6 = the MTP layer (history not stored)
            n = T if l != 6 else 3
            c.dsa_kv[l], c.dsa_ki[l] = torch.randn(1, n, 5), torch.randn(1, n, 3).to(torch.bfloat16)
            c.dsa_gs[l] = torch.randn(1, n, 3).to(torch.bfloat16)
        return c

    c = make_cache(T)
    snap, store = ad.export_eager(c)
    check(store.n == T and 6 not in store.kv and set(store.kv) == set(dsa_layers) and store.indexer_gap is None,
          "eager export: main layers only")
    check(snap.nbytes > 0 and store.nbytes > 0, "byte accounting")
    r = ad.restore_cache(store, snap, T)
    check(r.pos == T and 6 not in r.dsa_kv and all(torch.equal(r.dsa_kv[l], c.dsa_kv[l]) for l in dsa_layers) and
          all(torch.equal(r.kda_state[l], c.kda_state[l]) for l in kda_layers) and
          all(torch.equal(a, b) for l in kda_layers for a, b in zip(r.kda_conv[l], c.kda_conv[l])), "restore == state")
    r.kda_state[0].add_(1.0)
    r.dsa_kv[3].add_(1.0)
    check(torch.equal(snap.kda[0], c.kda_state[0]) and torch.equal(store.kv[3], c.dsa_kv[3]), "restore does not alias")
    c.kda_state[1].add_(1.0)
    check(not torch.equal(snap.kda[1], c.kda_state[1]), "export copies (no alias with the live cache)")
    # prompt-end snapshot + truncated restore from a later store
    c5 = make_cache(5)
    s5 = ad.snapshot_kda(c5, 5, h_last=torch.randn(1, 1, 7))
    r5 = ad.restore_cache(store, s5, 5)
    check(r5.pos == 5 and r5.dsa_kv[3].shape[1] == 5 and torch.equal(r5.dsa_kv[3], c.dsa_kv[3][:, :5]) and
          s5.h_last is not None, "truncated restore")
    try:
        ad.restore_cache(store, s5, 6)
        raise AssertionError("mismatched point accepted")
    except RuntimeError:
        pass

    class GD:                                                      # GraphDecoder static buffers (indexer off)
        t_max = 16

    gd = GD()
    P0, n = 5, 8
    pc = make_cache(P0)
    gd.kda = {l: {"state": torch.randn(1, 2, 4, 4), "cq": torch.randn(1, 3, 8), "ck": torch.randn(1, 3, 8),
                  "cv": torch.randn(1, 3, 8)} for l in kda_layers}
    gd.dsa = {l: {"kv": torch.randn(1, 16, 5)} for l in dsa_layers}
    for l in dsa_layers:
        gd.dsa[l]["kv"][:, :P0] = pc.dsa_kv[l]
    gd.pos_t = torch.tensor([n])
    gs, gstore = ad.export_graph(gd, pc, P0)
    check(gstore.n == n and gstore.indexer_gap == P0 and gs.n == n, "graph export: n from pos_t, gap at the prompt end")
    check(all(torch.equal(gstore.kv[l], gd.dsa[l]["kv"][:, :n]) for l in dsa_layers) and
          all(torch.equal(gstore.ki[l][:, :P0], pc.dsa_ki[l]) and gstore.ki[l].dtype == pc.dsa_ki[l].dtype and
              float(gstore.ki[l][:, P0:].abs().sum()) == 0.0 for l in dsa_layers), "graph export DSA + zero ki/gs")
    check(all(torch.equal(gs.kda[l], gd.kda[l]["state"]) for l in kda_layers), "graph export KDA")
    gd.kda[0]["state"].add_(1.0)
    check(not torch.equal(gs.kda[0], gd.kda[0]["state"]), "graph export copies")
    for l in dsa_layers:
        gd.dsa[l]["ki"] = torch.randn(1, 16, 3)
        gd.dsa[l]["gs"] = torch.randn(1, 16, 3)
    _, gstore2 = ad.export_graph(gd, pc, P0)
    check(gstore2.indexer_gap is None and gstore2.ki[3].dtype == torch.bfloat16, "graph export with in-graph indexer")
    return "snapshot / eager export / graph export / restore on CPU tensors"


def test_adapter_hooks_toy_runtime():
    """GLMRunnerAdapter + ServeState against a toy runtime whose generate() follows the INTEGRATION_S.md hooks H1-H5
    (and a GraphDecoder-like static-buffer decode with on_token, H3b). The toy state has the RealCache layout (KDA
    recurrent + conv, DSA kv / ki / gs) and exact integer-valued float64 math, so a correctly restored state gives
    bit-identical answers: reuse on vs off must match on every turn. The toy reads the indexer cache (ki) only past
    index_topk, like the real DSA: a restore point above an indexer gap used outside the causal regime changes the
    answer, so the gap rule is checked too (graph decode without in-graph indexer caches)."""
    try:
        import torch
    except ImportError:
        return "SKIP: torch not installed"
    f64 = torch.float64
    EOS = 154827
    KDA_L, DSA_L, N_MAIN, TOPK = (0, 1, 2), (3,), 4, 70

    class ToyCache:
        def __init__(self):
            self.pos = 0
            self.kda_state, self.kda_conv, self.dsa_kv, self.dsa_ki, self.dsa_gs = {}, {}, {}, {}, {}

    def tvec(tok, n, mod):
        return torch.tensor([[[float((tok * (7 + k) + k) % mod) for k in range(n)]]], dtype=f64)

    def kda_update(state, conv, tok, l):
        x = tvec(tok, 4, 101)
        cq, ck, cv = (torch.cat([c[:, 1:], x * (j + 1)], dim=1) for j, c in enumerate(conv))
        return torch.remainder(state * 3 + (cq.sum() + ck.sum() - cv.sum()) + l, 1009.0), (cq, ck, cv)

    def kv_row(tok, pos):
        return tvec(tok + 3 * pos, 5, 89)

    def ki_row(tok):
        return tvec(tok, 3, 53)

    def gs_row(tok):
        return tvec(tok * 5, 3, 47)

    def next_token(states, kv, ki, T):
        v = sum(float(states[l].sum()) * (l + 1) for l in KDA_L) + float(kv[:, :T].sum())
        if T > TOPK:                                                     # the "indexer" regime reads ki
            v += 3.0 * float(ki[:, :T].sum())
        h = int(v) % 1000
        return EOS if h % 23 == 0 else 1000 + 97 + h % 26

    class Cfg:
        num_hidden_layers, index_topk = N_MAIN, TOPK

    class BB:
        dsa_skip_indexer_short = True

    class ToyGD:
        def __init__(self, t_max, indexer):
            self.t_max, self.dsa_indexer = t_max, indexer
            z = lambda *s: torch.zeros(*s, dtype=f64)                    # noqa: E731
            self.kda = {l: {"state": z(1, 2, 2, 2), "cq": z(1, 3, 4), "ck": z(1, 3, 4), "cv": z(1, 3, 4)} for l in KDA_L}
            self.dsa = {l: {"kv": z(1, t_max, 5)} for l in DSA_L}
            if indexer:
                for l in DSA_L:
                    self.dsa[l]["ki"], self.dsa[l]["gs"] = z(1, t_max, 3), z(1, t_max, 3)
            self.pos_t = torch.zeros(1, dtype=torch.long)

        def load_cache(self, cache):
            T = int(cache.pos)
            if T >= self.t_max:
                return False
            for l, st in self.kda.items():
                st["state"].copy_(cache.kda_state[l])
                for k, c in zip(("cq", "ck", "cv"), cache.kda_conv[l]):
                    st[k].copy_(c)
            for l, st in self.dsa.items():
                st["kv"].zero_()
                st["kv"][:, :T].copy_(cache.dsa_kv[l])
                if "ki" in st:
                    st["ki"].zero_()
                    st["gs"].zero_()
                    st["ki"][:, :T].copy_(cache.dsa_ki[l])
                    st["gs"][:, :T].copy_(cache.dsa_gs[l])
            self.pos_t.fill_(T)
            return True

        def step(self, tok):
            p = int(self.pos_t.item())
            for l, st in self.kda.items():
                new, conv = kda_update(st["state"], (st["cq"], st["ck"], st["cv"]), tok, l)
                st["state"].copy_(new)
                for k, c in zip(("cq", "ck", "cv"), conv):
                    st[k].copy_(c)
            for l, st in self.dsa.items():
                st["kv"][:, p:p + 1] = kv_row(tok, p)
                if "ki" in st:
                    st["ki"][:, p:p + 1] = ki_row(tok)
                    st["gs"][:, p:p + 1] = gs_row(tok)
            self.pos_t += 1

        def decode(self, first_tok, cache, max_new, eos_ids, on_token=None):    # GraphDecoder.decode + H3b
            if not self.load_cache(cache):
                return None
            out, tok, pos = [], int(first_tok), int(cache.pos)
            for _ in range(max_new):
                if pos >= self.t_max - 1:
                    break
                self.step(tok)
                pos += 1
                tok = next_token({l: st["state"] for l, st in self.kda.items()}, self.dsa[3]["kv"],
                                 self.dsa[3].get("ki"), pos)
                out.append(tok)
                if on_token is not None and on_token(tok):
                    break
                if tok in eos_ids:
                    break
            return out

    class ToyRuntime:
        SERVE_HOOKS = ("state", "skip_prefill")

        def __init__(self, graphs, indexer, t_max=48):
            self.cfg, self.bb, self.device = Cfg(), BB(), torch.device("cpu")
            self.moe_layers = {}
            self.graph_decoder = ToyGD(t_max, indexer) if graphs else None
            self.prefilled = 0

        def _restore_prefix(self, ids):
            return ToyCache(), 0, None

        def _step(self, c, tok):
            for l in KDA_L:
                s, conv = c.kda_state.get(l), c.kda_conv.get(l)
                if s is None:
                    s, conv = torch.zeros(1, 2, 2, 2, dtype=f64), tuple(torch.zeros(1, 3, 4, dtype=f64) for _ in range(3))
                c.kda_state[l], c.kda_conv[l] = kda_update(s, conv, tok, l)
            for l in DSA_L:
                for d, row in ((c.dsa_kv, kv_row(tok, c.pos)), (c.dsa_ki, ki_row(tok)), (c.dsa_gs, gs_row(tok))):
                    d[l] = torch.cat([d[l], row], dim=1) if l in d else row
            c.pos += 1

        def generate(self, prompt_ids, max_new_tokens=48, eos_ids=None, use_mtp=False, serve_state=None):
            eos_ids = eos_ids or set()
            if serve_state is None:
                cache, n_pre, _ = self._restore_prefix(prompt_ids)
            else:
                cache, n_pre = serve_state.begin(self, prompt_ids)                     # H1
            skip_prefill = serve_state is not None and n_pre >= len(prompt_ids)        # H5
            if not skip_prefill:
                for t in prompt_ids[n_pre:]:
                    self._step(cache, t)
                self.prefilled += len(prompt_ids) - n_pre
            if skip_prefill:
                next_tok, h_last = int(serve_state.first_token), serve_state.h_last
            else:
                next_tok = next_token(cache.kda_state, cache.dsa_kv[3], cache.dsa_ki[3], cache.pos)
                h_last = torch.full((1, 1, 2), float(cache.pos), dtype=f64)
            gen_ids = [next_tok]
            if serve_state is not None:
                serve_state.after_prefill(self, cache, next_tok, h_last)               # H2
            graph_done = False
            if self.graph_decoder is not None and next_tok not in eos_ids and not (
                    serve_state is not None and serve_state.stopped):
                if serve_state is None:
                    g = self.graph_decoder.decode(next_tok, cache, max_new_tokens - 1, eos_ids)
                else:
                    g = self.graph_decoder.decode(next_tok, cache, max_new_tokens - 1, eos_ids,
                                                  on_token=serve_state.on_token)         # H3
                if g is not None:
                    gen_ids.extend(g)
                    next_tok = gen_ids[-1]
                    graph_done = True
            while not graph_done and len(gen_ids) < max_new_tokens and next_tok not in eos_ids:
                if serve_state is not None and serve_state.stopped:                      # H3
                    break
                self._step(cache, next_tok)
                next_tok = next_token(cache.kda_state, cache.dsa_kv[3], cache.dsa_ki[3], cache.pos)
                gen_ids.append(next_tok)
                if serve_state is not None:
                    serve_state.on_token(next_tok)                                       # H3
            if serve_state is not None:
                serve_state.after_decode(self, cache, gen_ids, graph_done)               # H4
            return {"generated_ids": gen_ids, "prompt_tokens": len(prompt_ids), "new_tokens": len(gen_ids)}

    class TokW:
        eos_ids = set(S.GLM53_EOS_IDS)

        def __init__(self):
            self.ft = S.FakeTokenizer()

        def encode_raw(self, t):
            return self.ft.encode(t)

        def decode(self, ids):
            return self.ft.decode(ids)

    tpl = S.ChatTemplate(TPL_SRC)
    out = []
    turns = [("hi", None, False, None), ("more please", 6, False, None), ("why?", None, True, None),
             ("ok", None, False, ["q"]), ("last one", None, True, None), ("bye", 9, False, None)]
    for graphs, indexer in ((False, False), (True, False), (True, True)):
        ra, rb = ToyRuntime(graphs, indexer), ToyRuntime(graphs, indexer)
        aa = S.GLMRunnerAdapter(ra, TokW(), cache_factory=ToyCache, log=lambda s: None)
        ab = S.GLMRunnerAdapter(rb, TokW(), cache_factory=ToyCache, log=lambda s: None)
        check(aa.hooked and {"reuse", "on_token", "skip_prefill"} <= aa.capabilities, f"capabilities {aa.capabilities}")
        # gap rule: safe while prompt + budget <= index_topk, or when the graphs do the whole decode (prompt < t_max)
        check(aa.indexer_gap_safe(60, 10) and not aa.indexer_gap_safe(60, 11) and
              aa.indexer_gap_safe(40, 1000) == graphs and not aa.indexer_gap_safe(48, 1000),
              f"indexer_gap_safe graphs={graphs}")
        ea = S.ServeEngine(aa, tpl, min_reuse=4, default_max_tokens=20, log=lambda s: None)
        eb = S.ServeEngine(ab, tpl, reuse=False, default_max_tokens=20, log=lambda s: None)
        # [R3.5-FIX serve-reuse-off-copies] the reuse-off engine commits nothing, so its runner must copy no state
        n_copies = [0]

        def counting_copy(t, _real=ab._store_copy, _n=n_copies):
            _n[0] += 1
            return _real(t)
        ab._store_copy = counting_copy
        hist = [{"role": "system", "content": "toy"}]
        cached = []
        for u, mt, think, stop in turns:
            hist.append({"role": "user", "content": u})
            body = {"messages": hist, "enable_thinking": think}
            if mt:
                body["max_tokens"] = mt
            if stop:
                body["stop"] = stop
            a, b = _chat(ea, body), _chat(eb, body)
            check(a["choices"][0]["message"] == b["choices"][0]["message"] and
                  a["choices"][0]["finish_reason"] == b["choices"][0]["finish_reason"] and
                  a["usage"]["completion_tokens"] == b["usage"]["completion_tokens"],
                  f"graphs={graphs} indexer={indexer} turn {u!r}: reuse changed the answer\n{a['choices'][0]}\n"
                  f"{b['choices'][0]}\nplan {a['apex_fold']['reuse']}")
            check("state_error" not in a["apex_fold"]["runner"], f"state error {a['apex_fold']['runner']}")
            check(b["apex_fold"]["reuse"]["prefilled_tokens"] == b["usage"]["prompt_tokens"],
                  f"reuse off: prefilled {b['apex_fold']['reuse']['prefilled_tokens']} != prompt "
                  f"{b['usage']['prompt_tokens']}")
            cached.append(a["usage"]["prompt_tokens_details"]["cached_tokens"])
            hist.append(_assistant_echo(a))
        regen = {"messages": hist[:-1], "enable_thinking": turns[-1][2], "max_tokens": turns[-1][1]}
        a, b = _chat(ea, regen), _chat(eb, regen)
        check(a["choices"][0]["message"] == b["choices"][0]["message"], f"regenerate graphs={graphs}")
        check(a["apex_fold"]["reuse"]["skip_prefill"], f"regenerate should skip the prefill: {a['apex_fold']['reuse']}")
        check(sum(1 for c in cached if c > 0) >= 3 and ra.prefilled < rb.prefilled,
              f"reuse ineffective: cached {cached}, prefilled {ra.prefilled} vs {rb.prefilled}")
        check(n_copies[0] == 0, f"graphs={graphs}: the reuse-off runner made {n_copies[0]} state copies")
        out.append(f"graphs={graphs}/indexer={indexer}: cached {cached}, prefilled {ra.prefilled} vs {rb.prefilled}, "
                   f"reuse-off copies {n_copies[0]}")
    return "; ".join(out)


# ============================================================================================ transformers (opt-in)
def test_transformers_parity():
    try:
        from transformers import AutoTokenizer
    except Exception as e:                                               # pragma: no cover
        return f"SKIP: transformers unavailable ({e})"
    hf = AutoTokenizer.from_pretrained(BACKBONE, trust_remote_code=True)
    jt = S.ChatTemplate(TPL_SRC)
    n = 0
    for name, c in CONVS.items():
        msgs, tools, kw = norm_conv(c)
        eff = kw.pop("reasoning_effort", "low")
        a = hf.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False, reasoning_effort=eff, **kw)
        b = jt.render(msgs, tools, add_generation_prompt=True, reasoning_effort=eff, **kw)
        check(a == b, f"{name}: transformers render differs")
        n += 1
    class HFRunner(S.FakeRunner):
        def encode(self, text):
            return hf.encode(text, add_special_tokens=False)

    eng = S.ServeEngine(HFRunner(), jt)
    for thinking in (False, True):
        prep = eng.prepare({"messages": [{"role": "user", "content": "What is 2+2?"}], "enable_thinking": thinking})
        text = hf.apply_chat_template([{"role": "user", "content": "What is 2+2?"}], tokenize=False,
                                      add_generation_prompt=True, enable_thinking=thinking, reasoning_effort="low")
        if not thinking and not text.rstrip().endswith("</think>"):
            text = text + "</think>"                                      # RealTokenizer.encode_chat
        check(prep.ids == hf.encode(text, add_special_tokens=False), f"encode_chat parity (thinking={thinking})")
    return f"{n} conversations == transformers apply_chat_template; encode_chat ids identical"


GROUPS = [
    ("template", test_template_python_vs_jinja2), ("template", test_template_expected_strings),
    ("normalize", test_normalize_and_errors), ("parse", test_tool_call_parsing),
    ("stream", test_stream_assembler_equivalence), ("reuse", test_cache_decisions), ("reuse", test_engine_reuse_oracle),
    ("reuse", test_engine_agentic_anchor),
    ("http", test_http_round_trip), ("tokenizer", test_real_tokenizer_prefix), ("torch", test_torch_state_store),
    ("torch", test_adapter_hooks_toy_runtime), ("transformers", test_transformers_parity),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-torch", action="store_true")
    ap.add_argument("--no-tokenizer", action="store_true")
    ap.add_argument("--with-transformers", action="store_true")
    ap.add_argument("-k", default="", help="run only tests whose name contains this")
    args = ap.parse_args()
    t_all = time.perf_counter()
    for grp, fn in GROUPS:
        if args.k and args.k not in fn.__name__:
            continue
        if (grp == "torch" and args.no_torch) or (grp == "tokenizer" and args.no_tokenizer) or \
                (grp == "transformers" and not args.with_transformers):
            SKIP.append(fn.__name__)
            print(f"SKIP {fn.__name__}", flush=True)
            continue
        t0 = time.perf_counter()
        try:
            msg = fn()
            if isinstance(msg, str) and msg.startswith("SKIP"):
                SKIP.append(fn.__name__)
                print(f"SKIP {fn.__name__}: {msg}", flush=True)
            else:
                PASS.append(fn.__name__)
                print(f"PASS {fn.__name__} ({time.perf_counter() - t0:.1f}s): {msg}", flush=True)
        except Exception as e:
            FAIL.append(fn.__name__)
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped in {time.perf_counter() - t_all:.1f}s "
          f"(runtime {RUNTIME})", flush=True)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
