#!/usr/bin/env python3
"""
glm53_fold_serve.py  [Round 3.5 - package S1: the --serve part of speed item 10(c)]
==================================================================================
OpenAI-compatible MULTI-TURN chat server for the APEX-FOLD GLM-5.3-Flash runtime, with conversation-state reuse.

It replaces the R3.x `--serve` handler of glm53_fold_generate.py, which kept only the LAST message (chat history and
agentic tool use were lost). [R3.5-INT] `glm53_fold_generate.py --serve ...` runs this server (the INTEGRATION_S.md
hooks are in the runtime); `--serve --serve-last-message-only` (env R35_SERVE_LAST_MESSAGE_ONLY=1) keeps the old
handler - exactly the base code - as an A/B baseline; `--serve-v2` is accepted for compatibility (the default now).
`python glm53_fold_serve.py --fake` serves a CPU stand-in model (protocol / client testing, no GPU).

Endpoints: POST /v1/chat/completions (alias /chat/completions), GET /v1/models, GET /health, GET /v1/serve/stats,
POST /v1/serve/reset (drops the conversation cache).

Request handling
  * messages: roles system / developer (= system) / user / assistant / tool (legacy "function" = tool); content as a
    string, null or a list of text parts (image / audio / video parts -> 400: the runtime is text-only); assistant
    tool_calls with `arguments` as a JSON string or object; assistant reasoning_content (alias reasoning); tool results
    with tool_call_id / name. tools (+ legacy functions), tool_choice auto | none (required / named: rendered normally,
    not enforced - reported), parallel_tool_calls=false keeps the first call only.
  * thinking on/off: chat_template_kwargs.enable_thinking, top-level enable_thinking, or thinking: {"type": "enabled" |
    "disabled"}; default = the server's --thinking. reasoning_effort (top level, chat_template_kwargs or
    reasoning.effort): low | high | max (minimal -> low, medium -> high); default --serve-effort (low, like
    RealTokenizer.encode_chat). chat_template_kwargs.clear_thinking is passed to the template.
  * max_tokens / max_completion_tokens (capped to the context, noted), stop (string or list; matched in the text AFTER
    the reasoning block, the stop string is not returned), stream (SSE: incremental when the runner calls back per
    token, else sent at the end), stream_options.include_usage, n = 1 only, logprobs -> 400.
  * [R3.9SMP] SAMPLING: temperature / top_p / top_k / seed are honoured (glm53_fold_sampler_r39: top-64 candidates,
    nucleus, Gumbel-max; temperature 0 = greedy, bit-identical to the old path). Defaults: --serve-temperature (0 =
    greedy) / --serve-top-p / --serve-top-k. MTP serving stays greedy (its verify rule is greedy). Penalties, min_p,
    logit_bias, ... are still accepted, ignored and listed (apex_fold.ignored_params; header X-Apex-Sampling).
  * Errors: OpenAI-style JSON {"error": {message, type, param, code}} with 400 / 401 / 404 / 411 / 413 / 500 / 503.

Prompt construction = GLM-5.3's own chat_template.jinja rendered exactly like transformers' apply_chat_template (the
same jinja2 ImmutableSandboxedEnvironment(trim_blocks, lstrip_blocks, loopcontrols) and tojson override); a pure-Python
port of this exact template (sha256-checked) is used only when jinja2 is missing. Thinking off appends "</think>" after
the generation prompt "<|assistant|><think>" - the RealTokenizer.encode_chat rule. Tokenized with
add_special_tokens=False (as encode_chat). OpenAI messages are normalized first: null content -> "" (the template
would print "None"), text parts joined, tool-call arguments JSON-decoded to objects (the template iterates them).

Output parsing: thinking on -> text before the FIRST "</think>" = message.reasoning_content (verbatim), the rest is
parsed for <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call> (the template's own format):
tool calls -> message.tool_calls (arguments typed by the tool's JSON schema: string parameters keep the raw text,
others are JSON-decoded when valid), the remaining text -> content (stripped). finish_reason: stop | length |
tool_calls. usage + prompt_tokens_details.cached_tokens (= tokens restored instead of prefilled).

State reuse (ConversationCache): remembers the TOKEN ids of served sequences (prompt + generated) together with exact
model-state restore points (state after the prompt; final state after decoding). A new request restores the largest
restore point that lies inside its longest common token prefix with a cached sequence and prefills only the rest -
token ids are compared, never strings, so template boundaries, cleared reasoning and re-tokenized tool calls are
handled exactly. Correct for generations ending with EOS (the EOS token is never fed through the model: the final
state covers prompt + generated[:-1]) or max_tokens; the runner reports how many tokens its state covers.
What the runtime must provide (RealCache restore / GraphDecoder export, MTP-layer history) is in INTEGRATION_S.md.
[R3.7-STABLE] Stable-prefix anchor (request field apex_fold.stable_messages, optional): a client whose proxy appends
per-request text to the END of the newest message (retrieved context, reminders) sends the same conversation without
that tail as apex_fold.stable_messages. The engine renders it with the same template settings, takes the common token
prefix K with the real prompt, prefills up to K in a one-token generation and keeps that prompt-end state as a
protected "stable" restore point. The next request of the conversation shares exactly this prefix (its own tail
differs), restores from K and prefills only what is new. Without the field nothing changes. Needs
--serve-cache-points 3 or more (final, prompt-end, stable).
[R3.5-INT] With graph-path MTP (--mtp --graphs and env R35_MTP_GRAPHS=1 - opt-in since R3.5-FIX) the MTP layer's
latent history is stored with every state and put back on reuse (--serve-mtp-history aligned, default;
GLMRunnerAdapter docstring); drafts only, never a token.

Concurrency: ThreadingHTTPServer; one generation at a time (model lock, bounded wait); tokenizer calls serialized.
Runner adapter interface: the ServeRunner protocol - GLMRunnerAdapter (real runtime; torch imported lazily) and
FakeRunner (CPU, used by build_run/test_serve_r35.py and --fake).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import hmac
import inspect
import json
import os
import re
import sys
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Protocol, Sequence, Set, Tuple

# ======================================================================================================== constants
SERVE_TAG = "R3.5-S1"
SERVE_VERSION = "r35-s1.0"
SYSTEM_FINGERPRINT = "apex-fold-r35-s1"
DEFAULT_MODEL_ID = "glm-5.3-flash-apex-fold"
DEFAULT_TEMPLATE_PATH = "C:/models/GLM53-backbone/chat_template.jinja"
GLM53_TEMPLATE_SHA256 = "acfc37c9cba68e0f2707bb16a51f7bf84cb4d29d6c6db0819d7b1761f98a52a1"  # all 3 backbone dirs

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
TOOL_OPEN, TOOL_CLOSE = "<tool_call>", "</tool_call>"
ARG_KEY_OPEN, ARG_KEY_CLOSE = "<arg_key>", "</arg_key>"
ARG_VAL_OPEN, ARG_VAL_CLOSE = "<arg_value>", "</arg_value>"

GLM53_EOS_IDS = (154820, 154827, 154829)  # <|endoftext|>, <|user|>, <|observation|> (generation_config.json)
GLM53_ADDED_TOKENS: Dict[str, int] = {     # tokenizer.json added_tokens (ids are the real ones)
    "<|endoftext|>": 154820, "[MASK]": 154821, "[gMASK]": 154822, "[sMASK]": 154823, "<sop>": 154824, "<eop>": 154825,
    "<|system|>": 154826, "<|user|>": 154827, "<|assistant|>": 154828, "<|observation|>": 154829,
    "<|begin_of_image|>": 154830, "<|end_of_image|>": 154831, "<|begin_of_video|>": 154832, "<|end_of_video|>": 154833,
    "<|begin_of_audio|>": 154834, "<|end_of_audio|>": 154835, "<|begin_of_transcription|>": 154836,
    "<|end_of_transcription|>": 154837, "<|code_prefix|>": 154838, "<|code_middle|>": 154839, "<|code_suffix|>": 154840,
    "<think>": 154841, "</think>": 154842, "<tool_call>": 154843, "</tool_call>": 154844, "<tool_response>": 154845,
    "</tool_response>": 154846, "<arg_key>": 154847, "</arg_key>": 154848, "<arg_value>": 154849, "</arg_value>": 154850,
    "/nothink": 154851, "<|begin_of_box|>": 154852, "<|end_of_box|>": 154853, "<|image|>": 154854, "<|video|>": 154855,
}
# markup the server parses / the template emits: a trimmed output head (--vocab-keep) must keep these rows
MARKUP_TOKENS = ("<think>", "</think>", "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>",
                 "</arg_value>", "<|user|>", "<|observation|>", "<|endoftext|>")
EFFORT_ALIASES = {"low": "low", "minimal": "low", "medium": "high", "high": "high", "max": "max", "maximum": "max",
                  "xhigh": "max"}
# accepted and ignored (reported back): the runtime decodes greedily, without logit processors
IGNORED_PARAMS = ("min_p", "typical_p", "presence_penalty", "frequency_penalty",   # [R3.9SMP] temperature/top_p/top_k/seed honoured
                  "repetition_penalty", "logit_bias", "user", "metadata", "store", "service_tier", "prediction",
                  "modalities", "audio", "web_search_options", "safety_identifier", "prompt_cache_key", "verbosity",
                  "min_tokens", "ignore_eos", "best_of", "use_beam_search", "length_penalty")
MAX_STOPS = 16
THINK_GRACE = 256                     # [R3.9SMP] tokens past the thinking budget to wait for a line break


def _now() -> float:
    return time.perf_counter()


# ======================================================================================================== errors
class RequestError(Exception):
    """A client-visible error, rendered as an OpenAI-style JSON error body with an HTTP status."""

    def __init__(self, message: str, status: int = 400, param: Optional[str] = None, code: Optional[str] = None,
                 etype: Optional[str] = None):
        super().__init__(message)
        self.message = str(message)
        self.status = int(status)
        self.param = param
        self.code = code
        self.etype = etype or ("invalid_request_error" if self.status < 500 else "server_error")

    def body(self) -> Dict[str, Any]:
        return {"error": {"message": self.message, "type": self.etype, "param": self.param, "code": self.code}}


# ======================================================================================================== chat template
def _jinja_env():
    """The jinja2 environment of transformers' chat-template renderer (utils/chat_template_utils.py, transformers 5.16):
    ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True) + loopcontrols, tojson without HTML escaping,
    raise_exception / strftime_now globals. {% generation %} (assistant-mask tracking) is accepted as a no-op block."""
    import jinja2
    import jinja2.ext
    from jinja2 import nodes
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    class _GenerationTag(jinja2.ext.Extension):
        tags = {"generation"}

        def parse(self, parser):
            lineno = next(parser.stream).lineno
            body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
            return nodes.CallBlock(self.call_method("_render"), [], [], body).set_lineno(lineno)

        def _render(self, caller):
            return caller()

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def strftime_now(fmt):
        return datetime.now().strftime(fmt)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[_GenerationTag, jinja2.ext.loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now
    return env


class ChatTemplate:
    """The model's chat template. engine: "auto" (jinja2 if importable, else the pure-Python port when the template is
    the known GLM-5.3 one), "jinja2" or "python" (the port; refused for any other template)."""

    def __init__(self, source: str, engine: str = "auto", origin: str = "<string>"):
        self.source = source
        self.origin = origin
        self.sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
        self.is_glm53 = self.sha256 == GLM53_TEMPLATE_SHA256
        self._tpl = None
        self.engine = ""
        if engine not in ("auto", "jinja2", "python"):
            raise ValueError(f"template engine {engine!r}")
        if engine in ("auto", "jinja2"):
            try:
                self._tpl = _jinja_env().from_string(source)
                self.engine = "jinja2"
            except ImportError:
                if engine == "jinja2":
                    raise
        if self._tpl is None:
            if not self.is_glm53:
                raise RuntimeError(f"{origin}: jinja2 is not available and this is not the GLM-5.3 template "
                                   f"(sha256 {self.sha256[:16]}): the pure-Python port cannot render it exactly")
            self.engine = "python"

    @classmethod
    def from_file(cls, path: str, engine: str = "auto") -> "ChatTemplate":
        with open(path, "r", encoding="utf-8") as f:
            return cls(f.read(), engine=engine, origin=path)

    def render(self, messages: List[dict], tools: Optional[List[dict]] = None, *, add_generation_prompt: bool = True,
               **kwargs) -> str:
        """kwargs = template variables (reasoning_effort, clear_thinking, ...); only the given ones are defined."""
        if self.engine == "jinja2":
            return self._tpl.render(messages=messages, tools=tools, documents=None,
                                    add_generation_prompt=add_generation_prompt, **kwargs)
        return render_glm53_python(messages, tools, add_generation_prompt=add_generation_prompt, **kwargs)


# ---- pure-Python port of GLM-5.3's chat_template.jinja (sha256 acfc37c9...). Mirrors jinja2's semantics for the
# message shapes normalize_messages() produces (string contents, dict tool-call arguments); the tool-result forms
# "list of outputs" / tool_reference raise NotImplementedError (render those with jinja2). build_run/test_serve_r35.py
# checks it against jinja2 on 13+ conversations.
_UNDEF = object()


def _jget(obj: Any, key: str) -> Any:
    """jinja2 `obj.key` on a mapping: obj[key]; missing -> _UNDEF."""
    if isinstance(obj, dict):
        return obj.get(key, _UNDEF)
    return getattr(obj, key, _UNDEF)


def _jtrue(v: Any) -> bool:
    return v is not _UNDEF and bool(v)


def _jstr(v: Any) -> str:
    """`{{ v }}`: str() (None -> 'None'); undefined -> ''."""
    if v is _UNDEF:
        return ""
    return v if isinstance(v, str) else str(v)


def _py_visible_text(content: Any) -> str:
    if content is _UNDEF:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        out = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                out.append(_jstr(item.get("text", _UNDEF)))
            elif isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict) and item.get("type") in ("image", "image_url"):
                out.append("<|begin_of_image|><|image|><|end_of_image|>")
            elif isinstance(item, dict) and item.get("type") in ("video", "video_url"):
                out.append("<|begin_of_video|><|video|><|end_of_video|>")
            elif isinstance(item, dict) and item.get("type") in ("audio", "audio_url", "input_audio"):
                out.append("<|begin_of_audio|><|end_of_audio|>")
        return "".join(out)
    return str(content)


def _py_first(c: Any) -> Any:
    if isinstance(c, (list, tuple, str)) and len(c):
        return c[0]
    if isinstance(c, dict):
        return c.get(0, _UNDEF)
    return _UNDEF


def _py_is_list_of_outputs(m: Any) -> bool:
    c = _jget(m, "content")
    if not _jtrue(c):
        return False
    first = _py_first(c)
    return isinstance(first, dict) and "output" in first


def _py_id_of(obj: Any) -> str:
    v = _jget(obj, "tool_call_id")
    if _jtrue(v):
        return _jstr(v)
    v = _jget(obj, "id")
    if _jtrue(v):
        return _jstr(v)
    return ""


def _py_render_tool_response(m: Any) -> str:
    c = _jget(m, "content")
    if isinstance(c, str):
        return "<tool_response>" + c + "</tool_response>"
    if _jtrue(c) and not isinstance(c, dict):
        first = _py_first(c)
        if isinstance(first, dict) and first.get("type") == "tool_reference":
            raise NotImplementedError("tool_reference tool results: render with jinja2")
    if _py_is_list_of_outputs(m):
        raise NotImplementedError("list-of-outputs tool results: render with jinja2")
    return "<tool_response>" + _py_visible_text(c) + "</tool_response>"


def _py_tool_to_json(tool: dict) -> str:
    parts = ['"' + _jstr(k) + '": ' + json.dumps(v, ensure_ascii=False)
             for k, v in tool.items() if k != "defer_loading" and k != "strict"]
    return "{" + ", ".join(parts) + "}"


_TOOLS_HEAD = ("<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
               "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n")
_TOOLS_TAIL = ("</tools>\n\nFor each function call, output the function name and arguments within the following XML "
               "format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>"
               "<arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>")


def render_glm53_python(messages: List[dict], tools: Optional[List[dict]] = None, add_generation_prompt: bool = False,
                        reasoning_effort: Any = _UNDEF, clear_thinking: Any = _UNDEF, **_unused) -> str:
    """Pure-Python rendering of GLM-5.3's chat_template.jinja (see the comment above)."""
    out = ["[gMASK]<sop>"]
    eff = reasoning_effort if (reasoning_effort is not _UNDEF and reasoning_effort in ("low", "high")) else "max"
    out.append("<|system|>Reasoning Effort: " + eff.capitalize())
    ct = True if clear_thinking is _UNDEF else clear_thinking
    if tools:
        out.append(_TOOLS_HEAD)
        for tool in tools:
            if "function" in tool:
                tool = tool["function"]
            dl = _jget(tool, "defer_loading")
            if dl is _UNDEF or not dl:
                out.append(_py_tool_to_json(tool) + "\n")
        out.append(_TOOLS_TAIL)
    last_user = -1
    for i, m in enumerate(messages):
        if _jget(m, "role") == "user":
            last_user = i
    for i, m in enumerate(messages):
        role = _jget(m, "role")
        if role == "user":
            out.append("<|user|>" + _py_visible_text(_jget(m, "content")))
        elif role == "assistant":
            out.append("<|assistant|>")
            content = _py_visible_text(_jget(m, "content"))
            rc: Any = _UNDEF                                   # loop-local in jinja2 (reset every iteration)
            mrc = _jget(m, "reasoning_content")
            if isinstance(mrc, str):
                rc = mrc
            elif "</think>" in content:
                rc = content.split("</think>")[0].split("<think>")[-1]
                content = content.split("</think>")[-1]
            if ((not ct) or i > last_user) and rc is not _UNDEF:
                out.append("<think>" + rc + "</think>")
            else:
                out.append("<think></think>")
            if content.strip():
                out.append(content.strip())
            tcs = _jget(m, "tool_calls")
            if _jtrue(tcs):
                for tc in tcs:
                    fn = _jget(tc, "function")
                    if _jtrue(fn):
                        tc = fn
                    out.append("<tool_call>" + _jstr(_jget(tc, "name")))
                    args = _jget(tc, "arguments")
                    if not isinstance(args, dict):
                        raise TypeError("tool-call arguments must be a mapping (normalize_messages decodes JSON strings)")
                    for k, v in args.items():
                        out.append("<arg_key>" + _jstr(k) + "</arg_key><arg_value>"
                                   + (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)) + "</arg_value>")
                    out.append("</tool_call>")
        elif role == "tool":
            if i == 0 or _jget(messages[i - 1], "role") != "tool":
                out.append("<|observation|>")
                start, end = i, i
                for j in range(start, len(messages)):
                    if _jget(messages[j], "role") == "tool":
                        end = j
                    else:
                        break
                tcalls = None
                if start > 0 and _jget(messages[start - 1], "role") == "assistant" and \
                        _jtrue(_jget(messages[start - 1], "tool_calls")):
                    tcalls = _jget(messages[start - 1], "tool_calls")
                can_sort = True
                if not tcalls:
                    can_sort = False
                else:
                    for k in range(start, end + 1):
                        mk = messages[k]
                        if _py_is_list_of_outputs(mk):
                            raise NotImplementedError("list-of-outputs tool results: render with jinja2")
                        tk = _py_id_of(mk)
                        if not tk:
                            can_sort = False
                        elif _py_has_dup_id(messages, start, end, tk):
                            can_sort = False
                        elif not any(_py_id_of(tc) == tk for tc in tcalls):
                            can_sort = False
                    for a in range(len(tcalls)):
                        tid = _py_id_of(tcalls[a])
                        if not tid:
                            can_sort = False
                        for b in range(a + 1, len(tcalls)):
                            if _py_id_of(tcalls[b]) == tid:
                                can_sort = False
                if can_sort:
                    for tc in tcalls:
                        tid = _py_id_of(tc)
                        for k in range(start, end + 1):
                            if _py_id_of(messages[k]) == tid:
                                out.append(_py_render_tool_response(messages[k]))
                else:
                    for k in range(start, end + 1):
                        out.append(_py_render_tool_response(messages[k]))
        elif role == "system":
            out.append("<|system|>" + _py_visible_text(_jget(m, "content")))
    if add_generation_prompt:
        out.append("<|assistant|><think>")
    return "".join(out)


def _py_has_dup_id(messages: List[dict], lo: int, hi: int, target: str) -> bool:
    n = 0
    for k in range(lo, hi + 1):
        m = messages[k]
        if _py_is_list_of_outputs(m):
            raise NotImplementedError("list-of-outputs tool results: render with jinja2")
        if _py_id_of(m) == target:
            n += 1
        if n > 1:
            break
    return n > 1


# ======================================================================================================== normalization
def _flatten_content(content: Any, where: str) -> str:
    """OpenAI content -> the string the template's visible_text() would print for it (text parts joined, no
    separator). null -> "" (the template itself would print "None"). Non-text parts are refused (text-only runtime)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for j, part in enumerate(content):
            if isinstance(part, str):
                parts.append(part)
                continue
            if isinstance(part, dict):
                t = part.get("type")
                if t in ("text", "input_text", "output_text"):
                    txt = part.get("text")
                    if not isinstance(txt, str):
                        raise RequestError(f"{where}[{j}].text must be a string", param=f"{where}[{j}].text")
                    parts.append(txt)
                    continue
                if t == "refusal" and isinstance(part.get("refusal"), str):
                    parts.append(part["refusal"])
                    continue
                if t in ("image", "image_url", "input_image", "video", "video_url", "audio", "audio_url",
                         "input_audio", "file", "input_file"):
                    raise RequestError(f"{where}[{j}]: content part type {t!r} is not supported (this runtime is "
                                       "text-only)", param=f"{where}[{j}]", code="unsupported_content")
            raise RequestError(f"{where}[{j}]: unsupported content part (expected a text part)", param=f"{where}[{j}]")
        return "".join(parts)
    raise RequestError(f"{where} must be a string, null or an array of text parts", param=where)


def _norm_tool_call(tc: Any, w: str) -> dict:
    if not isinstance(tc, dict):
        raise RequestError(f"{w} must be an object", param=w)
    fn = tc.get("function", tc)
    if not isinstance(fn, dict):
        raise RequestError(f"{w}.function must be an object", param=f"{w}.function")
    name = fn.get("name")
    if not isinstance(name, str) or not name:
        raise RequestError(f"{w}.function.name must be a non-empty string", param=f"{w}.function.name")
    args = fn.get("arguments", {})
    if args is None or (isinstance(args, str) and not args.strip()):
        args = {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError as e:
            raise RequestError(f"{w}.function.arguments is not valid JSON: {e}", param=f"{w}.function.arguments")
    if not isinstance(args, dict):
        raise RequestError(f"{w}.function.arguments must be a JSON object", param=f"{w}.function.arguments")
    out = {"type": "function", "function": {"name": name, "arguments": args}}
    if tc.get("id") is not None:
        out["id"] = tc["id"]
    return out


def normalize_messages(raw: Any) -> List[dict]:
    """OpenAI chat messages -> the template's message shapes (see the module docstring). Raises RequestError."""
    if not isinstance(raw, list) or not raw:
        raise RequestError("'messages' must be a non-empty array", param="messages")
    out: List[dict] = []
    for i, m in enumerate(raw):
        where = f"messages[{i}]"
        if not isinstance(m, dict):
            raise RequestError(f"{where} must be an object", param=where)
        role = m.get("role")
        if role == "developer":
            role = "system"
        elif role == "function":
            role = "tool"
        if role not in ("system", "user", "assistant", "tool"):
            raise RequestError(f"{where}.role {role!r} is not supported (system, developer, user, assistant, tool)",
                               param=f"{where}.role")
        msg: Dict[str, Any] = {"role": role, "content": _flatten_content(m.get("content"), f"{where}.content")}
        if role == "assistant":
            rc = m.get("reasoning_content", m.get("reasoning"))
            if rc is not None:
                if not isinstance(rc, str):
                    raise RequestError(f"{where}.reasoning_content must be a string", param=f"{where}.reasoning_content")
                msg["reasoning_content"] = rc
            tcs = m.get("tool_calls")
            if tcs:
                if not isinstance(tcs, list):
                    raise RequestError(f"{where}.tool_calls must be an array", param=f"{where}.tool_calls")
                msg["tool_calls"] = [_norm_tool_call(tc, f"{where}.tool_calls[{j}]") for j, tc in enumerate(tcs)]
            elif isinstance(m.get("function_call"), dict):                 # legacy single function call
                msg["tool_calls"] = [_norm_tool_call({"function": m["function_call"]}, f"{where}.function_call")]
        elif role == "tool":
            if m.get("tool_call_id") is not None:
                msg["tool_call_id"] = m["tool_call_id"]
            if m.get("name") is not None:
                msg["name"] = m["name"]
        out.append(msg)
    return out


def normalize_tools(tools: Any, functions: Any = None) -> Optional[List[dict]]:
    """tools (or legacy functions) -> [{"type": "function", "function": {...}}] with each function object deep-copied
    in its original key order (the template prints it in that order), or None when there are none."""
    src = tools
    if src is None and functions:
        if not isinstance(functions, list):
            raise RequestError("'functions' must be an array", param="functions")
        src = [{"type": "function", "function": f} for f in functions]
    if src is None or src == []:
        return None
    if not isinstance(src, list):
        raise RequestError("'tools' must be an array", param="tools")
    out = []
    for i, t in enumerate(src):
        if not isinstance(t, dict):
            raise RequestError(f"tools[{i}] must be an object", param=f"tools[{i}]")
        typ = t.get("type", "function")
        if typ != "function":
            raise RequestError(f"tools[{i}].type {typ!r} is not supported (only 'function')", param=f"tools[{i}].type")
        fn = t.get("function")
        if fn is None and isinstance(t.get("name"), str):                  # flat {"type": "function", "name": ...}
            fn = {k: v for k, v in t.items() if k != "type"}
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"]:
            raise RequestError(f"tools[{i}].function.name must be a non-empty string", param=f"tools[{i}].function")
        out.append({"type": "function", "function": copy.deepcopy(fn)})
    return out


def _as_bool(v: Any, param: str) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, int) and v in (0, 1):
        return bool(v)
    if isinstance(v, str) and v.strip().lower() in ("true", "false", "1", "0", "yes", "no", "on", "off", "enabled",
                                                    "disabled", "enable", "disable"):
        return v.strip().lower() in ("true", "1", "yes", "on", "enabled", "enable")
    raise RequestError(f"{param} must be a boolean", param=param)


# ======================================================================================================== output parsing
_TOOL_BLOCK_RE = re.compile(re.escape(TOOL_OPEN) + r"(.*?)" + re.escape(TOOL_CLOSE), re.DOTALL)
_ARG_RE = re.compile(re.escape(ARG_KEY_OPEN) + r"(.*?)" + re.escape(ARG_KEY_CLOSE) + r"\s*" + re.escape(ARG_VAL_OPEN)
                     + r"(.*?)" + re.escape(ARG_VAL_CLOSE), re.DOTALL)


def _json_type(v: Any) -> str:
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return "null"


def _schema_types(s: Any) -> Set[str]:
    if not isinstance(s, dict):
        return set()
    out: Set[str] = set()
    t = s.get("type")
    if isinstance(t, str):
        out.add(t)
    elif isinstance(t, list):
        out.update(x for x in t if isinstance(x, str))
    for k in ("anyOf", "oneOf", "allOf"):
        for sub in s.get(k) or ():
            out |= _schema_types(sub)
    if not out and isinstance(s.get("enum"), list):
        out.update(_json_type(v) for v in s["enum"])
    if not out and "const" in s:
        out.add(_json_type(s["const"]))
    return out


def coerce_tool_arg(fn_schema: Optional[dict], key: str, raw: str) -> Any:
    """<arg_value> text -> JSON value, mirroring the template (string values are printed raw, others as JSON):
    string-typed parameters keep the raw text verbatim; otherwise the stripped text is JSON-decoded when it parses and
    its type is allowed by the schema (no schema: whenever it parses), else the raw text is kept."""
    props = ((fn_schema or {}).get("parameters") or {}).get("properties") or {}
    types = _schema_types(props.get(key)) if isinstance(props, dict) else set()
    if types and types <= {"string", "null"}:
        return None if ("null" in types and raw.strip() == "null") else raw
    try:
        val = json.loads(raw.strip())
    except ValueError:
        return raw
    if not types:
        return val
    jt = _json_type(val)
    if jt in types or (jt == "integer" and "number" in types):
        return val
    return raw if "string" in types else val


def parse_tool_block(inner: str, schemas: Dict[str, dict]) -> Optional[dict]:
    """Inside of <tool_call>...</tool_call> -> {"name", "arguments"} or None (no / invalid name)."""
    k = inner.find(ARG_KEY_OPEN)
    name = (inner if k < 0 else inner[:k]).strip()
    if not name or "<" in name or ">" in name:
        return None
    args: Dict[str, Any] = {}
    if k >= 0:
        for mm in _ARG_RE.finditer(inner, k):
            key = mm.group(1).strip()
            args[key] = coerce_tool_arg(schemas.get(name), key, mm.group(2))
    return {"name": name, "arguments": args}


def split_tool_calls(post: str, schemas: Optional[Dict[str, dict]]) -> Tuple[List[dict], str]:
    """Text after the reasoning -> (tool calls, the remaining text with every parsed block removed). Blocks without a
    valid name and an unterminated <tool_call> stay in the text."""
    calls: List[dict] = []
    pieces: List[str] = []
    pos = 0
    for mm in _TOOL_BLOCK_RE.finditer(post):
        call = parse_tool_block(mm.group(1), schemas or {})
        if call is None:
            continue
        pieces.append(post[pos:mm.start()])
        pos = mm.end()
        calls.append(call)
    pieces.append(post[pos:])
    return calls, "".join(pieces)


@dataclass(eq=False)
class ParsedOutput:
    content: str
    reasoning: Optional[str]
    tool_calls: List[dict]            # [{"name", "arguments": dict}] (ids / JSON encoding are added by the engine)
    stop_string: Optional[str] = None


def _holdback(text: str, markers: Sequence[str]) -> int:
    """Length of the longest suffix of `text` that is a proper prefix of one of `markers` (may still become a marker)."""
    best = 0
    n = len(text)
    for m in markers:
        for L in range(min(len(m) - 1, n), best, -1):
            if text.endswith(m[:L]):
                best = L
                break
    return best


class OutputAssembler:
    """Incremental parse of the generated text, shared by the streaming and non-streaming paths.

    feed(text_delta) -> [(kind, text)] with kind "reasoning" | "content", safe to stream: text that could still become
    "</think>", "<tool_call>" or a stop string is held back. Thinking on: text before the first "</think>" is reasoning.
    Content streaming pauses at the first "<tool_call>" (tool calls are reported at the end). A stop string (matched
    after the reasoning, including tool markup) truncates the text there and sets .stopped. finish(schemas) ->
    (remaining events, ParsedOutput). Concatenated streamed reasoning == ParsedOutput.reasoning and streamed content
    .rstrip() == ParsedOutput.content (the non-streaming content is stripped; streaming skips leading whitespace)."""

    def __init__(self, *, thinking: bool, stops: Sequence[str] = (), parse_tools: bool = False):
        self.thinking = bool(thinking)
        self.stops = [s for s in (stops or ()) if s]
        self._max_stop = max((len(s) for s in self.stops), default=0)
        self.parse_tools = bool(parse_tools)
        self.in_reasoning = self.thinking
        self.reasoning: Optional[str] = "" if self.thinking else None
        self.post = ""
        self.r_emitted = 0
        self.c_begin = 0
        self.c_emitted = 0
        self.c_started = False
        self.frozen = False
        self.stopped = False
        self.stop_string: Optional[str] = None

    def feed(self, delta: str) -> List[Tuple[str, str]]:
        if self.stopped or not delta:
            return []
        if self.in_reasoning:
            ev: List[Tuple[str, str]] = []
            prev = len(self.reasoning)
            self.reasoning += delta
            k = self.reasoning.find(THINK_CLOSE, max(0, prev - len(THINK_CLOSE) + 1))
            if k >= 0:
                tail = self.reasoning[k + len(THINK_CLOSE):]
                self.reasoning = self.reasoning[:k]
                self.in_reasoning = False
                if len(self.reasoning) > self.r_emitted:
                    ev.append(("reasoning", self.reasoning[self.r_emitted:]))
                    self.r_emitted = len(self.reasoning)
                if tail:
                    ev.extend(self._feed_post(tail))
                return ev
            safe = len(self.reasoning) - _holdback(self.reasoning, (THINK_CLOSE,))
            if safe > self.r_emitted:
                ev.append(("reasoning", self.reasoning[self.r_emitted:safe]))
                self.r_emitted = safe
            return ev
        return self._feed_post(delta)

    def _feed_post(self, delta: str) -> List[Tuple[str, str]]:
        prev = len(self.post)
        self.post += delta
        if self.stops:
            lo = max(0, prev - self._max_stop + 1)
            hit = None
            for s in self.stops:
                k = self.post.find(s, lo)
                if k >= 0 and (hit is None or k < hit[0]):
                    hit = (k, s)
            if hit is not None:
                self.post = self.post[:hit[0]]
                self.stopped = True
                self.stop_string = hit[1]
        if self.frozen:
            return []
        limit = len(self.post)
        if self.parse_tools:
            k = self.post.find(TOOL_OPEN, max(0, self.c_emitted - len(TOOL_OPEN)))
            if k >= 0:
                limit = k
                self.frozen = True
        if not self.stopped:
            # stop strings are held back even when the tool opener just froze the content: a stop string may begin
            # before the opener and end after it
            marks = list(self.stops) + ([TOOL_OPEN] if (self.parse_tools and not self.frozen) else [])
            if marks:
                limit = min(limit, len(self.post) - _holdback(self.post, marks))
        if not self.c_started:
            j = self.c_emitted
            while j < limit and self.post[j].isspace():
                j += 1
            self.c_emitted = j
            if j < limit:
                self.c_started = True
                self.c_begin = j
        if limit > self.c_emitted:
            out = [("content", self.post[self.c_emitted:limit])]
            self.c_emitted = limit
            return out
        return []

    def finish(self, schemas: Optional[Dict[str, dict]] = None) -> Tuple[List[Tuple[str, str]], ParsedOutput]:
        ev: List[Tuple[str, str]] = []
        if self.in_reasoning and self.reasoning is not None and len(self.reasoning) > self.r_emitted:
            ev.append(("reasoning", self.reasoning[self.r_emitted:]))      # "</think>" never came: all reasoning
            self.r_emitted = len(self.reasoning)
        if self.parse_tools:
            calls, text = split_tool_calls(self.post, schemas)
        else:
            calls, text = [], self.post
        full = text.lstrip()
        emitted = self.post[self.c_begin:self.c_emitted] if self.c_started else ""
        rest = full[len(emitted):].rstrip() if full.startswith(emitted) else ""
        if rest:
            ev.append(("content", rest))
        return ev, ParsedOutput(content=text.strip(), reasoning=self.reasoning, tool_calls=calls,
                                stop_string=self.stop_string)


def parse_generation(text: str, *, thinking: bool, schemas: Optional[Dict[str, dict]] = None,
                     parse_tools: bool = True, stops: Sequence[str] = ()) -> ParsedOutput:
    """One-shot parse of a generated text (the non-streaming result of OutputAssembler)."""
    asm = OutputAssembler(thinking=thinking, stops=stops, parse_tools=parse_tools)
    asm.feed(text)
    return asm.finish(schemas)[1]


class IncrementalDetokenizer:
    """Token-by-token text deltas without splitting multi-byte characters (the prefix/read-offset scheme of HF TGI):
    a delta is released only when the decoded window grows and does not end in U+FFFD. skip_ids (EOS) print nothing."""

    def __init__(self, decode: Callable[[Sequence[int]], str], skip_ids: Iterable[int] = ()):
        self._decode = decode
        self.skip = frozenset(int(i) for i in skip_ids)
        self.ids: List[int] = []
        self.prefix_offset = 0
        self.read_offset = 0

    def push(self, tok: int) -> str:
        tok = int(tok)
        if tok in self.skip:
            return ""
        self.ids.append(tok)
        prefix = self._decode(self.ids[self.prefix_offset:self.read_offset]) if self.read_offset > self.prefix_offset else ""
        new = self._decode(self.ids[self.prefix_offset:])
        if len(new) > len(prefix) and not new.endswith("\ufffd"):
            self.prefix_offset, self.read_offset = self.read_offset, len(self.ids)
            return new[len(prefix):]
        return ""

    def flush(self) -> str:
        if self.read_offset >= len(self.ids):
            return ""
        prefix = self._decode(self.ids[self.prefix_offset:self.read_offset]) if self.read_offset > self.prefix_offset else ""
        new = self._decode(self.ids[self.prefix_offset:])
        self.prefix_offset = self.read_offset = len(self.ids)
        return new[len(prefix):] if len(new) > len(prefix) else ""


# ======================================================================================================== state reuse
@dataclass(eq=False)
class RestorePoint:
    """An exact model state after the first n_tokens tokens of an entry's sequence. handle = runner-owned state (for
    GLMRunnerAdapter: KDA recurrent + conv states (+ h_last)); the DSA history lives in the entry's store.
    next_token = the greedy token predicted FROM this state, when known (prompt-end points: the first generated token)
    -> an identical re-request can skip the prefill entirely (runner capability "skip_prefill").
    anchor = prompt-end point of a request whose last message is a user message: with clear_thinking (the template
    default) a later user turn renders every earlier assistant turn without its reasoning, so the next prompt diverges
    right there - the newest anchor is protected when points are trimmed (agentic tool loops).
    stable = [R3.7-STABLE] prompt-end point of the one-token generation at the end of the client's stable prefix
    (apex_fold.stable_messages): the newest one is protected, ahead of the anchor."""
    n_tokens: int
    handle: Any = None
    next_token: Optional[int] = None
    nbytes: int = 0
    kind: str = ""
    anchor: bool = False
    stable: bool = False


@dataclass(eq=False)
class ConversationEntry:
    tokens: List[int]                 # prompt + generated ids of the served sequence (the EOS included)
    store: Any                        # runner-owned history for tokens[:n_state] (DSA latents / indexer caches)
    store_nbytes: int
    n_state: int                      # tokens the store covers (= the largest restore point)
    points: List[RestorePoint]        # largest n first
    indexer_gap: Optional[int]        # first position whose DSA indexer caches are placeholders (None = complete)
    last_used: float
    uid: int


@dataclass(eq=False)
class ReusePlan:
    entry: Optional[ConversationEntry] = None
    point: Optional[RestorePoint] = None
    keep: int = 0                     # tokens restored (0 = fresh prefill)
    lcp: int = 0                      # longest common token prefix found
    skip_prefill: bool = False        # keep == len(prompt): no prefill, first token = point.next_token
    reason: str = "fresh"

    def describe(self) -> Dict[str, Any]:
        return {"cached_tokens": self.keep, "common_prefix": self.lcp, "point": self.point.kind if self.point else None,
                "skip_prefill": self.skip_prefill, "reason": self.reason}


def common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    i, step = 0, 256
    while i + step <= n and a[i:i + step] == b[i:i + step]:
        i += step
    while i < n and a[i] == b[i]:
        i += 1
    return i


class ConversationCache:
    """Token-level conversation cache: entries (most recent first), each = a served token sequence + restore points.

    plan(prompt_ids) -> ReusePlan: over all entries, the largest restore point n with n <= LCP(prompt, entry.tokens)
    and n <= len(prompt) - 1 (n == len(prompt) only with skip-prefill), n >= min_reuse; a point above the entry's
    indexer gap only with allow_gap_points (the runner guarantees this request never runs an eager DSA forward past
    index_topk, where the placeholder indexer caches would be read). commit(plan, prompt_ids, outcome): the new sequence replaces the entry it was
    restored from and inherits that entry's points n <= keep (still exact: tokens[:n] are unchanged); limits: entries,
    points per entry (largest n kept), total bytes (least recently used entries first, then the smallest points)."""

    def __init__(self, *, max_entries: int = 2, max_points: int = 3, max_bytes: int = 1 << 30, min_reuse: int = 32,
                 release: Optional[Callable[[Any], None]] = None):
        self.max_entries = max(1, int(max_entries))
        self.max_points = max(1, int(max_points))
        self.max_bytes = int(max_bytes)
        self.min_reuse = max(1, int(min_reuse))
        self._release_fn = release or (lambda obj: None)
        self.entries: List[ConversationEntry] = []
        self._uid = 0
        self.counters = {"plans": 0, "reused": 0, "fresh": 0, "skip_prefill": 0, "reused_tokens": 0, "commits": 0,
                         "evicted_entries": 0, "dropped_points": 0}

    # ---- accounting
    @staticmethod
    def entry_bytes(e: ConversationEntry) -> int:
        return int(e.store_nbytes) + sum(int(p.nbytes) for p in e.points)

    def total_bytes(self) -> int:
        return sum(self.entry_bytes(e) for e in self.entries)

    def stats(self) -> Dict[str, Any]:
        return {"entries": len(self.entries), "bytes": self.total_bytes(), "max_bytes": self.max_bytes,
                "points": [[p.n_tokens for p in e.points] for e in self.entries],
                "tokens": [len(e.tokens) for e in self.entries], **self.counters}

    def _release(self, obj: Any) -> None:
        if obj is not None:
            try:
                self._release_fn(obj)
            except Exception:
                pass

    # ---- decision
    def plan(self, prompt_ids: Sequence[int], *, allow_skip_prefill: bool = False,
             allow_gap_points: bool = False) -> ReusePlan:
        self.counters["plans"] += 1
        P = len(prompt_ids)
        best = ReusePlan(reason="no cached conversation" if not self.entries else
                         "no restore point inside the common token prefix")
        best_lcp = 0
        for e in self.entries:
            lcp = common_prefix_len(prompt_ids, e.tokens)
            best_lcp = max(best_lcp, lcp)
            for p in sorted(e.points, key=lambda q: -q.n_tokens):
                n = int(p.n_tokens)
                if n > lcp or n > e.n_state or n <= 0:
                    continue
                if n < self.min_reuse:
                    break
                skip = False
                if n >= P:
                    if not (allow_skip_prefill and n == P and p.next_token is not None):
                        continue
                    skip = True
                if e.indexer_gap is not None and n > e.indexer_gap and not allow_gap_points:
                    continue
                if n > best.keep:
                    best = ReusePlan(entry=e, point=p, keep=n, lcp=lcp, skip_prefill=skip,
                                     reason=f"restore {p.kind or 'point'}@{n} (common prefix {lcp} of {P} tokens"
                                            f"{', skip_prefill' if skip else ''})")
                break
        if best.point is None:
            best.lcp = best_lcp
            if self.entries:
                best.reason += f" (best common prefix {best_lcp} of {P} tokens)"
            self.counters["fresh"] += 1
        else:
            self.counters["reused"] += 1
            self.counters["reused_tokens"] += best.keep
            self.counters["skip_prefill"] += int(best.skip_prefill)
        return best

    # ---- update
    def commit(self, plan: Optional[ReusePlan], prompt_ids: Sequence[int], outcome: "GenerateOutcome"
               ) -> Optional[ConversationEntry]:
        self.counters["commits"] += 1
        src = None
        if plan is not None and plan.entry is not None and any(e is plan.entry for e in self.entries):
            src = plan.entry
        if outcome is None or outcome.store is None or outcome.n_state <= 0 or not outcome.points:
            return None                                       # nothing new to remember; existing entries stay valid
        P = len(prompt_ids)
        S = [int(t) for t in prompt_ids] + [int(t) for t in outcome.generated_ids]
        if not (P <= int(outcome.n_state) <= len(S)):
            raise ValueError(f"runner state covers {outcome.n_state} tokens, sequence has {P} + "
                             f"{len(outcome.generated_ids)}")
        keep = int(plan.keep) if src is not None else 0
        inherited = [p for p in src.points if p.n_tokens <= keep] if src is not None else []
        gap = outcome.indexer_gap
        if src is not None and src.indexer_gap is not None and src.indexer_gap < keep:
            gap = src.indexer_gap if gap is None else min(gap, src.indexer_gap)
        by_n: Dict[int, RestorePoint] = {}
        for p in list(outcome.points) + inherited:            # new points win over inherited ones at the same n
            if 0 < p.n_tokens <= outcome.n_state and p.n_tokens not in by_n:
                by_n[p.n_tokens] = p
        points = sorted(by_n.values(), key=lambda q: -q.n_tokens)
        kept = self._select_points(points, self.max_points)
        dropped = [p for p in points if all(p is not q for q in kept)]
        kept_ids = {id(p) for p in kept}                      # identity only: handles hold tensors
        released: Set[int] = set()

        def rel_point(p):
            if id(p) not in kept_ids and id(p) not in released:
                released.add(id(p))
                self._release(p.handle)
                self.counters["dropped_points"] += 1

        for p in outcome.points:
            rel_point(p)                                      # invalid / duplicate / trimmed new points (kept: no-op)
        if src is not None:
            self.entries = [e for e in self.entries if e is not src]
            self._release(src.store)                          # superseded: the new store holds tokens[:keep] as well
            for p in src.points:
                rel_point(p)
        for p in dropped:
            rel_point(p)
        self._uid += 1
        e = ConversationEntry(tokens=S, store=outcome.store, store_nbytes=int(outcome.store_nbytes),
                              n_state=int(outcome.n_state), points=kept, indexer_gap=gap, last_used=time.time(),
                              uid=self._uid)
        self.entries.insert(0, e)
        self._enforce()
        return e if any(x is e for x in self.entries) else None

    @staticmethod
    def _protected(points: List[RestorePoint]) -> List[RestorePoint]:
        """The newest final point, the newest prompt-end point, the newest stable point ([R3.7-STABLE]) and the newest
        anchor (largest n of each kind). Order matters when max_points is smaller than this list: stable before anchor,
        because a client that sends stable prefixes diverges at the stable point, not at the anchor."""
        by_n = sorted(points, key=lambda q: -q.n_tokens)
        out: List[RestorePoint] = []
        for pick in (next((p for p in by_n if p.kind == "final"), None),
                     next((p for p in by_n if p.kind == "prompt-end"), None),
                     next((p for p in by_n if getattr(p, "stable", False)), None),
                     next((p for p in by_n if p.anchor), None)):
            if pick is not None and all(pick is not q for q in out):
                out.append(pick)
        return out

    @classmethod
    def _select_points(cls, points: List[RestorePoint], k: int) -> List[RestorePoint]:
        """Points kept per entry: the protected ones first, then the largest n; returned largest n first."""
        chosen: List[RestorePoint] = []
        for p in cls._protected(points) + sorted(points, key=lambda q: -q.n_tokens):
            if len(chosen) < k and all(p is not q for q in chosen):
                chosen.append(p)
        return sorted(chosen, key=lambda q: -q.n_tokens)

    def _evict(self, e: ConversationEntry) -> None:
        self.entries = [x for x in self.entries if x is not e]
        self._release(e.store)
        for p in e.points:
            self._release(p.handle)
        self.counters["evicted_entries"] += 1

    def _enforce(self) -> None:
        while len(self.entries) > self.max_entries:
            self._evict(self.entries[-1])
        while self.entries and self.total_bytes() > self.max_bytes:
            if len(self.entries) > 1:
                self._evict(self.entries[-1])
                continue
            e = self.entries[0]
            if len(e.points) > 1:
                prot = self._protected(e.points)
                rest = [q for q in e.points if all(q is not r for r in prot)] or list(e.points)
                p = min(rest, key=lambda q: q.n_tokens)       # oldest unprotected turn boundary first
                e.points = [q for q in e.points if q is not p]
                self._release(p.handle)
                self.counters["dropped_points"] += 1
                continue
            self._evict(e)

    def clear(self) -> None:
        for e in list(self.entries):
            self._evict(e)

    def snapshot_entries(self) -> List[ConversationEntry]:
        """(gates / tests) shallow copy of the entry list; restore_entries() puts it back. Only safe with runners
        whose release() does not free memory eagerly (GLMRunnerAdapter, FakeRunner)."""
        return [copy.copy(e) for e in self.entries]

    def restore_entries(self, entries: List[ConversationEntry]) -> None:
        self.entries = [copy.copy(e) for e in entries]


# ======================================================================================================== runner protocol
@dataclass(eq=False)
class GenerateRequest:
    prompt_ids: List[int]
    max_new_tokens: int
    plan: ReusePlan
    on_token: Optional[Callable[[int], bool]] = None   # every generated token in order (first included); True = stop
    # [R3.5-FIX serve-reuse-off-copies] False: the engine will not commit this request's state (reuse off for it), so
    # the runner makes no prompt-end snapshot and no final export (two full state copies per request, thrown away)
    store: bool = True
    # [R3.9SMP] sampler settings {temperature, top_p, top_k, seed} (None = greedy: internal one-token generations)
    sampling: Optional[Dict[str, Any]] = None
    # [R3.9SMP] rewrite(tok) -> tok, called before each decoded token is committed (thinking budget); None = off
    rewrite: Optional[Callable[[int], int]] = None


@dataclass(eq=False)
class GenerateOutcome:
    generated_ids: List[int]
    n_state: int = 0                  # tokens of prompt + generated the final state covers (0 = nothing reusable)
    store: Any = None                 # history store for tokens[:n_state] (None = nothing reusable)
    store_nbytes: int = 0
    points: List[RestorePoint] = field(default_factory=list)
    indexer_gap: Optional[int] = None # first position (of THIS request's new tokens) without indexer caches
    prefilled: int = 0                # tokens actually prefilled
    cached: int = 0                   # tokens restored (or taken from the exact prefix cache)
    stop_reason: str = ""             # "eos" | "length" | "callback" | "context"
    telemetry: Dict[str, Any] = field(default_factory=dict)


class ServeRunner(Protocol):
    """What ServeEngine needs from a model. capabilities: "reuse" (honours GenerateRequest.plan and returns restore
    points), "on_token" (calls on_token during generation; else the engine replays the tokens afterwards), "stream",
    "skip_prefill" (accepts plan.skip_prefill)."""
    model_id: str
    eos_ids: FrozenSet[int]
    capabilities: FrozenSet[str]

    def indexer_gap_safe(self, n_prompt: int, max_new: int) -> bool:
        """May a restore point whose indexer caches hold placeholders serve this request (see GLMRunnerAdapter)?"""
        ...

    def encode(self, text: str) -> List[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...

    def generate(self, req: GenerateRequest) -> GenerateOutcome: ...

    def release(self, obj: Any) -> None: ...

    def describe(self) -> Dict[str, Any]: ...


# ======================================================================================================== engine
@dataclass(eq=False)
class Prepared:
    cid: str
    created: int
    messages: List[dict]
    tools: Optional[List[dict]]
    schemas: Dict[str, dict]
    parse_tools: bool
    parallel_tool_calls: bool
    thinking: bool
    effort: str
    clear_thinking: Optional[bool]
    text: str
    ids: List[int]
    max_tokens: int
    stops: List[str]
    stream: bool
    include_usage: bool
    reuse: bool
    ignored: Dict[str, Any]
    notes: List[str]
    t_received: float
    t_prepared: float
    stable_n: int = 0                 # [R3.7-STABLE] tokens of the client's stable prefix (0 = none given / not usable)
    sampling: Optional[Dict[str, Any]] = None   # [R3.9SMP] {temperature, top_p, top_k, seed}
    thinking_budget: int = 0                    # [R3.9SMP] reasoning tokens before </think> is forced (0 = off)


STABLE_MARGIN = 4                     # [R3.7-STABLE] tokens cut from the common prefix (boundary tokenization)
STABLE_MIN = 256                      # a shorter stable prefix is not worth a restore point


class ServeEngine:
    """Request validation + prompt construction (outside the model lock), generation with state reuse (one at a
    time), response / SSE chunk construction. See the module docstring for the request semantics."""

    def __init__(self, runner: ServeRunner, template: ChatTemplate, *, model_id: Optional[str] = None,
                 default_thinking: bool = False, default_effort: str = "low", default_max_tokens: int = 2048,
                 max_context: int = 16384, reuse: bool = True, cache_entries: int = 2, cache_points: int = 3,
                 cache_bytes: int = 1 << 30, min_reuse: int = 32, prefill_chunk: int = 0,
                 restore_reasoning: bool = False, queue_timeout: float = 600.0,
                 log: Optional[Callable[[str], None]] = None,
                 default_sampling: Optional[Dict[str, Any]] = None, default_thinking_budget: int = 0):
        self.runner = runner
        # [R3.9SMP] server-wide sampler defaults (a request's temperature / top_p / top_k / seed override them)
        self.default_sampling: Dict[str, Any] = dict(default_sampling or {"temperature": 0.0})
        # [R3.9SMP] thinking budget: after this many reasoning tokens the end-of-thinking token is forced at the next
        # line break (at the latest THINK_GRACE tokens later). 0 = off. apex_fold.thinking_budget overrides per request.
        self.default_thinking_budget = max(0, int(default_thinking_budget or 0))
        self.template = template
        self.model_id = model_id or getattr(runner, "model_id", DEFAULT_MODEL_ID)
        self.default_thinking = bool(default_thinking)
        de = EFFORT_ALIASES.get(str(default_effort).lower())
        if de is None:
            raise ValueError(f"default reasoning effort {default_effort!r}")
        self.default_effort = de
        self.default_max_tokens = max(1, int(default_max_tokens))
        self.max_context = int(max_context)
        self.reuse = bool(reuse)
        self.prefill_chunk = max(0, int(prefill_chunk))
        self.restore_reasoning = bool(restore_reasoning)
        self.queue_timeout = float(queue_timeout)
        self.cache = ConversationCache(max_entries=cache_entries, max_points=cache_points, max_bytes=cache_bytes,
                                       min_reuse=min_reuse, release=getattr(runner, "release", None))
        self.model_lock = threading.Lock()
        self.tok_lock = threading.Lock()
        self._mem_lock = threading.Lock()
        self._reasoning_mem: "OrderedDict[str, str]" = OrderedDict()   # tool-call id -> reasoning (opt-in)
        self.log = log or (lambda s: print(s, flush=True))
        self.counters = {"requests": 0, "errors": 0, "stream": 0, "prompt_tokens": 0, "completion_tokens": 0,
                         "prefilled_tokens": 0, "cached_tokens": 0}
        self.t_start = time.time()

    # ------------------------------------------------------------------ info
    def health(self) -> Dict[str, Any]:
        return {"status": "ok", "model": self.model_id, "busy": self.model_lock.locked(), "version": SERVE_VERSION,
                "sampling": ("greedy" if not float(self.default_sampling.get("temperature") or 0) > 0
                             else dict(self.default_sampling)), "reuse": self.reuse and "reuse" in self.runner.capabilities,
                "capabilities": sorted(self.runner.capabilities), "uptime_s": round(time.time() - self.t_start, 1)}

    def models(self) -> Dict[str, Any]:
        return {"object": "list", "data": [self.model_card()]}

    def model_card(self) -> Dict[str, Any]:
        return {"id": self.model_id, "object": "model", "created": int(self.t_start), "owned_by": "apex-fold",
                "max_context": self.max_context}

    def stats(self) -> Dict[str, Any]:
        d = {"counters": dict(self.counters), "cache": self.cache.stats(), "template": {
            "origin": self.template.origin, "engine": self.template.engine, "sha256": self.template.sha256}}
        try:
            d["runner"] = self.runner.describe()
        except Exception as e:                                             # pragma: no cover
            d["runner"] = {"error": repr(e)}
        return d

    def reset(self) -> None:
        with self.model_lock:
            self.cache.clear()

    # ------------------------------------------------------------------ request preparation (no model access)
    def prepare(self, body: Any) -> Prepared:
        t0 = _now()
        if not isinstance(body, dict):
            raise RequestError("the request body must be a JSON object")
        ignored: Dict[str, Any] = {}
        notes: List[str] = []
        n = body.get("n", 1)
        if n not in (None, 1):
            raise RequestError("only n=1 is supported (greedy decoding gives one answer)", param="n")
        if body.get("logprobs") or body.get("top_logprobs"):
            raise RequestError("logprobs are not supported by this runtime", param="logprobs")
        for k in IGNORED_PARAMS:
            if body.get(k) is not None:
                ignored[k] = body[k]
        from glm53_fold_sampler_r39 import parse_request_sampling        # [R3.9SMP]
        try:
            sampling = parse_request_sampling(body, self.default_sampling)
        except ValueError as e:
            raise RequestError(f"{e.args[0]} has an invalid value (temperature 0..5, top_p (0, 1], top_k >= 0, "
                               "seed an integer)", param=str(e.args[0]))
        rf = body.get("response_format")
        if isinstance(rf, dict) and rf.get("type") not in (None, "text"):
            ignored["response_format"] = rf.get("type")
            notes.append("response_format is not enforced (no constrained decoding)")
        messages = normalize_messages(body.get("messages"))
        tools = normalize_tools(body.get("tools"), body.get("functions"))
        tool_choice = body.get("tool_choice", body.get("function_call"))
        parse_tools = tools is not None
        tools_prompt = tools
        if tool_choice == "none":
            tools_prompt, parse_tools = None, False
        elif tool_choice not in (None, "auto"):
            ignored["tool_choice"] = tool_choice
            notes.append("tool_choice is not enforced (no constrained decoding); tools are offered as with 'auto'")
        ptc = body.get("parallel_tool_calls")
        ptc = True if ptc is None else _as_bool(ptc, "parallel_tool_calls")
        kw = body.get("chat_template_kwargs") or {}
        if not isinstance(kw, dict):
            raise RequestError("chat_template_kwargs must be an object", param="chat_template_kwargs")
        thinking = self.default_thinking
        th = body.get("thinking")
        if isinstance(th, dict) and "type" in th:
            thinking = _as_bool(th["type"], "thinking.type")
        elif isinstance(th, (bool, str)):
            thinking = _as_bool(th, "thinking")
        if body.get("enable_thinking") is not None:
            thinking = _as_bool(body["enable_thinking"], "enable_thinking")
        if kw.get("enable_thinking") is not None:
            thinking = _as_bool(kw["enable_thinking"], "chat_template_kwargs.enable_thinking")
        effort_raw = kw.get("reasoning_effort", body.get("reasoning_effort"))
        if effort_raw is None and isinstance(body.get("reasoning"), dict):
            effort_raw = body["reasoning"].get("effort")
        effort = self.default_effort if effort_raw is None else EFFORT_ALIASES.get(str(effort_raw).strip().lower())
        if effort is None:
            raise RequestError(f"reasoning_effort {effort_raw!r} is not supported (low, high, max; minimal -> low, "
                               "medium -> high)", param="reasoning_effort")
        clear_thinking = kw.get("clear_thinking")
        if clear_thinking is not None:
            clear_thinking = _as_bool(clear_thinking, "chat_template_kwargs.clear_thinking")
        stop = body.get("stop")
        if stop is None:
            stops: List[str] = []
        elif isinstance(stop, str):
            stops = [stop] if stop else []
        elif isinstance(stop, list) and all(isinstance(s, str) for s in stop):
            stops = [s for s in stop if s]
        else:
            raise RequestError("stop must be a string or an array of strings", param="stop")
        if len(stops) > MAX_STOPS:
            raise RequestError(f"at most {MAX_STOPS} stop strings are supported", param="stop")
        mt = body.get("max_completion_tokens", body.get("max_tokens"))
        if mt is not None and (isinstance(mt, bool) or not isinstance(mt, int) or mt < 1):
            raise RequestError("max_tokens must be a positive integer", param="max_tokens")
        stream = False if body.get("stream") is None else _as_bool(body["stream"], "stream")   # explicit null = false
        so = body.get("stream_options") or {}
        include_usage = bool(isinstance(so, dict) and so.get("include_usage"))
        reuse = True
        if body.get("cache_prompt") is not None:
            reuse = _as_bool(body["cache_prompt"], "cache_prompt")
        ext = body.get("apex_fold")
        if isinstance(ext, dict) and ext.get("reuse") is not None:
            reuse = _as_bool(ext["reuse"], "apex_fold.reuse")
        thinking_budget = self.default_thinking_budget                      # [R3.9SMP]
        if isinstance(ext, dict) and ext.get("thinking_budget") is not None:
            tb = ext["thinking_budget"]
            if isinstance(tb, bool) or not isinstance(tb, int) or tb < 0:
                raise RequestError("apex_fold.thinking_budget must be a non-negative integer (0 = off)",
                                   param="apex_fold.thinking_budget")
            thinking_budget = int(tb)
        if self.restore_reasoning:
            self._inject_reasoning(messages, notes)
        tkw: Dict[str, Any] = {"reasoning_effort": effort}
        if clear_thinking is not None:
            tkw["clear_thinking"] = clear_thinking
        try:
            text = self.template.render(messages, tools_prompt, add_generation_prompt=True, **tkw)
        except RequestError:
            raise
        except Exception as e:                                             # jinja2 TemplateError / UndefinedError ...
            raise RequestError(f"chat template rejected the conversation: {type(e).__name__}: {e}",
                               param="messages", code="template_error")
        if not thinking and not text.rstrip().endswith(THINK_CLOSE):        # RealTokenizer.encode_chat's thinking-off rule
            text = text + THINK_CLOSE
        with self.tok_lock:
            ids = [int(i) for i in self.runner.encode(text)]
        P = len(ids)
        if P >= self.max_context:
            raise RequestError(f"the prompt has {P} tokens; this server's maximum context is {self.max_context} "
                               "tokens", param="messages", code="context_length_exceeded")
        room = self.max_context - P
        if mt is None:
            max_tokens = min(self.default_max_tokens, room)
        else:
            max_tokens = min(int(mt), room)
            if max_tokens < int(mt):
                notes.append(f"max_tokens capped from {mt} to {max_tokens} (context {self.max_context}, prompt {P})")
        schemas = {t["function"]["name"]: t["function"] for t in (tools or [])}
        # [R3.7-STABLE] the client's stable prefix: the same conversation without the per-request tail of its newest
        # message. K = common token prefix of the two renderings, minus a margin; a problem here never fails the request.
        stable_n = 0
        sm_raw = ext.get("stable_messages") if isinstance(ext, dict) else None
        if sm_raw is not None and reuse and self.reuse:
            try:
                sm = normalize_messages(sm_raw)
                if self.restore_reasoning:
                    self._inject_reasoning(sm, [])
                stext = self.template.render(sm, tools_prompt, add_generation_prompt=False, **tkw)
                with self.tok_lock:
                    sids = [int(i) for i in self.runner.encode(stext)]
                k, lim = 0, min(len(sids), P)
                while k < lim and sids[k] == ids[k]:
                    k += 1
                k -= STABLE_MARGIN
                if k >= STABLE_MIN and P - k >= 8:
                    stable_n = k
                else:
                    notes.append(f"apex_fold.stable_messages: common prefix {max(k, 0)} of {P} tokens, no stable point")
            except Exception as e:
                notes.append(f"apex_fold.stable_messages ignored: {type(e).__name__}: {str(e)[:160]}")
        return Prepared(cid="chatcmpl-" + uuid.uuid4().hex, created=int(time.time()), messages=messages, tools=tools_prompt,
                        schemas=schemas, parse_tools=parse_tools, parallel_tool_calls=ptc, thinking=thinking,
                        effort=effort, clear_thinking=clear_thinking, text=text, ids=ids, max_tokens=max_tokens,
                        stops=stops, stream=stream, include_usage=include_usage, reuse=reuse, ignored=ignored,
                        notes=notes, t_received=t0, t_prepared=_now(), stable_n=stable_n, sampling=sampling,
                        thinking_budget=thinking_budget if thinking else 0)

    def _think_close_ids(self) -> List[int]:
        """[R3.9SMP] token ids of "</think>" (cached; one id expected)."""
        ids = getattr(self, "_close_ids_cache", None)
        if ids is None:
            with self.tok_lock:
                ids = [int(i) for i in self.runner.encode(THINK_CLOSE)]
            self._close_ids_cache = ids
        return ids

    def _inject_reasoning(self, messages: List[dict], notes: List[str]) -> None:
        """--serve-restore-reasoning: an assistant message after the last user message (a tool loop) that comes back
        without reasoning_content gets the reasoning this server generated with its tool calls (looked up by the
        tool-call ids it issued) - the template keeps reasoning in tool loops, clients often drop it."""
        last_user = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=-1)
        with self._mem_lock:
            for i, m in enumerate(messages):
                if i <= last_user or m["role"] != "assistant" or "reasoning_content" in m or not m.get("tool_calls"):
                    continue
                found = {self._reasoning_mem.get(str(tc.get("id"))) for tc in m["tool_calls"] if tc.get("id") is not None}
                if len(found) == 1 and None not in found:
                    m["reasoning_content"] = found.pop()
                    notes.append(f"messages[{i}]: reasoning_content restored from this server's earlier response")

    def _remember_reasoning(self, tool_calls: List[dict], reasoning: Optional[str]) -> None:
        if not reasoning or not tool_calls:
            return
        with self._mem_lock:
            for tc in tool_calls:
                self._reasoning_mem[str(tc["id"])] = reasoning
            while len(self._reasoning_mem) > 4096:
                self._reasoning_mem.popitem(last=False)

    # ------------------------------------------------------------------ generation
    def first_chunk(self, prep: Prepared) -> Dict[str, Any]:
        return self._chunk(prep, {"role": "assistant", "content": ""})

    def _chunk(self, prep: Prepared, delta: Dict[str, Any], finish_reason: Optional[str] = None) -> Dict[str, Any]:
        return {"id": prep.cid, "object": "chat.completion.chunk", "created": prep.created, "model": self.model_id,
                "system_fingerprint": SYSTEM_FINGERPRINT,
                "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish_reason}]}

    def execute(self, prep: Prepared, emit: Optional[Callable[[Dict[str, Any]], None]] = None
                ) -> Optional[Dict[str, Any]]:
        """Runs one request. emit=None -> returns the chat.completion object; emit=callable -> streams
        chat.completion.chunk objects through it (the caller sent the first chunk and writes [DONE]) and returns
        None. Raises RequestError (busy -> 503) or the runner's exception (the conversation cache is then cleared)."""
        t_q0 = _now()
        if not self.model_lock.acquire(timeout=self.queue_timeout):
            raise RequestError(f"server busy: waited {self.queue_timeout:.0f} s for the model", status=503,
                               code="server_busy")
        try:
            return self._execute_locked(prep, emit, t_q0)
        except Exception:
            self.counters["errors"] += 1
            raise
        finally:
            self.model_lock.release()

    def _decode_locked(self, ids: Sequence[int]) -> str:
        """Tokenizer calls of the generation thread (streaming deltas) and of concurrent prepare() calls share one lock
        (HF fast tokenizers may mutate their state inside encode)."""
        with self.tok_lock:
            return self.runner.decode(ids)

    def _plan(self, ids: List[int], max_new: int) -> ReusePlan:
        r = self.runner
        f = getattr(r, "indexer_gap_safe", None)
        gap_ok = bool(f(len(ids), int(max_new))) if callable(f) else False
        return self.cache.plan(ids, allow_skip_prefill="skip_prefill" in r.capabilities, allow_gap_points=gap_ok)

    def _chunked_prefill(self, prep: Prepared, plan: ReusePlan) -> Tuple[ReusePlan, Dict[str, Any]]:
        """--serve-prefill-chunk N: prefill a long suffix in N-token pieces, each one a 1-token generation whose prompt
        state becomes the next piece's restore point (bounds the eager prefill's activation / DSA score memory; every
        piece re-streams its experts, so only for prompts that would not fit otherwise)."""
        P, C = len(prep.ids), self.prefill_chunk
        cuts = list(range(plan.keep + C, P, C))
        t0 = _now()
        for c in cuts:
            sub = prep.ids[:c]
            sp = self._plan(sub, 1)
            out = self.runner.generate(GenerateRequest(prompt_ids=sub, max_new_tokens=1, plan=sp, on_token=None))
            self.cache.commit(sp, sub, out)
        return self._plan(prep.ids, prep.max_tokens), {"chunks": len(cuts) + 1, "chunk_tokens": C,
                                                       "seconds": round(_now() - t0, 3)}

    def _execute_locked(self, prep: Prepared, emit, t_q0: float) -> Optional[Dict[str, Any]]:
        t0 = _now()
        r = self.runner
        caps = r.capabilities
        self.counters["requests"] += 1
        self.counters["stream"] += int(emit is not None)
        use_reuse = self.reuse and prep.reuse and "reuse" in caps
        if use_reuse:
            plan = self._plan(prep.ids, prep.max_tokens)
        else:
            plan = ReusePlan(reason="reuse disabled" if (not self.reuse or not prep.reuse) else
                             "runner has no state hooks (INTEGRATION_S.md H1-H4)")
        # [R3.7-STABLE] a restore point at the end of the client's stable prefix: prefill up to K in a one-token
        # generation (as a prefill piece does), mark its prompt-end state stable, then plan again from there.
        stable_info = None
        K = int(getattr(prep, "stable_n", 0) or 0)
        if use_reuse and K > plan.keep and len(prep.ids) - K >= 8:
            sub = prep.ids[:K]
            sp = self._plan(sub, 1)
            t_s = _now()
            try:
                out_s = r.generate(GenerateRequest(prompt_ids=sub, max_new_tokens=1, plan=sp, on_token=None))
            except Exception:
                self.cache.clear()                                         # the device state may be inconsistent
                raise
            for p in out_s.points:
                if p.kind == "prompt-end":
                    p.stable = True
            self.cache.commit(sp, sub, out_s)
            self.counters["stable_points"] = self.counters.get("stable_points", 0) + 1
            stable_info = {"n": K, "cached": int(out_s.cached), "prefilled": int(out_s.prefilled),
                           "seconds": round(_now() - t_s, 3)}
            plan = self._plan(prep.ids, prep.max_tokens)
        chunk_info = None
        if use_reuse and self.prefill_chunk > 0 and len(prep.ids) - plan.keep > self.prefill_chunk:
            plan, chunk_info = self._chunked_prefill(prep, plan)
        asm = OutputAssembler(thinking=prep.thinking, stops=prep.stops, parse_tools=prep.parse_tools)
        detok = IncrementalDetokenizer(self._decode_locked, skip_ids=r.eos_ids)
        st = {"t_first": None, "n": 0, "gone": False}

        def send(obj):
            if emit is None or st["gone"]:
                return
            try:
                emit(obj)
            except Exception:
                st["gone"] = True                                          # client went away: stop generating

        def push(events):
            for kind, text in events:
                if text:
                    send(self._chunk(prep, {"reasoning_content": text} if kind == "reasoning" else {"content": text}))

        def on_token(tok: int) -> bool:
            st["n"] += 1
            if st["t_first"] is None:
                st["t_first"] = _now()
            if asm.stopped or st["gone"]:
                return True
            d = detok.push(int(tok))
            if d:
                push(asm.feed(d))
            return asm.stopped or st["gone"]

        # [R3.9SMP] thinking budget: count the decoded tokens while the reply is still in its reasoning block; past the
        # budget, the next token that contains a line break (at the latest THINK_GRACE tokens later) is replaced by
        # "</think>", which the decoder then feeds to the model: the answer follows from where the reasoning stopped.
        tb = {"budget": int(prep.thinking_budget or 0), "n": 0, "forced_at": None}
        rewrite = None
        close_ids = self._think_close_ids() if tb["budget"] > 0 else []
        if tb["budget"] > 0 and len(close_ids) == 1:
            close_id = close_ids[0]

            def rewrite(tok: int) -> int:
                if tb["forced_at"] is not None or not asm.in_reasoning:
                    return tok
                tb["n"] += 1
                if tb["n"] < tb["budget"] or tok == close_id:
                    return tok
                if tb["n"] >= tb["budget"] + THINK_GRACE or "\n" in self._decode_locked([int(tok)]):
                    tb["forced_at"] = tb["n"]
                    return close_id
                return tok
        elif tb["budget"] > 0:
            prep.notes.append("thinking budget off: the tokenizer does not encode </think> as one token")

        req = GenerateRequest(prompt_ids=list(prep.ids), max_new_tokens=prep.max_tokens, plan=plan, on_token=on_token,
                              store=use_reuse,                                 # [R3.5-FIX] no state copies without reuse
                              sampling=prep.sampling,                          # [R3.9SMP]
                              rewrite=rewrite)
        try:
            out = r.generate(req)
        except Exception:
            self.cache.clear()                                             # the device state may be inconsistent
            raise
        t_gen = _now()
        gen = [int(t) for t in out.generated_ids]
        if st["n"] == 0 and gen:                                           # runner without per-token callbacks
            for t in gen:
                if on_token(t):
                    break
        if not asm.stopped:
            tail = detok.flush()
            if tail:
                push(asm.feed(tail))
        fin, parsed = asm.finish(prep.schemas)
        push(fin)
        eos = r.eos_ids
        if parsed.stop_string is not None:
            finish = "stop"
        elif gen and gen[-1] in eos:
            finish = "stop"
        elif out.stop_reason == "context" or len(gen) >= prep.max_tokens:
            finish = "length"
        else:
            finish = "stop"
        calls = parsed.tool_calls if prep.parse_tools else []
        if calls and not prep.parallel_tool_calls:
            calls = calls[:1]
        tool_calls = [{"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                       "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                      for c in calls]
        if tool_calls and finish == "stop":
            finish = "tool_calls"
        committed = None
        if use_reuse:
            if prep.messages and prep.messages[-1]["role"] == "user":     # a user turn: the next divergence point
                for p in out.points:
                    if p.kind == "prompt-end":
                        p.anchor = True
            committed = self.cache.commit(plan, prep.ids, out)
        if self.restore_reasoning:
            self._remember_reasoning(tool_calls, parsed.reasoning)
        P = len(prep.ids)
        t_end = _now()
        t_first = st["t_first"] or t_gen
        timing = {"queue_s": round(t0 - t_q0, 4), "prepare_s": round(prep.t_prepared - prep.t_received, 4),
                  "ttft_s": round(t_first - t0, 4), "generate_s": round(t_gen - t0, 4),
                  "decode_tok_s": round((len(gen) - 1) / max(t_gen - t_first, 1e-6), 3) if len(gen) > 1 else None,
                  "total_s": round(t_end - prep.t_received, 4)}
        # [R3.7-STABLE] with a stable step the request restored stable_info["cached"] tokens and prefilled the rest in
        # two parts; report the request's real figures, not those of its second part.
        cached_req = int(out.cached) if stable_info is None else int(stable_info["cached"])
        prefilled_req = int(out.prefilled) if stable_info is None else int(out.prefilled) + int(stable_info["prefilled"])
        usage = {"prompt_tokens": P, "completion_tokens": len(gen), "total_tokens": P + len(gen),
                 "prompt_tokens_details": {"cached_tokens": cached_req}}
        self.counters["prompt_tokens"] += P
        self.counters["completion_tokens"] += len(gen)
        self.counters["prefilled_tokens"] += prefilled_req
        self.counters["cached_tokens"] += cached_req
        if tb["forced_at"] is not None:
            self.counters["thinking_budget_forced"] = self.counters.get("thinking_budget_forced", 0) + 1
            self.log(f"[serve] [{SERVE_TAG}] {prep.cid[-8:]} thinking budget {tb['budget']}: </think> forced after "
                     f"{tb['forced_at']} reasoning tokens")
        info = {"sampling": (out.telemetry or {}).get("sampling", "greedy"),
                "thinking_budget": {"budget": tb["budget"], "reasoning_tokens": tb["n"], "forced_at": tb["forced_at"]},
                "ignored_params": prep.ignored, "notes": prep.notes, "thinking": prep.thinking,
                "reasoning_effort": prep.effort,
                "reuse": dict(plan.describe(), prefilled_tokens=prefilled_req, committed=committed is not None,
                              stop_reason=out.stop_reason, prefill_chunks=chunk_info, stable=stable_info),
                "timing": timing, "runner": out.telemetry}
        if stable_info is not None:
            info["reuse"]["cached_tokens"] = cached_req
        self.log(f"[serve] [{SERVE_TAG}] {prep.cid[-8:]} prompt {P} tok (reused {cached_req}, prefilled {prefilled_req}"
                 f"{', stable point at ' + str(stable_info['n']) if stable_info else ''}) "
                 f"-> {len(gen)} tok, finish {finish}{' +' + str(len(tool_calls)) + ' tool calls' if tool_calls else ''}"
                 f", ttft {timing['ttft_s']:.2f}s, total {timing['total_s']:.2f}s | {plan.reason}")
        if emit is not None:
            if tool_calls:
                send(self._chunk(prep, {"tool_calls": [dict(tc, index=i) for i, tc in enumerate(tool_calls)]}))
            last = self._chunk(prep, {}, finish_reason=finish)
            last["apex_fold"] = info
            send(last)
            if prep.include_usage:
                u = {k: v for k, v in self._chunk(prep, {}).items() if k != "choices"}
                u["choices"] = []
                u["usage"] = usage
                send(u)
            return None
        msg: Dict[str, Any] = {"role": "assistant", "content": parsed.content if (parsed.content or not tool_calls) else None}
        if prep.thinking:
            msg["reasoning_content"] = parsed.reasoning if parsed.reasoning is not None else ""
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return {"id": prep.cid, "object": "chat.completion", "created": prep.created, "model": self.model_id,
                "system_fingerprint": SYSTEM_FINGERPRINT,
                "choices": [{"index": 0, "message": msg, "logprobs": None, "finish_reason": finish}],
                "usage": usage, "apex_fold": info}


# ======================================================================================================== HTTP layer
class ServeHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = os.name != "nt"     # Windows SO_REUSEADDR would let a second server bind the same port
    request_queue_size = 64


def make_handler(engine: ServeEngine, *, api_key: str = "", cors: bool = False, max_body: int = 32 << 20,
                 verbose: bool = False):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "apex-fold-serve/" + SERVE_VERSION
        timeout = 900

        def log_message(self, fmt, *args):
            if verbose:
                sys.stderr.write("[serve] %s - %s\n" % (self.address_string(), fmt % args))

        def _cors(self):
            if cors:
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _json(self, status: int, obj: Any) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Apex-Sampling", "greedy")
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def _err(self, e: RequestError) -> None:
            self._json(e.status, e.body())

        def _route(self) -> str:
            p = self.path.split("?", 1)[0]
            return p.rstrip("/") or "/"

        def _authorized(self) -> bool:
            if not api_key:
                return True
            h = self.headers.get("Authorization", "")
            tok = h[7:].strip() if h[:7].lower() == "bearer " else ""
            if tok and hmac.compare_digest(tok.encode("utf-8"), api_key.encode("utf-8")):
                return True
            self.close_connection = True
            self._err(RequestError("invalid or missing API key (Authorization: Bearer ...)", status=401,
                                   code="invalid_api_key", etype="authentication_error"))
            return False

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            p = self._route()
            if p == "/health":
                return self._json(200, engine.health())
            if not self._authorized():
                return
            if p in ("/v1/models", "/models"):
                return self._json(200, engine.models())
            if p.startswith("/v1/models/"):
                mid = p[len("/v1/models/"):]
                if mid == engine.model_id:
                    return self._json(200, engine.model_card())
                return self._err(RequestError(f"model {mid!r} not found", status=404, code="model_not_found"))
            if p == "/v1/serve/stats":
                return self._json(200, engine.stats())
            return self._err(RequestError(f"no route for GET {p}", status=404, code="not_found"))

        def do_POST(self):
            p = self._route()
            if not self._authorized():
                return
            if p not in ("/v1/chat/completions", "/chat/completions", "/v1/serve/reset"):
                self.close_connection = True
                return self._err(RequestError(f"no route for POST {p}", status=404, code="not_found"))
            length = self.headers.get("Content-Length")
            if length is None:
                self.close_connection = True
                return self._err(RequestError("Content-Length is required", status=411, code="length_required"))
            try:
                n = int(length)
            except ValueError:
                self.close_connection = True
                return self._err(RequestError("invalid Content-Length", status=400))
            if n < 0 or n > max_body:
                self.close_connection = True
                return self._err(RequestError(f"request body too large ({n} > {max_body} bytes)", status=413,
                                              code="body_too_large"))
            raw = self.rfile.read(n)
            if p == "/v1/serve/reset":
                engine.reset()
                return self._json(200, {"status": "ok", "cache": engine.cache.stats()})
            try:
                body = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as e:
                return self._err(RequestError(f"invalid JSON body: {e}", code="invalid_json"))
            try:
                prep = engine.prepare(body)
            except RequestError as e:
                return self._err(e)
            except Exception as e:
                traceback.print_exc()
                return self._err(RequestError(f"internal error while preparing the prompt: {type(e).__name__}: {e}",
                                              status=500))
            if prep.stream:
                return self._stream(prep)
            try:
                resp = engine.execute(prep)
            except RequestError as e:
                return self._err(e)
            except Exception as e:
                traceback.print_exc()
                return self._err(RequestError(f"internal error during generation: {type(e).__name__}: {e} "
                                              "(conversation cache cleared)", status=500))
            self._json(200, resp)

        def _stream(self, prep: Prepared) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("X-Apex-Sampling", "greedy")
            self._cors()
            self.end_headers()
            self.close_connection = True

            def emit(obj):
                self.wfile.write(b"data: " + json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n\n")
                self.wfile.flush()

            try:
                emit(engine.first_chunk(prep))
                engine.execute(prep, emit=emit)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return
            except RequestError as e:
                try:
                    emit(e.body())
                except OSError:
                    return
            except Exception as e:
                traceback.print_exc()
                try:
                    emit(RequestError(f"internal error during generation: {type(e).__name__}: {e} "
                                      "(conversation cache cleared)", status=500).body())
                except OSError:
                    return
            try:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except OSError:
                pass

    return Handler


def make_server(engine: ServeEngine, host: str = "127.0.0.1", port: int = 8088, **kw) -> ServeHTTPServer:
    return ServeHTTPServer((host, int(port)), make_handler(engine, **kw))


def run_http(engine: ServeEngine, host: str = "127.0.0.1", port: int = 8088, **kw) -> None:
    srv = make_server(engine, host, port, **kw)
    h, p = srv.server_address[:2]
    print(f"[serve] [{SERVE_TAG}] listening on http://{h}:{p}/v1/chat/completions (model {engine.model_id}, greedy, "
          f"reuse {'on' if engine.reuse and 'reuse' in engine.runner.capabilities else 'off'}, "
          f"capabilities {sorted(engine.runner.capabilities)})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


# ======================================================================================================== fake model (CPU)
class FakeTokenizer:
    """Deterministic stand-in tokenizer: GLM-5.3's added tokens (real ids) are matched first, as HF does; all other
    text is UTF-8 bytes (ids BYTE_BASE + b; multi-byte characters span several tokens)."""
    BYTE_BASE = 1000

    def __init__(self, added: Optional[Dict[str, int]] = None):
        self.added = dict(added or GLM53_ADDED_TOKENS)
        self.inv = {v: k for k, v in self.added.items()}
        self._re = re.compile("|".join(re.escape(t) for t in sorted(self.added, key=len, reverse=True)))

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        pos = 0
        for mm in self._re.finditer(text):
            ids.extend(self.BYTE_BASE + b for b in text[pos:mm.start()].encode("utf-8"))
            ids.append(self.added[mm.group(0)])
            pos = mm.end()
        ids.extend(self.BYTE_BASE + b for b in text[pos:].encode("utf-8"))
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        out: List[str] = []
        buf = bytearray()
        for i in ids:
            i = int(i)
            if self.BYTE_BASE <= i < self.BYTE_BASE + 256:
                buf.append(i - self.BYTE_BASE)
                continue
            if buf:
                out.append(buf.decode("utf-8", errors="replace"))
                buf = bytearray()
            out.append(self.inv.get(i, "\ufffd"))
        if buf:
            out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)


def _fake_hash(h: int, toks: Iterable[int]) -> int:
    for t in toks:
        h = (h * 1000003 + int(t) + 1) & 0xFFFFFFFFFFFF
    return h


class FakeRunner:
    """ServeRunner on the CPU with a VERIFIABLE recurrent state: state(n) = rolling hash of tokens[:n] (it cannot be
    rewound, like the KDA state). Restore points hold (n, hash); generate() raises AssertionError when a restored
    point's hash or the stored token history differs from the prompt prefix - an oracle for the reuse logic. As the
    real runtime, the last generated token is not fed through the model (process_last_token=True mimics an MTP
    accepted pair, whose state covers it). Output = script(prompt_text, prompt_ids, state_hash) -> text or ids (EOS
    appended when missing); default script: a text derived from the state hash (so a wrong state changes the
    answer). indexer_gap_after_prompt=True reports the decoded tokens as indexer-less (graph decode without
    R34_DSA_INDEXER)."""

    def __init__(self, tokenizer: Optional[FakeTokenizer] = None, script: Optional[Callable] = None, *,
                 eos_id: int = 154827, capabilities: Optional[Iterable[str]] = None, index_topk: int = 2048,
                 indexer_short_ok: bool = True, process_last_token: bool = False,
                 indexer_gap_after_prompt: bool = False, point_nbytes: int = 1 << 20, store_bytes_per_token: int = 64,
                 model_id: str = "glm-5.3-flash-fake", delay_s: float = 0.0):
        self.tok = tokenizer or FakeTokenizer()
        self.script = script
        self.eos_id = int(eos_id)
        self.eos_ids: FrozenSet[int] = frozenset(GLM53_EOS_IDS)
        self.capabilities: FrozenSet[str] = frozenset(capabilities if capabilities is not None else
                                                      ("reuse", "on_token", "stream", "skip_prefill"))
        self.index_topk = int(index_topk)
        self.indexer_short_ok = bool(indexer_short_ok)
        self.process_last_token = bool(process_last_token)
        self.indexer_gap_after_prompt = bool(indexer_gap_after_prompt)
        self.point_nbytes = int(point_nbytes)
        self.store_bytes_per_token = int(store_bytes_per_token)
        self.model_id = model_id
        self.delay_s = float(delay_s)
        self.calls: List[Dict[str, Any]] = []
        self.released = 0
        self._busy = threading.Lock()

    def encode(self, text: str) -> List[int]:
        return self.tok.encode(text)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tok.decode(ids)

    def release(self, obj: Any) -> None:
        self.released += 1

    def indexer_gap_safe(self, n_prompt: int, max_new: int) -> bool:
        return self.indexer_short_ok and n_prompt + max_new <= self.index_topk

    def describe(self) -> Dict[str, Any]:
        return {"runner": "fake", "capabilities": sorted(self.capabilities), "calls": len(self.calls)}

    def _output(self, prompt: List[int], h: int) -> List[int]:
        if self.script is None:
            text = f"ans{h % 9973}-{(h >> 13) % 997}"
        else:
            text = self.script(self.tok.decode(prompt), prompt, h)
        ids = list(text) if isinstance(text, list) else self.tok.encode(text)
        if not ids or ids[-1] not in self.eos_ids:
            ids.append(self.eos_id)
        return ids

    def generate(self, req: GenerateRequest) -> GenerateOutcome:
        if not self._busy.acquire(blocking=False):
            raise AssertionError("FakeRunner.generate entered concurrently (the engine must serialize model access)")
        try:
            return self._generate(req)
        finally:
            self._busy.release()

    def _generate(self, req: GenerateRequest) -> GenerateOutcome:
        P = [int(t) for t in req.prompt_ids]
        plan = req.plan
        use = "reuse" in self.capabilities and plan is not None and plan.point is not None and plan.keep > 0
        keep, h = 0, 0
        if use:
            keep = int(plan.keep)
            if plan.entry.store["tokens"][:keep] != P[:keep]:
                raise AssertionError("restore: the stored token history differs from the prompt prefix")
            hp = plan.point.handle
            if hp["n"] != keep or hp["h"] != _fake_hash(0, P[:keep]):
                raise AssertionError(f"restore: the state of point@{hp['n']} is not the state of the prompt prefix")
            h = hp["h"]
        skip = bool(use and plan.skip_prefill)
        if skip:
            if "skip_prefill" not in self.capabilities or keep != len(P):
                raise AssertionError("skip_prefill plan without capability or with tokens left to prefill")
        else:
            if keep >= len(P):
                raise AssertionError("restored the whole prompt without skip_prefill: nothing left to prefill")
            h = _fake_hash(h, P[keep:])
        if self.delay_s:
            time.sleep(self.delay_s)
        out = self._output(P, h)
        first = out[0]
        if skip and plan.point.next_token != first:
            raise AssertionError("skip_prefill: the stored next token is not the model's prediction")
        prompt_point = RestorePoint(n_tokens=len(P), handle={"n": len(P), "h": h}, next_token=first,
                                    nbytes=self.point_nbytes, kind="prompt-end")
        gen = [first]
        stopped = bool(req.on_token(first)) if req.on_token else False
        hs = h
        while not stopped and gen[-1] not in self.eos_ids and len(gen) < req.max_new_tokens and len(gen) < len(out):
            hs = _fake_hash(hs, [gen[-1]])                    # feed the last generated token
            nxt = out[len(gen)]
            if getattr(req, "rewrite", None) is not None:    # [R3.9SMP] the real decoders honour the same hook
                nxt = int(req.rewrite(nxt))
            gen.append(nxt)
            if req.on_token and req.on_token(nxt):
                stopped = True
        n_state = len(P) + len(gen) - 1
        if self.process_last_token and gen[-1] not in self.eos_ids:
            hs = _fake_hash(hs, [gen[-1]])
            n_state += 1
        final = RestorePoint(n_tokens=n_state, handle={"n": n_state, "h": hs}, nbytes=self.point_nbytes, kind="final")
        S = P + gen
        reason = "eos" if gen[-1] in self.eos_ids else ("callback" if stopped else "length")
        self.calls.append({"prompt": len(P), "keep": keep, "prefilled": 0 if skip else len(P) - keep, "gen": len(gen),
                           "skip": skip})
        if "reuse" not in self.capabilities:
            return GenerateOutcome(generated_ids=gen, prefilled=len(P), stop_reason=reason)
        gap = len(P) if (self.indexer_gap_after_prompt and n_state > len(P)) else None
        return GenerateOutcome(generated_ids=gen, n_state=n_state, store={"tokens": S[:n_state]},
                               store_nbytes=n_state * self.store_bytes_per_token, points=[prompt_point, final],
                               indexer_gap=gap, prefilled=0 if skip else len(P) - keep, cached=keep,
                               stop_reason=reason, telemetry={"fake": True})


# ======================================================================================================== real runtime
def _tbytes(t: Any) -> int:
    return int(t.numel()) * int(t.element_size()) if t is not None else 0


def _same_device(t: Any, dev: Any) -> bool:
    d = t.device
    return d.type == dev.type and (dev.index is None or d.index == dev.index or dev.type == "cpu")


class _KdaSnap:
    """Restore-point handle: KDA recurrent states [1, H, hd, hd] + conv states (3 x [1, K-1, H*hd]) of every main KDA
    layer after n tokens, plus h_last (collapsed hidden of position n-1, [1, 1, D]) when known (MTP draft input)."""
    __slots__ = ("n", "kda", "conv", "h_last", "nbytes", "open")

    def __init__(self, n: int, kda: Dict[int, Any], conv: Dict[int, Tuple[Any, Any, Any]], h_last: Any = None,
                 open_rows: Optional[Dict[int, Tuple[Any, Any]]] = None):
        self.n = int(n)
        self.kda = kda
        self.conv = conv
        self.h_last = h_last
        # [R3.8] coded history: the indexer rows (ki, gs) [1, P, hdi] of the pool that is open after n tokens, per DSA
        # layer. The history itself keeps only the keys of complete pools, so a point needs them to carry on.
        self.open = open_rows
        self.nbytes = (sum(_tbytes(t) for t in kda.values()) + sum(_tbytes(t) for cs in conv.values() for t in cs)
                       + _tbytes(h_last) + sum(_tbytes(t) for pair in (open_rows or {}).values() for t in pair))


class _DsaStore:
    """Entry store: DSA latent KV [1, n, C] and indexer caches (ki / gs) of every main DSA layer for tokens[:n].
    indexer_gap = first position whose ki / gs are zero placeholders (decoded by the graphs without R34_DSA_INDEXER).
    [R3.5-INT] mtp_kv / mtp_start: the MTP layer's (45) latent history of graph-path MTP decoding, entries
    [mtp_start, mtp_start + k) (entry i = (h_i, emb(t_{i+1})), k <= n - 1 - mtp_start), or None (eager MTP / no MTP)."""
    __slots__ = ("n", "kv", "ki", "gs", "indexer_gap", "nbytes", "mtp_kv", "mtp_start")

    def __init__(self, n: int, kv: Dict[int, Any], ki: Dict[int, Any], gs: Dict[int, Any],
                 indexer_gap: Optional[int] = None):
        self.n = int(n)
        self.kv, self.ki, self.gs = kv, ki, gs
        self.indexer_gap = indexer_gap
        self.mtp_kv, self.mtp_start = None, None
        self.nbytes = sum(_tbytes(t) for d in (kv, ki, gs) for t in d.values())

    def set_mtp(self, kv: Any, start: int) -> None:
        """[R3.5-INT] attach the layer-45 history (a stored copy) of the same token sequence."""
        self.nbytes -= _tbytes(self.mtp_kv)
        self.mtp_kv, self.mtp_start = kv, int(start)
        self.nbytes += _tbytes(kv)


class _QStore:
    """[R3.8] Entry store in the coded format of glm53_fold_qstate_r38 for tokens[:n]: per main DSA layer the latent
    codes kvc [1, n, code width], their scales kvs [1, n, 1] (none for the fp32 arm) and the pool keys pk
    [1, n // P, hdi]. No per-token indexer rows and therefore no indexer gap; the rows of the open pool live in the
    restore points (_KdaSnap.open). Truncating to an earlier restore point is slicing."""
    __slots__ = ("n", "kvc", "kvs", "pk", "bits", "kpool", "indexer_gap", "nbytes", "mtp_kv", "mtp_start")

    def __init__(self, n: int, kvc: Dict[int, Any], kvs: Dict[int, Any], pk: Dict[int, Any], bits: int, kpool: int):
        self.n = int(n)
        self.kvc, self.kvs, self.pk = kvc, kvs, pk
        self.bits, self.kpool = int(bits), int(kpool)
        self.indexer_gap = None
        self.mtp_kv, self.mtp_start = None, None
        self.nbytes = sum(_tbytes(t) for d in (kvc, kvs, pk) for t in d.values())

    def set_mtp(self, kv: Any, start: int) -> None:
        raise RuntimeError("the coded history has no MTP-layer history")


class ServeState:
    """The hook object passed as ApexFoldMoEModelRunner.generate(..., serve_state=st) (INTEGRATION_S.md H1-H5).

    begin(runner, prompt_ids) -> (cache, n_cached)       [H1: replaces `self._restore_prefix(prompt_ids)`]
    after_prefill(runner, cache, first_token, h_last)      [H2: right after the prefill, before any MTP draft]
    on_token(tok) -> bool / .stopped                       [H3: every later generated token, eager loop and graphs]
    after_decode(runner, cache, gen_ids, graph_done, mtp_graph=False)
                                                           [H4: once decoding ended, before generate() returns]
    first_token / h_last                                   [H5, optional: the plan restored the whole prompt]
    mtp_keep / mtp_restore                                 [R3.5-INT: graph-MTP layer-45 history, keep / put back]
    Failures in H2 / H4 never fail the request: the state is then simply not stored (logged).
    [R3.5-FIX] req.store False (reuse off for this request: nothing will be committed): no prompt-end snapshot, no final
    export and no layer-45 history kept - only the counters and the per-token callbacks run."""

    def __init__(self, adapter: "GLMRunnerAdapter", req: GenerateRequest):
        self.a = adapter
        self.req = req
        self.plan = req.plan
        self.stopped = False
        self.n_cached = 0
        self.first_token: Optional[int] = None
        self.h_last: Any = None
        self.prompt_point: Optional[RestorePoint] = None
        self.final_point: Optional[RestorePoint] = None
        self.store: Optional[_DsaStore] = None
        self.prefilled = 0
        self.graph_done = False
        self.t: Dict[str, float] = {}
        self.error: Optional[str] = None
        self.rewrite = getattr(req, "rewrite", None)                       # [R3.9SMP] thinking budget hook (None = off)
        # [R3.5-INT] graph-path MTP: keep the layer-45 history complete (decode_mtp keep_history) and put a stored one
        # back (decode_mtp mtp_restore) - the "aligned" policy; drafts only, never an emitted token
        # [R3.5-FIX] (drafts only: no emitted token depends on it) not for a request whose state is not stored
        self.mtp_keep = bool(getattr(adapter, "mtp_aligned", False)) and bool(getattr(req, "store", True))
        self.mtp_restore = None
        self.t_mtp_kept: Optional[int] = None                              # layer-45 entries stored with this state

    # H1
    def begin(self, runner, prompt_ids):
        plan = self.plan
        if plan is None or plan.point is None or plan.keep <= 0:
            cache, n_pre, _ = runner._restore_prefix(prompt_ids)           # exactly the base path
            gd38 = self.a.r38_decoder()
            if gd38 is not None:                                           # [R3.8] the prompt pass writes the coded history
                if n_pre:
                    raise RuntimeError("exact prefix caches are not available with the coded history (R38_KV_BITS)")
                cache.r38 = gd38.r38_history(clear=True)
            self.n_cached = int(n_pre)
            return cache, int(n_pre)
        t0 = _now()
        cache = self.a.restore_cache(plan.entry.store, plan.point.handle, int(plan.keep))
        if self.mtp_keep:                                                  # [R3.5-INT] aligned layer-45 history
            self.mtp_restore = self.a.mtp_restore_payload(plan.entry.store, plan.point.handle, int(plan.keep),
                                                          len(prompt_ids), bool(plan.skip_prefill))
        self.t["restore_ms"] = round((_now() - t0) * 1000.0, 2)
        self.n_cached = int(plan.keep)
        if plan.skip_prefill:
            self.first_token = int(plan.point.next_token)
            hl = getattr(plan.point.handle, "h_last", None)
            self.h_last = self.a.to_device(hl) if hl is not None else None
        return cache, int(plan.keep)

    # H2
    def after_prefill(self, runner, cache, first_token, h_last) -> bool:
        if not getattr(self.req, "store", True):                          # [R3.5-FIX] reuse off: no snapshot
            self.prefilled = max(0, int(cache.pos) - self.n_cached)
            return self.on_token(first_token)
        if getattr(self.a, "early_first_token", False):
            # [R3.5-T] --serve-early-token: the first token reaches the client (SSE) BEFORE the prompt-end state copy
            # (~146 MiB D2H); the copy still happens before any decode step changes the state
            stop = self.on_token(first_token)
            self._store_prompt_end(cache, first_token, h_last)
            return stop
        self._store_prompt_end(cache, first_token, h_last)
        return self.on_token(first_token)

    def _store_prompt_end(self, cache, first_token, h_last) -> None:
        """H2's state copy: the prompt-end restore point (KDA / conv states + h_last, next token known)."""
        n = int(cache.pos)
        P = len(self.req.prompt_ids)
        self.prefilled = max(0, n - self.n_cached)
        try:
            if n != P:
                raise RuntimeError(f"cache.pos {n} after the prefill != prompt length {P}")
            t0 = _now()
            snap = self.a.snapshot_kda(cache, n, h_last)
            self.prompt_point = RestorePoint(n_tokens=n, handle=snap, next_token=int(first_token), nbytes=snap.nbytes,
                                             kind="prompt-end")
            self.t["snapshot_ms"] = round((_now() - t0) * 1000.0, 2)
        except Exception as e:
            self.error = f"after_prefill: {type(e).__name__}: {e}"
            self.a.log(f"[serve] [{SERVE_TAG}] state not stored ({self.error})")

    # H3
    def on_token(self, tok) -> bool:
        cb = self.req.on_token
        if cb is not None and not self.stopped and cb(int(tok)):
            self.stopped = True
        return self.stopped

    # H4
    def after_decode(self, runner, cache, gen_ids, graph_done, mtp_graph: bool = False) -> None:
        self.graph_done = bool(graph_done)
        if self.error is not None or not getattr(self.req, "store", True):   # [R3.5-FIX] reuse off: no export
            return
        P = len(self.req.prompt_ids)
        try:
            t0 = _now()
            if graph_done:
                snap, store = self.a.export_graph(runner.graph_decoder, cache, P)
            else:
                snap, store = self.a.export_eager(cache)
            n = store.n
            if not (P <= n <= P + len(gen_ids)):
                raise RuntimeError(f"final state covers {n} tokens; prompt {P}, generated {len(gen_ids)}")
            if graph_done and mtp_graph and self.mtp_keep:                 # [R3.5-INT] graph-MTP layer-45 history
                self.t_mtp_kept = self.a.attach_mtp(runner.graph_decoder, snap, store, n)
            self.final_point = RestorePoint(n_tokens=n, handle=snap, nbytes=snap.nbytes, kind="final")
            self.store = store
            self.t["export_ms"] = round((_now() - t0) * 1000.0, 2)
        except Exception as e:
            self.error = f"after_decode: {type(e).__name__}: {e}"
            self.a.log(f"[serve] [{SERVE_TAG}] state not stored ({self.error})")

    def outcome(self, res: Dict[str, Any]) -> GenerateOutcome:
        gen = [int(t) for t in res.get("generated_ids", [])]
        eos = self.a.eos_ids
        if gen and gen[-1] in eos:
            reason = "eos"
        elif self.stopped:
            reason = "callback"
        elif len(gen) >= self.req.max_new_tokens:
            reason = "length"
        else:
            reason = "context"                                             # e.g. graph decode stopped at t_max
        tel = _slim_telemetry(res)
        tel.update(self.t)
        tel["graph_decode"] = self.graph_done
        if self.mtp_keep:                                                  # [R3.5-INT] layer-45 history (aligned)
            mg = res.get("mtp_graph") or {}
            tel["mtp_history"] = {"restored_entries": None if self.mtp_restore is None else
                                  int(self.mtp_restore[0].shape[1]) + int(self.mtp_restore[2] is not None),
                                  "start": mg.get("history_start"), "kept_entries": self.t_mtp_kept,
                                  "restored": mg.get("history_restored")}
        if self.error:
            tel["state_error"] = self.error
        ok = self.error is None and self.store is not None and self.prompt_point is not None
        # prompt-end first: at equal n (nothing decoded) ConversationCache keeps the first, which carries next_token
        points = [p for p in (self.prompt_point, self.final_point) if p is not None] if ok else []
        return GenerateOutcome(generated_ids=gen, n_state=self.store.n if ok else 0, store=self.store if ok else None,
                               store_nbytes=self.store.nbytes if ok else 0, points=points,
                               indexer_gap=self.store.indexer_gap if ok else None, prefilled=self.prefilled,
                               cached=self.n_cached, stop_reason=reason, telemetry=tel)


def _slim_telemetry(res: Dict[str, Any]) -> Dict[str, Any]:
    keep = ("prompt_tokens", "new_tokens", "prefill_s", "prefill_tok_s", "prefill_moe_s", "decode_s", "decode_tok_s",
            "step_ms_mean", "mtp_accept_rate", "vram_peak_gib")
    out = {k: res[k] for k in keep if k in res}
    pool = res.get("pool")
    if isinstance(pool, dict):
        out["pool"] = {k: pool[k] for k in ("hit_rate", "misses", "gib_h2d") if k in pool}
    for k in ("prefill_prefetch", "continued_past_tmax"):   # [R3.5-T] present only when those options acted
        if k in res:
            out[k] = res[k]
    return out


class GLMRunnerAdapter:
    """ServeRunner over the real runtime: ApexFoldMoEModelRunner + RealTokenizer (glm53_fold_generate.py).

    With the INTEGRATION_S.md hooks (runner.SERVE_HOOKS contains "state" and generate() takes serve_state=) it reuses
    conversation state: restore = a fresh RealCache built from a stored restore point (KDA / conv states) and the
    entry's DSA history truncated to n; the prompt-end point is a copy of the KDA / conv states right after the
    prefill; the final point is exported from the eager cache or from the GraphDecoder's static buffers after graph
    decode. States are copied to `state_device` ("cpu" = pageable host RAM, default: no VRAM held between requests;
    "cuda" = device clones). Without the hooks it calls the base generate() (no reuse, stop strings applied
    afterwards) - multi-turn prompts are still complete and exact.
    MTP-layer (45) history [R3.5-INT]: eager MTP (no graphs / R35_MTP_GRAPHS unset or 0 - the default since R3.5-FIX /
    prompts >= t_max) never builds a prompt history, so restored requests start it empty exactly as fresh ones do
    (reset). Graph-path MTP (R3.5-M2, --mtp with --graphs and env R35_MTP_GRAPHS=1) builds the full prompt history at
    every prefill; with mtp_history="aligned" (default) the
    history is kept complete during decoding, stored with the entry (entries [s, n - 1)) together with h_{n-1} of every
    restore point, and put back below a restored prefix (entry n - 1 recomputed from h_{n-1} and the new token at n), so
    a restored request drafts from the history a full prefill gives; "reset" stores none. Drafts only: never a token."""

    def __init__(self, runner, tokenizer, *, use_mtp: bool = False, state_device: str = "cpu",
                 model_id: str = DEFAULT_MODEL_ID, cache_factory: Optional[Callable[[], Any]] = None,
                 log: Optional[Callable[[str], None]] = None, mtp_history: str = "aligned",
                 early_first_token: bool = False):
        self.r = runner
        self.tok = tokenizer
        self.log = log or (lambda s: print(s, flush=True))
        self.early_first_token = bool(early_first_token)                   # [R3.5-T] --serve-early-token
        cfg = getattr(runner, "cfg", None)
        self.n_main = int(getattr(cfg, "num_hidden_layers", 45))
        mtp_loaded = self.n_main in getattr(runner, "moe_layers", {})
        self.use_mtp = bool(use_mtp) and mtp_loaded
        # [R3.5-INT] graph-path MTP (generate() decodes --mtp on the graphs when the decoder enables it)
        if mtp_history not in ("aligned", "reset"):
            raise ValueError(f"mtp_history {mtp_history!r} (aligned | reset)")
        gd = getattr(runner, "graph_decoder", None)
        gm = getattr(gd, "mtp_enabled", None)
        self.graph_mtp = bool(self.use_mtp and gd is not None and callable(gm) and gm())
        self.mtp_aligned = bool(self.graph_mtp and mtp_history == "aligned"
                                and callable(getattr(gd, "mtp_history_export", None)))
        self.mtp_history = "aligned" if self.mtp_aligned else "reset"
        self.model_id = model_id
        self.eos_ids: FrozenSet[int] = frozenset(int(i) for i in tokenizer.eos_ids)
        hooks = set(getattr(runner, "SERVE_HOOKS", ()) or ())
        try:
            has_kw = "serve_state" in inspect.signature(runner.generate).parameters
        except (TypeError, ValueError):
            has_kw = False
        self.hooked = bool(has_kw and "state" in hooks)
        caps = {"stream"}
        if self.hooked:
            caps |= {"reuse", "on_token"}
            if "skip_prefill" in hooks:
                caps.add("skip_prefill")
        self.capabilities: FrozenSet[str] = frozenset(caps)
        self.hooks = sorted(hooks)
        self.index_topk = int(getattr(cfg, "index_topk", 2048))
        self.indexer_short_ok = bool(getattr(getattr(runner, "bb", None), "dsa_skip_indexer_short", False))
        if state_device not in ("cpu", "cuda"):
            raise ValueError(f"state_device {state_device!r}")
        self.state_device = state_device
        self._cf = cache_factory
        self.device = getattr(runner, "device", None)
        self.warnings: List[str] = []
        self._check_tokenizer()

    # ---- tokenizer side
    def _check_tokenizer(self) -> None:
        if hasattr(self.tok, "_tok") and getattr(self.tok, "_tok") is None:
            raise RuntimeError("the HF tokenizer did not load (RealTokenizer fell back to raw bytes): refusing to serve")
        try:
            ids = [self.tok.encode_raw(t) for t in MARKUP_TOKENS]
        except Exception:
            return
        bad = [t for t, i in zip(MARKUP_TOKENS, ids) if len(i) != 1]
        if bad:
            self.warnings.append(f"tokenizer splits markup tokens {bad}")
        vids = getattr(getattr(self.r, "bb", None), "vocab_ids", None)    # [R3.4-V] trimmed output head
        if vids is not None:
            try:
                kept = set(int(x) for x in vids.tolist())
                miss = [t for t, i in zip(MARKUP_TOKENS, ids) if len(i) == 1 and i[0] not in kept]
                if miss:
                    self.warnings.append(f"--vocab-keep head cannot emit {miss}: generations cannot end / call tools")
            except Exception:
                pass
        gd = getattr(self.r, "graph_decoder", None)
        if self.hooked and gd is not None and not getattr(gd, "dsa_indexer", True):
            self.warnings.append("graph decode runs without DSA indexer caches (R34_DSA_INDEXER=0): states of "
                                 "graph-decoded turns are reused only by requests that stay on the graphs or within "
                                 "index_topk; past ~2k tokens conversations re-prefill more (serve with "
                                 "R34_DSA_INDEXER=1)")
        for w in self.warnings:
            self.log(f"[serve] [{SERVE_TAG}] WARNING: {w}")

    def encode(self, text: str) -> List[int]:
        return [int(i) for i in self.tok.encode_raw(text)]

    def decode(self, ids: Sequence[int]) -> str:
        return self.tok.decode([int(i) for i in ids])

    def release(self, obj: Any) -> None:
        pass                                                              # host / device copies: freed by refcount

    def indexer_gap_safe(self, n_prompt: int, max_new: int) -> bool:
        """A restore point above an indexer gap (zero ki / gs placeholders of graph-decoded positions) is exact for
        this request only if no eager DSA forward with more than index_topk positions can run (only those read ki /
        gs; requires bb.dsa_skip_indexer_short): prompt + budget <= index_topk, or graph decode does the whole decode
        (graphs on, no MTP or graph-path MTP [R3.5-INT], prompt < t_max <= index_topk: the suffix prefill stays below
        index_topk and GraphDecoder never reads ki / gs and stops at t_max - 1)."""
        if not self.indexer_short_ok:
            return False
        if getattr(self.r, "graph_decoder_std", None) is not None:
            # [R3.7 Stage 7] R37_DUAL_GRAPHS: the long-context decoder refuses a state with an indexer gap, so such a
            # restore point is usable only by a request that the window decoder serves (the same rule as
            # ApexFoldMoEModelRunner.select_graph_decoder, which generate() applies)
            gd_sel = self.r.select_graph_decoder(n_prompt, max_new)
            return gd_sel is self.r.graph_decoder_std and (not self.use_mtp or self.graph_mtp)
        if n_prompt + max_new <= self.index_topk:
            return True
        gd = getattr(self.r, "graph_decoder", None)
        t_max = int(getattr(gd, "t_max", 0) or 0) if gd is not None else 0
        # [R3.5-INT] graph-path MTP decodes on the graphs as well (decode_mtp, same t_max - 1 bound, never reads ki / gs)
        return gd is not None and (not self.use_mtp or self.graph_mtp) and n_prompt < t_max <= self.index_topk

    def describe(self) -> Dict[str, Any]:
        return {"runner": "glm53_fold_generate.ApexFoldMoEModelRunner", "hooks": self.hooks, "hooked": self.hooked,
                "capabilities": sorted(self.capabilities), "use_mtp": self.use_mtp, "state_device": self.state_device,
                "graphs": getattr(self.r, "graph_decoder", None) is not None, "index_topk": self.index_topk,
                "indexer_short_ok": self.indexer_short_ok, "graph_mtp": self.graph_mtp,
                "mtp_history": self.mtp_history, "early_first_token": bool(getattr(self, "early_first_token", False)),
                "warnings": self.warnings}

    # ---- generation
    def _apply_sampling(self, sampling: Optional[Dict[str, Any]]) -> Any:
        """[R3.9SMP] write the request's sampler settings to the backbone's sampler (created on first use); MTP serving
        stays greedy (its draft-verify rule compares against the argmax). Returns what the response reports."""
        bb = getattr(self.r, "bb", None)
        if bb is None:
            return "greedy"
        from glm53_fold_sampler_r39 import ensure_sampler
        smp = ensure_sampler(bb, self.device if self.device is not None else "cuda")
        if smp is None:
            return "greedy (R39_SAMPLER=0)"
        s = dict(sampling or {})
        if self.use_mtp and float(s.get("temperature") or 0) > 0:
            smp.set(0.0)
            return "greedy (MTP serving decodes greedily)"
        return smp.set(s.get("temperature"), s.get("top_p"), s.get("top_k"), s.get("seed"))

    def generate(self, req: GenerateRequest) -> GenerateOutcome:
        smp_info = self._apply_sampling(getattr(req, "sampling", None))   # [R3.9SMP]
        out = self._generate(req)
        try:
            out.telemetry = dict(out.telemetry or {}, sampling=smp_info)
        except Exception:                                                  # pragma: no cover
            pass
        return out

    def _generate(self, req: GenerateRequest) -> GenerateOutcome:
        P = [int(t) for t in req.prompt_ids]
        if not self.hooked:
            res = self.r.generate(P, max_new_tokens=int(req.max_new_tokens), eos_ids=set(self.eos_ids),
                                  use_mtp=self.use_mtp)
            gen = [int(t) for t in res["generated_ids"]]
            reason = "eos" if gen and gen[-1] in self.eos_ids else (
                "length" if len(gen) >= req.max_new_tokens else "context")
            if req.on_token is not None:                                   # stop strings, after the fact
                for i, t in enumerate(gen):
                    if req.on_token(t):
                        gen = gen[:i + 1]
                        reason = "callback"
                        break
            return GenerateOutcome(generated_ids=gen, prefilled=len(P), stop_reason=reason,
                                   telemetry=_slim_telemetry(res))
        st = ServeState(self, req)
        res = self.r.generate(P, max_new_tokens=int(req.max_new_tokens), eos_ids=set(self.eos_ids),
                              use_mtp=self.use_mtp, serve_state=st)
        return st.outcome(res)

    # ---- state copies (torch is imported by the runtime; nothing here runs without a ServeState hook call)
    def _store_copy(self, t):
        t = t.detach()
        if self.state_device == "cpu":
            return t.to("cpu", copy=True)
        return t.clone()

    def to_device(self, t):
        if _same_device(t, self.device):
            return t.clone()                                               # never alias the store
        return t.to(self.device).contiguous()

    def _new_cache(self):
        if self._cf is None:
            from glm53_fold_backbone import RealCache
            self._cf = RealCache
        return self._cf()

    def snapshot_kda(self, cache, n: int, h_last=None) -> _KdaSnap:
        kda = {int(l): self._store_copy(s) for l, s in cache.kda_state.items() if int(l) < self.n_main}
        conv = {int(l): tuple(self._store_copy(c) for c in cs) for l, cs in cache.kda_conv.items() if int(l) < self.n_main}
        hl = self._store_copy(h_last) if h_last is not None else None
        hist = getattr(cache, "r38", None)
        return _KdaSnap(n, kda, conv, hl, self._open_rows(hist.dsa) if hist is not None else None)

    # ---- [R3.8] coded history (R38_KV_BITS): the decoder's buffers are the working copy, the store keeps codes
    def r38_decoder(self):
        """The long-context decoder when it keeps the coded history, else None."""
        gd = getattr(self.r, "graph_decoder_long", None) or getattr(self.r, "graph_decoder", None)
        return gd if gd is not None and getattr(gd, "r38_bits", 0) else None

    def _open_rows(self, dsa) -> Dict[int, Tuple[Any, Any]]:
        return {int(l): (self._store_copy(st["ki"]), self._store_copy(st["gs"])) for l, st in dsa.items() if int(l) < self.n_main}

    def _q_store(self, dsa, lpk, codec, n: int) -> _QStore:
        P = int(self.r.cfg.index_kpool)
        main = [int(l) for l in dsa if int(l) < self.n_main]
        kvc = {l: self._store_copy(dsa[l]["kvc"][:, :n]) for l in main}
        kvs = {l: self._store_copy(dsa[l]["kvs"][:, :n]) for l in main} if codec.bits != 32 else {}
        pk = {l: self._store_copy(lpk[l][:, :n // P]) for l in main}
        return _QStore(n, kvc, kvs, pk, codec.bits, P)

    def _restore_q(self, store: _QStore, snap: _KdaSnap, n: int):
        gd = self.r38_decoder()
        if gd is None or int(gd.kvcodec.bits) != store.bits:
            raise RuntimeError("this stored state is coded; the decoder does not keep a coded history of the same width")
        if n >= int(gd.t_max):
            raise RuntimeError(f"{n} stored tokens do not fit the capacity {int(gd.t_max)}")
        P = store.kpool
        if n % P and not snap.open:
            raise RuntimeError(f"restore point {n} has no rows for its open pool")
        hist = gd.r38_history(clear=True)
        for l, st in hist.dsa.items():
            st["kvc"][:, :n].copy_(store.kvc[l][:, :n])
            if store.bits != 32:
                st["kvs"][:, :n].copy_(store.kvs[l][:, :n])
            hist.lpk[l][:, :n // P].copy_(store.pk[l][:, :n // P])
            if snap.open:
                st["ki"].copy_(snap.open[l][0])
                st["gs"].copy_(snap.open[l][1])
        c = self._new_cache()
        c.pos = int(n)
        for l, x in snap.kda.items():
            c.kda_state[l] = self.to_device(x)
        for l, cs in snap.conv.items():
            c.kda_conv[l] = tuple(self.to_device(x) for x in cs)
        c.r38 = hist
        return c

    def export_eager(self, cache, copied: bool = False) -> Tuple[_KdaSnap, _DsaStore]:
        """Final state from an eager RealCache (eager / MTP decode): n = cache.pos; main DSA layers must hold n
        positions; the MTP layer's DSA history is skipped (reset policy). [R3.5-T] copied=True: `cache` already holds
        store copies (GraphDecoder.export_cache(cache, copy=self._store_copy)) - they are kept, not copied again."""
        n = int(cache.pos)
        hist = getattr(cache, "r38", None)
        if hist is not None:                                               # [R3.8] eager decode on the coded history
            return self.snapshot_kda(cache, n, None), self._q_store(hist.dsa, hist.lpk, hist.codec, n)
        if copied:                                                         # [R3.5-T H4b] one copy, made by the export
            keep = lambda t: t                                              # noqa: E731
            snap = _KdaSnap(n, {int(l): s for l, s in cache.kda_state.items() if int(l) < self.n_main},
                            {int(l): tuple(cs) for l, cs in cache.kda_conv.items() if int(l) < self.n_main}, None)
        else:
            keep = self._store_copy
            snap = self.snapshot_kda(cache, n, None)
        kv, ki, gs = {}, {}, {}
        for l, t in cache.dsa_kv.items():
            l = int(l)
            if l >= self.n_main:
                continue
            if int(t.shape[1]) != n:
                raise RuntimeError(f"DSA layer {l} holds {int(t.shape[1])} positions, cache.pos = {n}")
            kv[l] = keep(t[:, :n])
            ki[l] = keep(cache.dsa_ki[l][:, :n])
            gs[l] = keep(cache.dsa_gs[l][:, :n])
        gap = None if getattr(cache, "indexer_complete", True) else int(getattr(cache, "indexer_gap", 0))
        return snap, _DsaStore(n, kv, ki, gs, gap)

    def export_graph(self, gd, cache, n_prompt: int) -> Tuple[_KdaSnap, _DsaStore]:
        """Final state after graph decode = the GraphDecoder's static buffers (the eager `cache` still holds the
        prompt-end state). Uses gd.export_cache(cache) when the runtime provides it (INTEGRATION_S.md H4b: a RealCache
        of the static state, main layers only, ki / gs for every position, attributes indexer_complete / indexer_gap),
        else reads gd.kda / gd.dsa / gd.pos_t directly (the N = 1 buffers, shared by the R3.5-M1 pair set). Without
        in-graph indexer caches (R34_DSA_INDEXER=0, default) the decoded positions get zero ki / gs and indexer_gap =
        the prompt length (such points are restored only for requests that indexer_gap_safe() admits).
        [R3.5-T] the runtime's export_cache takes copy=: it then writes the store copies itself (one copy per tensor,
        the same tensors the direct read below makes); an export_cache without copy= is copied once more here."""
        if getattr(gd, "r38_bits", 0):                                     # [R3.8] the decoder's own coded buffers
            n = int(gd.pos_t.item())
            if not (int(cache.pos) <= n <= int(gd.t_max)):
                raise RuntimeError(f"graph position {n} outside [{int(cache.pos)}, {int(gd.t_max)}]")
            kda = {int(l): self._store_copy(st["state"]) for l, st in gd.kda.items() if int(l) < self.n_main}
            conv = {int(l): (self._store_copy(st["cq"]), self._store_copy(st["ck"]), self._store_copy(st["cv"]))
                    for l, st in gd.kda.items() if int(l) < self.n_main}
            return _KdaSnap(n, kda, conv, None, self._open_rows(gd.dsa)), self._q_store(gd.dsa, gd.lpk, gd.kvcodec, n)
        if hasattr(gd, "export_cache"):
            try:
                sig = inspect.signature(gd.export_cache).parameters
            except (TypeError, ValueError):
                sig = {}
            if "copy" in sig:
                return self.export_eager(gd.export_cache(cache, copy=self._store_copy), copied=True)
            return self.export_eager(gd.export_cache(cache))
        import torch
        n = int(gd.pos_t.item())
        P0 = int(cache.pos)
        if not (P0 <= n <= int(gd.t_max)):
            raise RuntimeError(f"graph position {n} outside [{P0}, {int(gd.t_max)}]")
        kda = {int(l): self._store_copy(st["state"]) for l, st in gd.kda.items() if int(l) < self.n_main}
        conv = {int(l): (self._store_copy(st["cq"]), self._store_copy(st["ck"]), self._store_copy(st["cv"]))
                for l, st in gd.kda.items() if int(l) < self.n_main}
        snap = _KdaSnap(n, kda, conv, None)
        kv, ki, gs = {}, {}, {}
        gap = None
        for l, st in gd.dsa.items():
            l = int(l)
            if l >= self.n_main:                                           # MTP-layer buffers (R3.5-M2): reset policy
                continue
            e_kv = cache.dsa_kv[l]
            kv[l] = self._store_copy(st["kv"][:, :n].to(e_kv.dtype))
            e_ki, e_gs = cache.dsa_ki[l], cache.dsa_gs[l]
            if "ki" in st and "gs" in st:
                ki[l] = self._store_copy(st["ki"][:, :n].to(e_ki.dtype))
                gs[l] = self._store_copy(st["gs"][:, :n].to(e_gs.dtype))
            else:
                pk = e_ki.new_zeros((e_ki.shape[0], n - P0, e_ki.shape[2]))
                pg = e_gs.new_zeros((e_gs.shape[0], n - P0, e_gs.shape[2]))
                ki[l] = self._store_copy(torch.cat([e_ki[:, :P0], pk], dim=1))
                gs[l] = self._store_copy(torch.cat([e_gs[:, :P0], pg], dim=1))
                if n > P0:
                    gap = P0
        return snap, _DsaStore(n, kv, ki, gs, gap)

    def restore_cache(self, store: _DsaStore, snap: _KdaSnap, n: int):
        """A fresh RealCache equal to the model state after tokens[:n]: KDA / conv from the restore point, DSA latents
        and indexer caches truncated to n, pos = n, no MTP-layer history."""
        if snap.n != n or n > store.n:
            raise RuntimeError(f"restore point {snap.n} / store {store.n} cannot restore {n} tokens")
        if getattr(store, "kvc", None) is not None:                        # [R3.8] a _QStore
            return self._restore_q(store, snap, n)
        r38_decoder = getattr(self, "r38_decoder", None)
        if r38_decoder is not None and r38_decoder() is not None:
            raise RuntimeError("this stored state is fp32; the decoder keeps a coded history (R38_KV_BITS)")
        c = self._new_cache()
        c.pos = int(n)
        if os.environ.get('R36_LONGCTX','0')=='1':
            gap=getattr(store,'indexer_gap',None)
            c.indexer_complete=gap is None or n<=gap
            c.indexer_gap=None if c.indexer_complete else int(gap)
        for l, s in snap.kda.items():
            c.kda_state[l] = self.to_device(s)
        for l, cs in snap.conv.items():
            c.kda_conv[l] = tuple(self.to_device(x) for x in cs)
        for l, t in store.kv.items():
            c.dsa_kv[l] = self.to_device(t[:, :n])
            c.dsa_ki[l] = self.to_device(store.ki[l][:, :n])
            c.dsa_gs[l] = self.to_device(store.gs[l][:, :n])
        return c

    # ---- [R3.5-INT] graph-path MTP: the layer-45 history ("aligned" policy)
    def attach_mtp(self, gd, snap: _KdaSnap, store: _DsaStore, n: int) -> Optional[int]:
        """After graph-MTP decoding (decode_mtp keep_history): store the history entries [s, n - 1) with the entry and
        h_{n-1} with the final point (copies on state_device). Returns the number of entries stored (None: none)."""
        f = getattr(gd, "mtp_history_export", None)
        info = f() if callable(f) else None
        if not info or int(info.get("n", -1)) != int(n):
            return None
        store.set_mtp(self._store_copy(info["kv"]), int(info["start"]))
        if info.get("h_last") is not None:
            snap.h_last = self._store_copy(info["h_last"])
            snap.nbytes = (sum(_tbytes(t) for t in snap.kda.values())
                           + sum(_tbytes(t) for cs in snap.conv.values() for t in cs) + _tbytes(snap.h_last))
        return int(info["kv"].shape[1])

    def mtp_restore_payload(self, store: _DsaStore, snap: _KdaSnap, n: int, n_prompt: int, skip_prefill: bool):
        """The decode_mtp(mtp_restore=) argument for a request restored at n tokens: (kv entries [s, n - 1), s, h_prev)
        with h_prev = h_{n-1} of the restore point (None with skip-prefill: the restored prompt end, entry n - 1 is
        written by the first draft from h_last). None when the entry's history does not reach n - 2 or h_{n-1} is
        unknown - the request then drafts from the prefilled suffix only (the reset policy)."""
        kv, s = getattr(store, "mtp_kv", None), getattr(store, "mtp_start", None)
        if kv is None or s is None:
            return None
        e = int(n) - 1
        if not (0 <= int(s) <= e <= int(s) + int(kv.shape[1])):
            return None
        h_prev = None
        if not (skip_prefill and int(n) == int(n_prompt)):
            hl = getattr(snap, "h_last", None)
            if hl is None:
                return None
            h_prev = self.to_device(hl)
        return self.to_device(kv[:, : e - int(s)]), int(s), h_prev


# ======================================================================================================== CLI glue
def add_serve_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The --serve-* options (INTEGRATION_S.md H0 adds them to glm53_fold_generate.py's parser)."""
    g = ap.add_argument_group(f"[{SERVE_TAG}] multi-turn OpenAI server (--serve)")
    g.add_argument("--serve-v2", action="store_true", default=False,
                   help=f"[{SERVE_TAG}] with --serve: the multi-turn server of glm53_fold_serve.py (full chat history, "
                        "tools, reasoning, SSE, conversation-state reuse). [R3.5-INT] the default of --serve now; kept "
                        "for compatibility")
    g.add_argument("--serve-last-message-only", action="store_true",
                   default=os.environ.get("R35_SERVE_LAST_MESSAGE_ONLY", "0") == "1",
                   help="[R3.5-INT] with --serve: the R3.x handler instead (keeps only the LAST message, no tools / "
                        "reasoning / SSE / reuse; exactly the base code while no R3.5 env option is set - e.g. "
                        "R35_MTP_GRAPHS=1 would put --graphs --mtp on graph MTP) - an A/B baseline. Env "
                        "R35_SERVE_LAST_MESSAGE_ONLY=1")
    g.add_argument("--serve-host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    g.add_argument("--serve-model-name", default=DEFAULT_MODEL_ID, help="model id reported by the API")
    g.add_argument("--serve-max-tokens", type=int, default=2048, help="max_tokens when a request gives none")
    g.add_argument("--serve-max-context", type=int, default=16384,
                   help="prompt + completion token limit (fresh one-shot prefills above ~2k tokens may not fit in "
                        "VRAM: see INTEGRATION_S.md H7)")
    g.add_argument("--serve-effort", default="low", choices=["low", "high", "max"],
                   help="default reasoning effort (the template's 'Reasoning Effort' line; encode_chat used low)")
    # [R3.9SMP] sampler defaults (glm53_fold_sampler_r39); a request's temperature / top_p / top_k / seed override them
    g.add_argument("--serve-temperature", type=float, default=0.0,
                   help="default temperature (0 = greedy, bit-identical to the pre-R3.9SMP decode)")
    g.add_argument("--serve-top-p", type=float, default=1.0, help="default nucleus cut (1 = off)")
    g.add_argument("--serve-top-k", type=int, default=0, help="default top-k (0 = the sampler's 64 candidates)")
    g.add_argument("--serve-thinking-budget", type=int, default=0,
                   help="reasoning tokens after which </think> is forced at the next line break (0 = off; "
                        "apex_fold.thinking_budget overrides per request)")
    g.add_argument("--serve-no-reuse", action="store_true", help="disable conversation-state reuse")
    g.add_argument("--serve-cache-entries", type=int, default=2, help="conversations kept for reuse")
    g.add_argument("--serve-cache-points", type=int, default=3, help="restore points kept per conversation")
    g.add_argument("--serve-state-gib", type=float, default=1.0,
                   help="memory budget of the stored states (each restore point ~146 MiB of KDA state, + ~33 KiB per "
                        "token of DSA history); host RAM by default - it competes with the expert page cache")
    g.add_argument("--serve-state-device", default="cpu", choices=["cpu", "cuda"],
                   help="where stored states live: cpu (pageable host RAM, no VRAM held between requests) or cuda")
    g.add_argument("--serve-min-reuse", type=int, default=32, help="restore only when at least this many tokens match")
    g.add_argument("--serve-prefill-chunk", type=int, default=0,
                   help="prefill long uncached suffixes in chunks of N tokens (0 = off; every chunk re-streams its "
                        "experts - use only when a one-shot prefill does not fit)")
    g.add_argument("--serve-mtp-history", default="aligned", choices=["aligned", "reset"],
                   help="[R3.5-INT] graph-path MTP (--mtp --graphs, env R35_MTP_GRAPHS=1): keep the MTP layer's latent "
                        "history with every "
                        "stored state and put it back on reuse (aligned, default: drafts see the history a full prefill "
                        "gives) or start it at the restored prefix (reset). Drafts only - never changes a token")
    g.add_argument("--serve-restore-reasoning", action="store_true",
                   help="re-attach this server's reasoning to returned tool-call turns that come back without it "
                        "(changes the prompt vs a stateless server; off by default)")
    g.add_argument("--serve-early-token", action="store_true",
                   help="[R3.5-T] stream the first token before the prompt-end state copy (~146 MiB D2H) instead of "
                        "after it: lower time to first token for SSE clients; the stored state is the same")
    g.add_argument("--serve-api-key", default=os.environ.get("R35_SERVE_API_KEY", ""),
                   help="require Authorization: Bearer <key> (env R35_SERVE_API_KEY)")
    g.add_argument("--serve-cors", action="store_true", help="send Access-Control-Allow-Origin: * (browser clients)")
    g.add_argument("--serve-queue-timeout", type=float, default=600.0, help="seconds a request waits for the model")
    g.add_argument("--serve-template", default="", help="chat template path (default <backbone-dir>/chat_template.jinja)")
    g.add_argument("--serve-template-engine", default="auto", choices=["auto", "jinja2", "python"])
    g.add_argument("--serve-verbose", action="store_true", help="log every HTTP request line")
    return ap


def serve_v2_selected(args) -> bool:
    """[R3.5-INT] which --serve handler glm53_fold_generate.main() runs: True = this module's multi-turn server (the
    default), False = the R3.x last-message handler (--serve-last-message-only / env R35_SERVE_LAST_MESSAGE_ONLY=1).
    ValueError when both --serve-v2 and --serve-last-message-only are given."""
    lmo = bool(getattr(args, "serve_last_message_only", False))
    if lmo and bool(getattr(args, "serve_v2", False)):
        raise ValueError("--serve-v2 and --serve-last-message-only exclude each other")
    return not lmo


def _engine_from_args(runner_adapter, template: ChatTemplate, args, model_id: str) -> ServeEngine:
    return ServeEngine(runner_adapter, template, model_id=model_id, default_thinking=bool(getattr(args, "thinking", False)),
                       default_effort=getattr(args, "serve_effort", "low"),
                       default_max_tokens=int(getattr(args, "serve_max_tokens", 2048)),
                       max_context=int(getattr(args, "serve_max_context", 16384)),
                       reuse=not bool(getattr(args, "serve_no_reuse", False)),
                       cache_entries=int(getattr(args, "serve_cache_entries", 2)),
                       cache_points=int(getattr(args, "serve_cache_points", 3)),
                       cache_bytes=int(float(getattr(args, "serve_state_gib", 1.0)) * (1 << 30)),
                       min_reuse=int(getattr(args, "serve_min_reuse", 32)),
                       prefill_chunk=int(getattr(args, "serve_prefill_chunk", 0)),
                       restore_reasoning=bool(getattr(args, "serve_restore_reasoning", False)),
                       queue_timeout=float(getattr(args, "serve_queue_timeout", 600.0)),
                       default_sampling={"temperature": float(getattr(args, "serve_temperature", 0.0)),   # [R3.9SMP]
                                         "top_p": float(getattr(args, "serve_top_p", 1.0)),
                                         "top_k": int(getattr(args, "serve_top_k", 0))},
                       default_thinking_budget=int(getattr(args, "serve_thinking_budget", 0)))


def serve_main(runner, tok, args) -> None:
    """Entry point from glm53_fold_generate.main() (INTEGRATION_S.md H0): serve `runner` (ApexFoldMoEModelRunner,
    graphs already captured) with `tok` (RealTokenizer) until interrupted."""
    tpath = getattr(args, "serve_template", "") or os.path.join(getattr(args, "backbone_dir", ""), "chat_template.jinja")
    if not os.path.exists(tpath):
        tpath = DEFAULT_TEMPLATE_PATH
    template = ChatTemplate.from_file(tpath, engine=getattr(args, "serve_template_engine", "auto"))
    adapter = GLMRunnerAdapter(runner, tok, use_mtp=bool(getattr(args, "mtp", False)),
                               state_device=getattr(args, "serve_state_device", "cpu"),
                               model_id=getattr(args, "serve_model_name", DEFAULT_MODEL_ID),
                               mtp_history=getattr(args, "serve_mtp_history", "aligned"),
                               early_first_token=bool(getattr(args, "serve_early_token", False)))
    gd38 = adapter.r38_decoder()
    if gd38 is not None:                                                   # [R3.8] refuse what would fail inside a request
        if int(getattr(args, "serve_max_context", 0) or 0) > int(gd38.t_max):
            raise SystemExit(f"--serve-max-context {args.serve_max_context} exceeds the coded history's capacity "
                             f"{int(gd38.t_max)} (R36_LONGCTX_MAX)")
        if getattr(runner, "graph_decoder_std", None) is not None:
            raise SystemExit("R37_DUAL_GRAPHS=1 cannot be combined with R38_KV_BITS (one decoder owns the coded history)")
        if getattr(runner, "prefix_caches", None):
            raise SystemExit("--prefix-cache cannot be combined with R38_KV_BITS")
        print(f"[serve] [R3.8] coded attention history: {int(gd38.kvcodec.bits)} bits, capacity {int(gd38.t_max)} tokens",
              flush=True)
    engine = _engine_from_args(adapter, template, args, adapter.model_id)
    mtp = ("off" if not adapter.use_mtp else
           f"on ({'graphs' if adapter.graph_mtp else 'eager'}, layer-45 history {adapter.mtp_history})")
    print(f"[serve] [{SERVE_TAG}] template {tpath} ({template.engine}, sha256 {template.sha256[:12]}"
          f"{', GLM-5.3' if template.is_glm53 else ', NOT the known GLM-5.3 template'}); runner hooks {adapter.hooks or 'none'}"
          f" -> reuse {'on' if engine.reuse and adapter.hooked else 'OFF (INTEGRATION_S.md H1-H4 missing)' if not adapter.hooked else 'off'}"
          f"; thinking default {'on' if engine.default_thinking else 'off'}, effort {engine.default_effort}, "
          f"MTP {mtp}, state store {adapter.state_device} "
          f"{engine.cache.max_bytes / 2 ** 30:.2f} GiB", flush=True)
    run_http(engine, host=getattr(args, "serve_host", "127.0.0.1"), port=int(getattr(args, "port", 8088)),
             api_key=getattr(args, "serve_api_key", ""), cors=bool(getattr(args, "serve_cors", False)),
             verbose=bool(getattr(args, "serve_verbose", False)))


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=f"[{SERVE_TAG}] multi-turn OpenAI server for GLM-5.3 APEX-FOLD. The real "
                                             "model is served by glm53_fold_generate.py --serve; this entry "
                                             "point offers --fake (CPU stand-in) and --render (prompt preview).")
    ap.add_argument("--fake", action="store_true", help="serve the CPU FakeRunner (protocol / client testing, no GPU)")
    ap.add_argument("--render", default="", help="JSON file with a chat request body: print its prompt and exit")
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--thinking", action="store_true", help="default thinking on")
    ap.add_argument("--backbone-dir", default="C:/models/GLM53-backbone")
    add_serve_args(ap)
    args = ap.parse_args(argv)
    tpath = args.serve_template or os.path.join(args.backbone_dir, "chat_template.jinja")
    template = ChatTemplate.from_file(tpath, engine=args.serve_template_engine)
    if args.render:
        body = json.load(open(args.render, "r", encoding="utf-8"))
        eng = _engine_from_args(FakeRunner(), template, args, DEFAULT_MODEL_ID)
        prep = eng.prepare(body)
        sys.stdout.write(prep.text + "\n")
        sys.stdout.write(f"# {len(prep.ids)} fake-tokenizer tokens; thinking {prep.thinking}, effort {prep.effort}\n")
        return
    if not args.fake:
        ap.error("serve the real model with glm53_fold_generate.py --serve (INTEGRATION_S.md); "
                 "use --fake for the CPU stand-in")
    engine = _engine_from_args(FakeRunner(), template, args, "glm-5.3-flash-fake")
    run_http(engine, host=args.serve_host, port=args.port, api_key=args.serve_api_key, cors=args.serve_cors,
             verbose=args.serve_verbose)


if __name__ == "__main__":
    main()
