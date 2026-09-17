#!/usr/bin/env python3
"""
Parity harness for the DeepSeek-V4.1 Jinja chat template.

Renders a set of conversations with BOTH the checkpoint's reference encoder
(encoding/encoding.py, the only prompt-format spec DeepSeek ships) and the Jinja template,
and diffs the strings. Also checks that the rendered markers tokenize as single special tokens.

    python test_chat_template_v41.py [--src ~/NewModels/DeepSeek-V4.1-Flash] [--template chat_template_v41.jinja]
"""
import argparse, difflib, importlib.util, json, os, sys


def load_reference(src):
    spec = importlib.util.spec_from_file_location("dsv41_encoding", os.path.join(src, "encoding", "encoding.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def jinja_env():
    """The same environment transformers uses for chat templates (tojson without HTML escaping)."""
    from transformers.utils.chat_template_utils import _compile_jinja_template
    return _compile_jinja_template


CASES = [
    ("plain user, thinking",
     dict(messages = [{"role": "user", "content": "How many r in strawberry?"}])),
    ("plain user, chat mode",
     dict(messages = [{"role": "user", "content": "How many r in strawberry?"}], enable_thinking = False)),
    ("system + user",
     dict(messages = [{"role": "system", "content": "You are terse."},
                      {"role": "user", "content": "Hi"}])),
    ("system + user, chat mode",
     dict(messages = [{"role": "system", "content": "You are terse."},
                      {"role": "user", "content": "Hi"}], enable_thinking = False)),
    ("reasoning effort low",
     dict(messages = [{"role": "user", "content": "Hi"}], reasoning_effort = "low")),
    ("reasoning effort int",
     dict(messages = [{"role": "user", "content": "Hi"}], reasoning_effort = 42)),
    ("multi turn, reasoning dropped before last user",
     dict(messages = [{"role": "user", "content": "First question"},
                      {"role": "assistant", "content": "First answer", "reasoning_content": "thinking hard"},
                      {"role": "user", "content": "Second question"}])),
    ("multi turn, chat mode",
     dict(messages = [{"role": "user", "content": "First question"},
                      {"role": "assistant", "content": "First answer"},
                      {"role": "user", "content": "Second question"}], enable_thinking = False)),
    ("consecutive user messages merge",
     dict(messages = [{"role": "user", "content": "part one"},
                      {"role": "user", "content": "part two"}])),
    ("mid-conversation system message",
     dict(messages = [{"role": "user", "content": "Hi"},
                      {"role": "assistant", "content": "Hello"},
                      {"role": "system", "content": "Be formal from now on."}])),
    ("tools in system message",
     dict(messages = [{"role": "system", "content": "You may use tools."},
                      {"role": "user", "content": "Weather in Paris?"}],
          tools = [{"type": "function", "function": {
              "name": "get_weather", "description": "Get weather",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}])),
    ("assistant tool call + tool result",
     dict(messages = [{"role": "system", "content": "You may use tools."},
                      {"role": "user", "content": "Weather in Paris?"},
                      {"role": "assistant", "content": "", "reasoning_content": "need the tool",
                       "tool_calls": [{"type": "function", "id": "c1", "function": {
                           "name": "get_weather", "arguments": {"city": "Paris", "days": 3, "metric": True}}}]},
                      {"role": "tool", "tool_call_id": "c1", "content": "18C, cloudy"},
                      {"role": "user", "content": "And tomorrow?"}],
          tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}])),
    ("two tool results merge into one user turn",
     dict(messages = [{"role": "user", "content": "Compare Paris and Rome"},
                      {"role": "assistant", "content": "", "tool_calls": [
                          {"type": "function", "id": "a", "function": {"name": "w", "arguments": {"city": "Paris"}}},
                          {"type": "function", "id": "b", "function": {"name": "w", "arguments": {"city": "Rome"}}}]},
                      {"role": "tool", "tool_call_id": "a", "content": "18C"},
                      {"role": "tool", "tool_call_id": "b", "content": "24C"}])),
    ("chat mode with tools",
     dict(messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Weather?"}],
          enable_thinking = False,
          tools = [{"type": "function", "function": {"name": "w", "parameters": {"type": "object"}}}])),
    ("response_format schema",
     dict(messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Hi"}],
          response_format = {"type": "json_object", "schema": {"type": "object", "properties": {"a": {"type": "string"}}}})),
    ("three turns with two assistant replies",
     dict(messages = [{"role": "user", "content": "q1"},
                      {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
                      {"role": "user", "content": "q2"},
                      {"role": "assistant", "content": "a2", "reasoning_content": "r2"},
                      {"role": "user", "content": "q3"}])),
    ("tool call arguments as an unparseable string",
     dict(messages = [{"role": "user", "content": "go"},
                      {"role": "assistant", "content": "", "tool_calls": [
                          {"type": "function", "id": "x", "function": {"name": "f", "arguments": "not json"}}]},
                      {"role": "tool", "tool_call_id": "x", "content": "ok"}])),
    ("out-of-order tool results",
     dict(messages = [{"role": "user", "content": "both"},
                      {"role": "assistant", "content": "", "tool_calls": [
                          {"type": "function", "id": "a", "function": {"name": "w", "arguments": {"c": "P"}}},
                          {"type": "function", "id": "b", "function": {"name": "w", "arguments": {"c": "R"}}}]},
                      {"role": "tool", "tool_call_id": "b", "content": "second"},
                      {"role": "tool", "tool_call_id": "a", "content": "first"}])),
    ("unicode and json-ish content",
     dict(messages = [{"role": "user", "content": "quote \" brace { } accent é 中文"}])),
]


# The `namespace` tool extension (reference encoding.py commit dba1be0) is deliberately NOT
# ported into the template. It is kept out of CASES (which assert byte-for-byte parity) because
# the template is expected to reject it, not match the reference. See check_namespace_deferral.
NAMESPACE_CASE = dict(
    messages = [{"role": "system", "content": "You may use tools."},
                {"role": "user", "content": "Search it"}],
    tools = [{"type": "function",
              "namespace": {"name": "search", "description": "Search tools."},
              "function": {"name": "lookup", "description": "Look up a value",
                           "parameters": {"type": "object",
                                          "properties": {"query": {"type": "string"}}}}}],
)


def reference_render(enc, case):
    """encode_messages with the same defaults the template targets."""
    msgs = json.loads(json.dumps(case["messages"]))
    tools = case.get("tools")
    if tools:
        msgs[0]["tools"] = tools
    kwargs = {}
    if "reasoning_effort" in case:
        kwargs["reasoning_effort"] = case["reasoning_effort"]
    if case.get("response_format"):
        msgs[0]["response_format"] = case["response_format"]
    return enc.encode_messages(
        msgs,
        thinking_mode = "thinking" if case.get("enable_thinking", True) else "chat",
        **kwargs,
    )


def check_namespace_deferral(enc, template):
    """Durable guard for the deliberate non-adoption of the `namespace` tool extension
    (reference encoding.py commit dba1be0). Upstream qualifies tool names as `namespace::name`
    and folds namespace.description into the description; this template does neither and must
    fail closed instead of silently diverging. Returns True if the deferral still holds.
    If this ever starts passing silently, someone dropped the guard without porting dba1be0."""
    try:
        template.render(messages = NAMESPACE_CASE["messages"], tools = NAMESPACE_CASE["tools"],
                        add_generation_prompt = True)
    except Exception as e:
        print(f"[OK]   template fails closed on namespaced tools ({type(e).__name__})")
        ok = True
    else:
        print("[FAIL] template rendered a namespaced tool silently -- guard missing; "
              "port encoding.py dba1be0 into template AND output parser, or restore the guard")
        ok = False
    # Best-effort: record whether the pinned reference checkout even carries the feature.
    if getattr(enc, "_tool_name_for_encoding", None) is not None:
        print("[INFO] reference checkout has tool namespaces (>= dba1be0); a port must emit "
              "`namespace::name` and prepend namespace.description")
    else:
        print("[INFO] reference checkout predates tool namespaces (< dba1be0)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--src", default = os.path.expanduser("~/NewModels/DeepSeek-V4.1-Flash"))
    ap.add_argument("--template", default = os.path.join(here, "chat_template_v41.jinja"))
    ap.add_argument("-v", "--verbose", action = "store_true")
    args = ap.parse_args()

    enc = load_reference(args.src)
    compile_template = jinja_env()
    template = compile_template(open(args.template).read())

    failures = 0
    for name, case in CASES:
        ref = reference_render(enc, case)
        got = template.render(
            messages = case["messages"],
            tools = case.get("tools"),
            add_generation_prompt = True,
            enable_thinking = case.get("enable_thinking", True),
            reasoning_effort = case.get("reasoning_effort"),
            response_format = case.get("response_format"),
        )
        if ref == got:
            print(f"[OK]   {name}")
            if args.verbose:
                print("       " + ref.replace("\n", "\\n")[:200])
        else:
            failures += 1
            print(f"[FAIL] {name}")
            for line in difflib.unified_diff(ref.splitlines(), got.splitlines(),
                                             "reference", "jinja", lineterm = "", n = 1):
                print("       " + line[:200])
    print(f"\n{len(CASES) - failures}/{len(CASES)} cases match the reference encoder")

    # Deliberate non-adoption of the `namespace` extension must stay fail-closed
    if not check_namespace_deferral(enc, template):
        failures += 1

    # Special tokens must survive tokenization as single ids
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(args.src, "tokenizer.json"))
    sample = template.render(messages = [{"role": "user", "content": "hi"}], add_generation_prompt = True)
    ids = tok.encode(sample, add_special_tokens = False).ids
    for marker in ["<｜begin▁of▁sentence｜>", "<｜System｜>", "<｜User｜>",
                   "<｜Assistant｜>", "<think>"]:
        tid = tok.token_to_id(marker)
        assert tid is not None and tid in ids, f"{marker!r} did not tokenize as a single special token"
    print(f"[OK]   markers tokenize as single ids; sample prompt = {len(ids)} tokens")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
