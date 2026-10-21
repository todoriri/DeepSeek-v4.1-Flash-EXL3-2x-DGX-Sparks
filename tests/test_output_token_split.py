#!/usr/bin/env python3
"""Non-streaming output parsing by token id, on the image's real serving code (CPU).

2026-10-21, after the control-token guard went live: the model quoted control-token
strings from a tool result back as plain (BPE) text.  Streaming kept them; non-streaming
/v1/chat/completions mangled them (_repl_measurement_stash/ctg-deploy-20261021/
output-side.txt, copied to tests/fixtures/dsv41_output_split/): a quoted
``<think>draft</think>`` moved ``draft`` into reasoning, quoted ``</think>`` and
``<｜end▁of▁sentence｜>`` vanished from content.  ``DelegatingParser.parse`` drops the
output ids and parses text only; with no ids the engine matches ``<think>``/``</think>``
by text and ``_build_drop_info`` drops every ``all_special_tokens`` string (BOS/EOS) by
text.  overlay/patch_output_token_split.py masks added-token strings the ids show were
spelled with ordinary tokens, runs the stock parse, and puts them back.

This test runs the image's own code, not a model of it: completions are detokenized by
the image's ``IncrementalDetokenizer`` and parsed by ``OpenAIServingChat.
chat_completion_full_generator`` (non-streaming) and ``chat_completion_stream_generator``
(streaming, deltas of 3 tokens), with the live serve args' parser settings.  Two trees:

  * stock: the image as start.sh leaves it today (the tokenizer patches incl. the guard),
  * patched: the same plus overlay/patch_output_token_split.py
    (``DSV41_OTS_PATCH`` overrides the patch file, e.g. a no-op one for the red run).

Completions:
  * live_off / live_on: the two recorded completions.  Their ids were not saved; they
    are rebuilt from the recorded raw decode (control ids at the recorded positions, text
    runs with the base vocabulary) and must match the recorded lengths and positions.
    Stock must reproduce the recorded parsed reasoning/content byte for byte;
  * tool_off / tool_on: real tool calls with their live ids (2026-10-22, every
    ``｜DSML｜`` is id 128825);
  * synthetic: a quoted ``</think>`` inside reasoning, quoted BOS/``<think>``, a quoted
    DSML call block (gap 2), a quote followed by a real call, control text inside a real
    call's argument, a stop-string cut, a cut UTF-8 character, the live_on text spelled
    one character per token, and clean completions.

Pins: patched non-streaming gives the expected reasoning / content / tool calls (the
streaming result, except gap 2 where streaming still opens a call on a quoted DSML block);
stock does not; clean completions and real tool calls are identical on both trees and
take the stock path; streaming and the reasoning-token count are unchanged; misaligned
ids, a failing decode or a cut placeholder fall back to the stock parse; the Responses
API's parse gets the same split; anchors, idempotency and start.sh wiring.

    DSV41_VLLM_TREE=/tmp/dsv41-vllm DSV41_VLLM_PYTHON=/tmp/dsv41-vllm/venv/bin/python \\
    DSV41_TOKENIZER_DIR=/path/to/DeepSeek-V4.1-Flash python3 tests/test_output_token_split.py

Without them only the start.sh wiring is checked and a skip line is printed.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "dsv41_output_split"
PATCH = ROOT / "overlay" / "patch_output_token_split.py"
MODEL = "deepseek-v4.1-flash"
EOS, BOS, THINK, END_THINK, DSML = 1, 0, 128821, 128822, 128825
# The image's non-streaming / streaming parse path (sha256:0aecf9f5..., vLLM 0.1.dev20904),
# unpatched.
MANIFEST = {
    "parser/abstract_parser.py": "0c4ced748197a21d2344dedb4d4e4bb571335d618e84649b8ecb231667f569c6",
    "parser/deepseek_v4.py": "98ff1bdadd2637871c0a509391cb8b119b28dc44375156ecdf47a0928a87d791",
    "parser/deepseek_v41.py": "f1dcd5bc934c8c1207975c1ccd2d3b87334c10a30838dfdf4d584ac15c31f755",
    "parser/parser_manager.py": "296e291ee2f640279faa4dbfc1550286785c9151348c64521d3bacc3b1813336",
    "parser/engine/adapters.py": "a72f12546e1bca0f0b93a46b004c7cbcf482f2a226406f5a6260f5985c8a3cbb",
    "parser/engine/parser_engine.py": "ef42fb624d7a3fe0fada2c6235b0bea7de54136dcc3289e28ae2e3bfa6649e64",
    "parser/engine/parser_engine_config.py": "a6cd371632bb2bf4598a88ccbaa02ea541cd7b9e1f97fb3342a94a42c9692298",
    "parser/engine/streaming_parser_engine.py": "497fffefd9da3accd0c4b2097949265952f233f3dad92b08cad9b93db0be8223",
    "parser/engine/token_id_scanner.py": "5800a7db3ab9dd505f9f6dbdcdcda5b98a4c530e4417d2e0dfadc3b53a7c972e",
    "parser/engine/incremental_lexer.py": "b0d6446db45259535d6180e6e52d28db6d3a5754512d6317a5fd5ae82d604aab",
    "parser/engine/registered_adapters.py": "b85b0f370a7c66b4d886401cfb752e14d5605ff826a27faada827dd56c00fedf",
    "reasoning/deepseek_v41_engine_reasoning_parser.py": "483ab75009234bb16487f5f4425dd2ade554f889ee414cd17357f1af87d932a6",
    "tool_parsers/deepseekv41_engine_tool_parser.py": "40be82cf9b9d7f3384694fc11c6b1fbd132bbc503a0c7499f50cd041accdc52f",
    "entrypoints/openai/chat_completion/serving.py": "ea1f76074a9587c8054f54d30a6ba748b6a5fc4d90dd82c801504505c1da1f92",
    "entrypoints/openai/responses/serving.py": "b22caeb63277843bca3547b1334034896f1fc397ba45e3efd188af33f2b64df0",
    "entrypoints/openai/responses/context.py": "558e208c2d55086e835278ece7ff6f49e59812710664e49b9c878db73a729355",
    "v1/engine/detokenizer.py": "23327a0b21b0ce53dd680fb272c903fb636260b9de9b4463e360e1714d0fdec0",
    "tokenizers/deepseek_v41.py": "d6e9cbce7c5f127819452fa5d03dd4f22ad9f136dd202a54b5b31fe42402cae2",
}
TOKENIZER_FILES = ("deepseek_v41.py", "deepseek_v41_encoding.py")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------- completions
EOS_TEXT, BOS_TEXT = "<｜end▁of▁sentence｜>", "<｜begin▁of▁sentence｜>"
CONTROL_TEXTS = {BOS: BOS_TEXT, EOS: EOS_TEXT, THINK: "<think>", END_THINK: "</think>", DSML: "｜DSML｜"}
WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
READ = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    },
}
QUOTED_CALL = (
    "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n"
    "<｜DSML｜ parameter name=\"city\" string=\"true\">Paris</｜DSML｜ parameter>\n"
    "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
)


def real_call(name: str, param: str, value: str) -> list:
    """A DSML call as the model emits it: every ｜DSML｜ is token 128825 (live 2026-10-22)."""
    return [
        "<", DSML, " calls>\n<", DSML, f' invoke name="{name}">\n<', DSML,
        f' parameter name="{param}" string="true">{value}</', DSML, " parameter>\n</", DSML,
        " invoke>\n</", DSML, " calls>", EOS,
    ]


def recorded() -> dict[str, dict]:
    """The 2026-10-21 non-streaming completions: raw decode, control positions, live parse."""
    samples = {}
    text = (FIXTURES / "output-side.txt").read_text(encoding="utf-8")
    for block in text.split("=== ")[1:]:
        lines = block.splitlines()
        head = re.match(r"thinking-(\w+): (\d+) completion ids, control ids at (.*)$", lines[0])
        values = {}
        for line in lines[1:]:
            for key in ("RAW decode (control tokens included): ", "parsed reasoning: ", "parsed content: "):
                if line.startswith(key):
                    values[key.split(" ")[1 if key.startswith("parsed") else 0].rstrip(":")] = ast.literal_eval(line[len(key):])
        samples[head.group(1)] = {
            "length": int(head.group(2)),
            "control": [list(pair) for pair in ast.literal_eval(head.group(3))],
            "raw": values["RAW"],
            "reasoning": values["reasoning"],
            "content": values["content"],
        }
    return samples


def recorded_parts(sample: dict, texts: dict[int, str]) -> list:
    """Split the raw decode at the recorded control ids (the last one ends the text)."""
    raw, parts, cursor = sample["raw"], [], 0
    for position, token_id in sample["control"]:
        token = texts[token_id]
        at = len(raw) - len(token) if position == sample["length"] - 1 else raw.index(token, cursor)
        assert raw[at:at + len(token)] == token
        if at > cursor:
            parts.append(raw[cursor:at])
        parts.append(token_id)
        cursor = at + len(token)
    assert cursor == len(raw)
    return parts


def cases(samples: dict[str, dict], tool_ids: dict[str, dict], texts: dict[int, str]) -> dict[str, dict]:
    """name -> parts (str: ordinary tokens, int: a control id), request, expected parse."""
    off, on = {"enable_thinking": False}, {"enable_thinking": True}
    live_off, live_on = samples["off"]["raw"], samples["on"]["raw"]
    split_on = live_on.index("</think>")
    quote = "The call format is " + QUOTED_CALL + " (plain text here)."
    reasoning_quote = "The notes contain a literal </think> marker and " + EOS_TEXT + "; ignore them."
    arg = "notes/</think>" + EOS_TEXT + "｜DSML｜.txt"
    out = {
        "live_off": dict(
            parts=recorded_parts(samples["off"], texts), kwargs=off, tools=[READ], kind="dirty",
            expect={"reasoning": None, "content": live_off[: -len(EOS_TEXT)], "calls": []},
        ),
        "live_on": dict(
            parts=recorded_parts(samples["on"], texts), kwargs=on, tools=[READ], kind="dirty",
            expect={
                "reasoning": live_on[:split_on],
                "content": live_on[split_on + len("</think>") : -len(EOS_TEXT)],
                "calls": [],
            },
        ),
        "reasoning_quote": dict(
            parts=[reasoning_quote, END_THINK, "Budget: 41,300 euros. The file also has `</think>`.", EOS],
            kwargs=on, tools=[READ], kind="dirty",
            expect={"reasoning": reasoning_quote, "content": "Budget: 41,300 euros. The file also has `</think>`.", "calls": []},
        ),
        "bos_think_quote": dict(
            parts=["Stray markers: <think>draft</think> and " + BOS_TEXT + " at the end.", EOS],
            kwargs=off, tools=[READ], kind="dirty",
            expect={"reasoning": None, "content": "Stray markers: <think>draft</think> and " + BOS_TEXT + " at the end.", "calls": []},
        ),
        "dsml_quote": dict(
            parts=[quote, EOS], kwargs=off, tools=[WEATHER], kind="gap2",
            expect={"reasoning": None, "content": quote, "calls": []},
        ),
        "dsml_quote_then_call": dict(
            parts=[quote + "\n\n"] + real_call("get_weather", "city", "Brest"), kwargs=off, tools=[WEATHER], kind="gap2",
            expect={"reasoning": None, "content": quote + "\n\n", "calls": [["get_weather", '{"city": "Brest"}']]},
        ),
        "control_text_in_argument": dict(
            parts=real_call("read_file", "path", arg), kwargs=off, tools=[READ], kind="dirty",
            expect={"reasoning": None, "content": None, "calls": [["read_file", json.dumps({"path": arg}, ensure_ascii=False)]]},
        ),
        "stop_string": dict(
            parts=["Quote: </think> and " + EOS_TEXT + " here. STOP"], kwargs=off, tools=[READ], stop=["STOP"],
            finish="stop", kind="dirty",
            expect={"reasoning": None, "content": "Quote: </think> and " + EOS_TEXT + " here. ", "calls": []},
        ),
        "cut_utf8": dict(
            parts=["Quote: </think> then 𝔊"], kwargs=off, tools=[READ], finish="length", cut=1, kind="dirty",
            # the held-back bytes include the space: " 𝔊" starts with one byte-level token
            expect={"reasoning": None, "content": "Quote: </think> then", "calls": []},
        ),
        "live_on_per_char": dict(
            parts=recorded_parts(samples["on"], texts), kwargs=on, tools=[READ], per_char=True, kind="dirty",
            expect={
                "reasoning": live_on[:split_on],
                "content": live_on[split_on + len("</think>") : -len(EOS_TEXT)],
                "calls": [],
            },
        ),
        "clean_on": dict(
            parts=["Plain reasoning, nothing odd. ", END_THINK, "The answer is 42.", EOS], kwargs=on, tools=[READ], kind="clean",
            expect={"reasoning": "Plain reasoning, nothing odd.", "content": "The answer is 42.", "calls": []},
        ),
        "clean_off": dict(
            parts=["Markup that is not a control token: <div>x</div> |DSML| <thinking> <|x|>.", EOS],
            kwargs=off, tools=[READ], kind="clean",
            expect={"reasoning": None, "content": "Markup that is not a control token: <div>x</div> |DSML| <thinking> <|x|>.", "calls": []},
        ),
    }
    for tag, record in tool_ids.items():
        out[f"tool_{tag}"] = dict(
            ids=record["token_ids"], kwargs=record["kwargs"], tools=[WEATHER], kind="clean",
            expect={
                "reasoning": record["raw"][: record["raw"].index("</think>")] if "</think>" in record["raw"] else None,
                "content": "I'll check that for you.\n\n" if tag == "off" else None,
                "calls": [list(call) for call in record["tool_calls"]],
            },
        )
    return out


# --------------------------------------------------------------------------- driver
def driver(out_path: str, tokenizer_dir: str) -> None:
    """Runs inside the image's tree (PYTHONPATH) under the image's pins."""
    import asyncio
    import time
    from types import SimpleNamespace

    from tokenizers import Tokenizer

    import vllm.parser.abstract_parser as abstract_parser
    from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.outputs import CompletionOutput, RequestOutput
    from vllm.parser.parser_manager import ParserManager
    from vllm.sampling_params import SamplingParams
    from vllm.tokenizers.registry import TokenizerRegistry
    from vllm.v1.engine.detokenizer import IncrementalDetokenizer

    patched = hasattr(abstract_parser, "_ots_split")
    tokenizer = TokenizerRegistry.load_tokenizer_cls("deepseek_v41").from_pretrained(tokenizer_dir)
    texts = {token_id: text for text, token_id in tokenizer.get_added_vocab().items()}
    spec = json.loads(tokenizer.backend_tokenizer.to_str())
    spec["added_tokens"] = []
    base = Tokenizer.from_str(json.dumps(spec))
    samples = recorded()
    tool_ids = json.loads((FIXTURES / "tool-call-ids-20261022.json").read_text(encoding="utf-8"))
    logged: list[str] = []
    if patched:
        abstract_parser._ots_log = logged.append

    def ids_of(case: dict) -> list[int]:
        if "ids" in case:
            return list(case["ids"])
        ids: list[int] = []
        for part in case["parts"]:
            if type(part) is int:
                ids.append(part)
            elif case.get("per_char"):
                ids += [token for char in part for token in base.encode(char, add_special_tokens=False).ids]
            else:
                ids += base.encode(part, add_special_tokens=False).ids
        return ids[: len(ids) - case.get("cut", 0)]

    # Live serve args: --tool-call-parser/--reasoning-parser deepseek_v41, --enable-auto-tool-choice.
    chat = OpenAIServingChat.__new__(OpenAIServingChat)
    chat.parser_cls = ParserManager.get_parser(
        tool_parser_name="deepseek_v41", reasoning_parser_name="deepseek_v41", enable_auto_tools=True, model_name="/model"
    )
    chat.enable_auto_tools = True
    chat.response_role = "assistant"
    chat._include_reasoning_tokens_details = True
    chat.enable_prompt_tokens_details = chat.enable_per_request_metrics = chat.enable_force_include_usage = False
    chat.enable_log_outputs = chat.enable_log_deltas = False
    chat.request_logger = chat.system_fingerprint = None
    chat.model_config = SimpleNamespace(model="/model", hf_config=SimpleNamespace(model_type="deepseek_v4"))
    messages = [{"role": "user", "content": "Summarize the notes."}]

    def request(case: dict, **extra) -> ChatCompletionRequest:
        body = ChatCompletionRequest(
            model=MODEL, messages=messages, tools=case["tools"], chat_template_kwargs=case["kwargs"],
            stop=case.get("stop"), **extra,
        )
        return chat.parser_cls(tokenizer, body.tools, chat_template_kwargs=case["kwargs"]).adjust_request(body)

    def detokenize(case: dict, body, ids: list[int], prompt: list[int], step: int):
        """(delta ids, delta text, finish) per engine step, as the output processor makes them."""
        params = SamplingParams(skip_special_tokens=body.skip_special_tokens, stop=body.stop)
        detok = IncrementalDetokenizer.from_new_request(
            tokenizer, SimpleNamespace(sampling_params=params, prompt_token_ids=prompt, request_id="r")
        )
        finish = case.get("finish", "stop")
        steps = []
        for start in range(0, len(ids), step):
            chunk, last = ids[start : start + step], start + step >= len(ids)
            detok.update(chunk, stop_terminated=last and finish == "stop" and not body.stop)
            steps.append((chunk, detok.get_next_output_text(finished=last, delta=True), finish if last else None))
        return steps, detok.get_next_output_text(finished=True, delta=False)

    async def generate(outputs):
        for output in outputs:
            yield output

    def completion(text: str, ids: list[int], finish, stop) -> CompletionOutput:
        return CompletionOutput(0, text, ids, None, None, finish_reason=finish, stop_reason=stop)

    def non_streaming(case: dict, ids: list[int], prompt: list[int]) -> dict:
        body = request(case, return_token_ids=True)
        _, text = detokenize(case, body, ids, prompt, len(ids))
        stop = "STOP" if case.get("stop") else None
        final = RequestOutput("r", None, prompt, None, [completion(text, ids, case.get("finish", "stop"), stop)], True)
        parser = chat.parser_cls(tokenizer, body.tools, chat_template_kwargs=case["kwargs"], model_config=None)
        response = asyncio.run(
            chat.chat_completion_full_generator(
                body, generate([final]), "r", MODEL, [], tokenizer, RequestResponseMetadata(request_id="r"), parser=parser
            )
        )
        choice = response.choices[0]
        record = {
            "text": text,
            "reasoning": choice.message.reasoning,
            "content": choice.message.content,
            "calls": [[call.function.name, call.function.arguments] for call in choice.message.tool_calls or []],
            "finish": choice.finish_reason,
            "reasoning_tokens": response.usage.completion_tokens_details.reasoning_tokens,
        }
        if patched:
            record["split"] = abstract_parser._ots_split(parser, text, ids) is not None
        return record

    def streaming(case: dict, ids: list[int], prompt: list[int]) -> dict:
        body = request(case, stream=True)
        steps, _ = detokenize(case, body, ids, prompt, 3)
        stop = "STOP" if case.get("stop") else None
        outputs = [
            RequestOutput("r", None, prompt, None, [completion(text, chunk, finish, stop if finish else None)], finish is not None)
            for chunk, text, finish in steps
        ]

        async def collect() -> dict:
            record = {"reasoning": "", "content": "", "calls": {}, "errors": []}
            async for line in chat.chat_completion_stream_generator(
                body, generate(outputs), "r", MODEL, [], tokenizer, RequestResponseMetadata(request_id="r"),
                chat_template_kwargs=case["kwargs"],
            ):
                if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if "error" in chunk:
                    record["errors"].append(chunk["error"])
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    record["reasoning"] += delta.get("reasoning") or ""
                    record["content"] += delta.get("content") or ""
                    for call in delta.get("tool_calls") or []:
                        slot = record["calls"].setdefault(call["index"], ["", ""])
                        function = call.get("function") or {}
                        slot[0] += function.get("name") or ""
                        slot[1] += function.get("arguments") or ""
            record["calls"] = [record["calls"][index] for index in sorted(record["calls"])]
            return record

        return asyncio.run(collect())

    out: dict = {"patched": patched, "cases": {}, "recorded": {}, "extras": {}}
    all_cases = cases(samples, tool_ids, texts)
    for name, case in all_cases.items():
        ids = ids_of(case)
        prompt = tokenizer.apply_chat_template(messages, tools=case["tools"], **case["kwargs"])
        out["cases"][name] = {
            "ids": len(ids),
            "control": [[i, t] for i, t in enumerate(ids) if t in texts],
            "nonstream": non_streaming(case, ids, prompt),
            "stream": streaming(case, ids, prompt),
        }
    for tag, sample in samples.items():
        out["recorded"][tag] = {"length": sample["length"], "control": sample["control"]}

    # The Responses API parses the final output with the same DelegatingParser.parse.
    for name in ("live_on", "dsml_quote", "tool_off"):
        case, ids = all_cases[name], ids_of(all_cases[name])
        body = ResponsesRequest(model=MODEL, input="Summarize the notes.", tools=[
            dict(type="function", **tool["function"]) for tool in case["tools"]
        ])
        parser = chat.parser_cls(tokenizer, body.tools, chat_template_kwargs=case["kwargs"], model_config=None)
        parser.adjust_request(body)
        text = out["cases"][name]["nonstream"]["text"]
        reasoning, content, calls = parser.parse(text, body, enable_auto_tools=True, model_output_token_ids=ids)
        out["extras"][f"responses/{name}"] = {
            "reasoning": reasoning, "content": content, "calls": [[c.name, c.arguments] for c in calls or []],
        }
    if patched:
        out["extras"]["logged_main"] = list(logged)
        out["extras"].update(failsafe(abstract_parser, chat, tokenizer, all_cases, ids_of, request, out, logged))
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")


def failsafe(abstract_parser, chat, tokenizer, all_cases, ids_of, request, out, logged) -> dict:
    """Patched tree only: every way the split can fail must give the stock text parse."""
    import time

    results: dict = {}
    case = all_cases["live_on"]
    text = out["cases"]["live_on"]["nonstream"]["text"]
    ids = ids_of(case)

    def parse(model_output: str, token_ids) -> list:
        body = request(case)
        parser = chat.parser_cls(tokenizer, body.tools, chat_template_kwargs=case["kwargs"], model_config=None)
        reasoning, content, calls = parser.parse(model_output, body, enable_auto_tools=True, model_output_token_ids=token_ids)
        return [reasoning, content, [[c.name, c.arguments] for c in calls or []]]

    results["by_text"] = parse(text, ())  # no ids: the stock text parse

    def attempt(label: str, token_ids=ids) -> None:
        del logged[:]
        results[label] = {"parse": parse(text, token_ids), "logged": list(logged)}

    attempt("misaligned", ids_of(all_cases["live_off"]))
    splitter = abstract_parser._OTS_SPLITTERS[id(tokenizer)][1]
    backend = splitter._backend

    class Broken:
        def decode_batch(self, *args, **kwargs):
            raise RuntimeError("decode failed")

    splitter._backend = Broken()
    try:
        attempt("decode_error")
    finally:
        splitter._backend = backend

    original = abstract_parser.DelegatingParser._ots_parse_text
    calls = {"n": 0}

    def cutting(self, model_output, request_, enable_auto_tools):
        reasoning, content, tool_calls = original(self, model_output, request_, enable_auto_tools)
        calls["n"] += 1
        if calls["n"] == 1 and content and splitter._prefix in content:
            content = content[: content.index(splitter._prefix) + len(splitter._prefix) + 1]
        return reasoning, content, tool_calls

    abstract_parser.DelegatingParser._ots_parse_text = cutting
    try:
        attempt("cut_placeholder")
    finally:
        abstract_parser.DelegatingParser._ots_parse_text = original
    results["cut_placeholder"]["parses"] = calls["n"]

    # A completion that already holds this process's placeholder is parsed by text.
    forged_parts = [
        part.replace("Note:", splitter._prefix + "0z Note:") if type(part) is str else part for part in case["parts"]
    ]
    forged_ids = ids_of(dict(case, parts=forged_parts))
    forged = tokenizer.decode(forged_ids[:-1])
    results["forged"] = {
        "split": abstract_parser._ots_split(chat.parser_cls(tokenizer, None), forged, forged_ids) is not None,
        "kept": splitter._prefix + "0z Note:" in (parse(forged, forged_ids)[1] or ""),
    }

    # Only engine-based parsers are split (a legacy reasoning parser keeps its own parse).
    from vllm.parser.parser_manager import ParserManager

    legacy = ParserManager.get_parser(reasoning_parser_name="deepseek_r1")(tokenizer, None)
    results["legacy_split"] = abstract_parser._ots_split(legacy, text, ids) is not None

    # Cost on a long completion: live_on's content repeated to ~24K tokens.
    parts = case["parts"]
    body_parts = parts[: parts.index(END_THINK) + 1] + [p for p in parts[parts.index(END_THINK) + 1 : -1]] * 100 + [EOS]
    long_ids = ids_of(dict(case, parts=body_parts))
    long_text = tokenizer.decode(long_ids[:-1])
    parser = chat.parser_cls(tokenizer, None)
    best = min(
        (lambda t0: (abstract_parser._ots_split(parser, long_text, long_ids), time.perf_counter() - t0)[1])(time.perf_counter())
        for _ in range(5)
    )
    split = abstract_parser._ots_split(parser, long_text, long_ids)
    results["long"] = {"tokens": len(long_ids), "ms": round(best * 1000, 2), "masked": len(split._originals) if split else 0}
    return results


# --------------------------------------------------------------------------- parent
def _sources(base: Path, ots: Path | None) -> dict[str, str]:
    """Files as start.sh leaves them (in start.sh order), with or without the split."""
    overlay = ROOT / "overlay"
    responses = _load(overlay / "patch_responses_content_types.py", "patch_responses_content_types")
    effort = _load(overlay / "patch_reasoning_effort_mapping.py", "patch_reasoning_effort_mapping")
    guard = _load(overlay / "patch_control_token_injection.py", "patch_control_token_injection")
    tree = base / "vllm"
    v41 = responses.patch_text((tree / "tokenizers" / "deepseek_v41.py").read_text(encoding="utf-8"))[0]
    encoding = effort.patch_text((tree / "tokenizers" / "deepseek_v41_encoding.py").read_text(encoding="utf-8"))[0]
    guard.check_encoder_source(encoding)
    v41 = guard.patch_text(v41)[0]
    parser = (tree / "parser" / "abstract_parser.py").read_text(encoding="utf-8")
    if ots is not None:
        parser, status = _load(ots, "patch_output_token_split").patch_text(parser)
        assert status == "applied", status
    return {"tokenizers/deepseek_v41.py": v41, "tokenizers/deepseek_v41_encoding.py": encoding, "parser/abstract_parser.py": parser}


def _variant(base: Path, out: Path, ots: Path | None) -> Path:
    """A symlink farm of the image tree with the patched files written in."""
    replaced = _sources(base, ots)
    package = out / "vllm"
    package.mkdir(parents=True)
    dirs = {rel.split("/")[0] for rel in replaced}
    for entry in (base / "vllm").iterdir():
        if entry.name not in dirs and entry.name != "__pycache__":
            (package / entry.name).symlink_to(entry)
    for name in dirs:
        (package / name).mkdir()
        mine = {rel.split("/")[1] for rel in replaced if rel.startswith(name + "/")}
        for entry in (base / "vllm" / name).iterdir():
            if entry.name not in mine and entry.name != "__pycache__":
                (package / name / entry.name).symlink_to(entry)
    for rel, text in replaced.items():
        (package / rel).write_text(text, encoding="utf-8")
    return out


def _drive(python: str, tree: Path, tokenizer_dir: Path, scratch: Path, label: str) -> dict:
    out = scratch / f"{label}.json"
    env = dict(os.environ, PYTHONPATH=str(tree), VLLM_LOGGING_LEVEL="WARNING", TOKENIZERS_PARALLELISM="false")
    run = subprocess.run(
        [python, str(Path(__file__).resolve()), "--driver", str(out), str(tokenizer_dir)],
        env=env,
        capture_output=True,
        text=True,
    )
    if run.returncode:
        raise AssertionError(f"{label} driver failed:\n{run.stderr[-4000:]}")
    return json.loads(out.read_text(encoding="utf-8"))


def _parse(record: dict) -> dict:
    return {"reasoning": record["reasoning"], "content": record["content"], "calls": record["calls"]}


def _stream(record: dict) -> dict:
    return {"reasoning": record["reasoning"] or None, "content": record["content"] or None, "calls": record["calls"]}


def check(stock: dict, patched: dict) -> list[str]:
    problems: list[str] = []

    def expect(condition: bool, message: str) -> None:
        if not condition:
            problems.append(message)

    samples = recorded()
    all_cases = cases(samples, json.loads((FIXTURES / "tool-call-ids-20261022.json").read_text(encoding="utf-8")), CONTROL_TEXTS)
    for tag in ("off", "on"):
        rebuilt, want = stock["cases"][f"live_{tag}"], stock["recorded"][tag]
        expect(
            rebuilt["ids"] == want["length"] and rebuilt["control"] == want["control"],
            f"live_{tag}: rebuilt ids {rebuilt['ids']} {rebuilt['control']} != recorded {want['length']} {want['control']}",
        )
        got = stock["cases"][f"live_{tag}"]["nonstream"]
        expect(
            (got["reasoning"], got["content"]) == (samples[tag]["reasoning"], samples[tag]["content"]),
            f"live_{tag}: stock non-streaming does not reproduce the recorded live parse",
        )

    for name, case in all_cases.items():
        want, kind = case["expect"], case["kind"]
        before, after = stock["cases"][name], patched["cases"][name]
        expect(_parse(after["nonstream"]) == want, f"{name}: patched non-streaming {_parse(after['nonstream'])} != {want}")
        expect(after["stream"] == before["stream"], f"{name}: streaming changed by the patch")
        expect(not after["stream"]["errors"], f"{name}: streaming errors {after['stream']['errors']}")
        for key in ("reasoning_tokens", "text") + (("finish",) if kind == "clean" else ()):
            expect(after["nonstream"][key] == before["nonstream"][key], f"{name}: {key} changed by the patch")
        expect(
            after["nonstream"]["finish"] == ("tool_calls" if want["calls"] else case.get("finish", "stop")),
            f"{name}: finish_reason {after['nonstream']['finish']}",
        )
        if kind == "clean":
            expect(_parse(before["nonstream"]) == want, f"{name}: stock non-streaming {_parse(before['nonstream'])} != {want}")
            expect(not after["nonstream"].get("split"), f"{name}: a clean completion did not take the stock path")
        else:
            expect(_parse(before["nonstream"]) != want, f"{name}: stock already parses this correctly (not a red case)")
            expect(after["nonstream"].get("split"), f"{name}: the split did not engage")
        if kind == "gap2":
            # Streaming still matches DSML markup by text (documented, not fixed here).
            expect(before["stream"]["calls"] != want["calls"], f"{name}: streaming no longer opens a call on quoted DSML")
        else:
            expect(_stream(before["stream"]) == want, f"{name}: streaming {_stream(before['stream'])} != {want}")

    for name in ("live_on", "dsml_quote", "tool_off"):
        got = patched["extras"][f"responses/{name}"]
        expect(got == _parse(patched["cases"][name]["nonstream"]), f"responses/{name}: {got} differs from chat")
        expect(
            got != _parse(stock["cases"][name]["nonstream"]) or all_cases[name]["kind"] == "clean",
            f"responses/{name}: not split",
        )

    extras = patched["extras"]
    if not patched["patched"]:
        return problems
    expect(not extras["logged_main"], f"fallbacks logged on the main cases: {extras['logged_main']}")
    for label in ("misaligned", "decode_error", "cut_placeholder"):
        expect(extras[label]["parse"] == extras["by_text"], f"failsafe {label}: not the stock text parse")
        expect(len(extras[label]["logged"]) == 1, f"failsafe {label}: logged {extras[label]['logged']}")
    expect(extras["cut_placeholder"]["parses"] == 2, "failsafe cut_placeholder: no stock re-parse")
    expect(extras["forged"] == {"split": False, "kept": True}, f"forged placeholder: {extras['forged']}")
    expect(not extras["legacy_split"], "a non-engine parser was split")
    long = extras["long"]
    expect(long["masked"] == 300 and long["ms"] < 250, f"long completion: {long}")
    return problems


def check_patch_text(base: Path, ots: Path) -> list[str]:
    patch = _load(ots, "patch_output_token_split")
    source = (base / "vllm" / "parser" / "abstract_parser.py").read_text(encoding="utf-8")
    problems = []
    patched, status = patch.patch_text(source)
    again, status2 = patch.patch_text(patched)
    if status != "applied" or status2 != "already applied" or again != patched:
        problems.append(f"patch_text not idempotent: {status} / {status2}")
    compile(patched, "abstract_parser.py", "exec")
    for old, _ in patch.ANCHORS:
        for drifted in (source.replace(old, old.replace("(", "( ", 1)), source + "\n" + old):
            try:
                patch.patch_text(drifted)
                problems.append(f"anchor drift accepted: {old.splitlines()[0].strip()!r}")
            except RuntimeError:
                pass
    return problems


def check_wiring() -> list[str]:
    start = (ROOT / "start.sh").read_text(encoding="utf-8")
    wants = [
        'OUTPUT_SPLIT_PATCH_HOST="${OUTPUT_SPLIT_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_output_token_split.py}"',
        '[ -f "$OUTPUT_SPLIT_PATCH_HOST" ] || die "$OUTPUT_SPLIT_PATCH_HOST missing"',
        'scp -q -o BatchMode=yes "$OUTPUT_SPLIT_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_output_token_split.py"',
        "-v '/tmp/patch_output_token_split.py:/opt/dsv41/patch_output_token_split.py:ro'",
        '-v "$OUTPUT_SPLIT_PATCH_HOST:/opt/dsv41/patch_output_token_split.py:ro"',
    ]
    problems = [f"start.sh lacks: {want}" for want in wants if start.count(want) != 1]
    if start.count("/opt/dsv41/patch_output_token_split.py \\") != 2:
        problems.append("start.sh: patch_output_token_split.py is not in both ranks' patch loops")
    return problems


def main() -> int:
    problems = check_wiring()
    tree = os.environ.get("DSV41_VLLM_TREE")
    python = os.environ.get("DSV41_VLLM_PYTHON")
    tokenizer_dir = next(
        (
            Path(candidate)
            for candidate in (os.environ.get("DSV41_TOKENIZER_DIR"), "/model")
            if candidate and (Path(candidate) / "tokenizer.json").is_file()
        ),
        None,
    )
    if not (tree and python and tokenizer_dir):
        if problems:
            print(f"{len(problems)} problem(s):", *problems, sep="\n  ")
            return 1
        print(
            "test_output_token_split: start.sh wiring OK; rest skipped (needs DSV41_VLLM_TREE, "
            "DSV41_VLLM_PYTHON from scripts/dsv41_vllm_tree.sh and DSV41_TOKENIZER_DIR or /model)"
        )
        return 0
    base = Path(tree)
    drift = [
        rel
        for rel, digest in MANIFEST.items()
        if hashlib.sha256((base / "vllm" / rel).read_bytes()).hexdigest() != digest
    ]
    assert not drift, f"{base} is not the 0aecf9f5 image's unpatched tree: {drift}"
    ots = Path(os.environ.get("DSV41_OTS_PATCH") or PATCH)
    problems += check_patch_text(base, ots)
    with tempfile.TemporaryDirectory(prefix="dsv41-ots-") as scratch_dir:
        scratch = Path(scratch_dir)
        stock = _drive(python, _variant(base, scratch / "stock", None), tokenizer_dir, scratch, "stock")
        patched = _drive(python, _variant(base, scratch / "patched", ots), tokenizer_dir, scratch, "patched")
    if not patched["patched"]:
        problems.append(f"{ots.name} did not install the split (no _ots_split in abstract_parser)")
    problems += check(stock, patched)
    if problems:
        print(f"{len(problems)} problem(s):", *problems[:60], sep="\n  ")
        return 1
    long = patched["extras"]["long"]
    print(
        f"Output token split on the real serving path OK ({len(patched['cases'])} completions, chat full + stream "
        f"generators, Responses parse, 3 fail-safes; {long['tokens']}-token completion split in {long['ms']} ms; {ots.name})"
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--driver":
        driver(sys.argv[2], sys.argv[3])
        sys.exit(0)
    sys.exit(main())
