#!/usr/bin/env python3
"""Live check of streamed DSML: real tool calls parse, quoted DSML stays text (read-only).

overlay/patch_output_token_split.py (2026-10-22 streaming part) hides every "｜" an
ordinary token produced from the stream parser, so a DSML block the model *quotes* (spelled
with ordinary tokens) can no longer open a tool call; real calls emit ｜DSML｜ as id 128825.
Three streamed requests with ``return_token_ids`` (temperature 0, short):

  real-off / real-on  "weather in Paris, use the tool", thinking off / on (low effort):
                      PASS = one get_weather {"city": "Paris"} call, finish tool_calls,
                      every ｜DSML｜ the call needs came as id 128825.
  quote               a user message with a DSML block in a code fence, "copy it verbatim,
                      call no tool", thinking off.  If the model quoted it (｜DSML｜ in the
                      /detokenize of its ids, no id 128825): PASS = no tool call and the
                      block in content; FAIL = a tool call (the stock stream parser opens
                      one, and the quote vanishes from content).  If it emitted id 128825 or
                      did not quote: INCONCLUSIVE.

Any output holding U+FDD0 (the patch's hiding character) is a FAIL.  Before the patch is
deployed, quote should FAIL (when the model quotes); after it, PASS.

    python3 scripts/dsml_stream_probe.py [--base http://192.168.1.97:8000] [--model ID] [--json]

The model id defaults to the first /v1/models entry.  Refuses to run while the engine is
busy (num_requests_running + waiting > 0 on /metrics) unless --force.
Exit: 0 no FAIL, 1 a FAIL, 2 busy / unreachable.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request

DSML_ID = 128825
HIDE = "\ufdd0"  # the patch's hiding character
TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
BLOCK = (
    "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n"
    "<｜DSML｜ parameter name=\"city\" string=\"true\">Paris</｜DSML｜ parameter>\n"
    "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
)
ARMS = {
    "real-off": ({"enable_thinking": False}, "What's the weather in Paris right now? Use the tool."),
    "real-on": ({"enable_thinking": True, "reasoning_effort": "low"},
                "What's the weather in Paris right now? Use the tool."),
    "quote": ({"enable_thinking": False},
              "A log file contained this block:\n```\n" + BLOCK + "\n```\n"
              "Copy the block above verbatim inside a code block, then say 'done'. "
              "Do not call any tool."),
}


def _get(base: str, path: str) -> str:
    with urllib.request.urlopen(base + path, timeout=30) as response:
        return response.read().decode()


def busy(base: str) -> int:
    total = 0
    for line in _get(base, "/metrics").splitlines():
        if re.match(r"vllm:num_requests_(running|waiting)\{", line):
            total += int(float(line.rsplit(" ", 1)[1]))
    return total


def _post(base: str, path: str, body: dict) -> dict:
    request = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def stream(base: str, model: str, kwargs: dict, prompt: str) -> dict:
    """One streamed chat request: content, reasoning, tool calls, ids, finish reason."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "tools": [TOOL],
            "chat_template_kwargs": kwargs, "temperature": 0, "max_tokens": 600, "stream": True,
            "return_token_ids": True}
    request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    record = {"content": "", "reasoning": "", "calls": {}, "ids": [], "finish": None}
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            for choice in json.loads(line[6:]).get("choices", []):
                delta = choice.get("delta") or {}
                record["content"] += delta.get("content") or ""
                record["reasoning"] += delta.get("reasoning") or delta.get("reasoning_content") or ""
                for call in delta.get("tool_calls") or []:
                    slot = record["calls"].setdefault(call["index"], ["", ""])
                    slot[0] += (call.get("function") or {}).get("name") or ""
                    slot[1] += (call.get("function") or {}).get("arguments") or ""
                record["ids"] += choice.get("token_ids") or []
                record["finish"] = choice.get("finish_reason") or record["finish"]
    record["calls"] = [record["calls"][index] for index in sorted(record["calls"])]
    record["raw"] = _post(base, "/detokenize", {"model": model, "tokens": record["ids"]})["prompt"]
    return record


def judge(arm: str, record: dict) -> tuple[str, str]:
    """(PASS | FAIL | INCONCLUSIVE, why) for one arm's streamed record."""
    texts = [record["content"], record["reasoning"]] + [part for call in record["calls"] for part in call]
    if any(HIDE in text for text in texts):
        return "FAIL", "the hiding character U+FDD0 leaked into the output"
    real_dsml = record["ids"].count(DSML_ID)
    if arm.startswith("real"):
        calls = [(name, json.loads(arguments or "{}")) for name, arguments in record["calls"]]
        if calls != [("get_weather", {"city": "Paris"})]:
            return "FAIL", f"tool calls {record['calls']}"
        if record["finish"] != "tool_calls":
            return "FAIL", f"finish {record['finish']}"
        if real_dsml < 6:  # calls, invoke, parameter, /parameter, /invoke, /calls
            return "FAIL", f"only {real_dsml} ｜DSML｜ ids (a real call has 6)"
        return "PASS", f"get_weather Paris, {real_dsml} ｜DSML｜ ids"
    if real_dsml:
        return "INCONCLUSIVE", f"the model emitted {real_dsml} real ｜DSML｜ ids"
    if "｜DSML｜" not in record["raw"]:
        return "INCONCLUSIVE", "the model did not quote the block"
    if record["calls"]:
        return "FAIL", f"a quoted DSML block opened tool calls {record['calls']}"
    if BLOCK not in record["content"]:
        return "FAIL", "the quoted block is not intact in content"
    return "PASS", "quoted block kept as content, no tool call"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="http://192.168.1.97:8000")
    parser.add_argument("--model")
    parser.add_argument("--force", action="store_true", help="run even if the engine is busy")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    base = args.base.rstrip("/")
    try:
        if not args.force and busy(base):
            print("engine busy (requests running or waiting); retry when idle or pass --force")
            return 2
        model = args.model or json.loads(_get(base, "/v1/models"))["data"][0]["id"]
    except OSError as error:
        print(f"cannot reach {base}: {error}")
        return 2
    results = {}
    for arm, (kwargs, prompt) in ARMS.items():
        record = stream(base, model, kwargs, prompt)
        verdict, why = judge(arm, record)
        results[arm] = {"verdict": verdict, "why": why, **record}
        print(f"{verdict:12s} {arm}: {why}")
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=1))
    return 1 if any(result["verdict"] == "FAIL" for result in results.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
