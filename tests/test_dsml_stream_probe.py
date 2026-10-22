#!/usr/bin/env python3
"""scripts/dsml_stream_probe.py's verdicts on canned streamed records (no server)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("dsml_stream_probe", ROOT / "scripts" / "dsml_stream_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

REAL_IDS = [30, probe.DSML_ID, 7] * 6 + [1]
PARIS = [["get_weather", '{"city": "Paris"}']]


def record(content="", reasoning="", calls=(), ids=(), finish="stop", raw=None) -> dict:
    """``raw``: the /detokenize of the ids (default: what the model wrote, content + reasoning)."""
    return {"content": content, "reasoning": reasoning, "calls": [list(c) for c in calls], "ids": list(ids),
            "finish": finish, "raw": reasoning + content if raw is None else raw}


CASES = [
    ("real-off", record(calls=PARIS, ids=REAL_IDS, finish="tool_calls"), "PASS"),
    ("real-on", record(reasoning="call it", calls=PARIS, ids=REAL_IDS, finish="tool_calls"), "PASS"),
    ("real-off", record(content="It is sunny.", ids=[5, 6, 1]), "FAIL"),  # no call
    ("real-off", record(calls=PARIS + PARIS, ids=REAL_IDS, finish="tool_calls"), "FAIL"),  # two calls
    ("real-off", record(calls=[["get_weather", '{"city": "Brest"}']], ids=REAL_IDS, finish="tool_calls"), "FAIL"),
    ("real-off", record(calls=PARIS, ids=REAL_IDS, finish="stop"), "FAIL"),
    ("real-off", record(calls=PARIS, ids=[5] * 20, finish="tool_calls"), "FAIL"),  # call from text-spelled DSML
    ("real-off", record(calls=[["get_weather", '{"city": "Pa﷐ris"}']], ids=REAL_IDS, finish="tool_calls"), "FAIL"),
    ("quote", record(content="```\n" + probe.BLOCK + "\n```\ndone", ids=[5] * 40), "PASS"),
    # stock streaming: the quote became a call and vanished from content
    ("quote", record(content="```\n\n```\ndone", calls=PARIS, ids=[5] * 40, finish="tool_calls",
                     raw="```\n" + probe.BLOCK + "\n```\ndone"), "FAIL"),
    ("quote", record(content="```\n<｜DSML｜ calls>", calls=PARIS, ids=[5] * 40, finish="tool_calls"), "FAIL"),
    ("quote", record(content="```\n<｜DSML｜ calls> cut", ids=[5] * 40), "FAIL"),  # quoted, not intact
    ("quote", record(content=probe.BLOCK, calls=PARIS, ids=REAL_IDS, finish="tool_calls"), "INCONCLUSIVE"),
    ("quote", record(content="I won't copy that.", ids=[5] * 9), "INCONCLUSIVE"),
    ("quote", record(content=probe.BLOCK.replace("｜", "﷐"), ids=[5] * 40), "FAIL"),
]


def main() -> int:
    problems = []
    for index, (arm, rec, want) in enumerate(CASES):
        got, why = probe.judge(arm, rec)
        if got != want:
            problems.append(f"case {index} ({arm}): {got} ({why}) != {want}")
    if problems:
        print(f"{len(problems)} problem(s):", *problems, sep="\n  ")
        return 1
    print(f"dsml_stream_probe verdicts OK ({len(CASES)} canned records)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
