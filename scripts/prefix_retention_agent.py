#!/usr/bin/env python3
"""Measure prefix-cache reuse on agent-shaped traffic (upstream retention checklist 2-4).

Each request's ``usage.prompt_tokens_details.cached_tokens`` is compared with the
longest common token prefix (LCP) it shares with any earlier prompt in the run
(``/tokenize`` with the same messages/tools/kwargs).  LCP is the reuse an ideal
cache could give; ``short`` = LCP - cached is what the hybrid-attention /
DSpark alignment plus retention spacing lost.  Tokens the model generated are
also cached by vLLM, so ``cached`` can exceed LCP slightly.

Scenarios (fresh random salt per run, so the first request is always cold):
  append      one tool-using session, N user turns (tool call -> tool result -> answer)
  interleave  two sessions alternating turns (A1 B1 A2 B2 ...)
  fork        re-send the append session truncated after turn 2 with a new question
  thinking    append-only session with thinking on (history drops prior reasoning)

    python3 scripts/prefix_retention_agent.py --base http://192.168.1.97:8000
"""
from __future__ import annotations

import argparse
import json
import random
import string
import sys
import time
import urllib.error
import urllib.request

TOOLS = [
    {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": list(props)}}}
    for name, desc, props in [
        ("read_file", "Read a UTF-8 text file from the workspace.",
         {"path": {"type": "string", "description": "Workspace-relative path"}}),
        ("write_file", "Overwrite a text file in the workspace.",
         {"path": {"type": "string"}, "content": {"type": "string"}}),
        ("run_shell", "Run a shell command and return stdout/stderr.",
         {"command": {"type": "string"}, "timeout_s": {"type": "integer"}}),
        ("search_code", "Regex search across the repository.",
         {"pattern": {"type": "string"}, "glob": {"type": "string"}}),
        ("list_dir", "List a directory.", {"path": {"type": "string"}}),
        ("http_get", "Fetch a URL.", {"url": {"type": "string"}}),
        ("git_diff", "Show the working-tree diff for a path.", {"path": {"type": "string"}}),
        ("ask_user", "Ask the user a clarifying question.", {"question": {"type": "string"}}),
    ]
]


class Client:
    def __init__(self, base: str, model: str):
        self.base, self.model = base, model

    def post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.base + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=1800) as f:
                return json.load(f)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{path} HTTP {e.code}: {e.read().decode()[:300]}") from None

    def kwargs(self, thinking: bool) -> dict:
        return {"enable_thinking": thinking}

    def tokenize(self, messages, thinking) -> list[int]:
        d = self.post("/tokenize", {"model": self.model, "messages": messages, "tools": TOOLS,
                                    "chat_template_kwargs": self.kwargs(thinking)})
        return d["tokens"]

    def chat(self, messages, thinking, max_tokens=384) -> dict:
        return self.post("/v1/chat/completions", {
            "model": self.model, "messages": messages, "tools": TOOLS,
            "temperature": 0.0, "max_tokens": max_tokens,
            "chat_template_kwargs": self.kwargs(thinking)})


def words(rng: random.Random, n: int) -> str:
    return " ".join("".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 9)))
                    for _ in range(n))


def fake_source(rng: random.Random, path: str, n_funcs: int = 25) -> str:
    out = [f"# {path}"]
    for i in range(n_funcs):
        fn = "_".join(words(rng, 2).split())
        out.append(f"def {fn}_{i}(x, y=None):\n    \"\"\"{words(rng, 12)}.\"\"\"\n"
                   f"    return (x or 0) + {rng.randint(1, 999)}\n")
    return "\n".join(out)


def lcp(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class Recorder:
    def __init__(self, client: Client):
        self.client, self.history, self.rows = client, [], []

    def send(self, scenario: str, step: str, messages: list, thinking: bool) -> dict:
        try:
            toks = self.client.tokenize(messages, thinking)
            best = max((lcp(toks, h) for h in self.history), default=0)
            self.history.append(toks)
        except RuntimeError as e:  # e.g. vllm#57730 on tool_calls history
            toks, best = None, None
            print(f"  tokenize failed: {e}", flush=True)
        d = self.client.chat(messages, thinking)
        u = d["usage"]
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
        msg = d["choices"][0]["message"]
        row = dict(scenario=scenario, step=step, prompt=u["prompt_tokens"], lcp=best,
                   cached=cached, short=None if best is None or cached is None else best - cached,
                   tool_calls=len(msg.get("tool_calls") or []),
                   content=(msg.get("content") or "")[:60].replace("\n", " "))
        self.rows.append(row)
        print(f"  {scenario:10} {step:10} prompt={row['prompt']:6} lcp={row['lcp']!s:>6} "
              f"cached={row['cached']!s:>6} short={row['short']!s:>6} "
              f"tools={row['tool_calls']} {row['content']!r}", flush=True)
        return msg


def agent_turn(rec: Recorder, rng, scenario: str, messages: list, turn: int, thinking: bool):
    """User asks about a file -> model (hopefully) calls read_file -> tool result -> answer."""
    path = f"src/{'_'.join(words(rng, 2).split())}_{turn}.py"
    messages.append({"role": "user", "content":
                     f"Use read_file to open {path}, then tell me in one sentence what it does."})
    msg = rec.send(scenario, f"t{turn}.call", messages, thinking)
    calls = msg.get("tool_calls") or []
    if not calls:
        messages.append({"role": "assistant", "content": msg.get("content") or ""})
        return
    messages.append({"role": "assistant", "content": msg.get("content") or "",
                     "tool_calls": [{"id": c["id"], "type": "function",
                                     "function": {"name": c["function"]["name"],
                                                  "arguments": c["function"]["arguments"]}}
                                    for c in calls]})
    for c in calls:
        messages.append({"role": "tool", "tool_call_id": c["id"],
                         "content": fake_source(rng, path)})
    msg = rec.send(scenario, f"t{turn}.answer", messages, thinking)
    messages.append({"role": "assistant", "content": msg.get("content") or ""})


def system_prompt(rng, n_words: int) -> dict:
    return {"role": "system", "content":
            "You are a coding agent working in a repository. Always use tools to inspect "
            "files before answering. Keep answers to one sentence.\n\nProject notes:\n"
            + words(rng, n_words)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://192.168.1.97:8000")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--system-words", type=int, default=4000, help="~2.2 tokens per word")
    ap.add_argument("--scenarios", default="append,fork,interleave,thinking")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rng = random.Random(time.time_ns())
    rec = Recorder(Client(args.base, args.model))
    todo = args.scenarios.split(",")
    append_msgs = None

    if "append" in todo or "fork" in todo:
        print("== append", flush=True)
        append_msgs = [system_prompt(rng, args.system_words)]
        snapshot = None
        for t in range(1, args.turns + 1):
            agent_turn(rec, rng, "append", append_msgs, t, False)
            if t == 2:
                snapshot = list(append_msgs)
        if "fork" in todo and snapshot:
            print("== fork (back to after turn 2)", flush=True)
            fork = snapshot + [{"role": "user", "content":
                                "Instead: which of the files you read has more functions?"}]
            rec.send("fork", "f1", fork, False)

    if "interleave" in todo:
        print("== interleave", flush=True)
        a = [system_prompt(rng, args.system_words)]
        b = [system_prompt(rng, args.system_words)]
        for t in range(1, max(2, args.turns // 2) + 1):
            agent_turn(rec, rng, "inter-A", a, t, False)
            agent_turn(rec, rng, "inter-B", b, t, False)

    if "thinking" in todo:
        print("== thinking", flush=True)
        m = [system_prompt(rng, args.system_words)]
        for t in range(1, max(2, args.turns // 2) + 1):
            agent_turn(rec, rng, "thinking", m, t, True)

    # A warm prefix needs at least one 128-token block; shorter LCPs are chat-template boilerplate.
    rows = [r for r in rec.rows if r["lcp"] and r["lcp"] >= 128 and r["cached"] is not None]
    print("\n=== summary (requests with a warm prefix available) ===")
    print(f"{'scenario':10} {'n':>3} {'sum lcp':>8} {'sum cached':>10} {'reuse':>6} "
          f"{'max short':>9} {'zero-hit':>8}")
    for sc in dict.fromkeys(r["scenario"] for r in rows):
        rs = [r for r in rows if r["scenario"] == sc]
        L = sum(r["lcp"] for r in rs)
        C = sum(min(r["cached"], r["lcp"]) for r in rs)
        print(f"{sc:10} {len(rs):>3} {L:>8} {C:>10} {C / L:>6.1%} "
              f"{max(r['short'] for r in rs):>9} {sum(r['cached'] == 0 for r in rs):>8}")
    if args.out:
        with open(args.out, "w") as fh:
            for r in rec.rows:
                fh.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
