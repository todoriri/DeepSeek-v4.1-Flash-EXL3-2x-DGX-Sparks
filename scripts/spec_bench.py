#!/usr/bin/env python3
"""Before/after bench for a speculative-decoding change (e.g. DSpark fixed k=3 vs
k=5 + enable_adaptive_verification).

Fixed long-output prompts (code, prose, math, structured), official sampling
(temperature 1.0, top_p 0.95), thinking on at effort 50 so the decode mixes
reasoning and content like live traffic. Each phase (c=1, then c=2) reports:

  * per request: TTFT and decode tok/s = (completion - 1) / (end - first token),
    streamed so prefill is excluded;
  * the phase: aggregate tok/s (all completion tokens / wall time), and the
    engine's own acceptance from /metrics deltas (vllm:spec_decode_*): accepted
    per draft token, mean acceptance length (1 + accepted/drafts), per position.

The /metrics deltas are only this bench's if nothing else runs on the engine:
the teacher must be otherwise idle.

    python3 scripts/spec_bench.py --label k3-fixed --out bench-k3.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import statistics
import sys
import time
import urllib.request

PROMPTS = [
    "Write a complete Python module implementing an LRU cache with TTL expiry, "
    "thread safety, and statistics, with docstrings and type hints. Then explain the design.",
    "Write a Rust function that parses an INI file into nested maps, with error handling "
    "and unit tests. Explain each part.",
    "Explain in detail how a modern CPU's out-of-order execution pipeline works, from fetch "
    "to retirement, including register renaming and the reorder buffer.",
    "Write a long, careful essay on the trade-offs between monoliths and microservices for a "
    "twenty-person engineering team.",
    "Solve step by step: how many lattice paths from (0,0) to (12,12) never rise above the "
    "diagonal? Then generalise, derive the formula, and verify it for small n.",
    "Derive the closed form of the sum of k^3 for k=1..n by three different methods, "
    "showing every step.",
    "Produce a JSON array of 40 fictional books, each with title, author, year, genre, a "
    "one-sentence summary and a list of three themes.",
    "Translate the following into French, German and Spanish, sentence by sentence, then "
    "comment on idioms: 'The committee postponed the decision until the budget was final, "
    "which frustrated everyone who had prepared for weeks.' Extend with ten more example "
    "sentences of your own and translate those too.",
]


def metrics(base: str) -> dict:
    with urllib.request.urlopen(base + "/metrics", timeout=30) as f:
        text = f.read().decode()
    out = {"pos": {}}
    for line in text.splitlines():
        if line.startswith("#") or not line.startswith("vllm:spec_decode"):
            continue
        name, val = line.rsplit(" ", 1)
        v = float(val)
        if name.startswith("vllm:spec_decode_num_drafts_total"):
            out["drafts"] = out.get("drafts", 0) + v
        elif name.startswith("vllm:spec_decode_num_draft_tokens_total"):
            out["draft_tokens"] = out.get("draft_tokens", 0) + v
        elif name.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            out["accepted"] = out.get("accepted", 0) + v
        elif name.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total"):
            p = int(re.search(r'position="(\d+)"', name).group(1))
            out["pos"][p] = out["pos"].get(p, 0) + v
    return out


def delta(a: dict, b: dict) -> dict:
    d = {k: b.get(k, 0) - a.get(k, 0) for k in ("drafts", "draft_tokens", "accepted")}
    d["pos"] = {p: b["pos"].get(p, 0) - a["pos"].get(p, 0) for p in sorted(b["pos"])}
    if d["drafts"] > 0:
        d["accept_per_draft_token"] = round(d["accepted"] / d["draft_tokens"], 4)
        d["mean_accept_len"] = round(1 + d["accepted"] / d["drafts"], 3)
        d["pos_rate"] = {p: round(v / d["drafts"], 4) for p, v in d["pos"].items()}
    return d


def one(base: str, model: str, prompt: str, max_tokens: int) -> dict:
    body = {"model": model, "stream": True, "max_tokens": max_tokens,
            "temperature": 1.0, "top_p": 0.95,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": True, "thinking": True,
                                     "reasoning_effort": 50}}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    usage = {}
    with urllib.request.urlopen(req, timeout=3600) as f:
        for raw in f:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if first is None and (d.get("content") or d.get("reasoning_content")
                                      or d.get("reasoning")):
                    first = time.time()
    end = time.time()
    comp = usage.get("completion_tokens", 0)
    dec = (comp - 1) / (end - first) if first and end > first and comp > 1 else None
    return {"ttft": round((first or end) - t0, 2), "elapsed": round(end - t0, 2),
            "completion_tokens": comp, "decode_tok_s": round(dec, 2) if dec else None}


def phase(base, model, conc, max_tokens):
    m0 = metrics(base)
    t0 = time.time()
    with cf.ThreadPoolExecutor(conc) as ex:
        rows = list(ex.map(lambda p: one(base, model, p, max_tokens), PROMPTS))
    wall = time.time() - t0
    m1 = metrics(base)
    decs = [r["decode_tok_s"] for r in rows if r["decode_tok_s"]]
    return {"concurrency": conc, "wall_s": round(wall, 1),
            "aggregate_tok_s": round(sum(r["completion_tokens"] for r in rows) / wall, 2),
            "decode_tok_s_median": round(statistics.median(decs), 2) if decs else None,
            "decode_tok_s_mean": round(statistics.mean(decs), 2) if decs else None,
            "ttft_median": round(statistics.median(r["ttft"] for r in rows), 2),
            "spec": delta(m0, m1), "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://192.168.1.97:8000")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--label", required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--concurrency", default="1,2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"label": a.label, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "max_tokens": a.max_tokens, "phases": []}
    for c in [int(x) for x in a.concurrency.split(",")]:
        ph = phase(a.base, a.model, c, a.max_tokens)
        res["phases"].append(ph)
        s = ph["spec"]
        print(f"[{a.label}] c={c}: aggregate {ph['aggregate_tok_s']} tok/s, per-request decode "
              f"median {ph['decode_tok_s_median']}, TTFT median {ph['ttft_median']} s, "
              f"accept/draft-token {s.get('accept_per_draft_token')}, mean accept len "
              f"{s.get('mean_accept_len')}, per pos {s.get('pos_rate')}", flush=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
