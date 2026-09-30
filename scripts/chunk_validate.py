#!/usr/bin/env python3
"""Post-restart checks for a prefill-chunk change (MAX_NUM_BATCHED_TOKENS /
LONG_PREFILL_TOKEN_THRESHOLD). Run from rai against the live head.

  kvprog     one cold prompt, /metrics polled every second: prefill tok/s, and
             whether vllm:kv_cache_usage_perc moves while prompt_tokens_total
             stays flat (the watchdog's prefill-aware progress signal).
  fairness   a long cold prefill, then a short chat sent into it: the short
             request's latency (head-of-line blocking) and the long one's tok/s.
  retention  fresh-salt prompts of given lengths sent twice: cached tokens on
             the repeat (prefix-cache retention still works at the new chunk).

Prompts are random lowercase words behind a unique salt, so nothing hits the
prefix cache; thinking off, temperature 0, max_tokens 1 unless noted.
"""
from __future__ import annotations

import argparse
import json
import random
import string
import sys
import threading
import time
import urllib.error
import urllib.request


def post(base: str, path: str, body: dict, timeout: float = 3600) -> dict:
    req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as f:
            return json.load(f)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{path} HTTP {e.code}: {e.read().decode()[:300]}") from None


def metrics(base: str) -> dict[str, float]:
    want = ("vllm:prompt_tokens_total", "vllm:generation_tokens_total",
            "vllm:kv_cache_usage_perc", "vllm:num_requests_running",
            "vllm:num_requests_waiting")
    out = {k: 0.0 for k in want}
    with urllib.request.urlopen(base + "/metrics", timeout=5) as f:
        for line in f.read().decode().splitlines():
            if line.startswith("#") or " " not in line:
                continue
            ident, val = line.rsplit(" ", 1)
            name = ident.split("{", 1)[0]
            if name in out:
                out[name] += float(val)
    return out


def words(rng: random.Random, n: int) -> str:
    return " ".join("".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 9)))
                    for _ in range(n))


def build_messages(base: str, model: str, target: int, seed: int) -> tuple[list, int]:
    """Messages whose rendered prompt is ~target tokens (one /tokenize correction)."""
    salt = f"[run {seed} {time.time_ns()}] "
    n = max(16, int(target / 2.2))
    for _ in range(3):
        text = salt + words(random.Random(seed), n)
        msgs = [{"role": "user", "content": text + "\nReply with the single word OK."}]
        got = len(post(base, "/tokenize", {"model": model, "messages": msgs,
                                           "chat_template_kwargs": {"enable_thinking": False}})["tokens"])
        if abs(got - target) <= max(64, target // 200):
            return msgs, got
        n = max(16, int(n * target / got))
    return msgs, got


def chat(base: str, model: str, msgs: list, max_tokens: int = 1) -> tuple[dict, float]:
    t0 = time.time()
    r = post(base, "/v1/chat/completions", {
        "model": model, "messages": msgs, "temperature": 0.0, "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False}})
    return r, time.time() - t0


class Poller(threading.Thread):
    def __init__(self, base: str, period: float = 1.0):
        super().__init__(daemon=True)
        self.base, self.period, self.samples, self.stop = base, period, [], threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                self.samples.append((time.time(), metrics(self.base)))
            except Exception:  # noqa: BLE001 - keep sampling
                pass
            self.stop.wait(self.period)


def cmd_kvprog(a) -> dict:
    msgs, n = build_messages(a.base, a.model, a.tokens, a.seed)
    p = Poller(a.base); p.start()
    time.sleep(2)
    t_send = time.time()
    r, dt = chat(a.base, a.model, msgs)
    time.sleep(2); p.stop.set(); p.join()
    during = [m for t, m in p.samples if t_send + 1 < t < t_send + dt - 1]
    kv = [m["vllm:kv_cache_usage_perc"] for m in during]
    pt = {m["vllm:prompt_tokens_total"] for m in during}
    out = {"mode": "kvprog", "prompt_tokens": r["usage"]["prompt_tokens"], "ttft_s": round(dt, 2),
           "prefill_tok_s": round(r["usage"]["prompt_tokens"] / dt, 1),
           "samples_during": len(during), "kv_distinct": len(set(kv)),
           "kv_first_last": [kv[0], kv[-1]] if kv else None,
           "kv_monotonic": all(b >= a_ for a_, b in zip(kv, kv[1:])),
           "prompt_tokens_total_distinct_during": len(pt)}
    out["verdict"] = ("kv moves while counters flat" if out["kv_distinct"] > 1 and len(pt) == 1
                      else "CHECK")
    return out


def cmd_fairness(a) -> dict:
    msgs, n = build_messages(a.base, a.model, a.tokens, a.seed)
    res = {}

    def long_req():
        r, dt = chat(a.base, a.model, msgs)
        res["long"] = {"prompt_tokens": r["usage"]["prompt_tokens"], "ttft_s": round(dt, 2),
                       "prefill_tok_s": round(r["usage"]["prompt_tokens"] / dt, 1)}

    th = threading.Thread(target=long_req); th.start()
    time.sleep(a.delay)
    short = [{"role": "user", "content": f"[{time.time_ns()}] What is 17*19? Answer with the number."}]
    r, dt = chat(a.base, a.model, short, max_tokens=16)
    res["short"] = {"latency_s": round(dt, 2), "sent_after_s": a.delay,
                    "answer": r["choices"][0]["message"]["content"]}
    th.join()
    res["mode"] = "fairness"
    return res


def cmd_retention(a) -> dict:
    rows = []
    for i, L in enumerate(a.lengths):
        msgs, n = build_messages(a.base, a.model, L, a.seed + i)
        r1, dt1 = chat(a.base, a.model, msgs)
        r2, dt2 = chat(a.base, a.model, msgs)
        c = (r2["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
        rows.append({"prompt_tokens": r2["usage"]["prompt_tokens"], "cached_on_repeat": c,
                     "cold_s": round(dt1, 2), "warm_s": round(dt2, 2)})
    return {"mode": "retention", "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["kvprog", "fairness", "retention"])
    ap.add_argument("--base", default="http://192.168.1.97:8000")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--tokens", type=int, default=48000)
    ap.add_argument("--delay", type=float, default=30.0, help="fairness: short request after N s")
    ap.add_argument("--lengths", type=int, nargs="+", default=[8193, 34357])
    ap.add_argument("--seed", type=int, default=20260930)
    a = ap.parse_args()
    out = {"kvprog": cmd_kvprog, "fairness": cmd_fairness, "retention": cmd_retention}[a.mode](a)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
