#!/usr/bin/env python3
"""Drive the gb10 teacher with concurrent long-context requests — wedge repro.

Reproduces the trigger documented in docs/exl3-moe-barrier.md: the wedge fires
under concurrency with a large prefill (MiaAI-Lab #22: two concurrent requests;
GLM-5.3 #128: large-prefill + KV-release churn). Every request carries a unique
nonce so no two share a prefix-cache hit, and streaming is used because the
wedge signature is *silence*: the front end accepts the POST, then the engine
stops advancing and no SSE content ever arrives.

The 180 s watchdog stall threshold means a wedge is confirmed ~3 min after the
engine freezes; this script only reports what it can see from the client side
(no bytes for N seconds = stall suspected).

Usage:
    python3 scripts/repro_wedge.py --prompt-tokens 64000 --requests 2
    python3 scripts/repro_wedge.py --prompt-tokens 300000 --requests 2 --max-tokens 512

Exit: 0 all requests completed, 2 at least one stalled, 1 transport/setup error.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request
import uuid

FILLER = "the quick brown fox jumps over the lazy dog. "


def build_prompt(tokens: int, nonce: str) -> str:
    """Filler sized by word count toward ``tokens`` (~1 token per word).

    The nonce is what keeps two requests from sharing a prefix-cache entry, so
    both actually prefill instead of one riding the other's cache.
    """
    words = max(1, tokens)
    reps = max(1, words // 9)
    return (
        f"[repro {nonce}] Ignore everything below and reply with exactly: OK.\n"
        + FILLER * reps
    )


def one_request(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    read_timeout: float,
    stall_after: float,
    label: str,
    start_barrier: threading.Barrier,
    results: dict,
) -> None:
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    started = time.time()
    first_content: float | None = None
    chunks = 0
    usage: dict = {}
    stalled = False
    error = ""
    start_barrier.wait()  # all requests hit the engine at the same moment

    try:
        with urllib.request.urlopen(req, timeout=read_timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    evt = json.loads(body)
                except ValueError:
                    continue
                if evt.get("usage"):
                    usage = evt["usage"]
                for choice in evt.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        if first_content is None:
                            first_content = time.time()
                        chunks += 1
                now = time.time()
                progress_at = first_content or started
                if now - progress_at > stall_after:
                    stalled = True
                    break
    except Exception as exc:  # noqa: BLE001 - timeout/reset all mean "no progress"
        error = f"{type(exc).__name__}: {exc}"

    elapsed = time.time() - started
    results[label] = {
        "label": label,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "sse_chunks": chunks,
        "first_content_s": round(first_content - started, 2) if first_content else None,
        "elapsed_s": round(elapsed, 1),
        "stalled": stalled,
        "error": error,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="http://192.168.1.97:8000")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--requests", type=int, default=2)
    ap.add_argument("--prompt-tokens", type=int, default=64000)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--read-timeout", type=float, default=180.0)
    ap.add_argument("--stall-after", type=float, default=150.0)
    args = ap.parse_args()

    nonce = uuid.uuid4().hex[:8]
    prompts = {
        f"r{i}": build_prompt(args.prompt_tokens, f"{nonce}-{i}")
        for i in range(args.requests)
    }
    print(
        json.dumps(
            {
                "event": "repro_start",
                "nonce": nonce,
                "requests": args.requests,
                "prompt_tokens_target": args.prompt_tokens,
                "max_tokens": args.max_tokens,
                "stall_after_s": args.stall_after,
            }
        ),
        flush=True,
    )

    barrier = threading.Barrier(args.requests)
    results: dict = {}
    threads = [
        threading.Thread(
            target=one_request,
            args=(
                args.base_url,
                args.model,
                prompts[label],
                args.max_tokens,
                args.read_timeout,
                args.stall_after,
                label,
                barrier,
                results,
            ),
            daemon=True,
        )
        for label in prompts
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=args.read_timeout + 60)

    for label in sorted(results):
        print(json.dumps({"event": "repro_result", **results[label]}), flush=True)

    stalled = [r for r in results.values() if r["stalled"] or r["error"]]
    if stalled:
        print(
            json.dumps(
                {
                    "event": "repro_stalled",
                    "count": len(stalled),
                    "note": "engine silent; check vllm-watchdog-data for stall_confirmed + head-gpu.txt",
                }
            ),
            flush=True,
        )
        return 2
    if len(results) != args.requests:
        return 1
    print(json.dumps({"event": "repro_clean", "count": len(results)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
