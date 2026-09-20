#!/usr/bin/env python3
"""Disturb the teacher's KV cache the way a second chat session does.

The 23:13:55Z onset did not come from a synthetic wave: it came from a long
decode over a resident ~300k prefix, with the second request arriving *after*
onset. What a second session actually contributes is (a) a disjoint prefix, so
its prefill is real work beside the first session's decode, (b) human-paced
arrival offsets, and (c) request boundaries that release KV blocks and prefill
workspaces -- the ``running N -> N-1`` churn that both #128 and our own onset
notes align with.

This models that, deliberately and repeatably:

  decoder arm   repeat {unique ~4k prompt, up to 8k output, streamed to
                completion}: every completion is a KV release, every new request
                a fresh prefill beside the other arm's work.
  prefill arm   per cycle, one large unique prompt (jittered) that is ABORTED
                mid-flight after a random read window: a client disconnect makes
                the engine abort and free blocks/workspaces.
  jitter        sizes and offsets are randomised, so contention is not identical
                every cycle (a barrier-synchronised pair contends the same way
                each time -- that is why the earlier repro produced false
                positives instead).

Guards, so a stall is never confused with queueing or OOM:
  * waits while ``num_requests_waiting_by_reason{reason="capacity"}`` > 0
  * backs off when KV cache usage is high (either cache metric name)
  * stops below ``--mem-guard-gb`` MemAvailable on the head (memguard is
    disabled in the kit, so this script is the guard)
  * stops firing new cycles once the decoder arm has been silent for
    ``--stall-after``, so a wedged engine does not accumulate queued requests

One JSON object per action on stdout. Exit 2 if a client-side stall was seen.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

FILLER = "the quick brown fox jumps over the lazy dog. "


def prompt_for(tokens: int, nonce: str, instruction: str = "Reply with exactly: OK.") -> str:
    """``tokens`` filler words behind an instruction.

    The decoder arm must *not* use the short "reply OK" instruction: a healthy
    model answers that in 2-3 tokens, which is churn but not a long decode. It
    gets a task that runs to ``max_tokens`` instead.
    """
    reps = max(1, tokens // 9)
    return f"[disturb {nonce}] {instruction}\n" + FILLER * reps


DECODER_TASK = (
    "Count upward from 1, one number per line, and keep going until you run out "
    "of room. Do not comment, do not summarise."
)


def stream_request(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    read_window: float | None,
    timeout: float,
    on_progress=None,
) -> dict:
    """Stream one request.

    ``read_window=None`` streams to completion (the decoder arm);
    a number aborts mid-stream after that many seconds (the prefill arm), which
    closes the socket -- vLLM notices the disconnect and aborts the request.
    """
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
    aborted = False
    error = ""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                if read_window is not None and time.time() - started > read_window:
                    aborted = True
                    break
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
                        if on_progress:
                            on_progress(time.time())
    except Exception as exc:  # noqa: BLE001 - timeout/reset both mean no progress
        error = f"{type(exc).__name__}: {exc}"
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "sse_chunks": chunks,
        "first_content_s": round(first_content - started, 2) if first_content else None,
        "elapsed_s": round(time.time() - started, 1),
        "aborted": aborted,
        "error": error,
    }


def read_metrics(base_url: str, timeout: float = 6.0) -> dict:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/metrics", timeout=timeout) as resp:
            text = resp.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - metrics are a guard, not a dependency
        return {}
    out: dict = {}
    for line in text.splitlines():
        if line.startswith("vllm:num_requests_running"):
            out["running"] = float(line.rsplit(" ", 1)[-1])
        elif line.startswith("vllm:num_requests_waiting_by_reason") and 'reason="capacity"' in line:
            out["waiting_capacity"] = float(line.rsplit(" ", 1)[-1])
        elif "cache_usage_perc" in line and line.startswith("vllm:"):
            out["kv_usage_perc"] = float(line.rsplit(" ", 1)[-1])
    return out


def mem_available_gb(host: str) -> float | None:
    try:
        proc = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=8",
                host,
                "awk '/MemAvailable/{print $2}' /proc/meminfo",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return int(proc.stdout.strip() or 0) / 1048576
    except Exception:  # noqa: BLE001 - best effort
        return None


def emit(**fields) -> None:
    print(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields}), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="http://192.168.1.97:8000")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--head-host", default="192.168.1.97", help="ssh target for the MemAvailable guard")
    ap.add_argument("--cycles", type=int, default=12)
    ap.add_argument("--decoder-prompt-tokens", type=int, default=4000)
    ap.add_argument("--decoder-max-tokens", type=int, default=8000)
    ap.add_argument("--prefill-min-tokens", type=int, default=60000)
    ap.add_argument("--prefill-max-tokens", type=int, default=180000)
    ap.add_argument("--prefill-max-tokens-cap", type=int, default=512)
    ap.add_argument("--abort-min-s", type=float, default=5.0)
    ap.add_argument("--abort-max-s", type=float, default=25.0)
    ap.add_argument("--inter-cycle-min-s", type=float, default=30.0)
    ap.add_argument("--inter-cycle-max-s", type=float, default=90.0)
    ap.add_argument("--stall-after-s", type=float, default=240.0)
    ap.add_argument("--mem-guard-gb", type=float, default=1.2)
    ap.add_argument("--max-minutes", type=float, default=30.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        emit(
            event="dry_run",
            cycles=args.cycles,
            prefill_range=[args.prefill_min_tokens, args.prefill_max_tokens],
            abort_range=[args.abort_min_s, args.abort_max_s],
            inter_cycle_range=[args.inter_cycle_min_s, args.inter_cycle_max_s],
            note="no requests fired",
        )
        return 0

    stop = threading.Event()
    state = {"last_progress": time.time(), "decoder_streams": 0}
    deadline = time.time() + args.max_minutes * 60
    nonce = uuid.uuid4().hex[:8]

    def decoder_arm() -> None:
        while not stop.is_set() and time.time() < deadline:
            result = stream_request(
                args.base_url,
                args.model,
                prompt_for(args.decoder_prompt_tokens, f"{nonce}-d{state['decoder_streams']}", DECODER_TASK),
                args.decoder_max_tokens,
                None,
                1800.0,
                on_progress=lambda t: state.__setitem__("last_progress", t),
            )
            state["decoder_streams"] += 1
            emit(event="decoder_cycle", n=state["decoder_streams"], **result)
            if result["error"]:
                time.sleep(10)

    decoder = threading.Thread(target=decoder_arm, daemon=True)
    decoder.start()
    emit(event="disturb_start", nonce=nonce, cycles=args.cycles, max_minutes=args.max_minutes)

    stalled = False
    for cycle in range(1, args.cycles + 1):
        if time.time() >= deadline:
            emit(event="deadline_reached", cycle=cycle)
            break

        silent_for = time.time() - state["last_progress"]
        if silent_for > args.stall_after_s:
            emit(event="client_stall_suspected", cycle=cycle, silent_for_s=round(silent_for, 1))
            stalled = True
            break  # stop firing: do not pile requests onto a wedged engine

        metrics = read_metrics(args.base_url)
        if metrics.get("waiting_capacity", 0.0) > 0:
            emit(event="skip_capacity_backoff", cycle=cycle, metrics=metrics)
            time.sleep(60)
            continue
        if metrics.get("kv_usage_perc", 0.0) > 0.9:
            emit(event="skip_kv_backoff", cycle=cycle, metrics=metrics)
            time.sleep(60)
            continue

        avail = mem_available_gb(args.head_host)
        if avail is not None and avail < args.mem_guard_gb:
            emit(event="mem_guard_stop", cycle=cycle, mem_available_gb=round(avail, 2))
            break

        size = random.randint(args.prefill_min_tokens, args.prefill_max_tokens)
        window = random.uniform(args.abort_min_s, args.abort_max_s)
        emit(event="prefill_abort_fire", cycle=cycle, prompt_tokens_target=size, abort_after_s=round(window, 1), metrics=metrics, mem_available_gb=avail)
        result = stream_request(
            args.base_url,
            args.model,
            prompt_for(size, f"{nonce}-p{cycle}"),
            args.prefill_max_tokens_cap,
            window,
            900.0,
        )
        emit(event="prefill_abort_done", cycle=cycle, **result)
        time.sleep(random.uniform(args.inter_cycle_min_s, args.inter_cycle_max_s))

    stop.set()
    decoder.join(timeout=30)
    emit(event="disturb_stop", stalled=stalled, decoder_streams=state["decoder_streams"])
    return 2 if stalled else 0


if __name__ == "__main__":
    sys.exit(main())
