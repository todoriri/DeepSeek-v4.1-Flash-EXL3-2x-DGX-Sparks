#!/usr/bin/env python3
"""Final sliding-window retention for the hybrid prefix cache, OFF by default.

With DSV41_SWA_FINAL_WINDOW=keep, a request that leaves the running set (finish, abort
or preemption) leaves a prefix-cache checkpoint at its end, not only at its prompt end
and every PREFIX_CACHE_RETENTION_INTERVAL tokens. The next agent turn, whose prompt holds
the previous output, then hits 64-192 tokens below the previous request's end instead of
prefilling that output again (repl PROGRESS 2026-10-19 #5). This is the vLLM form of
DeepInfer's decode-time SWA store (deepseek-ai/DeepInfer @ 774ce24, Apache-2.0,
doc/inference_deployment_qa.md "SWA KV cache"). DeepInfer copies the window into a
replica slot at every 128-token crossing and stores the newest copy at finish. vLLM's SWA
cache is paged, so nothing is copied. Instead:

  v1/core/single_type_kv_cache_manager.py
      SlidingWindowManager keeps `hold` extra tokens below the window allocated, the
      engine's own extra_retained_tokens mechanism, so the newest checkpoint's tail
      (window + DSpark peek block) is still allocated at the free. The per-request
      admission cap grows by the same blocks, so sum(reservations) <= pool still bounds
      the real-held blocks. Under DSV41_YOCO_PREFILL=compact the hold also covers the
      checkpoint KEEP + peek further back.
  v1/core/kv_cache_manager.py
      free() and pop_blocks_for_free() first hash those tails (only blocks of processed
      tokens, below any in-flight step). They then join the LRU tail like any cached
      block.

Exact: hit selection, hashing, retention and the EAGLE block drop are unchanged; a
checkpoint at the request end is the same kind of block the 4096 retention already
caches inside a decode. It does not touch the lines patch_yoco_prefill.py anchors on, and
either may run first.

start.sh runs this only when DSV41_SWA_FINAL_WINDOW != off, before the YOCO patch, and
aborts the container if it fails. When the variable is unset or off, the image's vLLM files
stay byte-identical. Every anchor must match exactly once. Otherwise nothing is written and
the exit status is 1. `--check` verifies the anchors only (Dockerfile). `--root` patches
another vllm tree (tests).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

MARK = "dsv41-swa-final-window"
MODES = ("off", "keep")

STM_FOOTER = '''

# [dsv41-swa-final-window]
def _dsv41_swa_final_require_enabled() -> None:
    import os

    mode = os.environ.get("DSV41_SWA_FINAL_WINDOW", "") or "off"
    if mode != "keep":
        raise RuntimeError(
            "[dsv41-swa-final-window] vLLM carries the final-window patch but "
            f"DSV41_SWA_FINAL_WINDOW={mode!r}; start a fresh container (start.sh only "
            "patches when keep)"
        )


def _dsv41_swa_final_keep() -> int:
    """Tokens the YOCO overlay recomputes at the end of every prefill, 0 when it is off.
    Its hits stop KEEP before the prompt end, so a second checkpoint is kept that far back."""
    import os

    if (os.environ.get("DSV41_YOCO_PREFILL", "") or "off") == "off":
        return 0
    raw = os.environ.get("DSV41_YOCO_KEEP_TOKENS", "") or "140"
    if not raw.isdigit() or int(raw) < 1:
        raise RuntimeError(f"[dsv41-swa-final-window] bad DSV41_YOCO_KEEP_TOKENS={raw!r}")
    return int(raw)


def _dsv41_swa_final_hold_tokens(window, block_size, alignment, peek_blocks, keep) -> int:
    """Tokens kept below the window so that, when the request is freed after C computed
    tokens (frees so far used <= C), the tail of each final checkpoint B is allocated.
    The tail is the blocks [B/bs + peek - need, B/bs + peek), need = ceil((W-1)/bs) + peek;
    the free keeps blocks from floor((C - W + 1 - e) / bs). Worst cases:
      B_end  = align(floor_bs(C) - peek*bs)         >= C - (A + peek*bs - 1)
      B_yoco = align(T - KEEP - peek*bs), T >= C    >= C - (KEEP + A + peek*bs - 1)"""
    lag = alignment + peek_blocks * block_size - 1 + keep
    span = -(-(window - 1) // block_size) * block_size
    return max(0, lag - (window - 1) + span)


def _dsv41_swa_final_boundaries(num_computed, num_tokens, block_size, alignment, use_eagle,
                                keep) -> list:
    """Hit lengths a freed request leaves: the newest checkpoint its computed blocks
    complete, and under YOCO the newest one any continuation (N >= T) may still use
    (a hit plus its peek block must stay <= N - KEEP). Only complete tails count."""
    peek = block_size if use_eagle else 0
    full = num_computed // block_size * block_size
    hits = {(full - peek) // alignment * alignment}
    if keep:
        hits.add((num_tokens - keep - peek) // alignment * alignment)
    return sorted(h for h in hits if h >= alignment and h + peek <= full)


def _dsv41_swa_final_cache(manager, request, num_computed: int) -> list:
    """Hash the final checkpoints' tails of one SWA group (see kv_cache_manager.free)."""
    blocks = manager.req_to_blocks.get(request.request_id)
    if not blocks or not manager.enable_caching:
        return []
    bs = manager.block_size
    num_computed = min(num_computed, len(blocks) * bs)
    need = manager._contiguous_blocks_for_hit(
        manager.sliding_window, bs, manager.use_eagle
    )
    shift = 1 if manager.use_eagle else 0
    hits, spans = [], []
    for hit in _dsv41_swa_final_boundaries(
        num_computed,
        request.num_tokens,
        bs,
        manager.cache_hit_alignment_tokens,
        manager.use_eagle,
        _dsv41_swa_final_keep(),
    ):
        end = hit // bs + shift
        start = max(0, end - need)  # a run from block 0 is a hit even when shorter
        # A block freed before the hold or a hit's null padding: no usable tail here.
        if any(blocks[i].is_null for i in range(start, end)):
            continue
        hits.append(hit)
        spans.append((start, end))
    if not spans:
        return []
    lo = min(s for s, _ in spans)
    hi = max(e for _, e in spans)
    mask = [False] * (hi - lo)
    for s, e in spans:
        for i in range(s, e):
            mask[i - lo] = blocks[i].block_hash is None
    if any(mask):
        manager.block_pool.cache_full_blocks(
            request=request,
            blocks=blocks,
            num_cached_blocks=lo,
            num_full_blocks=hi,
            block_size=bs,
            kv_cache_group_id=manager.kv_cache_group_id,
            block_mask=mask,
        )
    return hits


_dsv41_swa_final_require_enabled()
'''

KVM_FOOTER = '''

# [dsv41-swa-final-window]
def _dsv41_swa_final_window(kv_cache_manager, request) -> None:
    """Before a request's blocks go back to the pool (finish, abort, preemption), cache
    the sliding-window tail of its final checkpoint(s). Only processed tokens count: not
    a step still in flight, not tokens multi-module MTP may re-prefill."""
    if not kv_cache_manager.enable_caching:
        return
    from vllm.v1.core.single_type_kv_cache_manager import (
        SlidingWindowManager,
        _dsv41_swa_final_cache,
    )

    coordinator = kv_cache_manager.coordinator
    num_computed = (
        min(
            request.num_computed_tokens - request.num_in_flight_tokens,
            request.num_computed_tokens - getattr(request, "num_output_placeholders", 0),
            request.num_tokens,
        )
        - coordinator.num_reprefillable_tokens
    )
    if num_computed <= 0:
        return
    try:
        hits = set()
        for manager in coordinator.single_type_managers:
            if isinstance(manager, SlidingWindowManager):
                hits.update(_dsv41_swa_final_cache(manager, request, num_computed))
        if hits:
            logger.info_once(
                "[dsv41-swa-final-window] caching final sliding windows (first: %s, "
                "%d computed tokens, checkpoints %s)",
                request.request_id,
                num_computed,
                sorted(hits),
            )
    except Exception:
        # Never block a free: the request's blocks are released uncached, as stock.
        logger.exception(
            "[dsv41-swa-final-window] caching the final window of %s failed",
            request.request_id,
        )
'''

JOBS: list[tuple[str, list[tuple[str, str]], str | None]] = [
    (
        "v1/core/single_type_kv_cache_manager.py",
        [
            (
                "        # Extra trailing tokens to retain below the window (never attended) so a\n"
                "        # multi-module MTP store-side lag can still reconstruct the window from\n"
                "        # cached blocks.\n"
                "        self.extra_retained_tokens = kv_cache_spec.extra_retained_tokens\n",
                "        # Extra trailing tokens to retain below the window (never attended) so a\n"
                "        # multi-module MTP store-side lag can still reconstruct the window from\n"
                "        # cached blocks.\n"
                "        self.extra_retained_tokens = kv_cache_spec.extra_retained_tokens\n"
                "        # [dsv41-swa-final-window] keep the newest aligned window allocated until\n"
                "        # the request is freed (assume the DSpark peek block: use_eagle is set\n"
                "        # later), and admit with the same extra blocks.\n"
                "        _dsv41_hold = _dsv41_swa_final_hold_tokens(\n"
                "            self.sliding_window,\n"
                "            self.block_size,\n"
                "            self.scheduler_block_size,\n"
                "            1,\n"
                "            _dsv41_swa_final_keep(),\n"
                "        )\n"
                "        self.extra_retained_tokens += _dsv41_hold\n"
                "        if self._max_admission_blocks_per_request is not None:\n"
                "            self._max_admission_blocks_per_request += cdiv(_dsv41_hold, self.block_size)\n"
                "        logger.info_once(\n"
                "            \"[dsv41-swa-final-window] sliding-window groups keep %d tokens below \"\n"
                "            \"the window (+%d admission blocks per request)\",\n"
                "            _dsv41_hold,\n"
                "            cdiv(_dsv41_hold, self.block_size),\n"
                "        )\n",
            ),
        ],
        STM_FOOTER,
    ),
    (
        "v1/core/kv_cache_manager.py",
        [
            (
                "        self.coordinator.free(request.request_id)\n",
                "        _dsv41_swa_final_window(self, request)  # [dsv41-swa-final-window]\n"
                "        self.coordinator.free(request.request_id)\n",
            ),
            (
                "        return self.coordinator.pop_blocks_for_free(request.request_id)\n",
                "        _dsv41_swa_final_window(self, request)  # [dsv41-swa-final-window]\n"
                "        return self.coordinator.pop_blocks_for_free(request.request_id)\n",
            ),
        ],
        KVM_FOOTER,
    ),
]


def patch_text(text: str, edits: list[tuple[str, str]], footer: str | None) -> tuple[str, str]:
    if MARK in text:
        return text, "already applied"
    for old, new in edits:
        n = text.count(old)
        if n != 1:
            first = old.strip().splitlines()[0][:80]
            raise RuntimeError(f"anchor found {n}x (want 1): {first!r}")
        text = text.replace(old, new, 1)
    if footer:
        if not text.endswith("\n"):
            text += "\n"
        text += footer
    return text, "applied"


def _vllm_root() -> Path:
    import vllm

    return Path(vllm.__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", help="vllm package directory (default: the installed one)")
    ap.add_argument("--check", action="store_true", help="verify anchors, write nothing")
    args = ap.parse_args(argv)

    if not args.check:
        mode = os.environ.get("DSV41_SWA_FINAL_WINDOW", "") or "off"
        if mode == "off":
            print("SWA final-window patch: skipped (DSV41_SWA_FINAL_WINDOW=off)")
            return 0
        if mode not in MODES:
            print(f"SWA final-window patch: DSV41_SWA_FINAL_WINDOW must be off or keep (got {mode!r})",
                  file=sys.stderr)
            return 2

    root = Path(args.root) if args.root else _vllm_root()
    out: list[tuple[Path, str, str]] = []
    errors: list[str] = []
    for rel, edits, footer in JOBS:
        path = root / rel
        if not path.is_file():
            errors.append(f"{rel}: missing")
            continue
        try:
            text, status = patch_text(path.read_text(), edits, footer)
        except RuntimeError as exc:
            errors.append(f"{rel}: {exc}")
            continue
        out.append((path, text, status))
    if errors:
        for err in errors:
            print(f"SWA final-window patch: {err}", file=sys.stderr)
        print("SWA final-window patch: FAILED (anchor drift), nothing written", file=sys.stderr)
        return 1
    if args.check:
        states = sorted({status for _, _, status in out})
        print(f"SWA final-window patch: check ok ({len(out)} files, {', '.join(states)})")
        return 0
    for path, text, status in out:
        if status == "applied":
            path.write_text(text)
        print(f"SWA final-window patch: {status} ({path.relative_to(root)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
