#!/usr/bin/env python3
"""Per-turn own-output re-prefill from the repl request record (read-only).

Measures what DSV41_SWA_FINAL_WINDOW=keep is meant to remove (repl PROGRESS 2026-10-19 #5)
and checks it after a deploy. A session's rows are chained the way repl's prefix-loss view
chains them: same session_id and upstream, caller traffic only, started_at order. For a
continuation (row n -> n+1), with P/O = row n's prompt/output tokens, T = P + O, N = row
n+1's prompt and H = its cached tokens:

  growth          N - T. When it is >= 0, the next prompt holds the whole previous output,
                  which repl verified for these clients (LCP(render n, render n+1) = len(render n)).
  reprefilled own the previous output tokens prefilled again: T - max(H, P), clipped to
                  [0, O]. Counted only when growth >= 0 and H is known.
  hit vs end      H - T (0 = resumed at the previous end; stock: about -(O + 59)).
  hit vs prompt   H - P (stock: -127..0 at the prompt tail, or a 4096 multiple inside O).
  predicted       the hit under keep, from lengths alone, assuming a normal finish
                  (C = T - 1 computed tokens):
                    off      B_end = align128(floor64(C) - 64), when B_end + 64 <= N - 1;
                    compact  the larger of B_end and align128(T - KEEP - 64), whichever
                             stays <= N - KEEP - 64 (YOCO's cap). This is a lower bound:
                             a 4096 checkpoint can sit above B_yoco.
                  The predicted value is max(observed, that). It does not model eviction.

Also printed per window: decode tok/s (output / (total - ttft), rows with >= 64 output
tokens and >= 1 s of decode) and prefill tok/s ((prompt - cached) / ttft, rows with >= 2048 uncached tokens).
Use them to check that decode is unchanged. The record holds no memory figure: head
MemAvailable comes from the head log / free -m (docs/swa-final-window-retention.md).

  swa_final_window_turns.py [--db PATH] [--from WHEN] [--to WHEN] [--session SID]...
                            [--upstream URL] [--path P] [--keep 140] [--per-turn] [--json]

WHEN = ISO-8601 (UTC, as the record stores it) or now / -15m / -2h / -7d (write a relative
one as --from=-7d). --db defaults
to repl's record (REQUEST_RECORD_DB, else LOG_DIR/requests.sqlite). The DB
is opened mode=ro. To join a turn to a dsh child session, match dsh's responseId to the
printed upstream_request_id. Exit: 0 ok, 2 no record / bad arguments.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import sys
from typing import Any, Iterable

BS, ALIGN, PEEK = 64, 128, 64
INTERNAL_ORIGINS = {"keepalive", "probe", "replay"}          # repl record_store
CACHED_NATIVE_SINCE = "2026-09-23T15:34:00.000Z"             # repl record_store.cached_known
COLUMNS = ("id", "started_at", "session_id", "upstream", "call_origin", "status_code", "path",
           "prompt_tokens", "output_tokens", "cached_tokens", "cached_source", "ttft_ms",
           "total_ms", "upstream_request_id")


def cached_known(row: dict[str, Any]) -> int | None:
    """repl's rule (record_store.cached_known): trust a backend/metrics source, a positive
    pre-v9 count, or a zero recorded after the backend started reporting zeros."""
    ct = int(row.get("cached_tokens") or 0)
    src = row.get("cached_source")
    if src:
        return ct if src in ("backend", "metrics") else None
    if ct > 0:
        return ct
    return ct if str(row.get("started_at") or "") >= CACHED_NATIVE_SINCE else None


def b_end(computed: int) -> int:
    return (computed // BS * BS - PEEK) // ALIGN * ALIGN


def b_yoco(num_tokens: int, keep: int) -> int:
    return (num_tokens - keep - PEEK) // ALIGN * ALIGN


def predicted_hit(P: int, O: int, N: int, observed: int, keep: int) -> int:
    """The hit with the final-window checkpoint, keep = 0 for YOCO off."""
    T = P + O
    cands = [b_end(T - 1)] + ([b_yoco(T, keep)] if keep else [])
    cap = N - keep if keep else N - 1
    usable = [b for b in cands if b >= ALIGN and b + PEEK <= cap and b + PEEK <= (T - 1) // BS * BS]
    return max([observed] + usable)


def reprefilled_own(P: int, O: int, hit: int) -> int:
    return max(0, min(O, P + O - max(hit, P)))


def chains(rows: Iterable[dict[str, Any]], path: str) -> dict[tuple, list[dict[str, Any]]]:
    out: dict[tuple, list[dict[str, Any]]] = {}
    for r in sorted(rows, key=lambda x: str(x.get("started_at") or "")):
        if (not r.get("session_id") or int(r.get("prompt_tokens") or 0) <= 0
                or str(r.get("call_origin") or "") in INTERNAL_ORIGINS
                or int(r.get("status_code") or 0) != 200 or (path and r.get("path") != path)):
            continue
        out.setdefault((r["session_id"], str(r.get("upstream") or "")), []).append(r)
    return out


def turns(rows: Iterable[dict[str, Any]], path: str, keep: int) -> list[dict[str, Any]]:
    out = []
    for (sid, upstream), chain in chains(rows, path).items():
        for prev, cur in zip(chain, chain[1:]):
            P, O = int(prev["prompt_tokens"]), int(prev["output_tokens"] or 0)
            N, H = int(cur["prompt_tokens"]), cached_known(cur)
            t = {"session_id": sid, "upstream": upstream, "started_at": cur["started_at"],
                 "upstream_request_id": cur.get("upstream_request_id"), "prompt": P,
                 "output": O, "next_prompt": N, "growth": N - P - O, "hit": H}
            if H is not None and N >= P + O:
                t.update(hit_vs_end=H - P - O, hit_vs_prompt=H - P,
                         reprefilled_own=reprefilled_own(P, O, H),
                         predicted_off=reprefilled_own(P, O, predicted_hit(P, O, N, H, 0)),
                         predicted_compact=reprefilled_own(P, O, predicted_hit(P, O, N, H, keep)))
            out.append(t)
    return out


def _dist(values: list[int]) -> dict[str, Any] | None:
    if not values:
        return None
    return {"n": len(values), "min": min(values), "median": statistics.median(values),
            "max": max(values)}


def summarize(rows: list[dict[str, Any]], path: str, keep: int) -> dict[str, Any]:
    ts = turns(rows, path, keep)
    full = [t for t in ts if "reprefilled_own" in t]
    decode = [int(r["output_tokens"]) / ((int(r["total_ms"]) - int(r["ttft_ms"])) / 1000)
              for r in rows if int(r.get("output_tokens") or 0) >= 64
              and int(r.get("total_ms") or 0) - int(r.get("ttft_ms") or 0) >= 1000
              and int(r.get("ttft_ms") or 0) > 0
              and str(r.get("call_origin") or "") not in INTERNAL_ORIGINS]
    prefill = []
    for r in rows:
        c = cached_known(r)
        unc = int(r.get("prompt_tokens") or 0) - (c or 0)
        if c is not None and unc >= 2048 and int(r.get("ttft_ms") or 0) > 0:
            prefill.append(unc / (int(r["ttft_ms"]) / 1000))
    return {
        "continuations": len(ts),
        "with_full_previous_output": len(full),
        "rewritten_history": sum(1 for t in ts if t["growth"] < 0),
        "unknown_cached": sum(1 for t in ts if t["hit"] is None),
        "own_output_tokens": sum(t["output"] for t in full),
        "reprefilled_own_tokens": sum(t["reprefilled_own"] for t in full),
        "predicted_keep_off_tokens": sum(t["predicted_off"] for t in full),
        "predicted_keep_compact_tokens": sum(t["predicted_compact"] for t in full),
        "hit_vs_end": _dist([t["hit_vs_end"] for t in full]),
        "hit_vs_prompt": _dist([t["hit_vs_prompt"] for t in full]),
        "hit_at_prompt_tail": sum(1 for t in full if -127 <= t["hit_vs_prompt"] <= 0),
        "hit_inside_output": sum(1 for t in full if 0 < t["hit_vs_prompt"] <= t["output"]),
        "decode_tok_s": _dist([round(x, 1) for x in decode]),
        "prefill_tok_s": _dist([round(x, 1) for x in prefill]),
        "keep": keep,
        "turns": ts,
    }


def when(text: str | None) -> str | None:
    """ISO-8601 as given, or now / -15m / -2h / -7d (repl window.py's relative forms), as
    the record's UTC `...Z` strings."""
    if text is None:
        return None
    import datetime as dt
    import re

    m = re.fullmatch(r"-(\d+)([mhd])", text)
    if text != "now" and not m:
        return text
    delta = dt.timedelta() if text == "now" else dt.timedelta(
        **{{"m": "minutes", "h": "hours", "d": "days"}[m.group(2)]: int(m.group(1))})
    t = dt.datetime.now(dt.timezone.utc) - delta
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"


def default_db() -> str:
    if os.environ.get("REQUEST_RECORD_DB"):
        return os.environ["REQUEST_RECORD_DB"]
    if os.environ.get("LOG_DIR"):
        return os.path.join(os.environ["LOG_DIR"], "requests.sqlite")
    return "/mnt/data0/logs/proxy/requests.sqlite"


def load_rows(db: str, since: str | None, until: str | None, sessions: list[str],
              upstream: str | None) -> list[dict[str, Any]]:
    if not os.path.exists(db):
        raise FileNotFoundError(db)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    have = {r[1] for r in con.execute("PRAGMA table_info(requests)")}
    cols = ", ".join(c if c in have else f"NULL AS {c}" for c in COLUMNS)
    sql, args = f"SELECT {cols} FROM requests WHERE 1=1", []
    if since:
        sql += " AND started_at >= ?"
        args.append(since)
    if until:
        sql += " AND started_at < ?"
        args.append(until)
    if sessions:
        sql += f" AND session_id IN ({', '.join('?' * len(sessions))})"
        args += sessions
    if upstream:
        sql += " AND upstream = ?"
        args.append(upstream)
    rows = [dict(r) for r in con.execute(sql + " ORDER BY started_at", args)]
    con.close()
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=default_db())
    ap.add_argument("--from", dest="since")
    ap.add_argument("--to", dest="until")
    ap.add_argument("--session", action="append", default=[])
    ap.add_argument("--upstream")
    ap.add_argument("--path", default="/v1/chat/completions")
    ap.add_argument("--keep", type=int, default=140, help="DSV41_YOCO_KEEP_TOKENS for the compact prediction")
    ap.add_argument("--per-turn", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if args.keep < 1:
        print("--keep must be >= 1", file=sys.stderr)
        return 2
    try:
        rows = load_rows(args.db, when(args.since), when(args.until), args.session, args.upstream)
    except FileNotFoundError:
        print(f"no request record at {args.db} (pass --db)", file=sys.stderr)
        return 2
    s = summarize(rows, args.path, args.keep)
    if not args.per_turn:
        s.pop("turns")
    if args.json:
        print(json.dumps(s, indent=1))
        return 0
    for k, v in s.items():
        if k != "turns":
            print(f"{k:32} {v}")
    if args.per_turn:
        print("\nstarted_at\tupstream_request_id\tP\tO\tN\thit\thit-T\treprefilled_own\tpred_off\tpred_compact")
        for t in s["turns"]:
            print("\t".join(str(t.get(k, "")) for k in (
                "started_at", "upstream_request_id", "prompt", "output", "next_prompt", "hit",
                "hit_vs_end", "reprefilled_own", "predicted_off", "predicted_compact")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
