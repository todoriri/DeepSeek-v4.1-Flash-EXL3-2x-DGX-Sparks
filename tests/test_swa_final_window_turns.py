#!/usr/bin/env python3
"""Offline checks for scripts/swa_final_window_turns.py on a synthetic request record.

  * the 40 recorded turns of session 3b75b1f3 give the record's own-output re-prefill
    (19,099 tokens), and the length-only predictions equal what the patched image code
    does in tests/test_swa_final_window_sim.py (4,639 YOCO off, 4,821 YOCO compact);
  * chaining: internal origins, other sessions/upstreams, errors and other paths are
    left out; a rewritten history and an unknown cached count are counted, not guessed;
  * decode / prefill rates; the DB is opened read-only; a missing DB exits 2;
  * history (2026-10-22): a turn whose next request drops the previous output's reasoning
    (repl request_bodies) is "dropped", not reusable; "echoed" / "no_reasoning" are; no
    body, an unreadable one or no assistant message is "unknown"; totals per agent_mode.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "swa_final_window_turns.py"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


W = _module(SCRIPT, "swa_final_window_turns")
SESSION = _module(ROOT / "tests" / "test_swa_final_window_sim.py", "swa_sim_rows").SESSION_3B75
SID = "3b75b1f3-498e-52b9-8632-3cb3ab2ac779"
UP = "http://192.168.1.97:8000"
_TMP = tempfile.TemporaryDirectory(prefix="swa-turns-")


def _row(i, **kw):
    r = dict(id=f"r{i}", started_at=f"2026-10-18T17:{i // 60:02d}:{i % 60:02d}.000Z",
             session_id=SID, upstream=UP, call_origin="caller", status_code=200,
             path="/v1/chat/completions", prompt_tokens=0, output_tokens=0, cached_tokens=0,
             cached_source="backend", ttft_ms=2000, total_ms=6000,
             upstream_request_id=f"chatcmpl-{i}")
    r.update(kw)
    return r


def _db(rows) -> Path:
    path = Path(tempfile.mkdtemp(dir=_TMP.name)) / "requests.sqlite"
    con = sqlite3.connect(path)
    cols = list(rows[0])
    con.execute(f"CREATE TABLE requests ({', '.join(cols)})")
    con.executemany(f"INSERT INTO requests VALUES ({', '.join('?' * len(cols))})",
                    [tuple(r[c] for c in cols) for r in rows])
    con.commit()
    con.close()
    return path


def _bodies(db: Path, bodies: dict) -> None:
    """repl's request_bodies table: (id, zlib JSON) or raw bytes for a broken one."""
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE request_bodies (seq INTEGER, id TEXT, encoding TEXT, size INTEGER, sha256 TEXT, body BLOB)")
    for seq, (rid, body) in enumerate(bodies.items()):
        blob = body if isinstance(body, bytes) else zlib.compress(json.dumps(body).encode())
        con.execute("INSERT INTO request_bodies VALUES (?, ?, 'zlib', ?, '', ?)", (seq, rid, len(blob), blob))
    con.commit()
    con.close()


def _history(reasoning: str | None, field: str = "reasoning_content") -> dict:
    """A chat body whose last assistant message (the previous output) has this reasoning."""
    assistant = {"role": "assistant", "content": "ok", "tool_calls": []}
    if reasoning is not None:
        assistant[field] = reasoning
    return {"model": "m", "messages": [{"role": "user", "content": "go"}, assistant,
                                        {"role": "tool", "content": "result"}]}


def _session_rows():
    return [_row(i, prompt_tokens=P, output_tokens=O, cached_tokens=c)
            for i, (P, O, c) in enumerate(SESSION)]


def _run(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], text=True, capture_output=True)


def test_session_numbers_match_the_simulation():
    db = _db(_session_rows())
    digest = hashlib.sha256(db.read_bytes()).hexdigest()
    r = _run("--db", str(db), "--json")
    assert r.returncode == 0, r.stderr
    s = json.loads(r.stdout)
    assert s["continuations"] == s["with_full_previous_output"] == 39, s
    assert s["own_output_tokens"] == 32221
    assert s["reprefilled_own_tokens"] == 19099
    assert s["predicted_keep_off_tokens"] == 4639          # == the patched image code
    assert s["predicted_keep_compact_tokens"] == 4821      # == patched + YOCO compact
    assert s["hit_at_prompt_tail"] + s["hit_inside_output"] == 39
    assert s["hit_inside_output"] == 9                     # all on 4096 checkpoints
    assert s["decode_tok_s"]["n"] == sum(1 for _, O, _ in SESSION if O >= 64)
    assert "turns" not in s
    assert hashlib.sha256(db.read_bytes()).hexdigest() == digest, "record modified"
    per = json.loads(_run("--db", str(db), "--json", "--per-turn").stdout)["turns"]
    assert per[0]["upstream_request_id"] == "chatcmpl-1" and per[0]["hit"] == 34432
    assert per[0]["predicted_off"] == W.reprefilled_own(34516, 263, 34688)


def test_chaining_rules():
    rows = _session_rows()[:4]
    n = len(rows)
    rows += [
        _row(n, call_origin="keepalive", prompt_tokens=10, output_tokens=1),
        _row(n + 1, session_id="other", prompt_tokens=500, output_tokens=10),
        _row(n + 2, upstream="http://elsewhere:8000", prompt_tokens=60000, output_tokens=10),
        _row(n + 3, status_code=500, prompt_tokens=60000),
        _row(n + 4, path="/v1/completions", prompt_tokens=60000),
        # history rewritten (a summary): shorter than the previous prompt + output
        _row(n + 5, prompt_tokens=30000, output_tokens=50, cached_tokens=29952),
        # unknown cached count (source "none")
        _row(n + 6, prompt_tokens=31000, output_tokens=50, cached_tokens=0, cached_source="none"),
    ]
    s = W.summarize(rows, "/v1/chat/completions", 140)
    assert s["continuations"] == 5, s                    # 3 in the session + 2 below
    assert s["rewritten_history"] == 1 and s["unknown_cached"] == 1
    assert s["with_full_previous_output"] == 3
    only = W.summarize([r for r in rows if r["session_id"] == SID], "/v1/chat/completions", 140)
    assert only["continuations"] == 5


def test_history_classes():
    # (P, O, R, cached) per request; every next prompt holds the whole previous output.
    turns = [(1000, 400, 300, 0), (1500, 200, 0, 1024), (1800, 300, 250, 1664), (2200, 100, 50, 1792),
             (2400, 90, 60, 2176), (2600, 80, 40, 2560), (2800, 70, 30, 2560), (3000, 10, 0, 2816)]
    rows = [_row(i, prompt_tokens=P, output_tokens=O, reasoning_tokens=R, cached_tokens=c, agent_mode="")
            for i, (P, O, R, c) in enumerate(turns)]
    rows += [_row(20 + i, session_id="dsh", agent_mode="coding", prompt_tokens=P, output_tokens=O,
                  reasoning_tokens=R, cached_tokens=c) for i, (P, O, R, c) in enumerate(turns[:2])]
    db = _db(rows)
    _bodies(db, {
        "r1": _history("thought"),                # r0's reasoning echoed
        "r2": _history(None),                     # r1 had none: no_reasoning, body irrelevant
        "r3": _history(None),                     # r2's reasoning dropped
        # r4: no body                             -> unknown
        "r5": b"not zlib",                        # unreadable -> unknown
        "r6": {"model": "m", "messages": [{"role": "user", "content": "x"}]},  # no assistant -> unknown
        "r7": _history("thought", "reasoning"),   # r6's reasoning echoed under "reasoning"
        "r21": _history(None),                    # dsh: r20's reasoning dropped
    })
    s = json.loads(_run("--db", str(db), "--json", "--per-turn").stdout)
    got = {t["upstream_request_id"]: t["history"] for t in s["turns"]}
    assert got == {"chatcmpl-1": "echoed", "chatcmpl-2": "no_reasoning", "chatcmpl-3": "dropped",
                   "chatcmpl-4": "unknown", "chatcmpl-5": "unknown", "chatcmpl-6": "unknown",
                   "chatcmpl-7": "echoed", "chatcmpl-21": "dropped"}, got
    h = s["by_history"]
    assert {k: v["turns"] for k, v in h.items()} == {"no_reasoning": 1, "echoed": 2, "dropped": 2, "unknown": 3}
    assert h["dropped"]["own_output_tokens"] == 300 + 400
    assert h["dropped"]["own_output_in_next_prompt"] == (300 - 250) + (400 - 300)
    assert s["reusable"]["turns"] == 3
    assert s["reusable"]["own_output_tokens"] == 400 + 200 + 70
    assert s["reusable"]["reprefilled_own_tokens"] == sum(
        W.reprefilled_own(P, O, turns[i + 1][3]) for i, (P, O, _, _) in enumerate(turns) if i in (0, 1, 6))
    assert sum(v["turns"] for v in h.values()) == s["with_full_previous_output"] == 8
    assert s["by_agent_mode"]["coding"]["turns"] == 1 and s["by_agent_mode"]["-"]["turns"] == 7
    # Without a request_bodies table every reasoning turn is unknown; R = 0 is still no_reasoning.
    plain = W.summarize(rows, "/v1/chat/completions", 140)
    assert {k: v["turns"] for k, v in plain["by_history"].items()} == {
        "no_reasoning": 1, "echoed": 0, "dropped": 0, "unknown": 7}
    assert W.history({"reasoning_tokens": None}, {"history_reasoning": True}) == "unknown"


def test_missing_db_and_args():
    assert _run("--db", "/nonexistent/requests.sqlite").returncode == 2
    db = _db(_session_rows())
    assert _run("--db", str(db), "--keep", "0").returncode == 2
    r = _run("--db", str(db), "--session", "nobody", "--json")
    assert r.returncode == 0 and json.loads(r.stdout)["continuations"] == 0
    r = _run("--db", str(db), "--from", "2026-10-18T17:00:10.000Z", "--json")
    assert json.loads(r.stdout)["continuations"] == 29


def test_relative_times():
    assert W.when("2026-10-18T17:00:00Z") == "2026-10-18T17:00:00Z"
    assert W.when(None) is None
    a, b = W.when("-7d"), W.when("now")
    assert a < b and a.endswith("Z") and len(a) == len("2026-10-18T17:00:00.000Z")
    db = _db(_session_rows())
    r = _run("--db", str(db), "--from=-36500d", "--to", "now", "--json")   # argparse: --from=-X
    assert r.returncode == 0 and json.loads(r.stdout)["continuations"] == 39


def test_cached_known_rule():
    assert W.cached_known({"cached_tokens": 5, "cached_source": "backend"}) == 5
    assert W.cached_known({"cached_tokens": 5, "cached_source": "none"}) is None
    assert W.cached_known({"cached_tokens": 7}) == 7
    assert W.cached_known({"cached_tokens": 0, "started_at": "2026-09-01T00:00:00Z"}) is None
    assert W.cached_known({"cached_tokens": 0, "started_at": "2026-10-01T00:00:00Z"}) == 0


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"swa_final_window_turns offline OK ({len(tests)} tests)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
