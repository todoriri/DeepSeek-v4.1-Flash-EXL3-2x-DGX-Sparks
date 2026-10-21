#!/usr/bin/env python3
"""Offline checks for scripts/swa_final_window_turns.py on a synthetic request record.

  * the 40 recorded turns of session 3b75b1f3 give the record's own-output re-prefill
    (19,099 tokens), and the length-only predictions equal what the patched image code
    does in tests/test_swa_final_window_sim.py (4,639 YOCO off, 4,821 YOCO compact);
  * chaining: internal origins, other sessions/upstreams, errors and other paths are
    left out; a rewritten history and an unknown cached count are counted, not guessed;
  * decode / prefill rates; the DB is opened read-only; a missing DB exits 2.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import subprocess
import sys
import tempfile
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
