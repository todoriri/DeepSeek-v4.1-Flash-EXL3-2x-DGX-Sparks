#!/usr/bin/env python3
"""scripts/control_token_probe.py against a local stand-in for the teacher.

The stand-in answers /tokenize and /v1/chat/completions (prompt_token_ids, as with
``return_token_ids``) from the image's tokenizer module the way the server builds
prompts (tests/test_control_token_injection.py ``_served``), in four behaviours:

  * stock: the unpatched tokenizer, which injects control tokens;
  * a1: what the 2026-10-21 A1 arm did. Every request whose prompt the guard
    changes gets HTTP 400 with the live error body, and the rest are served normally;
  * guarded: overlay/patch_control_token_injection.py as it is now;
  * chat_differs: guarded, but chat renders the control-text request with one more
    ordinary token than /tokenize does (same control ids).

On A1 the probe printed ERROR for the refused requests, and its "tokenize=false ->
HTTP 400" check passed. Now a refused control-text request is a FAIL on both arms,
the explicit ``tokenize: false`` request must equal the default one, and only a
fully guarded server ends with GUARDED. With the real tokenizer
(``DSV41_TOKENIZER_DIR`` or /model) the guarded stand-in's clean ids also match the
live pre-patch baseline in docs/.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
import test_control_token_injection as ctg  # noqa: E402

BASELINE = ROOT / "docs" / "control-token-injection-baseline-20261020.json"
A1_ERROR = (
    "DeepSeek V4.1: message text contains control-token strings that tokenize=False "
    "would turn into control tokens; remove 'tokenize' from chat_template_kwargs"
)


def _server(behaviour: str) -> ThreadingHTTPServer:
    e = ctg.env()
    lock = threading.Lock()

    def prompt_ids(body: dict) -> tuple[int, dict]:
        kwargs = body.get("chat_template_kwargs") or {}
        messages, tools = body["messages"], body.get("tools")
        with lock:
            legacy = ctg._served(e.legacy, messages, tools, **kwargs)
            guarded = ctg._served(e.guarded, messages, tools, **kwargs)
        if behaviour == "stock":
            return 200, legacy
        if behaviour == "a1" and guarded != legacy:
            return 400, {"error": {"message": A1_ERROR, "type": "BadRequestError", "param": None, "code": 400}}
        return 200, guarded

    def chat_ids(body: dict) -> tuple[int, dict]:
        status, ids = prompt_ids(body)
        if behaviour == "chat_differs" and status == 200 and "<think>y" in json.dumps(body, ensure_ascii=False):
            ids = ids[:-2] + [223] + ids[-2:]
        return status, ids

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            status, ids = (chat_ids if self.path == "/v1/chat/completions" else prompt_ids)(body)
            if status != 200:
                payload = ids
            elif self.path == "/tokenize":
                payload = {"tokens": ids, "count": len(ids), "max_model_len": 393216}
            elif self.path == "/v1/chat/completions":
                assert body["max_tokens"] == 1 and body["return_token_ids"] is True, body
                payload = {
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "x"}, "finish_reason": "length"}],
                    "prompt_token_ids": ids,
                }
            else:
                status, payload = 404, {"error": {"message": self.path}}
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _probe(behaviour: str, *args: str) -> tuple[int, list[str]]:
    server = _server(behaviour)
    try:
        run = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "control_token_probe.py"),
             "--base", f"http://127.0.0.1:{server.server_address[1]}", *args],
            capture_output=True,
            text=True,
            timeout=600,
        )
    finally:
        server.shutdown()
    assert not run.stderr.strip(), run.stderr
    return run.returncode, run.stdout.splitlines()


def test_stock_fails_on_injection() -> None:
    code, lines = _probe("stock")
    assert code == 1 and lines[-1].endswith("failing check(s)"), lines
    injected = [line for line in lines if line.startswith("FAIL") and "injected control ids" in line]
    assert {line.split()[1] for line in injected} == {"tokenize", "chat"}, lines
    assert not [line for line in lines if line.startswith("ERROR") or "refused" in line], lines


def test_a1_refusals_are_failures() -> None:
    code, lines = _probe("a1")
    assert code == 1, lines
    assert not [line for line in lines if line.startswith("ERROR")], lines
    refused = [line for line in lines if line.startswith("FAIL") and "control text refused, HTTP 400" in line]
    assert {line.split()[1] for line in refused} == {"tokenize", "chat"}, lines
    assert len(refused) == 2 * 11, refused  # 6 cases x 2 modes, minus reasoning history with thinking off
    assert lines[-1] == "23 failing check(s)", lines[-1]  # + the explicit tokenize=false request
    assert any(line.startswith("PASS tokenize reasoning_history {'enable_thinking': False}") for line in lines)
    assert any(line.startswith("FAIL tokenize=false with control text -> HTTP 400") for line in lines), lines


def test_guarded_passes_and_pins_clean_ids() -> None:
    with tempfile.TemporaryDirectory() as scratch:
        before = str(Path(scratch) / "before.json")
        code, lines = _probe("stock", "--tokenize-only", "--save", before)
        assert code == 1 and not [line for line in lines if " chat " in line], lines
        saved = json.loads(Path(before).read_text(encoding="utf-8"))
        assert set(saved) == set(json.loads(BASELINE.read_text(encoding="utf-8"))), sorted(saved)
        code, lines = _probe("guarded", "--compare", before)
    assert code == 0 and lines[-1] == "GUARDED", lines
    assert sum(line.startswith("PASS chat ") for line in lines) == 12, lines
    assert sum(line.startswith("PASS tokenize ") for line in lines) == 12, lines
    assert any(line.startswith("PASS tokenize=false with control text -> HTTP 200") for line in lines), lines
    assert any(line.startswith("PASS clean prompts id-identical") for line in lines), lines
    if ctg.env().real:
        code, lines = _probe("guarded", "--tokenize-only", "--compare", str(BASELINE))
        assert code == 0 and lines[-2].startswith("PASS clean prompts id-identical"), lines


def test_chat_must_match_tokenize() -> None:
    code, lines = _probe("chat_differs")
    differ = [line for line in lines if line.startswith("FAIL chat ") and "prompt ids differ from /tokenize" in line]
    assert code == 1 and len(differ) == 12, lines
    assert not [line for line in lines if line.startswith("FAIL tokenize ")], lines


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    source = "real" if ctg.env().real else "synthetic"
    print(f"Control-token probe OK ({len(tests)} tests, {source} tokenizer)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
