#!/usr/bin/env python3
"""Host checks for scripts/deepinfer_bench.py (DeepInfer's AIME/GPQA protocol over the
OpenAI-compatible endpoint). No network: the run loop is exercised against an in-process
fake vLLM server.

Expected prompts, extractions and estimator values are written out by hand from DeepInfer's
own examples (eval/README.md record formats, the docstrings of aime/bench.py, gpqa/bench.py and
common/passk.py), not read back from the vendored constants.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "deepinfer_bench.py"
spec = importlib.util.spec_from_file_location("deepinfer_bench", SCRIPT)
assert spec and spec.loader
db = importlib.util.module_from_spec(spec)
sys.modules["deepinfer_bench"] = db
spec.loader.exec_module(db)

STASH = Path("/mnt/data1/Projects/stash/DeepInfer/eval")
REFERENCE_ENCODER = Path("/tmp/ds1017/DeepSeek-V4.1-Flash-1017/encoding/encoding.py")

AIME_EXAMPLE = {"task_id": "aime_2026_0", "question": "Patrick started walking ...", "answer": "277"}
GPQA_EXAMPLE = {
    "task_id": "gpqa_diamond_0",
    "question": "Two quantum states ...\n\n(A) 10^-11 eV\n(B) 10^-8 eV\n(C) 10^-9 eV\n(D) 10^-4 eV",
    "answer": "D",
    "category": "Physics",
}


def _jsonl(rows, directory) -> Path:
    path = Path(directory) / "data.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


# --------------------------------------------------------------------------- vendored code
def test_vendored_files_are_unmodified_deepinfer() -> None:
    source = json.loads((db.VENDOR / "SOURCE.json").read_text())
    assert source["repo"] == "https://github.com/deepseek-ai/DeepInfer"
    assert source["license"] == "Apache-2.0"
    assert (db.VENDOR / "LICENSE").read_text().startswith(" " * 33 + "Apache License")
    assert set(source["files"]) >= {"aime/bench.py", "gpqa/bench.py", "common/passk.py", "common/io.py"}
    for rel, digest in source["files"].items():
        data = (db.VENDOR / rel).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, f"{rel} was edited; vendored files stay verbatim"
        if STASH.is_dir():
            assert data == (STASH / rel).read_bytes(), f"{rel} drifted from {STASH}"


# --------------------------------------------------------------------------- prompts
def test_aime_prompt_is_deepinfers() -> None:
    with tempfile.TemporaryDirectory() as td:
        records = db.load_records("aime", _jsonl([AIME_EXAMPLE], td))
    assert [r["task_id"] for r in records] == ["aime_2026_0"]
    assert records[0]["answer"] == "277"
    assert records[0]["messages"] == [{
        "role": "user",
        "content": "Patrick started walking ...\n"
                   "Please reason step by step, and put your final answer within \\boxed{}.",
    }]


def test_gpqa_prompt_is_deepinfers() -> None:
    with tempfile.TemporaryDirectory() as td:
        records = db.load_records("gpqa", _jsonl([GPQA_EXAMPLE], td))
    assert records[0]["answer"] == "D"
    assert records[0]["messages"] == [{
        "role": "user",
        "content": "Answer the following multiple choice question. The last line of your response "
                   "should be of the following format: 'ANSWER: $LETTER' (without quotes) where "
                   "LETTER is one of ABCD. Think step by step before answering.\n\n"
                   + GPQA_EXAMPLE["question"],
    }]


def test_gpqa_public_mirror_rows_are_normalised() -> None:
    # hendrydong/gpqa_diamond_mc shape: problem/solution, a trailing answer-format instruction
    # (stripped), and \boxed{D} as the answer (the "B" of "boxed" must not be taken).
    row = {"problem": "Q?\n\n(A) 1\n(B) 2\n(C) 3\n(D) 4\n\nPlease write your final answer in the "
                      "form of \\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}",
           "solution": "\\boxed{D}"}
    with tempfile.TemporaryDirectory() as td:
        records = db.load_records("gpqa", _jsonl([row], td))
    assert records[0]["answer"] == "D"
    assert records[0]["question"] == "Q?\n\n(A) 1\n(B) 2\n(C) 3\n(D) 4"
    assert records[0]["task_id"] == "gpqa_diamond_0"


def test_subset_selection() -> None:
    rows = [dict(AIME_EXAMPLE, task_id=f"aime_2026_{i}") for i in range(5)]
    with tempfile.TemporaryDirectory() as td:
        path = _jsonl(rows, td)
        assert [r["task_id"] for r in db.load_records("aime", path, limit=2)] == ["aime_2026_0", "aime_2026_1"]
        assert [r["task_id"] for r in db.load_records("aime", path, offset=3)] == ["aime_2026_3", "aime_2026_4"]
        picked = db.load_records("aime", path, task_ids=["aime_2026_4", "aime_2026_1"])
        assert [r["task_id"] for r in picked] == ["aime_2026_4", "aime_2026_1"]
        try:
            db.load_records("aime", path, task_ids=["nope"])
        except SystemExit:
            pass
        else:
            raise AssertionError("unknown task id accepted")


def test_effort_parsing_follows_run_eval() -> None:
    assert [db.parse_effort(v) for v in ("low", "high", "max", "1", "75", "100")] == \
        ["low", "high", "max", 1, 75, 100]
    for bad in ("0", "101", "xhigh", "medium", "", "7.5"):
        try:
            db.parse_effort(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted effort {bad!r}")
    assert [db.effort_budget(e) for e in ("low", "high", "max", 37)] == [50, 75, 100, 37]


def test_request_body() -> None:
    record = {"messages": [{"role": "user", "content": "q"}]}
    body = db.build_request(record, "max", model="m", prompt_tokens=100, seed=5)
    assert body == {
        "model": "m",
        "messages": [{"role": "user", "content": "q"}],
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 1024 * 144 - 100,
        "seed": 5,
        "skip_special_tokens": False,  # DeepInfer decodes with skip_special_tokens=False
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "max"},
    }
    body = db.build_request(record, 75, model="m", prompt_tokens=10, seed=0, max_len=20)
    assert body["chat_template_kwargs"]["reasoning_effort"] == 75
    assert body["max_tokens"] == 10
    assert db.build_request(record, 75, model="m", prompt_tokens=30, seed=0, max_len=20)["max_tokens"] == 1


def _reference_render(messages, effort):
    return ("<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: "
            f"{db.effort_budget(effort)} (range 1-100, the higher the value, the more thorough the "
            f"reasoning)\n\n<｜User｜>{messages[0]['content']}<｜Assistant｜><think>")


def test_rendered_prompt_check() -> None:
    messages = [{"role": "user", "content": "q"}]
    for effort in ("low", "high", "max", 37):
        assert db.check_rendered_prompt(_reference_render(messages, effort), messages, effort) == []
    # The live 0909 image's stale mapping rendered "high" as 50 (2026-09-23); must be refused.
    stale = _reference_render(messages, 50)
    assert db.check_rendered_prompt(stale, messages, "high")
    # Thinking off renders </think> and no effort line.
    assert db.check_rendered_prompt("<｜begin▁of▁sentence｜><｜User｜>q<｜Assistant｜></think>", messages, "max")
    if REFERENCE_ENCODER.is_file():
        spec_ = importlib.util.spec_from_file_location("ckpt_encoding", REFERENCE_ENCODER)
        enc = importlib.util.module_from_spec(spec_)
        spec_.loader.exec_module(enc)
        assert enc.REASONING_EFFORT_MAPPINGS == db.EFFORT_LEVELS
        for effort in ("low", "high", "max", 37):
            ref = enc.encode_messages(messages, thinking_mode="thinking", reasoning_effort=effort)
            assert ref == _reference_render(messages, effort)
            assert db.check_rendered_prompt(ref, messages, effort, reference=ref) == []
        assert db.check_rendered_prompt(ref + " ", messages, 37, reference=ref)


# --------------------------------------------------------------------------- extraction
def test_raw_text_is_what_deepinfer_decodes() -> None:
    assert db.rollout_raw_text("think", "ans", "stop") == "think</think>ans"
    assert db.rollout_raw_text("think", "", "stop") == "think</think>"
    assert db.rollout_raw_text("think", None, "length") == "think"  # cut inside the thinking
    assert db.rollout_raw_text("think", "partial", "length") == "think</think>partial"


def _score(bench, reasoning, content, gold, finish="stop"):
    row = {"task_id": "t", "sample_idx": 0, "reasoning": reasoning, "content": content,
           "finish_reason": finish}
    return db.score_rollout(bench, row, {"task_id": "t", "answer": gold})


def test_aime_extraction() -> None:
    s = _score("aime", "draft \\boxed{12} hmm", "So the answer is \\boxed{277}.", "277")
    assert (s["pred"], s["extract_source"], s["correct"]) == ("277", "boxed", True)
    s = _score("aime", "", "\\boxed{\\frac{1}{2}}", "277")  # nested braces (extract_boxed docstring)
    assert (s["pred"], s["correct"]) == ("\\frac{1}{2}", False)
    assert _score("aime", "", "\\boxed{1{,}234}", "1234")["pred"] == "1234"  # LaTeX thousands sep
    assert _score("aime", "", "$\\boxed{042}$", "42")["correct"] is True
    # No \boxed: the fallback scans only after </think>, so the draft 999 is not taken.
    s = _score("aime", "maybe 999", "The answer is 277", "277")
    assert (s["pred"], s["extract_source"], s["correct"]) == ("277", "tail_number", True)
    # Upstream quirk, kept: the tail regex refuses a number followed by "." (end of a sentence).
    assert _score("aime", "", "The answer is 277.", "277")["extract_source"] == "none"
    # Cut inside the thinking: no </think>, so the last 400 characters are scanned.
    s = _score("aime", "x" * 500 + " so far 5, 6, 7", None, "7", finish="length")
    assert (s["pred"], s["extract_source"]) == ("7", "tail_number")
    s = _score("aime", "", "no idea", "277")
    assert (s["pred"], s["extract_source"], s["correct"]) == (None, "none", False)


def test_gpqa_extraction() -> None:
    s = _score("gpqa", "", "reasoning\nANSWER: D", "D")
    assert (s["pred"], s["extract_source"], s["correct"]) == ("D", "answer_line", True)
    assert _score("gpqa", "ANSWER: A", "Final.\nANSWER: B", "B")["pred"] == "B"  # last line wins
    # Upstream quirk, kept: the ANSWER regex is line-anchored, so an answer glued to </think>
    # loses to a draft line in the thinking (vLLM passes content through unstripped).
    assert _score("gpqa", "ANSWER: A", "ANSWER: B", "B")["pred"] == "A"
    assert _score("gpqa", "", "\nanswer: (b)", "B")["pred"] == "B"
    s = _score("gpqa", "", "\\boxed{C}", "C")
    assert (s["pred"], s["extract_source"]) == ("C", "boxed")
    s = _score("gpqa", "A or B?", "I pick C.", "C")
    assert (s["pred"], s["extract_source"]) == ("C", "tail_letter")


def test_blacklisted_tokens_are_flagged() -> None:
    # DeepInfer blocks these at decode time; the endpoint cannot (no logit_bias with DSpark).
    assert _score("aime", "", "\\boxed{1}", "1")["blacklisted_tokens"] == []
    assert _score("aime", "<｜▁pad▁｜>", "\\boxed{1}", "1")["blacklisted_tokens"] == ["<｜▁pad▁｜>"]


# --------------------------------------------------------------------------- scoring math
def test_passk_is_deepinfers() -> None:
    est = db.passk().estimator
    assert est(4, 2, 1) == 0.5
    assert abs(est(4, 2, 2) - (1 - 1 / 6)) < 1e-12
    assert est(3, 2, 2) == 1.0
    k_list = db.passk().default_k_list
    assert k_list(32) == [1, 2, 4, 8, 16, 32]
    assert k_list(8) == [1, 2, 4, 8]
    assert k_list(3) == [1, 2, 3]
    assert k_list(1) == [1]
    rows = [{"task_id": "t1", "correct": True}, {"task_id": "t1", "correct": False},
            {"task_id": "t2", "correct": True}, {"task_id": "t2", "correct": True}]
    summary = db.passk().summarize(rows, n_sample=2)
    assert summary["Pass@1"] == 0.75 and summary["Pass@2"] == 1.0
    assert summary["task_accuracy"] == {"t1": 0.5, "t2": 1.0}


def test_paired_tests() -> None:
    assert db.mcnemar_exact(2, 0) == 0.5  # docs/effort-ab-75-vs-100-20260924.md
    assert db.mcnemar_exact(0, 0) == 1.0
    assert db.mcnemar_exact(5, 0) == 0.0625
    assert db.mcnemar_exact(3, 1) == 0.625
    assert db.sign_flip_p([1, 1]) == 0.5
    assert db.sign_flip_p([1] * 5) == 0.0625
    assert db.sign_flip_p([0.5, -0.5, 0]) == 1.0
    assert db.sign_flip_p([]) == 1.0
    lo, hi = db.bootstrap_ci([0.25] * 6, seed=1)
    assert lo == hi == 0.25
    assert db.bootstrap_ci([0, 1, 0, 1], seed=3) == db.bootstrap_ci([0, 1, 0, 1], seed=3)


def _scored(task, effort, sample, ok, tokens=100):
    return {"task_id": task, "effort": effort, "effort_label": effort, "sample_idx": sample,
            "correct": ok, "finish_reason": "stop", "completion_tokens": tokens, "elapsed_s": 2.0,
            "pred": "1", "gold": "1", "extract_source": "boxed"}


def test_compare_arms_is_item_clustered() -> None:
    rows = [_scored("a", 75, 0, True), _scored("a", 75, 1, False),
            _scored("a", 100, 0, True), _scored("a", 100, 1, True),
            _scored("b", 75, 0, True), _scored("b", 100, 0, True),
            _scored("c", 75, 0, True)]  # c has no 100 rollout: not comparable
    cmp_ = db.compare_arms(rows, 75, 100)
    assert cmp_["items"] == 2
    assert cmp_["per_item_diff"] == {"a": 0.5, "b": 0.0}
    assert cmp_["mean_diff"] == 0.25
    assert cmp_["discordant"] == {"75_only": [], "100_only": [["a", 1]]}
    assert cmp_["mcnemar_p"] == 1.0
    assert cmp_["sign_flip_p"] == 1.0


def test_paired_export_feeds_effort_ab_paired() -> None:
    rows = [_scored(t, e, s, not (t == "aime_2026_1" and e == 75 and s == 0), 100 + e)
            for t in ("aime_2026_0", "aime_2026_1") for e in (75, 100) for s in (0, 1)]
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "paired.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in db.paired_rows(rows, "aime")))
        for extra in ([], ["--efforts", "75,100"]):
            out = subprocess.run([sys.executable, str(ROOT / "scripts" / "effort_ab_paired.py"),
                                  str(path), *extra], capture_output=True, text=True)
            assert out.returncode == 0, out.stderr
            assert "rows 8  errors 0  complete pairs 4" in out.stdout
            assert "75 right/100 wrong = 0 []; 100 right/75 wrong = 1 [('aime_2026_1', 0)]" in out.stdout


# --------------------------------------------------------------------------- plan / resume
def test_plan_is_sample_major_and_interleaved() -> None:
    jobs = db.plan_jobs(["a", "b"], [75, 100], n_sample=2, seed=0)
    assert len(jobs) == 8
    assert [j.sample_idx for j in jobs] == [0, 0, 0, 0, 1, 1, 1, 1]
    for i in range(0, 8, 2):  # each (task, sample) pair runs back to back, both arms
        assert jobs[i].task_id == jobs[i + 1].task_id and jobs[i].sample_idx == jobs[i + 1].sample_idx
        assert {jobs[i].effort, jobs[i + 1].effort} == {75, 100}
        assert jobs[i].seed == jobs[i + 1].seed  # common seed across arms
    assert jobs == db.plan_jobs(["a", "b"], [75, 100], n_sample=2, seed=0)
    assert len({j.seed for j in jobs}) == 4
    # Seeds do not depend on n_sample or the item order, so a top-up run keeps them.
    assert db.sample_seed("aime_2026_0", 0) == db.plan_jobs(["x", "aime_2026_0"], [75], 5, 0)[1].seed
    assert db.sample_seed("aime_2026_0", 0) != db.sample_seed("aime_2026_0", 1)


def test_resume_skips_finished_rollouts_only() -> None:
    raw = [{"task_id": "a", "effort": 75, "sample_idx": 0, "status": "done"},
           {"task_id": "a", "effort": 100, "sample_idx": 0, "status": "yielded"},
           {"task_id": "b", "effort": 75, "sample_idx": 0, "status": "error"},
           {"task_id": "b", "effort": 100, "sample_idx": 0, "status": "budget"}]
    assert db.done_keys(raw) == {("a", 75, 0)}
    todo = [j for j in db.plan_jobs(["a", "b"], [75, 100], 1, 0) if j.key not in db.done_keys(raw)]
    assert sorted(j.key for j in todo) == [("a", 100, 0), ("b", 75, 0), ("b", 100, 0)]


def test_resume_refuses_a_different_protocol() -> None:
    old = {"bench": "aime", "data_sha256": "x", "model": "m", "temperature": 1.0, "top_p": 1.0,
           "max_len": 147456, "base": "http://h:8000"}
    assert db.resume_mismatches(old, dict(old)) == []
    for key, value in (("temperature", 0.6), ("top_p", 0.95), ("max_len", 32768),
                       ("data_sha256", "y"), ("model", "n"), ("bench", "gpqa")):
        assert db.resume_mismatches(old, dict(old, **{key: value})) == [key]


# --------------------------------------------------------------------------- idle gate
METRICS = """# HELP vllm:num_requests_running x
vllm:num_requests_running{engine="0",model_name="m"} %d.0
vllm:num_requests_waiting{engine="0",model_name="m"} %d.0
vllm:num_requests_waiting_by_reason{engine="0",model_name="m",reason="capacity"} 9.0
vllm:prompt_tokens_total{engine="0",model_name="m"} %s
"""


def test_metrics_parsing() -> None:
    m = db.parse_metrics(METRICS % (1, 2, "4.48187781e+08"))
    assert (m.running, m.waiting, m.prompt_tokens) == (1, 2, 448187781)


def test_foreign_detector() -> None:
    det = db.ForeignDetector()
    assert det.update(db.parse_metrics(METRICS % (0, 0, "1000")), ours=0) == ""  # baseline
    assert det.update(db.parse_metrics(METRICS % (0, 0, "1000")), ours=0) == ""
    det.expect_prompt(15)  # our request was sent; its prompt will be counted
    assert det.update(db.parse_metrics(METRICS % (1, 0, "1015")), ours=1) == ""
    assert det.update(db.parse_metrics(METRICS % (2, 0, "1015")), ours=1)  # a second request
    assert det.update(db.parse_metrics(METRICS % (1, 1, "1015")), ours=1)  # a queued request
    assert det.update(db.parse_metrics(METRICS % (1, 0, "1060")), ours=1)  # one came and went
    assert det.update(db.parse_metrics(METRICS % (0, 0, "1060")), ours=0) == ""


def test_proxy_last_activity() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "requests.sqlite"
        con = sqlite3.connect(path)
        con.execute("CREATE TABLE requests (id TEXT PRIMARY KEY, started_at TEXT NOT NULL, "
                    "ended_at TEXT, upstream TEXT, call_origin TEXT NOT NULL DEFAULT 'caller')")
        con.executemany("INSERT INTO requests VALUES (?,?,?,?,?)", [
            ("1", "2026-10-20T19:30:00.000Z", "2026-10-20T19:31:00.000Z", "http://192.168.1.97:8000", "caller"),
            ("2", "2026-10-20T19:35:00.000Z", "2026-10-20T19:36:30.500Z", "http://192.168.1.97:8000", "caller"),
            ("3", "2026-10-20T19:40:00.000Z", "2026-10-20T19:41:00.000Z", "http://127.0.0.1:8081", "caller"),
            # The proxy's own prefix keepalive (1 output token every 6 min per session) is not a
            # live session; counting it would keep the gate shut forever (2026-10-20 smoke run).
            ("4", "2026-10-20T19:49:15.000Z", "2026-10-20T19:49:16.200Z", "http://192.168.1.97:8000", "keepalive"),
        ])
        con.commit()
        con.close()
        now = db.iso_ts("2026-10-20T19:50:00Z")
        assert db.proxy_last_activity(str(path), "http://192.168.1.97:8000/", now) == \
            db.iso_ts("2026-10-20T19:36:30.500Z")
        assert db.proxy_last_activity(str(path), "http://10.0.0.1:8000", now) is None


# --------------------------------------------------------------------------- run loop (fake server)
class FakeVLLM:
    """/health, /v1/models, /metrics, /tokenize, /detokenize and streaming chat, like the teacher."""

    def __init__(self, answers, foreign_after_chunks=None, blip_s=0.3, repeat=False):
        self.answers = answers  # (task question, budget) -> boxed answer
        self.foreign_after_chunks = foreign_after_chunks  # report a foreign request mid-stream
        self.blip_s, self.repeat = blip_s, repeat  # for this long; once, or in every rollout
        self.running = 0
        self.foreign = 0
        self.prompt_tokens = 0
        self.chats = []
        self.lock = threading.Lock()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, obj, code=200):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/health":
                    self._json({})
                elif self.path == "/v1/models":
                    self._json({"data": [{"id": "deepseek-v4.1-flash", "max_model_len": 393216}]})
                elif self.path == "/metrics":
                    with fake.lock:
                        text = METRICS % (fake.running + fake.foreign, 0, str(float(fake.prompt_tokens)))
                    data = text.encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self._json({}, 404)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/tokenize":
                    effort = body["chat_template_kwargs"]["reasoning_effort"]
                    text = _reference_render(body["messages"], effort)
                    self._json({"count": len(text), "tokens": [ord(c) for c in text]})
                elif self.path == "/detokenize":
                    self._json({"prompt": "".join(chr(t) for t in body["tokens"])})
                elif self.path == "/v1/chat/completions":
                    fake._chat(self, body)
                else:
                    self._json({}, 404)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def _chat(self, handler, body):
        import time as _t
        effort = body["chat_template_kwargs"]["reasoning_effort"]
        question = body["messages"][0]["content"]
        answer = self.answers[(question.split("\n")[0], db.effort_budget(effort))]
        with self.lock:
            self.chats.append((question.split("\n")[0], effort, body["seed"]))
            self.running += 1
            self.prompt_tokens += 10
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.end_headers()
        pieces = [("reasoning", "thinking ")] * 6 + [("content", f"\\boxed{{{answer}}}")]
        try:
            for n, (field, text) in enumerate(pieces, 1):
                if self.foreign_after_chunks == n:
                    if not self.repeat:
                        self.foreign_after_chunks = None
                    with self.lock:
                        self.foreign = 1
                    _t.sleep(self.blip_s)  # long enough for the client's metrics poll
                    with self.lock:
                        self.foreign = 0
                chunk = {"choices": [{"index": 0, "delta": {field: text}, "finish_reason": None}],
                         "usage": {"prompt_tokens": 10, "completion_tokens": n}}
                handler.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                handler.wfile.flush()
                _t.sleep(0.01)
            final = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 10, "completion_tokens": len(pieces),
                               "completion_tokens_details": {"reasoning_tokens": 6}}}
            handler.wfile.write(f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n".encode())
            handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with self.lock:
                self.running -= 1
                self.foreign = 0

    def close(self):
        self.server.shutdown()


def _run(fake, out, data, *extra):
    return db.main(["run", "--bench", "aime", "--data-file", str(data), "--out-dir", str(out),
                    "--base", fake.base, "--efforts", "75,max", "--limit", "2", "--n-sample", "1",
                    "--idle-quiet-s", "0", "--poll-s", "0.05", "--yield-after-s", "0.1",
                    "--proxy-db", "none", *extra])


def test_run_score_resume_against_fake_server() -> None:
    rows = [{"task_id": "aime_2026_0", "question": "Q0", "answer": "1"},
            {"task_id": "aime_2026_1", "question": "Q1", "answer": "2"},
            {"task_id": "aime_2026_2", "question": "Q2", "answer": "3"}]
    answers = {("Q0", 75): "1", ("Q0", 100): "1", ("Q1", 75): "9", ("Q1", 100): "2",
               ("Q2", 75): "3", ("Q2", 100): "3"}
    fake = FakeVLLM(answers, foreign_after_chunks=3)
    try:
        with tempfile.TemporaryDirectory() as td:
            data, out = _jsonl(rows, td), Path(td) / "run"
            assert _run(fake, out, data) == 0
            raw = [json.loads(line) for line in (out / "raw.jsonl").read_text().splitlines()]
            statuses = [r["status"] for r in raw]
            assert statuses.count("yielded") == 1, statuses  # the foreign request made us back off
            assert statuses.count("done") == 4
            assert len(fake.chats) == 5
            assert {e for _, e, _ in fake.chats} == {75, "max"}  # the label goes to the server as given
            meta = json.loads((out / "meta.json").read_text())
            assert meta["task_ids"] == ["aime_2026_0", "aime_2026_1"]
            assert meta["efforts"] == [75, 100] and meta["effort_labels"] == [75, "max"]
            assert all(p["ok"] for p in meta["preflight"]["prompts"])

            assert _run(fake, out, data) == 0  # resume: nothing left to do
            assert len(fake.chats) == 5

            assert db.main(["score", "--out-dir", str(out)]) == 0
            summary = json.loads((out / "summary.json").read_text())
            assert summary["arms"]["75"]["Pass@1"] == 0.5
            assert summary["arms"]["100"]["Pass@1"] == 1.0
            assert summary["arms"]["100"]["coverage_complete"] is True
            assert summary["arms"]["75"]["yielded_attempts"] + summary["arms"]["100"]["yielded_attempts"] == 1
            cmp_ = summary["comparisons"]["75_vs_100"]
            assert cmp_["discordant"] == {"75_only": [], "100_only": [["aime_2026_1", 0]]}
            paired = [json.loads(line) for line in (out / "paired.jsonl").read_text().splitlines()]
            assert len(paired) == 4 and {p["effort"] for p in paired} == {75, 100}

            # Token budget: a third item would need more than 3 tokens; the rollout is aborted,
            # recorded as `budget`, never scored, and redone by the next run. The poll interval is
            # longer than the whole rollout, so only the per-chunk check can stop it in time.
            assert _run(fake, out, data, "--limit", "3", "--budget-tokens", "3", "--poll-s", "0.5") == 0
            raw = [json.loads(line) for line in (out / "raw.jsonl").read_text().splitlines()]
            assert raw[-1]["status"] == "budget" and raw[-1]["task_id"] == "aime_2026_2"
            assert raw[-1]["completion_tokens"] == 4
            assert len(raw) == 6
            assert db.main(["score", "--out-dir", str(out)]) == 0
            scored = [json.loads(line) for line in (out / "scored.jsonl").read_text().splitlines()]
            assert len(scored) == 4 and all(r["status"] == "done" for r in scored)
            summary = json.loads((out / "summary.json").read_text())
            assert sum(a["budget_attempts"] for a in summary["arms"].values()) == 1
            assert summary["arms"]["100"]["coverage_complete"] is False  # the plan grew to 3 items
    finally:
        fake.close()


def test_short_foreign_blip_does_not_yield() -> None:
    # A keepalive-sized foreign request (shorter than --yield-after-s) shares the engine
    # briefly instead of costing a whole rollout.
    # Every rollout carries a 0.2 s blip; rollouts start >= 0.6 s apart, so blips are separate
    # episodes and none lasts the 0.5 s needed to yield.
    rows = [{"task_id": "aime_2026_0", "question": "Q0", "answer": "1"}]
    fake = FakeVLLM({("Q0", 75): "1", ("Q0", 100): "1"}, foreign_after_chunks=3, blip_s=0.2, repeat=True)
    try:
        with tempfile.TemporaryDirectory() as td:
            data, out = _jsonl(rows, td), Path(td) / "run"
            assert _run(fake, out, data, "--yield-after-s", "0.5", "--idle-quiet-s", "0.6") == 0
            statuses = [json.loads(line)["status"] for line in (out / "raw.jsonl").read_text().splitlines()]
            assert statuses == ["done", "done"], statuses
    finally:
        fake.close()


def test_dry_run_sends_nothing() -> None:
    fake = FakeVLLM({})
    try:
        with tempfile.TemporaryDirectory() as td:
            data = _jsonl([AIME_EXAMPLE], td)
            assert _run(fake, Path(td) / "run", data, "--dry-run") == 0
            assert fake.chats == [] and not (Path(td) / "run" / "raw.jsonl").exists()
    finally:
        fake.close()


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"deepinfer_bench OK ({len(tests)} tests)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
