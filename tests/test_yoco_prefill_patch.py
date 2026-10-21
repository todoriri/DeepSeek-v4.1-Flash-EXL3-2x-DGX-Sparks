#!/usr/bin/env python3
"""Host checks for overlay/patch_yoco_prefill.py (decoder-SWA bounded replay).

The patch is applied to unpatched copies of the image's seven target files
(tests/fixtures/dsv41_yoco). Coverage:

  * every anchor applies exactly once and the patched files still compile;
  * a second run is a no-op;
  * anchor drift (missing or duplicated) fails loudly and writes nothing, in any file;
  * --check verifies without writing;
  * the patch is a no-op unless DSV41_YOCO_PREFILL=compact;
  * start.sh and the Dockerfile wire it like the other overlays, both ranks.

Set DSV41_YOCO_IMAGE_VLLM=/path/to/extracted/vllm to also --check a full image tree.
"""
from __future__ import annotations

import importlib.util
import os
import py_compile
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATCH_PATH = ROOT / "overlay" / "patch_yoco_prefill.py"
FIXTURES = ROOT / "tests" / "fixtures" / "dsv41_yoco" / "vllm"
spec = importlib.util.spec_from_file_location("patch_yoco_prefill", PATCH_PATH)
assert spec and spec.loader
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)

TARGETS = [
    "models/deepseek_v4_1/nvidia/model.py",
    "models/deepseek_v4_1/attention.py",
    "models/deepseek_v4_1/nvidia/model_state.py",
    "v1/core/kv_cache_manager.py",
    "v1/core/single_type_kv_cache_manager.py",
    "v1/engine/input_processor.py",
    "v1/worker/gpu/model_runner.py",
]


def _tree() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="yoco-patch-"))
    shutil.copytree(FIXTURES, tmp / "vllm")
    return tmp / "vllm"


def _run(root: Path, *args: str, mode: str | None = "compact") -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("DSV41_YOCO")}
    if mode is not None:
        env["DSV41_YOCO_PREFILL"] = mode
    return subprocess.run([sys.executable, str(PATCH_PATH), "--root", str(root), *args],
                          env=env, text=True, capture_output=True)


def _snapshot(root: Path) -> dict[str, str]:
    return {rel: (root / rel).read_text() for rel in TARGETS}


def test_jobs_cover_targets():
    assert sorted(rel for rel, _, _ in patch.JOBS) == sorted(TARGETS)
    for rel, edits, _ in patch.JOBS:
        assert edits, rel
        for old, new in edits:
            assert patch.MARK in new, (rel, old[:40])


def test_apply_compile_idempotent():
    root = _tree()
    before = _snapshot(root)
    r = _run(root)
    assert r.returncode == 0, r.stderr
    after = _snapshot(root)
    for rel in TARGETS:
        assert after[rel] != before[rel], rel
        assert f"[{patch.MARK}]" in after[rel], rel
        py_compile.compile(str(root / rel), doraise=True)
    r2 = _run(root)
    assert r2.returncode == 0 and "already applied" in r2.stdout, r2
    assert _snapshot(root) == after


def test_patched_semantics_spotcheck():
    root = _tree()
    assert _run(root).returncode == 0
    t = _snapshot(root)
    kvm = t["v1/core/kv_cache_manager.py"]
    assert "max_cache_hit_length = request.num_tokens - 1\n" not in kvm
    assert "max(0, request.num_tokens - _DSV41_YOCO_KEEP)" in kvm
    stm = t["v1/core/single_type_kv_cache_manager.py"]
    assert "request.num_prompt_tokens - _DSV41_YOCO_KEEP" in stm
    assert "reachable_boundaries = [request.num_prompt_tokens - 1]" in stm, "replay boundary kept"
    ip = t["v1/engine/input_processor.py"]
    assert "params.prompt_logprobs is not None" in ip
    mr = t["v1/worker/gpu/model_runner.py"]
    assert "need_eager=is_profile or skip_compiled or _dsv41_yoco_need_eager(batch_req_state)" in mr
    att = t["models/deepseek_v4_1/attention.py"]
    assert att.count('getattr(self, "_dsv41_yoco_skip_kv", False)') == 2
    model = t["models/deepseek_v4_1/nvidia/model.py"]
    assert "dsv41_yoco_split: typing.Any = None" in model
    assert "_dsv41_yoco_plan = _dsv41_yoco_begin(self, positions)" in model
    assert "_dsv41_yoco_plan.finish(hidden_states, aux_hidden_states)" in model
    ms = t["models/deepseek_v4_1/nvidia/model_state.py"]
    assert "def prepare_attn(" in ms and "super().prepare_attn(" in ms


def test_drift_fails_closed_atomically():
    for rel, edits, _ in patch.JOBS:
        for old, _new in edits:
            for mutate in ("drop", "dup"):
                root = _tree()
                path = root / rel
                text = path.read_text()
                path.write_text(text.replace(old, "") if mutate == "drop" else text + "\n" + old)
                before = _snapshot(root)
                r = _run(root)
                assert r.returncode != 0, (rel, mutate, old[:50])
                assert rel in r.stderr, r.stderr
                assert _snapshot(root) == before, (rel, mutate, "a file was written")


def test_check_mode_writes_nothing():
    root = _tree()
    before = _snapshot(root)
    r = _run(root, "--check", mode=None)
    assert r.returncode == 0 and "check ok" in r.stdout, r
    assert _snapshot(root) == before
    path = root / TARGETS[0]
    path.write_text(path.read_text().replace(patch.JOBS[0][1][0][0], ""))
    assert _run(root, "--check", mode=None).returncode != 0


def test_off_is_a_noop():
    for mode in (None, "", "off"):
        root = _tree()
        before = _snapshot(root)
        r = _run(root, mode=mode)
        assert r.returncode == 0 and "skipped" in r.stdout, (mode, r)
        assert _snapshot(root) == before
    r = _run(_tree(), mode="mask")
    assert r.returncode != 0 and "DSV41_YOCO_PREFILL" in r.stderr


def test_real_image_tree_if_available():
    tree = os.environ.get("DSV41_YOCO_IMAGE_VLLM")
    if not tree:
        return
    r = _run(Path(tree), "--check", mode=None)
    assert r.returncode == 0, r.stderr


def test_recipe_wiring():
    start = (ROOT / "start.sh").read_text()
    # both ranks: mounted (head with host paths, worker with scp'd /tmp copies) ...
    for name in ("patch_yoco_prefill.py", "dsv41_yoco.py"):
        assert start.count(f"/opt/dsv41/{name}:ro") == 2, name
        assert f'"${{WORKER_SSH}}:/tmp/{name}"' in start, name
    # ... and run only when enabled, fatally on failure, after the other patches
    guard = 'if [ "${DSV41_YOCO_PREFILL:-off}" != "off" ]; then'
    assert start.count(guard) == 2
    for script in ("HEAD_SCRIPT", "WORKER_SCRIPT"):
        body = start.split(f'cat > "${script}" <<\'EOF\'\n', 1)[1].split("\nEOF", 1)[0]
        assert body.index("patch_memory_log.py; do") < body.index(guard) < body.index("exec vllm serve")
        tail = body[body.index(guard):]
        assert "python3 /opt/dsv41/patch_yoco_prefill.py ||" in tail.split("fi\n", 1)[0]
        assert "exit 1" in tail.split("fi\n", 1)[0]
    docker = (ROOT / "Dockerfile").read_text()
    assert "COPY overlay/patch_yoco_prefill.py /opt/dsv41/patch_yoco_prefill.py" in docker
    assert "COPY overlay/dsv41_yoco.py /opt/dsv41/dsv41_yoco.py" in docker
    assert "RUN python3 /opt/dsv41/patch_yoco_prefill.py --check" in docker


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"YOCO prefill patch OK ({len(tests)} tests)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
