#!/usr/bin/env python3
"""Host checks for overlay/patch_swa_final_window.py (final SWA window retention).

The patch is applied to unpatched copies of the image's files (tests/fixtures/dsv41_yoco,
shared with the YOCO overlay's tests). Coverage:

  * every anchor applies exactly once and the patched files still compile;
  * a second run is a no-op;
  * anchor drift (missing or duplicated) fails loudly and writes nothing, in any file;
  * --check verifies without writing;
  * the patch is a no-op unless DSV41_SWA_FINAL_WINDOW=keep;
  * it coexists with patch_yoco_prefill.py in either order: neither touches the other's
    anchors, each one's anchors still match exactly once after the other ran;
  * the pure math in the footer (hold, checkpoint boundaries), checked against brute force;
  * start.sh and the Dockerfile wire it like the other overlays, both ranks, before YOCO.

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
PATCH_PATH = ROOT / "overlay" / "patch_swa_final_window.py"
YOCO_PATH = ROOT / "overlay" / "patch_yoco_prefill.py"
FIXTURES = ROOT / "tests" / "fixtures" / "dsv41_yoco" / "vllm"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


patch = _module(PATCH_PATH, "patch_swa_final_window")
yoco = _module(YOCO_PATH, "patch_yoco_prefill_for_final_window")

TARGETS = [
    "v1/core/single_type_kv_cache_manager.py",
    "v1/core/kv_cache_manager.py",
]
_TMP = tempfile.TemporaryDirectory(prefix="swa-final-patch-")


def _tree() -> Path:
    dst = Path(tempfile.mkdtemp(dir=_TMP.name)) / "vllm"
    shutil.copytree(FIXTURES, dst)
    return dst


def _run(root: Path, *args: str, mode: str | None = "keep", script: Path = PATCH_PATH):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("DSV41_SWA_FINAL", "DSV41_YOCO"))}
    if mode is not None:
        env["DSV41_SWA_FINAL_WINDOW"] = mode
    env["DSV41_YOCO_PREFILL"] = "compact"          # only read by the YOCO patch
    env["DSV41_YOCO_HELPER"] = str(ROOT / "overlay" / "dsv41_yoco.py")
    return subprocess.run([sys.executable, str(script), "--root", str(root), *args],
                          env=env, text=True, capture_output=True)


def _snapshot(root: Path) -> dict[str, str]:
    files = set(TARGETS) | {rel for rel, _, _ in yoco.JOBS}
    return {rel: (root / rel).read_text() for rel in sorted(files)}


def test_jobs_cover_targets():
    assert sorted(rel for rel, _, _ in patch.JOBS) == sorted(TARGETS)
    for rel, edits, footer in patch.JOBS:
        assert edits and footer, rel
        for old, new in edits:
            assert patch.MARK in new, (rel, old[:40])
            assert new.startswith(old) or new.endswith(old) or old in new, (rel, old[:40])
    assert "dsv41-yoco-prefill" not in PATCH_PATH.read_text(), "would read as YOCO-applied"


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
    for rel in set(after) - set(TARGETS):
        assert after[rel] == before[rel], rel
    r2 = _run(root)
    assert r2.returncode == 0 and "already applied" in r2.stdout, r2
    assert _snapshot(root) == after


def test_patched_semantics_spotcheck():
    root = _tree()
    assert _run(root).returncode == 0
    t = _snapshot(root)
    stm = t["v1/core/single_type_kv_cache_manager.py"]
    assert "self.extra_retained_tokens = kv_cache_spec.extra_retained_tokens\n" in stm
    assert "self.extra_retained_tokens += _dsv41_hold" in stm
    assert "self._max_admission_blocks_per_request += cdiv(_dsv41_hold, self.block_size)" in stm
    assert stm.count("_dsv41_swa_final_require_enabled()\n") == 1
    kvm = t["v1/core/kv_cache_manager.py"]
    for fn in ("def free(self, request: Request) -> None:",
               "def pop_blocks_for_free(self, request: Request) -> list[KVCacheBlock]:"):
        body = kvm.split(fn, 1)[1].split("\n    def ", 1)[0]
        assert body.count("_dsv41_swa_final_window(self, request)") == 1, fn
        assert body.index("_dsv41_swa_final_window(self, request)") < body.index(
            "self.coordinator."), fn
    # stock hit cap and the replay boundary are untouched (YOCO's anchors)
    assert "max_cache_hit_length = request.num_tokens - 1\n" in kvm
    assert "reachable_boundaries = [request.num_prompt_tokens - 1]\n" in stm


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
    for bad in ("on", "KEEP", "1"):
        r = _run(_tree(), mode=bad)
        assert r.returncode == 2 and "DSV41_SWA_FINAL_WINDOW" in r.stderr, (bad, r)


def test_coexists_with_yoco_in_either_order():
    for order in ((PATCH_PATH, YOCO_PATH), (YOCO_PATH, PATCH_PATH)):
        root = _tree()
        for script in order:
            r = _run(root, script=script)
            assert r.returncode == 0 and "applied" in r.stdout, (script.name, r.stderr)
        both = _snapshot(root)
        for rel in both:
            py_compile.compile(str(root / rel), doraise=True)
        for rel in TARGETS:
            assert f"[{patch.MARK}]" in both[rel] and f"[{yoco.MARK}]" in both[rel], rel
        # the other's anchors are intact in a fresh tree patched by one of them only
    for first, other in ((patch, yoco), (yoco, patch)):
        root = _tree()
        r = _run(root, script=PATCH_PATH if first is patch else YOCO_PATH)
        assert r.returncode == 0, r.stderr
        for rel, edits, _ in other.JOBS:
            text = (root / rel).read_text()
            for old, _new in edits:
                assert text.count(old) == 1, (first.MARK, rel, old[:60])


def _footer_ns() -> dict:
    ns: dict = {}
    saved = os.environ.get("DSV41_SWA_FINAL_WINDOW")
    os.environ["DSV41_SWA_FINAL_WINDOW"] = "keep"   # the footer refuses to load otherwise
    try:
        exec(compile(patch.STM_FOOTER, "STM_FOOTER", "exec"), ns)
    finally:
        if saved is None:
            os.environ.pop("DSV41_SWA_FINAL_WINDOW")
        else:
            os.environ["DSV41_SWA_FINAL_WINDOW"] = saved
    return ns


def test_footer_math_against_brute_force():
    ns = _footer_ns()
    hold_tokens = ns["_dsv41_swa_final_hold_tokens"]
    boundaries = ns["_dsv41_swa_final_boundaries"]
    W, bs = 128, 64
    for align in (64, 128, 256):
        for peek in (0, 1):
            need = -(-(W - 1) // bs) + peek
            for keep in (0, 140, 333):
                hold = hold_tokens(W, bs, align, peek, keep)
                worst = 0
                for C in range(0, 3000):
                    for T in (C, C + 1, C + 4, C + 1000):   # C + 1000: aborted mid-prefill
                        hits = boundaries(C, T, bs, align, bool(peek), keep)
                        full = C // bs * bs
                        want = {(full - peek * bs) // align * align}
                        if keep:
                            want.add((T - keep - peek * bs) // align * align)
                        want = {h for h in want if h >= align and h + peek * bs <= full}
                        assert set(hits) == want, (align, peek, keep, C, T, hits, want)
                        for h in hits:
                            assert h % align == 0
                            lowest = max(0, (h // bs + peek - need) * bs)
                            # the stock free keeps blocks from floor((C - W + 1 - e) / bs)
                            freed = max(0, C - W + 1 - hold) // bs * bs
                            assert lowest >= freed, (align, peek, keep, C, T, h, hold)
                            worst = max(worst, C - W + 1 - lowest)
                # not wasteful: a block less would lose some tail
                assert hold - bs < worst, (align, peek, keep, hold, worst)
    assert hold_tokens(128, 64, 128, 1, 0) == 192
    assert hold_tokens(128, 64, 128, 1, 140) == 332


def test_real_image_tree_if_available():
    tree = os.environ.get("DSV41_YOCO_IMAGE_VLLM")
    if not tree:
        return
    r = _run(Path(tree), "--check", mode=None)
    assert r.returncode == 0, r.stderr


def test_recipe_wiring():
    start = (ROOT / "start.sh").read_text()
    name = "patch_swa_final_window.py"
    assert start.count(f"/opt/dsv41/{name}:ro") == 2, name
    assert f'"${{WORKER_SSH}}:/tmp/{name}"' in start, name
    assert '[ -f "$SWA_FINAL_PATCH_HOST" ] || die' in start
    guard = 'if [ "${DSV41_SWA_FINAL_WINDOW:-off}" != "off" ]; then'
    yoco_guard = 'if [ "${DSV41_YOCO_PREFILL:-off}" != "off" ]; then'
    assert start.count(guard) == 2
    for script in ("HEAD_SCRIPT", "WORKER_SCRIPT"):
        body = start.split(f'cat > "${script}" <<\'EOF\'\n', 1)[1].split("\nEOF", 1)[0]
        assert (body.index("patch_memory_log.py; do") < body.index(guard)
                < body.index(yoco_guard) < body.index("exec vllm serve"))
        tail = body[body.index(guard):].split("fi\n", 1)[0]
        assert f"python3 /opt/dsv41/{name} ||" in tail and "exit 1" in tail
    docker = (ROOT / "Dockerfile").read_text()
    assert f"COPY overlay/{name} /opt/dsv41/{name}" in docker
    assert f"RUN python3 /opt/dsv41/{name} --check" in docker


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"SWA final-window patch OK ({len(tests)} tests)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
