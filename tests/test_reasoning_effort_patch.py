#!/usr/bin/env python3
"""Host checks for the DeepSeek V4.1 reasoning-effort mapping patch."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATCH_PATH = ROOT / "overlay" / "patch_reasoning_effort_mapping.py"
spec = importlib.util.spec_from_file_location("patch_reasoning_effort_mapping", PATCH_PATH)
assert spec and spec.loader
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)

# Trimmed copy of the image's deepseek_v41_encoding.py around the anchor
# (vllm/vllm-openai:deepseekv41-flash-0909).
FIXTURE = '''from typing import Dict, Union

REASONING_EFFORT_TEMPLATE = (
    "Reasoning Effort: {budget} "
    "(range 1-100, the higher the value, the more thorough the reasoning)\\n\\n"
)
REASONING_EFFORT_MAPPINGS: Dict[str, int] = {
    "low": 25,
    "high": 50,
    "xhigh": 75,
    "max": 100,
}

DEFAULT_REASONING_EFFORT = "high"


def render_reasoning_effort(index: int, thinking_mode: str, effort: Union[str, int, None]) -> str:
    if effort is None:
        effort = DEFAULT_REASONING_EFFORT
    assert (
        type(effort) is int and 1 <= effort <= 100
    ) or effort in REASONING_EFFORT_MAPPINGS
    if type(effort) is str:
        effort = REASONING_EFFORT_MAPPINGS[effort]
    if index == 0 and thinking_mode == "thinking":
        return REASONING_EFFORT_TEMPLATE.format(budget=effort)
    return ""
'''


def _render(source: str, effort: object) -> str:
    namespace: dict[str, object] = {}
    exec(compile(source, "fixture.py", "exec"), namespace)
    return namespace["render_reasoning_effort"](0, "thinking", effort)  # type: ignore[operator]


def test_official_mapping_and_idempotence() -> None:
    out, status = patch.patch_text(FIXTURE)
    assert status == "applied"
    expected = {"low": 50, "high": 75, "xhigh": 75, "max": 100, None: 75, 42: 42}
    for effort, budget in expected.items():
        assert _render(out, effort).startswith(f"Reasoning Effort: {budget} "), (effort, budget)
    again, repeated = patch.patch_text(out)
    assert again == out
    assert repeated == "already applied"


def test_drift_fails_closed() -> None:
    for source in (FIXTURE.replace(patch.OLD, ""), FIXTURE + FIXTURE):
        try:
            patch.patch_text(source)
        except RuntimeError:
            pass
        else:
            raise AssertionError("drifted source was accepted")


def test_recipe_wiring() -> None:
    start = (ROOT / "start.sh").read_text()
    assert str(PATCH_PATH.relative_to(ROOT)) in start
    assert start.count("/opt/dsv41/patch_reasoning_effort_mapping.py \\") == 2
    assert start.count("/opt/dsv41/patch_reasoning_effort_mapping.py:ro") == 2
    assert '"$EFFORT_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_reasoning_effort_mapping.py"' in start


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"Reasoning-effort mapping patch OK ({len(tests)} tests)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
