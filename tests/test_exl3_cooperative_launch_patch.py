#!/usr/bin/env python3
"""Focused host checks for the exl3_moe cooperative-launch patch."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PATCH_PATH = next(
    path
    for path in (
        ROOT / "overlay" / "patch_exl3_cooperative_launch.py",
        HERE / "patch_exl3_cooperative_launch.py",
    )
    if path.is_file()
)
spec = importlib.util.spec_from_file_location("patch_exl3_cooperative_launch", PATCH_PATH)
assert spec and spec.loader
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)

FIXTURE = (
    "void exl3_moe(\n"
    "    const torch::Tensor& hidden_state,\n"
    "    int num_sms\n"
    ")\n"
    "{\n"
    "    dim3 grid_dim(1, 1, 1);\n" + patch.LAUNCH_ANCHOR + "\n"
    "    cuda_check(cudaPeekAtLastError());\n"
    "}\n"
)


def test_prepare_patches_once_then_noops() -> None:
    out, action = patch.prepare(FIXTURE)
    assert action == "patched"
    assert "cudaLaunchCooperativeKernel" in out
    assert patch.LAUNCH_ANCHOR not in out
    assert "cudaLaunchCooperativeKernel" in out and "TORCH_CHECK" in out
    assert "cudaErrorCooperativeLaunchTooLarge" in out
    again, action = patch.prepare(out)
    assert again == out
    assert action == "already present"


def test_drift_and_ambiguity_fail_closed() -> None:
    for source in (
        FIXTURE + FIXTURE,
        FIXTURE.replace(patch.LAUNCH_ANCHOR, ""),
        FIXTURE.replace("    stream\n", "    stream_renamed\n"),
        FIXTURE.replace("        grid_dim,\n", ""),
    ):
        try:
            patch.prepare(source)
        except ValueError:
            pass
        else:
            raise AssertionError("drifted or ambiguous source was accepted")


def test_missing_extension_root_fails() -> None:
    with tempfile.TemporaryDirectory() as raw:
        try:
            patch.target_of(Path(raw))
        except ValueError:
            pass
        else:
            raise AssertionError("accepted a tree without quant/exl3_moe.cu")


def _run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PATCH_PATH), str(root), *args],
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_check_then_apply_then_idempotent() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        (root / "quant").mkdir()
        target = root / "quant" / "exl3_moe.cu"
        target.write_text(FIXTURE)

        check = _run(root, "--check")
        assert check.returncode == 0, check.stderr
        assert target.read_text() == FIXTURE

        applied = _run(root)
        assert applied.returncode == 0, applied.stderr
        text = target.read_text()
        assert "cudaLaunchCooperativeKernel" in text
        assert patch.LAUNCH_ANCHOR not in text

        repeated = _run(root)
        assert repeated.returncode == 0, repeated.stderr
        assert "already present" in repeated.stdout
        assert target.read_text() == text


def test_cli_refuses_drifted_source() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        (root / "quant").mkdir()
        target = root / "quant" / "exl3_moe.cu"
        drifted = FIXTURE.replace("        block_dim,\n", "")
        target.write_text(drifted)
        result = _run(root)
        assert result.returncode != 0, result.stdout
        assert target.read_text() == drifted


def test_recipe_wiring() -> None:
    dockerfile = ROOT / "Dockerfile"
    if not dockerfile.is_file():
        return
    text = dockerfile.read_text()
    assert (
        "COPY overlay/patch_exl3_cooperative_launch.py "
        "/opt/dsv41/patch_exl3_cooperative_launch.py" in text
    )
    # Opt-in build toggle, OFF by default so the pinned recipe image is unchanged.
    assert "ARG EXLLAMAV3_MOE_COOP_LAUNCH=0" in text
    assert "patch_exl3_cooperative_launch.py" in text
    assert text.count("patch_exl3_cooperative_launch.py") >= 2
    # The build must never silently patch: the gate is the explicit build arg.
    assert "EXLLAMAV3_MOE_COOP_LAUNCH" in text


def test_live_source_if_enabled() -> None:
    live = __import__("os").environ.get("EXL3_MOE_COOP_LIVE_SOURCE")
    if not live:
        return
    source = Path(live).read_text()
    out, action = patch.prepare(source)
    assert action == "patched", action
    assert "cudaLaunchCooperativeKernel" in out


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"exl3_moe cooperative-launch patch OK ({len(tests)} tests)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
