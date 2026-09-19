#!/usr/bin/env python3
"""Launch exl3_moe cooperatively, the way exl3_gemm already does.

Why
---
``exl3_moe_kernel.cuh`` synchronizes through five *device-global* group barriers
(``barrier_counters_sense`` = ``DevCtx::get_locks() + BARRIER_LOCKS_OFFSET``,
cudaMalloc'd and zeroed exactly **once** per process) plus a device-global ticket
scheduler at ``MOE_SCHED_OFFSET``. ``exl3_moe.cu`` states the requirement itself:

    // Launch. All blocks of the grid must be co-resident for the group
    // barriers, so groups * width <= num_sms.

The grid is sized to fill the device entirely, so that requirement has zero
headroom -- and the kernel is nevertheless launched with a plain
``cudaLaunchKernel`` (exl3_moe.cu:290), which enforces nothing. A block that
never becomes resident leaves the barrier spinning forever: no error is raised,
and the half-advanced arrivals stay in the device-global buffer for the rest of
the process. Observed on gb10 2026-09-19: the engine wedges permanently (front
end stays green, token counters freeze, restart is the only recovery), ~7 stalls
in one day.

The rest of the *same* extension gets this right: ``exl3_gemm.cu`` launches
through ``cudaLaunchCooperativeKernel`` (lines 296 and 619) and ships with
``#include <cooperative_groups.h>``, and the dense EXL3 GEMMs run inside vLLM's
captured decode graphs in this production stack -- so cooperative launch is
already proven capturable here. ``exl3_moe.cu`` also includes
``<cooperative_groups.h>``; only the launch call differs.

What the swap buys (CUDA Programming Guide, Cooperative Groups 4.4.8.1):
"cudaLaunchCooperativeKernel ensures that the kernel launch is atomic, i.e. if
the API call succeeds, then the provided number of thread blocks will launch on
the specified device." A grid that cannot be launched cooperatively returns
``cudaErrorCooperativeLaunchTooLarge`` instead of hanging, so the same condition
surfaces as an exception in vLLM rather than a silent deadlock.

Scope
-----
The launch only. The kernel's synchronization logic is untouched; replacing the
five hand-rolled barriers with cooperative groups' ``grid.sync()`` (what the
yeasah/exllamav3 fork does for ``exl3_gemm_kernel.cuh``) is a larger change that
needs a control-flow audit -- ``docs/exl3-moe-barrier.md`` lists the five sites.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

LAUNCH_ANCHOR = (
    "    cudaLaunchKernel\n"
    "    (\n"
    "        (void*) kernel,\n"
    "        grid_dim,\n"
    "        block_dim,\n"
    "        kernelArgs,\n"
    "        SMEM_MAX,\n"
    "        stream\n"
    "    );\n"
)

LAUNCH_REPLACEMENT = (
    "    // [dsv41] Cooperative launch. exl3_moe_kernel.cuh synchronizes through\n"
    "    // device-global group barriers that require every block of the grid to be\n"
    "    // co-resident (see the comment above), and a plain cudaLaunchKernel\n"
    "    // enforces nothing: a block that never becomes resident spins the barrier\n"
    "    // forever with no error, wedging the engine for the life of the process.\n"
    "    // Cooperative launch is atomic -- on success all blocks launch -- and a\n"
    "    // grid that cannot be launched cooperatively returns\n"
    "    // cudaErrorCooperativeLaunchTooLarge instead of hanging. exl3_gemm.cu\n"
    "    // already launches this way (296/619) and runs inside vLLM's captured\n"
    "    // decode graphs, so the mechanism is proven in this stack.\n"
    "    cudaError_t coop_err = cudaLaunchCooperativeKernel\n"
    "    (\n"
    "        (void*) kernel,\n"
    "        grid_dim,\n"
    "        block_dim,\n"
    "        kernelArgs,\n"
    "        SMEM_MAX,\n"
    "        stream\n"
    "    );\n"
    "    TORCH_CHECK\n"
    "    (\n"
    "        coop_err == cudaSuccess,\n"
    "        \"exl3_moe cooperative launch failed (\",\n"
    "        grid_dim.x * grid_dim.y * grid_dim.z,\n"
    "        \" blocks, \", num_sms, \" SMs): \",\n"
    "        cudaGetErrorString(coop_err)\n"
    "    );\n"
)


def prepare(text: str) -> tuple[str, str]:
    """Return (patched_text, action). Fails closed on drifted or ambiguous source."""
    if "cudaLaunchCooperativeKernel" in text and LAUNCH_ANCHOR not in text:
        return text, "already present"
    if text.count(LAUNCH_ANCHOR) != 1:
        raise ValueError(
            "exl3_moe launch anchor must appear exactly once "
            f"(found {text.count(LAUNCH_ANCHOR)}); upstream source drifted"
        )
    return text.replace(LAUNCH_ANCHOR, LAUNCH_REPLACEMENT, 1), "patched"


def target_of(ext_root: Path) -> Path:
    target = ext_root / "quant" / "exl3_moe.cu"
    if not target.is_file():
        raise ValueError(f"not an exllamav3 extension root (no {target})")
    return target


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ext_root", type=Path, help="path to .../exllamav3_ext")
    ap.add_argument(
        "--check",
        action="store_true",
        help="verify the patch applies and write nothing",
    )
    args = ap.parse_args()

    target = target_of(args.ext_root.resolve())
    text = target.read_text()
    patched, action = prepare(text)

    if args.check:
        print(f"{target.name}: {action} (check only, nothing written)")
        return 0
    if action == "already present":
        print(f"{target.name}: already present")
        return 0

    target.write_text(patched)
    print(f"{target.name}: patched (exl3_moe now launched via cudaLaunchCooperativeKernel)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        raise SystemExit(f"patch refused: {exc}")
