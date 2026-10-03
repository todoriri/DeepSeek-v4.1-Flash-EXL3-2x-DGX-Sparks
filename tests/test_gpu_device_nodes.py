#!/usr/bin/env python3
"""CPU-only coverage of the explicit NVIDIA --device nodes on every GPU container.

2026-10-03 EngineDead (Triton load_binary 'operation not permitted', rank0 only): with
`--gpus all` on the legacy (non-CDI) hook the nvidia char devices (195 = nvidia,
498 = nvidia-uvm) are allowed only in the BPF program the hook wrote, not in the
systemd scope's DeviceAllow. Any `systemctl daemon-reload` (snapd runs them) rebuilds
the BPF from DeviceAllow and every new open() of /dev/nvidia* fails EPERM. Reproduced
on gb10 (load_binary failed at load #17 after the reload); explicit --device nodes put
195:0/255 + 498:0/1 into the scope and survive the reload (20000 loads OK). Every
`docker run --gpus all` must therefore also pass the nodes."""
from __future__ import annotations

from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / 'start.sh').read_text()
NODES = ('/dev/nvidia0', '/dev/nvidiactl', '/dev/nvidia-uvm', '/dev/nvidia-uvm-tools')


def gpu_runs():
    """Each `docker run ... --gpus all` command, from `docker run` to the image arg."""
    runs = []
    for m in re.finditer(r'docker run ', SOURCE):
        end = SOURCE.find('--entrypoint', m.start())
        cmd = SOURCE[m.start():end]
        if '--gpus all' in cmd:
            runs.append(cmd)
    return runs


def test_every_gpu_run_passes_device_nodes():
    runs = gpu_runs()
    assert len(runs) == 3, 'overlay self-check + worker + head'
    for cmd in runs:
        assert '${GPU_DEVICE_ARGS}' in cmd, cmd[:120]


def test_device_args_render_all_nodes():
    line = next(l for l in SOURCE.splitlines() if l.startswith('GPU_DEVICE_ARGS='))
    out = subprocess.check_output(['bash', '-c', line + '\nprintf "%s" "$GPU_DEVICE_ARGS"'],
                                  env={'PATH': '/usr/bin:/bin'}, text=True)
    assert out.split() == [w for n in NODES for w in ('--device', n)], out
