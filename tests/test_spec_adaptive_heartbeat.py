#!/usr/bin/env python3
"""CPU-only coverage of DSPARK_ADAPTIVE (DSpark enable_adaptive_verification) and
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: validation, the rendered --speculative-config on
both ranks, and propagation to both containers. Unset = today's launch, byte for byte."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / 'start.sh').read_text()
KEYS = ('DSPARK_ADAPTIVE', 'TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC', 'SPEC_METHOD',
        'ENFORCE_EAGER', 'DSPARK_TOKENS', 'EXTRA_ARGS')


def clean_env(**over):
    env = {k: v for k, v in os.environ.items() if k not in KEYS}
    env['LC_ALL'] = 'C'
    env.update(over)
    return env


def validate(**over):
    begin = SOURCE.index('# GLM53 numeric config guard (begin)')
    end = SOURCE.index('# GLM53 numeric config guard (end)', begin)
    script = SOURCE[begin:end] + '''
GPU_MEM_UTIL=0.88; MAX_MODEL_LEN=600000; MAX_NUM_SEQS=2
MAX_NUM_BATCHED_TOKENS=1536; GLM53_SPINWAIT_MS=stock
validate_numeric_config || exit $?
printf '%s|%s' "${DSPARK_ADAPTIVE-}" "${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC-}"
'''
    return subprocess.run(['bash', '-c', script], env=clean_env(**over), text=True,
                          capture_output=True)


SPEC_RE = re.compile(r"ARGS\+=\(--speculative-config \"\$\(python3 -S -c '(.*?)'\)\"\)", re.S)


def spec_snippets():
    return SPEC_RE.findall(SOURCE)


def render(**env):
    (snippet,) = set(spec_snippets())
    return subprocess.check_output(['python3', '-S', '-c', snippet], env=clean_env(**env),
                                   text=True).strip()


def test_validation():
    assert validate().returncode == 0, validate()
    for v in ('0', '1'):
        r = validate(DSPARK_ADAPTIVE=v)
        assert r.returncode == 0 and r.stdout.startswith(v + '|'), (v, r)
    for bad in ('2', 'yes', 'true', ' 1', '01'):
        r = validate(DSPARK_ADAPTIVE=bad)
        assert r.returncode == 2 and 'DSPARK_ADAPTIVE' in r.stderr, (bad, r)
    r = validate(DSPARK_ADAPTIVE='1', SPEC_METHOD='none')
    assert r.returncode == 2 and 'dspark' in r.stderr, r
    r = validate(DSPARK_ADAPTIVE='1', ENFORCE_EAGER='1')
    assert r.returncode == 2 and 'ENFORCE_EAGER' in r.stderr, r
    for raw, canon in (('300', '300'), ('0300', '300'), ('86400', '86400')):
        r = validate(TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=raw)
        assert r.returncode == 0 and r.stdout.endswith('|' + canon), (raw, r)
    for bad in ('0', '-5', 'abc', '3.5', '86401'):
        r = validate(TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=bad)
        assert r.returncode == 2 and 'TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC' in r.stderr, (bad, r)


def test_speculative_config_render():
    snippets = spec_snippets()
    assert len(snippets) == 2 and len(set(snippets)) == 1, 'head and worker render the same config'
    # unset / 0 = today's JSON, byte for byte
    assert render(DSPARK_TOKENS='3') == '{"method":"dspark","num_speculative_tokens":3}'
    assert render(DSPARK_TOKENS='3', DSPARK_ADAPTIVE='0') == '{"method":"dspark","num_speculative_tokens":3}'
    on = json.loads(render(DSPARK_TOKENS='5', DSPARK_ADAPTIVE='1'))
    assert on == {"method": "dspark", "num_speculative_tokens": 5,
                  "enable_adaptive_verification": True}, on


def test_propagation():
    assert len(re.findall(r'^DSPARK_ADAPTIVE="\$\{DSPARK_ADAPTIVE:-0\}"$', SOURCE, re.M)) == 1, \
        'one default'
    assert '-e DSPARK_ADAPTIVE="${DSPARK_ADAPTIVE:-0}"' in SOURCE, 'head container'
    loop = SOURCE[SOURCE.index('local serve_env=""'):SOURCE.index('serve_env+=" -e VLLM_API_KEY')]
    assert re.search(r'\bDSPARK_ADAPTIVE\b', loop), 'worker container (serve_env)'
    # heartbeat: both ranks via nccl_common, and only when set
    block = SOURCE[SOURCE.index('local -a nccl_common=('):SOURCE.index('local worker_nccl=""')]
    assert 'TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC' in block
    probe = block + '\nfor e in "${nccl_common[@]}"; do printf "%s\\n" "$e"; done\n'
    probe = 'f(){\n' + probe + '\n}\nf\n'
    env = {'NCCL_CROSS_NIC': '0', 'NCCL_DEBUG': 'WARN', 'NCCL_BUFFSIZE': '1',
           'NCCL_LL128_BUFFSIZE': '1', 'NCCL_PROTO': 'x', 'NCCL_MAX_NCHANNELS': '8'}
    unset = subprocess.check_output(['bash', '-c', probe], env=clean_env(**env), text=True)
    assert 'TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC' not in unset, 'unset = not passed (torch default)'
    set_ = subprocess.check_output(['bash', '-c', probe],
                                   env=clean_env(TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC='300', **env),
                                   text=True)
    assert 'TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300' in set_.splitlines()


if __name__ == '__main__':
    test_validation()
    test_speculative_config_render()
    test_propagation()
    print('ok')
