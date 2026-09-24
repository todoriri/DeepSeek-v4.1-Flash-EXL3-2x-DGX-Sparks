#!/usr/bin/env python3
"""CPU-only coverage of TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC (validation + both ranks, passed
only when set) and of the adaptive-verification refusal: this image's
DeepseekV4IndexerBackend rejects enable_adaptive_verification at KV init (2026-09-24
trial), so the launcher refuses it up front. The rendered --speculative-config is
today's, byte for byte."""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / 'start.sh').read_text()
KEYS = ('TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC', 'DSPARK_TOKENS', 'EXTRA_ARGS')


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
printf '%s' "${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC-}"
'''
    return subprocess.run(['bash', '-c', script], env=clean_env(**over), text=True,
                          capture_output=True)


SPEC_RE = re.compile(r"ARGS\+=\(--speculative-config \"\$\(python3 -S -c '(.*?)'\)\"\)", re.S)


def test_heartbeat_validation():
    r = validate()
    assert r.returncode == 0 and r.stdout == '', r
    for raw, canon in (('300', '300'), ('0300', '300'), ('86400', '86400')):
        r = validate(TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=raw)
        assert r.returncode == 0 and r.stdout == canon, (raw, r)
    for bad in ('0', '-5', 'abc', '3.5', '86401'):
        r = validate(TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=bad)
        assert r.returncode == 2 and 'TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC' in r.stderr, (bad, r)


def test_adaptive_refused():
    extra = '--speculative-config {"method":"dspark","enable_adaptive_verification":true}'
    r = validate(EXTRA_ARGS=extra)
    assert r.returncode == 2 and 'DeepseekV4IndexerBackend' in r.stderr, r
    assert validate(EXTRA_ARGS='--served-model-name a b').returncode == 0


def test_speculative_config_unchanged():
    snippets = SPEC_RE.findall(SOURCE)
    assert len(snippets) == 2 and len(set(snippets)) == 1, 'head and worker render the same config'
    for k in ('3', '5'):
        out = subprocess.check_output(['python3', '-S', '-c', snippets[0]],
                                      env=clean_env(DSPARK_TOKENS=k), text=True).strip()
        assert out == '{"method":"dspark","num_speculative_tokens":%s}' % k, out
    assert 'DSPARK_ADAPTIVE' not in SOURCE.replace('# DSpark adaptive', '')


def test_heartbeat_propagation():
    block = SOURCE[SOURCE.index('local -a nccl_common=('):SOURCE.index('local worker_nccl=""')]
    probe = 'f(){\n' + block + '\nfor e in "${nccl_common[@]}"; do printf "%s\\n" "$e"; done\n}\nf\n'
    env = {'NCCL_CROSS_NIC': '0', 'NCCL_DEBUG': 'WARN', 'NCCL_BUFFSIZE': '1',
           'NCCL_LL128_BUFFSIZE': '1', 'NCCL_PROTO': 'x', 'NCCL_MAX_NCHANNELS': '8'}
    unset = subprocess.check_output(['bash', '-c', probe], env=clean_env(**env), text=True)
    assert 'TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC' not in unset, 'unset = not passed (torch default)'
    set_ = subprocess.check_output(['bash', '-c', probe],
                                   env=clean_env(TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC='300', **env),
                                   text=True)
    assert 'TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300' in set_.splitlines()


if __name__ == '__main__':
    test_heartbeat_validation()
    test_adaptive_refused()
    test_speculative_config_unchanged()
    test_heartbeat_propagation()
    print('ok')
