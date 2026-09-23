#!/usr/bin/env python3
"""CPU-only coverage of retention validation, rank argv, and Docker propagation."""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / 'start.sh').read_text()
KEY = 'PREFIX_CACHE_RETENTION_INTERVAL'
FLAG = '--prefix-cache-retention-interval'


def clean_env():
    env = {k: v for k, v in os.environ.items() if k not in (KEY, 'EXTRA_ARGS')}
    env['LC_ALL'] = 'C'
    return env


def validate(value=None, extra=''):
    begin = SOURCE.index('# GLM53 numeric config guard (begin)')
    end = SOURCE.index('# GLM53 numeric config guard (end)', begin)
    env = clean_env()
    if value is not None:
        env[KEY] = value
    env['EXTRA_ARGS'] = extra
    script = SOURCE[begin:end] + '''
GPU_MEM_UTIL=0.88; MAX_MODEL_LEN=600000; MAX_NUM_SEQS=2
MAX_NUM_BATCHED_TOKENS=1536; GLM53_SPINWAIT_MS=stock
validate_numeric_config || exit $?
printf '%s' "$PREFIX_CACHE_RETENTION_INTERVAL"
'''
    return subprocess.run(['bash', '-c', script], env=env, text=True, capture_output=True)


def test_values():
    for raw, expected in [(None, '4096'), ('4096', '4096'), ('0004096', '4096'),
                          ('0', '0'), ('000', '0'), ('128', '128'), ('1048576', '1048576')]:
        result = validate(raw)
        assert result.returncode == 0 and result.stdout == expected, (raw, result)
    for raw in ['', ' ', '4096 ', ' 4096', '4096\n', '4096\r', '-128', '+128',
                '0.0', 'nan', 'None', '1', '64', '129', '1048704', '9' * 80]:
        result = validate(raw)
        assert result.returncode == 2 and KEY in result.stderr, (raw, result)
    for extra in [f'{FLAG} 4096', f'{FLAG}=4096', f'--foo bar\t{FLAG}\t4096']:
        result = validate('4096', extra)
        assert result.returncode == 2 and 'EXTRA_ARGS' in result.stderr, result
    assert validate('4096', '--cudagraph-capture-sizes 1 2').returncode == 0


def test_defaults_and_propagation():
    assignment = f'{KEY}="${{{KEY}-4096}}"'
    assert SOURCE.count(assignment) == 1
    for raw, expected in [(None, '4096'), ('0', '0'), ('8192', '8192')]:
        env = clean_env()
        if raw is not None:
            env[KEY] = raw
        result = subprocess.check_output(['bash', '-c', assignment + f'; printf "%s" "${KEY}"'], env=env, text=True)
        assert result == expected
    lines = (ROOT / '.env.example').read_text().splitlines()
    assert [s for s in lines if s.startswith(KEY + '=')] == [KEY + '=4096']
    assert SOURCE.count(f'-e {KEY}="${KEY}"') == 1, 'head Docker environment'
    worker_loop = SOURCE.split('for v in SERVED_MODEL_NAME', 1)[1].split('; do', 1)[0]
    assert worker_loop.split().count(KEY) == 1, 'worker Docker environment'
    main = SOURCE.split('main() {', 1)[1]
    assert main.index('start|restart) validate_numeric_config') < main.index('restart)  stop; start')


def test_both_rank_argv():
    # Evaluate the actual base argv from each generated rank script without
    # running imports, Docker, SSH, patches, model loading, or GPU operations.
    for rank, name in [(0, 'HEAD_SCRIPT'), (1, 'WORKER_SCRIPT')]:
        script = SOURCE.split(f'cat > "${name}" <<\'EOF\'\n', 1)[1].split('\nEOF', 1)[0]
        argv = re.search(r'^ARGS=\(\n.*?^\)', script, re.M | re.S)
        assert argv is not None, name
        for value in ['0', '4096', '8192']:
            env = clean_env()
            env.update({KEY: value, 'SERVED_MODEL_NAME': 'test', 'PORT': '8888', 'TP': '2',
                        'NNODES': '2', 'HEAD_IP': '127.0.0.1', 'MASTER_PORT': '29521',
                        'ENGRAM_MOUNT': '/unused'})
            result = subprocess.check_output(['bash', '-c', 'set -eu\n' + argv.group() + '\nprintf "%s\\0" "${ARGS[@]}"'], env=env)
            args = result.decode().rstrip('\0').split('\0')
            assert args.count(FLAG) == 1, (name, args)
            assert args[args.index(FLAG) + 1] == value
            assert args[args.index('--node-rank') + 1] == str(rank)


if __name__ == '__main__':
    test_values()
    test_defaults_and_propagation()
    test_both_rank_argv()
    print('prefix-cache retention: PASS')
