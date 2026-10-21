#!/usr/bin/env python3
"""CPU-only coverage of DSV41_SWA_FINAL_WINDOW in start.sh.

  * validation: off|keep (unset and empty = off), anything else refused before a restart;
  * unset = today's launch: both ranks' vllm argv are byte-identical with the feature
    unset, off or keep, alone or with DSV41_YOCO_PREFILL=compact (it adds no vLLM flag);
  * the inner scripts run the patch only when it is not off, fatally on failure, and
    before the YOCO patch (which runs after every other patch);
  * both containers receive the variable (the TP ranks must agree).
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / 'start.sh').read_text()
KEY = 'DSV41_SWA_FINAL_WINDOW'
KEYS = (KEY, 'DSV41_YOCO_PREFILL', 'DSV41_YOCO_KEEP_TOKENS', 'DSPARK_TOKENS', 'SPEC_METHOD',
        'LANGUAGE_MODEL_ONLY', 'EXTRA_ARGS')
GUARD = 'if [ "${DSV41_SWA_FINAL_WINDOW:-off}" != "off" ]; then'


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
printf '%s' "$DSV41_SWA_FINAL_WINDOW"
'''
    over = {'LANGUAGE_MODEL_ONLY': '1', **over}
    return subprocess.run(['bash', '-c', script], env=clean_env(**over), text=True,
                          capture_output=True)


def test_values():
    for over, want in (({}, 'off'), ({KEY: ''}, 'off'), ({KEY: 'off'}, 'off'),
                       ({KEY: 'keep'}, 'keep'),
                       ({KEY: 'keep', 'DSV41_YOCO_PREFILL': 'compact'}, 'keep'),
                       ({KEY: 'keep', 'SPEC_METHOD': 'none'}, 'keep')):
        r = validate(**over)
        assert r.returncode == 0 and r.stdout == want, (over, r)
    for bad in ('on', 'KEEP', 'keep ', '1', 'true', 'dense'):
        r = validate(**{KEY: bad})
        assert r.returncode == 2 and KEY in r.stderr, (bad, r)
    main = SOURCE.split('main() {', 1)[1]
    assert main.index('start|restart) validate_numeric_config') < main.index('restart)  stop; start')


def _rank_argv(name, **over):
    script = SOURCE.split(f'cat > "${name}" <<\'EOF\'\n', 1)[1].split('\nEOF', 1)[0]
    block = re.search(r'^ARGS=\(\n.*?^\)$.*?(?=^\[ -f "\$\{MODEL_DIR\}/config.json" \])', script,
                      re.M | re.S)
    assert block is not None, name
    env = clean_env(SERVED_MODEL_NAME='m', PORT='8000', TP='2', NNODES='2', HEAD_IP='10.0.0.1',
                    MASTER_PORT='29521', ENGRAM_MOUNT='/e', PREFIX_CACHE_RETENTION_INTERVAL='4096',
                    MODEL_DIR='/model', DSPARK_TOKENS='3', ENFORCE_EAGER='0',
                    LANGUAGE_MODEL_ONLY='1', **over)
    out = subprocess.check_output(['bash', '-c', 'set -eu\nsay() { :; }\n' + block.group()
                                   + '\nprintf "%s\\0" "${ARGS[@]}"'], env=env)
    return out.decode().rstrip('\0').split('\0')


def test_argv_byte_identical():
    for name in ('HEAD_SCRIPT', 'WORKER_SCRIPT'):
        base = _rank_argv(name)
        assert '--prefix-cache-retention-interval' in base and '--enable-prefix-caching' in base
        for over in ({KEY: 'off'}, {KEY: 'keep'},
                     {KEY: 'keep', 'DSV41_YOCO_PREFILL': 'compact'},
                     {KEY: 'off', 'DSV41_YOCO_PREFILL': 'compact', 'DSV41_YOCO_KEEP_TOKENS': '256'}):
            assert _rank_argv(name, **over) == base, (name, over)


def test_inner_guard_only_when_enabled_and_before_yoco():
    for name in ('HEAD_SCRIPT', 'WORKER_SCRIPT'):
        script = SOURCE.split(f'cat > "${name}" <<\'EOF\'\n', 1)[1].split('\nEOF', 1)[0]
        start = script.index(GUARD)
        yoco = script.index('if [ "${DSV41_YOCO_PREFILL:-off}" != "off" ]; then')
        assert script.index('patch_memory_log.py; do') < start < yoco
        # from our guard through the YOCO guard: both patches, in order
        both = script[start:script.index('\nfi\n', yoco) + 4]
        probe = 'say() { echo "$*"; }\npython3() { echo "RAN $*"; [ "$1" != "$FAIL" ]; }\n' + both
        cases = [('', '', False, 0), ('off', '', False, 0), ('keep', '', True, 0),
                 ('keep', '/opt/dsv41/patch_swa_final_window.py', True, 1)]
        for mode, fail, want_ran, want_rc in cases:
            for yoco_mode in ('off', 'compact'):
                env = clean_env(FAIL=fail, DSV41_YOCO_PREFILL=yoco_mode)
                if mode:
                    env[KEY] = mode
                r = subprocess.run(['bash', '-c', probe], env=env, text=True, capture_output=True)
                ran = [ln.split()[1] for ln in r.stdout.splitlines() if ln.startswith('RAN ')]
                assert ('/opt/dsv41/patch_swa_final_window.py' in ran) == want_ran, (name, mode, r)
                assert r.returncode == want_rc, (name, mode, fail, r)
                if want_rc == 0 and yoco_mode == 'compact':
                    assert ran[-1] == '/opt/dsv41/patch_yoco_prefill.py', (name, ran)
                if want_rc:
                    assert '/opt/dsv41/patch_yoco_prefill.py' not in ran, (name, ran)


def test_propagation_both_ranks():
    assert SOURCE.count(f'{KEY}="${{{KEY}-off}}"') == 1
    assert SOURCE.count(f'-e {KEY}="${KEY}"') == 1, 'head container'
    loop = SOURCE.split('for v in SERVED_MODEL_NAME', 1)[1].split('; do', 1)[0]
    assert loop.split().count(KEY) == 1, 'worker container (serve_env)'
    assert SOURCE.count('SWA_FINAL_PATCH_HOST="${SWA_FINAL_PATCH_HOST:-$SCRIPT_DIR/overlay/'
                        'patch_swa_final_window.py}"') == 1
    assert SOURCE.count('-v "$SWA_FINAL_PATCH_HOST:/opt/dsv41/patch_swa_final_window.py:ro"') == 1
    assert SOURCE.count("-v '/tmp/patch_swa_final_window.py:/opt/dsv41/patch_swa_final_window.py:ro'") == 1
    example = (ROOT / '.env.example').read_text().splitlines()
    assert f'# {KEY}=off' in example


if __name__ == '__main__':
    test_values()
    test_argv_byte_identical()
    test_inner_guard_only_when_enabled_and_before_yoco()
    test_propagation_both_ranks()
    print('swa final-window launcher: PASS')
