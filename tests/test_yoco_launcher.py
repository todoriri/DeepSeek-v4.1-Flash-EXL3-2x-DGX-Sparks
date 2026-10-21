#!/usr/bin/env python3
"""CPU-only coverage of DSV41_YOCO_PREFILL / DSV41_YOCO_KEEP_TOKENS in start.sh.

  * validation: off|compact, and a positive KEEP; compact needs
    KEEP >= sliding window (128) + max speculative backtrack (DSPARK_TOKENS + 1);
    vision on (LANGUAGE_MODEL_ONLY=0) is refused;
  * unset = today's launch: both ranks' vllm argv are byte-identical with the
    feature unset, off or compact (it adds no vLLM flag), and the inner scripts
    only run the patch when it is not off;
  * both containers receive both variables (TP ranks must agree).
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / 'start.sh').read_text()
KEYS = ('DSV41_YOCO_PREFILL', 'DSV41_YOCO_KEEP_TOKENS', 'DSPARK_TOKENS', 'SPEC_METHOD',
        'LANGUAGE_MODEL_ONLY', 'EXTRA_ARGS')


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
printf '%s|%s' "$DSV41_YOCO_PREFILL" "$DSV41_YOCO_KEEP_TOKENS"
'''
    over = {'LANGUAGE_MODEL_ONLY': '1', **over}   # live config: vision off
    return subprocess.run(['bash', '-c', script], env=clean_env(**over), text=True,
                          capture_output=True)


def test_defaults_and_values():
    r = validate()
    assert r.returncode == 0 and r.stdout == 'off|140', r
    ok = [
        ({'DSV41_YOCO_PREFILL': 'off'}, 'off|140'),
        ({'DSV41_YOCO_PREFILL': ''}, 'off|140'),
        ({'DSV41_YOCO_PREFILL': 'compact'}, 'compact|140'),
        ({'DSV41_YOCO_PREFILL': 'compact', 'DSV41_YOCO_KEEP_TOKENS': '0256'}, 'compact|256'),
        ({'DSV41_YOCO_PREFILL': 'compact', 'DSV41_YOCO_KEEP_TOKENS': '132', 'DSPARK_TOKENS': '3'},
         'compact|132'),
        ({'DSV41_YOCO_PREFILL': 'compact', 'DSV41_YOCO_KEEP_TOKENS': '129', 'SPEC_METHOD': 'none'},
         'compact|129'),
        # off ignores KEEP's relation to the window (still a positive int)
        ({'DSV41_YOCO_PREFILL': 'off', 'DSV41_YOCO_KEEP_TOKENS': '5'}, 'off|5'),
    ]
    for over, want in ok:
        r = validate(**over)
        assert r.returncode == 0 and r.stdout == want, (over, r)


def test_refusals():
    bad = [
        ({'DSV41_YOCO_PREFILL': 'mask'}, 'DSV41_YOCO_PREFILL'),
        ({'DSV41_YOCO_PREFILL': 'on'}, 'DSV41_YOCO_PREFILL'),
        ({'DSV41_YOCO_PREFILL': 'COMPACT'}, 'DSV41_YOCO_PREFILL'),
        ({'DSV41_YOCO_PREFILL': 'compact', 'DSV41_YOCO_KEEP_TOKENS': '0'}, 'DSV41_YOCO_KEEP_TOKENS'),
        ({'DSV41_YOCO_PREFILL': 'compact', 'DSV41_YOCO_KEEP_TOKENS': '1.5e2'}, 'DSV41_YOCO_KEEP_TOKENS'),
        ({'DSV41_YOCO_PREFILL': 'compact', 'DSV41_YOCO_KEEP_TOKENS': ''}, 'DSV41_YOCO_KEEP_TOKENS'),
        ({'DSV41_YOCO_PREFILL': 'compact', 'DSV41_YOCO_KEEP_TOKENS': '2000000'}, 'DSV41_YOCO_KEEP_TOKENS'),
        # window 128 + backtrack (k + 1)
        ({'DSV41_YOCO_PREFILL': 'compact', 'DSV41_YOCO_KEEP_TOKENS': '131', 'DSPARK_TOKENS': '3'},
         'window'),
        ({'DSV41_YOCO_PREFILL': 'compact', 'DSV41_YOCO_KEEP_TOKENS': '133', 'DSPARK_TOKENS': '5'},
         'window'),
        ({'DSV41_YOCO_PREFILL': 'compact', 'DSV41_YOCO_KEEP_TOKENS': '128', 'SPEC_METHOD': 'none'},
         'window'),
        ({'DSV41_YOCO_PREFILL': 'compact', 'LANGUAGE_MODEL_ONLY': '0'}, 'LANGUAGE_MODEL_ONLY'),
        ({'DSV41_YOCO_PREFILL': 'compact', 'LANGUAGE_MODEL_ONLY': ''}, 'LANGUAGE_MODEL_ONLY'),
        ({'DSV41_YOCO_PREFILL': 'compact', 'DSPARK_TOKENS': 'x'}, 'DSPARK_TOKENS'),
    ]
    for over, word in bad:
        r = validate(**over)
        assert r.returncode == 2 and word in r.stderr, (over, r)


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
        assert '--speculative-config' in base, base
        for over in ({'DSV41_YOCO_PREFILL': 'off'},
                     {'DSV41_YOCO_PREFILL': 'compact', 'DSV41_YOCO_KEEP_TOKENS': '140'},
                     {'DSV41_YOCO_PREFILL': 'compact', 'DSV41_YOCO_KEEP_TOKENS': '512'}):
            assert _rank_argv(name, **over) == base, (name, over)


def test_inner_guard_only_when_enabled():
    for name in ('HEAD_SCRIPT', 'WORKER_SCRIPT'):
        script = SOURCE.split(f'cat > "${name}" <<\'EOF\'\n', 1)[1].split('\nEOF', 1)[0]
        guard = script[script.index('if [ "${DSV41_YOCO_PREFILL:-off}" != "off" ]; then'):]
        guard = guard[:guard.index('\nfi\n') + 4]
        probe = 'say() { echo "$*"; }\npython3() { echo "RAN $*"; return ${RC:-0}; }\n' + guard
        for mode, rc, want_ran, want_rc in (('', '0', False, 0), ('off', '0', False, 0),
                                            ('compact', '0', True, 0), ('compact', '1', True, 1)):
            env = clean_env(RC=rc)
            if mode:
                env['DSV41_YOCO_PREFILL'] = mode
            r = subprocess.run(['bash', '-c', probe], env=env, text=True, capture_output=True)
            assert ('RAN /opt/dsv41/patch_yoco_prefill.py' in r.stdout) == want_ran, (name, mode, r)
            assert r.returncode == want_rc, (name, mode, rc, r)


def test_propagation_both_ranks():
    assert SOURCE.count('DSV41_YOCO_PREFILL="${DSV41_YOCO_PREFILL-off}"') == 1
    assert SOURCE.count('DSV41_YOCO_KEEP_TOKENS="${DSV41_YOCO_KEEP_TOKENS-140}"') == 1
    assert SOURCE.count('-e DSV41_YOCO_PREFILL="$DSV41_YOCO_PREFILL"') == 1, 'head container'
    assert SOURCE.count('-e DSV41_YOCO_KEEP_TOKENS="$DSV41_YOCO_KEEP_TOKENS"') == 1, 'head container'
    loop = SOURCE.split('for v in SERVED_MODEL_NAME', 1)[1].split('; do', 1)[0]
    assert loop.split().count('DSV41_YOCO_PREFILL') == 1, 'worker container (serve_env)'
    assert loop.split().count('DSV41_YOCO_KEEP_TOKENS') == 1, 'worker container (serve_env)'
    example = (ROOT / '.env.example').read_text().splitlines()
    assert '# DSV41_YOCO_PREFILL=off' in example and '# DSV41_YOCO_KEEP_TOKENS=140' in example


if __name__ == '__main__':
    test_defaults_and_values()
    test_refusals()
    test_argv_byte_identical()
    test_inner_guard_only_when_enabled()
    test_propagation_both_ranks()
    print('yoco launcher: PASS')
