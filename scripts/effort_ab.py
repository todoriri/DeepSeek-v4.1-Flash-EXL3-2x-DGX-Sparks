#!/usr/bin/env python3
"""A/B the DeepSeek V4.1 reasoning-effort budget on the live server.

Integer efforts bypass the name->budget table, so one server can compare the
pre-patch default (50) with the official "high" (75) without a restart.

Graded set: math with answers computed here (exact integer match) and small
coding tasks whose returned function is run against unit tests in a
network-less subprocess (``unshare -rn``) with a timeout.

Thinking on, official sampling (temperature 1.0, top_p 0.95).  Arms are
interleaved per prompt so load and cache state hit both arms equally.

    python3 scripts/effort_ab.py --base http://192.168.1.97:8000 --reps 2 --concurrency 2
    python3 scripts/effort_ab.py --set hard --efforts 75,100 --max-tokens 32768
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------- math set
def _sum_digits(n: int) -> int:
    return sum(map(int, str(n)))


def _primes_below(n: int) -> list[int]:
    sieve = bytearray([1]) * n
    sieve[:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(n) if sieve[i]]


def _trailing_zeros_fact(n: int) -> int:
    z, p = 0, 5
    while p <= n:
        z += n // p
        p *= 5
    return z


def _partitions(n: int) -> int:
    p = [1] + [0] * n
    for k in range(1, n + 1):
        for i in range(k, n + 1):
            p[i] += p[i - k]
    return p[n]


def _derangements(n: int) -> int:
    d = [1, 0]
    for i in range(2, n + 1):
        d.append((i - 1) * (d[-1] + d[-2]))
    return d[n]


def _collatz_len(n: int) -> int:
    steps = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps


MATH = [
    ("How many positive integers n <= 10000 are divisible by 3 or 5 but not by 7?",
     sum(1 for n in range(1, 10001) if (n % 3 == 0 or n % 5 == 0) and n % 7)),
    ("What is the sum of the decimal digits of 2**200?", _sum_digits(2 ** 200)),
    ("What are the last four digits of 7**2026? Give the integer formed by those four digits.",
     pow(7, 2026, 10000)),
    ("How many primes are there below 20000?", len(_primes_below(20000))),
    ("How many trailing zeros does 2026! have?", _trailing_zeros_fact(2026)),
    ("In how many ways can the integer 40 be written as a sum of positive integers, "
     "ignoring order?", _partitions(40)),
    ("How many derangements (permutations with no fixed point) of 11 elements are there?",
     _derangements(11)),
    ("How many monotone lattice paths go from (0,0) to (9,7) using unit steps right or up "
     "that never pass through the point (4,3)?",
     math.comb(16, 7) - math.comb(7, 3) * math.comb(9, 4)),
    ("What is the multiplicative inverse of 1234 modulo 100003? Answer with the integer in "
     "[1, 100002].", pow(1234, -1, 100003)),
    ("How many steps does the Collatz sequence starting at 837799 take to reach 1?",
     _collatz_len(837799)),
    ("What is the sum of all primes p < 1000 such that p + 2 is also prime?",
     sum(p for p in _primes_below(1000) if p + 2 in set(_primes_below(1002)))),
    ("How many integers between 1 and 1000000 inclusive are perfect squares or perfect cubes?",
     len({i * i for i in range(1, 1001)} | {i ** 3 for i in range(1, 101)})),
]

MATH_SUFFIX = "\n\nEnd your reply with a final line of the form `ANSWER: <integer>`."

# ---------------------------------------------------------------- code set
CODE = [
    ("evaluate",
     "Write a Python function `evaluate(expr: str) -> int` that evaluates an arithmetic "
     "expression with non-negative integer literals, + - * /, parentheses, spaces and unary "
     "minus. `/` is integer division truncating toward zero. Do not use eval/exec.",
     """
assert evaluate("1 + 2 * 3") == 7
assert evaluate("(1+2)*3") == 9
assert evaluate("-(2+3)*4") == -20
assert evaluate("7/-2") == -3
assert evaluate("- -3") == 3
assert evaluate("10 - 4 - 3") == 3
assert evaluate("2*(3+(4-1))/4") == 3
assert evaluate("-7/2") == -3
"""),
    ("merge_intervals",
     "Write a Python function `merge_intervals(xs: list[list[int]]) -> list[list[int]]` "
     "that merges overlapping or touching closed intervals and returns them sorted.",
     """
assert merge_intervals([]) == []
assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]
assert merge_intervals([[1,4],[4,5]]) == [[1,5]]
assert merge_intervals([[5,7],[1,2],[2,3],[6,6]]) == [[1,3],[5,7]]
"""),
    ("min_window",
     "Write a Python function `min_window(s: str, t: str) -> str` returning the shortest "
     "substring of s containing every character of t with multiplicity (leftmost on ties), "
     "or '' if none.",
     """
assert min_window("ADOBECODEBANC", "ABC") == "BANC"
assert min_window("a", "aa") == ""
assert min_window("aa", "aa") == "aa"
assert min_window("abcabdebac", "cda") == "cabd"
assert min_window("xyz", "") == ""
"""),
    ("to_roman",
     "Write Python functions `to_roman(n: int) -> str` and `from_roman(s: str) -> int` for "
     "standard Roman numerals 1..3999.",
     """
assert to_roman(1994) == "MCMXCIV"
assert to_roman(3999) == "MMMCMXCIX"
assert from_roman("MCMXCIV") == 1994
assert all(from_roman(to_roman(i)) == i for i in range(1, 4000))
"""),
    ("edit_distance",
     "Write a Python function `edit_distance(a: str, b: str) -> int` computing the "
     "Damerau-Levenshtein distance (optimal string alignment variant: insert, delete, "
     "substitute, transpose adjacent characters).",
     """
assert edit_distance("", "") == 0
assert edit_distance("kitten", "sitting") == 3
assert edit_distance("ca", "ac") == 1
assert edit_distance("abcdef", "abdcef") == 1
assert edit_distance("ca", "abc") == 3
"""),
    ("next_permutation",
     "Write a Python function `next_permutation(xs: list[int]) -> list[int]` returning the "
     "lexicographically next permutation of xs (wrapping to the smallest when xs is the "
     "largest). Duplicates are allowed. Do not mutate the input.",
     """
assert next_permutation([1,2,3]) == [1,3,2]
assert next_permutation([3,2,1]) == [1,2,3]
assert next_permutation([1,1,5]) == [1,5,1]
assert next_permutation([1,5,1]) == [5,1,1]
assert next_permutation([2,3,3,1]) == [3,1,2,3]
x=[1,2]; next_permutation(x); assert x == [1,2]
"""),
]

CODE_SUFFIX = ("\n\nReturn only one ```python code block containing the complete "
               "implementation (no tests, no input reading).")


# ---------------------------------------------------------------- client
def chat(base: str, model: str, prompt: str, effort: int, max_tokens: int) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": effort},
    }
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as f:
        d = json.load(f)
    d["_elapsed"] = time.time() - t0
    return d


def grade_math(content: str, answer: int) -> bool:
    m = re.findall(r"ANSWER:\s*\**\s*(-?[\d,]+)", content or "")
    return bool(m) and int(m[-1].replace(",", "")) == answer


def grade_code(content: str, tests: str) -> bool:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", content or "", re.S)
    if not blocks:
        return False
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.py"
        p.write_text(blocks[-1] + "\n\n" + tests)
        try:
            r = subprocess.run(["unshare", "-rn", sys.executable, "-I", str(p)], cwd=td,
                               capture_output=True, timeout=20)
        except subprocess.TimeoutExpired:
            return False
        return r.returncode == 0


def run_one(args, job):
    kind, name, prompt, key, effort, rep = job
    try:
        d = chat(args.base, args.model, prompt, effort, args.max_tokens)
    except Exception as e:  # noqa: BLE001 - record and keep going
        return dict(kind=kind, name=name, effort=effort, rep=rep, error=str(e)[:200])
    ch = d["choices"][0]
    msg = ch["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    ok = grade_math(content, key) if kind == "math" else grade_code(content, key)
    comp = d["usage"]["completion_tokens"]
    return dict(kind=kind, name=name, effort=effort, rep=rep, ok=ok,
                finish=ch.get("finish_reason"), completion_tokens=comp,
                reasoning_chars=len(reasoning), content_chars=len(content),
                elapsed=round(d["_elapsed"], 1), tok_s=round(comp / d["_elapsed"], 2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://192.168.1.97:8000")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--efforts", default="50,75")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", choices=("basic", "hard"), default="basic",
                    help="basic = the 2026-09-23 set; hard = scripts/effort_items_hard.py")
    args = ap.parse_args()
    efforts = [int(e) for e in args.efforts.split(",")]

    math_set, code_set = MATH, CODE
    if args.set == "hard":
        from effort_items_hard import CODE as code_set, MATH as math_set
    items = [("math", f"m{i}", q + MATH_SUFFIX, a) for i, (q, a) in enumerate(math_set)]
    items += [("code", n, p + CODE_SUFFIX, t) for n, p, t in code_set]
    jobs = []
    rng = random.Random(args.seed)
    for rep in range(args.reps):
        for kind, name, prompt, key in items:
            arms = efforts[:]
            rng.shuffle(arms)
            jobs += [(kind, name, prompt, key, e, rep) for e in arms]

    out = Path(args.out or f"/tmp/effort_ab-{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    print(f"{len(jobs)} requests, efforts={efforts}, c={args.concurrency} -> {out}", flush=True)
    rows = []
    with cf.ThreadPoolExecutor(args.concurrency) as ex, out.open("w") as fh:
        for i, row in enumerate(ex.map(lambda j: run_one(args, j), jobs), 1):
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(f"[{i}/{len(jobs)}] {row}", flush=True)

    print("\n=== summary ===")
    print(f"{'effort':>6} {'kind':>5} {'n':>3} {'acc':>6} {'med tok':>8} {'mean tok':>9} "
          f"{'med s':>7} {'tok/s':>6} {'len-cut':>7} {'err':>4}")
    for e in efforts:
        for kind in ("math", "code", "all"):
            rs = [r for r in rows if r["effort"] == e and (kind == "all" or r["kind"] == kind)]
            good = [r for r in rs if "error" not in r]
            if not good:
                continue
            acc = sum(r["ok"] for r in good) / len(good)
            toks = [r["completion_tokens"] for r in good]
            secs = [r["elapsed"] for r in good]
            print(f"{e:>6} {kind:>5} {len(good):>3} {acc:>6.1%} {statistics.median(toks):>8.0f} "
                  f"{statistics.mean(toks):>9.0f} {statistics.median(secs):>7.1f} "
                  f"{statistics.median(r['tok_s'] for r in good):>6.1f} "
                  f"{sum(r['finish'] == 'length' for r in good):>7} {len(rs) - len(good):>4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
