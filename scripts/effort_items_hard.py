"""Harder graded set for scripts/effort_ab.py (`--set hard`).

The 2026-09-23 set scored 100% at both 50 and 75, so it could not separate budgets.
These items aim near the model's ceiling: competition-style math with numbers changed
from the well-known versions (answers computed here, exact match) and coding tasks whose
tests compare against a brute-force reference on seeded random inputs.
"""
from __future__ import annotations

import math
from fractions import Fraction
from itertools import permutations


# ---------------------------------------------------------------- math answers
def _pairs_sq_sum_div13() -> int:
    return sum(1 for x in range(1, 1001) for y in range(1, 1001) if (x * x + y * y) % 13 == 0)


def _subsets_sum_mod7() -> int:
    ways = [1] + [0] * 6
    for v in range(1, 31):
        ways = [ways[r] + ways[(r - v) % 7] for r in range(7)]
    return ways[0] % 1000


def _binary_no00_no111(n: int) -> int:
    # state: (last bit, run length of that bit)
    st = {(0, 1): 1, (1, 1): 1}
    for _ in range(n - 1):
        nx: dict = {}
        for (b, r), c in st.items():
            for nb in (0, 1):
                nr = r + 1 if nb == b else 1
                if (nb == 0 and nr >= 2) or (nb == 1 and nr >= 3):
                    continue
                nx[(nb, nr)] = nx.get((nb, nr), 0) + c
        st = nx
    return sum(st.values())


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _lcm_pairs_gcd_gt1() -> int:
    n = 2 ** 5 * 3 ** 4 * 5 ** 2
    ds = _divisors(n)
    return sum(1 for a in ds for b in ds if a * b // math.gcd(a, b) == n and math.gcd(a, b) > 1)


def _product_mult6() -> int:
    good = total = 0
    for a in range(1, 51):
        for b in range(a + 1, 51):
            total += 1
            good += (a * b) % 6 == 0
    f = Fraction(good, total)
    return f.numerator + f.denominator


def _sqrt_step_squares() -> int:
    a, count = 1, 0
    while a <= 10 ** 6:
        r = math.isqrt(a)
        count += r * r == a
        a += r
    return count


def _bounded_compositions() -> int:
    ways = [1] + [0] * 30
    for _ in range(6):
        ways = [sum(ways[s - x] for x in range(0, 9) if s - x >= 0) for s in range(31)]
    return ways[30]


def _menage8() -> int:
    n = 8
    return sum(1 for p in permutations(range(n))
               if all(p[i] != i and p[i] != (i + 1) % n for i in range(n)))


def _lattice_disc() -> int:
    return sum(1 for x in range(-46, 47) for y in range(-46, 47) if x * x + y * y <= 2026)


def _v_p_fact(n: int, p: int) -> int:
    v, q = 0, p
    while q <= n:
        v += n // q
        q *= p
    return v


def _smallest_fact_div() -> int:
    # smallest n with 2026**1013 | n!  (2026 = 2 * 1013; the 1013-adic valuation binds)
    n = 1013
    while _v_p_fact(n, 1013) < 1013:
        n += 1013
    assert _v_p_fact(n, 2) >= 1013
    return n


def _phi_third_sum() -> int:
    phi = list(range(1001))
    for i in range(2, 1001):
        if phi[i] == i:
            for j in range(i, 1001, i):
                phi[j] -= phi[j] // i
    return sum(n for n in range(1, 1001) if 3 * phi[n] == n)


def _pal7_div11() -> int:
    c = 0
    for a in range(1, 10):
        for b in range(10):
            for cc in range(10):
                for d in range(10):
                    if int(f"{a}{b}{cc}{d}{cc}{b}{a}") % 11 == 0:
                        c += 1
    return c


def _domino_4x7() -> int:
    rows, cols = 4, 7
    # column-by-column transfer over 4-bit masks of cells already filled from the left
    def fill(col_mask, row, next_mask, out):
        if row == rows:
            out.append(next_mask)
            return
        if col_mask >> row & 1:
            fill(col_mask, row + 1, next_mask, out)
            return
        fill(col_mask, row + 1, next_mask | 1 << row, out)  # horizontal domino
        if row + 1 < rows and not col_mask >> (row + 1) & 1:
            fill(col_mask, row + 2, next_mask, out)  # vertical domino
    dp = {0: 1}
    for _ in range(cols):
        nd: dict = {}
        for m, c in dp.items():
            outs: list = []
            fill(m, 0, 0, outs)
            for o in outs:
                nd[o] = nd.get(o, 0) + c
        dp = nd
    return dp.get(0, 0)


def _binom_mod1000() -> int:
    return math.comb(2026, 1013) % 1000


MATH = [
    ("How many ordered pairs (x, y) of integers with 1 <= x <= 1000 and 1 <= y <= 1000 "
     "satisfy that x^2 + y^2 is divisible by 13?", _pairs_sq_sum_div13()),
    ("Let N be the number of subsets of {1, 2, ..., 30} (the empty set included) whose "
     "elements sum to a multiple of 7. What is the remainder when N is divided by 1000?",
     _subsets_sum_mod7()),
    ("How many binary strings of length 20 contain no two consecutive 0s and no three "
     "consecutive 1s?", _binary_no00_no111(20)),
    ("Let n = 2^5 * 3^4 * 5^2. How many ordered pairs (a, b) of positive integers satisfy "
     "lcm(a, b) = n and gcd(a, b) > 1?", _lcm_pairs_gcd_gt1()),
    ("What are the last three digits of 3^(3^(3^3))? (Exponentiation is right-associative: "
     "the exponent is 3^27.) Give the integer formed by those three digits.",
     pow(3, 3 ** 27, 1000)),
    ("Two distinct integers are chosen uniformly at random from {1, 2, ..., 50}. The "
     "probability that their product is a multiple of 6 is m/n in lowest terms. Find m + n.",
     _product_mult6()),
    ("A sequence is defined by a_1 = 1 and a_(k+1) = a_k + floor(sqrt(a_k)). How many terms "
     "of the sequence that are at most 1,000,000 are perfect squares?", _sqrt_step_squares()),
    ("How many ordered 6-tuples (x1, ..., x6) of integers with 0 <= xi <= 8 for every i "
     "satisfy x1 + x2 + x3 + x4 + x5 + x6 = 30?", _bounded_compositions()),
    ("How many permutations s of {0, 1, ..., 7} satisfy s(i) != i and s(i) != (i + 1) mod 8 "
     "for every i?", _menage8()),
    ("How many points (x, y) with integer coordinates satisfy x^2 + y^2 <= 2026?",
     _lattice_disc()),
    ("What is the smallest positive integer n such that n! is divisible by 2026^1013?",
     _smallest_fact_div()),
    ("What is the sum of all positive integers n <= 1000 for which Euler's totient "
     "satisfies phi(n) = n / 3?", _phi_third_sum()),
    ("How many 7-digit palindromes (first digit nonzero) are divisible by 11?",
     _pal7_div11()),
    ("In how many ways can a 4 x 7 rectangle be tiled with 1 x 2 dominoes (placed "
     "horizontally or vertically)?", _domino_4x7()),
    ("What is the remainder when the binomial coefficient C(2026, 1013) is divided by 1000?",
     _binom_mod1000()),
]

# ---------------------------------------------------------------- code set
_NO_RE = '''
import re as _re_t
_src = open(__file__).read().split("# ---TESTS---")[0]
assert not _re_t.search(r"^\\s*(import\\s+re\\b|from\\s+re\\s+import)", _src, _re_t.M), "uses re"
'''

CODE = [
    ("regex_match",
     "Write a Python function `regex_match(s: str, p: str) -> bool` that returns True iff "
     "the WHOLE string s matches the pattern p. Pattern language: a lowercase letter "
     "matches itself; `.` matches any single character; a character class `[...]` holds "
     "lowercase letters and ranges such as `a-c`, and a leading `^` negates it; every atom "
     "(letter, `.` or class) may be followed by at most one quantifier: `*` (zero or more), "
     "`+` (one or more) or `?` (zero or one). There is no other syntax. Do not use the "
     "`re` module.",
     "# ---TESTS---" + _NO_RE + '''
import random
assert regex_match("", "")
assert regex_match("", "a*")
assert not regex_match("a", "")
assert regex_match("ab", ".*")
assert regex_match("abc", "a[^a]+")
assert regex_match("aaa", "a?a?a?aaa")
assert not regex_match("ab", "a")
assert regex_match("cab", "[a-c]+")
assert not regex_match("d", "[a-c]")
rng = random.Random(7)
atoms = ["a", "b", "c", ".", "[ab]", "[^a]", "[a-b]", "[^bc]"]
for _ in range(600):
    p = "".join(rng.choice(atoms) + rng.choice(["", "", "*", "+", "?"])
                for _ in range(rng.randint(0, 5)))
    s = "".join(rng.choice("abc") for _ in range(rng.randint(0, 7)))
    assert regex_match(s, p) == bool(_re_t.fullmatch(p, s)), (s, p)
'''),
    ("count_pal_subseq",
     "Write a Python function `count_pal_subseq(s: str) -> int` returning the number of "
     "DISTINCT non-empty palindromic subsequences of s (distinct as strings), modulo "
     "10**9 + 7. s has length up to 1000 and uses only the letters a, b, c, d. It must run "
     "in well under a second for length 1000.",
     "# ---TESTS---" + '''
import random
from itertools import combinations
def _brute(s):
    seen = set()
    for r in range(1, len(s) + 1):
        for idx in combinations(range(len(s)), r):
            t = "".join(s[i] for i in idx)
            if t == t[::-1]:
                seen.add(t)
    return len(seen)
def _ref(s):
    M = 10**9 + 7
    n = len(s)
    if n == 0:
        return 0
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 1
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] != s[j]:
                dp[i][j] = dp[i + 1][j] + dp[i][j - 1] - dp[i + 1][j - 1]
            else:
                lo, hi = i + 1, j - 1
                while lo <= hi and s[lo] != s[i]:
                    lo += 1
                while lo <= hi and s[hi] != s[i]:
                    hi -= 1
                if lo > hi:
                    dp[i][j] = 2 * dp[i + 1][j - 1] + 2
                elif lo == hi:
                    dp[i][j] = 2 * dp[i + 1][j - 1] + 1
                else:
                    dp[i][j] = 2 * dp[i + 1][j - 1] - dp[lo + 1][hi - 1]
            dp[i][j] %= M
    return dp[0][n - 1]
assert count_pal_subseq("bccb") == 6
assert count_pal_subseq("a") == 1
rng = random.Random(11)
for _ in range(120):
    s = "".join(rng.choice("abcd") for _ in range(rng.randint(1, 11)))
    assert count_pal_subseq(s) == _brute(s), s
import time
big = "".join(rng.choice("abcd") for _ in range(1000))
t0 = time.time(); got = count_pal_subseq(big); dt = time.time() - t0
assert got == _ref(big)
assert dt < 6, dt
'''),
    ("skyline",
     "Write a Python function `skyline(buildings: list[list[int]]) -> list[list[int]]`. "
     "Each building is [left, right, height] with left < right and height > 0, a "
     "rectangle on flat ground. Return the skyline as key points [x, h] sorted by x: each "
     "key point is the left end of a horizontal segment of the outline, the last one has "
     "h = 0, and no two consecutive key points have the same height. Buildings may be "
     "given in any order; an empty list gives [].",
     "# ---TESTS---" + '''
import random
def _ref(bs):
    xs = sorted({x for l, r, h in bs for x in (l, r)})
    out = []
    for x in xs:
        h = max([hh for l, r, hh in bs if l <= x < r], default=0)
        if not out or out[-1][1] != h:
            out.append([x, h])
    return out
assert skyline([]) == []
assert skyline([[2,9,10],[3,7,15],[5,12,12],[15,20,10],[19,24,8]]) == [[2,10],[3,15],[7,12],[12,0],[15,10],[20,8],[24,0]]
assert skyline([[0,2,3],[2,5,3]]) == [[0,3],[5,0]]
rng = random.Random(13)
for _ in range(400):
    bs = []
    for _ in range(rng.randint(0, 8)):
        l = rng.randint(0, 18); r = rng.randint(l + 1, 20)
        bs.append([l, r, rng.randint(1, 6)])
    assert skyline([b[:] for b in bs]) == _ref(bs), bs
'''),
    ("lisp_eval",
     "Write a Python function `lisp_eval(expr: str) -> int` for this language. An "
     "expression is an integer (possibly negative), a variable, or a parenthesised form: "
     "`(add e1 e2)` is e1 + e2; `(mult e1 e2)` is e1 * e2; `(let v1 e1 v2 e2 ... vn en "
     "body)` assigns each vi the value of ei in order (later assignments see earlier ones) "
     "and then evaluates body. A variable starts with a lowercase letter followed by "
     "lowercase letters or digits; add, let and mult are never variable names. A variable "
     "takes the value from the innermost enclosing let that assigned it. Tokens are "
     "separated by single spaces; inputs are always valid.",
     "# ---TESTS---" + '''
cases = {
    "(add 1 2)": 3,
    "(mult 3 (add 2 3))": 15,
    "(let x 2 (mult x 5))": 10,
    "(let x 2 (mult x (let x 3 y 4 (add x y))))": 14,
    "(let x 3 x 2 x)": 2,
    "(let x 1 y 2 x (add x y) (add x y))": 5,
    "(let x 2 (add (let x 3 (let x 4 x)) x))": 6,
    "(let a1 3 b2 (add a1 1) b2)": 4,
    "(add -2 (mult -3 4))": -14,
    "(let x 7 (let y (add x 1) (let x (mult y 2) (add x y))))": 24,
    "(let x -1 (let x (add x x) (mult x x)))": 4,
    "-12": -12,
    "(let v1 5 v2 (mult v1 v1) (let v1 (add v2 1) (add v1 v2)))": 51,
}
for e, want in cases.items():
    assert lisp_eval(e) == want, (e, lisp_eval(e), want)
'''),
    ("merge_stones",
     "Write a Python function `merge_stones(stones: list[int], k: int) -> int`. There are "
     "len(stones) piles in a row. A move merges exactly k CONSECUTIVE piles into one pile "
     "and costs the total number of stones in those k piles. Return the minimum total cost "
     "to merge all piles into one pile, or -1 if that is impossible. A single pile costs 0. "
     "k >= 2 and len(stones) <= 30.",
     "# ---TESTS---" + '''
import random
from functools import lru_cache
@lru_cache(maxsize=None)
def _brute(t, k):
    if len(t) == 1:
        return 0
    best = None
    for i in range(len(t) - k + 1):
        nt = t[:i] + (sum(t[i:i + k]),) + t[i + k:]
        sub = _brute(nt, k)
        if sub >= 0:
            c = sum(t[i:i + k]) + sub
            best = c if best is None else min(best, c)
    return -1 if best is None else best
assert merge_stones([3,2,4,1], 2) == 20
assert merge_stones([3,2,4,1], 3) == -1
assert merge_stones([3,5,1,2,6], 3) == 25
assert merge_stones([5], 3) == 0
rng = random.Random(17)
for _ in range(250):
    k = rng.randint(2, 4)
    t = tuple(rng.randint(1, 9) for _ in range(rng.randint(1, 8)))
    assert merge_stones(list(t), k) == _brute(t, k), (t, k)
'''),
    ("max_points",
     "Write a Python function `max_points(points: list[list[int]]) -> int` returning the "
     "largest number of the given points that lie on one straight line. Points may repeat; "
     "every copy counts. Coordinates are integers; use exact arithmetic.",
     "# ---TESTS---" + '''
import random
def _ref(ps):
    n = len(ps)
    if n <= 2:
        return n
    best = 0
    for i in range(n):
        for j in range(n):
            if ps[i] == ps[j]:
                continue
            (x1, y1), (x2, y2) = ps[i], ps[j]
            c = sum(1 for (x, y) in ps if (x2 - x1) * (y - y1) == (y2 - y1) * (x - x1))
            best = max(best, c)
    return best if best else n
assert max_points([]) == 0
assert max_points([[1,1]]) == 1
assert max_points([[1,1],[1,1],[1,1]]) == 3
assert max_points([[1,1],[2,2],[3,3]]) == 3
assert max_points([[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]]) == 4
assert max_points([[0,0],[94911151,94911150],[94911152,94911151]]) == 2
rng = random.Random(19)
for _ in range(400):
    ps = [[rng.randint(-3, 3), rng.randint(-3, 3)] for _ in range(rng.randint(0, 9))]
    assert max_points([p[:] for p in ps]) == _ref(ps), ps
'''),
]
