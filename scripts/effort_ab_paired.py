#!/usr/bin/env python3
"""Paired per-item summary of an effort_ab.py rows.jsonl (two efforts).

Each item ran at both efforts in every rep, so the comparison is paired:
  * accuracy per arm, and the discordant pairs (same item+rep, one arm right,
    the other wrong), which carry all the quality signal when both arms are
    near ceiling;
  * per item: mean completion tokens and latency per arm, and the ratio;
    the sign count (items where the high arm spent more) and the geometric
    mean of the per-item ratio, so one rambling run cannot dominate.

    python3 scripts/effort_ab_paired.py rows.jsonl [--efforts 75,100]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--efforts", default="")
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(a.rows) if line.strip()]
    effs = ([int(x) for x in a.efforts.split(",")] if a.efforts
            else sorted({r["effort"] for r in rows}))
    lo, hi = effs
    errs = [r for r in rows if "error" in r]
    good = [r for r in rows if "error" not in r]
    by = defaultdict(dict)                      # (name, rep) -> effort -> row
    for r in good:
        by[(r["name"], r["rep"])][r["effort"]] = r
    pairs = {k: v for k, v in by.items() if lo in v and hi in v}

    print(f"rows {len(rows)}  errors {len(errs)}  complete pairs {len(pairs)}")
    for e in effs:
        rs = [r for r in good if r["effort"] == e]
        for kind in ("math", "code", "all"):
            sel = [r for r in rs if kind == "all" or r["kind"] == kind]
            if sel:
                toks = [r["completion_tokens"] for r in sel]
                print(f"  effort {e:>3} {kind:>4}: {sum(r['ok'] for r in sel)}/{len(sel)} correct, "
                      f"tokens mean {statistics.mean(toks):.0f} median {statistics.median(toks):.0f} "
                      f"sum {sum(toks)}, median {statistics.median(r['elapsed'] for r in sel):.1f} s, "
                      f"length-cut {sum(r['finish'] == 'length' for r in sel)}")
    lo_only = [k for k, v in pairs.items() if v[lo]["ok"] and not v[hi]["ok"]]
    hi_only = [k for k, v in pairs.items() if v[hi]["ok"] and not v[lo]["ok"]]
    print(f"discordant pairs: {lo} right/{hi} wrong = {len(lo_only)} {lo_only}; "
          f"{hi} right/{lo} wrong = {len(hi_only)} {hi_only}")

    items = sorted({k[0] for k in pairs}, key=lambda n: (n[0] != "m", n))
    print(f"\n{'item':<22} {'acc ' + str(lo):>7} {'acc ' + str(hi):>7} {'tok ' + str(lo):>8} "
          f"{'tok ' + str(hi):>8} {'ratio':>6} {'s ' + str(lo):>7} {'s ' + str(hi):>7}")
    ratios, more = [], 0
    for n in items:
        ps = [v for k, v in pairs.items() if k[0] == n]
        tl = statistics.mean(p[lo]["completion_tokens"] for p in ps)
        th = statistics.mean(p[hi]["completion_tokens"] for p in ps)
        sl = statistics.mean(p[lo]["elapsed"] for p in ps)
        sh = statistics.mean(p[hi]["elapsed"] for p in ps)
        ratio = th / tl if tl else float("nan")
        ratios.append(ratio)
        more += th > tl
        print(f"{n:<22} {sum(p[lo]['ok'] for p in ps)}/{len(ps):>5} {sum(p[hi]['ok'] for p in ps)}/{len(ps):>5} "
              f"{tl:>8.0f} {th:>8.0f} {ratio:>6.2f} {sl:>7.1f} {sh:>7.1f}")
    tot_lo = sum(p[lo]["completion_tokens"] for p in pairs.values())
    tot_hi = sum(p[hi]["completion_tokens"] for p in pairs.values())
    s_lo = sum(p[lo]["elapsed"] for p in pairs.values())
    s_hi = sum(p[hi]["elapsed"] for p in pairs.values())
    gm = math.exp(statistics.mean(math.log(r) for r in ratios if r > 0))
    print(f"\n{hi} spent more on {more}/{len(items)} items; per-item ratio geomean {gm:.2f}, "
          f"median {statistics.median(ratios):.2f}, range {min(ratios):.2f}-{max(ratios):.2f}")
    print(f"total tokens {tot_lo} -> {tot_hi} ({tot_hi / tot_lo:.2f}x); "
          f"total seconds {s_lo:.0f} -> {s_hi:.0f} ({s_hi / s_lo:.2f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
