#!/usr/bin/env python3
"""Teacher-forced logprob A/B for DSV41_YOCO_PREFILL (DeepInfer's YoCo L2, adapted).

DeepInfer's L2 (examples/yoco/yoco_equiv.py `logp` + `compare`; deepseek-ai/DeepInfer
@ 774ce24, Apache-2.0) replays one fixed teacher-forced trajectory under its mask and
compact modes and requires bitwise-equal logprob and top-K. The image has no mask mode
(the CPU half of that check is tests/test_yoco_metadata.py), so here the arms are the
served teacher with the feature off and on. Equality is not expected; the yardstick is
the off-vs-off noise floor:

    capture --arm off-a --out RUN                       # feature off: defines trajectories
    capture --arm off-b --out RUN --trajectory RUN/off-a.json   # same server: noise floor
    # operator restarts both ranks with DSV41_YOCO_PREFILL=compact
    capture --arm yoco  --out RUN --trajectory RUN/off-a.json
    compare RUN/off-a.json RUN/off-b.json RUN/yoco.json

Per context length (default 8k / 45k / 128k tokens of --context-file):
  * decode path: one greedy request (temperature 0, max_tokens=--steps, logprobs=K).
    The off-a arm's output is the trajectory. Other arms rerun the same request and are
    compared up to their first divergence (decode reads the decoder SWA that the
    truncated prefill wrote).
  * prefill path, teacher-forced: for each probe t, prompt = context + trajectory[:t]
    as token ids, max_tokens=1. It records the logprob of trajectory[t] and the top-K
    at the last prompt position, the row YOCO recomputes inside its rectangle.
The teacher must be otherwise idle. Prompts are token ids
(return_tokens_as_token_ids=true), so nothing is re-tokenized between arms.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

PROBES = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 63)


def post(base: str, path: str, body: dict, timeout: float = 3600) -> dict:
    req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def token_id(tok: str) -> int:
    if not tok.startswith("token_id:"):
        raise ValueError(f"expected return_tokens_as_token_ids output, got {tok!r}")
    return int(tok.split(":", 1)[1])


def parse_choice(choice: dict) -> list[dict]:
    """One row per generated position: chosen id, its logprob, top-K {id: logprob}."""
    lp = choice["logprobs"]
    rows = []
    for tok, val, top in zip(lp["tokens"], lp["token_logprobs"], lp["top_logprobs"]):
        rows.append({"id": token_id(tok), "logprob": val,
                     "top": {str(token_id(k)): v for k, v in (top or {}).items()}})
    return rows


def completion(base: str, model: str, prompt_ids: list[int], max_tokens: int, k: int) -> list[dict]:
    body = {"model": model, "prompt": prompt_ids, "max_tokens": max_tokens, "temperature": 0,
            "logprobs": k, "return_tokens_as_token_ids": True}
    return parse_choice(post(base, "/v1/completions", body)["choices"][0])


def capture(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    traj = json.loads(Path(args.trajectory).read_text()) if args.trajectory else None
    text = Path(args.context_file).read_text()
    ids = post(args.base, "/tokenize", {"model": args.model, "prompt": text})["tokens"]
    result = {"arm": args.arm, "model": args.model, "k": args.k, "steps": args.steps,
              "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "lengths": {}}
    for L in args.lengths:
        if len(ids) < L:
            raise SystemExit(f"--context-file has {len(ids)} tokens < {L}")
        ctx = ids[:L]
        t0 = time.time()
        decode = completion(args.base, args.model, ctx, args.steps, args.k)
        entry = {"decode": decode, "decode_s": round(time.time() - t0, 2)}
        trajectory = ([r["id"] for r in decode] if traj is None
                      else traj["lengths"][str(L)]["trajectory"])
        entry["trajectory"] = trajectory
        probes = {}
        for t in [p for p in PROBES if p < len(trajectory)]:
            row = completion(args.base, args.model, ctx + trajectory[:t], 1, args.k)[0]
            row["target"] = trajectory[t]
            row["target_logprob"] = row["top"].get(str(trajectory[t]))
            probes[str(t)] = row
        entry["probes"] = probes
        result["lengths"][str(L)] = entry
        print(f"[{args.arm}] L={L}: decode {entry['decode_s']} s, {len(probes)} probes", flush=True)
    path = out_dir / f"{args.arm}.json"
    path.write_text(json.dumps(result, indent=1))
    print(f"wrote {path}")
    return 0


def _overlap(a: dict, b: dict) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa | sb))


def compare_arms(ref: dict, cand: dict) -> dict:
    """Per length: decode agreement to first divergence; teacher-forced probe deltas."""
    out = {}
    for L, r in ref["lengths"].items():
        c = cand["lengths"].get(L)
        if c is None:
            continue
        div = next((i for i, (x, y) in enumerate(zip(r["decode"], c["decode"])) if x["id"] != y["id"]),
                   min(len(r["decode"]), len(c["decode"])))
        dec_d = [abs(x["logprob"] - y["logprob"]) for x, y in zip(r["decode"][:div], c["decode"][:div])]
        dl, top1, jac = [], [], []
        for t, rp in r["probes"].items():
            cp = c["probes"].get(t)
            if cp is None:
                continue
            a, b = rp["target_logprob"], cp["target_logprob"]
            if a is not None and b is not None:
                dl.append(abs(a - b))
            top1.append(max(rp["top"], key=rp["top"].get) == max(cp["top"], key=cp["top"].get))
            jac.append(_overlap(rp["top"], cp["top"]))
        mean = (lambda xs: sum(xs) / len(xs) if xs else math.nan)
        out[L] = {"decode_first_divergence": div, "decode_steps": len(r["decode"]),
                  "decode_mean_abs_dlogp": mean(dec_d),
                  "probe_mean_abs_dlogp": mean(dl), "probe_max_abs_dlogp": max(dl, default=math.nan),
                  "probe_top1_agree": mean([float(x) for x in top1]),
                  "probe_topk_jaccard": mean(jac), "probes": len(top1)}
    return out


def verdict(floor: dict, test: dict, slack: float = 3.0, abs_tol: float = 0.05) -> list[str]:
    """Heuristic flags: worse than `slack` x the off-vs-off floor and than abs_tol."""
    flags = []
    for L, t in test.items():
        f = floor.get(L, {})
        lim = max(abs_tol, slack * f.get("probe_mean_abs_dlogp", 0.0))
        if t["probe_mean_abs_dlogp"] > lim:
            flags.append(f"L={L}: probe mean |dlogp| {t['probe_mean_abs_dlogp']:.4f} > {lim:.4f}")
        if t["probe_top1_agree"] < f.get("probe_top1_agree", 1.0) - 0.1:
            flags.append(f"L={L}: probe top-1 agreement {t['probe_top1_agree']:.2f} "
                         f"< floor {f.get('probe_top1_agree', 1.0):.2f} - 0.10")
    return flags


def compare(args: argparse.Namespace) -> int:
    ref, floor_arm, test_arm = (json.loads(Path(p).read_text()) for p in (args.ref, args.floor, args.test))
    floor, test = compare_arms(ref, floor_arm), compare_arms(ref, test_arm)
    print(f"{'L':>8} {'arm':>6} {'div/steps':>10} {'dec|dlp|':>9} {'probe|dlp|':>11} {'max':>7} {'top1':>5} {'jac':>5}")
    for L in sorted(set(floor) | set(test), key=int):
        for name, d in (("floor", floor.get(L)), ("test", test.get(L))):
            if d:
                print(f"{L:>8} {name:>6} {d['decode_first_divergence']:>4}/{d['decode_steps']:<5} "
                      f"{d['decode_mean_abs_dlogp']:>9.4f} {d['probe_mean_abs_dlogp']:>11.4f} "
                      f"{d['probe_max_abs_dlogp']:>7.3f} {d['probe_top1_agree']:>5.2f} {d['probe_topk_jaccard']:>5.2f}")
    flags = verdict(floor, test)
    for f in flags:
        print("FLAG", f)
    print("PASS (within the heuristic bounds)" if not flags else "REVIEW")
    if args.json:
        Path(args.json).write_text(json.dumps({"floor": floor, "test": test, "flags": flags}, indent=1))
    return 0 if not flags else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--base", default="http://192.168.1.97:8000")
    c.add_argument("--model", default="deepseek-v4.1-flash")
    c.add_argument("--arm", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--context-file", required=True, help="long plain text (e.g. the needle haystack corpus)")
    c.add_argument("--lengths", type=int, nargs="+", default=[8000, 45000, 128000])
    c.add_argument("--steps", type=int, default=64)
    c.add_argument("--k", type=int, default=20)
    c.add_argument("--trajectory", help="baseline JSON whose trajectories to teacher-force")
    m = sub.add_parser("compare")
    m.add_argument("ref")
    m.add_argument("floor")
    m.add_argument("test")
    m.add_argument("--json")
    args = ap.parse_args(argv)
    return capture(args) if args.cmd == "capture" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
