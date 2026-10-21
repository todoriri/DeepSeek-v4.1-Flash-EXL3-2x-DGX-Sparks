#!/usr/bin/env python3
"""Offline checks for scripts/yoco_logprob_ab.py (no server): parsing, comparison, flags."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("yoco_logprob_ab", ROOT / "scripts" / "yoco_logprob_ab.py")
assert spec and spec.loader
ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)


def _choice(ids, lps, tops):
    return {"logprobs": {"tokens": [f"token_id:{i}" for i in ids], "token_logprobs": lps,
                         "top_logprobs": [{f"token_id:{k}": v for k, v in t.items()} for t in tops]}}


def _arm(decode_ids, decode_lps, probe_tops, traj):
    decode = [{"id": i, "logprob": lp, "top": {}} for i, lp in zip(decode_ids, decode_lps)]
    probes = {str(t): {"id": max(top, key=top.get), "logprob": max(top.values()),
                       "top": {str(k): v for k, v in top.items()}, "target": traj[t],
                       "target_logprob": top.get(traj[t])}
              for t, top in probe_tops.items()}
    return {"lengths": {"8000": {"decode": decode, "trajectory": traj, "probes": probes}}}


def test_parse_choice():
    rows = ab.parse_choice(_choice([5, 7], [-0.1, -0.2], [{5: -0.1, 6: -2.0}, {7: -0.2}]))
    assert rows == [{"id": 5, "logprob": -0.1, "top": {"5": -0.1, "6": -2.0}},
                    {"id": 7, "logprob": -0.2, "top": {"7": -0.2}}]
    try:
        ab.token_id("hello")
    except ValueError:
        pass
    else:
        raise AssertionError("text tokens accepted (need return_tokens_as_token_ids)")


def test_compare_and_verdict():
    traj = [5, 7, 9]
    ref = _arm([5, 7, 9], [-0.1, -0.2, -0.3], {0: {5: -0.1, 6: -2.0}, 1: {7: -0.2, 8: -3.0}}, traj)
    same = _arm([5, 7, 9], [-0.1, -0.21, -0.3], {0: {5: -0.1, 6: -2.0}, 1: {7: -0.21, 8: -3.0}}, traj)
    drift = _arm([5, 8, 9], [-0.1, -0.9, -0.3], {0: {5: -0.6, 6: -0.7}, 1: {8: -0.1, 7: -2.5}}, traj)
    floor = ab.compare_arms(ref, same)["8000"]
    assert floor["decode_first_divergence"] == 3 and abs(floor["probe_mean_abs_dlogp"] - 0.005) < 1e-9
    assert floor["probe_top1_agree"] == 1.0 and floor["probe_topk_jaccard"] == 1.0
    test = ab.compare_arms(ref, drift)["8000"]
    assert test["decode_first_divergence"] == 1 and test["probe_top1_agree"] == 0.5
    assert ab.verdict({"8000": floor}, {"8000": floor}) == []
    flags = ab.verdict({"8000": floor}, {"8000": test})
    assert any("dlogp" in f for f in flags) and any("top-1" in f for f in flags)
    d = Path(tempfile.mkdtemp())
    for name, arm in (("a", ref), ("b", same), ("c", drift)):
        (d / f"{name}.json").write_text(json.dumps(arm))
    assert ab.main(["compare", str(d / "a.json"), str(d / "b.json"), str(d / "b.json")]) == 0
    assert ab.main(["compare", str(d / "a.json"), str(d / "b.json"), str(d / "c.json")]) == 1


if __name__ == "__main__":
    test_parse_choice()
    test_compare_and_verdict()
    print("yoco_logprob_ab offline: PASS")
