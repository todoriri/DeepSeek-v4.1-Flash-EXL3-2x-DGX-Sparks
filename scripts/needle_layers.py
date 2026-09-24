#!/usr/bin/env python3
"""Long-context retrieval check for the layer 18-22 hypothesis.

The checkpoint gives routed experts their fewest bits (K=2) on layers 18-22.
In DeepSeek-V4.1's encoder/decoder design every decoder layer's long-range KV,
and the indexer's candidate pool, are projected from the stream leaving the
encoder (layers 18-19). If that hurts anything, it hurts long-context
retrieval. A single needle is the easy case (upstream ran one at 1.03M), so
each haystack carries:

  * 5 key/value needles ("The access code for strongbox Kestrel-17 is 482913.")
    at depths 5/25/50/75/95 %, each with a NEAR-MISS decoy (Kestrel-71) plus
    10 unrelated strongbox codes: retrieval must match the key exactly;
  * 3 two-hop chains ("Cabinet Heron-52 opens with the code filed in
    pigeonhole Maple-9." ... "Pigeonhole Maple-9 holds the code 731904.") whose
    hops sit far apart, in both directions, plus decoy pigeonholes/cabinets.

Every question is appended after the SAME haystack, so one full prefill per
length and the rest hit the prefix cache. Thinking OFF, greedy (temperature
0): a pure retrieval signal. The hop questions are asked again with thinking
ON (effort 75) to see whether reasoning rescues a failed hop. Sequential
(concurrency 1): the teacher must be otherwise idle (no parallel prefill).

A wrong answer is classified: `decoy` (the value of a near-miss key or of the
wrong chain link) vs `invented` (a number the document never states).

    python3 scripts/needle_layers.py --haystack corpus.txt --out rows.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import urllib.request

BIRDS = ["Kestrel", "Heron", "Plover", "Osprey", "Curlew", "Merlin", "Bittern",
         "Dunlin", "Gannet", "Shrike", "Tanager", "Warbler", "Avocet", "Godwit",
         "Lapwing", "Petrel", "Redstart", "Siskin", "Vireo", "Whimbrel"]
TREES = ["Maple", "Rowan", "Alder", "Hazel", "Linden", "Cedar", "Juniper",
         "Larch", "Aspen", "Sorrel", "Yarrow", "Tamarind"]
SINGLE_DEPTHS = [0.05, 0.25, 0.50, 0.75, 0.95]
HOP_DEPTHS = [(0.10, 0.85), (0.80, 0.15), (0.40, 0.60)]  # (cabinet, pigeonhole)


def post(base: str, path: str, body: dict, timeout: float = 3600) -> dict:
    req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as f:
        return json.loads(f.read())


def chars_per_token(base: str, model: str, text: str) -> float:
    sample = text[:200_000]
    n = post(base, "/tokenize", {"model": model, "prompt": sample})["count"]
    return len(sample) / n


def build_facts(rng: random.Random, haystack: str):
    used_vals: set[int] = set()

    def val() -> int:
        while True:
            v = rng.randint(100000, 999999)
            if v not in used_vals and str(v) not in haystack:
                used_vals.add(v)
                return v

    def key(pool, used) -> str:
        while True:
            a, b = rng.randint(1, 9), rng.randint(1, 9)
            if a == b:
                continue
            k = f"{rng.choice(pool)}-{a}{b}"
            rev = f"{k.split('-')[0]}-{b}{a}"
            if k not in used and rev not in used:
                used.update((k, rev))
                return k

    used: set[str] = set()
    facts = []            # (depth, sentence)
    questions = []        # dicts
    for d in SINGLE_DEPTHS:
        k = key(BIRDS, used)
        name, digits = k.split("-")
        decoy_k = f"{name}-{digits[::-1]}"
        v, dv = val(), val()
        facts.append((d, f"The access code for strongbox {k} is {v}."))
        facts.append((rng.uniform(0.02, 0.98), f"The access code for strongbox {decoy_k} is {dv}."))
        questions.append({"kind": "single", "depth": d, "key": k, "answer": v, "decoys": [dv],
                          "q": f"What is the access code for strongbox {k}? "
                               "Reply with the 6-digit code only."})
    for _ in range(10):
        facts.append((rng.uniform(0.02, 0.98),
                      f"The access code for strongbox {key(BIRDS, used)} is {val()}."))
    tused: set[str] = set()
    for d_cab, d_pig in HOP_DEPTHS:
        cab, pig = key(BIRDS, used), key(TREES, tused)
        v = val()
        decoy_pig = f"{pig.split('-')[0]}-{pig.split('-')[1][::-1]}"
        dv = val()
        facts.append((d_cab, f"Cabinet {cab} opens with the code filed in pigeonhole {pig}."))
        facts.append((d_pig, f"Pigeonhole {pig} holds the code {v}."))
        facts.append((rng.uniform(0.02, 0.98), f"Pigeonhole {decoy_pig} holds the code {dv}."))
        questions.append({"kind": "hop", "depth": [d_cab, d_pig], "key": cab, "via": pig,
                          "answer": v, "decoys": [dv],
                          "q": f"Which code opens cabinet {cab}? Follow the statements in the "
                               "document. Reply with the 6-digit code only."})
    for _ in range(5):   # unrelated chains: a cabinet and a pigeonhole each
        c2, p2 = key(BIRDS, used), key(TREES, tused)
        facts.append((rng.uniform(0.02, 0.98), f"Cabinet {c2} opens with the code filed in pigeonhole {p2}."))
        facts.append((rng.uniform(0.02, 0.98), f"Pigeonhole {p2} holds the code {val()}."))
    all_vals = used_vals
    return facts, questions, all_vals


def make_doc(corpus: str, n_chars: int, facts) -> str:
    """The first n_chars of the corpus (cut at a paragraph), each fact inserted as its
    own paragraph at the paragraph boundary nearest its depth."""
    body = corpus[:n_chars]
    cut = body.rfind("\n\n")
    body = body[:cut] if cut > n_chars * 0.9 else body
    paras = body.split("\n\n")
    offs, o = [], 0
    for p in paras:
        offs.append(o)
        o += len(p) + 2
    inserts: dict[int, list[str]] = {}
    for depth, sent in facts:
        target = depth * len(body)
        i = min(range(len(offs)), key=lambda j: abs(offs[j] - target))
        inserts.setdefault(i, []).append(sent)
    out = []
    for i, p in enumerate(paras):
        out.extend(inserts.get(i, []))
        out.append(p)
    return "\n\n".join(out)


def ask(base, model, doc, q, thinking, max_tokens):
    prompt = ("Read the document below. Among other things it contains statements about "
              "strongboxes, cabinets and pigeonholes and their codes.\n\n<document>\n"
              + doc + "\n</document>\n\n" + q)
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "stream": False,
            "chat_template_kwargs": {"thinking": thinking, "enable_thinking": thinking}}
    if thinking:   # effort in chat_template_kwargs, as effort_ab.py sends it
        body["chat_template_kwargs"]["reasoning_effort"] = 75
        body.update(temperature=1.0, top_p=0.95)
    else:
        body.update(temperature=0.0)
    t0 = time.time()
    r = post(base, "/v1/chat/completions", body)
    el = time.time() - t0
    msg = r["choices"][0]["message"]
    u = r.get("usage") or {}
    return {"content": msg.get("content") or "", "reasoning_chars": len(
        msg.get("reasoning_content") or msg.get("reasoning") or ""),
        "finish": r["choices"][0].get("finish_reason"), "elapsed": round(el, 1),
        "prompt_tokens": u.get("prompt_tokens"),
        "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "completion_tokens": u.get("completion_tokens")}


def grade(content: str, qd: dict, all_vals: set[int]) -> str:
    nums = [int(x) for x in re.findall(r"(?<!\d)\d{6}(?!\d)", content)]
    if not nums:
        return "no_answer"
    if nums[0] == qd["answer"]:
        return "ok"
    if nums[0] in qd["decoys"] or nums[0] in all_vals:
        return "decoy"
    return "invented"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://192.168.1.97:8000")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--haystack", required=True)
    ap.add_argument("--lengths", default="8000,64000,128000,256000,300000")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-think-hops", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    corpus = open(a.haystack, encoding="utf-8").read()
    sha = hashlib.sha256(corpus.encode()).hexdigest()
    cpt = chars_per_token(a.base, a.model, corpus)
    lengths = [int(x) for x in a.lengths.split(",")]
    meta = {"haystack": a.haystack, "sha256": sha, "chars_per_token": round(cpt, 4),
            "lengths": lengths, "seed": a.seed, "model": a.model, "base": a.base}
    print("META " + json.dumps(meta), flush=True)
    rows = []
    with open(a.out, "w") as out:
        for L in lengths:
            rng = random.Random(a.seed * 1000 + L)
            n_chars = int(L * cpt) - 600          # room for the facts + the question
            if n_chars > len(corpus):
                print(f"skip {L}: corpus holds ~{int(len(corpus) / cpt)} tokens", flush=True)
                continue
            facts, qs, all_vals = build_facts(rng, corpus[:n_chars])
            doc = make_doc(corpus, n_chars, facts)
            plan = [(qd, False, 64) for qd in qs]
            if not a.no_think_hops:
                plan += [(qd, True, 8192) for qd in qs if qd["kind"] == "hop"]
            for qd, thinking, mt in plan:
                try:
                    res = ask(a.base, a.model, doc, qd["q"], thinking, mt)
                    verdict = grade(res["content"], qd, all_vals)
                except Exception as e:  # recorded, not fatal
                    res, verdict = {"error": repr(e)[:300]}, "error"
                row = {"length": L, "kind": qd["kind"], "depth": qd["depth"], "key": qd["key"],
                       "thinking": thinking, "answer": qd["answer"], "verdict": verdict,
                       **{k: v for k, v in res.items() if k != "content"},
                       "content": res.get("content", "")[:200]}
                rows.append(row)
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(json.dumps({k: row[k] for k in ("length", "kind", "depth", "thinking",
                                                     "verdict", "elapsed", "prompt_tokens",
                                                     "cached_tokens") if k in row}), flush=True)

    print("\nlength  kind    think  ok/n")
    for L in lengths:
        for kind in ("single", "hop"):
            for th in (False, True):
                sel = [r for r in rows if r["length"] == L and r["kind"] == kind and r["thinking"] == th]
                if sel:
                    ok = sum(r["verdict"] == "ok" for r in sel)
                    bad = ",".join(sorted({r["verdict"] for r in sel if r["verdict"] != "ok"}))
                    print(f"{L:>7} {kind:<7} {str(th):<6} {ok}/{len(sel)} {bad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
