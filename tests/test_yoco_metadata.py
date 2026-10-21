#!/usr/bin/env python3
"""CPU port of DeepInfer's YoCo L1 "mask-compact" metadata check, for the vLLM layout.

DeepInfer (deepseek-ai/DeepInfer @ 774ce24, Apache-2.0) checks its truncated prefill
metadata in examples/yoco/yoco_equiv.py (L1). It compares the SWA pivots, visible
keys and store slots of the decoder ("consumer") layers against an analytic
reference: for a kept row at position pos of a sequence of length N, the visible keys
are [max(pos - W + 1, N - K, 0), pos]. Its reference calls the compiled
prepare_prompt_v4, so it needs a GPU. This port needs neither weights nor a GPU. Its
pieces:

  * the row selection `overlay/dsv41_yoco.py` actually uses (build_keep_plan);
  * a line-by-line Python port of the image's `_compute_swa_indices_and_lens_kernel`
    (vllm/v1/attention/backends/mla/sparse_swa.py, causal, text-only), fed with the
    compacted batch exactly as the shadow builder sees it;
  * the window clamp the overlay applies afterwards (clamp_swa_rows).

On top of DeepInfer's checks, it simulates whole multi-chunk prefills through the
paged SWA cache. A kept row must only ever read decoder-SWA slots that this prefill
already wrote (same or earlier chunk). Decode must then see a full, valid window.
"""
from __future__ import annotations

import importlib.util
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
# repo: overlay/dsv41_yoco.py; image: /opt/dsv41/dsv41_yoco.py next to this file
HELPER = next(p for p in (HERE.parent / "overlay" / "dsv41_yoco.py", HERE / "dsv41_yoco.py")
              if p.is_file())
spec = importlib.util.spec_from_file_location("dsv41_yoco", HELPER)
assert spec and spec.loader
yoco = importlib.util.module_from_spec(spec)
sys.modules["dsv41_yoco"] = yoco  # dataclasses resolve the module by name
spec.loader.exec_module(yoco)

W = 128            # sliding_window (config.json text_config)
K = 140            # DSV41_YOCO_KEEP_TOKENS default
BS = 64            # SWA block size on SM12x (patch_sm120_block64)
DECODE_THRESHOLD = 4  # 1 + num_speculative_tokens (DSpark k=3)


# --------------------------------------------------------------------------------------
# Port of the image kernel (sparse_swa.py _compute_swa_indices_and_lens_kernel), causal,
# HAS_IMAGE=False. One "program" per row; writes index_width entries.
# --------------------------------------------------------------------------------------
def kernel_swa_indices(qsl, seq_lens, token_to_req, is_valid, block_table, block_size,
                       window, index_width, rows):
    out = np.full((len(rows), index_width), -1, dtype=np.int64)
    lens = np.zeros(len(rows), dtype=np.int64)
    for pid, token_idx in enumerate(rows):
        if not is_valid[token_idx]:
            continue
        req = token_to_req[token_idx]
        query_start, query_end = qsl[req], qsl[req + 1]
        prefix_len = seq_lens[req] - (query_end - query_start)
        pos = prefix_len + token_idx - query_start
        start_pos = max(pos - (window - 1), 0)
        swa_len = pos + 1 - start_pos
        lens[pid] = swa_len
        for off in range(min(swa_len, index_width)):
            p = start_pos + off
            out[pid, off] = block_table[req][p // block_size] * block_size + p % block_size
    return out, lens


def split_decodes_and_prefills(qsl, threshold):
    """Port of vllm/v1/attention/backends/utils.py split_decodes_and_prefills
    (require_uniform=False, treat_short_extends_as_decodes=True)."""
    q = np.diff(qsl)
    num_reqs, num_tokens = len(q), int(qsl[-1])
    if q.max() <= threshold:
        return num_reqs, 0, num_tokens, 0
    if q[0] > threshold:
        return 0, num_reqs, 0, num_tokens
    first = int(np.argmax(q > threshold))
    return first, num_reqs - first, int(qsl[first]), num_tokens - int(qsl[first])


@dataclass
class Req:
    N: int          # prefill_len: tokens this prefill must reach (prompt + resumed output)
    start: int      # first position scheduled this step
    q_len: int      # scheduled rows this step
    prefilling: bool = True


def block_tables_for(reqs, seed=0):
    """Distinct, shuffled physical blocks per request (slots never collide)."""
    rng = random.Random(seed)
    tables, nxt = [], 1  # block 0 = null block
    for r in reqs:
        n = (max(r.N, r.start + r.q_len) + BS - 1) // BS + 2
        ids = list(range(nxt, nxt + n))
        rng.shuffle(ids)
        tables.append(ids)
        nxt += n
    return tables


def slot_of(table, p):
    return table[p // BS] * BS + p % BS


def compact_rows(reqs, keep=K, threshold=DECODE_THRESHOLD, tables=None):
    """Run the overlay's row selection + the image kernel + the clamp on one step.

    Returns per compact row: (req, pos, visible_positions, store_slot, placeholder)."""
    qsl = np.concatenate([[0], np.cumsum([r.q_len for r in reqs])]).astype(np.int64)
    plan = yoco.build_keep_plan(
        query_start_loc=qsl,
        start_pos=np.array([r.start for r in reqs], dtype=np.int64),
        prefill_len=np.array([r.N for r in reqs], dtype=np.int64),
        is_prefilling=np.array([r.prefilling for r in reqs], dtype=bool),
        keep=keep,
    )
    tables = tables or block_tables_for(reqs)
    cq = plan.compact_qsl
    full_pos = np.concatenate([np.arange(r.start, r.start + r.q_len) for r in reqs])
    full_req = np.concatenate([[i] * r.q_len for i, r in enumerate(reqs)]).astype(np.int64)
    pos_c = full_pos[plan.keep_idx]
    req_c = full_req[plan.keep_idx]
    # Shadow-builder inputs: compact slot mapping (-1 on placeholder rows), seq_lens kept.
    slot_c = np.array([slot_of(tables[r], p) for r, p in zip(req_c, pos_c)], dtype=np.int64)
    slot_c[plan.placeholder] = -1
    seq_lens = np.array([r.start + r.q_len for r in reqs], dtype=np.int64)
    token_to_req = np.repeat(np.arange(len(reqs)), np.diff(cq))
    is_valid = slot_c >= 0
    nd, _, ndt, _ = split_decodes_and_prefills(cq, threshold)
    rows = list(range(len(pos_c)))
    idx, lens = kernel_swa_indices(cq, seq_lens, token_to_req, is_valid, tables, BS, W, W, rows)
    # The overlay clamps decode-split and prefill-split tensors separately.
    for sl in (slice(0, ndt), slice(ndt, len(rows))):
        if sl.stop - sl.start == 0:
            continue
        ti = torch.from_numpy(idx[sl].copy()).to(torch.int32).unsqueeze(1)
        tl = torch.from_numpy(lens[sl].copy()).to(torch.int32)
        yoco.clamp_swa_rows(ti, tl, torch.from_numpy(pos_c[sl]), torch.from_numpy(plan.lower_bound[sl]))
        idx[sl] = ti.squeeze(1).numpy()
        lens[sl] = tl.numpy()
    inv = [{slot_of(t, p): p for p in range(len(t) * BS)} for t in tables]
    out = []
    for i in rows:
        vis = [inv[req_c[i]][s] for s in idx[i, : lens[i]]]
        assert all(s == -1 for s in idx[i, lens[i]:]), "entries past swa_len must be -1"
        out.append((int(req_c[i]), int(pos_c[i]), vis, int(slot_c[i]), bool(plan.placeholder[i])))
    return plan, out, tables


def reference_visible(r: Req, pos, keep=K, window=W):
    """DeepInfer's analytic reference (yoco_equiv.py _reference_visible)."""
    if not r.prefilling or r.N <= keep:
        lo = max(pos - window + 1, 0)
    else:
        lo = max(pos - window + 1, r.N - keep, 0)
    return list(range(lo, pos + 1)) if lo <= pos else []


def expected_rows(r: Req, keep=K):
    """Kept positions of one request this step (+ placeholder flag)."""
    lo, hi = r.start, r.start + r.q_len
    if not r.prefilling or r.N <= keep:
        return list(range(lo, hi)), False
    first = max(lo, r.N - keep)
    if first >= hi:
        return [hi - 1], True       # chunk wholly before the rectangle: one placeholder
    return list(range(first, hi)), False


def check_case(name, reqs, keep=K):
    plan, rows, tables = compact_rows(reqs, keep)
    by_req = {}
    for req, pos, vis, slot, ph in rows:
        by_req.setdefault(req, []).append((pos, vis, slot, ph))
    for i, r in enumerate(reqs):
        want, want_ph = expected_rows(r, keep)
        got = by_req[i]
        assert [p for p, *_ in got] == want, (name, i, [p for p, *_ in got][:5], want[:5])
        for pos, vis, slot, ph in got:
            assert ph == want_ph, (name, i, pos)
            if ph:
                assert vis == [] and slot == -1, (name, i, pos, len(vis), slot)
                continue
            assert vis == reference_visible(r, pos, keep), (name, i, pos, vis[:3], vis[-3:])
            assert len(vis) <= W
            assert slot == slot_of(tables[i], pos), (name, i, pos)
    # compact rows of each request are a contiguous suffix of its scheduled rows
    for i in range(len(reqs)):
        lo, hi = plan.compact_qsl[i], plan.compact_qsl[i + 1]
        idx = plan.keep_idx[lo:hi]
        assert hi > lo and np.all(np.diff(idx) == 1), (name, i)
        full_end = sum(r.q_len for r in reqs[: i + 1])
        assert idx[-1] == full_end - 1, (name, i, "last row (logits row) must be kept")
    return plan


# --------------------------------------------------------------------------------------
# DeepInfer's L1 case matrix (build_mask_cases), with vLLM's chunk sizes.
# --------------------------------------------------------------------------------------
def deepinfer_cases(chunk):
    long_n = 2 * chunk + 904
    exact = 2 * chunk + K
    inside = 2 * chunk + K + 60
    return [
        ("short_lt_keep", [Req(64, 0, 64)]),
        ("eq_keep", [Req(K, 0, K)]),
        ("gt_keep_single_chunk", [Req(4 * W, 0, 4 * W)]),
        ("chunked_c0_all_clipped", [Req(long_n, 0, chunk)]),
        ("chunked_c1_all_clipped", [Req(long_n, chunk, chunk)]),
        ("chunked_c2_partial", [Req(long_n, 2 * chunk, long_n - 2 * chunk)]),
        ("rect_at_chunk_start", [Req(exact, 2 * chunk, K)]),
        ("rect_inside_chunk", [Req(inside, 2 * chunk, inside - 2 * chunk)]),
        ("rect_cuts_history", [Req(2000, 1900, 100)]),
        ("mixed_batch", [Req(64, 0, 64), Req(long_n, 2 * chunk, long_n - 2 * chunk),
                         Req(4 * W, 0, 4 * W), Req(long_n, chunk, chunk), Req(K, 0, K)]),
    ]


def test_deepinfer_case_matrix():
    for chunk in (1280, 1536, 2048):          # LPTT, MNBT, DeepInfer's default
        for name, reqs in deepinfer_cases(chunk):
            check_case(f"{name}@{chunk}", reqs)


def test_vllm_specific_batches():
    cases = [
        # decodes first (V2 order), then a prefill crossing the rectangle
        ("decode+prefill", [Req(5000, 4999, 4, prefilling=False), Req(9000, 7680, 1320)]),
        ("two_decodes+early_chunk", [Req(800, 799, 4, False), Req(70000, 799, 1, False),
                                     Req(300000, 1280, 1280)]),
        # final chunk shorter than the decode threshold (classified as decode)
        ("tiny_final_chunk", [Req(9000, 8997, 3)]),
        ("tiny_chunk_inside_rect", [Req(9000, 8870, 3), Req(20000, 0, 1280)]),
        # a 1-token chunk wholly before the rectangle (budget remainder)
        ("one_token_early", [Req(9000, 100, 1)]),
        # resumed after preemption: N = prompt + generated, chunk from 0
        ("resumed", [Req(6000 + 37, 4800, 1237)]),
        # rectangle boundary exactly at a block edge / one past it
        ("rect_block_edge", [Req(64 * 40 + K, 64 * 40 - 3, 200)]),
        ("rect_block_edge+1", [Req(64 * 40 + K + 1, 64 * 40 - 3, 200)]),
    ]
    for name, reqs in cases:
        check_case(name, reqs)


def test_random_batches():
    rng = random.Random(20261020)
    for _ in range(300):
        reqs = []
        for _ in range(rng.randint(1, 3)):
            N = rng.choice([rng.randint(1, 300), rng.randint(300, 20000)])
            if rng.random() < 0.3:
                reqs.append(Req(N, N - 1, rng.randint(1, 4), prefilling=False))
            else:
                start = rng.randint(0, N - 1)
                reqs.append(Req(N, start, rng.randint(1, min(1536, N - start))))
        reqs.sort(key=lambda r: r.prefilling)  # V2: decodes first
        check_case("random", reqs)


def test_mask_compact_equivalence():
    """DeepInfer _compare_mask: shared rows identical; mask-only rows see nothing."""
    for chunk in (1280, 1536):
        for name, reqs in deepinfer_cases(chunk):
            _, rows, tables = compact_rows(reqs)
            compact = {(q, p): (v, s) for q, p, v, s, ph in rows if not ph}
            for i, r in enumerate(reqs):
                for pos in range(r.start, r.start + r.q_len):
                    vis = reference_visible(r, pos)
                    in_rect = not (r.N > K and pos < r.N - K)
                    mask_store = slot_of(tables[i], pos) if in_rect else -1
                    if (i, pos) in compact:
                        assert compact[(i, pos)] == (vis, mask_store), (name, i, pos)
                    else:
                        assert vis == [], (name, i, pos, "dropped row must see nothing")


def simulate_prefill(N, chunks, hit=0, keep=K):
    """Run a whole prefill chunk by chunk; track decoder-SWA writes per position."""
    written = set()
    pos0 = hit
    for q_len in chunks:
        r = Req(N, pos0, q_len)
        tables = block_tables_for([Req(N, 0, N)], seed=N)   # one stable table per request
        _, rows, _ = compact_rows([r], keep, tables=tables)
        for _, pos, vis, slot, ph in rows:
            missing = [p for p in vis if p not in written and p != pos]
            assert not missing, (N, chunks, hit, pos, missing[:4], "read unwritten decoder SWA")
            assert all(p >= hit for p in vis), "read a cached (unwritten) decoder-SWA position"
            if slot >= 0:
                written.add(pos)
        pos0 += q_len
    assert pos0 == N
    # first decode step: positions N..N+k read [n-127, n]; all < N must be written
    for n in range(N, N + DECODE_THRESHOLD):
        need = [p for p in range(max(n - W + 1, 0), N)]
        assert all(p in written for p in need), (N, chunks, hit, n)
    return written


def test_multichunk_never_reads_unwritten():
    rng = random.Random(7)
    for _ in range(400):
        N = rng.randint(1, 30000)
        hit = 0
        if rng.random() < 0.5 and N > K + 192:
            hit = yoco.capped_hit(N, keep=K, align=128, peek=64, cached=rng.randint(0, N))
        chunks, left = [], N - hit
        while left:
            c = min(left, rng.choice([1, 2, 3, 4, 5, 64, 139, 140, 141, 1280, 1536]))
            chunks.append(c)
            left -= c
        simulate_prefill(N, chunks, hit)


def test_hit_cap_and_retention_boundary():
    for N in range(1, 3000):
        cap = yoco.max_cache_hit_length(N, K)
        assert cap == max(0, N - K)
        for cached in (0, N // 2, N - 1, N):
            hit = yoco.capped_hit(N, keep=K, align=128, peek=64, cached=cached)
            assert hit % 128 == 0 and hit <= cached and hit + 64 <= cap or hit == 0
            assert hit <= max(0, N - K - 64)
        # the YOCO retention boundary keeps a checkpoint that a retry can use
        b = yoco.yoco_reachable_boundary(N, K, block_size=64, use_eagle=True)
        if b > 0:
            a = b // 128 * 128
            assert a + 64 <= cap, (N, b, a)
            assert N - a <= K + 64 + 127, "recompute after a retry is bounded"


def test_clamp_brute_force():
    g = torch.Generator().manual_seed(3)
    for _ in range(200):
        R = 17
        pos = torch.randint(0, 600, (R,), generator=g)
        start = torch.clamp(pos - (W - 1), min=0)
        lens = (pos + 1 - start).to(torch.int32)
        idx = torch.full((R, 1, W), -1, dtype=torch.int32)
        for i in range(R):
            n = int(lens[i])
            idx[i, 0, :n] = torch.arange(int(start[i]), int(start[i]) + n) + 10000
        lower = torch.randint(0, 700, (R,), generator=g)
        lower[::3] = 0
        want = []
        for i in range(R):
            lo = max(int(start[i]), int(lower[i]))
            want.append([p + 10000 for p in range(lo, int(pos[i]) + 1)])
        yoco.clamp_swa_rows(idx, lens, pos, lower)
        for i in range(R):
            n = int(lens[i])
            assert idx[i, 0, :n].tolist() == want[i] and n == len(want[i])
            assert bool((idx[i, 0, n:] == -1).all())


def test_toy_attention_compact_equals_mask():
    """Gather keys through the clamped paged indices and attend. Compact (kept rows,
    chunked, cache carried across chunks) must equal a full-row reference whose
    decoder SWA is masked to the rectangle (DeepInfer's mask == compact, bitwise)."""
    torch.manual_seed(0)
    D = 8
    for N, chunks in ((900, [300, 300, 300]), (1500, [1280, 220]), (700, [700]),
                      (5000, [1536, 1536, 1536, 392]), (420, [279, 1, 140])):
        keys = torch.randn(N, D)
        vals = torch.randn(N, D)
        tables = block_tables_for([Req(N, 0, N)], seed=N)
        n_slots = (max(tables[0]) + 1) * BS
        cache_k = torch.full((n_slots, D), float("nan"))   # unwritten = NaN poison
        cache_v = torch.full((n_slots, D), float("nan"))
        out = {}
        pos0 = 0
        for q_len in chunks:
            _, rows, _ = compact_rows([Req(N, pos0, q_len)], tables=tables)
            for _, pos, vis, slot, ph in rows:            # store first (paged-direct)
                if slot >= 0:
                    cache_k[slot], cache_v[slot] = keys[pos], vals[pos]
            for _, pos, vis, slot, ph in rows:
                if ph or not vis:
                    continue
                s = torch.tensor([slot_of(tables[0], p) for p in vis])
                att = torch.softmax(cache_k[s] @ keys[pos], 0)
                out[pos] = att @ cache_v[s]
            pos0 += q_len
        for pos in range(max(N - K, 0), N):
            lo = max(pos - W + 1, N - K if N > K else 0)
            att = torch.softmax(keys[lo:pos + 1] @ keys[pos], 0)
            ref = att @ vals[lo:pos + 1]
            assert torch.equal(out[pos], ref), (N, chunks, pos)


def test_settings_and_min_keep():
    assert yoco.min_keep_tokens(window=128, num_speculative_tokens=3) == 132
    assert yoco.min_keep_tokens(window=128, num_speculative_tokens=0) == 129
    s = yoco.Settings.from_env({})
    assert (s.mode, s.keep) == ("off", 140) and not s.enabled
    s = yoco.Settings.from_env({"DSV41_YOCO_PREFILL": "compact", "DSV41_YOCO_KEEP_TOKENS": "0200"})
    assert (s.mode, s.keep, s.enabled) == ("compact", 200, True)
    for bad in ({"DSV41_YOCO_PREFILL": "mask"}, {"DSV41_YOCO_PREFILL": "on"},
                {"DSV41_YOCO_PREFILL": "compact", "DSV41_YOCO_KEEP_TOKENS": "0"},
                {"DSV41_YOCO_PREFILL": "compact", "DSV41_YOCO_KEEP_TOKENS": "1e3"}):
        try:
            yoco.Settings.from_env(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)


class _Cfg:
    num_hidden_layers = 40
    sliding_window = 128
    compress_ratios = [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]
    kv_source_layer_ids = [2, 8, 14, 20]
    index_source_layer_ids = [2, 8, 14, 20, 24, 28, 32, 36]
    candidate_source_layer_id = 20
    engram_layer_ids = [1, 14]
    dspark_target_layer_ids = [37, 38, 39]


def test_topology_shipped_config():
    assert yoco.decoder_boundary(_Cfg()) == 20
    bad = []
    for field, value in (("engram_layer_ids", [1, 20]), ("kv_source_layer_ids", [2, 8, 14, 20, 30]),
                         ("candidate_source_layer_id", 14), ("dspark_target_layer_ids", [19, 38, 39]),
                         ("index_source_layer_ids", [2, 8, 14, 24]), ("compress_ratios", [0] * 43)):
        cfg = _Cfg()
        setattr(cfg, field, value)
        try:
            yoco.decoder_boundary(cfg)
        except ValueError:
            continue
        bad.append(field)
    assert not bad, f"accepted an incompatible topology: {bad}"


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"YOCO L1 metadata port OK ({len(tests)} tests)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
