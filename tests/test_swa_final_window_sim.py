#!/usr/bin/env python3
"""CPU run of the image's own prefix-cache code, with and without
overlay/patch_swa_final_window.py (DSV41_SWA_FINAL_WINDOW=keep).

The modules are the image's files in tests/fixtures/dsv41_yoco/vllm (block pool, hybrid
coordinator, sliding-window manager, KV cache manager, specs). Only their leaf imports are
stubbed (logger, hashing, kv events, platform). A small driver plays the scheduler's part:
it runs chunked prefill, DSpark-style verify steps (k = 3, random acceptance, rejections
roll back num_computed_tokens), and the free at finish. Coverage:

  * fidelity: on the stock files, 40 turns of repl session 3b75b1f3 (2026-10-18, lengths
    from the request record) reproduce every recorded continuation hit exactly. Each
    lands at the previous prompt's tail or at a 4096 checkpoint inside the previous output;
  * patched (YOCO off): every continuation hits the final checkpoint
    B_end = align128(floor64(C) - 64), C = computed tokens at finish. The re-prefilled
    own output drops accordingly;
  * patched + YOCO compact, in both patch orders: hits respect YOCO's cap N - KEEP, and
    land on B_end or on the extra checkpoint align128(T - KEEP - 64) when the continuation
    is too short for B_end;
  * a sweep over prompt / output residues and acceptance patterns;
  * the hold keeps a request's real-held sliding-window blocks under its admission cap;
  * the cached tail is bounded (no dense decode caching), and stays below the processed
    tokens when a step is still in flight;
  * preemption, abort mid-prefill, the deferred (async) free path and an empty request
    are safe;
  * the patched files refuse to import while DSV41_SWA_FINAL_WINDOW is not keep.
"""
from __future__ import annotations

import enum
import hashlib
import importlib.util
import inspect
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "dsv41_yoco" / "vllm"
PATCH = ROOT / "overlay" / "patch_swa_final_window.py"
YOCO_PATCH = ROOT / "overlay" / "patch_yoco_prefill.py"
YOCO_HELPER = ROOT / "overlay" / "dsv41_yoco.py"

BS, ALIGN, PEEK, WINDOW, K = 64, 128, 64, 128, 3
KEEP = 140
ENV_KEYS = ("DSV41_SWA_FINAL_WINDOW", "DSV41_YOCO_PREFILL", "DSV41_YOCO_KEEP_TOKENS",
            "DSV41_YOCO_HELPER")

# repl request record, session 3b75b1f3-498e-52b9-8632-3cb3ab2ac779 (10-18 17:07Z on), first 40
# rows in started_at order: prompt_tokens | output_tokens | cached_tokens (backend usage).
SESSION_3B75 = [tuple(map(int, r.split("|"))) for r in """
34516|263|28672 35773|188|34432 53227|383|35712 55171|145|53248 59967|231|55168
62949|175|59904 64183|132|62848 66927|112|64128 69076|106|66816 74127|106|68992
79286|88|74112 84040|88|79232 87416|459|83968 90610|655|87296 91304|149|90496
91466|134|91264 91626|233|91392 93616|189|91520 97352|519|93568 114289|1436|97280
115852|246|114688 117165|333|115840 118654|110|117120 121041|2025|118528 126072|1467|122880
128099|5416|126976 135223|478|131072 140157|131|135168 143699|1366|140032 146052|1903|143616
148236|3977|147456 157902|3107|151552 162797|500|159744 164830|299|162688 166734|132|164736
171358|700|166656 174798|770|171264 175704|1004|174720 177735|2466|176128 182183|6606|177664
""".split()]


# --------------------------------------------------------------------- expected values
def b_end(computed: int) -> int:
    """Newest checkpoint a freed request completes: window + DSpark peek block below
    its last full block, aligned to the 128-token hit granularity."""
    return (computed // BS * BS - PEEK) // ALIGN * ALIGN


def b_yoco(num_tokens: int, keep: int = KEEP) -> int:
    """Under YOCO compact: the newest checkpoint any continuation (N >= T) can use."""
    return (num_tokens - keep - PEEK) // ALIGN * ALIGN


# ------------------------------------------------------------------- stubbed leaf imports
class _Log(logging.Logger):
    def info_once(self, *a, **k):
        self.info(*a)

    def warning_once(self, *a, **k):
        self.warning(*a)

    def debug_once(self, *a, **k):
        self.debug(*a)


class RequestStatus(enum.IntEnum):
    WAITING = 0
    RUNNING = 1
    PREEMPTED = 2
    FINISHED_STOPPED = 3


class _Event:
    def __init__(self, *a, **k):
        pass


def _install_stubs() -> None:
    logging.setLoggerClass(_Log)
    for name in list(sys.modules):
        if name == "vllm" or name.startswith("vllm.") or name == "dsv41_yoco":
            del sys.modules[name]

    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__path__ = []
        m.__dict__.update(attrs)
        sys.modules[name] = m
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(sys.modules[parent], child, m)
        return m

    def cdiv(a, b):
        return -(a // -b)

    def digest(obj):
        return hashlib.sha256(repr(obj).encode()).digest()

    class Mamba(enum.Enum):
        MAMBA1, MAMBA2, SHORT_CONV, LINEAR, GDN_ATTN, MAMBA = range(6)

    for name in ("vllm", "vllm.utils", "vllm.distributed", "vllm.v1", "vllm.v1.core",
                 "vllm.v1.metrics", "vllm.v1.attention", "vllm.v1.attention.backends"):
        mod(name)
    mod("vllm.logger", init_logger=logging.getLogger)
    mod("vllm.envs")
    mod("vllm.config", VllmConfig=type("VllmConfig", (), {}))
    mod("vllm.platforms", current_platform=SimpleNamespace(
        is_device_capability_family=lambda fam: fam == 120,
        register_custom_kv_cache_specs=lambda cfg: None))
    mod("vllm.utils.math_utils", cdiv=cdiv, round_down=lambda a, b: a // b * b,
        round_up=lambda a, b: cdiv(a, b) * b)
    mod("vllm.utils.torch_utils", get_dtype_size=lambda dt: 1)
    mod("vllm.utils.hashing", xxhash=digest, xxhash_cbor=digest, sha256=digest,
        sha256_cbor=digest)
    mod("vllm.utils.mem_utils", format_gib=lambda b: f"{b / 2**30:.2f}")
    mod("vllm.distributed.kv_events", BlockStored=_Event, BlockRemoved=_Event,
        AllBlocksCleared=_Event, KVCacheEvent=_Event, MEDIUM_GPU="GPU")
    mod("vllm.v1.metrics.stats", PrefixCacheStats=_Event, KVCacheEvictionEvent=_Event)
    mod("vllm.v1.attention.backends.registry", MambaAttentionBackendEnum=Mamba)
    mod("vllm.v1.utils", tensor_data=lambda t: t)
    mod("vllm.v1.request", Request=type("Request", (), {}), RequestStatus=RequestStatus)


ORDER = [
    ("vllm.v1.kv_cache_layout", "v1/kv_cache_layout.py"),
    ("vllm.v1.kv_cache_spec_registry", "v1/kv_cache_spec_registry.py"),
    ("vllm.v1.kv_cache_interface", "v1/kv_cache_interface.py"),
    ("vllm.v1.core.kv_cache_utils", "v1/core/kv_cache_utils.py"),
    ("vllm.v1.core.kv_cache_metrics", "v1/core/kv_cache_metrics.py"),
    ("vllm.v1.core.block_pool", "v1/core/block_pool.py"),
    ("vllm.v1.core.single_type_kv_cache_manager", "v1/core/single_type_kv_cache_manager.py"),
    ("vllm.v1.core.kv_cache_coordinator", "v1/core/kv_cache_coordinator.py"),
    ("vllm.v1.core.kv_cache_manager", "v1/core/kv_cache_manager.py"),
]


def load(tree: Path) -> SimpleNamespace:
    """Import the cache modules of `tree` (a vllm package directory) under the stubs."""
    _install_stubs()
    mods = {}
    for name, rel in ORDER:
        spec = importlib.util.spec_from_file_location(name, tree / rel)
        assert spec and spec.loader
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        parent, _, child = name.rpartition(".")
        setattr(sys.modules[parent], child, m)
        spec.loader.exec_module(m)
        mods[child] = m
    mods["single_type_kv_cache_manager"].register_all_kvcache_specs(None)
    return SimpleNamespace(**mods)


@contextmanager
def environ(**values):
    saved = {k: os.environ.get(k) for k in ENV_KEYS}
    for k in ENV_KEYS:
        os.environ.pop(k, None)
    os.environ.update({k: v for k, v in values.items() if v is not None})
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


_TMP = tempfile.TemporaryDirectory(prefix="swa-final-sim-")


def tree(*patches: str) -> Path:
    """A copy of the fixture tree with the named patches applied in order
    ("final" = this overlay, "yoco" = the YOCO prefill overlay)."""
    dst = Path(tempfile.mkdtemp(dir=_TMP.name)) / "vllm"
    shutil.copytree(FIXTURES, dst)
    env = {k: v for k, v in os.environ.items() if k not in ENV_KEYS}
    env.update(DSV41_SWA_FINAL_WINDOW="keep", DSV41_YOCO_PREFILL="compact",
               DSV41_YOCO_HELPER=str(YOCO_HELPER))
    for name in patches:
        script = {"final": PATCH, "yoco": YOCO_PATCH}[name]
        r = subprocess.run([sys.executable, str(script), "--root", str(dst)], env=env,
                           text=True, capture_output=True)
        assert r.returncode == 0, (name, r.stdout, r.stderr)
    return dst


MODES = {
    # name: (patches, env while importing and running)
    "stock": ((), {}),
    "final": (("final",), {"DSV41_SWA_FINAL_WINDOW": "keep"}),
    "yoco": (("yoco",), {"DSV41_YOCO_PREFILL": "compact", "DSV41_YOCO_HELPER": str(YOCO_HELPER)}),
    "final+yoco": (("final", "yoco"), {"DSV41_SWA_FINAL_WINDOW": "keep",
                                       "DSV41_YOCO_PREFILL": "compact",
                                       "DSV41_YOCO_HELPER": str(YOCO_HELPER)}),
    "yoco+final": (("yoco", "final"), {"DSV41_SWA_FINAL_WINDOW": "keep",
                                       "DSV41_YOCO_PREFILL": "compact",
                                       "DSV41_YOCO_HELPER": str(YOCO_HELPER)}),
}
_TREES: dict[str, Path] = {}


class _Errors(logging.Handler):
    def __init__(self):
        super().__init__(logging.ERROR)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


@contextmanager
def engine(mode: str):
    """The image's cache modules for `mode`, loaded with that mode's environment. Fails
    if anything logged an error (the free hook logs and swallows its exceptions)."""
    patches, env = MODES[mode]
    if mode not in _TREES:
        _TREES[mode] = FIXTURES if not patches else tree(*patches)
    errors = _Errors()
    logging.getLogger().addHandler(errors)
    try:
        with environ(**env):
            yield Engine(load(_TREES[mode]), yoco="yoco" in mode)
    finally:
        logging.getLogger().removeHandler(errors)
    assert not errors.records, [r.getMessage() for r in errors.records]


# -------------------------------------------------------------------- scheduler stand-in
class Req:
    """The fields of vllm.v1.request.Request the cache manager reads."""

    def __init__(self, rid: str, prompt: list[int]):
        self.request_id = rid
        self.prompt_token_ids = list(prompt)
        self.out: list[int] = []
        self.num_computed_tokens = 0
        self.num_in_flight_tokens = 0
        self.num_output_placeholders = 0
        self.shared_prefix_boundary = 0
        self.status = RequestStatus.WAITING
        self.num_preemptions = 0
        self.skip_reading_prefix_cache = False
        self.lora_request = None
        self.cache_salt = None
        self.mm_features: list = []
        self.block_hashes: list[bytes] = []
        self._hash()

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.out

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.out)

    def _hash(self) -> None:
        # Chained per-64-token hashes, as Request.update_block_hashes keeps them.
        toks = self.all_token_ids
        while (len(self.block_hashes) + 1) * BS <= len(toks):
            i = len(self.block_hashes)
            parent = self.block_hashes[-1] if i else b""
            self.block_hashes.append(
                hashlib.sha256(parent + repr(toks[i * BS:(i + 1) * BS]).encode()).digest())

    def append(self, toks: list[int]) -> None:
        self.out.extend(toks)
        self._hash()


class Engine:
    MNBT = 1536

    def __init__(self, mods: SimpleNamespace, yoco: bool):
        self.m = mods
        self.yoco = yoco
        I = mods.kv_cache_interface

        def swa():
            return I.SlidingWindowMLASpec(
                block_size=BS, num_kv_heads=1, head_size=512, dtype="uint8",
                sliding_window=WINDOW, cache_dtype_str="fp8_ds_mla", alignment=576,
                model_version="deepseek_v4")

        def mla(bs, ratio):
            return I.MLAAttentionSpec(
                block_size=bs, num_kv_heads=1, head_size=512, dtype="uint8",
                tokens_per_state=ratio, cache_dtype_str="fp8_ds_mla", alignment=576,
                model_version="deepseek_v4")

        # The image's DS V4.1 kinds: ratio-2 compressed KV (128-token pages on SM12x),
        # ratio-1 compressed KV, SWA groups {i, i+15, i+30} (the one holding the last
        # registered layer is the DSpark/EAGLE group), the compressor state ring.
        ring = I.CircularBufferSpec(block_size=8, num_kv_heads=1, head_size=1024,
                                    head_size_v=0, dtype="float32")
        groups = [I.KVCacheGroupSpec(["c2"], mla(2 * BS, 2)),
                  I.KVCacheGroupSpec(["c1"], mla(BS, 1))]
        groups += [I.KVCacheGroupSpec([f"swa.{i}"], swa(), is_eagle_group=(i == 2))
                   for i in range(3)]
        groups += [I.KVCacheGroupSpec(["ring"], ring)]
        cfg = I.KVCacheConfig(num_blocks=40000, kv_cache_tensors=[], kv_cache_groups=groups,
                              prefix_cache_retention_interval=4096)
        self.kvm = mods.kv_cache_manager.KVCacheManager(
            cfg, max_model_len=400000, scheduler_block_size=ALIGN, hash_block_size=BS,
            max_in_flight_tokens=2 * self.MNBT, enable_caching=True, use_eagle=True,
            num_prefill_lookahead=1)
        self.swa = [mgr for mgr in self.kvm.coordinator.single_type_managers
                    if isinstance(mgr, mods.single_type_kv_cache_manager.SlidingWindowManager)]
        assert len(self.swa) == 3 and all(mgr.use_eagle for mgr in self.swa)
        self.max_held_over_cap = 0

    def hit(self, req: Req) -> int:
        return self.kvm.get_computed_blocks(req)[1]

    def _check_cap(self, req: Req) -> None:
        for mgr in self.swa:
            held = sum(1 for b in mgr.req_to_blocks.get(req.request_id, ()) if not b.is_null)
            cap = mgr._max_admission_blocks_per_request
            self.max_held_over_cap = max(self.max_held_over_cap, held - cap)

    def _alloc(self, req: Req, n: int, **kw):
        blocks = self.kvm.allocate_slots(req, n, **kw)
        assert blocks is not None, "pool exhausted"
        self._check_cap(req)

    def prefill(self, req: Req, rng: random.Random, out: list[int], stop_after: int | None = None):
        blocks, hit, spb = self.kvm.get_computed_blocks(req)
        req.shared_prefix_boundary = spb
        req.hit = hit
        n = min(self.MNBT, req.num_tokens - hit)
        self._alloc(req, n, num_new_computed_tokens=hit, new_computed_blocks=blocks,
                    num_lookahead_tokens=K if hit + n == req.num_tokens else 0)
        req.status = RequestStatus.RUNNING
        req.num_computed_tokens = hit + n
        chunks = 1
        while req.num_computed_tokens < req.num_tokens:
            if stop_after is not None and chunks >= stop_after:
                return
            n = min(self.MNBT, req.num_tokens - req.num_computed_tokens)
            self._alloc(req, n, num_lookahead_tokens=K)
            req.num_computed_tokens += n
            chunks += 1
        req.append(out[:1])

    def decode(self, req: Req, rng: random.Random, out: list[int], steps: int | None = None):
        done = 0
        while len(req.out) < len(out) and (steps is None or done < steps):
            self._alloc(req, 1 + K, num_lookahead_tokens=K)
            req.num_computed_tokens += 1 + K
            accepted = rng.randint(0, K)
            req.num_computed_tokens -= K - accepted
            produced = len(req.out)
            req.append(out[produced:produced + accepted + 1])
            done += 1

    def in_flight_step(self, req: Req) -> None:
        """Async scheduling: the next verify step is scheduled but not processed."""
        self._alloc(req, 1 + K, num_lookahead_tokens=K)
        req.num_computed_tokens += 1 + K
        req.num_in_flight_tokens += 1 + K
        req.num_output_placeholders += 1 + K

    def in_flight_chunk(self, req: Req) -> None:
        """Async scheduling: the next prefill chunk is scheduled but not processed (no
        output placeholders: a non-final chunk samples nothing)."""
        n = min(self.MNBT, req.num_tokens - req.num_computed_tokens)
        self._alloc(req, n, num_lookahead_tokens=K)
        req.num_computed_tokens += n
        req.num_in_flight_tokens += n

    def run(self, req: Req, out: list[int], rng: random.Random) -> Req:
        self.prefill(req, rng, out)
        self.decode(req, rng, out)
        self.kvm.free(req)
        return req


def finish_computed(req: Req) -> int:
    return min(req.num_computed_tokens - req.num_in_flight_tokens, req.num_tokens)


def replay_session(mode: str, rows=SESSION_3B75, seed: int = 7):
    """Feed the record's turns through `mode`; each prompt = previous prompt + previous
    output + the recorded growth. Returns [(P, O, hit_next, C_finish, T)]."""
    rng = random.Random(seed)
    out = []
    with engine(mode) as eng:
        toks = [rng.randrange(1, 100000) for _ in range(rows[0][0])]
        for i, (P, O, _cached) in enumerate(rows):
            assert len(toks) == P, (i, len(toks), P)
            req = Req(f"turn{i}", toks)
            eng.run(req, [rng.randrange(1, 100000) for _ in range(O)], rng)
            if i + 1 == len(rows):
                break
            nxt = rows[i + 1][0]
            assert nxt - req.num_tokens >= 0
            toks = req.all_token_ids + [rng.randrange(1, 100000) for _ in range(nxt - req.num_tokens)]
            out.append((P, O, eng.hit(Req("probe", toks)), finish_computed(req), req.num_tokens))
        assert eng.max_held_over_cap <= 0, eng.max_held_over_cap
    return out


def own_output_reprefill(P: int, T: int, hit: int) -> int:
    """Tokens of the previous output [P, T) the next prompt had to prefill again."""
    return max(0, T - max(hit, P))


# ---------------------------------------------------------------------------------- tests
def test_stock_reproduces_the_record():
    turns = replay_session("stock")
    for i, (P, O, hit, C, T) in enumerate(turns):
        assert hit == SESSION_3B75[i + 1][2], (i, P, O, hit, SESSION_3B75[i + 1][2])
    # The #5 loss: 19,099 of the 32,221 output tokens of these 39 turns were prefilled again.
    lost = sum(own_output_reprefill(P, T, hit) for P, O, hit, C, T in turns)
    record = sum(own_output_reprefill(P, P + O, SESSION_3B75[i + 1][2])
                 for i, (P, O, _c) in enumerate(SESSION_3B75[:-1]))
    assert lost == record == 19099, (lost, record)


def test_final_window_hits_the_end():
    stock = replay_session("stock")
    final = replay_session("final")
    for (P, O, h0, C0, T0), (P1, O1, hit, C, T) in zip(stock, final):
        assert (P, O, T) == (P1, O1, T0)
        assert hit == b_end(C), (P, O, hit, C, b_end(C))
        assert hit >= h0 and 64 <= T - hit <= 192, (P, O, hit, T)
        assert own_output_reprefill(P, T, hit) <= min(O, 192), (P, O, hit)
    # 19,099 -> 4,639 tokens (-76%): what is left is the 64-token peek block plus the
    # 128-token hit alignment, under 192 per turn.
    before = sum(own_output_reprefill(P, T, h) for P, O, h, C, T in stock)
    after = sum(own_output_reprefill(P, T, h) for P, O, h, C, T in final)
    assert (before, after) == (19099, 4639), (before, after)


def _yoco_expect(hit: int, P_next: int, C: int, T: int) -> None:
    cap = P_next - KEEP
    assert hit % ALIGN == 0 and hit + PEEK <= cap, (hit, P_next)
    usable = [b for b in (b_end(C), b_yoco(T)) if b >= ALIGN and b + PEEK <= cap]
    assert usable and hit >= max(usable), (hit, usable, P_next, C, T)


def test_yoco_compact_both_orders():
    yoco = replay_session("yoco")
    for mode in ("final+yoco", "yoco+final"):
        turns = replay_session(mode)
        for i, ((P, O, hit, C, T), (_, _, h_yoco, _, _)) in enumerate(zip(turns, yoco)):
            _yoco_expect(hit, SESSION_3B75[i + 1][0], C, T)
            assert hit >= h_yoco
        before = sum(own_output_reprefill(P, T, h) for P, O, h, C, T in yoco)
        after = sum(own_output_reprefill(P, T, h) for P, O, h, C, T in turns)
        assert after < before / 2, (mode, before, after)


def test_residue_sweep():
    rng = random.Random(11)
    cases = []
    for P in range(8000, 8000 + 128, 9):
        for O in (1, 2, 50, 63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 300, 700):
            for S in (13, 150, 400):
                cases.append((P, O, S))
    rng.shuffle(cases)
    for mode in ("final", "final+yoco"):
        with engine(mode) as eng:
            for j, (P, O, S) in enumerate(cases[:240]):
                base = [rng.randrange(1, 100000) for _ in range(P)]
                req = Req(f"{mode}-{j}", base)
                eng.run(req, [rng.randrange(1, 100000) for _ in range(O)], rng)
                C, T = finish_computed(req), req.num_tokens
                nxt = req.all_token_ids + [rng.randrange(1, 100000) for _ in range(S)]
                hit = eng.hit(Req("probe", nxt))
                if mode == "final":
                    assert hit == b_end(C), (P, O, S, hit, b_end(C))
                else:
                    _yoco_expect(hit, len(nxt), C, T)
            assert eng.max_held_over_cap <= 0, (mode, eng.max_held_over_cap)


def test_hold_is_accounted_in_the_admission_cap():
    caps = {}
    for mode in ("stock", "final", "final+yoco"):
        with engine(mode) as eng:
            caps[mode] = {(m.extra_retained_tokens, m._max_admission_blocks_per_request)
                          for m in eng.swa}
            # a long chunked prefill and a decode, cap checked at every allocation
            rng = random.Random(3)
            req = Req("long", [rng.randrange(1, 100000) for _ in range(20000)])
            eng.run(req, [rng.randrange(1, 100000) for _ in range(600)], rng)
            assert eng.max_held_over_cap <= 0, (mode, eng.max_held_over_cap)
    (e0, cap0), = caps["stock"]
    (e1, cap1), = caps["final"]
    (e2, cap2), = caps["final+yoco"]
    assert e0 == 0 and e1 == 192 and e2 == 192 + KEEP, caps
    assert cap1 == cap0 + 3 and cap2 == cap0 + 6, caps


def _swa_tail_blocks(eng: Engine, req: Req) -> dict[int, list[int]]:
    """Per SWA group: indices of the request's blocks that carry a block hash."""
    out = {}
    for mgr in eng.swa:
        blocks = mgr.req_to_blocks.get(req.request_id, [])
        out[mgr.kv_cache_group_id] = [i for i, b in enumerate(blocks)
                                      if not b.is_null and b.block_hash is not None]
    return out


def _tails(C: int, T: int, yoco: bool) -> set[int]:
    """Block indices of the final checkpoints' tails (window + peek block)."""
    out = set()
    for b in [b_end(C)] + ([b_yoco(T)] if yoco else []):
        if b >= ALIGN and b + PEEK <= C // BS * BS:
            out |= set(range(b // BS + 1 - 3, b // BS + 1))
    return out


def test_bounded_and_below_processed():
    rng = random.Random(5)
    for mode in ("final", "final+yoco"):
        with engine(mode) as eng:
            sig = inspect.signature(eng.kvm.block_pool.cache_full_blocks)
            for in_flight in (False, True, "chunk"):
                req = Req(f"r{in_flight}", [rng.randrange(1, 100000) for _ in range(9000)])
                out = [rng.randrange(1, 100000) for _ in range(5000)]
                if in_flight == "chunk":     # aborted while a prefill chunk is in flight
                    eng.prefill(req, rng, out, stop_after=3)
                    eng.in_flight_chunk(req)
                else:
                    eng.prefill(req, rng, out)
                    eng.decode(req, rng, out, steps=1500)
                if in_flight is True:
                    eng.in_flight_step(req)
                before = _swa_tail_blocks(eng, req)
                C, T = finish_computed(req), req.num_tokens
                pool = eng.kvm.block_pool
                hashed: dict[int, set[int]] = {}
                orig = pool.cache_full_blocks

                def spy(*a, **kw):
                    args = sig.bind(*a, **kw).arguments
                    lo, hi, mask = (args["num_cached_blocks"], args["num_full_blocks"],
                                    args["block_mask"])
                    assert mask is not None and len(mask) == hi - lo
                    idx = {i for i in range(lo, hi) if mask[i - lo]}
                    assert not any(args["blocks"][i].block_hash is not None for i in idx)
                    hashed.setdefault(args["kv_cache_group_id"], set()).update(idx)
                    return orig(*a, **kw)

                pool.cache_full_blocks = spy
                eng.kvm.free(req)
                pool.cache_full_blocks = orig
                want = _tails(C, T, "yoco" in mode)
                assert want, (C, T)
                for mgr in eng.swa:
                    gid = mgr.kv_cache_group_id
                    new = hashed.get(gid, set())
                    assert new <= want, (mode, in_flight, gid, sorted(new), sorted(want))
                    assert want <= new | set(before[gid]), (mode, gid, sorted(want))
                    assert all((i + 1) * BS <= C for i in new), (mode, in_flight, C)
                assert set(hashed) <= {m.kv_cache_group_id for m in eng.swa}


def test_abort_mid_prefill_under_yoco():
    rng = random.Random(13)
    for mode in ("final+yoco", "yoco+final"):
        with engine(mode) as eng:
            req = Req("abort", [rng.randrange(1, 100000) for _ in range(9000)])
            eng.prefill(req, rng, [1], stop_after=3)
            C = req.num_computed_tokens
            assert b_yoco(req.num_tokens) > C          # only B_end is complete
            eng.kvm.free(req)
            retry = Req("abort2", req.all_token_ids)
            assert eng.hit(retry) == b_end(C), (mode, eng.hit(retry), C, b_end(C))


def test_preemption_and_abort():
    rng = random.Random(9)
    with engine("final") as eng:
        # preempted mid-decode: the resumed request hits the preemption point
        req = Req("pre", [rng.randrange(1, 100000) for _ in range(6000)])
        out = [rng.randrange(1, 100000) for _ in range(900)]
        eng.prefill(req, rng, out)
        eng.decode(req, rng, out, steps=150)
        C = finish_computed(req)
        eng.kvm.free(req)               # Scheduler._preempt_request frees before the reset
        resumed = Req("pre2", req.all_token_ids)
        assert eng.hit(resumed) == b_end(C), (eng.hit(resumed), b_end(C))
        # aborted in the middle of its chunked prefill: freed without error
        req = Req("abort", [rng.randrange(1, 100000) for _ in range(9000)])
        eng.prefill(req, rng, [1], stop_after=3)
        C = req.num_computed_tokens
        eng.kvm.free(req)
        retry = Req("abort2", req.all_token_ids)
        assert eng.hit(retry) == b_end(C), (eng.hit(retry), C, b_end(C))
        # never scheduled
        eng.kvm.free(Req("never", [1] * 300))


def test_deferred_free_path():
    """Scheduler._free_request_blocks under async scheduling: pop the blocks while a step may
    still be in flight, return them to the pool once its fence has passed."""
    rng = random.Random(17)
    for mode in ("final", "final+yoco"):
        with engine(mode) as eng:
            req = Req("deferred", [rng.randrange(1, 100000) for _ in range(7000)])
            out = [rng.randrange(1, 100000) for _ in range(800)]
            eng.prefill(req, rng, out)
            eng.decode(req, rng, out, steps=120)
            eng.in_flight_step(req)
            C, T = finish_computed(req), req.num_tokens
            blocks = eng.kvm.pop_blocks_for_free(req)
            nxt = req.all_token_ids + [rng.randrange(1, 100000) for _ in range(300)]
            # popped but not yet returned: the tails are already hashed and hittable
            hit_before = eng.hit(Req("probe1", nxt))
            eng.kvm.block_pool.free_blocks(reversed(blocks))
            hit_after = eng.hit(Req("probe2", nxt))
            assert hit_before == hit_after, (mode, hit_before, hit_after)
            if mode == "final":
                assert hit_after == b_end(C), (hit_after, C, b_end(C))
            else:
                _yoco_expect(hit_after, len(nxt), C, T)


def test_patched_files_refuse_without_the_env():
    dst = tree("final")
    with environ():
        try:
            load(dst)
        except RuntimeError as exc:
            assert "DSV41_SWA_FINAL_WINDOW" in str(exc)
        else:
            raise AssertionError("patched files imported with the feature off")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"SWA final-window simulation OK ({len(tests)} tests)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
