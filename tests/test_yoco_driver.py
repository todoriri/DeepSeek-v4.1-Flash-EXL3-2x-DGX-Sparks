#!/usr/bin/env python3
"""CPU run of the YOCO prefill driver (overlay/dsv41_yoco.py) against fake vLLM modules.

The real driver only runs inside the image (V2 runner, GB10). This file executes every
line of it on CPU. Coverage:

  * make_driver: config checks (the KEEP floor, the boundary layer's role);
  * attach: when a plan is made; compact builder inputs; placeholder slots; the window
    clamp; untouched entries;
  * begin / split / after_split / finish: the boundary layer's KV work runs on all rows
    under the full metadata; the decoder gets compact rows and metadata; outputs are
    scattered back; the forward context and the skip flag are restored.

It checks the driver's own logic against the attribute names read from the image's
source (see docs/yoco-prefill-truncation.md). It does not validate the real kernels.
"""
from __future__ import annotations

import enum
import importlib.util
import logging
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
W, H = 128, 16


# ------------------------------------------------------------------ fake vllm modules
class CUDAGraphMode(enum.Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2


_FC: list = [None]


def _install_fake_vllm():
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    mod("vllm")
    mod("vllm.config")
    mod("vllm.config.compilation", CUDAGraphMode=CUDAGraphMode)
    mod("vllm.forward_context", get_forward_context=lambda: _FC[0],
        is_forward_context_available=lambda: _FC[0] is not None)
    mod("vllm.platforms", current_platform=SimpleNamespace(
        is_device_capability_family=lambda fam: fam == 120))
    mod("vllm.logger", init_logger=logging.getLogger)
    for name in ("vllm.v1", "vllm.v1.worker", "vllm.v1.worker.gpu"):
        mod(name)
    mod("vllm.v1.worker.gpu.attn_utils", build_attn_metadata=fake_build_attn_metadata,
        prepare_kernel_block_sizes=lambda cfg, groups: [64] * len(groups))


# ---------------------------------------------------------------- fake metadata layer
@dataclass
class SWAMeta:
    num_decode_tokens: int
    num_prefill_tokens: int
    decode_swa_indices: torch.Tensor
    decode_swa_lens: torch.Tensor
    prefill_swa_indices: torch.Tensor
    prefill_swa_lens: torch.Tensor
    common: dict


class FakeBuilder:
    """SWA: image-kernel semantics with slot = position + group*10**6 (no paging)."""

    def __init__(self, kind, gid, kbs):
        self.kind, self.gid, self.kbs = kind, gid, kbs
        self.calls = []

    def build(self, common):
        self.calls.append(common)
        if self.kind != "swa":
            return SimpleNamespace(kind=self.kind, common=common)
        qsl = common["query_start_loc_cpu"].numpy()
        q = np.diff(qsl)
        nd_req = 0 if q.max() > 4 and q[0] > 4 else int(np.argmax(q > 4)) if q.max() > 4 else len(q)
        nd = int(qsl[nd_req])
        pos = common["positions"].numpy()
        slots = common["slot_mapping"].numpy()
        n = len(pos)
        idx = torch.full((n, 1, W), -1, dtype=torch.int32)
        lens = torch.zeros(n, dtype=torch.int32)
        for i in range(n):
            if slots[i] < 0:
                continue
            start = max(pos[i] - (W - 1), 0)
            ln = pos[i] + 1 - start
            lens[i] = ln
            idx[i, 0, :ln] = torch.arange(start, pos[i] + 1) + self.gid * 10**6
        return SWAMeta(nd, n - nd, idx[:nd], lens[:nd], idx[nd:], lens[nd:], common)


@dataclass
class FakeGroup:
    backend: str
    layer_names: list
    kv_cache_spec: str
    kv_cache_group_id: int
    metadata_builders: list = field(default_factory=list)

    def create_metadata_builders(self, vllm_config, device, kernel_block_size=None,
                                 num_metadata_builders=1):
        kind = "swa" if self.kv_cache_spec == "swa" else self.kv_cache_spec
        self.metadata_builders = [FakeBuilder(kind, self.kv_cache_group_id, kernel_block_size)
                                  for _ in range(num_metadata_builders)]

    def get_metadata_builder(self, i=0):
        return self.metadata_builders[i]


def fake_build_attn_metadata(*, attn_groups, num_reqs, num_tokens, query_start_loc_gpu,
                             query_start_loc_cpu, max_query_len, seq_lens, max_seq_len,
                             block_tables, slot_mappings, kv_cache_config,
                             seq_lens_cpu_upper_bound=None, dcp_local_seq_lens=None,
                             positions=None, is_prefilling=None, rswa_prefix_lens=None):
    out = {}
    assert slot_mappings.shape[1] == num_tokens == positions.shape[0]
    for gid, groups in enumerate(attn_groups):
        for g in groups:
            common = dict(query_start_loc_cpu=query_start_loc_cpu, positions=positions,
                          slot_mapping=slot_mappings[gid], num_reqs=num_reqs,
                          num_tokens=num_tokens, max_query_len=max_query_len,
                          seq_lens=seq_lens, max_seq_len=max_seq_len)
            md = g.get_metadata_builder(0).build(common)
            for name in g.layer_names:
                out[name] = md
    return out


# ------------------------------------------------------------------- fake model tree
class _Cfg:
    num_hidden_layers = 40
    sliding_window = 128
    compress_ratios = [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]
    kv_source_layer_ids = [2, 8, 14, 20]
    index_source_layer_ids = [2, 8, 14, 20, 24, 28, 32, 36]
    candidate_source_layer_id = 20
    engram_layer_ids = [1, 14]
    dspark_target_layer_ids = [37, 38, 39]


LOG: list = []


class FakeCompressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fused_wkv_wgate = torch.nn.Linear(H, 6, bias=False)

    def forward(self, kv_score, positions):
        LOG.append(("compress", kv_score.shape[0], positions.shape[0], id(_FC[0].attn_metadata)))
        assert kv_score.dtype == torch.float32
        return kv_score[:, :3]

    def insert_cache(self, latent, positions, rot):
        LOG.append(("insert", latent.shape[0], id(_FC[0].attn_metadata)))


class FakeIndexer(torch.nn.Module):
    def __init__(self, layer, owns_k):
        super().__init__()
        self.owns_k = owns_k
        self.k_cache = SimpleNamespace(prefix=f"lm.layers.20.attn.indexer.k_cache")

    def _produce_k(self, latent, positions, rot):
        LOG.append(("index_k", latent.shape[0], id(_FC[0].attn_metadata)))


class FakeAttn(torch.nn.Module):
    def __init__(self, i):
        super().__init__()
        self.swa_cache_layer = SimpleNamespace(prefix=f"lm.layers.{i}.attn.swa_cache")
        self.compressor = FakeCompressor() if i in (2, 8, 14, 20) else None
        self.indexer = (FakeIndexer(i, owns_k=i in (2, 8, 14, 20))
                        if i in _Cfg.index_source_layer_ids else None)
        self.max_image_tokens = 0
        self.rotary_emb = self.indexer_rotary_emb = object()


class FakeLayer(torch.nn.Module):
    def __init__(self, i):
        super().__init__()
        self.attn = FakeAttn(i)
        self.engram = object() if i in (1, 14) else None


class DeepseekV4Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _Cfg()
        self.layers = torch.nn.ModuleList(FakeLayer(i) for i in range(40))
        self.start_layer, self.end_layer = 0, 40
        self.aux_hidden_state_layers = (37, 38, 39)
        self.use_sequence_parallel = False


def _load_helper():
    spec = importlib.util.spec_from_file_location("dsv41_yoco", ROOT / "overlay" / "dsv41_yoco.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dsv41_yoco"] = mod
    spec.loader.exec_module(mod)
    return mod


_install_fake_vllm()
yoco = _load_helper()
# torch.mm(..., out_dtype=) is CUDA-only; the image runs it (like vLLM's own GEMM) on GPU.
_real_mm = torch.mm
MM_CALLS: list = []


def _mm(a, b, *, out_dtype=None):
    MM_CALLS.append(out_dtype)
    r = _real_mm(a.float(), b.float())
    return r if out_dtype is None else r.to(out_dtype)


torch.mm = _mm
VCFG = SimpleNamespace(speculative_config=SimpleNamespace(num_speculative_tokens=3))


def _groups():
    """15 SWA groups {i, i+15, i+30} (+ draft 40-42) and the layer-20 compressed/index group."""
    groups = []
    for g in range(15):
        names = [f"lm.layers.{i}.attn.swa_cache" for i in (g, g + 15, g + 30) if i < 43]
        groups.append([FakeGroup("swa", names, "swa", g)])
    groups.append([FakeGroup("mla", ["lm.layers.20.attn"], "mla", 15),
                   FakeGroup("idx", ["lm.layers.20.attn.indexer.k_cache"], "idx", 15)])
    for row in groups:
        for g in row:
            g.create_metadata_builders(None, "cpu", 64)
    return groups


def _batch(reqs):
    """reqs: (start, q_len, prefill_len, prefilling)."""
    qsl = np.concatenate([[0], np.cumsum([r[1] for r in reqs])]).astype(np.int32)
    pos = np.concatenate([np.arange(s, s + q) for s, q, _, _ in reqs])
    n = int(qsl[-1])
    return SimpleNamespace(
        has_prefill=any(r[3] for r in reqs), num_reqs=len(reqs), query_start_loc_np=qsl,
        num_computed_prefill_tokens_np=np.array([r[0] for r in reqs]),
        prefill_len_np=np.array([r[2] for r in reqs]),
        is_prefilling_np=np.array([r[3] for r in reqs]),
        num_tokens_after_padding=n, query_start_loc=torch.from_numpy(qsl),
        positions=torch.from_numpy(pos).long(),
        seq_lens=torch.tensor([s + q for s, q, _, _ in reqs]),
        seq_lens_cpu_upper_bound=torch.tensor([s + q for s, q, _, _ in reqs]),
        dcp_local_seq_lens=None, prompt_lens=None)


def _env(**kv):
    for k in ("DSV41_YOCO_PREFILL", "DSV41_YOCO_KEEP_TOKENS"):
        os.environ.pop(k, None)
    os.environ.update(kv)


def test_make_driver_checks():
    model = torch.nn.Module()
    model.language_model = torch.nn.Module()
    model.language_model.model = DeepseekV4Model()
    _env(DSV41_YOCO_PREFILL="compact")
    d = yoco.make_driver(VCFG, model, torch.device("cpu"))
    assert d.boundary == 20 and d.keep == 140
    assert d.decoder_swa_prefixes == [f"lm.layers.{i}.attn.swa_cache" for i in range(20, 40)]
    assert d.index_prefix == "lm.layers.20.attn.indexer.k_cache"
    _env(DSV41_YOCO_PREFILL="compact", DSV41_YOCO_KEEP_TOKENS="131")
    try:
        yoco.make_driver(VCFG, model, torch.device("cpu"))
    except RuntimeError as e:
        assert "131" in str(e)
    else:
        raise AssertionError("KEEP below window + backtrack accepted")
    _env(DSV41_YOCO_PREFILL="off")
    try:
        yoco.make_driver(VCFG, model, torch.device("cpu"))
    except RuntimeError:
        pass
    else:
        raise AssertionError("patched vLLM with the feature off must refuse")
    _env(DSV41_YOCO_PREFILL="compact")
    model.language_model.model.layers[20].attn.indexer.owns_k = False
    try:
        yoco.make_driver(VCFG, model, torch.device("cpu"))
    except RuntimeError:
        pass
    else:
        raise AssertionError("boundary without index-K accepted")


def _driver():
    model = DeepseekV4Model()
    _env(DSV41_YOCO_PREFILL="compact")
    return yoco.make_driver(VCFG, model, torch.device("cpu")), model


def test_attach_passthrough_cases():
    d, _ = _driver()
    full = {"x": 1}
    groups = _groups()
    sm = torch.zeros(16, 1284, dtype=torch.int64)
    b = _batch([(4999, 4, 5003, False), (8000, 1280, 9300, True)])
    for mode, cap in ((CUDAGraphMode.PIECEWISE, False), (CUDAGraphMode.FULL, False),
                      (CUDAGraphMode.NONE, True)):
        assert d.attach(full, b, mode, (), sm, groups, None, cap) is full
    short = _batch([(0, 100, 100, True)])
    assert d.attach(full, short, CUDAGraphMode.NONE, (), sm[:, :100], groups, None, False) is full
    dec = _batch([(4999, 4, 5003, False)])
    dec.has_prefill = False
    assert d.attach(full, dec, CUDAGraphMode.NONE, (), sm[:, :4], groups, None, False) is full


def test_attach_builds_compact_decoder_metadata():
    d, _ = _driver()
    groups = _groups()
    # decode (4 rows) | prefill chunk [8000, 9280) of 9300 -> rect [9160, 9300)
    # | chunk [0, 1000) of 50000 -> placeholder (row 999)
    b = _batch([(4999, 4, 5003, False), (8000, 1280, 9300, True), (0, 1000, 50000, True)])
    n = b.num_tokens_after_padding
    sm = torch.arange(16 * n, dtype=torch.int64).view(16, n) + 7
    full = {name: f"full:{name}" for row in groups for g in row for name in g.layer_names}
    out = d.attach(full, b, CUDAGraphMode.NONE, ("bt",), sm, groups, "kvc", False)
    assert isinstance(out, yoco.YocoAttnMetadata) and dict(out) == full
    plan = out.dsv41_yoco
    keep = plan.keep_idx.tolist()
    assert keep == [0, 1, 2, 3] + list(range(4 + 1160, 4 + 1280)) + [4 + 1280 + 999]
    assert plan.num_rows == n and plan.boundary == 20
    dec = plan.decoder_metadata
    for name, val in full.items():
        layer = int(name.split(".")[2])
        if name in d.needed_names:
            assert not isinstance(dec[name], str), name
        else:
            assert dec[name] == val, name        # encoder SWA, layer-20 compressed: full
        if layer < 20 and name.endswith("swa_cache"):
            assert dec[name] == val
    # primary builders untouched; shadow builders got the compact batch
    assert all(not g.get_metadata_builder(0).calls for row in groups for g in row)
    md = dec["lm.layers.21.attn.swa_cache"]
    c = md.common
    assert c["num_tokens"] == len(keep) and c["query_start_loc_cpu"].tolist() == [0, 4, 124, 125]
    assert c["positions"].tolist() == [4999, 5000, 5001, 5002] + list(range(9160, 9280)) + [999]
    assert c["slot_mapping"][-1].item() == -1, "placeholder row stores nothing"
    assert c["slot_mapping"][:-1].tolist() == sm[6, plan.keep_idx[:-1]].tolist()
    # window clamp: decode rows untouched, rectangle rows start at 9160, placeholder empty
    assert md.decode_swa_lens.tolist() == [128] * 4
    lens = md.prefill_swa_lens.tolist()
    want = [p - max(p - 127, 9160) + 1 for p in range(9160, 9280)] + [0]
    assert lens == want, (lens[:4], want[:4])
    first = md.prefill_swa_indices[0, 0, :2].tolist()
    assert first == [9160 + 6 * 10**6, -1]
    row = md.prefill_swa_indices[119, 0]   # pos 9279: [9160, 9279]
    assert row[:120].tolist() == [p + 6 * 10**6 for p in range(9160, 9280)]
    assert bool((row[120:] == -1).all())
    # every SWA group's object clamped exactly once (shared by 20/35 etc.)
    a, b2 = dec["lm.layers.20.attn.swa_cache"], dec["lm.layers.35.attn.swa_cache"]
    assert a is b2 and a.prefill_swa_lens.tolist() == want
    idx = dec["lm.layers.20.attn.indexer.k_cache"]
    assert idx.kind == "idx" and idx.common["num_tokens"] == len(keep)
    # second step reuses the shadow builders
    shadows_before = dict(d._shadow)
    d.attach(full, b, CUDAGraphMode.NONE, ("bt",), sm, groups, "kvc", False)
    assert d._shadow == shadows_before


def test_forward_flow():
    d, model = _driver()
    groups = _groups()
    b = _batch([(4999, 4, 5003, False), (8000, 1280, 9300, True)])
    n = b.num_tokens_after_padding
    sm = torch.arange(16 * n, dtype=torch.int64).view(16, n)
    full = {name: "full" for row in groups for g in row for name in g.layer_names}
    out = d.attach(full, b, CUDAGraphMode.NONE, (), sm, groups, None, False)
    _FC[0] = SimpleNamespace(attn_metadata=out, cudagraph_runtime_mode=CUDAGraphMode.NONE)
    pos = b.positions
    step = yoco.begin(model, pos)
    assert step is out.dsv41_yoco and step.full_metadata is out
    assert step.split_kwargs(19) == {} and "dsv41_yoco_split" in step.split_kwargs(20)
    layer = model.layers[20]
    x = torch.randn(n, H)
    residual, post_mix = torch.randn(n, 4, H), torch.randn(n, 4, 1)
    res_mix, attn_pre = torch.randn(n, 4, 4), torch.randn(n, 4)
    ids = torch.arange(n) + 100
    LOG.clear()
    got = step.split(layer, x, residual, post_mix, res_mix, attn_pre, pos, ids)
    full_id = id(out)
    assert MM_CALLS and MM_CALLS[-1] is torch.float32, 'dense compressor GEMM in fp32'
    assert LOG == [("compress", n, n, full_id), ("insert", n, full_id), ("index_k", n, full_id)]
    k = step.keep_idx
    for g, src in zip(got[:5], (x, residual, post_mix, res_mix, attn_pre)):
        assert torch.equal(g, src[k])
    assert torch.equal(got[5], pos[k]) and torch.equal(got[6], ids[k])
    assert _FC[0].attn_metadata is step.decoder_metadata
    assert layer.attn._dsv41_yoco_skip_kv is True
    p2, i2 = step.after_split()
    assert layer.attn._dsv41_yoco_skip_kv is False and torch.equal(p2, pos[k])
    hid = torch.randn(len(k), H)
    aux = [torch.randn(len(k), H) for _ in range(3)]
    buf = torch.full((2000, 4 * H), 9.0)
    step.scatter_rows_into(buf, torch.ones(len(k), 4 * H))
    assert buf[:n].sum().item() == len(k) * 4 * H and buf[n:].eq(9).all()
    h_full, aux_full = step.finish(hid, aux)
    assert _FC[0].attn_metadata is out
    assert h_full.shape == (n, H) and torch.equal(h_full[k], hid)
    mask = torch.ones(n, dtype=torch.bool)
    mask[k] = False
    assert h_full[mask].abs().sum().item() == 0
    assert all(torch.equal(a[k], s) for a, s in zip(aux_full, aux))
    try:
        yoco.begin(model, pos)
    except RuntimeError:
        pass
    else:
        raise AssertionError("plan reused")
    # wrong row count / cudagraph forward
    out2 = d.attach(full, b, CUDAGraphMode.NONE, (), sm, groups, None, False)
    _FC[0] = SimpleNamespace(attn_metadata=out2, cudagraph_runtime_mode=CUDAGraphMode.NONE)
    for bad_pos, mode in ((pos[:-1], CUDAGraphMode.NONE), (pos, CUDAGraphMode.PIECEWISE)):
        _FC[0].cudagraph_runtime_mode = mode
        try:
            yoco.begin(model, bad_pos)
        except RuntimeError:
            pass
        else:
            raise AssertionError(mode)
    _FC[0] = SimpleNamespace(attn_metadata={"plain": 1})
    assert yoco.begin(model, pos) is None
    _FC[0] = None
    assert yoco.begin(model, pos) is None


def test_produce_kv_full_exl3_branch():
    attn = FakeAttn(20)
    layer = attn.compressor.fused_wkv_wgate
    calls = []

    class QM:
        def apply(self, lyr, x):
            calls.append(x.shape)
            return torch.zeros(x.shape[0], 6, dtype=torch.float16)

    del layer.weight
    layer.quant_method = QM()
    _FC[0] = SimpleNamespace(attn_metadata={})
    LOG.clear()
    yoco.produce_kv_full(attn, torch.arange(5), torch.randn(5, H))
    assert calls == [torch.Size([5, H])] and [e[0] for e in LOG] == ["compress", "insert", "index_k"]
    _FC[0] = None


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"YOCO driver (fake vLLM) OK ({len(tests)} tests)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
