"""Decoder-SWA bounded replay ("YOCO prefill truncation") for DeepSeek-V4.1 on vLLM.

DeepSeek-V4.1-Flash is CED 20+20. Layers 0-19 form a causal encoder. Layers 20-39 form
a decoder whose compressed and index KV comes from the stream leaving the encoder:
layer 20 is the only kv-source after layer 14, and it owns the index-K cache. During
prefill, a decoder layer is therefore only needed for its own sliding-window KV (window
128), and that KV is only read by decode.

With DSV41_YOCO_PREFILL=compact, a prefill runs layers 0-19 over every scheduled
token. Layer 20 also writes its compressed and index KV for every token. Layer 20's
attention and MoE, and all of layers 21-39, run only on the rows with
position >= N - KEEP, where N = prefill_len. The decoder SWA keys are clipped to
[max(pos - 127, N - KEEP), pos]. Decode is unchanged.

The result is approximate, which DeepSeek accepts in production: the replayed rows
attend only within the replayed segment through the SWA. The long-range path, which
goes through the compressed KV and the indexer, is complete. Prompt logprobs are
refused.

Ported from DeepInfer (deepseek-ai/DeepInfer @ 774ce24, Apache License 2.0;
DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact). Sources:
  * deepinfer/environ.py;
  * csrc/engine/input_metadata/prepare_tensor_input_v4.cpp (rectangle
    [N - K, N) ∩ chunk, one placeholder row for a chunk wholly before it, SWA store
    only inside it);
  * deepinfer/model_executor/{input_metadata,v4_modules}/yoco.py and decoder_layer.py
    (layer-20 split: full-row KV production, then row compaction);
  * deepinfer/worker/model_runner.py (KEEP >= window + max backtracking).
The vLLM wiring targets the image's V2 runner (vLLM 0.1.dev20904+g179dd0fa9).

Loaded from /opt/dsv41 by the footers that overlay/patch_yoco_prefill.py adds. The pure
functions (Settings, build_keep_plan, clamp_swa_rows, decoder_boundary, hit math) are
CPU-testable: tests/test_yoco_metadata.py.
"""
from __future__ import annotations

import dataclasses
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

MODES = ("off", "compact")
DEFAULT_KEEP = 140
MAX_KEEP = 1048576
_DIGITS = re.compile(r"[0-9]+")


# ----------------------------------------------------------------------------- settings
@dataclass(frozen=True)
class Settings:
    mode: str
    keep: int

    @property
    def enabled(self) -> bool:
        return self.mode == "compact"

    @classmethod
    def from_env(cls, environ: Any = None) -> "Settings":
        env = os.environ if environ is None else environ
        mode = env.get("DSV41_YOCO_PREFILL", "") or "off"
        if mode not in MODES:
            raise ValueError(f"DSV41_YOCO_PREFILL must be one of {MODES} (got: {mode!r})")
        raw = env.get("DSV41_YOCO_KEEP_TOKENS", "") or str(DEFAULT_KEEP)
        if not _DIGITS.fullmatch(raw) or not 1 <= int(raw) <= MAX_KEEP:
            raise ValueError(
                f"DSV41_YOCO_KEEP_TOKENS must be an integer in [1, {MAX_KEEP}] (got: {raw!r})"
            )
        return cls(mode, int(raw))

    def require_enabled(self) -> "Settings":
        if not self.enabled:
            raise RuntimeError(
                "[dsv41-yoco] vLLM carries the YOCO prefill patch but DSV41_YOCO_PREFILL="
                f"{self.mode!r}; start a fresh container (start.sh only patches when compact)"
            )
        return self


def min_keep_tokens(window: int, num_speculative_tokens: int) -> int:
    """DeepInfer's KEEP >= window + max backtracking, backtracking = spec tokens + 1."""
    return int(window) + int(num_speculative_tokens) + 1


# ------------------------------------------------------------------- prefix-cache math
def max_cache_hit_length(num_tokens: int, keep: int) -> int:
    """Cap on a prefix-cache hit (stock: num_tokens - 1). The whole rectangle
    [N - KEEP, N) must be recomputed so no decoder-SWA row is read from a cached
    block (cached decoder SWA is garbage below the writer's own rectangle)."""
    return max(0, int(num_tokens) - max(int(keep), 1))


def capped_hit(num_tokens: int, keep: int, align: int, peek: int, cached: int) -> int:
    """Model of the coordinator's hit under the cap: the boundary is aligned and the
    EAGLE peek block past it must still fit under the cap and inside the cache."""
    limit = min(int(cached), max_cache_hit_length(num_tokens, keep)) - int(peek)
    return max(0, limit // align * align)


def yoco_reachable_boundary(num_prompt_tokens: int, keep: int, block_size: int,
                            use_eagle: bool) -> int:
    """Extra sparse-retention boundary: a retry of the same prompt hits at
    align_down(boundary), i.e. recomputes at most KEEP + peek + alignment - 1 tokens."""
    return int(num_prompt_tokens) - int(keep) - (int(block_size) if use_eagle else 0)


# ------------------------------------------------------------------------- topology
def decoder_boundary(cfg: Any) -> int:
    """First decoder ("YOCO consumer") layer = first layer with compress ratio 1
    (DeepInfer hugging_face/config.py _n_prefill_layer). Raises if the checkpoint's
    sharing topology does not allow the split (DeepInfer deepseek.py:128-138,
    cross_layer.py:115-116)."""
    n = int(cfg.num_hidden_layers)
    ratios = list(getattr(cfg, "compress_ratios", None) or [])[:n]
    try:
        boundary = ratios.index(1)
    except ValueError:
        raise ValueError("no compress-ratio-1 layer: not a CED (YOCO) checkpoint") from None
    if any(int(r) != 1 for r in ratios[boundary:]):
        raise ValueError(f"layers {boundary}..{n - 1} are not all compress-ratio 1")
    kv_src = [int(i) for i in (getattr(cfg, "kv_source_layer_ids", None) or [])]
    if boundary not in kv_src or max(kv_src) != boundary:
        raise ValueError(f"layer {boundary} must be the last kv-source layer (got {kv_src})")
    idx_src = [int(i) for i in (getattr(cfg, "index_source_layer_ids", None) or [])]
    if boundary not in idx_src:
        raise ValueError(f"layer {boundary} must be an index-source layer (got {idx_src})")
    cand = int(getattr(cfg, "candidate_source_layer_id", -1))
    if 0 <= cand < boundary:
        raise ValueError(f"candidate source {cand} is on the encoder side")
    engram = [int(i) for i in (getattr(cfg, "engram_layer_ids", None) or [])]
    if any(i >= boundary for i in engram):
        raise ValueError(f"Engram layers {engram} must all precede layer {boundary}")
    targets = [int(i) for i in (getattr(cfg, "dspark_target_layer_ids", None) or [])]
    if any(t <= boundary for t in targets):
        raise ValueError(f"DSpark target layers {targets} must all follow layer {boundary}")
    return boundary


# --------------------------------------------------------------------------- row plan
@dataclass
class KeepPlan:
    keep_idx: np.ndarray      # [n_kept] int64, rows of the full batch (sorted)
    compact_qsl: np.ndarray   # [num_reqs + 1] int32
    lower_bound: np.ndarray   # [n_kept] int64, lowest visible SWA position (0 = none)
    placeholder: np.ndarray   # [n_kept] bool, row kept only to keep its request alive
    needed: bool              # some prefilling request has prefill_len > KEEP


def build_keep_plan(query_start_loc, start_pos, prefill_len, is_prefilling,
                    keep: int) -> KeepPlan:
    """Rows the decoder runs this step (DeepInfer prepare_tensor_input_v4.cpp:222-264).

    A prefilling request with N = prefill_len > KEEP keeps its rows with position
    >= N - KEEP; a chunk wholly before that keeps one placeholder row (its last, the
    logits row, so every request keeps >= 1 row and per-request metadata need no
    renumbering). Decoding requests and prompts <= KEEP keep every row. Kept rows are
    always a contiguous suffix of the request's scheduled rows."""
    qsl = np.asarray(query_start_loc, dtype=np.int64)
    num_reqs = len(qsl) - 1
    rows, lower, ph = [], [], []
    compact = np.zeros(num_reqs + 1, dtype=np.int32)
    needed = False
    for r in range(num_reqs):
        q0, q1 = int(qsl[r]), int(qsl[r + 1])
        n = q1 - q0
        assert n > 0, "every scheduled request has >= 1 row"
        lb, first, placeholder = 0, 0, False
        if bool(is_prefilling[r]) and int(prefill_len[r]) > keep:
            needed = True
            lb = int(prefill_len[r]) - keep
            first = min(max(lb - int(start_pos[r]), 0), n)
            if first == n:
                first, placeholder = n - 1, True
        kept = np.arange(q0 + first, q1, dtype=np.int64)
        rows.append(kept)
        lower.append(np.full(len(kept), lb, dtype=np.int64))
        ph.append(np.full(len(kept), placeholder, dtype=bool))
        compact[r + 1] = compact[r] + len(kept)
    cat = (lambda xs, dt: np.concatenate(xs) if xs else np.zeros(0, dtype=dt))
    return KeepPlan(cat(rows, np.int64), compact, cat(lower, np.int64), cat(ph, bool), needed)


def clamp_swa_rows(indices: torch.Tensor, lens: torch.Tensor, pos: torch.Tensor,
                   lower: torch.Tensor) -> None:
    """Raise each row's SWA window start to ``lower`` in place.

    ``indices`` ([R, 1, width] or [R, width]) and ``lens`` ([R]) are what the image's
    _compute_swa_indices_and_lens_kernel wrote: entry j is the slot of position
    start + j for j < len, then -1, with start = pos + 1 - len. Entries are shifted left
    by max(0, lower - start), clipped to len. Invalid rows (len 0) stay empty."""
    if indices.shape[0] == 0:
        return
    flat = indices.view(indices.shape[0], -1)
    width = flat.shape[1]
    dev = flat.device
    pos = pos.to(device=dev, dtype=torch.int64)
    lower = lower.to(device=dev, dtype=torch.int64)
    lens64 = lens.to(device=dev, dtype=torch.int64)
    start = pos + 1 - lens64
    shift = torch.minimum(torch.clamp(lower - start, min=0), lens64)
    new_len = lens64 - shift
    ar = torch.arange(width, device=dev, dtype=torch.int64)
    src = torch.clamp(ar.unsqueeze(0) + shift.unsqueeze(1), max=width - 1)
    vals = torch.gather(flat, 1, src)
    flat.copy_(torch.where(ar.unsqueeze(0) < new_len.unsqueeze(1), vals, torch.full_like(vals, -1)))
    lens.copy_(new_len.to(lens.dtype))


# ----------------------------------------------------------------- model-side driver
class YocoAttnMetadata(dict):
    """The step's attention-metadata dict, carrying this step's plan to the forward
    (set_forward_context stores the object as is)."""

    dsv41_yoco: "StepPlan | None" = None


def _log(msg: str) -> None:
    try:
        from vllm.logger import init_logger

        init_logger("vllm.dsv41_yoco").info(msg)
    except Exception:  # pragma: no cover
        print(msg, flush=True)


def _find_decoder_stack(model: torch.nn.Module) -> torch.nn.Module:
    found = [m for m in model.modules()
             if type(m).__name__ == "DeepseekV4Model" and hasattr(m, "layers")
             and hasattr(m, "aux_hidden_state_layers")]
    if len(found) != 1:
        raise RuntimeError(f"[dsv41-yoco] expected one DeepseekV4Model, found {len(found)}")
    return found[0]


def produce_kv_full(attn: torch.nn.Module, positions: torch.Tensor, x: torch.Tensor) -> None:
    """The boundary layer's kv-source work, on every row and with the full metadata:
    compressor scores (same GEMM as DeepseekV4Attention._run_parallel_input_projections,
    including the packed-EXL3 branch), compressed-KV store, index-K store."""
    comp = attn.compressor
    layer = comp.fused_wkv_wgate
    qm = getattr(layer, "quant_method", None)
    if qm is not None and not hasattr(layer, "weight"):
        kv_score = qm.apply(layer, x).float()
    else:
        kv_score = torch.mm(x, layer.weight.T, out_dtype=torch.float32)
    latent = comp(kv_score, positions)
    comp.insert_cache(latent, positions, attn.rotary_emb)
    attn.indexer._produce_k(latent, positions, attn.indexer_rotary_emb)


class StepPlan:
    """One truncated forward: the row gather at the boundary layer and the scatter of
    the final / DSpark aux hidden states back to the full batch."""

    def __init__(self, keep_idx: torch.Tensor, num_rows: int, boundary: int,
                 decoder_metadata: dict[str, Any]):
        self.keep_idx = keep_idx
        self.num_rows = num_rows          # rows of the forward (num_tokens_after_padding)
        self.boundary = boundary
        self.decoder_metadata = decoder_metadata
        self.full_metadata: Any = None
        self.positions: torch.Tensor | None = None
        self.input_ids: torch.Tensor | None = None
        self.used = False
        self._attn: torch.nn.Module | None = None

    def split_kwargs(self, idx: int) -> dict[str, Any]:
        return {"dsv41_yoco_split": self.split} if idx == self.boundary else {}

    def split(self, layer, x, residual, post_mix, res_mix, attn_pre, positions, input_ids):
        """Called by the boundary layer between its attention pre-mix and attention."""
        from vllm.forward_context import get_forward_context

        attn = layer.attn
        produce_kv_full(attn, positions, x)
        k = self.keep_idx
        x, residual = x.index_select(0, k), residual.index_select(0, k)
        post_mix, res_mix = post_mix.index_select(0, k), res_mix.index_select(0, k)
        attn_pre = attn_pre.index_select(0, k)
        positions = positions.index_select(0, k)
        if input_ids is not None:
            input_ids = input_ids.index_select(0, k)
        self.positions, self.input_ids = positions, input_ids
        get_forward_context().attn_metadata = self.decoder_metadata
        attn._dsv41_yoco_skip_kv = True
        self._attn = attn
        return x, residual, post_mix, res_mix, attn_pre, positions, input_ids

    def after_split(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self._attn is not None:
            self._attn._dsv41_yoco_skip_kv = False
        assert self.positions is not None, "boundary layer did not call split()"
        return self.positions, self.input_ids

    def _scatter(self, rows: torch.Tensor) -> torch.Tensor:
        out = rows.new_zeros((self.num_rows, *rows.shape[1:]))
        out.index_copy_(0, self.keep_idx, rows)
        return out

    def scatter_rows_into(self, buf: torch.Tensor, rows: torch.Tensor) -> None:
        buf[: self.num_rows].zero_()
        buf.index_copy_(0, self.keep_idx, rows)

    def finish(self, hidden_states: torch.Tensor, aux: list[torch.Tensor]):
        from vllm.forward_context import get_forward_context

        get_forward_context().attn_metadata = self.full_metadata
        if self._attn is not None:
            self._attn._dsv41_yoco_skip_kv = False
        return self._scatter(hidden_states), [self._scatter(a) for a in aux]


def begin(model: torch.nn.Module, positions: torch.Tensor) -> StepPlan | None:
    """DeepseekV4Model.forward: this step's plan, or None (decode, capture, profile)."""
    from vllm.config.compilation import CUDAGraphMode
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if not is_forward_context_available():
        return None
    fc = get_forward_context()
    step = getattr(fc.attn_metadata, "dsv41_yoco", None)
    if step is None:
        return None
    if getattr(fc, "cudagraph_runtime_mode", CUDAGraphMode.NONE) != CUDAGraphMode.NONE:
        raise RuntimeError("[dsv41-yoco] truncation plan in a cudagraph forward")
    if positions.shape[0] != step.num_rows:
        raise RuntimeError(f"[dsv41-yoco] plan for {step.num_rows} rows, forward has "
                           f"{positions.shape[0]}")
    if step.used:
        raise RuntimeError("[dsv41-yoco] truncation plan reused")
    step.used = True
    step.full_metadata = fc.attn_metadata
    return step


class YocoPrefill:
    """Per-rank driver owned by DeepseekV41ModelState (both TP ranks build identical
    plans from the identical scheduler output)."""

    def __init__(self, vllm_config: Any, model: torch.nn.Module, device: torch.device,
                 settings: Settings):
        from vllm.platforms import current_platform

        self.vllm_config = vllm_config
        self.device = device
        self.keep = settings.keep
        stack = _find_decoder_stack(model)
        cfg = stack.config
        self.boundary = decoder_boundary(cfg)
        n = int(cfg.num_hidden_layers)
        if (stack.start_layer, stack.end_layer) != (0, n):
            raise RuntimeError("[dsv41-yoco] pipeline parallelism is not supported")
        if getattr(stack, "use_sequence_parallel", False):
            raise RuntimeError("[dsv41-yoco] sequence-parallel MoE is not supported")
        if not current_platform.is_device_capability_family(120):
            raise RuntimeError("[dsv41-yoco] only the SM12x FlashInfer paged-direct SWA "
                               "path is supported (the FlashMLA gather path is not clamped)")
        spec = vllm_config.speculative_config
        k = int(spec.num_speculative_tokens) if spec is not None else 0
        need = min_keep_tokens(cfg.sliding_window, k)
        if self.keep < need:
            raise RuntimeError(f"[dsv41-yoco] DSV41_YOCO_KEEP_TOKENS={self.keep} < window "
                               f"{cfg.sliding_window} + backtrack {k + 1} = {need}")
        layer = stack.layers[self.boundary]
        attn = layer.attn
        if getattr(layer, "engram", None) is not None or attn.compressor is None \
                or attn.indexer is None or not attn.indexer.owns_k:
            raise RuntimeError(f"[dsv41-yoco] layer {self.boundary} is not a kv/index-K source")
        dec = [stack.layers[i].attn for i in range(self.boundary, n)]
        if any(getattr(a, "max_image_tokens", 0) for a in dec):
            raise RuntimeError("[dsv41-yoco] vision-widened SWA windows are not supported")
        self.decoder_swa_prefixes = [a.swa_cache_layer.prefix for a in dec]
        self.index_prefix = attn.indexer.k_cache.prefix
        self.needed_names = set(self.decoder_swa_prefixes) | {self.index_prefix}
        self._shadow: dict[int, tuple[Any, Any]] = {}
        self._kernel_block_sizes: list[int] | None = None
        _log(f"[dsv41-yoco] compact prefill truncation: keep={self.keep} decoder layers "
             f"{self.boundary}..{n - 1} (window {cfg.sliding_window}, spec tokens {k})")

    def _shadow_groups(self, attn_groups, kv_cache_config):
        """Second metadata builders for the groups the decoder reads (all SWA groups
        and the boundary layer's index-K group): the full step metadata lives in the
        primary builders' persistent buffers and must stay intact for layers 0-19."""
        if self._kernel_block_sizes is None:
            from vllm.v1.worker.gpu import attn_utils

            self._kernel_block_sizes = list(
                attn_utils.prepare_kernel_block_sizes(kv_cache_config, attn_groups))
        cuda = torch.device(self.device).type == "cuda"
        created, before = 0, torch.cuda.memory_allocated(self.device) if cuda else 0
        out = []
        for gid, groups in enumerate(attn_groups):
            row = []
            for g in groups:
                if self.needed_names.isdisjoint(g.layer_names):
                    continue
                entry = self._shadow.get(id(g))
                if entry is None or entry[0] is not g:
                    shadow = dataclasses.replace(g, metadata_builders=[])
                    kbs = self._kernel_block_sizes
                    shadow.create_metadata_builders(
                        vllm_config=self.vllm_config, device=self.device,
                        kernel_block_size=kbs[gid] if gid < len(kbs) else None,
                        num_metadata_builders=1)
                    entry = (g, shadow)
                    self._shadow[id(g)] = entry
                    created += 1
                row.append(entry[1])
            out.append(row)
        if created:
            mib = ((torch.cuda.memory_allocated(self.device) if cuda else 0) - before) / 2**20
            _log(f"[dsv41-yoco] created {created} shadow metadata builders (+{mib:.1f} MiB)")
        return out

    def attach(self, attn_metadata, input_batch, cudagraph_mode, block_tables,
               slot_mappings, attn_groups, kv_cache_config, for_capture: bool):
        from vllm.config.compilation import CUDAGraphMode
        from vllm.v1.worker.gpu.attn_utils import build_attn_metadata

        if (for_capture or cudagraph_mode != CUDAGraphMode.NONE
                or not isinstance(attn_metadata, dict) or not input_batch.has_prefill):
            return attn_metadata
        num_reqs = input_batch.num_reqs
        qsl = input_batch.query_start_loc_np[: num_reqs + 1]
        plan = build_keep_plan(qsl, input_batch.num_computed_prefill_tokens_np[:num_reqs],
                               input_batch.prefill_len_np[:num_reqs],
                               input_batch.is_prefilling_np[:num_reqs], self.keep)
        if not plan.needed:
            return attn_metadata
        num_rows = input_batch.num_tokens_after_padding
        dev = input_batch.query_start_loc.device
        keep_idx = torch.from_numpy(plan.keep_idx).to(dev)
        n_kept = int(plan.keep_idx.shape[0])
        qsl_c_cpu = torch.from_numpy(plan.compact_qsl)
        positions_c = input_batch.positions[:num_rows].index_select(0, keep_idx)
        slots_c = slot_mappings.index_select(1, keep_idx)
        if plan.placeholder.any():
            ph = torch.from_numpy(np.flatnonzero(plan.placeholder)).to(dev)
            slots_c[:, ph] = -1       # placeholder rows: no SWA store, no visible SWA
        compact = build_attn_metadata(
            attn_groups=self._shadow_groups(attn_groups, kv_cache_config),
            num_reqs=num_reqs,
            num_tokens=n_kept,
            query_start_loc_gpu=qsl_c_cpu.to(dev),
            query_start_loc_cpu=qsl_c_cpu,
            max_query_len=int(np.diff(plan.compact_qsl).max()),
            seq_lens=input_batch.seq_lens,
            max_seq_len=int(input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item()),
            block_tables=block_tables,
            slot_mappings=slots_c,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            positions=positions_c,
            is_prefilling=torch.from_numpy(input_batch.is_prefilling_np),
            rswa_prefix_lens=input_batch.prompt_lens,
        )
        decoder = dict(attn_metadata)
        for name in self.needed_names:
            if name not in compact:
                raise RuntimeError(f"[dsv41-yoco] no compact metadata for {name}")
            decoder[name] = compact[name]
        lower = torch.from_numpy(plan.lower_bound).to(dev)
        seen: set[int] = set()
        for name in self.decoder_swa_prefixes:
            md = decoder[name]
            if id(md) in seen:
                continue
            seen.add(id(md))
            nd, npf = md.num_decode_tokens, md.num_prefill_tokens
            if nd:
                clamp_swa_rows(md.decode_swa_indices, md.decode_swa_lens,
                               positions_c[:nd], lower[:nd])
            if npf:
                clamp_swa_rows(md.prefill_swa_indices, md.prefill_swa_lens,
                               positions_c[nd:nd + npf], lower[nd:nd + npf])
        out = YocoAttnMetadata(attn_metadata)
        out.dsv41_yoco = StepPlan(keep_idx, num_rows, self.boundary, decoder)
        return out


def make_driver(vllm_config: Any, model: torch.nn.Module, device: torch.device) -> YocoPrefill:
    return YocoPrefill(vllm_config, model, device, Settings.from_env().require_enabled())
