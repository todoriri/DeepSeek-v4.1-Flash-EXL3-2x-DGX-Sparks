#!/usr/bin/env python3
"""Decoder-SWA bounded replay (YOCO prefill truncation) for DeepSeek-V4.1, OFF by default.

With DSV41_YOCO_PREFILL=compact, decoder layers 20-39 run each prefill only on the
last DSV41_YOCO_KEEP_TOKENS (default 140) positions of the sequence. Layer 20 still
writes its compressed and index KV for every token. This is DeepInfer's
DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact, the approximation DeepSeek's API
runs. The logic lives in overlay/dsv41_yoco.py (mounted at /opt/dsv41). This script
edits seven image files:

  models/deepseek_v4_1/nvidia/model.py     boundary hook in the decoder layer; plan
                                           begin / after_split / finish in the loop
  models/deepseek_v4_1/attention.py        boundary layer skips its kv-source work on
                                           the compact call (done on all rows first)
  models/deepseek_v4_1/nvidia/model_state.py
                                           builds the per-step plan and the compact
                                           decoder metadata
  v1/core/kv_cache_manager.py              prefix-cache hit <= num_tokens - KEEP
  v1/core/single_type_kv_cache_manager.py  extra retention checkpoint at the capped
                                           replay point
  v1/engine/input_processor.py             refuses prompt_logprobs
  v1/worker/gpu/model_runner.py            prefill steps run eager (no piecewise graph)

start.sh runs this only when DSV41_YOCO_PREFILL != off, after every other patch, and
aborts the container if it fails. When the variable is unset or off, the image's vLLM
files stay byte-identical. Every anchor must match exactly once. Otherwise nothing
is written and the exit status is 1. `--check` verifies the anchors only, e.g. in the
Dockerfile. `--root` patches another vllm tree (tests).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

MARK = "dsv41-yoco-prefill"

LOADER = '''

# [dsv41-yoco-prefill]
def _dsv41_yoco_load():
    """overlay/dsv41_yoco.py, mounted by start.sh (DSV41_YOCO_HELPER overrides)."""
    import importlib.util
    import os
    import sys

    mod = sys.modules.get("dsv41_yoco")
    if mod is None:
        path = os.environ.get("DSV41_YOCO_HELPER", "/opt/dsv41/dsv41_yoco.py")
        spec = importlib.util.spec_from_file_location("dsv41_yoco", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"[dsv41-yoco] cannot load {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["dsv41_yoco"] = mod
        spec.loader.exec_module(mod)
    return mod


_DSV41_YOCO_KEEP = _dsv41_yoco_load().Settings.from_env().require_enabled().keep
'''

MODEL_FOOTER = LOADER + '''

def _dsv41_yoco_begin(model, positions):
    return _dsv41_yoco_load().begin(model, positions)
'''

RUNNER_FOOTER = LOADER + '''

def _dsv41_yoco_need_eager(batch_req_state) -> bool:
    """A step with a prefill longer than KEEP runs eager: the decoder truncation and its
    SWA window bound only exist in the eager forward (no piecewise replay)."""
    if batch_req_state is None or not batch_req_state.has_prefill:
        return False
    prefilling = batch_req_state.is_prefilling_np
    return bool((batch_req_state.prefill_len_np[prefilling] > _DSV41_YOCO_KEEP).any())
'''

JOBS: list[tuple[str, list[tuple[str, str]], str | None]] = [
    (
        "models/deepseek_v4_1/nvidia/model.py",
        [
            (
                "        engram_hashes: torch.Tensor | None = None,\n"
                "        engram_mask: torch.Tensor | None = None,\n"
                "    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:\n",
                "        engram_hashes: torch.Tensor | None = None,\n"
                "        engram_mask: torch.Tensor | None = None,\n"
                "        dsv41_yoco_split: typing.Any = None,  # [dsv41-yoco-prefill]\n"
                "    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:\n",
            ),
            (
                "        if self.use_sequence_parallel:\n"
                "            x = sp_all_gather(x)[: positions.shape[0]]\n"
                "\n"
                "        x = self.attn(positions, x, None)\n",
                "        if dsv41_yoco_split is not None:\n"
                "            # [dsv41-yoco-prefill] boundary layer: compressed/index KV for every\n"
                "            # row, then attention and MoE on the kept rows only.\n"
                "            x, residual, post_mix, res_mix, attn_pre, positions, input_ids = (\n"
                "                dsv41_yoco_split(\n"
                "                    self, x, residual, post_mix, res_mix, attn_pre, positions,\n"
                "                    input_ids,\n"
                "                )\n"
                "            )\n"
                "\n"
                "        if self.use_sequence_parallel:\n"
                "            x = sp_all_gather(x)[: positions.shape[0]]\n"
                "\n"
                "        x = self.attn(positions, x, None)\n",
            ),
            (
                "        final_aux_recon: torch.Tensor | None = None  # avoid duplicate mhc_post call\n"
                "        for idx, layer in enumerate(\n"
                "            islice(self.layers, self.start_layer, self.end_layer),\n"
                "            start=self.start_layer,\n"
                "        ):\n"
                "            hidden_states, residual, post_mix, res_mix, pre_mix = layer(\n"
                "                hidden_states,\n"
                "                positions,\n"
                "                input_ids,\n"
                "                pre_mix,\n"
                "                post_mix,\n"
                "                res_mix,\n"
                "                residual,\n"
                "                engram_hashes,\n"
                "                engram_mask,\n"
                "            )\n",
                "        final_aux_recon: torch.Tensor | None = None  # avoid duplicate mhc_post call\n"
                "        _dsv41_yoco_plan = _dsv41_yoco_begin(self, positions)  # [dsv41-yoco-prefill]\n"
                "        for idx, layer in enumerate(\n"
                "            islice(self.layers, self.start_layer, self.end_layer),\n"
                "            start=self.start_layer,\n"
                "        ):\n"
                "            hidden_states, residual, post_mix, res_mix, pre_mix = layer(\n"
                "                hidden_states,\n"
                "                positions,\n"
                "                input_ids,\n"
                "                pre_mix,\n"
                "                post_mix,\n"
                "                res_mix,\n"
                "                residual,\n"
                "                engram_hashes,\n"
                "                engram_mask,\n"
                "                **(\n"
                "                    _dsv41_yoco_plan.split_kwargs(idx)\n"
                "                    if _dsv41_yoco_plan is not None\n"
                "                    else {}\n"
                "                ),\n"
                "            )\n"
                "            if _dsv41_yoco_plan is not None and idx == _dsv41_yoco_plan.boundary:\n"
                "                positions, input_ids = _dsv41_yoco_plan.after_split()\n",
            ),
            (
                "            num_tokens = hidden_states.shape[0]\n"
                "            self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))\n",
                "            if _dsv41_yoco_plan is not None:  # [dsv41-yoco-prefill]\n"
                "                _dsv41_yoco_plan.scatter_rows_into(\n"
                "                    self._mtp_hidden_buffer, hidden_states.flatten(1)\n"
                "                )\n"
                "            else:\n"
                "                num_tokens = hidden_states.shape[0]\n"
                "                self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))\n",
            ),
            (
                "        if len(aux_hidden_states) > 0:\n"
                "            return hidden_states, aux_hidden_states\n"
                "        return hidden_states\n",
                "        if _dsv41_yoco_plan is not None:  # [dsv41-yoco-prefill] back to all rows\n"
                "            hidden_states, aux_hidden_states = _dsv41_yoco_plan.finish(hidden_states, aux_hidden_states)\n"
                "        if len(aux_hidden_states) > 0:\n"
                "            return hidden_states, aux_hidden_states\n"
                "        return hidden_states\n",
            ),
        ],
        MODEL_FOOTER,
    ),
    (
        "models/deepseek_v4_1/attention.py",
        [
            (
                "        if self.compressor is not None:\n"
                "            # Local ref so the closure keeps a non-None type for mypy.\n"
                "            compressor = self.compressor\n",
                "        if self.compressor is not None and not getattr(self, \"_dsv41_yoco_skip_kv\", False):\n"
                "            # [dsv41-yoco-prefill] skipped on the boundary layer's compact call.\n"
                "            # Local ref so the closure keeps a non-None type for mypy.\n"
                "            compressor = self.compressor\n",
            ),
            (
                "            def compressor_kv_score() -> torch.Tensor:\n"
                "                # Packed EXL3 compressor has no dense .weight; apply the\n"
                "                # fused wkv+wgate LinearEXL3 and match the fp32 mm dtype.\n"
                "                _layer = compressor.fused_wkv_wgate\n"
                "                _qm = getattr(_layer, \"quant_method\", None)\n"
                "                if _qm is not None and not hasattr(_layer, \"weight\"):\n"
                "                    return _qm.apply(_layer, hidden_states).float()\n"
                "                return torch.mm(\n"
                "                    hidden_states,\n"
                "                    _layer.weight.T,\n"
                "                    out_dtype=torch.float32,\n"
                "                )\n",
                "            def compressor_kv_score() -> torch.Tensor:\n"
                "                # [dsv41-yoco-prefill] mirrored by dsv41_yoco.produce_kv_full.\n"
                "                # Packed EXL3 compressor has no dense .weight; apply the\n"
                "                # fused wkv+wgate LinearEXL3 and match the fp32 mm dtype.\n"
                "                _layer = compressor.fused_wkv_wgate\n"
                "                _qm = getattr(_layer, \"quant_method\", None)\n"
                "                if _qm is not None and not hasattr(_layer, \"weight\"):\n"
                "                    return _qm.apply(_layer, hidden_states).float()\n"
                "                return torch.mm(\n"
                "                    hidden_states,\n"
                "                    _layer.weight.T,\n"
                "                    out_dtype=torch.float32,\n"
                "                )\n",
            ),
            (
                "        indexer = self.indexer\n"
                "        compressor = self.compressor\n"
                "        aux_streams = self.aux_stream_list\n",
                "        indexer = self.indexer\n"
                "        # [dsv41-yoco-prefill] no compressor/insert/index-K on the compact call\n"
                "        # (latent stays None, so Indexer._produce_k returns early).\n"
                "        compressor = None if getattr(self, \"_dsv41_yoco_skip_kv\", False) else self.compressor\n"
                "        aux_streams = self.aux_stream_list\n",
            ),
        ],
        None,
    ),
    (
        "models/deepseek_v4_1/nvidia/model_state.py",
        [
            (
                "        if depth > 0:\n"
                "            # Persistent so a captured graph can read it on replay.\n"
                "            self.lookback_token_ids = torch.full(\n"
                "                (self.max_num_reqs, depth), -1, dtype=torch.int32, device=device\n"
                "            )\n",
                "        if depth > 0:\n"
                "            # Persistent so a captured graph can read it on replay.\n"
                "            self.lookback_token_ids = torch.full(\n"
                "                (self.max_num_reqs, depth), -1, dtype=torch.int32, device=device\n"
                "            )\n"
                "        # [dsv41-yoco-prefill] validates the topology / KEEP at load time\n"
                "        self._dsv41_yoco = _dsv41_yoco_load().make_driver(vllm_config, model, device)\n",
            ),
            (
                "            model_inputs[\"lookback_token_ids\"] = self.lookback_token_ids\n"
                "        return model_inputs\n",
                "            model_inputs[\"lookback_token_ids\"] = self.lookback_token_ids\n"
                "        return model_inputs\n"
                "\n"
                "    def prepare_attn(  # [dsv41-yoco-prefill]\n"
                "        self,\n"
                "        input_batch: InputBatch,\n"
                "        cudagraph_mode: Any,\n"
                "        block_tables: tuple[torch.Tensor, ...],\n"
                "        slot_mappings: torch.Tensor,\n"
                "        attn_groups: Any,\n"
                "        kv_cache_config: Any,\n"
                "        for_capture: bool = False,\n"
                "    ) -> dict[str, Any]:\n"
                "        attn_metadata = super().prepare_attn(\n"
                "            input_batch,\n"
                "            cudagraph_mode,\n"
                "            block_tables,\n"
                "            slot_mappings,\n"
                "            attn_groups,\n"
                "            kv_cache_config,\n"
                "            for_capture=for_capture,\n"
                "        )\n"
                "        # Eager prefill steps get a plan + compact decoder metadata; others pass.\n"
                "        return self._dsv41_yoco.attach(\n"
                "            attn_metadata,\n"
                "            input_batch,\n"
                "            cudagraph_mode,\n"
                "            block_tables,\n"
                "            slot_mappings,\n"
                "            attn_groups,\n"
                "            kv_cache_config,\n"
                "            for_capture,\n"
                "        )\n",
            ),
        ],
        LOADER,
    ),
    (
        "v1/core/kv_cache_manager.py",
        [
            (
                "        max_cache_hit_length = request.num_tokens - 1\n",
                "        # [dsv41-yoco-prefill] the decoder rectangle [N - KEEP, N) is recomputed on\n"
                "        # every prefill: cached blocks hold decoder SWA only for their writer's\n"
                "        # rectangle, so a hit may not reach into it.\n"
                "        max_cache_hit_length = max(0, request.num_tokens - _DSV41_YOCO_KEEP)\n",
            ),
        ],
        LOADER,
    ),
    (
        "v1/core/single_type_kv_cache_manager.py",
        [
            (
                "        reachable_boundaries = [request.num_prompt_tokens - 1]\n"
                "        if request.shared_prefix_boundary:\n",
                "        reachable_boundaries = [request.num_prompt_tokens - 1]\n"
                "        # [dsv41-yoco-prefill] hits are capped at num_tokens - KEEP: keep the tail\n"
                "        # a retry of this prompt can still hit (peek block included).\n"
                "        _dsv41_boundary = (\n"
                "            request.num_prompt_tokens - _DSV41_YOCO_KEEP\n"
                "            - (self.block_size if self.use_eagle else 0)\n"
                "        )\n"
                "        if _dsv41_boundary > 0:\n"
                "            reachable_boundaries.append(_dsv41_boundary)\n"
                "        if request.shared_prefix_boundary:\n",
            ),
        ],
        LOADER,
    ),
    (
        "v1/engine/input_processor.py",
        [
            (
                "            params.verify(\n"
                "                self.model_config,\n"
                "                self.speculative_config,\n"
                "                self.structured_outputs_config,\n"
                "                self.tokenizer,\n"
                "            )\n",
                "            params.verify(\n"
                "                self.model_config,\n"
                "                self.speculative_config,\n"
                "                self.structured_outputs_config,\n"
                "                self.tokenizer,\n"
                "            )\n"
                "            if params.prompt_logprobs is not None:  # [dsv41-yoco-prefill]\n"
                "                raise VLLMValidationError(\n"
                "                    \"prompt_logprobs is not available on this server: \"\n"
                "                    \"DSV41_YOCO_PREFILL runs decoder layers 20-39 only on the last \"\n"
                "                    f\"{_DSV41_YOCO_KEEP} prompt tokens, so earlier prompt positions \"\n"
                "                    \"have no logits\"\n"
                "                )\n",
            ),
        ],
        LOADER,
    ),
    (
        "v1/worker/gpu/model_runner.py",
        [
            (
                "            need_eager=is_profile or skip_compiled,\n",
                "            need_eager=is_profile or skip_compiled or _dsv41_yoco_need_eager(batch_req_state),  # [dsv41-yoco-prefill]\n",
            ),
        ],
        RUNNER_FOOTER,
    ),
]


def patch_text(text: str, edits: list[tuple[str, str]], footer: str | None) -> tuple[str, str]:
    if MARK in text:
        return text, "already applied"
    for old, new in edits:
        n = text.count(old)
        if n != 1:
            first = old.strip().splitlines()[0][:80]
            raise RuntimeError(f"anchor found {n}x (want 1): {first!r}")
        text = text.replace(old, new, 1)
    if footer:
        if not text.endswith("\n"):
            text += "\n"
        text += footer
    return text, "applied"


def _vllm_root() -> Path:
    import vllm

    return Path(vllm.__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", help="vllm package directory (default: the installed one)")
    ap.add_argument("--check", action="store_true", help="verify anchors, write nothing")
    args = ap.parse_args(argv)

    if not args.check:
        mode = os.environ.get("DSV41_YOCO_PREFILL", "") or "off"
        if mode == "off":
            print("YOCO prefill patch: skipped (DSV41_YOCO_PREFILL=off)")
            return 0
        if mode != "compact":
            print(f"YOCO prefill patch: DSV41_YOCO_PREFILL must be off or compact (got {mode!r})",
                  file=sys.stderr)
            return 2
        helper = Path(os.environ.get("DSV41_YOCO_HELPER", "/opt/dsv41/dsv41_yoco.py"))
        if args.root is None and not helper.is_file():
            print(f"YOCO prefill patch: helper {helper} missing", file=sys.stderr)
            return 1

    root = Path(args.root) if args.root else _vllm_root()
    out: list[tuple[Path, str, str]] = []
    errors: list[str] = []
    for rel, edits, footer in JOBS:
        path = root / rel
        if not path.is_file():
            errors.append(f"{rel}: missing")
            continue
        try:
            text, status = patch_text(path.read_text(), edits, footer)
        except RuntimeError as exc:
            errors.append(f"{rel}: {exc}")
            continue
        out.append((path, text, status))
    if errors:
        for err in errors:
            print(f"YOCO prefill patch: {err}", file=sys.stderr)
        print("YOCO prefill patch: FAILED (anchor drift), nothing written", file=sys.stderr)
        return 1
    if args.check:
        states = sorted({status for _, _, status in out})
        print(f"YOCO prefill patch: check ok ({len(out)} files, {', '.join(states)})")
        return 0
    for path, text, status in out:
        if status == "applied":
            path.write_text(text)
        print(f"YOCO prefill patch: {status} ({path.relative_to(root)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
