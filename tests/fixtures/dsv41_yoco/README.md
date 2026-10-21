Unpatched copies of image files, used by two overlays' host tests. Paths mirror the
`vllm/` package. They were taken from `/usr/local/lib/python3.12/dist-packages/vllm/` in
`ghcr.io/miaai-lab/deepseek-v4.1-flash-exl3-2x-dgx-sparks:2.9bpw`, the live head image
(`sha256:0aecf9f5f158…`). That image's vLLM tree is identical to
`sha256:4cdba4e946da…` (repo digest `sha256:2f0cf3ad…`): vLLM
`0.1.dev20904+g179dd0fa9`, base `vllm/vllm-openai:deepseekv41-flash-0909`.

- 2026-10-20: the seven files that `overlay/patch_yoco_prefill.py` edits, used by
  `tests/test_yoco_prefill_patch.py`.
- 2026-10-21: `v1/core/block_pool.py`, `kv_cache_coordinator.py`, `kv_cache_utils.py`,
  `kv_cache_metrics.py`, and `v1/kv_cache_interface.py`, `kv_cache_spec_registry.py`,
  `kv_cache_layout.py`. Together with the two `v1/core` cache-manager files already here,
  `tests/test_swa_final_window_sim.py` runs the image's own prefix-cache code on CPU, and
  `tests/test_swa_final_window_patch.py` checks `overlay/patch_swa_final_window.py`'s
  anchors. They are byte-identical in the live tree and in `2f0cf3ad`.

They already contain the build-time overlay patches (`[dsv41-sm120-block64]`,
`[dsv41-exl3-…]`). The runtime patches that start.sh re-applies to these files are
idempotent: their markers are already present, so they skip.
vLLM is Apache-2.0 (SPDX headers kept).
