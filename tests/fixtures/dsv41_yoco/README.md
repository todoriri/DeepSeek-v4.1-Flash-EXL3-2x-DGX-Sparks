Unpatched-by-YOCO copies of the seven image files that `overlay/patch_yoco_prefill.py`
edits, used by `tests/test_yoco_prefill_patch.py`. Paths mirror the `vllm/` package.
They were taken on 2026-10-20 from
`/usr/local/lib/python3.12/dist-packages/vllm/` in
`ghcr.io/miaai-lab/deepseek-v4.1-flash-exl3-2x-dgx-sparks:2.9bpw`, the live head image
(`sha256:0aecf9f5f158…`). That image's vLLM tree is identical to
`sha256:4cdba4e946da…` (repo digest `sha256:2f0cf3ad…`): vLLM
`0.1.dev20904+g179dd0fa9`, base `vllm/vllm-openai:deepseekv41-flash-0909`.

They already contain the build-time overlay patches (`[dsv41-sm120-block64]`,
`[dsv41-exl3-…]`). The runtime patches that start.sh re-applies to these files are
idempotent: their markers are already present, so they skip.
vLLM is Apache-2.0 (SPDX headers kept).
