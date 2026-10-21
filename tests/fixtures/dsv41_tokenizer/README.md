Unpatched copies of the image's DeepSeek V4.1 prompt tokenizer, used by
`tests/test_control_token_injection.py`. Taken from
`/usr/local/lib/python3.12/dist-packages/vllm/tokenizers/` in
`ghcr.io/miaai-lab/deepseek-v4.1-flash-exl3-2x-dgx-sparks:2.9bpw`
(image `sha256:0aecf9f5f158…`, base `vllm/vllm-openai:deepseekv41-flash-0909`), 2026-10-20.
vLLM code, Apache-2.0.

    d6e9cbce7c5f127819452fa5d03dd4f22ad9f136dd202a54b5b31fe42402cae2  deepseek_v41.py
    fa501415211cd0e0c9f06ee6033c5466046e293b6153bca6200b70695b88fc73  deepseek_v41_encoding.py

The live files differ only by `overlay/patch_responses_content_types.py` and
`overlay/patch_reasoning_effort_mapping.py`, which `start.sh` applies at container start.
