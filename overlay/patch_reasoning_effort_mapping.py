#!/usr/bin/env python3
"""Align the DeepSeek V4.1 reasoning-effort names with the official mapping.

The image's ``deepseek_v41`` encoder predates DeepSeek's HF commit df42c10 and
still maps ``low``/``high`` to 25/50.  The official encoder, the V4.1 tech
report (p. 30) and deepseek-recipe use low=50, high=75, max=100.  Because
``--tokenizer-mode deepseek_v41`` renders prompts itself (the Jinja
``--chat-template`` is never consulted), this table is what every request
actually gets, including the default effort ``"high"``.

``xhigh`` stays at 75 so existing callers keep working (DeepSeek's API maps
xhigh to high).  Mirrors vllm-project/vllm#58316.
"""
from __future__ import annotations

from pathlib import Path


TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/tokenizers/deepseek_v41_encoding.py"
)
OLD = '''REASONING_EFFORT_MAPPINGS: Dict[str, int] = {
    "low": 25,
    "high": 50,
    "xhigh": 75,
    "max": 100,
}
'''
NEW = '''REASONING_EFFORT_MAPPINGS: Dict[str, int] = {
    "low": 50,
    "high": 75,
    "xhigh": 75,
    "max": 100,
}
'''


def patch_text(source: str) -> tuple[str, str]:
    if NEW in source:
        return source, "already applied"
    count = source.count(OLD)
    if count != 1:
        raise RuntimeError(
            f"expected one DeepSeek V4.1 reasoning-effort mapping anchor, found {count}"
        )
    return source.replace(OLD, NEW, 1), "applied"


def main() -> None:
    source = TARGET.read_text(encoding="utf-8")
    patched, status = patch_text(source)
    if patched != source:
        TARGET.write_text(patched, encoding="utf-8")
    print(f"DeepSeek V4.1 reasoning-effort mapping patch: {status}")


if __name__ == "__main__":
    main()
