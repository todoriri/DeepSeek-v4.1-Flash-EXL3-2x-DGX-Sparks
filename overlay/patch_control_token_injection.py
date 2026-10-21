#!/usr/bin/env python3
"""Stop control-token text in request content from becoming control tokens.

The image's ``deepseek_v41`` tokenizer renders the whole chat to one string
and calls ``self.encode(prompt, add_special_tokens=False)``.  The HF tokenizer
matches added tokens anywhere in that string, so ``</think>``,
``<｜end▁of▁sentence｜>`` or ``｜DSML｜`` inside user content, tool results or
tool-call arguments became real control tokens (live /tokenize probe
2026-10-20: ids 128822, 1 and 128825 inside the user turn).  Only BOS/EOS/pad
are ``special`` in the V4.1 vocab, so ``split_special_tokens=True`` would
still emit ``</think>`` and ``｜DSML｜``.

The patched tokenizer renders each conversation twice: as today, and through
a private copy of the encoder module whose control-token constants are nonce
markers.  The markers are exactly the control tokens the template inserted.
When no other text holds an added-token string, the prompt is encoded exactly
as before (same call, same ids).  Otherwise it is encoded segment-wise:
template tokens straight to their ids, every text segment with the base
vocabulary only (a copy of the tokenizer without added tokens), which is
what DeepInfer's doc says DeepSeek's own serving does (the open-source
deepseek-recipe 0.1.1 still encodes the flat string).

The server always asks for a string: ``OnlineRenderer.preprocess_chat``
merges ``tokenize=False`` under the request's chat_template_kwargs (chat,
/tokenize and Responses alike), and the renderer encodes that string with
the full vocabulary.  So with ``tokenize`` false a clean prompt is returned
as today's string, and a prompt holding control-token text is returned as
the guarded ids, which ``parse_dec_only_prompt`` accepts as a TokensPrompt.
A request's explicit ``"tokenize": false`` is indistinguishable from that
default and gets the same ids.  (The first version refused it with
HTTP 400, which on the live server refused every such request: 2026-10-21
arm A1.)  Output parsing is untouched: the parsers resolve ``<think>`` /
``</think>`` through ``get_vocab()`` and match by id.
"""
from __future__ import annotations

import ast
from pathlib import Path


SITE = Path("/usr/local/lib/python3.12/dist-packages/vllm/tokenizers")
TARGET = SITE / "deepseek_v41.py"
ENCODING_TARGET = SITE / "deepseek_v41_encoding.py"
SENTINEL = "DSV41_CONTROL_TOKEN_GUARD"
# Every V4.1 added token contains one of these (1283 tokens, checked by the test).
NEEDLES = ("｜", "<|", "<think>", "</think>", "<dsml:", "</dsml:")

GUARD = r'''
# --- DSV41_CONTROL_TOKEN_GUARD: overlay/patch_control_token_injection.py ---
import importlib.util as _ctg_importlib_util
import json as _ctg_json
import re as _ctg_re
import secrets as _ctg_secrets
import threading as _ctg_threading

from . import deepseek_v41_encoding as _ctg_encoding


def _ctg_trie_pattern(words) -> str:
    """Regex for any of ``words``; one branch per prefix, longest match wins."""
    trie: dict[str, dict] = {}
    for word in words:
        node = trie
        for char in word:
            node = node.setdefault(char, {})
        node[""] = {}

    def build(node: dict[str, dict]) -> str:
        branches = [
            _ctg_re.escape(char) + build(child)
            for char, child in sorted(node.items())
            if char
        ]
        if not branches:
            return ""
        body = branches[0] if len(branches) == 1 else "(?:" + "|".join(branches) + ")"
        return "(?:" + body + ")?" if "" in node else body

    return build(trie)


class _ControlTokenGuard:
    """Keep control tokens to the positions the V4.1 encoder put them.

    ``encoding`` is a private copy of the encoder module whose control-token
    constants are nonce markers, so a render through it marks exactly the
    template's control tokens.  Text between markers is request text and is
    encoded with the base vocabulary only.
    """

    def __init__(self, tokenizer, added_vocab: dict[str, int]) -> None:
        self._ids = dict(added_vocab)
        if not self._ids:
            raise RuntimeError("DeepSeek V4.1 control-token guard: no added tokens")
        self._scan = _ctg_re.compile(_ctg_trie_pattern(self._ids))
        self._prefix = "\ue000" + _ctg_secrets.token_hex(16) + "#"
        self._marker = _ctg_re.compile(_ctg_re.escape(self._prefix) + "(\\d+)\ue001")
        self._tokens: list[str] = []
        self._markers: dict[str, str] = {}
        self.encoding = self._marked_encoding()
        self.image_placeholder = self._mark(IMAGE_PLACEHOLDER)
        # Snapshot now: the Rust backend is shared with the serving copies.
        spec = _ctg_json.loads(tokenizer.backend_tokenizer.to_str())
        spec["added_tokens"] = []
        spec["truncation"] = None
        spec["padding"] = None
        self._base_spec: str | None = _ctg_json.dumps(spec, ensure_ascii=False)
        self._base = None
        self._lock = _ctg_threading.Lock()

    def _mark(self, text: str) -> str:
        return self._scan.sub(self._marker_for, text)

    def _marker_for(self, match) -> str:
        token = match.group(0)
        marker = self._markers.get(token)
        if marker is None:
            marker = f"{self._prefix}{len(self._tokens)}\ue001"
            self._tokens.append(token)
            self._markers[token] = marker
        return marker

    def _mark_value(self, value):
        if type(value) is str:
            return self._mark(value)
        if type(value) is dict:
            return {key: self._mark_value(item) for key, item in value.items()}
        if type(value) in (list, tuple):
            return type(value)(self._mark_value(item) for item in value)
        return value

    def _marked_encoding(self):
        spec = _ctg_importlib_util.spec_from_file_location(
            _ctg_encoding.__name__ + "_control_marked", _ctg_encoding.__file__
        )
        module = _ctg_importlib_util.module_from_spec(spec)
        module.__package__ = _ctg_encoding.__package__
        spec.loader.exec_module(module)
        for name, value in list(vars(module).items()):
            if not name.startswith("__"):
                setattr(module, name, self._mark_value(value))
        return module

    def encode(self, tokenizer, prompt: str, marked: str, kwargs) -> str | list[int]:
        pieces = self._marker.split(marked)
        texts = pieces[0::2]
        indices = [int(index) for index in pieces[1::2]]
        if any(index >= len(self._tokens) for index in indices):
            raise ValueError("DeepSeek V4.1 control-token guard: unknown marker")
        tokens = [self._tokens[index] for index in indices]
        rebuilt = [texts[0]]
        for token, text in zip(tokens, texts[1:]):
            rebuilt += (token, text)
        if "".join(rebuilt) != prompt:
            raise ValueError("DeepSeek V4.1 control-token guard: marked render differs")
        dirty = any(self._scan.search(text) for text in texts)
        if not kwargs.get("tokenize", True):
            # The served path: OnlineRenderer.preprocess_chat defaults tokenize to
            # False (a request's own "tokenize": false is the same value), and the
            # renderer encodes a returned string with the full vocabulary. A clean
            # prompt goes back as that string, unchanged. A dirty one goes back as
            # ids: parse_dec_only_prompt makes them a TokensPrompt, which is never
            # re-encoded and still gets the request's truncation and length checks.
            return self._encode_segments(texts, tokens) if dirty else prompt
        options = {key: kwargs[key] for key in ("truncation", "max_length") if key in kwargs}
        if not dirty:
            return tokenizer.encode(prompt, add_special_tokens=False, **options)
        return self._truncate(tokenizer, self._encode_segments(texts, tokens), options)

    def _encode_segments(self, texts: list[str], tokens: list[str]) -> list[int]:
        encoded = iter(
            self._base_tokenizer().encode_batch(
                [text for text in texts if text], add_special_tokens=False
            )
        )
        ids: list[int] = []
        for index, text in enumerate(texts):
            if index:
                ids.append(self._ids[tokens[index - 1]])
            if text:
                ids.extend(next(encoded).ids)
        return ids

    def _base_tokenizer(self):
        with self._lock:
            if self._base is None:
                from tokenizers import Tokenizer

                self._base = Tokenizer.from_str(self._base_spec)
                self._base_spec = None
            return self._base

    @staticmethod
    def _truncate(tokenizer, ids: list[int], options) -> list[int]:
        max_length = options.get("max_length")
        if (
            max_length is None
            or options.get("truncation") in (False, "do_not_truncate")
            or len(ids) <= max_length
        ):
            return ids
        if tokenizer.truncation_side == "left":
            return ids[-max_length:]
        return ids[:max_length]
'''

ANCHORS: list[tuple[str, str]] = [
    (
        "from .hf import HfTokenizer, get_cached_tokenizer\n"
        "from .protocol import TokenizerLike\n",
        "from .hf import HfTokenizer, get_cached_tokenizer\n"
        "from .protocol import TokenizerLike\n" + GUARD,
    ),
    (
        "def _normalize_messages(\n"
        "    messages: list[ChatCompletionMessageParam],\n"
        ") -> list[dict[str, Any]]:\n",
        "def _normalize_messages(\n"
        "    messages: list[ChatCompletionMessageParam],\n"
        "    image_placeholder: str = IMAGE_PLACEHOLDER,\n"
        ") -> list[dict[str, Any]]:\n",
    ),
    (
        "                    parts.append(IMAGE_PLACEHOLDER)\n",
        "                    parts.append(image_placeholder)\n",
    ),
    (
        "    wrapped = copy.copy(tokenizer)\n"
        "    added_vocab = tokenizer.get_added_vocab()\n",
        "    wrapped = copy.copy(tokenizer)\n"
        "    added_vocab = tokenizer.get_added_vocab()\n"
        "    guard = _ControlTokenGuard(tokenizer, added_vocab)\n",
    ),
    (
        "            conversation = _normalize_messages(messages)\n"
        "            if tools:\n"
        '                system = next((m for m in conversation if m["role"] == "system"), None)\n'
        "                if system is None:\n"
        '                    system = {"role": "system", "content": ""}\n'
        "                    conversation.insert(0, system)\n"
        '                system["tools"] = tools\n',
        "            conversation = _normalize_messages(messages)\n"
        "            marked_conversation = _normalize_messages(messages, guard.image_placeholder)\n"
        "            if tools:\n"
        "                for rendered in (conversation, marked_conversation):\n"
        '                    system = next((m for m in rendered if m["role"] == "system"), None)\n'
        "                    if system is None:\n"
        '                        system = {"role": "system", "content": ""}\n'
        "                        rendered.insert(0, system)\n"
        '                    system["tools"] = tools\n',
    ),
    (
        "            prompt = encode_messages(\n"
        "                conversation,\n"
        '                thinking_mode="thinking" if thinking else "chat",\n'
        '                drop_thinking=kwargs.get("drop_thinking", True),\n'
        "                reasoning_effort=effort,\n"
        "            )\n"
        '            if kwargs.get("tokenize", True):\n'
        "                tokenizer_kwargs = {\n"
        "                    key: kwargs[key]\n"
        '                    for key in ("truncation", "max_length")\n'
        "                    if key in kwargs\n"
        "                }\n"
        "                return self.encode(prompt, add_special_tokens=False, **tokenizer_kwargs)\n"
        "            return prompt\n",
        "            encode_options = dict(\n"
        '                thinking_mode="thinking" if thinking else "chat",\n'
        '                drop_thinking=kwargs.get("drop_thinking", True),\n'
        "                reasoning_effort=effort,\n"
        "            )\n"
        "            prompt = encode_messages(conversation, **encode_options)\n"
        "            marked = guard.encoding.encode_messages(marked_conversation, **encode_options)\n"
        "            return guard.encode(self, prompt, marked, kwargs)\n",
    ),
]


def patch_text(source: str) -> tuple[str, str]:
    if SENTINEL in source:
        return source, "already applied"
    for old, _ in ANCHORS:
        count = source.count(old)
        if count != 1:
            raise RuntimeError(
                f"expected one DeepSeek V4.1 control-token guard anchor, found {count}: "
                f"{old.splitlines()[0].strip()!r}"
            )
    for old, new in ANCHORS:
        source = source.replace(old, new, 1)
    return source, "applied"


def check_encoder_source(source: str) -> None:
    """Refuse encoders that emit control tokens from anywhere but module constants.

    The guard marks the encoder's module-level constants; a control token
    written inline in a function, or bound as a default argument, would be
    rendered unmarked and encoded as plain text.
    """
    tree = ast.parse(source)
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }

    def bearing(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            and any(needle in node.value for needle in NEEDLES)
        )

    problems: list[int] = []
    for statement in tree.body:
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            scopes = [node for node in ast.walk(statement) if isinstance(node, ast.Lambda)]
        else:
            scopes = [statement]
        for scope in scopes:
            for node in ast.walk(scope):
                if bearing(node):
                    problems.append(node.lineno)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    defaults = node.args.defaults + [d for d in node.args.kw_defaults if d]
                    problems += [d.lineno for d in defaults if not isinstance(d, ast.Constant)]
    if problems:
        raise RuntimeError(
            "DeepSeek V4.1 encoder has control tokens outside module constants "
            f"(lines {sorted(set(problems))}); the control-token guard cannot mark them"
        )


def main() -> None:
    check_encoder_source(ENCODING_TARGET.read_text(encoding="utf-8"))
    source = TARGET.read_text(encoding="utf-8")
    patched, status = patch_text(source)
    if patched != source:
        TARGET.write_text(patched, encoding="utf-8")
    print(f"DeepSeek V4.1 control-token injection guard patch: {status}")


if __name__ == "__main__":
    main()
