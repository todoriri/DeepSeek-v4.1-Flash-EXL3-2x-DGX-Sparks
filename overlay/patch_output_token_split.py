#!/usr/bin/env python3
"""Non-streaming output parsing: control-token strings count only where the model emitted the token.

Since the control-token guard (98abfc7, live 2026-10-21) the model reads control-token
strings in tool results as plain text, and it may quote them back as plain text: in the
live record the only control ids in its completions were its own ``</think>`` (128822)
and EOS (1).  Streaming keeps those quotes, because once a delta has carried ids the
parser engine treats ``<think>``/``</think>``/BOS/EOS that its lexer found in text as
content (``StreamingParserEngine._process_lex_tokens``, "strict" mode).

Non-streaming does not.  ``OpenAIServingChat.chat_completion_full_generator`` (and batch
chat, Responses and the Responses finalizer) call
``parser.parse(output.text, ..., model_output_token_ids=output.token_ids)``, and
``DelegatingParser.parse`` drops the ids: ``extract_reasoning`` feeds the whole text with
no ids, then the tool parser re-parses the content, again text only.  With no ids the
engine matches by text, so in the live record (``_repl_measurement_stash/
ctg-deploy-20261021/output-side.txt``):

  * a quoted ``<think>draft</think>`` in a thinking-off answer moved ``draft`` into
    ``reasoning``, and a quoted ``</think>`` was absorbed (CONTENT --THINK_END--> CONTENT);
  * a quoted ``<｜end▁of▁sentence｜>`` was dropped: ``_build_drop_info`` makes every
    ``tokenizer.all_special_tokens`` string (BOS/EOS here) a DROP terminal by id *and*
    by text, and ``_on_terminal`` returns nothing for DROP;
  * a quoted ``</think>`` inside reasoning ended it, and the rest came back as content.

The same text matching opens a tool call on a quoted ``<｜DSML｜ calls>``; real calls
carry every ``｜DSML｜`` as token 128825 (live check 2026-10-22,
``_repl_measurement_stash/ots-dsml-ids-20261022``).

The patch makes ``DelegatingParser.parse`` (engine-based parsers only) use the ids it is
given.  The ids are split into runs at added-token ids and the runs are decoded with the
detokenizer's own backend.  ``output.text`` must be a prefix of that decode (a stripped
stop token, a stop string or held-back UTF-8 only shorten it), else the text is parsed
as before.  An added-token string inside a run was spelled with ordinary tokens, so it
is replaced by a nonce placeholder for the parse and restored in its result (reasoning,
content, tool-call names and arguments).  The parse itself is the image's, unchanged,
and a text with no such string goes through it untouched.  Streaming is not changed
(its tool markers are still matched by text: docs/output-token-split-20261022.md).
"""
from __future__ import annotations

from pathlib import Path


TARGET = Path("/usr/local/lib/python3.12/dist-packages/vllm/parser/abstract_parser.py")
SENTINEL = "DSV41_OUTPUT_TOKEN_SPLIT"

HELPERS = r'''

# --- DSV41_OUTPUT_TOKEN_SPLIT: overlay/patch_output_token_split.py ---
import re as _ots_re
import secrets as _ots_secrets
import threading as _ots_threading


def _ots_trie_pattern(words) -> str:
    """Regex for any of ``words``; one branch per prefix, longest match wins."""
    trie: dict[str, dict] = {}
    for word in words:
        node = trie
        for char in word:
            node = node.setdefault(char, {})
        node[""] = {}

    def build(node: dict[str, dict]) -> str:
        branches = [
            _ots_re.escape(char) + build(child)
            for char, child in sorted(node.items())
            if char
        ]
        if not branches:
            return ""
        body = branches[0] if len(branches) == 1 else "(?:" + "|".join(branches) + ")"
        return "(?:" + body + ")?" if "" in node else body

    return build(trie)


class _OutputTokenSplitResult:
    """A masked output text and how to put the masked strings back."""

    __slots__ = ("text", "_prefix", "_marker", "_originals")

    def __init__(self, text: str, prefix: str, marker, originals: list[str]) -> None:
        self.text = text
        self._prefix = prefix
        self._marker = marker
        self._originals = originals

    def _restore(self, value):
        if not value:
            return value
        value = self._marker.sub(lambda match: self._originals[int(match.group(1))], value)
        if self._prefix in value:
            raise ValueError("a placeholder did not survive the parse")
        return value

    def restore(self, reasoning, content, tool_calls):
        """The parse result with the masked strings back, or None if one was cut."""
        try:
            restored = [
                (self._restore(call.name), self._restore(call.arguments))
                for call in tool_calls or ()
            ]
            reasoning, content = self._restore(reasoning), self._restore(content)
        except (ValueError, IndexError):
            return None
        for call, (name, arguments) in zip(tool_calls or (), restored):
            call.name, call.arguments = name, arguments
        return reasoning, content, tool_calls


class _OutputTokenSplit:
    """Find the added-token strings a completion spelled with ordinary tokens."""

    def __init__(self, tokenizer) -> None:
        added = tokenizer.get_added_vocab()
        self._backend = tokenizer.backend_tokenizer
        self._texts = {token_id: text for text, token_id in added.items()}
        words = [text for text in added if text and text == text.strip()]
        self._scan = _ots_re.compile(_ots_trie_pattern(words)) if words else None
        self._prefix = "DSV41OTS" + _ots_secrets.token_hex(12) + "n"
        self._marker = _ots_re.compile(_ots_re.escape(self._prefix) + "([0-9]+)z")

    def split(self, text: str, token_ids) -> _OutputTokenSplitResult | None:
        scan = self._scan
        if scan is None or scan.search(text) is None or self._prefix in text:
            return None
        texts = self._texts
        runs: list[list[int]] = []
        layout: list[int | str] = []  # run index, or an added token's text
        run: list[int] = []
        for token_id in token_ids:
            token_text = texts.get(token_id)
            if token_text is None:
                run.append(token_id)
                continue
            if run:
                layout.append(len(runs))
                runs.append(run)
                run = []
            layout.append(token_text)
        if run:
            layout.append(len(runs))
            runs.append(run)
        decoded = self._backend.decode_batch(runs, skip_special_tokens=False) if runs else []
        pieces = [decoded[item] if type(item) is int else item for item in layout]
        if not "".join(pieces).startswith(text):
            _ots_log("the output text is not a prefix of its decoded token ids")
            return None
        originals: list[str] = []

        def mask(match) -> str:
            originals.append(match.group(0))
            return f"{self._prefix}{len(originals) - 1}z"

        out: list[str] = []
        position = 0
        for item, piece in zip(layout, pieces):
            visible = piece[: len(text) - position]
            out.append(scan.sub(mask, visible) if type(item) is int else visible)
            position += len(visible)
            if position >= len(text):
                break
        if not originals:
            return None
        return _OutputTokenSplitResult("".join(out), self._prefix, self._marker, originals)


_OTS_SPLITTERS: dict[int, tuple[object, _OutputTokenSplit | None]] = {}
_OTS_LOCK = _ots_threading.Lock()


def _ots_log(reason: str) -> None:
    logger.warning_once(
        "DeepSeek V4.1 output token split: %s; that output is parsed by text", reason
    )


def _ots_split(parser, text: str, token_ids) -> _OutputTokenSplitResult | None:
    """Mask BPE-spelled added-token strings in ``text`` (None: parse it as is)."""
    if not text or not token_ids or not getattr(parser, "_engine_based", False):
        return None
    tokenizer = parser.model_tokenizer
    try:
        entry = _OTS_SPLITTERS.get(id(tokenizer))
        if entry is None or entry[0] is not tokenizer:
            with _OTS_LOCK:
                entry = _OTS_SPLITTERS.get(id(tokenizer))
                if entry is None or entry[0] is not tokenizer:
                    usable = hasattr(tokenizer, "get_added_vocab") and hasattr(
                        tokenizer, "backend_tokenizer"
                    )
                    entry = (tokenizer, _OutputTokenSplit(tokenizer) if usable else None)
                    _OTS_SPLITTERS[id(tokenizer)] = entry
        splitter = entry[1]
        return splitter.split(text, list(token_ids)) if splitter is not None else None
    except Exception as error:  # never fail a request over this
        _ots_log(f"{type(error).__name__}: {error}")
        return None
'''

ANCHORS: list[tuple[str, str]] = [
    (
        "logger = init_logger(__name__)\n",
        "logger = init_logger(__name__)\n" + HELPERS,
    ),
    (
        "    def parse(\n"
        "        self,\n"
        "        model_output: str,\n"
        "        request: ChatCompletionRequest | ResponsesRequest,\n"
        "        enable_auto_tools: bool = False,\n"
        "        model_output_token_ids: Sequence[int] = (),\n"
        "    ) -> tuple[str | None, str | None, list[FunctionCall] | None]:\n"
        "        self._initialize_history_tool_call_cnt(request)\n"
        "        reasoning, content = self.extract_reasoning(model_output, request)\n"
        "        tool_calls, content = self._extract_tool_calls(\n"
        "            content=content,\n"
        "            request=request,\n"
        "            enable_auto_tools=enable_auto_tools,\n"
        "        )\n"
        "        return reasoning, content, tool_calls\n",
        "    def parse(\n"
        "        self,\n"
        "        model_output: str,\n"
        "        request: ChatCompletionRequest | ResponsesRequest,\n"
        "        enable_auto_tools: bool = False,\n"
        "        model_output_token_ids: Sequence[int] = (),\n"
        "    ) -> tuple[str | None, str | None, list[FunctionCall] | None]:\n"
        "        # [DSV41_OUTPUT_TOKEN_SPLIT] control-token strings the model spelled with\n"
        "        # ordinary tokens are text, as in streaming.\n"
        "        split = _ots_split(self, model_output, model_output_token_ids)\n"
        "        if split is not None:\n"
        "            restored = split.restore(\n"
        "                *self._ots_parse_text(split.text, request, enable_auto_tools)\n"
        "            )\n"
        "            if restored is not None:\n"
        "                return restored\n"
        '            _ots_log("a placeholder did not survive the parse")\n'
        "        return self._ots_parse_text(model_output, request, enable_auto_tools)\n"
        "\n"
        "    def _ots_parse_text(\n"
        "        self,\n"
        "        model_output: str,\n"
        "        request: ChatCompletionRequest | ResponsesRequest,\n"
        "        enable_auto_tools: bool,\n"
        "    ) -> tuple[str | None, str | None, list[FunctionCall] | None]:\n"
        "        self._initialize_history_tool_call_cnt(request)\n"
        "        reasoning, content = self.extract_reasoning(model_output, request)\n"
        "        tool_calls, content = self._extract_tool_calls(\n"
        "            content=content,\n"
        "            request=request,\n"
        "            enable_auto_tools=enable_auto_tools,\n"
        "        )\n"
        "        return reasoning, content, tool_calls\n",
    ),
]


def patch_text(source: str) -> tuple[str, str]:
    if SENTINEL in source:
        return source, "already applied"
    for old, _ in ANCHORS:
        count = source.count(old)
        if count != 1:
            raise RuntimeError(
                f"expected one DeepSeek V4.1 output token split anchor, found {count}: "
                f"{old.splitlines()[0].strip()!r}"
            )
    for old, new in ANCHORS:
        source = source.replace(old, new, 1)
    return source, "applied"


def main() -> None:
    source = TARGET.read_text(encoding="utf-8")
    patched, status = patch_text(source)
    if patched != source:
        TARGET.write_text(patched, encoding="utf-8")
    print(f"DeepSeek V4.1 output token split patch: {status}")


if __name__ == "__main__":
    main()
