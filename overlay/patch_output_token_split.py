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
and a text with no such string goes through it untouched.

The non-streaming reasoning-token count (``ParserEngineReasoningAdapter.
count_reasoning_tokens``) re-parses ``tokenizer.decode(ids)`` in one feed, where DSML
markers are matched by text too: a DSML quote in the reasoning ended it (4 reasoning
tokens instead of 62 in the test).  With such a quote the count parses the masked text.
(The ``rfind`` anchor rebuild does not move the count: its token counts follow the id
groups.)

Streaming (2026-10-22): its DSML markers are matched by text, so a quoted
``<｜DSML｜ calls>`` block opened a tool call.  Every DSML marker holds ``｜DSML｜``, which
a real call emits as id 128825, and a quote spells with ordinary tokens
(``｜`` ``DS`` ``ML`` ``｜``).  ``DelegatingParser.parse_delta`` now follows where each
added token's text lands in the stream (runs decoded as above) and hides every ``｜`` an
ordinary token produced behind U+FDD0, a noncharacter no tokenizer emits, so a quote can
never complete a marker; the delta it returns gets the ``｜`` back.  One character at a
time, so a quote split over deltas is covered.  The engine parser's ``adjust_request``
sets skip_special_tokens=False, so the text holds every added token's string, specials
included (the stop token's is never streamed).  If an added token's text does not land
where its ids put it, or is still held back, that delta is left as it is (and a mismatch
stops the hiding for the request, logged once).  Clean streams are untouched.
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


def _ots_splitter(tokenizer) -> _OutputTokenSplit | None:
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
    return entry[1]


def _ots_split(parser, text: str, token_ids) -> _OutputTokenSplitResult | None:
    """Mask BPE-spelled added-token strings in ``text`` (None: parse it as is)."""
    if not text or not token_ids or not getattr(parser, "_engine_based", False):
        return None
    try:
        splitter = _ots_splitter(parser.model_tokenizer)
        return splitter.split(text, list(token_ids)) if splitter is not None else None
    except Exception as error:  # never fail a request over this
        _ots_log(f"{type(error).__name__}: {error}")
        return None


def _ots_count_reasoning(parser, token_ids) -> int | None:
    """The non-streaming reasoning-token count on the masked text (None: count as is)."""
    adapter = parser._reasoning_parser
    if (
        not token_ids
        or getattr(adapter, "_streaming_count_valid", True)
        or not hasattr(adapter, "_counting_parser_engine")
    ):
        return None
    try:
        split = _ots_split(parser, adapter.model_tokenizer.decode(token_ids), token_ids)
        if split is None:
            return None
        if adapter._counting_parser_engine is None:
            adapter._counting_parser_engine = adapter._parser_engine_cls(
                adapter.model_tokenizer, **adapter._parser_engine_kwargs
            )
        adapter._counting_parser_engine._single_pass_parse(split.text, token_ids)
        return adapter._counting_parser_engine.count_reasoning_tokens(token_ids)
    except Exception as error:  # never fail a request over this
        _ots_log(f"{type(error).__name__}: {error}")
        return None


_OTS_PIPE = "\uff5c"  # "｜": every DSML marker holds two (inside ｜DSML｜, id 128825)
_OTS_HIDE = "\ufdd0"  # a Unicode noncharacter: no tokenizer's text holds it


class _OutputTokenStream:
    """One streamed completion: hide the "｜"s its ordinary tokens produced."""

    __slots__ = ("_splitter", "_run", "_end", "_spans", "_seen", "active", "hidden")

    def __init__(self, splitter: _OutputTokenSplit) -> None:
        self._splitter = splitter
        self._run: list[int] = []  # ordinary ids since the last added token
        self._end = 0  # length of the decode of the ids before self._run
        self._spans: list[tuple[int, str]] = []  # added tokens with a "｜", not all streamed
        self._seen = 0  # characters streamed so far
        self.active = True
        self.hidden = 0  # "｜"s hidden so far (the deltas need them back)

    def _stop(self, reason: str, text: str) -> str:
        self.active = False
        _ots_log(f"streaming: {reason}")
        return text

    def hide(self, text: str, token_ids) -> str:
        """``text`` with every "｜" an ordinary token produced replaced by _OTS_HIDE."""
        if not self.active:
            return text
        texts = self._splitter._texts
        for token_id in token_ids:
            token_text = texts.get(token_id)
            if token_text is None:
                self._run.append(token_id)
                continue
            if self._run:
                self._end += len(self._splitter._backend.decode(self._run, skip_special_tokens=False))
                self._run = []
            if self._end < self._seen:
                return self._stop("the text ran ahead of its token ids", text)
            if _OTS_PIPE in token_text:
                self._spans.append((self._end, token_text))
            self._end += len(token_text)
        start = self._seen
        self._seen += len(text)
        if _OTS_HIDE in text:
            return self._stop("the text holds the hiding character", text)
        real: set[int] = set()
        held = False
        kept: list[tuple[int, str]] = []
        for span_start, token_text in self._spans:
            low, high = max(span_start, start), min(span_start + len(token_text), self._seen)
            if low < high:
                if text[low - start : high - start] != token_text[low - span_start : high - span_start]:
                    return self._stop("an added token's text is not where its ids put it", text)
                real.update(range(low - start, high - start))
            if span_start + len(token_text) > self._seen:
                kept.append((span_start, token_text))
                held = held or span_start >= self._seen
        self._spans = kept
        if held:
            return text  # an added token's text is not streamed yet: which "｜" is real is open
        at = text.find(_OTS_PIPE)
        if at < 0:
            return text
        out: list[str] = []
        last = 0
        while at >= 0:
            if at not in real:
                out.append(text[last:at])
                out.append(_OTS_HIDE)
                last = at + 1
                self.hidden += 1
            at = text.find(_OTS_PIPE, at + 1)
        if not out:
            return text
        out.append(text[last:])
        return "".join(out)


def _ots_stream(parser) -> _OutputTokenStream | None:
    stream = getattr(parser, "_ots_stream_state", None)
    if stream is None:
        stream = False
        if getattr(parser, "_engine_based", False):
            try:
                splitter = _ots_splitter(parser.model_tokenizer)
            except Exception as error:  # never fail a request over this
                _ots_log(f"{type(error).__name__}: {error}")
                splitter = None
            if splitter is not None:
                stream = _OutputTokenStream(splitter)
        parser._ots_stream_state = stream
    return stream or None


def _ots_hide(stream: _OutputTokenStream, text: str, token_ids) -> str:
    try:
        return stream.hide(text, token_ids)
    except Exception as error:  # never fail a request over this
        return stream._stop(f"{type(error).__name__}: {error}", text)


def _ots_unhide(delta):
    """Put the hidden "｜"s back in a DeltaMessage."""
    if delta is None:
        return delta
    for field in ("content", "reasoning"):
        value = getattr(delta, field, None)
        if value and _OTS_HIDE in value:
            setattr(delta, field, value.replace(_OTS_HIDE, _OTS_PIPE))
    for call in getattr(delta, "tool_calls", None) or ():
        function = getattr(call, "function", None)
        for field in ("name", "arguments"):
            value = getattr(function, field, None) if function is not None else None
            if value and _OTS_HIDE in value:
                setattr(function, field, value.replace(_OTS_HIDE, _OTS_PIPE))
    return delta
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
    (
        "    def parse_delta(\n"
        "        self,\n"
        "        delta_text: str,\n"
        "        delta_token_ids: list[int],\n"
        "        request: ChatCompletionRequest | ResponsesRequest,\n"
        "        prompt_token_ids: list[int] | None = None,\n"
        "        *,\n"
        "        finished: bool,\n"
        "    ) -> DeltaMessage | None:\n"
        "        self._initialize_history_tool_call_cnt(request)\n"
        "        state = self._stream_state\n",
        "    def parse_delta(\n"
        "        self,\n"
        "        delta_text: str,\n"
        "        delta_token_ids: list[int],\n"
        "        request: ChatCompletionRequest | ResponsesRequest,\n"
        "        prompt_token_ids: list[int] | None = None,\n"
        "        *,\n"
        "        finished: bool,\n"
        "    ) -> DeltaMessage | None:\n"
        "        # [DSV41_OUTPUT_TOKEN_SPLIT] a \"｜\" an ordinary token produced cannot\n"
        "        # complete DSML markup: hidden from the parse, put back in the delta.\n"
        "        stream = _ots_stream(self)\n"
        "        if stream is not None:\n"
        "            delta_text = _ots_hide(stream, delta_text, delta_token_ids)\n"
        "        delta_message = self._ots_parse_delta(\n"
        "            delta_text, delta_token_ids, request, prompt_token_ids, finished=finished\n"
        "        )\n"
        "        if stream is not None and stream.hidden:\n"
        "            delta_message = _ots_unhide(delta_message)\n"
        "        return delta_message\n"
        "\n"
        "    def _ots_parse_delta(\n"
        "        self,\n"
        "        delta_text: str,\n"
        "        delta_token_ids: list[int],\n"
        "        request: ChatCompletionRequest | ResponsesRequest,\n"
        "        prompt_token_ids: list[int] | None = None,\n"
        "        *,\n"
        "        finished: bool,\n"
        "    ) -> DeltaMessage | None:\n"
        "        self._initialize_history_tool_call_cnt(request)\n"
        "        state = self._stream_state\n",
    ),
    (
        "    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:\n"
        '        """Count reasoning tokens through the configured reasoning parser."""\n'
        "        if self._reasoning_parser is None:\n"
        "            return 0\n"
        "        return self._reasoning_parser.count_reasoning_tokens(token_ids)\n",
        "    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:\n"
        '        """Count reasoning tokens through the configured reasoning parser."""\n'
        "        if self._reasoning_parser is None:\n"
        "            return 0\n"
        "        # [DSV41_OUTPUT_TOKEN_SPLIT] the non-streaming count, on the masked text.\n"
        "        count = _ots_count_reasoning(self, token_ids)\n"
        "        if count is not None:\n"
        "            return count\n"
        "        return self._reasoning_parser.count_reasoning_tokens(token_ids)\n",
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
