# Non-streaming output parsing by token id (2026-10-22)

**Status:** prepared, **not deployed**. `overlay/patch_output_token_split.py` (runtime, both
ranks, wired in `start.sh`), `tests/test_output_token_split.py`. Deploying it needs one restart,
on the operator's go; see [Deploy](#deploy-operator-only).

## The problem

After the control-token guard went live (2026-10-21, see
[control-token-injection-20261020.md](control-token-injection-20261020.md#deployed-2026-10-21)),
the model reads control-token strings in tool results as plain text. It may quote them back as
plain BPE text: the only control ids in its completions were its own `</think>` (128822) and
EOS (1).

- **Streaming is correct.**
- **Non-streaming `/v1/chat/completions` mangled the quotes** (`output-side.txt`):
  - a quoted `<think>draft</think>` moved `draft` into `reasoning`;
  - quoted `</think>` and `<｜end▁of▁sentence｜>` were missing from `content`;
  - in one thinking-on sample, a quoted `</think>` inside the reasoning ended it.

## Trace (image `0aecf9f5`, vLLM `0.1.dev20904`)

Live serve args: `--tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41
--enable-auto-tool-choice`. `ParserManager.get_parser` composes a `DelegatingParser` subclass
from `DeepSeekV41ParserReasoningAdapter` and `DeepSeekV41EngineToolParser`. Both adapters wrap
their own `DeepSeekV41Parser` (`ParserEngine`).

### Where non-streaming splits reasoning and content

1. **`OpenAIServingChat.chat_completion_full_generator`** calls
   `parser.parse(output.text, request, enable_auto_tools=..., model_output_token_ids=output.token_ids)`.
   - `output.text` comes from `FastIncrementalDetokenizer`, with `skip_special_tokens=False`
     (`ParserEngine.adjust_request`), so a real `</think>` or `｜DSML｜` appears in it as text.
   - The final EOS is not in the text (`stop_terminated`), but it is in the ids.
   - The same call is made by batch chat (`batch_serving.py`), the Responses API
     (`responses/serving.py`) and the Responses finalizer (`responses/context.py`).
2. **`DelegatingParser.parse`** (`parser/abstract_parser.py`) **drops `model_output_token_ids`**:
   - `extract_reasoning(model_output)`, then `ParserEngineReasoningAdapter.extract_reasoning`,
     then `ParserEngine.extract_reasoning`, which runs `self._feed(model_output, [])`: the
     whole text, no ids, tool parsing skipped;
   - then `_extract_tool_calls(content)`, then
     `ParserEngineToolAdapter.extract_tool_calls`, then
     `ParserEngine.extract_tool_calls_from_content`, which runs
     `_single_pass_parse(content, [], initial_state=CONTENT)`: the content a second time, text
     only.
3. **`StreamingParserEngine._process_lex_tokens`** switches to "strict" mode only after a delta
   has carried ids (`_ever_had_token_ids`). In strict mode, `THINK_START`/`THINK_END`/DROP
   lexemes that the lexer found in *text* become content. Streaming deltas carry ids, so
   streaming is strict. Non-streaming never is.

### Why each quote broke

- **`</think>` dropped from content:** a text `</think>` in CONTENT hits the transition
  `(CONTENT, THINK_END) → CONTENT` with no events ("absorb a bare `</think>`").
- **`<think>draft</think>` → `reasoning: "draft"`:** `(CONTENT, THINK_START) → REASONING`, then
  `THINK_END` back. In the tool pass, a `<think>` left in the content would discard the text
  after it (that pass's reasoning is thrown away).
- **`</think>` inside reasoning:** `(REASONING, THINK_END) → CONTENT` at the quote. The real
  `</think>` later is then absorbed in CONTENT.
- **`<｜end▁of▁sentence｜>` dropped (not gap 1):** `_build_drop_info`
  (`parser/engine/streaming_parser_engine.py`) takes every `tokenizer.all_special_tokens` entry
  that is not a configured terminal.
  - For V4.1 that is `<｜begin▁of▁sentence｜>` and `<｜end▁of▁sentence｜>` (pad = EOS).
  - It registers them as `DROP_TERMINAL` **by id and by text**: `drop_terminal_defs` are
    literal lexer terminals.
  - `_on_terminal` returns `[]` for DROP, so in non-strict mode a quoted EOS or BOS disappears
    from content, reasoning and tool arguments alike.
  - In strict (streaming) mode the text lexeme is content, and only the real ids are dropped.
- **Gap 2, tool markers, matched by text:** `｜DSML｜` (128825) is not a token-id terminal; the
  V4.1 markers `<｜DSML｜ calls>` etc. are multi-token text terminals in both modes. Non-streaming,
  stock, on the image (test cases):
  - a quoted DSML call block came back as a bogus `get_weather({"city": "Paris"})` call, with
    `finish_reason: tool_calls`;
  - a quote followed by a real call came back as **two** calls;
  - `</think>` / EOS text inside a real call's argument was deleted from the argument.

**Real tool calls use the token.** Live 2026-10-22, two requests while the teacher was idle,
raw output in `_repl_measurement_stash/ots-dsml-ids-20261022/`, fixture
`tests/fixtures/dsv41_output_split/tool-call-ids-20261022.json`. With thinking off and on,
every one of the 6 `｜DSML｜` in each call was id 128825. A BPE-spelled `｜DSML｜` in model
output is therefore a quote.

### Why "just pass the ids" is not the fix

`ParserEngine.parse(model_output, ..., model_output_token_ids)` exists in the image, but it feeds
the whole text and all ids as one delta. `TokenIDScanner` checks the per-token decode against
the text. That check fails when:

- EOS is in the ids but not the text;
- a multi-byte character is split across tokens.

On a failed check, `_rebuild_from_anchors` binds each id terminal to the **rightmost**
occurrence of its text (`rfind`). On the recorded thinking-on completion, the real `</think>`
and EOS bind to the quotes in the answer. The answer's first part goes into `reasoning` and
both quotes are lost (checked on the image tree), which is worse than stock. Streaming is safe
only because deltas are a few tokens long.

## The patch: `overlay/patch_output_token_split.py`

The patch targets `vllm/parser/abstract_parser.py`, at two anchors. Each anchor must match
exactly once or the patch raises `RuntimeError`, and it is idempotent (`already applied`).
`start.sh` runs it on both ranks, after the control-token guard.

`DelegatingParser.parse` (engine-based parsers only) now uses the ids it is given:

1. **Prefilter.** If the text holds no added-token string, the stock parse runs on the
   unchanged text.
2. **Split the ids into runs** at added-token ids (all 1,283 V4.1 added tokens). Decode the
   runs with the tokenizer's Rust backend, the one `FastIncrementalDetokenizer` streams with.
3. **Align.** `output.text` must be a **prefix** of the decoded pieces. A stripped stop token,
   a stop string or held-back UTF-8 only shorten it. Otherwise the text is parsed as before,
   with one warning per process.
4. **Mask.** An added-token string found inside a run's decoded text was spelled with ordinary
   tokens. Each one is replaced by a nonce placeholder (`DSV41OTS<96-bit hex>n<i>z`: ASCII, no
   lexer or JSON-special characters). Real control tokens stay as text, exactly as before.
   - If there was nothing to mask, the stock parse runs on the unchanged text.
5. **Parse.** The image's own two-pass parse runs on the masked text, unchanged.
6. **Restore.** The originals go back into `reasoning`, `content` and every tool call's `name`
   and `arguments`. The V4.1 added-token strings need no JSON escaping, so they are restored
   literally inside argument JSON.
7. **Fail-safe:**
   - a placeholder that did not survive intact: re-parse the original text (stock);
   - any exception: stock;
   - a text already holding the placeholder prefix: stock.

   Misalignment, an exception and a cut placeholder log
   `DeepSeek V4.1 output token split: <reason>; that output is parsed by text` once per
   process per reason. A forged prefix is not logged.

Properties:

- **Clean completions are untouched.** They never reach the masked parse: a text with no
  added-token string, or with only real ones, takes the stock path with the identical string.
  Real tool calls are therefore unchanged.
- **The result is the streaming result.** The parse sees control-token text only where the
  model emitted the control token. This also holds for BOS/EOS and for DSML inside arguments.
- **Gap 2 is fixed for non-streaming.** A quoted DSML block stays content. A real call that
  follows a quote is the only call.
- **Not changed:**
  - streaming;
  - the prompt side;
  - `count_reasoning_tokens`, which recounts from the ids; the test pins
    `usage.completion_tokens_details` unchanged.
- **Cost:** 3.4 ms for an 18,349-token completion holding 300 quotes, on the rai CPU. One
  `decode_batch` and one trie-regex scan; the regex is built once per tokenizer.

### Not fixed (documented)

- **Streaming still matches DSML by text (gap 2, streaming).** A quoted `<｜DSML｜ calls>`
  block still opens a tool call in a stream; the test pins this as a characterization. A fix
  needs the lexer to accept `｜DSML｜` inside its marker terminals only when it came from id
  128825. The masking approach does not carry over: delta text and delta ids do not align
  exactly (detokenizer hold-back, stop-string buffering), and a quote can straddle deltas.
- **`count_reasoning_tokens`** (non-streaming usage) feeds the decoded text plus all ids in
  one delta, so it shares the `rfind` anchor issue above. It only matters when the per-token
  decode differs from the text. On both recorded completions it is correct (0 and 47).

## Verification: `tests/test_output_token_split.py`

The test runs the image's own code on CPU (tree and venv from `scripts/dsv41_vllm_tree.sh`, live
`tokenizer.json`), with 18 parse-path files sha256-pinned:

- completions are detokenized by the image's `IncrementalDetokenizer`;
- they are parsed by `OpenAIServingChat.chat_completion_full_generator` (non-streaming) and
  `chat_completion_stream_generator` (deltas of 3 tokens);
- one Responses-API `parse` per case.

Two trees: the image as `start.sh` leaves it today (tokenizer patches including the guard), and
the same plus this patch.

**Completions (14):**

- **`live_off` / `live_on`**, the recorded ones. Their ids were not saved, so they are rebuilt
  from the recorded raw decode: control ids at the recorded positions, text runs in the base
  vocabulary. They reproduce the recorded lengths and positions exactly (169 with EOS at 168;
  232 with `</think>` at 47 and EOS at 231). Stock reproduces the recorded live
  `reasoning`/`content` byte for byte.
- **`tool_off` / `tool_on`**: real calls, live ids.
- **Synthetic:**
  - `</think>` + EOS quoted inside reasoning;
  - `<think>draft</think>` + BOS quoted;
  - a quoted DSML block;
  - a quote followed by a real call;
  - `</think>` + EOS + `｜DSML｜` inside a real call's argument;
  - a stop-string cut;
  - a cut UTF-8 character (`finish_reason: length`);
  - `live_on` spelled one character per token;
  - two clean completions.

**Pins:**

- **Patched non-streaming** gives the expected reasoning, content and tool calls, and
  `finish_reason` follows them.
- **Expected = the stream.** The expectation equals the streaming result, except the two gap-2
  cases, where streaming still opens a call.
- **Stock is red** on every dirty case.
- **Clean completions and real calls:** identical on both trees, and they take the stock path.
- **Unchanged:** streaming output, detokenized text and reasoning-token usage.
- **Responses:** the API's parse equals chat's.
- **Fail-safes:** misaligned ids, a failing decode, a cut placeholder (a stock re-parse
  follows) and a forged prefix all give the stock text parse, with exactly one log line.
- **No fallback** is logged on the main cases.
- **Patch file:** anchors, drift refused, idempotency, and the `start.sh` wiring.

**Red first.** A no-op patch (`DSV41_OTS_PATCH`, the parse as it is live today) gives 32
problems: every dirty case, both Responses cases, and the wiring/idempotency checks. With the
patch the suite is green:

```text
Output token split on the real serving path OK (14 completions, chat full + stream generators,
Responses parse, 3 fail-safes; 18349-token completion split in 3.35 ms; patch_output_token_split.py)
```

### Mutants

11 of 11 were killed (`/tmp/ots-mutants/run.py` style: one text edit of the patch, the full
suite run against it):

| Mutant | Killed by |
|---|---|
| mask real control tokens too (scan every piece) | 21 problems (`live_on`, tool calls, …) |
| no restore in tool-call name/arguments | `control_text_in_argument` |
| exact text match instead of prefix | 26 problems (every EOS-terminated case) |
| no leftover-placeholder check | `cut_placeholder` fail-safe |
| no forged-prefix check | forged placeholder rewritten |
| split even when nothing was masked | clean completions leave the stock path |
| no truncation of the last visible piece | `stop_string` |
| mask only `<think>`/`</think>` | 13 problems (EOS, BOS, DSML cases) |
| no engine-based gate | legacy parser split |
| no restore in reasoning | `reasoning_quote` |
| no stock re-parse after a cut placeholder | driver error (`None` unpacked) |

```bash
scripts/dsv41_vllm_tree.sh /tmp/dsv41-vllm   # once; read-only on gb10
DSV41_VLLM_TREE=/tmp/dsv41-vllm DSV41_VLLM_PYTHON=/tmp/dsv41-vllm/venv/bin/python \
  DSV41_TOKENIZER_DIR=/path/to/DeepSeek-V4.1-Flash python3 tests/test_output_token_split.py
```

Without the tree it checks only the `start.sh` wiring and prints a skip line.

## Deploy (operator only)

Not done. One restart. The patch only edits a file inside the container at start, so nothing
persists across a rollback.

1. **Before the window, on rai:**
   - run `tests/test_output_token_split.py` (above) and the guard suites
     (`test_control_token_injection.py`, `test_control_token_probe.py`,
     `test_control_token_render_path.py`);
   - run `bash -n start.sh`;
   - check that `git diff c35177c -- start.sh` shows only this patch's 7 lines.
2. **Prepare the kit** (`gb10:/mnt/storage1/dsv41/exl3-kit`, a non-git snapshot):
   - copy `overlay/patch_output_token_split.py` into the kit's `overlay/`;
   - back up `start.sh` (`cp -p start.sh start.sh.bak-pre-ots-<stamp>`);
   - copy this repo's `start.sh` over it;
   - check that the kit's `overlay/patch_control_token_injection.py` is still md5
     `b1124b1f…` (the `start.sh` copy also wires the guard).
3. **Wait for idle.** `num_requests_running` and `num_requests_waiting` must be 0 on
   `:8000/metrics`, and repl must show no live session.
4. **Restart on gb10 in the kit.** `start.sh` pauses `vllm-watchdog` and resumes it at
   `/health`:

   ```bash
   SKIP_BUILD=1 SKIP_PULL=1 setsid nohup ./start.sh restart > logs/ots-restart.log 2>&1 &
   ```

   Replacing a file under `overlay/` moves the recipe stamp; `SKIP_BUILD=1` keeps the pinned
   image.
5. **Head and worker logs** must show both of these (no `WARN`):
   - `DeepSeek V4.1 output token split patch: applied`
   - `DeepSeek V4.1 control-token injection guard patch: applied`

   The usual YOCO/SWA lines must be unchanged.
6. **Verify:**
   - `bash tests/test_smoke.sh http://192.168.1.97:8000 deepseek-v4.1-flash` → 323.
   - Re-run the deploy record's `ctg_output_side.py` (non-streaming, both modes). Every quoted
     control string must now be in `content`:
     - `<think>draft</think>` in content;
     - `reasoning` null with thinking off;
     - the thinking-on reasoning ending at the real `</think>`.

     `ctg_stream_split.py` must be unchanged.
   - Re-run `_repl_measurement_stash/ots-dsml-ids-20261022/dsml_ids.py`. It must return
     `get_weather {"city": "Paris"}`, `finish_reason: tool_calls`, both modes.
   - Grep the head log for `output token split:`. Any line means a fallback fired; collect it.
   - Repeat the `live-checks.txt` streaming tool-call checks.

**Rollback:** `cp -p start.sh.bak-pre-ots-<stamp> start.sh`, then
`SKIP_BUILD=1 SKIP_PULL=1 ./start.sh restart`.
