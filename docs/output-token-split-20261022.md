# Non-streaming output parsing by token id (2026-10-22)

**Status:**

- **Non-streaming:** **live on the gb10 kit since 2026-10-22 15:38 CEST** (one restart; record in
  [Deploy record](#deploy-record-2026-10-22)).
- **Streaming DSML + the non-streaming reasoning-token count:** prepared 2026-10-22, **not
  deployed** ([Streaming](#streaming-and-the-reasoning-token-count-2026-10-22-prepared)). It is
  the same file, so deploying it is one file copy and one restart; `start.sh` is unchanged.

`overlay/patch_output_token_split.py` (runtime, both ranks, wired in `start.sh`),
`tests/test_output_token_split.py`, `scripts/dsml_stream_probe.py` (live check).

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
- **Not changed:** the prompt side. (Streaming and the reasoning-token count were not changed
  by the deployed version; the 2026-10-22 extension below changes both.)
- **Cost:** 3.4 ms for an 18,349-token completion holding 300 quotes, on the rai CPU. One
  `decode_batch` and one trie-regex scan; the regex is built once per tokenizer.

## Streaming and the reasoning-token count (2026-10-22, prepared)

**Why it matters more than the non-streaming fix:** 7,696 of the 7,715 caller requests in
repl's record since 10-14 are `stream: true` (dsh and Hermes). The deployed non-streaming fix
covers almost none of the agent traffic.

**Live evidence** (`scripts/dsml_stream_probe.py`, gb10 kit, 2026-10-22, before this change;
`_repl_measurement_stash/ots-stream-20261022/probe-before.json`):

- Asked to copy a DSML block verbatim, the model wrote it with ordinary tokens (no id 128825
  in its 59 ids).
- The stock stream returned a `get_weather {"city": "Paris"}` tool call and the content
  `` ```\n\n```\n\ndone ``. An agent would run a call the model never made.
- Real calls, streamed, thinking off and on: correct, 6 × id 128825 each.

**Mechanism.** The stream parser (`StreamingParserEngine`) gets each delta's text and ids. Its
id-matched terminals are `</think>`, `<think>` and the specials (DROP). The V4.1 DSML markers
(`<｜DSML｜ calls>`, invoke, parameter and their closers) are multi-token strings, so
`token_id_terminals` cannot resolve them and the lexer matches them by text. The same text
matching inside a real call's arguments ends the call early. A path argument holding quoted
markers became `read_file {"path": "docs/"}` plus a call to a tool named `x`.

**The change** is in `DelegatingParser.parse_delta`, the path that both chat and Responses
streaming use:

1. **Track where the added tokens' text lands.** Per request, the ids are followed as runs
   between added-token ids. At each added token, the run before it is decoded with the same
   Rust backend as the non-streaming split, so the token's character position in the stream
   is known. Only tokens whose text holds `｜` are kept.
2. **Hide the other `｜`s.** Every DSML marker holds `｜DSML｜`, which a quote spells
   `｜` `DS` `ML` `｜`. Each `｜` in the delta text that is not inside a real added token's
   text is replaced by U+FDD0, a Unicode noncharacter that no tokenizer emits, before the
   stock `parse_delta` runs. A quote can then never complete a marker. This works one
   character at a time, so a quote split over deltas is covered.
3. **Put them back.** The returned delta's content, reasoning and tool-call names and
   arguments get `｜` back. All JSON on this path is `ensure_ascii=False`, so the character
   survives argument conversion.
4. **Never worse than stock:**
   - a real added token's text must be where its ids put it; otherwise the hiding stops for
     the request, logged once (`DeepSeek V4.1 output token split: streaming: …`);
   - while a real `｜`-token's text is still held back (a stop string makes the detokenizer
     hold back its length − 1 characters), that delta is left as it is;
   - any exception stops the hiding for the request, logged once;
   - a literal U+FDD0 in the text stops the hiding before anything is hidden. If it comes
     after a hidden quote, it comes back as `｜` (logged; no model output has held one).
5. **Specials.** The engine parser's `adjust_request` sets `skip_special_tokens=False` on every
   request, so the stream holds every added token's string, specials included. The stop
   token's text is never streamed.

**The non-streaming reasoning-token count.** `ParserEngineReasoningAdapter.
count_reasoning_tokens` re-parses `tokenizer.decode(ids)` in one feed. DSML markers are text
there too, so a DSML quote in the reasoning ended it: the count was 4 instead of 62
(`dsml_quote_in_reasoning`). With such a quote, the count now parses the masked text. The
`rfind` concern documented here before does not move the count: when the anchor rebuild
misplaces `</think>`, the token counts still follow the id groups (`count_quote` pins 7 on
both trees).

**Cost:** 10–12 ms over 40 ms of stock parsing for an 18,349-token stream in 4-token deltas
(4,588 deltas, about 2.4 µs per delta), on the rai CPU.

**Not covered:** the stock stream's own reasoning-token usage at 1-token deltas undercounts by
the tokens whose text is empty (a character split over byte tokens: 24 instead of 26). It is
the image's, unchanged, and the test pins it as such.

## Verification: `tests/test_output_token_split.py`

The test runs the image's own code on CPU (tree and venv from `scripts/dsv41_vllm_tree.sh`, live
`tokenizer.json`), with 18 parse-path files sha256-pinned:

- completions are detokenized by the image's `IncrementalDetokenizer`;
- they are parsed by `OpenAIServingChat.chat_completion_full_generator` (non-streaming) and
  `chat_completion_stream_generator` (deltas of 3 tokens, and since the streaming extension
  also 1 and 4: one DSpark step is up to 4 tokens), with streamed usage;
- one Responses-API `parse` per case.

Two trees: the image as `start.sh` leaves it today (tokenizer patches including the guard), and
the same plus this patch.

**Completions (21; the 7 added with the streaming extension are marked +):**

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
  - two clean completions;
  - + a DSML quote in reasoning before a real call (`dsml_quote_in_reasoning`);
  - + DSML markers quoted inside a real call's argument (`dsml_markers_in_argument`);
  - + a character split over byte tokens before a quoted `</think>` (`count_quote`);
  - + a quote then a real call under a stop string (`dsml_quote_then_call_held`);
  - + a real BOS mid-output before a quote and a call (`special_then_quote_call`);
  - + bare `｜` text (`pipe_text`, clean).
- **+ `live_stream_quote`:** the live probe's 59 ids. Stock must reproduce the live stream
  (the bogus `get_weather` call and the emptied code block).

**Pins:**

- **Patched non-streaming and streaming** (every delta size) give the expected reasoning,
  content and tool calls, and `finish_reason` follows them.
- **Stock is red** on every dirty case: non-streaming everywhere, streaming on every DSML case.
- **Clean completions and real calls:** identical on both trees, and they take the stock path.
- **Unchanged:** detokenized text; streams (and their usage) of every case without a DSML quote.
- **Reasoning tokens:** non-streaming equals the ids before `</think>` (0 with thinking off);
  so does the streamed count of the DSML cases at 3- and 4-token deltas.
- **Responses:** the API's parse equals chat's.
- **Fail-safes, non-streaming:** misaligned ids, a failing decode, a cut placeholder (a stock
  re-parse follows) and a forged prefix all give the stock text parse, with exactly one log
  line.
- **Fail-safes, streaming:** misaligned ids and a literal U+FDD0 give the stock stream; a
  failing decode and ids ahead of the text (the held-back rule) keep the real call. One log
  line each.
- **No fallback** is logged on the main cases.
- **Cost:** the long stream's overhead stays under 0.05 ms per delta.
- **Patch file:** anchors, drift refused, idempotency, and the `start.sh` wiring.

**Red first.** A no-op patch (`DSV41_OTS_PATCH`, the parse as it is live today) gives 32
problems: every dirty case, both Responses cases, and the wiring/idempotency checks. With the
patch the suite is green:

```text
Output token split on the real serving path OK (14 completions, chat full + stream generators,
Responses parse, 3 fail-safes; 18349-token completion split in 3.35 ms; patch_output_token_split.py)
```

**Streaming extension, red first.** Against the deployed version (`DSV41_OTS_PATCH` = the
`d68569a` file) the extended suite gives 26 problems: every DSML case's stream at all three
delta sizes, the 4-vs-62 reasoning-token counts, and the missing stream path. With the
extension it is green:

```text
Output token split on the real serving path OK (21 completions, chat full + stream generators
in deltas of 1, 3, 4, Responses parse, 7 fail-safes; 18349-token completion split in 3.57 ms;
18349-token stream (4588 deltas) +10.0 ms over 40.6 ms; patch_output_token_split.py)
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

Streaming extension: 13 of 15 killed, the 2 survivors equivalent (`/tmp/ots_mut/run.py`):

| Mutant | Killed by |
|---|---|
| hide nothing | 29 problems (every DSML stream) |
| hide real `｜` too | 33 problems (every real call) |
| no check that an added token's text is where its ids put it | `stream_misaligned` |
| no held-back rule | `stream_ids_ahead` (the real call lost) |
| no restore | 69 problems |
| restore content only | `reasoning_quote`, … |
| restore no arguments | `control_text_in_argument`, … |
| no stop on a literal U+FDD0 | `stream_literal_hide` |
| run not reset after a decode | `dsml_markers_in_argument` (and fallbacks logged) |
| count not patched | `dsml_quote_in_reasoning` 4 != 62 |
| count on the unmasked text | `dsml_quote_in_reasoning` 4 != 62 |
| count condition inverted | streamed usage changed |
| drop partly streamed spans | `dsml_quote_then_call_held` (1-token deltas) |
| *survived:* no "text ran ahead of its ids" check | equivalent: the span check catches the same desync at the next `｜` token, before a real one can be hidden |
| *survived:* track every added token, not only those with `｜` | equivalent: detects a desync earlier, never changes which `｜` is hidden |

```bash
scripts/dsv41_vllm_tree.sh /tmp/dsv41-vllm   # once; read-only on gb10
DSV41_VLLM_TREE=/tmp/dsv41-vllm DSV41_VLLM_PYTHON=/tmp/dsv41-vllm/venv/bin/python \
  DSV41_TOKENIZER_DIR=/path/to/DeepSeek-V4.1-Flash python3 tests/test_output_token_split.py
```

Without the tree it checks only the `start.sh` wiring and prints a skip line.

## Deploy the streaming extension (operator only, not done)

One restart. Only `overlay/patch_output_token_split.py` changes; the kit's `start.sh` already
runs it on both ranks.

1. **Before the window, on rai:** run `tests/test_output_token_split.py`,
   `tests/test_dsml_stream_probe.py` and the guard suites (as below).
2. **Prepare the kit** (`gb10:/mnt/storage1/dsv41/exl3-kit`): back up the live file
   (`cp -p overlay/patch_output_token_split.py overlay/patch_output_token_split.py.bak-pre-stream-<stamp>`,
   md5 `95333605…`) and copy this repo's file over it.
3. **Wait for idle** and **restart** exactly as in steps 3–4 below.
4. **Logs:** head and worker show `DeepSeek V4.1 output token split patch: applied` and the guard
   `applied`, no WARN; the SWA/YOCO lines are unchanged.
5. **Verify:**
   - `bash tests/test_smoke.sh http://192.168.1.97:8000` → 323;
   - `python3 scripts/dsml_stream_probe.py`: `real-off` and `real-on` PASS, `quote` PASS (or
     INCONCLUSIVE if the model did not quote; before the deploy it was FAIL);
   - `_repl_measurement_stash/ots-dsml-ids-20261022/dsml_ids.py` still returns
     `get_weather {"city": "Paris"}`, both modes;
   - the head log has no `output token split: streaming:` line after the probe.
6. **Rollback:** copy the `.bak-pre-stream-<stamp>` file back and restart the same way.

## Deploy (operator only)

Done on 2026-10-22 (record below). One restart. The patch only edits a file inside the container at start, so nothing
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

## Deploy record (2026-10-22)

One restart, `SKIP_BUILD=1 SKIP_PULL=1 ./start.sh restart`, launched 15:28:28 CEST, `/health` 200
at 15:38:43. The engine was idle: 0 running and 0 waiting, and repl's last client request was
about 9.6 h earlier. Raw data: `_repl_measurement_stash/ots-deploy-20261022/`.

- **Before the restart, on rai:**
  - `test_output_token_split.py`, plus the guard suites `test_control_token_injection.py` (15),
    `test_control_token_probe.py` (4) and `test_control_token_render_path.py` (84): all green, on
    the live tokenizer.json (md5 `8a8245dc…`, equal to the kit's model copy);
  - `bash -n start.sh` OK;
  - `git diff c35177c -- start.sh` = the patch's 7 lines.
- **Kit changes:**
  - `overlay/patch_output_token_split.py` added (md5 `95333605…`);
  - `start.sh` `0062235d` → `85a22ced` (backup `start.sh.bak-pre-ots-20261022-152816`);
  - the guard file is still `b1124b1f…`, and `.env` is untouched (SWA keep, YOCO compact).
- **Logs:** head and worker both show `output token split patch: applied` and the guard `applied`,
  with no WARN, and the SWA/YOCO lines are unchanged.
- **Verify:**
  - `tests/test_smoke.sh` → 323, both with the id given and with it read from `/v1/models`.
  - **`ctg_output_side.py` (non-streaming):**
    - Thinking on: the raw output holds the real `</think>` (id 128822) and two quoted ones. The
      reasoning keeps one quote and ends at the real token; the content keeps the other quote and
      a quoted `｜DSML｜`.
    - Thinking off: `reasoning` is null. This sample quoted no control text, so it does not test
      the masking.
  - `ctg_stream_split.py` is unchanged (finish `stop`, 0 tool-call deltas, both modes).
  - `dsml_ids.py`: `get_weather {"city": "Paris"}`, `finish_reason: tool_calls`, both modes.
  - **`ctg_live_checks.py`:** 27 PASS / 2 FAIL, against 25 / 4 before the deploy.
    - The two FAILs are thinking-off cold and repeat on that script's "no control text in content"
      check: the model quoted the notes' markers in its summary, and the content now keeps those
      quotes by design.
    - No tool call was produced (`tool_calls=None`).
    - Prompt ids still equal `/tokenize`, and the YOCO prefix hit stays inside N-140.
  - The head log has 0 `output token split:` fallback lines and 0 tracebacks.
- **Rollback (unused):** `cp -p start.sh.bak-pre-ots-20261022-152816 start.sh`, then
  `SKIP_BUILD=1 SKIP_PULL=1 ./start.sh restart`.
