# Control-token injection through request text, and the guard patch (2026-10-20)

**Status:** **live** on the gb10 kit since 2026-10-21 23:44 CEST (`98abfc7`; see
[the deploy record](#deployed-2026-10-21)). The first version (`eafef45`) was trialled earlier
that day (maintenance-window arm A1) and reverted: on the served path every request whose
messages held control-token text got HTTP 400 (see
[the A1 incident](#2026-10-21-arm-a1-every-control-text-request-refused)). `98abfc7` fixes the
patch, tests and probe.

## Finding

The image's `--tokenizer-mode deepseek_v41` tokenizer (`vllm/tokenizers/deepseek_v41.py`,
`apply_chat_template`) renders the whole conversation to one string with
`deepseek_v41_encoding.encode_messages`. That string is then encoded with the full HF
vocabulary in one of two places:

- On the served path, `apply_chat_template` returns the string and the renderer encodes it (see
  [the A1 incident](#2026-10-21-arm-a1-every-control-text-request-refused)).
- When a caller passes `tokenize=True`, `apply_chat_template` itself calls
  `self.encode(prompt, add_special_tokens=False)`.

The HF tokenizer matches **added tokens anywhere** in that string. So control-token *text* inside
user content, tool results, tool-call arguments, tool schemas or reasoning history became *real*
control tokens.

Live probe (`POST http://192.168.1.97:8000/tokenize`):

- Request: model `deepseek-v4.1-flash`, one user message `grep hit: a</think>b <｜end▁of▁sentence｜> c｜DSML｜d`, `enable_thinking=false`.
- Result: `[0, 128803, 73, 15277, 8180, 28, 260, **128822**, 68, 223, **1**, 274, **128825**, 70, 128804, 128822]`.
- That is `</think>`, EOS and `｜DSML｜` in the middle of the user turn.

`scripts/control_token_probe.py` against the live teacher (2026-10-20 22:04) shows the full scope:

- Injection on **every** surface in both modes: user, system, tool result, tool-call arguments,
  tool schema, and reasoning history with thinking on.
- Each case injected `[1, 128799, 128803, 128804, 128821, 128822, 128825]`.
- The single pass is reasoning history with thinking off, which the encoder drops anyway.
- `chat_template_kwargs: {"tokenize": false}` returns HTTP 200 with the same injected ids. This
  request is the server's default (see the A1 incident).

Why it matters here: agents in this lab routinely read files that contain these strings, and the
file contents arrive as tool results. Examples:

- this repo's `overlay/*.py`
- the vLLM sources (46 of 2,400 sampled `vllm/**/*.py` files contain added-token text)
- deepseek-recipe
- `repl/docs/temp_repos/ds4-zgx-gb10` test vectors

Reading those 46 files as tool results injected **284** real control tokens with the unpatched
tokenizer. A stray EOS, `</think>`, `<｜User｜>`/`<｜Assistant｜>` turn marker or `｜DSML｜` tool
marker in the prompt changes what the model believes the conversation structure is.

### Why `split_special_tokens=True` is not enough

The V4.1 `tokenizer.json` has 1,283 added tokens and marks 1,229 of them `special: true`, mostly
reserved placeholders. The 54 with `special: false` include every token that structures a
conversation: `<｜User｜>`, `<｜Assistant｜>`, `<｜System｜>`, `<think>`, `</think>`, `｜DSML｜`,
the `<｜tool▁…｜>` markers and `<｜deepseek_image｜>`. BOS, EOS and pad (ids 0–2) are special.
`split_special_tokens` (`encode_special_tokens` in `tokenizers`) skips only special tokens.
Measured:

```
encode(probe, split_special_tokens=True) -> [... 128822, 68, 818, 28217, ..., 128825, 70]
```

That neutralizes EOS only. Text segments need a tokenizer **without** the added vocabulary.
The tokenizer's normalizer is an empty sequence, so added tokens match literally, which is what
the guard's scan assumes.

### deepseek-recipe does not do this (yet)

DeepInfer's `doc/inference_deployment_qa.md` ("DeepSeek Recipe") describes DeepSeek's serving:

- It records the special tokens the template inserted, with their positions.
- It encodes ordinary text with the base vocabulary only.

The open-source **deepseek-recipe 0.1.1** (repl's `router_v2/.venv`; local checkout `530895b`) does
not. `PromptEncoding::encode` renders the prompt and calls `tokenizer.encode(prompt, false)`.
`DeepseekV41Encoding().with_tokenizer(...).encode(...)` on the same probe returns
`[0, 128803, …, 128822, 68, 223, 1, 274, 128825, 70, 128804, 128822]`, so delegating to it would
change nothing.

## 2026-10-21 arm A1: every control-text request refused

The first version of the patch (`eafef45`) went live in the 2026-10-21 maintenance window as arm
A1 (gb10 kit `/mnt/storage1/dsv41/exl3-kit`, image `0aecf9f5`). Results:

- **Clean prompts were fine.** Their ids were identical to the A0 capture (18 lists,
  `_repl_measurement_stash/maint-20261021/ctg-A0.json`, which equals this repo's 2026-10-20
  baseline). A1's prefix-reuse and decode numbers were in line with A0's: append reuse 99.6%
  on both, c=1 decode median 43.24 vs 43.33 tok/s.
- **Every request with control-token text was refused.** This held on `/tokenize` and
  `/v1/chat/completions`, with no `tokenize` key sent, and it returned HTTP 400:
  `DeepSeek V4.1: message text contains control-token strings that tokenize=False would turn
  into control tokens; remove 'tokenize' from chat_template_kwargs`. The only case that passed
  was reasoning history with thinking off, which the encoder drops.
- **The probe misreported it.** It printed the refusals as `ERROR` (11 rows), and its own
  "`tokenize=false` → HTTP 400" check passed for the wrong reason.

This is worse than the injection for agents: once a tool result holding such text is in a
session's history, every later turn of that session fails. No client traffic reached A1. The
guard was removed at the A3 restart, back to `422973a`'s `start.sh`, which does not run it.
The window's full record is [maintenance-window-20261021.md](maintenance-window-20261021.md).

### Root cause

The served path asks the template for a **string**. In the image (vLLM `0.1.dev20904`):

1. `OnlineRenderer.preprocess_chat` (`renderers/online_renderer.py`) merges these defaults under
   the request's own kwargs:
   `tools=tool_dicts, tokenize=(is_mistral_tokenizer(...) or model_config.enable_prompt_embeds)`.
   The merge is `ChatParams.with_defaults` → `merge_kwargs`: defaults, then the request's
   `chat_template_kwargs` with `None` values dropped. For this deploy that is
   **`tokenize=False`**, for chat completions, `/tokenize` and the Responses API alike.
2. `DeepseekV4Renderer.render_messages` passes those kwargs to `apply_chat_template`.
   `parse_dec_only_prompt` turns a returned string into a `TextPrompt`.
3. `BaseRenderer._tokenize_prompt` encodes the string with the full vocabulary:
   `tokenizer(prompt, add_special_tokens=False, truncation=…, max_length=…)`.

Three consequences:

- On the live server, the guard's id path never ran. For a dirty prompt only its refusal branch
  ran.
- The injection itself happens in step 3, not in the template.
- A request's explicit `"tokenize": false` is the same value as the default. Refusing it
  refused every dirty request.

The unit test and the local stand-in server both called `apply_chat_template` without the
server's merged kwargs, where `tokenize` defaults to `True`. Neither saw this path. The doc's
earlier note that "the renderer re-encodes the returned string" was right. The conclusion that
only an explicit `tokenize: false` reaches that point was wrong.

Reproduced offline on the image's own serving code (CPU, see
[Verification](#verification)): with `eafef45`, chat, `/tokenize` and Responses all reach
`apply_chat_template` with `tokenize: False` and raise A1's exact message.

## The patch: `overlay/patch_control_token_injection.py`

The patch is applied at container start on both ranks, right after the effort patch, in the same
style as the other runtime patches:

- anchored replacements
- exactly one match per anchor or `RuntimeError`
- idempotent (`already applied`)

### Rendering: marked copy

`_ControlTokenGuard` loads a **private copy** of `deepseek_v41_encoding.py` and replaces every
added-token string in its module-level constants with a nonce marker
(`<128-bit hex>#<n>`). The constants are `bos_token`, `eos_token`, the think tokens,
`dsml_token`, `USER_SP_TOKEN`, … and the derived `assistant_msg_template`. `_normalize_messages`
takes the marked image placeholder.

Each request is rendered twice: once as today, once through the marked copy. The markers are
**exactly** the control tokens the template inserted. Everything between them is text, wherever it
came from, including strings the encoder transforms itself. For example, `json.loads` turns
`</think>` in tool arguments into a raw `</think>`. That is why provenance comes from the
encoder's constants and not from escaping the request.

A prompt is **dirty** when a text segment holds an added-token string; otherwise it is clean.
Dirty text segments are encoded with a base-vocabulary tokenizer: `tokenizers.Tokenizer` built
from the backend's JSON with `added_tokens` removed. It is snapshotted at load and built lazily
on the first dirty request. Template markers map straight to their ids.

### What `apply_chat_template` returns

| `tokenize` | clean prompt | dirty prompt |
|---|---|---|
| false (the served path; also a request's explicit `"tokenize": false`) | the same string as today; the renderer encodes it exactly as before | the guarded ids (`list[int]`) |
| true (direct callers; a request's explicit `"tokenize": true`) | the same `self.encode(prompt, …)` call as today | the guarded ids, with HF `truncation`/`max_length` semantics |

- **Clean prompts:** ids are identical by construction on both paths.
- **Dirty prompts on the served path:** `parse_dec_only_prompt` makes the ids a
  `TokensPrompt`, which the renderer never re-encodes. `apply_post_tokenization` still applies
  the request's `truncate_prompt_tokens`/`truncation_side` and the context-length check, and
  `echo` decodes the ids back to the same prompt string. `chat_template_kwargs`
  `truncation`/`max_length` are ignored there, as they were for the string.
- **The explicit `tokenize: false` case returns ids; it is not refused.** It merges to the same
  value as the server default, so a refusal cannot be targeted at it (A1). Ids are safe because
  a token list is never re-encoded. There is no safe string to return. In the image, the only
  caller of this tokenizer's `apply_chat_template` is `DeepseekV4Renderer`, and it takes a token
  list. An offline caller that asks for a string gets ids for a dirty prompt.

Differences that remain for **dirty prompts on the served path** (clean prompts are unchanged):

- `return_prompt_text: true` returns `null`: a token prompt carries no text into the engine.
- `return_token_offsets` returns no offsets (vLLM computes them for text prompts only).
- An over-length prompt is refused by the token-count check, not the character pre-check. Both
  are HTTP 400 "maximum context length".
- The request log shows ids instead of the prompt text.

### Fail-closed checks

- If the marked render does not map back to the prompt string byte for byte, the request is
  refused (`ValueError`, HTTP 400). Causes would be a diverged in-memory encoder or a forged marker.
- At patch time, `check_encoder_source` refuses an encoder that writes a control token anywhere
  but a module constant: an inline literal or a non-constant default argument. The guard could
  not mark such a token. On refusal, `start.sh` prints `WARN` and the engine starts unpatched.

### Output side (unchanged)

These conclusions come from reading `parser/deepseek_v4.py`, `parser/deepseek_v41.py`,
`parser/engine/*` and `reasoning/deepseek_v41_engine_reasoning_parser.py` in the image:

- The `deepseek_v41` parser resolves `THINK_START`/`THINK_END` via `tokenizer.get_vocab()`
  (`<think>` 128821, `</think>` 128822). In streaming, deltas carry ids, so strict mode matches
  those **by id**. A model-written BPE `</think>` stays content. The patch does not touch
  `get_vocab`, `get_added_vocab`, `decode` or `encode`. The test pins all four.
- `is_reasoning_end(prompt_token_ids)` scans backwards for the last 128822/128821. The template's
  trailing `<think>`/`</think>` after `<｜Assistant｜>` is still the last one, so normal prompts
  behave as before. When the last message is an assistant turn, a `<think>` in its content used to
  win the scan; now the scan reaches the template's token, which is an improvement.

Pre-existing output-side gaps the patch does **not** change (noted, not fixed):

1. Non-streaming `DelegatingParser.parse` ignores `model_output_token_ids`. It matches
   `<think>`/`</think>` by text, so a literal `</think>` in a final answer is deleted and a literal
   `<think>` moves the rest into `reasoning`.
2. Tool markers (`<｜DSML｜ calls>`, invoke and parameter tags) are matched by text in both modes.
   A BPE-spelled `｜DSML｜` in model output can still open a tool call.

## Verification

### Tests

Each test was red first, against `eafef45`, the A1 version.

- **`tests/test_control_token_render_path.py`: the image's own serving code on CPU.**
  - What it drives: `OnlineRenderer.render_chat` (what `OpenAIServingChat.render_chat_request`
    delegates to), `ServingTokenization.create_tokenize` and
    `OpenAIServingResponses._make_request`. These go through `DeepseekV4Renderer`,
    `parse_dec_only_prompt`, `TokenizeParams` and `process_for_engine`, with the live serve
    args' parser and tool settings.
  - Inputs: the image's unpatched tree, pinned by sha256 for 14 render-path files, the image's
    package pins and the live `tokenizer.json`.
  - It compares two trees, as `start.sh` leaves them without and with the guard, over 3
    surfaces × 7 cases × 4 modes = 84 requests. Cases: user, user parts, system, tool result,
    tool arguments, tool schema, reasoning history. Modes: thinking off, thinking on, effort
    max, no kwargs.
  - Red on `eafef45`: 88 problems, one for every dirty request refused.
  - Green now:
    - every request succeeds;
    - control ids equal the plain-text twin's;
    - ids decode to the stock prompt string;
    - stock still injects on all three surfaces;
    - clean and plain prompts give stock's ids, and the template still hands the renderer a
      string;
    - an explicit `tokenize` false or true gives the default request's ids;
    - `echo`, `truncate_prompt_tokens` and the chat context-length check apply to guarded ids
      (`/tokenize` has no length limit, on either tree);
    - the probe's 18 clean `/tokenize` lists equal the 2026-10-20 baseline and the A0 capture,
      on both trees.
  - Setup: `scripts/dsv41_vllm_tree.sh` builds the tree and venv, read-only on gb10 (a
    never-started `docker create`). Without the environment the test prints a skip line.
- **`tests/test_control_token_injection.py` (15 tests).** It runs the image's tokenizer module
  (`tests/fixtures/dsv41_tokenizer/`, with `patch_responses_content_types` and
  `patch_reasoning_effort_mapping` applied as `start.sh` does). It uses the real tokenizer when
  `DSV41_TOKENIZER_DIR` or `/model` has one, otherwise a synthetic byte-level tokenizer with the
  V4.1 id layout.
  - Every surface is checked as a direct call and as the server makes it (`_served`: merged
    `tokenize=False`, then the renderer's encode). This is red on `eafef45` with A1's
    `ValueError`.
  - Coverage spans 17 surfaces × 5 modes: user, text parts, image part, system, developer,
    `response_format`, tool result (string/parts), tool arguments (JSON, JSON with `<`
    escapes, nested dict), tool name, tool schema, assistant content + reasoning, the
    `reasoning` alias, mid-conversation system, and assistant-last. For each, the control-id
    sequence equals the one for the same request with plain text, and the ids decode to
    today's prompt string.
  - Ordinary prompts give identical ids and text: every surface × mode with clean text,
    including near-misses `<｜User`, `</think`, `｜DSML`, `<|x|>` and `<thinking>`.
  - `tokenize` false, None or 0 behaves like the stock template: a string for clean prompts,
    guarded ids for dirty ones. Template truncation kwargs are ignored on that path.
  - Truncation with `tokenize=True` follows HF semantics. A diverged encoder is refused. The
    result holds under 8 threads and after pickle. Output-side vocab and decode are unchanged.
    Anchor drift fails closed. The encoder AST check works, and the `start.sh` wiring is in
    place.
  - Passes on the real tokenizer with transformers 5.17.0 / tokenizers 0.23.2 (the image's
    versions), and on the synthetic tokenizer with those versions and with host 5.13.1 / 0.22.2.
- **`tests/test_control_token_probe.py` (4 tests).** Runs `scripts/control_token_probe.py`
  against a local stand-in in four behaviours:
  - **stock** injects: `FAIL … injected` on both arms.
  - **a1** reproduces A1's 400s: every refusal is a `FAIL … control text refused, HTTP 400` on
    both arms, never `ERROR`, for 23 failing checks.
  - **guarded** gives `GUARDED`, with clean ids equal to the 2026-10-20 baseline on the real
    tokenizer.
  - **chat_differs** has chat render a different prompt than `/tokenize`, and the probe catches
    it.
- **Mutants:** 20 of 21 killed (guard and probe). The survivor routes clean prompts with
  `tokenize=True` through the segment path too. It is equivalent: segment encoding was
  id-identical to legacy on every clean case.

### Earlier checks (`eafef45`, still valid for the encoding itself)

- **One-off check with the image's real `hf.py` wrappers** (`CachedTokenizer`, `TokenizerPool`
  deep copies, pickle) on agent-shaped transcripts built from vLLM sources:
  - 180 clean renders were identical across legacy, guarded and pooled.
  - All 46 dirty files came out clean; legacy had injected 284 control tokens across them.
- **Latency on ~1.5 MB prompts** (best of 5):

  | prompt | tokens | unpatched | patched |
  |---|---:|---:|---:|
  | clean | 396K | 747 ms | 755 ms |
  | dirty | 380K | 728 ms | 104 ms |

  Dirty prompts are faster because segments are batch-encoded in parallel. On the served path, a
  dirty prompt is now encoded once, by the guard, instead of by the renderer.
- **Per-process cost:**
  - tokenizer load: about +0.3 s, with ~4 MB of tokenizer JSON held
  - first dirty request: builds the base-vocab tokenizer, ~0.2–0.35 s and ~40–50 MiB RSS
    (isolated measurement)

## Deploy (operator only)

Done on 2026-10-21; see [the record below](#deployed-2026-10-21). These are the steps for a
redeploy, for example after the kit is rebuilt or copied:

- The kit's `overlay/patch_control_token_injection.py` must be this repo's version (`98abfc7`,
  md5 `b1124b1f…`), **never `eafef45`'s** (md5 `e1b3f26f…`). With a `start.sh` that wires the
  guard, the `eafef45` file re-creates A1.
- `start.sh` changes nothing else for the guard, and the kit's `.env` needs no line for it.

Steps:

1. Before the window, run the tests:
   - `tests/test_control_token_injection.py` and `tests/test_control_token_probe.py`, with
     `DSV41_TOKENIZER_DIR` set;
   - `tests/test_control_token_render_path.py`, after `scripts/dsv41_vllm_tree.sh` (see
     [Files](#files)).

   Then save a fresh clean-id baseline, or use the 2026-10-20 one in this repo:

   ```bash
   python3 scripts/control_token_probe.py --tokenize-only --save /tmp/ctg-before.json
   ```

2. Copy `overlay/patch_control_token_injection.py` over the kit's copy in `overlay/`. Check that
   its md5 matches this repo's, not `e1b3f26f…`. Back up the kit's `start.sh`
   (`cp -p start.sh start.sh.bak-pre-ctg-<stamp>`), then copy this repo's `start.sh` over it.
3. On gb10 in the kit (`start.sh` pauses `vllm-watchdog` and resumes it at `/health`, so it
   needs no manual stop):

   ```bash
   SKIP_BUILD=1 SKIP_PULL=1 setsid nohup ./start.sh restart > logs/ctg-restart.log 2>&1 &
   ```

   Replacing a file under `overlay/` moves the recipe stamp. `SKIP_BUILD=1` keeps the pinned
   image (start.sh warns `stamp … != repo …` and does not rebuild).
4. The head **and** worker logs must show `DeepSeek V4.1 control-token injection guard patch:
   applied` (no `WARN`).
5. Verify:

   ```bash
   python3 scripts/control_token_probe.py --compare /tmp/ctg-before.json
   ```

   It must end with `GUARDED`:
   - every case PASS on both arms, 12 `tokenize` and 12 `chat` rows (the chat arm sends 12
     one-token requests; `--tokenize-only` skips it);
   - `tokenize=false with control text -> HTTP 200, same ids as the default request`;
   - clean prompts id-identical.

   Any `FAIL … control text refused` means the A1 version is running: roll back.

   Then run the smoke, with the kit's model id (the script's default id gets a 404):
   `bash tests/test_smoke.sh http://192.168.1.97:8000 deepseek-v4.1-flash`.

**Rollback:** restore the `start.sh` backup and restart. The patch only edits files inside the
container, so nothing persists.

## Deployed 2026-10-21

`98abfc7` went live on the gb10 kit with one restart at 23:34 CEST; it was healthy at 23:44.
Raw output: `_repl_measurement_stash/ctg-deploy-20261021/` (`ctg-before.json`,
`ctg-after-probe.txt`, `live-checks.txt`, `output-side.txt`, `output-side-stream.txt`,
`ctg-restart.log`, `ctg-head-20261021.log` and the check scripts).

**Before the restart:**

- Tests on rai, all green:
  - `test_control_token_injection.py` (15) and `test_control_token_probe.py` (4), on the live
    `tokenizer.json` (md5 `8a8245dc…`, read from the head container);
  - `test_control_token_render_path.py` (84 requests), on the tree `dsv41_vllm_tree.sh`
    extracted from image `0aecf9f5`;
  - the five `test_yoco_*.py` and four `test_swa_final_window_*.py` suites;
  - `bash -n start.sh`.

  `git diff 422973a HEAD -- start.sh` is the guard's 7 lines only.
- Fresh baseline (`--tokenize-only --save`): 11 injection FAILs, as expected before the guard.
  Its 18 clean lists equal the 2026-10-20 baseline in this repo.
- Kit changes:
  - `overlay/patch_control_token_injection.py`: md5 `e1b3f26f…` → `b1124b1f…` (this repo's);
  - `start.sh`: this repo's (unchanged since `eafef45`, md5 `0062235d…`), backup
    `start.sh.bak-pre-ctg-20261021-232939` (`422973a`'s, md5 `659208e0…`).

  `.env` is untouched (SWA `keep`, YOCO `compact`, KEEP 140). Image `0aecf9f5` on both nodes;
  start.sh warned about the stamp mismatch and did not rebuild.
- The teacher was idle: `num_requests_running`/`waiting` 0, and the last non-keepalive request in
  repl's record was at 04:22 UTC.

**Boot, head and worker logs:**

- `DeepSeek V4.1 control-token injection guard patch: applied`, no `WARN`;
- all 7 `YOCO prefill patch: applied` (`keep=140 decoder layers 20..39`, 16 shadow metadata
  builders, +58.8 MiB);
- both `SWA final-window patch: applied` (`keep 332 tokens below the window`).

start.sh paused `vllm-watchdog` and resumed it at health (clean `watchdog_start`, then
heartbeats). Head `MemAvailable` in the banner: 4.2 GiB.

**Probe** (`--compare /tmp/ctg-before.json`):

```text
PASS tokenize user {'enable_thinking': False}
PASS chat user {'enable_thinking': False}
PASS tokenize system {'enable_thinking': False}
PASS chat system {'enable_thinking': False}
PASS tokenize tool_result {'enable_thinking': False}
PASS chat tool_result {'enable_thinking': False}
PASS tokenize tool_arguments {'enable_thinking': False}
PASS chat tool_arguments {'enable_thinking': False}
PASS tokenize tool_schema {'enable_thinking': False}
PASS chat tool_schema {'enable_thinking': False}
PASS tokenize reasoning_history {'enable_thinking': False}
PASS chat reasoning_history {'enable_thinking': False}
PASS tokenize user {'enable_thinking': True}
PASS chat user {'enable_thinking': True}
PASS tokenize system {'enable_thinking': True}
PASS chat system {'enable_thinking': True}
PASS tokenize tool_result {'enable_thinking': True}
PASS chat tool_result {'enable_thinking': True}
PASS tokenize tool_arguments {'enable_thinking': True}
PASS chat tool_arguments {'enable_thinking': True}
PASS tokenize tool_schema {'enable_thinking': True}
PASS chat tool_schema {'enable_thinking': True}
PASS tokenize reasoning_history {'enable_thinking': True}
PASS chat reasoning_history {'enable_thinking': True}
PASS tokenize=false with control text -> HTTP 200, same ids as the default request
PASS clean prompts id-identical to /tmp/ctg-before.json (18 lists)
GUARDED
```

**Other live checks** (`live-checks.txt`):

| Check | Result |
|---|---|
| smoke `17*19` | 323 |
| Dirty tool result longer than YOCO's KEEP: 21 paragraphs of prose holding `</think>`, `<think>`, `｜DSML｜`, `<｜end▁of▁sentence｜>` and `<｜User｜>`; 1,304 prompt tokens with thinking off, 1,330 with it on | Both modes: HTTP 200, `stop`, correct answer (the budget from the notes). Chat prompt ids equal `/tokenize`'s, and the control ids equal the plain twin's (28 / 29). |
| The same request again | prefix hit, 1,024 tokens cached of 1,304 / 1,330 (inside YOCO's N − 140 cap) |
| `prompt_logprobs: 1` on a clean short and the dirty long prompt | HTTP 400 with the YOCO message |
| Streaming tool call (`get_weather`): clean with thinking off and on, and after the dirty long history | `finish_reason: tool_calls`; the arguments parse as JSON (`{"city": "Paris"}`, `{"city": "Brest"}`) |
| Head and worker logs after all checks | no traceback, no `ERROR` |

**Output side: pre-existing, not a deploy failure.** Asked to summarize the notes, the model
quoted their control strings back. It wrote them as plain BPE text. The only control ids in its
completions were its own `</think>` (128822) and EOS (`output-side.txt`).

- **Streaming is correct.** The quotes stay in `content`, and reasoning ends only at the real
  `</think>` (`output-side-stream.txt`).
- **Non-streaming mangles the quotes.** This is gap 1 under
  [Output side](#output-side-unchanged): the non-streaming parse matches text, not ids.
  - A quoted `<think>draft</think>` in a thinking-off answer moved `draft` into `reasoning`.
  - Quoted `</think>` and `<｜end▁of▁sentence｜>` were missing from `content`. The
    `<｜end▁of▁sentence｜>` drop is not covered by gap 1's description; its cause was not traced.
  - In one thinking-on sample (temperature 1.0), the model quoted `</think>` inside its
    reasoning. The rest of the reasoning came back as `content`.
- **The guard does not cause this.** It rewrites only `tokenizers/deepseek_v41.py`, not the
  parsers. Before the guard, the model got these strings as real control tokens.

The check script's strict "no control text in content" row counted these 4 non-streaming
requests as FAIL (25 of 29 checks pass). All 4 are this output-side effect.

**Rollback:** in the kit, `cp -p start.sh.bak-pre-ctg-20261021-232939 start.sh`, then
`SKIP_BUILD=1 SKIP_PULL=1 ./start.sh restart`.

## Files

- `overlay/patch_control_token_injection.py`: the patch and `check_encoder_source`
- `tests/test_control_token_injection.py`, `tests/fixtures/dsv41_tokenizer/`: unpatched image
  tokenizer sources
- `tests/test_control_token_render_path.py`: the image's serving code (chat, `/tokenize`,
  Responses)
- `scripts/dsv41_vllm_tree.sh`: builds that test's environment, read-only on gb10:

  ```bash
  scripts/dsv41_vllm_tree.sh /tmp/dsv41-vllm
  DSV41_VLLM_TREE=/tmp/dsv41-vllm DSV41_VLLM_PYTHON=/tmp/dsv41-vllm/venv/bin/python \
    DSV41_TOKENIZER_DIR=/path/to/DeepSeek-V4.1-Flash python3 tests/test_control_token_render_path.py
  ```

  It uses `tests/fixtures/dsv41_render/requirements.txt`, the image's pins.
- `tests/test_control_token_probe.py`: the probe against a local stand-in
- `scripts/control_token_probe.py`: live probe, `/tokenize` plus a chat-completions arm
  (`max_tokens=1`)
- `docs/control-token-injection-baseline-20261020.json`: clean-prompt ids from the live teacher before the patch
- `start.sh`: `CONTROL_TOKEN_PATCH_HOST`, preflight, scp, mounts, and patch loop on both ranks
