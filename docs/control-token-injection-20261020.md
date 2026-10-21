# Control-token injection through request text, and the guard patch (2026-10-20)

**Status:** patch, tests and probe are ready. **Not deployed.** The live teacher is unchanged;
deploying needs a restart, which only the operator does (see [Deploy](#deploy-operator-only)).

## Finding

The image's `--tokenizer-mode deepseek_v41` tokenizer (`vllm/tokenizers/deepseek_v41.py`,
`apply_chat_template`) does two things:

1. It renders the whole conversation to one string with `deepseek_v41_encoding.encode_messages`.
2. It calls `self.encode(prompt, add_special_tokens=False)`.

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
- `chat_template_kwargs: {"tokenize": false}` returns HTTP 200 with the same injected ids (see below).

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

In the V4.1 `tokenizer.json`, only ids 0–2 (BOS, EOS, pad) are `special: true`. `<｜User｜>`,
`<｜Assistant｜>`, `<think>`, `</think>`, `｜DSML｜`, `<｜System｜>`, `<｜deepseek_image｜>` and the
rest are added tokens with `special: false`. `split_special_tokens` (`encode_special_tokens` in
`tokenizers`) skips only special ones. Measured:

```
encode(probe, split_special_tokens=True) -> [... 128822, 68, 818, 28217, ..., 128825, 70]
```

That neutralizes EOS only. Text segments need a tokenizer **without** the added vocabulary.

### deepseek-recipe does not do this (yet)

DeepInfer's `doc/inference_deployment_qa.md` ("DeepSeek Recipe") describes DeepSeek's serving:

- It records the special tokens the template inserted, with their positions.
- It encodes ordinary text with the base vocabulary only.

The open-source **deepseek-recipe 0.1.1** (repl's `router_v2/.venv`; local checkout `530895b`) does
not. `PromptEncoding::encode` renders the prompt and calls `tokenizer.encode(prompt, false)`.
`DeepseekV41Encoding().with_tokenizer(...).encode(...)` on the same probe returns
`[0, 128803, …, 128822, 68, 223, 1, 274, 128825, 70, 128804, 128822]`, so delegating to it would
change nothing.

## The patch: `overlay/patch_control_token_injection.py`

The patch is applied at container start on both ranks, right after the effort patch, in the same
style as the other runtime patches:

- anchored replacements
- exactly one match per anchor or `RuntimeError`
- idempotent (`already applied`)

### Rendering: marked copy

`_ControlTokenGuard` loads a **private copy** of `deepseek_v41_encoding.py` and replaces every
added-token string in its module-level constants with a nonce marker
(`<128-bit hex>#<n>`). The constants are `bos_token`, `eos_token`, the think tokens,
`dsml_token`, `USER_SP_TOKEN`, … and the derived `assistant_msg_template`. `_normalize_messages`
takes the marked image placeholder.

Each request is rendered twice: once as today, once through the marked copy. The markers are
**exactly** the control tokens the template inserted. Everything between them is text, wherever it
came from, including strings the encoder transforms itself. For example, `json.loads` turns
`</think>` in tool arguments into a raw `</think>`. That is why provenance comes from the
encoder's constants and not from escaping the request.

### Encoding the rendered prompt

- **Clean prompts:** if no text segment holds an added-token string, the prompt goes to the same
  `self.encode(prompt, add_special_tokens=False, …)` call as today. Ids are identical by
  construction.
- **Dirty prompts:** encoded segment-wise.
  - Template markers map straight to their ids.
  - Text segments go to a base-vocabulary tokenizer: `tokenizers.Tokenizer` built from the
    backend's JSON with `added_tokens` removed. It is snapshotted at load and built lazily on the
    first dirty request.
  - `truncation`/`max_length` follow HF semantics.

### Fail-closed checks

- If the marked render does not map back to the prompt string byte for byte, the request is
  refused (`ValueError`, HTTP 400). Causes would be a diverged in-memory encoder or a forged marker.
- `tokenize=False` with dirty text is refused (HTTP 400). Today the renderer re-encodes the
  returned string with the plain tokenizer (`renderers/base.py`), which re-injects.
  `chat_template_kwargs` from the request pass through unfiltered, so this was an open bypass.
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

- **`tests/test_control_token_injection.py` (14 tests).** Red first: the probe test failed with the
  live ids `… 128822, 68, 223, 1, 274, 128825, 70` on a no-op patch. It runs the image's own
  `deepseek_v41.py`/`deepseek_v41_encoding.py` (`tests/fixtures/dsv41_tokenizer/`, with
  `patch_responses_content_types` and `patch_reasoning_effort_mapping` applied as `start.sh` does)
  on the real tokenizer when `DSV41_TOKENIZER_DIR` or `/model` has one. Otherwise it uses a
  synthetic byte-level tokenizer with the V4.1 id layout. Passing results:
  - real tokenizer, transformers 5.17.0 / tokenizers 0.23.2 (the image's versions)
  - synthetic tokenizer on the same versions
  - synthetic tokenizer on host 5.13.1 / 0.22.2

  The pins:

  - The probe's content span holds none of 1/128803/128804/128822/128825 (or any added id). The
    template's `[0, 128803 … 128804, 128822]` stay, and the span decodes back to the probe text.
  - Coverage spans 17 surfaces × 5 modes: user, text parts, image part, system, developer,
    `response_format`, tool result (string/parts), tool arguments (JSON, JSON with `<`
    escapes, nested dict), tool name, tool schema, assistant content + reasoning, the `reasoning`
    alias, mid-conversation system, and assistant-last. For each, the control-id sequence equals
    the one for the same request with plain text, and `decode(ids)` equals today's prompt string.
  - Ordinary prompts give identical ids and text: every surface × mode with clean text, including
    near-misses `<｜User`, `</think`, `｜DSML`, `<|x|>` and `<thinking>`.
  - `tokenize=False` with dirty text is refused; truncation follows HF; a diverged encoder is
    refused; the result holds under 8 threads and after pickle; output-side vocab/decode is
    unchanged; anchor drift fails closed; the encoder AST check works; `start.sh` wiring is in place.
- **Mutants:** 8 of 9 killed. The survivor routes clean prompts through the segment path too. It
  is an equivalent mutant: segment encoding was id-identical to legacy on every clean case.
- **One-off check with the image's real `hf.py` wrappers** (`CachedTokenizer`, `TokenizerPool`
  deep copies, pickle) on agent-shaped transcripts built from vLLM sources:
  - 180 clean renders were identical across legacy, guarded and pooled.
  - All 46 dirty files came out clean; legacy had injected 284 control tokens across them.
- **Latency on ~1.5 MB prompts** (best of 5):

  | prompt | tokens | unpatched | patched |
  |---|---:|---:|---:|
  | clean | 396K | 747 ms | 755 ms |
  | dirty | 380K | 728 ms | 104 ms |

  Dirty prompts are faster because segments are batch-encoded in parallel.
- **Per-process cost:**
  - tokenizer load: about +0.3 s, with ~4 MB of tokenizer JSON held
  - first dirty request: builds the base-vocab tokenizer, ~0.2–0.35 s and ~40–50 MiB RSS
    (isolated measurement)

## Deploy (operator only)

The gb10 kit's `start.sh` (`/mnt/storage1/dsv41/exl3-kit`) is byte-identical to this repo's
`949c96a` copy (md5 `711befc8…`), so the new `start.sh` drops in. The kit's `.env` stays
untouched.

1. Before the window, save a fresh clean-id baseline (or use the 2026-10-20 one in this repo):

   ```bash
   python3 scripts/control_token_probe.py --save /tmp/ctg-before.json
   ```

2. Copy `overlay/patch_control_token_injection.py` to the kit's `overlay/`. Back up the kit's
   `start.sh` (`cp -p start.sh start.sh.bak-pre-ctg-<stamp>`), then copy this repo's `start.sh` over it.
3. Stop `vllm-watchdog` first (see the maintenance-window watchdog footgun). Then on gb10 in the kit:

   ```bash
   SKIP_BUILD=1 SKIP_PULL=1 ./start.sh restart
   ```

   Re-arm the watchdog after `/health` is up.
4. The head log must show `DeepSeek V4.1 control-token injection guard patch: applied` (no `WARN`).
5. Verify:

   ```bash
   python3 scripts/control_token_probe.py --compare /tmp/ctg-before.json
   ```

   It must end with `GUARDED`:
   - every surface PASS
   - `tokenize=false` with control text → HTTP 400
   - clean prompts id-identical

   Then run the usual smoke (`tests/test_smoke.sh`, 17*19).

**Rollback:** restore the `start.sh` backup and restart. The patch only edits files inside the
container, so nothing persists.

## Files

- `overlay/patch_control_token_injection.py`: the patch and `check_encoder_source`
- `tests/test_control_token_injection.py`, `tests/fixtures/dsv41_tokenizer/`: unpatched image sources
- `scripts/control_token_probe.py`: live `/tokenize` probe (no generation)
- `docs/control-token-injection-baseline-20261020.json`: clean-prompt ids from the live teacher before the patch
- `start.sh`: `CONTROL_TOKEN_PATCH_HOST`, preflight, scp, mounts, and patch loop on both ranks
