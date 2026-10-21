# DeepInfer's AIME 2026 / GPQA Diamond protocol over the teacher endpoint (2026-10-20)

`scripts/deepinfer_bench.py` runs DeepSeek's own benchmark protocol from DeepInfer against the live
teacher's OpenAI-compatible API. That gives the effort A/Bs and quantization checks
an official, item-clustered quality signal instead of our home-made sets. The
[09-24 A/B](effort-ab-75-vs-100-20260924.md) used such a set: 84 rows, McNemar p = 0.5,
inconclusive.

## 1. What is reused and what is replaced

**Reused, verbatim.** DeepInfer is `deepseek-ai/DeepInfer` @ `774ce24`, Apache-2.0. The pieces
are vendored under [scripts/vendor/deepinfer_eval](../scripts/vendor/deepinfer_eval/NOTICE.md),
with a sha256 manifest and a drift test against the stash checkout:
- `aime/bench.py`: the prompt (`{question}\nPlease reason step by step, and put your final answer within \boxed{}.`),
  the last-`\boxed{}` extraction with the tail-number fallback, and the rule judge.
- `gpqa/bench.py`: the prompt (`... 'ANSWER: $LETTER' ...`), `ANSWER:` → `\boxed{}` → tail-letter
  extraction, and the public-mirror row normalisation.
- `common/passk.py`: the unbiased Pass@k estimator and `summarize`, so Pass@1 = mean over samples.

**Replaced.**
- `run_eval.py` (the offline engine) becomes one streamed `/v1/chat/completions` per rollout,
  with `chat_template_kwargs={"enable_thinking": true, "reasoning_effort": <low|high|max|1-100>}`.
- `score.py` keeps the rule judge only, so there is no judge service and no API keys. DeepInfer
  reports that the rule judge agreed with its LLM judge on all 960 AIME 2026 rollouts.
- HLE is not wired up: DeepInfer's `score.py` refuses rule judging for it.

## 2. Protocol fidelity

| Aspect | DeepInfer | Here | Status |
| :--- | :--- | :--- | :--- |
| Prompt text | `encoding.encode_messages(msgs, thinking_mode="thinking", reasoning_effort=E)` | server-rendered from `chat_template_kwargs` | **identical**: preflight compares `/tokenize`→`/detokenize` with the checkpoint `encoding.py` byte for byte, every item and arm; any mismatch refuses the run |
| Effort | `REASONING_EFFORT` override, default 100; low/high/max = 50/75/100 | `--efforts`, default `$REASONING_EFFORT` or 100; same mapping | names are sent as given; the arm is the rendered budget, so `high` and `75` are one arm |
| Sampling | temperature 1.0, top_p 1.0 | same | `effort_ab.py` used top_p 0.95; results are not pooled with it |
| Length | `max_len` 147,456 = prompt + output | same (`--max-len`) | live `max_model_len` 393,216 |
| Model output | decoded with `skip_special_tokens=False` | `reasoning + "</think>" + content`, sent with `skip_special_tokens=False` | `</think>` only when an answer started or the rollout stopped; content is not stripped by vLLM's parser (no tools) |
| Decode blacklist | tokens 0, 2, 129262–129264 blocked | **not enforced**: vLLM rejects `logit_bias` with speculative decoding (HTTP 400) | a rollout that emits one is flagged (`blacklisted_tokens`, `flag` column) |
| Seeds | `index * n_sample + sample` | `crc32(task_id#sample) + --seed`, shared by both arms | stable under top-ups; vLLM + DSpark is not batch-invariant, so not reproducible |
| Engine | DeepInfer, MTP 5, EP 8 | vLLM, EXL3 2.9 bpw, DSpark k=3, TP 2 | this difference is what a quantization check measures |

Two DeepInfer extraction quirks are kept, since fidelity beats tidiness. Both are pinned in the
tests:
- The AIME tail fallback refuses a number followed by `.`, so `The answer is 277.` extracts
  nothing.
- The GPQA `ANSWER:` regex is line-anchored. An answer glued directly to `</think>` loses to a
  draft line in the thinking.

### Reference numbers

DeepInfer's [eval/README.md](https://github.com/deepseek-ai/DeepInfer/blob/774ce24/eval/README.md)
§7 does publish "reference figures for the V4.1 model under the default configuration": **AIME
2026 99.9%, GPQA Diamond 93.4%, HLE-Full 42.6%**. That corrects the brief's "ships no reference
scores". Their limits:
- **Aggregate only.** There are no per-item outputs, so they cannot pair with our rollouts.
- **Indicative.** The README itself says "indicative rather than exact".
- **Their conditions.** Every task, 32 / 8 samples, effort 100, 144K length, the LLM judge, and
  DeepInfer's engine on unquantized weights. The run scripts default to
  `DeepSeek-V4.1-Flash-1017`, the weights live since 10-18.

**What the reference is good for.** On AIME, 99.9% means about 1 failure in 960, so every failure
we see carries information. If the true rate matched the reference, then in 30 rollouts at
effort 100 with no truncation:
- P(≥ 1 failure) = 3%;
- P(≥ 2 failures) = 0.04%.

That holds even against a pessimistic 0.5% failure rate (the Poisson upper bound of 1/960): P(≥ 2
in 30) ≈ 1%. So a 30×1 AIME pass at 100 is already a meaningful quantization check, provided the
conditions match. GPQA is weaker on both counts: at 93.4%, a 198×1 pass carries ±2–3 points of
sampling error, and DeepInfer reports rule-vs-LLM agreement for AIME only.

I did not check the Hugging Face model card's numbers, so none are cited.

## 3. Sharing the teacher

The teacher runs `MAX_NUM_SEQS=2` next to live agent sessions, at ~40 tok/s at c=1. The script is
built to stay out of their way.

- **Small by default.** `--limit 3`, `--n-sample 1`, `--concurrency 1`. `--dry-run` prints the
  plan and the ETA and sends nothing.
- **Hard budgets per invocation.**
  - `--budget-tokens` (default 120,000) caps completion tokens. It is checked after every streamed
    chunk.
  - `--budget-seconds` (default 3,600) caps wall clock, idle waits included.
  - A rollout that would cross either budget is aborted. It is recorded as `budget`, never scored,
    and redone later.
- **Idle gate before every start.** The engine must be free of foreign requests for
  `--idle-quiet-s` (default 120). "Free" means running + waiting ≤ our own requests, and the
  prompt-token counter grew by no more than our prompts. The counter catches requests that come
  and go between 2 s polls. `request_success_total` is useless here, because aborted requests
  never reach it on this build.
- **Proxy check.** The repl proxy (`requests.sqlite`, read-only) must not have served this
  upstream for `--proxy-quiet-s` (default 600). This covers a live session between turns, which
  the engine gauges cannot see. The proxy's own prefix keepalives (`call_origin='keepalive'`,
  1 token every ~6 min per idle session) are ignored. The first smoke attempt showed that
  counting them keeps the gate shut forever.
- **Yield.** With `--yield` (default), foreign use lasting `--yield-after-s` (default 10 s) aborts
  our in-flight rollout. It is recorded as `yielded` and redone first, with the same seed. A
  ~1 s keepalive only shares the engine briefly. `--no-yield` shares a slot instead.
- **Resumable.** `raw.jsonl` is append-only, one fsynced row per attempt.
  - Rerunning the same command skips finished `(task, effort, sample)` keys.
  - Raising `--limit`, `--n-sample` or adding an arm tops the run up; seeds stay stable.
  - Changing the data file, model, temperature, top_p, max_len or seed is refused.
- **Bias note.** Yields are redone rather than dropped, but long rollouts are more exposed to
  interruption. Under heavy interruption a redo can therefore favour shorter samples. `summary.json`
  reports `yielded_attempts` per arm. If that number is high, rerun in a quiet window or use
  `--no-yield`.

## 4. Usage

The script needs DeepInfer-format JSONL (DeepInfer `eval/README.md` §3). **Datasets are not
downloaded yet: get the operator's OK first.**
- AIME 2026: `MathArena/aime_2026`, public.
- GPQA Diamond: gated upstream (`Idavidrein/gpqa`); public mirror `hendrydong/gpqa_diamond_mc`.
  DeepInfer's loader accepts the mirror's `problem`/`solution` rows as they are.

A conversion sketch follows. The column names are unverified until the download: check
`df.columns` first.

```bash
hf download MathArena/aime_2026 --repo-type dataset --local-dir /mnt/data1/Projects/_repl_measurement_stash/deepinfer-bench-data/aime_2026
```

```bash
python3 -c "import glob, json, pandas as pd; d = '/mnt/data1/Projects/_repl_measurement_stash/deepinfer-bench-data'; df = pd.concat(pd.read_parquet(p) for p in sorted(glob.glob(d + '/aime_2026/**/*.parquet', recursive=True))); print(df.columns.tolist()); open(d + '/aime_2026.jsonl', 'w').writelines(json.dumps({'task_id': f'aime_2026_{i}', 'question': r['problem'], 'answer': str(r['answer'])}) + '\n' for i, r in enumerate(df.to_dict('records')))"
```

Plan, run, resume, score. Run from rai against the vLLM head (`--base` defaults to
`http://192.168.1.97:8000`), not the proxy:

```bash
python3 scripts/deepinfer_bench.py run --bench aime --data-file D/aime_2026.jsonl --out-dir RUN --efforts high,max --limit 30 --dry-run
```

```bash
python3 scripts/deepinfer_bench.py run --bench aime --data-file D/aime_2026.jsonl --out-dir RUN --efforts high,max --limit 30 --budget-tokens 1500000 --budget-seconds 43200
```

- Rerun the same command to resume after a budget stop or Ctrl-C.
- Add `--n-sample 4` later to top up the same run.

```bash
python3 scripts/deepinfer_bench.py score --out-dir RUN
```

```bash
python3 scripts/effort_ab_paired.py RUN/paired.jsonl --efforts 75,100
```

`run` also scores when it finishes. Outputs in `RUN/`:
- `meta.json`: protocol, plan, DeepInfer commit, deviations, the preflight with the rendered
  prompt and per-item prompt tokens, and every invocation's budgets and gate.
- `raw.jsonl`: one row per attempt: `status` (done / yielded / budget / error / interrupted),
  `finish_reason`, tokens, `elapsed_s`, `ttft_s`, `seed`, and the full `reasoning` and `content`.
  Misses can be classified afterwards, which the 09-24 harness could not do.
- `scored.jsonl`: the final `done` row per key, plus DeepInfer's `pred`, `extract_source`, `gold`
  and `correct`.
- `summary.json`, per arm:
  - DeepInfer's `Pass@k`, `task_accuracy` and coverage, and `pass1_ci95_items` (item bootstrap);
  - `finish_reasons` and `truncated_rollouts`, extraction sources and the blacklist flag;
  - token stats, decode tok/s, and yield/budget/error attempt counts.

  `comparisons` holds each arm pair.
- `paired.jsonl`: `effort_ab.py`-format rows (`name` = task, `rep` = sample), so
  `effort_ab_paired.py` works unchanged.

**Statistics.** The comparison is item-clustered. Samples of one item share its difficulty, so
the unit is the item:
- `mean_diff`: the mean per-item Pass@1 difference;
- `ci95_items`: a bootstrap over items;
- `sign_flip_p`: an exact paired permutation test over items (Monte Carlo above 16 non-zero items).

McNemar on sample-index pairs is kept for continuity with the 09-24 doc. It treats rollouts as
independent, so it overstates the evidence when an item contributes several pairs.

**Truncation and effort.** A length-cut rollout scores as DeepInfer would score it, usually
wrong. Truncation hits the higher-effort arm harder, so for effort A/Bs keep `--max-len` at the
protocol's 144K. Check `truncated_rollouts` (the `cut` column) before reading a difference.

## 5. Expected wall-clock

Generation time is completion tokens ÷ decode rate. Prompts are 60–300 tokens, so prefill is
negligible.
- **c=1.** 40 tok/s. The k=3 bench measured 41.5. The 1017 reasoning turns run 42–46 tok/s. The
  smoke run measured 43–52 tok/s, but on rollouts of 6K tokens or less (§6).
- **c=2.** Aggregate ~60 tok/s, so ~0.7× wall-clock. But it holds both slots, so every live
  request waits for our yield. Use it only in a maintenance window.

The mean rollout length on AIME 2026 at each effort is **not measured yet**, so the table brackets
it. For calibration, our hard set (easier than AIME) averaged 4.9K tokens at 75 and 6.4K at 100,
with a p90 of 18.9K at 100. After the first real pass, `--dry-run` re-estimates from the run's own
rows.

| Configuration | Rollouts | mean 10K tok | mean 20K tok | mean 30K tok |
| :--- | ---: | ---: | ---: | ---: |
| defaults: 3 AIME items × 1 sample × 1 arm | 3 | 13 min | 25 min | 38 min (the 1 h budget may stop it) |
| AIME 30 × 1, one arm (quant check at 100) | 30 | 2.1 h | 4.2 h | 6.3 h |
| AIME 30 × 1, two arms (effort A/B pass 1) | 60 | 4.2 h | 8.3 h | 12.5 h |
| AIME 30 × 4, two arms | 240 | 17 h | 33 h | 50 h |
| AIME 30 × 32, one arm (full protocol) | 960 | 2.8 d | 5.6 d | 8.3 d |

| Configuration | Rollouts | mean 5K tok | mean 10K tok |
| :--- | ---: | ---: | ---: |
| GPQA 198 × 1, one arm | 198 | 6.9 h | 14 h |
| GPQA 198 × 1, two arms | 396 | 14 h | 28 h |
| GPQA 198 × 8, one arm (full protocol) | 1,584 | 2.3 d | 4.6 d |

These are generation hours only. Idle-gate waits and yields add to them, so an unattended night
covers roughly one 8–10 h row. Set `--budget-tokens` to rollouts × the assumed mean, plus 10%;
the budget stop is clean.

**Suggested order:**
1. AIME 30 × 1 at `max`, the quantization check against the 99.9% reference.
2. The same run directory topped up with `--efforts high,max`, which is effort A/B pass 1 and
   reuses the max rollouts.
3. Raise `--n-sample` only if `comparisons.75_vs_100` is still ambiguous.

## 6. Smoke run (2026-10-20)

Raw data: `_repl_measurement_stash/deepinfer-bench-smoke-20261020/` (fixtures, logs, run dirs).

**Setup.** No dataset was downloaded; the fixtures are DeepInfer-format JSONL built here:
- 2 AIME-format items from `scripts/effort_items_hard.py`: m4 (answer 387) and m12 (answer 818,
  the item that missed at 75 on 09-24). Their answers are computed in-script.
- 1 hand-written GPQA-format item (answer B).

The run used the default gate and yield settings. Budgets were 60K tokens / 2 h for AIME and
20K / 1 h for GPQA.

| item | arm | tokens | wall | decode | pred / gold | extract |
| :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| hard_m4 | high (75) | 1,259 | 24 s | 52 tok/s | 387 / 387 | boxed |
| hard_m4 | max (100) | 2,777 | 59 s | 47 tok/s | 387 / 387 | boxed |
| hard_m12 | high (75) | 4,315 | 94 s | 46 tok/s | 818 / 818 | boxed |
| hard_m12 | max (100) | 5,886 | 132 s | 45 tok/s | 818 / 818 | boxed |
| smoke_gpqa_0 | max (100) | 174 | 4 s | 43 tok/s | B / B | answer_line |

**Results.**
- **Rollouts.** 5/5 correct, no length cuts, no blacklisted tokens, no yields, no errors.
- **Prompts.** Preflight found all 5 byte-identical to the 1017 checkpoint's `encoding.py`.
- **Paired export.** `effort_ab_paired.py` read `paired.jsonl` unchanged.
- **Resume.** Rerunning the command sent nothing; `--top-p 0.95` on the same directory was
  refused.
- **Decode.** 43–52 tok/s on these short (≤ 6K) rollouts. Longer AIME rollouts run at growing
  context, so §5 keeps 40 tok/s.

**What the gate did on live traffic.**
- **First attempt.** It never opened. The only proxy rows in the window were the proxy's own
  keepalives (session `8b59dd3c`, 5,163 prompt tokens, 1 output token, every 6 min). Counting
  them kept the 600 s proxy-quiet window from ever elapsing; they would also have aborted every
  long rollout. The fix (`call_origin='keepalive'` ignored; yields need `--yield-after-s` of
  sustained use) was added red-test first, and the attempt was stopped before sending anything.
- **Second attempt.**
  - It started 120 s after preflight.
  - A keepalive landed mid-rollout at 22:13:16 (running = 2 for ~2 s). It was shared, not
    yielded, and the next start waited out a fresh 120 s window.
  - AIME took 9.1 min for 4 rollouts and 14,237 tokens. About 3.7 min of that was gate
    waiting: 2.0 min before the first start and 1.7 min after the keepalive.

## 7. Tests

`python3 tests/test_deepinfer_bench.py` (or pytest) runs 25 checks, all CPU-only:
- **Vendored code:** the files are verbatim (sha256 + stash drift).
- **Prompts:** AIME, GPQA and mirror prompts against DeepInfer's own README examples; the
  rendered-prompt check against the checkpoint `encoding.py`, including the stale 0909
  `high = 50` mapping.
- **Extraction:** the docstring cases (nested `\boxed`, `{,}`, tail fallback, truncation) and the
  two pinned quirks.
- **Scoring math:** the Pass@k estimator and `default_k_list`; McNemar (2 vs 0 → p = 0.5, as in
  the 09-24 doc) and the sign-flip test.
- **Compatibility:** `effort_ab_paired.py` run on the exported rows.
- **Run loop, against an in-process fake vLLM:**
  - resume;
  - yield + redo;
  - a short blip that must not yield;
  - the per-chunk token budget;
  - dry run;
  - keepalive filtering.

The suite killed 22 of 22 hand-made mutants.
