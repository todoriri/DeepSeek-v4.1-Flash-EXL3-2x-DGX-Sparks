# Effort mapping + prefix retention: deployment and live validation (2026-09-23)

Live 2×GB10 stack right after the effort-mapping patch (6e19496) and the upstream
#12/#21 merge: EXL3 2.9 bpw, DSpark k=3, `MAX_NUM_SEQS=2`, MNBT 1024, text-only,
4 GiB KV pool (1,253,996 tokens), `PREFIX_CACHE_RETENTION_INTERVAL=4096`.

## 0. What was deployed

| Change | Source | Effect on the live pair |
|---|---|---|
| Reasoning-effort mapping patch | `overlay/patch_reasoning_effort_mapping.py` (6e19496, mirrors vllm#58316) | The image's `deepseek_v41` encoder mapped low/high to 25/50; now low=50, high=75, xhigh=75, max=100. Default effort (`"high"`) goes 50 → 75 |
| Prefix-cache retention 4096 | upstream MiaAI #21 | `--prefix-cache-retention-interval 4096` on both ranks |
| Usage details | upstream MiaAI #12 | `--enable-prompt-tokens-details`: `usage.prompt_tokens_details.cached_tokens` per request |
| Responses API content types | upstream MiaAI #12 | `input_text` / `output_text` parts accepted |
| Engram IO threads | upstream MiaAI #12 | `DSV41_IO_THREADS` 32 → 96 (start.sh default; not overridden in `.env`) |
| Live `.env` | gb10 kit | `LONG_PREFILL_TOKEN_THRESHOLD=1792` commented out: it exceeded `MAX_NUM_BATCHED_TOKENS=1024`, so it never clamped; behaviour unchanged |

Procedure: kit backed up to `/mnt/storage1/dsv41/exl3-kit-predeploy-20260923-152044.tgz`
(+ `.env.bak-effort-*`), changed files copied from this repo, then
`SKIP_BUILD=1 SKIP_PULL=1 ./start.sh restart`. The skip flags are required because the
pulled image's recipe label is `unknown`, which never equals the repo hash, so a plain
restart would rebuild. Both patches are runtime-mounted and log
`... patch: applied` on head and worker. Healthy in ~460 s; KV pool unchanged
(1,253,996 tokens, 3.19× at 393,216); watchdog paused and resumed by start.sh; idle
MemAvailable 4.9 GiB head / 5.0 GiB worker.

Post-deploy checks: `/tokenize` renders `Reasoning Effort: 75` for `"high"` and for no
effort (50 low, 75 xhigh, 100 max, integers verbatim); a Responses API `input_text`
request answers `323`; upstream's retention boundary cases cache 8064 / 4096 / 4096 /
8192 / 32768 tokens at 8192 / 8193 / 8256 / 8257 / 34357 input tokens, matching upstream's
retention-4096 numbers (the 8193, 8256 and 34357 cases were zero hits at retention 0).

## 1. Reasoning effort 50 (old "high") vs 75 (official "high")

`scripts/effort_ab.py --reps 2 --concurrency 2`: 12 math prompts (answers computed
by the script, exact match) + 6 coding tasks (unit tests run network-less), thinking
on, `temperature=1.0 top_p=0.95`, arms interleaved per prompt. Integer efforts
bypass the name table, so both arms ran on the same patched server.

| effort | n | accuracy | median tok | mean tok | median s | tok/s | length-cut |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 36 | 100% | 452 | 989 | 11.1 | 36.9 | 0 |
| 75 | 36 | 100% | 584 | 1330 | 15.1 | 37.7 | 0 |

- **Cost:** 75 spent 1.34× the completion tokens overall (35,615 → 47,874) and
  +36% median latency; decode speed unchanged.
- **Quality:** no difference measurable: both arms were perfect, so this set is
  below the model's ceiling at either budget. A discriminating comparison needs
  harder items (multi-step agent tasks, competition math).
- **Variance dominates per item:** ratios ranged 0.26× (m1, one 50-arm run
  rambled to 9.6k tokens) to 8.95× (m9 Collatz, 75 spent 6.5k). Budget shifts the
  mean, not a hard cap. Two reps per arm is noisy for per-item claims.

## 2. Prefix reuse on agent-shaped traffic

`scripts/prefix_retention_agent.py`: ~15k-token system prompt + 8 tool schemas;
each turn = user asks → model calls `read_file` → ~1.9k-token tool result → answer
(temperature 0, real model outputs appended). `lcp` = longest common token prefix
with any earlier prompt via `/tokenize` (ideal reuse); `short` = lcp − cached.

| scenario | warm requests | Σ lcp | Σ cached | reuse | max short | warm zero-hits |
|---|---:|---:|---:|---:|---:|---:|
| append (6 turns) | 11 | 222,287 | 219,520 | 98.8% | 1,932 | 0 |
| fork (back to turn 2) | 1 | 18,791 | 18,688 | 99.5% | 103 | 0 |
| interleave A | 5 | 86,100 | 85,376 | 99.2% | 403 | 0 |
| interleave B | 5 | 85,839 | 83,584 | 97.4% | 1,931 | 0 |
| thinking on | 5 | 85,475 | 82,432 | 96.4% | 2,458 | 0 |

(The first request of each session is cold by construction; its ~30-token overlap
with other sessions is template boilerplate and is excluded.)

- Typical shortfall is 20–140 tokens (128-token alignment + DSpark lookahead).
- Occasionally (3 of 27 warm requests) the hit falls back one checkpoint and
  ~1.9–2.5k tokens are recomputed — bounded well below the 4096 interval, never
  a zero hit.
- Tool-call history, forks, A/B interleaving and thinking-on history (prior
  reasoning dropped) all reuse the prefix. At this scale (~20k-token sessions)
  there was no cross-session eviction; the 1.25M-token pool is far from pressure.
- Not covered: sessions near `MAX_MODEL_LEN`, eviction under real multi-session
  pressure, a retention-0 A/B on this workload (upstream's boundary table shows
  0-hit cliffs at 8193/8256/34357; with 4096 live we measured 4096/4096/32768).
