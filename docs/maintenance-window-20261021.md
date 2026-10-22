# Maintenance window 2026-10-21: control-token guard, SWA final window, YOCO prefill truncation

Operator-approved window on the gb10 kit (`/mnt/storage1/dsv41/exl3-kit`), about 17:30–19:40 CEST.
Raw data, the driver script and every log: `_repl_measurement_stash/maint-20261021/`
(`A0.log`, `A1.log`, `A3.log`, `A3-needle.log`, `logp/`, `logp-compare.txt`, `spec-*.json`,
`pra-*.json`, `ctg-A0.json`). Head `MemAvailable` was sampled every 1 s on gb10 throughout
(archived as `head-memavailable.log` in the same directory; `<epoch> <kB>` per line).

## Outcome: what is live now

| Feature | Repo | Live | Verdict |
|---|---|---|---|
| Final sliding-window retention (`DSV41_SWA_FINAL_WINDOW=keep`) | `422973a`, log fix `9063305` | **on** | pass |
| YOCO prefill truncation (`DSV41_YOCO_PREFILL=compact`, KEEP 140) | `292d4e2` | **on** | pass, memory margin accepted by the operator |
| Control-token injection guard | `eafef45`, fix `98abfc7` | **on** since 2026-10-21 23:44 CEST (`98abfc7`) | `eafef45` failed in the window (HTTP 400 on the served chat path) and was reverted; `98abfc7` deployed after the window, probe `GUARDED` ([record](control-token-injection-20261020.md#deployed-2026-10-21)) |

Kit state after the window:
- `start.sh` = this repo's `422973a` version: YOCO and SWA hunks, no control-token hunks. The
  kit copy was byte-identical to `949c96a` (md5 `711befc8…`) before the window, so the YOCO
  and SWA docs' "the kit's start.sh has diverged, port the hunks" was out of date. Backup:
  `start.sh.bak-pre-maint-20261021`.
- `.env` gained two lines, `DSV41_SWA_FINAL_WINDOW=keep` and `DSV41_YOCO_PREFILL=compact`.
  Backup: `.env.bak-pre-maint-20261021`.
- `overlay/` gained `patch_swa_final_window.py` (the `9063305` version), `patch_yoco_prefill.py`
  and `dsv41_yoco.py`. `patch_control_token_injection.py` is present but not referenced.
- **Later on 2026-10-21 (after the window):** the guard was deployed with one more restart.
  `overlay/patch_control_token_injection.py` is now the `98abfc7` version, and `start.sh` is this
  repo's (unchanged since `eafef45`: `422973a` plus the guard's 7 lines). Backup:
  `start.sh.bak-pre-ctg-20261021-232939`. `.env` was not changed. Record:
  [control-token-injection-20261020.md](control-token-injection-20261020.md#deployed-2026-10-21).

Rollback: delete either `.env` line (or set it to `off`) and run
`SKIP_BUILD=1 SKIP_PULL=1 ./start.sh restart`. For the full pre-window state, restore both backups.

Live values that differ from the README's shipped defaults: `MAX_MODEL_LEN=393216` and
`KV_CACHE_MEMORY_BYTES=4294967296` (4 GiB), with `LANGUAGE_MODEL_ONLY=1`.

## Arms

One restart each (`SKIP_BUILD=1 SKIP_PULL=1 ./start.sh restart`, about 10 min to healthy):

- **A0:** the pre-window server, up 3 days, all features off.
- **A1:** control-token guard plus SWA `keep`.
- **A3:** SWA `keep` plus YOCO `compact`, with the guard removed after A1.

A2 (YOCO alone) was skipped: `keep` only changes what is cached at a free, and the YOCO
logprob baselines were taken on A1 so that A1 and A3 differ only by YOCO.

repl's `sovereign-router-keepalive.timer` was paused for the window. No client traffic
reached the teacher during it.

## Results

### Cold prefill (`chunk_validate.py kvprog`, fresh salt)

| Prompt | A0 TTFT | A0 tok/s | A1 tok/s | A3 TTFT | A3 tok/s | A3 vs A0 |
|---|---:|---:|---:|---:|---:|---:|
| 45K | 43.8 s | 1,029 | | 26.3 s | 1,710 | 1.66× |
| 128K | 130.3 s | 984 | | 75.8 s | 1,693 | 1.72× |
| 380K | 433.5 s | 876 | 906 (419.4 s) | 232.9 s | 1,632 | 1.86× |

The YOCO doc's model (§9) predicted a 45K TTFT of 30–35 s and 1.4–2k tok/s at 380K. Both landed
at the fast end.

### Retention (`chunk_validate.py retention`, prompt sent twice)

| Prompt | A0 cached | A1 cached | A3 cached |
|---|---:|---:|---:|
| 8,195 | 4,096 | **8,064** | 7,936 |
| 34,302 | 34,176 | 34,176 | 34,048 |

- A1's 8,064 is the final window at work: stock vLLM only had the 4096 checkpoint.
- A3's values are exactly the YOCO doc's prediction, from the `N − 140 − 64` boundary.

### Agent prefix reuse (`prefix_retention_agent.py`, append/fork/interleave, 8 turns)

| Arm | append | fork | inter-A | inter-B | worst miss per turn |
|---|---:|---:|---:|---:|---:|
| A0 | 99.6 % | 99.2 % | 98.1 % | 99.5 % | 1,924 |
| A1 | 99.6 % | 99.2 % | 99.5 % | 99.5 % | 154 |
| A3 | 99.4 % | 98.8 % | 99.1 % | 99.2 % | 302 |

A3's worst miss stays inside the YOCO cap (KEEP + 64 + 191 = 395). This script's LCP is taken
against the previous *prompt*, so it cannot show own-output reuse; that is the organic-traffic
check under Follow-ups.

### Decode (`spec_bench.py`, 8 prompts × 1024 tokens, official sampling)

| Arm | c=1 aggregate | c=1 per-request median | c=1 accept len | c=2 aggregate | c=2 accept len |
|---|---:|---:|---:|---:|---:|
| A0 | 40.9 | 43.3 | 2.44 | 59.6 | 2.45 |
| A1 | 40.4 | 43.2 | 2.40 | 59.6 | 2.43 |
| A3 | 39.0 | 41.0 | 2.33 | 59.1 | 2.43 |

All spec_bench prompts are under 140 tokens, so YOCO takes its exact path on them (N ≤ KEEP).
A3's lower c=1 figure is therefore sampling noise at temperature 1.0; c=2 matches. DSpark
acceptance after a *long* truncated prefill is not measured by this script. The decode half of
the logprob A/B below covers it at 8K/45K/128K.

### YOCO quality

**Teacher-forced logprob A/B** (`yoco_logprob_ab.py`; off-a/off-b on A1, yoco on A3; haystack
= the needle corpus). Verdict: **PASS** within the script's bounds.

| L | arm | greedy steps before divergence | decode mean abs Δlogprob | probe mean abs Δlogprob | max | top-1 agree | top-K Jaccard |
|---:|---|---|---:|---:|---:|---:|---:|
| 8K | floor | 53/64 | 0.028 | 0.017 | 0.115 | 1.00 | 0.90 |
| 8K | yoco | 26/64 | 0.034 | 0.017 | 0.078 | 1.00 | 0.84 |
| 45K | floor | 64/64 | 0.026 | 0.030 | 0.260 | 1.00 | 0.86 |
| 45K | yoco | 64/64 | 0.023 | 0.045 | 0.324 | 1.00 | 0.87 |
| 128K | floor | 7/64 | 0.066 | 0.052 | 0.352 | 0.92 | 0.93 |
| 128K | yoco | 64/64 | 0.050 | 0.087 | 0.601 | 1.00 | 0.92 |

- Top-1 agrees with the off arm at every probe.
- The probe-row perturbation is 1.0× / 1.5× / 1.7× the off-vs-off floor at 8K / 45K / 128K.
  So it is small and grows with context, as an approximation should.
- Greedy first-divergence is reported, not gated: near-ties flip under the floor too.

**Needle battery** (`needle_layers.py`, 8K–300K), against the 10-18 run on the same weights
(`_repl_measurement_stash/needle-layers-1017-20261018`):

| | 10-18 baseline | A3 |
|---|---:|---:|
| single needles | 25/25 | 25/25 |
| two-hop, thinking on | 15/15 | 15/15 |
| two-hop, thinking off (known weak, decoy) | 2/15 | 1/15 |

**Functional checks on A3:**
- smoke: 17*19 → 323;
- `prompt_logprobs: 1` → HTTP 400 with the YOCO message;
- a 13-token prompt answers;
- head and worker logs show all 7 YOCO files `applied`, `keep=140 decoder layers 20..39`, and
  `created 16 shadow metadata builders (+58.8 MiB)`;
- SWA reports `keep 332 tokens below the window (+6 admission blocks per request)`.

### Head memory

`MemAvailable` on the head, gb10:

| Arm | idle before the 380K run | floor during the 380K prefill | dip |
|---|---:|---:|---:|
| A0 (up 3 days) | 5.2 GiB | 5.15 GiB | ~0.05 |
| A1 (fresh boot) | ~3.95 GiB | 3.84 GiB | ~0.15 |
| A3 (fresh boot) | 3.74 GiB | 3.14 GiB | ~0.6 |

**The A0→A1 offset is the restart, not the SWA patch.** Every fresh boot since 10-18, all
without these patches, reports 4.4–4.5 GiB on the head in the start.sh banner. A0's 5.2 GiB is a
server that had run for 3 days.

**YOCO costs about 0.5 GiB more under a long prefill, and it is allocator cache.** The dip
persists after the prefill because start.sh only calls `empty_cache` once `MemAvailable` drops
below `DSV41_PREFILL_EMPTY_CACHE_MEMAVAIL_GIB` (2.5 GiB). Through the needle battery's
back-to-back 256K/300K cold prefills, that release fired as designed:
- first at 2.44 GiB, which it recovered to 3.36;
- then repeatedly, the lowest sample being **2.14 GiB**.

That misses the YOCO doc's pass rule (floor ≥ baseline − 0.2 GiB). The operator kept YOCO on
as tested (2026-10-21): 2.14 GiB is about the 2.1 GiB the README already records the head
surviving at the end of a 601K prompt, and the release mechanism recovers it each time. If it
ever needs more margin, the lever is that threshold (e.g. 3.0 GiB), at some prefill cost
(untested).

## Incidents

1. **The control-token guard returned HTTP 400 on real chat requests (A1).**
   - Clean prompts were id-identical to the A0 baseline (18/18 lists).
   - But every `/v1/chat/completions` and `/tokenize` request whose messages held control-token
     text got 400: "message text contains control-token strings that tokenize=False would turn
     into control tokens".
   - Calling the patched tokenizer directly in the container encodes correctly. So the served
     path hands `apply_chat_template` a falsy `tokenize`, and the guard's refusal branch fires.
   - For agent traffic this is worse than the injection: one tool result with `</think>` text
     in the history fails every later turn.
   - No client traffic reached A1. The guard was removed at the A3 restart; the fix is a
     separate task.
2. **The SWA patch logged a TypeError traceback on every free (A3).**
   - `logger.info_once(..., sorted(hits))` passed a list, and vLLM's `*_once` lru-caches its
     args. It raised after the caching had already run, so retention was unaffected (the A1/A3
     numbers above), but each finished request logged a full traceback.
   - Fixed in `9063305` (test stub now enforces hashable args; red first) and deployed at the
     final restart.
3. **`tests/test_smoke.sh` defaults its model id to `DeepSeek-v4.1-Flash-EXL3`.** The kit serves
   `deepseek-v4.1-flash`, so pass it as the 2nd argument, or the curl gets a 404.

## Follow-ups

- **Organic own-output reuse — measured 2026-10-22: NOT the predicted saving (open).**
  `scripts/swa_final_window_turns.py`, read-only on repl's record:
  - From the final restart (10-21 21:44Z) to the 10-22 restart: 95 continuations, 55 of which
    hold the whole previous output. 19,314 of 20,355 own-output tokens were re-prefilled (95 %).
    The length-only prediction under keep + compact is 6,188.
  - In the pre-window week (10-14 to 10-21 10:00Z): 4,009,469 of 4,881,676 were re-prefilled
    (82 %), with a prediction under keep + compact of 1,117,232.
  - The hit still lands at the prompt tail in 32 of 55 (median −266 from the previous end).
  - The head log shows the patch applied and caching final windows, so the retention runs. What
    is not established is why organic turns miss it. Candidates: eviction across idle gaps,
    YOCO's cap on short (~370-token) outputs, and the small sample.
  - Re-run on a busier window, with `--per-turn`, before drawing a verdict.
  - Raw data: `_repl_measurement_stash/ots-deploy-20261022/swa-turns-*.txt`.
- **Long-context DSpark acceptance.** Compare the repl record's decode tok/s for prompts > 32K
  before and after the window.
  - Same tool, all rows: the median is 42.0 tok/s after the window (n = 92) against 51.5
    before (n = 7,449).
  - That is not like for like: the post-window sample is small and its prompt-length mix is
    unchecked. Split it by prompt length before reading anything into it.
- **Head memory.** Watch the `dsv41-mem` release lines and `MemAvailable` on long sessions.
- **Control-token guard:** done. The served-path fix (`98abfc7`) was tested and deployed on
  2026-10-21 at 23:44 CEST (see the record linked above).

### Pending in this repo (cross-repo sweep, 2026-10-22; updated the same day)

The repl index is `docs/PROGRESS.md` 2026-10-22 (repl `d0d0823`).

- **Done 2026-10-22:**
  - the output-token split is live (one restart, 15:28–15:38 CEST;
    [output-token-split-20261022.md](output-token-split-20261022.md) §Deploy record);
  - the #23 review committed (`6c0e02a`);
  - the 09-21 wedge diagram moved under `docs/exl3-moe-wedge-diagram-20260921/` and marked
    superseded (`f9392f5`);
  - `tests/test_smoke.sh` reads the served id from `/v1/models` when none is given;
  - pushed to `origin/main`.
- **Open:**
  - the organic own-output reuse shortfall (above);
  - the > 32K decode split (above);
  - head memory.
- **Open, documented gaps:**
  - streaming still matches DSML by text;
  - non-streaming `count_reasoning_tokens` shares the `rfind` anchor issue
    ([output-token-split-20261022.md](output-token-split-20261022.md) §Not fixed).
- **Optional:** the AIME 32-sample protocol, to separate 99.9 % from ~95 %
  ([deepinfer-bench-protocol.md](deepinfer-bench-protocol.md) §8).
