# Stability and performance research (2026-09-30)

## Scope

This covers DeepSeek V4.1 Flash EXL3 2.9 bpw on 2× GB10 (TP=2). Stability comes first, then
performance; vision is out of scope.

- **Upstream.** The kit repo has no new commits since 09-19. It also has no new image tags,
  quants, releases or HF discussions, and nothing new on #22.
- **What this pass covered:**
  - open PRs, issues and forks of the kit;
  - the sibling GLM kit;
  - vLLM main (v0.30.0 and the nightlies);
  - ExLlamaV3 v1.4.6 to v1.5.3 and its `dev` branch;
  - the NVIDIA forums and community recipes;
  - live data from gb10 and kgb10: watchdog `events.jsonl`, `/metrics`, the router log, and
    kernel logs.

## 1. The "wedge" is a client timeout plus a progress metric that cannot see prefill

**Main finding.** The recorded 09-20 and 09-21 stalls did not hang the engine. Neither did the
four on 09-24. Each one was a long prefill in progress.

- **Why the counters freeze.** vLLM counts a request's prompt tokens only when the request emits
  its first token (`v1/engine/output_processor.py:675` in the live image). The watchdog
  measures progress as `prompt_tokens_total` + `generation_tokens_total`, so both stay flat for
  the whole prefill.
- **Why that trips the watchdog.** At the measured 730–775 tok/s (needle test, 64K–300K), any
  cold prefill over ~135K tokens outlasts the watchdog's 180 s `STALL_SECONDS`.
- **Why the probe fails.** The probe times out because a running prefill takes the whole
  1024-token step budget.
- **The loop that stretches it.** The repl router's upstream `httpx` timeout is
  `HTTP_TIMEOUT_S`, default 300 s, and it is not overridden live. A streaming request sends no
  bytes before its first token, so any prefill that takes over 300 s is killed with
  `httpx.ReadTimeout` at `logger_proxy.py:1167`. The client then retries, and each retry
  prefills again from the last cached point.

### Every stall event since 09-19 has one running request, and its "recovery" is a prompt finishing

| Watchdog stall (UTC) | Running / waiting | Progress jump at recovery | Router log (local = UTC+2) |
|---|---|---|---|
| 09-20 10:29 ("terminal_device_spin") | 1 / 0 | ~2.0M (several cached 275–288K turns) | 12:20:30 → 12:25:31 → 12:30:33 → 12:35:36 → 12:40:40: one retry every ~300 s after `ReadTimeout`. The first completion was at 12:43 (275,233 prompt tokens, 79% cached), with no intervention |
| 09-20 11:27 | 1 / 0 | 229,429 | one ~229K prompt |
| 09-21 12:20 (the "484 s" wedge) | 1 / 0 | 261,367 | request 14:17:15 → `ReadTimeout` 14:22:15 → retry → `ReadTimeout` 14:27:17 → retry → done 14:30:27, **260,579** prompt tokens |
| 09-21 13:00 (the "484 s" wedge) | 1 / 0 | 793,230 | request 14:57:20 → `ReadTimeout` 15:02:21 → retry → `ReadTimeout` 15:07:22 → retry → done 15:08:44 (**263,224**). Two cached turns (263,935 and 264,660) also finished by 15:09:04, and the three add up to the jump |
| 09-24 15:47 / 15:53 / 15:59 / 16:07 | 1 / 0 | 513K / 257K / 2.4M / 902K | the needle test (300K single requests). Stacks were captured on both ranks. The worker was at `apply_exl3_fused_moe` exl3.py:1655, the same pin as the 09-21 12:20 wedge |

There has been no stall event since the needle run on 09-24, six days of serving.

### What this changes

- **The "~480 s recovery clock".** Two ~260K prompts, each killed twice at 300 s, took about the
  same wall time. No NCCL timer is needed to explain it. `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300`
  is harmless, but it has lost its reason to exist.
- **"Pins on different kernels".** A host blocked on a full launch queue, or on the first sync,
  during a GPU-bound prefill shows up wherever that happens to be. ExLlamaV3 review:
  - `build_grouped_fat_tables` / `apply_exl3_grouped_fat` contain **no** host sync;
  - the exllamav3 `exl3.py:214` pin is the plain `had_r_128` launch;
  - 96 % SM is the prefill doing real work.
- **The peer-desync, barrier and co-residency hypotheses.** Our captures contain no evidence for
  them. The 2026-09-19 repro had already recorded that the watchdog raises false positives on
  long prefills, but the 09-21 analysis did not carry that forward.
- **Upstream #22 "Mode 2".** Its numbers fit the same reading: two concurrent 260K cold
  prefills, 947.6 vs 473.5 tok/s, 550 s ≈ 2 × 286 s, and every request completed. It is
  probably not a hang. #22 Mode 1 (the shared-experts stream) was a real hang: HTTP 500 and
  EngineCore force-killed. It is fixed here by `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`.
- **Still open.**
  - On 09-21 the effective prefill rate across the retries was ~380 tok/s, about half the
    09-24 needle rate. Possible causes are work lost on abort, the arm-B image, or concurrent
    cron traffic. It is not established.
  - The GLM kit's PR #224 (sparse-MLA decode slicing) reports the same *symptom* with a
    soak-tested fix. Those events may be the same artefact. Only port it if a stall reproduces
    under a prefill-aware progress signal (§2 S1).

## Deployed 2026-09-30 (S1 and S3), validated live

- **Prefill chunk (S3).** The gb10 kit `.env` got `MAX_NUM_BATCHED_TOKENS` 1024 → 1536 and
  `LONG_PREFILL_TOKEN_THRESHOLD=1280`. The backup is `.env.bak-mnbt1536-20260930-114815`.
  - Restarted at 11:48 with `SKIP_BUILD=1 SKIP_PULL=1 ./start.sh restart` while the backend was
    idle. Healthy at 11:58 (the health check passed after 550 s).
  - Both ranks carry both flags. Cooperative MoE is enabled on both, and the EXL3 pretune ran
    130 launches in 2.8 s.
  - Smoke test `17*19` → `323`.
  - The repo `.env.example` stays upstream's default (threshold disabled by omission, pinned by
    `tests/test_numeric_config.py`).
- **Watchdog (S1).** `progress_of()` = prompt + generation tokens + `kv_cache_usage_perc`, in
  `logging_stack/vllm-watchdog`, with 5 new tests (24/24 pass). Deployed to gb10 and only the
  watchdog was restarted. The backup is `vllm_watchdog.py.bak-prekv-20260930`. Worker capture is
  still armed.

Measured with `scripts/chunk_validate.py` and `scripts/spec_bench.py` from rai, directly against
the head:

| Check | Before | After |
|---|---|---|
| Cold 48K prefill | — | 48,074 tok in 46.8 s = **1,027 tok/s** |
| Cold long prefill | 300K in 407 s = 737 tok/s (09-24 needle, 1024 chunk) | **379,930 tok in 417.9 s = 909 tok/s** (+23 %; each step is a 1,280-token chunk) |
| Short chat sent 30 s into that prefill | queued for the whole prefill (the watchdog probe always timed out) | **4.18 s**, correct |
| `kv_cache_usage_perc` during the 48K prefill | — | 37 distinct values in 45 one-second samples, monotonic, while `prompt_tokens_total` stayed flat |
| Watchdog events across the 418 s prefill | would have fired `stall_suspected` at 180 s | **none** |
| Head / worker `MemAvailable` low-water (380K run) | — | **2.51 / 3.77 GiB**. The head's low came 2.5 min in, during tokenisation and the early prefill, not at the end; it was 3.3 GiB at the end |
| Retention (cached on repeat) | 8193 → 4096 (09-23) | 8195 → **4096**, 34302 → **34176**, 19978 → **16384** |
| Decode, c=1 (agg / per-req median) | 41.5 / 43.9 tok/s | **41.6 / 44.5** |
| Decode, c=2 (agg / per-req median) | 60.2 / 32.8 tok/s | **60.7 / 33.3** |
| Mean acceptance length c=1 / c=2 | 2.46 / 2.47 | 2.44 / 2.49 |

`NV_ERR_NO_MEMORY` lines during the run: 5 (gb10) and 4 (kgb10). This is the known
allocator-handled noise; nothing failed.

**Still open.** S2, the router `HTTP_TIMEOUT_S=300`. A cold 393K prompt now takes ~430 s, so a
cold prefill over ~270K through the router still hits the timeout-and-retry loop.

## 2. Stability recommendations (in order)

**S1. Give the watchdog a progress signal that moves during prefill (no teacher restart).**

- `SchedulerStats` is sent every engine step, even when no request produced output
  (`v1/core/sched/scheduler.py:2264`, "We must return the stats even if there are no request
  outputs this step").
- So `vllm:kv_cache_usage_perc` grows chunk by chunk during a cold prefill.
- Proposed rule: treat any change in it as progress, and stop treating a probe timeout alone
  as confirmation.
- A real freeze still shows: KV usage flat, counters flat.
- File: `~/logging_stack_gb10/vllm-watchdog/vllm_watchdog.py`, the progress sum at ~L983, L1148
  and L1243.

**S2. Stop the retry loop at the router (repl, no teacher restart).**

- Raise the teacher backend's read timeout well above the worst-case TTFT. A cold 393K prefill
  at 750 tok/s is ~524 s, and roughly doubles when a second prefill shares the step, so use
  e.g. `httpx.Timeout(connect=10, read=1800)`.
- Alternatively, send SSE keep-alive comments downstream before the first token.
- The dsh undici `bodyTimeout` fix (`fix/http-proxy-body-timeout`) is still undeployed. It is
  the same class of bug on the client side.

**S3. `MAX_NUM_BATCHED_TOKENS` 1024 → 1536, plus `LONG_PREFILL_TOKEN_THRESHOLD` (restart
required).** This is upstream's shipped default.

- **Prefill speed.** On the upstream agent replay, 995 vs 760 tok/s (+31 %), and median TTFT
  45.3 → 34.6 s. That shortens every window above.
- **Memory headroom.**
  - #19 measured *more* headroom at 1536 than at 1024.
  - The thomwebb fork (09-29) validated our exact 4 GiB pool at 600K with a 2048 chunk: a cold
    595K prefill left 4.46 GiB free on the head and 3.57 GiB on the worker.
  - Do **not** go above 2048: #19 reports a 3072 chunk wedging at ~140K.
- **Fairness.** vLLM caps any single request's chunk at the threshold, e.g. 1280. A solo long
  prefill still runs larger chunks than today, and 256 tokens per step are left for the second
  slot. Upstream's 2048/1792 run answered a 17-token chat sent into a running 181K prefill in
  3.6 s. Our probe currently cannot get scheduled at all in that situation.
- **Optional.** `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128`, down from 256; the coolbho3k recipe uses
  it to bound the indexer transient at larger chunks.
- **Validate** head `MemAvailable` at the end of a ~390K cold prompt before calling it done.

**S4. Host (no change today, just don't regress).**

- Both nodes run kernel `6.17.0-1029-nvidia`, so they are not affected by the DGX OS 7.6 /
  kernel `7.0.0-1019` KHO regression (NVIDIA forum t/383023: NCCL `ibv_reg_mr` ENOMEM,
  `NV_ERR_NO_MEMORY`; the fix is `kho=off` / `cma=128M`).
- One user also had to lower GPU memory utilisation after that update. Hold the update, or read
  that thread first.
- Keep swap on: ~4.5 GiB of cold vLLM and desktop heap lives there and forms part of the
  headroom.
- `NV_ERR_NO_MEMORY` count since 09-24: gb10 6, kgb10 55. This is known allocator noise; watch
  the trend.

**S5. Small, optional.**

- **vLLM #59119.** It adds `do_not_specialize` to `_ring_slot_mapping_kernel` (in
  `models/deepseek_v4_1/compressor.py`, unpatched in the live image) to prevent mid-serve Triton
  recompiles that stall the peer rank. It never JIT-compiled mid-serve this boot: the 9 JIT
  compiles were all sampler and speculation kernels during boot warmup.
- **Effort=max loops.** Effort=max can loop in reasoning up to its 65,536-token cap (forum
  t/383242, ~25 min). With 2 slots, that holds half the capacity, and the teacher's `coding`
  profile sends `effort=max`. Consider a router-side `max_tokens` cap.

## 3. Performance recommendations

**P1. S3 is also the biggest measured performance lever** for this prefill-dominated agent
workload.

**P2. Both CX7 PCIe functions for NCCL.**

- `roceP2p1s0f0` / `enP2p1s0f0np0` is up at 200 Gb/s on both nodes (10.10.13.x) and unused.
- One function tops out at ~92–110 Gb/s. With both, others measure a 179.5 Gb/s NCCL
  all-reduce (NVIDIA dgx-spark-playbooks; forum t/376298, t/377290).
- Decode all-reduces are latency-bound, so gains are for prefill only. Estimate: ~3–7 % at a
  1536 chunk.
- Needs a `start.sh` change: the GID preflight indexes `/sys/class/infiniband/${HEAD_CX7_IB}`,
  which breaks on a comma list.

**P3. DSpark.**

- Keep k=3. k=4 at c=2 means 2×5 = 10 rows, outside the cooperative kernel's 1–8-row envelope.
- `num_speculative_tokens_per_batch_size` exists in the image. It is disabled only for DP>1, and
  we run DP=1.
  - An A/B of `[[1,1,3],[2,2,2]]` (k=2 at c=2) is possible.
  - It is untested with DSpark on this image, so an 8–10 min boot failure is the downside: run
    it in a window.
- Adaptive verification stays dead on SM12x, even on vLLM main:
  `supports_device_cpu_query_lens_mismatch` is SM100/SM90-only.
- Probabilistic drafting (`draft_sample_method`, default `greedy`) would suit our temp=1.0
  traffic. Wait for the fix to vLLM #59355 (out-of-bounds read in block verification).

**P4. Later, as image work.** ExLlamaV3 v1.5.x brings the `EXL3_MOE_MTILE` 32/64-row fused-MoE
tiles (30–47 % less fused-kernel time at 24–128 rows) and an fp16-accumulate hgemm with SM12x
tiles.

- **Cost:** an image rebuild, an adapter for the new 35-argument `exl3_moe`, and re-gating the
  cooperative kernel.
- **Benefit:** no stability fix, and the prefill gain is unmeasured on GB10.

## 4. Checked, not worth doing now

- **vLLM v0.30.0 / nightly.** It has native `DeepseekV41`, and #58316 (effort mapping) was
  merged 09-26. On SM12x, though:
  - it fails to start (no common block size, #56461/#59203; fixes open);
  - it gives NaN outputs with CUDA graphs (#57156/#58560);
  - MoE decode is ~15 % slower (#58624).

  Stay on the 0909 image. Drop `patch_reasoning_effort_mapping.py` only when moving.
- **Kit PR #6 `DSV41_EXL3_MOE_X`.** The cooperative MoE we run is still faster on decode
  (39.75 vs 35.39 tok/s C1). Whether the two stack is untested, and it does not affect the
  "Mode 2" events.
- **Other quants.**
  - jeet0733 C4h36 (same size, better perplexity): its serving branch is not public, and it
    relies on adaptive verification.
  - sfxnz 2.0 bpw: 50 / 81 tok/s prose, 21–23 GiB free per node, 1M context. This is the option
    if headroom ever becomes the binding constraint, but it has a quality cost. Run
    `scripts/effort_items_hard.py` before considering it.
- **Headless display-RAM reclaim** (~2 GiB per node, `nvidia_drm modeset=1 fbdev=0`). It is
  experimental, driver-specific, and failed on driver 595.84.
- **`iommu.passthrough=1`** (currently `0` on both nodes). One forum report claims −27 % TTFT at
  32K on TP2, and it conflicts with other CX7 throughput reports. It is a boot and security
  setting; the operator decides.
- **NCCL.** The image ships NCCL 2.29.7. The reported GB10 deadlock was the pip 2.28.9 wheel
  (vLLM #46097). NCCL PR #2393, an ARM memory-ordering race, would not self-clear, and ours did.
