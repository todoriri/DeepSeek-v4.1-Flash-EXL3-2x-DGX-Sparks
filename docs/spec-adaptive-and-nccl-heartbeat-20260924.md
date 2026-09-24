# Adaptive speculative verification (not possible here) + the NCCL heartbeat clock (2026-09-24)

These are two maintenance-window items that shared one planned teacher restart.
1. The 09-23 paper review's adaptive-verification trial: DSpark picks its verify budget
   per request instead of a fixed k.
2. The 09-21 wedge investigation's §9 item 2: set `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`
   explicitly, to see whether the ~480 s stall-recovery clock follows it.

## Baseline (before any change): DSpark fixed k=3

`scripts/spec_bench.py --label k3-fixed`, 18:16–18:22 CEST, with the teacher otherwise
idle (repl keepalive paused). The bench runs 8 long-output prompts (code, prose, math,
structured) with thinking on at effort 50, official sampling and 1024 max tokens.
Acceptance comes from `/metrics` deltas.

| concurrency | aggregate tok/s | per-request decode tok/s (median) | TTFT median | accepted / draft token | mean acceptance length | per position |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 41.5 | 43.9 | 0.35 s | 0.486 | 2.46 | 0.69 / 0.46 / 0.31 |
| 2 | 60.2 | 32.8 | 0.47 s | 0.491 | 2.47 | 0.69 / 0.47 / 0.32 |

Raw data: `_repl_measurement_stash/spec-trial-20260924/bench-k3-fixed.json`. For
comparison, the lifetime counters before the restart gave 57.8 % accepted and a mean
length of 2.73 on live (agentic) traffic.

## The trial restart failed: adaptive verification is refused by this image

- **Launch.** `start.sh` (with the `DSPARK_ADAPTIVE` switch of `8ba63cf`) went to the gb10
  kit, with a backup of `start.sh` + `.env` as `*.bak-spectrial-20260924-182230`. Then
  `SKIP_BUILD=1 SKIP_PULL=1 DSPARK_TOKENS=5 DSPARK_ADAPTIVE=1
  TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300 ./start.sh restart` at 18:23:12.
  - The rendered flag was correct on both ranks:
    `{"method":"dspark","num_speculative_tokens":5,"enable_adaptive_verification":true}`.
- **Failure.** After ~9 min of weight loading, both ranks failed at KV-cache init:

  ```
  ValueError: Adaptive verification trims verification requests on device, which the
  DeepseekV4IndexerBackend attention backend does not support. Pass
  enable_adaptive_verification=false in the speculative config, or use a backend that does.
  ```

  (`vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:463`, from
  `model_runner.initialize_kv_cache`.)
  - The config validator (`vllm.py _validate_adaptive_verification`) only checks LoRA,
    eager mode and PP, so this surfaces only after the full load.
  - `start.sh` reported "server did not become healthy". The head container exited (1).
    The watchdog stayed stopped, because start.sh re-arms it only when the server is ready.
  - Logs: `/mnt/storage1/dsv41/spectrial-fail-20260924-182312/` on gb10, copied to
    `_repl_measurement_stash/spec-trial-20260924/fail-logs/`.
- **Recovery.** `SKIP_BUILD=1 SKIP_PULL=1 DSPARK_ADAPTIVE=0
  TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300 ./start.sh restart` at 18:35:17, with k=3 from
  `.env`. Healthy at 18:45:27.
  - KV pool unchanged: 1,253,996 tokens, 3.19× at 393,216. Served ids
    `deepseek-v4.1-flash` + `teacher`. Watchdog re-armed. repl wire smoke OK.
  - **Teacher outage: 18:23–18:45, about 22 minutes.** The repl request record shows no
    client traffic in that window.
- **Consequence for the recipe.** The `DSPARK_ADAPTIVE` switch is removed, since on this
  image it could only fail, and only after a long load. `start.sh` now refuses an
  `enable_adaptive_verification` smuggled in through `EXTRA_ARGS` before anything
  starts. The rendered `--speculative-config` is byte-identical to before
  (`tests/test_nccl_heartbeat.py`).
- **Revisit** only with an image whose DeepSeek V4 indexer backend supports on-device
  verify trimming.

## NCCL heartbeat: live at 300 s on both ranks

- The recovery restart carried `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300`. It is verified in
  the container env of both `dsv41-exl3-head` and `dsv41-exl3-worker`, and it is also
  pinned in the kit `.env`, so a plain restart keeps it.
- `start.sh` validates it (a positive int ≤ 86400) and passes it to both ranks only when
  set. Unset means torch's default, 480 s.
- **What it tests (09-21 doc §5).** Two stalls recovered about 484 s after the grace
  window opened, which matches torch's 480 s default.
  - If the next stall recovers (or the rank aborts) about 300 s in, the heartbeat monitor
    owns that clock.
  - If it still takes about 480 s, the clock is something else.
- **Risk, accepted with the item.** If the ProcessGroupNCCL watchdog thread is itself stuck
  for 300 s, the heartbeat monitor aborts the process. A stall that used to self-recover
  in about 8 minutes could then become a crash and a reload. Before, a 480 s abort never
  fired (RestartCount 0), so the watchdog thread was never stuck that long. A stall of
  300–480 s is the case that changes.
- **Nothing to measure until the next stall.** The vllm-watchdog capture (grace window,
  late SM%) records it.
