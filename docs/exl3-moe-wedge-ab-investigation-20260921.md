# EXL3 MoE Wedge — A/B Investigation (2026-09-21, final)

> **2026-09-22 UPDATE — root cause reframed by upstream issue [#22]; the desync hypothesis
> (§6) is REFUTED.** An operator on an independent 2×GB10 pair captured *all three processes
> including both worker ranks simultaneously* during the identical ~260 k stall: both ranks
> sit at the same instant in `build_grouped_fat_tables` / `apply_exl3_grouped_fat` with **no
> NCCL frames on either rank** — a two-rank-*symmetric* host-side spin, not one rank waiting
> on the other. The wedge is the **stock EXL3 fat-grouped prefill path contending the single
> per-device split-K lock buffer** under concurrency + long prefill — the same
> "one-lock-buffer-per-device" hazard as the now-fixed shared-experts-stream deadlock, reached
> by a different route. See the new **§10** below; §1–§9 are preserved as the 2026-09-21
> record. [#22]: https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks/issues/22

**Status (2026-09-21, superseded by §10):** Arm B closed (negative). Launch-atomicity lead
retired. Root cause not yet confirmed; leading hypothesis is a two-rank collective desync
cleared by a ~480 s timeout.

**One-line verdict:** The cooperative-launch fix does nothing to the wedge, the wedge is
*not* a single-kernel deadlock and *not* driver OOM — it is a **whole-engine
forward-progress freeze that clears on a fixed ~480 s timeout**, and every capture so far
has only ever seen one of the two ranks.

---

## 1. What was tested

Single-variable A/B on the fused-MoE launch site (`exl3_moe.cu:290`):

| | Arm A (baseline) | Arm B (test) |
|---|---|---|
| Image | GHCR `…:2.9bpw`, `sha256:4cdba4e9…` | rebuild, `sha256:e34293fd…` |
| `exl3_moe.cu:290` | `cudaLaunchKernel` | `cudaLaunchCooperativeKernel` + `TORCH_CHECK` |
| Toggle | `EXLLAMAV3_MOE_COOP_LAUNCH=0` | `=1` |

Deploy path corrected mid-run: `restart` (`stop; start`) builds *inside* `start()` **after**
the engine is down, and there is no build-only verb — so the safe order is **build first
while the engine serves** (`docker build … -t <image> .`), then a build-free
`SKIP_BUILD=1 SKIP_PULL=1 ./start.sh restart`. The watchdog is auto-paused by that path
(`pause_watchdog → ensure_image → resume in on_ready`, gated on `DSV41_MANAGE_WATCHDOG=1`).

**Arm B verified genuinely live** (independent of the build log): `strings` on the shipped
`exllamav3_ext…so` contains the patch's own message `exl3_moe cooperative launch failed (`
exactly once. Both ranks ran `sha256:e34293fd…` from 11:28:53Z, `RestartCount 0`.

## 2. Arm B result — confirmed negative

The 12:20Z production-shape wedge fired on Arm B. The cooperative launch **neither prevented
the spin nor raised the fail-fast**: no `cudaErrorCooperativeLaunchTooLarge` /
`cooperative launch failed` / `cudaError` in `docker logs` since 11:28Z. This is exactly the
pre-registered "B shows no error and no change ⇒ atomicity doesn't own it" branch, and the
documented Caveat #3: `exl3_moe.cu:214` already size-guards the grid to fit `num_sms`, so
`cudaLaunchCooperativeKernel` succeeds-and-waits instead of failing.

Caveat: n = 1, and the Arm-A comparison is unmatched in time/memory state. But the fail-fast
result is *absolute*, not comparative — the check is in the binary and never tripped.

## 3. The launch-atomicity family is dead — not just Arm B

Two afternoon wedges, **different signatures, identical recovery:**

| | 12:20 stall | 13:00 stall |
|---|---|---|
| `in_launch` | **false** | **true** |
| `in_collective` (head py-spy) | false | false |
| pin | `apply_exl3_fused_moe` (exl3.py:1655) | `reconstruct_hgemm` (exllamav3/…/exl3.py:**214**) |
| `frozen_progress` (prompt+gen) | 137,265 | 4,525,892 |
| recovered_after (from grace-open) | **484.3 s** | **484.3 s** |

The `in_launch:true` case is on `reconstruct_hgemm`, a **different kernel** than the
`exl3_moe` launch Arm B patched — so even where a launch was caught, the patch could not
have helped. `frozen_progress` = total prompt+gen tokens (137265 = 95197 + 42068 from the
12:27 metrics sample) and stayed frozen for the whole window with the GPU at 96 % SM: a
genuine **zero-throughput whole-engine freeze**, not slow-drip decode, that parks wherever
execution happens to sit.

## 4. OOM was a red herring

43 NVRM `NV_ERR_NO_MEMORY … @ mem_desc.c:1359` lines on 2026-09-21, in three bursts:
**13:29 (×29), 13:38 (×7), 13:50 (×7)**. **None** coincide with a watchdog stall; the two
stalls (12:20, 13:00) had **zero** OOM. The driver OOMs are allocator-handled noise from a
memory-tight box (`MemAvailable ≈ 4.6 GiB` at boot, mem-guards disabled) — a real
constraint worth fixing for headroom, but **not** the wedge mechanism.

## 5. The ~480 s recovery clock

Both stalls recovered ~484 s after `grace_window_open` — identical across two structurally
different stalls ⇒ recovery is **external/timeout-driven**, not kernel self-heal. Sourcing it:

- **Prime suspect:** `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` is **unset → PyTorch default 480 s**,
  and the NCCL flight recorder **fired during the stall** (`flight_pipe triggered=1`,
  `/tmp/dsv41_nccl_flight_rank_0`, with `TORCH_NCCL_DUMP_ON_TIMEOUT=1`,
  `TORCH_NCCL_TRACE_BUFFER_SIZE=20000`). That is a collective-level fingerprint.
- **Ruled out:** `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800`, `UV_HTTP_TIMEOUT=500` (pkg-mgr).
- **Tension:** an NCCL heartbeat timeout normally *aborts* the rank, but the engine never
  restarted (`RestartCount 0`). So either it is a recover-not-abort variant, or the ~480 s
  value is a sampling artifact (60 s grace cadence bucketing recovery at the 480 s tick).
  Unresolved — resolve by setting the timeout explicitly and seeing if recovery time moves.

## 6. Leading hypothesis — two-rank collective desync  ⚠️ REFUTED (see §10)

> Refuted 2026-09-22 by upstream #22's simultaneous both-rank capture: the stall is
> two-rank-*symmetric* at a host-side fat-table function with **no NCCL frames on either
> rank**, which is the opposite of one rank waiting on a collective. Kept for the record.

A spin-with-no-progress that lands on *different* kernels and clears on a *fixed* timeout is
the classic profile of one rank waiting on the other. The head rank spins (sometimes in a
launch, sometimes host-side); the NCCL flight recorder engages; a ~480 s timeout clears it.

**We have never captured the worker rank.** `WORKER_CAPTURE_CMD` is unset in every capture
(the `vllm-watchdog` container has no `ssh` binary and no `/root/.ssh`). We have been
diagnosing half the system.

## 7. Trigger lane

- The wedge fires on a **production ~260 k-token continue-prefill** (2× on 09-20 arm A, 1×
  today arm B). The engine-direct `disturb_wedge.py` **script lane is low-yield** (0 stalls
  today and in earlier controlled runs) — treat stall-free script runs as uninformative, and
  never run the script alongside a live DSH session (makes any stall unattributable).
- **Self-amplifier:** undici's 300 s `bodyTimeout` kills the turn and retries the whole
  ~260 k context → a second resident prefill.
- **The timeout "fix" is inert:** committed on branch `fix/http-proxy-body-timeout@5c0bfb85a6`,
  but `dsh web` runs **master** (Sep-19 `apps/cli/lib/bin.js` build) which has no such knob.
  `DSH_HTTP_BODY_TIMEOUT_MS=1800000` is exported but dead code → undici 300 s still governs.

## 8. Ruled out

- Launch atomicity of the fused MoE (Arm B, confirmed).
- Driver/UMA OOM as the stall cause (timing anti-correlated).
- Fat-path grouped kernels (earlier: co-residency-free, only `__syncthreads`; and
  `EXL3_FAT_GROUPED=0` falls back to the *stock* co-residency-dependent path → more exposure).

## 9. Next actions (prioritized)

> **2026-09-22:** item 1 is **done and superseded** — worker capture was armed
> (`vllm-watchdog:ssh`, `worker_capture:true`, verified live) *and* the both-rank capture now
> exists upstream (#22), which killed the desync hypothesis it was meant to test. The live
> priority is now the **Mode-2 per-device-lock-buffer** thread in §10, not more captures.

1. ~~**Arm worker-rank capture** so the next stall snapshots *both* ranks.~~ **DONE** (key
   mounted at `/keys/worker_capture`, forced-command on kgb10) **+ superseded by #22's
   both-rank capture.** This is the one instrument that can confirm/kill the desync
   hypothesis — and it did: refuted, see §10.
2. **Nail the 480 s clock:** set `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` explicitly (e.g. 300 s)
   and check whether recovery time tracks it. If yes → NCCL heartbeat confirmed → collective
   desync all but proven.
3. **Deploy the real bodyTimeout fix** (checkout `fix/http-proxy-body-timeout`, rebuild CLI
   artifacts, restart `dsh web`). This is a **severity mitigation** (stops the retry doubling
   resident prefill), not a root cause. Do **not** flip it *between* A/B arms.
4. **Headroom** (secondary): lower KV pool / `gpu-mem-util` / `--max-num-seqs`, or enable
   `DSV41_MEM_GUARD=1` to convert a driver-OOM spin into a clean request kill.

`CUDA_LAUNCH_BLOCKING=1` drops down the list: it answers "which kernel is slow," but if the
head is slow because it is *waiting on the peer*, that is the wrong question.

## 10. Upstream #22 reframe — the per-device EXL3 lock buffer (2026-09-22)

Upstream issue [#22] ("Two concurrent requests deadlock the pair, 2×GB10 TP=2", closed
2026-09-22) resolves most of this investigation and splits what we called "the wedge" into
**two distinct modes**.

### Mode 1 — shared-experts-stream lock corruption (root-caused; **we are already immune**)
`DSV41_EXL3_SERIAL_STREAMS=1` only nulls the *model's* `aux_stream_list`. vLLM's MoE
**shared-experts** runner takes a *separate* process-global stream
(`vllm/utils/torch_utils.py::aux_stream()`, placed on the GEMM in
`fused_moe/runner/shared_experts.py`, gated by `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD`,
default 256 → every decode qualifies). The shared-expert EXL3 GEMM on that aux stream races
the routed EXL3 MoE on the main stream and they corrupt each other's split-K tile locks —
the documented one-lock-buffer-per-device constraint, on a stream the recipe's patch cannot
see. Symptom: two concurrent *short* requests deadlock, `shm_broadcast` 60 s timeout,
EngineCore force-killed, HTTP 500 at 243 s. **Fix = `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`**
(plain vLLM env, default `False`). Our live `.env` already sets it → our earlier 5-way decode
burst does not wedge. Narrower lever: `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=0`.

### Mode 2 — fat-grouped prefill spin (**still open = our remaining wedge**)
Reproduced by the operator *with the Mode-1 fix in place*, via **two concurrent cold ~260 k
prefills** — our exact trigger lane (§7). They captured all three processes, **both worker
ranks at the same instant**:

- **EngineCore** host-spin: `sched_yield ← shm_broadcast.wait ← acquire_read ← get_response`
  (`multiproc_executor.py:433`) — waiting on the workers.
- **Worker_TP0 and Worker_TP1, same instant:** `build_grouped_fat_tables` (`exl3.py:1354/1349`)
  under `apply_exl3_grouped_fat` (`1408`). **No NCCL frames on either rank.** GPU 96 % util
  spin (not progress). Freeze cycles ~3.5–4.5 min, self-clear, engine never dies.

**Mechanism, grounded in `overlay/exl3.py`.** With `EXL3_FAT_GROUPED=1`
(`_exl3_fat_effective_tier == "grouped"`), the large-prefill branch (`apply_exl3_fused_moe`,
lines 1690–1708) is deliberately **host-sync-free / graph-capturable**: it launches the thin
fused `exllamav3_ext.exl3_moe` kernel *and* `apply_exl3_grouped_fat` (which fires
`exl3_fat_moe_gather/gateup/down`) and returns with **no `.item()`**. So on our config there
is no host sync to hang on — the *device* wedges and the host thread parks launching the
sync-free `build_grouped_fat_tables` device ops (`searchsorted`/`cumsum`/`index_select`)
against a saturated queue. That is exactly the captured pin. The overlay does **no**
request-level serialization of EXL3 launches (grep: only `_stage_counts_to_host`'s side
stream, unused on the grouped path); concurrency safety rests entirely on the native `.so`'s
**single per-device split-K lock buffer**, which tolerates only one EXL3 GEMM family per
device at a time. Under `MAX_NUM_SEQS>1` + long prefill, two requests' fat-expert MoE work
becomes device-co-resident and contends that one lock buffer → symmetric spin on both ranks.
The `EXL3_FAT_GROUPED=0` fallback makes the same wedge *visible* as the host sync
`int(counts.max().item())` (line 1736) / `count_stream.synchronize()` (1728) "that never
returns", because there the host explicitly waits on the wedged device.

This also **corrects §8**, which prematurely "ruled out" the fat-grouped kernels as
co-residency-free: the *Python* grouped path is host-sync-free, but the *device* fat kernels
still share the per-device lock buffer, so they are the co-residency exposure, not the
exception to it. And it **re-confirms §2–§3**: the wedge is stock upstream code, independent
of our cooperative-MoE overlay (which only routes thin/decode launches).

### What this changes for us
- The risky local reproduction (deliberate ~8-min freeze) to "capture the worker" (§9.1) is
  **no longer needed** — the both-rank capture exists upstream. Our own worker-capture arming
  (`vllm-watchdog:ssh`, `worker_capture:true`) is verified live and stays as a safety net.
- **Our exposure:** live `.env` = `EXL3_FAT_GROUPED=1`, `EXL3_FUSED_MOE=1`,
  `EXL3_TEMP_ROWS_FUSED=16`, `MAX_NUM_SEQS=3` → on the Mode-2 path with concurrency enabled;
  `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` (immune to Mode 1); `LANGUAGE_MODEL_ONLY=1`
  (vision off → not exposed to the #28 encoder-cache livelock); `MAX_NUM_BATCHED_TOKENS=1024`
  (tightest per #19; 1536 would likely gain head headroom, restart-gated).
- **No upstream PR fixes Mode 2** (Mode 1 was resolved by operators setting the env, not
  code). The real fix is native: make the split-K lock buffer per-invocation, or serialize
  EXL3 GEMM families per device. Cheap operational mitigations to test (no code): force
  `MAX_NUM_SEQS=1` (kills the co-residency, at a concurrency cost), or gate concurrent long
  prefills so two fat-expert passes never overlap.

### Candidate confirmations (do not require inducing a full freeze locally)
1. Ask the operator (offered) for the full forensic write-up and the
   `EXL3_FAT_GROUPED=0` / `EXL3_TEMP_ROWS_FUSED` sweep they volunteered.
2. If we do use a maintenance window: `MAX_NUM_SEQS=1` should make the ~260 k×2 stall vanish
   (positive control for the co-residency mechanism); an `EXL3_TEMP_ROWS_FUSED` bump changes
   the fat/thin boundary and should move the onset.

## Appendix — verification commands

```bash
# Arm B live + arm-A revert anchor
ssh gb10 "docker inspect -f '{{.Image}}' dsv41-exl3-head; \
          docker inspect -f '{{.Id}}' dsv41-flash-exl3:armA-ghcr-20260912"   # must differ
ssh gb10 "docker exec dsv41-exl3-head sh -c 'strings \$(find / -name \"exllamav3_ext*.so\" | head -1) | grep -c \"cooperative launch failed\"'"

# OOM vs stall timing
ssh gb10 "sudo dmesg -T | grep 'Sep 21' | grep -iE 'NV_ERR_NO_MEMORY|_memdescAllocInternal'"
ssh gb10 "grep -E '\"event\": \"(stall_confirmed|stall_recovered)\"' \
          ~/logging_stack_gb10/vllm-watchdog-data/events.jsonl | grep 2026-09-21T1"

# Stall pins (in the diagnostics_captured event, not stall_confirmed)
ssh gb10 "grep diagnostics_captured ~/logging_stack_gb10/vllm-watchdog-data/events.jsonl | \
          grep 2026-09-21T1 | grep -oE '\"pin\": \"[^\"]*\"|\"in_launch\": (true|false)'"

# Timeout env sourcing the 480 s
ssh gb10 "docker inspect dsv41-exl3-head --format '{{range .Config.Env}}{{println .}}{{end}}' | \
          grep -iE 'NCCL|TIMEOUT|HEARTBEAT'"
```
