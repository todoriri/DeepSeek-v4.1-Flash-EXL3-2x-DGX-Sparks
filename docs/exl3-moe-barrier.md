# The exl3_moe barrier: make the launch atomic, or don't launch at all

*Verified 2026-09-19 against the recipe image's pinned ExLlamaV3 commit
`e648f1a131365aae15920073e761a3fa5a527654` (v1.4.5, 2026-08-31).*

The gb10 teacher wedges silently and permanently: the front end stays green
(`/v1/models`, `/metrics` 200), `vllm:num_requests_running` freezes at 1, token
counters stop advancing, and only a cluster restart recovers it — ~7 stalls on
2026-09-19 alone. This note records the one **structural** defect found in the
path that wedge is pinned to, the minimal patch for it, and — just as
importantly — what is still unproven.

## What is verified

Read from the running container and the pinned upstream source, not from docs:

| | |
|---|---|
| Image | `ghcr.io/miaai-lab/…:2.9bpw`; exllamav3 **1.4.5**, upstream `turboderp-org/exllamav3` at `e648f1a1`, built in-image from `/tmp/exllamav3` |
| MoE path | `vllm/.../quantization/exl3.py` → `exllamav3_ext.exl3_moe` (live: `EXL3_FAT_GROUPED=1`, `EXL3_TEMP_ROWS_FUSED=16`, `max-num-seqs 3`, dspark spec-decode) |
| Kernel barriers | `exl3_moe_kernel.cuh` **111, 162, 185, 234, 265** — all `group_barrier(...)`, **unconditional**, over `locks + BARRIER_LOCKS_OFFSET` |
| Shared state | `exl3_devctx.cu:59-70` — the `locks` buffer is `cudaMalloc`'d and **`cudaMemset` to zero exactly once** (lazily, first use). Stale arrivals survive for the process lifetime. |
| Launcher | `exl3_moe.cu:208` takes `locks`; `:210` states the requirement — *"All blocks of the grid must be co-resident for the group barriers"*; `:214-216` size the grid to fill the device; `:290` launches with a plain **`cudaLaunchKernel`** |
| Same extension, other choice | `exl3_gemm.cu:296` and `:619` launch with **`cudaLaunchCooperativeKernel`**, and dense EXL3 GEMMs run inside vLLM's captured decode graphs in this stack |

The fused MoE kernel is therefore the one kernel of the family that depends on
whole-grid co-residency *and* does nothing to obtain it. A plain launch enforces
nothing; there is no timeout, no error and no recovery — the barrier simply
spins, and the half-advanced arrivals stay in the device-global buffer, which is
why the wedge is permanent rather than transient.

## Why this is worth fixing even though it is not yet proven to be *the* cause

The 2026-09-19 py-spy captures localize the wedge only weakly: the "active"
worker thread pins in five different places (MoE fat tables, a DeepGEMM FP8
`o_proj` with no EXL3 anywhere in the stack, a plain `exl3` linear, and raw
`libcuda.so` frames). The 08:56 native stack shows the host inside
`cuLaunchKernel` launching a trivial `.half()` cast — async launch-queue
back-pressure, the artifact yeasah's own notes describe ("the faulthandler stack
lies under an async launch queue"). A single kernel that never completes upstream
in the queue produces exactly this signature, so those captures can neither
confirm nor exclude the barrier.

What they can do is *exclude the alternative*. The watchdog now samples GPU state
at the stall (`stall-<ts>-head-gpu.txt`, `CAPTURE_GPU_STATE=1`): **SM% pinned** ⇒
a kernel is spinning on the device, and this patch targets it; **SM% ~0** ⇒ the
device is idle while the host sits in a driver call (the hypothesis MiaAI-Lab
recorded in `GLM-5.3-Flash-EXL3-2x-DGX-Sparks#128`), in which case this patch is
not the fix and the allocator path is where to look.

## 2026-09-20 — the SM% discriminator fired: `terminal_device_spin`, with a self-recovery caveat

The instrument added earlier on 2026-09-20 (watchdog grace window + late GPU
resample, logging_stack `a31d95b`) caught its first post-deploy terminal-class
stall, and it was on the **production shape** — a ~275k-resident session continued
after a long idle, i.e. one large prefill, not the synthetic 2-concurrent
`disturb_wedge` load. Sequence:

- freeze → `stall_suspected` / `stall_confirmed` (forward-progress probe timed
  out); early GPU sample 96 % (the ambiguity zone); py-spy pin
  `build_grouped_fat_tables` (`exl3.py:1353`), `in_collective=false`, NCCL
  `only active collectives: 0`;
- `grace_window_open` (600 s), during which the engine **micro-recovered**
  (+39 combined tokens) then **re-froze**;
- the re-freeze held the full window: **late GPU resample +300 s = 94 %,
  +600 s = 96 %** (both pinned) → `stall_persists`, `outcome:
  terminal_device_spin`.

So the discriminator this doc describes finally produced a reading, and it is the
**SM-pinned** one: the device is spinning, not host-blocked (`SM ~0`). That
corroborates the barrier / co-residency reading over the MiaAI-Lab allocator
hypothesis *for this event*, and it is the cell the cooperative-launch patch
targets.

**The caveat that keeps the toggle OFF.** The engine **self-recovered ~1–5 min
after the 600 s verdict, with no cluster restart** — post-verdict token counters
climbed from the wedged values (11.358 M → 13.341 M prompt) rather than resetting.
Total spin ≈ 11–15 min. A permanently dead barrier does not self-clear, so the
honest classification is a **long-but-recoverable device-spin**, not the permanent
wedge this doc's opening describes (those — 2026-09-18 19:46 / 19:52 — needed an
operator reboot). Consequences:

1. The 600 s grace window is **too short**: it declared terminal at a boundary a
   ~11–15 min spin overshoots. Either extend it or add a distinct
   `terminal_device_spin_self_recovered` outcome that keeps sampling until real
   recovery or a restart. **(RESOLVED 2026-09-20 — see the next subsection:
   device-spin extension, logging_stack `1d603f4`, deployed + verified live.)**
2. Cooperative-launch fail-fast would convert *this* event from an ~11 min
   self-heal into an immediate `cudaErrorCooperativeLaunchTooLarge`. Bounded
   failure may beat an 11 min stall, but that is a different value proposition
   than "cures a deadlock" — frame the trial that way.
3. The confirm-first gate for the **permanent** wedge is still not satisfied by a
   self-recovering event. Trial the `9a9c44f0…` cooperative-MoE overlay in a
   maintenance window (overlay-OFF baseline first) against the real terminal
   trigger before repinning.

### A concurrency source the repro had not accounted for: the TTFT keep-alive

The `repl` proxy runs a systemd timer (`sovereign-router-keepalive.timer`, every
5 min) that fires one tiny worker-tier completion at the teacher to avoid the
~40 s first-forward-after-idle stall. It is **meant** to be idle-gated
(`scripts/teacher_keepalive.sh` skips when `cc_metrics.log` was touched within
240 s), but that activity file went **stale for ~6 h across this window**
(`idle_age` climbing 16 k → 22 k s in `keepalive.log`), so the gate never engaged
and the ping fired **every 5 min through the disturb campaign and the wedge** — its
`total` times (8.7 s, 19.7 s, 28.9 s, then a 90 s `rc=28` timeout against the
wedged engine at 10:21:48Z) show it contending with live load, not pinging an idle
one.

Since the deadlock is **concurrency-gated, not length-gated** (`#22`: two
concurrent requests, even small, is enough), an un-gated 5-min keep-alive is a
standing second consumer that can supply the co-residency race — and it fires
preferentially at the idle→active transition, which is exactly when a user resumes
a session. This does **not** prove it caused any specific wedge (the dominant load
here was the 275k prefill), but it is an **unaccounted concurrency source** that
any repro or mitigation must control for. Fix the activity signal (or disable the
timer) before attributing wedge frequency to user traffic alone.

### 2026-09-20 (later) — a second self-recovering device-spin, and the grace window extended

A second stall the same day (~11:27–11:34Z, the same 275k session resumed warm)
reproduced the pattern and sharpened it. The py-spy pin this time was
`apply_exl3_fused_moe` (`exl3.py:1655`) with **`in_launch=true`** — caught *inside*
the fused-MoE `cudaLaunchKernel` at `exl3_moe.cu:290`, the exact site this patch
rewrites, rather than the fat-table builder. The grace window resampled **+300 s =
93 % SM (pinned)** and the engine then **self-recovered at 424 s** (a re-prefill
burst, `prompt throughput 19,896 t/s`, prefix hit ~46 %). So there are now **two**
independent `pinned + self-recovers` captures (94–96 % and 93 %), both released
without a restart — the device-spin arm is real but *releasing*, and there is still
**zero** `pinned + no-recovery` (permanent) capture under the late-sample instrument.

**Consequence #1 above is now fixed** (logging_stack `1d603f4`, deployed to gb10 +
verified live in the `watchdog_start` config). When the 600 s grace window expires
with SM pinned, the watchdog no longer declares terminal at that edge: it holds a
bounded **device-spin extension** (`RECOVERY_DEVICE_SPIN_EXTEND_S`, default 1200 s →
up to 30 min total) that keeps sampling counters and a coarse GPU trajectory, and
splits three ways —

- counters resume → `stall_recovered` with `outcome=device_spin_self_recovered` (no restart);
- extension also expires still pinned → `terminal_device_spin` (the genuine `pinned + no-recovery` cell — the only one that justifies the overlay/patch);
- extension expires with SM decayed to ~0 → reclassified `terminal_host_block`.

Host-block and unknown still escalate at the 600 s edge (waiting buys nothing). New
event `device_spin_extend_open`; `stall_recovered` now carries `outcome` +
`recovered_after_s`; 19 offline tests green. This closes the false-terminal that
mislabeled the first 2026-09-20 capture, so a terminal-repro now classifies
honestly: **only an extension that *also* expires pinned** is the confirm-first datum
the overlay trial waits on.

## The patch

`overlay/patch_exl3_cooperative_launch.py` rewrites the single launch call in
`exl3_moe.cu`:

```diff
-    cudaLaunchKernel
+    cudaError_t coop_err = cudaLaunchCooperativeKernel
     (
         (void*) kernel, grid_dim, block_dim, kernelArgs, SMEM_MAX, stream
     );
+    TORCH_CHECK(
+        coop_err == cudaSuccess,
+        "exl3_moe cooperative launch failed (",
+        grid_dim.x * grid_dim.y * grid_dim.z, " blocks, ", num_sms, " SMs): ",
+        cudaGetErrorString(coop_err));
```

Per the CUDA Programming Guide (Cooperative Groups §4.4.8.1): *"cudaLaunchCooperativeKernel
ensures that the kernel launch is atomic, i.e. if the API call succeeds, then the
provided number of thread blocks will launch on the specified device."* A grid
that cannot be launched cooperatively returns `cudaErrorCooperativeLaunchTooLarge`
instead of hanging, so the same condition becomes an exception in vLLM.

The kernel's synchronization logic is untouched. When the build toggle is off the
build runs the patch with `--check`, so **upstream drift fails the build** rather
than silently producing an unpatched image.
`tests/test_exl3_cooperative_launch_patch.py` covers the patch, the drift cases,
the CLI and the recipe wiring, including an apply against the real pinned file
(`EXL3_MOE_COOP_LIVE_SOURCE=/path/to/exl3_moe.cu`).

**Known limitation, stated plainly.** The guide's guarantee is about launch
atomicity; it is not a promise of priority over other work when the device is
busy. An NVIDIA engineer on the developer forums is explicit that cooperative
launch provides "a valid environment for the operation of grid sync" and that a
resource-hogging kernel **in the same process** is not preempted. This stack
already serializes the EXL3 streams (`DSV41_EXL3_SERIAL_STREAMS=1`,
`VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`), so what this closes is the contention
window the launch itself creates.

## Building a candidate image

```sh
docker build --build-arg EXLLAMAV3_MOE_COOP_LAUNCH=1 -t dsv41-flash-exl3:coop-launch .
```

The toggle is **OFF by default** on purpose: the pinned recipe image and the
`9a9c44f0…` cooperative-MoE artifact were validated as they are, and this is a
CUDA binary change.

## Gate before repinning

1. Import check — `exllamav3_ext` still exposes `exl3_moe`, `exl3_fat_gemm`, `exl3_fat_gemm_scatter`.
2. Graph capture still succeeds at the pinned sizes (`--cudagraph-capture-sizes 1 2 3 4 6 8 12 18 24`); watch for `cudaErrorStreamCaptureUnsupported`.
3. Numerics — `tests/test_exl3_overlay.py`, plus the two-node smoke in `tests/test_smoke.sh`.
4. Soak the trigger: two concurrent requests at long context, repeatedly; then read `vllm-watchdog-data/` for stalls **and** the new `-head-gpu.txt` (SM% during any stall).
5. Throughput before/after — cooperative launch changes scheduling.
6. Rollback: rebuild without the build arg; the previous digest stays available.

## Operator-side confirmation (the agent must not run this)

The agent serving this session rides the same teacher, so a
`CUDA_LAUNCH_BLOCKING=1` boot is an operator action for a maintenance window:

- `CUDA_LAUNCH_BLOCKING=1` serializes launches, so the py-spy pin lands on the
  kernel that never completes instead of the op that filled the launch queue.
  Expect a large slowdown; do not leave it on.
- Confirm capture still works at boot (`cudaErrorStreamCaptureUnsupported` in the
  log means graphs are off for that run).
- Reproduce the known trigger (two concurrent requests, ~300k context) and
  capture with `pyspy_dump.sh` as soon as a stall is suspected.

## The deeper fix, if the barrier is confirmed

Replacing the five hand-rolled barriers with cooperative groups `grid.sync()` —
what the yeasah/exllamav3 fork does for `exl3_gemm_kernel.cuh` — removes the
dependency on a device-global buffer entirely. It is deliberately **not** part of
this patch: `exl3_moe_kernel.cuh` has no `cooperative_groups` context, and three
of the five call sites sit inside conditionals, so a blind substitution can
deadlock on its own. Audit first; the sites are `exl3_moe_kernel.cuh:111, 162,
185, 234, 265`. The upstream `exl3_gemm_kernel.cuh` sites are `95, 184, 219` (all
`#if __CUDA_ARCH__ > 890`, therefore active on sm_121 and unpatched in this image
— yeasah's fork gates them behind `EXL3_SM90_BARRIER`).

## Sources

- CUDA Programming Guide, Cooperative Groups §4.4.8.1 —
  <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html>
- NVIDIA developer forums, "cudaLaunchCooperativeKernel behaviour" (Aug 2025) —
  <https://forums.developer.nvidia.com/t/cudalaunchcooperativekernel-behaviour/341549>
- Upstream source at the pinned commit —
  <https://github.com/turboderp-org/exllamav3/tree/e648f1a131365aae15920073e761a3fa5a527654>
