# Wedge repro runbook: arm the instrument, cause the wedge, read the verdict

*Operator playbook, 2026-09-20. Companion to [`exl3-moe-barrier.md`](exl3-moe-barrier.md)
(the structural defect) and [`cooperative-moe-quickstart.md`](cooperative-moe-quickstart.md)
(the overlay we already hold). All steps below ride the live teacher → **maintenance
window, operator-run.***

## The one datum this campaign is missing

Every downstream decision — whether the cooperative-launch patch / coop overlay is
the fix, or whether it's a host-side syndrome the overlay can't touch — turns on a
single measurement we have never captured: a **TIME-confirmed TERMINAL wedge, with SM%
read LATE**. The 2×2:

| | counters resume (recoverable) | counters never resume (terminal) |
|---|---|---|
| **SM pinned late (> ~50%)** | benign long kernel | **device barrier-spin → the coop path is the fix** |
| **SM ~0 late** | driver/launch stall *(every capture so far, incl. 4d4da228)* | host-block (allocator/#128 class → coop path will NOT help) |

We only ever land in the recoverable row. The bottom row is the missing corpus.

## Why the wedge is hard to cause on demand

It is a **co-residency race, not a load threshold** (full mechanism: `exl3-moe-barrier.md`).
The fused-MoE grid is sized to fill the device and synchronizes through device-global
barriers that need *every block co-resident at once*. It goes **terminal** only when
that grid's launch overlaps in time with another SM consumer (a second request at a
*different* pipeline stage, spec-decode draft, shared-expert stream, or the cross-rank
all-reduce) such that a block can never become resident. Whether the overlap is
permanent (wedge) or transient (crawl-then-recover) is probabilistic.

Note on shape: two *identical simultaneous* prefills tend to **batch into one MoE
launch** (`--max-num-seqs 3`) = a bigger batch, not two racing launches. The
reliable-in-theory shape is **one long resident DECODE beside a large concurrent
PREFILL** — kernels at different stages, genuinely concurrent (matches #128 and our own
onsets). `disturb_wedge.py` models exactly this.

The default `disturb_wedge.py` **retreats at the contention peak** (silence-stop,
capacity back-off, KV back-off) — which is why every controlled run so far only
crawled and recovered. `--sustain` disables those retreats so contention is held
*through* the peak (host-OOM mem-guard stays on).

> **Honest caveat:** there is no *proven* deterministic terminal trigger. #22 reports
> 2×800-tok is reliable for them; we have only crawled. Expect to run `--sustain`
> repeatedly, and be ready for the outcome "it still only crawls" — that itself is a
> finding (terminal may need true production shape / longer hold). This is exactly why
> the instrument is armed first: whichever run finally tips terminal — deliberate or
> incidental — the watchdog self-captures the cell.

## Step 1 — arm the instrument (do this FIRST; it only captures, it cannot cause a wedge)

Deploy the recovery-aware grace window + late SM% resample (logging_stack `a31d95b`):

```bash
scp -r /mnt/data1/Projects/logging_stack/vllm-watchdog gb10:~/logging_stack_gb10/
ssh gb10 'cd ~/logging_stack_gb10 && docker compose -f docker-compose.gb10-watchdog.yml up -d'
```

Set the peer-rank capture so the two-rank asymmetry lands in the event (not hand-grabbed):

```bash
# in the watchdog service env (compose / .env on gb10):
WORKER_CAPTURE_CMD="ssh -o BatchMode=yes kgb10 'docker exec dsv41-exl3-worker bash /opt/dsv41/pyspy_dump.sh'"
```

Confirm it loaded: the `watchdog_start` event should show `recovery_grace_enabled: true`,
`recovery_gpu_resample_offsets: "300,600"`, `worker_capture: true`.

(Free insurance, next teacher boot only: rename `TORCH_NCCL_TRACE_BUFFER_SIZE` →
`TORCH_FR_BUFFER_SIZE` in `start.sh` so the NCCL flight recorder is actually on.)

## Step 2 — cause the repro (BASELINE, overlay OFF)

Run the sustained production-shape trigger against the stock stack:

```bash
# long decode beside sustained, aborting large prefills — held through the peak
python3 scripts/disturb_wedge.py --sustain --max-minutes 45 \
  --decoder-prompt-tokens 4000 --decoder-max-tokens 8000 \
  --prefill-min-tokens 120000 --prefill-max-tokens 300000
```

If that only crawls, also try the blunt #22 shape (no back-offs by construction):

```bash
python3 scripts/repro_wedge.py --requests 3 --prompt-tokens 200000 --max-tokens 512
```

Do **not** stage the overlay yet: a baseline terminal cell must exist before an
overlay run can prove anything (otherwise "no wedge" is confounded with "never hit
terminal").

## Step 3 — read the verdict (from the watchdog events / durable files)

The watchdog now decides the cell for you. Watch for, in order:

- `stall_suspected` → `stall_confirmed` (`engine_forward_progress_stall`, probe timed out)
- `grace_window_open` → periodic `grace_gpu_resample` (`mean_sm_pct` at +300s, +600s)
- then exactly one of:
  - **`stall_recovered`** — counters resumed (or probe answered) → recoverable, no fix justified.
  - **`stall_persists` with `outcome: terminal_device_spin`** (`last_mean_sm_pct` > 50) →
    **THE cell that justifies the coop path.** Proceed to Step 4.
  - **`stall_persists` with `outcome: terminal_host_block`** (`last_mean_sm_pct` ~0) →
    host/allocator syndrome (#128 / vLLM #41725 class) → the coop overlay will NOT help;
    pivot to the allocator/attention-GEMM path, not a MoE-kernel swap.
  - **`terminal_unknown`** — no late SM reading; fix the capture and re-run.

Durable artifacts land in `~/logging_stack_gb10/vllm-watchdog-data/`:
`stall-<ts>-head-gpu-late-{300,600}s.txt`, `-head-pyspy.txt`, `-worker-pyspy.txt`, and
the event carries `asymmetry: head=… worker=…`.

## Step 4 — treat, and re-test (ONLY if Step 3 = terminal_device_spin)

Stage the validated coop overlay (`9a9c44f0`) per `cooperative-moe-quickstart.md`, select
it on both ranks (`EXL3_OVERLAY_HOST`), and re-run the **same** Step 2 trigger. It swaps
only the DECODE MoE barrier path, so it is a real test, not an assumed cure:

- wedge no longer fires (or now reads recoverable) → the decode-MoE barrier was the arm; adopt.
- wedge still fires terminal, pin in attention-GEMM/prefill (our o_proj `reconstruct_hgemm`
  / #128) → the shared GB10-TP2 syndrome, overlay is not sufficient.

Do **not** attempt `EXL3_P2B_MOE` (it exists nowhere upstream). A from-scratch repin to
upstream `dev`/`exl3_moe_coop` is the fallback only, not the first move.
