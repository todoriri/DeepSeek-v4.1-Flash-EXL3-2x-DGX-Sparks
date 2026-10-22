# Upstream issue #23 — display-reserve KV (3.1M) vs. this kit

**Reviewed:** 2026-10-09 (UTC). **Upstream artifact:** `MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks#23`
(opened 2026-09-18 by @coolbho3k, still **open**, label `enhancement`).
**Subject:** whether the display-reservation KV unlock is implemented on our backend, and where the
claimed 3M tokens actually come from.

Basis tags follow `KV_CACHE_ISSUE.md`: **[verified]** = read the live artifact / ran the command this
session; **[inferred]** = reasoned from evidence; **[open]** = not established.

---

## TL;DR

- **Not implemented here.** The live kit serves the MiaAI 2.9 bpw image with a 4 GiB ordinary-RAM KV
  pin, **1,223,846 tokens** of pool, and `fp8_ds_mla` KV chosen by the image. No DRM dumb-buffer
  allocator, no FP4 KV, no DCP2. **[verified]**
- **The 2 GB display region is one of three things you need, not the whole story.** 1.75 GiB at our
  measured KV cost buys ~535k tokens — about **17%** of 3.15M. The rest comes from a different KV
  format (FP4 main + FP8 SWA) and DCP2 across both ranks. **[verified arithmetic, inferred attribution]**
- **The display region is not freed memory.** It is a firmware scanout reservation CUDA cannot see;
  the unlock allocates a DRM dumb buffer inside it and registers that buffer as CUDA-accessible
  memory. RAM does not increase. **[verified: upstream source]**
- **Our hosts are configured the opposite way** to the prerequisite: boot target `graphical.target`,
  `nvidia-drm.fbdev=1`, desktop session active on gb10. Adopting it costs a headless reboot.
  **[verified]**

---

## 1. What the issue proposes

From the issue body **[verified: `gh api`]**:

| Piece | Claim |
|---|---|
| KV | **3.1M tokens** stored inside the ~2 GB display-reserved region, 1.75 GiB used, 256 MB spare |
| Layout | **DCP2 + FP4 KV** ("same as DeepSeek's paper") |
| Quant | 3.0 bpw experts (`coolbho3k/DeepSeek-V4.1-Flash-EXL3-3bpw`), nothing else quantized |
| Drafter | Quantized EXL3 drafter, ~1% acceptance loss |
| Concurrency | C6; "over 6 simultaneous 500k context sessions"; requests tested to the 3.1M limit |
| Vision | Preserved |
| Host config | `sudo rmmod nvidia_drm && sudo modprobe nvidia_drm modeset=1 fbdev=0`; `sudo systemctl set-default multi-user.target` (fully headless) |
| Code | Allocator `display_kv.c` L50–92; PyTorch integration `display_kv.py` L31–59; offers PRs |

Status **[verified]**: `state: open`, no assignee, author association `NONE`. Two comments, both
thank-you/encouragement (Capicua25x, tuolaku) — **no maintainer commitment**. The kit repo has not
been pushed since 2026-09-19T13:38Z (the #21 prefix-retention merge); the issue is dated the same day.

The sibling port exists elsewhere in MiaAI **[verified: code search]**:
`MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold` carries `patches/0072-glm-display-kv.patch`
(the `DISPLAY_KV_MIB=1792` shared-pool carve-out, credited to @gabewillen, by @ezoushen), plus
`patches/0087-glm-display-kv-dispram.patch` and `tools/display_kv_check.py`. That is the GLM
TensorFold stack, **not** the DS4.1 vLLM kit we run.

---

## 2. Is it implemented on our backend? — **No** [verified]

Live stack: `dsv41-exl3-head` (gb10, 192.168.1.97) + `dsv41-exl3-worker` (kgb10:10.10.10.2),
image `ghcr.io/miaai-lab/deepseek-v4.1-flash-exl3-2x-dgx-sparks:2.9bpw`, up 6 days.
Hermes' `custom-teacher` provider (`base_url: http://127.0.0.1:8000/v1`, `key_env:
HERMES_CUSTOM_DS4FLASH_API_KEY`) resolves through `rai:8000` → gb10:8000.

### 2.1 Actual serve args (from the container's own boot log)

```
vllm serve /model --served-model-name deepseek-v4.1-flash --host 0.0.0.0 --port 8000
  --tensor-parallel-size 2 --nnodes 2 --node-rank 0
  --prefix-cache-retention-interval 4096
  --distributed-executor-backend mp
  --tokenizer-mode deepseek_v41 --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41
  --enable-auto-tool-choice --enable-prefix-caching --enable-prompt-tokens-details
  --no-enable-flashinfer-autotune
  --hf-overrides {"engram_table_dir":"/engram-src"}
  --quantization exl3 --max-model-len 393216 --gpu-memory-utilization 0.88
  --max-num-seqs 2 --max-num-batched-tokens 1536 --long-prefill-token-threshold 1280
  --kv-cache-memory-bytes 4294967296 --block-size 64
  --speculative-config {"method":"dspark","num_speculative_tokens":3}
  --chat-template /opt/dsv41/chat_template.jinja --language-model-only
  --cudagraph-capture-sizes 1 2 3 4 6 8 12 18 24
```

Absent from these args: any `--decode-context-parallel-size` / `dcp` flag, any FP4-KV flag, any
`--kv-cache-dtype` (correct for this kit — the image picks the layout itself).

### 2.2 The pool number

```
(Worker_TP0 pid=428) Initial free memory 112.96 GiB, reserved 4.0 GiB memory for KV Cache as
  specified by kv_cache_memory_bytes config and skipped memory profiling. This does not respect
  the gpu_memory_utilization config.
(EngineCore pid=361) GPU KV cache size: 1,223,846 tokens,
  Maximum concurrency for 393,216 tokens per request: 3.11x
```

**1.22M aggregate, 3.11× at 393K — not 3.1M.** Note the pin also disables the
`gpu_memory_utilization 0.88` path entirely, so raising `GPU_MEM_UTIL` here would do nothing.

### 2.3 Artifact-level absence

- Container `/opt/dsv41/` (31 entries) carries the EXL3/Engram/binding patches — `exl3.py`,
  `engram_file_backend.py`, `patch_engram_file.py`, `patch_exl3_ext_aarch64.py`,
  `patch_sm120_block64.py`, `patch_scheduler_decode_floor.py`, `patch_spinwait.py`, etc.
  **No `display_kv.py`, no `libds41_display_kv.so`.** **[verified: `find` in container]**
- `/dev/dri/card0` *is* present in the container (`crw------- root video`), so the device is not the
  blocker — nothing in the image opens it.
- The stock allocator is unmodified. `/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/utils.py:429`
  still reads `buf = torch.zeros(buf_size, dtype=torch.int8, device=device)` — exactly the line
  coolbo's `display_kv.py` rewrites via `_compile(...)`. No `display` string anywhere in
  `v1/worker/`. **[verified]**
- Grep across our trees (`dsparkx2`, `dsv41-flash-exl3-2x-spark`, `vllm-exl3-plugin`,
  `DeepSeek-V4-Flash-Dual-DGX-Spark-1M-Context`, `DeepSeek-V4-Flash-Vision-2x-DGX-Spark`,
  both GLM EXL3 repos): **zero hits** for `display_kv`, `ds41_display`, `CREATE_DUMB`, `fbdev=0`.
  **[verified]**

### 2.4 Host state is the *opposite* of the prerequisite

| Check | Required by the recipe | Ours (gb10) |
|---|---|---|
| `systemctl get-default` | `multi-user.target` | **`graphical.target`** |
| `/proc/cmdline` | `nvidia-drm.modeset=1 nvidia-drm.fbdev=0` | `nvidia-drm.modeset=1 nvidia-drm.**fbdev=1**` |
| Uptime | — | 43 days |
| Desktop | none / headless | active session, 6 users |

`modeset=1` is already correct. `fbdev=1` is not, and the boot target is not headless. **[verified]**

---

## 3. Where the 3M KV actually comes from

Three separate things, and only the first is "the freed-up RAM".

### 3.1 The display reservation (a fixed 1.75 GiB ceiling, not RAM freed)

The GB10 firmware carves ~2 GiB for a screen. A headless Spark never uses it and `MemAvailable`
never counts it. The unlock does **not** return it to ordinary RAM — it allocates a DRM dumb buffer
*inside* the reservation and registers that buffer for CUDA:

`release/runtime/sources/display_kv.c` **[verified: upstream source]**:
- rejects anything other than ≤1 GiB ordinary + **exactly 1.75 GiB (1792 MiB)** display
- `open("/dev/dri/card0", O_RDWR|O_CLOEXEC)` → `DRM_IOCTL_MODE_CREATE_DUMB` (4096 × 114688 × 32bpp
  = 1792 MiB) → `DRM_IOCTL_MODE_MAP_DUMB` → `mmap(MAP_FIXED|MAP_SHARED)`
- `cuMemHostRegister(..., CU_MEMHOSTREGISTER_DEVICEMAP | CU_MEMHOSTREGISTER_IOMEMORY)` → device
  pointer; a mismatch between the host pointer and the returned device pointer is a hard failure
- owns the span for process lifetime; releases only via `ds41_display_destroy` (graphs/tensors first)

`release/runtime/serving/ds41/display_kv.py` **[verified]**:
- patches `allocate_kv_cache` in **two** modules (`vllm.v1.worker.utils`,
  `vllm.v1.worker.gpu.attn_utils`), refusing to run unless both files match pinned SHA-256 hashes —
  i.e. it is coupled to an exact vLLM build
- budgets: `credited_budgets()` requires **zero ordinary RAM** for the pool and adds
  `DISPLAY_BYTES` per rank; it raises if given positive ordinary budgets
- complains if MemAvailable drops more than `ordinary + 128 MiB` during allocation, and fails if
  it ends below 512 MiB

Upstream README, verbatim: *"An experimental display-buffer allocator supplies 1.75 GiB (1792 MiB)
of KV backing per GPU from the display reservation. The nominal remaining 256 MiB is not extra
host-RAM safety margin."* And: *"It does not increase physical RAM or turn 2 GiB into ordinary CUDA
memory."* Bandwidth note: *"Its raw read bandwidth measured below ordinary CUDA-backed RAM in our
probes."*

### 3.2 What 1.75 GiB buys at our current KV cost

Our measured cost (4 GiB pin → 1,223,846 tokens): **3,510 bytes/token** (`4096 MiB ÷ 1,223,846`,
≈3.43 KiB/token). **[verified arithmetic]**

```
1792 MiB = 1,879,048,192 B  ÷  3,510 B/token  ≈  535,000 tokens   (~535k)
```

So moving the whole pool into the display region at *today's* format buys roughly **535k tokens** —
about **17%** of the 3.15M target, and less than our current 1.22M. Cross-check against a sibling
data point: MiaAI's GLM TensorFold README states `DISPLAY_KV_MIB=1792` *"adds ~277k tokens at
PARALLEL=8"* — same order of magnitude, different model and KV layout. **[verified: upstream README]**

### 3.3 The part that isn't memory at all

Reaching 3.15M requires the **KV format** change, which is why the issue bundles it:

- **FP4 main KV** (E2M1/E4M3, searched NVFP4 scales) — versus our `fp8_ds_mla`
- **NVFP4 indexer keys**
- **FP8 sliding-window KV** (FP8 per 32 non-RoPE values, BF16 RoPE preserved)
- **DCP2** — decode context parallel across the two ranks, so the pool spans both GPUs

Upstream's own framing **[verified: README/docs]**:

| Statement | Source |
|---|---|
| "The six-session profile supports up to 1,048,576 tokens per request; historical testing reached about **3.15M input tokens in aggregate**, not 6M" | README |
| "This is aggregate capacity, **not six full million-token contexts**" | `docs/display-memory.md` |
| "completed six simultaneous independent contexts containing 3,146,968 input tokens (3,161,062 including generated)" | `docs/display-memory.md` |
| The 3.15M test used the *preceding* runtime and has **not** been repeated with rc2's kernels | README |
| "Release candidate, **not yet clean-install-qualified**"; the fresh-clone launcher has CPU-only validation | README |

**Summary of the attribution:** display region = a fixed **1.75 GiB/GPU** ceiling, no RAM increase;
the 3M figure = that region **plus** FP4/SWA KV packing **plus** DCP2, measured as aggregate input
across six sessions at a 1,048,576 per-request limit. It is not a single setting. **[verified]**

---

## 4. What it would cost to adopt here

### 4.1 Host (both Sparks) — manual, one-time, and it takes the desktop down

```bash
sudo systemctl set-default multi-user.target
sudoedit /etc/modprobe.d/ds41-display-kv.conf      # options nvidia_drm modeset=1 fbdev=0
sudo update-initramfs -u -k all
# reboot both idle machines; then verify:
sudo cat /sys/module/nvidia_drm/parameters/modeset   # Y
sudo cat /sys/module/nvidia_drm/parameters/fbdev     # N
ls -l /dev/dri/card*
```

Constraints **[verified: upstream docs]**:
- UEFI/BIOS display reservation **stays at 2 GB**; zero removes the region this uses
- no monitor / no desktop on either node; the GLM port refuses the setting while any `card0` output
  reports `connected`
- driver: 580.173.02 baseline; 595.84 passes on an NVIDIA-branded Spark but **fails** on at least one
  ASUS GX10 at `register display IO: CUDA_ERROR_INVALID_VALUE` (issue #3 there). Our gb10 runs
  6.17.0-1029-nvidia; the upstream notes above were measured on that same kernel
- the container needs the DRM card mapped (upstream: map the selected card to `/dev/dri/card0`)

**gb10 currently runs a desktop** (`graphical.target`, 6 sessions, 43 days uptime). This is the
substantive cost — not the flags.

### 4.2 Our kit config

`.env` today vs. what the recipe needs:

```diff
--- .env (live values, 2026-10-09)
+++ .env (display-KV + FP4/DCP2 profile)
-GPU_MEM_UTIL=0.88
-KV_CACHE_MEMORY_BYTES=4294967296
+GPU_MEM_UTIL=0.88                    # may be ignored once a display pool is pinned
+KV_CACHE_MEMORY_BYTES=0              # credited_budgets() requires ZERO ordinary KV
 KV_BLOCK_SIZE=64
 MAX_MODEL_LEN=393216
 MAX_NUM_SEQS=2
 MAX_NUM_BATCHED_TOKENS=1536
 SPEC_METHOD=dspark
 DSPARK_TOKENS=3
+# --- new, not present in our .env ---
+DCP=2                    # decode context parallel (pool spans both ranks)
+FP4_KV_MODE=nvfp4_4over6 # FP4 main KV writer; 'legacy' = earlier writer
+SWA_KV_GROUP_SIZE=32     # FP8 sliding-window KV grouping
+DISPLAY_KV_MIB=1792      # GLM-kit name; the DS4.1 allocator hardcodes 1792
+HEAD_DRM_CARD=card0
+WORKER_DRM_CARD=card0
```

**Caveat:** the DS4.1 fork does *not* expose `DISPLAY_KV_MIB` — its C allocator hard-requires exactly
1.75 GiB. Those names come from the GLM TensorFold port. Treat the DS4.1 upstream as the source of
truth for the flag surface. **[verified: source read; inferred: naming]**

### 4.3 start.sh (this repo) — one device is genuinely missing

```diff
--- a/start.sh
+++ b/start.sh
@@
 # Explicit NVIDIA nodes next to every `--gpus all`. Without CDI the legacy hook
 GPU_DEVICE_ARGS="--device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools"
+GPU_DEVICE_ARGS="${GPU_DEVICE_ARGS} --device /dev/dri/${HEAD_DRM_CARD:-card0}"
```

Every container run in `start.sh` (head 1652, worker 1611, the two `--rm` probes at 976/1071/1084)
passes `--gpus all ${GPU_DEVICE_ARGS}` and **does not** map `/dev/dri/card0`. Under CDI the card may
arrive anyway, but CDI is not what this kit relies on — commit `949c96a` documents the legacy-hook
path on gb10 and adds explicit NVIDIA nodes for exactly that reason. Mapping the card explicitly keeps
both paths honest. **[verified: read from start.sh; the card mapping is not applied]**

### 4.4 The decisive constraint: the runtime

The upstream display-KV path is not a flag layered onto our image. `display_kv.py` **hard-pins
SHA-256 hashes of two installed vLLM files** and refuses to patch if they differ, and the FP4/SWA/DCP2
KV layouts live in that fork's kernels. Adopting it means running
`ghcr.io/coolbho3k/deepseek-v4.1-flash-exl3-3bpw-2x-dgx-spark` (pinned by digest in their
`recipe-lock.json`) on their 3 bpw checkpoint, not overlaying it on MiaAI 2.9 bpw. Our kit's own
overlay is hash-pinned to *our* image. **[verified: source read]**

---

## 5. Risks and open questions

- **Read bandwidth.** Upstream states the display region reads *below* ordinary CUDA RAM in their
  probes. No decode/prefill impact number is given for DS4.1. **[open]**
- **Driver fragility.** The birth of this technique is a `cuMemHostRegister` failure on 595.84 on one
  vendor's box; cause still unestablished upstream. Our hosts would need a pinned driver. **[open]**
- **The requantization cost is not just memory.** 3 bpw experts change quality; upstream reports KLD
  0.04137 vs 0.04475 for its own target and a 93.63% top-choice agreement — teacher-forced, not a task
  benchmark. Our current kit is a *different* codebook (2.9 bpw mul1). Any comparison needs
  teacher-forced KLD, per this repo's own cook policy. **[verified: upstream numbers]**
- **The 3.15M figure is un-revalidated** on rc2's kernels, and the launcher's clean two-node GPU boot
  is deferred upstream. Treat 3.15M as an upper bound from an earlier runtime. **[verified]**
- **Desktop trade-off.** Headless on both Sparks removes a display path — relevant if gb10 is also used
  interactively. **[verified: host state]**
- **Does the MiaAI kit even accept the patch?** `display_kv.py`'s hash pins are for the coolbo fork's
  vLLM build. Whether they match the MiaAI 2.9 bpw image is unverified. **[open]**

---

## 6. Verification commands

```bash
# Our current pool (expect ~1,223,846 tokens at 393,216 max-len)
ssh gb10 'docker logs dsv41-exl3-head 2>&1 | grep -E "GPU KV cache size|reserved .* for KV"'

# Blend of args actually in use
ssh gb10 'docker logs dsv41-exl3-head 2>&1 | grep -m1 "non-default args"'

# Display-KV artifacts (expect nothing)
ssh gb10 'docker exec dsv41-exl3-head sh -c "find / -name \"*display_kv*\" -not -path \"/proc/*\" 2>/dev/null"'

# Stock allocator line the patch rewrites (expect torch.zeros)
ssh gb10 'docker exec dsv41-exl3-head sh -c "grep -n \"buf = torch.zeros\" /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/utils.py"'

# Host prerequisite state (expect graphical.target, fbdev=1)
ssh gb10 'systemctl get-default; cat /proc/cmdline | tr " " "\n" | grep nvidia-drm'
```

---

## 7. Sources

| What | Where | Access |
|---|---|---|
| Issue #23 body/labels/state, 2 comments | `gh api repos/MiaAI-Lab/…/issues/23` (+ `/comments`) | 2026-10-09 |
| Kit repo metadata (last push 09-19T13:38Z) | `gh api repos/MiaAI-Lab/…` | 2026-10-09 |
| coolbo allocator + PyTorch integration | `raw.githubusercontent.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark/878e0ee/release/runtime/{sources/display_kv.c,serving/ds41/display_kv.py}` | 2026-10-09 |
| coolbo README + `docs/display-memory.md` | same repo, `main` | 2026-10-09 |
| GLM TensorFold display-KV port | `gh api repos/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/contents/patches/0072-glm-display-kv.patch` | 2026-10-09 |
| Live serve args, KV pool, allocator, host state | gb10/kgb10 containers + host | 2026-10-09 |
