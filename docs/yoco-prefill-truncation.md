# Decoder-SWA bounded replay (YOCO prefill truncation): prepared, OFF by default

**Status 2026-10-20: prepared, not deployed.** The overlay, launcher wiring and CPU tests
are in this repo. Building or deploying the image and restarting the teacher are
operator-only; the maintenance-window plan is in §10. With `DSV41_YOCO_PREFILL` unset or
`off`, the launch is today's launch: the vLLM argv is byte-identical and the patch never
runs, so the image's vLLM files are untouched.

## 1. What and why

DeepSeek-V4.1-Flash is CED 20+20: a 20-layer causal encoder (layers 0-19) and a 20-layer
decoder (layers 20-39). Each decoder layer reads three things:

- the compressed KV (`compress_ratio` 1) that layer 20 projects from the stream leaving
  the encoder;
- the index keys of layer 20's indexer;
- its own sliding-window KV (window 128).

In prefill, a decoder layer is therefore only needed for its own SWA KV, and only decode
reads that KV. DeepInfer's `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact`
(deepseek-ai/DeepInfer @ 774ce24, Apache-2.0) runs layers 20-39 only on the last
`KEEP_TOKENS` positions of the sequence. The replayed queries attend through the SWA
only inside the replayed segment, i.e. to keys `max(N-KEEP, pos-127) .. pos`. The
result is approximate. DeepSeek runs this in production and says it "nearly halves
total prefill computation". Their QA doc says an exact rebuild would need 2,560 tokens
(20 layers x 128), so they replay 128 instead.

Today a cold prefill runs at about 909 tok/s at 380K (13f1924), and a cold 45K session
has about 57 s TTFT.

## 2. Step 0: the image does not truncate

The image vLLM is `0.1.dev20904+g179dd0fa9`. The live head runs local rebuild
`sha256:0aecf9f5…`. Its vLLM tree is identical to `sha256:4cdba4e9…` (repo digest
`2f0cf3ad…`): files were extracted read-only via `docker create` + `docker cp`, and
`diff -r` shows no difference.

- `models/deepseek_v4_1/nvidia/model.py` `DeepseekV4Model.forward` (~L639) runs every
  layer over every scheduled token. There is no yoco / fast-prefill / truncation code in
  `models/deepseek_v4_1`.
- vLLM's generic YOCO hook (`CacheConfig.kv_sharing_fast_prefill`, Gemma3n) exists in
  the image, but it cannot serve DS V4.1, for three independent reasons:
  - `config/vllm.py:1633` rejects it with any EAGLE-family speculator, and DSpark is one.
  - The V2 runner lists the flag as unsupported (`:2643`), and V1 rejects DSpark (`:2661`).
  - Its eligible layers are trailing `Attention` layers with a kv-sharing target, and
    `DeepseekV4Attention` has none.

  Its metadata transform would also be wrong here: it keeps one row per request, keeps
  the full slot mapping, and has no positions. So this port has its own overlay.

## 3. Topology in the shipped config (checkpoint `config.json`, `text_config`)

| | value |
|---|---|
| `compress_ratios` | `[0,0, 2 x18 (2-19), 1 x20 (20-39), 0,0,0 (DSpark 40-42)]` |
| `kv_source_layer_ids` | `[2, 8, 14, 20]` |
| `index_source_layer_ids` | `[2, 8, 14, 20, 24, 28, 32, 36]` |
| `candidate_source_layer_id` | 20 |
| `engram_layer_ids` | `[1, 14]` |
| `dspark_target_layer_ids` | `[37, 38, 39]` |
| `sliding_window` | 128 |

- The boundary is the first layer with compress ratio 1, as in DeepInfer's
  `hugging_face/config.py` `_n_prefill_layer`. That is **layer 20**, and the consumers
  are layers **20-39**.
- `dsv41_yoco.decoder_boundary()` refuses a checkpoint whose topology does not allow the
  split, in the cases DeepInfer guards (`deepseek.py:128-138`, `cross_layer.py:115`):
  - a kv source after the boundary;
  - a boundary that is not an index source;
  - an encoder-side candidate source;
  - an Engram layer at or after the boundary;
  - a DSpark target at or before the boundary.

Per layer, under truncation, the image code splits as follows:

| layer | all scheduled rows (full metadata) | kept rows only (compact metadata) |
|---|---|---|
| 0-19 | everything | — |
| 20 (kv + index-K + candidate source) | HC post/pre + attention norm; compressor scores (EXL3 or dense GEMM), `compressor()` into compressed KV, `insert_cache`, `indexer._produce_k` into index K | `fused_wqa_wkv`, q, own SWA insert, indexer q / weights, top-k, candidate write, sparse attention, o-proj, MoE |
| 24/28/32/36 (index sources, no K) | — | own indexer q / weights over layer 20's index K, candidate mask, attention, MoE |
| 21-39 other | — | reuse the nearest top-k, own SWA, attention, MoE |

The decoder's compressed and index keys come only from layer 20. DeepInfer's
`FlopsEstimator` agrees: "Packed KV and KProject always consume the full input".

## 4. Semantics (DeepInfer parity)

Let N be `prefill_len`: the prompt, plus generated tokens for a request resumed after
preemption. That matches DeepInfer's `seq_data->get_len()`. The rectangle is
`[N-KEEP, N)`, anchored to the whole sequence and not to the chunk
(`prepare_tensor_input_v4.cpp:222`).

- **Kept rows per step.** In each step, a prefilling request with N > KEEP keeps its rows
  with `pos >= N-KEEP`. When the final chunk is shorter than KEEP, the rectangle spans
  the last two chunks. The earlier chunk then writes the decoder SWA that the later one
  reads.
- **Placeholder row.** A chunk wholly before the rectangle keeps exactly one placeholder
  row (its last row, which is the logits row). The placeholder has slot −1, so it stores
  nothing in SWA and attends nothing. This keeps every request alive in the decoder
  metadata, so block tables and per-request fields need no renumbering. The placeholder's
  output feeds only its own discarded sample.
- **Rows that are always kept.** Decode and spec-verify rows, and prompts with N <= KEEP,
  keep every row. Kept rows are always a contiguous suffix of the request's scheduled
  rows. Positions stay correct because `seq_lens` is unchanged: every kernel recovers a
  position as `seq_len − q_len + offset`.
- **Visible SWA keys** for a kept row are `[max(pos−127, N−KEEP, 0), pos]`. The image
  kernel writes `[max(pos−127,0), pos]`, and the overlay clamps it afterwards
  (`clamp_swa_rows`). Decode is unchanged. Its window `[n−127, n]` with n ≥ N lies
  entirely inside written SWA because KEEP ≥ 128 + k + 1.
- **KEEP floor.** KEEP ≥ window + max backtracking (DeepInfer `model_runner.py:107`,
  B = MTP + 1). Here B = `DSPARK_TOKENS + 1`, so KEEP ≥ 132 at k = 3. The default is 140,
  as in DeepInfer, which leaves margin up to k = 11. start.sh refuses smaller values,
  and the model state re-checks against the checkpoint's `sliding_window` and the
  speculative config at load time.
- **prompt_logprobs is refused** with HTTP 400 (`InputProcessor._validate_params`).
  Earlier prompt positions never reach the LM head. DeepInfer forces it off
  (`llm_engine.py:529`). Completions `echo` + `logprobs` maps to `prompt_logprobs`, so it
  is refused too. No repl client sends either.

## 5. Prefix caching, retention, and the 128-token hit alignment

- **Decoder SWA cannot get its own cache group.** The 43 SWA caches pack into 15
  KV-cache groups `{i, i+15, i+30}`. Every SWA group mixes encoder and decoder (or draft)
  layers. One block ID per group spans all three layers' tensors, and the scheduler
  merges the 15 specs into one lookup group. Decoder SWA therefore cannot be excluded
  per group without a new KV spec (Phase 2 below).
- **Cached blocks carry garbage decoder SWA.** Under truncation, a cached block holds
  valid decoder SWA only where its writer computed it: the writer's own rectangle, plus
  its decode region.
- **The rule: never let a hit reach into the reader's rectangle.**
  `get_computed_blocks` caps `max_cache_hit_length` at `max(0, num_tokens − KEEP)`; the
  stock cap is `num_tokens − 1`. The whole rectangle is then recomputed by this request,
  and every decoder-SWA key a kept row reads was written by this prefill.
  `test_multichunk_never_reads_unwritten` checks this over 400 random chunkings and hits.
  - Encoder SWA, compressed KV and index-K hits stay exact, and outputs do not depend on
    cache state.
  - Garbage decoder pages in cached blocks become dead data.
  - Hashing, retention and the EAGLE checks are unchanged.
  - DeepInfer does the same: its QA doc caps hits at L − 134, and the engine never caches
    the "dropped region".
- **Alignment and peek.** With 128-token alignment and the 64-token EAGLE peek block, a
  hit B needs `B + 64 <= N − KEEP`. A hit therefore recomputes 204-331 tokens of the
  prompt (KEEP + 64 + up to 127 of alignment), versus 65-192 today. That is about 140
  extra encoder tokens per hit, roughly 0.1-0.2 s at long context.
- **Retention (`--prefix-cache-retention-interval 4096`).** The periodic checkpoints are
  unchanged. A tail is still kept at `num_prompt_tokens − 1` (the replay boundary).
  `cache_blocks` adds a second boundary at `num_prompt_tokens − KEEP − 64` (64 = the peek
  block). An exact retry or fork of the same prompt then hits at
  `align_down(N − KEEP − 64, 128)` instead of falling back to the last 4096 checkpoint.
  For example:
  - N = 34357: the retry hits 34048 (recompute 309). Today it hits 32768 (recompute 1589),
    because the replay boundary at 34304 cannot fit its peek block.
  - N = 8193: the retry hits 7936.
  - The extra checkpoint is about 3 blocks x 15 SWA groups per request. It is evictable
    and lives in the pinned pool.
- **Follow-up agent turns.** A turn that extends the previous request by S tokens is
  unaffected while S ≥ KEEP + (prompt end − replay checkpoint). Otherwise it falls back
  to the next checkpoint at or below `N2 − KEEP − 64`.
- **Phase 2 (not done).** Move decoder SWA (20-39) and draft SWA (40-42) into their own
  non-prefix-cacheable spec, KpoolTail-style. That would shrink each retained checkpoint
  about 3x and remove the garbage pages. It is a KV-layout change and needs its own
  trial.

**Coordination with the final-window overlay (2026-10-21).** "Keep final SWA window at
request end" landed as `overlay/patch_swa_final_window.py` (`DSV41_SWA_FINAL_WINDOW=keep`,
off by default; docs/swa-final-window-retention.md).

- It does not use `reachable_boundaries`. It holds the newest aligned window in
  `SlidingWindowManager.__init__` and hashes its tail in `KVCacheManager.free` /
  `pop_blocks_for_free`. None of this patch's anchors are touched, and either patch may run
  first. start.sh runs it before this one.
- Under compact it also keeps `align128(T − KEEP − 64)` at the request end, as asked here.
  That checkpoint is usable by any continuation, because the request-end checkpoint needs
  the next turn to add about 12-140 tokens under the `N − KEEP` cap.
- This patch's anchor is still the two lines starting
  `reachable_boundaries = [request.num_prompt_tokens - 1]`, and drift fails it loudly.

## 6. DSpark

- **What the draft reads.** DSpark reads the attention inputs of target layers 37-39
  (`aux_hidden_state_layers`, captured after loop indices 36-38). All of them are decoder
  layers.
- **`propose()` runs every step, including non-final chunks.** It projects every target
  row through `combine_hidden_states`. `precompute_and_store_context_kv` writes the
  draft's SWA context KV (layers 40-42, ratio 0) for those rows.
- **Draft window.** The draft block is non-causal over `[anchor−128, anchor)` plus the
  block itself (`sparse_swa.py:1189`). Unchanged first-step drafts therefore need correct
  aux states only for `[N−128, N)`, and the target's first verify step needs `[N−127, N)`.
  Both lie inside the rectangle.
- **What the patch passes the draft.** The aux states, and the final hidden states the LM
  head reads, are scattered back to full length. Rows dropped by the decoder carry zeros.
  The draft therefore writes zero-derived context KV only at positions below N − KEEP,
  outside every later window. Drafts made on non-final chunks are discarded, as today.
- **Backtracking.** A verify step rolls `num_computed_tokens` back by at most k = 3 and
  never below N. The KEEP floor carries DeepInfer's B = k + 1 for parity.
- **DeepInfer's MTP + YOCO.** `--mtp N` needs a DFlash + YoCo checkpoint. DFlash's main
  cache is written from the compact rows of target layers 37-39 with compact store
  slots, and its rings live in the never-persisted dropped region. DeepInfer's L2 also
  compares the DFlash confidence bitwise. The draft only needs the last W target
  features before decode, which K ≥ W + B guarantees. B = max(env, MTP + 1) = 6 at
  MTP 5, so 134 is the floor and 140 the default. The extra 6 is undocumented.

## 7. Engram, CUDA graphs, memory, EXL3 MoE paths

- **Engram (layers 1, 14) is encoder-only.** Hashes are computed once over all tokens
  before the layer loop, from layer 0's SWA metadata. Neither layer 0's metadata entry
  nor its builder buffers are touched: the compact metadata comes from separate builders.
- **CUDA graphs.**
  - The live capture sizes 1 2 3 4 6 8 12 18 24 (`ENFORCE_EAGER=0`) serve uniform decode
    as FULL graphs, and mixed steps of 24 tokens or fewer as breakable PIECEWISE.
  - The model is not torch.compiled (`DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES` sets
    compilation mode NONE).
  - Truncation lives only in the eager forward. Any step that contains a prefill longer
    than KEEP is forced eager (`need_eager`). Otherwise a ≤ 24-token chunk inside the
    rectangle could replay without the SWA bound and read unwritten decoder SWA.
  - Decode graphs and capture are unchanged: no plan is made for FULL, PIECEWISE,
    capture or profile runs.
- **Memory.**
  - The compact decoder metadata needs a second set of metadata builders, so the full
    step metadata (encoder layers) is never aliased. That is 15 SWA builders (about
    1.6 MiB each) plus layer 20's indexer builder (about 38 MiB, mostly
    `expanded_block_table_buffer` of 1536 x 6144 int32). Total: about 65 MiB per rank,
    allocated on the first truncated step (the boot shape warmup) and logged as
    `[dsv41-yoco] created N shadow metadata builders (+X MiB)`.
  - The KV pool is unchanged (pinned at 4 GiB).
  - Decoder activations shrink: rows ≤ KEEP + 1 per prefilling request plus decode rows,
    instead of up to 1536. So does the ratio-1 indexer's logits work, which at 380K is
    the largest prefill buffer. The head MemAvailable floor (2.51 GiB) is expected to
    hold or improve. It must be measured (§10).
- **EXL3 MoE paths.** The decoder MoE now sees about 1-145 rows per prefill step instead
  of up to 1536 (~0.4-2.3 rows per expert instead of ~24). With
  `EXL3_TEMP_ROWS_FUSED=16`, nearly every decoder expert stays on the fused `exl3_moe`
  kernel, which uses the cooperative launch in the live overlay. The E3 grouped fat
  kernels see few or no decoder calls. These shapes are already exercised: fused handles
  the ≤ 16-row experts of every prefill chunk today, and decode is M = 4-8. The encoder
  half is unchanged. The first truncated prefill may JIT a new Triton/TileLang shape
  once (watch `logs/head.log`).

## 8. Implementation

| file | role |
|---|---|
| `overlay/dsv41_yoco.py` | the logic: settings, `build_keep_plan`, `clamp_swa_rows`, topology check, hit math, `YocoPrefill` driver (shadow builders, `attach`), `StepPlan` (split / gather / scatter / finish). DeepInfer attribution in the header |
| `overlay/patch_yoco_prefill.py` | 14 anchored edits in 7 image files, plus loader footers. Applies all or nothing, is idempotent, and fails loudly (exit 1, nothing written) on any anchor drift. `--check` only verifies |
| `start.sh` | `DSV41_YOCO_PREFILL=off\|compact` (default off) and `DSV41_YOCO_KEEP_TOKENS` (default 140): validated, passed to both containers, helper and patch mounted on both ranks. The patch runs after every other patch only when not off, and the container exits if it fails. The TP ranks must agree |
| `Dockerfile` | copies both files, runs `patch_yoco_prefill.py --check` against the image's vLLM after the build-time patches, runs `test_yoco_metadata.py` |

Edits (each anchor matches exactly once in the image):

| file | what changes |
|---|---|
| `nvidia/model.py` | decoder-layer hook between the attention pre-mix and attention; plan begin/after_split in the loop; scatter of `_mtp_hidden_buffer`; `finish` (scatter final and aux hidden states, restore metadata) |
| `attention.py` | the boundary layer's compact call skips compressor scores, `compressor()`, `insert_cache` and index-K (the latent stays None, so `_produce_k` returns early). The kv-score GEMM it mirrors is pinned by an anchor |
| `nvidia/model_state.py` | builds the driver at load (topology/KEEP checks); `prepare_attn` attaches the plan and compact decoder metadata on eager prefill steps |
| `v1/core/kv_cache_manager.py` | hit cap `num_tokens − KEEP` |
| `v1/core/single_type_kv_cache_manager.py` | retention boundary `num_prompt_tokens − KEEP − peek` |
| `v1/engine/input_processor.py` | `prompt_logprobs` refused |
| `v1/worker/gpu/model_runner.py` | `need_eager` for steps with a prefill longer than KEEP |

The decoder metadata is the full step dict with the 20 decoder SWA entries and layer 20's
`indexer.k_cache` entry replaced. Those entries are built by the shadow builders from a
compact `CommonAttentionMetadata`:
- changed: `query_start_loc`, `num_tokens`, `max_query_len`, positions, and per-group
  slot mappings, with −1 on placeholder rows;
- unchanged: `seq_lens`, block tables, `is_prefilling`.

Layer 20's compressed-cache entry (block table only, for the decoder's `forward_mqa`)
stays full. The plan travels with the step's metadata dict, a `dict` subclass carrying
`.dsv41_yoco`, so nothing is global.

Tests (CPU, `python3 tests/<file>`):
- `test_yoco_metadata.py` is the L1 port of DeepInfer's mask-compact check. It contains:
  - a Python port of the image's `_compute_swa_indices_and_lens_kernel`;
  - DeepInfer's 10-case matrix at chunk sizes 1280, 1536 and 2048;
  - vLLM-specific batches (decode + prefill, tiny chunks, resumed requests, block edges);
  - 300 random batches;
  - mask == compact;
  - the multi-chunk "never reads unwritten decoder SWA" simulation over random hits;
  - hit / retention arithmetic;
  - a toy attention through the clamped paged indices with NaN-poisoned unwritten slots
    (compact == mask, bitwise).
- `test_yoco_driver.py` executes every line of the driver against fake vLLM modules.
- `test_yoco_prefill_patch.py` checks the anchors against the image fixtures
  (`tests/fixtures/dsv41_yoco`): applies cleanly, compiles, idempotent, atomic on drift,
  `--check`, no-op when off, wiring.
- `test_yoco_launcher.py` covers validation, byte-identical argv with the feature unset,
  off or compact on both ranks, the inner guard, and propagation.
- `test_yoco_logprob_ab.py` covers the A/B script offline.

Mutation runs: 32 of 32 mutants were killed (11 helper, 11 driver, 10 launcher / patch /
Dockerfile).

## 9. Expected effect (model, not measured)

Per token at context L, compute splits roughly as follows:

- dense + MoE: about 0.8 GFLOP per layer;
- sparse attention: about 0.08 GFLOP per layer;
- indexer logits: L x 32 x 128 x 2 per index-source layer at ratio 1, half that at
  ratio 2.

The decoder has 5 ratio-1 index sources, and the encoder has 3 ratio-2 ones. So the
decoder holds about 52% of prefill FLOPs at 45K context and about 61% at 380K. If time
followed FLOPs, a cold prefill would speed up about 2x at 45K and about 2.5x at 380K.

EXL3 dequantisation, memory-bound small-M MoE, eager overheads, and the encoder's own
indexer will make the real gain smaller. A plausible guess is a 45K TTFT of about 30-35 s
(from 57 s) and 380K at about 1.4-2k tok/s (from 909). The trial measures it.

## 10. Maintenance-window validation plan (operator only; not run)

0. **Deploy.**
   - On rai: run all `tests/test_yoco_*.py` and `bash -n start.sh`.
   - The live launcher is the non-git kit `gb10:/mnt/storage1/dsv41/exl3-kit`. Copy
     `overlay/patch_yoco_prefill.py` and `overlay/dsv41_yoco.py`, and **port** the
     start.sh hunks: the kit's start.sh has diverged, so never overwrite it, and never
     copy the rai `.env`.
   - Restart with `SKIP_BUILD=1 SKIP_PULL=1`. The recipe stamp moves; the runtime mounts
     carry the change.
   - start.sh pauses and resumes the watchdog itself. Boot takes about 12-17 min.
1. **Baseline on today's config (feature off).** This must use the current 1017 weights,
   because the 909 tok/s / 57 s figures predate them.
   - `scripts/yoco_logprob_ab.py capture --arm off-a` then `--arm off-b --trajectory
     RUN/off-a.json`. Use `--context-file` = the needle haystack corpus, lengths 8k / 45k /
     128k.
   - `scripts/chunk_validate.py kvprog --tokens 45000`, then 128000 and 380000: prefill
     tok/s and TTFT. Sample the head MemAvailable floor during the 380K run (`dsv41-mem`
     log lines, or `free -m` every 1 s on gb10).
   - `scripts/chunk_validate.py retention --lengths 8193 34357`.
   - `scripts/spec_bench.py --label off` (acceptance length).
   - `scripts/needle_layers.py` at 8k-300k (single and two-hop).
2. **Enable.** Set `DSV41_YOCO_PREFILL=compact` in the kit `.env`, leave
   `DSV41_YOCO_KEEP_TOKENS` unset (140), and run
   `SKIP_BUILD=1 SKIP_PULL=1 ./start.sh restart`. Both `logs/head.log` and `logs/worker.log`
   must show:
   - `YOCO prefill patch: applied` for all 7 files;
   - `[dsv41-yoco] compact prefill truncation: keep=140 decoder layers 20..39`;
   - after warmup, `created … shadow metadata builders (+~65 MiB)`.

   Any `FATAL: YOCO prefill patch failed` means the container stopped. Roll back (step 5).
3. **Smoke.**
   - `bash tests/test_smoke.sh` (17*19 → 323).
   - A completions request with `prompt_logprobs: 1` must return HTTP 400.
   - A 100-token prompt (N ≤ KEEP, exact path) must answer normally.
4. **Measure (feature on)**, against step 1:
   - `yoco_logprob_ab.py capture --arm yoco --trajectory RUN/off-a.json`, then
     `compare RUN/off-a.json RUN/off-b.json RUN/yoco.json`. This is the teacher-forced
     logprob / top-K comparison, DeepInfer's L2 adapted. Pass: no FLAG. Decode
     first-divergence is reported but not gated: greedy trajectories diverge at near-ties.
   - `kvprog` at 45k / 128k / 380K: expect ≥ 1.3x prefill tok/s at 45k as the go
     threshold; §9 gives the model. Compare the MemAvailable floor at 380K with the
     baseline (≥ baseline − 0.2 GiB).
   - `retention --lengths 8193 34357`: repeats should report **7936 and 34048** cached
     tokens, from the new boundary at `N − 140 − 64`. Compare with the step-1 baseline;
     34357 recovered 32768 on 2026-09-18.
   - `spec_bench.py --label yoco`: mean acceptance length within the off run's
     run-to-run noise.
   - `needle_layers.py`, same lengths: single needles 25/25 and two-hop thinking-on 15/15,
     as on 2026-09-24. Any new miss is a stop.
   - Optional: the 1017 copy-defect battery and a small `deepinfer_bench.py` AIME slice,
     both within budget.
5. **Rollback.** Remove the `DSV41_YOCO_PREFILL` line or set it to `off`, then run
   `SKIP_BUILD=1 SKIP_PULL=1 ./start.sh restart`. The fresh containers carry unpatched
   vLLM files.

**Open risks the GPU trial must clear** (each was settled from source only):
- the C++ fused SWA insert treats slot −1 as padding, as cudagraph padding relies on;
- FlashInfer SM120 prefill accepts rows with a clamped `swa_len` of 1-127 next to full
  rows;
- the shadow indexer metadata's chunk split with compact queries;
- no new mid-serve JIT stalls.
