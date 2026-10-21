# Final sliding-window retention (a): prepared, OFF by default. Encoder SWA bounded replay (b): designed, not implemented

**Status 2026-10-21.**

- **(a)** Port of DeepInfer's decode-time SWA store, as `DSV41_SWA_FINAL_WINDOW=keep`. It is
  prepared: the overlay, launcher wiring and CPU tests are in this repo. It is exact.
- **(b)** Encoder SWA bounded replay. It is approximate. Its design and the reasons to defer
  it are in §7.

Building, deploying and restarting the teacher are operator-only. The measurement plan for
the maintenance window is in §9. With `DSV41_SWA_FINAL_WINDOW` unset or `off`:

- the vLLM argv is byte-identical;
- the patch never runs;
- the image's vLLM files are untouched.

## 1. The loss: an agent turn prefills its own previous output again

repl PROGRESS 2026-10-19, item #5, session `3b75b1f3` (10-18). Read-only, from the repl
request record with `scripts/swa_final_window_turns.py`:

| window | continuations (whole previous output in the next prompt) | own-output tokens | prefilled again |
|---|---:|---:|---:|
| session `3b75b1f3` | 216 | 153,039 | **114,667** |
| 10-14 → 10-21, all caller sessions | 7,242 | 4,856,860 | **3,984,653** |
| since the 1017 weights swap (10-18 12:26Z → 10-21) | 2,370 | 1,638,086 | **1,317,339** |

- In that session the hit lands at the previous prompt's tail in 187 steps (median −59).
  In the other 29 it lands on the highest 4096 multiple inside the previous output.
- Prefill runs at a median of 794 tok/s that week. So about 4.0 M tokens is roughly 84 minutes
  of teacher prefill per week, all of it on the agent's critical path.

**Mechanism** (image `2f0cf3ad`, vLLM `0.1.dev20904`; files in
`tests/fixtures/dsv41_yoco/vllm`):

- `SingleTypeKVCacheManager.cache_blocks` keeps sliding-window (SWA) tails at only three kinds
  of boundary:
  - the replay boundary `num_prompt_tokens − 1`;
  - a shared-prefix junction;
  - one tail per `PREFIX_CACHE_RETENTION_INTERVAL` (4096) segment.
- Decode blocks that leave the window are freed unhashed. `BlockPool.free_blocks` prepends
  them to the free queue, so they are reused first.
- At finish the request still holds only the last ~2 full window blocks.
- A hit at B under DSpark needs 3 contiguous cached blocks: the 2 blocks of the window
  [B−128, B) plus the 64-token peek block [B, B+64) that the EAGLE drop removes again. So no
  hit can land near the end of a decode.

**Fidelity check.** `tests/test_swa_final_window_sim.py` loads the image's own block pool,
hybrid coordinator, SWA manager and KV cache manager on CPU. It feeds them the record's
lengths for the session's first 40 turns, with DSpark k = 3 verify steps and random
acceptance. **All 39 recorded continuation hits come out exactly**, including the nine on
4096 checkpoints inside an output.

## 2. What DeepInfer does

Source: deepseek-ai/DeepInfer @ 774ce24, Apache-2.0, read-only. The QA doc
(`doc/inference_deployment_qa.md`, sections "SWA KV cache" and "KV cache hits and
recomputation") describes the policy. The open-source tree carries only the mechanism:

- `LLMEngine.should_copy_state` returns `False`.
- `do_save_kvcache` and the PE save hooks are empty "virtual functions"
  (`deepinfer/engine/llm_engine.py:809-823`).
- `copy_state` is never allocated.
- The hit length is injected from outside (`Sequence.num_cached_tokens` setter).

So this port follows the doc's policy and the code's data layout.

- **Slot layout.** Each sequence owns one state slot:
  `[stored: SWA of the 20 encoder layers | dropped: consumer (decoder) SWA, the 3 DFlash
  rings, C2 compression state]` (`cache_engine.py`). The ring holds W + B = 128 + 6
  positions (`get_inference_swa_block_size`).
- **Store positions.** Stores land only on 128 multiples (`KVSTORE_GRAIN`, asserted at
  `cache_engine.py:785`).
  - Prefill stores once, at the end, after reserving a token (`(L−1)//128·128`).
  - Decode copies the whole slot into a replica at each 128 crossing
    (`do_copy_state`, `llm_engine.py:790-798`; `last_copy_state_token_idx = num_cached_tokens`
    = computed tokens).
  - The finish-time store is taken from the newest copy. The ≤ 5-token overshoot lands in
    the 6 slack positions.
- **Replica memory.** `state_size *= 2  # state store & copy buffer` (`worker/base.py:243`).
  That is a second full slot per sequence: about 3.13 MB at V4.1-Flash dims.
- **Prefill-side extra.** `get_kvstore_swa_block_size` adds 128 + KEEP positions to the
  prefill ring. Its docstring says the reason: "prefill may not land on a 128 boundary;
  storing an extra segment guarantees multi-turn hits; with YoCo add the anchor back-off
  margin". §3's hold is the same idea in vLLM's paged layout.
- **Hit policy and bounded replay (b).**
  - The QA doc: the hit is decided by block KV alone, capped at L − 134. If no SWA state
    exists in [L1−128, L1), back off 128 and replay that segment with SWA keys clamped to
    `max(s, i−127)..i`; block KV is "reused as-is, neither recomputed nor overwritten".
  - The code: `swa_valid_start` → `prepare_tensor_input_v4.cpp:505-523`. The replayed rows'
    main KV, indexer K and YOCO full KV are written through the ordinary prefill slot
    mappings (`:46-69`). Block KV *is* recomputed and overwritten, in the prefill machine's
    private copy.

## 3. (a) in vLLM: hold the newest aligned window, hash it at the free

vLLM's SWA cache is paged and position-addressed, not a ring. So the replica becomes "do not
free those blocks yet", and the store becomes "register their hashes before the free".

**Hold.** `SlidingWindowManager.__init__` adds `hold` to `extra_retained_tokens`. That is the
engine's own knob for keeping blocks below the window (multi-module MTP uses it).
`remove_skipped_blocks` therefore keeps enough blocks for the newest checkpoint's tail to
still be allocated when the request is freed.

```
hold = (A + peek·bs − 1 + KEEP_yoco) − (W − 1) + ceil((W − 1)/bs)·bs
     = 192 tokens (3 blocks), YOCO off;   332 (6 blocks), YOCO compact (KEEP 140)
```

- A = 128 is the scheduler block and hit alignment, bs = 64 the SWA block, W = 128.
- peek = 1 is assumed: `use_eagle` is set only after the managers are built.
- The formula is the worst case over every residue of the computed count C mod 64 and of
  the alignment. `test_footer_math_against_brute_force` checks it over all C < 3000, and also
  checks that one block less would lose a tail.
- The per-request admission cap (`_max_admission_blocks_per_request`) grows by
  `cdiv(hold, bs)`. So `sum(reservations) ≤ pool` still bounds the real-held blocks (the
  #39734 deadlock guard). The simulation checks real-held ≤ cap at every allocation, through
  1536-token chunked prefill and decode.

**Hash at the free.** `KVCacheManager.free()` and `pop_blocks_for_free()` call the hook before
the coordinator frees. Those are the only two ways a request's blocks go back:

- finish (`_free_request`);
- abort (`finish_requests`);
- preemption (`_preempt_request` → `_free_request_blocks`, before `num_computed_tokens` is
  reset);
- the deferred-free path under async scheduling.

For each SWA group, the hook hashes the tail of:

- `B_end = align128(floor64(C) − 64)` — the newest checkpoint its computed blocks complete;
- under YOCO compact, also `B_yoco = align128(T − KEEP − 64)` (§5).

Each tail is the blocks [B−128, B+64). Only blocks without a hash are passed (masked
`BlockPool.cache_full_blocks`). A tail with a freed or null block is skipped, so partial tails
are never cached.

```
C = min(num_computed − num_in_flight, num_computed − num_output_placeholders, num_tokens)
    − num_reprefillable_tokens
```

- C counts processed tokens only.
- A step or chunk still in flight writes positions ≥ C, so it never touches a hashed block.
- The `num_tokens` cap leaves out positions after a stop inside accepted drafts.
- `num_reprefillable_tokens` is 0 for DSpark (multi-module MTP only).

The hashed blocks then go to the LRU end of the free queue, like any cached block.

**Exact.** Hit selection, block hashing, the retention interval, the shared-prefix junction
and the EAGLE peek / drop are unchanged. The checkpoint at the end of a decode is the same
kind of block the 4096 retention already caches inside decodes (the 29 hits above). B_end is
at or above every checkpoint the stock policy leaves for that request, so a hit is never
worse. The hook is fail-safe: an exception is logged, and the request is freed uncached as
stock.

**Cost.**

- At most 3 hashed blocks per SWA group per freed request (6 under YOCO). That is ≤ 45 / 90
  block ids for the 15 groups.
- No dense decode caching: blocks inside a decode are still freed unhashed, which the tests
  pin.
- CPU: one masked `cache_full_blocks` call per SWA group per free.

**Never at the very end.** The residue is 64-192 tokens below T:

- the 64-token peek block, which a DSpark hit must match and then recompute;
- up to 127 tokens of 128 alignment.

On session `3b75b1f3` this is 4,639 tokens over 39 turns, against 19,099 before. DeepInfer
pays a similar residue: 128 granularity and one reserved token.

## 4. EAGLE/DSpark peek block and 128 alignment

The stock behaviour (docs/prefix-cache-retention.md):

- hits are 128-aligned;
- the DSpark/EAGLE group needs the block after the hit;
- the scheduler merges the 15 identical SWA specs into one lookup group, so `use_eagle`
  applies to all of them.

(a) works inside that rule and does not relax it. B is a 128 multiple whose peek block
[B, B+64) is full and below C. The token after B is therefore known, so the draft KV at B−1
(written with token B) matches what any reader with the same prefix needs. That is the
condition the drop protects.

It also removes the "one-output-token cliff". A request that ends right after prefill
(`max_tokens` 1) still leaves a usable checkpoint, at the B_end of its prompt.

## 5. The mixed encoder/decoder SWA groups (292d4e2) and YOCO compact

The 15 SWA KV groups `{i, i+15, i+30}` mix encoder, decoder and draft layers. One block id
spans all three layers' pages.

- **YOCO off.** Every page of a decode-region block was written by a full forward. A
  checkpoint's decoder and draft SWA is valid.
- **YOCO compact.**
  - Below the writer's own rectangle [N−KEEP, N), the decoder pages are garbage.
  - YOCO's rule makes that harmless: hits are capped at `N − KEEP`, and each reader
    recomputes its own rectangle. A reader never reads a cached decoder page.
  - That cap is why one checkpoint is not enough. A continuation of N tokens can use a
    checkpoint B only if B + 64 ≤ N − KEEP.
  - B_end needs the next turn to add about 12-140 tokens. B_yoco = align128(T − KEEP − 64)
    is usable by any continuation (N ≥ T). It mirrors the `num_prompt_tokens − KEEP − 64`
    boundary that YOCO adds at the prompt end, as the YOCO doc's coordination note asked.
  - On session `3b75b1f3` (both patches): 4,821 tokens over 39 turns, with every hit within
    YOCO's cap.
- **Coexistence.**
  - This patch does not touch YOCO's anchors (`max_cache_hit_length = request.num_tokens - 1`
    and `reachable_boundaries = [request.num_prompt_tokens - 1]` / `if
    request.shared_prefix_boundary:`). Its own anchors do not overlap YOCO's.
  - Both apply in either order, and each one's anchors still match exactly once after the
    other (`test_coexists_with_yoco_in_either_order`). start.sh runs this patch before YOCO,
    which still runs after every other patch.
  - If YOCO's anchor is rebased later, this patch needs no change.

## 6. Memory and the C2 boundary state

**Memory.**

- There is no replica copy.
- The hold keeps ≤ 3 (YOCO: 6) extra blocks per SWA group per *running* request inside the
  pinned KV pool. With `MAX_NUM_SEQS=2` that is ≤ 90 / 180 block ids.
- The live pool is `KV_CACHE_MEMORY_BYTES=4294967296` (4 GiB; the README still says 2.5).
  It holds at least ~28.7 K block ids: 3.11 × the ≥ 9,216 that a 393,216-token request needs
  in its two compressed-KV groups alone. So the hold is under 0.7 % of the pool, taken from
  evictable cache, not new memory.
- Nothing is allocated on the GPU, and no tensor changes shape. Head `MemAvailable` (2.51 GiB
  low-water at 380 K; the floor comes during long prefills) is not expected to move. §9
  measures it.
- During a 1536-token prefill chunk, a request's real-held SWA blocks per group rise from
  about 27 to 30 / 33. That is still under its raised admission cap.

**C2 (compress-ratio-2) boundary state.**

- Layers 2-19 compress pairs of tokens with non-overlapping windows. The open pair's
  [kv, score] rows live in a per-request ring (`CompressorStateCache`, a `CircularBufferSpec`
  that is not prefix-cacheable). The compressor rejects any ratio but 1 and 2.
- At any even boundary the open group is empty. B_end and B_yoco are 128 multiples, so a hit
  needs no compressor state. A reader starts from a fresh ring.
- DeepInfer says the same thing for its dropped `ca_state`: "compression window boundaries are
  naturally aligned to multiples of the 128-token store granularity". V4.1 has no `ica_state`
  (`use_kproj`).
- The compressed KV (ratio 2 and 1) and the indexer K are full-attention groups. They are
  hashed as their blocks fill, decode included, so [0, B) is already cached at the free.

## 7. (b) Encoder SWA bounded replay: design and the case for deferring it

**What it would add.**

- Hits wherever block KV is still cached, at any 128 boundary, without an SWA checkpoint.
  The cost is 128 replayed tokens and an approximation.
- For #5 it adds nothing over (a).
- Its headroom in the 10-14 → 10-21 record:
  - Continuations whose hit fell more than 127 tokens below the previous prompt: 175 of
    7,242. 174 of those fell back at most one 4096 interval (median 309 tokens).
  - First turns of 127 sessions: 68 % of the first-turn hits sit on a 4096 checkpoint of the
    shared head (median 16,384 cached).
  - Upper bound: on the order of 0.4 M tokens per week, against (a)'s ~3.1 M.

**vLLM design, the variant faithful to DeepInfer's code (b1).**

1. *Hit policy* (`SlidingWindowManager.find_longest_cache_hit` + coordinator):
   - L1 is the hit of the full-attention groups (compressed KV, indexer K), capped at
     N − 134 (N − KEEP under YOCO).
   - If the SWA group has a real checkpoint at L1, use it.
   - Otherwise, hit h = L1 − 128 with all-null SWA blocks below h.
2. *Replay rows* [h, L1) are scheduled as ordinary prefill. Block KV for them is recomputed
   into the request's own fresh blocks, from approximate hidden states. This is what
   DeepInfer's code does, in its private copy. Nothing shared is written.
3. *SWA clamp*: every SWA layer clamps keys to `max(pos − 127, h)` for that request's rows.
   That covers encoder, decoder and draft layers, which share pages. It generalises
   `dsv41_yoco.clamp_swa_rows`, and needs a per-request `swa_valid_start` on the worker. That
   value could be carried in the scheduler output, or detected from null blocks inside a row's
   window.
4. *CUDA graphs*: steps with a replay row run eager, like YOCO's `need_eager`.
5. *Decode is unaffected*: N − h ≥ 134 + 128, so every decode window [n−127, n] is inside
   written SWA. The first draft's window [N−128, N) is too.
6. *C2 state at h*: h is even, so the ring starts empty.
7. *Memory*: none (no replica).

**The variant faithful to the QA doc (b2)** reuses block KV for [h, L1) as is. It needs two
things:

- per-group hit lengths (full groups at L1, SWA at h) through the coordinator, the KV
  manager's allocation and the scheduler's single `num_computed_tokens`;
- slot −1 for the replay rows in every full-attention group, so shared cached blocks are never
  written.

It has no reference implementation.

**Why deferred.**

1. (a) already removes about 78 % of the #5 loss, exactly.
2. (b) is a model-side overlay of YOCO's size. It touches prefill metadata for all 43 SWA
   layers and the scheduler's hit policy, and it is approximate.
3. Its approximation has to pass a teacher-forced logprob A/B at fork points on the GPU
   (`scripts/yoco_logprob_ab.py` style).
4. Its measured headroom is about an order of magnitude smaller than (a)'s.

Revisit after (a) is measured: the record script's "below prompt tail" and first-turn figures
are the inputs.

## 8. Implementation, tests, mutants

| file | role |
|---|---|
| `overlay/patch_swa_final_window.py` | 3 anchors in 2 image files (`v1/core/single_type_kv_cache_manager.py`: the hold in `SlidingWindowManager.__init__`; `v1/core/kv_cache_manager.py`: the hook in `free` and `pop_blocks_for_free`) plus footers. Applies all or nothing, is idempotent, fails loudly (exit 1, nothing written) on drift, `--check` only verifies. The patched files refuse to import unless `DSV41_SWA_FINAL_WINDOW=keep` |
| `start.sh` | `DSV41_SWA_FINAL_WINDOW=off\|keep` (default off): validated, passed to both containers. The patch is mounted on both ranks and run before YOCO only when not off; the container exits if it fails |
| `Dockerfile` | copies the patch and runs `--check` against the image's vLLM |
| `scripts/swa_final_window_turns.py` | per-turn own-output re-prefill, hit position and predictions from the repl record (read-only) |

Tests (CPU, `python3 tests/<file>`). All were red before the overlay existed.

- `test_swa_final_window_sim.py` runs the image's prefix-cache code under stubs. It covers:
  - fidelity to the record, 39/39;
  - the B_end hit on every turn, and 19,099 → 4,639;
  - both YOCO orders: 4,821, within the cap;
  - a residue sweep (240 prompt / output / growth cases × 2 modes);
  - the admission cap at every allocation;
  - bounded tails below the processed tokens, with a verify step or a prefill chunk in
    flight;
  - preemption, abort mid-prefill (also under YOCO), the deferred (async) free path, and a
    never-scheduled request;
  - the refusal to import when the feature is off;
  - no error logged by the fail-safe hook.

  It uses the shared fixture tree `tests/fixtures/dsv41_yoco` (7 more image files added).
- `test_swa_final_window_patch.py` covers the anchors against the image fixtures, compile,
  idempotency, atomic drift, `--check`, off is a no-op, coexistence with YOCO in either order,
  the footer math against brute force, and the wiring.
- `test_swa_final_window_launcher.py` covers validation, byte-identical argv on both ranks
  (alone and with YOCO), the inner guard running only when enabled, fatally, before YOCO, and
  propagation to both ranks.
- `test_swa_final_window_turns.py` checks the record script on a synthetic record. Its
  length-only predictions equal the patched image code's: 4,639 / 4,821.

Mutation runs: 23 of 24 mutants were killed (17 overlay, 6 launcher / Dockerfile). The
simulation alone kills 17 of the 18 overlay mutants by behaviour, not by source checks. The
survivor drops the null-block check before hashing a tail. It is equivalent:
`cache_full_blocks` skips null blocks itself, and every non-null block of such a tail
comes from a prefix hit, so it is already hashed. The check stays so that a partial tail is
never hashed. Two mutants survived a first run and got scenarios of their own: in-flight
tokens counted, with a prefill chunk in flight and no placeholders; and a YOCO checkpoint past
the computed tokens, on an abort mid-prefill.

## 9. Maintenance-window measurement plan (operator only; not run)

The live launcher is the non-git kit `gb10:/mnt/storage1/dsv41/exl3-kit`.

0. **Deploy.**
   - Copy `overlay/patch_swa_final_window.py`, and **port** the start.sh hunks (the kit's
     start.sh has diverged; never overwrite it, never copy the rai `.env`).
   - Restart with `SKIP_BUILD=1 SKIP_PULL=1 ./start.sh restart`. start.sh pauses and resumes
     the watchdog itself; boot takes about 12-17 min.
   - Both `logs/head.log` and `logs/worker.log` must show:
     - `SWA final-window patch: applied` for both files;
     - after the first request, `sliding-window groups keep 192 tokens below the window (+3
       admission blocks per request)` (332 / +6 under YOCO);
     - after the first finished request, `caching final sliding windows`.
   - Any `FATAL: SWA final-window patch failed` stopped the container. Roll back (step 5).

1. **Arms**, one restart each:
   - A0 = today (off, YOCO off);
   - A1 = `keep`, YOCO off;
   - if YOCO's own trial goes ahead: A2 = YOCO compact, A3 = YOCO compact + `keep`.

   In each arm, bracket the load with repl's window (`tools/window.py start --name swa-A1`
   … `end`) and run:
   - **Controlled agent load.** `scripts/prefix_retention_agent.py --base http://<head>:8000
     --scenarios append,fork,interleave --turns 8`. Each next prompt carries *this run's*
     output, and `short = LCP − cached` is printed per turn.
     - Expect `short` after an answer to drop from about output + 59 to 64-192 (A1), or to
       stay ≤ KEEP + 64 + 191 = 395 (A3).
     - Forks and interleaving must not get worse.
   - **Organic agent traffic**, at least 60 min, then:
     ```
     python3 scripts/swa_final_window_turns.py --from <start> --to <end> --json
     ```
     - Per turn: `reprefilled_own` against the A0 window, and `hit_vs_end`. That is the hit
       position relative to the previous prompt + output: A0 about −(O + 59); A1 −192..−64;
       A3 YOCO-capped.
     - A1's measured re-prefilled own output should be close to A0's
       `predicted_keep_off_tokens` (the 10-14 → 10-21 prediction is −78 %). A large gap means
       eviction or a client change. Check with `--per-turn`, and join dsh's `responseId` to
       `upstream_request_id`.
   - **Do not use `tools/record_to_trace.py` replays for this.** A replayed turn sends the
     *recorded* next prompt, which holds the original output, not the replay's. Own-output
     reuse cannot show up in a replay, with or without the patch. Replays still measure
     prompt-region reuse and timing.
   - **Head memory.** Sample `MemAvailable` (`free -m` every 1 s on gb10, or the `dsv41-mem`
     log lines) through one 380 K cold prefill (`scripts/chunk_validate.py kvprog --tokens
     380000`) and the agent load. Expect a floor ≥ A0's − 0.1 GiB; the pool is pinned.
   - **Decode unchanged.** `scripts/spec_bench.py --label <arm>` (c=1,2) gives tok/s and
     acceptance length. The record script's `decode_tok_s` median must stay within
     run-to-run noise (48.7 tok/s since the 1017 swap; prefill median 829 tok/s).
   - **Retention unchanged.** `scripts/chunk_validate.py retention --lengths 8193 34357` and
     `bash tests/test_smoke.sh`.

2. **Pass.** All of:
   - A1's re-prefilled own output is ≤ 35 % of A0's on comparable traffic;
   - no turn's hit moves *down* against A0's for the same lengths;
   - the MemAvailable floor holds;
   - decode tok/s is within noise;
   - retention and smoke checks match A0.

3. **Rollback.** Remove the `DSV41_SWA_FINAL_WINDOW` line or set it to `off`, then run
   `SKIP_BUILD=1 SKIP_PULL=1 ./start.sh restart`. The fresh containers carry unpatched vLLM
   files.

**Open risks the trial must clear** (each settled from source and the CPU simulation only):

- The real scheduler's async path frees with `num_in_flight_tokens` / placeholders as modelled.
- No other overlay touches the two files at runtime. Today only YOCO does, and the two
  coexist.
