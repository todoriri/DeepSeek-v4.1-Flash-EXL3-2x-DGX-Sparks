# Build and repin the cooperative MoE binary (fork maintainer path)

This is the **maintenance-lab** procedure for producing a fresh
`cooperative_moe.so`, validating it on the two Sparks, and repinning it on this
fork. It is the counterpart to the operator
[opt-in quickstart](cooperative-moe-quickstart.md): the quickstart *installs* an
already-validated, already-pinned binary; this document is how that pinned
binary comes to exist in the first place.

Upstream (MiaAI-Lab) never published the original `a09a589c…` artifact — see
their issue #17, still open. This fork is self-contained instead: we build,
gate, repin, and commit the binary here (`.gitignore` already whitelists
`extensions/cooperative_moe/artifacts/cooperative_moe.so`).

## Read first — non-negotiables

- **A clean rebuild will not match the old hash, by design.** `nvcc -lineinfo`
  and the GNU build-id vary across runs, so `prepare_profile.py` correctly
  rejects a rebuilt binary until you repin it. Repinning is legitimate here
  because we own this fork; it is *not* a way to bypass validation.
- **Build on a GB10-class box (aarch64, SM121a).** The `.so` carries aarch64
  host code and is loaded by aarch64 Python on gb10. The rai dev host is
  `x86_64` and **cannot** produce a usable binary. Do the `nvcc` step on gb10
  (or another SM121a aarch64 machine with the recipe image).
- **Repin only *after* the 54-case GPU gate passes on both nodes.** Apply the
  pin edits provisionally to run the gate; keep them (commit) only on a pass,
  revert them on a fail.
- **Maintenance window.** The gate stops DS4.1 and runs GPU fixtures. Do not run
  it alongside active user workloads.
- Hosts referenced below: **rai** = this fork's git checkout
  (`/mnt/data1/Projects/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks`); **gb10** =
  the two-Spark serving cluster (aarch64), with the pinned recipe image and
  passwordless SSH to its worker.

## Shared variables

```bash
# Pinned, reproducible build/validation image (same one the quickstart uses).
COOP_IMAGE='ghcr.io/miaai-lab/deepseek-v4.1-flash-exl3-2x-dgx-sparks@sha256:2f0cf3adc0f989c1d446be274df864eb799630175f604c3b22b71b7205971dce'
# Its resolved local image id must be:
COOP_IMAGE_ID='sha256:4cdba4e946da2d19bf5b5a20c6d3a1a4bf421fa4d6db5082f271a986168176cb'
# Pinned ExLlamaV3 source the native kernel is compiled against.
EXLLAMAV3_URL='https://github.com/turboderp-org/exllamav3.git'
EXLLAMAV3_PIN='02aef45cd681b960a00afcd0749a4ab99e6c1bfe'
# This fork's extension directory (relative to the repo root).
EXT='extensions/cooperative_moe'
```

## 1. Archive the pinned upstream headers (git host)

`archive_upstream.sh` needs `git`; the recipe image does not have it. Run this on
any host with git (rai is fine — this step is architecture-independent, it only
extracts headers).

```bash
cd /mnt/data1/Projects/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks   # the fork checkout
WORK="$(mktemp -d /tmp/coop-build.XXXXXX)"
git clone "$EXLLAMAV3_URL" "$WORK/exllamav3-src"
git -C "$WORK/exllamav3-src" checkout "$EXLLAMAV3_PIN"
bash "$EXT/archive_upstream.sh" "$WORK/exllamav3-src" "$WORK/upstream-tree"
# archive_upstream.sh hard-checks the pin and extracts exllamav3/exllamav3_ext.
```

If you archived on rai, ship the extracted tree **and** this fork's `$EXT`
directory to gb10 (the recipe image build reads both):

```bash
GB10=gb10                                  # or user@host from your .env
GB10_WORK='/mnt/storage1/dsv41/coop-build' # scratch on the aarch64 box
ssh "$GB10" "mkdir -p '$GB10_WORK'"
rsync -a "$WORK/upstream-tree" "$EXT" "$GB10:$GB10_WORK/"
```

## 2. Compile in the recipe image (gb10, aarch64)

`build.sh` never calls git and refuses a non-empty output dir. Run it inside the
pinned image so the toolchain/ABI match the validated lane.

```bash
# On gb10:
GB10_WORK='/mnt/storage1/dsv41/coop-build'
docker run --rm --gpus all \
  -e MAX_JOBS=2 \
  -v "$GB10_WORK:/build" \
  --entrypoint bash "$COOP_IMAGE" \
  /build/cooperative_moe/build.sh /build/upstream-tree /build/out
# Outputs in $GB10_WORK/out: cooperative_moe.so, runtime.py, cooperative_moe-build.log
NEW_SO="$(ssh gb10 "sha256sum '$GB10_WORK/out/cooperative_moe.so'" | cut -d' ' -f1)"
echo "new binary sha256: $NEW_SO"    # expect this to DIFFER from a09a589c… — that is normal
```

Keep `cooperative_moe-build.log` (ptxas verbose output) with the artifact for
review; the docs require reviewing the build before repinning.

**Note — `out/runtime.py` is informational.** `build.sh` compiles only the `.so`
and copies the repo's *current* (pre-edit) `runtime.py` beside it for
co-location. The adapter you actually pin and stage is the repo's `runtime.py`
*after* you edit its `SHA256` in step 3 — not `out/runtime.py`. `runtime.py` is
pure Python glue; it is not compiled, so the only change it needs is the new
`SHA256` value.

## 3. Provisional repin (to make the gate loadable)

The GPU gate loads the adapter through a `prepare_profile.py`-generated overlay,
and both `prepare_profile.py` and `runtime.py` verify hashes. Update the pins to
the new binary so the generator accepts it. **Do this on a working copy first**
(gb10 scratch or a scratch branch on rai) — it is not committed until step 6.

Order matters, because editing `runtime.py` changes its own hash, which feeds
`ADAPTER_SHA`:

```bash
# Run against the $EXT copy you will stage for the gate. NEW_SO from step 2.
cd "$EXT"
install -m 644 /path/to/out/cooperative_moe.so artifacts/cooperative_moe.so

# a. binary hash -> runtime.py's self-check and prepare_profile's binary pin
sed -i "s|^SHA256 = \".*\"|SHA256 = \"${NEW_SO}\"|"       runtime.py
sed -i "s|^BINARY_SHA = \".*\"|BINARY_SHA = \"${NEW_SO}\"|" prepare_profile.py
printf '%s  cooperative_moe.so\n' "${NEW_SO}" > artifacts/SHA256SUMS

# b. NOW recompute runtime.py's hash (after step a edited it) -> adapter pin
NEW_RT="$(sha256sum runtime.py | cut -d' ' -f1)"
sed -i "s|^ADAPTER_SHA = \".*\"|ADAPTER_SHA = \"${NEW_RT}\"|" prepare_profile.py

# Sanity: STOCK_SHA is unchanged (overlay/exl3.py was not touched).
grep -E "^(STOCK_SHA|BINARY_SHA|ADAPTER_SHA) =" prepare_profile.py
grep -E "^SHA256 =" runtime.py
cd -
```

## 4. Generate the overlay and stage identical artifacts on both nodes

Same mechanism as quickstart step 4 — the launcher does not copy the `.so` or
adapter, so stage them explicitly on head **and** worker at the same
container-visible path.

```bash
# On gb10, with the repinned $EXT copy from step 3 and your .env sourced:
STAGE="$CACHE_ROOT/coop-build"      # a versioned staging dir under the head cache
RUNTIME_DIR='/root/.cache/vllm/coop-build'   # identical container path on both ranks
mkdir -p "$STAGE"
# Stage the .so copied in at step 3 AND the runtime.py you edited at step 3
# (its hash == ADAPTER_SHA). Do NOT stage build.sh's out/runtime.py — it is unedited.
install -m 644 "$EXT/artifacts/cooperative_moe.so" "$EXT/runtime.py" "$STAGE/"
python3 "$EXT/prepare_profile.py" \
  --stock overlay/exl3.py \
  --artifacts "$STAGE" \
  --runtime-directory "$RUNTIME_DIR" \
  --output "$STAGE/exl3-cooperative.py"
install -m 644 "$EXT/test_cuda_integration.py" tests/test_exl3_overlay.py "$STAGE/"

# Mirror to the worker and verify byte-identical staging.
ssh "$WORKER" "mkdir -p '$WORKER_STAGE'"
scp "$STAGE"/{cooperative_moe.so,runtime.py,exl3-cooperative.py,test_cuda_integration.py,test_exl3_overlay.py} "$WORKER:$WORKER_STAGE/"
(cd "$STAGE" && sha256sum cooperative_moe.so runtime.py exl3-cooperative.py test_cuda_integration.py test_exl3_overlay.py) > "$STAGE/SHA256SUMS"
scp "$STAGE/SHA256SUMS" "$WORKER:$WORKER_STAGE/SHA256SUMS"
ssh "$WORKER" "cd '$WORKER_STAGE' && sha256sum -c SHA256SUMS"
```

If `prepare_profile.py` errors here, the pins in step 3 do not match the staged
files — fix the pins, do not touch the generated overlay by hand.

## 5. Run the 54-case GPU gate on each node

DS4.1 and every other GPU workload must be stopped. These containers do not load
the checkpoint; they run the synthetic integration fixtures.

```bash
docker run --rm --network none --gpus all --cpus 2 --memory 6g --memory-swap 6g \
  -e DSV41_COOP_MAINTENANCE_TEST=1 -e MAX_JOBS=2 -e OMP_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$CACHE_ROOT:/root/.cache/vllm" \
  -v "$STAGE/exl3-cooperative.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py:ro" \
  -v "$STAGE/test_exl3_overlay.py:/opt/dsv41/test_exl3_overlay.py:ro" \
  -v "$STAGE/test_cuda_integration.py:/opt/dsv41/test_cuda_integration.py:ro" \
  --entrypoint python3 "$COOP_IMAGE" /opt/dsv41/test_cuda_integration.py \
  2>&1 | tee gate-head.log
# ...and the same command over ssh on the worker (mount $WORKER_CACHE / $WORKER_STAGE).
```

**Pass criterion, required on BOTH nodes:** the final JSON line is
`{"stage": "complete", "checks": 54, "status": "pass", ...}` and the process exit
code is `0`. Any fixture line with a non-`pass` status, a `checks` count other
than 54, or a nonzero exit is a **fail** — roll back the provisional pins
(step 6, fail branch). A pass is the documented peak-normalized numerical screen,
not bitwise parity with stock.

## 6. Keep or revert the repin

**On pass** — make it authoritative on the fork (do this on rai, in the git
checkout, with the validated `.so` copied back from gb10):

```bash
cd /mnt/data1/Projects/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks
scp gb10:"$GB10_WORK/out/cooperative_moe.so"        "$EXT/artifacts/cooperative_moe.so"
scp gb10:"$GB10_WORK/out/cooperative_moe-build.log" "$EXT/artifacts/"    # optional, for provenance
# Apply the SAME pin edits from step 3 to the committed files (NEW_SO recorded above):
sed -i "s|^SHA256 = \".*\"|SHA256 = \"${NEW_SO}\"|"        "$EXT/runtime.py"
sed -i "s|^BINARY_SHA = \".*\"|BINARY_SHA = \"${NEW_SO}\"|" "$EXT/prepare_profile.py"
printf '%s  cooperative_moe.so\n' "${NEW_SO}" > "$EXT/artifacts/SHA256SUMS"
NEW_RT="$(sha256sum "$EXT/runtime.py" | cut -d' ' -f1)"
sed -i "s|^ADAPTER_SHA = \".*\"|ADAPTER_SHA = \"${NEW_RT}\"|" "$EXT/prepare_profile.py"

# Re-run the host-side CPU suites (they use fixtures/stubs, so they stay green
# and confirm the pins are self-consistent):
( cd "$EXT" && python3 test_dispatch.py && python3 test_profile.py && python3 test_build.py \
  && bash -n build.sh archive_upstream.sh )

git add "$EXT/artifacts/cooperative_moe.so" "$EXT/artifacts/SHA256SUMS" \
        "$EXT/runtime.py" "$EXT/prepare_profile.py"
git commit -m "cooperative_moe: repin rebuilt .so <NEW_SO short> after 54-case GPU gate (both nodes)"
git push origin main
```

Record the gate logs (`gate-head.log`, gate-worker log) and the build log with
the commit or in the PR description as the validation evidence.

**On fail** — discard everything; nothing is committed:

```bash
cd /mnt/data1/Projects/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks
git checkout -- "$EXT/runtime.py" "$EXT/prepare_profile.py" "$EXT/artifacts/SHA256SUMS"
git clean -f "$EXT/artifacts/cooperative_moe.so"   # if you had copied it in
```

## 7. Deploy the freshly pinned binary

Once committed, operators (and gb10's manually-copied kit) install the new
`.so` exactly as the [opt-in quickstart](cooperative-moe-quickstart.md)
describes — the digest it verifies is now your `NEW_SO`. Because this fork's
serving config is text-only and tuned (`LANGUAGE_MODEL_ONLY=1`,
`MAX_MODEL_LEN=393216`, `MAX_NUM_SEQS=3`), follow the quickstart but keep those
`.env` values; the coop kernel's activation gates (≤8 decode rows, hidden 5120 /
intermediate 1152 / top-k 6, K2/K3 mul1) are satisfied by that profile, and the
only serving prerequisites — `DSV41_EXL3_SERIAL_STREAMS=1`,
`VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`, `EXL3_FUSED_MOE=1` — are already set.
Activation remains a single added line: `EXL3_OVERLAY_HOST=<generated overlay>`.

## Cleanup

```bash
rm -rf "$WORK"                       # rai scratch
ssh gb10 "rm -rf '$GB10_WORK'"       # gb10 scratch (keep the build/gate logs elsewhere first)
```
