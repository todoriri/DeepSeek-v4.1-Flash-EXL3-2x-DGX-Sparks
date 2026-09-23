# Periodic prefix-cache retention for agent workloads

## Default and configuration

The launcher explicitly passes `--prefix-cache-retention-interval 4096` on **both
ranks**. This applies even to existing `.env` files that omit the new setting.

```bash
# .env
PREFIX_CACHE_RETENTION_INTERVAL=4096
```

The launcher accepts `0` or a positive multiple of 128 up to 1048576, normalizes
leading zeroes, and rejects invalid/empty values before an approved restart can
stop the running service. The engine remains responsible for validating other
cache geometries. Use this setting rather than a duplicate retention flag in
`EXTRA_ARGS`; duplicates are rejected to avoid silently overriding the policy.
An explicit CLI value also takes precedence over the image's deprecated
`VLLM_PREFIX_CACHE_RETENTION_INTERVAL` environment default.

Restart both ranks with the usual launcher procedure to apply. Roll back with
`PREFIX_CACHE_RETENTION_INTERVAL=0` and another approved restart. Zero restores
the image's latest-reachable-boundary-only behavior; it does not disable APC.

## Why this is needed

Image `sha256:2f0cf3adc0f989c1d446be274df864eb799630175f604c3b22b71b7205971dce`
(vLLM `v0.1.dev20904+g179dd0fa9`) defaults `CacheConfig` retention to **0**. Do not
confuse that with the lower-level `KVCacheConfig` dataclass default of `None`,
which means dense checkpoint retention.

In the tested hybrid-attention / DSpark profile, hits align to 128 tokens. An
EAGLE-marked group also needs a 64-token lookahead block before safely dropping
it. At some prompt lengths the newest retained checkpoint cannot satisfy this
lookup. The previous checkpoint's sliding-window state was not retained, so a
safe common hit collapses to zero even though prefix caching is enabled.

Fresh-salt, one-output-token request pairs on the unchanged server demonstrated:

| Input tokens | Cached on second request, retention 0 |
|---:|---:|
| 8192 | 8064 |
| 8193 | 0 |
| 8256 | 0 |
| 8257 | 8192 |
| 34357 | 0 |
| 34370 | 34304 |

The failure also reproduces with append-only chat history and eight tool
schemas: a 11189-token first turn followed by its ACK and a new user message
had zero hits; a nearby 11236-token first turn reused 11136 tokens. Both returned
correct ACK/pong answers. Thus a successful short cache smoke test does not
establish reliable reuse for agent workloads.

Periodic checkpoints preserve earlier fallback states. This is a **retention
mitigation**, not a change to the engine's hit-selection algorithm. It preserves
all hybrid-group and speculative block-drop correctness checks. Never work
around misses by pretending that missing sliding-window KV is a valid hit.

## Operational consequences

- This does not enlarge the configured KV allocation or change weights/native
  kernels. Cached checkpoints are evictable within the existing pool.
- More entries compete with other prefixes and require cache bookkeeping;
  scheduling, cache pressure and workload-specific performance must be measured.
- A periodic fallback can require recomputing roughly the last interval of
  tokens, rather than resuming immediately before the end of the prompt.
- Sessions and forks can share an identical token prefix, but there is no pinned
  per-session tree, durable storage, or guarantee that an old fork remains warm.
- Interleaved sessions can evict each other's checkpoints. Parallel summarization
  also competes for compute and active-request slots; it has no priority isolation.
- Replacing history with a summary changes the prefix. Retention cannot reuse
  the old suffix KV under a different summarized history. Background summary
  requests themselves benefit only to the extent their actual prefix matches.
- The number 4096 is a conservative checkpoint spacing, not a measured universal
  optimum, a cache lifetime in seconds, or a guaranteed maximum recomputation bound
  after eviction.

## Validation

Per-request cached-token fields require the vLLM usage-reporting option
`--enable-prompt-tokens-details`, which the launcher enables. Retention controls
cache reuse independently of reporting and does not alter API usage schemas.
Engine prefix-cache metrics can also establish reuse when fields are absent.

Host-only launcher regression:

```bash
python3 tests/test_prefix_cache_retention.py
python3 tests/test_numeric_config.py
bash -n start.sh
```

The dedicated test evaluates the actual generated rank argv, checks head/worker
Docker propagation, defaults and explicit rollback, validates numeric edge cases,
and rejects duplicate overrides before stop/restart.

A CPU metadata reproduction using the image's cache manager and representative
hybrid groups tested all 128 residues around 8k plus five longer boundaries.
Retention 0 reproduced the live alignment cliff. Retention 4096 recovered a
positive hit at all 133 tested lengths; for a 34357-token input it recovered
32768 tokens. This synthetic manager test is not a model-output quality test.

Two-Spark validation on 2026-09-18 used DSpark k=3, batch 1536, 96 IO threads,
vision on, packed Engram, two active sequences, a 2.5 GiB KV pool per rank, and
the existing GPU-validated cooperative safety overlay. A 34357-token Pi replay
went from zero hits on all repeats to **0 / 32768 / 32768** cached tokens for
cold/second/third requests with retention 4096. This validates reuse for that
profile, not k=5, full 600k inference, or arbitrary concurrent long-context
capacity. Before deploying a different profile, check:

1. Cold/repeated requests on both sides of the boundary, especially 8193/8256/8257.
2. Append-only tool-bearing sessions with exact-prefix verification.
3. Interleaved A/B/A sessions and forks to earlier history.
4. A background summary fork alongside foreground continuation.
5. Useful output on cache hits, API/tool/image smoke checks, and memory/cache
   pressure during longer prompts and generation.
