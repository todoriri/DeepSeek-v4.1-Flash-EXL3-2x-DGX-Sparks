# EXL3 MoE wedge diagram (2026-09-21) — superseded

An architecture diagram of the EXL3 MoE "wedge" as it was understood on 2026-09-21:
`exl3-moe-wedge.html` (built from `exl3-moe-wedge.architecture.json`), plus its visual check
(`exl3-moe-wedge.visual-check.html` / `.json` and four light/dark screenshots).

**Superseded on 2026-09-30 (`13f1924`).** The recorded "wedges" were long cold prefills (vLLM counts
prompt tokens only at the first token, so the watchdog saw flat progress), plus the repl router's
300 s ReadTimeout retry loop. They were not a GPU hang. Read
[`../research-stability-perf-20260930.md`](../research-stability-perf-20260930.md) for the current account;
[`../exl3-moe-barrier.md`](../exl3-moe-barrier.md) keeps the kernel analysis.

Kept as a record: the files were generated at the repo root and left untracked until 2026-10-22.
The `artifact.path` in `exl3-moe-wedge.visual-check.json` still names that original root location.
