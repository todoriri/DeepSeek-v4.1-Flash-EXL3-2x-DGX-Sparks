# Vendored: DeepInfer benchmark evaluation (subset)

These files are copied **unmodified** from DeepSeek's DeepInfer,
<https://github.com/deepseek-ai/DeepInfer>, directory `eval/`, at commit
`774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`. They are © DeepSeek and licensed under the Apache
License 2.0; a copy of that license is in [LICENSE](LICENSE). DeepInfer ships no NOTICE file.

| File | Upstream | Used for |
| :--- | :--- | :--- |
| `aime/bench.py` | `eval/aime/bench.py` | AIME 2026 loader, prompt, `\boxed{}` extraction, rule judge |
| `gpqa/bench.py` | `eval/gpqa/bench.py` | GPQA Diamond loader (incl. public-mirror rows), prompt, `ANSWER:` extraction, rule judge |
| `common/passk.py` | `eval/common/passk.py` | unbiased Pass@k estimator, `summarize` |
| `common/io.py` | `eval/common/io.py` | `require_file` (imported by both `bench.py`) |
| `common/__init__.py` | `eval/common/__init__.py` | package marker |

Not vendored:
- `run_eval.py`, which drives DeepInfer's offline engine. `scripts/deepinfer_bench.py` replaces it
  with the OpenAI-compatible endpoint.
- `score.py` and `common/judge.py`. Only the rule judge is used, and it lives in `bench.py`.
- `hle/`. DeepInfer's `score.py` refuses rule judging for HLE.

`SOURCE.json` pins the sha256 of every file. `tests/test_deepinfer_bench.py` fails if a file is
edited, and also when it drifts from the stash checkout at
`/mnt/data1/Projects/stash/DeepInfer` (only when that checkout is present).

To resync after an upstream change:
1. Copy the same files over from the new commit.
2. Regenerate `SOURCE.json` and update the commit hash above.
3. Rerun the tests.
