#!/usr/bin/env bash
# tests/test_smoke.sh — 17*19, tools, optional image. Requires a healthy server.
set -euo pipefail
BASE="${1:-http://127.0.0.1:8888}"
AUTH=()
[ -n "${VLLM_API_KEY:-}" ] && AUTH=(-H "Authorization: Bearer ${VLLM_API_KEY}")
# The model id: argument 2, else SERVED_MODEL_NAME, else the id the server itself serves
# (a kit can override start.sh's default name), else start.sh's default.
MODEL="${2:-${SERVED_MODEL_NAME:-}}"
if [ -z "$MODEL" ]; then
  MODEL=$(curl -fsS "${AUTH[@]}" "$BASE/v1/models" 2>/dev/null \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null) || MODEL=""
  MODEL="${MODEL:-DeepSeek-v4.1-Flash-EXL3}"
fi
echo "model: $MODEL"

body=$(curl -fsS "${AUTH[@]}" "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"What is 17*19? Reply with the integer only."}],"max_tokens":32,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}')
echo "$body"
echo "$body" | grep -E '323' >/dev/null \
  || { echo "smoke FAIL: expected 323 in the completion" >&2; exit 1; }
echo "smoke OK: 17*19 -> 323"
