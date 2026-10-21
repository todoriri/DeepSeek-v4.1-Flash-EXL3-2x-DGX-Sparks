#!/usr/bin/env bash
# Build what tests/test_control_token_render_path.py needs to run the image's real
# serving code on CPU:
#   $DEST/vllm   the image's unpatched vLLM Python tree (no .so/.cubin/__pycache__)
#   $DEST/venv   Python 3.12 with the image's pins (tests/fixtures/dsv41_render/
#                requirements.txt) and the CPU build of the image's torch
#
#   scripts/dsv41_vllm_tree.sh [DEST]        (default /tmp/dsv41-vllm)
#   HOST=gb10 IMAGE=sha256:0aecf9f5... scripts/dsv41_vllm_tree.sh
#
# Read-only on HOST: the tree comes from a `docker create`d container that is never
# started, then removed. Running containers are not touched. Needs ssh to HOST, uv,
# and python3.12 here.
set -euo pipefail

DEST="${1:-/tmp/dsv41-vllm}"
HOST="${HOST:-gb10}"
IMAGE="${IMAGE:-sha256:0aecf9f5f1589945430acc2eca09fa972311bd0d80fd66de5868bb9acfbc72b5}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SITE=/usr/local/lib/python3.12/dist-packages/vllm

mkdir -p "$DEST"
if [ ! -f "$DEST/vllm/__init__.py" ]; then
    echo "extracting $SITE from $IMAGE on $HOST ..." >&2
    ssh -o BatchMode=yes "$HOST" python3 - "$IMAGE" "$SITE" <<'EOF' | tar -C "$DEST" -xf -
import subprocess, sys, tarfile
image, site = sys.argv[1:3]
cid = subprocess.run(["docker", "create", image, "true"], check=True, capture_output=True, text=True).stdout.strip()
try:
    cp = subprocess.Popen(["docker", "cp", f"{cid}:{site}", "-"], stdout=subprocess.PIPE)
    src = tarfile.open(fileobj=cp.stdout, mode="r|")
    dst = tarfile.open(fileobj=sys.stdout.buffer, mode="w|")
    for member in src:
        if member.name.endswith((".so", ".cubin", ".pyc")) or "__pycache__" in member.name.split("/"):
            continue
        dst.addfile(member, src.extractfile(member) if member.isfile() else None)
    dst.close()
    if cp.wait():
        sys.exit("docker cp failed")
finally:
    subprocess.run(["docker", "rm", "-f", cid], capture_output=True)
EOF
fi
sha256sum "$DEST/vllm/tokenizers/deepseek_v41.py" "$DEST/vllm/tokenizers/deepseek_v41_encoding.py" >&2

if [ ! -x "$DEST/venv/bin/python" ]; then
    uv venv -q -p python3.12 "$DEST/venv"
    VIRTUAL_ENV="$DEST/venv" uv pip install -q --no-deps -r "$ROOT/tests/fixtures/dsv41_render/requirements.txt"
    VIRTUAL_ENV="$DEST/venv" uv pip install -q --no-deps torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
fi

echo "DSV41_VLLM_TREE=$DEST DSV41_VLLM_PYTHON=$DEST/venv/bin/python"
