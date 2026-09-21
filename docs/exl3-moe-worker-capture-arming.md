# Arming worker-rank capture on the vLLM watchdog (draft)

**Why:** the 2026-09-21 A/B closed the launch-atomicity lead and pointed at a **two-rank
collective desync** (see `exl3-moe-wedge-ab-investigation-20260921.md`). Every capture so far
has only ever snapshotted the **head** rank; `WORKER_CAPTURE_CMD` has been unset in every
stall. We cannot confirm or kill the desync hypothesis without both ranks in one snapshot.

> This is a **draft / proposal**. It creates a credential and changes standing watchdog
> config on the live cluster — do not deploy without an explicit go-ahead. Nothing here has
> been applied.

## Verified facts (2026-09-21, read-only)

- Watchdog is a **host-networked** `python:3.12-slim` container `vllm-watchdog` on gb10,
  running `WORKER_CAPTURE_CMD` via `/bin/sh -c` at capture time, stdout → `stall-<ts>-worker-pyspy.txt`
  (`vllm_watchdog.py:209-216`, `WORKER_CAPTURE_TIMEOUT_S` default 150 s).
- **Blocker confirmed:** `docker exec vllm-watchdog command -v ssh` → `NO-SSH-IN-CONTAINER`,
  and no key is mounted. (Host networking means network reachability to kgb10 is *not* the
  problem — only the ssh client + key are missing.)
- gb10 host → `kgb10` passwordless ssh works (`BatchMode=yes`).
- Worker container `dsv41-exl3-worker` is up on kgb10 and **has** `/opt/dsv41/pyspy_dump.sh`
  (same teacher image as the head → the head's capture script is present worker-side too).

## Recommended: dedicated forced-command key + ssh client in the image

Rationale: the watchdog container also mounts `/var/run/docker.sock` (root-equivalent on
gb10). Mounting the host's *general* ssh key there would grant that container full control of
kgb10 too. Instead, use a **single-purpose key locked to the pyspy dump** via a forced
command, so a compromised container can only trigger a stack dump.

### 1. Dedicated keypair (on gb10 host)
```bash
ssh-keygen -t ed25519 -N "" -C "vllm-watchdog worker-capture" \
  -f ~/.ssh/dsv41_worker_capture
```

### 2. Lock the pubkey to one command (kgb10 `~/.ssh/authorized_keys`)
```
command="docker exec dsv41-exl3-worker bash /opt/dsv41/pyspy_dump.sh",no-agent-forwarding,no-port-forwarding,no-pty,no-X11-forwarding ssh-ed25519 AAAA...worker-capture...
```
The client's requested command is ignored; only the forced `pyspy_dump.sh` runs, and its
stdout returns to the watchdog. (Populate `AAAA...` from `~/.ssh/dsv41_worker_capture.pub`.)

### 3. Add an ssh client to the watchdog image
`~/logging_stack_gb10/vllm-watchdog/Dockerfile`:
```dockerfile
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends openssh-client \
 && rm -rf /var/lib/apt/lists/*
```

### 4. Wire it in `docker-compose.gb10-watchdog.yml`
```yaml
    # image: python:3.12-slim          # <- replace with a local build:
    build: ./vllm-watchdog
    image: vllm-watchdog:ssh
    volumes:
      - ./vllm-watchdog/vllm_watchdog.py:/app/vllm_watchdog.py:ro
      - /var/run/docker.sock:/var/run/docker.sock
      - ./vllm-watchdog-data:/var/log/vllm-watchdog
      - /home/todoriri/.ssh/dsv41_worker_capture:/keys/worker_capture:ro   # NEW
    environment:
      # ... existing env ...
      WORKER_CAPTURE_CMD: >-
        ssh -i /keys/worker_capture -o BatchMode=yes
        -o StrictHostKeyChecking=accept-new
        -o ConnectTimeout=8 todoriri@kgb10
      WORKER_CAPTURE_TIMEOUT_S: "150"
```
(The remote command is supplied by the forced command in step 2, so none is needed here.
`accept-new` trusts kgb10's host key on first connect — acceptable on the mgmt LAN; harden to
a mounted `known_hosts` later if desired.)

## Deploy + verify
```bash
cd ~/logging_stack_gb10
docker compose -f docker-compose.gb10-watchdog.yml up -d --build
# dry-run the exact capture the watchdog will fire:
docker exec vllm-watchdog sh -c \
  "ssh -i /keys/worker_capture -o BatchMode=yes -o StrictHostKeyChecking=accept-new todoriri@kgb10" \
  | head            # expect a py-spy dump of the worker's vLLM, NOT a shell error
```
Then on the next real stall, confirm `vllm-watchdog-data/stall-<ts>-worker-pyspy.txt` is
written and the event's `worker_rank` field is populated (the code classifies it alongside
the head — `vllm_watchdog.py:592+`). The payoff read: **head ACTIVE-in-a-launch while worker
IDLE-on-RPC (or vice-versa) confirms the desync**; both ACTIVE in the same collective points
elsewhere.

## Alternative (rejected): host-side trigger relay
Have `WORKER_CAPTURE_CMD` `touch` a trigger in the bind-mounted data dir and run a host-side
`inotifywait`/systemd-path unit on gb10 that ssh's to kgb10 with the *host* key. Keeps the key
off the container entirely, but adds a second moving part and the capture lands in a separate
file instead of the watchdog's own `stall-<ts>-worker-pyspy.txt` (loses the auto-classification).
Prefer the forced-command key above.

## Pairs with
- Set `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` explicitly (e.g. 300 s) at engine launch and re-check
  whether recovery time tracks it — if the ~480 s recovery follows, the NCCL-heartbeat /
  collective-timeout story is confirmed.
