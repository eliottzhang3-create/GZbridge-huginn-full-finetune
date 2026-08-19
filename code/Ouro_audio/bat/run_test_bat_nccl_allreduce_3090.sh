#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"
OUTPUT="$(printenv BAT_NCCL_OUTPUT 2>/dev/null || true)"
if [ -z "$OUTPUT" ]; then echo "Set BAT_NCCL_OUTPUT" >&2; exit 2; fi
CMD_PREFIX="BAT_NCCL_OUTPUT=$(printf '%q' "$OUTPUT") "
append_env() { local name="$1"; local value; value="$(printenv "$name" 2>/dev/null || true)"; if [ -n "$value" ]; then CMD_PREFIX="$CMD_PREFIX$name=$(printf '%q' "$value") "; fi; }
for name in BAT_NCCL_WORLD_SIZE BAT_NCCL_WARMUP BAT_NCCL_ITERATIONS BAT_NCCL_TENSOR_ELEMENTS NCCL_DEBUG NCCL_DEBUG_SUBSYS NCCL_SOCKET_IFNAME NCCL_IB_DISABLE NCCL_SHM_DISABLE NCCL_CUMEM_HOST_ENABLE; do append_env "$name"; done
vc submit -p pdgpu-3090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 \
  -j "bat-nccl-allreduce-3090-$(date +%m%d%H%M)" -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/bat-nccl-allreduce-3090.JOB.log" \
  --cmd "$CMD_PREFIX bash scripts/test_bat_nccl_allreduce_remote.sh"

