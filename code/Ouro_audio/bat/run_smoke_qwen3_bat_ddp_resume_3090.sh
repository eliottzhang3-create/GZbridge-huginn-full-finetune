#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"
: "${QWEN3_BAT_DDP_RESUME_DATASET:?Set QWEN3_BAT_DDP_RESUME_DATASET before submitting}"

DATASET="$QWEN3_BAT_DDP_RESUME_DATASET"
ROOT_DIR="${QWEN3_BAT_DDP_RESUME_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/qwen3/ddp_resume_smoke_8x3090-$(date +%Y%m%d-%H%M%S)}"
CMD_PREFIX="QWEN3_BAT_DDP_RESUME_DATASET=$(printf '%q' "$DATASET") QWEN3_BAT_DDP_RESUME_ROOT=$(printf '%q' "$ROOT_DIR") "
for name in QWEN3_MODEL_PATH QWEN3_BAT_PLUGIN_PATH MASTER_PORT OMP_NUM_THREADS; do
  value="${!name:-}"
  if [ -n "$value" ]; then
    printf -v quoted_value '%q' "$value"
    CMD_PREFIX="${CMD_PREFIX}${name}=${quoted_value} "
  fi
done

vc submit \
  -p pdgpu-3090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 \
  -j "qwen3-bat-ddp-resume-3090-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/qwen3_bat_ddp_resume_3090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/run_qwen3_bat_ddp_resume_remote.sh"
