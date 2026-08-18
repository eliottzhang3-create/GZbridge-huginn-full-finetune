#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"
: "${QWEN3_BAT_COMPILE_DDP_RESUME_CHECKPOINT:?Set QWEN3_BAT_COMPILE_DDP_RESUME_CHECKPOINT before submitting}"

CHECKPOINT="$QWEN3_BAT_COMPILE_DDP_RESUME_CHECKPOINT"
DATASET="${QWEN3_BAT_COMPILE_DDP_RESUME_DATASET:-/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/manifests/qwen3_bat_compile_ddp_128.jsonl}"
ROOT_DIR="${QWEN3_BAT_COMPILE_DDP_RESUME_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/qwen3/compile_ddp_resume-$(date +%Y%m%d-%H%M%S)}"

CMD_PREFIX="QWEN3_BAT_COMPILE_DDP_RESUME_CHECKPOINT=$(printf '%q' "$CHECKPOINT") QWEN3_BAT_COMPILE_DDP_RESUME_DATASET=$(printf '%q' "$DATASET") QWEN3_BAT_COMPILE_DDP_RESUME_OUTPUT_DIR=$(printf '%q' "$ROOT_DIR") "
for name in QWEN3_MODEL_PATH QWEN3_BAT_PLUGIN_PATH QWEN3_BAT_COMPILE_DATALOADER_NUM_WORKERS QWEN3_BAT_COMPILE_THREADS MASTER_PORT; do
  value="${!name:-}"
  if [ -n "$value" ]; then
    printf -v quoted_value '%q' "$value"
    CMD_PREFIX="${CMD_PREFIX}${name}=${quoted_value} "
  fi
done

vc submit \
  -p pdgpu-3090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 64 -m 256G -g 8 -n 1 \
  -j "qwen3-bat-compile-ddp-resume-3090-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/qwen3_bat_compile_ddp_resume_3090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/run_smoke_qwen3_bat_compile_ddp_resume_remote.sh"
