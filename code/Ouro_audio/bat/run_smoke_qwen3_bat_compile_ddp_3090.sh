#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"

CMD_PREFIX=""
for name in QWEN3_MODEL_PATH QWEN3_BAT_PLUGIN_PATH QWEN3_BAT_COMPILE_SOURCE_DATASET QWEN3_BAT_COMPILE_DATASET QWEN3_BAT_COMPILE_DDP_OUTPUT_DIR QWEN3_BAT_COMPILE_DDP_REPORT QWEN3_BAT_COMPILE_DATALOADER_NUM_WORKERS QWEN3_BAT_COMPILE_THREADS MASTER_PORT; do
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
  -j "qwen3-bat-compile-ddp-3090-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/qwen3_bat_compile_ddp_3090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/run_smoke_qwen3_bat_compile_ddp_remote.sh"
