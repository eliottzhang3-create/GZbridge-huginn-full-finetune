#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

: "${QWEN3_BAT_SMOKE_DATASET:?Set QWEN3_BAT_SMOKE_DATASET before submitting}"

CMD_PREFIX="QWEN3_BAT_SMOKE_DATASET=$(printf '%q' "$QWEN3_BAT_SMOKE_DATASET") "
for name in QWEN3_MODEL_PATH QWEN3_BAT_PLUGIN_PATH QWEN3_BAT_SMOKE_RUN_TAG QWEN3_BAT_SMOKE_OUTPUT_DIR QWEN3_BAT_SMOKE_OUTPUT_REPORT QWEN3_BAT_DEVICE; do
  value="${!name:-}"
  if [ -n "$value" ]; then
    printf -v quoted_value '%q' "$value"
    CMD_PREFIX="${CMD_PREFIX}${name}=${quoted_value} "
  fi
done

vc submit \
  -p pdgpu-3090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j smoke-qwen3-bat-lora-3090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/smoke_qwen3_bat_lora_3090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/run_smoke_qwen3_bat_lora_remote.sh"
