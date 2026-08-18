#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in QWEN3_MODEL_PATH QWEN3_BAT_PLUGIN_PATH BAT_QA_ROOT QWEN3_BAT_MULTIMODAL_OUTPUT_REPORT QWEN3_BAT_DEVICE; do
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
  -j inspect-qwen3-bat-multimodal-3090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/inspect_qwen3_bat_multimodal_3090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/run_qwen3_bat_multimodal_audit_remote.sh"
