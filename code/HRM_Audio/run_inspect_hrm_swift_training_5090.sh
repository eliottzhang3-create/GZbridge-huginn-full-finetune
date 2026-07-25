#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in \
  HRM_TEXT_MODEL_PATH \
  HRM_TEXT_SWIFT_PLUGIN_PATH \
  HRM_SWIFT_TRAIN_RUN_TAG \
  HRM_SWIFT_TRAIN_RUN_DIR \
  HRM_SWIFT_TRAIN_OUTPUT_REPORT \
  HRM_SWIFT_TRAIN_ADAPTER_DIR \
  HRM_SWIFT_TRAIN_LORA_RANK \
  HRM_SWIFT_TRAIN_LORA_ALPHA \
  HRM_SWIFT_TRAIN_LEARNING_RATE; do
  value="${!name:-}"
  if [ -n "$value" ]; then
    printf -v quoted_value '%q' "$value"
    CMD_PREFIX="${CMD_PREFIX}${name}=${quoted_value} "
  fi
done

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j inspect-hrm-swift-training-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/inspect_hrm_swift_training_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/inspect_hrm_swift_training.sh"
