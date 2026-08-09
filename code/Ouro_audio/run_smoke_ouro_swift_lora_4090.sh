#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in OURO_MODEL_PATH OURO_TEXT_PLUGIN_PATH OURO_LORA_TINY_DATASET OURO_LORA_RUN_TAG OURO_LORA_OUTPUT_DIR OURO_LORA_OUTPUT_REPORT; do
  value="${!name:-}"
  if [ -n "$value" ]; then
    printf -v quoted_value '%q' "$value"
    CMD_PREFIX="${CMD_PREFIX}${name}=${quoted_value} "
  fi
done

vc submit \
  -p pdgpu-4090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j smoke-ouro-swift-lora-4090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/smoke_ouro_swift_lora_4090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/smoke_ouro_swift_lora.sh"
