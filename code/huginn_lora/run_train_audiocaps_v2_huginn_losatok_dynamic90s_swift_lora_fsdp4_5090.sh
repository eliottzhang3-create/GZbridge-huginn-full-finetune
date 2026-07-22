#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in \
  LOSATOK_DYNAMIC_FSDP4_TRAIN_MANIFEST \
  LOSATOK_DYNAMIC_FSDP4_OUTPUT_DIR \
  LOSATOK_DYNAMIC_FSDP4_LOGGING_DIR \
  LOSATOK_DYNAMIC_FSDP4_LEARNING_RATE \
  LOSATOK_DYNAMIC_FSDP4_ALIGNER_LR \
  LOSATOK_DYNAMIC_FSDP4_LOGGING_STEPS \
  LOSATOK_DYNAMIC_FSDP4_REPORT_TO; do
  if [ -n "${!name:-}" ]; then
    CMD_PREFIX="${CMD_PREFIX}${name}=${!name} "
  fi
done

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 128G -g 4 \
  -n 1 \
  -j train-audiocaps-v2-losatok-dynamic90s-lora-fsdp4-e3-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/train_audiocaps_v2_huginn_losatok_dynamic90s_swift_lora_fsdp4_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/train_audiocaps_v2_huginn_losatok_dynamic90s_swift_lora_fsdp4.sh"

