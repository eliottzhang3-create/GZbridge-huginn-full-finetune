#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in \
  LOSATOK_DYNAMIC_FSDP2_TRAIN_MANIFEST \
  LOSATOK_DYNAMIC_FSDP2_OUTPUT_DIR \
  LOSATOK_DYNAMIC_FSDP2_LOGGING_DIR \
  LOSATOK_DYNAMIC_FSDP2_LEARNING_RATE \
  LOSATOK_DYNAMIC_FSDP2_ALIGNER_LR \
  LOSATOK_DYNAMIC_FSDP2_LOGGING_STEPS \
  LOSATOK_DYNAMIC_FSDP2_REPORT_TO; do
  if [ -n "${!name:-}" ]; then
    CMD_PREFIX="${CMD_PREFIX}${name}=${!name} "
  fi
done

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 16 -m 64G -g 2 \
  -n 1 \
  -j losatok-dyn90-f2-b4ga4-e3-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/train_audiocaps_v2_huginn_losatok_dynamic90s_swift_lora_fsdp2_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/train_audiocaps_v2_huginn_losatok_dynamic90s_swift_lora_fsdp2.sh"
