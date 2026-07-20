#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in \
  LOSATOK_TRAIN_MANIFEST \
  LOSATOK_OUTPUT_DIR \
  LOSATOK_LOGGING_DIR \
  LOSATOK_NUM_TRAIN_EPOCHS \
  LOSATOK_MAX_STEPS \
  LOSATOK_SAVE_STRATEGY \
  LOSATOK_SAVE_STEPS \
  LOSATOK_SAVE_TOTAL_LIMIT \
  LOSATOK_LOGGING_STEPS \
  LOSATOK_REPORT_TO \
  LOSATOK_RESUME_FROM_CHECKPOINT \
  LOSATOK_LEARNING_RATE \
  LOSATOK_ALIGNER_LR \
  HUGINN_LOSATOK_TRAIN_CHAIN_AUDIT; do
  if [ -n "${!name:-}" ]; then
    CMD_PREFIX="${CMD_PREFIX}${name}=${!name} "
  fi
done

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j train-audiocaps-v2-losatok-e3-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/train_audiocaps_v2_huginn_losatok_swift_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/train_audiocaps_v2_huginn_losatok_swift_5090.sh"
