#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in \
  AUDIOCAPS_TRAIN_MANIFEST \
  AUDIOCAPS_OUTPUT_DIR \
  AUDIOCAPS_LOGGING_DIR \
  AUDIOCAPS_NUM_TRAIN_EPOCHS \
  AUDIOCAPS_MAX_STEPS \
  AUDIOCAPS_SAVE_STRATEGY \
  AUDIOCAPS_SAVE_STEPS \
  AUDIOCAPS_SAVE_TOTAL_LIMIT \
  AUDIOCAPS_LOGGING_STEPS \
  AUDIOCAPS_REPORT_TO \
  AUDIOCAPS_RESUME_FROM_CHECKPOINT; do
  if [ -n "${!name:-}" ]; then
    CMD_PREFIX="${CMD_PREFIX}${name}=${!name} "
  fi
done

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j train-audiocaps-v2-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/train_audiocaps_v2_huginn_audio_swift_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/train_audiocaps_v2_huginn_audio_swift_5090.sh"
