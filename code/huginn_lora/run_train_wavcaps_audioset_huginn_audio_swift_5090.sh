#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in \
  WAVCAPS_TRAIN_MANIFEST WAVCAPS_INIT_CHECKPOINT WAVCAPS_OUTPUT_DIR WAVCAPS_LOGGING_DIR \
  WAVCAPS_NUM_TRAIN_EPOCHS WAVCAPS_MAX_STEPS WAVCAPS_SAVE_STRATEGY WAVCAPS_SAVE_STEPS \
  WAVCAPS_SAVE_TOTAL_LIMIT WAVCAPS_LOGGING_STEPS WAVCAPS_LEARNING_RATE WAVCAPS_ALIGNER_LR; do
  if [ -n "${!name:-}" ]; then
    CMD_PREFIX="${CMD_PREFIX}${name}=${!name} "
  fi
done

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j train-wavcaps-audioset-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/train_wavcaps_audioset_huginn_audio_swift_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/train_wavcaps_audioset_huginn_audio_swift_5090.sh"
