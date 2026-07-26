#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in \
  HRM_TEXT_MODEL_PATH \
  HRM_AUDIO_WHISPER_MODEL_PATH \
  HRM_AUDIOCAPS_TRAIN_MANIFEST \
  HRM_AUDIO_TRAINER_SMOKE_RUN_TAG \
  HRM_AUDIO_TRAINER_SMOKE_RUN_DIR \
  HRM_AUDIO_TRAINER_SMOKE_OUTPUT_REPORT \
  HRM_AUDIO_TRAINER_SMOKE_MICRO_BATCH_SIZE \
  HRM_AUDIO_TRAINER_SMOKE_GRADIENT_ACCUMULATION_STEPS \
  HRM_AUDIO_TRAINER_SMOKE_LORA_RANK \
  HRM_AUDIO_TRAINER_SMOKE_LORA_ALPHA \
  HRM_AUDIO_TRAINER_SMOKE_LORA_DROPOUT \
  HRM_AUDIO_TRAINER_SMOKE_LEARNING_RATE \
  HRM_AUDIO_TRAINER_SMOKE_ALIGNER_LEARNING_RATE; do
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
  -j smoke-hrm-audio-trainer-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/smoke_hrm_audio_swift_trainer_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/smoke_hrm_audio_swift_trainer.sh"
