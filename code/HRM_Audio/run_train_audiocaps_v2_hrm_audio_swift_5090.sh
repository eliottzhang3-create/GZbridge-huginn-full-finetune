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
  HRM_AUDIOCAPS_TRAIN_STATS \
  HRM_AUDIO_FORMAL_RUN_TAG \
  HRM_AUDIO_FORMAL_RUN_ROOT \
  HRM_AUDIO_FORMAL_AUDIT_REPORT \
  HRM_AUDIO_FORMAL_RESUME_FROM_CHECKPOINT \
  HRM_AUDIO_FORMAL_REPORT_TO \
  HRM_AUDIO_FORMAL_RESOURCE_INTERVAL_SECONDS; do
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
  -j train-hrm-audio-audiocaps-e2-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/train_hrm_audio_audiocaps_e2_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/train_audiocaps_v2_hrm_audio_swift_5090.sh"
