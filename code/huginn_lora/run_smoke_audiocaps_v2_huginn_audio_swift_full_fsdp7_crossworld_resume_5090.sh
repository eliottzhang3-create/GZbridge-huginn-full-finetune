#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in \
  AUDIOCAPS_FSDP7_CROSSWORLD_TRAIN_MANIFEST \
  AUDIOCAPS_FSDP7_CROSSWORLD_CHECKPOINT \
  AUDIOCAPS_FSDP7_CROSSWORLD_EXPECTED_STEP \
  AUDIOCAPS_FSDP7_CROSSWORLD_SMOKE_UPDATES \
  AUDIOCAPS_FSDP7_CROSSWORLD_RUN_TAG \
  AUDIOCAPS_FSDP7_CROSSWORLD_OUTPUT_DIR \
  AUDIOCAPS_FSDP7_CROSSWORLD_MIN_FREE_GB; do
  value="${!name:-}"
  if [ -n "$value" ]; then
    printf -v quoted_value '%q' "$value"
    CMD_PREFIX="${CMD_PREFIX}${name}=${quoted_value} "
  fi
done

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 7 \
  -n 1 \
  -j audiocaps-fsdp8to7-resume-smoke-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/smoke_audiocaps_v2_huginn_audio_swift_full_fsdp7_crossworld_resume_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/smoke_audiocaps_v2_huginn_audio_swift_full_fsdp7_crossworld_resume.sh"
