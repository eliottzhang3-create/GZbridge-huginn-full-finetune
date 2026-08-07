#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in \
  HUGINN_AUDIO_DYNAMIC30S_ACAV_FORMAL_INIT_CHECKPOINT \
  ACAVCAPS_WDS_QUARTER_MANIFEST \
  HUGINN_AUDIO_DYNAMIC30S_ACAV_WARMSTART_GATE \
  HUGINN_AUDIO_DYNAMIC30S_ACAV_FORMAL_RUN_ROOT \
  HUGINN_AUDIO_DYNAMIC30S_ACAV_FORMAL_SAVE_STEPS \
  HUGINN_AUDIO_DYNAMIC30S_ACAV_FORMAL_LOGGING_STEPS \
  HUGINN_AUDIO_DYNAMIC30S_ACAV_FORMAL_REPORT_TO \
  HUGINN_AUDIO_DYNAMIC30S_ACAV_ALLOW_UNGATED_FORMAL; do
  value="${!name:-}"
  if [ -n "$value" ]; then
    printf -v quoted_value '%q' "$value"
    CMD_PREFIX="${CMD_PREFIX}${name}=${quoted_value} "
  fi
done

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 \
  -n 1 \
  -j train-acav-quarter-whisper-dyn30-fsdp8-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/train_huginn_audio_whisper_dynamic30s_acavcaps_fsdp8_warmstart_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/train_huginn_audio_whisper_dynamic30s_acavcaps_fsdp8_warmstart.sh"
