#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in \
  HUGINN_AUDIO_DYNAMIC30S_ACAV_WARMSTART_CHECKPOINT \
  ACAVCAPS_FLAT_MANIFEST \
  ACAVCAPS_FLAT_MAX_TARS \
  HUGINN_AUDIO_DYNAMIC30S_ACAV_SMOKE_ROOT; do
  value="${!name:-}"
  if [ -n "$value" ]; then
    printf -v quoted_value '%q' "$value"
    CMD_PREFIX="${CMD_PREFIX}${name}=${quoted_value} "
  fi
done

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 128G -g 8 \
  -n 1 \
  -j smoke-acav-flat-whisper-warmstart-fsdp8-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/smoke_huginn_audio_whisper_dynamic30s_acavcaps_fsdp8_warmstart_save_resume_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/smoke_huginn_audio_whisper_dynamic30s_acavcaps_fsdp8_warmstart_save_resume.sh"
