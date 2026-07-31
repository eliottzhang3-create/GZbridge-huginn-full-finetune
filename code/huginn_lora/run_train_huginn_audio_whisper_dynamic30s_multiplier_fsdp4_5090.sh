#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in \
  HUGINN_MULTIPLIER_POOL_REGISTRY \
  HUGINN_MULTIPLIER_POOL_AUDIT \
  HUGINN_MULTIPLIER_RESUME_SMOKE_GATE \
  HUGINN_MULTIPLIER_FORMAL_RUN_ROOT \
  HUGINN_MULTIPLIER_FORMAL_RESUME_CHECKPOINT \
  HUGINN_MULTIPLIER_FORMAL_REPORT_TO; do
  value="${!name:-}"
  if [ -n "$value" ]; then
    printf -v quoted_value '%q' "$value"
    CMD_PREFIX="${CMD_PREFIX}${name}=${quoted_value} "
  fi
done

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 128G -g 4 \
  -n 1 \
  -j train-whisper-dyn30-multiplier-fsdp4-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/train_huginn_audio_whisper_dynamic30s_multiplier_fsdp4_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/train_huginn_audio_whisper_dynamic30s_multiplier_fsdp4.sh"
