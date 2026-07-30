#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in \
  HUGINN_AUDIO_DYNAMIC90S_CHECKPOINT_RUN_ROOT \
  HUGINN_DYNAMIC90S_POOL_REGISTRY \
  HUGINN_DYNAMIC90S_REAL_DATA_CHAIN_REPORT \
  HUGINN_DYNAMIC90S_SAMPLER_REPORT \
  HUGINN_DYNAMIC90S_MIXTURE_SEED; do
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
  -j smoke-whisper-dyn30-ckpt-resume-fsdp4-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/smoke_huginn_audio_whisper_dynamic90s_checkpoint_resume_fsdp4_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/smoke_huginn_audio_whisper_dynamic90s_checkpoint_resume_fsdp4.sh"
