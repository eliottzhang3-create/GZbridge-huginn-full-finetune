#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in \
  HUGINN_AUDIO_DYNAMIC90S_FORMAL_RUN_ROOT \
  HUGINN_AUDIO_DYNAMIC90S_FORMAL_RESUME_CHECKPOINT \
  HUGINN_AUDIO_DYNAMIC90S_FORMAL_MIN_FREE_GB \
  HUGINN_AUDIO_DYNAMIC90S_FORMAL_REPORT_TO \
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
  -j train-whisper-dyn90-multitask-4000h-fsdp4-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/train_huginn_audio_whisper_dynamic90s_multitask_fsdp4_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/train_huginn_audio_whisper_dynamic90s_multitask_fsdp4.sh"
