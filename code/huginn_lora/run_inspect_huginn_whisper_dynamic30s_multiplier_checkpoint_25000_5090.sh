#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in HUGINN_AUDIO_DYNAMIC30S_MULTIPLIER_CHECKPOINT_25000 HUGINN_AUDIO_DYNAMIC30S_MULTIPLIER_CHECKPOINT_25000_REPORT; do
  value="${!name:-}"
  if [ -n "$value" ]; then
    printf -v quoted_value '%q' "$value"
    CMD_PREFIX="${CMD_PREFIX}${name}=${quoted_value} "
  fi
done

vc submit \
  -p pdgpu-4090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j inspect-whisper-dyn30-checkpoint25000-4090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/inspect_huginn_whisper_dynamic30s_multiplier_checkpoint_25000_4090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/inspect_huginn_whisper_dynamic30s_multiplier_checkpoint_25000_5090.sh"
