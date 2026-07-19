#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in \
  AUDIOCAPS_FULL_FSDP_TRAIN_MANIFEST \
  AUDIOCAPS_FULL_FSDP_OUTPUT_DIR \
  AUDIOCAPS_FULL_FSDP_LOGGING_DIR \
  AUDIOCAPS_FULL_FSDP_MIN_FREE_GB; do
  value="${!name:-}"
  if [ -n "$value" ]; then
    printf -v quoted_value '%q' "$value"
    CMD_PREFIX="${CMD_PREFIX}${name}=${quoted_value} "
  fi
done

vc submit -p pdgpu-5090 -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 -c 28 -m 224G -g 7 -n 1 -j train-audiocaps-full-fsdp7-e2-5090-$(date +%m%d%H%M) -d "$SCRIPT_DIR" JOB=1:1 "$SCRIPT_DIR/log/train_audiocaps_v2_huginn_audio_swift_full_fsdp8_5090.JOB.log" --cmd "${CMD_PREFIX}bash scripts/train_audiocaps_v2_huginn_audio_swift_full_fsdp8.sh"
