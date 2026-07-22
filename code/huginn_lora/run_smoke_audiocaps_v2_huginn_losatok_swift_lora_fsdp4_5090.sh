#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in LOSATOK_FSDP4_SMOKE_MANIFEST LOSATOK_FSDP4_SMOKE_OUTPUT_DIR LOSATOK_FSDP4_SMOKE_LOGGING_DIR; do
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
  -j smoke-losatok-lora-fsdp4-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/smoke_audiocaps_v2_huginn_losatok_swift_lora_fsdp4_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/smoke_audiocaps_v2_huginn_losatok_swift_lora_fsdp4.sh"
