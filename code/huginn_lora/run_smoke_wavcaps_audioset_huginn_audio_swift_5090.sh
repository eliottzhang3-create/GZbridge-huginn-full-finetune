#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in WAVCAPS_INIT_CHECKPOINT WAVCAPS_OUTPUT_DIR WAVCAPS_MAX_STEPS; do
  if [ -n "${!name:-}" ]; then
    CMD_PREFIX="${CMD_PREFIX}${name}=${!name} "
  fi
done

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j smoke-wavcaps-audioset-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/smoke_wavcaps_audioset_huginn_audio_swift_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/smoke_wavcaps_audioset_huginn_audio_swift_5090.sh"
