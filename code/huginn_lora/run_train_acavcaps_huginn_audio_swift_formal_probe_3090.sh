#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p log

CMD_PREFIX=""
if [ -n "${FORMAL_MANIFEST:-}" ]; then
  CMD_PREFIX="${CMD_PREFIX}FORMAL_MANIFEST=${FORMAL_MANIFEST} "
fi
if [ -n "${FORMAL_PROBE_NAME:-}" ]; then
  CMD_PREFIX="${CMD_PREFIX}FORMAL_PROBE_NAME=${FORMAL_PROBE_NAME} "
fi
if [ -n "${FORMAL_MAX_STEPS:-}" ]; then
  CMD_PREFIX="${CMD_PREFIX}FORMAL_MAX_STEPS=${FORMAL_MAX_STEPS} "
fi
if [ -n "${FORMAL_BATCH_SIZE:-}" ]; then
  CMD_PREFIX="${CMD_PREFIX}FORMAL_BATCH_SIZE=${FORMAL_BATCH_SIZE} "
fi
if [ -n "${FORMAL_GRADIENT_ACCUMULATION_STEPS:-}" ]; then
  CMD_PREFIX="${CMD_PREFIX}FORMAL_GRADIENT_ACCUMULATION_STEPS=${FORMAL_GRADIENT_ACCUMULATION_STEPS} "
fi

vc submit \
  -p pdgpu-3090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j train-acavcaps-formal-probe-3090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/train_acavcaps_huginn_audio_swift_formal_probe_3090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/train_acavcaps_huginn_audio_swift_formal_probe.sh"
