#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p log

CMD_PREFIX=""
if [ -n "${FORMAL_OUTPUT_DIR:-}" ]; then
  CMD_PREFIX="${CMD_PREFIX}FORMAL_OUTPUT_DIR=${FORMAL_OUTPUT_DIR} "
fi
if [ -n "${FORMAL_LOGGING_DIR:-}" ]; then
  CMD_PREFIX="${CMD_PREFIX}FORMAL_LOGGING_DIR=${FORMAL_LOGGING_DIR} "
fi
if [ -n "${FORMAL_MAX_STEPS:-}" ]; then
  CMD_PREFIX="${CMD_PREFIX}FORMAL_MAX_STEPS=${FORMAL_MAX_STEPS} "
fi
if [ -n "${FORMAL_SAVE_STEPS:-}" ]; then
  CMD_PREFIX="${CMD_PREFIX}FORMAL_SAVE_STEPS=${FORMAL_SAVE_STEPS} "
fi
if [ -n "${FORMAL_LOGGING_STEPS:-}" ]; then
  CMD_PREFIX="${CMD_PREFIX}FORMAL_LOGGING_STEPS=${FORMAL_LOGGING_STEPS} "
fi
if [ -n "${FORMAL_RESUME_FROM_CHECKPOINT:-}" ]; then
  CMD_PREFIX="${CMD_PREFIX}FORMAL_RESUME_FROM_CHECKPOINT=${FORMAL_RESUME_FROM_CHECKPOINT} "
fi
if [ -n "${HUGINN_AUDIO_TARFILE_CACHE_LIMIT:-}" ]; then
  CMD_PREFIX="${CMD_PREFIX}HUGINN_AUDIO_TARFILE_CACHE_LIMIT=${HUGINN_AUDIO_TARFILE_CACHE_LIMIT} "
fi

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j train-acavcaps-formal-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/train_acavcaps_huginn_audio_swift_formal_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/train_acavcaps_huginn_audio_swift_formal_5090.sh"
