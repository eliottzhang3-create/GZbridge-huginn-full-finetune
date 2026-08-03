#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in HUGINN_XARES_CONDA_ENV HUGINN_XARES_ROOT HUGINN_XARES_CHECKPOINT HUGINN_XARES_PLUGIN_PATH HUGINN_XARES_VOXCELEB1_ROOT HUGINN_XARES_REAL_SAMPLE_COUNT HUGINN_XARES_ENCODER_SMOKE_OUTPUT_DIR HUGINN_XARES_CUDA_VISIBLE_DEVICES; do
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
  -j smoke-huginn-xares-encoder-4090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/smoke_huginn_xares_encoder_4090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/smoke_huginn_whisper_xares_encoder.sh"
