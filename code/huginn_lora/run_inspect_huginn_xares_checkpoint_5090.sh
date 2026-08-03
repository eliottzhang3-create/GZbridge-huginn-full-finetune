#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in HUGINN_XARES_CHECKPOINT HUGINN_XARES_EXPECTED_STEP HUGINN_XARES_EXPECTED_PHASE HUGINN_XARES_WORLD_SIZE HUGINN_XARES_MODEL_CONFIG HUGINN_XARES_INSPECT_OUTPUT_DIR HUGINN_XARES_SKIP_TENSOR_PROBES; do
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
  -j inspect-huginn-xares-checkpoint-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/inspect_huginn_xares_checkpoint_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/inspect_huginn_xares_checkpoint.sh"
