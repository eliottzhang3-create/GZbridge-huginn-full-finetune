#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in SWIFT_RETRIEVAL_CHECKPOINTS SWIFT_RETRIEVAL_OUTPUT_DIR SWIFT_RETRIEVAL_SAMPLE_COUNT SWIFT_RETRIEVAL_PLUGIN_PATH HUGINN_AUDIO_FSDP_EVAL_EXPORT_DIR; do
  value="${!name:-}"
  if [ -n "$value" ]; then
    printf -v quoted_value '%q' "$value"
    CMD_PREFIX="${CMD_PREFIX}${name}=${quoted_value} "
  fi
done

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j eval-swift-audio-retrieval-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/eval_huginn_audio_text_retrieval_swift_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/eval_huginn_audio_text_retrieval_swift.sh"
