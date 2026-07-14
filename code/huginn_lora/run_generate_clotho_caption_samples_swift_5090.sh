#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in CLOTHO_CAPTION_CHECKPOINT CLOTHO_CAPTION_OUTPUT_DIR CLOTHO_CAPTION_SAMPLE_COUNT CLOTHO_CAPTION_MAX_NEW_TOKENS CLOTHO_CAPTION_USE_CACHE; do
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
  -j generate-clotho-caption-samples-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/generate_clotho_caption_samples_swift_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/generate_clotho_caption_samples_swift.sh"
