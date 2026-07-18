#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in \
  AUDIOCAPS_TRAIN_MANIFEST \
  CLOTHO_CAPTION_ROOT \
  CLOTHO_CAPTION_MANIFEST \
  AUDIOCAPS_CLOTHO_ARTIFACT_DIR \
  CLOTHO_CAPTION_SWIFT_MANIFEST \
  AUDIOCAPS_CLOTHO_MIXTURE_MANIFEST; do
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
  -j prepare-audiocaps-clotho-mixture-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/prepare_audiocaps_clotho_caption_mixture_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/prepare_audiocaps_clotho_caption_mixture.sh"
