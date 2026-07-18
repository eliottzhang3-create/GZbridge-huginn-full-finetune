#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in \
  CONTINUATION_INIT_CHECKPOINT \
  CLOTHO_CAPTION_ROOT \
  CLOTHO_CAPTION_MANIFEST \
  CLOTHO_AQA_ROOT \
  CLOTHO_AQA_MANIFEST \
  CLOTHO_CONTINUATION_ARTIFACT_DIR \
  CONTINUATION_CHECKPOINT_REPORT \
  CLOTHO_CONTINUATION_INSPECT_REPORT; do
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
  -j inspect-clotho-continuation-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/inspect_clotho_continuation_inputs_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/inspect_clotho_continuation_inputs.sh"
