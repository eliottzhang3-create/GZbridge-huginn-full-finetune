#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in CLOTHOAQA_DATASET_ROOT CLOTHOAQA_SOURCE_MANIFEST CLOTHOAQA_TRAIN_MANIFEST; do
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
  -j prepare-clothoaqa-losatok-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/prepare_clotho_aqa_huginn_losatok_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/prepare_clotho_aqa_huginn_losatok_swift_dataset.sh"
