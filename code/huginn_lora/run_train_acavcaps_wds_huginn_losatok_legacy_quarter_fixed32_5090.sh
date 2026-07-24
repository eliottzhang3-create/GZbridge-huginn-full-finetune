#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in \
  LOSATOK_LEGACY_ACAV_WDS_INIT_CHECKPOINT \
  LOSATOK_LEGACY_ACAV_WDS_QUARTER_OUTPUT_DIR \
  LOSATOK_LEGACY_ACAV_WDS_QUARTER_LOGGING_DIR \
  LOSATOK_LEGACY_ACAV_WDS_QUARTER_LOGGING_STEPS \
  LOSATOK_LEGACY_ACAV_WDS_QUARTER_REPORT_TO \
  ACAVCAPS_WDS_QUARTER_MANIFEST \
  ACAVCAPS_WDS_BUFFER_SIZE; do
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
  -j train-acav-wds-losatok-quarter-fixed32-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/train_acavcaps_wds_huginn_losatok_legacy_quarter_fixed32_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/train_acavcaps_wds_huginn_losatok_legacy_quarter_fixed32_5090.sh"
