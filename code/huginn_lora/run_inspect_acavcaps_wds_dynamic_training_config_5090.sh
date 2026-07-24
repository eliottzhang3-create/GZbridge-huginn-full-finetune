#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in ACAVCAPS_WDS_FULL_MANIFEST ACAVCAPS_WDS_FULL_STATS ACAVCAPS_NUM_TRAIN_EPOCHS; do
  value="${!name:-}"
  if [ -n "$value" ]; then
    printf -v quoted_value '%q' "$value"
    CMD_PREFIX="${CMD_PREFIX}${name}=${quoted_value} "
  fi
done

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 4 -m 16G -g 1 \
  -n 1 \
  -j inspect-acav-dynamic-config-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/inspect_acavcaps_wds_dynamic_training_config_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/inspect_acavcaps_wds_dynamic_training_config_5090.sh"
