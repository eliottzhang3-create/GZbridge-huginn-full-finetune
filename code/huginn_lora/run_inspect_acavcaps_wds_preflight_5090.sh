#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p log

CMD_PREFIX=""
for name in ACAVCAPS_WDS_PRIVATE_ROOT ACAVCAPS_WDS_MANIFEST_OUT ACAVCAPS_WDS_SEED ACAVCAPS_WDS_SAMPLE_SHUFFLE_BUFFER ACAVCAPS_WDS_SCAN_MODE ACAVCAPS_WDS_SCAN_TARS_PER_STAGE; do
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
  -j inspect-acav-wds-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/inspect_acavcaps_wds_preflight_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/inspect_acavcaps_wds_preflight.sh"
