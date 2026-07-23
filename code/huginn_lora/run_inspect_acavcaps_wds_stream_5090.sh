#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in ACAVCAPS_WDS_STREAM_MANIFEST ACAVCAPS_WDS_STREAM_BUFFER_SIZE ACAVCAPS_WDS_STREAM_MAX_TARS_PER_STAGE ACAVCAPS_WDS_STREAM_DECODE_EVERY ACAVCAPS_WDS_STREAM_LOG_EVERY; do
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
  -j inspect-acav-wds-stream-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/inspect_acavcaps_wds_stream_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/inspect_acavcaps_wds_stream.sh"
