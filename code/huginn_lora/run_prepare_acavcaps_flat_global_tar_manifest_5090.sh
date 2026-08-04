#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in ACAVCAPS_FULL_SOURCE_MANIFEST ACAVCAPS_FLAT_MANIFEST ACAVCAPS_FLAT_TAR_SHUFFLE_SEED ACAVCAPS_WDS_BUFFER_SIZE; do
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
  -j prepare-acav-flat-global-tar-4090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/prepare_acavcaps_flat_global_tar_manifest_4090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/prepare_acavcaps_flat_global_tar_manifest_5090.sh"
