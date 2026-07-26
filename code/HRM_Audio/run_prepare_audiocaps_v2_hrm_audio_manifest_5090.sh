#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in \
  HRM_AUDIOCAPS_SOURCE_MANIFEST \
  HRM_AUDIOCAPS_SOURCE_STATS \
  HRM_AUDIOCAPS_TRAIN_MANIFEST \
  HRM_AUDIOCAPS_EXPECTED_RECORD_COUNT \
  HRM_AUDIOCAPS_MANIFEST_OVERWRITE; do
  value="${!name:-}"
  if [ -n "$value" ]; then
    printf -v quoted_value '%q' "$value"
    CMD_PREFIX="${CMD_PREFIX}${name}=${quoted_value} "
  fi
done

# This is a CPU/I/O preparation job.  The established cluster submission
# route currently uses the 5090 image/partition for a reproducible filesystem
# and conda environment; the Python program itself never initializes CUDA.
vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j prepare-hrm-audiocaps-manifest-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/prepare_audiocaps_v2_hrm_audio_manifest_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/prepare_audiocaps_v2_hrm_audio_manifest.sh"
