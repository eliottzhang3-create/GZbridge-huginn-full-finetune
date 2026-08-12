#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"

CMD_PREFIX=""
for name in OWL_BIDEPTH_ROOT OWL_REVERB_ARCHIVE OWL_PHASE1_DEEP_OUTPUT OWL_SAMPLE_NPY_COUNT OWL_AUDIO_ROOT OWL_SOURCE_ROOT OWL_PHASE1_DEEP_SHA256; do
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
  -j inspect-owl-phase1-deep-4090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/inspect_owl_phase1_deep_4090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/inspect_phase1_deep_assets.sh"
