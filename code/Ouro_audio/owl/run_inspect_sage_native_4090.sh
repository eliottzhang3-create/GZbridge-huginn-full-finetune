#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"

CMD_PREFIX=""
for name in OWL_SAGE_PATH OWL_BIDEPTH_ROOT OWL_SOURCE_ROOT OWL_AUDIO_ROOT OWL_SAGE_NATIVE_OUTPUT OWL_SAGE_DEVICE OWL_SAGE_REAL_SAMPLE_COUNT; do
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
  -j inspect-owl-sage-native-4090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/inspect_owl_sage_native_4090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/inspect_sage_native.sh"
