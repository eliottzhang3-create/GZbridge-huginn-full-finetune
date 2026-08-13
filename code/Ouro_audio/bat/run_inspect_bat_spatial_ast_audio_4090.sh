#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"

CMD_PREFIX=""
for name in BAT_SPATIAL_AST_CODE_ROOT BAT_SPATIAL_AST_CHECKPOINT BAT_DATA_ROOT \
  BAT_QA_ROOT BAT_AUDIO_ROOT BAT_REVERB_ROOT BAT_QFORMER_SOURCE \
  BAT_SPATIAL_AST_AUDIO_OUTPUT BAT_SPATIAL_AST_DEVICE; do
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
  -j "inspect-bat-spatial-ast-audio-5090-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/inspect_bat_spatial_ast_audio_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/run_bat_spatial_ast_audio_remote.sh"
