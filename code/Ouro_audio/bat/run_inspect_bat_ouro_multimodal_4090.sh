#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"

CMD_PREFIX=""
for name in OURO_MODEL_PATH OURO_BAT_PLUGIN_PATH BAT_SPATIAL_AST_CODE_ROOT \
  BAT_SPATIAL_AST_CHECKPOINT BAT_QFORMER_SOURCE BAT_AUDIO_ROOT BAT_REVERB_ROOT \
  BAT_QA_ROOT OURO_BAT_MULTIMODAL_AUDIT_OUTPUT OURO_BAT_MULTIMODAL_DEVICE; do
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
  -j "inspect-bat-ouro-multimodal-5090-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/inspect_bat_ouro_multimodal_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/run_bat_ouro_multimodal_audit_remote.sh"
