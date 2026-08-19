#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"
MANIFEST="${QWEN3_BAT_STAGE3_AB_CDE_MANIFEST:?Set QWEN3_BAT_STAGE3_AB_CDE_MANIFEST}"
REPORT="${QWEN3_BAT_STAGE3_AB_CDE_REPORT:?Set QWEN3_BAT_STAGE3_AB_CDE_REPORT}"
OUTPUT_DIR="${QWEN3_BAT_STAGE3_AB_CDE_OUTPUT_DIR:?Set QWEN3_BAT_STAGE3_AB_CDE_OUTPUT_DIR}"

CMD_PREFIX="QWEN3_BAT_STAGE3_AB_CDE_MANIFEST=$(printf '%q' "$MANIFEST") QWEN3_BAT_STAGE3_AB_CDE_REPORT=$(printf '%q' "$REPORT") QWEN3_BAT_STAGE3_AB_CDE_OUTPUT_DIR=$(printf '%q' "$OUTPUT_DIR") "
for name in QWEN3_MODEL_PATH QWEN3_BAT_PLUGIN_PATH QWEN3_BAT_STAGE3_AB_CDE_RESUME_FROM_CHECKPOINT BAT_LOCAL_CACHE_ROOT BAT_RUNTIME_MONITOR_INTERVAL_STEPS BAT_ARROW_PREWARM_REPORT BAT_MAX_STEPS_OVERRIDE MASTER_PORT; do
  value="${!name:-}"
  if [ -n "$value" ]; then
    printf -v quoted_value '%q' "$value"
    CMD_PREFIX="${CMD_PREFIX}${name}=${quoted_value} "
  fi
done

vc submit \
  -p pdgpu-3090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 \
  -j "qwen3-bat-stage3-ab-cde-3090-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/qwen3_bat_stage3_ab_cde_3090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/train_qwen3_bat_stage3_ab_cde_remote.sh"
