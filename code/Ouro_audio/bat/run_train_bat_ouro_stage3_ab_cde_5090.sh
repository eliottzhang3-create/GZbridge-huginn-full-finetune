#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"
MANIFEST="${BAT_STAGE3_AB_CDE_MANIFEST:?Set BAT_STAGE3_AB_CDE_MANIFEST}"
REPORT="${BAT_STAGE3_AB_CDE_REPORT:?Set BAT_STAGE3_AB_CDE_REPORT}"
OUTPUT_DIR="${BAT_STAGE3_AB_CDE_OUTPUT_DIR:?Set BAT_STAGE3_AB_CDE_OUTPUT_DIR}"
RESUME_CHECKPOINT="${BAT_STAGE3_AB_CDE_RESUME_FROM_CHECKPOINT:-}"
COMPILE_THREADS="${BAT_OURO_COMPILE_THREADS:-2}"

if [[ "${BAT_GPU_COUNT:-8}" != "8" ]]; then
  echo "Stage-III A+B -> C+D+E route requires 8 GPUs" >&2
  exit 2
fi

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 \
  -j "bat-ouro-stage3-ab-cde-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/bat_ouro_stage3_ab_cde_5090.JOB.log" \
  --cmd "BAT_STAGE3_AB_CDE_MANIFEST=$(printf '%q' "$MANIFEST") BAT_STAGE3_AB_CDE_REPORT=$(printf '%q' "$REPORT") BAT_STAGE3_AB_CDE_OUTPUT_DIR=$(printf '%q' "$OUTPUT_DIR") BAT_STAGE3_AB_CDE_RESUME_FROM_CHECKPOINT=$(printf '%q' "$RESUME_CHECKPOINT") BAT_OURO_COMPILE_THREADS=$(printf '%q' "$COMPILE_THREADS") BAT_LAUNCH_MODE=ddp bash scripts/train_bat_ouro_stage3_ab_cde_remote.sh"
