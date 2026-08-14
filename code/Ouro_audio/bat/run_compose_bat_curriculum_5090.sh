#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"
STAGE1="${BAT_STAGE1_MANIFEST:?Set BAT_STAGE1_MANIFEST to a private Stage-I manifest}"
STAGE2="${BAT_STAGE2_MANIFEST:?Set BAT_STAGE2_MANIFEST to a private Stage-II manifest}"
STAGE3="${BAT_STAGE3_MANIFEST:?Set BAT_STAGE3_MANIFEST to a private Stage-III manifest}"
OUTPUT="${BAT_CURRICULUM_MANIFEST:?Set BAT_CURRICULUM_MANIFEST to a private output path}"
REPORT="${BAT_CURRICULUM_REPORT:?Set BAT_CURRICULUM_REPORT to a private output path}"

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "bat-curriculum-compose-5090-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/bat_curriculum_compose_5090.JOB.log" \
  --cmd "BAT_STAGE1_MANIFEST=$(printf '%q' "$STAGE1") BAT_STAGE2_MANIFEST=$(printf '%q' "$STAGE2") BAT_STAGE3_MANIFEST=$(printf '%q' "$STAGE3") BAT_CURRICULUM_MANIFEST=$(printf '%q' "$OUTPUT") BAT_CURRICULUM_REPORT=$(printf '%q' "$REPORT") bash scripts/compose_bat_curriculum_remote.sh"
