#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"
SOURCE="${BAT_STAGE3_SOURCE_MANIFEST:?Set BAT_STAGE3_SOURCE_MANIFEST}"
OUTPUT="${BAT_STAGE3_AB_CDE_MANIFEST:?Set BAT_STAGE3_AB_CDE_MANIFEST}"
REPORT="${BAT_STAGE3_AB_CDE_REPORT:?Set BAT_STAGE3_AB_CDE_REPORT}"

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "bat-compose-stage3-ab-cde-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/bat_compose_stage3_ab_cde.JOB.log" \
  --cmd "BAT_STAGE3_SOURCE_MANIFEST=$(printf '%q' "$SOURCE") BAT_STAGE3_AB_CDE_MANIFEST=$(printf '%q' "$OUTPUT") BAT_STAGE3_AB_CDE_REPORT=$(printf '%q' "$REPORT") BAT_STAGE3_SHUFFLE_SEED=${BAT_STAGE3_SHUFFLE_SEED:-42} bash scripts/compose_bat_stage3_ab_cde_manifest_remote.sh"
