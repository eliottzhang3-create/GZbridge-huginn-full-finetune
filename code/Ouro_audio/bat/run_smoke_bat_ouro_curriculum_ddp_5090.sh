#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"
STAGE1="${BAT_STAGE1_MANIFEST:?Set BAT_STAGE1_MANIFEST}"
STAGE2="${BAT_STAGE2_MANIFEST:?Set BAT_STAGE2_MANIFEST}"
STAGE3="${BAT_STAGE3_MANIFEST:?Set BAT_STAGE3_MANIFEST}"
ROOT="${BAT_CURRICULUM_SMOKE_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/bat/curriculum_smoke-$(date +%Y%m%d-%H%M%S)}"

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 \
  -j "bat-ouro-curriculum-smoke-5090-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/bat_ouro_curriculum_smoke_5090.JOB.log" \
  --cmd "BAT_STAGE1_MANIFEST=$(printf '%q' "$STAGE1") BAT_STAGE2_MANIFEST=$(printf '%q' "$STAGE2") BAT_STAGE3_MANIFEST=$(printf '%q' "$STAGE3") BAT_CURRICULUM_SMOKE_ROOT=$(printf '%q' "$ROOT") bash scripts/smoke_bat_ouro_curriculum_ddp_remote.sh"
