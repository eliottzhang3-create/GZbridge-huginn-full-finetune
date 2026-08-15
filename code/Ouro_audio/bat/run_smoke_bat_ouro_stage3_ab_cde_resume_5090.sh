#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"

SOURCE_MANIFEST="${BAT_STAGE3_AB_CDE_SMOKE_SOURCE_MANIFEST:?Set BAT_STAGE3_AB_CDE_SMOKE_SOURCE_MANIFEST}"
ROOT_DIR="${BAT_STAGE3_AB_CDE_SMOKE_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/bat/stage3_ab_cde_resume_smoke-$(date +%Y%m%d-%H%M%S)}"
case "$ROOT_DIR" in
  /hpc_stor03/public|/hpc_stor03/public/*) echo "Refusing public output" >&2; exit 2;;
esac

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 \
  -j "bat-stage3-resume-smoke-5090-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/bat_stage3_resume_smoke_5090.JOB.log" \
  --cmd "BAT_STAGE3_AB_CDE_SMOKE_SOURCE_MANIFEST=$(printf '%q' "$SOURCE_MANIFEST") BAT_STAGE3_AB_CDE_SMOKE_ROOT=$(printf '%q' "$ROOT_DIR") bash scripts/run_bat_ouro_stage3_ab_cde_resume_smoke_remote.sh"
