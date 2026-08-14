#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"
REPORT="${BAT_CURRICULUM_REPORT:?Set BAT_CURRICULUM_REPORT to the private curriculum report}"

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "bat-curriculum-callback-5090-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/bat_curriculum_callback_5090.JOB.log" \
  --cmd "BAT_CURRICULUM_REPORT=$(printf '%q' "$REPORT") bash scripts/run_bat_curriculum_callback_smoke_remote.sh"
