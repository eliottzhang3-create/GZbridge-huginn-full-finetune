#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"
MANIFEST="${BAT_CURRICULUM_MANIFEST:?Set BAT_CURRICULUM_MANIFEST}"
REPORT="${BAT_CURRICULUM_REPORT:?Set BAT_CURRICULUM_REPORT}"
AUDIT_REPORT="${BAT_CURRICULUM_AUDIT_REPORT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/bat/curriculum_manifest_audit-$(date +%Y%m%d-%H%M%S).json}"

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "bat-curriculum-audit-5090-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/bat_curriculum_audit_5090.JOB.log" \
  --cmd "BAT_CURRICULUM_MANIFEST=$(printf '%q' "$MANIFEST") BAT_CURRICULUM_REPORT=$(printf '%q' "$REPORT") BAT_CURRICULUM_AUDIT_REPORT=$(printf '%q' "$AUDIT_REPORT") bash scripts/audit_bat_curriculum_remote.sh"
