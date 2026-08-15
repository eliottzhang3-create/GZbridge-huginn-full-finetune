#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"

MANIFEST="${BAT_STAGE3_AB_CDE_MANIFEST:?Set BAT_STAGE3_AB_CDE_MANIFEST}"
REPORT="${BAT_STAGE3_AB_CDE_REPORT:?Set BAT_STAGE3_AB_CDE_REPORT}"
OUTPUT_REPORT="${BAT_STAGE3_AB_CDE_MANIFEST_AUDIT_REPORT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/bat/stage3_ab_cde_manifest_audit.json}"
case "$OUTPUT_REPORT" in
  /hpc_stor03/public|/hpc_stor03/public/*) echo "Refusing public output" >&2; exit 2;;
esac

vc submit \
  -p pdgpu-3090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "bat-stage3-manifest-audit-3090-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/bat_stage3_manifest_audit_3090.JOB.log" \
  --cmd "BAT_STAGE3_AB_CDE_MANIFEST=$(printf '%q' "$MANIFEST") BAT_STAGE3_AB_CDE_REPORT=$(printf '%q' "$REPORT") BAT_STAGE3_AB_CDE_MANIFEST_AUDIT_REPORT=$(printf '%q' "$OUTPUT_REPORT") bash scripts/audit_bat_stage3_ab_cde_manifest_remote.sh"
