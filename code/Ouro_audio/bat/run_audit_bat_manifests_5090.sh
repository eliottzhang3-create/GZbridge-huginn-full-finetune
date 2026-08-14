#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"
STAGE2="${BAT_STAGE2_MANIFEST:?Set BAT_STAGE2_MANIFEST to a private Stage-II manifest}"
STAGE3="${BAT_STAGE3_MANIFEST:?Set BAT_STAGE3_MANIFEST to a private Stage-III manifest}"
AUDIT_ROOT="${BAT_MANIFEST_AUDIT_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/bat/manifest_audits-$(date +%Y%m%d-%H%M%S)}"

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "bat-manifest-audit-5090-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/bat_manifest_audit_5090.JOB.log" \
  --cmd "BAT_STAGE2_MANIFEST=$(printf '%q' "$STAGE2") BAT_STAGE3_MANIFEST=$(printf '%q' "$STAGE3") BAT_MANIFEST_AUDIT_ROOT=$(printf '%q' "$AUDIT_ROOT") bash scripts/audit_bat_manifests_remote.sh"
