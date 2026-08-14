#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"
MANIFEST="${BAT_CURRICULUM_MANIFEST:?Set BAT_CURRICULUM_MANIFEST}"
REPORT="${BAT_CURRICULUM_REPORT:?Set BAT_CURRICULUM_REPORT}"
OUTPUT="${BAT_ARROW_SCHEMA_REPORT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/bat/curriculum_arrow_schema_audit-$(date +%Y%m%d-%H%M%S).json}"

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "bat-arrow-schema-audit-5090-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/bat_arrow_schema_audit_5090.JOB.log" \
  --cmd "BAT_CURRICULUM_MANIFEST=$(printf '%q' "$MANIFEST") BAT_CURRICULUM_REPORT=$(printf '%q' "$REPORT") BAT_ARROW_SCHEMA_REPORT=$(printf '%q' "$OUTPUT") bash scripts/audit_bat_jsonl_arrow_schema_remote.sh"
