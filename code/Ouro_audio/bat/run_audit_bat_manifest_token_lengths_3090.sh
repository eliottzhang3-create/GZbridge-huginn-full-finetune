#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"

MANIFEST="${BAT_TOKEN_AUDIT_MANIFEST:-/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/manifests/stage3_ab_cde_2epoch.jsonl}"
OUTPUT_REPORT="${BAT_TOKEN_AUDIT_OUTPUT_REPORT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/bat/stage3_ab_cde_token_length_audit.json}"
PROGRESS_EVERY="${BAT_TOKEN_AUDIT_PROGRESS_EVERY:-10000}"
TAIL_RECORDS="${BAT_TOKEN_AUDIT_TAIL_RECORDS:-650000}"

case "$OUTPUT_REPORT" in
  /hpc_stor03/public|/hpc_stor03/public/*)
    echo "Refusing public audit output: $OUTPUT_REPORT" >&2
    exit 2
    ;;
esac

vc submit \
  -p pdgpu-3090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "bat-token-length-audit-3090-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/bat_token_length_audit_3090.JOB.log" \
  --cmd "BAT_TOKEN_AUDIT_MANIFEST=$(printf '%q' "$MANIFEST") BAT_TOKEN_AUDIT_OUTPUT_REPORT=$(printf '%q' "$OUTPUT_REPORT") BAT_TOKEN_AUDIT_PROGRESS_EVERY=$(printf '%q' "$PROGRESS_EVERY") BAT_TOKEN_AUDIT_TAIL_RECORDS=$(printf '%q' "$TAIL_RECORDS") bash scripts/audit_bat_manifest_token_lengths_remote.sh"
