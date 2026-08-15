#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"

REVERB_ROOT="${BAT_REVERB_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/SpatialSoundQA/mp3d_reverb}"
OUTPUT="${BAT_RIR_AUDIT_OUTPUT:?Set BAT_RIR_AUDIT_OUTPUT to a private output path}"
SAMPLE_RATE="${BAT_RIR_SAMPLE_RATE:-32000}"
TARGET_SECONDS="${BAT_RIR_TARGET_SECONDS:-2.0}"
LIMIT="${BAT_RIR_AUDIT_LIMIT:-0}"
PREVIEW_COUNT="${BAT_RIR_AUDIT_PREVIEW_COUNT:-20}"

case "$OUTPUT" in
  /hpc_stor03/public|/hpc_stor03/public/*)
    echo "Refusing public output path: $OUTPUT" >&2
    exit 2
    ;;
esac

vc submit \
  -p pdgpu-3090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "bat-rir-length-audit-3090-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/bat_rir_length_audit_3090.JOB.log" \
  --cmd "BAT_REVERB_ROOT=$(printf '%q' "$REVERB_ROOT") BAT_RIR_AUDIT_OUTPUT=$(printf '%q' "$OUTPUT") BAT_RIR_SAMPLE_RATE=$(printf '%q' "$SAMPLE_RATE") BAT_RIR_TARGET_SECONDS=$(printf '%q' "$TARGET_SECONDS") BAT_RIR_AUDIT_LIMIT=$(printf '%q' "$LIMIT") BAT_RIR_AUDIT_PREVIEW_COUNT=$(printf '%q' "$PREVIEW_COUNT") bash scripts/audit_bat_rir_lengths_remote.sh"
