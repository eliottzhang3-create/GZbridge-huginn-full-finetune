#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"

SOURCE_MANIFEST="${BAT_FEATURE_SOURCE_MANIFEST:?Set BAT_FEATURE_SOURCE_MANIFEST}"
CACHE_DIR="${BAT_FEATURE_CACHE_DIR:?Set BAT_FEATURE_CACHE_DIR}"
OUTPUT="${BAT_FEATURE_AUDIT_OUTPUT:?Set BAT_FEATURE_AUDIT_OUTPUT to a private output path}"
QUEUE="${BAT_AUDIT_QUEUE:-pdgpu-5090}"
FINITE_MODE="${BAT_FEATURE_AUDIT_FINITE_MODE:-all}"
SOURCE_LIMIT="${BAT_FEATURE_AUDIT_SOURCE_LIMIT:-0}"

case "$QUEUE" in
  pdgpu-5090|pdgpu-3090) ;;
  *) echo "BAT_AUDIT_QUEUE must be pdgpu-5090 or pdgpu-3090, got: $QUEUE" >&2; exit 2 ;;
esac
case "$OUTPUT" in
  /hpc_stor03/public|/hpc_stor03/public/*)
    echo "Refusing public output path: $OUTPUT" >&2
    exit 2
    ;;
esac

vc submit \
  -p "$QUEUE" \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "bat-ast-cache-audit-${QUEUE#pdgpu-}-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/bat_ast_cache_audit_${QUEUE#pdgpu-}.JOB.log" \
  --cmd "BAT_FEATURE_SOURCE_MANIFEST=$(printf '%q' "$SOURCE_MANIFEST") BAT_FEATURE_CACHE_DIR=$(printf '%q' "$CACHE_DIR") BAT_FEATURE_AUDIT_OUTPUT=$(printf '%q' "$OUTPUT") BAT_FEATURE_AUDIT_FINITE_MODE=$(printf '%q' "$FINITE_MODE") BAT_FEATURE_AUDIT_SOURCE_LIMIT=$(printf '%q' "$SOURCE_LIMIT") bash scripts/audit_bat_spatial_ast_bf16_cache_remote.sh"
