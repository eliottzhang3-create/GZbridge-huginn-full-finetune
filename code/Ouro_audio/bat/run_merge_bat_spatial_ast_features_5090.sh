#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"

SOURCE_SHARD_DIR="${BAT_SOURCE_SHARD_DIR:?Set BAT_SOURCE_SHARD_DIR}"
FEATURE_ROOT="${BAT_FEATURE_OUTPUT_ROOT:?Set BAT_FEATURE_OUTPUT_ROOT to a private output directory}"
SHARD_COUNT="${BAT_SOURCE_SHARD_COUNT:-16}"

case "$FEATURE_ROOT" in
  /hpc_stor03/public|/hpc_stor03/public/*)
    echo "Refusing public output directory: $FEATURE_ROOT" >&2
    exit 2
    ;;
esac

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "bat-ast-feature-merge-5090-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/bat_ast_feature_merge_5090.JOB.log" \
  --cmd "BAT_SOURCE_SHARD_DIR=$(printf '%q' "$SOURCE_SHARD_DIR") BAT_FEATURE_OUTPUT_ROOT=$(printf '%q' "$FEATURE_ROOT") BAT_SOURCE_SHARD_COUNT=$(printf '%q' "$SHARD_COUNT") bash scripts/merge_bat_spatial_ast_feature_indices_remote.sh"
