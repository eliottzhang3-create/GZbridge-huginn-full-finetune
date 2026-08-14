#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"

SOURCE_MANIFEST="${BAT_UNIQUE_SOURCE_MANIFEST:?Set BAT_UNIQUE_SOURCE_MANIFEST}"
OUTPUT_DIR="${BAT_SOURCE_SHARD_OUTPUT_DIR:?Set BAT_SOURCE_SHARD_OUTPUT_DIR to a private output directory}"
SHARD_COUNT="${BAT_SOURCE_SHARD_COUNT:?Set BAT_SOURCE_SHARD_COUNT to a positive integer}"

case "$OUTPUT_DIR" in
  /hpc_stor03/public|/hpc_stor03/public/*)
    echo "Refusing public output directory: $OUTPUT_DIR" >&2
    exit 2
    ;;
esac
if ! [[ "$SHARD_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "BAT_SOURCE_SHARD_COUNT must be a positive integer, got: $SHARD_COUNT" >&2
  exit 2
fi

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "bat-source-split-5090-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/bat_source_split_5090.JOB.log" \
  --cmd "BAT_UNIQUE_SOURCE_MANIFEST=$(printf '%q' "$SOURCE_MANIFEST") BAT_SOURCE_SHARD_OUTPUT_DIR=$(printf '%q' "$OUTPUT_DIR") BAT_SOURCE_SHARD_COUNT=$(printf '%q' "$SHARD_COUNT") bash scripts/split_bat_source_manifest_remote.sh"
