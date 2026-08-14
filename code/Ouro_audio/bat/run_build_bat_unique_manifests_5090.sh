#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"

QA_ROOT="${BAT_QA_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/SpatialSoundQA/closed-end}"
OUTPUT_DIR="${BAT_UNIQUE_OUTPUT_DIR:?Set BAT_UNIQUE_OUTPUT_DIR to a private output directory}"
SHARD_COUNT="${BAT_SOURCE_SHARD_COUNT:-0}"
EXPECTED_MIN="${BAT_EXPECTED_QA_MIN:-870000}"
EXPECTED_MAX="${BAT_EXPECTED_QA_MAX:-880000}"

case "$OUTPUT_DIR" in
  /hpc_stor03/public|/hpc_stor03/public/*)
    echo "Refusing public output directory: $OUTPUT_DIR" >&2
    exit 2
    ;;
esac
if ! [[ "$SHARD_COUNT" =~ ^[0-9]+$ ]]; then
  echo "BAT_SOURCE_SHARD_COUNT must be a non-negative integer, got: $SHARD_COUNT" >&2
  exit 2
fi

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "bat-unique-manifests-5090-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/bat_unique_manifests_5090.JOB.log" \
  --cmd "BAT_QA_ROOT=$(printf '%q' "$QA_ROOT") BAT_UNIQUE_OUTPUT_DIR=$(printf '%q' "$OUTPUT_DIR") BAT_SOURCE_SHARD_COUNT=$(printf '%q' "$SHARD_COUNT") BAT_EXPECTED_QA_MIN=$(printf '%q' "$EXPECTED_MIN") BAT_EXPECTED_QA_MAX=$(printf '%q' "$EXPECTED_MAX") bash scripts/build_bat_unique_manifests_remote.sh"
