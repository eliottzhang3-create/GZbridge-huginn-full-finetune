#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"
STAGE="${BAT_STAGE:?Set BAT_STAGE=I, II or III}"
OUTPUT="${BAT_MANIFEST_OUTPUT:-/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/manifests/stage${STAGE}_train.jsonl}"
LIMIT="${BAT_MANIFEST_LIMIT:-0}"
SEED="${BAT_MANIFEST_SEED:-42}"
QA_ROOT="${BAT_QA_ROOT:-}"

if ! [[ "$LIMIT" =~ ^[0-9]+$ ]]; then
  echo "BAT_MANIFEST_LIMIT must be a non-negative integer, got: $LIMIT" >&2
  exit 2
fi
if ! [[ "$SEED" =~ ^-?[0-9]+$ ]]; then
  echo "BAT_MANIFEST_SEED must be an integer, got: $SEED" >&2
  exit 2
fi

QA_ROOT_ASSIGN=""
if [ -n "$QA_ROOT" ]; then
  QA_ROOT_ASSIGN="BAT_QA_ROOT=$(printf '%q' "$QA_ROOT") "
fi

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "bat-manifest-${STAGE}-5090-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/bat_manifest_${STAGE}_5090.JOB.log" \
  --cmd "${QA_ROOT_ASSIGN}BAT_STAGE=$(printf '%q' "$STAGE") BAT_MANIFEST_OUTPUT=$(printf '%q' "$OUTPUT") BAT_MANIFEST_LIMIT=$(printf '%q' "$LIMIT") BAT_MANIFEST_SEED=$(printf '%q' "$SEED") bash scripts/prepare_bat_manifest_remote.sh"
