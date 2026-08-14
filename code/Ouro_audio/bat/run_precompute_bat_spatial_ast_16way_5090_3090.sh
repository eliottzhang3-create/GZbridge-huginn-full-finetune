#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"

SOURCE_SHARD_DIR="${BAT_SOURCE_SHARD_DIR:?Set BAT_SOURCE_SHARD_DIR containing shard-000-of-016.jsonl ... shard-015-of-016.jsonl}"
FEATURE_ROOT="${BAT_FEATURE_OUTPUT_ROOT:?Set BAT_FEATURE_OUTPUT_ROOT to a private feature-cache root}"
SPATIAL_AST_ROOT="${BAT_SPATIAL_AST_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/Spatial-AST}"
SPATIAL_AST_CHECKPOINT="${BAT_SPATIAL_AST_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/models/BAT/SpatialAST/finetuned.pth}"
AUDIO_ROOT="${BAT_AUDIO_ROOT:-/hpc_stor03/public/shared/data/raa/AudioSet}"
REVERB_ROOT="${BAT_REVERB_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/SpatialSoundQA/mp3d_reverb}"
BATCH_SIZE="${BAT_PRECOMPUTE_BATCH_SIZE:-4}"
CHUNK_SIZE="${BAT_PRECOMPUTE_CHUNK_SIZE:-512}"

case "$FEATURE_ROOT" in
  /hpc_stor03/public|/hpc_stor03/public/*)
    echo "Refusing public feature output root: $FEATURE_ROOT" >&2
    exit 2
    ;;
esac
if [ ! -d "$SOURCE_SHARD_DIR" ]; then
  echo "Source shard directory does not exist: $SOURCE_SHARD_DIR" >&2
  exit 2
fi

for SHARD_ID in $(seq 0 15); do
  SHARD_NAME="shard-$(printf '%03d' "$SHARD_ID")-of-016.jsonl"
  SHARD_MANIFEST="$SOURCE_SHARD_DIR/$SHARD_NAME"
  if [ ! -f "$SHARD_MANIFEST" ]; then
    echo "Missing source shard: $SHARD_MANIFEST" >&2
    exit 2
  fi
done

for SHARD_ID in $(seq 0 15); do
  SHARD_NAME="shard-$(printf '%03d' "$SHARD_ID")-of-016.jsonl"
  SHARD_MANIFEST="$SOURCE_SHARD_DIR/$SHARD_NAME"
  SHARD_OUTPUT="$FEATURE_ROOT/shard-$(printf '%03d' "$SHARD_ID")"
  if [ "$SHARD_ID" -lt 8 ]; then
    QUEUE="pdgpu-5090"
  else
    QUEUE="pdgpu-3090"
  fi

  vc submit \
    -p "$QUEUE" \
    -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
    -c 8 -m 32G -g 1 -n 1 \
    -j "bat-ast-bf16-shard-${SHARD_ID}-$(date +%m%d%H%M)" \
    -d "$SCRIPT_DIR" \
    JOB=1:1 "$SCRIPT_DIR/log/bat_ast_bf16_shard_${SHARD_ID}.JOB.log" \
    --cmd "BAT_SOURCE_SHARD_MANIFEST=$(printf '%q' "$SHARD_MANIFEST") BAT_FEATURE_SHARD_OUTPUT_DIR=$(printf '%q' "$SHARD_OUTPUT") BAT_SPATIAL_AST_ROOT=$(printf '%q' "$SPATIAL_AST_ROOT") BAT_SPATIAL_AST_CHECKPOINT=$(printf '%q' "$SPATIAL_AST_CHECKPOINT") BAT_AUDIO_ROOT=$(printf '%q' "$AUDIO_ROOT") BAT_REVERB_ROOT=$(printf '%q' "$REVERB_ROOT") BAT_PRECOMPUTE_DEVICE=cuda:0 BAT_PRECOMPUTE_BATCH_SIZE=$(printf '%q' "$BATCH_SIZE") BAT_PRECOMPUTE_CHUNK_SIZE=$(printf '%q' "$CHUNK_SIZE") bash scripts/precompute_bat_spatial_ast_bf16_remote.sh"
done
