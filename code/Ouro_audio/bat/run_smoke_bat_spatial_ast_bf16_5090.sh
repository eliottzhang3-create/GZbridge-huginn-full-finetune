#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"

SOURCE_SHARD_MANIFEST="${BAT_SOURCE_SHARD_MANIFEST:?Set BAT_SOURCE_SHARD_MANIFEST}"
OUTPUT_DIR="${BAT_FEATURE_SMOKE_OUTPUT_DIR:?Set BAT_FEATURE_SMOKE_OUTPUT_DIR to a private output directory}"
SPATIAL_AST_ROOT="${BAT_SPATIAL_AST_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/Spatial-AST}"
SPATIAL_AST_CHECKPOINT="${BAT_SPATIAL_AST_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/models/BAT/SpatialAST/finetuned.pth}"
AUDIO_ROOT="${BAT_AUDIO_ROOT:-/hpc_stor03/public/shared/data/raa/AudioSet}"
REVERB_ROOT="${BAT_REVERB_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/SpatialSoundQA/mp3d_reverb}"
LIMIT="${BAT_PRECOMPUTE_LIMIT:-32}"
QUEUE="${BAT_SMOKE_QUEUE:-pdgpu-5090}"

case "$QUEUE" in
  pdgpu-5090|pdgpu-3090) ;;
  *) echo "BAT_SMOKE_QUEUE must be pdgpu-5090 or pdgpu-3090, got: $QUEUE" >&2; exit 2 ;;
esac

case "$OUTPUT_DIR" in
  /hpc_stor03/public|/hpc_stor03/public/*)
    echo "Refusing public smoke output directory: $OUTPUT_DIR" >&2
    exit 2
    ;;
esac

vc submit \
  -p "$QUEUE" \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "bat-ast-bf16-smoke-${QUEUE#pdgpu-}-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/bat_ast_bf16_smoke_${QUEUE#pdgpu-}.JOB.log" \
  --cmd "BAT_SOURCE_SHARD_MANIFEST=$(printf '%q' "$SOURCE_SHARD_MANIFEST") BAT_FEATURE_SHARD_OUTPUT_DIR=$(printf '%q' "$OUTPUT_DIR") BAT_SPATIAL_AST_ROOT=$(printf '%q' "$SPATIAL_AST_ROOT") BAT_SPATIAL_AST_CHECKPOINT=$(printf '%q' "$SPATIAL_AST_CHECKPOINT") BAT_AUDIO_ROOT=$(printf '%q' "$AUDIO_ROOT") BAT_REVERB_ROOT=$(printf '%q' "$REVERB_ROOT") BAT_PRECOMPUTE_DEVICE=cuda:0 BAT_PRECOMPUTE_BATCH_SIZE=2 BAT_PRECOMPUTE_CHUNK_SIZE=8 BAT_PRECOMPUTE_LIMIT=$(printf '%q' "$LIMIT") bash scripts/precompute_bat_spatial_ast_bf16_remote.sh"
