#!/usr/bin/env bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_ouro"

REPO_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/code/Ouro_audio:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

SHARD_MANIFEST="${BAT_SOURCE_SHARD_MANIFEST:?Set BAT_SOURCE_SHARD_MANIFEST}"
OUTPUT_DIR="${BAT_FEATURE_SHARD_OUTPUT_DIR:?Set BAT_FEATURE_SHARD_OUTPUT_DIR to a private output directory}"
SPATIAL_AST_ROOT="${BAT_SPATIAL_AST_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/Spatial-AST}"
SPATIAL_AST_CHECKPOINT="${BAT_SPATIAL_AST_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/models/BAT/SpatialAST/finetuned.pth}"
AUDIO_ROOT="${BAT_AUDIO_ROOT:-/hpc_stor03/public/shared/data/raa/AudioSet}"
REVERB_ROOT="${BAT_REVERB_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/SpatialSoundQA/mp3d_reverb}"
DEVICE="${BAT_PRECOMPUTE_DEVICE:-cuda:0}"
BATCH_SIZE="${BAT_PRECOMPUTE_BATCH_SIZE:-4}"
CHUNK_SIZE="${BAT_PRECOMPUTE_CHUNK_SIZE:-512}"
LIMIT="${BAT_PRECOMPUTE_LIMIT:-0}"

case "$OUTPUT_DIR" in
  /hpc_stor03/public|/hpc_stor03/public/*)
    echo "Refusing public output directory: $OUTPUT_DIR" >&2
    exit 2
    ;;
esac

python -u code/Ouro_audio/bat/scripts/precompute_bat_spatial_ast_bf16.py \
  --source-manifest "$SHARD_MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --spatial-ast-root "$SPATIAL_AST_ROOT" \
  --spatial-ast-checkpoint "$SPATIAL_AST_CHECKPOINT" \
  --audio-root "$AUDIO_ROOT" \
  --reverb-root "$REVERB_ROOT" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --chunk-size "$CHUNK_SIZE" \
  --limit "$LIMIT"
