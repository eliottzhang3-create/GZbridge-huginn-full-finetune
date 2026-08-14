#!/usr/bin/env bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_ouro"

REPO_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/code/Ouro_audio:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

SOURCE_SHARD_DIR="${BAT_SOURCE_SHARD_DIR:?Set BAT_SOURCE_SHARD_DIR}"
FEATURE_ROOT="${BAT_FEATURE_OUTPUT_ROOT:?Set BAT_FEATURE_OUTPUT_ROOT to a private output directory}"
SHARD_COUNT="${BAT_SOURCE_SHARD_COUNT:-16}"

case "$FEATURE_ROOT" in
  /hpc_stor03/public|/hpc_stor03/public/*)
    echo "Refusing public output directory: $FEATURE_ROOT" >&2
    exit 2
    ;;
esac

python -u code/Ouro_audio/bat/scripts/merge_bat_spatial_ast_feature_indices.py \
  --source-shard-dir "$SOURCE_SHARD_DIR" \
  --feature-root "$FEATURE_ROOT" \
  --shard-count "$SHARD_COUNT"
