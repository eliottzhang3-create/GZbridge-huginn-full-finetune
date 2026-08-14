#!/usr/bin/env bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_ouro"

REPO_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/code/Ouro_audio:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

SOURCE_MANIFEST="${BAT_UNIQUE_SOURCE_MANIFEST:?Set BAT_UNIQUE_SOURCE_MANIFEST}"
OUTPUT_DIR="${BAT_SOURCE_SHARD_OUTPUT_DIR:?Set BAT_SOURCE_SHARD_OUTPUT_DIR to a private output directory}"
SHARD_COUNT="${BAT_SOURCE_SHARD_COUNT:?Set BAT_SOURCE_SHARD_COUNT to a positive integer}"

case "$OUTPUT_DIR" in
  /hpc_stor03/public|/hpc_stor03/public/*)
    echo "Refusing public output directory: $OUTPUT_DIR" >&2
    exit 2
    ;;
esac

python -u code/Ouro_audio/bat/scripts/split_bat_source_manifest.py \
  --source-manifest "$SOURCE_MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --shard-count "$SHARD_COUNT"
