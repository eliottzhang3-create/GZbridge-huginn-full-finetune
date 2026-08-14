#!/usr/bin/env bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_ouro"

REPO_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/code/Ouro_audio:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

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

python -u code/Ouro_audio/bat/scripts/build_bat_unique_manifests.py \
  --qa-root "$QA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --source-shard-count "$SHARD_COUNT" \
  --expected-qa-min "$EXPECTED_MIN" \
  --expected-qa-max "$EXPECTED_MAX"
