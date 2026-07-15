#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
DATASET_ROOT="${WAVCAPS_DATASET_ROOT:-/hpc_stor03/public/shared/data/raa/WavCaps}"
OUTPUT_MANIFEST="${WAVCAPS_TRAIN_MANIFEST:-$REPO_ROOT/data/audio_swift/wavcaps_audioset/wavcaps_audioset_sl_train_swift.jsonl}"
LIMIT_RECORDS="${WAVCAPS_LIMIT_RECORDS:-}"

echo "========== PREPARE WAVCAPS AUDIOSET SWIFT DATASET =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "dataset_root=$DATASET_ROOT"
echo "output_manifest=$OUTPUT_MANIFEST"
echo "limit_records=${LIMIT_RECORDS:-<full>}"

ARGS=(
  --dataset_root "$DATASET_ROOT"
  --output_manifest "$OUTPUT_MANIFEST"
  --invalid_row_policy skip
)
if [ -n "$LIMIT_RECORDS" ]; then
  ARGS+=(--limit_records "$LIMIT_RECORDS")
fi

python -u code/huginn_lora/scripts/prepare_wavcaps_audioset_swift_dataset.py "${ARGS[@]}"
