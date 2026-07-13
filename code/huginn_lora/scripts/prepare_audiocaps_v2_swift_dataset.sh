#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
DATASET_ROOT="${AUDIOCAPS_DATASET_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/data/audiocaps_v2}"
OUTPUT_MANIFEST="${AUDIOCAPS_TRAIN_MANIFEST:-$REPO_ROOT/data/audio_swift/audiocaps_v2/audiocaps_v2_train_swift.jsonl}"

echo "========== PREPARE AUDIOCAPS V2 TRAIN MANIFEST =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "dataset_root=$DATASET_ROOT"
echo "output_manifest=$OUTPUT_MANIFEST"

python -u code/huginn_lora/scripts/prepare_audiocaps_v2_swift_dataset.py \
  --dataset_root "$DATASET_ROOT" \
  --split train \
  --output_manifest "$OUTPUT_MANIFEST"
