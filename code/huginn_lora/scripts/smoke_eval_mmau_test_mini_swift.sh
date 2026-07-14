#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

CHECKPOINT="${MMAU_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_audio_audiocaps_v2_train_e5_b8ga4_5090/v0-20260713-155848/checkpoint-8406}"
DATASET_PATH="${MMAU_TEST_MINI_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAU test_mini/test_mini.parquet}"
OUTPUT_DIR="${MMAU_SMOKE_OUTPUT_DIR:-$REPO_ROOT/outputs/mmau_test_mini_smoke}"
SAMPLE_COUNT="${MMAU_SMOKE_SAMPLE_COUNT:-5}"
SAMPLE_OFFSET="${MMAU_SMOKE_SAMPLE_OFFSET:-0}"

echo "========== RUN MMAU TEST-MINI SWIFT SMOKE =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "checkpoint=$CHECKPOINT"
echo "dataset_path=$DATASET_PATH"
echo "sample_count=$SAMPLE_COUNT sample_offset=$SAMPLE_OFFSET"

python -u code/huginn_lora/scripts/smoke_eval_mmau_test_mini_swift.py \
  --checkpoint "$CHECKPOINT" \
  --dataset-path "$DATASET_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --sample-count "$SAMPLE_COUNT" \
  --sample-offset "$SAMPLE_OFFSET"
