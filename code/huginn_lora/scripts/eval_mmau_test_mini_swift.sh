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
OUTPUT_DIR="${MMAU_OUTPUT_DIR:-$REPO_ROOT/outputs/mmau_test_mini_full_checkpoint_8406_ffmpeg_bytes_v2}"
START_OFFSET="${MMAU_START_OFFSET:-0}"
MAX_SAMPLES="${MMAU_MAX_SAMPLES:-}"
LOG_EVERY="${MMAU_LOG_EVERY:-10}"
NUM_STEPS="${MMAU_NUM_STEPS:-}"

MAX_SAMPLES_ARGS=()
if [ -n "$MAX_SAMPLES" ]; then
  MAX_SAMPLES_ARGS=(--max-samples "$MAX_SAMPLES")
fi
NUM_STEPS_ARGS=()
if [ -n "$NUM_STEPS" ]; then
  NUM_STEPS_ARGS=(--num-steps "$NUM_STEPS")
fi

echo "========== RUN MMAU TEST-MINI SWIFT FULL EVAL =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "checkpoint=$CHECKPOINT"
echo "dataset_path=$DATASET_PATH"
echo "output_dir=$OUTPUT_DIR"
echo "start_offset=$START_OFFSET max_samples=${MAX_SAMPLES:-<all>} log_every=$LOG_EVERY"
echo "num_steps=${NUM_STEPS:-<config.mean_recurrence>}"

python -u code/huginn_lora/scripts/eval_mmau_test_mini_swift.py \
  --checkpoint "$CHECKPOINT" \
  --dataset-path "$DATASET_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --start-offset "$START_OFFSET" \
  --log-every "$LOG_EVERY" \
  "${MAX_SAMPLES_ARGS[@]}" \
  "${NUM_STEPS_ARGS[@]}"
