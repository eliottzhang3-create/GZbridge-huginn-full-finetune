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
export HUGINN_AUDIO_FSDP2_NONPERSISTENT_ROPE=1

CHECKPOINT="${MMAU_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_audio_audiocaps_v2_train_e5_b8ga4_5090/v0-20260713-155848/checkpoint-8406}"
DATASET_PATH="${MMAU_TEST_MINI_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAU test_mini/test_mini.parquet}"
OUTPUT_DIR="${MMAU_OUTPUT_DIR:-$REPO_ROOT/outputs/mmau_test_mini_full_checkpoint_8406_ffmpeg_bytes_v2}"
START_OFFSET="${MMAU_START_OFFSET:-0}"
MAX_SAMPLES="${MMAU_MAX_SAMPLES:-}"
LOG_EVERY="${MMAU_LOG_EVERY:-10}"
NUM_STEPS="${MMAU_NUM_STEPS:-}"
CHECKPOINTS_RAW="${MMAU_CHECKPOINTS:-}"
FSDP_EXPORT_DIR="${HUGINN_AUDIO_FSDP_EVAL_EXPORT_DIR:-}"
PLUGIN_PATH="${MMAU_PLUGIN_PATH:-$REPO_ROOT/code/huginn_lora/plugins/huginn_audio_swift.py}"

MAX_SAMPLES_ARGS=()
if [ -n "$MAX_SAMPLES" ]; then
  MAX_SAMPLES_ARGS=(--max-samples "$MAX_SAMPLES")
fi
NUM_STEPS_ARGS=()
if [ -n "$NUM_STEPS" ]; then
  NUM_STEPS_ARGS=(--num-steps "$NUM_STEPS")
fi

checkpoint_slug() {
  local checkpoint="${1%/}"
  local run_dir
  run_dir="$(basename "$(dirname "$checkpoint")")"
  printf '%s_%s' "$run_dir" "$(basename "$checkpoint")"
}

evaluate_one_checkpoint() {
  local checkpoint="$1"
  local output_dir="$2"
  if [ ! -d "$checkpoint" ]; then
    echo "MMAU checkpoint directory does not exist: $checkpoint" >&2
    exit 1
  fi

  echo "========== RUN MMAU TEST-MINI SWIFT FULL EVAL =========="
  echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
  echo "checkpoint=$checkpoint"
  echo "dataset_path=$DATASET_PATH"
  echo "output_dir=$output_dir"
  echo "start_offset=$START_OFFSET max_samples=${MAX_SAMPLES:-<all>} log_every=$LOG_EVERY"
  echo "num_steps=${NUM_STEPS:-<config.mean_recurrence>}"
  echo "fsdp_export_dir=${FSDP_EXPORT_DIR:-<checkpoint-sibling-default>}"
  echo "plugin_path=$PLUGIN_PATH"

  CMD=(python -u code/huginn_lora/scripts/eval_mmau_test_mini_swift.py \
    --checkpoint "$checkpoint" \
    --dataset-path "$DATASET_PATH" \
    --output-dir "$output_dir" \
    --plugin-path "$PLUGIN_PATH" \
    --start-offset "$START_OFFSET" \
    --log-every "$LOG_EVERY" \
    "${MAX_SAMPLES_ARGS[@]}" \
    "${NUM_STEPS_ARGS[@]}")
  if [ -n "$FSDP_EXPORT_DIR" ]; then
    CMD+=(--fsdp-export-dir "$FSDP_EXPORT_DIR")
  fi
  "${CMD[@]}"
}

if [ -z "$CHECKPOINTS_RAW" ]; then
  evaluate_one_checkpoint "$CHECKPOINT" "$OUTPUT_DIR"
  exit 0
fi

IFS=';' read -r -a CHECKPOINTS <<< "$CHECKPOINTS_RAW"
echo "========== RUN MMAU TEST-MINI SWIFT MULTI-CHECKPOINT EVAL =========="
echo "checkpoint_count=${#CHECKPOINTS[@]} output_root=$OUTPUT_DIR"
for checkpoint in "${CHECKPOINTS[@]}"; do
  if [ -z "$checkpoint" ]; then
    echo "MMAU_CHECKPOINTS contains an empty checkpoint entry" >&2
    exit 1
  fi
  evaluate_one_checkpoint "$checkpoint" "$OUTPUT_DIR/$(checkpoint_slug "$checkpoint")"
done
