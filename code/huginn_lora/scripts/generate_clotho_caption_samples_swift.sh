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

CHECKPOINT="${CLOTHO_CAPTION_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_audio_audiocaps_v2_train_e5_b8ga4_5090/v0-20260713-155848/checkpoint-8406}"
OUTPUT_DIR="${CLOTHO_CAPTION_OUTPUT_DIR:-$REPO_ROOT/outputs/huginn_audio_clotho_caption_samples}"
SAMPLE_COUNT="${CLOTHO_CAPTION_SAMPLE_COUNT:-3}"
MAX_NEW_TOKENS="${CLOTHO_CAPTION_MAX_NEW_TOKENS:-64}"
CHECKPOINTS_RAW="${CLOTHO_CAPTION_CHECKPOINTS:-}"
FSDP_EXPORT_DIR="${HUGINN_AUDIO_FSDP_EVAL_EXPORT_DIR:-}"
PLUGIN_PATH="${CLOTHO_CAPTION_PLUGIN_PATH:-$REPO_ROOT/code/huginn_lora/plugins/huginn_audio_swift.py}"

checkpoint_slug() {
  local checkpoint="${1%/}"
  local run_dir
  run_dir="$(basename "$(dirname "$checkpoint")")"
  printf '%s_%s' "$run_dir" "$(basename "$checkpoint")"
}

generate_one_checkpoint() {
  local checkpoint="$1"
  local output_dir="$2"
  if [ ! -d "$checkpoint" ]; then
    echo "Clotho generation checkpoint directory does not exist: $checkpoint" >&2
    exit 1
  fi

  echo "========== GENERATE CLOTHO CAPTION SAMPLES =========="
  echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
  echo "checkpoint=$checkpoint"
  echo "output_dir=$output_dir"
  echo "sample_count=$SAMPLE_COUNT"
  echo "max_new_tokens=$MAX_NEW_TOKENS"
  echo "generation_path=audio_manual_cache"
  echo "fsdp_export_dir=${FSDP_EXPORT_DIR:-<checkpoint-sibling-default>}"
  echo "plugin_path=$PLUGIN_PATH"

  CMD=(python -u code/huginn_lora/scripts/generate_clotho_caption_samples_swift.py \
    --checkpoint "$checkpoint" \
    --output-dir "$output_dir" \
    --sample-count "$SAMPLE_COUNT" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --plugin-path "$PLUGIN_PATH")
  if [ -n "$FSDP_EXPORT_DIR" ]; then
    CMD+=(--fsdp-export-dir "$FSDP_EXPORT_DIR")
  fi
  "${CMD[@]}"
}

if [ -z "$CHECKPOINTS_RAW" ]; then
  generate_one_checkpoint "$CHECKPOINT" "$OUTPUT_DIR"
  exit 0
fi

IFS=';' read -r -a CHECKPOINTS <<< "$CHECKPOINTS_RAW"
echo "========== GENERATE CLOTHO CAPTION SAMPLES (MULTI-CHECKPOINT) =========="
echo "checkpoint_count=${#CHECKPOINTS[@]} output_root=$OUTPUT_DIR"
for checkpoint in "${CHECKPOINTS[@]}"; do
  if [ -z "$checkpoint" ]; then
    echo "CLOTHO_CAPTION_CHECKPOINTS contains an empty checkpoint entry" >&2
    exit 1
  fi
  generate_one_checkpoint "$checkpoint" "$OUTPUT_DIR/$(checkpoint_slug "$checkpoint")"
done
