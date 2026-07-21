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

OUTPUT_DIR="${SWIFT_RETRIEVAL_OUTPUT_DIR:-$REPO_ROOT/outputs/huginn_audio_retrieval_clotho_v2_swift}"
SAMPLE_COUNT="${SWIFT_RETRIEVAL_SAMPLE_COUNT:-all}"
CHECKPOINTS_RAW="${SWIFT_RETRIEVAL_CHECKPOINTS:-}"
FSDP_EXPORT_DIR="${HUGINN_AUDIO_FSDP_EVAL_EXPORT_DIR:-}"
PLUGIN_PATH="${SWIFT_RETRIEVAL_PLUGIN_PATH:-$REPO_ROOT/code/huginn_lora/plugins/huginn_audio_swift.py}"
CHECKPOINT_ARGS=()
FSDP_EXPORT_ARGS=()

if [ -n "$FSDP_EXPORT_DIR" ]; then
  FSDP_EXPORT_ARGS=(--fsdp_export_dir "$FSDP_EXPORT_DIR")
fi

if [ -n "$CHECKPOINTS_RAW" ]; then
  IFS=';' read -r -a CHECKPOINTS <<< "$CHECKPOINTS_RAW"
  for checkpoint in "${CHECKPOINTS[@]}"; do
    if [ -z "$checkpoint" ]; then
      echo "SWIFT_RETRIEVAL_CHECKPOINTS contains an empty checkpoint entry" >&2
      exit 1
    fi
    if [ ! -d "$checkpoint" ]; then
      echo "Retrieval checkpoint directory does not exist: $checkpoint" >&2
      exit 1
    fi
    CHECKPOINT_ARGS+=(--checkpoint "$checkpoint")
  done
fi

echo "========== EVAL SWIFT HUGINN AUDIO-TEXT RETRIEVAL =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "output_dir=$OUTPUT_DIR"
echo "sample_count=$SAMPLE_COUNT"
echo "checkpoints=${CHECKPOINTS_RAW:-<script-defaults>}"
echo "fsdp_export_dir=${FSDP_EXPORT_DIR:-<checkpoint-sibling-default>}"
echo "plugin_path=$PLUGIN_PATH"

python -u code/huginn_lora/scripts/eval_huginn_audio_text_retrieval_swift.py \
  --output_dir "$OUTPUT_DIR" \
  --sample_count "$SAMPLE_COUNT" \
  --plugin_path "$PLUGIN_PATH" \
  "${FSDP_EXPORT_ARGS[@]}" \
  "${CHECKPOINT_ARGS[@]}"
