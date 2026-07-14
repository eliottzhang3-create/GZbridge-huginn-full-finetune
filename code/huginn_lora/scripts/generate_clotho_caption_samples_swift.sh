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

CHECKPOINT="${CLOTHO_CAPTION_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_audio_audiocaps_v2_train_e5_b8ga4_5090/v0-20260713-155848/checkpoint-8406}"
OUTPUT_DIR="${CLOTHO_CAPTION_OUTPUT_DIR:-$REPO_ROOT/outputs/huginn_audio_clotho_caption_samples}"
SAMPLE_COUNT="${CLOTHO_CAPTION_SAMPLE_COUNT:-3}"
MAX_NEW_TOKENS="${CLOTHO_CAPTION_MAX_NEW_TOKENS:-64}"

echo "========== GENERATE CLOTHO CAPTION SAMPLES =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "checkpoint=$CHECKPOINT"
echo "output_dir=$OUTPUT_DIR"
echo "sample_count=$SAMPLE_COUNT"
echo "max_new_tokens=$MAX_NEW_TOKENS"
echo "generation_path=audio_manual_cache"

python -u code/huginn_lora/scripts/generate_clotho_caption_samples_swift.py \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --sample-count "$SAMPLE_COUNT" \
  --max-new-tokens "$MAX_NEW_TOKENS"
