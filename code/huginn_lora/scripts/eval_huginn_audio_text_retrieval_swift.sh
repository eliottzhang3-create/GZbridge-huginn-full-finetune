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

OUTPUT_DIR="${SWIFT_RETRIEVAL_OUTPUT_DIR:-$REPO_ROOT/outputs/huginn_audio_retrieval_clotho_v2_swift}"
SAMPLE_COUNT="${SWIFT_RETRIEVAL_SAMPLE_COUNT:-all}"

echo "========== EVAL SWIFT HUGINN AUDIO-TEXT RETRIEVAL =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "output_dir=$OUTPUT_DIR"
echo "sample_count=$SAMPLE_COUNT"

python -u code/huginn_lora/scripts/eval_huginn_audio_text_retrieval_swift.py \
  --output_dir "$OUTPUT_DIR" \
  --sample_count "$SAMPLE_COUNT"
