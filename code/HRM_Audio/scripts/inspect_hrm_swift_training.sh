#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_HRM"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${HRM_TEXT_MODEL_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/models/HRM-text}"
PLUGIN_PATH="${HRM_TEXT_SWIFT_PLUGIN_PATH:-$REPO_ROOT/code/HRM_Audio/plugins/hrm_text_swift.py}"
RUN_TAG="${HRM_SWIFT_TRAIN_RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${HRM_SWIFT_TRAIN_RUN_DIR:-$REPO_ROOT/outputs/hrm_text/swift_training_inspect/$RUN_TAG}"
OUTPUT_REPORT="${HRM_SWIFT_TRAIN_OUTPUT_REPORT:-$RUN_DIR/training_inspect.json}"
ADAPTER_OUTPUT_DIR="${HRM_SWIFT_TRAIN_ADAPTER_DIR:-$RUN_DIR/adapter}"
LORA_RANK="${HRM_SWIFT_TRAIN_LORA_RANK:-8}"
LORA_ALPHA="${HRM_SWIFT_TRAIN_LORA_ALPHA:-16}"
LEARNING_RATE="${HRM_SWIFT_TRAIN_LEARNING_RATE:-1e-4}"

echo "========== INSPECT HRM SWIFT TRAINING =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "PYTHON=$(which python)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "MODEL_PATH=$MODEL_PATH"
echo "PLUGIN_PATH=$PLUGIN_PATH"
echo "RUN_DIR=$RUN_DIR"
echo "OUTPUT_REPORT=$OUTPUT_REPORT"
echo "ADAPTER_OUTPUT_DIR=$ADAPTER_OUTPUT_DIR"
echo "LORA_RANK=$LORA_RANK"
echo "LORA_ALPHA=$LORA_ALPHA"
echo "LEARNING_RATE=$LEARNING_RATE"
echo "OFFLINE_MODE=true"

python -u code/HRM_Audio/scripts/inspect_hrm_swift_training.py \
  --model-path "$MODEL_PATH" \
  --plugin-path "$PLUGIN_PATH" \
  --device cuda:0 \
  --output-report "$OUTPUT_REPORT" \
  --adapter-output-dir "$ADAPTER_OUTPUT_DIR" \
  --lora-rank "$LORA_RANK" \
  --lora-alpha "$LORA_ALPHA" \
  --learning-rate "$LEARNING_RATE"
