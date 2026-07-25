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
DATASET_PATH="${HRM_SWIFT_TRAINER_DATASET:-$REPO_ROOT/code/HRM_Audio/data/hrm_text_trainer_smoke.jsonl}"
RUN_TAG="${HRM_SWIFT_TRAINER_RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${HRM_SWIFT_TRAINER_OUTPUT_DIR:-$REPO_ROOT/outputs/hrm_text/swift_trainer_smoke/$RUN_TAG}"
OUTPUT_REPORT="${HRM_SWIFT_TRAINER_OUTPUT_REPORT:-$OUTPUT_DIR/trainer_smoke_report.json}"

echo "========== RUN HRM SWIFT TRAINER ONE-STEP SMOKE =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "PYTHON=$(which python)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "MODEL_PATH=$MODEL_PATH"
echo "PLUGIN_PATH=$PLUGIN_PATH"
echo "DATASET_PATH=$DATASET_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "OUTPUT_REPORT=$OUTPUT_REPORT"
echo "OFFLINE_MODE=true"

python -u code/HRM_Audio/scripts/smoke_hrm_swift_trainer.py \
  --model-path "$MODEL_PATH" \
  --plugin-path "$PLUGIN_PATH" \
  --dataset "$DATASET_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --output-report "$OUTPUT_REPORT"
