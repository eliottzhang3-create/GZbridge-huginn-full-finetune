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
OUTPUT_REPORT="${HRM_MODEL_LOAD_OUTPUT_REPORT:-$REPO_ROOT/outputs/hrm_text/model_load_inspect.json}"

echo "========== INSPECT HRM-TEXT MODEL LOAD =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "PYTHON=$(which python)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "MODEL_PATH=$MODEL_PATH"
echo "OUTPUT_REPORT=$OUTPUT_REPORT"
echo "OFFLINE_MODE=true"

python -u code/HRM_Audio/scripts/inspect_hrm_model_load.py \
  --model-path "$MODEL_PATH" \
  --device cuda:0 \
  --output-report "$OUTPUT_REPORT"
