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
TEMPLATE_TYPE="${HRM_SWIFT_INFERENCE_TEMPLATE:-hrm_text_synth_cot}"
QUESTION="${HRM_SWIFT_INFERENCE_QUESTION:-What is 1 + 1?}"
MAX_NEW_TOKENS="${HRM_SWIFT_INFERENCE_MAX_NEW_TOKENS:-128}"
OUTPUT_REPORT="${HRM_SWIFT_INFERENCE_OUTPUT_REPORT:-$REPO_ROOT/outputs/hrm_text/swift_model_inference_inspect.json}"

echo "========== INSPECT HRM SWIFT MODEL + INFERENCE =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "PYTHON=$(which python)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "MODEL_PATH=$MODEL_PATH"
echo "PLUGIN_PATH=$PLUGIN_PATH"
echo "TEMPLATE_TYPE=$TEMPLATE_TYPE"
echo "QUESTION=$QUESTION"
echo "MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
echo "OUTPUT_REPORT=$OUTPUT_REPORT"
echo "OFFLINE_MODE=true"

python -u code/HRM_Audio/scripts/inspect_hrm_swift_model_inference.py \
  --model-path "$MODEL_PATH" \
  --plugin-path "$PLUGIN_PATH" \
  --template-type "$TEMPLATE_TYPE" \
  --question "$QUESTION" \
  --device cuda:0 \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --output-report "$OUTPUT_REPORT"
