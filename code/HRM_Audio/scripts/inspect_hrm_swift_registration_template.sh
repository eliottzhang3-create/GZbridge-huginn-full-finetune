#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_HRM"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${HRM_TEXT_MODEL_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/models/HRM-text}"
PLUGIN_PATH="${HRM_TEXT_SWIFT_PLUGIN_PATH:-$REPO_ROOT/code/HRM_Audio/plugins/hrm_text_swift.py}"
OUTPUT_REPORT="${HRM_SWIFT_TEMPLATE_OUTPUT_REPORT:-$REPO_ROOT/outputs/hrm_text/swift_registration_template_inspect.json}"

echo "========== INSPECT HRM SWIFT REGISTRATION + TEMPLATE =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "PYTHON=$(which python)"
echo "MODEL_PATH=$MODEL_PATH"
echo "PLUGIN_PATH=$PLUGIN_PATH"
echo "OUTPUT_REPORT=$OUTPUT_REPORT"
echo "OFFLINE_MODE=true"
echo "LOAD_MODEL_WEIGHTS=false"

python -u code/HRM_Audio/scripts/inspect_hrm_swift_registration_template.py \
  --model-path "$MODEL_PATH" \
  --plugin-path "$PLUGIN_PATH" \
  --output-report "$OUTPUT_REPORT"
