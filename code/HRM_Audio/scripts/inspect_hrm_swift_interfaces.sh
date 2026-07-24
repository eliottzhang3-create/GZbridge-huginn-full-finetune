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
OUTPUT_REPORT="${HRM_SWIFT_INTERFACE_OUTPUT_REPORT:-$REPO_ROOT/outputs/hrm_text/swift_interface_inspect.json}"
SOURCE_HIT_LIMIT="${HRM_SWIFT_SOURCE_HIT_LIMIT:-120}"

echo "========== INSPECT HRM SWIFT INTERFACES =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "PYTHON=$(which python)"
echo "MODEL_PATH=$MODEL_PATH"
echo "OUTPUT_REPORT=$OUTPUT_REPORT"
echo "SOURCE_HIT_LIMIT=$SOURCE_HIT_LIMIT"
echo "OFFLINE_MODE=true"
echo "MUTATES_SWIFT_REGISTRIES=false"

python -u code/HRM_Audio/scripts/inspect_hrm_swift_interfaces.py \
  --model-path "$MODEL_PATH" \
  --output-report "$OUTPUT_REPORT" \
  --source-hit-limit "$SOURCE_HIT_LIMIT"
