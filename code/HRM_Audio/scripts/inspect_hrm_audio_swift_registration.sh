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

RUN_TAG="${HRM_AUDIO_SWIFT_REG_RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${HRM_AUDIO_SWIFT_REG_RUN_DIR:-$REPO_ROOT/outputs/hrm_text/audio_swift_registration/$RUN_TAG}"
OUTPUT_REPORT="${HRM_AUDIO_SWIFT_REG_OUTPUT_REPORT:-$RUN_DIR/registration_inspect.json}"

echo "========== INSPECT HRM AUDIO SWIFT REGISTRATION =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "PYTHON=$(which python)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "WRAPPER_MODEL_PATH=$REPO_ROOT/models/hrm-text-audio-v1"
echo "TEXT_PLUGIN_PATH=$REPO_ROOT/code/HRM_Audio/plugins/hrm_text_swift.py"
echo "AUDIO_PLUGIN_PATH=$REPO_ROOT/code/HRM_Audio/plugins/hrm_text_audio_swift.py"
echo "OUTPUT_REPORT=$OUTPUT_REPORT"
echo "OFFLINE_MODE=true"

python -u code/HRM_Audio/scripts/inspect_hrm_audio_swift_registration.py \
  --wrapper-model-path "$REPO_ROOT/models/hrm-text-audio-v1" \
  --text-plugin-path "$REPO_ROOT/code/HRM_Audio/plugins/hrm_text_swift.py" \
  --audio-plugin-path "$REPO_ROOT/code/HRM_Audio/plugins/hrm_text_audio_swift.py" \
  --output-report "$OUTPUT_REPORT" \
  --device cuda:0
