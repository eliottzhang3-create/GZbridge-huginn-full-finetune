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

RUN_TAG="${HRM_AUDIO_SWIFT_TRAINABILITY_RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${HRM_AUDIO_SWIFT_TRAINABILITY_RUN_DIR:-$REPO_ROOT/outputs/hrm_text/audio_swift_trainability/$RUN_TAG}"
OUTPUT_REPORT="${HRM_AUDIO_SWIFT_TRAINABILITY_OUTPUT_REPORT:-$RUN_DIR/trainability_inspect.json}"

echo "========== INSPECT HRM AUDIO SWIFT TRAINABILITY =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "PYTHON=$(which python)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "WRAPPER_MODEL_PATH=$REPO_ROOT/models/hrm-text-audio-v1"
echo "PLUGIN_PATH=$REPO_ROOT/code/HRM_Audio/plugins/hrm_text_audio_swift.py"
echo "RUN_DIR=$RUN_DIR"
echo "OUTPUT_REPORT=$OUTPUT_REPORT"
echo "OFFLINE_MODE=true"
echo "POLICY=lora_llm frozen_whisper frozen_hrm_base trainable_aligner trainable_H_L_lora"

python -u code/HRM_Audio/scripts/inspect_hrm_audio_swift_trainability.py \
  --wrapper-model-path "$REPO_ROOT/models/hrm-text-audio-v1" \
  --plugin-path "$REPO_ROOT/code/HRM_Audio/plugins/hrm_text_audio_swift.py" \
  --run-dir "$RUN_DIR" \
  --output-report "$OUTPUT_REPORT"
