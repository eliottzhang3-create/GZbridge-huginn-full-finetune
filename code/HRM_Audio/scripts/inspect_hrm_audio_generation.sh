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

HRM_MODEL_PATH="${HRM_TEXT_MODEL_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/models/HRM-text}"
WHISPER_MODEL_PATH="${HRM_AUDIO_WHISPER_MODEL_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/models/whisper-large}"
WRAPPER_MODEL_PATH="${HRM_AUDIO_WRAPPER_MODEL_PATH:-$REPO_ROOT/models/hrm-text-audio-v1}"
RUN_TAG="${HRM_AUDIO_GENERATION_RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${HRM_AUDIO_GENERATION_RUN_DIR:-$REPO_ROOT/outputs/hrm_text/audio_generation_inspect/$RUN_TAG}"
OUTPUT_REPORT="${HRM_AUDIO_GENERATION_OUTPUT_REPORT:-$RUN_DIR/generation_inspect.json}"

echo "========== INSPECT HRM AUDIO GENERATION =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "PYTHON=$(which python)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "HRM_MODEL_PATH=$HRM_MODEL_PATH"
echo "WHISPER_MODEL_PATH=$WHISPER_MODEL_PATH"
echo "WRAPPER_MODEL_PATH=$WRAPPER_MODEL_PATH"
echo "OUTPUT_REPORT=$OUTPUT_REPORT"
echo "OFFLINE_MODE=true"

python -u code/HRM_Audio/scripts/inspect_hrm_audio_generation.py \
  --hrm-model-path "$HRM_MODEL_PATH" \
  --whisper-model-path "$WHISPER_MODEL_PATH" \
  --wrapper-model-path "$WRAPPER_MODEL_PATH" \
  --output-report "$OUTPUT_REPORT" \
  --device cuda:0 \
  --min-new-tokens 2 \
  --max-new-tokens 4
