#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_ouro"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${OURO_MODEL_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/models/Ouro-1.4B}"
PLUGIN_PATH="${OURO_TEXT_PLUGIN_PATH:-$REPO_ROOT/code/Ouro_audio/plugins/ouro_text_swift.py}"
OUTPUT_REPORT="${OURO_SWIFT_OUTPUT_REPORT:-$REPO_ROOT/outputs/ouro/swift_text_inference.json}"
QUESTION="${OURO_SWIFT_QUESTION:-The future of artificial intelligence is}"
MAX_NEW_TOKENS="${OURO_SWIFT_MAX_NEW_TOKENS:-32}"

echo "========== INSPECT OURO MS-SWIFT TEXT INFERENCE =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "PYTHON=$(which python)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "MODEL_PATH=$MODEL_PATH"
echo "PLUGIN_PATH=$PLUGIN_PATH"
echo "OUTPUT_REPORT=$OUTPUT_REPORT"
echo "OFFLINE_MODE=true"

python -u code/Ouro_audio/scripts/inspect_ouro_swift_inference.py \
  --model-path "$MODEL_PATH" \
  --plugin-path "$PLUGIN_PATH" \
  --question "$QUESTION" \
  --device cuda:0 \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --output-report "$OUTPUT_REPORT"
