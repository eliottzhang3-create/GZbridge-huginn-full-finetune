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
DATASET_PATH="${OURO_LORA_TINY_DATASET:-$REPO_ROOT/code/Ouro_audio/data/ouro_lora_tiny.jsonl}"
RUN_TAG="${OURO_LORA_RUN_TAG:-$(date +%m%d%H%M%S)}"
OUTPUT_DIR="${OURO_LORA_OUTPUT_DIR:-$REPO_ROOT/outputs/ouro/lora_smoke-$RUN_TAG}"
OUTPUT_REPORT="${OURO_LORA_OUTPUT_REPORT:-$REPO_ROOT/outputs/ouro/lora_smoke-$RUN_TAG.json}"

echo "========== OURO MS-SWIFT TEXT LORA ONE-STEP SMOKE =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "PYTHON=$(which python)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "MODEL_PATH=$MODEL_PATH"
echo "PLUGIN_PATH=$PLUGIN_PATH"
echo "DATASET_PATH=$DATASET_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "OUTPUT_REPORT=$OUTPUT_REPORT"
echo "OFFLINE_MODE=true"

python -u code/Ouro_audio/scripts/smoke_ouro_swift_lora.py \
  --model-path "$MODEL_PATH" \
  --plugin-path "$PLUGIN_PATH" \
  --dataset "$DATASET_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --output-report "$OUTPUT_REPORT"
