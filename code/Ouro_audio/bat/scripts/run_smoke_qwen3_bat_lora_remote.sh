#!/usr/bin/env bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate swift_ouro

REPO_ROOT="/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/code/Ouro_audio:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export BAT_AUDIO_AUDIT=1

: "${QWEN3_BAT_SMOKE_DATASET:?Set QWEN3_BAT_SMOKE_DATASET to a private two-record JSONL manifest}"

MODEL_PATH="${QWEN3_MODEL_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/models/Qwen3-4B-Base}"
PLUGIN_PATH="${QWEN3_BAT_PLUGIN_PATH:-$REPO_ROOT/code/Ouro_audio/plugins/qwen3_bat_spatial_ast_swift.py}"
RUN_TAG="${QWEN3_BAT_SMOKE_RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${QWEN3_BAT_SMOKE_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/qwen3/lora_smoke-$RUN_TAG}"
OUTPUT_REPORT="${QWEN3_BAT_SMOKE_OUTPUT_REPORT:-$OUTPUT_DIR/audit.json}"

case "$OUTPUT_DIR:$OUTPUT_REPORT" in
  /hpc_stor03/public*|*:/hpc_stor03/public*)
    echo "Refusing public output path: $OUTPUT_DIR $OUTPUT_REPORT" >&2
    exit 2
    ;;
esac

echo "========== QWEN3 BAT LORA + Q-FORMER STRICT SMOKE =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "PYTHON=$(which python)"
echo "MODEL_PATH=$MODEL_PATH"
echo "PLUGIN_PATH=$PLUGIN_PATH"
echo "DATASET=$QWEN3_BAT_SMOKE_DATASET"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "OUTPUT_REPORT=$OUTPUT_REPORT"
echo "DEVICE=${QWEN3_BAT_DEVICE:-cuda:0}"

python -u code/Ouro_audio/bat/scripts/smoke_qwen3_bat_lora.py \
  --model-path "$MODEL_PATH" \
  --plugin-path "$PLUGIN_PATH" \
  --dataset "$QWEN3_BAT_SMOKE_DATASET" \
  --output-dir "$OUTPUT_DIR" \
  --output-report "$OUTPUT_REPORT"
