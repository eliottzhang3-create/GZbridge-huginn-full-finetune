#!/usr/bin/env bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_ouro"
REPO_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/code/Ouro_audio:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
MODEL_PATH="${OURO_MODEL_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/models/Ouro-1.4B}"
PLUGIN_PATH="${OURO_BAT_PLUGIN_PATH:-$REPO_ROOT/code/Ouro_audio/plugins/ouro_bat_spatial_ast_swift.py}"
DATASET="${BAT_SMOKE_DATASET:?Set BAT_SMOKE_DATASET to a private BAT JSONL manifest}"
OUTPUT_DIR="${BAT_SMOKE_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/bat/lora_qformer_smoke-$(date +%Y%m%d-%H%M%S)}"
REPORT="${BAT_SMOKE_REPORT:-$OUTPUT_DIR/audit.json}"

case "$OUTPUT_DIR:$REPORT" in
  /hpc_stor03/public*|*:/hpc_stor03/public*) echo "Refusing public output" >&2; exit 2;;
esac

python -u code/Ouro_audio/bat/scripts/smoke_bat_ouro_lora.py \
  --model-path "$MODEL_PATH" --plugin-path "$PLUGIN_PATH" --dataset "$DATASET" \
  --output-dir "$OUTPUT_DIR" --output-report "$REPORT"
