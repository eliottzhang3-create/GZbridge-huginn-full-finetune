#!/usr/bin/env bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_ouro"

REPO_ROOT="/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/code/Ouro_audio:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
MODEL_PATH="${OURO_MODEL_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/models/Ouro-1.4B}"
PLUGIN_PATH="${OURO_BAT_PLUGIN_PATH:-$REPO_ROOT/code/Ouro_audio/plugins/ouro_bat_spatial_ast_swift.py}"
QA_ROOT="${BAT_QA_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/SpatialSoundQA/closed-end}"
OUTPUT="${OURO_BAT_MULTIMODAL_AUDIT_OUTPUT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/bat/ouro_multimodal_forward_audit.json}"

case "$OUTPUT" in
  /hpc_stor03/public|/hpc_stor03/public/*)
    echo "Refusing public output path: $OUTPUT" >&2
    exit 2
    ;;
esac

python -u code/Ouro_audio/bat/scripts/audit_bat_ouro_multimodal.py \
  --model-path "$MODEL_PATH" \
  --plugin-path "$PLUGIN_PATH" \
  --qa-root "$QA_ROOT" \
  --output "$OUTPUT" \
  --device "${OURO_BAT_MULTIMODAL_DEVICE:-cuda:0}"
