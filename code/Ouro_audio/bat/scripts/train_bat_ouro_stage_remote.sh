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
DATASET="${BAT_STAGE_DATASET:?Set BAT_STAGE_DATASET to a private JSONL manifest}"
STAGE="${BAT_STAGE:?Set BAT_STAGE=I, II or III}"
OUTPUT_DIR="${BAT_STAGE_OUTPUT_DIR:?Set BAT_STAGE_OUTPUT_DIR to a private checkpoint directory}"
WORLD_SIZE="${BAT_WORLD_SIZE:-1}"
GRAD_ACCUM="${BAT_GRADIENT_ACCUMULATION_STEPS:-1}"

case "$OUTPUT_DIR" in /hpc_stor03/public|/hpc_stor03/public/*) echo "Refusing public output" >&2; exit 2;; esac
ARGS=(--model-path "$MODEL_PATH" --plugin-path "$PLUGIN_PATH" --dataset "$DATASET" --stage "$STAGE" --output-dir "$OUTPUT_DIR" --world-size "$WORLD_SIZE" --gradient-accumulation-steps "$GRAD_ACCUM")
if [[ -n "${BAT_RESUME_FROM_CHECKPOINT:-}" ]]; then
  ARGS+=(--resume-from-checkpoint "$BAT_RESUME_FROM_CHECKPOINT")
fi
python -u code/Ouro_audio/bat/scripts/train_bat_ouro_stage.py "${ARGS[@]}"
