#!/usr/bin/env bash
set -euo pipefail

ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune
source /hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/etc/profile.d/conda.sh
conda activate /hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/envs/swift_ouro
cd "$ROOT"
export PYTHONPATH="$ROOT/code/Ouro_audio:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${BAT_FIXED_DDP_OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${BAT_FIXED_DDP_MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${BAT_FIXED_DDP_OPENBLAS_NUM_THREADS:-1}"

MODEL_TYPE="${BAT_FIXED_DDP_MODEL_TYPE:?Set BAT_FIXED_DDP_MODEL_TYPE (ouro or qwen3)}"
MODEL_PATH="${BAT_FIXED_DDP_MODEL_PATH:?Set BAT_FIXED_DDP_MODEL_PATH}"
PLUGIN_PATH="${BAT_FIXED_DDP_PLUGIN_PATH:?Set BAT_FIXED_DDP_PLUGIN_PATH}"
OUTPUT="${BAT_FIXED_DDP_OUTPUT:?Set BAT_FIXED_DDP_OUTPUT}"
WORLD="${BAT_FIXED_DDP_WORLD_SIZE:-8}"
case "$OUTPUT" in /hpc_stor03/public|/hpc_stor03/public/*) echo "Refusing public output" >&2; exit 2;; esac
torchrun --standalone --nproc_per_node="$WORLD" code/Ouro_audio/bat/scripts/test_bat_fixed_waveform_ddp.py \
  --model-type "$MODEL_TYPE" --model-path "$MODEL_PATH" --plugin-path "$PLUGIN_PATH" \
  --output-report "$OUTPUT" --local-batch-size "${BAT_FIXED_DDP_LOCAL_BATCH_SIZE:-2}" --steps "${BAT_FIXED_DDP_STEPS:-2}" --sequence-length 176
