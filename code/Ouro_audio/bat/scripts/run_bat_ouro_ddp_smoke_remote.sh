#!/usr/bin/env bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_ouro"

REPO_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/code/Ouro_audio:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29519}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${OURO_MODEL_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/models/Ouro-1.4B}"
PLUGIN_PATH="${OURO_BAT_PLUGIN_PATH:-$REPO_ROOT/code/Ouro_audio/plugins/ouro_bat_spatial_ast_swift.py}"
DATASET="${BAT_DDP_SMOKE_DATASET:?Set BAT_DDP_SMOKE_DATASET to a private 16-record JSONL manifest}"
OUTPUT_DIR="${BAT_DDP_SMOKE_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/bat/ddp_smoke_8x5090-$(date +%Y%m%d-%H%M%S)}"
REPORT="${BAT_DDP_SMOKE_REPORT:-$OUTPUT_DIR/audit.json}"

case "$OUTPUT_DIR:$REPORT" in
  /hpc_stor03/public*|*:/hpc_stor03/public*) echo "Refusing public output" >&2; exit 2;;
esac
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite DDP smoke output: $OUTPUT_DIR" >&2
  exit 2
fi

echo "========== BAT OURO DDP 8-RANK SMOKE =========="
echo "world_size=8 per_device_batch_size=2 gradient_accumulation_steps=1 global_batch_size=16"
echo "dataset=$DATASET"
echo "output_dir=$OUTPUT_DIR"
echo "launch=torchrun --standalone --nproc_per_node=8"

torchrun --standalone --nproc_per_node=8 \
  code/Ouro_audio/bat/scripts/smoke_bat_ouro_ddp.py \
  --model-path "$MODEL_PATH" \
  --plugin-path "$PLUGIN_PATH" \
  --dataset "$DATASET" \
  --output-dir "$OUTPUT_DIR" \
  --output-report "$REPORT"
