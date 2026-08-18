#!/usr/bin/env bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate swift_ouro

REPO_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/code/Ouro_audio:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29631}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

: "${QWEN3_BAT_DDP_SMOKE_DATASET:?Set QWEN3_BAT_DDP_SMOKE_DATASET to a private 16-record Stage-III JSONL manifest}"

MODEL_PATH="${QWEN3_MODEL_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/models/Qwen3-4B-Base}"
PLUGIN_PATH="${QWEN3_BAT_PLUGIN_PATH:-$REPO_ROOT/code/Ouro_audio/plugins/qwen3_bat_spatial_ast_swift.py}"
OUTPUT_DIR="${QWEN3_BAT_DDP_SMOKE_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/qwen3/ddp_smoke_8x3090-$(date +%Y%m%d-%H%M%S)}"
REPORT="${QWEN3_BAT_DDP_SMOKE_REPORT:-$OUTPUT_DIR/audit.json}"

case "$OUTPUT_DIR:$REPORT" in
  /hpc_stor03/public*|*:/hpc_stor03/public*) echo "Refusing public output" >&2; exit 2;;
esac
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite DDP smoke output: $OUTPUT_DIR" >&2
  exit 2
fi

echo "========== QWEN3 BAT STAGE-III 8-RANK DDP SMOKE =========="
echo "world_size=8 per_device_batch_size=2 gradient_accumulation_steps=1 global_batch_size=16"
echo "route=stage3_ab_cde curriculum=false"
echo "dataset=$QWEN3_BAT_DDP_SMOKE_DATASET"
echo "output_dir=$OUTPUT_DIR"

torchrun --standalone --nproc_per_node=8 \
  code/Ouro_audio/bat/scripts/smoke_qwen3_bat_ddp.py \
  --model-path "$MODEL_PATH" \
  --plugin-path "$PLUGIN_PATH" \
  --dataset "$QWEN3_BAT_DDP_SMOKE_DATASET" \
  --output-dir "$OUTPUT_DIR" \
  --output-report "$REPORT" \
  --expected-records 16 \
  --max-steps 2 \
  --save-steps 2
