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
export MASTER_PORT="${MASTER_PORT:-29647}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS="${QWEN3_BAT_COMPILE_THREADS:-2}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export BAT_AUDIO_AUDIT=1
export BAT_MAX_SEQUENCE_LENGTH=176

SOURCE_DATASET="${QWEN3_BAT_COMPILE_SOURCE_DATASET:-/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/manifests/stage3_ab_cde_2epoch.jsonl}"
DATASET="${QWEN3_BAT_COMPILE_DATASET:-/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/manifests/qwen3_bat_compile_ddp_128.jsonl}"
MODEL_PATH="${QWEN3_MODEL_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/models/Qwen3-4B-Base}"
PLUGIN_PATH="${QWEN3_BAT_PLUGIN_PATH:-$REPO_ROOT/code/Ouro_audio/plugins/qwen3_bat_spatial_ast_swift.py}"
OUTPUT_DIR="${QWEN3_BAT_COMPILE_DDP_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/qwen3/compile_ddp_8x_b8-$(date +%Y%m%d-%H%M%S)}"
REPORT="${QWEN3_BAT_COMPILE_DDP_REPORT:-$OUTPUT_DIR/audit.json}"
DATALOADER_NUM_WORKERS="${QWEN3_BAT_COMPILE_DATALOADER_NUM_WORKERS:-0}"

if [[ "$DATALOADER_NUM_WORKERS" =~ ^[0-9]+$ ]]; then
  :
else
  echo "Invalid QWEN3_BAT_COMPILE_DATALOADER_NUM_WORKERS=$DATALOADER_NUM_WORKERS" >&2
  exit 2
fi

case "$DATASET:$OUTPUT_DIR:$REPORT" in
  /hpc_stor03/public*|*:/hpc_stor03/public*) echo "Refusing public path" >&2; exit 2;;
esac
if [[ ! -f "$DATASET" ]]; then
  [[ -f "$SOURCE_DATASET" ]] || { echo "Missing source dataset: $SOURCE_DATASET" >&2; exit 2; }
  mkdir -p "$(dirname "$DATASET")"
  tmp="$DATASET.tmp.$$"
  head -n 128 "$SOURCE_DATASET" > "$tmp"
  mv "$tmp" "$DATASET"
fi
[[ "$(grep -cve '^$' "$DATASET")" -eq 128 ]] || { echo "Compile DDP manifest must contain exactly 128 records: $DATASET" >&2; exit 2; }
[[ ! -e "$OUTPUT_DIR" ]] || { echo "Refusing to overwrite: $OUTPUT_DIR" >&2; exit 2; }

echo "========== QWEN3 BAT COMPILE 8-RANK DDP SMOKE =========="
echo "model=$MODEL_PATH"
echo "dataset=$DATASET records=128"
echo "world_size=8 per_device_batch_size=8 global_batch_size=64 max_steps=2 sequence_length=176"
echo "dataloader_num_workers_per_rank=$DATALOADER_NUM_WORKERS total_worker_processes=$((8 * DATALOADER_NUM_WORKERS))"
echo "inductor_compile_threads_per_rank=$TORCHINDUCTOR_COMPILE_THREADS total_compile_workers=$((8 * TORCHINDUCTOR_COMPILE_THREADS))"
echo "compile_target=Qwen3ForCausalLM.model dynamic=false mode=default"
echo "compile_excluded=Spatial-AST,Q-Former,audio-renderer,lm_head"

torchrun --standalone --nproc_per_node=8 \
  code/Ouro_audio/bat/scripts/smoke_qwen3_bat_ddp.py \
  --model-path "$MODEL_PATH" \
  --plugin-path "$PLUGIN_PATH" \
  --dataset "$DATASET" \
  --output-dir "$OUTPUT_DIR" \
  --output-report "$REPORT" \
  --expected-records 128 \
  --max-steps 2 \
  --save-steps 2 \
  --per-device-batch-size 8 \
  --dataloader-num-workers "$DATALOADER_NUM_WORKERS" \
  --torch-compile \
  --compile-mode default \
  --no-compile-dynamic
