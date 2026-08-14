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
export MASTER_PORT="${MASTER_PORT:-29521}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${OURO_MODEL_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/models/Ouro-1.4B}"
PLUGIN_PATH="${OURO_BAT_PLUGIN_PATH:-$REPO_ROOT/code/Ouro_audio/plugins/ouro_bat_spatial_ast_swift.py}"
DATASET="${BAT_DDP_PREFLIGHT_DATASET:?Set BAT_DDP_PREFLIGHT_DATASET to a private 160-record Stage-I manifest}"
OUTPUT_DIR="${BAT_DDP_PREFLIGHT_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/bat/ddp_preflight_160-$(date +%Y%m%d-%H%M%S)}"
REPORT="${BAT_DDP_PREFLIGHT_REPORT:-$OUTPUT_DIR/audit.json}"

case "$OUTPUT_DIR:$REPORT" in
  /hpc_stor03/public*|*:/hpc_stor03/public*) echo "Refusing public output" >&2; exit 2;;
esac
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite DDP preflight output: $OUTPUT_DIR" >&2
  exit 2
fi

echo "========== BAT OURO DDP 160-RECORD PREFLIGHT =========="
echo "world_size=8 per_device_batch_size=2 gradient_accumulation_steps=1 global_batch_size=16"
echo "dataset_records=160 expected_stage1_steps=20"
echo "dataset=$DATASET"
echo "output_dir=$OUTPUT_DIR"

TRAIN_PID=""
MONITOR_PID=""
resource_monitor() {
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    echo "========== BAT DDP PREFLIGHT RESOURCE SNAPSHOT =========="
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
    sleep 20
  done
}
cleanup() {
  status=$?
  if [[ -n "$MONITOR_PID" ]] && kill -0 "$MONITOR_PID" 2>/dev/null; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
  echo "========== BAT OURO DDP 160-RECORD PREFLIGHT EXIT status=$status =========="
  exit "$status"
}
trap cleanup EXIT

torchrun --standalone --nproc_per_node=8 \
  code/Ouro_audio/bat/scripts/smoke_bat_ouro_ddp.py \
  --model-path "$MODEL_PATH" \
  --plugin-path "$PLUGIN_PATH" \
  --dataset "$DATASET" \
  --output-dir "$OUTPUT_DIR" \
  --output-report "$REPORT" \
  --expected-records 160 \
  --max-steps 20 \
  --save-steps 20 &
TRAIN_PID=$!
resource_monitor &
MONITOR_PID=$!
wait "$TRAIN_PID"
