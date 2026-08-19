#!/usr/bin/env bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_ouro"

REPO_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/code/Ouro_audio:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29667}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export BAT_MAX_SEQUENCE_LENGTH=176
export BAT_AUDIO_AUDIT=0
export BAT_RUNTIME_MONITOR_INTERVAL_STEPS="${BAT_RUNTIME_MONITOR_INTERVAL_STEPS:-500}"
export PYTHONFAULTHANDLER=1

WORLD_SIZE=8
MODEL_PATH="${QWEN3_MODEL_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/models/Qwen3-4B-Base}"
PLUGIN_PATH="${QWEN3_BAT_PLUGIN_PATH:-$REPO_ROOT/code/Ouro_audio/plugins/qwen3_bat_spatial_ast_swift.py}"
DATASET="${QWEN3_BAT_STAGE3_AB_CDE_MANIFEST:?Set QWEN3_BAT_STAGE3_AB_CDE_MANIFEST}"
REPORT="${QWEN3_BAT_STAGE3_AB_CDE_REPORT:?Set QWEN3_BAT_STAGE3_AB_CDE_REPORT}"
OUTPUT_DIR="${QWEN3_BAT_STAGE3_AB_CDE_OUTPUT_DIR:?Set QWEN3_BAT_STAGE3_AB_CDE_OUTPUT_DIR}"

# Build the local Arrow cache exactly once before torchrun.  All ranks on this
# single node then read the completed files; they do not race to create them.
LOCAL_ARROW_CACHE="${BAT_LOCAL_CACHE_ROOT:-/tmp/bat_qwen3_arrow_cache_${USER:-user}_$$}/datasets"
PREWARM_REPORT="${BAT_ARROW_PREWARM_REPORT:-${OUTPUT_DIR}.arrow_cache_prewarm.json}"
case "$LOCAL_ARROW_CACHE" in
  /tmp/*) ;;
  *) echo "Refusing non-local Arrow cache path: $LOCAL_ARROW_CACHE" >&2; exit 2 ;;
esac
export BAT_LOCAL_ARROW_CACHE="$LOCAL_ARROW_CACHE"
export HF_DATASETS_CACHE="$LOCAL_ARROW_CACHE"
export BAT_ARROW_PREWARM_REPORT="$PREWARM_REPORT"

mkdir -p "$LOCAL_ARROW_CACHE"
echo "[cache] prewarming local Arrow cache=$LOCAL_ARROW_CACHE"
python -u code/Ouro_audio/bat/scripts/prewarm_bat_arrow_cache.py \
  --manifest "$DATASET" \
  --cache-dir "$LOCAL_ARROW_CACHE" \
  --report "$PREWARM_REPORT"

VISIBLE_GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())')"
if [[ "$VISIBLE_GPU_COUNT" != "8" ]]; then
  echo "Expected 8 visible GPUs, got $VISIBLE_GPU_COUNT" >&2
  exit 2
fi

ARGS=(
  --model-path "$MODEL_PATH"
  --plugin-path "$PLUGIN_PATH"
  --dataset "$DATASET"
  --report "$REPORT"
  --output-dir "$OUTPUT_DIR"
  --world-size "$WORLD_SIZE"
)
if [[ -n "${QWEN3_BAT_STAGE3_AB_CDE_RESUME_FROM_CHECKPOINT:-}" ]]; then
  ARGS+=(--resume-from-checkpoint "$QWEN3_BAT_STAGE3_AB_CDE_RESUME_FROM_CHECKPOINT")
fi

case "$OUTPUT_DIR:$REPORT" in
  /hpc_stor03/public*|*:/hpc_stor03/public*) echo "Refusing public output path" >&2; exit 2;;
esac

echo "========== QWEN3-4B BAT STAGE-III A+B -> C+D+E LAUNCH =========="
echo "world_size=$WORLD_SIZE per_device_batch_size=8 global_batch_size=64"
echo "dataset=$DATASET"
echo "report=$REPORT"
echo "output_dir=$OUTPUT_DIR"
echo "dataloader_num_workers_per_rank=0"
echo "torch_compile=false eager_transformer=true"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "runtime_monitor_interval_steps=$BAT_RUNTIME_MONITOR_INTERVAL_STEPS"
echo "arrow_prewarm_report=$PREWARM_REPORT"
echo "python_faulthandler=$PYTHONFAULTHANDLER"

torchrun --standalone --nproc_per_node=8 \
  code/Ouro_audio/bat/scripts/train_qwen3_bat_stage3_ab_cde.py "${ARGS[@]}"
