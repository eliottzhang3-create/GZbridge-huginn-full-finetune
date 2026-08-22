#!/usr/bin/env bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_ouro"
REPO_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/code/Ouro_audio:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export BAT_AUDIO_AUDIT=0
export BAT_FIXED_SEQUENCE_LENGTH=false
export BAT_MAX_SEQUENCE_LENGTH=512
export BAT_RUNTIME_MONITOR_INTERVAL_STEPS="${BAT_RUNTIME_MONITOR_INTERVAL_STEPS:-500}"
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

WORLD_SIZE=8
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi

MODEL_PATH="${OURO_MODEL_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/models/Ouro-1.4B}"
PLUGIN_PATH="${OURO_BAT_PLUGIN_PATH:-$REPO_ROOT/code/Ouro_audio/plugins/ouro_bat_spatial_ast_swift.py}"
DATASET="${BAT_STAGE3_AB_CDE_MANIFEST:?Set BAT_STAGE3_AB_CDE_MANIFEST}"
REPORT="${BAT_STAGE3_AB_CDE_REPORT:?Set BAT_STAGE3_AB_CDE_REPORT}"
OUTPUT_DIR="${BAT_STAGE3_AB_CDE_OUTPUT_DIR:?Set BAT_STAGE3_AB_CDE_OUTPUT_DIR}"

# Arrow metadata is built once by this single process and then reused by all
# eight DDP ranks.  /tmp is job-local; a fresh job therefore intentionally
# rebuilds the cache.  The report lives beside (not inside) OUTPUT_DIR so the
# fresh-run non-empty-output guard remains meaningful.
LOCAL_CACHE_ROOT="${BAT_LOCAL_CACHE_ROOT:-/tmp/bat_ouro_arrow_cache_${USER:-user}_$$}"
LOCAL_ARROW_CACHE="$LOCAL_CACHE_ROOT/datasets"
LOCAL_MODELSCOPE_CACHE="$LOCAL_CACHE_ROOT/modelscope"
PREWARM_REPORT="${BAT_ARROW_PREWARM_REPORT:-${OUTPUT_DIR}.arrow_cache_prewarm.json}"
case "$LOCAL_CACHE_ROOT" in
  /tmp/*) ;;
  *) echo "Refusing non-local cache path: $LOCAL_CACHE_ROOT" >&2; exit 2 ;;
esac
export BAT_LOCAL_ARROW_CACHE="$LOCAL_ARROW_CACHE"
export BAT_LOCAL_CACHE_ROOT="$LOCAL_CACHE_ROOT"
export HF_DATASETS_CACHE="$LOCAL_ARROW_CACHE"
export MODELSCOPE_CACHE="$LOCAL_MODELSCOPE_CACHE"
export BAT_ARROW_PREWARM_REPORT="$PREWARM_REPORT"

mkdir -p "$LOCAL_ARROW_CACHE"
mkdir -p "$LOCAL_MODELSCOPE_CACHE"
echo "[cache] prewarming local Arrow cache=$LOCAL_ARROW_CACHE"
echo "[cache] local ModelScope cache=$MODELSCOPE_CACHE"
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
if [[ -n "${BAT_STAGE3_AB_CDE_RESUME_FROM_CHECKPOINT:-}" ]]; then
  ARGS+=(--resume-from-checkpoint "$BAT_STAGE3_AB_CDE_RESUME_FROM_CHECKPOINT")
fi

echo "========== BAT OURO STAGE-III A+B -> C+D+E LAUNCH =========="
echo "world_size=$WORLD_SIZE per_device_batch_size=8 global_batch_size=64"
echo "dataset=$DATASET"
echo "report=$REPORT"
echo "output_dir=$OUTPUT_DIR"
echo "dataloader_num_workers=0 pin_memory=false"
echo "padding=dynamic_batch max_length_ceiling=512 fixed_sequence_length=false"
echo "torch_compile=false eager_transformer=true"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "MODELSCOPE_CACHE=$MODELSCOPE_CACHE"
echo "runtime_monitor_interval_steps=$BAT_RUNTIME_MONITOR_INTERVAL_STEPS"
echo "arrow_prewarm_report=$PREWARM_REPORT"
echo "python_faulthandler=$PYTHONFAULTHANDLER"

torchrun --standalone --nproc_per_node=8 \
  code/Ouro_audio/bat/scripts/train_bat_ouro_stage3_ab_cde.py "${ARGS[@]}"
