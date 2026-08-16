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

# The profiler intentionally makes CPU threading explicit.  The production
# launcher historically used OMP_NUM_THREADS=4 while also creating four
# workers per rank, which can oversubscribe a 32-core allocation.  Override
# these variables from the submission wrapper for controlled comparisons.
export OMP_NUM_THREADS="${BAT_PROFILE_OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${BAT_PROFILE_MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${BAT_PROFILE_OPENBLAS_NUM_THREADS:-1}"

WORLD_SIZE="${BAT_PROFILE_WORLD_SIZE:-8}"
MODEL_PATH="${OURO_MODEL_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/models/Ouro-1.4B}"
PLUGIN_PATH="${OURO_BAT_PLUGIN_PATH:-$REPO_ROOT/code/Ouro_audio/plugins/ouro_bat_spatial_ast_swift.py}"
DATASET="${BAT_PROFILE_DATASET:?Set BAT_PROFILE_DATASET}"
OUTPUT_REPORT="${BAT_PROFILE_OUTPUT_REPORT:?Set BAT_PROFILE_OUTPUT_REPORT}"
STEPS="${BAT_PROFILE_STEPS:-20}"
WARMUP_STEPS="${BAT_PROFILE_WARMUP_STEPS:-3}"
START_INDEX="${BAT_PROFILE_START_INDEX:-0}"
LOCAL_BATCH_SIZE="${BAT_PROFILE_LOCAL_BATCH_SIZE:-2}"
NUM_WORKERS="${BAT_PROFILE_NUM_WORKERS:-4}"
PREFETCH_FACTOR="${BAT_PROFILE_PREFETCH_FACTOR:-2}"
PERSISTENT_WORKERS="${BAT_PROFILE_PERSISTENT_WORKERS:-true}"
TORCH_COMPILE="${BAT_PROFILE_TORCH_COMPILE:-false}"
COMPILE_MODE="${BAT_PROFILE_COMPILE_MODE:-default}"
COMPILE_DYNAMIC="${BAT_PROFILE_COMPILE_DYNAMIC:-true}"
ATTENTION_PROFILE="${BAT_PROFILE_ATTENTION_PROFILE:-false}"

case "$OUTPUT_REPORT" in
  /hpc_stor03/public|/hpc_stor03/public/*)
    echo "Refusing public profiler output: $OUTPUT_REPORT" >&2
    exit 2
    ;;
esac

ARGS=(
  --model-path "$MODEL_PATH"
  --plugin-path "$PLUGIN_PATH"
  --dataset "$DATASET"
  --output-report "$OUTPUT_REPORT"
  --steps "$STEPS"
  --warmup-steps "$WARMUP_STEPS"
  --start-index "$START_INDEX"
  --local-batch-size "$LOCAL_BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --prefetch-factor "$PREFETCH_FACTOR"
  --expected-world-size "$WORLD_SIZE"
)
if [[ "$PERSISTENT_WORKERS" == "true" ]]; then
  ARGS+=(--persistent-workers)
else
  ARGS+=(--no-persistent-workers)
fi
if [[ "$TORCH_COMPILE" == "true" ]]; then
  ARGS+=(--torch-compile --compile-mode "$COMPILE_MODE")
else
  ARGS+=(--no-torch-compile)
fi
if [[ "$COMPILE_DYNAMIC" == "true" ]]; then
  ARGS+=(--compile-dynamic)
else
  ARGS+=(--no-compile-dynamic)
fi
if [[ "$ATTENTION_PROFILE" == "true" ]]; then
  ARGS+=(--attention-profile)
else
  ARGS+=(--no-attention-profile)
fi

echo "========== BAT OURO PURE PROFILING LAUNCH =========="
echo "world_size=$WORLD_SIZE local_batch_size=$LOCAL_BATCH_SIZE global_batch_size=$((WORLD_SIZE * LOCAL_BATCH_SIZE))"
echo "dataset=$DATASET start_index=$START_INDEX steps=$STEPS warmup_steps=$WARMUP_STEPS"
echo "workers=$NUM_WORKERS prefetch_factor=$PREFETCH_FACTOR persistent_workers=$PERSISTENT_WORKERS"
echo "torch_compile=$TORCH_COMPILE compile_mode=$COMPILE_MODE compile_dynamic=$COMPILE_DYNAMIC attention_profile=$ATTENTION_PROFILE"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$MKL_NUM_THREADS OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"

if [[ "$WORLD_SIZE" -le 1 ]]; then
  python -u code/Ouro_audio/bat/scripts/profile_bat_ouro_pipeline.py "${ARGS[@]}"
else
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  export MASTER_PORT="${MASTER_PORT:-29519}"
  torchrun --standalone --nproc_per_node="$WORLD_SIZE" \
    code/Ouro_audio/bat/scripts/profile_bat_ouro_pipeline.py "${ARGS[@]}"
fi
