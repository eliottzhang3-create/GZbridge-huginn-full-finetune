#!/usr/bin/env bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate swift_ouro

REPO_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/code/Ouro_audio:$REPO_ROOT/code/Ouro_audio/bat/scripts:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# Each DataLoader worker is a process.  Keep one BLAS/OpenMP thread per
# worker so the 8-worker case does not oversubscribe the 8 allocated CPUs.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

DATASET="${OURO_BAT_DATALOADER_DATASET:-/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/manifests/stage3_ab_cde_2epoch.jsonl}"
MODEL_PATH="${OURO_MODEL_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/models/Ouro-1.4B}"
PLUGIN_PATH="${OURO_BAT_PLUGIN_PATH:-$REPO_ROOT/code/Ouro_audio/plugins/ouro_bat_spatial_ast_swift.py}"
OUTPUT_REPORT="${OURO_BAT_DATALOADER_OUTPUT_REPORT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/bat/dataloader_profile_stage3_ab_cde.json}"

case "$OUTPUT_REPORT" in
  /hpc_stor03/public|/hpc_stor03/public/*) echo "Refusing public output" >&2; exit 2;;
esac

echo "========== OURO BAT DATALOADER-ONLY PROFILING =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "PYTHON=$(which python)"
echo "dataset=$DATASET"
echo "model=$MODEL_PATH"
echo "plugin=$PLUGIN_PATH"
echo "output=$OUTPUT_REPORT"
echo "workers=0,2,4,8 local_batch_size=8 assumed_global_batch_size=64"
echo "warmup=${OURO_BAT_DATALOADER_WARMUP_BATCHES:-2} measure=${OURO_BAT_DATALOADER_MEASURE_BATCHES:-5}"

python -u code/Ouro_audio/bat/scripts/profile_ouro_bat_dataloader.py \
  --model-path "$MODEL_PATH" \
  --plugin-path "$PLUGIN_PATH" \
  --dataset "$DATASET" \
  --output-report "$OUTPUT_REPORT" \
  --start-index "${OURO_BAT_DATALOADER_START_INDEX:-0}" \
  --warmup-batches "${OURO_BAT_DATALOADER_WARMUP_BATCHES:-2}" \
  --measure-batches "${OURO_BAT_DATALOADER_MEASURE_BATCHES:-5}" \
  --prefetch-factor "${OURO_BAT_DATALOADER_PREFETCH_FACTOR:-2}" \
  --persistent-workers \
  --pin-memory
