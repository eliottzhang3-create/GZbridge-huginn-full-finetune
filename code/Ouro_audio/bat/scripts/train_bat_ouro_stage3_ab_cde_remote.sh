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
export BAT_AUDIO_AUDIT="${BAT_AUDIO_AUDIT:-0}"
export BAT_MAX_SEQUENCE_LENGTH="${BAT_MAX_SEQUENCE_LENGTH:-176}"
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
echo "max_sequence_length=176"
echo "torch_compile=false eager_transformer=true"

torchrun --standalone --nproc_per_node=8 \
  code/Ouro_audio/bat/scripts/train_bat_ouro_stage3_ab_cde.py "${ARGS[@]}"
