#!/usr/bin/env bash
set -euo pipefail

ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune
source /hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/etc/profile.d/conda.sh
conda activate /hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/envs/swift_ouro
cd "$ROOT"
export PYTHONPATH="$ROOT/code/Ouro_audio:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS="${BAT_NCCL_OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${BAT_NCCL_MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${BAT_NCCL_OPENBLAS_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

OUTPUT="${BAT_NCCL_OUTPUT:?Set BAT_NCCL_OUTPUT}"
WORLD="${BAT_NCCL_WORLD_SIZE:-8}"
ARGS=(--output-report "$OUTPUT" --warmup "${BAT_NCCL_WARMUP:-20}" --iterations "${BAT_NCCL_ITERATIONS:-200}" --tensor-elements "${BAT_NCCL_TENSOR_ELEMENTS:-1048576}")
torchrun --standalone --nproc_per_node="$WORLD" code/Ouro_audio/bat/scripts/test_bat_nccl_allreduce.py "${ARGS[@]}"
