#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export NPROC_PER_NODE=7
export OMP_NUM_THREADS=4

echo "========== RUN HUGINN AUDIO SWIFT FULL FSDP7 INSPECT =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" code/huginn_lora/scripts/inspect_huginn_audio_swift_full_fsdp.py
