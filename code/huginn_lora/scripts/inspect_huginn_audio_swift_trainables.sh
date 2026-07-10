#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
which python || true
python -V || true

export CUDA_VISIBLE_DEVICES=0

mkdir -p data/audio_swift
mkdir -p outputs/huginn_audio_swift_inspect

DATASET_DIR=/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_aqa_huginn_tiny_train32
RAW_MANIFEST=train.jsonl
SWIFT_MANIFEST="$REPO_ROOT/data/audio_swift/clotho_aqa_tiny_train32_swift.jsonl"

python code/huginn_lora/scripts/prepare_huginn_audio_dataset.py \
  --dataset_dir "$DATASET_DIR" \
  --input_manifest "$RAW_MANIFEST" \
  --output_manifest "$SWIFT_MANIFEST" \
  --task aqa

python code/huginn_lora/scripts/inspect_huginn_audio_swift_trainables.py
