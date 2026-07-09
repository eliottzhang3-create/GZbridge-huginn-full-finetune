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
mkdir -p outputs/huginn_audio_swift_smoke

DATASET_DIR=/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_aqa_huginn_tiny_train32
RAW_MANIFEST=train.jsonl
SWIFT_MANIFEST="$REPO_ROOT/data/audio_swift/clotho_aqa_tiny_train32_swift.jsonl"
PLUGIN_PATH="$REPO_ROOT/code/huginn_lora/plugins/huginn_audio_swift.py"
MODEL_PATH="$REPO_ROOT/models/huginn-audio-whisper-v1"

python code/huginn_lora/scripts/prepare_huginn_audio_dataset.py \
  --dataset_dir "$DATASET_DIR" \
  --input_manifest "$RAW_MANIFEST" \
  --output_manifest "$SWIFT_MANIFEST" \
  --task aqa

python code/huginn_lora/scripts/smoke_huginn_audio_swift.py

swift sft \
  --model "$MODEL_PATH" \
  --model_type huginn_audio_raven \
  --template huginn_audio_text \
  --external_plugins "$PLUGIN_PATH" \
  --dataset "$SWIFT_MANIFEST" \
  --max_length 192 \
  --output_dir outputs/huginn_audio_swift_smoke \
  --tuner_type lora_llm \
  --freeze_vit true \
  --freeze_aligner false \
  --learning_rate 1e-4 \
  --aligner_lr 1e-4 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --max_steps 4 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --logging_steps 1 \
  --save_steps 4 \
  --save_total_limit 2 \
  --dataloader_num_workers 0 \
  --dataloader_pin_memory false \
  --dataset_num_proc 1 \
  --save_only_model true \
  --report_to none \
  --bf16 true
