#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

export CUDA_VISIBLE_DEVICES=0

mkdir -p outputs/huginn_sft_lora

swift sft \
  --model /hpc_stor03/sjtu_home/jinwei.zhang/models/huginn-0125 \
  --model_type huginn_raven \
  --template huginn_text \
  --external_plugins plugins/huginn_swift.py \
  --dataset data/gsm8k_1000.jsonl \
  --max_length 1024 \
  --output_dir outputs/huginn_sft_lora \
  --tuner_type lora \
  --optim_target_modules all-linear \
  --learning_rate 1e-4 \
  --max_steps 5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --logging_steps 1 \
  --save_steps 5 \
  --save_total_limit 2 \
  --report_to none \
  --bf16 true
