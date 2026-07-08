#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

export CUDA_VISIBLE_DEVICES=0

mkdir -p outputs/huginn_scienceqa_lora


swift sft \
  --model /hpc_stor03/sjtu_home/jinwei.zhang/models/huginn-0125 \
  --model_type huginn_raven \
  --template huginn_text \
  --external_plugins plugins/huginn_swift.py \
  --dataset data/scienceqa/scienceqa_answer_train_sft.jsonl \
  --max_length 512 \
  --output_dir outputs/huginn_scienceqa_lora_alpaca_fullanswer \
  --tuner_type lora \
  --optim_target_modules all-linear \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --resume_from_checkpoint outputs/huginn_scienceqa_lora_alpaca_fullanswer/v7-20260604-105253/checkpoint-1400 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --logging_steps 10 \
  --save_steps 200 \
  --save_total_limit 2 \
  --dataloader_num_workers 0 \
  --dataloader_pin_memory false \
  --dataset_num_proc 1 \
  --report_to none \
  --lora_rank 16 \
  --lora_alpha 64 \
  --bf16 true
