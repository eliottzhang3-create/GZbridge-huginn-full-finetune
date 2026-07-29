#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

MASTER_MANIFEST="$REPO_ROOT/data/audio_swift/acavcaps/acavcaps_subset_56_full_master_shuffled.jsonl"
PLUGIN_PATH="$REPO_ROOT/code/huginn_lora/plugins/huginn_audio_swift.py"
MODEL_PATH="$REPO_ROOT/models/huginn-audio-whisper-v1"
OUTPUT_DIR="outputs/huginn_audio_acavcaps_formal_stress100_b4ga4"

if [ ! -s "$MASTER_MANIFEST" ]; then
  echo "Formal master manifest is missing or empty: $MASTER_MANIFEST" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
echo "========== ACAVCAPS HUGINN AUDIO SWIFT FORMAL STRESS 100 =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "mode=lora_llm generator_frozen_audio_encoder aligner_trainable"
echo "dataset=$MASTER_MANIFEST"
echo "output_dir=$OUTPUT_DIR"
echo "max_steps=100"
echo "per_device_train_batch_size=4"
echo "gradient_accumulation_steps=4"
echo "effective_batch_size=16"
echo "save_steps=50"
echo "save_total_limit=2"
echo "save_only_model=false"
echo "report_to=none"

python -u code/huginn_lora/scripts/smoke_acavcaps_huginn_audio_swift.py \
  --manifest "$MASTER_MANIFEST"
python -u code/huginn_lora/scripts/debug_huginn_audio_swift_env.py

swift sft \
  --model "$MODEL_PATH" \
  --model_type huginn_audio_raven \
  --template huginn_audio_text \
  --external_plugins "$PLUGIN_PATH" \
  --dataset "$MASTER_MANIFEST" \
  --max_length 192 \
  --output_dir "$OUTPUT_DIR" \
  --tuner_type lora_llm \
  --freeze_aligner false \
  --learning_rate 1e-4 \
  --aligner_lr 1e-4 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --max_steps 100 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --logging_steps 5 \
  --save_steps 50 \
  --save_total_limit 2 \
  --dataloader_num_workers 0 \
  --dataloader_pin_memory false \
  --dataset_num_proc 1 \
  --save_only_model false \
  --report_to none \
  --bf16 true
