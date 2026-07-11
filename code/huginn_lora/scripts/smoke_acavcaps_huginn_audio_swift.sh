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

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

echo "PYTHONUNBUFFERED=$PYTHONUNBUFFERED"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

mkdir -p outputs/huginn_audio_acavcaps_caption_smoke_generator

PILOT_MANIFEST="$REPO_ROOT/data/audio_swift/acavcaps/acavcaps_caption_long_pilot_swift.jsonl"
PLUGIN_PATH="$REPO_ROOT/code/huginn_lora/plugins/huginn_audio_swift.py"
MODEL_PATH="$REPO_ROOT/models/huginn-audio-whisper-v1"

if [ ! -s "$PILOT_MANIFEST" ]; then
  echo "ACAVCAPS pilot manifest is missing or empty: $PILOT_MANIFEST" >&2
  exit 1
fi

python -u code/huginn_lora/scripts/smoke_acavcaps_huginn_audio_swift.py \
  --manifest "$PILOT_MANIFEST"
python -u code/huginn_lora/scripts/debug_huginn_audio_swift_env.py

echo "========== ACAVCAPS HUGINN AUDIO SWIFT SMOKE =========="
echo "mode=lora_llm generator_frozen_audio_encoder aligner_trainable"
echo "dataset=$PILOT_MANIFEST"

swift sft \
  --model "$MODEL_PATH" \
  --model_type huginn_audio_raven \
  --template huginn_audio_text \
  --external_plugins "$PLUGIN_PATH" \
  --dataset "$PILOT_MANIFEST" \
  --max_length 192 \
  --output_dir outputs/huginn_audio_acavcaps_caption_smoke_generator \
  --tuner_type lora_llm \
  --freeze_aligner false \
  --learning_rate 1e-4 \
  --aligner_lr 1e-4 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --max_steps 4 \
  --per_device_train_batch_size 1 \
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
