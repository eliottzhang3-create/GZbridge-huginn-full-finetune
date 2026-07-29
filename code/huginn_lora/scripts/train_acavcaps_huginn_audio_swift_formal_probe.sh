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

FORMAL_MANIFEST="${FORMAL_MANIFEST:-$REPO_ROOT/data/audio_swift/acavcaps/subset_56_full_1tar_chunks/acavcaps_caption_long_formal_chunk_000.jsonl}"
FORMAL_PROBE_NAME="${FORMAL_PROBE_NAME:-formal_chunk000_smoke_b1ga2}"
FORMAL_MAX_STEPS="${FORMAL_MAX_STEPS:-4}"
FORMAL_BATCH_SIZE="${FORMAL_BATCH_SIZE:-1}"
FORMAL_GRADIENT_ACCUMULATION_STEPS="${FORMAL_GRADIENT_ACCUMULATION_STEPS:-2}"
OUTPUT_DIR="outputs/huginn_audio_acavcaps_${FORMAL_PROBE_NAME}"
PLUGIN_PATH="$REPO_ROOT/code/huginn_lora/plugins/huginn_audio_swift.py"
MODEL_PATH="$REPO_ROOT/models/huginn-audio-whisper-v1"

if [ ! -s "$FORMAL_MANIFEST" ]; then
  echo "Formal ACAVCAPS manifest is missing or empty: $FORMAL_MANIFEST" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
echo "========== ACAVCAPS HUGINN AUDIO SWIFT FORMAL PROBE =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "mode=lora_llm generator_frozen_audio_encoder aligner_trainable"
echo "dataset=$FORMAL_MANIFEST"
echo "output_dir=$OUTPUT_DIR"
echo "max_steps=$FORMAL_MAX_STEPS"
echo "per_device_train_batch_size=$FORMAL_BATCH_SIZE"
echo "gradient_accumulation_steps=$FORMAL_GRADIENT_ACCUMULATION_STEPS"
echo "effective_batch_size=$((FORMAL_BATCH_SIZE * FORMAL_GRADIENT_ACCUMULATION_STEPS))"

python -u code/huginn_lora/scripts/smoke_acavcaps_huginn_audio_swift.py \
  --manifest "$FORMAL_MANIFEST"
python -u code/huginn_lora/scripts/debug_huginn_audio_swift_env.py

swift sft \
  --model "$MODEL_PATH" \
  --model_type huginn_audio_raven \
  --template huginn_audio_text \
  --external_plugins "$PLUGIN_PATH" \
  --dataset "$FORMAL_MANIFEST" \
  --max_length 192 \
  --output_dir "$OUTPUT_DIR" \
  --tuner_type lora_llm \
  --freeze_aligner false \
  --learning_rate 1e-4 \
  --aligner_lr 1e-4 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --max_steps "$FORMAL_MAX_STEPS" \
  --per_device_train_batch_size "$FORMAL_BATCH_SIZE" \
  --gradient_accumulation_steps "$FORMAL_GRADIENT_ACCUMULATION_STEPS" \
  --logging_steps 1 \
  --save_steps "$FORMAL_MAX_STEPS" \
  --save_total_limit 1 \
  --dataloader_num_workers 0 \
  --dataloader_pin_memory false \
  --dataset_num_proc 1 \
  --save_only_model true \
  --report_to none \
  --bf16 true
