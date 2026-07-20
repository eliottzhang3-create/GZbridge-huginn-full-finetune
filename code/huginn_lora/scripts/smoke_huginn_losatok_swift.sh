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
export HUGINN_LOSATOK_TRAIN_CHAIN_AUDIT=1

SOURCE_MANIFEST="$REPO_ROOT/data/audio_swift/audiocaps_v2/audiocaps_v2_train_swift.jsonl"
RUNTIME_DIR="$(mktemp -d /tmp/huginn_losatok_smoke.XXXXXX)"
SMOKE_MANIFEST="$RUNTIME_DIR/audiocaps_v2_one_record.jsonl"
PLUGIN_PATH="$REPO_ROOT/code/huginn_lora/plugins/huginn_losatok_swift.py"
MODEL_PATH="$REPO_ROOT/models/huginn-audio-losatok-v1"
OUTPUT_DIR="${LOSATOK_SMOKE_OUTPUT_DIR:-outputs/huginn_losatok_audiocaps_v2_smoke1_5090}"

on_exit() {
  status=$?
  trap - EXIT
  echo "========== HUGINN LOSATOK SWIFT SMOKE EXIT =========="
  echo "exit_status=$status"
  echo "exit_time=$(date '+%Y-%m-%d %H:%M:%S')"
  exit "$status"
}
trap on_exit EXIT

echo "========== HUGINN LOSATOK SWIFT SMOKE =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "source_manifest=$SOURCE_MANIFEST"
echo "runtime_manifest=$SMOKE_MANIFEST"
echo "model=$MODEL_PATH"
echo "output_dir=$OUTPUT_DIR"
echo "mode=lora_llm frozen_losatok aligner_trainable"
echo "batch_size=1 gradient_accumulation_steps=1 max_steps=1"
echo "checkpoint_saving=disabled tensorboard=disabled"

python - <<'PY'
import torch
import torchaudio
print(f"[env] torch={torch.__version__} torchaudio={torchaudio.__version__} cuda={torch.version.cuda}")
if torch.__version__ != torchaudio.__version__:
    raise SystemExit("Torch and torchaudio versions must match for the LoSATok smoke")
PY

python -u code/huginn_lora/scripts/smoke_huginn_losatok_swift.py \
  --source_manifest "$SOURCE_MANIFEST" \
  --output_manifest "$SMOKE_MANIFEST"

swift sft \
  --model "$MODEL_PATH" \
  --model_type huginn_losatok_raven \
  --template huginn_losatok_text \
  --external_plugins "$PLUGIN_PATH" \
  --dataset "$SMOKE_MANIFEST" \
  --dataset_shuffle false \
  --train_dataloader_shuffle false \
  --sortish_sampler false \
  --group_by_length false \
  --max_length 192 \
  --output_dir "$OUTPUT_DIR" \
  --tuner_type lora_llm \
  --freeze_aligner false \
  --learning_rate 1e-4 \
  --aligner_lr 1e-4 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --max_steps 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing true \
  --logging_steps 1 \
  --save_strategy no \
  --dataloader_num_workers 0 \
  --dataloader_pin_memory false \
  --dataset_num_proc 1 \
  --save_only_model true \
  --report_to none \
  --bf16 true
