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
export ACAVCAPS_WDS_MANIFEST="${ACAVCAPS_WDS_MANIFEST:-/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/data/audio_swift/acavcaps_wds/acavcaps_wds_stage_schedule_sampled.json}"
export ACAVCAPS_WDS_BUFFER_SIZE="${ACAVCAPS_WDS_BUFFER_SIZE:-512}"
export ACAVCAPS_WDS_MAX_TARS_PER_STAGE="${ACAVCAPS_WDS_MAX_TARS_PER_STAGE:-2}"
unset HUGINN_LOSATOK_DYNAMIC_AUDIO_TOKENS

PLUGIN_PATH="$REPO_ROOT/code/huginn_lora/plugins/huginn_losatok_acavcaps_wds_swift.py"
MODEL_PATH="$REPO_ROOT/models/huginn-audio-losatok-v1"
OUTPUT_DIR="${ACAVCAPS_WDS_SMOKE_OUTPUT_DIR:-outputs/huginn_losatok_acavcaps_wds_smoke20_b8ga4_5090}"

for required_path in "$ACAVCAPS_WDS_MANIFEST" "$PLUGIN_PATH" "$MODEL_PATH" \
  "/hpc_stor03/sjtu_home/jinwei.zhang/models/LoSATok/ckpts/losatok_kl1e-3.pth" \
  "/hpc_stor03/sjtu_home/jinwei.zhang/models/LoSATok/ckpts/semantic_encoder.pth"; do
  if [ ! -e "$required_path" ]; then
    echo "Required ACAVCAPS LoSATok smoke path is missing: $required_path" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_DIR"
echo "========== ACAVCAPS WEBDATASET HUGINN LOSATOK SWIFT SMOKE =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "manifest=$ACAVCAPS_WDS_MANIFEST"
echo "buffer_size=$ACAVCAPS_WDS_BUFFER_SIZE"
echo "max_tars_per_stage=$ACAVCAPS_WDS_MAX_TARS_PER_STAGE"
echo "model=$MODEL_PATH"
echo "plugin=$PLUGIN_PATH"
echo "output_dir=$OUTPUT_DIR"
echo "mode=legacy_fixed32 frozen_losatok_encoder aligner_trainable huginn_lora_trainable"
echo "selected_data=first_2_tars_per_stage_full_stream"
echo "max_steps=20 per_device_train_batch_size=8 gradient_accumulation_steps=4 effective_batch_size=32"
echo "dataset_shuffle=false train_dataloader_shuffle=false"
echo "checkpoint_saving=disabled dynamic_audio_tokens=disabled"

python - <<'PY'
import torch
import torchaudio
print(f"[env] torch={torch.__version__} torchaudio={torchaudio.__version__} cuda={torch.version.cuda}")
if torch.__version__ != torchaudio.__version__:
    raise SystemExit("Torch and torchaudio versions must match for the LoSATok smoke")
PY

swift sft \
  --model "$MODEL_PATH" \
  --model_type huginn_losatok_raven \
  --template huginn_losatok_text \
  --external_plugins "$PLUGIN_PATH" \
  --dataset "$ACAVCAPS_WDS_MANIFEST" \
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
  --max_steps 20 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing true \
  --logging_steps 1 \
  --save_strategy no \
  --dataloader_num_workers 0 \
  --dataloader_pin_memory false \
  --dataset_num_proc 1 \
  --save_only_model true \
  --report_to none \
  --bf16 true
