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
export HUGINN_LOSATOK_FORCE_ALIGNER_TRAINABLE=1
unset HUGINN_LOSATOK_DYNAMIC_AUDIO_TOKENS
unset HUGINN_AUDIO_FSDP2_NONPERSISTENT_ROPE

INIT_CHECKPOINT="${LOSATOK_LEGACY_ACAV_WDS_INIT_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_losatok_audiocaps_v2_train_e3_b8ga4_5090/v1-20260720-162632/checkpoint-2802}"
export ACAVCAPS_WDS_MANIFEST="${ACAVCAPS_WDS_MANIFEST:-/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/data/audio_swift/acavcaps_wds/acavcaps_wds_stage_schedule_sampled.json}"
export ACAVCAPS_WDS_BUFFER_SIZE="${ACAVCAPS_WDS_BUFFER_SIZE:-512}"
export ACAVCAPS_WDS_MAX_TARS_PER_STAGE="${ACAVCAPS_WDS_MAX_TARS_PER_STAGE:-2}"

PLUGIN_PATH="$REPO_ROOT/code/huginn_lora/plugins/huginn_losatok_acavcaps_wds_swift.py"
MODEL_PATH="$REPO_ROOT/models/huginn-audio-losatok-v1"
RUN_ROOT="${LOSATOK_LEGACY_ACAV_WDS_SMOKE_ROOT:-outputs/huginn_losatok_acavcaps_wds_legacy_warmstart_save_reload/run-$(date +%Y%m%d_%H%M%S)}"
SAVE_OUTPUT_DIR="$RUN_ROOT/save_phase"
RELOAD_OUTPUT_DIR="$RUN_ROOT/reload_phase"
SAVE_STEPS=2
RELOAD_STEPS=1
BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=4
EXPECTED_LORA_TENSORS=66
EXPECTED_ALIGNER_TENSORS=20

if [ -e "$RUN_ROOT" ]; then
  echo "Smoke run root already exists; choose a fresh LOSATOK_LEGACY_ACAV_WDS_SMOKE_ROOT: $RUN_ROOT" >&2
  exit 1
fi
for required_path in \
  "$ACAVCAPS_WDS_MANIFEST" \
  "$PLUGIN_PATH" \
  "$MODEL_PATH" \
  "$INIT_CHECKPOINT/adapter_model.safetensors" \
  "$INIT_CHECKPOINT/vit.safetensors" \
  "/hpc_stor03/sjtu_home/jinwei.zhang/models/LoSATok/ckpts/losatok_kl1e-3.pth" \
  "/hpc_stor03/sjtu_home/jinwei.zhang/models/LoSATok/ckpts/semantic_encoder.pth"; do
  if [ ! -e "$required_path" ]; then
    echo "Required legacy ACAVCAPS warm-start path is missing: $required_path" >&2
    exit 1
  fi
done

mkdir -p "$SAVE_OUTPUT_DIR" "$RELOAD_OUTPUT_DIR"

inspect_checkpoint() {
  local checkpoint=$1
  local report=$2
  python -u code/huginn_lora/scripts/inspect_swift_huginn_audio_checkpoints.py \
    --checkpoint "$checkpoint" \
    --output_report "$report" \
    --expected_lora_tensor_count "$EXPECTED_LORA_TENSORS" \
    --expected_aligner_tensor_count "$EXPECTED_ALIGNER_TENSORS" \
    --require_boundary_embeddings
}

find_checkpoint() {
  local output_dir=$1
  local checkpoint_name=$2
  mapfile -t matches < <(find "$output_dir" -type d -name "$checkpoint_name" -print | sort)
  if [ "${#matches[@]}" -ne 1 ]; then
    echo "Expected exactly one $checkpoint_name below $output_dir; found ${#matches[@]}" >&2
    printf '  %s\n' "${matches[@]:-<none>}" >&2
    exit 1
  fi
  printf '%s\n' "${matches[0]}"
}

echo "========== ACAVCAPS LEGACY LOSATOK WARM-START SAVE/RELOAD SMOKE =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "mode=legacy_fixed32 dynamic_audio_tokens=disabled fsdp=disabled"
echo "init_checkpoint=$INIT_CHECKPOINT"
echo "manifest=$ACAVCAPS_WDS_MANIFEST buffer_size=$ACAVCAPS_WDS_BUFFER_SIZE max_tars_per_stage=$ACAVCAPS_WDS_MAX_TARS_PER_STAGE"
echo "save_phase=max_steps_$SAVE_STEPS save_steps_$SAVE_STEPS reload_phase=max_steps_$RELOAD_STEPS"
echo "per_device_train_batch_size=$BATCH_SIZE gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS effective_batch_size=$((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "checkpoint_contract=66_lora_plus_20_aligner_plus_audio_bos_eos"

python - <<'PY'
import torch
import torchaudio
print(f"[env] torch={torch.__version__} torchaudio={torchaudio.__version__} cuda={torch.version.cuda}")
if torch.__version__ != torchaudio.__version__:
    raise SystemExit("Torch and torchaudio versions must match for the legacy LoSATok smoke")
PY

echo "========== INPUT CHECKPOINT INSPECT =========="
inspect_checkpoint "$INIT_CHECKPOINT" "$RUN_ROOT/input_checkpoint.inspect.json"

export HUGINN_LOSATOK_INIT_ALIGNER_CHECKPOINT="$INIT_CHECKPOINT"
echo "========== SAVE PHASE =========="
swift sft \
  --model "$MODEL_PATH" \
  --model_type huginn_losatok_raven \
  --template huginn_losatok_text \
  --external_plugins "$PLUGIN_PATH" \
  --dataset "$ACAVCAPS_WDS_MANIFEST" \
  --streaming true \
  --dataset_shuffle false \
  --train_dataloader_shuffle false \
  --sortish_sampler false \
  --group_by_length false \
  --max_length 192 \
  --output_dir "$SAVE_OUTPUT_DIR" \
  --tuner_type lora_llm \
  --freeze_aligner false \
  --learning_rate 1e-4 \
  --aligner_lr 1e-4 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --adapters "$INIT_CHECKPOINT" \
  --load_args false \
  --max_steps "$SAVE_STEPS" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --gradient_checkpointing true \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit 1 \
  --dataloader_num_workers 0 \
  --dataloader_pin_memory false \
  --dataset_num_proc 1 \
  --save_only_model false \
  --report_to none \
  --bf16 true

SAVE_CHECKPOINT="$(find_checkpoint "$SAVE_OUTPUT_DIR" "checkpoint-$SAVE_STEPS")"
echo "========== SAVED CHECKPOINT INSPECT =========="
echo "saved_checkpoint=$SAVE_CHECKPOINT"
inspect_checkpoint "$SAVE_CHECKPOINT" "$RUN_ROOT/saved_checkpoint.inspect.json"

export HUGINN_LOSATOK_INIT_ALIGNER_CHECKPOINT="$SAVE_CHECKPOINT"
echo "========== FRESH WEIGHT RELOAD PHASE =========="
swift sft \
  --model "$MODEL_PATH" \
  --model_type huginn_losatok_raven \
  --template huginn_losatok_text \
  --external_plugins "$PLUGIN_PATH" \
  --dataset "$ACAVCAPS_WDS_MANIFEST" \
  --streaming true \
  --dataset_shuffle false \
  --train_dataloader_shuffle false \
  --sortish_sampler false \
  --group_by_length false \
  --max_length 192 \
  --output_dir "$RELOAD_OUTPUT_DIR" \
  --tuner_type lora_llm \
  --freeze_aligner false \
  --learning_rate 1e-4 \
  --aligner_lr 1e-4 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --adapters "$SAVE_CHECKPOINT" \
  --load_args false \
  --max_steps "$RELOAD_STEPS" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --gradient_checkpointing true \
  --logging_steps 1 \
  --save_strategy no \
  --dataloader_num_workers 0 \
  --dataloader_pin_memory false \
  --dataset_num_proc 1 \
  --save_only_model false \
  --report_to none \
  --bf16 true

echo "========== ACAVCAPS LEGACY LOSATOK WARM-START SAVE/RELOAD SMOKE PASSED =========="
echo "input_checkpoint=$INIT_CHECKPOINT"
echo "saved_checkpoint=$SAVE_CHECKPOINT"
echo "inspection_reports=$RUN_ROOT/input_checkpoint.inspect.json,$RUN_ROOT/saved_checkpoint.inspect.json"
