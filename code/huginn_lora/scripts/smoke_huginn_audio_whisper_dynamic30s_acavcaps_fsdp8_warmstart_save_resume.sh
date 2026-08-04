#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORLD_SIZE=8
PER_DEVICE_BATCH=1
GRADIENT_ACCUMULATION_STEPS=1
SAVE_STEP=2
RESUME_STEP=3
SEED="${ACAVCAPS_FLAT_SEED:-20260723}"
MAX_TARS="${ACAVCAPS_FLAT_MAX_TARS:-2}"
INIT_CHECKPOINT="${HUGINN_AUDIO_DYNAMIC30S_ACAV_WARMSTART_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_whisper_dynamic30s_multiplier_single_epoch_fsdp4/run-20260731_084946/swift_output/v0-20260731-085036/checkpoint-46050}"
MANIFEST="${ACAVCAPS_FLAT_MANIFEST:-$REPO_ROOT/data/audio_swift/acavcaps/acavcaps_flat_global_tar_shuffle_seed20260723.json}"
RUN_ROOT="${HUGINN_AUDIO_DYNAMIC30S_ACAV_SMOKE_ROOT:-outputs/huginn_audio_whisper_dynamic30s_acavcaps_fsdp8_warmstart_save_resume/run-$(date +%Y%m%d_%H%M%S)}"
SAVE_OUTPUT_DIR="$RUN_ROOT/save_phase"
RESUME_OUTPUT_DIR="$RUN_ROOT/resume_phase"
AUDIT_DIR="$RUN_ROOT/audit"
FSDP_CONFIG_PATH="$RUN_ROOT/fsdp2_acavcaps_fsdp8.json"
REPORT_PATH="$RUN_ROOT/acavcaps_warmstart_resume_report.json"
PLUGIN_PATH="$REPO_ROOT/code/huginn_lora/plugins/huginn_audio_whisper_dynamic30s_acavcaps_swift.py"
MODEL_PATH="$REPO_ROOT/models/huginn-audio-whisper-dynamic90s-v1"
MANIFEST_INSPECTOR="$REPO_ROOT/code/huginn_lora/scripts/inspect_acavcaps_flat_global_tar_manifest.py"
CHECKPOINT_INSPECTOR="$REPO_ROOT/code/huginn_lora/scripts/inspect_huginn_audio_whisper_dynamic30s_acavcaps_warmstart_resume.py"
MODULES_TO_SAVE=(temporal_compressor audio_projector audio_boundary_embeddings)

if [ -e "$RUN_ROOT" ]; then
  echo "ACAVCAPS FSDP8 smoke root already exists: $RUN_ROOT" >&2
  exit 1
fi
if ! [[ "$MAX_TARS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ACAVCAPS_FLAT_MAX_TARS must be a positive integer, got: $MAX_TARS" >&2
  exit 1
fi
for required_path in \
  "$INIT_CHECKPOINT/pytorch_model_fsdp_0" \
  "$MANIFEST" \
  "${MANIFEST%.json}.stats.json" \
  "$PLUGIN_PATH" \
  "$MODEL_PATH" \
  "$MANIFEST_INSPECTOR" \
  "$CHECKPOINT_INSPECTOR"; do
  if [ ! -e "$required_path" ]; then
    echo "Required ACAVCAPS FSDP8 smoke path is missing: $required_path" >&2
    exit 1
  fi
done

mkdir -p "$SAVE_OUTPUT_DIR" "$RESUME_OUTPUT_DIR" "$AUDIT_DIR"
RUN_ROOT="$(cd "$RUN_ROOT" && pwd)"
SAVE_OUTPUT_DIR="$RUN_ROOT/save_phase"
RESUME_OUTPUT_DIR="$RUN_ROOT/resume_phase"
AUDIT_DIR="$RUN_ROOT/audit"
FSDP_CONFIG_PATH="$RUN_ROOT/fsdp2_acavcaps_fsdp8.json"
REPORT_PATH="$RUN_ROOT/acavcaps_warmstart_resume_report.json"

FSDP_CONFIG='{"fsdp":"full_shard auto_wrap","fsdp_config":{"activation_checkpointing":true,"auto_wrap_policy":"TRANSFORMER_BASED_WRAP","cpu_ram_efficient_loading":true,"fsdp_version":2,"reshard_after_forward":true,"state_dict_type":"SHARDED_STATE_DICT"}}'
printf '%s\n' "$FSDP_CONFIG" > "$FSDP_CONFIG_PATH"

export ACAVCAPS_FLAT_MANIFEST="$MANIFEST"
export ACAVCAPS_FLAT_SAMPLE_SHUFFLE_BUFFER=512
export ACAVCAPS_FLAT_MAX_TARS="$MAX_TARS"
export HUGINN_AUDIO_DYNAMIC90S_FSDP2_NONPERSISTENT_ROPE=1
export HUGINN_AUDIO_DYNAMIC90S_TRAIN_CHAIN_AUDIT=1
export HUGINN_AUDIO_DYNAMIC90S_PEFT_ALIGNER_MODULES_TO_SAVE=1
export HUGINN_AUDIO_DYNAMIC90S_FULL_MODEL_DCP=1
export HUGINN_AUDIO_DYNAMIC30S_RECURRENT_CORE_RESHARD_AFTER_FORWARD_FALSE=1
export HUGINN_AUDIO_DYNAMIC30S_RECURRENT_CORE_EXPECTED_WORLD_SIZE="$WORLD_SIZE"
export HUGINN_AUDIO_DYNAMIC90S_MODEL_ONLY_WARMSTART_AUDIT_DIR="$AUDIT_DIR"
export HUGINN_AUDIO_DYNAMIC90S_MODEL_ONLY_WARMSTART_EXPECTED_WORLD_SIZE="$WORLD_SIZE"
export HUGINN_AUDIO_DYNAMIC90S_MODEL_ONLY_WARMSTART_AUDIT_MODE=fresh
export HUGINN_AUDIO_DYNAMIC90S_MODEL_ONLY_WARMSTART_EXPECTED_START_STEP=0
export HUGINN_AUDIO_DYNAMIC90S_MODEL_ONLY_WARMSTART_PHASE=acavcaps_warmstart
export HUGINN_AUDIO_DYNAMIC90S_INIT_FSDP_DCP_CHECKPOINT="$INIT_CHECKPOINT"
unset HUGINN_AUDIO_DYNAMIC90S_INIT_ALIGNER_CHECKPOINT
unset HUGINN_AUDIO_DYNAMIC90S_CHECKPOINT_AUDIT_DIR

echo "========== ACAVCAPS FLAT GLOBAL MANIFEST AUDIT =========="
python -u "$MANIFEST_INSPECTOR" \
  --manifest "$MANIFEST" \
  --world_size "$WORLD_SIZE" \
  --per_device_batch_size "$PER_DEVICE_BATCH" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --check_tar_files

echo "========== ACAVCAPS FSDP8 MODEL-ONLY WARM-START SAVE SMOKE =========="
echo "input_checkpoint=$INIT_CHECKPOINT"
echo "manifest=$MANIFEST"
echo "manifest_scope=all_1071_tars_flat_global_order smoke_max_tars=$MAX_TARS"
echo "world_size=$WORLD_SIZE per_device_batch=$PER_DEVICE_BATCH gradient_accumulation=$GRADIENT_ACCUMULATION_STEPS"
echo "phase1=model_weights_only_new_optimizer_scheduler_global_step_rng_data_position"

swift sft \
  --model "$MODEL_PATH" \
  --model_type huginn_audio_whisper_dynamic90s \
  --template huginn_audio_whisper_dynamic90s \
  --external_plugins "$PLUGIN_PATH" \
  --dataset huginn_audio_whisper_dynamic30s_acavcaps \
  --streaming true \
  --dataset_shuffle false \
  --train_dataloader_shuffle false \
  --sortish_sampler false \
  --group_by_length false \
  --max_length 192 \
  --output_dir "$SAVE_OUTPUT_DIR" \
  --tuner_type lora_llm \
  --freeze_vit false \
  --freeze_aligner false \
  --modules_to_save "${MODULES_TO_SAVE[@]}" \
  --learning_rate 1e-4 \
  --aligner_lr 1e-4 \
  --vit_lr 1e-4 \
  --lr_scheduler_type constant \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --adapters "$INIT_CHECKPOINT" \
  --load_args false \
  --fsdp "$FSDP_CONFIG_PATH" \
  --max_steps "$SAVE_STEP" \
  --per_device_train_batch_size "$PER_DEVICE_BATCH" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --gradient_checkpointing false \
  --vit_gradient_checkpointing false \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps "$SAVE_STEP" \
  --save_total_limit 1 \
  --dataloader_num_workers 0 \
  --dataloader_pin_memory false \
  --dataset_num_proc 1 \
  --save_only_model false \
  --report_to none \
  --bf16 true \
  --seed "$SEED" \
  --data_seed "$SEED"

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

SAVE_CHECKPOINT="$(find_checkpoint "$SAVE_OUTPUT_DIR" "checkpoint-$SAVE_STEP")"
if [ ! -s "$SAVE_CHECKPOINT/model_only_warmstart.json" ]; then
  echo "Model-only warm-start audit was not written into the save checkpoint" >&2
  exit 1
fi

echo "========== ACAVCAPS FSDP8 COLD RESUME SMOKE =========="
echo "saved_checkpoint=$SAVE_CHECKPOINT"
echo "phase2=resume_new_8card_smoke_checkpoint_only"
unset HUGINN_AUDIO_DYNAMIC90S_INIT_FSDP_DCP_CHECKPOINT
export HUGINN_AUDIO_DYNAMIC90S_MODEL_ONLY_WARMSTART_AUDIT_MODE=resume
export HUGINN_AUDIO_DYNAMIC90S_MODEL_ONLY_WARMSTART_EXPECTED_START_STEP="$SAVE_STEP"
export HUGINN_AUDIO_DYNAMIC90S_MODEL_ONLY_WARMSTART_PHASE=acavcaps_cold_resume

swift sft \
  --model "$MODEL_PATH" \
  --model_type huginn_audio_whisper_dynamic90s \
  --template huginn_audio_whisper_dynamic90s \
  --external_plugins "$PLUGIN_PATH" \
  --dataset huginn_audio_whisper_dynamic30s_acavcaps \
  --streaming true \
  --dataset_shuffle false \
  --train_dataloader_shuffle false \
  --sortish_sampler false \
  --group_by_length false \
  --max_length 192 \
  --output_dir "$RESUME_OUTPUT_DIR" \
  --tuner_type lora_llm \
  --freeze_vit false \
  --freeze_aligner false \
  --modules_to_save "${MODULES_TO_SAVE[@]}" \
  --learning_rate 1e-4 \
  --aligner_lr 1e-4 \
  --vit_lr 1e-4 \
  --lr_scheduler_type constant \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --fsdp "$FSDP_CONFIG_PATH" \
  --resume_from_checkpoint "$SAVE_CHECKPOINT" \
  --ignore_data_skip true \
  --max_steps "$RESUME_STEP" \
  --per_device_train_batch_size "$PER_DEVICE_BATCH" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --gradient_checkpointing false \
  --vit_gradient_checkpointing false \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps "$RESUME_STEP" \
  --save_total_limit 1 \
  --dataloader_num_workers 0 \
  --dataloader_pin_memory false \
  --dataset_num_proc 1 \
  --save_only_model false \
  --report_to none \
  --bf16 true \
  --seed "$SEED" \
  --data_seed "$SEED"

RESUME_CHECKPOINT="$(find_checkpoint "$RESUME_OUTPUT_DIR" "checkpoint-$RESUME_STEP")"
python -u "$CHECKPOINT_INSPECTOR" \
  --save-checkpoint "$SAVE_CHECKPOINT" \
  --resume-checkpoint "$RESUME_CHECKPOINT" \
  --save-step "$SAVE_STEP" \
  --resume-step "$RESUME_STEP" \
  --world-size "$WORLD_SIZE" \
  --output-report "$REPORT_PATH"

echo "========== ACAVCAPS FSDP8 WARM-START/SAVE/RESUME SMOKE PASSED =========="
echo "input_checkpoint=$INIT_CHECKPOINT"
echo "saved_checkpoint=$SAVE_CHECKPOINT"
echo "resume_checkpoint=$RESUME_CHECKPOINT"
echo "report=$REPORT_PATH"
