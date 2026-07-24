#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export OMP_NUM_THREADS=4
export HUGINN_AUDIO_FSDP2_NONPERSISTENT_ROPE=1
export HUGINN_LOSATOK_DYNAMIC_AUDIO_TOKENS=1
export HUGINN_LOSATOK_TRAIN_CHAIN_AUDIT=1
unset HUGINN_LOSATOK_FORCE_ALIGNER_TRAINABLE

TRAIN_MANIFEST="${LOSATOK_DYNAMIC_MODULES_SAVE_SMOKE_MANIFEST:-$REPO_ROOT/data/audio_swift/audiocaps_v2/audiocaps_v2_train_swift.jsonl}"
TRAIN_STATS="$TRAIN_MANIFEST.stats.json"
RUN_ROOT="${LOSATOK_DYNAMIC_MODULES_SAVE_SMOKE_ROOT:-outputs/huginn_losatok_dynamic90s_modules_save_fsdp2_smoke_resume/run-$(date +%Y%m%d_%H%M%S)}"
SAVE_OUTPUT_DIR="$RUN_ROOT/save_phase"
RESUME_OUTPUT_DIR="$RUN_ROOT/resume_phase"
PLUGIN_PATH="$REPO_ROOT/code/huginn_lora/plugins/huginn_losatok_swift.py"
MODEL_PATH="$REPO_ROOT/models/huginn-audio-losatok-v1"
SAVE_STEPS=2
RESUME_MAX_STEPS=3
WORLD_SIZE=2
MICRO_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
MODULES_TO_SAVE=(temporal_compressor audio_projector audio_boundary_embeddings)
FSDP_CONFIG='{"fsdp":"full_shard auto_wrap","fsdp_config":{"activation_checkpointing":false,"auto_wrap_policy":"TRANSFORMER_BASED_WRAP","cpu_ram_efficient_loading":true,"fsdp_version":2,"reshard_after_forward":true,"state_dict_type":"SHARDED_STATE_DICT"}}'

if [ -e "$RUN_ROOT" ]; then
  echo "Modules-to-save FSDP2 smoke root already exists; choose a fresh LOSATOK_DYNAMIC_MODULES_SAVE_SMOKE_ROOT: $RUN_ROOT" >&2
  exit 1
fi
for required_path in "$TRAIN_MANIFEST" "$TRAIN_STATS" "$PLUGIN_PATH" "$MODEL_PATH"; do
  if [ ! -e "$required_path" ]; then
    echo "Required modules-to-save FSDP2 smoke path is missing: $required_path" >&2
    exit 1
  fi
done

python - "$TRAIN_STATS" <<'PY'
import json
import sys
from dataclasses import fields

with open(sys.argv[1], encoding='utf-8') as handle:
    stats = json.load(handle)
if stats.get('dataset') != 'audiocaps_v2' or stats.get('split') != 'train':
    raise SystemExit(f"Unexpected AudioCaps stats: {stats.get('dataset')!r}/{stats.get('split')!r}")
if not isinstance(stats.get('record_count'), int) or stats['record_count'] <= 0:
    raise SystemExit(f"Unexpected AudioCaps record_count: {stats.get('record_count')!r}")
if stats.get('audio_path_verification') != 'passed' or stats.get('wav_readability_verification') != 'passed':
    raise SystemExit('AudioCaps manifest verification is not passed')
from swift.arguments.sft_args import SftArguments
available = {field.name for field in fields(SftArguments)}
required = {'fsdp', 'modules_to_save', 'resume_from_checkpoint', 'save_strategy', 'save_steps', 'save_only_model'}
missing = sorted(required - available)
if missing:
    raise SystemExit(f'Installed Swift lacks required modules-to-save FSDP arguments: {missing}')
print(f"[preflight] audiocaps_record_count={stats['record_count']}")
print('[preflight] swift_modules_to_save_and_resume_arguments=present')
PY

mkdir -p "$SAVE_OUTPUT_DIR" "$RESUME_OUTPUT_DIR"
printf '%s\n' "$FSDP_CONFIG" > "$SAVE_OUTPUT_DIR/fsdp2.json"
printf '%s\n' "$FSDP_CONFIG" > "$RESUME_OUTPUT_DIR/fsdp2.json"

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

audit_checkpoint() {
  python -u code/huginn_lora/scripts/inspect_losatok_dynamic_fsdp_checkpoint.py \
    --checkpoint "$1" \
    --require_complete
}

echo "========== LOSATOK DYNAMIC FSDP2 MODULES-TO-SAVE SAVE/RESUME SMOKE =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "world_size=$WORLD_SIZE per_device_batch=$MICRO_BATCH_SIZE accumulation=$GRADIENT_ACCUMULATION_STEPS"
echo "dynamic_audio_prefix=90_seconds compressor=kernel11_stride6 adaptive_pool=false"
echo "modules_to_save=${MODULES_TO_SAVE[*]}"
echo "required_dcp_tensor_contract=lora_66+aligner_20"
echo "phase1=max_steps_$SAVE_STEPS save_checkpoint_$SAVE_STEPS"
echo "phase2=fresh_process_resume_checkpoint_$SAVE_STEPS max_steps_$RESUME_MAX_STEPS save_checkpoint_$RESUME_MAX_STEPS"

echo "========== SAVE PHASE =========="
swift sft \
  --model "$MODEL_PATH" \
  --model_type huginn_losatok_raven \
  --template huginn_losatok_text \
  --external_plugins "$PLUGIN_PATH" \
  --dataset "$TRAIN_MANIFEST" \
  --dataset_shuffle false \
  --train_dataloader_shuffle false \
  --sortish_sampler false \
  --group_by_length false \
  --max_length 192 \
  --output_dir "$SAVE_OUTPUT_DIR" \
  --tuner_type lora_llm \
  --freeze_aligner false \
  --modules_to_save "${MODULES_TO_SAVE[@]}" \
  --learning_rate 1e-4 \
  --aligner_lr 1e-4 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --fsdp "$SAVE_OUTPUT_DIR/fsdp2.json" \
  --max_steps "$SAVE_STEPS" \
  --per_device_train_batch_size "$MICRO_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --gradient_checkpointing false \
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
echo "========== SAVE CHECKPOINT DCP AUDIT =========="
echo "save_checkpoint=$SAVE_CHECKPOINT"
audit_checkpoint "$SAVE_CHECKPOINT"

echo "========== FRESH FSDP RESUME PHASE =========="
swift sft \
  --model "$MODEL_PATH" \
  --model_type huginn_losatok_raven \
  --template huginn_losatok_text \
  --external_plugins "$PLUGIN_PATH" \
  --dataset "$TRAIN_MANIFEST" \
  --dataset_shuffle false \
  --train_dataloader_shuffle false \
  --sortish_sampler false \
  --group_by_length false \
  --max_length 192 \
  --output_dir "$RESUME_OUTPUT_DIR" \
  --tuner_type lora_llm \
  --freeze_aligner false \
  --modules_to_save "${MODULES_TO_SAVE[@]}" \
  --learning_rate 1e-4 \
  --aligner_lr 1e-4 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --fsdp "$RESUME_OUTPUT_DIR/fsdp2.json" \
  --resume_from_checkpoint "$SAVE_CHECKPOINT" \
  --max_steps "$RESUME_MAX_STEPS" \
  --per_device_train_batch_size "$MICRO_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --gradient_checkpointing false \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps "$RESUME_MAX_STEPS" \
  --save_total_limit 1 \
  --dataloader_num_workers 0 \
  --dataloader_pin_memory false \
  --dataset_num_proc 1 \
  --save_only_model false \
  --report_to none \
  --bf16 true

RESUME_CHECKPOINT="$(find_checkpoint "$RESUME_OUTPUT_DIR" "checkpoint-$RESUME_MAX_STEPS")"
echo "========== RESUMED CHECKPOINT DCP AUDIT =========="
echo "resume_checkpoint=$RESUME_CHECKPOINT"
audit_checkpoint "$RESUME_CHECKPOINT"

python - "$SAVE_CHECKPOINT/trainer_state.json" "$RESUME_CHECKPOINT/trainer_state.json" "$SAVE_STEPS" "$RESUME_MAX_STEPS" <<'PY'
import json
import sys
from pathlib import Path

save_state = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
resume_state = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
expected_save, expected_resume = map(int, sys.argv[3:])
if save_state.get('global_step') != expected_save:
    raise SystemExit(f"Save-phase global_step mismatch: {save_state.get('global_step')} != {expected_save}")
if resume_state.get('global_step') != expected_resume:
    raise SystemExit(f"Resume-phase global_step mismatch: {resume_state.get('global_step')} != {expected_resume}")
print(f"[resume] checkpoint_steps=({expected_save},{expected_resume})")
PY

echo "========== LOSATOK DYNAMIC FSDP2 MODULES-TO-SAVE SAVE/RESUME SMOKE PASSED =========="
echo "save_checkpoint=$SAVE_CHECKPOINT"
echo "resume_checkpoint=$RESUME_CHECKPOINT"
