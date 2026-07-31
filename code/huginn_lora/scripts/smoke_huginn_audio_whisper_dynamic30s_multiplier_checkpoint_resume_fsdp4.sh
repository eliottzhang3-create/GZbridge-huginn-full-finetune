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
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORLD_SIZE=4
PER_DEVICE_BATCH=1
GRADIENT_ACCUMULATION_STEPS=1
SAVE_STEP=4
RESUME_STEP=6
REGISTRY="${HUGINN_MULTIPLIER_POOL_REGISTRY:-$REPO_ROOT/data/audio_swift/huginn_whisper_dynamic30s_multiplier/v1_gigaspeech_m/multiplier_pool_registry.json}"
POOL_AUDIT="${HUGINN_MULTIPLIER_POOL_AUDIT:-$(dirname "$REGISTRY")/multiplier_pool_audit.json}"
RUN_ROOT="${HUGINN_MULTIPLIER_CHECKPOINT_RUN_ROOT:-$REPO_ROOT/outputs/huginn_whisper_dynamic30s_multiplier_checkpoint_resume_fsdp4/run-$(date +%Y%m%d_%H%M%S)}"
if [ -e "$RUN_ROOT" ]; then
  echo "Multiplier checkpoint smoke root already exists: $RUN_ROOT" >&2
  exit 1
fi
mkdir -p "$RUN_ROOT"
RUN_ROOT="$(cd "$RUN_ROOT" && pwd)"

MODEL_PATH="$REPO_ROOT/models/huginn-audio-whisper-dynamic90s-v1"
PLUGIN_PATH="$REPO_ROOT/code/huginn_lora/plugins/huginn_audio_whisper_dynamic30s_multiplier_swift.py"
MARKER_INSPECTOR="$SCRIPT_DIR/inspect_huginn_whisper_dynamic30s_multiplier_checkpoint_resume.py"
CHECKPOINT_INSPECTOR="$SCRIPT_DIR/inspect_huginn_whisper_dynamic30s_smoke_fsdp_checkpoints.py"
REAL_DATA_INSPECTOR="$SCRIPT_DIR/inspect_huginn_whisper_dynamic30s_multiplier_real_data.py"
SAVE_OUTPUT_DIR="$RUN_ROOT/save_phase"
RESUME_OUTPUT_DIR="$RUN_ROOT/resume_phase"
SAVE_AUDIT_DIR="$RUN_ROOT/save_rank_audits"
RESUME_AUDIT_DIR="$RUN_ROOT/resume_rank_audits"
DATA_AUDIT_DIR="$RUN_ROOT/data_position_audits"
FORWARD_AUDIT_DIR="$RUN_ROOT/forward_consumption_audits"
TRAINING_STATS_DIR="$RUN_ROOT/training_statistics"
FSDP_CONFIG_PATH="$RUN_ROOT/fsdp2_checkpoint_accelerated.json"
MARKER_REPORT="$RUN_ROOT/multiplier_checkpoint_resume_report.json"
CONTENT_REPORT="$RUN_ROOT/checkpoint_content_report.json"
REAL_DATA_REPORT="$RUN_ROOT/multiplier_real_data_report.json"
SMOKE_GATE="$(dirname "$REGISTRY")/checkpoint_resume_smoke_gate.json"
MODULES_TO_SAVE=(temporal_compressor audio_projector audio_boundary_embeddings)

for path in \
  "$MODEL_PATH" "$PLUGIN_PATH" "$REGISTRY" "$POOL_AUDIT" \
  "$MARKER_INSPECTOR" "$CHECKPOINT_INSPECTOR" "$REAL_DATA_INSPECTOR"; do
  if [ ! -e "$path" ]; then
    echo "Required multiplier checkpoint path is missing: $path" >&2
    exit 1
  fi
done

read -r SEED TOTAL_RECORDS MAX_STEPS < <(python - "$REGISTRY" "$POOL_AUDIT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

registry_path = Path(sys.argv[1]).resolve()
audit_path = Path(sys.argv[2]).resolve()
registry = json.loads(registry_path.read_text(encoding='utf-8'))
audit = json.loads(audit_path.read_text(encoding='utf-8'))
if (
    not audit.get('validation_passed')
    or audit.get('gate') != 'huginn_whisper_dynamic30s_multiplier_pool_audit_v1'
    or Path(audit.get('registry', '')).resolve() != registry_path
):
    raise SystemExit(f'Multiplier pool prerequisite has not passed: {audit_path}')
schedule = Path(registry['schedule_path'])
digest = hashlib.sha256()
with schedule.open('rb') as handle:
    while chunk := handle.read(8 * 1024 * 1024):
        digest.update(chunk)
if digest.hexdigest() != registry['schedule_sha256']:
    raise SystemExit('Multiplier schedule identity changed after prerequisite audit')
if int(registry['total_records']) < 64 or int(registry['total_records']) % 32:
    raise SystemExit(f"Invalid multiplier schedule length: {registry['total_records']}")
print(registry['seed'], registry['total_records'], registry['max_steps'])
PY
)

FSDP_CONFIG='{"fsdp":"full_shard auto_wrap","fsdp_config":{"activation_checkpointing":true,"auto_wrap_policy":"TRANSFORMER_BASED_WRAP","cpu_ram_efficient_loading":true,"fsdp_version":2,"reshard_after_forward":true,"state_dict_type":"SHARDED_STATE_DICT"}}'
printf '%s\n' "$FSDP_CONFIG" > "$FSDP_CONFIG_PATH"
mkdir -p \
  "$SAVE_OUTPUT_DIR" "$RESUME_OUTPUT_DIR" "$SAVE_AUDIT_DIR" "$RESUME_AUDIT_DIR" \
  "$DATA_AUDIT_DIR" "$FORWARD_AUDIT_DIR" "$TRAINING_STATS_DIR"

export HUGINN_MULTIPLIER_POOL_REGISTRY="$REGISTRY"
export HUGINN_DYNAMIC90S_POOL_REGISTRY="$REGISTRY"
export HUGINN_DYNAMIC90S_MIXTURE_SEED="$SEED"
export HUGINN_MULTIPLIER_MAX_SAMPLES=64
export HUGINN_AUDIO_DYNAMIC90S_FSDP2_NONPERSISTENT_ROPE=1
export HUGINN_AUDIO_DYNAMIC90S_TRAIN_CHAIN_AUDIT=1
export HUGINN_AUDIO_DYNAMIC90S_PEFT_ALIGNER_MODULES_TO_SAVE=1
export HUGINN_AUDIO_DYNAMIC90S_FSDP_SAVE_DEBUG=1
export HUGINN_AUDIO_DYNAMIC90S_FULL_MODEL_DCP=1
export HUGINN_AUDIO_DYNAMIC90S_DATA_POSITION_AUDIT_DIR="$DATA_AUDIT_DIR"
export HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_DIR="$TRAINING_STATS_DIR"
export HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_LOG_STEPS=1
export HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_CHECKPOINT_STEPS="$SAVE_STEP,$RESUME_STEP"
export HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_FORWARD_AUDIT_DIR="$FORWARD_AUDIT_DIR"
export HUGINN_AUDIO_DYNAMIC30S_RECURRENT_CORE_RESHARD_AFTER_FORWARD_FALSE=1
unset HUGINN_AUDIO_DYNAMIC30S_FORMAL_PLAN_PATH
unset HUGINN_AUDIO_DYNAMIC90S_INIT_ALIGNER_CHECKPOINT

python -u "$REAL_DATA_INSPECTOR" \
  --registry "$REGISTRY" \
  --output-report "$REAL_DATA_REPORT"

echo "========== HUGINN WHISPER DYNAMIC30S MULTIPLIER CHECKPOINT RESUME START =========="
echo "registry=$REGISTRY seed=$SEED total_records=$TOTAL_RECORDS formal_max_steps=$MAX_STEPS"
echo "smoke=real_audio fsdp4 save_step=$SAVE_STEP resume_step=$RESUME_STEP"
echo "model_contract=whisper+aligner+lora_trainable huginn_base_frozen"

find_checkpoint() {
  local output_dir=$1
  local name=$2
  mapfile -t matches < <(find "$output_dir" -type d -name "$name" -print | sort)
  if [ "${#matches[@]}" -ne 1 ]; then
    echo "Expected exactly one $name below $output_dir; found ${#matches[@]}" >&2
    exit 1
  fi
  printf '%s\n' "${matches[0]}"
}

run_save_phase() {
  export HUGINN_MULTIPLIER_START_POSITION=0
  export HUGINN_AUDIO_DYNAMIC90S_DATA_POSITION_AUDIT_PHASE=save
  export HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_PHASE=save
  unset HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_RESUME_STATE || true
  export HUGINN_AUDIO_DYNAMIC90S_CHECKPOINT_AUDIT_DIR="$SAVE_AUDIT_DIR"
  export HUGINN_AUDIO_DYNAMIC90S_CHECKPOINT_PHASE=save
  export HUGINN_AUDIO_DYNAMIC90S_CHECKPOINT_START_STEP=0
  export HUGINN_AUDIO_DYNAMIC90S_CHECKPOINT_END_STEP="$SAVE_STEP"
  export HUGINN_AUDIO_DYNAMIC90S_CHECKPOINT_LAUNCH_ID="multiplier-save-$(date +%s%N)-$$"
  swift sft \
    --model "$MODEL_PATH" \
    --model_type huginn_audio_whisper_dynamic90s \
    --template huginn_audio_whisper_dynamic90s \
    --external_plugins "$PLUGIN_PATH" \
    --dataset "$REGISTRY" \
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
    --bf16 true
}

run_save_phase
SAVE_CHECKPOINT="$(find_checkpoint "$SAVE_OUTPUT_DIR" "checkpoint-$SAVE_STEP")"
SAVE_STATS_STATE="$SAVE_CHECKPOINT/audio_training_statistics.json"
if [ ! -s "$SAVE_STATS_STATE" ]; then
  echo "Multiplier save checkpoint lacks statistics state: $SAVE_STATS_STATE" >&2
  exit 1
fi
SAVE_GLOBAL_POSITION=$((SAVE_STEP * WORLD_SIZE * PER_DEVICE_BATCH * GRADIENT_ACCUMULATION_STEPS))

run_resume_phase() {
  export HUGINN_MULTIPLIER_START_POSITION="$SAVE_GLOBAL_POSITION"
  export HUGINN_AUDIO_DYNAMIC90S_DATA_POSITION_AUDIT_PHASE=resume
  export HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_PHASE=resume
  export HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_RESUME_STATE="$SAVE_STATS_STATE"
  export HUGINN_AUDIO_DYNAMIC90S_CHECKPOINT_AUDIT_DIR="$RESUME_AUDIT_DIR"
  export HUGINN_AUDIO_DYNAMIC90S_CHECKPOINT_PHASE=resume
  export HUGINN_AUDIO_DYNAMIC90S_CHECKPOINT_START_STEP="$SAVE_STEP"
  export HUGINN_AUDIO_DYNAMIC90S_CHECKPOINT_END_STEP="$RESUME_STEP"
  export HUGINN_AUDIO_DYNAMIC90S_CHECKPOINT_LAUNCH_ID="multiplier-resume-$(date +%s%N)-$$"
  swift sft \
    --model "$MODEL_PATH" \
    --model_type huginn_audio_whisper_dynamic90s \
    --template huginn_audio_whisper_dynamic90s \
    --external_plugins "$PLUGIN_PATH" \
    --dataset "$REGISTRY" \
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
    --bf16 true
}

run_resume_phase
RESUME_CHECKPOINT="$(find_checkpoint "$RESUME_OUTPUT_DIR" "checkpoint-$RESUME_STEP")"
RESUME_STATS_STATE="$RESUME_CHECKPOINT/audio_training_statistics.json"
if [ ! -s "$RESUME_STATS_STATE" ]; then
  echo "Multiplier resume checkpoint lacks statistics state: $RESUME_STATS_STATE" >&2
  exit 1
fi

python -u "$MARKER_INSPECTOR" \
  --save-audit-dir "$SAVE_AUDIT_DIR" \
  --resume-audit-dir "$RESUME_AUDIT_DIR" \
  --data-audit-dir "$DATA_AUDIT_DIR" \
  --forward-audit-dir "$FORWARD_AUDIT_DIR" \
  --save-stats-state "$SAVE_STATS_STATE" \
  --resume-stats-state "$RESUME_STATS_STATE" \
  --registry "$REGISTRY" \
  --save-step "$SAVE_STEP" \
  --resume-step "$RESUME_STEP" \
  --world-size "$WORLD_SIZE" \
  --output-report "$MARKER_REPORT"

python -u "$CHECKPOINT_INSPECTOR" \
  --save-checkpoint "$SAVE_CHECKPOINT" \
  --resume-checkpoint "$RESUME_CHECKPOINT" \
  --save-step "$SAVE_STEP" \
  --resume-step "$RESUME_STEP" \
  --world-size "$WORLD_SIZE" \
  --output-report "$CONTENT_REPORT"

python - "$REGISTRY" "$MARKER_REPORT" "$CONTENT_REPORT" "$REAL_DATA_REPORT" "$SMOKE_GATE" <<'PY'
import json
import os
import sys
from pathlib import Path

registry, marker_path, content_path, real_data_path, output = map(
    lambda value: Path(value).resolve(), sys.argv[1:]
)
marker = json.loads(marker_path.read_text(encoding='utf-8'))
content = json.loads(content_path.read_text(encoding='utf-8'))
real_data = json.loads(real_data_path.read_text(encoding='utf-8'))
if (
    not marker.get('validation_passed')
    or not content.get('validation_passed')
    or not real_data.get('validation_passed')
):
    raise SystemExit('Multiplier checkpoint gate cannot be written from failed reports')
payload = {
    'gate': 'huginn_whisper_dynamic30s_multiplier_checkpoint_resume_gate_v1',
    'validation_passed': True,
    'registry': str(registry),
    'marker_report': str(marker_path),
    'content_report': str(content_path),
    'real_data_report': str(real_data_path),
    'save_step': marker['save_step'],
    'resume_step': marker['resume_step'],
    'cumulative_provenance_sha256': marker['cumulative']['provenance_sha256'],
}
temporary = output.with_name(output.name + '.tmp')
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
os.replace(temporary, output)
print(f'[multiplier-smoke-gate] output={output}')
PY

echo "========== HUGINN WHISPER DYNAMIC30S MULTIPLIER CHECKPOINT RESUME PASSED =========="
echo "save_checkpoint=$SAVE_CHECKPOINT"
echo "resume_checkpoint=$RESUME_CHECKPOINT"
echo "marker_report=$MARKER_REPORT"
echo "content_report=$CONTENT_REPORT"
echo "real_data_report=$REAL_DATA_REPORT"
echo "smoke_gate=$SMOKE_GATE"
