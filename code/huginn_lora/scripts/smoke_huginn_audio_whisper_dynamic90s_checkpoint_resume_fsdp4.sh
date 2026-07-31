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
SEED="${HUGINN_DYNAMIC90S_MIXTURE_SEED:-20260730}"
DATASET_MAX_SAMPLES="${HUGINN_DYNAMIC90S_CHECKPOINT_DATASET_MAX_SAMPLES:-64}"
RUN_ROOT="${HUGINN_AUDIO_DYNAMIC90S_CHECKPOINT_RUN_ROOT:-outputs/huginn_audio_whisper_dynamic30s_checkpoint_resume_fsdp4/run-$(date +%Y%m%d_%H%M%S)}"
if [ -e "$RUN_ROOT" ]; then
  echo "Checkpoint smoke root already exists; choose a fresh HUGINN_AUDIO_DYNAMIC90S_CHECKPOINT_RUN_ROOT: $RUN_ROOT" >&2
  exit 1
fi
mkdir -p "$RUN_ROOT"
RUN_ROOT="$(cd "$RUN_ROOT" && pwd)"
SAVE_OUTPUT_DIR="$RUN_ROOT/save_phase"
RESUME_OUTPUT_DIR="$RUN_ROOT/resume_phase"
SAVE_AUDIT_DIR="$RUN_ROOT/save_rank_audits"
RESUME_AUDIT_DIR="$RUN_ROOT/resume_rank_audits"
DATA_AUDIT_DIR="$RUN_ROOT/data_position_audits"
FORWARD_AUDIT_DIR="$RUN_ROOT/forward_consumption_audits"
TRAINING_STATS_DIR="$RUN_ROOT/training_statistics"
FSDP_CONFIG_PATH="$RUN_ROOT/fsdp2_checkpoint_accelerated.json"
CONTENT_REPORT="$RUN_ROOT/checkpoint_content_report.json"
ENHANCED_AUDIT_REPORT="$RUN_ROOT/enhanced_checkpoint_resume_audit.json"
DATA_ARTIFACT_FINGERPRINT="$RUN_ROOT/data_artifact_fingerprint.json"
MODEL_PATH="$REPO_ROOT/models/huginn-audio-whisper-dynamic90s-v1"
PLUGIN_PATH="$REPO_ROOT/code/huginn_lora/plugins/huginn_audio_whisper_dynamic90s_mixture_swift.py"
REGISTRY="${HUGINN_DYNAMIC90S_POOL_REGISTRY:-$REPO_ROOT/data/audio_swift/huginn_whisper_dynamic90s_multitask/v2_dynamic30s/pool_registry.json}"
REALDATA_REPORT="${HUGINN_DYNAMIC90S_REAL_DATA_CHAIN_REPORT:-$REPO_ROOT/data/audio_swift/huginn_whisper_dynamic90s_multitask/v2_dynamic30s/real_data_chain_report.json}"
SAMPLER_REPORT="${HUGINN_DYNAMIC90S_SAMPLER_REPORT:-$REPO_ROOT/data/audio_swift/huginn_whisper_dynamic90s_multitask/v2_dynamic30s/sampler/mixture_sampler_report.json}"
MARKER_INSPECTOR="$REPO_ROOT/code/huginn_lora/scripts/inspect_huginn_whisper_dynamic90s_checkpoint_resume_markers.py"
CHECKPOINT_INSPECTOR="$REPO_ROOT/code/huginn_lora/scripts/inspect_huginn_whisper_dynamic30s_smoke_fsdp_checkpoints.py"
DATA_INTEGRITY_INSPECTOR="$REPO_ROOT/code/huginn_lora/scripts/inspect_huginn_whisper_dynamic30s_smoke_data_integrity.py"
MODULES_TO_SAVE=(temporal_compressor audio_projector audio_boundary_embeddings)

for required_path in \
  "$MODEL_PATH" "$PLUGIN_PATH" "$REGISTRY" "$REALDATA_REPORT" "$SAMPLER_REPORT" \
  "$MARKER_INSPECTOR" "$CHECKPOINT_INSPECTOR" "$DATA_INTEGRITY_INSPECTOR"; do
  if [ ! -e "$required_path" ]; then
    echo "Required checkpoint smoke path is missing: $required_path" >&2
    exit 1
  fi
done

python -u "$REPO_ROOT/code/huginn_lora/scripts/inspect_huginn_whisper_dynamic30s_contract.py"

python - "$REALDATA_REPORT" "$SAMPLER_REPORT" "$DATASET_MAX_SAMPLES" "$SEED" <<'PY'
import json
import inspect
import sys
from dataclasses import fields
from pathlib import Path

repo_root = Path.cwd()
sys.path.insert(0, str(repo_root / 'code' / 'huginn_lora'))

from accelerate.utils import fsdp_utils
from data_pipeline.dynamic90s_mixture_rows import EXPECTED_TASKS, TASK_PROMPTS
from swift.arguments.sft_args import SftArguments
from swift.tuner_plugin.lora_llm import LoRALLMTuner
from swift.utils import get_multimodal_target_regex

report = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
if (
    not report.get('validation_passed')
    or report.get('gate') != 'huginn_whisper_dynamic30s_real_data_chain_v2'
    or report.get('contract_version') != 'huginn_whisper_dynamic30s_data_v2'
):
    raise SystemExit(f'Real data-chain prerequisite has not passed: {sys.argv[1]}')
if report.get('duration_policy') != 'retain_all_then_cap_at30s':
    raise SystemExit(f'Real data-chain uses the obsolete duration policy: {report.get("duration_policy")!r}')
sampler_report = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
if (
    not sampler_report.get('validation_passed')
    or sampler_report.get('sampler_version') != 'deterministic_hierarchical_no_replacement_v2'
    or sampler_report.get('gate') != 'huginn_whisper_dynamic30s_indexed_mixture_no_replacement_v2'
    or sampler_report.get('contract_version') != 'huginn_whisper_dynamic30s_data_v2'
):
    raise SystemExit(f'No-replacement sampler prerequisite has not passed: {sys.argv[2]}')
if sampler_report.get('duration_policy') != 'retain_all_then_cap_at30s':
    raise SystemExit(f'Sampler report uses the obsolete duration policy: {sampler_report.get("duration_policy")!r}')
if int(report.get('seed', -1)) != int(sys.argv[4]) or int(sampler_report.get('seed', -1)) != int(sys.argv[4]):
    raise SystemExit(
        f'Prerequisite seed mismatch: real={report.get("seed")} sampler={sampler_report.get("seed")} '
        f'current={sys.argv[4]}'
    )
if int(sys.argv[3]) < 24:
    raise SystemExit(f'Checkpoint dataset quota must be at least 24, got {sys.argv[3]}')
if set(TASK_PROMPTS) != {'AAC', 'ASR'} or TASK_PROMPTS['AAC'] == TASK_PROMPTS['ASR']:
    raise SystemExit(f'AAC and ASR require distinct task prompts: {TASK_PROMPTS}')
if EXPECTED_TASKS.get('gigaspeech_l_asr') != 'ASR' or any(
    EXPECTED_TASKS.get(name) != 'AAC'
    for name in ('wavcaps_no_bbc_aac', 'audiocaps_v2_aac', 'clotho_v2_aac')
):
    raise SystemExit(f'Dataset-to-task mapping is invalid: {EXPECTED_TASKS}')
available = {field.name for field in fields(SftArguments)}
required = {
    'fsdp', 'modules_to_save', 'resume_from_checkpoint', 'ignore_data_skip', 'vit_lr',
    'vit_gradient_checkpointing', 'gradient_checkpointing_kwargs',
    'save_strategy', 'save_steps', 'save_only_model', 'lr_scheduler_type',
    'streaming', 'max_steps',
}
missing = sorted(required - available)
if missing:
    raise SystemExit(f'Installed Swift lacks required checkpoint smoke arguments: {missing}')
prepare_source = inspect.getsource(LoRALLMTuner.prepare_model)
target_signature = inspect.signature(get_multimodal_target_regex)
if (
    'get_multimodal_target_regex(model)' not in prepare_source
    or 'model_arch.vision_tower + model_arch.aligner' not in prepare_source
    or target_signature.parameters['freeze_vit'].default is not True
    or target_signature.parameters['freeze_aligner'].default is not True
):
    raise SystemExit(
        'Installed Swift lora_llm does not provide the required contract: '
        'Huginn-only LoRA plus full-parameter vision_tower/aligner training'
    )
for function_name in ('_get_model_state_dict', '_set_model_state_dict'):
    function = getattr(fsdp_utils, function_name, None)
    if function is None or 'adapter_only' not in inspect.signature(function).parameters:
        raise SystemExit(
            f'Installed Accelerate lacks paired full-model FSDP API {function_name}(..., adapter_only=...): '
            f'{function!r}'
        )
print(
    '[precheck] real_data_chain=passed no_replacement_sampler=passed checkpoint_arguments=present '
    'dataset_quota=sufficient task_prompts=distinct lora_llm_mixed_tuning_contract=present'
)
PY

python -u "$DATA_INTEGRITY_INSPECTOR" self-test
python -u "$DATA_INTEGRITY_INSPECTOR" freeze \
  --registry "$REGISTRY" \
  --output "$DATA_ARTIFACT_FINGERPRINT"

FSDP_CONFIG='{"fsdp":"full_shard auto_wrap","fsdp_config":{"activation_checkpointing":true,"auto_wrap_policy":"TRANSFORMER_BASED_WRAP","cpu_ram_efficient_loading":true,"fsdp_version":2,"reshard_after_forward":true,"state_dict_type":"SHARDED_STATE_DICT"}}'
printf '%s\n' "$FSDP_CONFIG" > "$FSDP_CONFIG_PATH"
mkdir -p \
  "$SAVE_OUTPUT_DIR" "$RESUME_OUTPUT_DIR" "$SAVE_AUDIT_DIR" "$RESUME_AUDIT_DIR" \
  "$DATA_AUDIT_DIR" "$FORWARD_AUDIT_DIR" "$TRAINING_STATS_DIR"

export HUGINN_DYNAMIC90S_POOL_REGISTRY="$REGISTRY"
export HUGINN_DYNAMIC90S_MIXTURE_SEED="$SEED"
export HUGINN_DYNAMIC90S_MIXTURE_MAX_SAMPLES="$DATASET_MAX_SAMPLES"
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
unset HUGINN_AUDIO_DYNAMIC30S_ACCELERATION_STAGE0_AUDIT_DIR
unset HUGINN_AUDIO_DYNAMIC30S_ACCELERATION_STAGE1_AUDIT_DIR
unset HUGINN_AUDIO_DYNAMIC30S_ACCELERATION_STAGE2_AUDIT_DIR
unset HUGINN_AUDIO_DYNAMIC90S_INIT_ALIGNER_CHECKPOINT

echo "========== HUGINN WHISPER DYNAMIC30S CHECKPOINT RESUME FSDP4 START =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "run_root=$RUN_ROOT"
echo "world_size=$WORLD_SIZE per_device_batch=$PER_DEVICE_BATCH accumulation=$GRADIENT_ACCUMULATION_STEPS"
echo "phase1=fresh_process_positions_0_15_train_to_step_4_save_checkpoint_4"
echo "phase2=new_process_group_positions_16_23_resume_checkpoint_4_train_to_step_6"
echo "checkpoint_model_contract=full_model_dcp_including_whisper+lora_66+aligner fsdp_state_dict=SHARDED_STATE_DICT"
echo "checkpoint_state=model+optimizer+scheduler+rng+trainer_global_step+no_replacement_sampler_position+cumulative_data_statistics"
echo "lr_scheduler=constant learning_rate=1e-4"
echo "modules_to_save=${MODULES_TO_SAVE[*]}"
echo "whisper_encoder=fully_trainable learning_rate=1e-4 aligner_lr=1e-4"
echo "audio=single_dynamic_chunk retain_all_retain_first30s token_rate=240ms"
echo "activation_checkpointing=true gradient_checkpointing=false vit_gradient_checkpointing=false use_reentrant=false"
echo "fsdp_reshard=whisper_true aligner_true prelude_true recurrent_core_false coda_true"
echo "pytorch_cuda_alloc_conf=$PYTORCH_CUDA_ALLOC_CONF"
echo "huginn_backbone=frozen lora_rank=8 lora_alpha=16 lora_dropout=0.05 full_model_dcp=true"

ACTIVE_PID=""
ACTIVE_PHASE=""
MONITOR_PID=""

print_resource_snapshot() {
  echo "========== DYNAMIC30S CHECKPOINT RESOURCE SNAPSHOT =========="
  echo "snapshot_time=$(date '+%Y-%m-%d %H:%M:%S') phase=${ACTIVE_PHASE:-none}"
  if [ -n "$ACTIVE_PID" ] && kill -0 "$ACTIVE_PID" 2>/dev/null; then
    ps -o pid,ppid,rss,vsz,%mem,etime,stat,cmd -p "$ACTIVE_PID" || true
  fi
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
}

resource_monitor() {
  while [ -n "$ACTIVE_PID" ] && kill -0 "$ACTIVE_PID" 2>/dev/null; do
    print_resource_snapshot
    sleep 30
  done
}

stop_resource_monitor() {
  if [ -n "$MONITOR_PID" ] && kill -0 "$MONITOR_PID" 2>/dev/null; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
  MONITOR_PID=""
}

run_phase() {
  local phase=$1
  shift
  ACTIVE_PHASE=$phase
  echo "========== CHECKPOINT PHASE START: $phase =========="
  "$@" &
  ACTIVE_PID=$!
  resource_monitor &
  MONITOR_PID=$!
  set +e
  wait "$ACTIVE_PID"
  local status=$?
  set -e
  stop_resource_monitor
  ACTIVE_PID=""
  if [ "$status" -ne 0 ]; then
    echo "========== CHECKPOINT PHASE FAILED: $phase status=$status =========="
    return "$status"
  fi
  echo "========== CHECKPOINT PHASE PASSED: $phase =========="
}

on_exit() {
  status=$?
  trap - EXIT
  stop_resource_monitor
  echo "========== HUGINN WHISPER DYNAMIC30S CHECKPOINT RESUME FSDP4 EXIT =========="
  echo "exit_status=$status"
  echo "exit_time=$(date '+%Y-%m-%d %H:%M:%S')"
  exit "$status"
}
trap on_exit EXIT

on_signal() {
  echo "received_signal=$1 phase=${ACTIVE_PHASE:-none}"
  print_resource_snapshot
  if [ -n "$ACTIVE_PID" ] && kill -0 "$ACTIVE_PID" 2>/dev/null; then
    kill -TERM "$ACTIVE_PID" 2>/dev/null || true
  fi
  exit 143
}
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT

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

run_save_phase() {
  export HUGINN_DYNAMIC90S_MIXTURE_START_POSITION=0
  export HUGINN_AUDIO_DYNAMIC90S_DATA_POSITION_AUDIT_PHASE=save
  export HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_PHASE=save
  unset HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_RESUME_STATE || true
  export HUGINN_AUDIO_DYNAMIC90S_CHECKPOINT_AUDIT_DIR="$SAVE_AUDIT_DIR"
  export HUGINN_AUDIO_DYNAMIC90S_CHECKPOINT_PHASE=save
  export HUGINN_AUDIO_DYNAMIC90S_CHECKPOINT_START_STEP=0
  export HUGINN_AUDIO_DYNAMIC90S_CHECKPOINT_END_STEP="$SAVE_STEP"
  export HUGINN_AUDIO_DYNAMIC90S_CHECKPOINT_LAUNCH_ID="save-$(date +%s%N)-$$"
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
run_phase save run_save_phase

SAVE_CHECKPOINT="$(find_checkpoint "$SAVE_OUTPUT_DIR" "checkpoint-$SAVE_STEP")"
echo "[checkpoint] saved=$SAVE_CHECKPOINT"
SAVE_STATS_STATE="$SAVE_CHECKPOINT/audio_training_statistics.json"
if [ ! -s "$SAVE_STATS_STATE" ]; then
  echo "Saved checkpoint is missing cumulative training statistics: $SAVE_STATS_STATE" >&2
  exit 1
fi
SAVE_RUNTIME_CONTRACT="$SAVE_CHECKPOINT/huginn_training_runtime_contract.json"
if [ ! -s "$SAVE_RUNTIME_CONTRACT" ]; then
  echo "Saved checkpoint is missing the accelerated runtime contract: $SAVE_RUNTIME_CONTRACT" >&2
  exit 1
fi
SAVE_GLOBAL_POSITION=$((SAVE_STEP * WORLD_SIZE * PER_DEVICE_BATCH * GRADIENT_ACCUMULATION_STEPS))
python -u "$DATA_INTEGRITY_INSPECTOR" verify \
  --registry "$REGISTRY" \
  --fingerprint "$DATA_ARTIFACT_FINGERPRINT"
python -u "$DATA_INTEGRITY_INSPECTOR" verify-resume-state \
  --registry "$REGISTRY" \
  --state "$SAVE_STATS_STATE" \
  --seed "$SEED" \
  --start-position "$SAVE_GLOBAL_POSITION"
cp "$DATA_ARTIFACT_FINGERPRINT" "$SAVE_CHECKPOINT/smoke_data_artifact_fingerprint.json"

# The first torchrun has completely exited before this function starts.
run_resume_phase() {
  export HUGINN_DYNAMIC90S_MIXTURE_START_POSITION="$SAVE_GLOBAL_POSITION"
  export HUGINN_AUDIO_DYNAMIC90S_DATA_POSITION_AUDIT_PHASE=resume
  export HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_PHASE=resume
  export HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_RESUME_STATE="$SAVE_STATS_STATE"
  export HUGINN_AUDIO_DYNAMIC90S_CHECKPOINT_AUDIT_DIR="$RESUME_AUDIT_DIR"
  export HUGINN_AUDIO_DYNAMIC90S_CHECKPOINT_PHASE=resume
  export HUGINN_AUDIO_DYNAMIC90S_CHECKPOINT_START_STEP="$SAVE_STEP"
  export HUGINN_AUDIO_DYNAMIC90S_CHECKPOINT_END_STEP="$RESUME_STEP"
  export HUGINN_AUDIO_DYNAMIC90S_CHECKPOINT_LAUNCH_ID="resume-$(date +%s%N)-$$"
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
run_phase resume run_resume_phase

RESUME_CHECKPOINT="$(find_checkpoint "$RESUME_OUTPUT_DIR" "checkpoint-$RESUME_STEP")"
echo "[checkpoint] resumed=$RESUME_CHECKPOINT"
RESUME_STATS_STATE="$RESUME_CHECKPOINT/audio_training_statistics.json"
if [ ! -s "$RESUME_STATS_STATE" ]; then
  echo "Resumed checkpoint is missing cumulative training statistics: $RESUME_STATS_STATE" >&2
  exit 1
fi
RESUME_RUNTIME_CONTRACT="$RESUME_CHECKPOINT/huginn_training_runtime_contract.json"
if [ ! -s "$RESUME_RUNTIME_CONTRACT" ]; then
  echo "Resumed checkpoint is missing the accelerated runtime contract: $RESUME_RUNTIME_CONTRACT" >&2
  exit 1
fi
python -u "$DATA_INTEGRITY_INSPECTOR" verify \
  --registry "$REGISTRY" \
  --fingerprint "$DATA_ARTIFACT_FINGERPRINT"
cp "$DATA_ARTIFACT_FINGERPRINT" "$RESUME_CHECKPOINT/smoke_data_artifact_fingerprint.json"

python -u "$MARKER_INSPECTOR" \
  --save-audit-dir "$SAVE_AUDIT_DIR" \
  --resume-audit-dir "$RESUME_AUDIT_DIR" \
  --data-audit-dir "$DATA_AUDIT_DIR" \
  --forward-audit-dir "$FORWARD_AUDIT_DIR" \
  --save-stats-state "$SAVE_STATS_STATE" \
  --resume-stats-state "$RESUME_STATS_STATE" \
  --registry "$REGISTRY" \
  --seed "$SEED" \
  --save-step "$SAVE_STEP" \
  --resume-step "$RESUME_STEP" \
  --world-size "$WORLD_SIZE" \
  --save-checkpoint "$SAVE_CHECKPOINT" \
  --resume-checkpoint "$RESUME_CHECKPOINT" \
  --artifact-fingerprint "$DATA_ARTIFACT_FINGERPRINT" \
  --output-report "$ENHANCED_AUDIT_REPORT"

python -u "$CHECKPOINT_INSPECTOR" \
  --save-checkpoint "$SAVE_CHECKPOINT" \
  --resume-checkpoint "$RESUME_CHECKPOINT" \
  --save-step "$SAVE_STEP" \
  --resume-step "$RESUME_STEP" \
  --world-size "$WORLD_SIZE" \
  --output-report "$CONTENT_REPORT"

echo "========== HUGINN WHISPER DYNAMIC30S CHECKPOINT RESUME FSDP4 PASSED =========="
echo "save_checkpoint=$SAVE_CHECKPOINT"
echo "resume_checkpoint=$RESUME_CHECKPOINT"
echo "content_report=$CONTENT_REPORT"
echo "enhanced_audit_report=$ENHANCED_AUDIT_REPORT"
