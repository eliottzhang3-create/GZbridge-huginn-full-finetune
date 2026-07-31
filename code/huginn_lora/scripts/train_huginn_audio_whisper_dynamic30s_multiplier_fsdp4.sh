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
PER_DEVICE_BATCH=2
GRADIENT_ACCUMULATION_STEPS=4
GLOBAL_BATCH=$((WORLD_SIZE * PER_DEVICE_BATCH * GRADIENT_ACCUMULATION_STEPS))
CHECKPOINT_INTERVAL=5000
LEARNING_RATE=1e-4
ALIGNER_LR=1e-4
WHISPER_LR=1e-4
WARMUP_RATIO=0.05
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0
LOGGING_STEPS=10
STATISTICS_LOG_STEPS=100
MIN_FREE_GB="${HUGINN_MULTIPLIER_FORMAL_MIN_FREE_GB:-100}"
REPORT_TO="${HUGINN_MULTIPLIER_FORMAL_REPORT_TO:-tensorboard}"
RESUME_CHECKPOINT="${HUGINN_MULTIPLIER_FORMAL_RESUME_CHECKPOINT:-}"
REGISTRY="${HUGINN_MULTIPLIER_POOL_REGISTRY:-$REPO_ROOT/data/audio_swift/huginn_whisper_dynamic30s_multiplier/v1_gigaspeech_m/multiplier_pool_registry.json}"
POOL_AUDIT="${HUGINN_MULTIPLIER_POOL_AUDIT:-$(dirname "$REGISTRY")/multiplier_pool_audit.json}"
SMOKE_GATE="${HUGINN_MULTIPLIER_RESUME_SMOKE_GATE:-$(dirname "$REGISTRY")/checkpoint_resume_smoke_gate.json}"
RUN_ROOT="${HUGINN_MULTIPLIER_FORMAL_RUN_ROOT:-$REPO_ROOT/outputs/huginn_whisper_dynamic30s_multiplier_single_epoch_fsdp4/run-$(date +%Y%m%d_%H%M%S)}"
if [ -e "$RUN_ROOT" ]; then
  echo "Multiplier formal run root already exists: $RUN_ROOT" >&2
  exit 1
fi
mkdir -p "$RUN_ROOT"
RUN_ROOT="$(cd "$RUN_ROOT" && pwd)"

MODEL_PATH="$REPO_ROOT/models/huginn-audio-whisper-dynamic90s-v1"
PLUGIN_PATH="$REPO_ROOT/code/huginn_lora/plugins/huginn_audio_whisper_dynamic30s_multiplier_swift.py"
PLANNER="$SCRIPT_DIR/plan_huginn_whisper_dynamic30s_multiplier_training.py"
CHECKPOINT_INSPECTOR="$SCRIPT_DIR/inspect_huginn_whisper_dynamic30s_multiplier_formal_checkpoints.py"
OUTPUT_DIR="$RUN_ROOT/swift_output"
LOGGING_DIR="$RUN_ROOT/tensorboard"
TRAINING_STATS_DIR="$RUN_ROOT/training_statistics"
PLAN_PATH="$RUN_ROOT/multiplier_formal_training_plan.json"
FSDP_CONFIG_PATH="$RUN_ROOT/fsdp2_multiplier_formal.json"
FINAL_AUDIT_REPORT="$RUN_ROOT/multiplier_formal_checkpoint_report.json"
MODULES_TO_SAVE=(temporal_compressor audio_projector audio_boundary_embeddings)

for path in \
  "$MODEL_PATH" "$PLUGIN_PATH" "$REGISTRY" "$POOL_AUDIT" "$SMOKE_GATE" \
  "$PLANNER" "$CHECKPOINT_INSPECTOR"; do
  if [ ! -e "$path" ]; then
    echo "Required multiplier formal path is missing: $path" >&2
    exit 1
  fi
done

python - "$REGISTRY" "$POOL_AUDIT" "$SMOKE_GATE" <<'PY'
import json
import sys
from pathlib import Path

registry, audit, gate = map(lambda value: Path(value).resolve(), sys.argv[1:])
audit_payload = json.loads(audit.read_text(encoding='utf-8'))
gate_payload = json.loads(gate.read_text(encoding='utf-8'))
if (
    not audit_payload.get('validation_passed')
    or audit_payload.get('gate') != 'huginn_whisper_dynamic30s_multiplier_pool_audit_v1'
    or Path(audit_payload.get('registry', '')).resolve() != registry
):
    raise SystemExit(f'Multiplier pool audit is invalid: {audit}')
if (
    not gate_payload.get('validation_passed')
    or gate_payload.get('gate') != 'huginn_whisper_dynamic30s_multiplier_checkpoint_resume_gate_v1'
    or Path(gate_payload.get('registry', '')).resolve() != registry
):
    raise SystemExit(f'Multiplier checkpoint/resume gate is invalid: {gate}')
print('[multiplier-formal-preflight] pool_audit=passed checkpoint_resume=passed')
PY

python -u "$PLANNER" \
  --registry "$REGISTRY" \
  --pool-audit "$POOL_AUDIT" \
  --output "$PLAN_PATH" \
  --world-size "$WORLD_SIZE" \
  --per-device-batch "$PER_DEVICE_BATCH" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --checkpoint-interval "$CHECKPOINT_INTERVAL"

read -r SEED TOTAL_RECORDS MAX_STEPS CHECKPOINT_STEPS_CSV < <(python - "$PLAN_PATH" <<'PY'
import json
import sys
from pathlib import Path
plan = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
print(plan['seed'], plan['total_records'], plan['max_steps'], ','.join(map(str, plan['checkpoint_steps'])))
PY
)

START_POSITION=0
RESUME_START_STEP=0
RESUME_STATS_STATE=""
RESUME_ARGS=()
if [ -n "$RESUME_CHECKPOINT" ]; then
  RESUME_CHECKPOINT="$(cd "$RESUME_CHECKPOINT" && pwd)"
  RESUME_STATS_STATE="$RESUME_CHECKPOINT/audio_training_statistics.json"
  CHECKPOINT_PLAN="$RESUME_CHECKPOINT/multiplier_formal_training_plan.json"
  for path in "$RESUME_STATS_STATE" "$CHECKPOINT_PLAN" "$RESUME_CHECKPOINT/trainer_state.json"; do
    if [ ! -s "$path" ]; then
      echo "Multiplier resume checkpoint is incomplete: $path" >&2
      exit 1
    fi
  done
  read -r RESUME_START_STEP START_POSITION < <(python - "$RESUME_CHECKPOINT" "$PLAN_PATH" "$GLOBAL_BATCH" <<'PY'
import json
import sys
from pathlib import Path
checkpoint = Path(sys.argv[1])
current_plan = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
saved_plan = json.loads((checkpoint / 'multiplier_formal_training_plan.json').read_text(encoding='utf-8'))
trainer = json.loads((checkpoint / 'trainer_state.json').read_text(encoding='utf-8'))
stats = json.loads((checkpoint / 'audio_training_statistics.json').read_text(encoding='utf-8'))
if saved_plan != current_plan:
    raise SystemExit('Multiplier resume plan differs from the current frozen schedule')
step = int(trainer['global_step'])
position = step * int(sys.argv[3])
if int(stats.get('global_step', -1)) != step or int(stats.get('next_global_position', -1)) != position:
    raise SystemExit(f'Multiplier resume statistics mismatch: step={step} position={position} stats={stats}')
if step <= 0 or step >= int(current_plan['max_steps']) or step not in current_plan['checkpoint_steps']:
    raise SystemExit(f'Invalid multiplier resume step: {step}')
print(step, position)
PY
)
  RESUME_ARGS+=(--resume_from_checkpoint "$RESUME_CHECKPOINT" --ignore_data_skip true)
fi

AVAILABLE_GB="$(df -BG "$REPO_ROOT" | awk 'NR==2 {gsub(/G/, "", $4); print $4}')"
if [ -z "$AVAILABLE_GB" ] || [ "$AVAILABLE_GB" -lt "$MIN_FREE_GB" ]; then
  echo "Insufficient multiplier checkpoint storage: available=${AVAILABLE_GB:-unknown}G required=${MIN_FREE_GB}G" >&2
  exit 1
fi

FSDP_CONFIG='{"fsdp":"full_shard auto_wrap","fsdp_config":{"activation_checkpointing":true,"auto_wrap_policy":"TRANSFORMER_BASED_WRAP","cpu_ram_efficient_loading":true,"fsdp_version":2,"reshard_after_forward":true,"state_dict_type":"SHARDED_STATE_DICT"}}'
printf '%s\n' "$FSDP_CONFIG" > "$FSDP_CONFIG_PATH"
mkdir -p "$OUTPUT_DIR" "$LOGGING_DIR" "$TRAINING_STATS_DIR"

export HUGINN_MULTIPLIER_POOL_REGISTRY="$REGISTRY"
export HUGINN_MULTIPLIER_START_POSITION="$START_POSITION"
export HUGINN_MULTIPLIER_MAX_SAMPLES=$((TOTAL_RECORDS - START_POSITION))
export HUGINN_DYNAMIC90S_POOL_REGISTRY="$REGISTRY"
export HUGINN_DYNAMIC90S_MIXTURE_SEED="$SEED"
export HUGINN_AUDIO_DYNAMIC90S_FSDP2_NONPERSISTENT_ROPE=1
export HUGINN_AUDIO_DYNAMIC90S_TRAIN_CHAIN_AUDIT=1
export HUGINN_AUDIO_DYNAMIC90S_PEFT_ALIGNER_MODULES_TO_SAVE=1
export HUGINN_AUDIO_DYNAMIC90S_FULL_MODEL_DCP=1
export HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_DIR="$TRAINING_STATS_DIR"
export HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_LOG_STEPS="$STATISTICS_LOG_STEPS"
export HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_CHECKPOINT_STEPS="$CHECKPOINT_STEPS_CSV"
export HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_PHASE=multiplier_formal
export HUGINN_MULTIPLIER_FORMAL_PLAN_PATH="$PLAN_PATH"
export HUGINN_AUDIO_DYNAMIC30S_RECURRENT_CORE_RESHARD_AFTER_FORWARD_FALSE=1
unset HUGINN_AUDIO_DYNAMIC30S_FORMAL_PLAN_PATH
unset HUGINN_AUDIO_DYNAMIC90S_INIT_ALIGNER_CHECKPOINT
unset HUGINN_AUDIO_DYNAMIC90S_FSDP_SAVE_DEBUG
unset HUGINN_AUDIO_DYNAMIC90S_CHECKPOINT_AUDIT_DIR
unset HUGINN_AUDIO_DYNAMIC90S_DATA_POSITION_AUDIT_DIR
unset HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_FORWARD_AUDIT_DIR
if [ -n "$RESUME_STATS_STATE" ]; then
  export HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_RESUME_STATE="$RESUME_STATS_STATE"
else
  unset HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_RESUME_STATE
fi

echo "========== HUGINN WHISPER DYNAMIC30S MULTIPLIER FORMAL START =========="
echo "registry=$REGISTRY seed=$SEED schedule=single_frozen_global_epoch"
echo "records=$TOTAL_RECORDS max_steps=$MAX_STEPS global_batch=$GLOBAL_BATCH"
echo "checkpoints=$CHECKPOINT_STEPS_CSV save_total_limit=4"
echo "world_size=4 per_device_batch=2 gradient_accumulation=4"
echo "audio=dynamic_first30s token_rate=160ms prompts=AAC+ASR_specific"
echo "trainable=whisper+aligner+huginn_lora frozen=huginn_native_backbone"
echo "resume_checkpoint=${RESUME_CHECKPOINT:-<fresh>} start_position=$START_POSITION"

CMD=(swift sft)
CMD+=(--model "$MODEL_PATH" --model_type huginn_audio_whisper_dynamic90s --template huginn_audio_whisper_dynamic90s)
CMD+=(--external_plugins "$PLUGIN_PATH" --dataset "$REGISTRY" --streaming true)
CMD+=(--dataset_shuffle false --train_dataloader_shuffle false --sortish_sampler false --group_by_length false)
CMD+=(--max_length 192 --output_dir "$OUTPUT_DIR" --logging_dir "$LOGGING_DIR")
CMD+=(--tuner_type lora_llm --freeze_vit false --freeze_aligner false)
CMD+=(--modules_to_save "${MODULES_TO_SAVE[@]}")
CMD+=(--learning_rate "$LEARNING_RATE" --aligner_lr "$ALIGNER_LR" --vit_lr "$WHISPER_LR")
CMD+=(--lora_rank 8 --lora_alpha 16 --lora_dropout 0.05)
CMD+=(--lr_scheduler_type cosine --warmup_ratio "$WARMUP_RATIO" --weight_decay "$WEIGHT_DECAY" --max_grad_norm "$MAX_GRAD_NORM")
CMD+=(--fsdp "$FSDP_CONFIG_PATH" --max_steps "$MAX_STEPS")
CMD+=(--per_device_train_batch_size "$PER_DEVICE_BATCH" --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS")
CMD+=(--gradient_checkpointing false --vit_gradient_checkpointing false --gradient_checkpointing_kwargs '{"use_reentrant": false}')
CMD+=(--logging_steps "$LOGGING_STEPS" --save_strategy steps --save_steps "$CHECKPOINT_INTERVAL" --save_total_limit 4)
CMD+=(--dataloader_num_workers 0 --dataloader_pin_memory false --dataset_num_proc 1)
CMD+=(--save_only_model false --report_to "$REPORT_TO" --bf16 true --seed "$SEED" --data_seed "$SEED")
CMD+=("${RESUME_ARGS[@]}")

"${CMD[@]}"

mapfile -t CHECKPOINT_PATHS < <(find "$OUTPUT_DIR" -type d -name 'checkpoint-*' -print | sort -V)
if [ "${#CHECKPOINT_PATHS[@]}" -lt 1 ] || [ "${#CHECKPOINT_PATHS[@]}" -gt 4 ]; then
  echo "Expected one to four retained multiplier checkpoints, found ${#CHECKPOINT_PATHS[@]}" >&2
  exit 1
fi
FINAL_CHECKPOINT_MATCHES=()
for path in "${CHECKPOINT_PATHS[@]}"; do
  if [ "$(basename "$path")" = "checkpoint-$MAX_STEPS" ]; then
    FINAL_CHECKPOINT_MATCHES+=("$path")
  fi
done
if [ "${#FINAL_CHECKPOINT_MATCHES[@]}" -ne 1 ]; then
  echo "Final multiplier checkpoint-$MAX_STEPS is missing or duplicated" >&2
  exit 1
fi

AUDIT_ARGS=(--plan "$PLAN_PATH" --world-size "$WORLD_SIZE" --output-report "$FINAL_AUDIT_REPORT")
for path in "${CHECKPOINT_PATHS[@]}"; do
  AUDIT_ARGS+=(--checkpoint "$path")
done
python -u "$CHECKPOINT_INSPECTOR" "${AUDIT_ARGS[@]}"

echo "========== HUGINN WHISPER DYNAMIC30S MULTIPLIER FORMAL PASSED =========="
echo "checkpoints=${CHECKPOINT_PATHS[*]}"
echo "formal_plan=$PLAN_PATH"
echo "formal_audit=$FINAL_AUDIT_REPORT"

