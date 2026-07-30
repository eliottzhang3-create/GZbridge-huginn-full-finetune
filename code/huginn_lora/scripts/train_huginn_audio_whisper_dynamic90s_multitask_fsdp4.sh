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
SEED="${HUGINN_DYNAMIC90S_MIXTURE_SEED:-20260730}"
TARGET_HOURS=3000
PLANNING_RESERVE_RATIO=1.05
STEP_ROUNDING=100
LEARNING_RATE=1e-4
ALIGNER_LR=1e-4
WHISPER_LR=1e-4
WARMUP_RATIO=0.05
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0
LOGGING_STEPS=10
STATISTICS_LOG_STEPS=100
MIN_FREE_GB="${HUGINN_AUDIO_DYNAMIC90S_FORMAL_MIN_FREE_GB:-100}"
REPORT_TO="${HUGINN_AUDIO_DYNAMIC90S_FORMAL_REPORT_TO:-tensorboard}"
RESUME_CHECKPOINT="${HUGINN_AUDIO_DYNAMIC90S_FORMAL_RESUME_CHECKPOINT:-}"
RUN_ROOT="${HUGINN_AUDIO_DYNAMIC90S_FORMAL_RUN_ROOT:-outputs/huginn_audio_whisper_dynamic30s_multitask_3000h_fsdp4/run-$(date +%Y%m%d_%H%M%S)}"

MODEL_PATH="$REPO_ROOT/models/huginn-audio-whisper-dynamic90s-v1"
PLUGIN_PATH="$REPO_ROOT/code/huginn_lora/plugins/huginn_audio_whisper_dynamic90s_mixture_swift.py"
REGISTRY="${HUGINN_DYNAMIC90S_POOL_REGISTRY:-$REPO_ROOT/data/audio_swift/huginn_whisper_dynamic90s_multitask/v2_dynamic30s/pool_registry.json}"
REALDATA_REPORT="${HUGINN_DYNAMIC90S_REAL_DATA_CHAIN_REPORT:-$REPO_ROOT/data/audio_swift/huginn_whisper_dynamic90s_multitask/v2_dynamic30s/real_data_chain_report.json}"
SAMPLER_REPORT="${HUGINN_DYNAMIC90S_SAMPLER_REPORT:-$REPO_ROOT/data/audio_swift/huginn_whisper_dynamic90s_multitask/v2_dynamic30s/sampler/mixture_sampler_report.json}"
PLANNER="$REPO_ROOT/code/huginn_lora/scripts/plan_huginn_whisper_dynamic90s_formal_training.py"
CHECKPOINT_INSPECTOR="$REPO_ROOT/code/huginn_lora/scripts/inspect_huginn_whisper_dynamic90s_fsdp_checkpoints.py"
MODULES_TO_SAVE=(temporal_compressor audio_projector audio_boundary_embeddings)

if [ -e "$RUN_ROOT" ]; then
  echo "Formal run root already exists; choose a fresh HUGINN_AUDIO_DYNAMIC90S_FORMAL_RUN_ROOT: $RUN_ROOT" >&2
  exit 1
fi
mkdir -p "$RUN_ROOT"
RUN_ROOT="$(cd "$RUN_ROOT" && pwd)"
OUTPUT_DIR="$RUN_ROOT/swift_output"
LOGGING_DIR="$RUN_ROOT/tensorboard"
TRAINING_STATS_DIR="$RUN_ROOT/training_statistics"
PLAN_PATH="$RUN_ROOT/formal_training_plan.json"
FSDP_CONFIG_PATH="$RUN_ROOT/fsdp2_lora_activation_checkpointing.json"
FINAL_AUDIT_REPORT="$RUN_ROOT/formal_checkpoint_content_report.json"

for required_path in \
  "$MODEL_PATH" "$PLUGIN_PATH" "$REGISTRY" "$REALDATA_REPORT" "$SAMPLER_REPORT" \
  "$PLANNER" "$CHECKPOINT_INSPECTOR"; do
  if [ ! -e "$required_path" ]; then
    echo "Required formal-training path is missing: $required_path" >&2
    exit 1
  fi
done

python -u "$REPO_ROOT/code/huginn_lora/scripts/inspect_huginn_whisper_dynamic30s_contract.py"

python - "$REALDATA_REPORT" "$SAMPLER_REPORT" "$SEED" <<'PY'
import inspect
import json
import sys
from dataclasses import fields
from pathlib import Path

repo_root = Path.cwd()
sys.path.insert(0, str(repo_root / "code" / "huginn_lora"))

from accelerate.utils import fsdp_utils
from data_pipeline.dynamic90s_mixture_rows import EXPECTED_TASKS, TASK_PROMPTS
from swift.arguments.sft_args import SftArguments
from swift.tuner_plugin.lora_llm import LoRALLMTuner
from swift.utils import get_multimodal_target_regex

real_data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
sampler = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if (
    not real_data.get("validation_passed")
    or real_data.get("gate") != "huginn_whisper_dynamic30s_real_data_chain_v2"
    or real_data.get("contract_version") != "huginn_whisper_dynamic30s_data_v2"
):
    raise SystemExit(f"Real data-chain prerequisite has not passed: {sys.argv[1]}")
if (
    not sampler.get("validation_passed")
    or sampler.get("sampler_version") != "deterministic_hierarchical_no_replacement_v2"
    or sampler.get("gate") != "huginn_whisper_dynamic30s_indexed_mixture_no_replacement_v2"
    or sampler.get("contract_version") != "huginn_whisper_dynamic30s_data_v2"
):
    raise SystemExit(f"No-replacement sampler prerequisite has not passed: {sys.argv[2]}")
if int(real_data.get("seed", -1)) != int(sys.argv[3]) or int(sampler.get("seed", -1)) != int(sys.argv[3]):
    raise SystemExit(
        f"Prerequisite seed mismatch: real={real_data.get('seed')} sampler={sampler.get('seed')} "
        f"current={sys.argv[3]}"
    )
if set(TASK_PROMPTS) != {"AAC", "ASR"} or TASK_PROMPTS["AAC"] == TASK_PROMPTS["ASR"]:
    raise SystemExit(f"AAC and ASR require distinct task prompts: {TASK_PROMPTS}")
if EXPECTED_TASKS.get("gigaspeech_l_asr") != "ASR" or any(
    EXPECTED_TASKS.get(name) != "AAC"
    for name in ("wavcaps_no_bbc_aac", "audiocaps_v2_aac", "clotho_v2_aac")
):
    raise SystemExit(f"Dataset-to-task mapping is invalid: {EXPECTED_TASKS}")
available = {field.name for field in fields(SftArguments)}
required = {
    "fsdp", "modules_to_save", "resume_from_checkpoint", "ignore_data_skip", "vit_lr",
    "vit_gradient_checkpointing", "gradient_checkpointing_kwargs", "save_strategy", "save_steps",
    "save_only_model", "lr_scheduler_type", "warmup_ratio", "weight_decay", "max_grad_norm",
    "streaming", "max_steps",
}
missing = sorted(required - available)
if missing:
    raise SystemExit(f"Installed Swift lacks required formal-training arguments: {missing}")
prepare_source = inspect.getsource(LoRALLMTuner.prepare_model)
target_signature = inspect.signature(get_multimodal_target_regex)
if (
    "get_multimodal_target_regex(model)" not in prepare_source
    or "model_arch.vision_tower + model_arch.aligner" not in prepare_source
    or target_signature.parameters["freeze_vit"].default is not True
    or target_signature.parameters["freeze_aligner"].default is not True
):
    raise SystemExit(
        "Installed Swift lacks the required Huginn-only LoRA plus full Whisper/aligner tuning contract"
    )
for function_name in ("_get_model_state_dict", "_set_model_state_dict"):
    function = getattr(fsdp_utils, function_name, None)
    if function is None or "adapter_only" not in inspect.signature(function).parameters:
        raise SystemExit(f"Installed Accelerate lacks paired full-model FSDP API: {function_name}")
print(
    "[precheck] real_data=passed sampler_v2=passed task_prompts=distinct "
    "swift_mixed_tuning=present full_model_dcp=present"
)
PY

python -u "$PLANNER" \
  --registry "$REGISTRY" \
  --output "$PLAN_PATH" \
  --seed "$SEED" \
  --target-hours "$TARGET_HOURS" \
  --reserve-ratio "$PLANNING_RESERVE_RATIO" \
  --step-rounding "$STEP_ROUNDING" \
  --world-size "$WORLD_SIZE" \
  --per-device-batch "$PER_DEVICE_BATCH" \
  --gradient-accumulation "$GRADIENT_ACCUMULATION_STEPS"

read -r MAX_STEPS HALFWAY_STEP TOTAL_SAMPLES < <(python - "$PLAN_PATH" <<'PY'
import json
import sys
from pathlib import Path

plan = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(plan["max_steps"], plan["halfway_step"], plan["total_scheduled_samples"])
PY
)
if [ -z "${MAX_STEPS:-}" ] || [ -z "${HALFWAY_STEP:-}" ] || [ -z "${TOTAL_SAMPLES:-}" ]; then
  echo "Unable to read the frozen formal training plan: $PLAN_PATH" >&2
  exit 1
fi
if (( MAX_STEPS % 100 != 0 || HALFWAY_STEP * 2 != MAX_STEPS || TOTAL_SAMPLES != MAX_STEPS * GLOBAL_BATCH )); then
  echo "Formal plan arithmetic is inconsistent: max=$MAX_STEPS half=$HALFWAY_STEP samples=$TOTAL_SAMPLES" >&2
  exit 1
fi

RESUME_ARGS=()
START_POSITION=0
RESUME_STATS_STATE=""
if [ -n "$RESUME_CHECKPOINT" ]; then
  if [ ! -d "$RESUME_CHECKPOINT/pytorch_model_fsdp_0" ]; then
    echo "Formal resume checkpoint has no full-model FSDP state: $RESUME_CHECKPOINT" >&2
    exit 1
  fi
  RESUME_STATS_STATE="$RESUME_CHECKPOINT/audio_training_statistics.json"
  if [ ! -s "$RESUME_STATS_STATE" ]; then
    echo "Formal resume checkpoint lacks cumulative audio statistics: $RESUME_STATS_STATE" >&2
    exit 1
  fi
  START_POSITION="$(python - "$RESUME_STATS_STATE" "$HALFWAY_STEP" "$GLOBAL_BATCH" "$SEED" <<'PY'
import json
import sys
from pathlib import Path

state = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
halfway = int(sys.argv[2])
global_batch = int(sys.argv[3])
seed = int(sys.argv[4])
if state.get("statistics_version") != "huginn_dynamic30s_training_statistics_v2":
    raise SystemExit(f"Resume checkpoint uses an incompatible audio contract: {state.get('statistics_version')!r}")
if int(state.get("global_step", -1)) != halfway:
    raise SystemExit(f"Resume checkpoint must be the halfway checkpoint-{halfway}: {state.get('global_step')}")
if int(state.get("sampler_seed", -1)) != seed:
    raise SystemExit(f"Resume sampler seed mismatch: state={state.get('sampler_seed')} current={seed}")
position = int(state.get("next_global_position", -1))
expected = halfway * global_batch
if position != expected or int(state.get("total_samples", -1)) != expected:
    raise SystemExit(f"Resume sample position mismatch: expected={expected} actual={position}")
print(position)
PY
)"
  RESUME_ARGS+=(--resume_from_checkpoint "$RESUME_CHECKPOINT" --ignore_data_skip true)
fi

AVAILABLE_GB="$(df -BG "$REPO_ROOT" | awk 'NR==2 {gsub(/G/, "", $4); print $4}')"
if [ -z "$AVAILABLE_GB" ] || [ "$AVAILABLE_GB" -lt "$MIN_FREE_GB" ]; then
  echo "Insufficient storage for two full FSDP checkpoints: available=${AVAILABLE_GB:-unknown}G required=${MIN_FREE_GB}G" >&2
  exit 1
fi

FSDP_CONFIG='{"fsdp":"full_shard auto_wrap","fsdp_config":{"activation_checkpointing":true,"auto_wrap_policy":"TRANSFORMER_BASED_WRAP","cpu_ram_efficient_loading":true,"fsdp_version":2,"reshard_after_forward":true,"state_dict_type":"SHARDED_STATE_DICT"}}'
printf '%s\n' "$FSDP_CONFIG" > "$FSDP_CONFIG_PATH"
mkdir -p "$OUTPUT_DIR" "$LOGGING_DIR" "$TRAINING_STATS_DIR"

export HUGINN_DYNAMIC90S_POOL_REGISTRY="$REGISTRY"
export HUGINN_DYNAMIC90S_MIXTURE_SEED="$SEED"
export HUGINN_DYNAMIC90S_MIXTURE_START_POSITION="$START_POSITION"
unset HUGINN_DYNAMIC90S_MIXTURE_MAX_SAMPLES
export HUGINN_AUDIO_DYNAMIC90S_FSDP2_NONPERSISTENT_ROPE=1
export HUGINN_AUDIO_DYNAMIC90S_TRAIN_CHAIN_AUDIT=1
export HUGINN_AUDIO_DYNAMIC90S_PEFT_ALIGNER_MODULES_TO_SAVE=1
export HUGINN_AUDIO_DYNAMIC90S_FULL_MODEL_DCP=1
export HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_DIR="$TRAINING_STATS_DIR"
export HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_LOG_STEPS="$STATISTICS_LOG_STEPS"
export HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_CHECKPOINT_STEPS="$HALFWAY_STEP,$MAX_STEPS"
export HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_PHASE=formal
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

echo "========== HUGINN WHISPER DYNAMIC30S MULTITASK FORMAL FSDP4 START =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "run_root=$RUN_ROOT"
echo "registry=$REGISTRY sampler=deterministic_hierarchical_no_replacement_v2 seed=$SEED start_position=$START_POSITION"
echo "tasks=AAC_60_percent+ASR_40_percent prompts=task_specific"
echo "world_size=$WORLD_SIZE per_device_batch=$PER_DEVICE_BATCH accumulation=$GRADIENT_ACCUMULATION_STEPS global_batch=$GLOBAL_BATCH"
echo "target_realized_hours_gt=$TARGET_HOURS planning_reserve_ratio=$PLANNING_RESERVE_RATIO"
echo "max_steps=$MAX_STEPS halfway_step=$HALFWAY_STEP total_scheduled_samples=$TOTAL_SAMPLES"
echo "checkpoints=checkpoint-$HALFWAY_STEP,checkpoint-$MAX_STEPS save_total_limit=2 full_model_dcp=true"
echo "audio=single_dynamic_chunk discard_gt90s retain_first30s token_rate=160ms padding=local_batch_longest labels=-100"
echo "whisper=fully_trainable aligner=fully_trainable huginn_backbone=frozen huginn_lora_only=true"
echo "learning_rates=whisper:$WHISPER_LR,aligner:$ALIGNER_LR,lora:$LEARNING_RATE"
echo "lora=rank8,alpha16,dropout0.05 scheduler=cosine warmup_ratio=$WARMUP_RATIO weight_decay=$WEIGHT_DECAY max_grad_norm=$MAX_GRAD_NORM"
echo "fsdp_units=whisper,aligner,prelude2,recurrent_adapter_plus_core4,coda2 reshard_after_forward=true"
echo "activation_checkpointing=true vit_gradient_checkpointing=true use_reentrant=false"
echo "resume_checkpoint=${RESUME_CHECKPOINT:-<fresh>}"
echo "free_storage_gb=$AVAILABLE_GB required_free_gb=$MIN_FREE_GB report_to=$REPORT_TO"

TRAIN_PID=""
MONITOR_PID=""

print_resource_snapshot() {
  echo "========== DYNAMIC30S FORMAL RESOURCE SNAPSHOT =========="
  echo "snapshot_time=$(date '+%Y-%m-%d %H:%M:%S')"
  if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    ps -o pid,ppid,rss,vsz,%mem,etime,stat,cmd -p "$TRAIN_PID" || true
  fi
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
  for cgroup_file in /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.events; do
    if [ -r "$cgroup_file" ]; then
      echo "[cgroup] $(basename "$cgroup_file")=$(tr '\n' ' ' < "$cgroup_file")"
    fi
  done
}

resource_monitor() {
  while [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; do
    print_resource_snapshot
    sleep 60
  done
}

stop_resource_monitor() {
  if [ -n "$MONITOR_PID" ] && kill -0 "$MONITOR_PID" 2>/dev/null; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
  MONITOR_PID=""
}

on_exit() {
  status=$?
  trap - EXIT
  stop_resource_monitor
  echo "========== HUGINN WHISPER DYNAMIC30S MULTITASK FORMAL FSDP4 EXIT =========="
  echo "exit_status=$status"
  echo "exit_time=$(date '+%Y-%m-%d %H:%M:%S')"
  exit "$status"
}

on_signal() {
  echo "received_signal=$1"
  print_resource_snapshot
  if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    kill -TERM "$TRAIN_PID" 2>/dev/null || true
  fi
  exit 143
}

trap on_exit EXIT
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT

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
CMD+=(--gradient_checkpointing false --vit_gradient_checkpointing true --gradient_checkpointing_kwargs '{"use_reentrant": false}')
CMD+=(--logging_steps "$LOGGING_STEPS" --save_strategy steps --save_steps "$HALFWAY_STEP" --save_total_limit 2)
CMD+=(--dataloader_num_workers 0 --dataloader_pin_memory false --dataset_num_proc 1)
CMD+=(--save_only_model false --report_to "$REPORT_TO" --bf16 true --seed "$SEED" --data_seed "$SEED")
CMD+=("${RESUME_ARGS[@]}")

"${CMD[@]}" &
TRAIN_PID=$!
resource_monitor &
MONITOR_PID=$!
set +e
wait "$TRAIN_PID"
TRAIN_STATUS=$?
set -e
if [ "$TRAIN_STATUS" -ne 0 ]; then
  exit "$TRAIN_STATUS"
fi
stop_resource_monitor

find_one_checkpoint() {
  local root=$1
  local step=$2
  mapfile -t matches < <(find "$root" -type d -name "checkpoint-$step" -print | sort)
  if [ "${#matches[@]}" -ne 1 ]; then
    echo "Expected exactly one checkpoint-$step below $root; found ${#matches[@]}" >&2
    printf '  %s\n' "${matches[@]:-<none>}" >&2
    exit 1
  fi
  printf '%s\n' "${matches[0]}"
}

FINAL_CHECKPOINT="$(find_one_checkpoint "$OUTPUT_DIR" "$MAX_STEPS")"
if [ -n "$RESUME_CHECKPOINT" ]; then
  HALF_CHECKPOINT="$RESUME_CHECKPOINT"
  SURVIVING_NEW_CHECKPOINTS="$(find "$OUTPUT_DIR" -type d -name 'checkpoint-*' -print | wc -l)"
  if [ "$SURVIVING_NEW_CHECKPOINTS" -ne 1 ]; then
    echo "A resumed formal output must contain only the final checkpoint; found $SURVIVING_NEW_CHECKPOINTS" >&2
    exit 1
  fi
else
  HALF_CHECKPOINT="$(find_one_checkpoint "$OUTPUT_DIR" "$HALFWAY_STEP")"
  SURVIVING_CHECKPOINTS="$(find "$OUTPUT_DIR" -type d -name 'checkpoint-*' -print | wc -l)"
  if [ "$SURVIVING_CHECKPOINTS" -ne 2 ]; then
    echo "Fresh formal training must retain exactly two checkpoints; found $SURVIVING_CHECKPOINTS" >&2
    exit 1
  fi
fi

python - "$FINAL_CHECKPOINT/audio_training_statistics.json" "$TARGET_HOURS" "$MAX_STEPS" "$TOTAL_SAMPLES" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
target_hours = float(sys.argv[2])
max_steps = int(sys.argv[3])
total_samples = int(sys.argv[4])
state = json.loads(path.read_text(encoding="utf-8"))
if int(state.get("global_step", -1)) != max_steps:
    raise SystemExit(f"Final statistics step mismatch: expected={max_steps} actual={state.get('global_step')}")
if int(state.get("total_samples", -1)) != total_samples:
    raise SystemExit(f"Final sample count mismatch: expected={total_samples} actual={state.get('total_samples')}")
hours = float(state.get("total_effective_duration_hours", -1.0))
if hours <= target_hours:
    raise SystemExit(f"Formal run did not exceed {target_hours} realized hours: actual={hours}")
pool_counts = {name: entry["sample_count"] for name, entry in state["pools"].items()}
pool_hours = {name: entry["effective_duration_hours"] for name, entry in state["pools"].items()}
print(f"[formal-final] global_step={max_steps} samples={total_samples} realized_hours={hours:.6f}")
print(f"[formal-final] pool_sample_counts={pool_counts}")
print(f"[formal-final] pool_effective_hours={pool_hours}")
PY

python -u "$CHECKPOINT_INSPECTOR" \
  --save-checkpoint "$HALF_CHECKPOINT" \
  --resume-checkpoint "$FINAL_CHECKPOINT" \
  --save-step "$HALFWAY_STEP" \
  --resume-step "$MAX_STEPS" \
  --world-size "$WORLD_SIZE" \
  --output-report "$FINAL_AUDIT_REPORT"

echo "========== HUGINN WHISPER DYNAMIC30S MULTITASK FORMAL FSDP4 PASSED =========="
echo "half_checkpoint=$HALF_CHECKPOINT"
echo "final_checkpoint=$FINAL_CHECKPOINT"
echo "formal_plan=$PLAN_PATH"
echo "checkpoint_audit=$FINAL_AUDIT_REPORT"
