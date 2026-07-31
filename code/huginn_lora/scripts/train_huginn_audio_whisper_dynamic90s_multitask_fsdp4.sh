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
FIXED_MAX_STEPS=20000
CHECKPOINT_INTERVAL=5000
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
FORMAL_CHECKPOINT_INSPECTOR="$REPO_ROOT/code/huginn_lora/scripts/inspect_huginn_whisper_dynamic30s_formal_checkpoints.py"
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
  "$PLANNER" "$FORMAL_CHECKPOINT_INSPECTOR"; do
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
if real_data.get("duration_policy") != "retain_all_then_cap_at30s":
    raise SystemExit(f"Real data-chain uses the obsolete duration policy: {real_data.get('duration_policy')!r}")
if (
    not sampler.get("validation_passed")
    or sampler.get("sampler_version") != "deterministic_hierarchical_no_replacement_v2"
    or sampler.get("gate") != "huginn_whisper_dynamic30s_indexed_mixture_no_replacement_v2"
    or sampler.get("contract_version") != "huginn_whisper_dynamic30s_data_v2"
):
    raise SystemExit(f"No-replacement sampler prerequisite has not passed: {sys.argv[2]}")
if sampler.get("duration_policy") != "retain_all_then_cap_at30s":
    raise SystemExit(f"Sampler report uses the obsolete duration policy: {sampler.get('duration_policy')!r}")
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
  --max-steps "$FIXED_MAX_STEPS" \
  --checkpoint-interval "$CHECKPOINT_INTERVAL" \
  --world-size "$WORLD_SIZE" \
  --per-device-batch "$PER_DEVICE_BATCH" \
  --gradient-accumulation "$GRADIENT_ACCUMULATION_STEPS"

read -r MAX_STEPS HALFWAY_STEP TOTAL_SAMPLES CHECKPOINT_INTERVAL_FROM_PLAN CHECKPOINT_STEPS_CSV < <(python - "$PLAN_PATH" <<'PY'
import json
import sys
from pathlib import Path

plan = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(
    plan["max_steps"],
    plan["halfway_step"],
    plan["total_scheduled_samples"],
    plan["checkpoint_interval"],
    ",".join(str(step) for step in plan["checkpoint_steps"]),
)
PY
)
if [ -z "${MAX_STEPS:-}" ] || [ -z "${HALFWAY_STEP:-}" ] || [ -z "${TOTAL_SAMPLES:-}" ] || [ -z "${CHECKPOINT_STEPS_CSV:-}" ]; then
  echo "Unable to read the frozen formal training plan: $PLAN_PATH" >&2
  exit 1
fi
if (( MAX_STEPS != 20000 || HALFWAY_STEP != 10000 || CHECKPOINT_INTERVAL_FROM_PLAN != 5000 || TOTAL_SAMPLES != MAX_STEPS * GLOBAL_BATCH )); then
  echo "Formal plan arithmetic is inconsistent: max=$MAX_STEPS half=$HALFWAY_STEP samples=$TOTAL_SAMPLES" >&2
  exit 1
fi
if [ "$CHECKPOINT_STEPS_CSV" != "5000,10000,15000,20000" ]; then
  echo "Formal checkpoint schedule is inconsistent: $CHECKPOINT_STEPS_CSV" >&2
  exit 1
fi

RESUME_ARGS=()
START_POSITION=0
RESUME_START_STEP=0
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
  read -r START_POSITION RESUME_START_STEP < <(python - "$RESUME_CHECKPOINT" "$PLAN_PATH" "$GLOBAL_BATCH" "$SEED" <<'PY'
import json
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1]).resolve()
current_plan = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
global_batch = int(sys.argv[3])
seed = int(sys.argv[4])
embedded_plan_path = checkpoint / "formal_training_plan.json"
runtime_path = checkpoint / "huginn_training_runtime_contract.json"
statistics_path = checkpoint / "audio_training_statistics.json"
trainer_state_path = checkpoint / "trainer_state.json"
scheduler_path = checkpoint / "scheduler.pt"
for path in (embedded_plan_path, runtime_path, statistics_path, trainer_state_path, scheduler_path):
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"Formal resume checkpoint is missing required state: {path}")
embedded_plan = json.loads(embedded_plan_path.read_text(encoding="utf-8"))
if embedded_plan != current_plan:
    raise SystemExit(
        f"Formal resume checkpoint plan differs from the current frozen plan: {embedded_plan_path}"
    )
runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
formal = runtime.get("formal_training", {})
resume_step = int(runtime.get("global_step", -1))
checkpoint_steps = [int(value) for value in current_plan["checkpoint_steps"]]
expected_formal = {
    "checkpoint_role": "scheduled",
    "checkpoint_step": resume_step,
    "checkpoint_index": checkpoint_steps.index(resume_step) + 1
    if resume_step in checkpoint_steps
    else -1,
    "plan_version": current_plan["plan_version"],
    "step_policy": current_plan["step_policy"],
    "sampler_version": current_plan["sampler_version"],
    "sampler_epoch_policy": current_plan["sampler_epoch_policy"],
    "sampler_seed": int(current_plan["sampler_seed"]),
    "duration_policy": current_plan["duration_policy"],
    "duration_estimate_used_for_max_steps": False,
    "target_realized_hours_minimum": float(current_plan["target_realized_hours_minimum"]),
    "max_steps": int(current_plan["max_steps"]),
    "halfway_step": int(current_plan["halfway_step"]),
    "checkpoint_interval": int(current_plan["checkpoint_interval"]),
    "checkpoint_steps": checkpoint_steps,
    "checkpoint_count": int(current_plan["checkpoint_count"]),
    "global_batch_size": int(current_plan["global_batch_size"]),
    "total_scheduled_samples": int(current_plan["total_scheduled_samples"]),
}
if (
    runtime.get("gate") != "huginn_whisper_dynamic30s_240ms_training_runtime_contract_v2"
    or runtime.get("phase") != "formal_checkpoint"
    or resume_step not in checkpoint_steps[:-1]
    or formal != expected_formal
):
    raise SystemExit(
        "Formal resume runtime contract mismatch: "
        f"actual={formal} expected={expected_formal} runtime={runtime}"
    )
trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
if int(trainer_state.get("global_step", -1)) != resume_step:
    raise SystemExit(f"Formal resume trainer state is not checkpoint-{resume_step}: {trainer_state}")
rng_files = sorted(checkpoint.glob("rng_state*.pth"))
if len(rng_files) != 4:
    raise SystemExit(f"Formal resume requires four per-rank RNG files, found: {rng_files}")
optimizer_dirs = [
    path
    for path in checkpoint.iterdir()
    if path.is_dir()
    and path.name != "pytorch_model_fsdp_0"
    and (path / ".metadata").is_file()
]
if not optimizer_dirs:
    raise SystemExit(f"Formal resume checkpoint has no optimizer DCP directory: {checkpoint}")
observed_sibling_steps = []
for sibling in checkpoint.parent.glob("checkpoint-*"):
    if not sibling.is_dir():
        continue
    try:
        observed_sibling_steps.append(int(sibling.name.split("-", 1)[1]))
    except (IndexError, ValueError) as exc:
        raise SystemExit(f"Unexpected checkpoint directory beside resume state: {sibling}") from exc
expected_sibling_steps = [step for step in checkpoint_steps if step <= resume_step]
if sorted(observed_sibling_steps) != expected_sibling_steps:
    raise SystemExit(
        "Formal resume requires one unbranched checkpoint chain through the selected step: "
        f"expected_siblings={expected_sibling_steps} actual={sorted(observed_sibling_steps)}"
    )
state = json.loads(statistics_path.read_text(encoding="utf-8"))
if state.get("statistics_version") != "huginn_dynamic30s_training_statistics_v2":
    raise SystemExit(f"Resume checkpoint uses an incompatible audio contract: {state.get('statistics_version')!r}")
if int(state.get("global_step", -1)) != resume_step:
    raise SystemExit(f"Resume statistics step mismatch: expected={resume_step} actual={state.get('global_step')}")
if int(state.get("sampler_seed", -1)) != seed:
    raise SystemExit(f"Resume sampler seed mismatch: state={state.get('sampler_seed')} current={seed}")
position = int(state.get("next_global_position", -1))
expected = resume_step * global_batch
if position != expected or int(state.get("total_samples", -1)) != expected:
    raise SystemExit(f"Resume sample position mismatch: expected={expected} actual={position}")
print(position, resume_step)
PY
)
  RESUME_ARGS+=(--resume_from_checkpoint "$RESUME_CHECKPOINT" --ignore_data_skip true)
fi

AVAILABLE_GB="$(df -BG "$REPO_ROOT" | awk 'NR==2 {gsub(/G/, "", $4); print $4}')"
if [ -z "$AVAILABLE_GB" ] || [ "$AVAILABLE_GB" -lt "$MIN_FREE_GB" ]; then
  echo "Insufficient storage for four planned full FSDP checkpoints: available=${AVAILABLE_GB:-unknown}G required=${MIN_FREE_GB}G" >&2
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
export HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_CHECKPOINT_STEPS="$CHECKPOINT_STEPS_CSV"
export HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_PHASE=formal
export HUGINN_AUDIO_DYNAMIC30S_FORMAL_PLAN_PATH="$PLAN_PATH"
export HUGINN_AUDIO_DYNAMIC30S_RECURRENT_CORE_RESHARD_AFTER_FORWARD_FALSE=1
unset HUGINN_AUDIO_DYNAMIC30S_ACCELERATION_STAGE0_AUDIT_DIR
unset HUGINN_AUDIO_DYNAMIC30S_ACCELERATION_STAGE1_AUDIT_DIR
unset HUGINN_AUDIO_DYNAMIC30S_ACCELERATION_STAGE2_AUDIT_DIR
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
echo "target_realized_hours_gt=$TARGET_HOURS max_steps_policy=user_fixed_no_duration_estimation"
echo "max_steps=$MAX_STEPS halfway_step=$HALFWAY_STEP total_scheduled_samples=$TOTAL_SAMPLES"
echo "checkpoints=$CHECKPOINT_STEPS_CSV checkpoint_interval=$CHECKPOINT_INTERVAL_FROM_PLAN save_total_limit=4 full_model_dcp=true"
echo "audio=single_dynamic_chunk retain_all_retain_first30s token_rate=240ms max_content_tokens=125 padding=local_batch_longest labels=-100"
echo "whisper=fully_trainable aligner=fully_trainable huginn_backbone=frozen huginn_lora_only=true"
echo "learning_rates=whisper:$WHISPER_LR,aligner:$ALIGNER_LR,lora:$LEARNING_RATE"
echo "lora=rank8,alpha16,dropout0.05 scheduler=cosine warmup_ratio=$WARMUP_RATIO weight_decay=$WEIGHT_DECAY max_grad_norm=$MAX_GRAD_NORM"
echo "fsdp_units=whisper,aligner,prelude2,recurrent_adapter_plus_core4,coda2 recurrent_core_reshard_after_forward=false all_other_units=true"
echo "activation_checkpointing=true vit_gradient_checkpointing=false whisper_outer_checkpoint=true use_reentrant=false"
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
CMD+=(--gradient_checkpointing false --vit_gradient_checkpointing false --gradient_checkpointing_kwargs '{"use_reentrant": false}')
CMD+=(--logging_steps "$LOGGING_STEPS" --save_strategy steps --save_steps "$CHECKPOINT_INTERVAL_FROM_PLAN" --save_total_limit 4)
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

IFS=',' read -r -a CHECKPOINT_STEPS <<< "$CHECKPOINT_STEPS_CSV"
CHECKPOINT_PATHS=()
OLD_CHECKPOINT_ROOT=""
if [ -n "$RESUME_CHECKPOINT" ]; then
  OLD_CHECKPOINT_ROOT="$(cd "$(dirname "$RESUME_CHECKPOINT")" && pwd)"
fi
for step in "${CHECKPOINT_STEPS[@]}"; do
  mapfile -t matches < <(
    {
      find "$OUTPUT_DIR" -type d -name "checkpoint-$step" -print
      if [ -n "$OLD_CHECKPOINT_ROOT" ]; then
        find "$OLD_CHECKPOINT_ROOT" -maxdepth 1 -type d -name "checkpoint-$step" -print
      fi
    } | sort -u
  )
  if [ "${#matches[@]}" -ne 1 ]; then
    echo "Expected exactly one retained checkpoint-$step across current/prior run roots; found ${#matches[@]}" >&2
    printf '  %s\n' "${matches[@]:-<none>}" >&2
    exit 1
  fi
  CHECKPOINT_PATHS+=("${matches[0]}")
done

SURVIVING_NEW_CHECKPOINTS="$(find "$OUTPUT_DIR" -type d -name 'checkpoint-*' -print | wc -l)"
EXPECTED_NEW_CHECKPOINTS=0
for step in "${CHECKPOINT_STEPS[@]}"; do
  if [ "$step" -gt "$RESUME_START_STEP" ]; then
    EXPECTED_NEW_CHECKPOINTS=$((EXPECTED_NEW_CHECKPOINTS + 1))
  fi
done
if [ "$SURVIVING_NEW_CHECKPOINTS" -ne "$EXPECTED_NEW_CHECKPOINTS" ]; then
  echo "Formal output checkpoint count mismatch: expected=$EXPECTED_NEW_CHECKPOINTS actual=$SURVIVING_NEW_CHECKPOINTS" >&2
  exit 1
fi

FORMAL_AUDIT_ARGS=(
  --plan "$PLAN_PATH"
  --world-size "$WORLD_SIZE"
  --output-report "$FINAL_AUDIT_REPORT"
)
for checkpoint_path in "${CHECKPOINT_PATHS[@]}"; do
  FORMAL_AUDIT_ARGS+=(--checkpoint "$checkpoint_path")
done
python -u "$FORMAL_CHECKPOINT_INSPECTOR" "${FORMAL_AUDIT_ARGS[@]}"

echo "========== HUGINN WHISPER DYNAMIC30S MULTITASK FORMAL FSDP4 PASSED =========="
echo "checkpoints=${CHECKPOINT_PATHS[*]}"
echo "formal_plan=$PLAN_PATH"
echo "checkpoint_audit=$FINAL_AUDIT_REPORT"
