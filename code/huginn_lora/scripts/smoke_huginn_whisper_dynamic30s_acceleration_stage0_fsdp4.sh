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
MAX_STEPS=1
SEED="${HUGINN_DYNAMIC90S_MIXTURE_SEED:-20260730}"
DATASET_MAX_SAMPLES="${HUGINN_DYNAMIC30S_ACCEL_STAGE0_DATASET_MAX_SAMPLES:-64}"
RUN_ROOT="${HUGINN_DYNAMIC30S_ACCEL_STAGE0_RUN_ROOT:-outputs/huginn_whisper_dynamic30s_acceleration_stage0_fsdp4/run-$(date +%Y%m%d_%H%M%S)}"

if [ -e "$RUN_ROOT" ]; then
  echo "Acceleration Stage 0 run root already exists; choose a fresh HUGINN_DYNAMIC30S_ACCEL_STAGE0_RUN_ROOT: $RUN_ROOT" >&2
  exit 1
fi
mkdir -p "$RUN_ROOT"
RUN_ROOT="$(cd "$RUN_ROOT" && pwd)"

AUDIT_DIR="$RUN_ROOT/rank_audits"
OUTPUT_DIR="$RUN_ROOT/swift_output"
FSDP_CONFIG_PATH="$RUN_ROOT/fsdp2_acceleration_stage0.json"
REPORT_PATH="$RUN_ROOT/acceleration_stage0_report.json"
MODEL_PATH="$REPO_ROOT/models/huginn-audio-whisper-dynamic90s-v1"
PLUGIN_PATH="$REPO_ROOT/code/huginn_lora/plugins/huginn_audio_whisper_dynamic90s_mixture_swift.py"
REGISTRY="${HUGINN_DYNAMIC90S_POOL_REGISTRY:-$REPO_ROOT/data/audio_swift/huginn_whisper_dynamic90s_multitask/v2_dynamic30s/pool_registry.json}"
REALDATA_REPORT="${HUGINN_DYNAMIC90S_REAL_DATA_CHAIN_REPORT:-$REPO_ROOT/data/audio_swift/huginn_whisper_dynamic90s_multitask/v2_dynamic30s/real_data_chain_report.json}"
SAMPLER_REPORT="${HUGINN_DYNAMIC90S_SAMPLER_REPORT:-$REPO_ROOT/data/audio_swift/huginn_whisper_dynamic90s_multitask/v2_dynamic30s/sampler/mixture_sampler_report.json}"
INSPECTOR="$REPO_ROOT/code/huginn_lora/scripts/inspect_huginn_whisper_dynamic30s_acceleration_stage0.py"
MODULES_TO_SAVE=(temporal_compressor audio_projector audio_boundary_embeddings)

for required_path in \
  "$MODEL_PATH" "$PLUGIN_PATH" "$REGISTRY" "$REALDATA_REPORT" "$SAMPLER_REPORT" "$INSPECTOR"; do
  if [ ! -e "$required_path" ]; then
    echo "Required acceleration Stage 0 path is missing: $required_path" >&2
    exit 1
  fi
done

python -u "$REPO_ROOT/code/huginn_lora/scripts/inspect_huginn_whisper_dynamic30s_contract.py"

python - "$REALDATA_REPORT" "$SAMPLER_REPORT" "$SEED" "$DATASET_MAX_SAMPLES" <<'PY'
import inspect
import json
import sys
from dataclasses import fields
from pathlib import Path

from swift.arguments.sft_args import SftArguments
from swift.tuner_plugin.lora_llm import LoRALLMTuner
from swift.utils import get_multimodal_target_regex

real_data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
sampler = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
seed = int(sys.argv[3])
dataset_max_samples = int(sys.argv[4])
if (
    not real_data.get("validation_passed")
    or real_data.get("gate") != "huginn_whisper_dynamic30s_real_data_chain_v2"
    or real_data.get("contract_version") != "huginn_whisper_dynamic30s_data_v2"
    or real_data.get("duration_policy") != "retain_all_then_cap_at30s"
):
    raise SystemExit(f"Dynamic30s real-data prerequisite has not passed: {sys.argv[1]}")
if (
    not sampler.get("validation_passed")
    or sampler.get("gate") != "huginn_whisper_dynamic30s_indexed_mixture_no_replacement_v2"
    or sampler.get("contract_version") != "huginn_whisper_dynamic30s_data_v2"
    or sampler.get("sampler_version") != "deterministic_hierarchical_no_replacement_v2"
    or sampler.get("duration_policy") != "retain_all_then_cap_at30s"
):
    raise SystemExit(f"Dynamic30s sampler prerequisite has not passed: {sys.argv[2]}")
if int(real_data.get("seed", -1)) != seed or int(sampler.get("seed", -1)) != seed:
    raise SystemExit(
        f"Prerequisite seed mismatch: real={real_data.get('seed')} "
        f"sampler={sampler.get('seed')} current={seed}"
    )
if dataset_max_samples < 32:
    raise SystemExit(f"Acceleration Stage 0 requires at least 32 dataset rows, got {dataset_max_samples}")

available = {field.name for field in fields(SftArguments)}
required = {
    "fsdp", "tuner_type", "freeze_vit", "freeze_aligner", "vit_lr",
    "vit_gradient_checkpointing", "gradient_checkpointing_kwargs", "lora_rank",
    "lora_alpha", "lora_dropout", "lr_scheduler_type", "max_steps", "save_strategy",
    "streaming", "seed", "data_seed",
}
missing = sorted(required - available)
if missing:
    raise SystemExit(f"Installed Swift lacks required acceleration Stage 0 arguments: {missing}")
prepare_source = inspect.getsource(LoRALLMTuner.prepare_model)
target_signature = inspect.signature(get_multimodal_target_regex)
if (
    "get_multimodal_target_regex(model)" not in prepare_source
    or "model_arch.vision_tower + model_arch.aligner" not in prepare_source
    or target_signature.parameters["freeze_vit"].default is not True
    or target_signature.parameters["freeze_aligner"].default is not True
):
    raise SystemExit(
        "Installed Swift lacks Huginn-only LoRA plus full Whisper/aligner mixed tuning"
    )
print(
    "[precheck] acceleration_stage0_prerequisites=passed "
    f"seed={seed} dataset_max_samples={dataset_max_samples}"
)
PY

FSDP_CONFIG='{"fsdp":"full_shard auto_wrap","fsdp_config":{"activation_checkpointing":true,"auto_wrap_policy":"TRANSFORMER_BASED_WRAP","cpu_ram_efficient_loading":true,"fsdp_version":2,"reshard_after_forward":true,"state_dict_type":"SHARDED_STATE_DICT"}}'
printf '%s\n' "$FSDP_CONFIG" > "$FSDP_CONFIG_PATH"
mkdir -p "$AUDIT_DIR" "$OUTPUT_DIR"

export HUGINN_DYNAMIC90S_POOL_REGISTRY="$REGISTRY"
export HUGINN_DYNAMIC90S_MIXTURE_SEED="$SEED"
export HUGINN_DYNAMIC90S_MIXTURE_START_POSITION=0
export HUGINN_DYNAMIC90S_MIXTURE_MAX_SAMPLES="$DATASET_MAX_SAMPLES"
export HUGINN_AUDIO_DYNAMIC90S_FSDP2_NONPERSISTENT_ROPE=1
export HUGINN_AUDIO_DYNAMIC90S_TRAIN_CHAIN_AUDIT=1
export HUGINN_AUDIO_DYNAMIC90S_PEFT_ALIGNER_MODULES_TO_SAVE=1
export HUGINN_AUDIO_DYNAMIC30S_ACCELERATION_STAGE0_AUDIT_DIR="$AUDIT_DIR"
unset HUGINN_AUDIO_DYNAMIC90S_FULL_MODEL_DCP
unset HUGINN_AUDIO_DYNAMIC90S_FSDP_SAVE_DEBUG
unset HUGINN_AUDIO_DYNAMIC90S_CHECKPOINT_AUDIT_DIR
unset HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_DIR
unset HUGINN_AUDIO_DYNAMIC90S_INIT_ALIGNER_CHECKPOINT

echo "========== HUGINN WHISPER DYNAMIC30S ACCELERATION STAGE 0 START =========="
echo "run_root=$RUN_ROOT"
echo "scope=diagnostic_only no_attention_change=true no_reshard_change=true no_checkpoint_save=true"
echo "data=real_no_replacement positions_start=0 dataset_max_samples=$DATASET_MAX_SAMPLES"
echo "world_size=$WORLD_SIZE per_device_batch=$PER_DEVICE_BATCH accumulation=$GRADIENT_ACCUMULATION_STEPS global_batch=$GLOBAL_BATCH max_steps=$MAX_STEPS"
echo "audio=retain_all_retain_first30s dynamic_240ms_tokens local_batch_longest_padding=true"
echo "fsdp_units=whisper,aligner,prelude2,recurrent_adapter_plus_core4,coda2 reshard_after_forward=true"
echo "activation_checkpointing=true vit_gradient_checkpointing=true use_reentrant=false"
echo "diagnostics=whisper_attention_class+sdpa_calls+checkpoint_wrapper_ownership+per_unit_reshard+peak_memory"

TRAIN_PID=""
MONITOR_PID=""

print_resource_snapshot() {
  echo "========== DYNAMIC30S ACCELERATION STAGE0 RESOURCE SNAPSHOT =========="
  echo "snapshot_time=$(date '+%Y-%m-%d %H:%M:%S')"
  if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    ps -o pid,ppid,rss,vsz,%mem,etime,stat,cmd -p "$TRAIN_PID" || true
  fi
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
}

resource_monitor() {
  while [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; do
    print_resource_snapshot
    sleep 10
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
  print_resource_snapshot
  echo "========== HUGINN WHISPER DYNAMIC30S ACCELERATION STAGE 0 EXIT =========="
  echo "exit_status=$status"
  echo "exit_time=$(date '+%Y-%m-%d %H:%M:%S')"
  exit "$status"
}
trap on_exit EXIT

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
  --output_dir "$OUTPUT_DIR" \
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
  --max_steps "$MAX_STEPS" \
  --per_device_train_batch_size "$PER_DEVICE_BATCH" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --gradient_checkpointing false \
  --vit_gradient_checkpointing true \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --logging_steps 1 \
  --save_strategy no \
  --dataloader_num_workers 0 \
  --dataloader_pin_memory false \
  --dataset_num_proc 1 \
  --report_to none \
  --bf16 true \
  --seed "$SEED" \
  --data_seed "$SEED" &
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

python -u "$INSPECTOR" \
  --audit-dir "$AUDIT_DIR" \
  --output-report "$REPORT_PATH" \
  --world-size "$WORLD_SIZE"

echo "========== HUGINN WHISPER DYNAMIC30S ACCELERATION STAGE 0 VALIDATION PASSED =========="
