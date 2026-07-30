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
MAX_ALLOCATED_GIB="${HUGINN_DYNAMIC30S_ACCEL_STAGE2_MAX_ALLOCATED_GIB:-29.0}"
MAX_RESERVED_GIB="${HUGINN_DYNAMIC30S_ACCEL_STAGE2_MAX_RESERVED_GIB:-30.0}"
RUN_ROOT="${HUGINN_DYNAMIC30S_ACCEL_STAGE2_RUN_ROOT:-outputs/huginn_whisper_dynamic30s_acceleration_stage2_fsdp4/run-$(date +%Y%m%d_%H%M%S)}"

if [ -e "$RUN_ROOT" ]; then
  echo "Acceleration Stage 2 run root already exists; choose a fresh HUGINN_DYNAMIC30S_ACCEL_STAGE2_RUN_ROOT: $RUN_ROOT" >&2
  exit 1
fi
mkdir -p "$RUN_ROOT"
RUN_ROOT="$(cd "$RUN_ROOT" && pwd)"

AUDIT_DIR="$RUN_ROOT/rank_audits"
OUTPUT_DIR="$RUN_ROOT/swift_output"
FSDP_CONFIG_PATH="$RUN_ROOT/fsdp2_acceleration_stage2.json"
REPORT_PATH="$RUN_ROOT/acceleration_stage2_report.json"
MODEL_PATH="$REPO_ROOT/models/huginn-audio-whisper-dynamic90s-v1"
PLUGIN_PATH="$REPO_ROOT/code/huginn_lora/plugins/huginn_audio_whisper_dynamic90s_swift.py"
PREPARE_SCRIPT="$REPO_ROOT/code/huginn_lora/scripts/prepare_huginn_whisper_dynamic30s_acceleration_stage2.py"
INSPECTOR="$REPO_ROOT/code/huginn_lora/scripts/inspect_huginn_whisper_dynamic30s_acceleration_stage2.py"
MODULES_TO_SAVE=(temporal_compressor audio_projector audio_boundary_embeddings)

for required_path in "$MODEL_PATH" "$PLUGIN_PATH" "$PREPARE_SCRIPT" "$INSPECTOR"; do
  if [ ! -e "$required_path" ]; then
    echo "Required acceleration Stage 2 path is missing: $required_path" >&2
    exit 1
  fi
done

python -u "$REPO_ROOT/code/huginn_lora/scripts/inspect_huginn_whisper_dynamic30s_contract.py"

python - <<'PY'
import inspect
from dataclasses import fields

from swift.arguments.sft_args import SftArguments
from swift.tuner_plugin.lora_llm import LoRALLMTuner
from swift.utils import get_multimodal_target_regex

available = {field.name for field in fields(SftArguments)}
required = {
    "fsdp", "tuner_type", "freeze_vit", "freeze_aligner", "vit_lr",
    "vit_gradient_checkpointing", "gradient_checkpointing_kwargs", "lora_rank",
    "lora_alpha", "lora_dropout", "lr_scheduler_type", "max_steps", "save_strategy",
    "seed", "data_seed",
}
missing = sorted(required - available)
if missing:
    raise SystemExit(f"Installed Swift lacks required acceleration Stage 2 arguments: {missing}")
prepare_source = inspect.getsource(LoRALLMTuner.prepare_model)
target_signature = inspect.signature(get_multimodal_target_regex)
if (
    "get_multimodal_target_regex(model)" not in prepare_source
    or "model_arch.vision_tower + model_arch.aligner" not in prepare_source
    or target_signature.parameters["freeze_vit"].default is not True
    or target_signature.parameters["freeze_aligner"].default is not True
):
    raise SystemExit("Installed Swift lacks Huginn-only LoRA plus full Whisper/aligner mixed tuning")
print("[precheck] acceleration_stage2_swift_contract=passed")
PY

python -u "$PREPARE_SCRIPT" --work-dir "$RUN_ROOT"
TRAIN_MANIFEST="$RUN_ROOT/fixture/dynamic30s_acceleration_stage2_fsdp4.jsonl"
FIXTURE_SUMMARY="$RUN_ROOT/fixture/dynamic30s_acceleration_stage2_fsdp4.summary.json"
if [ ! -s "$TRAIN_MANIFEST" ] || [ ! -s "$FIXTURE_SUMMARY" ]; then
  echo "Acceleration Stage 2 fixture is incomplete: manifest=$TRAIN_MANIFEST summary=$FIXTURE_SUMMARY" >&2
  exit 1
fi

python - "$FIXTURE_SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "gate": "huginn_whisper_dynamic30s_acceleration_stage2_fixture_v1",
    "record_count": 64,
    "duration_seconds": 30.0,
    "sample_rate": 16000,
    "sample_count": 480000,
    "segments_per_sample": 1,
    "audio_tokens_per_sample": 187,
    "prefix_tokens_per_sample": 189,
    "global_batch_size": 32,
}
observed = {key: summary.get(key) for key in expected}
if observed != expected:
    raise SystemExit(f"Acceleration Stage 2 fixture contract mismatch: observed={observed} expected={expected}")
print(f"[precheck] acceleration_stage2_fixture_contract=passed {observed}")
PY

FSDP_CONFIG='{"fsdp":"full_shard auto_wrap","fsdp_config":{"activation_checkpointing":true,"auto_wrap_policy":"TRANSFORMER_BASED_WRAP","cpu_ram_efficient_loading":true,"fsdp_version":2,"reshard_after_forward":true,"state_dict_type":"SHARDED_STATE_DICT"}}'
printf '%s\n' "$FSDP_CONFIG" > "$FSDP_CONFIG_PATH"
mkdir -p "$AUDIT_DIR" "$OUTPUT_DIR"

export HUGINN_AUDIO_DYNAMIC90S_FSDP2_NONPERSISTENT_ROPE=1
export HUGINN_AUDIO_DYNAMIC90S_TRAIN_CHAIN_AUDIT=1
export HUGINN_AUDIO_DYNAMIC90S_PEFT_ALIGNER_MODULES_TO_SAVE=1
export HUGINN_AUDIO_DYNAMIC30S_ACCELERATION_STAGE2_AUDIT_DIR="$AUDIT_DIR"
unset HUGINN_AUDIO_DYNAMIC30S_ACCELERATION_STAGE0_AUDIT_DIR
unset HUGINN_AUDIO_DYNAMIC30S_ACCELERATION_STAGE1_AUDIT_DIR
unset HUGINN_AUDIO_DYNAMIC90S_MEMORY90_AUDIT_DIR
unset HUGINN_AUDIO_DYNAMIC90S_STAGE34_AUDIT_DIR
unset HUGINN_AUDIO_DYNAMIC90S_STAGE5_AUDIT_DIR
unset HUGINN_AUDIO_DYNAMIC90S_REALDATA_AUDIT_DIR
unset HUGINN_AUDIO_DYNAMIC90S_CHECKPOINT_AUDIT_DIR
unset HUGINN_AUDIO_DYNAMIC90S_FULL_MODEL_DCP
unset HUGINN_AUDIO_DYNAMIC90S_FSDP_SAVE_DEBUG
unset HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_DIR
unset HUGINN_AUDIO_DYNAMIC90S_INIT_ALIGNER_CHECKPOINT

echo "========== HUGINN WHISPER DYNAMIC30S ACCELERATION STAGE 2 START =========="
echo "run_root=$RUN_ROOT"
echo "scope=recurrent_core_reshard_only no_checkpoint_save=true"
echo "data=synthetic_exact30s records=64 consumed_global_samples=32"
echo "audio=one_chunk content_tokens=187 prefix_tokens=189 no_audio_padding=true"
echo "world_size=$WORLD_SIZE per_device_batch=$PER_DEVICE_BATCH accumulation=$GRADIENT_ACCUMULATION_STEPS global_batch=$GLOBAL_BATCH max_steps=$MAX_STEPS"
echo "checkpointing=fsdp_activation_true whisper_internal_false whisper_outer_true"
echo "reshard_before=all_true reshard_after=core_false_all_other_true"
echo "memory_gate=allocated_lt_${MAX_ALLOCATED_GIB}GiB reserved_lt_${MAX_RESERVED_GIB}GiB"
echo "loss=response_only_shifted_next_token_prediction prefix_labels_minus100=true"

TRAIN_PID=""
MONITOR_PID=""

print_resource_snapshot() {
  echo "========== DYNAMIC30S ACCELERATION STAGE2 RESOURCE SNAPSHOT =========="
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
  echo "========== HUGINN WHISPER DYNAMIC30S ACCELERATION STAGE 2 EXIT =========="
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
  --dataset "$TRAIN_MANIFEST" \
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
  --vit_gradient_checkpointing false \
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
  --world-size "$WORLD_SIZE" \
  --max-allocated-gib "$MAX_ALLOCATED_GIB" \
  --max-reserved-gib "$MAX_RESERVED_GIB"

echo "========== HUGINN WHISPER DYNAMIC30S ACCELERATION STAGE 2 VALIDATION PASSED =========="
