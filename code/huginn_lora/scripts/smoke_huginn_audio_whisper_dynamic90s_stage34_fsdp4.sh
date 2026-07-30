#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
export OMP_NUM_THREADS=4

RUN_ROOT="${HUGINN_AUDIO_DYNAMIC90S_STAGE34_RUN_ROOT:-outputs/huginn_audio_whisper_dynamic90s_stage34_fsdp4/run-$(date +%Y%m%d_%H%M%S)}"
if [ -e "$RUN_ROOT" ]; then
  echo "Stage 3-4 run root already exists; choose a fresh HUGINN_AUDIO_DYNAMIC90S_STAGE34_RUN_ROOT: $RUN_ROOT" >&2
  exit 1
fi
mkdir -p "$RUN_ROOT"
RUN_ROOT="$(cd "$RUN_ROOT" && pwd)"
FIXTURE_DIR="$RUN_ROOT/fixture"
AUDIT_DIR="$RUN_ROOT/rank_audits"
OUTPUT_DIR="$RUN_ROOT/swift_output"
FSDP_CONFIG_PATH="$RUN_ROOT/fsdp2_lora_no_activation.json"
MODEL_PATH="$REPO_ROOT/models/huginn-audio-whisper-dynamic90s-v1"
PLUGIN_PATH="$REPO_ROOT/code/huginn_lora/plugins/huginn_audio_whisper_dynamic90s_swift.py"
PREPARE_SCRIPT="$REPO_ROOT/code/huginn_lora/scripts/prepare_huginn_audio_whisper_dynamic90s_stage34.py"
MARKER_INSPECTOR="$REPO_ROOT/code/huginn_lora/scripts/inspect_huginn_audio_whisper_dynamic90s_stage34_markers.py"

for required_path in "$MODEL_PATH" "$PLUGIN_PATH" "$PREPARE_SCRIPT" "$MARKER_INSPECTOR"; do
  if [ ! -e "$required_path" ]; then
    echo "Required Stage 3-4 path is missing: $required_path" >&2
    exit 1
  fi
done

python - <<'PY'
from dataclasses import fields
from swift.arguments.sft_args import SftArguments

available = {field.name for field in fields(SftArguments)}
required = {
    'fsdp', 'tuner_type', 'freeze_vit', 'freeze_aligner', 'lora_rank',
    'lora_alpha', 'lora_dropout', 'max_steps', 'save_strategy',
}
missing = sorted(required - available)
if missing:
    raise SystemExit(f'Installed Swift lacks required Stage 3-4 arguments: {missing}')
print('[precheck] swift_stage34_arguments=present')
PY

python -u "$PREPARE_SCRIPT" --work-dir "$RUN_ROOT"
TRAIN_MANIFEST="$FIXTURE_DIR/dynamic90s_stage34_fsdp4.jsonl"
if [ ! -s "$TRAIN_MANIFEST" ]; then
  echo "Stage 3-4 synthetic manifest is missing or empty: $TRAIN_MANIFEST" >&2
  exit 1
fi

FSDP_CONFIG='{"fsdp":"full_shard auto_wrap","fsdp_config":{"activation_checkpointing":false,"auto_wrap_policy":"TRANSFORMER_BASED_WRAP","cpu_ram_efficient_loading":true,"fsdp_version":2,"reshard_after_forward":true,"state_dict_type":"SHARDED_STATE_DICT"}}'
printf '%s\n' "$FSDP_CONFIG" > "$FSDP_CONFIG_PATH"
mkdir -p "$AUDIT_DIR" "$OUTPUT_DIR"

export HUGINN_AUDIO_DYNAMIC90S_FSDP2_NONPERSISTENT_ROPE=1
export HUGINN_AUDIO_DYNAMIC90S_TRAIN_CHAIN_AUDIT=1
export HUGINN_AUDIO_DYNAMIC90S_STAGE34_AUDIT_DIR="$AUDIT_DIR"

echo "========== HUGINN WHISPER DYNAMIC90S MERGED STAGE 3-4 FSDP4 =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "launch_mode=swift_cli_internal_torchrun"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "dataset=job_generated_synthetic_wav"
echo "formal_dataset_dependency=none"
echo "run_root=$RUN_ROOT"
echo "model_type=huginn_audio_whisper_dynamic90s"
echo "audio_tokens=dynamic complete_120ms_per_token"
echo "audio_over_90s=retain_first_90s no_duration_discard=true"
echo "first_step_prefix_tokens_across_ranks=10,252,502,752"
echo "whisper_encoder=fully_trainable learning_rate=1e-4"
echo "aligner=trainable tensors=14 learning_rate=1e-4"
echo "huginn=lora_llm tensors=66 rank=8 alpha=16 dropout=0.05 learning_rate=1e-4"
echo "fsdp=custom_fsdp2_json world_size=4 sharded_dtensor_required=true"
echo "fsdp_units=whisper_whole,aligner_whole,prelude_2blocks,core_adapter_plus_4blocks,coda_2blocks"
echo "fsdp_reshard_after_forward=true for_all_units=true"
echo "fsdp_activation_checkpointing=false gradient_checkpointing=false"
echo "per_device_train_batch_size=1 gradient_accumulation_steps=1 global_batch_size=4"
echo "max_steps=1 expected_optimizer_updates=1 save_strategy=no"

TRAIN_PID=""
MONITOR_PID=""

print_resource_snapshot() {
  echo "========== DYNAMIC90S STAGE 3-4 RESOURCE SNAPSHOT =========="
  echo "snapshot_time=$(date '+%Y-%m-%d %H:%M:%S')"
  if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    ps -o pid,ppid,rss,vsz,%mem,etime,stat,cmd -p "$TRAIN_PID" || true
  fi
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
}

resource_monitor() {
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    print_resource_snapshot
    sleep 30
  done
}

stop_resource_monitor() {
  if [ -n "$MONITOR_PID" ] && kill -0 "$MONITOR_PID" 2>/dev/null; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
}

on_exit() {
  status=$?
  trap - EXIT
  stop_resource_monitor
  echo "========== HUGINN WHISPER DYNAMIC90S STAGE 3-4 EXIT =========="
  echo "exit_status=$status"
  echo "exit_time=$(date '+%Y-%m-%d %H:%M:%S')"
  exit "$status"
}

on_signal() {
  signal_name=$1
  echo "========== HUGINN WHISPER DYNAMIC90S STAGE 3-4 SIGNAL =========="
  echo "received_signal=$signal_name"
  print_resource_snapshot
  if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    kill -TERM "$TRAIN_PID" 2>/dev/null || true
  fi
  exit 143
}

trap on_exit EXIT
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT

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
  --learning_rate 1e-4 \
  --aligner_lr 1e-4 \
  --vit_lr 1e-4 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --fsdp "$FSDP_CONFIG_PATH" \
  --max_steps 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing false \
  --logging_steps 1 \
  --save_strategy no \
  --dataloader_num_workers 0 \
  --dataloader_pin_memory false \
  --dataset_num_proc 1 \
  --report_to none \
  --bf16 true &
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

python -u "$MARKER_INSPECTOR" --audit-dir "$AUDIT_DIR"

echo "========== HUGINN WHISPER DYNAMIC90S MERGED STAGE 3-4 VALIDATION PASSED =========="
