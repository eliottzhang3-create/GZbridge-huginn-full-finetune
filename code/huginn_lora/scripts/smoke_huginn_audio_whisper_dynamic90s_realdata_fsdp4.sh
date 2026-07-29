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

MAX_STEPS="${HUGINN_AUDIO_DYNAMIC90S_REALDATA_MAX_STEPS:-8}"
SEED="${HUGINN_DYNAMIC90S_MIXTURE_SEED:-20260730}"
DATASET_MAX_SAMPLES="${HUGINN_DYNAMIC90S_REALDATA_DATASET_MAX_SAMPLES:-128}"
RUN_ROOT="${HUGINN_AUDIO_DYNAMIC90S_REALDATA_RUN_ROOT:-outputs/huginn_audio_whisper_dynamic90s_realdata_fsdp4/run-$(date +%Y%m%d_%H%M%S)}"
if [ -e "$RUN_ROOT" ]; then
  echo "Real-data FSDP4 run root already exists; choose a fresh HUGINN_AUDIO_DYNAMIC90S_REALDATA_RUN_ROOT: $RUN_ROOT" >&2
  exit 1
fi
mkdir -p "$RUN_ROOT"
RUN_ROOT="$(cd "$RUN_ROOT" && pwd)"
AUDIT_DIR="$RUN_ROOT/rank_audits"
OUTPUT_DIR="$RUN_ROOT/swift_output"
FSDP_CONFIG_PATH="$RUN_ROOT/fsdp2_lora_no_activation.json"
MODEL_PATH="$REPO_ROOT/models/huginn-audio-whisper-dynamic90s-v1"
PLUGIN_PATH="$REPO_ROOT/code/huginn_lora/plugins/huginn_audio_whisper_dynamic90s_mixture_swift.py"
REGISTRY="${HUGINN_DYNAMIC90S_POOL_REGISTRY:-$REPO_ROOT/data/audio_swift/huginn_whisper_dynamic90s_multitask/v1/pool_registry.json}"
DATA_CHAIN_REPORT="${HUGINN_DYNAMIC90S_REAL_DATA_CHAIN_REPORT:-$REPO_ROOT/data/audio_swift/huginn_whisper_dynamic90s_multitask/v1/real_data_chain_report.json}"
MARKER_INSPECTOR="$REPO_ROOT/code/huginn_lora/scripts/inspect_huginn_whisper_dynamic90s_realdata_fsdp4_markers.py"

for required_path in "$MODEL_PATH" "$PLUGIN_PATH" "$REGISTRY" "$DATA_CHAIN_REPORT" "$MARKER_INSPECTOR"; do
  if [ ! -e "$required_path" ]; then
    echo "Required real-data FSDP4 path is missing: $required_path" >&2
    exit 1
  fi
done

python - "$DATA_CHAIN_REPORT" "$MAX_STEPS" "$DATASET_MAX_SAMPLES" <<'PY'
import json
import sys
from dataclasses import fields
from pathlib import Path
from swift.arguments.sft_args import SftArguments

report = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
if not report.get('validation_passed'):
    raise SystemExit(f'Real data-chain prerequisite has not passed: {sys.argv[1]}')
max_steps = int(sys.argv[2])
dataset_max_samples = int(sys.argv[3])
if max_steps <= 1:
    raise SystemExit(f'Real-data FSDP4 gate requires max_steps > 1, got {max_steps}')
if dataset_max_samples < max_steps * 4:
    raise SystemExit(
        f'Dataset quota must cover all global samples: quota={dataset_max_samples} required={max_steps * 4}'
    )
available = {field.name for field in fields(SftArguments)}
required = {
    'fsdp', 'tuner_type', 'freeze_vit', 'freeze_aligner', 'lora_rank',
    'lora_alpha', 'lora_dropout', 'max_steps', 'save_strategy', 'streaming',
}
missing = sorted(required - available)
if missing:
    raise SystemExit(f'Installed Swift lacks required real-data gate arguments: {missing}')
print('[precheck] real_data_chain=passed swift_arguments=present dataset_quota=sufficient')
PY

FSDP_CONFIG='{"fsdp":"full_shard auto_wrap","fsdp_config":{"activation_checkpointing":false,"auto_wrap_policy":"TRANSFORMER_BASED_WRAP","cpu_ram_efficient_loading":true,"fsdp_version":2,"reshard_after_forward":true,"state_dict_type":"SHARDED_STATE_DICT"}}'
printf '%s\n' "$FSDP_CONFIG" > "$FSDP_CONFIG_PATH"
mkdir -p "$AUDIT_DIR" "$OUTPUT_DIR"

export HUGINN_DYNAMIC90S_POOL_REGISTRY="$REGISTRY"
export HUGINN_DYNAMIC90S_MIXTURE_SEED="$SEED"
export HUGINN_DYNAMIC90S_MIXTURE_START_POSITION=0
export HUGINN_DYNAMIC90S_MIXTURE_MAX_SAMPLES="$DATASET_MAX_SAMPLES"
export HUGINN_AUDIO_DYNAMIC90S_FSDP2_NONPERSISTENT_ROPE=1
export HUGINN_AUDIO_DYNAMIC90S_TRAIN_CHAIN_AUDIT=1
export HUGINN_AUDIO_DYNAMIC90S_REALDATA_AUDIT_DIR="$AUDIT_DIR"
export HUGINN_AUDIO_DYNAMIC90S_REALDATA_MAX_STEPS="$MAX_STEPS"

echo "========== HUGINN WHISPER DYNAMIC90S REALDATA FSDP4 START =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "launch_mode=swift_cli_internal_torchrun"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES NPROC_PER_NODE=$NPROC_PER_NODE"
echo "dataset=indexed_four_pool_real_mixture registry=$REGISTRY"
echo "mixture_seed=$SEED start_position=0 dataset_max_samples=$DATASET_MAX_SAMPLES"
echo "task_weights=AAC:0.60,ASR:0.40 aac_weights=WavCaps:0.60,AudioCaps:0.30,Clotho:0.10"
echo "gigaspeech=read_only_opus_segment_decode_in_memory"
echo "audio_tokens=dynamic_complete_120ms_blocks runtime_accumulation=true"
echo "audio_over_90s=retain_first_90s no_duration_discard=true"
echo "whisper_encoder=frozen_fp32"
echo "aligner=trainable tensors=14 learning_rate=1e-4"
echo "huginn=lora_llm tensors=66 rank=8 alpha=16 dropout=0.05 learning_rate=1e-4"
echo "lora_scope=huginn_transformer_only"
echo "fsdp=custom_fsdp2_json world_size=4 sharded_dtensor_required=true"
echo "fsdp_units=whisper_whole,aligner_whole,prelude_2blocks,core_adapter_plus_4blocks,coda_2blocks"
echo "fsdp_reshard_after_forward=true for_all_units=true"
echo "per_device_train_batch_size=1 gradient_accumulation_steps=1 global_batch_size=4"
echo "max_steps=$MAX_STEPS save_strategy=no checkpoint_resume=next_separate_gate"
echo "run_root=$RUN_ROOT"

TRAIN_PID=""
MONITOR_PID=""

print_resource_snapshot() {
  echo "========== DYNAMIC90S REALDATA RESOURCE SNAPSHOT =========="
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
  echo "========== HUGINN WHISPER DYNAMIC90S REALDATA FSDP4 EXIT =========="
  echo "exit_status=$status"
  echo "exit_time=$(date '+%Y-%m-%d %H:%M:%S')"
  exit "$status"
}

on_signal() {
  signal_name=$1
  echo "========== HUGINN WHISPER DYNAMIC90S REALDATA FSDP4 SIGNAL =========="
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
  --dataset "$REGISTRY" \
  --streaming true \
  --dataset_shuffle false \
  --train_dataloader_shuffle false \
  --sortish_sampler false \
  --group_by_length false \
  --max_length 192 \
  --output_dir "$OUTPUT_DIR" \
  --tuner_type lora_llm \
  --freeze_vit true \
  --freeze_aligner false \
  --learning_rate 1e-4 \
  --aligner_lr 1e-4 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --fsdp "$FSDP_CONFIG_PATH" \
  --max_steps "$MAX_STEPS" \
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

python -u "$MARKER_INSPECTOR" \
  --audit-dir "$AUDIT_DIR" \
  --registry "$REGISTRY" \
  --seed "$SEED" \
  --max-steps "$MAX_STEPS" \
  --world-size 4 \
  --per-device-batch-size 1

echo "========== HUGINN WHISPER DYNAMIC90S REALDATA FSDP4 PASSED =========="
