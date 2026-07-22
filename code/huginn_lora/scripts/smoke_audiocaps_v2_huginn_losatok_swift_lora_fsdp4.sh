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

# Keep the exact FSDP2 compatibility settings that passed on the Huginn
# multimodal full-parameter route.  In particular, Huginn's recurrent integer
# step-state must not be recomputed by activation checkpointing.
export HUGINN_AUDIO_FSDP2_NONPERSISTENT_ROPE=1
export HUGINN_LOSATOK_TRAIN_CHAIN_AUDIT=1

TRAIN_MANIFEST="${LOSATOK_FSDP4_SMOKE_MANIFEST:-$REPO_ROOT/data/audio_swift/audiocaps_v2/audiocaps_v2_train_swift.jsonl}"
TRAIN_STATS="$TRAIN_MANIFEST.stats.json"
OUTPUT_DIR="${LOSATOK_FSDP4_SMOKE_OUTPUT_DIR:-outputs/huginn_losatok_lora_fsdp4_smoke20}"
LOGGING_DIR="${LOSATOK_FSDP4_SMOKE_LOGGING_DIR:-$OUTPUT_DIR/tensorboard}"
PLUGIN_PATH="$REPO_ROOT/code/huginn_lora/plugins/huginn_losatok_swift.py"
MODEL_PATH="$REPO_ROOT/models/huginn-audio-losatok-v1"

WORLD_SIZE=4
MICRO_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
MAX_STEPS=20
LEARNING_RATE=1e-4
ALIGNER_LR=1e-4

FSDP_CONFIG='{"fsdp":"full_shard auto_wrap","fsdp_config":{"activation_checkpointing":false,"auto_wrap_policy":"TRANSFORMER_BASED_WRAP","cpu_ram_efficient_loading":true,"fsdp_version":2,"reshard_after_forward":true,"state_dict_type":"SHARDED_STATE_DICT"}}'

if [ ! -s "$TRAIN_MANIFEST" ] || [ ! -s "$TRAIN_STATS" ]; then
  echo "LoSATok FSDP4 smoke manifest or stats is missing: manifest=$TRAIN_MANIFEST stats=$TRAIN_STATS" >&2
  exit 1
fi
for required_path in "$MODEL_PATH" "$PLUGIN_PATH"; do
  if [ ! -e "$required_path" ]; then
    echo "Required LoSATok FSDP4 smoke path is missing: $required_path" >&2
    exit 1
  fi
done

python - "$TRAIN_STATS" <<'PY'
import json
import sys

with open(sys.argv[1], encoding='utf-8') as handle:
    stats = json.load(handle)
if stats.get('dataset') != 'audiocaps_v2' or stats.get('split') != 'train':
    raise SystemExit(
        f"Unexpected AudioCaps stats: dataset={stats.get('dataset')!r} split={stats.get('split')!r}"
    )
if not isinstance(stats.get('record_count'), int) or stats['record_count'] <= 0:
    raise SystemExit(f"Unexpected AudioCaps record_count: {stats.get('record_count')!r}")
if stats.get('audio_path_verification') != 'passed' or stats.get('wav_readability_verification') != 'passed':
    raise SystemExit('AudioCaps manifest verification is not marked passed')

from swift.arguments.sft_args import SftArguments
from dataclasses import fields

available = {field.name for field in fields(SftArguments)}
required = {'fsdp', 'save_strategy', 'tuner_type', 'freeze_aligner'}
missing = sorted(required - available)
if missing:
    raise SystemExit(f'Installed Swift lacks required FSDP LoRA arguments: {missing}')
print(f"[precheck] record_count={stats['record_count']}")
print('[precheck] manifest_verification=passed')
print('[precheck] swift_fsdp_lora_arguments=present')
PY

mkdir -p "$OUTPUT_DIR" "$LOGGING_DIR"
if find "$OUTPUT_DIR" -type d -name 'checkpoint-*' -print -quit | grep -q .; then
  echo "Smoke output already contains a checkpoint; choose a fresh LOSATOK_FSDP4_SMOKE_OUTPUT_DIR: $OUTPUT_DIR" >&2
  exit 1
fi

FSDP_CONFIG_PATH="$OUTPUT_DIR/fsdp2_lora_no_activation.json"
printf '%s\n' "$FSDP_CONFIG" > "$FSDP_CONFIG_PATH"

echo "========== HUGINN LOSATOK LORA FSDP4 20-STEP SMOKE =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "launch_mode=swift_cli_internal_torchrun"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "dataset=$TRAIN_MANIFEST"
echo "output_dir=$OUTPUT_DIR"
echo "model_arch=huginn_losatok_raven"
echo "architecture=current_fixed_audio_prefix"
echo "audio_encoder_policy=frozen"
echo "tuner_type=lora_llm"
echo "huginn_base_policy=frozen"
echo "aligner_policy=trainable_including_audio_bos_audio_eos"
echo "expected_losatok_trainable_parameters=0"
echo "expected_aligner_trainable_parameters=47224608"
echo "expected_huginn_lora_trainable_parameters=12541440"
echo "expected_huginn_base_trainable_parameters=0"
echo "fsdp=custom_fsdp2_json full_shard_auto_wrap"
echo "fsdp_version=2 state_dict_type=SHARDED_STATE_DICT"
echo "fsdp_activation_checkpointing=false gradient_checkpointing=false"
echo "per_device_train_batch_size=$MICRO_BATCH_SIZE"
echo "gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS"
echo "global_effective_batch_size=$((WORLD_SIZE * MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "max_steps=$MAX_STEPS"
echo "save_strategy=no save_checkpoint=false"
echo "learning_rate=$LEARNING_RATE aligner_lr=$ALIGNER_LR"
echo "train_chain_audit=true"

TRAIN_PID=""
MONITOR_PID=""

print_resource_snapshot() {
  echo "========== LOSATOK LORA FSDP4 RESOURCE SNAPSHOT =========="
  echo "snapshot_time=$(date '+%Y-%m-%d %H:%M:%S')"
  if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    ps -o pid,ppid,rss,vsz,%mem,etime,stat,cmd -p "$TRAIN_PID" || true
  fi
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
  for cgroup_file in /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.max \
    /sys/fs/cgroup/memory.events /sys/fs/cgroup/memory/memory.usage_in_bytes \
    /sys/fs/cgroup/memory/memory.limit_in_bytes; do
    if [ -r "$cgroup_file" ]; then
      echo "[cgroup] $(basename "$cgroup_file")=$(tr '\n' ' ' < "$cgroup_file")"
    fi
  done
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
  echo "========== HUGINN LOSATOK LORA FSDP4 20-STEP SMOKE EXIT =========="
  echo "exit_status=$status"
  echo "exit_time=$(date '+%Y-%m-%d %H:%M:%S')"
  if find "$OUTPUT_DIR" -type d -name 'checkpoint-*' -print -quit | grep -q .; then
    echo "ERROR: smoke was configured not to save checkpoints, but one was found" >&2
    status=1
  fi
  exit "$status"
}

on_signal() {
  local signal_name=$1
  echo "========== LOSATOK LORA FSDP4 SMOKE SIGNAL =========="
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

# Swift rejects save_only_model=true with SHARDED_STATE_DICT even when
# save_strategy=no. Keep the FSDP state-dict type used by the validated
# route, disable model-only saving, and rely on save_strategy=no for the
# no-checkpoint smoke contract.
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
  --output_dir "$OUTPUT_DIR" \
  --logging_dir "$LOGGING_DIR" \
  --tuner_type lora_llm \
  --freeze_aligner false \
  --learning_rate "$LEARNING_RATE" \
  --aligner_lr "$ALIGNER_LR" \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --fsdp "$FSDP_CONFIG_PATH" \
  --max_steps "$MAX_STEPS" \
  --per_device_train_batch_size "$MICRO_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --gradient_checkpointing false \
  --logging_steps 1 \
  --save_strategy no \
  --dataloader_num_workers 0 \
  --dataloader_pin_memory false \
  --dataset_num_proc 1 \
  --save_only_model false \
  --report_to none \
  --bf16 true &
TRAIN_PID=$!
resource_monitor &
MONITOR_PID=$!

set +e
wait "$TRAIN_PID"
TRAIN_STATUS=$?
set -e
exit "$TRAIN_STATUS"
