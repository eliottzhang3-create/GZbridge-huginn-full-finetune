#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export NPROC_PER_NODE=7
export OMP_NUM_THREADS=4
export HUGINN_AUDIO_FSDP2_NONPERSISTENT_ROPE=1
export HUGINN_AUDIO_TRAIN_CHAIN_AUDIT=1

TRAIN_MANIFEST="${AUDIOCAPS_FULL_FSDP_TRAIN_MANIFEST:-$REPO_ROOT/data/audio_swift/audiocaps_v2/audiocaps_v2_train_swift.jsonl}"
TRAIN_STATS="$TRAIN_MANIFEST.stats.json"
OUTPUT_DIR="${AUDIOCAPS_FULL_FSDP_OUTPUT_DIR:-outputs/huginn_audio_audiocaps_v2_full_fsdp7_e2_b1ga4}"
LOGGING_DIR="${AUDIOCAPS_FULL_FSDP_LOGGING_DIR:-$OUTPUT_DIR/tensorboard}"
MIN_FREE_GB="${AUDIOCAPS_FULL_FSDP_MIN_FREE_GB:-200}"

WORLD_SIZE=7
MICRO_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
NUM_TRAIN_EPOCHS=2
LOGGING_STEPS=10
LEARNING_RATE=1e-5
ALIGNER_LR=1e-4
WARMUP_RATIO=0.05
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0

# Swift's `--fsdp fsdp2` is an immutable preset. A full custom config passed
# directly to `--fsdp` is required to retain FSDP2 while disabling activation checkpointing.
FSDP_CONFIG='{"fsdp":"full_shard auto_wrap","fsdp_config":{"activation_checkpointing":false,"auto_wrap_policy":"TRANSFORMER_BASED_WRAP","cpu_ram_efficient_loading":true,"fsdp_version":2,"reshard_after_forward":true,"state_dict_type":"SHARDED_STATE_DICT"}}'

if [ ! -s "$TRAIN_MANIFEST" ] || [ ! -s "$TRAIN_STATS" ]; then
  echo "AudioCaps manifest or stats is missing: manifest=$TRAIN_MANIFEST stats=$TRAIN_STATS" >&2
  exit 1
fi
CALCULATED_TRAINING_STEPS="$(python - "$TRAIN_STATS" "$WORLD_SIZE" "$MICRO_BATCH_SIZE" "$GRADIENT_ACCUMULATION_STEPS" "$NUM_TRAIN_EPOCHS" <<'PY'
import json
import math
import sys
from dataclasses import fields

stats_path, world_size, micro_batch, grad_accum, epochs = sys.argv[1:]
world_size = int(world_size)
micro_batch = int(micro_batch)
grad_accum = int(grad_accum)
epochs = int(epochs)
with open(stats_path, encoding='utf-8') as handle:
    stats = json.load(handle)
if stats.get('dataset') != 'audiocaps_v2' or stats.get('split') != 'train':
    raise SystemExit(f"Unexpected AudioCaps stats: dataset={stats.get('dataset')!r} split={stats.get('split')!r}")
record_count = stats.get('record_count')
if not isinstance(record_count, int) or record_count <= 0:
    raise SystemExit(f"Unexpected AudioCaps record_count: {record_count!r}")
if stats.get('audio_path_verification') != 'passed' or stats.get('wav_readability_verification') != 'passed':
    raise SystemExit('AudioCaps manifest verification is not marked passed')

# DistributedSampler pads to an even number of examples across ranks. Trainer then
# groups each rank's B=1 batches into gradient-accumulation optimizer updates.
per_rank_samples = math.ceil(record_count / world_size)
per_rank_batches = math.ceil(per_rank_samples / micro_batch)
steps_per_epoch = math.ceil(per_rank_batches / grad_accum)

from swift.arguments.sft_args import SftArguments
available_fields = {field.name for field in fields(SftArguments)}
required_fields = {
    'fsdp', 'num_train_epochs', 'save_strategy', 'save_steps', 'save_total_limit',
    'lr_scheduler_type', 'warmup_ratio', 'weight_decay', 'max_grad_norm',
}
missing_fields = sorted(required_fields - available_fields)
if missing_fields:
    raise SystemExit(f'Installed Swift SftArguments lacks formal FSDP fields: {missing_fields}')

print(record_count, steps_per_epoch, steps_per_epoch * epochs)
PY
)"
read -r RECORD_COUNT STEPS_PER_EPOCH TOTAL_STEPS <<< "$CALCULATED_TRAINING_STEPS"

# Seven ranks produce an odd 3203 optimizer updates per epoch. Saving on epoch
# boundaries avoids pretending that a non-integer optimizer step is a half epoch.
SAVE_STRATEGY=epoch
SAVE_TOTAL_LIMIT=2

AVAILABLE_GB="$(df -BG "$REPO_ROOT" | awk 'NR==2 {gsub(/G/, "", $4); print $4}')"
if [ -z "$AVAILABLE_GB" ] || [ "$AVAILABLE_GB" -lt "$MIN_FREE_GB" ]; then
  echo "Insufficient free storage for FSDP sharded checkpoints: available=${AVAILABLE_GB:-unknown}G required=${MIN_FREE_GB}G" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$LOGGING_DIR"
if find "$OUTPUT_DIR" -type d -name 'checkpoint-*' -print -quit | grep -q .; then
  echo "Fresh-training output directory already contains a checkpoint; choose a new AUDIOCAPS_FULL_FSDP_OUTPUT_DIR: $OUTPUT_DIR" >&2
  exit 1
fi
FSDP_CONFIG_PATH="$OUTPUT_DIR/fsdp2_full_train_no_activation.json"
printf '%s\n' "$FSDP_CONFIG" > "$FSDP_CONFIG_PATH"

echo "========== AUDIOCAPS V2 HUGINN FULL FSDP7 FRESH TRAIN =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "launch_mode=swift_cli_internal_torchrun"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "dataset=$TRAIN_MANIFEST"
echo "record_count=$RECORD_COUNT"
echo "output_dir=$OUTPUT_DIR"
echo "logging_dir=$LOGGING_DIR"
echo "tuner_type=full"
echo "freeze_llm=false freeze_vit=true freeze_aligner=false"
echo "audio_encoder_policy=frozen"
echo "fsdp=custom_fsdp2_json full_shard_auto_wrap"
echo "fsdp_version=2 state_dict_type=SHARDED_STATE_DICT"
echo "fsdp_activation_checkpointing=false gradient_checkpointing=false"
echo "per_device_train_batch_size=$MICRO_BATCH_SIZE"
echo "gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS"
echo "global_effective_batch_size=$((WORLD_SIZE * MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "num_train_epochs=$NUM_TRAIN_EPOCHS max_steps=$TOTAL_STEPS"
echo "steps_per_epoch_7rank=$STEPS_PER_EPOCH total_steps=$TOTAL_STEPS"
echo "initialization=original_huginn_audio_model_no_resume_checkpoint"
echo "save_strategy=$SAVE_STRATEGY save_total_limit=$SAVE_TOTAL_LIMIT"
echo "expected_checkpoints=$STEPS_PER_EPOCH,$TOTAL_STEPS"
echo "learning_rate=$LEARNING_RATE aligner_lr=$ALIGNER_LR"
echo "lr_scheduler_type=cosine warmup_ratio=$WARMUP_RATIO weight_decay=$WEIGHT_DECAY max_grad_norm=$MAX_GRAD_NORM"
echo "logging_steps=$LOGGING_STEPS report_to=tensorboard"
echo "save_only_model=false"
echo "free_storage_gb=$AVAILABLE_GB min_required_gb=$MIN_FREE_GB"

TRAIN_PID=""
MONITOR_PID=""

print_resource_snapshot() {
  echo "========== AUDIOCAPS FULL FSDP7 RESOURCE SNAPSHOT =========="
  echo "snapshot_time=$(date '+%Y-%m-%d %H:%M:%S')"
  if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    ps -o pid,ppid,rss,vsz,%mem,etime,stat,cmd -p "$TRAIN_PID" || true
  fi
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
  for cgroup_file in \
    /sys/fs/cgroup/memory.current \
    /sys/fs/cgroup/memory.max \
    /sys/fs/cgroup/memory/memory.usage_in_bytes \
    /sys/fs/cgroup/memory/memory.limit_in_bytes; do
    if [ -r "$cgroup_file" ]; then
      echo "[cgroup] $(basename "$cgroup_file")=$(tr -d '\n' < "$cgroup_file")"
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
  echo "========== AUDIOCAPS FULL FSDP7 FRESH TRAIN EXIT =========="
  echo "exit_status=$status"
  echo "exit_time=$(date '+%Y-%m-%d %H:%M:%S')"
  exit "$status"
}

on_signal() {
  signal_name=$1
  echo "========== AUDIOCAPS FULL FSDP7 FRESH TRAIN SIGNAL =========="
  echo "received_signal=$signal_name"
  echo "signal_time=$(date '+%Y-%m-%d %H:%M:%S')"
  if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    kill -TERM "$TRAIN_PID" 2>/dev/null || true
  fi
  exit 143
}

trap on_exit EXIT
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT

CMD=(swift sft)
CMD+=(--model "$REPO_ROOT/models/huginn-audio-whisper-v1")
CMD+=(--model_type huginn_audio_raven --template huginn_audio_text)
CMD+=(--external_plugins "$REPO_ROOT/code/huginn_lora/plugins/huginn_audio_swift.py")
CMD+=(--dataset "$TRAIN_MANIFEST")
CMD+=(--dataset_shuffle true --train_dataloader_shuffle true --sortish_sampler false --group_by_length false)
CMD+=(--max_length 192 --output_dir "$OUTPUT_DIR" --logging_dir "$LOGGING_DIR")
CMD+=(--tuner_type full --freeze_llm false --freeze_vit true --freeze_aligner false --fsdp "$FSDP_CONFIG_PATH")
CMD+=(--learning_rate "$LEARNING_RATE" --aligner_lr "$ALIGNER_LR")
CMD+=(--lr_scheduler_type cosine --warmup_ratio "$WARMUP_RATIO" --weight_decay "$WEIGHT_DECAY" --max_grad_norm "$MAX_GRAD_NORM")
CMD+=(--gradient_checkpointing false --num_train_epochs "$NUM_TRAIN_EPOCHS" --max_steps "$TOTAL_STEPS")
CMD+=(--per_device_train_batch_size "$MICRO_BATCH_SIZE" --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS")
CMD+=(--logging_steps "$LOGGING_STEPS" --save_strategy "$SAVE_STRATEGY" --save_total_limit "$SAVE_TOTAL_LIMIT")
CMD+=(--dataloader_num_workers 0 --dataloader_pin_memory false --dataset_num_proc 1)
CMD+=(--save_only_model false --report_to tensorboard --bf16 true --seed 42 --data_seed 42)

"${CMD[@]}" &
TRAIN_PID=$!
resource_monitor &
MONITOR_PID=$!

set +e
wait "$TRAIN_PID"
TRAIN_STATUS=$?
set -e
if [ "$TRAIN_STATUS" -eq 0 ]; then
  FINAL_CHECKPOINT="$(find "$OUTPUT_DIR" -type d -name "checkpoint-$TOTAL_STEPS" -print -quit)"
  if [ -z "$FINAL_CHECKPOINT" ]; then
    echo "Training reported success but expected final FSDP checkpoint was not found: checkpoint-$TOTAL_STEPS" >&2
    TRAIN_STATUS=1
  else
    echo "========== AUDIOCAPS FULL FSDP7 FINAL CHECKPOINT VERIFIED =========="
    echo "final_checkpoint=$FINAL_CHECKPOINT"
  fi
fi
exit "$TRAIN_STATUS"
