#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

MASTER_MANIFEST="${FORMAL_MASTER_MANIFEST:-$REPO_ROOT/data/audio_swift/acavcaps/acavcaps_subset_56_full_master_shuffled.jsonl}"
MASTER_STATS="$MASTER_MANIFEST.stats.json"
PLUGIN_PATH="$REPO_ROOT/code/huginn_lora/plugins/huginn_audio_swift.py"
MODEL_PATH="$REPO_ROOT/models/huginn-audio-whisper-v1"
FORMAL_OUTPUT_DIR="${FORMAL_OUTPUT_DIR:-outputs/huginn_audio_acavcaps_formal_master_b8ga4_5090}"
FORMAL_LOGGING_DIR="${FORMAL_LOGGING_DIR:-$FORMAL_OUTPUT_DIR/tensorboard}"
FORMAL_MAX_STEPS="${FORMAL_MAX_STEPS:-7500}"
FORMAL_SAVE_STEPS="${FORMAL_SAVE_STEPS:-200}"
FORMAL_LOGGING_STEPS="${FORMAL_LOGGING_STEPS:-10}"
FORMAL_RESUME_FROM_CHECKPOINT="${FORMAL_RESUME_FROM_CHECKPOINT:-}"

# The master manifest samples all 56 tar shards globally. Keep every tar index
# open in the single-worker loader to avoid repeatedly indexing gzip archives.
export HUGINN_AUDIO_TARFILE_CACHE_LIMIT="${HUGINN_AUDIO_TARFILE_CACHE_LIMIT:-64}"

if [ ! -s "$MASTER_MANIFEST" ]; then
  echo "Formal master manifest is missing or empty: $MASTER_MANIFEST" >&2
  exit 1
fi
if [ ! -s "$MASTER_STATS" ]; then
  echo "Formal master stats are missing or empty: $MASTER_STATS" >&2
  exit 1
fi

python -c '
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    stats = json.load(f)
record_count = stats.get("record_count")
if record_count != 239854:
    raise SystemExit(f"Unexpected master record_count: {record_count}")
if stats.get("audio_caption_pair_verification") != "passed":
    raise SystemExit("Master audio/caption pair verification is not marked passed")
' "$MASTER_STATS"

mkdir -p "$FORMAL_OUTPUT_DIR" "$FORMAL_LOGGING_DIR"
echo "========== ACAVCAPS HUGINN AUDIO SWIFT FORMAL TRAIN 5090 =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "mode=lora_llm generator_frozen_audio_encoder aligner_trainable"
echo "dataset=$MASTER_MANIFEST"
echo "output_dir=$FORMAL_OUTPUT_DIR"
echo "logging_dir=$FORMAL_LOGGING_DIR"
echo "max_steps=$FORMAL_MAX_STEPS"
echo "per_device_train_batch_size=8"
echo "gradient_accumulation_steps=4"
echo "effective_batch_size=32"
echo "tarfile_cache_limit=$HUGINN_AUDIO_TARFILE_CACHE_LIMIT"
echo "save_steps=$FORMAL_SAVE_STEPS"
echo "save_total_limit=2"
echo "logging_steps=$FORMAL_LOGGING_STEPS"
echo "report_to=tensorboard"
echo "save_only_model=false"
if [ -n "$FORMAL_RESUME_FROM_CHECKPOINT" ]; then
  echo "resume_from_checkpoint=$FORMAL_RESUME_FROM_CHECKPOINT"
fi

RESUME_ARGS=()
if [ -n "$FORMAL_RESUME_FROM_CHECKPOINT" ]; then
  if [ ! -d "$FORMAL_RESUME_FROM_CHECKPOINT" ]; then
    echo "Resume checkpoint directory does not exist: $FORMAL_RESUME_FROM_CHECKPOINT" >&2
    exit 1
  fi
  RESUME_ARGS+=(--resume_from_checkpoint "$FORMAL_RESUME_FROM_CHECKPOINT")
fi

TRAIN_PID=""
MONITOR_PID=""

print_resource_snapshot() {
  echo "========== FORMAL TRAIN RESOURCE SNAPSHOT =========="
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
  echo "========== ACAVCAPS FORMAL TRAIN EXIT =========="
  echo "exit_status=$status"
  echo "exit_time=$(date '+%Y-%m-%d %H:%M:%S')"
  exit "$status"
}

on_signal() {
  signal_name=$1
  echo "========== ACAVCAPS FORMAL TRAIN SIGNAL =========="
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

swift sft \
  --model "$MODEL_PATH" \
  --model_type huginn_audio_raven \
  --template huginn_audio_text \
  --external_plugins "$PLUGIN_PATH" \
  --dataset "$MASTER_MANIFEST" \
  --max_length 192 \
  --output_dir "$FORMAL_OUTPUT_DIR" \
  --logging_dir "$FORMAL_LOGGING_DIR" \
  --tuner_type lora_llm \
  --freeze_aligner false \
  --learning_rate 1e-4 \
  --aligner_lr 1e-4 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --max_steps "$FORMAL_MAX_STEPS" \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --logging_steps "$FORMAL_LOGGING_STEPS" \
  --save_steps "$FORMAL_SAVE_STEPS" \
  --save_total_limit 2 \
  --dataloader_num_workers 0 \
  --dataloader_pin_memory false \
  --dataset_num_proc 1 \
  --save_only_model false \
  --report_to tensorboard \
  --bf16 true "${RESUME_ARGS[@]}" &
TRAIN_PID=$!
resource_monitor &
MONITOR_PID=$!

set +e
wait "$TRAIN_PID"
TRAIN_STATUS=$?
set -e
exit "$TRAIN_STATUS"
