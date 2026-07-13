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

TRAIN_MANIFEST="${AUDIOCAPS_TRAIN_MANIFEST:-$REPO_ROOT/data/audio_swift/audiocaps_v2/audiocaps_v2_train_swift.jsonl}"
TRAIN_STATS="$TRAIN_MANIFEST.stats.json"
PLUGIN_PATH="$REPO_ROOT/code/huginn_lora/plugins/huginn_audio_swift.py"
MODEL_PATH="$REPO_ROOT/models/huginn-audio-whisper-v1"
OUTPUT_DIR="${AUDIOCAPS_OUTPUT_DIR:-outputs/huginn_audio_audiocaps_v2_train_e5_b8ga4_5090}"
LOGGING_DIR="${AUDIOCAPS_LOGGING_DIR:-$OUTPUT_DIR/tensorboard}"
NUM_TRAIN_EPOCHS="${AUDIOCAPS_NUM_TRAIN_EPOCHS:-5}"
MAX_STEPS="${AUDIOCAPS_MAX_STEPS:-}"
SAVE_STRATEGY="${AUDIOCAPS_SAVE_STRATEGY:-epoch}"
SAVE_STEPS="${AUDIOCAPS_SAVE_STEPS:-20}"
SAVE_TOTAL_LIMIT="${AUDIOCAPS_SAVE_TOTAL_LIMIT:-5}"
LOGGING_STEPS="${AUDIOCAPS_LOGGING_STEPS:-10}"
REPORT_TO="${AUDIOCAPS_REPORT_TO:-tensorboard}"
RESUME_FROM_CHECKPOINT="${AUDIOCAPS_RESUME_FROM_CHECKPOINT:-}"

if [ ! -s "$TRAIN_MANIFEST" ]; then
  echo "AudioCaps train manifest is missing or empty: $TRAIN_MANIFEST" >&2
  exit 1
fi
if [ ! -s "$TRAIN_STATS" ]; then
  echo "AudioCaps train stats are missing or empty: $TRAIN_STATS" >&2
  exit 1
fi
if [ "$SAVE_STRATEGY" != "epoch" ] && [ "$SAVE_STRATEGY" != "steps" ]; then
  echo "AUDIOCAPS_SAVE_STRATEGY must be epoch or steps, got: $SAVE_STRATEGY" >&2
  exit 1
fi

python - "$TRAIN_STATS" <<'PY'
import json
import sys
from dataclasses import fields

with open(sys.argv[1], encoding="utf-8") as f:
    stats = json.load(f)
if stats.get("dataset") != "audiocaps_v2" or stats.get("split") != "train":
    raise SystemExit(f"Unexpected AudioCaps manifest stats: dataset={stats.get('dataset')!r} split={stats.get('split')!r}")
if not isinstance(stats.get("record_count"), int) or stats["record_count"] <= 0:
    raise SystemExit(f"Unexpected AudioCaps record_count: {stats.get('record_count')!r}")
if stats.get("audio_path_verification") != "passed":
    raise SystemExit("AudioCaps audio-path verification is not marked passed")
if stats.get("wav_readability_verification") != "passed":
    raise SystemExit("AudioCaps WAV readability verification is not marked passed")

from swift.arguments.sft_args import SftArguments
available_fields = {field.name for field in fields(SftArguments)}
required_fields = {"num_train_epochs", "save_strategy", "save_total_limit"}
missing_fields = sorted(required_fields - available_fields)
if missing_fields:
    raise SystemExit(f"Installed Swift SftArguments lacks required epoch-checkpoint fields: {missing_fields}")
PY

mkdir -p "$OUTPUT_DIR" "$LOGGING_DIR"
echo "========== AUDIOCAPS V2 HUGINN AUDIO SWIFT TRAIN 5090 =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "mode=lora_llm generator_frozen_audio_encoder aligner_trainable"
echo "dataset=$TRAIN_MANIFEST"
echo "output_dir=$OUTPUT_DIR"
echo "logging_dir=$LOGGING_DIR"
echo "num_train_epochs=$NUM_TRAIN_EPOCHS"
echo "max_steps=${MAX_STEPS:-<unset>}"
echo "per_device_train_batch_size=8"
echo "gradient_accumulation_steps=4"
echo "effective_batch_size=32"
echo "dataset_shuffle=true"
echo "train_dataloader_shuffle=true"
echo "save_strategy=$SAVE_STRATEGY"
echo "save_steps=$SAVE_STEPS"
echo "save_total_limit=$SAVE_TOTAL_LIMIT"
echo "logging_steps=$LOGGING_STEPS"
echo "report_to=$REPORT_TO"
echo "save_only_model=false"
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
  echo "resume_from_checkpoint=$RESUME_FROM_CHECKPOINT"
fi

TRAIN_LENGTH_ARGS=(--num_train_epochs "$NUM_TRAIN_EPOCHS")
if [ -n "$MAX_STEPS" ]; then
  TRAIN_LENGTH_ARGS=(--max_steps "$MAX_STEPS")
fi
SAVE_ARGS=(--save_strategy "$SAVE_STRATEGY" --save_total_limit "$SAVE_TOTAL_LIMIT")
if [ "$SAVE_STRATEGY" = "steps" ]; then
  SAVE_ARGS+=(--save_steps "$SAVE_STEPS")
fi
RESUME_ARGS=()
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
  if [ ! -d "$RESUME_FROM_CHECKPOINT" ]; then
    echo "Resume checkpoint directory does not exist: $RESUME_FROM_CHECKPOINT" >&2
    exit 1
  fi
  RESUME_ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

TRAIN_PID=""
MONITOR_PID=""

print_resource_snapshot() {
  echo "========== AUDIOCAPS TRAIN RESOURCE SNAPSHOT =========="
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
  echo "========== AUDIOCAPS V2 TRAIN EXIT =========="
  echo "exit_status=$status"
  echo "exit_time=$(date '+%Y-%m-%d %H:%M:%S')"
  exit "$status"
}

on_signal() {
  signal_name=$1
  echo "========== AUDIOCAPS V2 TRAIN SIGNAL =========="
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
  --dataset "$TRAIN_MANIFEST" \
  --dataset_shuffle true \
  --train_dataloader_shuffle true \
  --sortish_sampler false \
  --group_by_length false \
  --max_length 192 \
  --output_dir "$OUTPUT_DIR" \
  --logging_dir "$LOGGING_DIR" \
  --tuner_type lora_llm \
  --freeze_aligner false \
  --learning_rate 1e-4 \
  --aligner_lr 1e-4 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  "${TRAIN_LENGTH_ARGS[@]}" \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --logging_steps "$LOGGING_STEPS" \
  "${SAVE_ARGS[@]}" \
  --dataloader_num_workers 0 \
  --dataloader_pin_memory false \
  --dataset_num_proc 1 \
  --save_only_model false \
  --report_to "$REPORT_TO" \
  --bf16 true "${RESUME_ARGS[@]}" &
TRAIN_PID=$!
resource_monitor &
MONITOR_PID=$!

set +e
wait "$TRAIN_PID"
TRAIN_STATUS=$?
set -e
exit "$TRAIN_STATUS"
