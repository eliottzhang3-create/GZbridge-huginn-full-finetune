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
export HUGINN_LOSATOK_TRAIN_CHAIN_AUDIT="${HUGINN_LOSATOK_TRAIN_CHAIN_AUDIT:-1}"

TRAIN_MANIFEST="${LOSATOK_TRAIN_MANIFEST:-$REPO_ROOT/data/audio_swift/audiocaps_v2/audiocaps_v2_train_swift.jsonl}"
TRAIN_STATS="$TRAIN_MANIFEST.stats.json"
PLUGIN_PATH="$REPO_ROOT/code/huginn_lora/plugins/huginn_losatok_swift.py"
MODEL_PATH="$REPO_ROOT/models/huginn-audio-losatok-v1"
LOSATOK_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/models/LoSATok
LOSATOK_CODE_DIR="$REPO_ROOT/code/huginn_lora/LosatokCode"
OUTPUT_DIR="${LOSATOK_OUTPUT_DIR:-outputs/huginn_losatok_audiocaps_v2_train_e3_b8ga4_5090}"
LOGGING_DIR="${LOSATOK_LOGGING_DIR:-$OUTPUT_DIR/tensorboard}"
NUM_TRAIN_EPOCHS="${LOSATOK_NUM_TRAIN_EPOCHS:-3}"
MAX_STEPS="${LOSATOK_MAX_STEPS:-}"
SAVE_STRATEGY="${LOSATOK_SAVE_STRATEGY:-epoch}"
SAVE_STEPS="${LOSATOK_SAVE_STEPS:-2802}"
SAVE_TOTAL_LIMIT="${LOSATOK_SAVE_TOTAL_LIMIT:-3}"
LOGGING_STEPS="${LOSATOK_LOGGING_STEPS:-10}"
REPORT_TO="${LOSATOK_REPORT_TO:-tensorboard}"
RESUME_FROM_CHECKPOINT="${LOSATOK_RESUME_FROM_CHECKPOINT:-}"
LEARNING_RATE="${LOSATOK_LEARNING_RATE:-1e-4}"
ALIGNER_LR="${LOSATOK_ALIGNER_LR:-1e-4}"
BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=4

if [ ! -s "$TRAIN_MANIFEST" ] || [ ! -s "$TRAIN_STATS" ]; then
  echo "AudioCaps manifest or stats are missing: manifest=$TRAIN_MANIFEST stats=$TRAIN_STATS" >&2
  exit 1
fi
for required_path in "$MODEL_PATH" "$PLUGIN_PATH" "$LOSATOK_ROOT/ckpts/losatok_kl1e-3.pth" \
  "$LOSATOK_ROOT/ckpts/semantic_encoder.pth" "$LOSATOK_ROOT/midashenglm" "$LOSATOK_CODE_DIR/config/16k_16k_25Hz_losatok.yml"; do
  if [ ! -e "$required_path" ]; then
    echo "Required LoSATok training path is missing: $required_path" >&2
    exit 1
  fi
done
if [ "$SAVE_STRATEGY" != epoch ] && [ "$SAVE_STRATEGY" != steps ]; then
  echo "LOSATOK_SAVE_STRATEGY must be epoch or steps, got: $SAVE_STRATEGY" >&2
  exit 1
fi

python - "$TRAIN_STATS" <<'PY'
import json
import sys
from dataclasses import fields

with open(sys.argv[1], encoding='utf-8') as handle:
    stats = json.load(handle)
if stats.get('dataset') != 'audiocaps_v2' or stats.get('split') != 'train':
    raise SystemExit(f"Unexpected AudioCaps stats: dataset={stats.get('dataset')!r} split={stats.get('split')!r}")
if not isinstance(stats.get('record_count'), int) or stats['record_count'] <= 0:
    raise SystemExit(f"Unexpected AudioCaps record_count: {stats.get('record_count')!r}")
if stats.get('audio_path_verification') != 'passed' or stats.get('wav_readability_verification') != 'passed':
    raise SystemExit('AudioCaps manifest verification is not marked passed')
if stats.get('limit_records') is None:
    if stats.get('source_csv_row_count') != stats['record_count'] + stats.get('excluded_row_count', -1):
        raise SystemExit('AudioCaps source/valid/excluded row accounting mismatch')

from swift.arguments.sft_args import SftArguments
available = {field.name for field in fields(SftArguments)}
required = {'num_train_epochs', 'save_strategy', 'save_total_limit', 'resume_from_checkpoint'}
missing = sorted(required - available)
if missing:
    raise SystemExit(f'Installed Swift lacks required formal-training arguments: {missing}')
PY

python - <<'PY'
import torch
import torchaudio
print(f"[env] torch={torch.__version__} torchaudio={torchaudio.__version__} cuda={torch.version.cuda}")
if torch.__version__ != torchaudio.__version__:
    raise SystemExit('Torch and torchaudio versions must match for LoSATok training')
PY

TRAIN_LENGTH_ARGS=(--num_train_epochs "$NUM_TRAIN_EPOCHS")
if [ -n "$MAX_STEPS" ]; then
  TRAIN_LENGTH_ARGS=(--max_steps "$MAX_STEPS")
fi
SAVE_ARGS=(--save_strategy "$SAVE_STRATEGY" --save_total_limit "$SAVE_TOTAL_LIMIT")
if [ "$SAVE_STRATEGY" = steps ]; then
  SAVE_ARGS+=(--save_steps "$SAVE_STEPS")
fi
RESUME_ARGS=()
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
  if [ ! -d "$RESUME_FROM_CHECKPOINT" ]; then
    echo "LoSATok resume checkpoint does not exist: $RESUME_FROM_CHECKPOINT" >&2
    exit 1
  fi
  export HUGINN_LOSATOK_FORCE_ALIGNER_TRAINABLE=1
  RESUME_ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

mkdir -p "$OUTPUT_DIR" "$LOGGING_DIR"
echo "========== AUDIOCAPS V2 HUGINN LOSATOK SWIFT FORMAL TRAIN =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "mode=lora_llm frozen_losatok_encoder aligner_trainable"
echo "dataset=$TRAIN_MANIFEST"
echo "output_dir=$OUTPUT_DIR"
echo "logging_dir=$LOGGING_DIR"
echo "num_train_epochs=$NUM_TRAIN_EPOCHS max_steps=${MAX_STEPS:-<unset>}"
echo "per_device_train_batch_size=$BATCH_SIZE gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS effective_batch_size=32"
echo "learning_rate=$LEARNING_RATE aligner_lr=$ALIGNER_LR"
echo "dataset_shuffle=true train_dataloader_shuffle=true"
echo "save_strategy=$SAVE_STRATEGY save_steps=$SAVE_STEPS save_total_limit=$SAVE_TOTAL_LIMIT save_only_model=false"
echo "logging_steps=$LOGGING_STEPS report_to=$REPORT_TO"
echo "resource_snapshot_interval_seconds=10"
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
  echo "resume_from_checkpoint=$RESUME_FROM_CHECKPOINT"
fi

TRAIN_PID=""
MONITOR_PID=""

print_resource_snapshot() {
  echo "========== LOSATOK FORMAL TRAIN RESOURCE SNAPSHOT =========="
  echo "snapshot_time=$(date '+%Y-%m-%d %H:%M:%S')"
  if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    ps -o pid,ppid,rss,vsz,%mem,etime,stat,cmd -p "$TRAIN_PID" || true
  fi
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
  for cgroup_file in /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.events \
    /sys/fs/cgroup/memory/memory.usage_in_bytes /sys/fs/cgroup/memory/memory.limit_in_bytes; do
    if [ -r "$cgroup_file" ]; then
      echo "[cgroup] $(basename "$cgroup_file")=$(tr '\n' ' ' < "$cgroup_file")"
    fi
  done
}

resource_monitor() {
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    print_resource_snapshot
    sleep 10
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
  echo "========== AUDIOCAPS V2 HUGINN LOSATOK FORMAL TRAIN EXIT =========="
  echo "exit_status=$status"
  echo "exit_time=$(date '+%Y-%m-%d %H:%M:%S')"
  exit "$status"
}

on_signal() {
  local signal_name=$1
  echo "========== LOSATOK FORMAL TRAIN SIGNAL =========="
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
  --model_type huginn_losatok_raven \
  --template huginn_losatok_text \
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
  --learning_rate "$LEARNING_RATE" \
  --aligner_lr "$ALIGNER_LR" \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  "${TRAIN_LENGTH_ARGS[@]}" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --gradient_checkpointing true \
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
