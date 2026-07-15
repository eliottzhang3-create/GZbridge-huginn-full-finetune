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

TRAIN_MANIFEST="${WAVCAPS_TRAIN_MANIFEST:-$REPO_ROOT/data/audio_swift/wavcaps_audioset/wavcaps_audioset_sl_train_swift.jsonl}"
TRAIN_STATS="$TRAIN_MANIFEST.stats.json"
INIT_CHECKPOINT="${WAVCAPS_INIT_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_audio_audiocaps_v2_train_e5_b8ga4_5090/v0-20260713-155848/checkpoint-5604}"
PLUGIN_PATH="$REPO_ROOT/code/huginn_lora/plugins/huginn_audio_swift.py"
MODEL_PATH="$REPO_ROOT/models/huginn-audio-whisper-v1"
OUTPUT_DIR="${WAVCAPS_OUTPUT_DIR:-outputs/huginn_audio_wavcaps_audioset_sl_e2_warmstart5604_b8ga4_5090}"
LOGGING_DIR="${WAVCAPS_LOGGING_DIR:-$OUTPUT_DIR/tensorboard}"
NUM_TRAIN_EPOCHS="${WAVCAPS_NUM_TRAIN_EPOCHS:-2}"
MAX_STEPS="${WAVCAPS_MAX_STEPS:-}"
SAVE_STRATEGY="${WAVCAPS_SAVE_STRATEGY:-epoch}"
SAVE_STEPS="${WAVCAPS_SAVE_STEPS:-20}"
SAVE_TOTAL_LIMIT="${WAVCAPS_SAVE_TOTAL_LIMIT:-2}"
LOGGING_STEPS="${WAVCAPS_LOGGING_STEPS:-10}"
LEARNING_RATE="${WAVCAPS_LEARNING_RATE:-1e-4}"
ALIGNER_LR="${WAVCAPS_ALIGNER_LR:-$LEARNING_RATE}"

if [ ! -s "$TRAIN_MANIFEST" ] || [ ! -s "$TRAIN_STATS" ]; then
  echo "WavCaps manifest or stats file is missing: manifest=$TRAIN_MANIFEST stats=$TRAIN_STATS" >&2
  exit 1
fi
if [ ! -d "$INIT_CHECKPOINT" ]; then
  echo "Initial AudioCaps checkpoint does not exist: $INIT_CHECKPOINT" >&2
  exit 1
fi
if [ ! -f "$INIT_CHECKPOINT/adapter_model.safetensors" ]; then
  echo "Initial checkpoint lacks adapter_model.safetensors: $INIT_CHECKPOINT" >&2
  exit 1
fi
if [ ! -f "$INIT_CHECKPOINT/vit.safetensors" ]; then
  echo "Initial checkpoint lacks vit.safetensors: $INIT_CHECKPOINT" >&2
  exit 1
fi
if [ "$SAVE_STRATEGY" != "epoch" ] && [ "$SAVE_STRATEGY" != "steps" ]; then
  echo "WAVCAPS_SAVE_STRATEGY must be epoch or steps, got: $SAVE_STRATEGY" >&2
  exit 1
fi

python - "$TRAIN_STATS" <<'PY'
import json
import sys
from dataclasses import fields

with open(sys.argv[1], encoding='utf-8') as f:
    stats = json.load(f)
if stats.get('dataset') != 'wavcaps' or stats.get('subset') != 'AudioSet_SL':
    raise SystemExit(f"Unexpected WavCaps stats: dataset={stats.get('dataset')!r} subset={stats.get('subset')!r}")
if not isinstance(stats.get('record_count'), int) or stats['record_count'] <= 0:
    raise SystemExit(f"Unexpected WavCaps record_count: {stats.get('record_count')!r}")
if stats.get('audio_path_verification') != 'passed' or stats.get('metadata_pairing_verification') != 'passed':
    raise SystemExit('WavCaps manifest verification is not marked passed')

from swift.arguments.sft_args import SftArguments
available = {field.name for field in fields(SftArguments)}
required = {'resume_from_checkpoint', 'resume_only_model', 'ignore_data_skip', 'num_train_epochs', 'save_strategy'}
missing = sorted(required - available)
if missing:
    raise SystemExit(f'Installed Swift lacks required WavCaps warm-start fields: {missing}')
PY

# Swift resume_only_model loads LoRA via adapters. The plugin restores vit.safetensors
# before PEFT wrapping so compressor/projector state is also warm-started.
export HUGINN_AUDIO_INIT_ALIGNER_CHECKPOINT="$INIT_CHECKPOINT"
mkdir -p "$OUTPUT_DIR" "$LOGGING_DIR"

echo "========== WAVCAPS AUDIOSET HUGINN AUDIO SWIFT TRAIN 5090 =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "mode=lora_llm frozen_audio_encoder trainable_aligner"
echo "dataset=$TRAIN_MANIFEST"
echo "output_dir=$OUTPUT_DIR"
echo "init_checkpoint=$INIT_CHECKPOINT"
echo "warm_start=resume_only_model+ignore_data_skip+plugin_aligner_restore"
echo "num_train_epochs=$NUM_TRAIN_EPOCHS"
echo "max_steps=${MAX_STEPS:-<unset>}"
echo "per_device_train_batch_size=8"
echo "gradient_accumulation_steps=4"
echo "effective_batch_size=32"
echo "dataset_shuffle=true"
echo "train_dataloader_shuffle=true"
echo "learning_rate=$LEARNING_RATE"
echo "aligner_lr=$ALIGNER_LR"
echo "save_strategy=$SAVE_STRATEGY"
echo "save_steps=$SAVE_STEPS"
echo "save_total_limit=$SAVE_TOTAL_LIMIT"
echo "report_to=tensorboard"
echo "save_only_model=false"

TRAIN_LENGTH_ARGS=(--num_train_epochs "$NUM_TRAIN_EPOCHS")
if [ -n "$MAX_STEPS" ]; then
  TRAIN_LENGTH_ARGS=(--max_steps "$MAX_STEPS")
fi
SAVE_ARGS=(--save_strategy "$SAVE_STRATEGY" --save_total_limit "$SAVE_TOTAL_LIMIT")
if [ "$SAVE_STRATEGY" = "steps" ]; then
  SAVE_ARGS+=(--save_steps "$SAVE_STEPS")
fi

TRAIN_PID=""
MONITOR_PID=""

print_resource_snapshot() {
  echo "========== WAVCAPS TRAIN RESOURCE SNAPSHOT =========="
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
  echo "========== WAVCAPS TRAIN EXIT =========="
  echo "exit_status=$status"
  echo "exit_time=$(date '+%Y-%m-%d %H:%M:%S')"
  exit "$status"
}

trap on_exit EXIT

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
  --learning_rate "$LEARNING_RATE" \
  --aligner_lr "$ALIGNER_LR" \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --resume_from_checkpoint "$INIT_CHECKPOINT" \
  --resume_only_model true \
  --ignore_data_skip true \
  --load_args false \
  "${TRAIN_LENGTH_ARGS[@]}" \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --logging_steps "$LOGGING_STEPS" \
  "${SAVE_ARGS[@]}" \
  --dataloader_num_workers 0 \
  --dataloader_pin_memory false \
  --dataset_num_proc 1 \
  --save_only_model false \
  --report_to tensorboard \
  --bf16 true &
TRAIN_PID=$!
resource_monitor &
MONITOR_PID=$!

set +e
wait "$TRAIN_PID"
TRAIN_STATUS=$?
set -e
exit "$TRAIN_STATUS"
