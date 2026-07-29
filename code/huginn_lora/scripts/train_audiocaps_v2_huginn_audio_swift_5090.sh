#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${HUGINN_AUDIO_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE=4
export OMP_NUM_THREADS=4
export HUGINN_AUDIO_FSDP2_NONPERSISTENT_ROPE=1
export HUGINN_AUDIO_TRAIN_CHAIN_AUDIT=1

TRAIN_MANIFEST="${AUDIOCAPS_TRAIN_MANIFEST:-$REPO_ROOT/data/audio_swift/audiocaps_v2/audiocaps_v2_train_swift.jsonl}"
TRAIN_STATS="$TRAIN_MANIFEST.stats.json"
PLUGIN_PATH="$REPO_ROOT/code/huginn_lora/plugins/huginn_audio_swift.py"
MODEL_PATH="$REPO_ROOT/models/huginn-audio-whisper-v1"
OUTPUT_DIR="${AUDIOCAPS_OUTPUT_DIR:-outputs/huginn_audio_audiocaps_v2_dynamic90s_lora_fsdp4_fresh}"
LOGGING_DIR="${AUDIOCAPS_LOGGING_DIR:-$OUTPUT_DIR/tensorboard}"
NUM_TRAIN_EPOCHS="${AUDIOCAPS_NUM_TRAIN_EPOCHS:-5}"
MAX_STEPS="${AUDIOCAPS_MAX_STEPS:-}"
SAVE_STRATEGY="${AUDIOCAPS_SAVE_STRATEGY:-epoch}"
SAVE_STEPS="${AUDIOCAPS_SAVE_STEPS:-20}"
SAVE_TOTAL_LIMIT="${AUDIOCAPS_SAVE_TOTAL_LIMIT:-5}"
LOGGING_STEPS="${AUDIOCAPS_LOGGING_STEPS:-10}"
REPORT_TO="${AUDIOCAPS_REPORT_TO:-tensorboard}"

WORLD_SIZE=4
MICRO_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE=1e-4
ALIGNER_LR=1e-4
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
# This is the same FSDP2 configuration used by the previously verified
# Whisper full-parameter route. LoRA changes trainability, not the sharding
# topology. Activation recomputation stays disabled for Huginn's recurrent
# forward path.
FSDP_CONFIG='{"fsdp":"full_shard auto_wrap","fsdp_config":{"activation_checkpointing":false,"auto_wrap_policy":"TRANSFORMER_BASED_WRAP","cpu_ram_efficient_loading":true,"fsdp_version":2,"reshard_after_forward":true,"state_dict_type":"SHARDED_STATE_DICT"}}'

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
source_rows = stats.get("source_csv_row_count")
excluded_rows = stats.get("excluded_row_count")
if not isinstance(source_rows, int) or not isinstance(excluded_rows, int):
    raise SystemExit("AudioCaps stats are missing source_csv_row_count or excluded_row_count")
if stats.get("limit_records") is None and source_rows != stats["record_count"] + excluded_rows:
    raise SystemExit(
        f"AudioCaps stats accounting mismatch: source={source_rows} "
        f"record_count={stats['record_count']} excluded={excluded_rows}")

from swift.arguments.sft_args import SftArguments
available_fields = {field.name for field in fields(SftArguments)}
required_fields = {"num_train_epochs", "save_strategy", "save_total_limit"}
missing_fields = sorted(required_fields - available_fields)
if missing_fields:
    raise SystemExit(f"Installed Swift SftArguments lacks required epoch-checkpoint fields: {missing_fields}")
PY

mkdir -p "$OUTPUT_DIR" "$LOGGING_DIR"
if find "$OUTPUT_DIR" -type d -name 'checkpoint-*' -print -quit | grep -q .; then
  echo "Fresh-training output directory already contains a checkpoint; choose a new AUDIOCAPS_OUTPUT_DIR: $OUTPUT_DIR" >&2
  exit 1
fi
FSDP_CONFIG_PATH="$OUTPUT_DIR/fsdp2_lora_no_activation.json"
printf '%s\n' "$FSDP_CONFIG" > "$FSDP_CONFIG_PATH"
echo "========== AUDIOCAPS V2 HUGINN AUDIO SWIFT TRAIN 5090 =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "mode=lora_llm generator_frozen_audio_encoder aligner_trainable"
echo "dataset=$TRAIN_MANIFEST"
echo "output_dir=$OUTPUT_DIR"
echo "logging_dir=$LOGGING_DIR"
echo "num_train_epochs=$NUM_TRAIN_EPOCHS"
echo "max_steps=${MAX_STEPS:-<unset>}"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "nproc_per_node=$NPROC_PER_NODE"
echo "world_size=$WORLD_SIZE"
echo "per_device_train_batch_size=$MICRO_BATCH_SIZE"
echo "gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS"
echo "effective_batch_size=$((WORLD_SIZE * MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "audio_dynamic_tokens=true audio_chunk_seconds=30 audio_max_seconds=90 audio_discard_seconds=120"
echo "audio_token_rate=120ms_per_token full_30s_tokens=250 max_audio_tokens=750"
echo "audio_batch_padding=per_batch_max_prefix attention_mask_zero labels_minus_100"
echo "audio_gt120_policy=discard_gate"
echo "fsdp=custom_fsdp2_json world_size=$WORLD_SIZE"
echo "whisper_audio_encoder=frozen"
echo "lora_rank=$LORA_RANK lora_alpha=$LORA_ALPHA lora_dropout=$LORA_DROPOUT"
echo "learning_rate=$LEARNING_RATE aligner_lr=$ALIGNER_LR"
echo "manifest_invalid_rows_prechecked=true"
echo "dataset_shuffle=true"
echo "train_dataloader_shuffle=true"
echo "save_strategy=$SAVE_STRATEGY"
echo "save_steps=$SAVE_STEPS"
echo "save_total_limit=$SAVE_TOTAL_LIMIT"
echo "logging_steps=$LOGGING_STEPS"
echo "report_to=$REPORT_TO"
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
  --freeze_vit true \
  --freeze_aligner false \
  --learning_rate "$LEARNING_RATE" \
  --aligner_lr "$ALIGNER_LR" \
  --lora_rank "$LORA_RANK" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --fsdp "$FSDP_CONFIG_PATH" \
  "${TRAIN_LENGTH_ARGS[@]}" \
  --per_device_train_batch_size "$MICRO_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --gradient_checkpointing false \
  --logging_steps "$LOGGING_STEPS" \
  "${SAVE_ARGS[@]}" \
  --dataloader_num_workers 0 \
  --dataloader_pin_memory false \
  --dataset_num_proc 1 \
  --save_only_model false \
  --report_to "$REPORT_TO" \
  --bf16 true &
TRAIN_PID=$!
resource_monitor &
MONITOR_PID=$!

set +e
wait "$TRAIN_PID"
TRAIN_STATUS=$?
set -e
exit "$TRAIN_STATUS"
