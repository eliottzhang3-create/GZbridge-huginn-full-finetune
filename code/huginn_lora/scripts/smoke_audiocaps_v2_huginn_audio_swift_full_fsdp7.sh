#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export NPROC_PER_NODE=6
export OMP_NUM_THREADS=4
export HUGINN_AUDIO_FSDP2_NONPERSISTENT_ROPE=1

# This smoke validates FSDP2 sharding, forward, backward, and optimizer setup.
# Huginn's recurrent scalar-index path needs a separate activation-checkpointing
# compatibility pass, so do not enable FSDP native activation checkpointing here.
FSDP_CONFIG='{"activation_checkpointing":false,"auto_wrap_policy":"TRANSFORMER_BASED_WRAP","cpu_ram_efficient_loading":true,"fsdp_version":2,"reshard_after_forward":true,"state_dict_type":"SHARDED_STATE_DICT"}'

TRAIN_MANIFEST="${AUDIOCAPS_FULL_TRAIN_MANIFEST:-$REPO_ROOT/data/audio_swift/audiocaps_v2/audiocaps_v2_train_swift.jsonl}"
TRAIN_STATS="$TRAIN_MANIFEST.stats.json"
OUTPUT_DIR="${AUDIOCAPS_FULL_FSDP_OUTPUT_DIR:-outputs/huginn_audio_audiocaps_v2_full_fsdp6_smoke1}"
LOGGING_DIR="${AUDIOCAPS_FULL_FSDP_LOGGING_DIR:-$OUTPUT_DIR/tensorboard}"

if [ ! -s "$TRAIN_MANIFEST" ] || [ ! -s "$TRAIN_STATS" ]; then
  echo "AudioCaps manifest or stats is missing: manifest=$TRAIN_MANIFEST stats=$TRAIN_STATS" >&2
  exit 1
fi

python - "$TRAIN_STATS" <<'PY'
import json
import sys
with open(sys.argv[1], encoding='utf-8') as f:
    stats = json.load(f)
if stats.get('dataset') != 'audiocaps_v2' or stats.get('split') != 'train':
    raise SystemExit(f"Unexpected AudioCaps stats: dataset={stats.get('dataset')!r} split={stats.get('split')!r}")
if stats.get('audio_path_verification') != 'passed' or stats.get('wav_readability_verification') != 'passed':
    raise SystemExit('AudioCaps manifest verification is not marked passed')
PY

mkdir -p "$OUTPUT_DIR" "$LOGGING_DIR"
FSDP_CONFIG_PATH="$OUTPUT_DIR/fsdp2_smoke_no_activation_checkpointing.json"
printf '%s\n' "$FSDP_CONFIG" > "$FSDP_CONFIG_PATH"
echo "========== HUGINN AUDIOCAPS V2 SWIFT FULL FSDP6 SMOKE =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "launch_mode=swift_cli_internal_torchrun"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "dataset=$TRAIN_MANIFEST"
echo "output_dir=$OUTPUT_DIR"
echo "tuner_type=full"
echo "freeze_llm=false freeze_vit=true freeze_aligner=false"
echo "audio_encoder_policy=frozen"
echo "fsdp=fsdp2"
echo "fsdp2_rope_buffer=nonpersistent"
echo "fsdp_activation_checkpointing=false"
echo "fsdp_config_path=$FSDP_CONFIG_PATH"
echo "per_device_train_batch_size=1"
echo "gradient_accumulation_steps=4"
echo "global_effective_batch_size=24"
echo "max_steps=1"
echo "learning_rate=1e-5 aligner_lr=1e-4"
echo "gradient_checkpointing=false"

TRAIN_PID=""
MONITOR_PID=""

resource_monitor() {
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    echo "========== AUDIOCAPS FULL FSDP6 RESOURCE SNAPSHOT =========="
    echo "snapshot_time=$(date '+%Y-%m-%d %H:%M:%S')"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
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
  echo "========== HUGINN AUDIOCAPS V2 SWIFT FULL FSDP6 SMOKE EXIT =========="
  echo "exit_status=$status"
  echo "exit_time=$(date '+%Y-%m-%d %H:%M:%S')"
  exit "$status"
}
trap on_exit EXIT

CMD=(swift sft)
CMD+=(--model "$REPO_ROOT/models/huginn-audio-whisper-v1")
CMD+=(--model_type huginn_audio_raven --template huginn_audio_text)
CMD+=(--external_plugins "$REPO_ROOT/code/huginn_lora/plugins/huginn_audio_swift.py")
CMD+=(--dataset "$TRAIN_MANIFEST")
CMD+=(--dataset_shuffle true --train_dataloader_shuffle true --sortish_sampler false --group_by_length false)
CMD+=(--max_length 192 --output_dir "$OUTPUT_DIR" --logging_dir "$LOGGING_DIR")
CMD+=(--tuner_type full --freeze_llm false --freeze_vit true --freeze_aligner false --fsdp fsdp2 --fsdp_config "$FSDP_CONFIG_PATH")
CMD+=(--learning_rate 1e-5 --aligner_lr 1e-4 --gradient_checkpointing false)
CMD+=(--max_steps 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 4)
CMD+=(--logging_steps 1 --save_strategy no --dataloader_num_workers 0 --dataloader_pin_memory false)
CMD+=(--dataset_num_proc 1 --save_only_model false --report_to none --bf16 true)
"${CMD[@]}" &
TRAIN_PID=$!
resource_monitor &
MONITOR_PID=$!

set +e
wait "$TRAIN_PID"
TRAIN_STATUS=$?
set -e
exit "$TRAIN_STATUS"
