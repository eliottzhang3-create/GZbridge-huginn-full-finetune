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

TRAIN_MANIFEST="${CLOTHOAQA_TRAIN_MANIFEST:-$REPO_ROOT/data/audio_swift/clotho_aqa/clotho_aqa_train_swift.jsonl}"
TRAIN_STATS="$TRAIN_MANIFEST.stats.json"
INIT_CHECKPOINT="${CLOTHOAQA_INIT_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_losatok_audiocaps_v2_train_e3_b8ga4_5090/v1-20260720-162632/checkpoint-5604}"
PLUGIN_PATH="$REPO_ROOT/code/huginn_lora/plugins/huginn_losatok_swift.py"
MODEL_PATH="$REPO_ROOT/models/huginn-audio-losatok-v1"
LOSATOK_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/models/LoSATok
LOSATOK_CODE_DIR="$REPO_ROOT/code/huginn_lora/LosatokCode"
OUTPUT_DIR="${CLOTHOAQA_OUTPUT_DIR:-outputs/huginn_losatok_clothoaqa_e1_warmstart5604_b8ga4_5090}"
LOGGING_DIR="${CLOTHOAQA_LOGGING_DIR:-$OUTPUT_DIR/tensorboard}"
NUM_TRAIN_EPOCHS="${CLOTHOAQA_NUM_TRAIN_EPOCHS:-1}"
MAX_STEPS="${CLOTHOAQA_MAX_STEPS:-}"
SAVE_STRATEGY="${CLOTHOAQA_SAVE_STRATEGY:-epoch}"
SAVE_STEPS="${CLOTHOAQA_SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${CLOTHOAQA_SAVE_TOTAL_LIMIT:-1}"
LOGGING_STEPS="${CLOTHOAQA_LOGGING_STEPS:-10}"
REPORT_TO="${CLOTHOAQA_REPORT_TO:-tensorboard}"
LEARNING_RATE="${CLOTHOAQA_LEARNING_RATE:-1e-4}"
ALIGNER_LR="${CLOTHOAQA_ALIGNER_LR:-1e-4}"
BATCH_SIZE="${CLOTHOAQA_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${CLOTHOAQA_GRADIENT_ACCUMULATION_STEPS:-4}"

if [ ! -s "$TRAIN_MANIFEST" ] || [ ! -s "$TRAIN_STATS" ]; then
  echo "ClothoAQA manifest or stats are missing: manifest=$TRAIN_MANIFEST stats=$TRAIN_STATS" >&2
  exit 1
fi
for required_path in "$MODEL_PATH" "$PLUGIN_PATH" "$INIT_CHECKPOINT/adapter_model.safetensors" \
  "$INIT_CHECKPOINT/vit.safetensors" "$LOSATOK_ROOT/ckpts/losatok_kl1e-3.pth" \
  "$LOSATOK_ROOT/ckpts/semantic_encoder.pth" "$LOSATOK_ROOT/midashenglm" \
  "$LOSATOK_CODE_DIR/config/16k_16k_25Hz_losatok.yml"; do
  if [ ! -e "$required_path" ]; then
    echo "Required ClothoAQA warm-start path is missing: $required_path" >&2
    exit 1
  fi
done
if [ "$SAVE_STRATEGY" != epoch ] && [ "$SAVE_STRATEGY" != steps ] && [ "$SAVE_STRATEGY" != no ]; then
  echo "CLOTHOAQA_SAVE_STRATEGY must be epoch, steps, or no; got: $SAVE_STRATEGY" >&2
  exit 1
fi
if ! [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ && "$GRADIENT_ACCUMULATION_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "CLOTHOAQA_BATCH_SIZE and CLOTHOAQA_GRADIENT_ACCUMULATION_STEPS must be positive integers" >&2
  exit 1
fi

python - "$TRAIN_STATS" <<'PY'
import json
import sys
from dataclasses import fields

with open(sys.argv[1], encoding='utf-8') as handle:
    stats = json.load(handle)
if stats.get('dataset') != 'clotho_aqa':
    raise SystemExit(f"Unexpected dataset stats: {stats.get('dataset')!r}")
if not isinstance(stats.get('record_count'), int) or stats['record_count'] <= 0:
    raise SystemExit(f"Invalid ClothoAQA record_count: {stats.get('record_count')!r}")
if stats.get('audio_path_verification') != 'passed' or stats.get('aqa_prompt_verification') != 'passed':
    raise SystemExit('ClothoAQA manifest is not fully verified')

from swift.arguments.sft_args import SftArguments
available = {field.name for field in fields(SftArguments)}
required = {'adapters', 'load_args', 'num_train_epochs', 'save_strategy', 'save_total_limit'}
missing = sorted(required - available)
if missing:
    raise SystemExit(f'Installed Swift lacks required LoSATok warm-start arguments: {missing}')
PY

python -u code/huginn_lora/scripts/inspect_swift_huginn_audio_checkpoints.py \
  --checkpoint "$INIT_CHECKPOINT" \
  --expected_lora_tensor_count 66 \
  --expected_aligner_tensor_count 20 \
  --require_boundary_embeddings \
  --output_report "$OUTPUT_DIR/init_checkpoint_inspect.json"

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
SAVE_ARGS=(--save_strategy "$SAVE_STRATEGY")
if [ "$SAVE_STRATEGY" != no ]; then
  SAVE_ARGS+=(--save_total_limit "$SAVE_TOTAL_LIMIT")
fi
if [ "$SAVE_STRATEGY" = steps ]; then
  SAVE_ARGS+=(--save_steps "$SAVE_STEPS")
fi

# This is a new task, not a Trainer resume: weights continue, but optimizer,
# scheduler, RNG, global_step, and data position are freshly initialized.
export HUGINN_LOSATOK_INIT_ALIGNER_CHECKPOINT="$INIT_CHECKPOINT"
export HUGINN_LOSATOK_FORCE_ALIGNER_TRAINABLE=1
mkdir -p "$OUTPUT_DIR" "$LOGGING_DIR"

echo "========== CLOTHOAQA HUGINN LOSATOK LORA WARM-START =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "mode=lora_llm frozen_losatok_encoder aligner_trainable"
echo "dataset=$TRAIN_MANIFEST"
echo "output_dir=$OUTPUT_DIR"
echo "init_checkpoint=$INIT_CHECKPOINT"
echo "warm_start=adapters_only+strict_plugin_aligner_restore+fresh_trainer_state"
echo "num_train_epochs=$NUM_TRAIN_EPOCHS max_steps=${MAX_STEPS:-<unset>}"
echo "per_device_train_batch_size=$BATCH_SIZE gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS effective_batch_size=$((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "learning_rate=$LEARNING_RATE aligner_lr=$ALIGNER_LR"
echo "dataset_shuffle=true train_dataloader_shuffle=true"
echo "save_strategy=$SAVE_STRATEGY save_steps=$SAVE_STEPS save_total_limit=$SAVE_TOTAL_LIMIT save_only_model=false"
echo "logging_steps=$LOGGING_STEPS report_to=$REPORT_TO"
echo "resource_snapshot_interval_seconds=10"

TRAIN_PID=""
MONITOR_PID=""

print_resource_snapshot() {
  echo "========== CLOTHOAQA LOSATOK RESOURCE SNAPSHOT =========="
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
  echo "========== CLOTHOAQA HUGINN LOSATOK LORA WARM-START EXIT =========="
  echo "exit_status=$status"
  echo "exit_time=$(date '+%Y-%m-%d %H:%M:%S')"
  exit "$status"
}

on_signal() {
  local signal_name=$1
  echo "========== CLOTHOAQA LOSATOK TRAIN SIGNAL =========="
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
  --adapters "$INIT_CHECKPOINT" \
  --load_args false \
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
  --bf16 true &
TRAIN_PID=$!
resource_monitor &
MONITOR_PID=$!

set +e
wait "$TRAIN_PID"
TRAIN_STATUS=$?
set -e
exit "$TRAIN_STATUS"
