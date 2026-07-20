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
export HUGINN_LOSATOK_TRAIN_CHAIN_AUDIT=1
export HUGINN_LOSATOK_FORCE_ALIGNER_TRAINABLE=1

SOURCE_MANIFEST="$REPO_ROOT/data/audio_swift/audiocaps_v2/audiocaps_v2_train_swift.jsonl"
RUNTIME_DIR="$(mktemp -d /tmp/huginn_losatok_checkpoint_resume.XXXXXX)"
SMOKE_MANIFEST="$RUNTIME_DIR/audiocaps_v2_32_records.jsonl"
PLUGIN_PATH="$REPO_ROOT/code/huginn_lora/plugins/huginn_losatok_swift.py"
MODEL_PATH="$REPO_ROOT/models/huginn-audio-losatok-v1"
RUN_ROOT="${LOSATOK_CHECKPOINT_TEST_ROOT:-outputs/huginn_losatok_audiocaps_v2_checkpoint_resume/run-$(date +%Y%m%d_%H%M%S)}"
SAVE_OUTPUT_DIR="$RUN_ROOT/save_phase"
RESUME_OUTPUT_DIR="$RUN_ROOT/resume_phase"
RECORD_COUNT=32
BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=4
EXPECTED_LORA_TENSORS=66
EXPECTED_ALIGNER_TENSORS=20
ACTIVE_PID=""
ACTIVE_STAGE=""
MONITOR_PID=""

print_resource_snapshot() {
  echo "========== LOSATOK CHECKPOINT RESOURCE SNAPSHOT =========="
  echo "snapshot_time=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "stage=${ACTIVE_STAGE:-none}"
  if [ -n "$ACTIVE_PID" ] && kill -0 "$ACTIVE_PID" 2>/dev/null; then
    ps -o pid,ppid,rss,vsz,%mem,etime,stat,cmd -p "$ACTIVE_PID" || true
  fi
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
  for cgroup_file in \
    /sys/fs/cgroup/memory.current \
    /sys/fs/cgroup/memory.max \
    /sys/fs/cgroup/memory.events \
    /sys/fs/cgroup/memory/memory.usage_in_bytes \
    /sys/fs/cgroup/memory/memory.limit_in_bytes \
    /sys/fs/cgroup/memory/memory.failcnt; do
    if [ -r "$cgroup_file" ]; then
      echo "[cgroup] $(basename "$cgroup_file")=$(tr '\n' ' ' < "$cgroup_file")"
    fi
  done
  df -h "$RUN_ROOT" 2>/dev/null || true
}

resource_monitor() {
  while [ -n "$ACTIVE_PID" ] && kill -0 "$ACTIVE_PID" 2>/dev/null; do
    print_resource_snapshot
    sleep 5
  done
}

stop_resource_monitor() {
  if [ -n "$MONITOR_PID" ] && kill -0 "$MONITOR_PID" 2>/dev/null; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
  MONITOR_PID=""
}

run_stage() {
  local stage=$1
  shift
  ACTIVE_STAGE=$stage
  echo "========== LOSATOK STAGE START =========="
  echo "stage=$ACTIVE_STAGE"
  "$@" &
  ACTIVE_PID=$!
  resource_monitor &
  MONITOR_PID=$!
  set +e
  wait "$ACTIVE_PID"
  local status=$?
  set -e
  stop_resource_monitor
  print_resource_snapshot
  ACTIVE_PID=""
  if [ "$status" -ne 0 ]; then
    echo "========== LOSATOK STAGE FAILED =========="
    echo "stage=$ACTIVE_STAGE exit_status=$status"
    return "$status"
  fi
  echo "========== LOSATOK STAGE PASSED =========="
  echo "stage=$ACTIVE_STAGE"
}

on_exit() {
  status=$?
  trap - EXIT
  stop_resource_monitor
  echo "========== HUGINN LOSATOK CHECKPOINT RESUME EXIT =========="
  echo "exit_status=$status"
  echo "exit_time=$(date '+%Y-%m-%d %H:%M:%S')"
  exit "$status"
}
trap on_exit EXIT

on_signal() {
  local signal_name=$1
  echo "========== LOSATOK CHECKPOINT SIGNAL =========="
  echo "signal=$signal_name stage=${ACTIVE_STAGE:-none}"
  print_resource_snapshot
  if [ -n "$ACTIVE_PID" ] && kill -0 "$ACTIVE_PID" 2>/dev/null; then
    kill -TERM "$ACTIVE_PID" 2>/dev/null || true
  fi
  exit 143
}
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT

find_checkpoint() {
  local root=$1
  local checkpoint_name=$2
  local found
  found="$(find "$root" -type d -name "$checkpoint_name" -print 2>/dev/null | sort | tail -n 1)"
  if [ -z "$found" ]; then
    echo "Expected $checkpoint_name was not created under $root" >&2
    exit 1
  fi
  printf '%s\n' "$found"
}

inspect_checkpoint() {
  local checkpoint=$1
  local report=$2
  python -u code/huginn_lora/scripts/inspect_swift_huginn_audio_checkpoints.py \
    --checkpoint "$checkpoint" \
    --output_report "$report" \
    --expected_lora_tensor_count "$EXPECTED_LORA_TENSORS" \
    --expected_aligner_tensor_count "$EXPECTED_ALIGNER_TENSORS" \
    --require_boundary_embeddings
}

echo "========== HUGINN LOSATOK CHECKPOINT RESUME VERIFICATION =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "run_root=$RUN_ROOT"
echo "source_manifest=$SOURCE_MANIFEST"
echo "record_count=$RECORD_COUNT batch_size=$BATCH_SIZE gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS"
echo "save_phase=max_steps_1 save_steps_1"
echo "resume_phase=checkpoint_1_to_max_steps_2"
echo "save_only_model=false"

python - <<'PY'
import torch
import torchaudio
print(f"[env] torch={torch.__version__} torchaudio={torchaudio.__version__} cuda={torch.version.cuda}")
if torch.__version__ != torchaudio.__version__:
    raise SystemExit("Torch and torchaudio versions must match for the LoSATok checkpoint test")
PY

python -u code/huginn_lora/scripts/smoke_huginn_losatok_swift.py \
  --source_manifest "$SOURCE_MANIFEST" \
  --output_manifest "$SMOKE_MANIFEST" \
  --record_count "$RECORD_COUNT"

echo "========== LOSATOK SAVE PHASE =========="
run_save_phase() {
swift sft \
  --model "$MODEL_PATH" \
  --model_type huginn_losatok_raven \
  --template huginn_losatok_text \
  --external_plugins "$PLUGIN_PATH" \
  --dataset "$SMOKE_MANIFEST" \
  --dataset_shuffle false \
  --train_dataloader_shuffle false \
  --sortish_sampler false \
  --group_by_length false \
  --max_length 192 \
  --output_dir "$SAVE_OUTPUT_DIR" \
  --tuner_type lora_llm \
  --freeze_aligner false \
  --learning_rate 1e-4 \
  --aligner_lr 1e-4 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --max_steps 1 \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --gradient_checkpointing true \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps 1 \
  --save_total_limit 2 \
  --dataloader_num_workers 0 \
  --dataloader_pin_memory false \
  --dataset_num_proc 1 \
  --save_only_model false \
  --report_to none \
  --bf16 true
}
run_stage save_phase run_save_phase

CHECKPOINT_1="$(find_checkpoint "$SAVE_OUTPUT_DIR" checkpoint-1)"
echo "========== LOSATOK CHECKPOINT-1 INSPECT =========="
echo "checkpoint_1=$CHECKPOINT_1"
inspect_checkpoint "$CHECKPOINT_1" "$RUN_ROOT/checkpoint-1.inspect.json"

echo "========== LOSATOK RESUME PHASE =========="
run_resume_phase() {
swift sft \
  --model "$MODEL_PATH" \
  --model_type huginn_losatok_raven \
  --template huginn_losatok_text \
  --external_plugins "$PLUGIN_PATH" \
  --dataset "$SMOKE_MANIFEST" \
  --dataset_shuffle false \
  --train_dataloader_shuffle false \
  --sortish_sampler false \
  --group_by_length false \
  --max_length 192 \
  --output_dir "$RESUME_OUTPUT_DIR" \
  --tuner_type lora_llm \
  --freeze_aligner false \
  --learning_rate 1e-4 \
  --aligner_lr 1e-4 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --max_steps 2 \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --gradient_checkpointing true \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps 1 \
  --save_total_limit 2 \
  --dataloader_num_workers 0 \
  --dataloader_pin_memory false \
  --dataset_num_proc 1 \
  --save_only_model false \
  --report_to none \
  --bf16 true \
  --resume_from_checkpoint "$CHECKPOINT_1"
}
run_stage resume_phase run_resume_phase

CHECKPOINT_2="$(find_checkpoint "$RESUME_OUTPUT_DIR" checkpoint-2)"
echo "========== LOSATOK CHECKPOINT-2 INSPECT =========="
echo "checkpoint_2=$CHECKPOINT_2"
inspect_checkpoint "$CHECKPOINT_2" "$RUN_ROOT/checkpoint-2.inspect.json"

echo "========== LOSATOK CHECKPOINT RESUME VERIFICATION PASSED =========="
echo "checkpoint_1=$CHECKPOINT_1"
echo "checkpoint_2=$CHECKPOINT_2"
