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
unset HUGINN_LOSATOK_DYNAMIC_AUDIO_TOKENS
unset HUGINN_AUDIO_FSDP2_NONPERSISTENT_ROPE
unset ACAVCAPS_WDS_MAX_TARS_PER_STAGE

INIT_CHECKPOINT="${LOSATOK_LEGACY_ACAV_WDS_INIT_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_losatok_audiocaps_v2_train_e3_b8ga4_5090/v1-20260720-162632/checkpoint-2802}"
export ACAVCAPS_WDS_MANIFEST="${ACAVCAPS_WDS_QUARTER_MANIFEST:-$REPO_ROOT/data/audio_swift/acavcaps_wds/acavcaps_wds_stage_schedule_quarter_ceil_seed20260723.json}"
export ACAVCAPS_WDS_BUFFER_SIZE="${ACAVCAPS_WDS_BUFFER_SIZE:-512}"

PLUGIN_PATH="$REPO_ROOT/code/huginn_lora/plugins/huginn_losatok_acavcaps_wds_swift.py"
MODEL_PATH="$REPO_ROOT/models/huginn-audio-losatok-v1"
WORLD_SIZE=1
BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=4
GLOBAL_EFFECTIVE_BATCH=$((WORLD_SIZE * BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
EXPECTED_LORA_TENSORS=66
EXPECTED_ALIGNER_TENSORS=20
OUTPUT_DIR="${LOSATOK_LEGACY_ACAV_WDS_QUARTER_OUTPUT_DIR:-outputs/huginn_losatok_acavcaps_wds_legacy_quarter_fixed32_warmstart2802_e1_b8ga4_5090/run-$(date +%Y%m%d_%H%M%S)}"
LOGGING_DIR="${LOSATOK_LEGACY_ACAV_WDS_QUARTER_LOGGING_DIR:-$OUTPUT_DIR/tensorboard}"
LOGGING_STEPS="${LOSATOK_LEGACY_ACAV_WDS_QUARTER_LOGGING_STEPS:-10}"
REPORT_TO="${LOSATOK_LEGACY_ACAV_WDS_QUARTER_REPORT_TO:-tensorboard}"
TRAIN_PID=""
MONITOR_PID=""

if [ "$ACAVCAPS_WDS_BUFFER_SIZE" != "512" ]; then
  echo "Formal quarter training requires ACAVCAPS_WDS_BUFFER_SIZE=512, got: $ACAVCAPS_WDS_BUFFER_SIZE" >&2
  exit 1
fi
if [ -e "$OUTPUT_DIR" ]; then
  echo "Formal output directory already exists; choose a fresh LOSATOK_LEGACY_ACAV_WDS_QUARTER_OUTPUT_DIR: $OUTPUT_DIR" >&2
  exit 1
fi
for required_path in \
  "$ACAVCAPS_WDS_MANIFEST" \
  "${ACAVCAPS_WDS_MANIFEST%.json}.stats.json" \
  "$PLUGIN_PATH" \
  "$MODEL_PATH" \
  "$INIT_CHECKPOINT/adapter_model.safetensors" \
  "$INIT_CHECKPOINT/vit.safetensors" \
  "/hpc_stor03/sjtu_home/jinwei.zhang/models/LoSATok/ckpts/losatok_kl1e-3.pth" \
  "/hpc_stor03/sjtu_home/jinwei.zhang/models/LoSATok/ckpts/semantic_encoder.pth"; do
  if [ ! -e "$required_path" ]; then
    echo "Required formal-quarter path is missing: $required_path" >&2
    exit 1
  fi
done

echo "========== ACAVCAPS LEGACY FIXED32 QUARTER FORMAL PREFLIGHT =========="
python -u code/huginn_lora/scripts/inspect_acavcaps_wds_quarter_manifest.py \
  --manifest "$ACAVCAPS_WDS_MANIFEST" \
  --world_size "$WORLD_SIZE" \
  --per_device_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"

read -r TOTAL_SAMPLES MAX_STEPS < <(python - "$ACAVCAPS_WDS_MANIFEST" "$GLOBAL_EFFECTIVE_BATCH" <<'PY'
import json
import math
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
global_batch = int(sys.argv[2])
stats_path = manifest_path.with_suffix('.stats.json')
stats = json.loads(stats_path.read_text(encoding='utf-8'))
total_samples = stats.get('sample_count')
if not isinstance(total_samples, int) or total_samples <= 0:
    raise SystemExit(f'Invalid quarter sample_count in {stats_path}: {total_samples!r}')
print(total_samples, math.ceil(total_samples / global_batch))
PY
)
if [ -z "${TOTAL_SAMPLES:-}" ] || [ -z "${MAX_STEPS:-}" ]; then
  echo "Unable to derive total samples/max steps from quarter manifest stats" >&2
  exit 1
fi

# Select the most frequent safe save cadence among 4, 3, and 2 equal partitions.
# The chosen interval divides MAX_STEPS exactly, so the final checkpoint is always
# checkpoint-MAX_STEPS.  Keeping the last two checkpoints provides a recovery point.
SAVE_PARTITIONS=1
for candidate in 4 3 2; do
  if (( MAX_STEPS % candidate == 0 )); then
    SAVE_PARTITIONS=$candidate
    break
  fi
done
SAVE_STEPS=$((MAX_STEPS / SAVE_PARTITIONS))
SAVE_TOTAL_LIMIT=2
SAVE_CHECKPOINT_NAME="checkpoint-$MAX_STEPS"

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

print_resource_snapshot() {
  echo "========== ACAVCAPS LEGACY QUARTER RESOURCE SNAPSHOT =========="
  echo "snapshot_time=$(date '+%Y-%m-%d %H:%M:%S')"
  if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    ps -o pid,ppid,rss,vsz,%mem,etime,stat,cmd -p "$TRAIN_PID" || true
  fi
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
  for cgroup_file in \
    /sys/fs/cgroup/memory.current \
    /sys/fs/cgroup/memory.max \
    /sys/fs/cgroup/memory.events \
    /sys/fs/cgroup/memory/memory.usage_in_bytes \
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
  echo "========== ACAVCAPS LEGACY QUARTER FORMAL TRAIN EXIT =========="
  echo "exit_status=$status"
  echo "exit_time=$(date '+%Y-%m-%d %H:%M:%S')"
  exit "$status"
}

on_signal() {
  local signal_name=$1
  echo "========== ACAVCAPS LEGACY QUARTER FORMAL TRAIN SIGNAL =========="
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

mkdir -p "$OUTPUT_DIR" "$LOGGING_DIR"
echo "========== ACAVCAPS LEGACY FIXED32 LOSATOK QUARTER FORMAL TRAIN =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "mode=legacy_fixed32 dynamic_audio_tokens=disabled fsdp=disabled"
echo "init_checkpoint=$INIT_CHECKPOINT"
echo "manifest=$ACAVCAPS_WDS_MANIFEST"
echo "buffer_size=$ACAVCAPS_WDS_BUFFER_SIZE max_tars_per_stage=all"
echo "stage_schedule=stage1(00A,0M0,S00)->stage2(S0A,SM0,0MA)->stage3(SMA)"
echo "data_order=source_stage_global_tar_shuffle_preserved_then_per_tar_webdataset_buffer_shuffle"
echo "streaming=true dataset_shuffle=false train_dataloader_shuffle=false"
echo "total_samples=$TOTAL_SAMPLES max_steps=$MAX_STEPS complete_dataset_passes=1"
echo "per_device_train_batch_size=$BATCH_SIZE gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS global_effective_batch=$GLOBAL_EFFECTIVE_BATCH"
echo "training_policy=frozen_losatok_encoder+trainable_aligner_including_audio_bos_eos+trainable_huginn_lora"
echo "checkpoint_contract=adapter_model_66_lora+vit_20_aligner+audio_bos_eos"
echo "save_strategy=steps save_partitions=$SAVE_PARTITIONS save_steps=$SAVE_STEPS save_total_limit=$SAVE_TOTAL_LIMIT save_only_model=false"
echo "logging_steps=$LOGGING_STEPS report_to=$REPORT_TO"

python - <<'PY'
import torch
import torchaudio
print(f"[env] torch={torch.__version__} torchaudio={torchaudio.__version__} cuda={torch.version.cuda}")
if torch.__version__ != torchaudio.__version__:
    raise SystemExit('Torch and torchaudio versions must match for legacy LoSATok training')
PY

echo "========== INPUT CHECKPOINT INSPECT =========="
inspect_checkpoint "$INIT_CHECKPOINT" "$OUTPUT_DIR/input_checkpoint.inspect.json"

export HUGINN_LOSATOK_INIT_ALIGNER_CHECKPOINT="$INIT_CHECKPOINT"
swift sft \
  --model "$MODEL_PATH" \
  --model_type huginn_losatok_raven \
  --template huginn_losatok_text \
  --external_plugins "$PLUGIN_PATH" \
  --dataset "$ACAVCAPS_WDS_MANIFEST" \
  --streaming true \
  --dataset_shuffle false \
  --train_dataloader_shuffle false \
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
  --adapters "$INIT_CHECKPOINT" \
  --load_args false \
  --max_steps "$MAX_STEPS" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --gradient_checkpointing true \
  --logging_steps "$LOGGING_STEPS" \
  --save_strategy steps \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
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
if [ "$TRAIN_STATUS" -ne 0 ]; then
  exit "$TRAIN_STATUS"
fi

mapfile -t CHECKPOINT_MATCHES < <(find "$OUTPUT_DIR" -type d -name "$SAVE_CHECKPOINT_NAME" -print | sort)
if [ "${#CHECKPOINT_MATCHES[@]}" -ne 1 ]; then
  echo "Expected exactly one final $SAVE_CHECKPOINT_NAME below $OUTPUT_DIR; found ${#CHECKPOINT_MATCHES[@]}" >&2
  printf '  %s\n' "${CHECKPOINT_MATCHES[@]:-<none>}" >&2
  exit 1
fi
FINAL_CHECKPOINT="${CHECKPOINT_MATCHES[0]}"

python - "$FINAL_CHECKPOINT/trainer_state.json" "$MAX_STEPS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_step = int(sys.argv[2])
if not path.is_file():
    raise SystemExit(f'Missing trainer state: {path}')
state = json.loads(path.read_text(encoding='utf-8'))
if state.get('global_step') != expected_step:
    raise SystemExit(f"Final global step mismatch: expected={expected_step} actual={state.get('global_step')}")
print(f"[checkpoint] final_global_step={state['global_step']}")
PY

echo "========== FINAL CHECKPOINT INSPECT =========="
echo "final_checkpoint=$FINAL_CHECKPOINT"
inspect_checkpoint "$FINAL_CHECKPOINT" "$OUTPUT_DIR/final_checkpoint.inspect.json"

echo "========== ACAVCAPS LEGACY FIXED32 LOSATOK QUARTER FORMAL TRAIN PASSED =========="
echo "final_checkpoint=$FINAL_CHECKPOINT"
echo "final_checkpoint_inspect=$OUTPUT_DIR/final_checkpoint.inspect.json"
