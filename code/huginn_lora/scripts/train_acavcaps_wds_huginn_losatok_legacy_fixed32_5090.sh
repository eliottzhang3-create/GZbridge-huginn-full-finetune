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

INIT_CHECKPOINT="${LOSATOK_LEGACY_ACAV_WDS_INIT_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_losatok_audiocaps_v2_train_e3_b8ga4_5090/v1-20260720-162632/checkpoint-2802}"
export ACAVCAPS_WDS_MANIFEST="${ACAVCAPS_WDS_MANIFEST:-/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/data/audio_swift/acavcaps_wds/acavcaps_wds_stage_schedule_full_seed20260723.json}"
export ACAVCAPS_WDS_BUFFER_SIZE="${ACAVCAPS_WDS_BUFFER_SIZE:-512}"
unset ACAVCAPS_WDS_MAX_TARS_PER_STAGE

PLUGIN_PATH="$REPO_ROOT/code/huginn_lora/plugins/huginn_losatok_acavcaps_wds_swift.py"
MODEL_PATH="$REPO_ROOT/models/huginn-audio-losatok-v1"
TOTAL_SAMPLES=4664169
WORLD_SIZE=1
BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=4
GLOBAL_EFFECTIVE_BATCH=$((WORLD_SIZE * BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
MAX_STEPS=145756
EXPECTED_LORA_TENSORS=66
EXPECTED_ALIGNER_TENSORS=20
OUTPUT_DIR="${LOSATOK_LEGACY_ACAV_WDS_OUTPUT_DIR:-outputs/huginn_losatok_acavcaps_wds_legacy_fixed32_warmstart2802_e1_b8ga4_5090/run-$(date +%Y%m%d_%H%M%S)}"
LOGGING_DIR="${LOSATOK_LEGACY_ACAV_WDS_LOGGING_DIR:-$OUTPUT_DIR/tensorboard}"
LOGGING_STEPS="${LOSATOK_LEGACY_ACAV_WDS_LOGGING_STEPS:-10}"
REPORT_TO="${LOSATOK_LEGACY_ACAV_WDS_REPORT_TO:-tensorboard}"
SAVE_CHECKPOINT_NAME="checkpoint-$MAX_STEPS"
TRAIN_PID=""
MONITOR_PID=""

if [ -e "$OUTPUT_DIR" ]; then
  echo "Formal output directory already exists; choose a fresh LOSATOK_LEGACY_ACAV_WDS_OUTPUT_DIR: $OUTPUT_DIR" >&2
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
    echo "Required legacy ACAVCAPS formal-training path is missing: $required_path" >&2
    exit 1
  fi
done

python - "$ACAVCAPS_WDS_MANIFEST" "$TOTAL_SAMPLES" "$GLOBAL_EFFECTIVE_BATCH" "$MAX_STEPS" <<'PY'
import json
import math
import os
import sys
from dataclasses import fields
from pathlib import Path

manifest_path = Path(sys.argv[1])
expected_total = int(sys.argv[2])
global_batch = int(sys.argv[3])
expected_steps = int(sys.argv[4])
stats_path = manifest_path.with_suffix('.stats.json')
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
stats = json.loads(stats_path.read_text(encoding='utf-8'))
expected_stage_order = ('stage1', 'stage2', 'stage3')
expected_stage_categories = {
    'stage1': ('00A', '0M0', 'S00'),
    'stage2': ('S0A', 'SM0', '0MA'),
    'stage3': ('SMA',),
}
expected_stage_tars = {'stage1': 651, 'stage2': 398, 'stage3': 22}
public_root = '/hpc_stor03/public/shared/data/raa/ACAVCAPS'

if manifest.get('scan_mode') != 'full' or stats.get('scan_mode') != 'full':
    raise SystemExit('Full ACAVCAPS manifest/stats are required')
if manifest.get('dataset_root') != public_root or stats.get('dataset_root') != public_root:
    raise SystemExit('Manifest/stats public dataset root mismatch')
if manifest.get('public_root_mutation') != 'forbidden' or stats.get('public_root_mutation') != 'forbidden':
    raise SystemExit('Public-root read-only policy is missing')
if stats.get('all_pairs_valid') is not True:
    raise SystemExit(f"ACAVCAPS pair validation is not passed: {stats.get('all_pairs_valid')!r}")
stages = manifest.get('stages')
if not isinstance(stages, list) or tuple(stage.get('name') for stage in stages) != expected_stage_order:
    raise SystemExit(f'Unexpected stage order: {[stage.get("name") for stage in stages or []]!r}')
total = 0
stage_counts = {}
for stage in stages:
    name = stage['name']
    if tuple(stage.get('categories', [])) != expected_stage_categories[name]:
        raise SystemExit(f'{name} categories mismatch: {stage.get("categories")!r}')
    tars = stage.get('tars')
    if not isinstance(tars, list) or len(tars) != expected_stage_tars[name]:
        raise SystemExit(f'{name} tar count mismatch: {len(tars or [])}')
    count = 0
    for entry in tars:
        if not isinstance(entry.get('json_count'), int) or entry['json_count'] != entry.get('flac_count'):
            raise SystemExit(f'{name} contains invalid JSON/FLAC counts: {entry!r}')
        count += entry['json_count']
    if stage.get('sample_count') != count:
        raise SystemExit(f'{name} sample count mismatch: manifest={stage.get("sample_count")} computed={count}')
    stage_counts[name] = count
    total += count
if total != expected_total or stats.get('sample_count') != expected_total:
    raise SystemExit(f'Total sample count mismatch: manifest={total} stats={stats.get("sample_count")} expected={expected_total}')
if stats.get('stage_sample_counts') != stage_counts:
    raise SystemExit(f'Stage sample count mismatch: {stats.get("stage_sample_counts")!r} != {stage_counts!r}')
if os.environ.get('ACAVCAPS_WDS_MAX_TARS_PER_STAGE', '').strip():
    raise SystemExit('Formal training must not set ACAVCAPS_WDS_MAX_TARS_PER_STAGE')
if os.environ.get('HUGINN_LOSATOK_DYNAMIC_AUDIO_TOKENS', '').strip():
    raise SystemExit('Legacy fixed-32 training must not enable HUGINN_LOSATOK_DYNAMIC_AUDIO_TOKENS')
actual_steps = math.ceil(total / global_batch)
if actual_steps != expected_steps:
    raise SystemExit(f'Max-step accounting mismatch: computed={actual_steps} expected={expected_steps}')

from swift.arguments.sft_args import SftArguments
available = {field.name for field in fields(SftArguments)}
required = {'adapters', 'load_args', 'max_steps', 'save_strategy', 'save_steps', 'save_total_limit'}
missing = sorted(required - available)
if missing:
    raise SystemExit(f'Installed Swift lacks required formal-training arguments: {missing}')
print(f'[preflight] total_samples={total} stage_sample_counts={stage_counts}')
print(f'[preflight] global_effective_batch={global_batch} max_steps={actual_steps}')
PY

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
  echo "========== ACAVCAPS LEGACY LOSATOK RESOURCE SNAPSHOT =========="
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
  echo "========== ACAVCAPS LEGACY LOSATOK FORMAL TRAIN EXIT =========="
  echo "exit_status=$status"
  echo "exit_time=$(date '+%Y-%m-%d %H:%M:%S')"
  exit "$status"
}

on_signal() {
  local signal_name=$1
  echo "========== ACAVCAPS LEGACY LOSATOK FORMAL TRAIN SIGNAL =========="
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
echo "========== ACAVCAPS LEGACY FIXED32 LOSATOK FORMAL TRAIN =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "mode=legacy_fixed32 dynamic_audio_tokens=disabled fsdp=disabled"
echo "init_checkpoint=$INIT_CHECKPOINT"
echo "manifest=$ACAVCAPS_WDS_MANIFEST"
echo "buffer_size=$ACAVCAPS_WDS_BUFFER_SIZE max_tars_per_stage=all"
echo "stage_schedule=stage1(00A,0M0,S00)->stage2(S0A,SM0,0MA)->stage3(SMA)"
echo "data_order=manifest_tar_order_then_per_tar_webdataset_buffer_shuffle"
echo "streaming=true dataset_shuffle=false train_dataloader_shuffle=false"
echo "total_samples=$TOTAL_SAMPLES max_steps=$MAX_STEPS complete_dataset_passes=1"
echo "per_device_train_batch_size=$BATCH_SIZE gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS global_effective_batch=$GLOBAL_EFFECTIVE_BATCH"
echo "training_policy=frozen_losatok_encoder+trainable_aligner+trainable_huginn_lora"
echo "checkpoint_contract=adapter_model_66_lora+vit_20_aligner+audio_bos_eos"
echo "save_strategy=steps save_steps=$MAX_STEPS save_total_limit=1 save_only_model=false"
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
  --save_steps "$MAX_STEPS" \
  --save_total_limit 1 \
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

echo "========== ACAVCAPS LEGACY FIXED32 LOSATOK FORMAL TRAIN PASSED =========="
echo "final_checkpoint=$FINAL_CHECKPOINT"
echo "final_checkpoint_inspect=$OUTPUT_DIR/final_checkpoint.inspect.json"
