#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export NPROC_PER_NODE=7
export OMP_NUM_THREADS=4
export HUGINN_AUDIO_FSDP2_NONPERSISTENT_ROPE=1
export HUGINN_AUDIO_TRAIN_CHAIN_AUDIT=1

TRAIN_MANIFEST="${AUDIOCAPS_FSDP7_CROSSWORLD_TRAIN_MANIFEST:-$REPO_ROOT/data/audio_swift/audiocaps_v2/audiocaps_v2_train_swift.jsonl}"
TRAIN_STATS="$TRAIN_MANIFEST.stats.json"
RESUME_FROM_CHECKPOINT="${AUDIOCAPS_FSDP7_CROSSWORLD_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_audio_audiocaps_v2_full_fsdp8_e2_b1ga4/v0-20260717-084419/checkpoint-2802}"
EXPECTED_RESUME_STEP="${AUDIOCAPS_FSDP7_CROSSWORLD_EXPECTED_STEP:-2802}"
SMOKE_UPDATES="${AUDIOCAPS_FSDP7_CROSSWORLD_SMOKE_UPDATES:-3}"
RUN_TAG="${AUDIOCAPS_FSDP7_CROSSWORLD_RUN_TAG:-$(date '+%Y%m%d_%H%M%S')}"
OUTPUT_ROOT="${AUDIOCAPS_FSDP7_CROSSWORLD_OUTPUT_DIR:-outputs/huginn_audio_audiocaps_v2_full_fsdp7_crossworld2802_smoke3/run-$RUN_TAG}"
LOGGING_DIR="$OUTPUT_ROOT/tensorboard"
MIN_FREE_GB="${AUDIOCAPS_FSDP7_CROSSWORLD_MIN_FREE_GB:-200}"

WORLD_SIZE=7
MICRO_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=1e-5
ALIGNER_LR=1e-4
WARMUP_RATIO=0.05
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0
FSDP_CONFIG='{"fsdp":"full_shard auto_wrap","fsdp_config":{"activation_checkpointing":false,"auto_wrap_policy":"TRANSFORMER_BASED_WRAP","cpu_ram_efficient_loading":true,"fsdp_version":2,"reshard_after_forward":true,"state_dict_type":"SHARDED_STATE_DICT"}}'

if [ ! -s "$TRAIN_MANIFEST" ] || [ ! -s "$TRAIN_STATS" ]; then
  echo "AudioCaps manifest or stats is missing: manifest=$TRAIN_MANIFEST stats=$TRAIN_STATS" >&2
  exit 1
fi
if [ ! -d "$RESUME_FROM_CHECKPOINT/pytorch_model_fsdp_0" ] || [ ! -d "$RESUME_FROM_CHECKPOINT/optimizer_0" ]; then
  echo "Expected 8-rank FSDP model and optimizer shard directories under: $RESUME_FROM_CHECKPOINT" >&2
  exit 1
fi
if ! [[ "$SMOKE_UPDATES" =~ ^[1-9][0-9]*$ ]]; then
  echo "AUDIOCAPS_FSDP7_CROSSWORLD_SMOKE_UPDATES must be a positive integer, got: $SMOKE_UPDATES" >&2
  exit 1
fi

CALCULATED_VALUES="$(python - "$TRAIN_STATS" "$RESUME_FROM_CHECKPOINT/trainer_state.json" "$EXPECTED_RESUME_STEP" "$SMOKE_UPDATES" <<'PY'
import json
import sys

stats_path, trainer_state_path, expected_step, smoke_updates = sys.argv[1:]
expected_step = int(expected_step)
smoke_updates = int(smoke_updates)
with open(stats_path, encoding='utf-8') as handle:
    stats = json.load(handle)
with open(trainer_state_path, encoding='utf-8') as handle:
    trainer_state = json.load(handle)
if stats.get('dataset') != 'audiocaps_v2' or stats.get('split') != 'train':
    raise SystemExit(f"Unexpected AudioCaps stats: dataset={stats.get('dataset')!r} split={stats.get('split')!r}")
if stats.get('audio_path_verification') != 'passed' or stats.get('wav_readability_verification') != 'passed':
    raise SystemExit('AudioCaps manifest verification is not marked passed')
if trainer_state.get('global_step') != expected_step:
    raise SystemExit(
        f"Resume checkpoint global_step={trainer_state.get('global_step')!r}, expected {expected_step}"
    )
print(stats['record_count'], expected_step + smoke_updates)
PY
)"
read -r RECORD_COUNT TARGET_GLOBAL_STEP <<< "$CALCULATED_VALUES"

AVAILABLE_GB="$(df -BG "$REPO_ROOT" | awk 'NR==2 {gsub(/G/, "", $4); print $4}')"
if [ -z "$AVAILABLE_GB" ] || [ "$AVAILABLE_GB" -lt "$MIN_FREE_GB" ]; then
  echo "Insufficient free storage: available=${AVAILABLE_GB:-unknown}G required=${MIN_FREE_GB}G" >&2
  exit 1
fi
if [ -e "$OUTPUT_ROOT" ]; then
  echo "Smoke output root already exists; choose a new run tag or output root: $OUTPUT_ROOT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT" "$LOGGING_DIR"
FSDP_CONFIG_PATH="$OUTPUT_ROOT/fsdp2_crossworld_resume_no_activation.json"
printf '%s\n' "$FSDP_CONFIG" > "$FSDP_CONFIG_PATH"

echo "========== AUDIOCAPS V2 FSDP 8-TO-7 CROSS-WORLD RESUME SMOKE =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "resume_checkpoint=$RESUME_FROM_CHECKPOINT"
echo "resume_checkpoint_world_size=8 target_world_size=$WORLD_SIZE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES NPROC_PER_NODE=$NPROC_PER_NODE"
echo "dataset=$TRAIN_MANIFEST record_count=$RECORD_COUNT"
echo "output_root=$OUTPUT_ROOT"
echo "resume_global_step=$EXPECTED_RESUME_STEP smoke_updates=$SMOKE_UPDATES target_global_step=$TARGET_GLOBAL_STEP"
echo "tuner_type=full freeze_llm=false freeze_vit=true freeze_aligner=false"
echo "audio_encoder_policy=frozen"
echo "fsdp_version=2 state_dict_type=SHARDED_STATE_DICT activation_checkpointing=false gradient_checkpointing=false"
echo "per_device_train_batch_size=$MICRO_BATCH_SIZE gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS"
echo "global_effective_batch_size=$((WORLD_SIZE * MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "save_strategy=steps save_steps=$TARGET_GLOBAL_STEP save_total_limit=1 save_only_model=false"
echo "resume_data_policy=ignore_old_8rank_data_skip"
echo "lr_scheduler_type=cosine warmup_ratio=$WARMUP_RATIO weight_decay=$WEIGHT_DECAY max_grad_norm=$MAX_GRAD_NORM"

TRAIN_PID=""
MONITOR_PID=""

resource_monitor() {
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    echo "========== FSDP 8-TO-7 CROSS-WORLD RESOURCE SNAPSHOT =========="
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
  echo "========== AUDIOCAPS V2 FSDP 8-TO-7 CROSS-WORLD RESUME SMOKE EXIT =========="
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
CMD+=(--max_length 192 --output_dir "$OUTPUT_ROOT" --logging_dir "$LOGGING_DIR")
CMD+=(--tuner_type full --freeze_llm false --freeze_vit true --freeze_aligner false --fsdp "$FSDP_CONFIG_PATH")
CMD+=(--learning_rate "$LEARNING_RATE" --aligner_lr "$ALIGNER_LR")
CMD+=(--lr_scheduler_type cosine --warmup_ratio "$WARMUP_RATIO" --weight_decay "$WEIGHT_DECAY" --max_grad_norm "$MAX_GRAD_NORM")
CMD+=(--gradient_checkpointing false --num_train_epochs 2 --max_steps "$TARGET_GLOBAL_STEP")
CMD+=(--per_device_train_batch_size "$MICRO_BATCH_SIZE" --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS")
CMD+=(--logging_steps 1 --save_strategy steps --save_steps "$TARGET_GLOBAL_STEP" --save_total_limit 1)
CMD+=(--dataloader_num_workers 0 --dataloader_pin_memory false --dataset_num_proc 1)
CMD+=(--save_only_model false --report_to tensorboard --bf16 true --seed 42 --data_seed 42)
CMD+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT" --ignore_data_skip true)

"${CMD[@]}" &
TRAIN_PID=$!
resource_monitor &
MONITOR_PID=$!

set +e
wait "$TRAIN_PID"
TRAIN_STATUS=$?
set -e
stop_resource_monitor
MONITOR_PID=""

if [ "$TRAIN_STATUS" -eq 0 ]; then
  FINAL_CHECKPOINT="$(find "$OUTPUT_ROOT" -type d -name "checkpoint-$TARGET_GLOBAL_STEP" -print -quit)"
  if [ -z "$FINAL_CHECKPOINT" ]; then
    echo "Smoke reported success but did not save checkpoint-$TARGET_GLOBAL_STEP" >&2
    TRAIN_STATUS=1
  else
    python - "$FINAL_CHECKPOINT" "$TARGET_GLOBAL_STEP" <<'PY'
import json
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1])
expected_step = int(sys.argv[2])
state = json.loads((checkpoint / 'trainer_state.json').read_text(encoding='utf-8'))
if state.get('global_step') != expected_step:
    raise SystemExit(f"Saved checkpoint global_step={state.get('global_step')!r}, expected {expected_step}")
for directory in ('pytorch_model_fsdp_0', 'optimizer_0'):
    path = checkpoint / directory
    if not path.is_dir() or not any(item.is_file() and item.stat().st_size > 0 for item in path.rglob('*')):
        raise SystemExit(f"Missing or empty saved FSDP state directory: {path}")
print(f'[checkpoint] cross_world_resume_saved_checkpoint={checkpoint}')
print(f'[checkpoint] verified_global_step={expected_step}')
PY
    echo "========== FSDP 8-TO-7 CROSS-WORLD RESUME SMOKE PASSED =========="
    echo "checkpoint=$FINAL_CHECKPOINT"
  fi
fi
exit "$TRAIN_STATUS"
