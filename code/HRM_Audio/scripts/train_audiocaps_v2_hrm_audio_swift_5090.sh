#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_HRM"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

TRAIN_MANIFEST="${HRM_AUDIOCAPS_TRAIN_MANIFEST:-$REPO_ROOT/data/audio_swift/audiocaps_v2/audiocaps_v2_train_hrm_audio.jsonl}"
TRAIN_STATS="${HRM_AUDIOCAPS_TRAIN_STATS:-$TRAIN_MANIFEST.stats.json}"
RUN_TAG="${HRM_AUDIO_FORMAL_RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${HRM_AUDIO_FORMAL_RUN_ROOT:-$REPO_ROOT/outputs/hrm_text/audio_audiocaps_v2_train_e2_b8ga4_r16_5090/$RUN_TAG}"
SWIFT_OUTPUT_DIR="$RUN_ROOT/swift_output"
LOGGING_DIR="$RUN_ROOT/tensorboard"
PREFLIGHT_REPORT="$RUN_ROOT/formal_preflight_report.json"
FINAL_AUDIT_REPORT="${HRM_AUDIO_FORMAL_AUDIT_REPORT:-$RUN_ROOT/formal_checkpoint_audit.json}"
RESUME_FROM_CHECKPOINT="${HRM_AUDIO_FORMAL_RESUME_FROM_CHECKPOINT:-}"
REPORT_TO="${HRM_AUDIO_FORMAL_REPORT_TO:-none}"
RESOURCE_INTERVAL="${HRM_AUDIO_FORMAL_RESOURCE_INTERVAL_SECONDS:-60}"

NUM_TRAIN_EPOCHS=2
MICRO_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=4
EFFECTIVE_BATCH_SIZE=32
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT=0.0
LEARNING_RATE=1e-4
ALIGNER_LEARNING_RATE=1e-4
LOGGING_STEPS=10
SAVE_TOTAL_LIMIT=2
EXPECTED_EPOCH1_STEP=2802
EXPECTED_EPOCH2_STEP=5604

if [ -e "$RUN_ROOT" ]; then
  echo "Formal run root already exists; refusing to overwrite: $RUN_ROOT" >&2
  exit 1
fi
if [ ! -s "$TRAIN_MANIFEST" ] || [ ! -s "$TRAIN_STATS" ]; then
  echo "Formal HRM AudioCaps manifest/stats missing: manifest=$TRAIN_MANIFEST stats=$TRAIN_STATS" >&2
  exit 1
fi
case "$REPORT_TO" in
  none|tensorboard) ;;
  *) echo "HRM_AUDIO_FORMAL_REPORT_TO must be none or tensorboard, got: $REPORT_TO" >&2; exit 1 ;;
esac
if ! [[ "$RESOURCE_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
  echo "HRM_AUDIO_FORMAL_RESOURCE_INTERVAL_SECONDS must be a positive integer, got: $RESOURCE_INTERVAL" >&2
  exit 1
fi
mkdir -p "$RUN_ROOT" "$LOGGING_DIR"

RESUME_PREFLIGHT_ARGS=()
RESUME_SWIFT_ARGS=()
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
  if [ ! -d "$RESUME_FROM_CHECKPOINT" ]; then
    echo "Formal resume checkpoint does not exist: $RESUME_FROM_CHECKPOINT" >&2
    exit 1
  fi
  RESUME_PREFLIGHT_ARGS=(--resume-checkpoint "$RESUME_FROM_CHECKPOINT")
  RESUME_SWIFT_ARGS=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

find_checkpoint() {
  local root=$1
  local name=$2
  local found
  found="$(find "$root" -type d -name "$name" -print 2>/dev/null | sort | tail -n 1)"
  if [ -z "$found" ]; then
    echo "Expected $name was not created under $root" >&2
    exit 1
  fi
  printf '%s\n' "$found"
}

TRAIN_PID=""
MONITOR_PID=""

print_resource_snapshot() {
  echo "========== HRM AUDIO FORMAL RESOURCE SNAPSHOT =========="
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
  while [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; do
    print_resource_snapshot
    sleep "$RESOURCE_INTERVAL"
  done
}

stop_resource_monitor() {
  if [ -n "$MONITOR_PID" ] && kill -0 "$MONITOR_PID" 2>/dev/null; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
  MONITOR_PID=""
}

on_exit() {
  local status=$?
  trap - EXIT
  stop_resource_monitor
  echo "========== HRM AUDIO FORMAL TRAIN EXIT =========="
  echo "exit_status=$status"
  echo "exit_time=$(date '+%Y-%m-%d %H:%M:%S')"
  exit "$status"
}

on_signal() {
  local signal_name=$1
  echo "========== HRM AUDIO FORMAL TRAIN SIGNAL =========="
  echo "signal=$signal_name"
  print_resource_snapshot
  if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    kill -TERM "$TRAIN_PID" 2>/dev/null || true
  fi
  exit 143
}

trap on_exit EXIT
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT

echo "========== HRM AUDIO AUDIOCAPS-V2 FORMAL SWIFT TRAIN =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "PYTHON=$(which python)"
echo "SWIFT=$(which swift)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "TRAIN_MANIFEST=$TRAIN_MANIFEST"
echo "TRAIN_STATS=$TRAIN_STATS"
echo "RUN_ROOT=$RUN_ROOT"
echo "SWIFT_OUTPUT_DIR=$SWIFT_OUTPUT_DIR"
echo "NUM_TRAIN_EPOCHS=$NUM_TRAIN_EPOCHS"
echo "MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE"
echo "GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS"
echo "EFFECTIVE_BATCH_SIZE=$EFFECTIVE_BATCH_SIZE"
echo "LORA_RANK=$LORA_RANK LORA_ALPHA=$LORA_ALPHA LORA_DROPOUT=$LORA_DROPOUT"
echo "LEARNING_RATE=$LEARNING_RATE ALIGNER_LEARNING_RATE=$ALIGNER_LEARNING_RATE"
echo "LAZY_TOKENIZE=true DATASET_SHUFFLE=true TRAIN_DATALOADER_SHUFFLE=true"
echo "SAVE_STRATEGY=epoch SAVE_TOTAL_LIMIT=$SAVE_TOTAL_LIMIT"
echo "EXPECTED_CHECKPOINTS=checkpoint-$EXPECTED_EPOCH1_STEP,checkpoint-$EXPECTED_EPOCH2_STEP"
echo "LOGGING_STEPS=$LOGGING_STEPS REPORT_TO=$REPORT_TO"
echo "RESOURCE_INTERVAL_SECONDS=$RESOURCE_INTERVAL"
echo "TRAINABLE=aligner+H/L_rank16_LoRA FROZEN=Whisper+HRM_base"
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
  echo "RESUME_FROM_CHECKPOINT=$RESUME_FROM_CHECKPOINT"
fi

python -u code/HRM_Audio/scripts/audit_hrm_audio_formal_training.py preflight \
  --manifest "$TRAIN_MANIFEST" \
  --stats "$TRAIN_STATS" \
  --output-report "$PREFLIGHT_REPORT" \
  "${RESUME_PREFLIGHT_ARGS[@]}"

swift sft \
  --model "$REPO_ROOT/models/hrm-text-audio-v1" \
  --model_type hrm_text_audio_whisper \
  --template hrm_text_audio \
  --external_plugins "$REPO_ROOT/code/HRM_Audio/plugins/hrm_text_audio_swift.py" \
  --dataset "$TRAIN_MANIFEST" \
  --split_dataset_ratio 0 \
  --dataset_shuffle true \
  --train_dataloader_shuffle true \
  --sortish_sampler false \
  --group_by_length false \
  --max_length 192 \
  --output_dir "$SWIFT_OUTPUT_DIR" \
  --logging_dir "$LOGGING_DIR" \
  --tuner_type lora_llm \
  --tuner_backend peft \
  --target_modules all-linear \
  --freeze_llm true \
  --freeze_vit true \
  --freeze_aligner false \
  --lora_rank "$LORA_RANK" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --learning_rate "$LEARNING_RATE" \
  --aligner_lr "$ALIGNER_LEARNING_RATE" \
  --lr_scheduler_type constant \
  --warmup_ratio 0 \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --per_device_train_batch_size "$MICRO_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --gradient_checkpointing false \
  --logging_steps "$LOGGING_STEPS" \
  --save_strategy epoch \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --save_only_model false \
  --dataloader_num_workers 0 \
  --dataloader_pin_memory false \
  --dataset_num_proc 1 \
  --lazy_tokenize true \
  --seed 42 \
  --data_seed 42 \
  --optim adamw_torch \
  --attn_impl sdpa \
  --bf16 true \
  --report_to "$REPORT_TO" \
  "${RESUME_SWIFT_ARGS[@]}" &
TRAIN_PID=$!
resource_monitor &
MONITOR_PID=$!

set +e
wait "$TRAIN_PID"
TRAIN_STATUS=$?
set -e
stop_resource_monitor
print_resource_snapshot
TRAIN_PID=""
if [ "$TRAIN_STATUS" -ne 0 ]; then
  exit "$TRAIN_STATUS"
fi

if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
  EPOCH1_CHECKPOINT="$RESUME_FROM_CHECKPOINT"
else
  EPOCH1_CHECKPOINT="$(find_checkpoint "$SWIFT_OUTPUT_DIR" "checkpoint-$EXPECTED_EPOCH1_STEP")"
fi
EPOCH2_CHECKPOINT="$(find_checkpoint "$SWIFT_OUTPUT_DIR" "checkpoint-$EXPECTED_EPOCH2_STEP")"

python -u code/HRM_Audio/scripts/audit_hrm_audio_formal_training.py audit \
  --manifest "$TRAIN_MANIFEST" \
  --preflight-report "$PREFLIGHT_REPORT" \
  --epoch1-checkpoint "$EPOCH1_CHECKPOINT" \
  --epoch2-checkpoint "$EPOCH2_CHECKPOINT" \
  --output-report "$FINAL_AUDIT_REPORT" \
  --lora-rank "$LORA_RANK" \
  --lora-alpha "$LORA_ALPHA" \
  --lora-dropout "$LORA_DROPOUT" \
  --learning-rate "$LEARNING_RATE"

echo "========== HRM AUDIO AUDIOCAPS-V2 FORMAL TRAIN PASSED =========="
echo "epoch1_checkpoint=$EPOCH1_CHECKPOINT"
echo "epoch2_checkpoint=$EPOCH2_CHECKPOINT"
echo "audit_report=$FINAL_AUDIT_REPORT"
