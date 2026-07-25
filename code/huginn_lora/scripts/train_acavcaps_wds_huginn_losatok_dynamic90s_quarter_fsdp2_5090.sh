#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export OMP_NUM_THREADS=4
export HUGINN_AUDIO_FSDP2_NONPERSISTENT_ROPE=1
export HUGINN_LOSATOK_DYNAMIC_AUDIO_TOKENS=1
export HUGINN_LOSATOK_PEFT_ALIGNER_MODULES_TO_SAVE=1
export HUGINN_LOSATOK_TRAIN_CHAIN_AUDIT=1
unset HUGINN_LOSATOK_FSDP_SAVE_DEBUG
unset HUGINN_LOSATOK_FORCE_ALIGNER_TRAINABLE
unset HUGINN_LOSATOK_INIT_ALIGNER_CHECKPOINT
unset ACAVCAPS_WDS_MAX_TARS_PER_STAGE

INIT_CHECKPOINT="${LOSATOK_DYNAMIC_ACAV_WDS_INIT_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_losatok_dynamic90s_audiocaps_v2_e2_b4ga4_fsdp2_complete/v0-20260724-115115/checkpoint-2802}"
export HUGINN_LOSATOK_INIT_FSDP_DCP_CHECKPOINT="$INIT_CHECKPOINT"
export ACAVCAPS_WDS_MANIFEST="${ACAVCAPS_WDS_QUARTER_MANIFEST:-$REPO_ROOT/data/audio_swift/acavcaps_wds/acavcaps_wds_stage_schedule_quarter_ceil_seed20260723.json}"
export ACAVCAPS_WDS_BUFFER_SIZE="${ACAVCAPS_WDS_BUFFER_SIZE:-512}"

PLUGIN_PATH="$REPO_ROOT/code/huginn_lora/plugins/huginn_losatok_acavcaps_wds_swift.py"
MODEL_PATH="$REPO_ROOT/models/huginn-audio-losatok-v1"
CHECKPOINT_AUDIT_SCRIPT="$REPO_ROOT/code/huginn_lora/scripts/inspect_losatok_dynamic_fsdp_checkpoint.py"
LOSATOK_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/models/LoSATok
LOSATOK_CODE_DIR="$REPO_ROOT/code/huginn_lora/LosatokCode"

WORLD_SIZE=2
MICRO_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
GLOBAL_EFFECTIVE_BATCH=$((WORLD_SIZE * MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
LEARNING_RATE="${LOSATOK_DYNAMIC_ACAV_WDS_LEARNING_RATE:-1e-4}"
ALIGNER_LR="${LOSATOK_DYNAMIC_ACAV_WDS_ALIGNER_LR:-1e-4}"
LOGGING_STEPS="${LOSATOK_DYNAMIC_ACAV_WDS_LOGGING_STEPS:-10}"
REPORT_TO="${LOSATOK_DYNAMIC_ACAV_WDS_REPORT_TO:-tensorboard}"
OUTPUT_DIR="${LOSATOK_DYNAMIC_ACAV_WDS_OUTPUT_DIR:-outputs/huginn_losatok_acavcaps_wds_dynamic90s_quarter_warmstart2802_fsdp2_e1_b4ga4/run-$(date +%Y%m%d_%H%M%S)}"
LOGGING_DIR="${LOSATOK_DYNAMIC_ACAV_WDS_LOGGING_DIR:-$OUTPUT_DIR/tensorboard}"
MODULES_TO_SAVE=(temporal_compressor audio_projector audio_boundary_embeddings)
FSDP_CONFIG='{"fsdp":"full_shard auto_wrap","fsdp_config":{"activation_checkpointing":false,"auto_wrap_policy":"TRANSFORMER_BASED_WRAP","cpu_ram_efficient_loading":true,"fsdp_version":2,"reshard_after_forward":true,"state_dict_type":"SHARDED_STATE_DICT"}}'
TRAIN_PID=""
MONITOR_PID=""

if [ "$ACAVCAPS_WDS_BUFFER_SIZE" != "512" ]; then
  echo "Formal dynamic ACAVCAPS quarter training requires ACAVCAPS_WDS_BUFFER_SIZE=512, got: $ACAVCAPS_WDS_BUFFER_SIZE" >&2
  exit 1
fi
if [ -e "$OUTPUT_DIR" ]; then
  echo "Formal output directory already exists; choose a fresh LOSATOK_DYNAMIC_ACAV_WDS_OUTPUT_DIR: $OUTPUT_DIR" >&2
  exit 1
fi
for required_path in \
  "$INIT_CHECKPOINT" \
  "$INIT_CHECKPOINT/pytorch_model_fsdp_0" \
  "$ACAVCAPS_WDS_MANIFEST" \
  "${ACAVCAPS_WDS_MANIFEST%.json}.stats.json" \
  "$PLUGIN_PATH" \
  "$MODEL_PATH" \
  "$CHECKPOINT_AUDIT_SCRIPT" \
  "$LOSATOK_ROOT/ckpts/losatok_kl1e-3.pth" \
  "$LOSATOK_ROOT/ckpts/semantic_encoder.pth" \
  "$LOSATOK_ROOT/midashenglm" \
  "$LOSATOK_CODE_DIR/config/16k_16k_25Hz_losatok.yml"; do
  if [ ! -e "$required_path" ]; then
    echo "Required dynamic ACAVCAPS formal-training path is missing: $required_path" >&2
    exit 1
  fi
done

python - <<'PY'
import torch
import torchaudio

print(f"[env] torch={torch.__version__} torchaudio={torchaudio.__version__} cuda={torch.version.cuda}")
if torch.__version__ != torchaudio.__version__:
    raise SystemExit("Torch and torchaudio versions must match for dynamic LoSATok ACAVCAPS training")
PY

echo "========== INPUT CHECKPOINT DCP AUDIT =========="
python -u "$CHECKPOINT_AUDIT_SCRIPT" --checkpoint "$INIT_CHECKPOINT" --require_complete

echo "========== ACAVCAPS QUARTER MANIFEST PREFLIGHT =========="
PREFLIGHT_OUTPUT="$(python -u code/huginn_lora/scripts/inspect_acavcaps_wds_quarter_manifest.py \
  --manifest "$ACAVCAPS_WDS_MANIFEST" \
  --world_size "$WORLD_SIZE" \
  --per_device_batch_size "$MICRO_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS")"
printf '%s\n' "$PREFLIGHT_OUTPUT"

read -r TOTAL_SAMPLES MAX_STEPS < <(python - "$ACAVCAPS_WDS_MANIFEST" "$GLOBAL_EFFECTIVE_BATCH" <<'PY'
import json
import math
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
global_batch = int(sys.argv[2])
stats = json.loads(manifest.with_suffix('.stats.json').read_text(encoding='utf-8'))
total_samples = stats.get('sample_count')
if not isinstance(total_samples, int) or total_samples <= 0:
    raise SystemExit(f'Invalid quarter sample_count: {total_samples!r}')
print(total_samples, math.ceil(total_samples / global_batch))
PY
)
if [ -z "${TOTAL_SAMPLES:-}" ] || [ -z "${MAX_STEPS:-}" ]; then
  echo "Unable to derive ACAVCAPS quarter total_samples/max_steps" >&2
  exit 1
fi

# Save at the most frequent exact partition among 4/3/2 equal partitions.
# The interval divides MAX_STEPS, therefore checkpoint-MAX_STEPS is guaranteed.
SAVE_PARTITIONS=1
for candidate in 4 3 2; do
  if (( MAX_STEPS % candidate == 0 )); then
    SAVE_PARTITIONS=$candidate
    break
  fi
done
SAVE_STEPS=$((MAX_STEPS / SAVE_PARTITIONS))
SAVE_TOTAL_LIMIT=2

mkdir -p "$OUTPUT_DIR" "$LOGGING_DIR"
FSDP_CONFIG_PATH="$OUTPUT_DIR/fsdp2_lora_no_activation.json"
printf '%s\n' "$FSDP_CONFIG" > "$FSDP_CONFIG_PATH"

print_resource_snapshot() {
  echo "========== ACAVCAPS DYNAMIC90S FSDP2 RESOURCE SNAPSHOT =========="
  echo "snapshot_time=$(date '+%Y-%m-%d %H:%M:%S')"
  if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    ps -o pid,ppid,rss,vsz,%mem,etime,stat,cmd -p "$TRAIN_PID" || true
  fi
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
  for cgroup_file in /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.events /sys/fs/cgroup/memory/memory.usage_in_bytes /sys/fs/cgroup/memory/memory.limit_in_bytes; do
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
  echo "========== ACAVCAPS DYNAMIC90S FSDP2 FORMAL TRAIN EXIT =========="
  echo "exit_status=$status"
  echo "exit_time=$(date '+%Y-%m-%d %H:%M:%S')"
  exit "$status"
}

on_signal() {
  local signal_name=$1
  echo "========== ACAVCAPS DYNAMIC90S FSDP2 FORMAL TRAIN SIGNAL =========="
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

echo "========== ACAVCAPS DYNAMIC LOSATOK QUARTER FSDP2 FORMAL TRAIN =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "init_semantics=adapter_weight_warm_start_only optimizer_scheduler_rng_global_step_and_data_position=fresh"
echo "init_checkpoint=$INIT_CHECKPOINT"
echo "manifest=$ACAVCAPS_WDS_MANIFEST"
echo "stage_schedule=stage1(00A,0M0,S00)->stage2(S0A,SM0,0MA)->stage3(SMA)"
echo "data_order=private_source_stage_tar_random_order_preserved_then_per_tar_webdataset_buffer_shuffle"
echo "streaming=true decode_policy=training_time_only max_tars_per_stage=all"
echo "total_samples=$TOTAL_SAMPLES num_train_epochs=1 max_steps=$MAX_STEPS"
echo "world_size=$WORLD_SIZE per_device_batch=$MICRO_BATCH_SIZE accumulation=$GRADIENT_ACCUMULATION_STEPS global_effective_batch=$GLOBAL_EFFECTIVE_BATCH"
echo "dynamic_audio_prefix=90_seconds compressor=kernel11_stride6 adaptive_pool=false max_audio_tokens=375"
echo "training_policy=frozen_losatok+trainable_peft_owned_aligner_including_audio_bos_eos+trainable_huginn_lora"
echo "checkpoint_contract=lora_66+aligner_20 fsdp_state_dict=SHARDED_STATE_DICT"
echo "save_strategy=steps save_partitions=$SAVE_PARTITIONS save_steps=$SAVE_STEPS save_total_limit=$SAVE_TOTAL_LIMIT"
echo "learning_rate=$LEARNING_RATE aligner_lr=$ALIGNER_LR logging_steps=$LOGGING_STEPS report_to=$REPORT_TO"

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
  --modules_to_save "${MODULES_TO_SAVE[@]}" \
  --learning_rate "$LEARNING_RATE" \
  --aligner_lr "$ALIGNER_LR" \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --adapters "$INIT_CHECKPOINT" \
  --load_args false \
  --fsdp "$FSDP_CONFIG_PATH" \
  --num_train_epochs 1 \
  --max_steps "$MAX_STEPS" \
  --per_device_train_batch_size "$MICRO_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --gradient_checkpointing false \
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

mapfile -t FINAL_CHECKPOINTS < <(find "$OUTPUT_DIR" -type d -name "checkpoint-$MAX_STEPS" -print | sort)
if [ "${#FINAL_CHECKPOINTS[@]}" -ne 1 ]; then
  echo "Expected exactly one final checkpoint-$MAX_STEPS below $OUTPUT_DIR" >&2
  exit 1
fi
FINAL_CHECKPOINT="${FINAL_CHECKPOINTS[0]}"

python - "$FINAL_CHECKPOINT/trainer_state.json" "$MAX_STEPS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
if not path.is_file():
    raise SystemExit(f'Missing trainer_state.json: {path}')
actual = json.loads(path.read_text(encoding='utf-8')).get('global_step')
if actual != expected:
    raise SystemExit(f'Final global_step mismatch: expected={expected} actual={actual}')
print(f'[checkpoint] final_global_step={actual}')
PY

mapfile -t SURVIVING_CHECKPOINTS < <(find "$OUTPUT_DIR" -type d -name 'checkpoint-*' -print | sort -V)
if [ "${#SURVIVING_CHECKPOINTS[@]}" -lt 1 ] || [ "${#SURVIVING_CHECKPOINTS[@]}" -gt "$SAVE_TOTAL_LIMIT" ]; then
  echo "Unexpected surviving checkpoint count: ${#SURVIVING_CHECKPOINTS[@]}" >&2
  printf '  %s\n' "${SURVIVING_CHECKPOINTS[@]:-<none>}" >&2
  exit 1
fi
for checkpoint_dir in "${SURVIVING_CHECKPOINTS[@]}"; do
  echo "========== FORMAL ACAVCAPS CHECKPOINT DCP AUDIT =========="
  echo "checkpoint=$checkpoint_dir"
  python -u "$CHECKPOINT_AUDIT_SCRIPT" --checkpoint "$checkpoint_dir" --require_complete
done

echo "========== ACAVCAPS DYNAMIC LOSATOK QUARTER FSDP2 FORMAL TRAIN PASSED =========="
echo "final_checkpoint=$FINAL_CHECKPOINT"
printf 'surviving_checkpoint=%s\n' "${SURVIVING_CHECKPOINTS[@]}"
