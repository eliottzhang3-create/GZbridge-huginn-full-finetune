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

# These are the exact compatibility switches used by the successful dynamic-
# prefix smoke. The formal run uses two ranks; Huginn recurrent activation
# recomputation remains off.
export HUGINN_AUDIO_FSDP2_NONPERSISTENT_ROPE=1
export HUGINN_LOSATOK_DYNAMIC_AUDIO_TOKENS=1
export HUGINN_LOSATOK_PEFT_ALIGNER_MODULES_TO_SAVE=1
export HUGINN_LOSATOK_TRAIN_CHAIN_AUDIT=1

TRAIN_MANIFEST="${LOSATOK_DYNAMIC_FSDP2_TRAIN_MANIFEST:-$REPO_ROOT/data/audio_swift/audiocaps_v2/audiocaps_v2_train_swift.jsonl}"
TRAIN_STATS="$TRAIN_MANIFEST.stats.json"
OUTPUT_DIR="${LOSATOK_DYNAMIC_FSDP2_OUTPUT_DIR:-outputs/huginn_losatok_dynamic90s_audiocaps_v2_e3_b4ga4_fsdp2}"
LOGGING_DIR="${LOSATOK_DYNAMIC_FSDP2_LOGGING_DIR:-$OUTPUT_DIR/tensorboard}"
PLUGIN_PATH="$REPO_ROOT/code/huginn_lora/plugins/huginn_losatok_swift.py"
MODEL_PATH="$REPO_ROOT/models/huginn-audio-losatok-v1"
LOSATOK_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/models/LoSATok
LOSATOK_CODE_DIR="$REPO_ROOT/code/huginn_lora/LosatokCode"

WORLD_SIZE=2
MICRO_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
NUM_TRAIN_EPOCHS=3
LEARNING_RATE="${LOSATOK_DYNAMIC_FSDP2_LEARNING_RATE:-1e-4}"
ALIGNER_LR="${LOSATOK_DYNAMIC_FSDP2_ALIGNER_LR:-1e-4}"
LOGGING_STEPS="${LOSATOK_DYNAMIC_FSDP2_LOGGING_STEPS:-10}"
REPORT_TO="${LOSATOK_DYNAMIC_FSDP2_REPORT_TO:-tensorboard}"

FSDP_CONFIG='{"fsdp":"full_shard auto_wrap","fsdp_config":{"activation_checkpointing":false,"auto_wrap_policy":"TRANSFORMER_BASED_WRAP","cpu_ram_efficient_loading":true,"fsdp_version":2,"reshard_after_forward":true,"state_dict_type":"SHARDED_STATE_DICT"}}'

if [ ! -s "$TRAIN_MANIFEST" ] || [ ! -s "$TRAIN_STATS" ]; then
  echo "AudioCaps-v2 manifest or stats is missing: manifest=$TRAIN_MANIFEST stats=$TRAIN_STATS" >&2
  exit 1
fi
for required_path in \
  "$MODEL_PATH" \
  "$PLUGIN_PATH" \
  "$LOSATOK_ROOT/ckpts/losatok_kl1e-3.pth" \
  "$LOSATOK_ROOT/ckpts/semantic_encoder.pth" \
  "$LOSATOK_ROOT/midashenglm" \
  "$LOSATOK_CODE_DIR/config/16k_16k_25Hz_losatok.yml"; do
  if [ ! -e "$required_path" ]; then
    echo "Required LoSATok FSDP2 training path is missing: $required_path" >&2
    exit 1
  fi
done

RECORD_COUNT="$(python - "$TRAIN_STATS" <<'PY'
import json
import sys
from dataclasses import fields

with open(sys.argv[1], encoding="utf-8") as handle:
    stats = json.load(handle)
if stats.get("dataset") != "audiocaps_v2" or stats.get("split") != "train":
    raise SystemExit(
        f"Unexpected AudioCaps stats: dataset={stats.get('dataset')!r} split={stats.get('split')!r}"
    )
record_count = stats.get("record_count")
if not isinstance(record_count, int) or record_count <= 0:
    raise SystemExit(f"Unexpected AudioCaps record_count: {record_count!r}")
if stats.get("audio_path_verification") != "passed" or stats.get("wav_readability_verification") != "passed":
    raise SystemExit("AudioCaps manifest verification is not marked passed")
if stats.get("limit_records") is None:
    source_count = stats.get("source_csv_row_count")
    excluded_count = stats.get("excluded_row_count")
    if source_count != record_count + excluded_count:
        raise SystemExit("AudioCaps source/valid/excluded row accounting mismatch")

from swift.arguments.sft_args import SftArguments

available = {field.name for field in fields(SftArguments)}
required = {
    "fsdp", "num_train_epochs", "save_strategy", "save_total_limit",
    "save_only_model", "tuner_type", "freeze_aligner",
}
missing = sorted(required - available)
if missing:
    raise SystemExit(f"Installed Swift lacks required formal FSDP LoRA arguments: {missing}")
print(record_count)
PY
)"

python - <<'PY'
import torch
import torchaudio

print(f"[env] torch={torch.__version__} torchaudio={torchaudio.__version__} cuda={torch.version.cuda}")
if torch.__version__ != torchaudio.__version__:
    raise SystemExit("Torch and torchaudio versions must match for LoSATok training")
PY

mkdir -p "$OUTPUT_DIR" "$LOGGING_DIR"
if find "$OUTPUT_DIR" -type d -name 'checkpoint-*' -print -quit | grep -q .; then
  echo "Formal output already contains checkpoints; choose a fresh LOSATOK_DYNAMIC_FSDP2_OUTPUT_DIR: $OUTPUT_DIR" >&2
  exit 1
fi

FSDP_CONFIG_PATH="$OUTPUT_DIR/fsdp2_lora_no_activation.json"
printf '%s\n' "$FSDP_CONFIG" > "$FSDP_CONFIG_PATH"

echo "========== AUDIOCAPS-V2 HUGINN LOSATOK DYNAMIC90S LORA FSDP2-2GPU TRAIN =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "launch_mode=swift_cli_internal_torchrun"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "dataset=$TRAIN_MANIFEST"
echo "record_count=$RECORD_COUNT"
echo "output_dir=$OUTPUT_DIR"
echo "architecture=dynamic_audio_prefix"
echo "audio_max_seconds=90"
echo "losatok_nominal_token_rate_hz=25"
echo "compressor_kernel_size=11 compressor_stride=6 adaptive_pool=false"
echo "max_compressed_audio_tokens=375 max_audio_prefix_tokens=377"
echo "max_text_tokens=192 max_combined_context_tokens=569"
echo "audio_batch_padding=per_batch_max attention_mask_zero labels_minus_100"
echo "training_policy=aligner_plus_huginn_lora_only"
echo "losatok_encoder_policy=frozen expected_trainable_parameters=0"
echo "huginn_base_policy=frozen expected_trainable_parameters=0"
echo "aligner_policy=trainable_including_audio_bos_audio_eos expected_trainable_parameters=62953248"
echo "huginn_lora_policy=trainable expected_trainable_parameters=12541440"
echo "expected_total_trainable_parameters=75494688"
echo "loss=next_token_prediction logits[t]_predict_labels[t+1] audio_prefix_labels=-100"
echo "fsdp=custom_fsdp2_json full_shard_auto_wrap world_size=$WORLD_SIZE"
echo "fsdp_version=2 state_dict_type=SHARDED_STATE_DICT"
echo "fsdp_activation_checkpointing=false gradient_checkpointing=false"
echo "num_train_epochs=$NUM_TRAIN_EPOCHS"
echo "per_device_train_batch_size=$MICRO_BATCH_SIZE"
echo "gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS"
echo "global_effective_batch_size=$((WORLD_SIZE * MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "learning_rate=$LEARNING_RATE aligner_lr=$ALIGNER_LR"
echo "dataset_shuffle=true train_dataloader_shuffle=true"
echo "save_strategy=epoch save_total_limit=3 save_only_model=false"
echo "logging_steps=$LOGGING_STEPS report_to=$REPORT_TO"
echo "train_chain_audit=true"

TRAIN_PID=""
MONITOR_PID=""

print_resource_snapshot() {
  echo "========== LOSATOK DYNAMIC90S FSDP2-2GPU RESOURCE SNAPSHOT =========="
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
  echo "========== AUDIOCAPS-V2 HUGINN LOSATOK DYNAMIC90S LORA FSDP2-2GPU EXIT =========="
  echo "exit_status=$status"
  echo "exit_time=$(date '+%Y-%m-%d %H:%M:%S')"
  exit "$status"
}

on_signal() {
  local signal_name=$1
  echo "========== LOSATOK DYNAMIC90S FSDP2-2GPU SIGNAL =========="
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
  --fsdp "$FSDP_CONFIG_PATH" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --per_device_train_batch_size "$MICRO_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --gradient_checkpointing false \
  --logging_steps "$LOGGING_STEPS" \
  --save_strategy epoch \
  --save_total_limit 3 \
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

mapfile -t CHECKPOINT_DIRS < <(find "$OUTPUT_DIR" -type d -name 'checkpoint-*' -print | sort -V)
if [ "${#CHECKPOINT_DIRS[@]}" -ne 3 ]; then
  echo "Expected exactly three epoch checkpoints below $OUTPUT_DIR, found ${#CHECKPOINT_DIRS[@]}" >&2
  printf '  %s\n' "${CHECKPOINT_DIRS[@]:-<none>}" >&2
  exit 1
fi

python - "$WORLD_SIZE" "${CHECKPOINT_DIRS[@]}" <<'PY'
import json
import sys
from pathlib import Path

world_size = int(sys.argv[1])
checkpoint_dirs = [Path(value) for value in sys.argv[2:]]
previous_step = -1
for index, checkpoint_dir in enumerate(checkpoint_dirs, start=1):
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    if not trainer_state_path.is_file():
        raise SystemExit(f"Missing trainer_state.json: {trainer_state_path}")
    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    step = trainer_state.get("global_step")
    epoch = trainer_state.get("epoch")
    if not isinstance(step, int) or step <= previous_step:
        raise SystemExit(f"Non-increasing checkpoint global_step: previous={previous_step} current={step}")
    if not isinstance(epoch, (int, float)) or abs(float(epoch) - index) > 0.05:
        raise SystemExit(f"Checkpoint epoch mismatch: expected={index} actual={epoch}")
    data_files = [
        path for path in checkpoint_dir.rglob("*")
        if path.is_file() and path.name != "trainer_state.json" and path.stat().st_size > 0
    ]
    if len(data_files) < world_size:
        raise SystemExit(
            f"Checkpoint has too few non-empty state files for world_size={world_size}: "
            f"path={checkpoint_dir} count={len(data_files)}"
        )
    print(
        f"[checkpoint] epoch={epoch} global_step={step} path={checkpoint_dir} "
        f"nonempty_state_file_count={len(data_files)}"
    )
    previous_step = step
PY

echo "========== AUDIOCAPS-V2 HUGINN LOSATOK DYNAMIC90S LORA FSDP2-2GPU TRAIN PASSED =========="
printf 'checkpoint=%s\n' "${CHECKPOINT_DIRS[@]}"
exit 0
