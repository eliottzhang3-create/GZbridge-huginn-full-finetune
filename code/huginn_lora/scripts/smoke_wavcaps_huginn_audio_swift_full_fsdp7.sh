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

TRAIN_MANIFEST="${WAVCAPS_TRAIN_MANIFEST:-$REPO_ROOT/data/audio_swift/wavcaps_audioset/wavcaps_audioset_sl_train_swift.jsonl}"
TRAIN_STATS="$TRAIN_MANIFEST.stats.json"
OUTPUT_DIR="${FULL_FSDP_OUTPUT_DIR:-outputs/huginn_audio_wavcaps_full_fsdp7_smoke1}"
LOGGING_DIR="${FULL_FSDP_LOGGING_DIR:-$OUTPUT_DIR/tensorboard}"

if [ ! -s "$TRAIN_MANIFEST" ] || [ ! -s "$TRAIN_STATS" ]; then
  echo "WavCaps manifest or stats is missing: manifest=$TRAIN_MANIFEST stats=$TRAIN_STATS" >&2
  exit 1
fi

python - "$TRAIN_STATS" <<'PY'
import json
import sys
with open(sys.argv[1], encoding='utf-8') as f:
    stats = json.load(f)
if stats.get('dataset') != 'wavcaps' or stats.get('subset') != 'AudioSet_SL':
    raise SystemExit(f"Unexpected dataset stats: {stats.get('dataset')!r}, {stats.get('subset')!r}")
if stats.get('audio_path_verification') != 'passed' or stats.get('metadata_pairing_verification') != 'passed':
    raise SystemExit('WavCaps manifest verification is not marked passed')
PY

mkdir -p "$OUTPUT_DIR" "$LOGGING_DIR"
echo "========== HUGINN AUDIO SWIFT FULL FSDP7 SMOKE =========="
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
echo "per_device_train_batch_size=1"
echo "gradient_accumulation_steps=4"
echo "global_effective_batch_size=28"
echo "max_steps=1"
echo "learning_rate=1e-5 aligner_lr=1e-4"
echo "gradient_checkpointing=false"

TRAIN_PID=""
MONITOR_PID=""

resource_monitor() {
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    echo "========== FULL FSDP7 RESOURCE SNAPSHOT =========="
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
  echo "========== HUGINN AUDIO SWIFT FULL FSDP7 SMOKE EXIT =========="
  echo "exit_status=$status"
  echo "exit_time=$(date '+%Y-%m-%d %H:%M:%S')"
  exit "$status"
}
trap on_exit EXIT

# NPROC_PER_NODE makes the `swift` CLI re-exec itself through torchrun. Calling
# SwiftSft directly under an outer torchrun bypasses this installed FSDP2 path.
CMD=(swift sft)
CMD+=(--model "$REPO_ROOT/models/huginn-audio-whisper-v1")
CMD+=(--model_type huginn_audio_raven --template huginn_audio_text)
CMD+=(--external_plugins "$REPO_ROOT/code/huginn_lora/plugins/huginn_audio_swift.py")
CMD+=(--dataset "$TRAIN_MANIFEST")
CMD+=(--dataset_shuffle true --train_dataloader_shuffle true --sortish_sampler false --group_by_length false)
CMD+=(--max_length 192 --output_dir "$OUTPUT_DIR" --logging_dir "$LOGGING_DIR")
CMD+=(--tuner_type full --freeze_llm false --freeze_vit true --freeze_aligner false --fsdp fsdp2)
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
