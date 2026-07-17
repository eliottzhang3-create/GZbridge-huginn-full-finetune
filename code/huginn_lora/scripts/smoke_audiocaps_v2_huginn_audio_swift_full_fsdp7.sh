#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export OMP_NUM_THREADS=4
export HUGINN_AUDIO_FSDP2_NONPERSISTENT_ROPE=1
export HUGINN_AUDIO_TRAIN_CHAIN_AUDIT=1

# This validation checks FSDP2 sharding, checkpoint save, and checkpoint resume.
# Swift treats `--fsdp fsdp2` as an immutable preset. Passing a complete config
# path directly to `--fsdp` is the supported way to override that preset.
FSDP_CONFIG='{"fsdp":"full_shard auto_wrap","fsdp_config":{"activation_checkpointing":false,"auto_wrap_policy":"TRANSFORMER_BASED_WRAP","cpu_ram_efficient_loading":true,"fsdp_version":2,"reshard_after_forward":true,"state_dict_type":"SHARDED_STATE_DICT"}}'

TRAIN_MANIFEST="${AUDIOCAPS_FULL_TRAIN_MANIFEST:-$REPO_ROOT/data/audio_swift/audiocaps_v2/audiocaps_v2_train_swift.jsonl}"
TRAIN_STATS="$TRAIN_MANIFEST.stats.json"
RUN_TAG="${AUDIOCAPS_FULL_FSDP_RUN_TAG:-$(date '+%Y%m%d_%H%M%S')}"
OUTPUT_ROOT="${AUDIOCAPS_FULL_FSDP_OUTPUT_DIR:-outputs/huginn_audio_audiocaps_v2_full_fsdp8_checkpoint_resume/run-$RUN_TAG}"
SAVE_OUTPUT_DIR="$OUTPUT_ROOT/save_phase"
RESUME_OUTPUT_DIR="$OUTPUT_ROOT/resume_phase"
SAVE_LOGGING_DIR="$SAVE_OUTPUT_DIR/tensorboard"
RESUME_LOGGING_DIR="$RESUME_OUTPUT_DIR/tensorboard"

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

mkdir -p "$OUTPUT_ROOT" "$SAVE_LOGGING_DIR" "$RESUME_LOGGING_DIR"
FSDP_CONFIG_PATH="$OUTPUT_ROOT/fsdp2_checkpoint_resume_no_activation.json"
printf '%s\n' "$FSDP_CONFIG" > "$FSDP_CONFIG_PATH"
echo "========== HUGINN AUDIOCAPS V2 SWIFT FULL FSDP8 CHECKPOINT RESUME =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "launch_mode=swift_cli_internal_torchrun"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "dataset=$TRAIN_MANIFEST"
echo "output_root=$OUTPUT_ROOT"
echo "save_output_dir=$SAVE_OUTPUT_DIR"
echo "resume_output_dir=$RESUME_OUTPUT_DIR"
echo "tuner_type=full"
echo "freeze_llm=false freeze_vit=true freeze_aligner=false"
echo "audio_encoder_policy=frozen"
echo "fsdp=custom_fsdp2_json"
echo "fsdp2_rope_buffer=nonpersistent"
echo "fsdp_activation_checkpointing=false"
echo "fsdp_config_path=$FSDP_CONFIG_PATH"
echo "train_chain_audit=true"
echo "per_device_train_batch_size=1"
echo "gradient_accumulation_steps=4"
echo "global_effective_batch_size=32"
echo "save_phase_max_steps=2 resume_phase_max_steps=3"
echo "save_strategy=steps save_only_model=false state_dict_type=SHARDED_STATE_DICT"
echo "learning_rate=1e-5 aligner_lr=1e-4"
echo "gradient_checkpointing=false"

TRAIN_PID=""
MONITOR_PID=""

resource_monitor() {
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    echo "========== AUDIOCAPS FULL FSDP8 CHECKPOINT RESOURCE SNAPSHOT =========="
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
  echo "========== HUGINN AUDIOCAPS V2 SWIFT FULL FSDP8 CHECKPOINT RESUME EXIT =========="
  echo "exit_status=$status"
  echo "exit_time=$(date '+%Y-%m-%d %H:%M:%S')"
  exit "$status"
}
trap on_exit EXIT

COMMON_CMD=(swift sft)
COMMON_CMD+=(--model "$REPO_ROOT/models/huginn-audio-whisper-v1")
COMMON_CMD+=(--model_type huginn_audio_raven --template huginn_audio_text)
COMMON_CMD+=(--external_plugins "$REPO_ROOT/code/huginn_lora/plugins/huginn_audio_swift.py")
COMMON_CMD+=(--dataset "$TRAIN_MANIFEST")
COMMON_CMD+=(--dataset_shuffle true --train_dataloader_shuffle true --sortish_sampler false --group_by_length false)
COMMON_CMD+=(--max_length 192)
COMMON_CMD+=(--tuner_type full --freeze_llm false --freeze_vit true --freeze_aligner false --fsdp "$FSDP_CONFIG_PATH")
COMMON_CMD+=(--learning_rate 1e-5 --aligner_lr 1e-4 --gradient_checkpointing false)
COMMON_CMD+=(--per_device_train_batch_size 1 --gradient_accumulation_steps 4)
COMMON_CMD+=(--logging_steps 1 --dataloader_num_workers 0 --dataloader_pin_memory false)
COMMON_CMD+=(--dataset_num_proc 1 --save_only_model false --report_to none --bf16 true)

run_stage() {
  local stage_name="$1"
  shift
  echo "========== FSDP CHECKPOINT STAGE $stage_name =========="
  "$@" &
  TRAIN_PID=$!
  resource_monitor &
  MONITOR_PID=$!

  set +e
  wait "$TRAIN_PID"
  local stage_status=$?
  set -e
  stop_resource_monitor
  MONITOR_PID=""
  TRAIN_PID=""
  if [ "$stage_status" -ne 0 ]; then
    echo "[checkpoint] stage=$stage_name status=$stage_status" >&2
    exit "$stage_status"
  fi
  echo "[checkpoint] stage=$stage_name status=0"
}

find_checkpoint() {
  local search_root="$1"
  local step="$2"
  local matches=()
  mapfile -t matches < <(find "$search_root" -type d -name "checkpoint-$step" -print | sort)
  if [ "${#matches[@]}" -ne 1 ]; then
    echo "Expected exactly one checkpoint-$step below $search_root, found ${#matches[@]}" >&2
    printf '  %s\n' "${matches[@]:-<none>}" >&2
    exit 1
  fi
  printf '%s\n' "${matches[0]}"
}

inspect_checkpoint() {
  local checkpoint_dir="$1"
  local expected_step="$2"
  python - "$checkpoint_dir" "$expected_step" "$NPROC_PER_NODE" <<'PY'
import json
import sys
from pathlib import Path

checkpoint_dir = Path(sys.argv[1])
expected_step = int(sys.argv[2])
world_size = int(sys.argv[3])
trainer_state_path = checkpoint_dir / 'trainer_state.json'
if not trainer_state_path.is_file():
    raise SystemExit(f'Missing trainer_state.json: {trainer_state_path}')
trainer_state = json.loads(trainer_state_path.read_text(encoding='utf-8'))
actual_step = trainer_state.get('global_step')
if actual_step != expected_step:
    raise SystemExit(f'Checkpoint global_step mismatch: expected={expected_step} actual={actual_step}')
files = sorted(path for path in checkpoint_dir.rglob('*') if path.is_file())
data_files = [path for path in files if path.name != 'trainer_state.json' and path.stat().st_size > 0]
if len(data_files) < world_size:
    raise SystemExit(
        f'Checkpoint has too few non-empty state files for world_size={world_size}: {len(data_files)}')
print(f'[checkpoint] verified_path={checkpoint_dir}')
print(f'[checkpoint] verified_global_step={actual_step}')
print(f'[checkpoint] nonempty_state_file_count={len(data_files)} world_size={world_size}')
for path in data_files[:24]:
    print(f'[checkpoint] file={path.relative_to(checkpoint_dir)} bytes={path.stat().st_size}')
PY
}

SAVE_CMD=("${COMMON_CMD[@]}")
SAVE_CMD+=(--output_dir "$SAVE_OUTPUT_DIR" --logging_dir "$SAVE_LOGGING_DIR")
SAVE_CMD+=(--max_steps 2 --save_strategy steps --save_steps 2 --save_total_limit 2)
run_stage save "${SAVE_CMD[@]}"

CHECKPOINT_2="$(find_checkpoint "$SAVE_OUTPUT_DIR" 2)"
inspect_checkpoint "$CHECKPOINT_2" 2

RESUME_CMD=("${COMMON_CMD[@]}")
RESUME_CMD+=(--output_dir "$RESUME_OUTPUT_DIR" --logging_dir "$RESUME_LOGGING_DIR")
RESUME_CMD+=(--max_steps 3 --save_strategy steps --save_steps 3 --save_total_limit 2)
RESUME_CMD+=(--resume_from_checkpoint "$CHECKPOINT_2")
run_stage resume "${RESUME_CMD[@]}"

CHECKPOINT_3="$(find_checkpoint "$RESUME_OUTPUT_DIR" 3)"
inspect_checkpoint "$CHECKPOINT_3" 3
echo "========== FSDP CHECKPOINT RESUME VERIFICATION PASSED =========="
echo "checkpoint_2=$CHECKPOINT_2"
echo "checkpoint_3=$CHECKPOINT_3"
