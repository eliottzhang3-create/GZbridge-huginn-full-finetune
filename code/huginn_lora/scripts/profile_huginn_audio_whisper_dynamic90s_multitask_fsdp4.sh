#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH

WORLD_SIZE=4
PER_DEVICE_BATCH=2
GRADIENT_ACCUMULATION_STEPS=4
GLOBAL_BATCH=$((WORLD_SIZE * PER_DEVICE_BATCH * GRADIENT_ACCUMULATION_STEPS))
SEED="${HUGINN_DYNAMIC90S_MIXTURE_SEED:-20260730}"
START_POSITION="${HUGINN_DYNAMIC90S_MIXTURE_START_POSITION:-0}"
MAX_STEPS="${HUGINN_AUDIO_DYNAMIC90S_PROFILE_MAX_STEPS:-8}"
FORMAL_MAX_STEPS="${HUGINN_AUDIO_DYNAMIC90S_PROFILE_FORMAL_MAX_STEPS:-17700}"
PROFILE_WAIT="${HUGINN_TORCH_PROFILER_WAIT:-4}"
PROFILE_WARMUP="${HUGINN_TORCH_PROFILER_WARMUP:-4}"
PROFILE_ACTIVE="${HUGINN_TORCH_PROFILER_ACTIVE:-8}"
PROFILE_WITH_STACK="${HUGINN_TORCH_PROFILER_WITH_STACK:-0}"
MIN_FREE_GB="${HUGINN_AUDIO_DYNAMIC90S_PROFILE_MIN_FREE_GB:-50}"
LEARNING_RATE=1e-4
ALIGNER_LR=1e-4
WHISPER_LR=1e-4
WARMUP_RATIO=0.05
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0

MODEL_PATH="$REPO_ROOT/models/huginn-audio-whisper-dynamic90s-v1"
PLUGIN_PATH="$REPO_ROOT/code/huginn_lora/plugins/huginn_audio_whisper_dynamic90s_mixture_profiler_swift.py"
REGISTRY="${HUGINN_DYNAMIC90S_POOL_REGISTRY:-$REPO_ROOT/data/audio_swift/huginn_whisper_dynamic90s_multitask/v1/pool_registry.json}"
REALDATA_REPORT="${HUGINN_DYNAMIC90S_REAL_DATA_CHAIN_REPORT:-$REPO_ROOT/data/audio_swift/huginn_whisper_dynamic90s_multitask/v1/real_data_chain_report.json}"
SAMPLER_REPORT="${HUGINN_DYNAMIC90S_SAMPLER_REPORT:-$REPO_ROOT/data/audio_swift/huginn_whisper_dynamic90s_multitask/v1/sampler/mixture_sampler_report.json}"
INSPECTOR="$REPO_ROOT/code/huginn_lora/scripts/inspect_huginn_whisper_dynamic90s_profiler_results.py"
OUTPUT_DIR="${HUGINN_AUDIO_DYNAMIC90S_PROFILE_OUTPUT_DIR:-$REPO_ROOT/outputs/huginn_audio_whisper_dynamic90s_multitask_fsdp4_profiler/run-$(date +%Y%m%d_%H%M%S)}"
MODULES_TO_SAVE=(temporal_compressor audio_projector audio_boundary_embeddings)

for value_name in MAX_STEPS FORMAL_MAX_STEPS PROFILE_WAIT PROFILE_WARMUP PROFILE_ACTIVE START_POSITION MIN_FREE_GB; do
  value="${!value_name}"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "$value_name must be a non-negative integer, got: $value" >&2
    exit 1
  fi
done
if (( MAX_STEPS <= 0 || FORMAL_MAX_STEPS <= 0 || PROFILE_ACTIVE <= 0 )); then
  echo "MAX_STEPS, FORMAL_MAX_STEPS, and PROFILE_ACTIVE must be positive" >&2
  exit 1
fi
AVAILABLE_GB="$(df -BG "$REPO_ROOT" | awk 'NR==2 {gsub(/G/, "", $4); print $4}')"
if [ -z "$AVAILABLE_GB" ] || [ "$AVAILABLE_GB" -lt "$MIN_FREE_GB" ]; then
  echo "Insufficient profiler storage: available=${AVAILABLE_GB:-unknown}G required=${MIN_FREE_GB}G" >&2
  exit 1
fi
TOTAL_MICROBATCHES=$((MAX_STEPS * GRADIENT_ACCUMULATION_STEPS))
PROFILED_MICROBATCHES=$((PROFILE_WAIT + PROFILE_WARMUP + PROFILE_ACTIVE))
if (( TOTAL_MICROBATCHES <= PROFILED_MICROBATCHES )); then
  echo "Profiler run needs post-active timing: total_microbatches=$TOTAL_MICROBATCHES schedule=$PROFILED_MICROBATCHES" >&2
  exit 1
fi
if [ -e "$OUTPUT_DIR" ]; then
  echo "Profiler output already exists; choose a fresh HUGINN_AUDIO_DYNAMIC90S_PROFILE_OUTPUT_DIR: $OUTPUT_DIR" >&2
  exit 1
fi
for required_path in "$MODEL_PATH" "$PLUGIN_PATH" "$REGISTRY" "$REALDATA_REPORT" "$SAMPLER_REPORT" "$INSPECTOR"; do
  if [ ! -e "$required_path" ]; then
    echo "Required profiler path is missing: $required_path" >&2
    exit 1
  fi
done

python - "$REALDATA_REPORT" "$SAMPLER_REPORT" <<'PY'
import json
import sys
from pathlib import Path

real_data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
sampler = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if not real_data.get("validation_passed"):
    raise SystemExit(f"Real data-chain prerequisite has not passed: {sys.argv[1]}")
if (
    not sampler.get("validation_passed")
    or sampler.get("sampler_version") != "deterministic_hierarchical_no_replacement_v2"
):
    raise SystemExit(f"No-replacement sampler prerequisite has not passed: {sys.argv[2]}")
print("[precheck] real_data=passed sampler=no_replacement_v2")
PY

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
PROFILER_DIR="$OUTPUT_DIR/torch_profiler"
TRAINING_STATS_DIR="$OUTPUT_DIR/training_statistics"
FSDP_CONFIG_PATH="$OUTPUT_DIR/fsdp2_lora_activation_checkpointing.json"
AGGREGATE_REPORT="$OUTPUT_DIR/profiler_aggregate.json"
RESOURCE_LOG="$OUTPUT_DIR/resource_samples.log"
mkdir -p "$PROFILER_DIR" "$TRAINING_STATS_DIR"
printf '%s\n' '{"fsdp":"full_shard auto_wrap","fsdp_config":{"activation_checkpointing":true,"auto_wrap_policy":"TRANSFORMER_BASED_WRAP","cpu_ram_efficient_loading":true,"fsdp_version":2,"reshard_after_forward":true,"state_dict_type":"SHARDED_STATE_DICT"}}' > "$FSDP_CONFIG_PATH"

export HUGINN_DYNAMIC90S_POOL_REGISTRY="$REGISTRY"
export HUGINN_DYNAMIC90S_MIXTURE_SEED="$SEED"
export HUGINN_DYNAMIC90S_MIXTURE_START_POSITION="$START_POSITION"
export HUGINN_DYNAMIC90S_MIXTURE_MAX_SAMPLES=$(((MAX_STEPS + 2) * GLOBAL_BATCH))
export HUGINN_AUDIO_DYNAMIC90S_FSDP2_NONPERSISTENT_ROPE=1
export HUGINN_AUDIO_DYNAMIC90S_TRAIN_CHAIN_AUDIT=1
export HUGINN_AUDIO_DYNAMIC90S_PEFT_ALIGNER_MODULES_TO_SAVE=1
export HUGINN_AUDIO_DYNAMIC90S_FULL_MODEL_DCP=1
export HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_DIR="$TRAINING_STATS_DIR"
export HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_LOG_STEPS=100
export HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_PHASE=profile
unset HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_CHECKPOINT_STEPS
unset HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_RESUME_STATE
unset HUGINN_AUDIO_DYNAMIC90S_FSDP_SAVE_DEBUG
unset HUGINN_AUDIO_DYNAMIC90S_CHECKPOINT_AUDIT_DIR
unset HUGINN_AUDIO_DYNAMIC90S_DATA_POSITION_AUDIT_DIR
unset HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_FORWARD_AUDIT_DIR
export HUGINN_TORCH_PROFILER_ENABLED=1
export HUGINN_TORCH_PROFILER_OUTPUT_DIR="$PROFILER_DIR"
export HUGINN_TORCH_PROFILER_WAIT="$PROFILE_WAIT"
export HUGINN_TORCH_PROFILER_WARMUP="$PROFILE_WARMUP"
export HUGINN_TORCH_PROFILER_ACTIVE="$PROFILE_ACTIVE"
export HUGINN_TORCH_PROFILER_WITH_STACK="$PROFILE_WITH_STACK"
export HUGINN_TORCH_PROFILER_MAX_EVENT_ROWS=1000
export NCCL_DEBUG_FILE="$OUTPUT_DIR/nccl.%h.%p.log"

python - "$OUTPUT_DIR/environment.json" <<'PY'
import json
import platform
import sys
from pathlib import Path

import torch
import transformers

payload = {
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "transformers": transformers.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    "torch_config": torch.__config__.show(),
}
if not payload["cuda_available"] or payload["cuda_device_count"] != 4:
    raise SystemExit(f"Profiler requires four visible CUDA devices: {payload}")
Path(sys.argv[1]).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[env] torch={payload['torch']} cuda={payload['cuda']} devices={payload['devices']}")
PY
nvidia-smi topo -m > "$OUTPUT_DIR/nvidia_topology.txt" 2>&1 || true
nvidia-smi -q > "$OUTPUT_DIR/nvidia_query.txt" 2>&1 || true

echo "========== HUGINN WHISPER DYNAMIC90S FULL TORCH PROFILER START =========="
echo "output_dir=$OUTPUT_DIR"
echo "registry=$REGISTRY sampler=deterministic_hierarchical_no_replacement_v2 seed=$SEED start_position=$START_POSITION"
echo "world_size=$WORLD_SIZE per_device_batch=$PER_DEVICE_BATCH accumulation=$GRADIENT_ACCUMULATION_STEPS global_batch=$GLOBAL_BATCH"
echo "max_steps=$MAX_STEPS total_microbatches=$TOTAL_MICROBATCHES checkpoint_saving=disabled"
echo "profiler_schedule=wait${PROFILE_WAIT},warmup${PROFILE_WARMUP},active${PROFILE_ACTIVE},post$((TOTAL_MICROBATCHES - PROFILED_MICROBATCHES))"
echo "formal_match=whisper_trainable+aligner_trainable+huginn_lora_rank8 fsdp5 reshard_true activation_checkpointing_true"
echo "capture=cpu,cuda,flops,shapes,memory,nccl,dispatch,module_ranges,recurrence,dynamic_lengths"
echo "free_storage_gb=$AVAILABLE_GB required_free_gb=$MIN_FREE_GB"

TRAIN_PID=""
MONITOR_PID=""

resource_monitor() {
  while [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; do
    {
      echo "snapshot_time=$(date '+%Y-%m-%d %H:%M:%S')"
      nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.sm,clocks.mem --format=csv,noheader,nounits || true
      for cgroup_file in /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.events; do
        if [ -r "$cgroup_file" ]; then
          echo "cgroup_$(basename "$cgroup_file")=$(tr '\n' ' ' < "$cgroup_file")"
        fi
      done
    } >> "$RESOURCE_LOG"
    sleep 5
  done
}

stop_monitor() {
  if [ -n "$MONITOR_PID" ] && kill -0 "$MONITOR_PID" 2>/dev/null; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
  MONITOR_PID=""
}

on_exit() {
  status=$?
  trap - EXIT
  stop_monitor
  echo "========== HUGINN WHISPER DYNAMIC90S FULL TORCH PROFILER EXIT =========="
  echo "exit_status=$status"
  echo "exit_time=$(date '+%Y-%m-%d %H:%M:%S')"
  exit "$status"
}
trap on_exit EXIT

CMD=(swift sft)
CMD+=(--model "$MODEL_PATH" --model_type huginn_audio_whisper_dynamic90s --template huginn_audio_whisper_dynamic90s)
CMD+=(--external_plugins "$PLUGIN_PATH" --dataset "$REGISTRY" --streaming true)
CMD+=(--dataset_shuffle false --train_dataloader_shuffle false --sortish_sampler false --group_by_length false)
CMD+=(--max_length 192 --output_dir "$OUTPUT_DIR/swift_output")
CMD+=(--tuner_type lora_llm --freeze_vit false --freeze_aligner false)
CMD+=(--modules_to_save "${MODULES_TO_SAVE[@]}")
CMD+=(--learning_rate "$LEARNING_RATE" --aligner_lr "$ALIGNER_LR" --vit_lr "$WHISPER_LR")
CMD+=(--lora_rank 8 --lora_alpha 16 --lora_dropout 0.05)
CMD+=(--lr_scheduler_type cosine --warmup_ratio "$WARMUP_RATIO" --weight_decay "$WEIGHT_DECAY" --max_grad_norm "$MAX_GRAD_NORM")
CMD+=(--fsdp "$FSDP_CONFIG_PATH" --max_steps "$MAX_STEPS")
CMD+=(--per_device_train_batch_size "$PER_DEVICE_BATCH" --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS")
CMD+=(--gradient_checkpointing false --vit_gradient_checkpointing true --gradient_checkpointing_kwargs '{"use_reentrant": false}')
CMD+=(--logging_steps 10 --save_strategy no)
CMD+=(--dataloader_num_workers 0 --dataloader_pin_memory false --dataset_num_proc 1)
CMD+=(--save_only_model false --report_to none --bf16 true --seed "$SEED" --data_seed "$SEED")

"${CMD[@]}" &
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
stop_monitor

if find "$OUTPUT_DIR" -type d -name 'checkpoint-*' -print -quit 2>/dev/null | grep -q .; then
  echo "Profiler run must not save checkpoints" >&2
  exit 1
fi

python -u "$INSPECTOR" \
  --profiler-dir "$PROFILER_DIR" \
  --output-report "$AGGREGATE_REPORT" \
  --world-size "$WORLD_SIZE" \
  --per-device-batch "$PER_DEVICE_BATCH" \
  --gradient-accumulation "$GRADIENT_ACCUMULATION_STEPS" \
  --max-steps "$MAX_STEPS" \
  --formal-max-steps "$FORMAL_MAX_STEPS"

echo "========== HUGINN WHISPER DYNAMIC90S FULL TORCH PROFILER PASSED =========="
echo "aggregate_report=$AGGREGATE_REPORT"
echo "tensorboard_trace_root=$PROFILER_DIR"
echo "resource_log=$RESOURCE_LOG"
