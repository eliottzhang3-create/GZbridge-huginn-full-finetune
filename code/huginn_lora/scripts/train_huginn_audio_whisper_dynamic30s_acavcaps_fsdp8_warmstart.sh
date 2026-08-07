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
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORLD_SIZE=8
PER_DEVICE_BATCH=4
GRADIENT_ACCUMULATION_STEPS=4
GLOBAL_BATCH_SIZE=$((WORLD_SIZE * PER_DEVICE_BATCH * GRADIENT_ACCUMULATION_STEPS))
if [ "$GLOBAL_BATCH_SIZE" -ne 128 ]; then
  echo "ACAVCAPS formal global batch mismatch: expected=128 actual=$GLOBAL_BATCH_SIZE" >&2
  exit 1
fi

SEED="${ACAVCAPS_FLAT_SEED:-20260723}"
AUDIO_ENCODER_LR="${HUGINN_AUDIO_DYNAMIC30S_AUDIO_ENCODER_LR:-1e-5}"
ALIGNER_LR="${HUGINN_AUDIO_DYNAMIC30S_ALIGNER_LR:-5e-5}"
LORA_LR="${HUGINN_AUDIO_DYNAMIC30S_LORA_LR:-5e-5}"
CHECKPOINT_INTERVAL="${HUGINN_AUDIO_DYNAMIC30S_ACAV_FORMAL_SAVE_STEPS:-5000}"
LOGGING_STEPS="${HUGINN_AUDIO_DYNAMIC30S_ACAV_FORMAL_LOGGING_STEPS:-10}"
REPORT_TO="${HUGINN_AUDIO_DYNAMIC30S_ACAV_FORMAL_REPORT_TO:-tensorboard}"
MANIFEST="${ACAVCAPS_FLAT_MANIFEST:-$REPO_ROOT/data/audio_swift/acavcaps/acavcaps_flat_global_tar_shuffle_seed20260723.json}"
INIT_CHECKPOINT="${HUGINN_AUDIO_DYNAMIC30S_ACAV_FORMAL_INIT_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_whisper_dynamic30s_multiplier_single_epoch_fsdp4/run-20260731_084946/swift_output/v0-20260731-085036/checkpoint-46050}"
WARMSTART_GATE="${HUGINN_AUDIO_DYNAMIC30S_ACAV_WARMSTART_GATE:-}"
RUN_ROOT="${HUGINN_AUDIO_DYNAMIC30S_ACAV_FORMAL_RUN_ROOT:-$REPO_ROOT/outputs/huginn_audio_whisper_dynamic30s_acavcaps_formal_fsdp8/run-$(date +%Y%m%d_%H%M%S)}"

MODEL_PATH="$REPO_ROOT/models/huginn-audio-whisper-dynamic90s-v1"
PLUGIN_PATH="$REPO_ROOT/code/huginn_lora/plugins/huginn_audio_whisper_dynamic30s_acavcaps_swift.py"
MANIFEST_INSPECTOR="$REPO_ROOT/code/huginn_lora/scripts/inspect_acavcaps_flat_global_tar_manifest.py"
MODULES_TO_SAVE=(temporal_compressor audio_projector audio_boundary_embeddings)

if [ -e "$RUN_ROOT" ]; then
  echo "ACAVCAPS formal run root already exists: $RUN_ROOT" >&2
  exit 1
fi
if [ ! -s "$MANIFEST" ]; then
  echo "ACAVCAPS flat manifest is missing or empty: $MANIFEST" >&2
  exit 1
fi
if [ ! -s "${MANIFEST%.json}.stats.json" ]; then
  echo "ACAVCAPS flat manifest stats are missing or empty: ${MANIFEST%.json}.stats.json" >&2
  exit 1
fi
if [ ! -d "$INIT_CHECKPOINT/pytorch_model_fsdp_0" ]; then
  echo "Model-only warm-start DCP directory is missing: $INIT_CHECKPOINT/pytorch_model_fsdp_0" >&2
  exit 1
fi
for required_path in "$MODEL_PATH" "$PLUGIN_PATH" "$MANIFEST_INSPECTOR"; do
  if [ ! -e "$required_path" ]; then
    echo "Required ACAVCAPS formal path is missing: $required_path" >&2
    exit 1
  fi
done

if [ -z "$WARMSTART_GATE" ]; then
  if [ "${HUGINN_AUDIO_DYNAMIC30S_ACAV_ALLOW_UNGATED_FORMAL:-0}" != "1" ]; then
    echo "The 8-card ACAVCAPS warm-start/save/resume smoke gate is required before formal training." >&2
    echo "Set HUGINN_AUDIO_DYNAMIC30S_ACAV_WARMSTART_GATE to the smoke report, or explicitly set" >&2
    echo "HUGINN_AUDIO_DYNAMIC30S_ACAV_ALLOW_UNGATED_FORMAL=1 for a deliberate ungated launch." >&2
    exit 1
  fi
  echo "[formal-warning] warm-start smoke gate bypassed by explicit override" >&2
else
  python - "$WARMSTART_GATE" "$INIT_CHECKPOINT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]).expanduser().resolve()
expected_checkpoint = Path(sys.argv[2]).expanduser().resolve()
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("gate") != "huginn_audio_whisper_dynamic30s_acavcaps_fsdp8_model_only_warmstart_resume_v1":
    raise SystemExit(f"Unexpected warm-start smoke gate: {payload.get('gate')!r}")
if payload.get("validation_passed") is not True or int(payload.get("world_size", -1)) != 8:
    raise SystemExit(f"Warm-start smoke gate is not a passed 8-card report: {path}")
semantics = payload.get("semantics", {})
if semantics.get("phase1_new_state") != "optimizer_scheduler_global_step_rng_data_position":
    raise SystemExit(f"Warm-start smoke semantics mismatch: {semantics}")
warmstart = payload.get("warmstart_report", {})
source = warmstart.get("warmstart_report", {}).get("checkpoint_dir")
if source is not None and Path(source).expanduser().resolve() != expected_checkpoint:
    raise SystemExit(
        f"Warm-start smoke source differs from formal source: smoke={source!r} formal={expected_checkpoint}"
    )
print(f"[acavcaps-formal-preflight] warmstart_smoke_gate=passed path={path}")
PY
fi

read -r TOTAL_SAMPLES TAR_COUNT < <(python - "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(int(payload["sample_count"]), int(payload["tar_count"]))
PY
)
MAX_STEPS=$(((TOTAL_SAMPLES + GLOBAL_BATCH_SIZE - 1) / GLOBAL_BATCH_SIZE))
if [ "$TOTAL_SAMPLES" -ne 4664169 ] || [ "$TAR_COUNT" -ne 1071 ]; then
  echo "Unexpected full ACAVCAPS manifest size: samples=$TOTAL_SAMPLES tars=$TAR_COUNT" >&2
  exit 1
fi
if [ "$CHECKPOINT_INTERVAL" -le 0 ] || [ "$LOGGING_STEPS" -le 0 ]; then
  echo "Checkpoint/logging intervals must be positive" >&2
  exit 1
fi

CHECKPOINT_STEPS=()
step=$CHECKPOINT_INTERVAL
while [ "$step" -lt "$MAX_STEPS" ]; do
  CHECKPOINT_STEPS+=("$step")
  step=$((step + CHECKPOINT_INTERVAL))
done
CHECKPOINT_STEPS+=("$MAX_STEPS")
CHECKPOINT_STEPS_CSV="$(IFS=,; echo "${CHECKPOINT_STEPS[*]}")"

mkdir -p "$RUN_ROOT"
RUN_ROOT="$(cd "$RUN_ROOT" && pwd)"
OUTPUT_DIR="$RUN_ROOT/swift_output"
LOGGING_DIR="$RUN_ROOT/tensorboard"
AUDIT_DIR="$RUN_ROOT/audit"
FSDP_CONFIG_PATH="$RUN_ROOT/fsdp2_acavcaps_formal_fsdp8.json"
PLAN_PATH="$RUN_ROOT/acavcaps_formal_training_plan.json"

python - "$MANIFEST" "$INIT_CHECKPOINT" "$PLAN_PATH" "$TOTAL_SAMPLES" "$TAR_COUNT" "$GLOBAL_BATCH_SIZE" "$MAX_STEPS" "$SEED" "$AUDIO_ENCODER_LR" "$ALIGNER_LR" "$LORA_LR" "$CHECKPOINT_INTERVAL" "$CHECKPOINT_STEPS_CSV" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    manifest_value,
    checkpoint_value,
    output_value,
    total_samples,
    tar_count,
    global_batch,
    max_steps,
    seed,
    audio_lr,
    aligner_lr,
    lora_lr,
    checkpoint_interval,
    checkpoint_steps_csv,
) = sys.argv[1:]
manifest = Path(manifest_value).expanduser().resolve()
digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
plan = {
    "plan_version": "huginn_audio_whisper_dynamic30s_acavcaps_flat_global_fsdp8_v1",
    "dataset": {
        "manifest": str(manifest),
        "manifest_sha256": digest,
        "schedule_policy": "global_tar_order_shuffle_all_stages_v1_per_tar_buffer_shuffle",
        "source_stage_order": ["stage1", "stage2", "stage3"],
        "tar_count": int(tar_count),
        "sample_count": int(total_samples),
        "sample_shuffle_buffer": 512,
        "max_tars": None,
        "public_root_mutation": "forbidden",
    },
    "warmstart": {
        "source_checkpoint": str(Path(checkpoint_value).expanduser().resolve()),
        "semantics": "model_weights_only_new_optimizer_scheduler_global_step_rng_data_position",
        "restored_groups": ["audio_encoder", "aligner", "lora"],
        "skipped_group": "huginn_base",
    },
    "world_size": 8,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "global_batch_size": int(global_batch),
    "max_steps": int(max_steps),
    "coverage_policy": "ceil_sample_count_divided_by_global_batch_with_terminal_partial_batch",
    "seed": int(seed),
    "learning_rates": {
        "audio_encoder": audio_lr,
        "aligner": aligner_lr,
        "lora": lora_lr,
    },
    "lr_scheduler_type": "constant",
    "lora": {"rank": 8, "alpha": 16, "dropout": 0.05},
    "checkpoint_interval": int(checkpoint_interval),
    "checkpoint_steps": [int(value) for value in checkpoint_steps_csv.split(",")],
    "save_total_limit": 2,
    "fsdp": {
        "version": 2,
        "state_dict_type": "SHARDED_STATE_DICT",
        "activation_checkpointing": True,
        "reshard_after_forward": True,
        "recurrent_core_reshard_after_forward": False,
    },
}
output = Path(output_value).expanduser().resolve()
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[acavcaps-formal-plan] path={output}")
print(f"[acavcaps-formal-plan] sample_count={total_samples} tar_count={tar_count} global_batch={global_batch} max_steps={max_steps}")
PY

FSDP_CONFIG='{"fsdp":"full_shard auto_wrap","fsdp_config":{"activation_checkpointing":true,"auto_wrap_policy":"TRANSFORMER_BASED_WRAP","cpu_ram_efficient_loading":true,"fsdp_version":2,"reshard_after_forward":true,"state_dict_type":"SHARDED_STATE_DICT"}}'
printf '%s\n' "$FSDP_CONFIG" > "$FSDP_CONFIG_PATH"
mkdir -p "$OUTPUT_DIR" "$LOGGING_DIR" "$AUDIT_DIR"

export ACAVCAPS_FLAT_MANIFEST="$MANIFEST"
export ACAVCAPS_FLAT_SAMPLE_SHUFFLE_BUFFER=512
unset ACAVCAPS_FLAT_MAX_TARS
export HUGINN_AUDIO_DYNAMIC90S_FSDP2_NONPERSISTENT_ROPE=1
export HUGINN_AUDIO_DYNAMIC90S_TRAIN_CHAIN_AUDIT=1
export HUGINN_AUDIO_DYNAMIC90S_PEFT_ALIGNER_MODULES_TO_SAVE=1
export HUGINN_AUDIO_DYNAMIC90S_FULL_MODEL_DCP=1
export HUGINN_AUDIO_DYNAMIC30S_RECURRENT_CORE_RESHARD_AFTER_FORWARD_FALSE=1
export HUGINN_AUDIO_DYNAMIC30S_RECURRENT_CORE_EXPECTED_WORLD_SIZE="$WORLD_SIZE"
export HUGINN_AUDIO_DYNAMIC90S_MODEL_ONLY_WARMSTART_AUDIT_DIR="$AUDIT_DIR"
export HUGINN_AUDIO_DYNAMIC90S_MODEL_ONLY_WARMSTART_EXPECTED_WORLD_SIZE="$WORLD_SIZE"
export HUGINN_AUDIO_DYNAMIC90S_MODEL_ONLY_WARMSTART_AUDIT_MODE=fresh
export HUGINN_AUDIO_DYNAMIC90S_MODEL_ONLY_WARMSTART_EXPECTED_START_STEP=0
export HUGINN_AUDIO_DYNAMIC90S_MODEL_ONLY_WARMSTART_PHASE=acavcaps_formal
export HUGINN_AUDIO_DYNAMIC90S_INIT_FSDP_DCP_CHECKPOINT="$INIT_CHECKPOINT"
export HUGINN_AUDIO_DYNAMIC30S_EXPECTED_AUDIO_ENCODER_LR="$AUDIO_ENCODER_LR"
export HUGINN_AUDIO_DYNAMIC30S_EXPECTED_ALIGNER_LR="$ALIGNER_LR"
export HUGINN_AUDIO_DYNAMIC30S_EXPECTED_LORA_LR="$LORA_LR"
unset HUGINN_AUDIO_DYNAMIC90S_INIT_ALIGNER_CHECKPOINT
unset HUGINN_AUDIO_DYNAMIC90S_CHECKPOINT_AUDIT_DIR
unset HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_DIR
unset HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_PHASE

echo "========== ACAVCAPS FLAT GLOBAL MANIFEST FORMAL PREFLIGHT =========="
python -u "$MANIFEST_INSPECTOR" \
  --manifest "$MANIFEST" \
  --world_size "$WORLD_SIZE" \
  --per_device_batch_size "$PER_DEVICE_BATCH" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --check_tar_files

echo "========== HUGINN WHISPER DYNAMIC30S ACAVCAPS FORMAL FSDP8 START =========="
echo "dataset=$MANIFEST"
echo "dataset_scope=all_1071_tars_one_global_stage_flattened_per_tar_shuffle_buffer_512"
echo "dataset_samples=$TOTAL_SAMPLES nominal_full_pass_updates=$MAX_STEPS"
echo "warmstart_checkpoint=$INIT_CHECKPOINT"
echo "warmstart=model_weights_only restored=whisper+aligner+66_lora fresh=optimizer+scheduler+global_step+rng+data_position"
echo "world_size=$WORLD_SIZE per_device_batch=$PER_DEVICE_BATCH gradient_accumulation=$GRADIENT_ACCUMULATION_STEPS global_batch=$GLOBAL_BATCH_SIZE"
echo "learning_rates=audio_encoder:$AUDIO_ENCODER_LR aligner:$ALIGNER_LR lora:$LORA_LR"
echo "lr_scheduler_type=constant lora_rank=8 lora_alpha=16 lora_dropout=0.05"
echo "trainable=whisper+aligner+huginn_lora frozen=huginn_native_backbone"
echo "fsdp=version2_full_shard_state_dict=SHARDED_STATE_DICT activation_checkpointing=true recurrent_core_reshard_after_forward=false"
echo "checkpoints=$CHECKPOINT_STEPS_CSV save_total_limit=2"
echo "acavcaps_flat_max_tars=<unset>"
echo "formal_plan=$PLAN_PATH"

swift sft \
  --model "$MODEL_PATH" \
  --model_type huginn_audio_whisper_dynamic90s \
  --template huginn_audio_whisper_dynamic90s \
  --external_plugins "$PLUGIN_PATH" \
  --dataset huginn_audio_whisper_dynamic30s_acavcaps \
  --streaming true \
  --dataset_shuffle false \
  --train_dataloader_shuffle false \
  --sortish_sampler false \
  --group_by_length false \
  --max_length 192 \
  --output_dir "$OUTPUT_DIR" \
  --logging_dir "$LOGGING_DIR" \
  --tuner_type lora_llm \
  --freeze_vit false \
  --freeze_aligner false \
  --modules_to_save "${MODULES_TO_SAVE[@]}" \
  --learning_rate "$LORA_LR" \
  --aligner_lr "$ALIGNER_LR" \
  --vit_lr "$AUDIO_ENCODER_LR" \
  --lr_scheduler_type constant \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --adapters "$INIT_CHECKPOINT" \
  --load_args false \
  --fsdp "$FSDP_CONFIG_PATH" \
  --max_steps "$MAX_STEPS" \
  --per_device_train_batch_size "$PER_DEVICE_BATCH" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --gradient_checkpointing false \
  --vit_gradient_checkpointing false \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --logging_steps "$LOGGING_STEPS" \
  --save_strategy steps \
  --save_steps "$CHECKPOINT_INTERVAL" \
  --save_total_limit 2 \
  --dataloader_num_workers 0 \
  --dataloader_pin_memory false \
  --dataset_num_proc 1 \
  --save_only_model false \
  --report_to "$REPORT_TO" \
  --bf16 true \
  --seed "$SEED" \
  --data_seed "$SEED"

FINAL_CHECKPOINT="$OUTPUT_DIR/checkpoint-$MAX_STEPS"
if [ ! -d "$FINAL_CHECKPOINT/pytorch_model_fsdp_0" ]; then
  echo "Formal ACAVCAPS final checkpoint is missing: $FINAL_CHECKPOINT" >&2
  exit 1
fi
for required_file in \
  "$FINAL_CHECKPOINT/trainer_state.json" \
  "$FINAL_CHECKPOINT/model_only_warmstart.json" \
  "$FINAL_CHECKPOINT/huginn_training_runtime_contract.json"; do
  if [ ! -s "$required_file" ]; then
    echo "Formal ACAVCAPS final checkpoint audit file is missing: $required_file" >&2
    exit 1
  fi
done

echo "========== HUGINN WHISPER DYNAMIC30S ACAVCAPS FORMAL FSDP8 PASSED =========="
echo "final_checkpoint=$FINAL_CHECKPOINT"
echo "formal_plan=$PLAN_PATH"
echo "audit_dir=$AUDIT_DIR"
