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
export HUGINN_LOSATOK_FSDP_SAVE_DEBUG=1
unset HUGINN_LOSATOK_FORCE_ALIGNER_TRAINABLE
unset HUGINN_LOSATOK_INIT_ALIGNER_CHECKPOINT

INIT_CHECKPOINT="${LOSATOK_DYNAMIC_ACAV_WDS_INIT_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_losatok_dynamic90s_audiocaps_v2_e2_b4ga4_fsdp2_complete/v0-20260724-115115/checkpoint-2802}"
export HUGINN_LOSATOK_INIT_FSDP_DCP_CHECKPOINT="$INIT_CHECKPOINT"
export ACAVCAPS_WDS_MANIFEST="${ACAVCAPS_WDS_QUARTER_MANIFEST:-$REPO_ROOT/data/audio_swift/acavcaps_wds/acavcaps_wds_stage_schedule_quarter_ceil_seed20260723.json}"
export ACAVCAPS_WDS_BUFFER_SIZE="${ACAVCAPS_WDS_BUFFER_SIZE:-512}"
# The manifest always describes all 271 quarter tars.  The cap is only for
# this smoke's bounded streaming run; the later formal script must unset it.
export ACAVCAPS_WDS_MAX_TARS_PER_STAGE="${ACAVCAPS_WDS_MAX_TARS_PER_STAGE:-2}"

RUN_ROOT="${LOSATOK_DYNAMIC_ACAV_WDS_QUARTER_SMOKE_ROOT:-outputs/huginn_losatok_acavcaps_wds_dynamic90s_quarter_warmstart2802_fsdp2_save_reload/run-$(date +%Y%m%d_%H%M%S)}"
SAVE_OUTPUT_DIR="$RUN_ROOT/save_phase"
RESUME_OUTPUT_DIR="$RUN_ROOT/resume_phase"
PLUGIN_PATH="$REPO_ROOT/code/huginn_lora/plugins/huginn_losatok_acavcaps_wds_swift.py"
MODEL_PATH="$REPO_ROOT/models/huginn-audio-losatok-v1"
WORLD_SIZE=2
MICRO_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
SAVE_STEPS=2
RESUME_MAX_STEPS=3
MODULES_TO_SAVE=(temporal_compressor audio_projector audio_boundary_embeddings)
FSDP_CONFIG='{"fsdp":"full_shard auto_wrap","fsdp_config":{"activation_checkpointing":false,"auto_wrap_policy":"TRANSFORMER_BASED_WRAP","cpu_ram_efficient_loading":true,"fsdp_version":2,"reshard_after_forward":true,"state_dict_type":"SHARDED_STATE_DICT"}}'

if [ "$ACAVCAPS_WDS_BUFFER_SIZE" != "512" ]; then
  echo "Dynamic quarter smoke requires ACAVCAPS_WDS_BUFFER_SIZE=512, got: $ACAVCAPS_WDS_BUFFER_SIZE" >&2
  exit 1
fi
if ! [[ "$ACAVCAPS_WDS_MAX_TARS_PER_STAGE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ACAVCAPS_WDS_MAX_TARS_PER_STAGE must be a positive integer for this smoke" >&2
  exit 1
fi
if [ -e "$RUN_ROOT" ]; then
  echo "Dynamic ACAVCAPS warm-start smoke root already exists: $RUN_ROOT" >&2
  exit 1
fi
for required_path in \
  "$INIT_CHECKPOINT" \
  "$INIT_CHECKPOINT/pytorch_model_fsdp_0" \
  "$ACAVCAPS_WDS_MANIFEST" \
  "${ACAVCAPS_WDS_MANIFEST%.json}.stats.json" \
  "$PLUGIN_PATH" \
  "$MODEL_PATH" \
  "/hpc_stor03/sjtu_home/jinwei.zhang/models/LoSATok/ckpts/losatok_kl1e-3.pth" \
  "/hpc_stor03/sjtu_home/jinwei.zhang/models/LoSATok/ckpts/semantic_encoder.pth"; do
  if [ ! -e "$required_path" ]; then
    echo "Required dynamic ACAVCAPS warm-start smoke path is missing: $required_path" >&2
    exit 1
  fi
done

python - "$ACAVCAPS_WDS_MANIFEST" <<'PY'
import json
import sys
from dataclasses import fields
from pathlib import Path

manifest = Path(sys.argv[1])
stats = manifest.with_suffix('.stats.json')
payload = json.loads(stats.read_text(encoding='utf-8'))
if payload.get('scan_mode') != 'derived_from_full' or payload.get('all_pairs_valid') is not True:
    raise SystemExit(f'Quarter manifest is not a validated private derivation: {stats}')
if payload.get('tar_count') != 271 or not isinstance(payload.get('sample_count'), int) or payload['sample_count'] <= 0:
    raise SystemExit(f'Unexpected ACAVCAPS quarter stats: {payload}')
from swift.arguments.sft_args import SftArguments
available = {field.name for field in fields(SftArguments)}
required = {'fsdp', 'modules_to_save', 'resume_from_checkpoint', 'adapters', 'load_args', 'save_strategy', 'save_steps', 'save_only_model'}
missing = sorted(required - available)
if missing:
    raise SystemExit(f'Installed Swift lacks required FSDP warm-start arguments: {missing}')
print(f"[preflight] quarter_tar_count={payload['tar_count']} quarter_sample_count={payload['sample_count']}")
print('[preflight] swift_dynamic_warmstart_arguments=present')
PY

audit_checkpoint() {
  python -u code/huginn_lora/scripts/inspect_losatok_dynamic_fsdp_checkpoint.py \
    --checkpoint "$1" \
    --require_complete
}

find_checkpoint() {
  local output_dir=$1
  local checkpoint_name=$2
  mapfile -t matches < <(find "$output_dir" -type d -name "$checkpoint_name" -print | sort)
  if [ "${#matches[@]}" -ne 1 ]; then
    echo "Expected exactly one $checkpoint_name below $output_dir; found ${#matches[@]}" >&2
    printf '  %s\n' "${matches[@]:-<none>}" >&2
    exit 1
  fi
  printf '%s\n' "${matches[0]}"
}

mkdir -p "$SAVE_OUTPUT_DIR" "$RESUME_OUTPUT_DIR"
printf '%s\n' "$FSDP_CONFIG" > "$SAVE_OUTPUT_DIR/fsdp2.json"
printf '%s\n' "$FSDP_CONFIG" > "$RESUME_OUTPUT_DIR/fsdp2.json"

echo "========== ACAVCAPS DYNAMIC LOSATOK QUARTER FSDP2 WARM-START SAVE/RELOAD SMOKE =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "init_semantics=adapter_weight_warm_start_only optimizer_scheduler_rng_global_step_and_data_position=fresh"
echo "init_checkpoint=$INIT_CHECKPOINT"
echo "input_dcp_contract=66_lora+20_aligner"
echo "manifest=$ACAVCAPS_WDS_MANIFEST"
echo "manifest_scope=all_271_quarter_tars stage_order=stage1_to_stage2_to_stage3"
echo "smoke_stream_cap_per_stage=$ACAVCAPS_WDS_MAX_TARS_PER_STAGE per_tar_buffer_shuffle=$ACAVCAPS_WDS_BUFFER_SIZE"
echo "world_size=$WORLD_SIZE per_device_batch=$MICRO_BATCH_SIZE accumulation=$GRADIENT_ACCUMULATION_STEPS global_effective_batch=$((WORLD_SIZE * MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "phase1=warm_start_then_train_to_step_${SAVE_STEPS}_and_save"
echo "phase2=fresh_process_resume_new_acav_checkpoint_to_step_${RESUME_MAX_STEPS}"

echo "========== INPUT CHECKPOINT DCP AUDIT =========="
audit_checkpoint "$INIT_CHECKPOINT"

echo "========== QUARTER MANIFEST CONFIG AUDIT =========="
python -u code/huginn_lora/scripts/inspect_acavcaps_wds_quarter_manifest.py \
  --manifest "$ACAVCAPS_WDS_MANIFEST" \
  --world_size "$WORLD_SIZE" \
  --per_device_batch_size "$MICRO_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"

echo "========== SAVE PHASE: DCP WEIGHT WARM-START =========="
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
  --output_dir "$SAVE_OUTPUT_DIR" \
  --tuner_type lora_llm \
  --freeze_aligner false \
  --modules_to_save "${MODULES_TO_SAVE[@]}" \
  --learning_rate 1e-4 \
  --aligner_lr 1e-4 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --adapters "$INIT_CHECKPOINT" \
  --load_args false \
  --fsdp "$SAVE_OUTPUT_DIR/fsdp2.json" \
  --max_steps "$SAVE_STEPS" \
  --per_device_train_batch_size "$MICRO_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --gradient_checkpointing false \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit 1 \
  --dataloader_num_workers 0 \
  --dataloader_pin_memory false \
  --dataset_num_proc 1 \
  --save_only_model false \
  --report_to none \
  --bf16 true

SAVE_CHECKPOINT="$(find_checkpoint "$SAVE_OUTPUT_DIR" "checkpoint-$SAVE_STEPS")"
echo "========== WARM-STARTED ACAVCAPS CHECKPOINT DCP AUDIT =========="
echo "save_checkpoint=$SAVE_CHECKPOINT"
audit_checkpoint "$SAVE_CHECKPOINT"

echo "========== FRESH FSDP RESUME PHASE =========="
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
  --output_dir "$RESUME_OUTPUT_DIR" \
  --tuner_type lora_llm \
  --freeze_aligner false \
  --modules_to_save "${MODULES_TO_SAVE[@]}" \
  --learning_rate 1e-4 \
  --aligner_lr 1e-4 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --fsdp "$RESUME_OUTPUT_DIR/fsdp2.json" \
  --resume_from_checkpoint "$SAVE_CHECKPOINT" \
  --max_steps "$RESUME_MAX_STEPS" \
  --per_device_train_batch_size "$MICRO_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --gradient_checkpointing false \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps "$RESUME_MAX_STEPS" \
  --save_total_limit 1 \
  --dataloader_num_workers 0 \
  --dataloader_pin_memory false \
  --dataset_num_proc 1 \
  --save_only_model false \
  --report_to none \
  --bf16 true

RESUME_CHECKPOINT="$(find_checkpoint "$RESUME_OUTPUT_DIR" "checkpoint-$RESUME_MAX_STEPS")"
echo "========== RESUMED ACAVCAPS CHECKPOINT DCP AUDIT =========="
echo "resume_checkpoint=$RESUME_CHECKPOINT"
audit_checkpoint "$RESUME_CHECKPOINT"

python - "$SAVE_CHECKPOINT/trainer_state.json" "$RESUME_CHECKPOINT/trainer_state.json" "$SAVE_STEPS" "$RESUME_MAX_STEPS" <<'PY'
import json
import sys
from pathlib import Path

saved = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
resumed = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
expected_saved, expected_resumed = map(int, sys.argv[3:])
if saved.get('global_step') != expected_saved:
    raise SystemExit(f"Warm-start save global_step mismatch: {saved.get('global_step')} != {expected_saved}")
if resumed.get('global_step') != expected_resumed:
    raise SystemExit(f"Fresh resume global_step mismatch: {resumed.get('global_step')} != {expected_resumed}")
print(f"[resume] checkpoint_steps=({expected_saved},{expected_resumed})")
PY

echo "========== ACAVCAPS DYNAMIC LOSATOK QUARTER FSDP2 WARM-START SAVE/RELOAD SMOKE PASSED =========="
echo "input_checkpoint=$INIT_CHECKPOINT"
echo "saved_checkpoint=$SAVE_CHECKPOINT"
echo "resume_checkpoint=$RESUME_CHECKPOINT"
