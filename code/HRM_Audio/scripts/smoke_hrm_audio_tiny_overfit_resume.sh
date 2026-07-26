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

SOURCE_MANIFEST="${HRM_AUDIOCAPS_TRAIN_MANIFEST:-$REPO_ROOT/data/audio_swift/audiocaps_v2/audiocaps_v2_train_hrm_audio.jsonl}"
SOURCE_STATS="${HRM_AUDIOCAPS_TRAIN_STATS:-$SOURCE_MANIFEST.stats.json}"
RUN_TAG="${HRM_AUDIO_TINY_RESUME_RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${HRM_AUDIO_TINY_RESUME_RUN_ROOT:-$REPO_ROOT/outputs/hrm_text/audio_tiny_overfit_resume/$RUN_TAG}"
FIXTURE_MANIFEST="$RUN_ROOT/audiocaps_v2_tiny_overfit_b32.jsonl"
PREPARE_REPORT="$RUN_ROOT/fixture_prepare_report.json"
PHASE1_OUTPUT="$RUN_ROOT/pre_resume_swift_output"
PHASE2_OUTPUT="$RUN_ROOT/resume_swift_output"
BOUNDARY_REPORT="$RUN_ROOT/resume_boundary_report.json"
RUNTIME_REPORT="$RUN_ROOT/resume_runtime_report.json"
FINAL_REPORT="${HRM_AUDIO_TINY_RESUME_OUTPUT_REPORT:-$RUN_ROOT/tiny_overfit_resume_report.json}"

UNIQUE_RECORDS=4
FIXTURE_RECORDS=32
MICRO_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=4
EFFECTIVE_BATCH_SIZE=32
STEP_BEFORE_RESUME=12
STEP_AFTER_RESUME=24
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT=0.0
LEARNING_RATE=1e-4
ALIGNER_LEARNING_RATE=1e-4
MINIMUM_RELATIVE_LOSS_REDUCTION="${HRM_AUDIO_TINY_MIN_RELATIVE_LOSS_REDUCTION:-0.10}"

if [ -e "$RUN_ROOT" ]; then
  echo "Run root already exists; refusing to overwrite: $RUN_ROOT" >&2
  exit 1
fi
mkdir -p "$RUN_ROOT"

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

echo "========== HRM AUDIO TINY-OVERFIT + FRESH RESUME GATE =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "PYTHON=$(which python)"
echo "SWIFT=$(which swift)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SOURCE_MANIFEST=$SOURCE_MANIFEST"
echo "SOURCE_STATS=$SOURCE_STATS"
echo "RUN_ROOT=$RUN_ROOT"
echo "FIXTURE=real AudioCaps-v2 unique=$UNIQUE_RECORDS repeated_records=$FIXTURE_RECORDS"
echo "FORMAL_CONFIG=B${MICRO_BATCH_SIZE}/GA${GRADIENT_ACCUMULATION_STEPS}/effective${EFFECTIVE_BATCH_SIZE}/rank${LORA_RANK}/alpha${LORA_ALPHA}"
echo "LEARNING_RATE=$LEARNING_RATE ALIGNER_LEARNING_RATE=$ALIGNER_LEARNING_RATE"
echo "PHASE1=steps_1_to_${STEP_BEFORE_RESUME}"
echo "PHASE2=fresh_process_resume_${STEP_BEFORE_RESUME}_to_${STEP_AFTER_RESUME}"
echo "MINIMUM_RELATIVE_LOSS_REDUCTION=$MINIMUM_RELATIVE_LOSS_REDUCTION"
echo "TRAINABLE=aligner+H/L_LoRA FROZEN=Whisper+HRM_base"
echo "FRAMEWORK=ms-swift_4.4.2"

python -u code/HRM_Audio/scripts/audit_hrm_audio_tiny_overfit_resume.py prepare \
  --source-manifest "$SOURCE_MANIFEST" \
  --source-stats "$SOURCE_STATS" \
  --fixture-manifest "$FIXTURE_MANIFEST" \
  --output-report "$PREPARE_REPORT" \
  --unique-records "$UNIQUE_RECORDS" \
  --fixture-records "$FIXTURE_RECORDS"

echo "========== HRM AUDIO TINY-OVERFIT PHASE 1: SWIFT SFT =========="
swift sft \
  --model "$REPO_ROOT/models/hrm-text-audio-v1" \
  --model_type hrm_text_audio_whisper \
  --template hrm_text_audio \
  --external_plugins "$REPO_ROOT/code/HRM_Audio/plugins/hrm_text_audio_swift.py" \
  --dataset "$FIXTURE_MANIFEST" \
  --split_dataset_ratio 0 \
  --dataset_shuffle false \
  --train_dataloader_shuffle false \
  --sortish_sampler false \
  --group_by_length false \
  --max_length 192 \
  --output_dir "$PHASE1_OUTPUT" \
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
  --max_steps "$STEP_BEFORE_RESUME" \
  --per_device_train_batch_size "$MICRO_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --gradient_checkpointing false \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps "$STEP_BEFORE_RESUME" \
  --save_total_limit 1 \
  --save_only_model false \
  --dataloader_num_workers 0 \
  --dataloader_pin_memory false \
  --dataset_num_proc 1 \
  --lazy_tokenize false \
  --seed 42 \
  --data_seed 42 \
  --optim adamw_torch \
  --attn_impl sdpa \
  --bf16 true \
  --report_to none

CHECKPOINT_BEFORE="$(find_checkpoint "$PHASE1_OUTPUT" "checkpoint-$STEP_BEFORE_RESUME")"
echo "[phase1] checkpoint_before_resume=$CHECKPOINT_BEFORE"

echo "========== HRM AUDIO TINY-OVERFIT PHASE 2: FRESH SWIFT RESUME =========="
python -u code/HRM_Audio/scripts/resume_hrm_audio_tiny_overfit_swift.py \
  --wrapper-model-path "$REPO_ROOT/models/hrm-text-audio-v1" \
  --plugin-path "$REPO_ROOT/code/HRM_Audio/plugins/hrm_text_audio_swift.py" \
  --dataset "$FIXTURE_MANIFEST" \
  --resume-from-checkpoint "$CHECKPOINT_BEFORE" \
  --output-dir "$PHASE2_OUTPUT" \
  --boundary-report "$BOUNDARY_REPORT" \
  --runtime-report "$RUNTIME_REPORT" \
  --step-before-resume "$STEP_BEFORE_RESUME" \
  --step-after-resume "$STEP_AFTER_RESUME" \
  --micro-batch-size "$MICRO_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --lora-rank "$LORA_RANK" \
  --lora-alpha "$LORA_ALPHA" \
  --lora-dropout "$LORA_DROPOUT" \
  --learning-rate "$LEARNING_RATE" \
  --aligner-learning-rate "$ALIGNER_LEARNING_RATE"

CHECKPOINT_AFTER="$(find_checkpoint "$PHASE2_OUTPUT" "checkpoint-$STEP_AFTER_RESUME")"
echo "[phase2] checkpoint_after_resume=$CHECKPOINT_AFTER"

python -u code/HRM_Audio/scripts/audit_hrm_audio_tiny_overfit_resume.py audit \
  --fixture-manifest "$FIXTURE_MANIFEST" \
  --prepare-report "$PREPARE_REPORT" \
  --checkpoint-before-resume "$CHECKPOINT_BEFORE" \
  --checkpoint-after-resume "$CHECKPOINT_AFTER" \
  --resume-boundary-report "$BOUNDARY_REPORT" \
  --resume-runtime-report "$RUNTIME_REPORT" \
  --output-report "$FINAL_REPORT" \
  --step-before-resume "$STEP_BEFORE_RESUME" \
  --step-after-resume "$STEP_AFTER_RESUME" \
  --lora-rank "$LORA_RANK" \
  --lora-alpha "$LORA_ALPHA" \
  --lora-dropout "$LORA_DROPOUT" \
  --learning-rate "$LEARNING_RATE" \
  --minimum-relative-loss-reduction "$MINIMUM_RELATIVE_LOSS_REDUCTION"

echo "========== HRM AUDIO TINY-OVERFIT + FRESH RESUME PASSED =========="
echo "checkpoint_before_resume=$CHECKPOINT_BEFORE"
echo "checkpoint_after_resume=$CHECKPOINT_AFTER"
echo "output_report=$FINAL_REPORT"
