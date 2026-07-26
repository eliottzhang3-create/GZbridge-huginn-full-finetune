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

SOURCE_MANIFEST="${HRM_AUDIOCAPS_TRAIN_MANIFEST:-$REPO_ROOT/data/audio_swift/audiocaps_v2/audiocaps_v2_train_swift.jsonl}"
RUN_TAG="${HRM_AUDIO_TRAINER_SMOKE_RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${HRM_AUDIO_TRAINER_SMOKE_RUN_DIR:-$REPO_ROOT/outputs/hrm_text/audio_trainer_smoke/$RUN_TAG}"
OUTPUT_REPORT="${HRM_AUDIO_TRAINER_SMOKE_OUTPUT_REPORT:-$RUN_DIR/trainer_smoke_report.json}"
MICRO_BATCH_SIZE="${HRM_AUDIO_TRAINER_SMOKE_MICRO_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${HRM_AUDIO_TRAINER_SMOKE_GRADIENT_ACCUMULATION_STEPS:-1}"
LORA_RANK="${HRM_AUDIO_TRAINER_SMOKE_LORA_RANK:-8}"
LORA_ALPHA="${HRM_AUDIO_TRAINER_SMOKE_LORA_ALPHA:-16}"
LORA_DROPOUT="${HRM_AUDIO_TRAINER_SMOKE_LORA_DROPOUT:-0}"
LEARNING_RATE="${HRM_AUDIO_TRAINER_SMOKE_LEARNING_RATE:-1e-4}"
ALIGNER_LEARNING_RATE="${HRM_AUDIO_TRAINER_SMOKE_ALIGNER_LEARNING_RATE:-1e-4}"

echo "========== RUN HRM AUDIO REAL AUDIOCAPS TRAINER SMOKE =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "PYTHON=$(which python)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SOURCE_MANIFEST=$SOURCE_MANIFEST"
echo "WRAPPER_MODEL_PATH=$REPO_ROOT/models/hrm-text-audio-v1"
echo "PLUGIN_PATH=$REPO_ROOT/code/HRM_Audio/plugins/hrm_text_audio_swift.py"
echo "RUN_DIR=$RUN_DIR"
echo "OUTPUT_REPORT=$OUTPUT_REPORT"
echo "OFFLINE_MODE=true"
echo "MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE"
echo "GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS"
echo "EFFECTIVE_BATCH_SIZE=$((MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "LORA_RANK=$LORA_RANK LORA_ALPHA=$LORA_ALPHA LORA_DROPOUT=$LORA_DROPOUT"
echo "LEARNING_RATE=$LEARNING_RATE ALIGNER_LEARNING_RATE=$ALIGNER_LEARNING_RATE"
echo "POLICY=AudioCaps-v2-real-audio one-update frozen-Whisper frozen-HRM trainable-aligner trainable-H-L-LoRA"

python -u code/HRM_Audio/scripts/smoke_hrm_audio_swift_trainer.py \
  --wrapper-model-path "$REPO_ROOT/models/hrm-text-audio-v1" \
  --plugin-path "$REPO_ROOT/code/HRM_Audio/plugins/hrm_text_audio_swift.py" \
  --source-manifest "$SOURCE_MANIFEST" \
  --run-dir "$RUN_DIR" \
  --output-report "$OUTPUT_REPORT" \
  --reload-script "$REPO_ROOT/code/HRM_Audio/scripts/reload_hrm_audio_swift_checkpoint.py" \
  --micro-batch-size "$MICRO_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --lora-rank "$LORA_RANK" \
  --lora-alpha "$LORA_ALPHA" \
  --lora-dropout "$LORA_DROPOUT" \
  --learning-rate "$LEARNING_RATE" \
  --aligner-learning-rate "$ALIGNER_LEARNING_RATE"
