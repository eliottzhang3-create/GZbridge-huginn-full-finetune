#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1

CHECKPOINT="${CONTINUATION_INIT_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_audio_wavcaps_audioset_sl_e2_warmstart5604_b8ga4_5090/v0-20260715-101351/checkpoint-6754}"
CAPTION_ROOT="${CLOTHO_CAPTION_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_caption_huginn}"
CAPTION_MANIFEST="${CLOTHO_CAPTION_MANIFEST:-train_expand.json}"
AQA_ROOT="${CLOTHO_AQA_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_aqa_huginn}"
AQA_MANIFEST="${CLOTHO_AQA_MANIFEST:-train.jsonl}"
ARTIFACT_DIR="${CLOTHO_CONTINUATION_ARTIFACT_DIR:-$REPO_ROOT/data/audio_swift/continuation}"
CHECKPOINT_REPORT="${CONTINUATION_CHECKPOINT_REPORT:-$ARTIFACT_DIR/wavcaps_checkpoint_6754_inspect.json}"
DATA_REPORT="${CLOTHO_CONTINUATION_INSPECT_REPORT:-$ARTIFACT_DIR/clotho_continuation_inputs_inspect.json}"

mkdir -p "$ARTIFACT_DIR"

echo "========== INSPECT LORA CONTINUATION INPUTS =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "checkpoint=$CHECKPOINT"
echo "caption_root=$CAPTION_ROOT"
echo "caption_manifest=$CAPTION_MANIFEST"
echo "aqa_root=$AQA_ROOT"
echo "aqa_manifest=$AQA_MANIFEST"
echo "checkpoint_report=$CHECKPOINT_REPORT"
echo "data_report=$DATA_REPORT"

set +e
python -u code/huginn_lora/scripts/inspect_swift_huginn_audio_checkpoints.py \
  --checkpoint "$CHECKPOINT" \
  --expected_lora_tensor_count 66 \
  --expected_aligner_tensor_count 20 \
  --require_boundary_embeddings \
  --output_report "$CHECKPOINT_REPORT"
CHECKPOINT_STATUS=$?

python -u code/huginn_lora/scripts/inspect_clotho_huginn_continuation_inputs.py \
  --caption_root "$CAPTION_ROOT" \
  --caption_manifest "$CAPTION_MANIFEST" \
  --aqa_root "$AQA_ROOT" \
  --aqa_manifest "$AQA_MANIFEST" \
  --output_report "$DATA_REPORT"
DATA_STATUS=$?
set -e

echo "========== INSPECT LORA CONTINUATION INPUTS EXIT =========="
echo "checkpoint_status=$CHECKPOINT_STATUS"
echo "data_status=$DATA_STATUS"
if [ "$CHECKPOINT_STATUS" -ne 0 ] || [ "$DATA_STATUS" -ne 0 ]; then
  exit 1
fi
