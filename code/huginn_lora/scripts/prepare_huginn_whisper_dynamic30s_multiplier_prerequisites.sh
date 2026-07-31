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

SOURCE_REGISTRY="${HUGINN_MULTIPLIER_SOURCE_REGISTRY:-$REPO_ROOT/data/audio_swift/huginn_whisper_dynamic90s_multitask/v2_dynamic30s/pool_registry.json}"
OUTPUT_ROOT="${HUGINN_MULTIPLIER_OUTPUT_ROOT:-$REPO_ROOT/data/audio_swift/huginn_whisper_dynamic30s_multiplier/v1_gigaspeech_m}"
SEED="${HUGINN_MULTIPLIER_SEED:-20260730}"
REGISTRY="$OUTPUT_ROOT/multiplier_pool_registry.json"
AUDIT_REPORT="$OUTPUT_ROOT/multiplier_pool_audit.json"

echo "========== HUGINN WHISPER DYNAMIC30S MULTIPLIER PREREQUISITES START =========="
echo "source_registry=$SOURCE_REGISTRY"
echo "output_root=$OUTPUT_ROOT"
echo "seed=$SEED"
echo "scope=metadata_only audio_decode=false audio_copy=false model_load=false"

python -u "$SCRIPT_DIR/inspect_huginn_whisper_dynamic30s_multiplier_pool.py" --self-test
python -u "$SCRIPT_DIR/prepare_huginn_whisper_dynamic30s_multiplier_pool.py" \
  --source-registry "$SOURCE_REGISTRY" \
  --output-root "$OUTPUT_ROOT" \
  --seed "$SEED"
python -u "$SCRIPT_DIR/inspect_huginn_whisper_dynamic30s_multiplier_pool.py" \
  --registry "$REGISTRY" \
  --output-report "$AUDIT_REPORT"

echo "========== HUGINN WHISPER DYNAMIC30S MULTIPLIER PREREQUISITES PASSED =========="
echo "registry=$REGISTRY"
echo "audit_report=$AUDIT_REPORT"

