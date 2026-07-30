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

DATA_ROOT="${HUGINN_DYNAMIC90S_FULL_POOL_ROOT:-$REPO_ROOT/data/audio_swift/huginn_whisper_dynamic90s_multitask/v2_dynamic30s}"
REGISTRY="$DATA_ROOT/pool_registry.json"
FULL_REPORT="$DATA_ROOT/full_pool_report.json"
SAMPLER_DIR="$DATA_ROOT/sampler"
SAMPLER_REPORT="$SAMPLER_DIR/mixture_sampler_report.json"
REALDATA_REPORT="$DATA_ROOT/real_data_chain_report.json"
PLAN_PREVIEW="$DATA_ROOT/formal_training_plan_3000h_preview.json"

export HUGINN_DYNAMIC90S_FULL_POOL_ROOT="$DATA_ROOT"
export HUGINN_DYNAMIC90S_POOL_REGISTRY="$REGISTRY"
export HUGINN_DYNAMIC90S_FULL_POOL_REPORT="$FULL_REPORT"
export HUGINN_DYNAMIC90S_SAMPLER_DIR="$SAMPLER_DIR"
export HUGINN_DYNAMIC90S_SAMPLER_REPORT="$SAMPLER_REPORT"
export HUGINN_DYNAMIC90S_REAL_DATA_CHAIN_REPORT="$REALDATA_REPORT"

echo "========== HUGINN WHISPER DYNAMIC30S TRAINING PREREQUISITES START =========="
echo "data_root=$DATA_ROOT"
echo "duration_policy=discard_gt90s_retain_first30s token_rate=160ms chunks_per_sample=1"
echo "scope=metadata_rebuild+sampler_cpu_audit+four_real_audio_decodes"

if [ ! -s "$REGISTRY" ] || [ ! -s "$FULL_REPORT" ]; then
  bash code/huginn_lora/scripts/prepare_huginn_whisper_dynamic90s_full_atomic_pools.sh
else
  python - "$REGISTRY" "$FULL_REPORT" <<'PY'
import json
import sys
from pathlib import Path

registry = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
report = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if registry.get("contract_version") != "huginn_whisper_dynamic30s_data_v2":
    raise SystemExit(f"Existing registry has incompatible contract: {registry.get('contract_version')!r}")
if not report.get("validation_passed") or report.get("gate") != "huginn_whisper_dynamic30s_full_atomic_pools_v2":
    raise SystemExit(f"Existing full-pool report is not a passed dynamic30s gate: {sys.argv[2]}")
print(f"[prerequisite] reuse_passed_full_pool={sys.argv[1]}")
PY
fi

bash code/huginn_lora/scripts/inspect_huginn_whisper_dynamic90s_indexed_mixture.sh
bash code/huginn_lora/scripts/inspect_huginn_whisper_dynamic90s_real_data_chain.sh
python -u code/huginn_lora/scripts/plan_huginn_whisper_dynamic90s_formal_training.py \
  --registry "$REGISTRY" \
  --output "$PLAN_PREVIEW" \
  --seed "${HUGINN_DYNAMIC90S_MIXTURE_SEED:-20260730}" \
  --target-hours 3000 \
  --reserve-ratio 1.05 \
  --step-rounding 100 \
  --world-size 4 \
  --per-device-batch 2 \
  --gradient-accumulation 4

python - "$SAMPLER_REPORT" "$REALDATA_REPORT" <<'PY'
import json
import sys
from pathlib import Path

sampler = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
realdata = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if (
    not sampler.get("validation_passed")
    or sampler.get("gate") != "huginn_whisper_dynamic30s_indexed_mixture_no_replacement_v2"
    or sampler.get("contract_version") != "huginn_whisper_dynamic30s_data_v2"
):
    raise SystemExit(f"Sampler audit did not pass: {sys.argv[1]}")
if (
    not realdata.get("validation_passed")
    or realdata.get("gate") != "huginn_whisper_dynamic30s_real_data_chain_v2"
    or realdata.get("contract_version") != "huginn_whisper_dynamic30s_data_v2"
):
    raise SystemExit(f"Real-data chain did not pass dynamic30s contract: {sys.argv[2]}")
print(f"[prerequisite] sampler_report={sys.argv[1]}")
print(f"[prerequisite] realdata_report={sys.argv[2]}")
PY

echo "formal_plan_preview=$PLAN_PREVIEW"
echo "========== HUGINN WHISPER DYNAMIC30S TRAINING PREREQUISITES PASSED =========="
