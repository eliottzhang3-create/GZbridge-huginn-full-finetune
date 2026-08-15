#!/usr/bin/env bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_ouro"

REPO_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/code/Ouro_audio:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29531}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${OURO_MODEL_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/models/Ouro-1.4B}"
PLUGIN_PATH="${OURO_BAT_PLUGIN_PATH:-$REPO_ROOT/code/Ouro_audio/plugins/ouro_bat_spatial_ast_swift.py}"
SOURCE_MANIFEST="${BAT_STAGE3_AB_CDE_SMOKE_SOURCE_MANIFEST:?Set BAT_STAGE3_AB_CDE_SMOKE_SOURCE_MANIFEST}"
ROOT_DIR="${BAT_STAGE3_AB_CDE_SMOKE_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/bat/stage3_ab_cde_resume_smoke-$(date +%Y%m%d-%H%M%S)}"
SMOKE_MANIFEST="$ROOT_DIR/smoke_manifest_128.jsonl"
TRAIN_DIR="$ROOT_DIR/train"
FRESH_REPORT="$ROOT_DIR/fresh_report.json"
RESUMED_REPORT="$ROOT_DIR/resumed_report.json"

case "$ROOT_DIR" in
  /hpc_stor03/public|/hpc_stor03/public/*) echo "Refusing public output" >&2; exit 2;;
esac
if [[ -e "$ROOT_DIR" ]]; then
  echo "Refusing to overwrite resume smoke root: $ROOT_DIR" >&2
  exit 2
fi

echo "========== BAT OURO STAGE-III SPATIAL-AST/CHECKPOINT/RESUME SMOKE =========="
echo "[source-manifest] $SOURCE_MANIFEST"
echo "[root] $ROOT_DIR"
echo "[contract] world_size=8 per_device_batch_size=8 global_batch_size=64"

python -u code/Ouro_audio/bat/scripts/prepare_bat_stage3_ab_cde_resume_manifest.py \
  --source-manifest "$SOURCE_MANIFEST" \
  --output "$SMOKE_MANIFEST" \
  --records-per-group 64

torchrun --standalone --nproc_per_node=8 \
  code/Ouro_audio/bat/scripts/smoke_bat_ouro_stage3_ab_cde_resume.py \
  --model-path "$MODEL_PATH" \
  --plugin-path "$PLUGIN_PATH" \
  --dataset "$SMOKE_MANIFEST" \
  --output-dir "$TRAIN_DIR" \
  --output-report "$FRESH_REPORT" \
  --expected-records 128 \
  --max-steps 1 \
  --save-steps 1

SOURCE_CHECKPOINT="$(python - "$FRESH_REPORT" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("status") != "ok":
    raise SystemExit(f"Fresh phase failed: {report.get('issues')}")
checkpoint = report.get("checkpoint") or {}
if int(checkpoint.get("global_step", -1)) != 1:
    raise SystemExit(f"Fresh checkpoint step mismatch: {checkpoint}")
print(checkpoint["path"])
PY
)"

if [[ ! -d "$SOURCE_CHECKPOINT" ]]; then
  echo "Fresh checkpoint directory is missing: $SOURCE_CHECKPOINT" >&2
  exit 1
fi
echo "[resume-source] $SOURCE_CHECKPOINT"

torchrun --standalone --nproc_per_node=8 \
  code/Ouro_audio/bat/scripts/smoke_bat_ouro_stage3_ab_cde_resume.py \
  --model-path "$MODEL_PATH" \
  --plugin-path "$PLUGIN_PATH" \
  --dataset "$SMOKE_MANIFEST" \
  --output-dir "$TRAIN_DIR" \
  --output-report "$RESUMED_REPORT" \
  --expected-records 128 \
  --max-steps 2 \
  --save-steps 2 \
  --resume-from-checkpoint "$SOURCE_CHECKPOINT"

python - "$RESUMED_REPORT" "$SOURCE_CHECKPOINT" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
source = str(Path(sys.argv[2]).resolve())
if report.get("status") != "ok":
    raise SystemExit(f"Resumed phase failed: {report.get('issues')}")
distributed = report.get("distributed", {})
if int(distributed.get("initial_global_step", -1)) != 1:
    raise SystemExit(f"Unexpected resume source step: {distributed}")
if int(distributed.get("target_global_step", -1)) != 2:
    raise SystemExit(f"Unexpected resumed target step: {distributed}")
if int(distributed.get("optimizer_steps", -1)) != 1:
    raise SystemExit(f"Resume did not execute one remaining step: {distributed}")
if str(distributed.get("resumed_from_checkpoint", "")) != source:
    raise SystemExit(f"Resume source mismatch: {distributed.get('resumed_from_checkpoint')} != {source}")
print("[resume] status=ok source_step=1 resumed_step=2 remaining_optimizer_steps=1")
PY

echo "========== BAT OURO STAGE-III RESUME SMOKE PASSED =========="
echo "fresh_report=$FRESH_REPORT"
echo "resumed_report=$RESUMED_REPORT"
