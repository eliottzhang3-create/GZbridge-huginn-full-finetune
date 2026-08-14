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
export MASTER_PORT="${MASTER_PORT:-29523}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${OURO_MODEL_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/models/Ouro-1.4B}"
PLUGIN_PATH="${OURO_BAT_PLUGIN_PATH:-$REPO_ROOT/code/Ouro_audio/plugins/ouro_bat_spatial_ast_swift.py}"
DATASET="${BAT_DDP_RESUME_DATASET:?Set BAT_DDP_RESUME_DATASET to a private 16-record Stage-I manifest}"
ROOT_DIR="${BAT_DDP_RESUME_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/bat/ddp_resume_smoke_8x5090-$(date +%Y%m%d-%H%M%S)}"
FRESH_DIR="$ROOT_DIR/fresh"
RESUME_DIR="$ROOT_DIR/resumed"
FRESH_REPORT="$FRESH_DIR/audit.json"
RESUME_REPORT="$RESUME_DIR/audit.json"

case "$ROOT_DIR" in
  /hpc_stor03/public|/hpc_stor03/public/*) echo "Refusing public output" >&2; exit 2;;
esac
if [[ -e "$ROOT_DIR" ]]; then
  echo "Refusing to overwrite resume smoke root: $ROOT_DIR" >&2
  exit 2
fi

echo "========== BAT OURO DDP CHECKPOINT RESUME SMOKE =========="
echo "world_size=8 per_device_batch_size=2 gradient_accumulation_steps=1 global_batch_size=16"
echo "phase1=target_global_step_1 phase2=resume_to_global_step_2"
echo "dataset=$DATASET"
echo "root_dir=$ROOT_DIR"

torchrun --standalone --nproc_per_node=8 \
  code/Ouro_audio/bat/scripts/smoke_bat_ouro_ddp.py \
  --model-path "$MODEL_PATH" \
  --plugin-path "$PLUGIN_PATH" \
  --dataset "$DATASET" \
  --output-dir "$FRESH_DIR" \
  --output-report "$FRESH_REPORT" \
  --expected-records 16 \
  --max-steps 1 \
  --save-steps 1

SOURCE_CHECKPOINT="$(python - "$FRESH_REPORT" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("status") != "ok":
    raise SystemExit("Fresh resume-smoke phase did not pass")
checkpoint = report.get("checkpoint", {})
if int(checkpoint.get("global_step", -1)) != 1:
    raise SystemExit(f"Fresh phase checkpoint step mismatch: {checkpoint}")
print(checkpoint["path"])
PY
)"

if [[ ! -d "$SOURCE_CHECKPOINT" ]]; then
  echo "Fresh checkpoint directory is missing: $SOURCE_CHECKPOINT" >&2
  exit 1
fi
echo "[resume-source] $SOURCE_CHECKPOINT"

torchrun --standalone --nproc_per_node=8 \
  code/Ouro_audio/bat/scripts/smoke_bat_ouro_ddp.py \
  --model-path "$MODEL_PATH" \
  --plugin-path "$PLUGIN_PATH" \
  --dataset "$DATASET" \
  --output-dir "$RESUME_DIR" \
  --output-report "$RESUME_REPORT" \
  --expected-records 16 \
  --max-steps 2 \
  --save-steps 2 \
  --resume-from-checkpoint "$SOURCE_CHECKPOINT"

python - "$RESUME_REPORT" "$SOURCE_CHECKPOINT" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
source = str(Path(sys.argv[2]).resolve())
distributed = report.get("distributed", {})
if report.get("status") != "ok":
    raise SystemExit("Resumed phase did not pass")
if int(distributed.get("target_global_step", -1)) != 2:
    raise SystemExit(f"Unexpected resumed target step: {distributed}")
if int(distributed.get("initial_global_step", -1)) != 1:
    raise SystemExit(f"Unexpected resumed initial step: {distributed}")
if int(distributed.get("optimizer_steps", -1)) != 1:
    raise SystemExit(f"Resume did not execute exactly one remaining step: {distributed}")
if str(distributed.get("resumed_from_checkpoint", "")) != source:
    raise SystemExit(f"Resume source mismatch: {distributed.get('resumed_from_checkpoint')!r} != {source!r}")
print("[resume] status=ok source_step=1 resumed_step=2 remaining_optimizer_steps=1")
PY

echo "========== BAT OURO DDP CHECKPOINT RESUME SMOKE PASSED =========="
echo "fresh_report=$FRESH_REPORT"
echo "resumed_report=$RESUME_REPORT"
