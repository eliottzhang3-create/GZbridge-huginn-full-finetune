#!/usr/bin/env bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate swift_ouro

REPO_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/code/Ouro_audio:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29637}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

: "${QWEN3_BAT_DDP_RESUME_DATASET:?Set QWEN3_BAT_DDP_RESUME_DATASET to a private 16-record Stage-III JSONL manifest}"

MODEL_PATH="${QWEN3_MODEL_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/models/Qwen3-4B-Base}"
PLUGIN_PATH="${QWEN3_BAT_PLUGIN_PATH:-$REPO_ROOT/code/Ouro_audio/plugins/qwen3_bat_spatial_ast_swift.py}"
ROOT_DIR="${QWEN3_BAT_DDP_RESUME_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/qwen3/ddp_resume_smoke_8x3090-$(date +%Y%m%d-%H%M%S)}"
FRESH_DIR="$ROOT_DIR/fresh"
RESUMED_DIR="$ROOT_DIR/resumed"
FRESH_REPORT="$FRESH_DIR/audit.json"
RESUMED_REPORT="$RESUMED_DIR/audit.json"

case "$ROOT_DIR" in
  /hpc_stor03/public|/hpc_stor03/public/*) echo "Refusing public output" >&2; exit 2;;
esac
if [[ -e "$ROOT_DIR" ]]; then
  echo "Refusing to overwrite resume smoke root: $ROOT_DIR" >&2
  exit 2
fi

echo "========== QWEN3 BAT STAGE-III DDP CHECKPOINT RESUME SMOKE =========="
echo "world_size=8 per_device_batch_size=2 gradient_accumulation_steps=1 global_batch_size=16"
echo "route=stage3_ab_cde curriculum=false"
echo "phase1=global_step_1 phase2=resume_to_global_step_2"
echo "dataset=$QWEN3_BAT_DDP_RESUME_DATASET"
echo "root_dir=$ROOT_DIR"

torchrun --standalone --nproc_per_node=8 \
  code/Ouro_audio/bat/scripts/smoke_qwen3_bat_ddp.py \
  --model-path "$MODEL_PATH" \
  --plugin-path "$PLUGIN_PATH" \
  --dataset "$QWEN3_BAT_DDP_RESUME_DATASET" \
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
    raise SystemExit("Fresh Qwen3 DDP resume phase did not pass")
checkpoint = report.get("checkpoint", {})
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
  code/Ouro_audio/bat/scripts/smoke_qwen3_bat_ddp.py \
  --model-path "$MODEL_PATH" \
  --plugin-path "$PLUGIN_PATH" \
  --dataset "$QWEN3_BAT_DDP_RESUME_DATASET" \
  --output-dir "$RESUMED_DIR" \
  --output-report "$RESUMED_REPORT" \
  --expected-records 16 \
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
    raise SystemExit("Resumed Qwen3 DDP phase did not pass")
distributed = report.get("distributed", {})
if int(distributed.get("target_global_step", -1)) != 2:
    raise SystemExit(f"Unexpected resumed target step: {distributed}")
if int(distributed.get("initial_global_step", -1)) != 1:
    raise SystemExit(f"Unexpected resumed initial step: {distributed}")
if int(distributed.get("optimizer_steps", -1)) != 1:
    raise SystemExit(f"Resume did not execute one remaining optimizer step: {distributed}")
if str(distributed.get("resumed_from_checkpoint", "")) != source:
    raise SystemExit(f"Resume source mismatch: {distributed.get('resumed_from_checkpoint')!r} != {source!r}")
print("[resume] status=ok source_step=1 resumed_step=2 remaining_optimizer_steps=1")
PY

echo "========== QWEN3 BAT STAGE-III DDP CHECKPOINT RESUME SMOKE PASSED =========="
echo "fresh_report=$FRESH_REPORT"
echo "resumed_report=$RESUMED_REPORT"
