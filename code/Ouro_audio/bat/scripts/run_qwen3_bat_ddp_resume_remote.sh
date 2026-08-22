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
export BAT_FIXED_SEQUENCE_LENGTH=false
export BAT_MAX_SEQUENCE_LENGTH=512

SOURCE_MANIFEST="${QWEN3_BAT_DDP_RESUME_SOURCE_MANIFEST:-}"
DATASET="${QWEN3_BAT_DDP_RESUME_DATASET:-}"

MODEL_PATH="${QWEN3_MODEL_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/models/Qwen3-4B-Base}"
PLUGIN_PATH="${QWEN3_BAT_PLUGIN_PATH:-$REPO_ROOT/code/Ouro_audio/plugins/qwen3_bat_spatial_ast_swift.py}"
ROOT_DIR="${QWEN3_BAT_DDP_RESUME_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/qwen3/ddp_resume_smoke_8x3090-$(date +%Y%m%d-%H%M%S)}"
case "$ROOT_DIR" in
  /hpc_stor03/public|/hpc_stor03/public/*) echo "Refusing public output" >&2; exit 2;;
esac
if [[ -e "$ROOT_DIR" ]]; then
  echo "Refusing to overwrite resume smoke root: $ROOT_DIR" >&2
  exit 2
fi
if [[ -n "$SOURCE_MANIFEST" ]]; then
  SMOKE_MANIFEST="$ROOT_DIR/smoke_manifest_128.jsonl"
  python -u code/Ouro_audio/bat/scripts/prepare_bat_stage3_ab_cde_resume_manifest.py \
    --source-manifest "$SOURCE_MANIFEST" \
    --output "$SMOKE_MANIFEST" \
    --records-per-group 64
  DATASET="$SMOKE_MANIFEST"
fi
if [[ -z "$DATASET" ]]; then
  echo "Set QWEN3_BAT_DDP_RESUME_SOURCE_MANIFEST or QWEN3_BAT_DDP_RESUME_DATASET" >&2
  exit 2
fi
FRESH_DIR="$ROOT_DIR/fresh"
RESUMED_DIR="$ROOT_DIR/resumed"
FRESH_REPORT="$FRESH_DIR/audit.json"
RESUMED_REPORT="$RESUMED_DIR/audit.json"
LOCAL_CACHE_ROOT="${BAT_LOCAL_CACHE_ROOT:-/tmp/bat_qwen3_resume_arrow_cache_${USER:-user}_$$}"
LOCAL_ARROW_CACHE="$LOCAL_CACHE_ROOT/datasets"
LOCAL_MODELSCOPE_CACHE="$LOCAL_CACHE_ROOT/modelscope"
PREWARM_REPORT="$ROOT_DIR/arrow_cache_prewarm.json"
case "$LOCAL_CACHE_ROOT" in
  /tmp/*) ;;
  *) echo "Refusing non-local cache path: $LOCAL_CACHE_ROOT" >&2; exit 2;;
esac
export BAT_LOCAL_CACHE_ROOT="$LOCAL_CACHE_ROOT"
export BAT_LOCAL_ARROW_CACHE="$LOCAL_ARROW_CACHE"
export HF_DATASETS_CACHE="$LOCAL_ARROW_CACHE"
export MODELSCOPE_CACHE="$LOCAL_MODELSCOPE_CACHE"
export BAT_ARROW_PREWARM_REPORT="$PREWARM_REPORT"
mkdir -p "$LOCAL_ARROW_CACHE" "$LOCAL_MODELSCOPE_CACHE"
python -u code/Ouro_audio/bat/scripts/prewarm_bat_arrow_cache.py \
  --manifest "$DATASET" \
  --cache-dir "$LOCAL_ARROW_CACHE" \
  --report "$PREWARM_REPORT"

echo "========== QWEN3 BAT STAGE-III DDP CHECKPOINT RESUME SMOKE =========="
echo "world_size=8 per_device_batch_size=8 gradient_accumulation_steps=1 global_batch_size=64 padding=dynamic_batch max_length_ceiling=512"
echo "route=stage3_ab_cde curriculum=false"
echo "phase1=global_step_1 phase2=resume_to_global_step_2"
echo "dataset=$DATASET"
echo "root_dir=$ROOT_DIR"

torchrun --standalone --nproc_per_node=8 \
  code/Ouro_audio/bat/scripts/smoke_qwen3_bat_ddp.py \
  --model-path "$MODEL_PATH" \
  --plugin-path "$PLUGIN_PATH" \
  --dataset "$DATASET" \
  --output-dir "$FRESH_DIR" \
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
  --dataset "$DATASET" \
  --output-dir "$RESUMED_DIR" \
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
