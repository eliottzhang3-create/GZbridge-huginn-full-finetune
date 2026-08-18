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
export MASTER_PORT="${MASTER_PORT:-29657}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS="${QWEN3_BAT_COMPILE_THREADS:-2}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export BAT_AUDIO_AUDIT=1
export BAT_MAX_SEQUENCE_LENGTH=176

: "${QWEN3_BAT_COMPILE_DDP_RESUME_CHECKPOINT:?Set QWEN3_BAT_COMPILE_DDP_RESUME_CHECKPOINT}"

MODEL_PATH="${QWEN3_MODEL_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/models/Qwen3-4B-Base}"
PLUGIN_PATH="${QWEN3_BAT_PLUGIN_PATH:-$REPO_ROOT/code/Ouro_audio/plugins/qwen3_bat_spatial_ast_swift.py}"
DATASET="${QWEN3_BAT_COMPILE_DDP_RESUME_DATASET:-/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/manifests/qwen3_bat_compile_ddp_128.jsonl}"
ROOT_DIR="${QWEN3_BAT_COMPILE_DDP_RESUME_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/qwen3/compile_ddp_resume-$(date +%Y%m%d-%H%M%S)}"
REPORT="$ROOT_DIR/audit.json"

case "$ROOT_DIR:$REPORT" in
  /hpc_stor03/public*|*:/hpc_stor03/public*) echo "Refusing public output" >&2; exit 2;;
esac
[[ -d "$QWEN3_BAT_COMPILE_DDP_RESUME_CHECKPOINT" ]] || { echo "Missing resume checkpoint: $QWEN3_BAT_COMPILE_DDP_RESUME_CHECKPOINT" >&2; exit 2; }
[[ -f "$QWEN3_BAT_COMPILE_DDP_RESUME_CHECKPOINT/trainer_state.json" ]] || { echo "Missing trainer_state.json" >&2; exit 2; }
[[ -f "$DATASET" ]] || { echo "Missing dataset: $DATASET" >&2; exit 2; }
[[ ! -e "$ROOT_DIR" ]] || { echo "Refusing to overwrite: $ROOT_DIR" >&2; exit 2; }

SOURCE_STEP="$(python - "$QWEN3_BAT_COMPILE_DDP_RESUME_CHECKPOINT/trainer_state.json" <<'PY'
import json
import sys
from pathlib import Path

state = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
step = int(state.get("global_step", -1))
if step < 0:
    raise SystemExit(f"Invalid source global_step: {step}")
print(step)
PY
)"
TARGET_STEP=$((SOURCE_STEP + 1))

echo "========== QWEN3 BAT COMPILE DDP RESUME SMOKE =========="
echo "source_checkpoint=$QWEN3_BAT_COMPILE_DDP_RESUME_CHECKPOINT"
echo "source_step=$SOURCE_STEP target_step=$TARGET_STEP remaining_optimizer_steps=1"
echo "world_size=8 per_device_batch_size=8 global_batch_size=64"
echo "dataset=$DATASET records=128"
echo "dataloader_num_workers_per_rank=${QWEN3_BAT_COMPILE_DATALOADER_NUM_WORKERS:-4}"
echo "inductor_compile_threads_per_rank=$TORCHINDUCTOR_COMPILE_THREADS total_compile_workers=$((8 * TORCHINDUCTOR_COMPILE_THREADS))"
echo "compile_target=Qwen3ForCausalLM.model dynamic=false mode=default"

torchrun --standalone --nproc_per_node=8 \
  code/Ouro_audio/bat/scripts/smoke_qwen3_bat_ddp.py \
  --model-path "$MODEL_PATH" \
  --plugin-path "$PLUGIN_PATH" \
  --dataset "$DATASET" \
  --output-dir "$ROOT_DIR" \
  --output-report "$REPORT" \
  --expected-records 128 \
  --max-steps "$TARGET_STEP" \
  --save-steps "$TARGET_STEP" \
  --per-device-batch-size 8 \
  --dataloader-num-workers "${QWEN3_BAT_COMPILE_DATALOADER_NUM_WORKERS:-4}" \
  --torch-compile \
  --compile-mode default \
  --no-compile-dynamic \
  --resume-from-checkpoint "$QWEN3_BAT_COMPILE_DDP_RESUME_CHECKPOINT"

python - "$REPORT" "$QWEN3_BAT_COMPILE_DDP_RESUME_CHECKPOINT" "$SOURCE_STEP" "$TARGET_STEP" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
source = str(Path(sys.argv[2]).resolve())
source_step = int(sys.argv[3])
target_step = int(sys.argv[4])
if report.get("status") != "ok":
    raise SystemExit(f"Resume report is not ok: {report.get('status')}")
distributed = report.get("distributed", {})
if int(distributed.get("initial_global_step", -1)) != source_step:
    raise SystemExit(f"Initial step mismatch: {distributed}")
if int(distributed.get("target_global_step", -1)) != target_step:
    raise SystemExit(f"Target step mismatch: {distributed}")
if int(distributed.get("optimizer_steps", -1)) != 1:
    raise SystemExit(f"Expected exactly one resumed optimizer step: {distributed}")
if str(Path(distributed.get("resumed_from_checkpoint", "")).resolve()) != source:
    raise SystemExit(f"Resume source mismatch: {distributed.get('resumed_from_checkpoint')!r}")
for item in distributed.get("rank_reports", []):
    compile_report = item.get("compile", {})
    if not compile_report.get("requested") or compile_report.get("dynamic"):
        raise SystemExit(f"Invalid compile contract: {compile_report}")
    if int(compile_report.get("unique_graphs", 0)) <= 0:
        raise SystemExit(f"No compiled graph observed: {compile_report}")
    if compile_report.get("reuse_observation") not in {
        "insufficient_steps_for_in_process_reuse",
        "verified_across_steps",
    }:
        raise SystemExit(f"Unexpected reuse observation: {compile_report}")
checkpoint = report.get("checkpoint", {})
if int(checkpoint.get("global_step", -1)) != target_step:
    raise SystemExit(f"Saved checkpoint step mismatch: {checkpoint}")
print(f"[resume] status=ok source_step={source_step} resumed_step={target_step} remaining_optimizer_steps=1")
PY

echo "========== QWEN3 BAT COMPILE DDP RESUME SMOKE PASSED =========="
echo "report=$REPORT"
