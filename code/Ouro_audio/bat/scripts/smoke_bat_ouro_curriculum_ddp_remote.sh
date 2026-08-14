#!/usr/bin/env bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_ouro"
REPO_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/code/Ouro_audio:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

STAGE1="${BAT_STAGE1_MANIFEST:?Set BAT_STAGE1_MANIFEST}"
STAGE2="${BAT_STAGE2_MANIFEST:?Set BAT_STAGE2_MANIFEST}"
STAGE3="${BAT_STAGE3_MANIFEST:?Set BAT_STAGE3_MANIFEST}"
ROOT="${BAT_CURRICULUM_SMOKE_ROOT:?Set BAT_CURRICULUM_SMOKE_ROOT to a private output root}"
MODEL_PATH="${OURO_MODEL_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/models/Ouro-1.4B}"
PLUGIN_PATH="${OURO_BAT_PLUGIN_PATH:-$REPO_ROOT/code/Ouro_audio/plugins/ouro_bat_spatial_ast_swift.py}"
MANIFEST="$ROOT/manifest.jsonl"
REPORT="$ROOT/curriculum_report.json"
TRAIN_OUTPUT="$ROOT/train"
mkdir -p "$ROOT"
case "$ROOT" in /hpc_stor03/public|/hpc_stor03/public/*) echo "Refusing public output" >&2; exit 2;; esac

python -u code/Ouro_audio/bat/scripts/compose_bat_curriculum_manifest.py \
  --stage1-manifest "$STAGE1" --stage2-manifest "$STAGE2" --stage3-manifest "$STAGE3" \
  --output "$MANIFEST" --report "$REPORT" --global-batch-size 16 \
  --limit-per-stage 16 --allow-count-drift
python -u code/Ouro_audio/bat/scripts/audit_bat_curriculum_manifest.py \
  --manifest "$MANIFEST" --report "$REPORT" \
  --output-report "$ROOT/manifest_audit.json" --global-batch-size 16

torchrun --standalone --nproc_per_node=8 \
  code/Ouro_audio/bat/scripts/train_bat_ouro_curriculum.py \
  --model-path "$MODEL_PATH" --plugin-path "$PLUGIN_PATH" \
  --dataset "$MANIFEST" --curriculum-report "$REPORT" \
  --output-dir "$TRAIN_OUTPUT" --world-size 8 --gradient-accumulation-steps 1

python - "$TRAIN_OUTPUT" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_candidates = sorted(
    path for path in root.iterdir()
    if path.is_dir() and (path / "checkpoint-7").is_dir()
)
if len(run_candidates) != 1:
    raise SystemExit(f"expected one Swift run directory containing checkpoint-7 under {root}, found: {run_candidates}")
root = run_candidates[0]
print(f"[checkpoint-audit] effective_run_dir={root}")
expected = {2: "I", 4: "II", 7: "III"}
required_any = {
    "model": ("adapter_model.safetensors", "pytorch_model.bin", "model.safetensors"),
    "optimizer": ("optimizer.pt", "optimizer.bin"),
    "scheduler": ("scheduler.pt",),
    "trainer": ("trainer_state.json",),
}
for step, stage in expected.items():
    checkpoint = root / f"checkpoint-{step}"
    if not checkpoint.is_dir():
        raise SystemExit(f"missing checkpoint: {checkpoint}")
    marker = json.loads((checkpoint / "curriculum_stage.json").read_text(encoding="utf-8"))
    if marker.get("stage") != stage or int(marker.get("global_step", -1)) != step:
        raise SystemExit(f"bad marker: {checkpoint}: {marker}")
    for kind, candidates in required_any.items():
        if not any((checkpoint / name).is_file() for name in candidates):
            raise SystemExit(f"missing {kind} state in {checkpoint}: {candidates}")
    rng_files = sorted(
        path.name
        for pattern in ("rng_state_*.pth", "rng_state_*.pt")
        for path in checkpoint.glob(pattern)
        if path.is_file()
    )
    rng_ranks = sorted(
        {
            int(match.group(1))
            for name in rng_files
            if (match := re.fullmatch(r"rng_state_(\d+)\.(?:pth|pt)", name)) is not None
        }
    )
    missing_rng = [rank for rank in range(8) if rank not in rng_ranks]
    if missing_rng:
        raise SystemExit(
            f"missing DDP RNG states in {checkpoint}: missing_ranks={missing_rng} present={rng_files}"
        )
print("========== BAT OURO CURRICULUM REAL CALLBACK SMOKE PASSED ==========")
PY
