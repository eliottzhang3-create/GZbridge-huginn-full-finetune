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
export BAT_MAX_SEQUENCE_LENGTH=176

STAGE1="${BAT_STAGE1_MANIFEST:?Set BAT_STAGE1_MANIFEST}"
STAGE2="${BAT_STAGE2_MANIFEST:?Set BAT_STAGE2_MANIFEST}"
STAGE3="${BAT_STAGE3_MANIFEST:?Set BAT_STAGE3_MANIFEST}"
ROOT="${BAT_COMPILE_SMOKE_ROOT:?Set BAT_COMPILE_SMOKE_ROOT to a private output root}"
MODEL_PATH="${OURO_MODEL_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/models/Ouro-1.4B}"
PLUGIN_PATH="${OURO_BAT_PLUGIN_PATH:-$REPO_ROOT/code/Ouro_audio/plugins/ouro_bat_spatial_ast_swift.py}"
MANIFEST="$ROOT/manifest.jsonl"
REPORT="$ROOT/curriculum_report.json"
TRAIN_OUTPUT="$ROOT/train"
FRESH_AUDIT="$ROOT/fresh_compile_audit.json"
RESUMED_AUDIT="$ROOT/resumed_compile_audit.json"

mkdir -p "$ROOT"
case "$ROOT" in /hpc_stor03/public|/hpc_stor03/public/*) echo "Refusing public output" >&2; exit 2;; esac

python -u code/Ouro_audio/bat/scripts/compose_bat_curriculum_manifest.py \
  --stage1-manifest "$STAGE1" --stage2-manifest "$STAGE2" --stage3-manifest "$STAGE3" \
  --output "$MANIFEST" --report "$REPORT" --global-batch-size 16 \
  --limit-per-stage 16 --allow-count-drift
python -u code/Ouro_audio/bat/scripts/audit_bat_curriculum_manifest.py \
  --manifest "$MANIFEST" --report "$REPORT" \
  --output-report "$ROOT/manifest_audit.json" --global-batch-size 16

run_train() {
  local audit="$1"
  shift
  torchrun --standalone --nproc_per_node=8 \
    code/Ouro_audio/bat/scripts/train_bat_ouro_curriculum.py \
    --model-path "$MODEL_PATH" --plugin-path "$PLUGIN_PATH" \
    --dataset "$MANIFEST" --curriculum-report "$REPORT" \
    --output-dir "$TRAIN_OUTPUT" --world-size 8 --gradient-accumulation-steps 1 \
    --max-sequence-length 176 --torch-compile --compile-mode reduce-overhead \
    --no-compile-dynamic --compile-audit-output "$audit" --logging-steps 1 "$@"
}

run_train "$FRESH_AUDIT"

FRESH_RUN_DIR="$(python - "$TRAIN_OUTPUT" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
candidates = sorted(p for p in root.iterdir() if p.is_dir() and (p / "checkpoint-7").is_dir())
if len(candidates) != 1:
    raise SystemExit(f"expected one fresh run directory with checkpoint-7, found={candidates}")
print(candidates[0])
PY
)"
RESUME_CHECKPOINT="$FRESH_RUN_DIR/checkpoint-2"
test -d "$RESUME_CHECKPOINT"

run_train "$RESUMED_AUDIT" --resume-from-checkpoint "$RESUME_CHECKPOINT"

python - "$ROOT" "$TRAIN_OUTPUT" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
train_root = Path(sys.argv[2])
fresh = json.loads((root / "fresh_compile_audit.json").read_text(encoding="utf-8"))
resumed = json.loads((root / "resumed_compile_audit.json").read_text(encoding="utf-8"))

def check_audit(name, report):
    if report.get("status") != "ok":
        raise SystemExit(f"{name} audit status is not ok: {report}")
    compile_report = report.get("compile", {})
    if not compile_report.get("enabled"):
        raise SystemExit(f"{name} compile was not enabled: {compile_report}")
    if compile_report.get("target") != "OuroForCausalLM.model":
        raise SystemExit(f"{name} wrong compile target: {compile_report}")
    if compile_report.get("dynamic") is not False:
        raise SystemExit(f"{name} dynamic compile is not false: {compile_report}")
    if compile_report.get("outer_multimodal_model_compiled") is not False:
        raise SystemExit(f"{name} outer model was compiled unexpectedly")
    counters = report.get("dynamo_counters", {})
    if int(counters.get("unique_graphs", 0)) != 1:
        raise SystemExit(f"{name} expected exactly one compiled graph: {counters}")
    if int(counters.get("graph_break_count", 0)) != 0:
        raise SystemExit(f"{name} observed graph breaks: {counters}")
    batch = report.get("batch_contract", {})
    if int(report.get("global_step", -1)) != 7:
        raise SystemExit(f"{name} expected final global_step=7: {report.get('global_step')}")
    if not batch.get("forward_calls_observed"):
        raise SystemExit(f"{name} observed no real training forward")
    if any(shape[-1] != 176 for shape in batch.get("input_ids_shapes", [])):
        raise SystemExit(f"{name} found non-static input width: {batch}")
    if any(shape[-1] != 176 for shape in batch.get("labels_shapes", [])):
        raise SystemExit(f"{name} found non-static label width: {batch}")
    if batch.get("padding_label_violation_count", 0) != 0:
        raise SystemExit(f"{name} padding contributes labels: {batch}")
    if any(shape[-2:] != [2, 320000] for shape in batch.get("audio_waveforms_shapes", [])):
        raise SystemExit(f"{name} audio waveform contract failed: {batch}")
    return {
        "global_step": report.get("global_step"),
        "first_step_wall_seconds": report.get("first_step_wall_seconds"),
        "steady_step_count": len(report.get("steady_state_step_wall_seconds", [])),
        "compile_counters": counters,
        "batch_contract": batch,
    }

fresh_summary = check_audit("fresh", fresh)
resumed_summary = check_audit("resumed", resumed)

def checkpoint_files(checkpoint):
    required = {
        "adapter": any((checkpoint / n).is_file() for n in ("adapter_model.safetensors", "pytorch_model.bin", "model.safetensors")),
        "optimizer": any((checkpoint / n).is_file() for n in ("optimizer.pt", "optimizer.bin")),
        "scheduler": (checkpoint / "scheduler.pt").is_file(),
        "trainer": (checkpoint / "trainer_state.json").is_file(),
    }
    if not all(required.values()):
        raise SystemExit(f"incomplete checkpoint {checkpoint}: {required}")
    rng = set()
    for path in checkpoint.glob("rng_state_*.pth"):
        match = re.fullmatch(r"rng_state_(\d+)\.pth", path.name)
        if match:
            rng.add(int(match.group(1)))
    for path in checkpoint.glob("rng_state_*.pt"):
        match = re.fullmatch(r"rng_state_(\d+)\.pt", path.name)
        if match:
            rng.add(int(match.group(1)))
    if rng != set(range(8)):
        raise SystemExit(f"incomplete DDP RNG state in {checkpoint}: {sorted(rng)}")
    return required | {"rng_ranks": sorted(rng)}

fresh_runs = sorted(p for p in train_root.iterdir() if p.is_dir() and (p / "checkpoint-7").is_dir())
if len(fresh_runs) < 2:
    raise SystemExit(f"expected fresh and resumed completed run directories under {train_root}, found={fresh_runs}")
fresh_run = fresh_runs[0]
resumed_run = fresh_runs[-1]
if resumed_run == fresh_run:
    raise SystemExit(f"fresh/resumed run directories are identical: {fresh_run}")
for step, stage in ((2, "I"), (4, "II"), (7, "III")):
    marker = json.loads((fresh_run / f"checkpoint-{step}" / "curriculum_stage.json").read_text(encoding="utf-8"))
    if marker.get("stage") != stage or int(marker.get("global_step", -1)) != step:
        raise SystemExit(f"bad fresh curriculum marker at step {step}: {marker}")
    checkpoint_files(fresh_run / f"checkpoint-{step}")
checkpoint_files(resumed_run / "checkpoint-7")

print(json.dumps({
    "status": "ok",
    "fresh": fresh_summary,
    "resumed": resumed_summary,
    "fresh_run": str(fresh_run),
    "resumed_run": str(resumed_run),
    "resume_source": str(fresh_run / "checkpoint-2"),
    "curriculum_boundaries": {"I": 2, "II": 4, "III": 7},
}, ensure_ascii=False, indent=2))
print("========== BAT OURO CURRICULUM STATIC-COMPILE RESUME SMOKE PASSED ==========")
PY
