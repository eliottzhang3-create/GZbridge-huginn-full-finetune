#!/usr/bin/env python3
"""Read-only audit of an already completed curriculum fresh/resume smoke."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


BOUNDARIES = {2: "I", 4: "II", 7: "III"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Smoke root containing train/ and audit JSON files")
    parser.add_argument("--output-report", type=Path, default=None)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--fresh-run-dir", type=Path, default=None)
    parser.add_argument("--resumed-run-dir", type=Path, default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def check_state_files(path: Path, world_size: int) -> dict[str, Any]:
    candidates = {
        "adapter": ("adapter_model.safetensors", "pytorch_model.bin", "model.safetensors"),
        "optimizer": ("optimizer.pt", "optimizer.bin"),
        "scheduler": ("scheduler.pt",),
        "trainer": ("trainer_state.json",),
    }
    present = {
        key: next((name for name in names if (path / name).is_file()), None)
        for key, names in candidates.items()
    }
    issues = [f"missing_{key}" for key, value in present.items() if value is None]
    rng_ranks: set[int] = set()
    for item in path.iterdir():
        match = re.fullmatch(r"rng_state_(\d+)\.(?:pth|pt)", item.name)
        if match and item.is_file():
            rng_ranks.add(int(match.group(1)))
    expected_rng = set(range(world_size))
    if rng_ranks != expected_rng:
        issues.append(f"rng_ranks={sorted(rng_ranks)} expected={sorted(expected_rng)}")
    trainer_state = None
    if present["trainer"] is not None:
        trainer_state = read_json(path / present["trainer"])
        if int(trainer_state.get("global_step", -1)) != int(path.name.split("-", 1)[1]):
            issues.append(
                f"trainer_state_global_step={trainer_state.get('global_step')} "
                f"expected={path.name.split('-', 1)[1]}"
            )
    return {
        "path": str(path),
        "present": present,
        "rng_ranks": sorted(rng_ranks),
        "trainer_global_step": trainer_state.get("global_step") if trainer_state else None,
        "issues": issues,
    }


def check_checkpoint(path: Path, stage: str, world_size: int) -> dict[str, Any]:
    issues: list[str] = []
    marker_path = path / "curriculum_stage.json"
    marker = read_json(marker_path) if marker_path.is_file() else None
    if marker is None:
        issues.append("missing_curriculum_stage_marker")
    else:
        if marker.get("status") != "ok":
            issues.append(f"marker_status={marker.get('status')}")
        if marker.get("stage") != stage:
            issues.append(f"marker_stage={marker.get('stage')} expected={stage}")
        try:
            step = int(path.name.split("-", 1)[1])
        except (IndexError, ValueError):
            step = -1
        if int(marker.get("global_step", -1)) != step:
            issues.append(f"marker_step={marker.get('global_step')} expected={step}")
    state = check_state_files(path, world_size)
    issues.extend(state.pop("issues"))
    return {"stage": stage, "checkpoint": state, "marker": marker, "issues": issues}


def check_compile_report(path: Path) -> dict[str, Any]:
    report = read_json(path)
    issues: list[str] = []
    if report.get("status") != "ok":
        issues.append(f"status={report.get('status')}")
    compile_report = report.get("compile", {})
    if compile_report.get("enabled") is not True:
        issues.append("compile_not_enabled")
    if compile_report.get("target") != "OuroForCausalLM.model":
        issues.append(f"compile_target={compile_report.get('target')}")
    if compile_report.get("dynamic") is not False:
        issues.append(f"compile_dynamic={compile_report.get('dynamic')}")
    counters = report.get("dynamo_counters", {})
    if int(counters.get("unique_graphs", 0)) != 1:
        issues.append(f"unique_graphs={counters.get('unique_graphs')}")
    if int(counters.get("graph_break_count", 0)) != 0:
        issues.append(f"graph_break_count={counters.get('graph_break_count')}")
    batch = report.get("batch_contract", {})
    if int(report.get("global_step", -1)) != 7:
        issues.append(f"audit_global_step={report.get('global_step')}")
    if not batch.get("forward_calls_observed"):
        issues.append("no_forward_calls_observed")
    if any(shape[-1] != 176 for shape in batch.get("input_ids_shapes", []) if shape):
        issues.append("non_176_input_width")
    if any(shape[-1] != 176 for shape in batch.get("labels_shapes", []) if shape):
        issues.append("non_176_label_width")
    if int(batch.get("padding_label_violation_count", 0)) != 0:
        issues.append("padding_labels_contribute_to_loss")
    return {
        "path": str(path),
        "status": report.get("status"),
        "global_step": report.get("global_step"),
        "compile": compile_report,
        "dynamo_counters": counters,
        "speed": {
            "first_step_wall_seconds": report.get("first_step_wall_seconds"),
            "steady_state_step_count": len(report.get("steady_state_step_wall_seconds", [])),
        },
        "batch_contract": batch,
        "issues": issues,
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    train_root = root / "train"
    if not train_root.is_dir():
        raise FileNotFoundError(train_root)

    if args.fresh_run_dir is not None:
        fresh_run = args.fresh_run_dir.resolve()
    else:
        runs = sorted(path for path in train_root.iterdir() if path.is_dir())
        fresh_candidates = [path for path in runs if (path / "checkpoint-7").is_dir()]
        if not fresh_candidates:
            raise FileNotFoundError(f"No completed run with checkpoint-7 below {train_root}")
        fresh_run = fresh_candidates[0]
    if args.resumed_run_dir is not None:
        resumed_run = args.resumed_run_dir.resolve()
    else:
        runs = sorted(path for path in train_root.iterdir() if path.is_dir() and path != fresh_run)
        resumed_candidates = [path for path in runs if (path / "checkpoint-7").is_dir()]
        if not resumed_candidates:
            raise FileNotFoundError(f"No second completed run with checkpoint-7 below {train_root}")
        resumed_run = resumed_candidates[-1]

    issues: list[str] = []
    fresh_checkpoints = {}
    for step, stage in BOUNDARIES.items():
        checkpoint = fresh_run / f"checkpoint-{step}"
        if not checkpoint.is_dir():
            issues.append(f"missing_fresh_checkpoint_{step}")
            continue
        fresh_checkpoints[str(step)] = check_checkpoint(checkpoint, stage, args.world_size)
        issues.extend(f"fresh_checkpoint_{step}:{item}" for item in fresh_checkpoints[str(step)]["issues"])

    # On resume from Stage-I, checkpoint-2 is inherited from the fresh run;
    # the new versioned run is expected to contain the later boundaries.
    resumed_checkpoints = {}
    for step, stage in ((4, "II"), (7, "III")):
        checkpoint = resumed_run / f"checkpoint-{step}"
        if not checkpoint.is_dir():
            issues.append(f"missing_resumed_checkpoint_{step}")
            continue
        resumed_checkpoints[str(step)] = check_checkpoint(checkpoint, stage, args.world_size)
        issues.extend(f"resumed_checkpoint_{step}:{item}" for item in resumed_checkpoints[str(step)]["issues"])

    compile_reports = {}
    for name in ("fresh_compile_audit.json", "resumed_compile_audit.json"):
        path = root / name
        if not path.is_file():
            issues.append(f"missing_{name}")
            continue
        compile_reports[name] = check_compile_report(path)
        issues.extend(f"{name}:{item}" for item in compile_reports[name]["issues"])

    report = {
        "status": "ok" if not issues else "incomplete",
        "root": str(root),
        "fresh_run": str(fresh_run),
        "resumed_run": str(resumed_run),
        "fresh_checkpoints": fresh_checkpoints,
        "resumed_checkpoints": resumed_checkpoints,
        "compile_reports": compile_reports,
        "issues": issues,
        "note": "The resumed run inherits Stage-I marker/checkpoint from fresh_run; it need not duplicate checkpoint-2.",
    }
    output = (args.output_report or (root / "resume_output_audit.json")).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "issues": issues, "report": str(output)}, ensure_ascii=False, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

