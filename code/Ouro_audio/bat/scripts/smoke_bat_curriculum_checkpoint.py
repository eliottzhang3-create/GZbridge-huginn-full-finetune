#!/usr/bin/env python3
"""Small, non-training validation of BAT curriculum checkpoint boundaries."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from bat.curriculum import load_report, validate_curriculum_report
from curriculum_checkpoint import CurriculumBoundaryCheckpointCallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum-report", type=Path, required=True)
    parser.add_argument("--global-batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = load_report(args.curriculum_report)
    validate_curriculum_report(report, args.global_batch_size)
    callback = CurriculumBoundaryCheckpointCallback(args.curriculum_report, args.global_batch_size)
    with tempfile.TemporaryDirectory(prefix="bat_curriculum_callback_") as temporary:
        output_dir = Path(temporary)
        trainer_args = SimpleNamespace(output_dir=str(output_dir))
        seen: list[dict[str, object]] = []
        for step, stage in sorted(callback.step_to_stage.items()):
            control = SimpleNamespace(should_save=False)
            state = SimpleNamespace(global_step=step)
            callback.on_step_end(trainer_args, state, control)
            if not control.should_save:
                raise RuntimeError(f"Boundary step {step} did not request a save")
            checkpoint_dir = output_dir / f"checkpoint-{step}"
            checkpoint_dir.mkdir(parents=True, exist_ok=False)
            callback.on_save(trainer_args, state, control)
            marker_path = checkpoint_dir / "curriculum_stage.json"
            if not marker_path.is_file():
                raise RuntimeError(f"Boundary step {step} did not write a marker")
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if marker.get("stage") != stage or int(marker.get("global_step", -1)) != step:
                raise RuntimeError(f"Unexpected marker at step {step}: {marker}")
            seen.append(marker)
        if callback.missing_boundary_steps():
            raise RuntimeError(f"Callback still reports missing steps: {callback.missing_boundary_steps()}")
        print(f"[callback] boundaries={seen}")
        print("========== BAT CURRICULUM CALLBACK SMOKE PASSED ==========")


if __name__ == "__main__":
    main()
