#!/usr/bin/env python3
"""Stage-boundary checkpoint callback for a continuous BAT curriculum run."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from bat.curriculum import load_report, validate_curriculum_report


class CurriculumBoundaryCheckpointCallback(TrainerCallback):
    """Request ordinary full Trainer checkpoints at exact global steps.

    The callback does not manually serialize model or optimizer objects.  It
    sets ``control.should_save`` and lets Transformers/Swift perform the
    standard full checkpoint save, which includes adapter weights, optimizer,
    scheduler, trainer state, and RNG state when ``save_only_model=false``.
    After the standard save completes, it writes a small stage marker inside
    the checkpoint directory for unambiguous stage identification.
    """

    def __init__(
        self,
        curriculum_report: Path,
        global_batch_size: int,
        checkpoint_root: Path | None = None,
    ):
        super().__init__()
        self.curriculum_report = curriculum_report.resolve()
        self.report = load_report(self.curriculum_report)
        validate_curriculum_report(self.report, global_batch_size)
        boundary_steps = self.report["boundary_steps"]
        self.step_to_stage = {int(value): str(stage) for stage, value in boundary_steps.items()}
        self.boundary_steps = frozenset(self.step_to_stage)
        self.checkpoint_root = checkpoint_root.resolve() if checkpoint_root is not None else None
        self.saved_steps: set[int] = set()
        self._load_existing_boundary_markers()

    def _load_existing_boundary_markers(self) -> None:
        """Treat previously completed boundary saves as already satisfied.

        This is needed when a job is resumed from a Stage-I or Stage-II
        checkpoint: the resumed Trainer will only execute later steps, but the
        post-training audit must still see the earlier boundary as complete.
        """
        if self.checkpoint_root is None or not self.checkpoint_root.is_dir():
            return
        for step, stage in self.step_to_stage.items():
            marker_path = self.checkpoint_root / f"checkpoint-{step}" / "curriculum_stage.json"
            if not marker_path.is_file():
                continue
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if marker.get("status") == "ok" and marker.get("stage") == stage and int(marker.get("global_step", -1)) == step:
                self.saved_steps.add(step)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        del args, kwargs
        step = int(state.global_step)
        if step in self.boundary_steps:
            control.should_save = True
        return control

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        del kwargs
        step = int(state.global_step)
        if step not in self.boundary_steps:
            return control
        self.saved_steps.add(step)
        process_zero = int(os.environ.get("RANK", "0")) == 0
        if not process_zero:
            return control
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{step}"
        if not checkpoint_dir.is_dir():
            raise RuntimeError(f"Trainer reported on_save but checkpoint is missing: {checkpoint_dir}")
        marker = {
            "status": "ok",
            "stage": self.step_to_stage[step],
            "global_step": step,
            "curriculum_report": str(self.curriculum_report),
            "checkpoint_dir": str(checkpoint_dir),
            "full_state_expected": True,
        }
        marker_path = checkpoint_dir / "curriculum_stage.json"
        temporary = marker_path.with_name(marker_path.name + ".tmp")
        temporary.write_text(json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(marker_path)
        return control

    def missing_boundary_steps(self) -> list[int]:
        return sorted(self.boundary_steps - self.saved_steps)
