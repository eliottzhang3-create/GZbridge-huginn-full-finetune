#!/usr/bin/env python3
"""Checkpoint callback for the two-epoch Stage-III A+B -> C+D+E route."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class Stage3EpochCheckpointCallback(TrainerCallback):
    """Ask Trainer/Swift for full checkpoints at the two epoch boundaries."""

    def __init__(self, report_path: Path, checkpoint_root: Path, resume_checkpoint: Path | None = None):
        super().__init__()
        self.report_path = report_path.resolve()
        self.report = json.loads(self.report_path.read_text(encoding="utf-8"))
        if self.report.get("status") != "ok" or self.report.get("route") != "stage3_ab_cde_2epoch":
            raise ValueError(f"Invalid Stage-III route report: {self.report_path}")
        raw_boundaries = self.report.get("epoch_boundary_steps")
        if not isinstance(raw_boundaries, dict) or set(raw_boundaries) != {"1", "2"}:
            raise ValueError(f"Expected epoch boundaries for epochs 1 and 2: {raw_boundaries}")
        self.step_to_epoch = {int(step): int(epoch) for epoch, step in raw_boundaries.items()}
        self.boundary_steps = frozenset(self.step_to_epoch)
        self.checkpoint_root = checkpoint_root.resolve()
        self.resume_checkpoint = resume_checkpoint.resolve() if resume_checkpoint else None
        self.saved_steps: set[int] = set()
        self._load_existing_markers()
        self._load_resume_marker()

    def _read_marker(self, path: Path) -> dict[str, Any] | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _load_existing_markers(self) -> None:
        if not self.checkpoint_root.is_dir():
            return
        for step, epoch in self.step_to_epoch.items():
            marker = self._read_marker(self.checkpoint_root / f"checkpoint-{step}" / "stage3_epoch.json")
            if marker and marker.get("status") == "ok" and int(marker.get("global_step", -1)) == step and int(marker.get("epoch", -1)) == epoch:
                self.saved_steps.add(step)

    def _load_resume_marker(self) -> None:
        if self.resume_checkpoint is None:
            return
        marker = self._read_marker(self.resume_checkpoint / "stage3_epoch.json")
        if marker and marker.get("status") == "ok":
            step = int(marker.get("global_step", -1))
            if step in self.boundary_steps and int(marker.get("epoch", -1)) == self.step_to_epoch[step]:
                self.saved_steps.add(step)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any) -> TrainerControl:
        del args, kwargs
        if int(state.global_step) in self.boundary_steps:
            control.should_save = True
        return control

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any) -> TrainerControl:
        del kwargs
        step = int(state.global_step)
        if step not in self.boundary_steps:
            return control
        self.saved_steps.add(step)
        if int(os.environ.get("RANK", "0")) != 0:
            return control
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{step}"
        if not checkpoint_dir.is_dir():
            raise RuntimeError(f"Trainer reported save but checkpoint is missing: {checkpoint_dir}")
        marker = {
            "status": "ok",
            "route": "stage3_ab_cde_2epoch",
            "stage": "III",
            "epoch": self.step_to_epoch[step],
            "global_step": step,
            "report": str(self.report_path),
            "checkpoint_dir": str(checkpoint_dir),
            "full_state_expected": True,
        }
        target = checkpoint_dir / "stage3_epoch.json"
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(target)
        return control

    def missing_boundary_steps(self) -> list[int]:
        return sorted(self.boundary_steps - self.saved_steps)
