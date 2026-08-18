#!/usr/bin/env python3
"""Checkpoint callback for the two-epoch Stage-III A+B -> C+D+E route."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class Stage3EpochCheckpointCallback(TrainerCallback):
    """Keep resumable periodic checkpoints and protected epoch boundaries.

    Trainer's native ``save_total_limit`` cannot express "retain all stage
    boundaries plus the latest N periodic checkpoints".  This callback asks
    Trainer to save at both kinds of steps and performs the narrow retention
    policy after each save.  All checkpoints are full resumable checkpoints;
    only old *non-boundary* checkpoints are removed.
    """

    def __init__(
        self,
        report_path: Path,
        checkpoint_root: Path,
        resume_checkpoint: Path | None = None,
        boundary_steps: dict[int, int] | None = None,
        periodic_save_steps: int = 2_000,
        max_periodic_checkpoints: int = 3,
    ):
        super().__init__()
        self.report_path = report_path.resolve()
        self.report = json.loads(self.report_path.read_text(encoding="utf-8"))
        if self.report.get("status") != "ok" or self.report.get("route") != "stage3_ab_cde_2epoch":
            raise ValueError(f"Invalid Stage-III route report: {self.report_path}")
        if boundary_steps is None:
            raw_boundaries = self.report.get("epoch_boundary_steps")
            if not isinstance(raw_boundaries, dict) or set(raw_boundaries) != {"1", "2"}:
                raise ValueError(f"Expected epoch boundaries for epochs 1 and 2: {raw_boundaries}")
            self.step_to_epoch = {int(step): int(epoch) for epoch, step in raw_boundaries.items()}
        else:
            if set(boundary_steps.values()) != {1, 2} or any(int(step) <= 0 for step in boundary_steps):
                raise ValueError(f"Invalid actual training epoch boundaries: {boundary_steps}")
            self.step_to_epoch = {int(step): int(epoch) for step, epoch in boundary_steps.items()}
        self.boundary_steps = frozenset(self.step_to_epoch)
        self.checkpoint_root = checkpoint_root.resolve()
        self.resume_checkpoint = resume_checkpoint.resolve() if resume_checkpoint else None
        if periodic_save_steps <= 0:
            raise ValueError(f"periodic_save_steps must be positive, got {periodic_save_steps}")
        if max_periodic_checkpoints < 0:
            raise ValueError(f"max_periodic_checkpoints must be non-negative, got {max_periodic_checkpoints}")
        self.periodic_save_steps = int(periodic_save_steps)
        self.max_periodic_checkpoints = int(max_periodic_checkpoints)
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

    @staticmethod
    def _checkpoint_step(path: Path) -> int | None:
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        return int(match.group(1)) if match else None

    def _is_periodic_step(self, step: int) -> bool:
        return step > 0 and step % self.periodic_save_steps == 0 and step not in self.boundary_steps

    def _write_periodic_marker(self, checkpoint_dir: Path, step: int) -> None:
        marker = {
            "status": "ok",
            "route": "stage3_ab_cde_2epoch",
            "checkpoint_kind": "periodic_resumable",
            "global_step": step,
            "periodic_save_steps": self.periodic_save_steps,
            "max_periodic_checkpoints": self.max_periodic_checkpoints,
            "report": str(self.report_path),
            "checkpoint_dir": str(checkpoint_dir),
            "full_state_expected": True,
        }
        target = checkpoint_dir / "periodic_checkpoint.json"
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(target)

    def _prune_periodic_checkpoints(self) -> None:
        if not self.checkpoint_root.is_dir():
            return
        candidates: list[tuple[int, Path]] = []
        for path in self.checkpoint_root.glob("checkpoint-*"):
            if not path.is_dir():
                continue
            step = self._checkpoint_step(path)
            if step is not None and step not in self.boundary_steps:
                candidates.append((step, path))
        candidates.sort(key=lambda item: item[0])
        remove_count = max(0, len(candidates) - self.max_periodic_checkpoints)
        for _, path in candidates[:remove_count]:
            # Only checkpoint-* directories that are not protected boundary
            # steps are eligible for deletion.
            shutil.rmtree(path)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any) -> TrainerControl:
        del args, kwargs
        step = int(state.global_step)
        if step in self.boundary_steps or self._is_periodic_step(step):
            control.should_save = True
        return control

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any) -> TrainerControl:
        del kwargs
        step = int(state.global_step)
        is_boundary = step in self.boundary_steps
        is_periodic = self._is_periodic_step(step)
        if not is_boundary and not is_periodic:
            return control
        if is_boundary:
            # Every rank observes the save callback and must consider the
            # boundary satisfied when the shared Trainer save completed.
            self.saved_steps.add(step)
        if int(os.environ.get("RANK", "0")) != 0:
            return control
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{step}"
        if not checkpoint_dir.is_dir():
            raise RuntimeError(f"Trainer reported save but checkpoint is missing: {checkpoint_dir}")
        if is_boundary:
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
        else:
            self._write_periodic_marker(checkpoint_dir, step)
        self._prune_periodic_checkpoints()
        return control

    def missing_boundary_steps(self) -> list[int]:
        return sorted(self.boundary_steps - self.saved_steps)
