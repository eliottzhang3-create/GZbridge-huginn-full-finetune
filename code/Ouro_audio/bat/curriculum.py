"""BAT language-model curriculum planning utilities.

This module is intentionally framework-independent.  It is shared by the
manifest composer, the manifest audit, and the eventual Swift training
launcher so that sample counts, stage boundaries, and scheduler steps cannot
drift between tools.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STAGE_ORDER = ("I", "II", "III")
STAGE_DATASET_DIRS = {
    "I": "stage1-clsdoa",
    "II": "stage2-single",
    "III": "stage3-mixup",
}
STAGE_EPOCHS = {"I": 2, "II": 2, "III": 3}
STAGE_TYPES = {
    "I": {"A", "B"},
    "II": {"A", "B", "C", "D"},
    "III": {"A", "B", "C", "D", "E"},
}


@dataclass(frozen=True)
class CurriculumBlock:
    stage: str
    epoch: int
    source_records: int
    padding_records: int
    written_records: int
    start_record: int
    end_record: int
    start_step: int
    end_step: int


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Expected object at {path}:{line_number}, got {type(item).__name__}")
            records.append(item)
    if not records:
        raise ValueError(f"Manifest is empty: {path}")
    return records


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def steps_for_records(record_count: int, global_batch_size: int) -> int:
    if record_count <= 0 or global_batch_size <= 0:
        raise ValueError("record_count and global_batch_size must be positive")
    if record_count % global_batch_size:
        raise ValueError(
            f"Curriculum block has {record_count} records, which is not divisible by "
            f"global batch size {global_batch_size}"
        )
    return record_count // global_batch_size


def load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Curriculum report must be an object: {path}")
    if payload.get("status") != "ok":
        raise ValueError(f"Curriculum report is not ok: {path}")
    return payload


def validate_curriculum_report(report: dict[str, Any], global_batch_size: int) -> None:
    if int(report.get("global_batch_size", -1)) != global_batch_size:
        raise ValueError(
            f"Curriculum global batch mismatch: report={report.get('global_batch_size')} "
            f"expected={global_batch_size}"
        )
    blocks = report.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("Curriculum report has no blocks")
    expected_sequence = [(stage, epoch) for stage in STAGE_ORDER for epoch in range(1, STAGE_EPOCHS[stage] + 1)]
    observed_sequence = [(str(item.get("stage")), int(item.get("epoch", -1))) for item in blocks]
    if observed_sequence != expected_sequence:
        raise ValueError(f"Unexpected curriculum block sequence: {observed_sequence}")

    previous_record = 0
    previous_step = 0
    for item in blocks:
        for key in ("start_record", "end_record", "start_step", "end_step", "written_records"):
            if key not in item:
                raise ValueError(f"Curriculum block is missing {key}: {item}")
        if int(item["start_record"]) != previous_record or int(item["start_step"]) != previous_step:
            raise ValueError(f"Non-contiguous curriculum block: {item}")
        written_records = int(item["written_records"])
        if written_records <= 0 or written_records % global_batch_size:
            raise ValueError(f"Curriculum block is not batch aligned: {item}")
        expected_end_step = previous_step + written_records // global_batch_size
        if int(item["end_record"]) != previous_record + written_records:
            raise ValueError(f"Curriculum record range mismatch: {item}")
        if int(item["end_step"]) != expected_end_step:
            raise ValueError(f"Curriculum step range mismatch: {item}")
        previous_record = int(item["end_record"])
        previous_step = int(item["end_step"])

    if int(report.get("total_records", -1)) != previous_record:
        raise ValueError("Curriculum total_records does not match block ranges")
    if int(report.get("total_steps", -1)) != previous_step:
        raise ValueError("Curriculum total_steps does not match block ranges")
    boundary_steps = report.get("boundary_steps")
    expected_boundaries = {
        "I": next(item for item in blocks if item["stage"] == "I" and item["epoch"] == 2)["end_step"],
        "II": next(item for item in blocks if item["stage"] == "II" and item["epoch"] == 2)["end_step"],
        "III": previous_step,
    }
    if boundary_steps != expected_boundaries:
        raise ValueError(f"Curriculum boundary_steps mismatch: {boundary_steps} vs {expected_boundaries}")
