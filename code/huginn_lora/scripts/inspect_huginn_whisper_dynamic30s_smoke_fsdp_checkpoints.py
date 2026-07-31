#!/usr/bin/env python3
"""Smoke-only strict optimizer and full-model FSDP checkpoint audit."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from inspect_huginn_whisper_dynamic90s_fsdp_checkpoints import (
    compare_model_states,
    inspect_checkpoint,
    write_json_atomic,
)


TRAINABLE_GROUPS = ("lora", "aligner", "audio_encoder")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-checkpoint", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path, required=True)
    parser.add_argument("--save-step", type=int, default=4)
    parser.add_argument("--resume-step", type=int, default=6)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def strict_optimizer_coverage(checkpoint: dict[str, Any]) -> dict[str, int]:
    directories = checkpoint["optimizer_dcp_dirs"]
    metadata_counts = checkpoint["optimizer_metadata_counts"]
    if len(directories) != 1 or len(metadata_counts) != 1:
        raise RuntimeError(
            f"Strict optimizer audit expected exactly one DCP directory: "
            f"checkpoint={checkpoint['path']} directories={directories} counts={metadata_counts}"
        )
    counts = next(iter(metadata_counts.values()))
    missing = {group: int(counts.get(group, 0)) for group in TRAINABLE_GROUPS if int(counts.get(group, 0)) <= 0}
    if missing or int(counts.get("huginn_base", -1)) != 0:
        raise RuntimeError(
            f"Strict optimizer DCP ownership mismatch at {checkpoint['path']}: "
            f"missing_trainable_groups={missing} counts={counts}"
        )
    return {name: int(value) for name, value in counts.items()}


def main() -> None:
    args = parse_args()
    if args.world_size != 4 or not (0 < args.save_step < args.resume_step):
        raise ValueError("Expected world_size=4 and 0 < save_step < resume_step")
    saved = inspect_checkpoint(
        args.save_checkpoint,
        args.save_step,
        args.world_size,
        "save",
    )
    resumed = inspect_checkpoint(
        args.resume_checkpoint,
        args.resume_step,
        args.world_size,
        "resume",
    )
    save_optimizer_counts = strict_optimizer_coverage(saved)
    resume_optimizer_counts = strict_optimizer_coverage(resumed)
    if save_optimizer_counts != resume_optimizer_counts:
        raise RuntimeError(
            "Optimizer DCP metadata coverage changed across cold resume: "
            f"save={save_optimizer_counts} resume={resume_optimizer_counts}"
        )
    comparison = compare_model_states(saved, resumed)
    report = {
        "gate": "huginn_whisper_dynamic30s_smoke_strict_fsdp_checkpoint_v1",
        "validation_passed": True,
        "strict_optimizer_coverage": {
            "save": save_optimizer_counts,
            "resume": resume_optimizer_counts,
            "trainable_groups_present": list(TRAINABLE_GROUPS),
            "frozen_huginn_optimizer_states": 0,
        },
        "save_checkpoint": {
            key: value
            for key, value in saved.items()
            if key not in {"state_metadata", "grouped_keys"}
        },
        "resume_checkpoint": {
            key: value
            for key, value in resumed.items()
            if key not in {"state_metadata", "grouped_keys"}
        },
        "model_comparison": comparison,
    }
    write_json_atomic(args.output_report, report)
    print(
        f"[strict-optimizer-dcp] save={save_optimizer_counts} "
        f"resume={resume_optimizer_counts}"
    )
    print(f"[report] path={args.output_report.resolve()}")
    print("========== HUGINN WHISPER DYNAMIC30S SMOKE STRICT FSDP CHECKPOINT PASSED ==========")


if __name__ == "__main__":
    main()
