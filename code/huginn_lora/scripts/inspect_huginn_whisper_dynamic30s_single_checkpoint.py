#!/usr/bin/env python3
"""Audit one current Whisper full-model FSDP checkpoint without a resume pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from inspect_huginn_whisper_dynamic30s_smoke_fsdp_checkpoints import strict_optimizer_coverage
from inspect_huginn_whisper_dynamic90s_fsdp_checkpoints import inspect_checkpoint, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--phase", default="multiplier_formal_checkpoint")
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--require-formal-training", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return payload


def main() -> None:
    args = parse_args()
    if args.step <= 0 or args.world_size <= 0:
        raise ValueError("step and world-size must be positive")
    checkpoint = args.checkpoint.expanduser().resolve()
    inspected = inspect_checkpoint(checkpoint, args.step, args.world_size, args.phase)
    optimizer_coverage = strict_optimizer_coverage(inspected)
    formal_training = inspected["training_runtime_contract"].get("formal_training")
    if args.require_formal_training:
        if not isinstance(formal_training, dict):
            raise RuntimeError(f"Formal-training contract is missing at {checkpoint}")
        if int(formal_training.get("checkpoint_step", -1)) != args.step:
            raise RuntimeError(
                "Formal-training checkpoint step mismatch: "
                f"expected={args.step} actual={formal_training.get('checkpoint_step')}"
            )
    report = {
        "gate": "huginn_whisper_dynamic30s_single_full_model_fsdp_checkpoint_v1",
        "validation_passed": True,
        "checkpoint": {
            key: value
            for key, value in inspected.items()
            if key not in {"state_metadata", "grouped_keys"}
        },
        "strict_optimizer_coverage": optimizer_coverage,
        "formal_training": formal_training,
    }
    write_json_atomic(args.output_report, report)
    print(f"[report] path={args.output_report.resolve()}")
    print("========== HUGINN WHISPER SINGLE FULL-MODEL CHECKPOINT AUDIT PASSED ==========")


if __name__ == "__main__":
    main()
