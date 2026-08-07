#!/usr/bin/env python3
"""Strictly audit the 8-card ACAVCAPS model-only warm-start/resume smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from inspect_huginn_whisper_dynamic90s_fsdp_checkpoints import (
    compare_model_states,
    inspect_checkpoint,
    write_json_atomic,
)


TRAINABLE_GROUPS = ("lora", "aligner", "audio_encoder")
EXPECTED_LEARNING_RATES = {
    "whisper_encoder": 1e-5,
    "audio_aligner": 5e-5,
    "huginn_lora": 5e-5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-checkpoint", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path, required=True)
    parser.add_argument("--save-step", type=int, default=2)
    parser.add_argument("--resume-step", type=int, default=3)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--save-phase", default="acavcaps_warmstart")
    parser.add_argument("--resume-phase", default="acavcaps_cold_resume")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return payload


def strict_optimizer_coverage(checkpoint: dict[str, Any]) -> dict[str, int]:
    directories = checkpoint["optimizer_dcp_dirs"]
    metadata_counts = checkpoint["optimizer_metadata_counts"]
    if len(directories) != 1 or len(metadata_counts) != 1:
        raise RuntimeError(
            "ACAVCAPS smoke expects exactly one optimizer DCP directory: "
            f"checkpoint={checkpoint['path']} directories={directories} counts={metadata_counts}"
        )
    counts = next(iter(metadata_counts.values()))
    missing = {
        group: int(counts.get(group, 0))
        for group in TRAINABLE_GROUPS
        if int(counts.get(group, 0)) <= 0
    }
    other_state_count = next(
        iter(checkpoint.get("optimizer_other_state_counts", {}).values()),
        0,
    )
    if missing or int(counts.get("huginn_base", -1)) != 0 or other_state_count != 0:
        raise RuntimeError(
            "ACAVCAPS smoke optimizer ownership mismatch: "
            f"checkpoint={checkpoint['path']} missing={missing} counts={counts} "
            f"other_state_count={other_state_count}"
        )
    return {name: int(value) for name, value in counts.items()}


def audit_warmstart_report(save_checkpoint: Path) -> dict[str, Any]:
    report_path = save_checkpoint / "model_only_warmstart.json"
    report = load_json(report_path)
    if report.get("audit_mode") != "fresh":
        raise RuntimeError(f"Warm-start checkpoint audit is not fresh/model-only mode: {report_path}")
    warmstart = report.get("warmstart_report")
    if not isinstance(warmstart, dict):
        raise RuntimeError(f"Warm-start report is missing: {report_path}")
    if warmstart.get("semantics") != "model_weights_only_new_optimizer_scheduler_global_step_rng_data_position":
        raise RuntimeError(f"Warm-start semantics mismatch: {warmstart}")
    restored = warmstart.get("restored_tensor_counts")
    skipped = warmstart.get("skipped_tensor_counts")
    if not isinstance(restored, dict) or not all(int(restored.get(group, 0)) > 0 for group in TRAINABLE_GROUPS):
        raise RuntimeError(f"Warm-start did not restore all trainable groups: {warmstart}")
    if int(warmstart.get("verified_tensor_count", -1)) != int(warmstart.get("restored_tensor_count", -2)):
        raise RuntimeError(f"Warm-start tensor verification is incomplete: {warmstart}")
    if not isinstance(skipped, dict) or int(skipped.get("huginn_base", 0)) <= 0 or int(skipped.get("other", -1)) != 0:
        raise RuntimeError(f"Warm-start skipped-group contract mismatch: {warmstart}")
    optimizer_groups = report.get("optimizer_group_audit")
    if not isinstance(optimizer_groups, list) or not optimizer_groups:
        raise RuntimeError(f"Warm-start optimizer-group audit is missing: {report_path}")
    expected_rates = {
        "lora": 5e-5,
        "aligner": 5e-5,
        "audio_encoder": 1e-5,
    }
    observed_groups: set[str] = set()
    for group_audit in optimizer_groups:
        if not isinstance(group_audit, dict):
            raise RuntimeError(f"Warm-start optimizer-group audit entry is invalid: {group_audit}")
        counts = group_audit.get("parameter_counts")
        if not isinstance(counts, dict):
            raise RuntimeError(f"Warm-start optimizer-group parameter counts are missing: {group_audit}")
        active = [name for name, count in counts.items() if int(count) > 0]
        if len(active) != 1 or active[0] not in expected_rates:
            raise RuntimeError(f"Warm-start optimizer group mixes/contains invalid parameters: {group_audit}")
        observed_groups.add(active[0])
        expected_rate = expected_rates[active[0]]
        if abs(float(group_audit.get("learning_rate", -1.0)) - expected_rate) > 1e-12:
            raise RuntimeError(f"Warm-start optimizer LR mismatch: {group_audit}")
        if abs(float(group_audit.get("configured_learning_rate", -1.0)) - expected_rate) > 1e-12:
            raise RuntimeError(f"Warm-start configured optimizer LR mismatch: {group_audit}")
    if observed_groups != set(expected_rates):
        raise RuntimeError(
            f"Warm-start optimizer groups are incomplete: observed={sorted(observed_groups)} "
            f"expected={sorted(expected_rates)}"
        )
    return report


def main() -> None:
    args = parse_args()
    if args.world_size != 8 or not (0 < args.save_step < args.resume_step):
        raise ValueError("Expected world_size=8 and 0 < save_step < resume_step")
    saved = inspect_checkpoint(
        args.save_checkpoint,
        args.save_step,
        args.world_size,
        args.save_phase,
        expected_learning_rates=EXPECTED_LEARNING_RATES,
    )
    resumed = inspect_checkpoint(
        args.resume_checkpoint,
        args.resume_step,
        args.world_size,
        args.resume_phase,
        expected_learning_rates=EXPECTED_LEARNING_RATES,
    )
    warmstart_report = audit_warmstart_report(args.save_checkpoint.resolve())
    save_optimizer_counts = strict_optimizer_coverage(saved)
    resume_optimizer_counts = strict_optimizer_coverage(resumed)
    if save_optimizer_counts != resume_optimizer_counts:
        raise RuntimeError(
            "Optimizer DCP ownership changed across the new-run cold resume: "
            f"save={save_optimizer_counts} resume={resume_optimizer_counts}"
        )
    comparison = compare_model_states(saved, resumed)
    report = {
        "gate": "huginn_audio_whisper_dynamic30s_acavcaps_fsdp8_model_only_warmstart_resume_v1",
        "validation_passed": True,
        "world_size": args.world_size,
        "semantics": {
            "phase1": "load_model_weights_only_from_4card_full_model_dcp",
            "phase1_new_state": "optimizer_scheduler_global_step_rng_data_position",
            "phase2": "normal_resume_of_new_8card_smoke_checkpoint",
        },
        "warmstart_report": warmstart_report,
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
    print(f"[report] path={args.output_report.resolve()}")
    print("========== HUGINN AUDIO WHISPER DYNAMIC30S ACAVCAPS FSDP8 WARMSTART/RESUME PASSED ==========")


if __name__ == "__main__":
    main()
