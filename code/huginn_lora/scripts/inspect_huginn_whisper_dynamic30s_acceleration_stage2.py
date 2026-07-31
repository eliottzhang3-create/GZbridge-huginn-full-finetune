#!/usr/bin/env python3
"""Validate the four-rank recurrent-core no-reshard Stage 2 smoke."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_UNITS = (
    "WhisperEncoderFSDPUnit",
    "AudioAlignerFSDPUnit",
    "HuginnPreludeFSDPUnit",
    "HuginnRecurrentCoreFSDPUnit",
    "HuginnCodaFSDPUnit",
)
CORE_UNIT = "HuginnRecurrentCoreFSDPUnit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--max-allocated-gib", type=float, default=29.0)
    parser.add_argument("--max-reserved-gib", type=float, default=30.0)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing or empty Stage 2 audit: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Stage 2 audit is not an object: {path}")
    return payload


def reshard_value(unit: dict[str, Any], *, rank: int, phase: str, class_name: str) -> bool:
    audit = unit.get("reshard_after_forward", {})
    effective = audit.get("effective")
    candidates = audit.get("candidates", [])
    explicit = {bool(candidate["value"]) for candidate in candidates if "value" in candidate}
    if effective not in (True, False) or explicit != {bool(effective)} or not audit.get("has_fsdp_state"):
        raise ValueError(
            f"Rank {rank} invalid {phase} reshard audit for {class_name}: {audit}"
        )
    return bool(effective)


def validate_optimizer_groups(payload: dict[str, Any], rank: int) -> None:
    audits = payload.get("optimizer_groups")
    if not isinstance(audits, list) or not audits:
        raise ValueError(f"Rank {rank} has no optimizer-group audit")
    observed = {name: 0 for name in ("lora", "aligner", "audio_encoder", "huginn_base", "other")}
    for audit in audits:
        if abs(float(audit.get("learning_rate", -1.0)) - 1e-4) > 1e-12:
            raise ValueError(f"Rank {rank} optimizer LR mismatch: {audits}")
        counts = audit.get("parameter_counts", {})
        for name in observed:
            observed[name] += int(counts.get(name, 0))
    if (
        observed["lora"] != 66
        or observed["aligner"] != 14
        or observed["audio_encoder"] <= 0
        or observed["huginn_base"] != 0
        or observed["other"] != 0
    ):
        raise ValueError(f"Rank {rank} optimizer ownership mismatch: {observed}")


def validate_rank(
    audit_dir: Path,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    fsdp = read_json(audit_dir / f"fsdp-rank-{rank}.json")
    if (
        fsdp.get("kind") != "fsdp"
        or fsdp.get("stage") != "acceleration_stage2"
        or int(fsdp.get("rank", -1)) != rank
        or int(fsdp.get("world_size", -1)) != world_size
        or fsdp.get("valid_prefix_tokens") != [127, 127]
    ):
        raise ValueError(f"Rank {rank} invalid Stage 2 FSDP marker: {fsdp}")
    fsdp_units = fsdp.get("fsdp_units", {})
    if set(fsdp_units) != set(EXPECTED_UNITS):
        raise ValueError(f"Rank {rank} FSDP topology mismatch: {sorted(fsdp_units)}")

    payload = read_json(audit_dir / f"acceleration-stage2-rank-{rank}.json")
    if (
        payload.get("kind") != "acceleration_stage2"
        or payload.get("gate") != "huginn_whisper_dynamic30s_acceleration_stage2_rank_v1"
        or payload.get("stage") != "acceleration_stage2"
        or int(payload.get("rank", -1)) != rank
        or int(payload.get("world_size", -1)) != world_size
        or int(payload.get("global_step", -1)) != 1
        or int(payload.get("per_device_train_batch_size", -1)) != 2
        or int(payload.get("gradient_accumulation_steps", -1)) != 4
        or int(payload.get("global_batch_size", -1)) != 32
    ):
        raise ValueError(f"Rank {rank} invalid Stage 2 runtime marker: {payload}")

    if (
        float(payload.get("synthetic_audio_seconds", -1.0)) != 30.0
        or int(payload.get("expected_audio_tokens", -1)) != 125
        or int(payload.get("expected_prefix_tokens", -1)) != 127
        or payload.get("observed_prefix_tokens") != [127] * 8
    ):
        raise ValueError(f"Rank {rank} exact-30-second prefix contract mismatch: {payload}")
    if int(payload.get("core_forward_calls", 0)) < 4:
        raise ValueError(f"Rank {rank} observed too few recurrent-core calls: {payload}")

    if (
        payload.get("vit_gradient_checkpointing_arg") is not False
        or payload.get("whisper_internal_gradient_checkpointing") is not False
        or payload.get("whisper_outer_activation_checkpointed") is not True
        or payload.get("whisper_double_checkpoint_candidate") is not False
    ):
        raise ValueError(f"Rank {rank} Stage 1 checkpoint contract regressed: {payload}")
    outer = payload.get("whisper_outer_checkpoint_wrappers", [])
    if (
        len(outer) != 1
        or not outer[0].get("path", "").endswith("audio_encoder.encoder")
        or "WhisperEncoder" not in outer[0].get("inner_mro", [])
    ):
        raise ValueError(f"Rank {rank} outer Whisper checkpoint mismatch: {outer}")
    wrappers = payload.get("checkpoint_wrappers", [])
    missing_wrappers = [
        suffix
        for suffix in payload.get("expected_wrapper_suffixes", [])
        if not any(wrapper.get("path", "").endswith(suffix) for wrapper in wrappers)
    ]
    if missing_wrappers:
        raise ValueError(f"Rank {rank} lost activation checkpoint wrappers: {missing_wrappers}")

    before = payload.get("fsdp_units_before", {})
    after = payload.get("fsdp_units_after", {})
    if set(before) != set(EXPECTED_UNITS) or set(after) != set(EXPECTED_UNITS):
        raise ValueError(f"Rank {rank} reshard unit set mismatch: before={before} after={after}")
    for class_name in EXPECTED_UNITS:
        if reshard_value(before[class_name], rank=rank, phase="before", class_name=class_name) is not True:
            raise ValueError(f"Rank {rank} {class_name} was not initially reshard=true")
        expected_after = class_name != CORE_UNIT
        if reshard_value(after[class_name], rank=rank, phase="after", class_name=class_name) is not expected_after:
            raise ValueError(
                f"Rank {rank} {class_name} final reshard mismatch: expected={expected_after}"
            )
    if not before[CORE_UNIT]["reshard_after_forward"].get("has_setter"):
        raise ValueError(f"Rank {rank} recurrent core exposed no runtime setter")

    gradients = payload.get("gradient_audit", {})
    for group in ("lora", "aligner", "audio_encoder"):
        audit = gradients.get(group, {})
        if (
            int(audit.get("gradient_tensors", 0)) <= 0
            or int(audit.get("finite_gradient_tensors", 0)) != int(audit.get("gradient_tensors", 0))
            or int(audit.get("nonzero_gradient_tensors", 0)) <= 0
        ):
            raise ValueError(f"Rank {rank} invalid {group} gradients: {audit}")
    for group in ("huginn_base", "other"):
        if int(gradients.get(group, {}).get("gradient_tensors", -1)) != 0:
            raise ValueError(f"Rank {rank} unexpected {group} gradients: {gradients.get(group)}")

    loss_contract = payload.get("loss_contract", {})
    if (
        loss_contract.get("prefix_tokens") != 127
        or loss_contract.get("prefix_labels_all_ignored") is not True
        or int(loss_contract.get("supervised_shift_tokens", 0)) <= 0
        or loss_contract.get("shift_length_valid") is not True
    ):
        raise ValueError(f"Rank {rank} shifted-NTP loss contract mismatch: {loss_contract}")
    if not payload.get("finite_losses") or not payload.get("finite_grad_norms"):
        raise ValueError(f"Rank {rank} lacks finite loss/grad-norm logs")
    if not all(math.isfinite(float(value)) for value in payload["finite_losses"] + payload["finite_grad_norms"]):
        raise ValueError(f"Rank {rank} contains non-finite logs")
    if float(payload.get("train_wall_seconds", 0.0)) <= 0.0:
        raise ValueError(f"Rank {rank} lacks a positive training wall time")
    validate_optimizer_groups(payload, rank)
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.world_size != 4:
        raise ValueError(f"Acceleration Stage 2 requires world_size=4, got {args.world_size}")
    if args.max_allocated_gib <= 0 or args.max_reserved_gib <= 0:
        raise ValueError("Stage 2 memory thresholds must be positive")
    audit_dir = args.audit_dir.expanduser().resolve()
    reports = [validate_rank(audit_dir, rank, args.world_size) for rank in range(args.world_size)]

    devices = {int(report.get("cuda_device", -1)) for report in reports}
    if devices != {0, 1, 2, 3}:
        raise ValueError(f"Stage 2 expected CUDA devices 0-3, observed {sorted(devices)}")
    core_calls = {str(rank): int(report["core_forward_calls"]) for rank, report in enumerate(reports)}
    if len(set(core_calls.values())) != 1:
        raise ValueError(f"Recurrent-core call counts differ across ranks: {core_calls}")

    memory = {
        str(rank): {
            "allocated_gib": float(report["peak_memory_allocated_gib"]),
            "reserved_gib": float(report["peak_memory_reserved_gib"]),
        }
        for rank, report in enumerate(reports)
    }
    max_allocated = max(entry["allocated_gib"] for entry in memory.values())
    max_reserved = max(entry["reserved_gib"] for entry in memory.values())
    memory_gate_passed = (
        max_allocated < args.max_allocated_gib
        and max_reserved < args.max_reserved_gib
    )

    report = {
        "gate": "huginn_whisper_dynamic30s_240ms_acceleration_stage2_fsdp4_v2",
        "validation_passed": memory_gate_passed,
        "world_size": args.world_size,
        "global_batch_size": 32,
        "synthetic_audio_seconds": 30.0,
        "audio_tokens_per_sample": 125,
        "prefix_tokens_per_sample": 127,
        "prefix_observations_per_rank": 8,
        "reshard_before": {class_name: True for class_name in EXPECTED_UNITS},
        "reshard_after": {
            class_name: class_name != CORE_UNIT for class_name in EXPECTED_UNITS
        },
        "core_forward_calls_by_rank": core_calls,
        "train_wall_seconds_by_rank": {
            str(rank): float(rank_report["train_wall_seconds"])
            for rank, rank_report in enumerate(reports)
        },
        "memory_by_rank": memory,
        "memory_thresholds_gib": {
            "allocated": args.max_allocated_gib,
            "reserved": args.max_reserved_gib,
        },
        "max_peak_allocated_gib": max_allocated,
        "max_peak_reserved_gib": max_reserved,
        "memory_gate_passed": memory_gate_passed,
        "checkpointing": {
            "whisper_internal": False,
            "whisper_outer": True,
            "double_checkpoint_candidate": False,
        },
        "rank_audit_dir": str(audit_dir),
    }
    output_report = args.output_report.expanduser().resolve()
    write_json(output_report, report)

    if not memory_gate_passed:
        raise RuntimeError(
            "Acceleration Stage 2 completed but failed the memory safety margin: "
            f"allocated={max_allocated:.3f}/{args.max_allocated_gib:.3f} GiB "
            f"reserved={max_reserved:.3f}/{args.max_reserved_gib:.3f} GiB "
            f"report={output_report}"
        )

    print(f"[stage2-reshard-before] {report['reshard_before']}")
    print(f"[stage2-reshard-after] {report['reshard_after']}")
    print(f"[stage2-prefix] per_rank={[127] * 8} global_samples=32")
    print(f"[stage2-core-calls] {core_calls}")
    print(f"[stage2-time] {report['train_wall_seconds_by_rank']}")
    print(
        "[stage2-memory] "
        f"by_rank={memory} max_allocated={max_allocated:.3f} "
        f"max_reserved={max_reserved:.3f}"
    )
    print(f"[stage2-report] {output_report}")
    print("========== HUGINN WHISPER DYNAMIC30S ACCELERATION STAGE 2 PASSED ==========")


if __name__ == "__main__":
    main()
