#!/usr/bin/env python3
"""Aggregate the four-rank dynamic-30s acceleration Stage 0 audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_UNITS = (
    "WhisperEncoderFSDPUnit",
    "AudioAlignerFSDPUnit",
    "HuginnPreludeFSDPUnit",
    "HuginnRecurrentCoreFSDPUnit",
    "HuginnCodaFSDPUnit",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    return parser.parse_args()


def load_rank_reports(audit_dir: Path, world_size: int) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for rank in range(world_size):
        path = audit_dir / f"rank-{rank}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing acceleration Stage 0 rank audit: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("gate") != "huginn_whisper_dynamic30s_acceleration_stage0_rank_v1":
            raise ValueError(f"Unexpected rank audit gate at {path}: {payload.get('gate')!r}")
        if int(payload.get("rank", -1)) != rank or int(payload.get("world_size", -1)) != world_size:
            raise ValueError(f"Rank identity mismatch at {path}: {payload}")
        if int(payload.get("global_step", -1)) != 1:
            raise ValueError(f"Rank {rank} did not finish one optimizer step")
        if int(payload.get("whisper_forward_calls", 0)) <= 0:
            raise ValueError(f"Rank {rank} observed no Whisper forward calls")
        if not payload.get("finite_losses") or not payload.get("finite_grad_norms"):
            raise ValueError(f"Rank {rank} lacks finite loss/gradient logs")
        if not isinstance(payload.get("gradient_audit"), dict):
            raise ValueError(f"Rank {rank} lacks the trainable-gradient audit")
        reports.append(payload)
    return reports


def same_across_ranks(reports: list[dict[str, Any]], key: str) -> Any:
    values = [report[key] for report in reports]
    canonical = json.dumps(values[0], sort_keys=True)
    mismatched = [rank for rank, value in enumerate(values) if json.dumps(value, sort_keys=True) != canonical]
    if mismatched:
        raise ValueError(f"Acceleration Stage 0 field differs across ranks: key={key} ranks={mismatched}")
    return values[0]


def summarize_reshard(reports: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for class_name in EXPECTED_UNITS:
        rank_values: dict[int, bool | None] = {}
        rank_details: dict[int, dict[str, Any]] = {}
        for rank, report in enumerate(reports):
            units = report.get("fsdp_units", {})
            if set(units) != set(EXPECTED_UNITS):
                raise ValueError(f"Rank {rank} FSDP unit set mismatch: {sorted(units)}")
            reshard = units[class_name].get("reshard_after_forward", {})
            rank_values[rank] = reshard.get("effective")
            rank_details[rank] = reshard
            if not reshard.get("has_fsdp_state"):
                raise ValueError(f"Rank {rank} {class_name} has no readable FSDP2 state")
        distinct = set(rank_values.values())
        if len(distinct) != 1:
            raise ValueError(f"{class_name} reshard state differs across ranks: {rank_values}")
        effective = next(iter(distinct))
        if effective is None:
            raise ValueError(
                f"Unable to resolve effective reshard_after_forward for {class_name}: {rank_details}"
            )
        summary[class_name] = {
            "effective": effective,
            "rank_details": rank_details,
        }
    return summary


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.world_size != 4:
        raise ValueError(f"Acceleration Stage 0 requires world_size=4, got {args.world_size}")
    audit_dir = args.audit_dir.expanduser().resolve()
    reports = load_rank_reports(audit_dir, args.world_size)

    attention = same_across_ranks(reports, "whisper_attention")
    checkpoint_wrappers = same_across_ranks(reports, "checkpoint_wrappers")
    internal_checkpointing = same_across_ranks(
        reports, "whisper_internal_gradient_checkpointing"
    )
    internal_modules = same_across_ranks(
        reports, "whisper_internal_gradient_checkpoint_modules"
    )
    outer_checkpointing = same_across_ranks(
        reports, "whisper_outer_activation_checkpointed"
    )
    double_checkpoint = same_across_ranks(
        reports, "whisper_double_checkpoint_candidate"
    )
    sdpa_values = {rank: bool(report["whisper_uses_sdpa"]) for rank, report in enumerate(reports)}
    if len(set(sdpa_values.values())) != 1:
        raise ValueError(f"Whisper SDPA use differs across ranks: {sdpa_values}")
    whisper_uses_sdpa = next(iter(sdpa_values.values()))
    if whisper_uses_sdpa and any(int(report["whisper_sdpa_calls"]) <= 0 for report in reports):
        raise ValueError("Whisper reports SDPA use but one rank has zero SDPA calls")
    if not whisper_uses_sdpa and any(report.get("first_whisper_sdpa_call") is not None for report in reports):
        raise ValueError("Whisper reports eager attention but captured an SDPA call")

    reshard = summarize_reshard(reports)
    expected_baseline = {class_name: True for class_name in EXPECTED_UNITS}
    observed_baseline = {
        class_name: bool(entry["effective"])
        for class_name, entry in reshard.items()
    }
    if observed_baseline != expected_baseline:
        raise ValueError(
            "Stage 0 baseline must leave every FSDP unit at reshard_after_forward=true: "
            f"observed={observed_baseline}"
        )

    report = {
        "gate": "huginn_whisper_dynamic30s_acceleration_stage0_fsdp4_v1",
        "validation_passed": True,
        "world_size": args.world_size,
        "whisper": {
            "uses_pytorch_sdpa": whisper_uses_sdpa,
            "sdpa_calls_by_rank": {
                str(rank): int(rank_report["whisper_sdpa_calls"])
                for rank, rank_report in enumerate(reports)
            },
            "forward_calls_by_rank": {
                str(rank): int(rank_report["whisper_forward_calls"])
                for rank, rank_report in enumerate(reports)
            },
            "attention": attention,
            "first_sdpa_calls": {
                str(rank): rank_report.get("first_whisper_sdpa_call")
                for rank, rank_report in enumerate(reports)
            },
            "cuda_sdpa_backends_enabled": same_across_ranks(
                reports, "cuda_sdpa_backends_enabled"
            ),
        },
        "checkpointing": {
            "whisper_internal_gradient_checkpointing": internal_checkpointing,
            "whisper_internal_gradient_checkpoint_modules": internal_modules,
            "whisper_outer_activation_checkpointed": outer_checkpointing,
            "double_checkpoint_candidate": double_checkpoint,
            "checkpoint_wrappers": checkpoint_wrappers,
        },
        "fsdp_reshard_after_forward": reshard,
        "memory_by_rank": {
            str(rank): {
                "allocated_gib": rank_report["peak_memory_allocated_gib"],
                "reserved_gib": rank_report["peak_memory_reserved_gib"],
            }
            for rank, rank_report in enumerate(reports)
        },
        "rank_audit_dir": str(audit_dir),
    }
    output_report = args.output_report.expanduser().resolve()
    write_json(output_report, report)

    print(
        "[stage0-whisper] "
        f"uses_pytorch_sdpa={whisper_uses_sdpa} "
        f"config_attn_implementation={attention.get('config_attn_implementation')!r} "
        f"attention_classes={attention.get('attention_classes')} "
        f"sdpa_calls_by_rank={report['whisper']['sdpa_calls_by_rank']}"
    )
    print(
        "[stage0-checkpoint] "
        f"internal={internal_checkpointing} outer_whisper={outer_checkpointing} "
        f"double_candidate={double_checkpoint} wrappers={checkpoint_wrappers}"
    )
    print(f"[stage0-reshard] {observed_baseline}")
    print(f"[stage0-memory] {report['memory_by_rank']}")
    print(f"[stage0-report] {output_report}")
    print("========== HUGINN WHISPER DYNAMIC30S ACCELERATION STAGE 0 PASSED ==========")


if __name__ == "__main__":
    main()
