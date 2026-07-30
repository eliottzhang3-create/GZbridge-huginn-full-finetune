#!/usr/bin/env python3
"""Aggregate the four-rank dynamic-30s acceleration Stage 1 audit."""

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

EXPECTED_GRADIENT_GROUPS = ("lora", "aligner", "audio_encoder")


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
            raise FileNotFoundError(f"Missing acceleration Stage 1 rank audit: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("gate") != "huginn_whisper_dynamic30s_acceleration_stage1_rank_v1":
            raise ValueError(f"Unexpected rank audit gate at {path}: {payload.get('gate')!r}")
        if int(payload.get("rank", -1)) != rank or int(payload.get("world_size", -1)) != world_size:
            raise ValueError(f"Rank identity mismatch at {path}: {payload}")
        if int(payload.get("global_step", -1)) != 1:
            raise ValueError(f"Rank {rank} did not finish one optimizer step")
        if payload.get("vit_gradient_checkpointing_arg") is not False:
            raise ValueError(f"Rank {rank} did not disable vit_gradient_checkpointing")
        if payload.get("whisper_internal_gradient_checkpointing") is not False:
            raise ValueError(f"Rank {rank} retained Whisper internal gradient checkpointing")
        if payload.get("whisper_internal_gradient_checkpoint_modules") != []:
            raise ValueError(f"Rank {rank} retained internal checkpoint modules")
        if payload.get("whisper_outer_activation_checkpointed") is not True:
            raise ValueError(f"Rank {rank} lost the outer WhisperEncoder checkpoint")
        if payload.get("whisper_double_checkpoint_candidate") is not False:
            raise ValueError(f"Rank {rank} still has double Whisper checkpointing")
        if not payload.get("finite_losses") or not payload.get("finite_grad_norms"):
            raise ValueError(f"Rank {rank} lacks finite loss/gradient logs")
        if float(payload.get("train_wall_seconds", 0.0)) <= 0.0:
            raise ValueError(f"Rank {rank} lacks a positive training wall time")

        outer_wrappers = payload.get("whisper_outer_checkpoint_wrappers", [])
        if len(outer_wrappers) != 1:
            raise ValueError(f"Rank {rank} outer Whisper wrapper count mismatch: {outer_wrappers}")
        outer = outer_wrappers[0]
        if (
            not outer.get("path", "").endswith("audio_encoder.encoder")
            or "WhisperEncoder" not in outer.get("inner_mro", [])
            or outer.get("contains_whisper_encoder") is not True
        ):
            raise ValueError(f"Rank {rank} outer Whisper wrapper ownership mismatch: {outer}")

        wrappers = payload.get("checkpoint_wrappers", [])
        expected_suffixes = payload.get("expected_wrapper_suffixes", [])
        missing = [
            suffix
            for suffix in expected_suffixes
            if not any(wrapper.get("path", "").endswith(suffix) for wrapper in wrappers)
        ]
        if missing:
            raise ValueError(f"Rank {rank} lost activation-checkpoint wrappers: {missing}")

        gradient_audit = payload.get("gradient_audit", {})
        for group in EXPECTED_GRADIENT_GROUPS:
            audit = gradient_audit.get(group, {})
            if (
                int(audit.get("gradient_tensors", 0)) <= 0
                or int(audit.get("finite_gradient_tensors", 0))
                != int(audit.get("gradient_tensors", 0))
                or int(audit.get("nonzero_gradient_tensors", 0)) <= 0
            ):
                raise ValueError(f"Rank {rank} invalid {group} gradients: {audit}")
        for group in ("huginn_base", "other"):
            if int(gradient_audit.get(group, {}).get("gradient_tensors", -1)) != 0:
                raise ValueError(f"Rank {rank} unexpected {group} gradients: {gradient_audit.get(group)}")

        units = payload.get("fsdp_units", {})
        if set(units) != set(EXPECTED_UNITS):
            raise ValueError(f"Rank {rank} FSDP unit set mismatch: {sorted(units)}")
        for class_name in EXPECTED_UNITS:
            reshard = units[class_name].get("reshard_after_forward", {})
            if reshard.get("effective") is not True or not reshard.get("has_fsdp_state"):
                raise ValueError(f"Rank {rank} invalid {class_name} reshard state: {reshard}")

        attention = payload.get("whisper_attention", {})
        if attention.get("config_attn_implementation") != "sdpa":
            raise ValueError(f"Rank {rank} Whisper attention changed: {attention}")
        reports.append(payload)
    return reports


def same_across_ranks(reports: list[dict[str, Any]], key: str) -> Any:
    values = [report[key] for report in reports]
    canonical = json.dumps(values[0], sort_keys=True)
    mismatched = [
        rank
        for rank, value in enumerate(values)
        if json.dumps(value, sort_keys=True) != canonical
    ]
    if mismatched:
        raise ValueError(f"Acceleration Stage 1 field differs across ranks: key={key} ranks={mismatched}")
    return values[0]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.world_size != 4:
        raise ValueError(f"Acceleration Stage 1 requires world_size=4, got {args.world_size}")
    audit_dir = args.audit_dir.expanduser().resolve()
    reports = load_rank_reports(audit_dir, args.world_size)

    report = {
        "gate": "huginn_whisper_dynamic30s_acceleration_stage1_fsdp4_v1",
        "validation_passed": True,
        "world_size": args.world_size,
        "checkpointing": {
            "vit_gradient_checkpointing_arg": False,
            "whisper_internal_gradient_checkpointing": False,
            "whisper_internal_gradient_checkpoint_modules": [],
            "whisper_outer_activation_checkpointed": True,
            "whisper_outer_checkpoint_wrappers": same_across_ranks(
                reports, "whisper_outer_checkpoint_wrappers"
            ),
            "double_checkpoint_candidate": False,
            "checkpoint_wrappers": same_across_ranks(reports, "checkpoint_wrappers"),
        },
        "whisper_attention": same_across_ranks(reports, "whisper_attention"),
        "fsdp_units": same_across_ranks(reports, "fsdp_units"),
        "memory_by_rank": {
            str(rank): {
                "allocated_gib": rank_report["peak_memory_allocated_gib"],
                "reserved_gib": rank_report["peak_memory_reserved_gib"],
            }
            for rank, rank_report in enumerate(reports)
        },
        "finite_losses_by_rank": {
            str(rank): rank_report["finite_losses"]
            for rank, rank_report in enumerate(reports)
        },
        "finite_grad_norms_by_rank": {
            str(rank): rank_report["finite_grad_norms"]
            for rank, rank_report in enumerate(reports)
        },
        "train_wall_seconds_by_rank": {
            str(rank): rank_report["train_wall_seconds"]
            for rank, rank_report in enumerate(reports)
        },
        "rank_audit_dir": str(audit_dir),
    }
    output_report = args.output_report.expanduser().resolve()
    write_json(output_report, report)

    outer_path = report["checkpointing"]["whisper_outer_checkpoint_wrappers"][0]["path"]
    reshard = {
        class_name: details["reshard_after_forward"]["effective"]
        for class_name, details in report["fsdp_units"].items()
    }
    print(
        "[stage1-checkpoint] "
        f"internal=false outer=true outer_path={outer_path!r} double_candidate=false"
    )
    print(
        "[stage1-whisper] "
        f"attention={report['whisper_attention'].get('config_attn_implementation')!r} "
        f"classes={report['whisper_attention'].get('attention_classes')}"
    )
    print(f"[stage1-reshard] {reshard}")
    print(f"[stage1-time] {report['train_wall_seconds_by_rank']}")
    print(f"[stage1-memory] {report['memory_by_rank']}")
    print(f"[stage1-report] {output_report}")
    print("========== HUGINN WHISPER DYNAMIC30S ACCELERATION STAGE 1 PASSED ==========")


if __name__ == "__main__":
    main()
