#!/usr/bin/env python3
"""Compose ordered BAT Stage-I/II/III training blocks into one JSONL manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from itertools import islice
from pathlib import Path
from typing import Any

from bat.curriculum import (
    STAGE_EPOCHS,
    STAGE_ORDER,
    STAGE_TYPES,
    count_jsonl,
    steps_for_records,
    update_order_digest,
    validate_curriculum_report,
)


EXPECTED_COUNTS = {"I": 278_784, "II": 514_784, "III": 872_312}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-manifest", type=Path, required=True)
    parser.add_argument("--stage2-manifest", type=Path, required=True)
    parser.add_argument("--stage3-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--limit-per-stage", type=int, default=0, help="For a small integration smoke only")
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--allow-count-drift", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank line in {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object in {path}:{line_number}")
            yield row


def validate_stage_manifest(
    path: Path,
    stage: str,
    allow_count_drift: bool,
    limit: int,
) -> int:
    available = count_jsonl(path)
    selected = min(available, limit) if limit else available
    if limit and available < limit:
        raise ValueError(f"{path} has only {available} records; --limit-per-stage={limit} requested")
    if not allow_count_drift and selected != EXPECTED_COUNTS[stage]:
        raise ValueError(f"{path} has {selected} selected records; expected {EXPECTED_COUNTS[stage]}")
    allowed = STAGE_TYPES[stage]
    observed: set[str] = set()
    for index, item in enumerate(islice(iter_jsonl(path), selected)):
        if item.get("bat_stage") != stage:
            raise ValueError(f"{path} record {index} has bat_stage={item.get('bat_stage')!r}, expected {stage!r}")
        kind = str(item.get("bat_type"))
        observed.add(kind)
        if kind not in allowed:
            raise ValueError(f"{path} record {index} has type {kind!r}, outside stage {stage}: {sorted(allowed)}")
    if not observed:
        raise ValueError(f"{path} has no selected records")
    return selected


def write_row(handle: Any, row: dict[str, Any], stage: str, epoch: int, block: int, padding: bool) -> None:
    item = dict(row)
    item["curriculum_stage"] = stage
    item["curriculum_epoch"] = epoch
    item["curriculum_block"] = block
    item["is_curriculum_padding"] = bool(padding)
    handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    if args.global_batch_size <= 0:
        raise ValueError("--global-batch-size must be positive")
    if args.limit_per_stage < 0:
        raise ValueError("--limit-per-stage must be non-negative")
    stage_paths = {"I": args.stage1_manifest, "II": args.stage2_manifest, "III": args.stage3_manifest}
    stage_counts = {
        stage: validate_stage_manifest(stage_paths[stage], stage, args.allow_count_drift, args.limit_per_stage)
        for stage in STAGE_ORDER
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    blocks: list[dict[str, Any]] = []
    record_cursor = 0
    step_cursor = 0
    block_index = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for stage in STAGE_ORDER:
            source_count = stage_counts[stage]
            # Keep only one stage in memory.  This permits deterministic
            # per-block shuffling without materializing the 4.2M-row global
            # curriculum manifest itself.
            source_rows = list(islice(iter_jsonl(stage_paths[stage]), source_count))
            for epoch in range(1, STAGE_EPOCHS[stage] + 1):
                block_index += 1
                block_seed = args.shuffle_seed + block_index - 1
                block_rows = list(source_rows)
                random.Random(block_seed).shuffle(block_rows)
                padding_count = (-source_count) % args.global_batch_size if stage == "III" else 0
                written_count = source_count + padding_count
                if stage != "III" and written_count % args.global_batch_size:
                    raise ValueError(
                        f"Stage {stage} epoch {epoch} has {written_count} records, not divisible by "
                        f"global batch {args.global_batch_size}"
                    )
                start_record = record_cursor
                start_step = step_cursor
                padding_seed: list[dict[str, Any]] = []
                order_digest = hashlib.sha256()
                for index, row in enumerate(block_rows):
                    if index < args.global_batch_size:
                        padding_seed.append(row)
                    item = dict(row)
                    item["curriculum_shuffle_seed"] = block_seed
                    update_order_digest(order_digest, item, padding=False)
                    write_row(handle, item, stage, epoch, block_index, padding=False)
                for index in range(padding_count):
                    item = dict(padding_seed[index % len(padding_seed)])
                    item["curriculum_shuffle_seed"] = block_seed
                    update_order_digest(order_digest, item, padding=True)
                    write_row(handle, item, stage, epoch, block_index, padding=True)
                record_cursor += written_count
                step_cursor += steps_for_records(written_count, args.global_batch_size)
                blocks.append(
                    {
                        "block": block_index,
                        "stage": stage,
                        "epoch": epoch,
                        "source_records": source_count,
                        "padding_records": padding_count,
                        "shuffle_seed": block_seed,
                        "order_sha256": order_digest.hexdigest(),
                        "written_records": written_count,
                        "start_record": start_record,
                        "end_record": record_cursor,
                        "start_step": start_step,
                        "end_step": step_cursor,
                    }
                )
    temporary.replace(args.output)

    boundary_steps = {
        "I": next(item["end_step"] for item in blocks if item["stage"] == "I" and item["epoch"] == 2),
        "II": next(item["end_step"] for item in blocks if item["stage"] == "II" and item["epoch"] == 2),
        "III": step_cursor,
    }
    report = {
        "status": "ok",
        "manifest": str(args.output),
        "source_manifests": {stage: str(path) for stage, path in stage_paths.items()},
        "global_batch_size": args.global_batch_size,
        "stage_epochs": STAGE_EPOCHS,
        "blocks": blocks,
        "boundary_steps": boundary_steps,
        "total_records": record_cursor,
        "total_steps": step_cursor,
        "warmup_steps": boundary_steps["I"],
        "lr_scheduler_type": "cosine",
        "padding_policy": "pad_stage_III_each_epoch_to_global_batch",
        "padding_records_total": sum(int(item["padding_records"]) for item in blocks),
        "shuffle_policy": "deterministic_per_curriculum_block",
        "shuffle_seed": args.shuffle_seed,
        "runtime_shuffle": False,
    }
    validate_curriculum_report(report, args.global_batch_size)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[curriculum] output={args.output}")
    print(f"[curriculum] total_records={record_cursor} total_steps={step_cursor}")
    print(f"[curriculum] boundaries={boundary_steps} padding={report['padding_records_total']}")
    print(f"[report] {args.report}")


if __name__ == "__main__":
    main()
