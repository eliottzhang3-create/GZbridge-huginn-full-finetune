#!/usr/bin/env python3
"""Compose a two-epoch BAT Stage-III route with A+B before C+D+E.

Each epoch is written as two deterministic blocks:

    1. shuffled A+B records
    2. shuffled C+D+E records

The manifest itself is the ordering contract.  Runtime dataset and dataloader
shuffle must remain disabled during training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


EXPECTED_TYPE_COUNTS = {
    "A": 139_392,
    "B": 139_392,
    "C": 118_000,
    "D": 118_000,
    "E": 357_528,
}
AB_TYPES = {"A", "B"}
CDE_TYPES = {"C", "D", "E"}
ROUTE = "stage3_ab_cde_2epoch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage3-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--warmup-ratio", type=float, default=0.13)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--allow-count-drift", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank line in {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_number}")
            yield row


def private_path(path: Path) -> None:
    normalized = str(path).replace("\\", "/")
    if normalized == "/hpc_stor03/public" or normalized.startswith("/hpc_stor03/public/"):
        raise ValueError(f"Refusing public output path: {path}")


def row_digest(digest: "hashlib._Hash", row: dict[str, Any], padding: bool) -> None:
    payload = {
        "question_id": row.get("question_id"),
        "bat_type": row.get("bat_type"),
        "source_shape": row.get("source_shape"),
        "padding": bool(padding),
    }
    digest.update(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\n")


def write_row(handle: Any, row: dict[str, Any], epoch: int, block: int, seed: int, padding: bool) -> None:
    item = dict(row)
    item["curriculum_stage"] = "III"
    item["curriculum_epoch"] = epoch
    item["curriculum_block"] = block
    item["curriculum_shuffle_seed"] = seed
    item["is_curriculum_padding"] = bool(padding)
    handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    if args.global_batch_size <= 0 or args.epochs != 2:
        raise ValueError("This route requires --epochs=2 and a positive global batch size")
    if not 0.0 < args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be between 0 and 1")
    if args.learning_rate != 0.002:
        raise ValueError("This route requires learning rate 0.002")
    if not args.stage3_manifest.is_file():
        raise FileNotFoundError(args.stage3_manifest)
    private_path(args.output)
    private_path(args.report)

    rows = list(iter_jsonl(args.stage3_manifest))
    type_counts = Counter(str(row.get("bat_type")) for row in rows)
    if not args.allow_count_drift and dict(type_counts) != EXPECTED_TYPE_COUNTS:
        raise ValueError(
            f"Stage-III type counts mismatch: actual={dict(sorted(type_counts.items()))} "
            f"expected={EXPECTED_TYPE_COUNTS}"
        )
    unknown = sorted(set(type_counts) - set(EXPECTED_TYPE_COUNTS))
    if unknown:
        raise ValueError(f"Unknown BAT types in Stage-III manifest: {unknown}")

    ab_rows = [row for row in rows if str(row.get("bat_type")) in AB_TYPES]
    cde_rows = [row for row in rows if str(row.get("bat_type")) in CDE_TYPES]
    if not ab_rows or not cde_rows:
        raise ValueError("Both A+B and C+D+E groups must be non-empty")
    for index, row in enumerate(rows):
        for field in ("messages", "audios", "bat_type", "question_id", "source_shape"):
            if field not in row:
                raise ValueError(f"Stage-III row {index} is missing {field!r}")

    blocks: list[dict[str, Any]] = []
    record_cursor = 0
    step_cursor = 0
    block_number = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for epoch in range(1, args.epochs + 1):
            for group_name, group_rows in (("A+B", ab_rows), ("C+D+E", cde_rows)):
                block_number += 1
                seed = args.shuffle_seed + block_number - 1
                shuffled = list(group_rows)
                random.Random(seed).shuffle(shuffled)
                padding_count = (-len(shuffled)) % args.global_batch_size if group_name == "C+D+E" else 0
                if (len(shuffled) + padding_count) % args.global_batch_size:
                    raise RuntimeError(f"Block {group_name} is not batch aligned after padding")

                start_record = record_cursor
                start_step = step_cursor
                digest = hashlib.sha256()
                for row in shuffled:
                    row_digest(digest, row, padding=False)
                    write_row(handle, row, epoch, block_number, seed, padding=False)
                for index in range(padding_count):
                    padding_row = shuffled[index % len(shuffled)]
                    row_digest(digest, padding_row, padding=True)
                    write_row(handle, padding_row, epoch, block_number, seed, padding=True)

                written = len(shuffled) + padding_count
                record_cursor += written
                step_cursor += written // args.global_batch_size
                blocks.append(
                    {
                        "block": block_number,
                        "epoch": epoch,
                        "group": group_name,
                        "types": sorted(AB_TYPES if group_name == "A+B" else CDE_TYPES),
                        "source_records": len(shuffled),
                        "padding_records": padding_count,
                        "written_records": written,
                        "shuffle_seed": seed,
                        "order_sha256": digest.hexdigest(),
                        "start_record": start_record,
                        "end_record": record_cursor,
                        "start_step": start_step,
                        "end_step": step_cursor,
                    }
                )
    temporary.replace(args.output)

    total_steps = step_cursor
    warmup_steps = int(math.ceil(total_steps * args.warmup_ratio))
    epoch_boundaries = {
        str(epoch): next(item["end_step"] for item in blocks if item["epoch"] == epoch and item["group"] == "C+D+E")
        for epoch in range(1, args.epochs + 1)
    }
    report = {
        "status": "ok",
        "route": ROUTE,
        "source_manifest": str(args.stage3_manifest),
        "manifest": str(args.output),
        "source_record_count": len(rows),
        "source_type_counts": dict(sorted(type_counts.items())),
        "group_record_counts": {"A+B": len(ab_rows), "C+D+E": len(cde_rows)},
        "epochs": args.epochs,
        "global_batch_size": args.global_batch_size,
        "per_device_batch_size": 8,
        "world_size": 8,
        "gradient_accumulation_steps": 1,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "warmup_steps": warmup_steps,
        "scheduler": "half-cycle cosine decay",
        "runtime_shuffle": False,
        "shuffle_policy": "deterministic_per_epoch_group",
        "group_order": ["A+B", "C+D+E"],
        "blocks": blocks,
        "epoch_boundary_steps": epoch_boundaries,
        "total_records": record_cursor,
        "total_steps": total_steps,
        "padding_records_total": sum(int(item["padding_records"]) for item in blocks),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[stage3-route] output={args.output}")
    print(f"[stage3-route] source_records={len(rows)} counts={dict(sorted(type_counts.items()))}")
    print(f"[stage3-route] blocks={len(blocks)} epoch_boundaries={epoch_boundaries}")
    print(f"[stage3-route] total_records={record_cursor} total_steps={total_steps} warmup_steps={warmup_steps}")
    print(f"[report] {args.report}")


if __name__ == "__main__":
    main()
