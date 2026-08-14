#!/usr/bin/env python3
"""Audit the materialized ordered BAT curriculum manifest line by line."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from bat.curriculum import STAGE_EPOCHS, STAGE_ORDER, load_report, validate_curriculum_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--global-batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    curriculum = load_report(args.report)
    validate_curriculum_report(curriculum, args.global_batch_size)
    expected_blocks = [
        (int(item["block"]), str(item["stage"]), int(item["epoch"]), int(item["written_records"]))
        for item in curriculum["blocks"]
    ]
    issues: list[str] = []
    observed_blocks: Counter[tuple[int, str, int]] = Counter()
    observed_padding: Counter[tuple[int, str, int]] = Counter()
    record_count = 0
    current_block_index = 0
    with args.manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                issues.append(f"blank_line:{line_number}")
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.append(f"invalid_json:{line_number}:{exc.msg}")
                continue
            if not isinstance(row, dict):
                issues.append(f"non_object:{line_number}")
                continue
            try:
                block = int(row.get("curriculum_block", -1))
                stage = str(row.get("curriculum_stage"))
                epoch = int(row.get("curriculum_epoch", -1))
                padding = bool(row.get("is_curriculum_padding", False))
            except (TypeError, ValueError) as exc:
                issues.append(f"bad_curriculum_metadata:{line_number}:{exc}")
                continue
            if block < 1 or block > len(expected_blocks):
                issues.append(f"invalid_block:{line_number}:{block}")
                continue
            expected_block, expected_stage, expected_epoch, _ = expected_blocks[block - 1]
            if (block, stage, epoch) != (expected_block, expected_stage, expected_epoch):
                issues.append(
                    f"block_metadata_mismatch:{line_number}:observed={(block, stage, epoch)} "
                    f"expected={(expected_block, expected_stage, expected_epoch)}"
                )
            if block < current_block_index:
                issues.append(f"curriculum_order_regressed:{line_number}:{block}<{current_block_index}")
            current_block_index = max(current_block_index, block)
            observed_blocks[(block, stage, epoch)] += 1
            if padding:
                observed_padding[(block, stage, epoch)] += 1
                if stage != "III":
                    issues.append(f"non_stage3_padding:{line_number}:stage={stage}")
            record_count += 1

    for block, stage, epoch, written_records in expected_blocks:
        observed = observed_blocks[(block, stage, epoch)]
        if observed != written_records:
            issues.append(f"block_count_mismatch:block={block}:observed={observed}:expected={written_records}")
        expected_padding = next(item for item in curriculum["blocks"] if int(item["block"]) == block)["padding_records"]
        if observed_padding[(block, stage, epoch)] != int(expected_padding):
            issues.append(
                f"block_padding_mismatch:block={block}:observed={observed_padding[(block, stage, epoch)]}:"
                f"expected={expected_padding}"
            )
    if record_count != int(curriculum["total_records"]):
        issues.append(f"total_record_mismatch:observed={record_count}:expected={curriculum['total_records']}")
    if current_block_index != len(expected_blocks):
        issues.append(f"final_block_mismatch:observed={current_block_index}:expected={len(expected_blocks)}")

    report = {
        "status": "ok" if not issues else "incomplete",
        "manifest": str(args.manifest),
        "curriculum_report": str(args.report),
        "record_count": record_count,
        "expected_record_count": int(curriculum["total_records"]),
        "observed_blocks": {f"{block}:{stage}:epoch{epoch}": count for (block, stage, epoch), count in sorted(observed_blocks.items())},
        "observed_padding": {f"{block}:{stage}:epoch{epoch}": count for (block, stage, epoch), count in sorted(observed_padding.items()) if count},
        "issues": issues,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[curriculum] records={record_count} blocks={current_block_index}/{len(expected_blocks)}")
    print(f"[report] {args.output_report}")
    print(f"[status] {report['status']} issues={issues[:10]}")
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
