#!/usr/bin/env python3
"""Audit the ordered Stage-III A+B -> C+D+E manifest against its report.

This is intentionally independent of the training process.  It streams the
manifest, reconstructs the composer order digest, checks every curriculum
annotation and verifies that the report really describes the file passed to
the trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROUTE = "stage3_ab_cde_2epoch"
EXPECTED_BLOCKS = ((1, "A+B"), (1, "C+D+E"), (2, "A+B"), (2, "C+D+E"))
GROUP_TYPES = {"A+B": {"A", "B"}, "C+D+E": {"C", "D", "E"}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--expected-global-batch-size", type=int, default=64)
    return parser.parse_args()


def private_output(path: Path) -> None:
    normalized = str(path).replace("\\", "/")
    if normalized == "/hpc_stor03/public" or normalized.startswith("/hpc_stor03/public/"):
        raise ValueError(f"Refusing public output path: {path}")


def resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
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
            yield line_number, row


def composer_digest_update(digest: "hashlib._Hash", row: dict[str, Any], padding: bool) -> None:
    """Exactly reproduce compose_bat_stage3_ab_cde_manifest.row_digest."""
    payload = {
        "question_id": row.get("question_id"),
        "bat_type": row.get("bat_type"),
        "source_shape": row.get("source_shape"),
        "padding": bool(padding),
    }
    digest.update(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(b"\n")


def canonical_content_digest_update(
    digest: "hashlib._Hash", row: dict[str, Any], padding: bool
) -> None:
    """Hash row content as an additional tamper/order diagnostic.

    Curriculum annotations are deliberately excluded because they are the
    order wrapper, while the underlying QA/source/message content is what must
    remain unchanged.  Padding remains explicit so the final duplicated rows
    cannot silently disappear from the audit.
    """
    payload = {
        key: value
        for key, value in row.items()
        if key not in {"curriculum_stage", "curriculum_epoch", "curriculum_block", "curriculum_shuffle_seed", "is_curriculum_padding"}
    }
    payload["_audit_padding"] = bool(padding)
    digest.update(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\n")


def load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Report is not a JSON object: {path}")
    return payload


def issue(issues: list[str], message: str) -> None:
    if message not in issues:
        issues.append(message)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    manifest = resolve(args.manifest)
    report_path = resolve(args.report)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if not report_path.is_file():
        raise FileNotFoundError(report_path)

    report = load_report(report_path)
    issues: list[str] = []
    warnings: list[str] = []

    if report.get("status") != "ok":
        issue(issues, "source_report_not_ok")
    if report.get("route") != ROUTE:
        issue(issues, "unexpected_route")
    reported_manifest = report.get("manifest")
    if not isinstance(reported_manifest, str):
        issue(issues, "report_manifest_missing")
    else:
        try:
            if resolve(Path(reported_manifest)) != manifest:
                issue(issues, "report_manifest_path_mismatch")
        except (OSError, ValueError):
            issue(issues, "report_manifest_path_unresolvable")

    if int(report.get("global_batch_size", -1)) != args.expected_global_batch_size:
        issue(issues, "global_batch_size_mismatch")
    if int(report.get("world_size", -1)) != 8:
        issue(issues, "world_size_mismatch")
    if int(report.get("per_device_batch_size", -1)) != 8:
        issue(issues, "per_device_batch_size_mismatch")
    if int(report.get("gradient_accumulation_steps", -1)) != 1:
        issue(issues, "gradient_accumulation_steps_mismatch")
    if report.get("runtime_shuffle") is not False:
        issue(issues, "runtime_shuffle_not_disabled")
    if report.get("group_order") != ["A+B", "C+D+E"]:
        issue(issues, "group_order_mismatch")

    blocks = report.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != len(EXPECTED_BLOCKS):
        issue(issues, "report_block_count_mismatch")
        blocks = []

    block_summaries: list[dict[str, Any]] = []
    report_by_block: dict[int, dict[str, Any]] = {}
    for item in blocks:
        if isinstance(item, dict):
            try:
                report_by_block[int(item["block"])] = item
            except (KeyError, TypeError, ValueError):
                issue(issues, "report_block_number_invalid")

    actual_record_count = 0
    actual_type_counts: Counter[str] = Counter()
    actual_nonpadding_type_counts: Counter[str] = Counter()
    block_number = 0
    current: dict[str, Any] | None = None
    digest: hashlib._Hash | None = None
    content_digest: hashlib._Hash | None = None
    block_rows = 0
    block_source_rows = 0
    block_padding_rows = 0
    block_padding_seen = False
    block_type_counts: Counter[str] = Counter()
    block_nonpadding_type_counts: Counter[str] = Counter()
    block_start_record = 0
    previous_block_end = 0
    previous_step_end = 0

    def begin_block(row: dict[str, Any]) -> None:
        nonlocal block_number, current, digest, content_digest, block_rows
        nonlocal block_source_rows, block_padding_rows, block_padding_seen
        nonlocal block_type_counts, block_nonpadding_type_counts, block_start_record
        block_number += 1
        epoch = row.get("curriculum_epoch")
        group = "A+B" if str(row.get("bat_type")) in GROUP_TYPES["A+B"] else "C+D+E"
        current = {"block": block_number, "epoch": epoch, "group": group, "seed": row.get("curriculum_shuffle_seed")}
        digest = hashlib.sha256()
        content_digest = hashlib.sha256()
        block_rows = 0
        block_source_rows = 0
        block_padding_rows = 0
        block_padding_seen = False
        block_type_counts = Counter()
        block_nonpadding_type_counts = Counter()
        block_start_record = actual_record_count

    def finish_block() -> None:
        nonlocal previous_block_end, previous_step_end
        if current is None or digest is None or content_digest is None:
            return
        expected_epoch, expected_group = EXPECTED_BLOCKS[current["block"] - 1] if 0 < current["block"] <= len(EXPECTED_BLOCKS) else (-1, "")
        if (int(current.get("epoch", -1)), str(current.get("group"))) != (expected_epoch, expected_group):
            issue(issues, f"block_{current['block']}_order_mismatch")
        expected = report_by_block.get(int(current["block"]))
        actual_end_record = block_start_record + block_rows
        actual_end_step = previous_step_end + block_rows // args.expected_global_batch_size if block_rows % args.expected_global_batch_size == 0 else -1
        if expected is None:
            issue(issues, f"block_{current['block']}_missing_from_report")
        else:
            comparisons = {
                "epoch": (int(current.get("epoch", -1)), int(expected.get("epoch", -2))),
                "group": (str(current.get("group")), str(expected.get("group"))),
                "written_records": (block_rows, int(expected.get("written_records", -1))),
                "source_records": (block_source_rows, int(expected.get("source_records", -1))),
                "padding_records": (block_padding_rows, int(expected.get("padding_records", -1))),
                "start_record": (block_start_record, int(expected.get("start_record", -1))),
                "end_record": (actual_end_record, int(expected.get("end_record", -1))),
                "start_step": (previous_step_end, int(expected.get("start_step", -1))),
                "end_step": (actual_end_step, int(expected.get("end_step", -1))),
                "shuffle_seed": (int(current.get("seed", -1)), int(expected.get("shuffle_seed", -2))),
                "order_sha256": (digest.hexdigest(), str(expected.get("order_sha256"))),
            }
            for name, (actual, expected_value) in comparisons.items():
                if actual != expected_value:
                    issue(issues, f"block_{current['block']}_{name}_mismatch")
            report_content_digest = expected.get("content_sha256")
            if report_content_digest is not None and report_content_digest != content_digest.hexdigest():
                issue(issues, f"block_{current['block']}_content_sha256_mismatch")
        if block_rows <= 0 or block_rows % args.expected_global_batch_size:
            issue(issues, f"block_{current['block']}_not_batch_aligned")
        if block_padding_rows and str(current.get("group")) != "C+D+E":
            issue(issues, f"block_{current['block']}_unexpected_padding")
        block_summaries.append({
            "block": current["block"],
            "epoch": current.get("epoch"),
            "group": current.get("group"),
            "rows": block_rows,
            "source_rows": block_source_rows,
            "padding_rows": block_padding_rows,
            "type_counts": dict(sorted(block_type_counts.items())),
            "nonpadding_type_counts": dict(sorted(block_nonpadding_type_counts.items())),
            "order_sha256": digest.hexdigest(),
            "content_sha256": content_digest.hexdigest(),
        })
        previous_block_end = actual_end_record
        if actual_end_step >= 0:
            previous_step_end = actual_end_step

    for line_number, row in iter_jsonl(manifest):
        row_block = row.get("curriculum_block")
        if not isinstance(row_block, int):
            issue(issues, "curriculum_block_missing_or_noninteger")
        if current is None or row_block != current.get("block"):
            if current is not None:
                finish_block()
            begin_block(row)
        actual_record_count += 1
        block_rows += 1
        bat_type = str(row.get("bat_type"))
        actual_type_counts[bat_type] += 1
        block_type_counts[bat_type] += 1
        if bat_type not in GROUP_TYPES.get(str(current.get("group")), set()):
            issue(issues, f"line_{line_number}_type_not_in_block_group")
        padding = row.get("is_curriculum_padding")
        if not isinstance(padding, bool):
            issue(issues, f"line_{line_number}_padding_flag_invalid")
            padding = bool(padding)
        if padding:
            block_padding_rows += 1
            block_padding_seen = True
        else:
            if block_padding_seen:
                issue(issues, f"block_{current['block']}_nonpadding_after_padding")
            block_source_rows += 1
            actual_nonpadding_type_counts[bat_type] += 1
            block_nonpadding_type_counts[bat_type] += 1
        if row.get("curriculum_stage") != "III":
            issue(issues, f"line_{line_number}_stage_invalid")
        if row.get("curriculum_epoch") != current.get("epoch"):
            issue(issues, f"line_{line_number}_epoch_mismatch")
        if row.get("curriculum_shuffle_seed") != current.get("seed"):
            issue(issues, f"line_{line_number}_seed_mismatch")
        if not isinstance(row.get("messages"), list) or not row.get("messages"):
            issue(issues, f"line_{line_number}_messages_missing")
        if not isinstance(row.get("audios"), list) or not row.get("audios"):
            issue(issues, f"line_{line_number}_audios_missing")
        composer_digest_update(digest, row, padding)  # type: ignore[arg-type]
        canonical_content_digest_update(content_digest, row, padding)  # type: ignore[arg-type]
    if current is not None:
        finish_block()

    if [
        (int(item.get("epoch", -1)), str(item.get("group"))) for item in block_summaries
    ] != list(EXPECTED_BLOCKS):
        issue(issues, "actual_block_order_mismatch")
    if int(report.get("total_records", -1)) != actual_record_count:
        issue(issues, "total_record_count_mismatch")
    if int(report.get("total_steps", -1)) != previous_step_end:
        issue(issues, "total_step_count_mismatch")
    if int(report.get("padding_records_total", -1)) != sum(item["padding_rows"] for item in block_summaries):
        issue(issues, "padding_total_mismatch")
    report_type_counts = report.get("source_type_counts")
    if isinstance(report_type_counts, dict):
        observed = dict(sorted(actual_nonpadding_type_counts.items()))
        expected_twice = {key: int(value) * 2 for key, value in report_type_counts.items()}
        if observed != dict(sorted(expected_twice.items())):
            issue(issues, "nonpadding_type_counts_mismatch")
    else:
        issue(issues, "source_type_counts_missing")

    for summary in block_summaries:
        expected_types = GROUP_TYPES.get(str(summary["group"]), set())
        if set(summary["nonpadding_type_counts"]) != expected_types:
            issue(issues, f"block_{summary['block']}_type_set_mismatch")

    return {
        "status": "ok" if not issues else "incomplete",
        "issues": issues,
        "warnings": warnings,
        "manifest": str(manifest),
        "source_report": str(report_path),
        "report_manifest": report.get("manifest"),
        "actual_record_count": actual_record_count,
        "actual_type_counts_including_padding": dict(sorted(actual_type_counts.items())),
        "actual_nonpadding_type_counts": dict(sorted(actual_nonpadding_type_counts.items())),
        "actual_total_steps": previous_step_end,
        "blocks": block_summaries,
        "expected_global_batch_size": args.expected_global_batch_size,
        "digest_contract": {
            "composer_order_sha256_recomputed": True,
            "additional_content_sha256_recomputed": True,
            "content_digest_in_source_report": any("content_sha256" in item for item in blocks if isinstance(item, dict)),
        },
    }


def main() -> None:
    args = parse_args()
    private_output(args.output_report)
    try:
        result = audit(args)
    except Exception as exc:
        result = {
            "status": "incomplete",
            "issues": [f"{type(exc).__name__}: {exc}"],
            "warnings": [],
            "manifest": str(args.manifest),
            "source_report": str(args.report),
        }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_report.with_name(args.output_report.name + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output_report)
    print(f"[report] {args.output_report}")
    print(f"[status] {result['status']} issues={result.get('issues', [])}")
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
