#!/usr/bin/env python3
"""Preflight a BAT JSONL manifest through the same Hugging Face JSON loader.

This catches nested Arrow schema errors before a GPU training job starts.  In
particular, it verifies that every sampled ``audios[0]`` object has the fixed
canonical BAT audio schema and that the complete JSONL can be materialized by
``datasets.load_dataset('json', ...)``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CANONICAL_AUDIO_FIELDS = {
    "audio_id",
    "reverb_id",
    "audio_id2",
    "reverb_id2",
    "question",
    "answer",
    "question_type",
    "question_id",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=0)
    parser.add_argument("--curriculum-report", type=Path, default=None)
    return parser.parse_args()


def sample_indices(count: int, curriculum_report: Path | None) -> list[int]:
    indices = {0, max(0, count // 2), max(0, count - 1)}
    if curriculum_report is not None and curriculum_report.is_file():
        report = json.loads(curriculum_report.read_text(encoding="utf-8"))
        for block in report.get("blocks", []):
            for key in ("start_record", "end_record"):
                value = int(block.get(key, -1))
                if 0 <= value < count:
                    indices.add(value)
            end = int(block.get("end_record", -1)) - 1
            if 0 <= end < count:
                indices.add(end)
    return sorted(indices)


def check_sample(row: dict[str, Any], index: int) -> list[str]:
    issues: list[str] = []
    audios = row.get("audios")
    if not isinstance(audios, list) or len(audios) != 1 or not isinstance(audios[0], dict):
        return [f"row[{index}] audios must contain one object"]
    source = audios[0]
    if set(source) != CANONICAL_AUDIO_FIELDS:
        issues.append(
            f"row[{index}] audio fields={sorted(source)} expected={sorted(CANONICAL_AUDIO_FIELDS)}"
        )
    non_string = sorted(name for name in CANONICAL_AUDIO_FIELDS if not isinstance(source.get(name), str))
    if non_string:
        issues.append(f"row[{index}] non-string audio fields={non_string}")
    if not isinstance(row.get("messages"), list) or len(row["messages"]) != 2:
        issues.append(f"row[{index}] messages must contain two entries")
    return issues


def main() -> None:
    args = parse_args()
    issues: list[str] = []
    report: dict[str, Any] = {
        "status": "incomplete",
        "manifest": str(args.manifest),
        "expected_count": args.expected_count or None,
    }
    try:
        if not args.manifest.is_file():
            raise FileNotFoundError(args.manifest)
        from datasets import load_dataset

        dataset = load_dataset("json", data_files=str(args.manifest), split="train")
        count = len(dataset)
        report["record_count"] = count
        report["features"] = str(dataset.features)
        if args.expected_count and count != args.expected_count:
            issues.append(f"record_count={count},expected={args.expected_count}")
        indices = sample_indices(count, args.curriculum_report)
        report["sample_indices"] = indices
        for index in indices:
            issues.extend(check_sample(dataset[index], index))
        report["status"] = "ok" if not issues else "incomplete"
    except Exception as exc:
        report["error"] = repr(exc)
        issues.append(f"dataset_load_error={exc}")
    report["issues"] = issues
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[schema] manifest={args.manifest}")
    print(f"[schema] records={report.get('record_count', 'unavailable')}")
    print(f"[report] {args.output_report}")
    print(f"[status] {report['status']} issues={issues[:10]}")
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
