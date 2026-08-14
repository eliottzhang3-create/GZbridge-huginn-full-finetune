#!/usr/bin/env python3
"""Strict audit for a prepared BAT ms-swift JSONL manifest."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from bat.scripts.prepare_bat_swift_manifest import bat_type, source_shape


EXPECTED_TYPES = {
    "I": {"A", "B"},
    "II": {"A", "B", "C", "D"},
    "III": {"A", "B", "C", "D", "E"},
}
EXPECTED_SOURCE_SHAPES = {
    "I": {"single"},
    "II": {"single", "dual"},
    "III": {"single", "dual"},
}
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
    parser.add_argument("--stage", choices=tuple(EXPECTED_TYPES), required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=0)
    parser.add_argument("--audio-root", type=Path, default=None)
    parser.add_argument("--reverb-root", type=Path, default=None)
    return parser.parse_args()


def present(value: Any) -> bool:
    return value is not None and str(value).strip().lower() not in {"", "null", "none"}


def normalize(value: Any) -> str:
    return str(value).replace("\\", "/").lstrip("./")


def resolve_audio(root: Path, audio_id: Any) -> Path | None:
    relative = normalize(audio_id)
    path = root / relative
    if path.suffix and path.is_file():
        return path
    for suffix in (".wav", ".flac", ".mp3", ".ogg"):
        candidate = path.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None


def resolve_reverb(root: Path, reverb_id: Any) -> Path | None:
    relative = normalize(reverb_id)
    candidates = (
        root / "binaural" / relative,
        root / relative,
        root / "mp3d_reverb" / "binaural" / relative,
    )
    return next((path for path in candidates if path.is_file()), None)


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    issues: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                issues.append(f"blank_line:{line_number}")
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.append(f"invalid_json:{line_number}:{exc.msg}")
                continue
            if not isinstance(item, dict):
                issues.append(f"non_object:{line_number}")
                continue
            records.append(item)
    return records, issues


def audit_record(record: dict[str, Any], stage: str, index: int, issues: list[str]) -> tuple[str | None, str | None]:
    prefix = f"record[{index}]"
    if record.get("bat_stage") != stage:
        issues.append(f"{prefix}:bat_stage={record.get('bat_stage')!r},expected={stage!r}")
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        issues.append(f"{prefix}:messages must contain user and assistant")
    else:
        roles = [item.get("role") if isinstance(item, dict) else None for item in messages]
        if roles != ["user", "assistant"]:
            issues.append(f"{prefix}:unexpected message roles={roles}")
        if any(not isinstance(item, dict) or not str(item.get("content", "")).strip() for item in messages):
            issues.append(f"{prefix}:empty message content")

    audios = record.get("audios")
    if not isinstance(audios, list) or len(audios) != 1 or not isinstance(audios[0], dict):
        issues.append(f"{prefix}:audios must contain one source record")
        source = {}
    else:
        source = audios[0]
    if source and set(source) != CANONICAL_AUDIO_FIELDS:
        issues.append(
            f"{prefix}:audios[0] schema mismatch; observed={sorted(source)} "
            f"expected={sorted(CANONICAL_AUDIO_FIELDS)}"
        )
    if source:
        non_string_fields = sorted(
            field for field in CANONICAL_AUDIO_FIELDS if field in source and not isinstance(source[field], str)
        )
        if non_string_fields:
            issues.append(f"{prefix}:audios[0] non-string fields={non_string_fields}")
    for field in ("audio_id", "reverb_id", "question", "answer", "question_type", "question_id"):
        if not present(source.get(field)):
            issues.append(f"{prefix}:source missing {field}")

    try:
        kind = bat_type(source)
    except Exception as exc:
        issues.append(f"{prefix}:type_error={exc}")
        kind = None
    try:
        shape = source_shape(source)
    except Exception as exc:
        issues.append(f"{prefix}:source_shape_error={exc}")
        shape = None
    if record.get("bat_type") != kind:
        issues.append(f"{prefix}:bat_type={record.get('bat_type')!r},inferred={kind!r}")
    if record.get("source_shape") != shape:
        issues.append(f"{prefix}:source_shape={record.get('source_shape')!r},inferred={shape!r}")
    if kind not in EXPECTED_TYPES[stage]:
        issues.append(f"{prefix}:type={kind!r} not allowed in stage {stage}")
    if shape not in EXPECTED_SOURCE_SHAPES[stage]:
        issues.append(f"{prefix}:source_shape={shape!r} not allowed in stage {stage}")
    return kind, shape


def main() -> None:
    args = parse_args()
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    records, issues = load_jsonl(args.manifest)
    type_counts: Counter[str] = Counter()
    shape_counts: Counter[str] = Counter()
    question_ids: Counter[str] = Counter()
    audio_refs: set[str] = set()
    reverb_refs: set[str] = set()
    for index, record in enumerate(records):
        kind, shape = audit_record(record, args.stage, index, issues)
        if kind is not None:
            type_counts[kind] += 1
        if shape is not None:
            shape_counts[shape] += 1
        source = record.get("audios", [{}])[0] if isinstance(record.get("audios"), list) and record["audios"] else {}
        if isinstance(source, dict):
            if present(source.get("question_id")):
                question_ids[str(source["question_id"])] += 1
            for audio_key, reverb_key in (("audio_id", "reverb_id"), ("audio_id2", "reverb_id2")):
                if present(source.get(audio_key)):
                    audio_refs.add(normalize(source[audio_key]))
                if present(source.get(reverb_key)):
                    reverb_refs.add(normalize(source[reverb_key]))

    if args.expected_count > 0 and len(records) != args.expected_count:
        issues.append(f"record_count={len(records)},expected={args.expected_count}")
    duplicate_question_ids = sum(count - 1 for count in question_ids.values() if count > 1)

    assets: dict[str, Any] = {
        "audio_reference_count": len(audio_refs),
        "reverb_reference_count": len(reverb_refs),
    }
    if args.audio_root is not None:
        missing = sorted(reference for reference in audio_refs if resolve_audio(args.audio_root, reference) is None)
        assets["audio_root"] = str(args.audio_root)
        assets["audio_missing_count"] = len(missing)
        assets["audio_missing_preview"] = missing[:20]
        if missing:
            issues.append(f"audio_missing={len(missing)}")
    if args.reverb_root is not None:
        missing = sorted(reference for reference in reverb_refs if resolve_reverb(args.reverb_root, reference) is None)
        assets["reverb_root"] = str(args.reverb_root)
        assets["reverb_missing_count"] = len(missing)
        assets["reverb_missing_preview"] = missing[:20]
        if missing:
            issues.append(f"reverb_missing={len(missing)}")

    report = {
        "status": "ok" if not issues else "incomplete",
        "manifest": str(args.manifest),
        "stage": args.stage,
        "expected_count": args.expected_count or None,
        "record_count": len(records),
        "question_type_counts": dict(sorted(type_counts.items())),
        "source_shape_counts": dict(sorted(shape_counts.items())),
        "duplicate_question_id_count": duplicate_question_ids,
        "assets": assets,
        "issues": issues,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[manifest] stage={args.stage} records={len(records)} types={dict(type_counts)} shapes={dict(shape_counts)}")
    print(f"[assets] audio_refs={len(audio_refs)} reverb_refs={len(reverb_refs)}")
    print(f"[report] {args.output_report}")
    print(f"[status] {report['status']} issues={issues[:10]}")
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
