#!/usr/bin/env python3
"""Build and audit the deduplicated BAT QA/source manifests.

The released BAT train files are cumulative in practice: later files contain
records that also occur in earlier files, while Stage-III adds the Type-E
reasoning records.  This script keeps one semantic QA record for every unique
question and one source record for every unique ordered audio/RIR tuple.

Outputs:

* ``unique_qa_manifest.jsonl``: one fixed-schema ms-swift QA row per unique
  A-E record (normally about 870--880K rows);
* ``by_type/*.jsonl``: the same QA rows split into A/B/C/D/E for inspection;
* ``unique_source_manifest.jsonl``: one row per unique source tuple, used by
  the later offline Spatial-AST feature precomputation;
* optional ``source_shards/shard-*.jsonl`` files, with a configurable shard
  count for however many accelerator jobs are available;
* a detailed JSON audit report.

This is manifest-only work.  It does not read AudioSet, RIR files, or run a
GPU model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from .prepare_bat_swift_manifest import (
        STAGES,
        bat_type,
        canonical_audio_record,
        load_records,
        present,
        source_shape,
    )
except ImportError:  # Direct ``python path/to/script.py`` execution.
    from prepare_bat_swift_manifest import (
        STAGES,
        bat_type,
        canonical_audio_record,
        load_records,
        present,
        source_shape,
    )


SOURCE_FIELDS = ("audio_id", "reverb_id", "audio_id2", "reverb_id2")
STAGE_ORDER = ("I", "II", "III")
STAGE_DIRS = {stage: STAGES[stage][0] for stage in STAGE_ORDER}
STAGE_ALLOWED_TYPES = {stage: set(STAGES[stage][1]) for stage in STAGE_ORDER}
ALL_TYPES = ("A", "B", "C", "D", "E")


def normalize_reference(value: Any) -> str:
    if not present(value):
        return ""
    return str(value).replace("\\", "/").lstrip("./")


def normalize_question_type(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(value).strip().upper()).strip("_")


def normalized_source_tuple(row: dict[str, Any]) -> tuple[str, str, str, str]:
    values = tuple(normalize_reference(row.get(field)) for field in SOURCE_FIELDS)
    if not values[0] or not values[1]:
        raise ValueError("audio_id and reverb_id are required")
    if bool(values[2]) != bool(values[3]):
        raise ValueError(
            "audio_id2 and reverb_id2 must either both be present or both be absent"
        )
    return values


def source_key(source: tuple[str, str, str, str]) -> str:
    encoded = json.dumps(list(source), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def semantic_qa_payload(row: dict[str, Any], source: tuple[str, str, str, str], kind: str) -> dict[str, Any]:
    """Fields that define one QA training example.

    The curriculum stage is deliberately excluded: a cumulative copy of the
    same question in Stage II/III must collapse to the same QA record.
    """

    return {
        "question_id": str(row.get("question_id")),
        "question_type": normalize_question_type(row.get("question_type")),
        "bat_type": kind,
        "question": str(row.get("question")),
        "answer": str(row.get("answer")),
        "source": list(source),
    }


def qa_key(row: dict[str, Any], source: tuple[str, str, str, str], kind: str) -> str:
    payload = semantic_qa_payload(row, source, kind)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_raw_row(row: dict[str, Any], path: Path, index: int) -> None:
    required = ("audio_id", "reverb_id", "question", "answer", "question_type", "question_id")
    missing = [field for field in required if not present(row.get(field))]
    if missing:
        raise ValueError(f"{path}:{index} missing required fields {missing}")


def private_output(path: Path) -> None:
    normalized = str(path).replace("\\", "/")
    if normalized == "/hpc_stor03/public" or normalized.startswith("/hpc_stor03/public/"):
        raise ValueError(f"Refusing to write output under read-only public storage: {path}")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    temporary.replace(path)
    return count


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank line in generated manifest {path}:{line_number}")
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Generated manifest row is not an object: {path}:{line_number}")
            rows.append(item)
    return rows


def verify_generated_outputs(
    qa_path: Path,
    source_path: Path,
    type_paths: dict[str, str],
    unique_rows: list[dict[str, Any]],
    unique_source_rows: list[dict[str, Any]],
    shard_paths: list[Path],
) -> dict[str, Any]:
    """Re-read generated files and verify the cross-manifest invariants."""

    qa_rows = read_jsonl(qa_path)
    source_rows = read_jsonl(source_path)
    expected_qa_keys = {str(row["qa_fingerprint"]) for row in unique_rows}
    actual_qa_keys = [str(row.get("qa_fingerprint")) for row in qa_rows]
    expected_source_keys = {str(row["source_key"]) for row in unique_source_rows}
    actual_source_keys = [str(row.get("source_key")) for row in source_rows]
    source_key_set = set(actual_source_keys)
    verification: dict[str, Any] = {
        "qa_file_count": len(qa_rows),
        "source_file_count": len(source_rows),
        "qa_count_matches_memory": len(qa_rows) == len(unique_rows),
        "source_count_matches_memory": len(source_rows) == len(unique_source_rows),
        "qa_keys_unique": len(actual_qa_keys) == len(set(actual_qa_keys)),
        "source_keys_unique": len(actual_source_keys) == len(set(actual_source_keys)),
        "qa_keys_match_memory": set(actual_qa_keys) == expected_qa_keys,
        "source_keys_match_memory": source_key_set == expected_source_keys,
        "qa_source_keys_resolve": True,
        "source_keys_rehash_ok": True,
        "type_files": {},
        "shards": [],
    }

    for row in qa_rows:
        source_items = row.get("audios")
        if not isinstance(source_items, list) or len(source_items) != 1 or str(row.get("source_key")) not in source_key_set:
            verification["qa_source_keys_resolve"] = False
            continue
        item = source_items[0]
        if not isinstance(item, dict):
            verification["qa_source_keys_resolve"] = False
            continue
        source = tuple(str(item.get(field, "")) for field in SOURCE_FIELDS)
        if source_key(source) != str(row.get("source_key")):
            verification["qa_source_keys_resolve"] = False

    for row in source_rows:
        source = tuple(str(row.get(field, "")) for field in SOURCE_FIELDS)
        if source_key(source) != str(row.get("source_key")):
            verification["source_keys_rehash_ok"] = False

    for kind, path_string in type_paths.items():
        rows = read_jsonl(Path(path_string))
        keys = [str(row.get("qa_fingerprint")) for row in rows]
        expected = {str(row["qa_fingerprint"]) for row in unique_rows if row["bat_type"] == kind}
        item = {
            "count": len(rows),
            "expected_count": len(expected),
            "keys_unique": len(keys) == len(set(keys)),
            "keys_match_parent": set(keys) == expected,
        }
        verification["type_files"][kind] = item

    shard_keys: list[str] = []
    for path in shard_paths:
        rows = read_jsonl(path)
        keys = [str(row.get("source_key")) for row in rows]
        shard_keys.extend(keys)
        verification["shards"].append(
            {
                "path": str(path),
                "count": len(rows),
                "keys_unique_within_shard": len(keys) == len(set(keys)),
            }
        )
    verification["shards_cover_sources_once"] = (
        not shard_paths
        or len(shard_keys) == len(set(shard_keys)) == len(expected_source_keys)
        and set(shard_keys) == expected_source_keys
    )
    verification["ok"] = all(
        [
            verification["qa_count_matches_memory"],
            verification["source_count_matches_memory"],
            verification["qa_keys_unique"],
            verification["source_keys_unique"],
            verification["qa_keys_match_memory"],
            verification["source_keys_match_memory"],
            verification["qa_source_keys_resolve"],
            verification["source_keys_rehash_ok"],
            all(item["keys_unique"] and item["keys_match_parent"] for item in verification["type_files"].values()),
            verification["shards_cover_sources_once"],
        ]
    )
    return verification


def load_stage_file(qa_root: Path, stage: str) -> tuple[Path, list[dict[str, Any]]]:
    path = qa_root / STAGE_DIRS[stage] / "train.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    records = load_records(path)
    return path, records


def add_source_occurrence(
    source_entries: dict[str, dict[str, Any]],
    source: tuple[str, str, str, str],
    kind: str,
    stage: str,
    qa_fp: str,
    question_id: str,
) -> None:
    key = source_key(source)
    entry = source_entries.get(key)
    if entry is None:
        entry = {
            "source_key": key,
            "audio_id": source[0],
            "reverb_id": source[1],
            "audio_id2": source[2],
            "reverb_id2": source[3],
            "source_shape": "dual" if source[2] else "single",
            "bat_types": set(),
            "stages": set(),
            "qa_fingerprints": set(),
            "question_ids": set(),
        }
        source_entries[key] = entry
    entry["bat_types"].add(kind)
    entry["stages"].add(stage)
    entry["qa_fingerprints"].add(qa_fp)
    entry["question_ids"].add(question_id)


def finalize_source_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_key": entry["source_key"],
        "audio_id": entry["audio_id"],
        "reverb_id": entry["reverb_id"],
        "audio_id2": entry["audio_id2"],
        "reverb_id2": entry["reverb_id2"],
        "source_shape": entry["source_shape"],
        "bat_types": sorted(entry["bat_types"]),
        "stages": sorted(entry["stages"], key=STAGE_ORDER.index),
        "qa_record_count": len(entry["qa_fingerprints"]),
        "question_id_examples": sorted(entry["question_ids"])[:10],
        "estimated_render_source_count": 2 if entry["source_shape"] == "dual" else 1,
    }


def split_sources(rows: list[dict[str, Any]], shard_count: int) -> list[list[dict[str, Any]]]:
    if shard_count <= 0:
        return []
    shards: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    costs = [0 for _ in range(shard_count)]
    # Greedy deterministic balancing by expected number of convolution calls.
    # This is better than raw row-count balancing because dual-source rows cost
    # approximately twice as much as single-source rows.
    for row in sorted(rows, key=lambda item: str(item["source_key"])):
        target = min(range(shard_count), key=lambda index: (costs[index], len(shards[index]), index))
        item = dict(row)
        item["precompute_shard"] = target
        item["precompute_shard_count"] = shard_count
        shards[target].append(item)
        costs[target] += int(item["estimated_render_source_count"])
    return shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-shard-count", type=int, default=0)
    parser.add_argument("--expected-qa-min", type=int, default=870_000)
    parser.add_argument("--expected-qa-max", type=int, default=880_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source_shard_count < 0:
        raise ValueError("--source-shard-count must be non-negative")
    if args.expected_qa_min < 0 or args.expected_qa_max < args.expected_qa_min:
        raise ValueError("Invalid expected QA count range")
    private_output(args.output_dir)

    print("========== BAT UNIQUE QA/SOURCE MANIFEST BUILD ==========")
    print(f"[qa] root={args.qa_root}")
    print(f"[output] dir={args.output_dir}")
    print(f"[source_shards] count={args.source_shard_count}")

    issues: list[str] = []
    warnings: list[str] = []
    raw_stage_counts: dict[str, int] = {}
    raw_stage_type_counts: dict[str, Counter[str]] = {}
    raw_stage_shape_counts: dict[str, Counter[str]] = {}
    stage_fingerprints: dict[str, set[str]] = {stage: set() for stage in STAGE_ORDER}
    stage_new_counts: Counter[str] = Counter()
    stage_reused_counts: Counter[str] = Counter()
    question_id_to_fingerprints: dict[str, set[str]] = defaultdict(set)
    unique_rows: dict[str, dict[str, Any]] = {}
    source_entries: dict[str, dict[str, Any]] = {}
    first_seen_order: list[str] = []
    raw_total = 0
    invalid_rows: list[dict[str, Any]] = []

    for stage in STAGE_ORDER:
        path, records = load_stage_file(args.qa_root, stage)
        raw_stage_counts[stage] = len(records)
        raw_stage_type_counts[stage] = Counter()
        raw_stage_shape_counts[stage] = Counter()
        for index, raw in enumerate(records):
            raw_total += 1
            try:
                validate_raw_row(raw, path, index)
                source = normalized_source_tuple(raw)
                shape = source_shape({**raw, **dict(zip(SOURCE_FIELDS, source))})
                kind = bat_type(raw)
                if kind not in STAGE_ALLOWED_TYPES[stage]:
                    raise ValueError(
                        f"inferred BAT type {kind!r} is not allowed in Stage-{stage}; "
                        f"allowed={sorted(STAGE_ALLOWED_TYPES[stage])}"
                    )
                fingerprint = qa_key(raw, source, kind)
            except Exception as exc:
                invalid_rows.append({"stage": stage, "path": str(path), "record_index": index, "error": repr(exc)})
                continue

            raw_stage_type_counts[stage][kind] += 1
            raw_stage_shape_counts[stage][shape] += 1
            stage_fingerprints[stage].add(fingerprint)
            question_id = str(raw["question_id"])
            question_id_to_fingerprints[question_id].add(fingerprint)

            if fingerprint not in unique_rows:
                normalized_raw = dict(raw)
                for field in SOURCE_FIELDS:
                    normalized_raw[field] = source[SOURCE_FIELDS.index(field)]
                converted = canonical_audio_record(normalized_raw)
                converted_row = {
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Based on the audio you've heard, refer to the instruction and provide a response.\n\n"
                                "### Instruction:\n"
                                f"{str(raw['question'])}\n\n"
                                "### Response:"
                            ),
                        },
                        {"role": "assistant", "content": str(raw["answer"])},
                    ],
                    "audios": [converted],
                    "bat_stage": stage,
                    "bat_type": kind,
                    "question_id": converted["question_id"],
                    "question_type": converted["question_type"],
                    "source_shape": shape,
                    "qa_fingerprint": fingerprint,
                    "source_key": source_key(source),
                    "source_stages": [stage],
                    "source_record_index": index,
                }
                unique_rows[fingerprint] = converted_row
                first_seen_order.append(fingerprint)
                stage_new_counts[stage] += 1
            else:
                unique_rows[fingerprint]["source_stages"].append(stage)
                stage_reused_counts[stage] += 1

            add_source_occurrence(source_entries, source, kind, stage, fingerprint, question_id)

    if invalid_rows:
        issues.append("invalid_or_unmapped_raw_records")
    conflicts = {
        question_id: sorted(fingerprints)
        for question_id, fingerprints in question_id_to_fingerprints.items()
        if len(fingerprints) > 1
    }
    if conflicts:
        issues.append("question_id_content_conflicts")

    unique_qa_rows = [unique_rows[fingerprint] for fingerprint in first_seen_order]
    for row in unique_qa_rows:
        row["source_stages"] = sorted(set(row["source_stages"]), key=STAGE_ORDER.index)
    unique_source_rows = [
        finalize_source_entry(source_entries[key]) for key in sorted(source_entries)
    ]

    final_type_counts = Counter(str(row["bat_type"]) for row in unique_qa_rows)
    final_shape_counts = Counter(str(row["source_shape"]) for row in unique_qa_rows)
    source_shape_counts = Counter(str(row["source_shape"]) for row in unique_source_rows)
    source_type_counts = Counter()
    for row in unique_source_rows:
        for kind in row["bat_types"]:
            source_type_counts[kind] += 1

    missing_types = sorted(set(ALL_TYPES) - set(final_type_counts))
    if missing_types:
        issues.append(f"missing_final_bat_types:{','.join(missing_types)}")
    if not args.expected_qa_min <= len(unique_qa_rows) <= args.expected_qa_max:
        warnings.append(
            f"unique_qa_count_outside_expected_range:{len(unique_qa_rows)} "
            f"not in [{args.expected_qa_min},{args.expected_qa_max}]"
        )

    containment: dict[str, Any] = {}
    for previous, current in (("I", "II"), ("II", "III")):
        missing = stage_fingerprints[previous] - stage_fingerprints[current]
        containment[f"{previous}_to_{current}"] = {
            "previous_unique_records": len(stage_fingerprints[previous]),
            "current_unique_records": len(stage_fingerprints[current]),
            "previous_is_subset_of_current": not missing,
            "missing_previous_records": len(missing),
            "missing_examples": sorted(missing)[:20],
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    qa_path = args.output_dir / "unique_qa_manifest.jsonl"
    source_path = args.output_dir / "unique_source_manifest.jsonl"
    report_path = args.output_dir / "dedup_report.json"
    qa_count = write_jsonl(qa_path, unique_qa_rows)
    source_count = write_jsonl(source_path, unique_source_rows)

    type_paths: dict[str, str] = {}
    for kind in ALL_TYPES:
        type_path = args.output_dir / "by_type" / f"type_{kind}.jsonl"
        write_jsonl(type_path, (row for row in unique_qa_rows if row["bat_type"] == kind))
        type_paths[kind] = str(type_path)

    shard_report: dict[str, Any] = {"requested_count": args.source_shard_count, "shards": []}
    shard_paths: list[Path] = []
    if args.source_shard_count:
        shards = split_sources(unique_source_rows, args.source_shard_count)
        shard_dir = args.output_dir / "source_shards"
        for shard_id, shard_rows in enumerate(shards):
            shard_path = shard_dir / f"shard-{shard_id:03d}-of-{args.source_shard_count:03d}.jsonl"
            count = write_jsonl(shard_path, shard_rows)
            shard_paths.append(shard_path)
            render_cost = sum(int(row["estimated_render_source_count"]) for row in shard_rows)
            shard_report["shards"].append(
                {
                    "shard_id": shard_id,
                    "path": str(shard_path),
                    "source_tuple_count": count,
                    "estimated_render_source_count": render_cost,
                }
            )

    verification = verify_generated_outputs(
        qa_path,
        source_path,
        type_paths,
        unique_qa_rows,
        unique_source_rows,
        shard_paths,
    )
    if not verification["ok"]:
        issues.append("generated_manifest_verification_failed")

    report = {
        "status": "incomplete" if issues else "ok",
        "warnings": warnings,
        "issues": issues,
        "paths": {
            "qa_root": str(args.qa_root),
            "output_dir": str(args.output_dir),
            "unique_qa_manifest": str(qa_path),
            "unique_source_manifest": str(source_path),
            "type_manifests": type_paths,
        },
        "input_stages": {
            stage: {
                "directory": STAGE_DIRS[stage],
                "raw_record_count": raw_stage_counts.get(stage, 0),
                "raw_type_counts": dict(sorted(raw_stage_type_counts[stage].items())),
                "raw_source_shape_counts": dict(sorted(raw_stage_shape_counts[stage].items())),
                "unique_fingerprint_count": len(stage_fingerprints[stage]),
                "new_records_added_to_union": stage_new_counts[stage],
                "records_reused_from_previous_inputs": stage_reused_counts[stage],
            }
            for stage in STAGE_ORDER
        },
        "deduplication": {
            "raw_record_count_total": raw_total,
            "unique_qa_record_count": len(unique_qa_rows),
            "duplicate_records_removed": raw_total - len(unique_qa_rows),
            "unique_bat_type_counts": dict(sorted(final_type_counts.items())),
            "unique_source_shape_counts": dict(sorted(final_shape_counts.items())),
            "unique_source_tuple_count": len(unique_source_rows),
            "unique_source_shape_counts_by_tuple": dict(sorted(source_shape_counts.items())),
            "source_tuple_type_membership_counts": dict(sorted(source_type_counts.items())),
            "source_tuple_reuse_extra_qa_records": sum(
                max(0, int(row["qa_record_count"]) - 1) for row in unique_source_rows
            ),
            "question_id_conflict_count": len(conflicts),
            "question_id_conflict_examples": dict(list(conflicts.items())[:20]),
            "invalid_record_count": len(invalid_rows),
            "invalid_record_examples": invalid_rows[:20],
        },
        "curriculum_relationships": containment,
        "source_shards": shard_report,
        "verification": verification,
        "expected_final_qa_range": {
            "min": args.expected_qa_min,
            "max": args.expected_qa_max,
            "within_range": args.expected_qa_min <= len(unique_qa_rows) <= args.expected_qa_max,
        },
        "contract": {
            "qa_dedup_key": "question_id + normalized question_type + bat_type + question + answer + ordered source tuple",
            "source_dedup_key": "sha256(ordered audio_id,reverb_id,audio_id2,reverb_id2)",
            "source_tuple_order_preserved": True,
            "audio_processing_performed": False,
            "spatial_ast_processing_performed": False,
            "public_storage_written": False,
        },
    }
    write_json(report_path, report)

    print(f"[input] raw_records={raw_total}")
    print(f"[dedup] unique_qa_records={qa_count} removed={raw_total - qa_count}")
    print(f"[dedup] unique_bat_types={dict(sorted(final_type_counts.items()))}")
    print(f"[dedup] unique_source_tuples={source_count} shapes={dict(sorted(source_shape_counts.items()))}")
    print(f"[output] qa={qa_path}")
    print(f"[output] sources={source_path}")
    print(f"[report] {report_path}")
    print(f"[status] {report['status']} issues={issues} warnings={warnings}")
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
