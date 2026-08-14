#!/usr/bin/env python3
"""Merge and audit the independent BAT Spatial-AST feature shard indices."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE_FIELDS = ("audio_id", "reverb_id", "audio_id2", "reverb_id2")


def private_output(path: Path) -> None:
    normalized = str(path).replace("\\", "/")
    if normalized == "/hpc_stor03/public" or normalized.startswith("/hpc_stor03/public/"):
        raise ValueError(f"Refusing public output under read-only storage: {path}")


def source_key(row: dict[str, Any]) -> str:
    values = [str(row.get(field, "")) for field in SOURCE_FIELDS]
    encoded = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank line in {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object in {path}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-shard-dir", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.shard_count <= 0:
        raise ValueError("--shard-count must be positive")
    private_output(args.feature_root)

    expected_keys: set[str] = set()
    expected_shards: dict[int, dict[str, Any]] = {}
    for shard_id in range(args.shard_count):
        shard_name = f"shard-{shard_id:03d}-of-{args.shard_count:03d}.jsonl"
        source_path = args.source_shard_dir / shard_name
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        rows = read_jsonl(source_path)
        keys = [str(row["source_key"]) for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate source_key values in source shard: {source_path}")
        expected_keys.update(keys)
        expected_shards[shard_id] = {"path": str(source_path), "count": len(rows), "keys": set(keys)}

    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    shard_reports: list[dict[str, Any]] = []
    for shard_id in range(args.shard_count):
        shard_dir = args.feature_root / f"shard-{shard_id:03d}"
        index_path = shard_dir / "index.jsonl"
        report_path = shard_dir / "precompute_report.json"
        if not index_path.is_file():
            raise FileNotFoundError(index_path)
        if not report_path.is_file():
            raise FileNotFoundError(report_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        index_rows = read_jsonl(index_path)
        source_keys = {str(row.get("source_key")) for row in index_rows}
        expected = expected_shards[shard_id]["keys"]
        missing = sorted(expected - source_keys)
        extra = sorted(source_keys - expected)
        if missing or extra or len(source_keys) != len(index_rows):
            raise RuntimeError(
                f"Shard {shard_id} index mismatch: missing={len(missing)} extra={len(extra)} "
                f"duplicate={len(index_rows) - len(source_keys)}"
            )
        if report.get("status") != "ok":
            raise RuntimeError(f"Shard {shard_id} precompute report is not ok: {report.get('status')}")
        for row in index_rows:
            key = str(row["source_key"])
            if key in seen:
                raise RuntimeError(f"Duplicate source_key across feature shards: {key}")
            seen.add(key)
            feature_file = shard_dir / "features" / str(row["feature_file"])
            if not feature_file.is_file():
                raise FileNotFoundError(feature_file)
            merged.append(
                {
                    "source_key": key,
                    "feature_shard": f"shard-{shard_id:03d}",
                    "feature_file": str(feature_file.relative_to(args.feature_root)).replace("\\", "/"),
                    "row": int(row["row"]),
                    "shape": row["shape"],
                    "dtype": row["dtype"],
                }
            )
        shard_reports.append(
            {
                "shard_id": shard_id,
                "source_count": expected_shards[shard_id]["count"],
                "feature_count": len(index_rows),
                "status": report.get("status"),
            }
        )

    if seen != expected_keys:
        raise RuntimeError(f"Global feature key mismatch: expected={len(expected_keys)} actual={len(seen)}")

    index_path = args.feature_root / "global_index.jsonl"
    report_path = args.feature_root / "global_feature_cache_report.json"
    write_jsonl(index_path, merged)
    report = {
        "status": "ok",
        "source_shard_dir": str(args.source_shard_dir),
        "feature_root": str(args.feature_root),
        "shard_count": args.shard_count,
        "expected_source_tuple_count": len(expected_keys),
        "merged_feature_count": len(merged),
        "global_index": str(index_path),
        "shards": shard_reports,
        "contract": {
            "all_source_keys_covered_once": True,
            "all_feature_files_exist": True,
            "feature_dtype": "bfloat16",
            "feature_shape": [515, 768],
            "public_storage_written": False,
        },
    }
    write_json(report_path, report)
    print(f"[merge] source_tuples={len(expected_keys)} features={len(merged)} shards={args.shard_count}")
    print(f"[index] {index_path}")
    print(f"[report] {report_path}")
    print("[status] ok")


if __name__ == "__main__":
    main()
