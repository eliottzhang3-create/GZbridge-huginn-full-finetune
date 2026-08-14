#!/usr/bin/env python3
"""Audit the actual payloads of a BAT BF16 Spatial-AST cache shard."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

try:
    from .build_bat_unique_manifests import SOURCE_FIELDS, source_key
except ImportError:  # Direct ``python path/to/script.py`` execution.
    from bat.scripts.build_bat_unique_manifests import SOURCE_FIELDS, source_key


EXPECTED_SHAPE = (515, 768)


def private_output(path: Path) -> None:
    normalized = str(path).replace("\\", "/")
    if normalized == "/hpc_stor03/public" or normalized.startswith("/hpc_stor03/public/"):
        raise ValueError(f"Refusing to write under read-only public storage: {path}")


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    private_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--finite-mode", choices=("all", "sample", "none"), default="all")
    parser.add_argument("--finite-sample-values", type=int, default=8192)
    parser.add_argument("--source-limit", type=int, default=0, help="Audit only the first N source rows; 0 means all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.finite_sample_values <= 0:
        raise ValueError("--finite-sample-values must be positive")
    if args.source_limit < 0:
        raise ValueError("--source-limit must be non-negative")
    private_output(args.output)
    source_path = args.source_manifest
    index_path = args.cache_dir / "index.jsonl"
    feature_root = args.cache_dir / "features"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not index_path.is_file():
        raise FileNotFoundError(index_path)

    source_rows = read_jsonl(source_path)
    if args.source_limit:
        source_rows = source_rows[: args.source_limit]
        if not source_rows:
            raise ValueError("--source-limit selected zero source rows")
    index_rows = read_jsonl(index_path)
    expected_keys = {str(row["source_key"]) for row in source_rows}
    index_keys = [str(row.get("source_key")) for row in index_rows]
    issues: list[str] = []
    failures: list[dict[str, Any]] = []
    if len(expected_keys) != len(source_rows):
        issues.append("duplicate_source_keys_in_source_manifest")
    if len(index_keys) != len(set(index_keys)):
        issues.append("duplicate_source_keys_in_cache_index")
    if set(index_keys) != expected_keys:
        issues.append("cache_index_source_key_coverage_mismatch")

    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in index_rows:
        required = {"source_key", "feature_file", "row", "shape", "dtype"}
        missing = sorted(required - set(row))
        if missing:
            failures.append({"kind": "index_row_missing_fields", "missing": missing, "row": row})
            continue
        if row["dtype"] != "bfloat16" or tuple(row["shape"]) != EXPECTED_SHAPE:
            failures.append({"kind": "index_metadata_mismatch", "row": row})
        by_file[str(row["feature_file"])].append(row)

    checked_values = 0
    checked_files = 0
    payload_values = 0
    start = time.perf_counter()
    file_summaries: list[dict[str, Any]] = []
    for relative_file, rows in sorted(by_file.items()):
        path = feature_root / relative_file
        if not path.is_file():
            failures.append({"kind": "missing_feature_file", "path": str(path)})
            continue
        try:
            payload = load_file(str(path), device="cpu")
            if set(payload) != {"features"}:
                failures.append({"kind": "unexpected_tensor_names", "path": str(path), "names": sorted(payload)})
                continue
            tensor = payload["features"]
            checked_files += 1
            payload_values += tensor.numel()
            file_item: dict[str, Any] = {
                "path": str(path),
                "index_rows": len(rows),
                "actual_shape": list(tensor.shape),
                "actual_dtype": str(tensor.dtype).replace("torch.", ""),
                "finite_checked": args.finite_mode != "none",
            }
            if tensor.dtype != torch.bfloat16:
                failures.append({"kind": "actual_dtype_mismatch", "path": str(path), "dtype": str(tensor.dtype)})
            if tensor.ndim != 3 or tuple(tensor.shape[1:]) != EXPECTED_SHAPE:
                failures.append({"kind": "actual_shape_mismatch", "path": str(path), "shape": list(tensor.shape)})
            row_numbers = [int(row["row"]) for row in rows]
            expected_row_numbers = set(range(int(tensor.shape[0]))) if tensor.ndim >= 1 else set()
            if len(row_numbers) != len(set(row_numbers)) or set(row_numbers) != expected_row_numbers:
                failures.append(
                    {
                        "kind": "row_index_mismatch",
                        "path": str(path),
                        "index_min_max": [min(row_numbers), max(row_numbers)] if row_numbers else None,
                        "actual_rows": int(tensor.shape[0]) if tensor.ndim >= 1 else None,
                    }
                )
            if args.finite_mode == "all":
                finite = bool(torch.isfinite(tensor).all().item())
                checked_values += tensor.numel()
            elif args.finite_mode == "sample":
                flat = tensor.reshape(-1)
                sample_indices = torch.linspace(
                    0,
                    max(0, flat.numel() - 1),
                    steps=min(args.finite_sample_values, flat.numel()),
                    dtype=torch.long,
                )
                finite = bool(torch.isfinite(flat[sample_indices]).all().item())
                checked_values += int(sample_indices.numel())
            else:
                finite = True
            file_item["finite"] = finite
            if not finite:
                failures.append({"kind": "non_finite_feature_values", "path": str(path)})
            file_summaries.append(file_item)
        except Exception as exc:
            failures.append({"kind": "feature_file_read_failed", "path": str(path), "error": repr(exc)})

    elapsed = time.perf_counter() - start
    if failures:
        issues.append("feature_payload_validation_failures")
    report = {
        "status": "incomplete" if issues else "ok",
        "source_manifest": str(source_path),
        "cache_dir": str(args.cache_dir),
        "index": str(index_path),
        "finite_mode": args.finite_mode,
        "source_limit": args.source_limit,
        "counts": {
            "source_rows": len(source_rows),
            "index_rows": len(index_rows),
            "expected_source_keys": len(expected_keys),
            "actual_index_keys": len(set(index_keys)),
            "feature_files_indexed": len(by_file),
            "feature_files_checked": checked_files,
            "payload_values": payload_values,
            "finite_values_checked": checked_values,
            "failure_count": len(failures),
        },
        "shape": list(EXPECTED_SHAPE),
        "dtype": "bfloat16",
        "elapsed_seconds": elapsed,
        "issues": issues,
        "failure_examples": failures[:50],
        "file_summary_examples": file_summaries[:20],
        "contract": {
            "actual_safetensors_payload_opened": True,
            "actual_dtype_checked": True,
            "actual_shape_checked": True,
            "row_coverage_checked": True,
            "finite_values_checked": args.finite_mode != "none",
        },
    }
    write_json(args.output, report)
    print(f"[audit] source_rows={len(source_rows)} index_rows={len(index_rows)} files={checked_files}/{len(by_file)}")
    print(f"[audit] finite_mode={args.finite_mode} finite_values_checked={checked_values}")
    print(f"[report] {args.output}")
    print(f"[status] {report['status']} issues={issues}")
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
