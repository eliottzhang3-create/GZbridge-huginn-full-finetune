#!/usr/bin/env python3
"""Inspect the HuggingFace Datasets/Arrow cache used by BAT JSONL loading.

No model, renderer, CUDA or NCCL is used.  The audit is intended to separate
Arrow/mmap/cache behaviour from AudioSet/RIR and model execution.  Set
``--cache-dir`` to a private local directory for the comparison run; omit it
to observe the current default cache placement.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from bat_diagnostics import filesystem_stats, process_stats, require_private_absolute, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=8500)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--local-batch-size", type=int, default=8)
    return parser.parse_args()


def cache_file_summary(dataset: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in getattr(dataset, "cache_files", []) or []:
        filename = item.get("filename") if isinstance(item, dict) else str(item)
        path = Path(filename) if filename else None
        result.append({
            "filename": str(path) if path else None,
            "exists": bool(path and path.is_file()),
            "size_bytes": path.stat().st_size if path and path.is_file() else None,
        })
    return result


def main() -> None:
    args = parse_args()
    output = require_private_absolute(args.output_report)
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if args.limit <= 0 or args.world_size <= 0 or not 0 <= args.rank < args.world_size:
        raise ValueError("invalid limit/world-size/rank")
    if args.cache_dir is not None:
        args.cache_dir = require_private_absolute(args.cache_dir)
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["HF_DATASETS_CACHE"] = str(args.cache_dir)
        os.environ["HF_HOME"] = str(args.cache_dir.parent)
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from datasets import load_dataset

    started = time.time()
    process_start = process_stats()
    print("========== BAT ARROW/CACHE PATH AUDIT ==========" , flush=True)
    print(f"[manifest] {args.manifest.resolve()} limit={args.limit}", flush=True)
    print(f"[cache] HF_DATASETS_CACHE={os.environ.get('HF_DATASETS_CACHE')} HF_HOME={os.environ.get('HF_HOME')}", flush=True)
    dataset = load_dataset("json", data_files=str(args.manifest.resolve()), split="train", cache_dir=str(args.cache_dir) if args.cache_dir else None)
    cache_files = cache_file_summary(dataset)
    row_accesses = 0
    last_row = None
    first_row = None
    for row_index, row in enumerate(dataset):
        # Match the contiguous global-batch assignment used by the DDP
        # training loader, without initializing a process group.
        global_batch = row_index // (args.world_size * args.local_batch_size)
        within_global = row_index % (args.world_size * args.local_batch_size)
        row_rank = within_global // args.local_batch_size
        if row_rank != args.rank:
            continue
        row_accesses += 1
        if first_row is None:
            first_row = row_index
        last_row = row_index
        if row_accesses >= args.limit:
            break
    report = {
        "status": "ok",
        "manifest": str(args.manifest.resolve()),
        "dataset_num_rows": len(dataset),
        "dataset_features": {key: str(value) for key, value in dataset.features.items()},
        "cache_environment": {
            "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE"),
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE"),
        },
        "cache_files": cache_files,
        "access_pattern": {
            "world_size": args.world_size,
            "rank": args.rank,
            "local_batch_size": args.local_batch_size,
            "row_accesses": row_accesses,
            "first_row_index": first_row,
            "last_row_index": last_row,
        },
        "process_start": process_start,
        "process_end": process_stats(),
        "filesystems": filesystem_stats((args.manifest, output, args.cache_dir or args.manifest.parent)),
        "elapsed_seconds": time.time() - started,
    }
    write_json(output, report)
    print(f"[summary] rows={len(dataset)} accessed={row_accesses} cache_files={len(cache_files)} seconds={report['elapsed_seconds']:.2f}", flush=True)
    print(f"[report] {output}", flush=True)
    print("[status] ok", flush=True)


if __name__ == "__main__":
    main()
