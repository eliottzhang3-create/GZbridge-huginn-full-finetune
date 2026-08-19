#!/usr/bin/env python3
"""Build and verify the job-local HuggingFace JSON/Arrow cache once.

This runs before ``torchrun``.  It must remain a single process so DDP ranks do
not race while creating the same Arrow cache files.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = args.manifest.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if str(report_path).replace("\\", "/").startswith("/hpc_stor03/public"):
        raise ValueError(f"Refusing public report path: {report_path}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    minimum_free_bytes = int(os.environ.get("BAT_ARROW_CACHE_MIN_FREE_BYTES", 4 * 1024**3))
    free_bytes_before = shutil.disk_usage(cache_dir).free
    if free_bytes_before < minimum_free_bytes:
        raise RuntimeError(
            f"Insufficient local cache space before load: free={free_bytes_before} bytes, "
            f"required_at_least={minimum_free_bytes} bytes, path={cache_dir}"
        )
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir)
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from datasets import load_dataset

    started = time.time()
    dataset = load_dataset(
        "json",
        data_files=str(manifest),
        split="train",
        cache_dir=str(cache_dir),
        keep_in_memory=False,
    )
    cache_files = []
    for item in dataset.cache_files:
        path = Path(item["filename"])
        cache_files.append({"filename": str(path), "exists": path.is_file(), "size_bytes": path.stat().st_size if path.is_file() else None})
    if not cache_files or not all(item["exists"] for item in cache_files):
        raise RuntimeError(f"Local Arrow cache was not materialized correctly: {cache_files}")
    free_bytes_after = shutil.disk_usage(cache_dir).free
    if free_bytes_after < minimum_free_bytes:
        raise RuntimeError(
            f"Insufficient local cache space after load: free={free_bytes_after} bytes, "
            f"required_at_least={minimum_free_bytes} bytes, path={cache_dir}"
        )

    payload = {
        "status": "ok",
        "manifest": str(manifest),
        "cache_dir": str(cache_dir),
        "dataset_num_rows": len(dataset),
        "dataset_features": {key: str(value) for key, value in dataset.features.items()},
        "cache_files": cache_files,
        "cache_bytes": sum(int(item["size_bytes"] or 0) for item in cache_files),
        "cache_free_bytes_before_load": free_bytes_before,
        "cache_free_bytes_after_load": free_bytes_after,
        "cache_minimum_free_bytes": minimum_free_bytes,
        "hf_datasets_cache": os.environ.get("HF_DATASETS_CACHE"),
        "modelscope_cache": os.environ.get("MODELSCOPE_CACHE"),
        "elapsed_seconds": time.time() - started,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[arrow-cache] status=ok rows={len(dataset)} files={len(cache_files)} bytes={payload['cache_bytes']}", flush=True)
    print(f"[arrow-cache] cache_dir={cache_dir}", flush=True)
    print(f"[arrow-cache] report={report_path}", flush=True)


if __name__ == "__main__":
    main()
