#!/usr/bin/env python3
"""Prepare a reproducible local manifest for the public X-ARES Zenodo archives.

The manifest records the exact WebDataset tar files published by each Zenodo
record, including per-file size, checksum, and download URL.  It is a small
control artifact; it does not download or extract any audio archive.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATASETS: tuple[tuple[str, str], ...] = (
    ("asvspoof2015", "14718430"),
    ("clotho", "14856454"),
    ("cremad", "14646870"),
    ("desed", "14808180"),
    ("esc50", "14614287"),
    ("fluentspeechcommands", "14722453"),
    ("freemusicarchive", "14725056"),
    ("fsd50k", "14868441"),
    ("fsdkaggle2018", "14725117"),
    ("gtzan_genre", "14722472"),
    ("libricount", "14722478"),
    ("librispeechmalefemale", "14716252"),
    ("maestro", "14858022"),
    ("nsynthinstument", "14725174"),
    ("ravdess", "14722524"),
    ("speechcommandsv1", "14722647"),
    ("speechocean762", "14725291"),
    ("urbansound8k", "14722683"),
    ("vocalsound", "14722710"),
    ("voxceleb1", "14811963"),
    ("voxlingua33", "14812245"),
    ("vocalimitations", "14862060"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "xares_zenodo_public_dataset_manifest.json",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    return parser.parse_args()


def fetch_record(record_id: str, timeout: float, retries: int, retry_delay: float) -> dict[str, Any]:
    url = f"https://zenodo.org/api/records/{record_id}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "GZbridge-huginn-full-finetune/xares-manifest"},
    )
    last_error: Exception | None = None
    for attempt in range(1, retries + 2):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            if not isinstance(payload, dict):
                raise ValueError(f"Zenodo record {record_id} did not return an object")
            return payload
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            if attempt > retries:
                break
            delay = retry_delay * (2 ** (attempt - 1))
            print(
                f"[xares-manifest] retry record={record_id} attempt={attempt + 1} "
                f"after={delay:.1f}s error={error}",
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError(f"Unable to fetch Zenodo record {record_id} after {retries + 1} attempts") from last_error


def normalize_file(file_payload: dict[str, Any]) -> dict[str, Any]:
    key = file_payload.get("key")
    size = file_payload.get("size")
    checksum = file_payload.get("checksum")
    links = file_payload.get("links")
    self_url = links.get("self") if isinstance(links, dict) else None
    if not isinstance(key, str) or not isinstance(size, int) or not isinstance(checksum, str):
        raise ValueError(f"Malformed Zenodo file metadata: {file_payload!r}")
    if not isinstance(self_url, str):
        raise ValueError(f"Zenodo file has no self URL: {file_payload!r}")
    return {
        "name": key,
        "size_bytes": size,
        "checksum": checksum,
        "download_url": self_url,
    }


def dataset_entry(dataset_name: str, record_id: str, record: dict[str, Any]) -> dict[str, Any]:
    raw_files = record.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError(f"Zenodo record {record_id} has no files")
    files = [normalize_file(item) for item in raw_files]
    record_bytes = sum(item["size_bytes"] for item in files)
    archive_url = f"https://zenodo.org/api/records/{record_id}/files-archive"
    return {
        "dataset": dataset_name,
        "record_id": record_id,
        "title": record.get("metadata", {}).get("title"),
        "archive_url": archive_url,
        "files": files,
        "file_count": len(files),
        "tar_bytes_sum": record_bytes,
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def partial_payload(datasets: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "xares_zenodo_public_dataset_manifest_v1",
        "status": "partial",
        "source": "https://github.com/jimbozhang/xares/blob/main/tools/download_manually.sh",
        "zenodo_api": "https://zenodo.org/api/records/{record_id}",
        "requested_dataset_count": len(DATASETS),
        "dataset_count": len(datasets),
        "file_count": sum(item["file_count"] for item in datasets),
        "tar_bytes_sum": sum(item["tar_bytes_sum"] for item in datasets),
        "datasets": datasets,
    }


def load_partial(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "xares_zenodo_public_dataset_manifest_v1":
        raise ValueError(f"Unexpected partial manifest schema: {path}")
    datasets = payload.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError(f"Malformed partial manifest datasets: {path}")
    cached: dict[str, dict[str, Any]] = {}
    for item in datasets:
        if isinstance(item, dict) and isinstance(item.get("dataset"), str):
            cached[item["dataset"]] = item
    return cached


def main() -> None:
    args = parse_args()
    if args.retries < 0:
        raise ValueError("--retries must be non-negative")
    if args.retry_delay < 0:
        raise ValueError("--retry-delay must be non-negative")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    cached = load_partial(partial)
    datasets: list[dict[str, Any]] = []

    for dataset_name, record_id in DATASETS:
        if dataset_name in cached and cached[dataset_name].get("record_id") == record_id:
            entry = cached[dataset_name]
            print(
                f"[xares-manifest] resume dataset={dataset_name} record={record_id} "
                f"files={entry['file_count']} bytes={entry['tar_bytes_sum']}",
                flush=True,
            )
        else:
            record = fetch_record(record_id, args.timeout, args.retries, args.retry_delay)
            entry = dataset_entry(dataset_name, record_id, record)
            cached[dataset_name] = entry
            write_json_atomic(partial, partial_payload([cached[name] for name, _ in DATASETS if name in cached]))
            print(
                f"[xares-manifest] dataset={dataset_name} record={record_id} "
                f"files={entry['file_count']} bytes={entry['tar_bytes_sum']}",
                flush=True,
            )
        datasets.append(entry)

    total_files = sum(item["file_count"] for item in datasets)
    total_bytes = sum(item["tar_bytes_sum"] for item in datasets)

    payload = {
        "schema_version": "xares_zenodo_public_dataset_manifest_v1",
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "https://github.com/jimbozhang/xares/blob/main/tools/download_manually.sh",
        "zenodo_api": "https://zenodo.org/api/records/{record_id}",
        "dataset_count": len(datasets),
        "file_count": total_files,
        "tar_bytes_sum": total_bytes,
        "datasets": datasets,
    }
    write_json_atomic(output, payload)
    partial.unlink(missing_ok=True)
    print(f"[xares-manifest] wrote={output}", flush=True)
    print(f"[xares-manifest] datasets={len(datasets)} files={total_files} bytes={total_bytes}", flush=True)


if __name__ == "__main__":
    main()
