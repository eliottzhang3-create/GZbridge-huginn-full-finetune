#!/usr/bin/env python3
"""Download and verify one X-ARES Zenodo dataset from the local manifest.

The downloader fetches the published WebDataset tar files directly, one file
at a time. It keeps a ``.part`` file during transfer and resumes interrupted
downloads when the Zenodo endpoint supports HTTP Range requests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


CHUNK_SIZE = 8 * 1024 * 1024
PROGRESS_INTERVAL_SECONDS = 5.0
USER_AGENT = "GZbridge-huginn-full-finetune/xares-dataset-downloader"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "xares_zenodo_public_dataset_manifest.json",
    )
    parser.add_argument("--dataset", help="One dataset name from the manifest")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Local staging root; the dataset is written below <output-root>/<dataset>",
    )
    parser.add_argument("--list", action="store_true", help="List datasets and exit")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-delay", type=float, default=5.0)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError(f"Manifest is not complete: {path}")
    datasets = payload.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError(f"Manifest has no datasets list: {path}")
    result: dict[str, dict[str, Any]] = {}
    for entry in datasets:
        if not isinstance(entry, dict) or not isinstance(entry.get("dataset"), str):
            raise ValueError(f"Malformed dataset entry in manifest: {entry!r}")
        if entry["dataset"] in result:
            raise ValueError(f"Duplicate dataset in manifest: {entry['dataset']}")
        result[entry["dataset"]] = entry
    return result


def parse_checksum(value: str) -> tuple[str, str]:
    try:
        algorithm, digest = value.split(":", 1)
    except ValueError as error:
        raise ValueError(f"Checksum must have algorithm:digest form: {value!r}") from error
    algorithm = algorithm.lower()
    if algorithm not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported checksum algorithm: {algorithm}")
    return algorithm, digest.lower()


def verify_file(path: Path, expected_size: int, checksum: str) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        return False, f"size={actual_size},expected_size={expected_size}"
    algorithm, expected_digest = parse_checksum(checksum)
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    actual_digest = digest.hexdigest().lower()
    if actual_digest != expected_digest:
        return False, f"checksum={actual_digest},expected_checksum={expected_digest}"
    return True, "ok"


def open_download(url: str, start: int, timeout: float):
    headers = {"User-Agent": USER_AGENT}
    if start:
        headers["Range"] = f"bytes={start}-"
    request = urllib.request.Request(url, headers=headers)
    response = urllib.request.urlopen(request, timeout=timeout)
    status = response.getcode()
    if start and status != 206:
        response.close()
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        response = urllib.request.urlopen(request, timeout=timeout)
        return response, 0
    return response, start


def download_one(
    file_entry: dict[str, Any],
    destination: Path,
    timeout: float,
    retries: int,
    retry_delay: float,
    resume: bool,
) -> None:
    name = file_entry.get("name")
    url = file_entry.get("download_url")
    expected_size = file_entry.get("size_bytes")
    checksum = file_entry.get("checksum")
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError(f"Unsafe or malformed filename: {name!r}")
    if not isinstance(url, str) or not isinstance(expected_size, int) or not isinstance(checksum, str):
        raise ValueError(f"Malformed file entry: {file_entry!r}")

    part = destination.with_name(destination.name + ".part")
    if destination.exists():
        valid, detail = verify_file(destination, expected_size, checksum)
        if valid:
            print(f"[xares-download] skip_verified file={destination.name}", flush=True)
            return
        print(f"[xares-download] existing_invalid file={destination.name} detail={detail}", flush=True)

    if not resume and part.exists():
        part.unlink()

    last_error: Exception | None = None
    for attempt in range(1, retries + 2):
        try:
            start = part.stat().st_size if part.exists() else 0
            if start >= expected_size:
                part.unlink()
                start = 0
            response, write_start = open_download(url, start, timeout)
            digest_algorithm, _ = parse_checksum(checksum)
            digest = hashlib.new(digest_algorithm)
            if write_start:
                with part.open("rb") as existing:
                    for chunk in iter(lambda: existing.read(CHUNK_SIZE), b""):
                        digest.update(chunk)
            mode = "ab" if write_start else "wb"
            transferred = write_start
            started_at = time.monotonic()
            last_report_at = started_at
            print(
                f"[xares-download] downloading file={destination.name} "
                f"resume_from={write_start} bytes={expected_size}",
                flush=True,
            )
            with response, part.open(mode) as output:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
                    transferred += len(chunk)
                    now = time.monotonic()
                    if now - last_report_at >= PROGRESS_INTERVAL_SECONDS or transferred == expected_size:
                        elapsed = max(now - started_at, 1e-6)
                        rate = (transferred - write_start) / elapsed
                        remaining = max(expected_size - transferred, 0)
                        eta = remaining / rate if rate > 0 else 0
                        print(
                            f"[xares-download] progress file={destination.name} "
                            f"percent={transferred / expected_size * 100:.1f} "
                            f"bytes={transferred}/{expected_size} "
                            f"speed={rate / 1024 / 1024:.1f}MiB/s eta={eta / 60:.1f}min",
                            flush=True,
                        )
                        last_report_at = now
            if transferred != expected_size:
                raise IOError(f"downloaded {transferred} bytes, expected {expected_size}")
            if digest.hexdigest().lower() != parse_checksum(checksum)[1]:
                raise IOError(f"checksum mismatch for {destination.name}")
            part.replace(destination)
            print(f"[xares-download] verified file={destination.name} bytes={expected_size}", flush=True)
            return
        except (OSError, urllib.error.URLError, TimeoutError) as error:
            last_error = error
            if attempt > retries:
                break
            delay = retry_delay * (2 ** (attempt - 1))
            print(
                f"[xares-download] retry file={destination.name} attempt={attempt + 1} "
                f"after={delay:.1f}s error={error}",
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError(f"Failed to download {destination.name} after {retries + 1} attempts") from last_error


def main() -> None:
    args = parse_args()
    if args.retries < 0 or args.retry_delay < 0:
        raise ValueError("--retries and --retry-delay must be non-negative")
    datasets = load_manifest(args.manifest)

    if args.list:
        for name, entry in datasets.items():
            print(f"{name}\tfiles={entry['file_count']}\tbytes={entry['tar_bytes_sum']}")
        return
    if not args.dataset:
        raise ValueError("--dataset is required unless --list is used")
    if args.dataset not in datasets:
        raise ValueError(f"Unknown dataset {args.dataset!r}; use --list")
    if not args.output_root:
        raise ValueError("--output-root is required")

    entry = datasets[args.dataset]
    destination_root = args.output_root.expanduser().resolve() / args.dataset
    destination_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[xares-download] dataset={args.dataset} files={entry['file_count']} "
        f"bytes={entry['tar_bytes_sum']} root={destination_root}",
        flush=True,
    )
    failures = []
    for file_entry in entry["files"]:
        destination = destination_root / file_entry["name"]
        if args.verify_only:
            valid, detail = verify_file(destination, file_entry["size_bytes"], file_entry["checksum"])
            print(f"[xares-download] verify file={destination.name} result={detail}", flush=True)
            if not valid:
                failures.append(destination.name)
        else:
            download_one(
                file_entry,
                destination,
                args.timeout,
                args.retries,
                args.retry_delay,
                resume=not args.no_resume,
            )
    if failures:
        raise SystemExit(f"Verification failed for {len(failures)} file(s): {', '.join(failures)}")
    print(f"[xares-download] dataset_complete={args.dataset}", flush=True)


if __name__ == "__main__":
    main()
