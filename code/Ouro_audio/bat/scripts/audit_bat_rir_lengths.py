#!/usr/bin/env python3
"""Audit the duration of the extracted BAT binaural RIR files.

The official Spatial-AST preprocessing contract pads or truncates each RIR to
two seconds at 32 kHz.  This script is read-only: it only inspects ``.npy``
headers/shapes and writes a JSON report.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reverb-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--target-seconds", type=float, default=2.0)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Inspect at most this many .npy files; 0 means all files.",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=20,
        help="Number of over-threshold and invalid paths to include in the report.",
    )
    return parser.parse_args()


def reject_public_output(path: Path) -> None:
    resolved = path.expanduser().resolve()
    public_root = Path("/hpc_stor03/public").resolve()
    try:
        resolved.relative_to(public_root)
    except ValueError:
        return
    raise RuntimeError(f"Refusing to write an audit report under the read-only public tree: {resolved}")


def iter_npy_files(root: Path, limit: int) -> Iterable[Path]:
    paths = sorted(path for path in root.rglob("*.npy") if path.is_file())
    if limit > 0:
        paths = paths[:limit]
    return paths


def infer_time_axis(shape: tuple[int, ...]) -> tuple[int, int, str]:
    """Return (time_samples, time_axis, interpretation).

    Official binaural RIRs are expected to be [channels, time].  The fallback
    also accepts [time, channels] so the audit can diagnose files without
    silently counting the channel dimension as time.
    """

    if len(shape) == 1:
        return int(shape[0]), 0, "mono_1d"
    if len(shape) != 2:
        raise ValueError(f"expected 1D or 2D RIR, got shape={shape}")

    first, second = int(shape[0]), int(shape[1])
    if first <= 8 and second > first:
        return second, 1, "channels_first"
    if second <= 8 and first > second:
        return first, 0, "channels_last"
    if first == second:
        raise ValueError(f"ambiguous square RIR shape={shape}")
    # Keep the file visible in the report instead of silently assigning a
    # potentially wrong channel axis.  This fallback is only for unusual
    # layouts where neither dimension looks like a channel count.
    if first > second:
        return first, 0, "fallback_larger_axis"
    return second, 1, "fallback_larger_axis"


def percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), fraction * 100.0))


def as_relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def main() -> int:
    args = parse_args()
    if args.sample_rate <= 0:
        raise ValueError("--sample-rate must be positive")
    if args.target_seconds <= 0:
        raise ValueError("--target-seconds must be positive")
    if args.limit < 0 or args.preview_count < 0:
        raise ValueError("--limit and --preview-count must be non-negative")

    root = args.reverb_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    reject_public_output(output)
    if not root.is_dir():
        raise FileNotFoundError(f"RIR root does not exist or is not a directory: {root}")
    output.parent.mkdir(parents=True, exist_ok=True)

    target_samples_float = args.sample_rate * args.target_seconds
    if not target_samples_float.is_integer():
        raise ValueError("sample-rate * target-seconds must be an integer number of samples")
    target_samples = int(target_samples_float)

    npy_paths = list(iter_npy_files(root, args.limit))
    lengths: list[int] = []
    shape_counts: Counter[str] = Counter()
    interpretation_counts: Counter[str] = Counter()
    channel_counts: Counter[str] = Counter()
    over_threshold: list[dict[str, Any]] = []
    invalid_files: list[dict[str, str]] = []

    for path in npy_paths:
        relative = as_relative(path, root)
        try:
            # mmap reads the .npy metadata without materializing the full RIR.
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            shape = tuple(int(value) for value in array.shape)
            del array
            time_samples, time_axis, interpretation = infer_time_axis(shape)
            lengths.append(time_samples)
            shape_counts[str(shape)] += 1
            interpretation_counts[interpretation] += 1
            if len(shape) == 1:
                channel_counts["1"] += 1
            elif time_axis == 1:
                channel_counts[str(shape[0])] += 1
            else:
                channel_counts[str(shape[1])] += 1
            if time_samples > target_samples and len(over_threshold) < args.preview_count:
                over_threshold.append(
                    {
                        "path": relative,
                        "shape": list(shape),
                        "time_axis": time_axis,
                        "time_samples": time_samples,
                        "duration_seconds": time_samples / args.sample_rate,
                        "excess_samples": time_samples - target_samples,
                        "excess_seconds": (time_samples - target_samples) / args.sample_rate,
                    }
                )
        except Exception as exc:  # noqa: BLE001 - report every bad file and continue
            invalid_files.append({"path": relative, "error": f"{type(exc).__name__}: {exc}"})

    valid_count = len(lengths)
    over_count = sum(length > target_samples for length in lengths)
    equal_count = sum(length == target_samples for length in lengths)
    under_count = sum(length < target_samples for length in lengths)

    report: dict[str, Any] = {
        "status": "ok" if not invalid_files else "incomplete",
        "issues": ["rir_files_failed_to_inspect"] if invalid_files else [],
        "findings": ["rir_longer_than_target_present"] if over_count else [],
        "contract": {
            "sample_rate": args.sample_rate,
            "target_seconds": args.target_seconds,
            "target_samples": target_samples,
            "comparison": "time_samples > target_samples means longer than the official 2-second target",
        },
        "input": {
            "reverb_root": str(root),
            "recursive_npy_file_count": len(npy_paths),
            "limit": args.limit,
            "read_mode": "numpy_npy_header_via_mmap",
        },
        "summary": {
            "valid_rir_count": valid_count,
            "invalid_rir_count": len(invalid_files),
            "shorter_than_target_count": under_count,
            "equal_to_target_count": equal_count,
            "longer_than_target_count": over_count,
            "longer_than_target_ratio": (over_count / valid_count) if valid_count else None,
            "longer_than_target_percentage": (100.0 * over_count / valid_count) if valid_count else None,
            "target_or_shorter_count": under_count + equal_count,
            "target_or_shorter_percentage": (100.0 * (under_count + equal_count) / valid_count)
            if valid_count
            else None,
        },
        "length_statistics": {
            "min_samples": min(lengths) if lengths else None,
            "max_samples": max(lengths) if lengths else None,
            "mean_samples": (sum(lengths) / valid_count) if valid_count else None,
            "median_samples": float(np.median(np.asarray(lengths))) if lengths else None,
            "p95_samples": percentile(lengths, 0.95),
            "min_seconds": (min(lengths) / args.sample_rate) if lengths else None,
            "max_seconds": (max(lengths) / args.sample_rate) if lengths else None,
            "mean_seconds": (sum(lengths) / valid_count / args.sample_rate) if valid_count else None,
            "median_seconds": (float(np.median(np.asarray(lengths))) / args.sample_rate) if lengths else None,
            "p95_seconds": (percentile(lengths, 0.95) / args.sample_rate) if lengths else None,
        },
        "shape_counts": dict(shape_counts),
        "interpretation_counts": dict(interpretation_counts),
        "channel_counts": dict(channel_counts),
        "over_threshold_preview": over_threshold,
        "invalid_preview": invalid_files[: args.preview_count],
    }

    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("========== BAT RIR LENGTH AUDIT ==========")
    print(f"[root] {root}")
    print(f"[files] discovered={len(npy_paths)} valid={valid_count} invalid={len(invalid_files)}")
    print(
        f"[target] seconds={args.target_seconds:g} samples={target_samples} "
        f"sample_rate={args.sample_rate}"
    )
    print(
        f"[summary] shorter={under_count} equal={equal_count} "
        f"longer={over_count} "
        f"longer_percentage={(100.0 * over_count / valid_count) if valid_count else math.nan:.6f}%"
    )
    if over_count:
        print(f"[finding] {over_count} RIR files exceed the 2-second target")
    if invalid_files:
        print(f"[issue] failed to inspect {len(invalid_files)} RIR files")
    print(f"[report] {output}")
    print(f"[status] {report['status']} issues={report['issues']} findings={report['findings']}")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # concise CLI failure, with no partial success claim
        print(f"[fatal] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
