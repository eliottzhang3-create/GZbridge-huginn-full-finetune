"""Build a private ACAVCAPS manifest with one global tar permutation.

The source manifest must be the completed read-only full ACAVCAPS preflight.
All source stages are flattened into one tar list, then shuffled once with a
fixed seed.  The stage/category fields remain provenance only; they do not
define training boundaries.  Audio is still decoded lazily from the public
tar files by the training plugin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any


PUBLIC_ROOT = Path("/hpc_stor03/public/shared/data/raa/ACAVCAPS")
DEFAULT_SOURCE = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/"
    "data/audio_swift/acavcaps_wds/acavcaps_wds_stage_schedule_full_seed20260723.json"
)
DEFAULT_OUTPUT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/"
    "data/audio_swift/acavcaps/acavcaps_flat_global_tar_shuffle_seed20260723.json"
)
EXPECTED_STAGE_NAMES = ("stage1", "stage2", "stage3")
EXPECTED_TAR_COUNT = 1071
EXPECTED_SAMPLE_COUNT = 4_664_169
DEFAULT_SAMPLE_SHUFFLE_BUFFER = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_manifest", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output_manifest", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--sample_shuffle_buffer", type=int, default=DEFAULT_SAMPLE_SHUFFLE_BUFFER)
    parser.add_argument("--expected_tar_count", type=int, default=EXPECTED_TAR_COUNT)
    parser.add_argument("--expected_sample_count", type=int, default=EXPECTED_SAMPLE_COUNT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(f"[flat-manifest] FAIL {message}")


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"missing JSON file: {path}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")
    if not isinstance(payload, dict):
        fail(f"JSON root must be an object: {path}")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_source(
    source_manifest: dict[str, Any],
    source_stats: dict[str, Any],
    source_path: Path,
    *,
    expected_tar_count: int,
    expected_sample_count: int,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    if Path(str(source_manifest.get("dataset_root", ""))).resolve() != PUBLIC_ROOT.resolve():
        fail(f"source dataset_root mismatch: {source_manifest.get('dataset_root')!r}")
    if source_stats.get("dataset_root") != str(PUBLIC_ROOT):
        fail(f"source stats dataset_root mismatch: {source_stats.get('dataset_root')!r}")
    if source_manifest.get("scan_mode") != "full" or source_stats.get("scan_mode") != "full":
        fail("source manifest and stats must both be full scans")
    if (
        source_manifest.get("public_root_mutation") != "forbidden"
        or source_stats.get("public_root_mutation") != "forbidden"
    ):
        fail("source manifest/stats do not enforce the read-only public-root policy")
    if source_stats.get("all_pairs_valid") is not True:
        fail(f"source all_pairs_valid is not true: {source_stats.get('all_pairs_valid')!r}")

    stages = source_manifest.get("stages")
    if not isinstance(stages, list) or tuple(stage.get("name") for stage in stages) != EXPECTED_STAGE_NAMES:
        fail(f"unexpected source stage order: {[stage.get('name') for stage in stages or []]!r}")

    flattened: list[dict[str, Any]] = []
    stage_tar_counts: dict[str, int] = {}
    stage_sample_counts: dict[str, int] = {}
    seen_paths: set[str] = set()
    category_counts: Counter[str] = Counter()
    for stage in stages:
        stage_name = str(stage["name"])
        tars = stage.get("tars")
        if not isinstance(tars, list):
            fail(f"{stage_name} does not contain a tar list")
        stage_samples = 0
        for source_order_index, raw_entry in enumerate(tars):
            if not isinstance(raw_entry, dict):
                fail(f"{stage_name}[{source_order_index}] is not an object")
            path = str(raw_entry.get("path", ""))
            if not path:
                fail(f"{stage_name}[{source_order_index}] has no tar path")
            resolved = str(Path(path).resolve())
            if PUBLIC_ROOT.resolve() not in Path(resolved).parents:
                fail(f"{stage_name}[{source_order_index}] tar is outside the read-only public root: {resolved}")
            if resolved in seen_paths:
                fail(f"duplicate source tar path: {resolved}")
            seen_paths.add(resolved)
            if raw_entry.get("scan_status") != "scanned" or raw_entry.get("valid") is not True:
                fail(f"{stage_name}[{source_order_index}] is not fully validated")
            json_count = raw_entry.get("json_count")
            flac_count = raw_entry.get("flac_count")
            if not isinstance(json_count, int) or json_count <= 0 or json_count != flac_count:
                fail(
                    f"{stage_name}[{source_order_index}] JSON/FLAC counts are invalid: "
                    f"json={json_count!r} flac={flac_count!r}"
                )
            category = str(raw_entry.get("category", ""))
            if not category:
                fail(f"{stage_name}[{source_order_index}] has no category")
            entry = dict(raw_entry)
            entry["source_stage"] = stage_name
            entry["source_stage_order_index"] = source_order_index
            entry["source_stage_seed"] = stage.get("seed")
            entry["path"] = resolved
            flattened.append(entry)
            stage_samples += json_count
            category_counts[category] += 1
        stage_tar_counts[stage_name] = len(tars)
        stage_sample_counts[stage_name] = stage_samples
        if stage.get("sample_count") != stage_samples:
            fail(
                f"{stage_name} sample_count mismatch: manifest={stage.get('sample_count')!r} "
                f"computed={stage_samples}"
            )

    if len(flattened) != expected_tar_count:
        fail(f"source tar count mismatch: expected={expected_tar_count} actual={len(flattened)}")
    total_samples = sum(int(entry["json_count"]) for entry in flattened)
    if total_samples != expected_sample_count:
        fail(f"source sample count mismatch: expected={expected_sample_count} actual={total_samples}")
    if source_stats.get("tar_count") != len(flattened) or source_stats.get("sample_count") != total_samples:
        fail(
            "source stats totals mismatch: "
            f"stats=({source_stats.get('tar_count')},{source_stats.get('sample_count')}) "
            f"computed=({len(flattened)},{total_samples})"
        )
    print(
        f"[flat-manifest] source_validated path={source_path} tar_count={len(flattened)} "
        f"sample_count={total_samples} stage_tar_counts={stage_tar_counts}",
        flush=True,
    )
    return flattened, stage_tar_counts, stage_sample_counts


def main() -> int:
    args = parse_args()
    if args.sample_shuffle_buffer <= 0:
        fail(f"sample_shuffle_buffer must be positive: {args.sample_shuffle_buffer}")
    if args.expected_tar_count <= 0 or args.expected_sample_count <= 0:
        fail("expected counts must be positive")

    source_path = Path(args.source_manifest).expanduser().resolve()
    output_path = Path(args.output_manifest).expanduser().resolve()
    stats_path = output_path.with_suffix(".stats.json")
    if not source_path.is_file():
        fail(f"source manifest does not exist: {source_path}")
    source_stats_path = source_path.with_suffix(".stats.json")
    if not source_stats_path.is_file():
        fail(f"source stats do not exist: {source_stats_path}")
    if output_path == source_path:
        fail("output manifest must differ from source manifest")
    if not args.overwrite and (output_path.exists() or stats_path.exists()):
        fail(f"output already exists: {output_path} or {stats_path}; use --overwrite")
    if PUBLIC_ROOT.resolve() in output_path.parents or PUBLIC_ROOT.resolve() in stats_path.parents:
        fail("refusing to write manifest/stats inside the public ACAVCAPS root")

    source_manifest = read_json(source_path)
    source_stats = read_json(source_stats_path)
    flattened, stage_tar_counts, stage_sample_counts = validate_source(
        source_manifest,
        source_stats,
        source_path,
        expected_tar_count=args.expected_tar_count,
        expected_sample_count=args.expected_sample_count,
    )

    random.Random(args.seed).shuffle(flattened)
    for order_index, entry in enumerate(flattened):
        entry["order_index"] = order_index

    source_digest = sha256(source_path)
    manifest = {
        "schema_version": 1,
        "dataset_root": str(PUBLIC_ROOT),
        "public_root_mutation": "forbidden",
        "scan_mode": "derived_from_full",
        "schedule_policy": "global_tar_order_shuffle_all_stages_v1_per_tar_buffer_shuffle",
        "seed": args.seed,
        "sample_shuffle_buffer": args.sample_shuffle_buffer,
        "source_stage_order": list(EXPECTED_STAGE_NAMES),
        "source_manifest": str(source_path),
        "source_manifest_sha256": source_digest,
        "tar_count": len(flattened),
        "sample_count": sum(int(entry["json_count"]) for entry in flattened),
        "tars": flattened,
    }
    category_tar_counts = dict(sorted(Counter(str(entry["category"]) for entry in flattened).items()))
    stats = {
        "schema_version": 1,
        "manifest_path": str(output_path),
        "dataset_root": str(PUBLIC_ROOT),
        "public_root_mutation": "forbidden",
        "scan_mode": "derived_from_full",
        "schedule_policy": manifest["schedule_policy"],
        "seed": args.seed,
        "sample_shuffle_buffer": args.sample_shuffle_buffer,
        "source_manifest": str(source_path),
        "source_manifest_sha256": source_digest,
        "source_stage_order": list(EXPECTED_STAGE_NAMES),
        "source_stage_tar_counts": stage_tar_counts,
        "source_stage_sample_counts": stage_sample_counts,
        "category_tar_counts": category_tar_counts,
        "tar_count": len(flattened),
        "sample_count": manifest["sample_count"],
        "all_pairs_valid": True,
    }
    atomic_json_write(output_path, manifest)
    atomic_json_write(stats_path, stats)
    print(f"[flat-manifest] wrote_manifest={output_path}")
    print(f"[flat-manifest] wrote_stats={stats_path}")
    print(
        f"[flat-manifest] seed={args.seed} tar_count={manifest['tar_count']} "
        f"sample_count={manifest['sample_count']} buffer={args.sample_shuffle_buffer}"
    )
    print("[flat-manifest] result=PASS public_root_changed=false global_stage_flattened=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
