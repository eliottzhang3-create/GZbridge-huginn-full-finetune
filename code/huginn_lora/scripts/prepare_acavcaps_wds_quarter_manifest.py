"""Derive a private per-category quarter ACAVCAPS WebDataset manifest.

The source manifest has already completed a full read-only tar-pair
validation.  This program intentionally does not reopen or decode any tar:
it selects metadata entries from that validated manifest and writes new,
private JSON manifest/stat files.  For each category it takes ceil(N / 4)
entries from the category's order in the source stage schedule; the selected
entries are then emitted in their original stage order.  The latter preserves
the source stage-level random tar order, while the runtime WebDataset loader
continues to buffer-shuffle samples within each tar.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


PUBLIC_ROOT = Path("/hpc_stor03/public/shared/data/raa/ACAVCAPS")
DEFAULT_SOURCE = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/"
    "data/audio_swift/acavcaps_wds/acavcaps_wds_stage_schedule_full_seed20260723.json"
)
DEFAULT_OUTPUT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/"
    "data/audio_swift/acavcaps_wds/acavcaps_wds_stage_schedule_quarter_ceil_seed20260723.json"
)
EXPECTED_STAGES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("stage1", ("00A", "0M0", "S00")),
    ("stage2", ("S0A", "SM0", "0MA")),
    ("stage3", ("SMA",)),
)
EXPECTED_SOURCE_CATEGORY_TARS = {
    "00A": 14,
    "0M0": 159,
    "S00": 478,
    "S0A": 98,
    "SM0": 293,
    "0MA": 7,
    "SMA": 22,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_manifest", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output_manifest", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing private derived manifest and its stats sidecar.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"Missing JSON file: {path}") from None
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON root must be an object: {path}")
    return payload


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_private_output(path: Path) -> None:
    resolved = path.resolve(strict=False)
    root = PUBLIC_ROOT.resolve()
    if resolved == root or root in resolved.parents:
        raise SystemExit(f"Refusing to write inside public ACAVCAPS root: {resolved}")


def atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_source(manifest: dict[str, Any], stats: dict[str, Any], source_path: Path) -> list[dict[str, Any]]:
    if Path(str(manifest.get("dataset_root", ""))).resolve() != PUBLIC_ROOT.resolve():
        raise SystemExit(f"Source manifest does not point to the public ACAVCAPS root: {source_path}")
    if stats.get("dataset_root") != str(PUBLIC_ROOT):
        raise SystemExit("Source stats dataset_root mismatch")
    if manifest.get("scan_mode") != "full" or stats.get("scan_mode") != "full":
        raise SystemExit("Quarter derivation requires a full source manifest and full source stats")
    if manifest.get("public_root_mutation") != "forbidden" or stats.get("public_root_mutation") != "forbidden":
        raise SystemExit("Source manifest/stats lack the read-only public-root policy")
    if stats.get("all_pairs_valid") is not True:
        raise SystemExit(f"Source tar-pair validation is not passed: {stats.get('all_pairs_valid')!r}")

    stages = manifest.get("stages")
    expected_names = tuple(name for name, _ in EXPECTED_STAGES)
    if not isinstance(stages, list) or tuple(stage.get("name") for stage in stages) != expected_names:
        raise SystemExit(f"Unexpected source stage order: {[stage.get('name') for stage in stages or []]!r}")
    return stages


def validate_and_select_stage(stage: dict[str, Any], expected_name: str, categories: tuple[str, ...]) -> tuple[dict[str, Any], dict[str, int]]:
    if tuple(stage.get("categories", [])) != categories:
        raise SystemExit(f"{expected_name} source categories mismatch: {stage.get('categories')!r}")
    entries = stage.get("tars")
    if not isinstance(entries, list) or not entries:
        raise SystemExit(f"{expected_name} source tars are missing")

    by_category: dict[str, list[tuple[int, dict[str, Any]]]] = {category: [] for category in categories}
    seen_paths: set[str] = set()
    for source_order_index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise SystemExit(f"{expected_name}[{source_order_index}] is not an object")
        category = entry.get("category")
        if category not in by_category:
            raise SystemExit(f"{expected_name}[{source_order_index}] unexpected category: {category!r}")
        tar_path = Path(str(entry.get("path", ""))).resolve()
        if PUBLIC_ROOT.resolve() not in tar_path.parents or not tar_path.is_file():
            raise SystemExit(f"{expected_name}[{source_order_index}] is outside/missing public tar: {tar_path}")
        if str(tar_path) in seen_paths:
            raise SystemExit(f"{expected_name} contains a duplicate tar path: {tar_path}")
        seen_paths.add(str(tar_path))
        required_valid = {
            "scan_status": "scanned",
            "valid": True,
        }
        for field, expected in required_valid.items():
            if entry.get(field) != expected:
                raise SystemExit(
                    f"{expected_name}[{source_order_index}] is not fully source-validated: "
                    f"{field}={entry.get(field)!r}"
                )
        json_count, flac_count = entry.get("json_count"), entry.get("flac_count")
        if not isinstance(json_count, int) or json_count <= 0 or json_count != flac_count:
            raise SystemExit(
                f"{expected_name}[{source_order_index}] invalid JSON/FLAC counts: "
                f"json={json_count!r} flac={flac_count!r}"
            )
        by_category[category].append((source_order_index, entry))

    selected_source_indices: set[int] = set()
    selected_per_category: dict[str, int] = {}
    source_per_category: dict[str, int] = {}
    for category in categories:
        category_entries = by_category[category]
        source_count = len(category_entries)
        expected_source_count = EXPECTED_SOURCE_CATEGORY_TARS[category]
        if source_count != expected_source_count:
            raise SystemExit(
                f"{expected_name}/{category} source tar count mismatch: "
                f"actual={source_count} expected={expected_source_count}"
            )
        take_count = math.ceil(source_count / 4)
        source_per_category[category] = source_count
        selected_per_category[category] = take_count
        selected_source_indices.update(index for index, _ in category_entries[:take_count])

    selected_entries: list[dict[str, Any]] = []
    for source_order_index, source_entry in enumerate(entries):
        if source_order_index not in selected_source_indices:
            continue
        entry = copy.deepcopy(source_entry)
        entry["source_order_index"] = source_order_index
        entry["order_index"] = len(selected_entries)
        selected_entries.append(entry)

    expected_selected = sum(selected_per_category.values())
    if len(selected_entries) != expected_selected:
        raise SystemExit(
            f"{expected_name} selected tar count mismatch: actual={len(selected_entries)} expected={expected_selected}"
        )
    selected_sample_count = sum(int(entry["json_count"]) for entry in selected_entries)
    return (
        {
            "name": expected_name,
            "categories": list(categories),
            "seed": stage.get("seed"),
            "tar_count": len(selected_entries),
            "sample_count": selected_sample_count,
            "source_tar_count": len(entries),
            "source_sample_count": stage.get("sample_count"),
            "source_category_tar_counts": source_per_category,
            "selected_category_tar_counts": selected_per_category,
            "selection_policy": "per_category_first_ceil_quarter_in_source_stage_random_order_then_preserve_source_stage_order",
            "tars": selected_entries,
        },
        selected_per_category,
    )


def main() -> int:
    args = parse_args()
    source_path = Path(args.source_manifest).expanduser().resolve()
    output_path = Path(args.output_manifest).expanduser().resolve()
    stats_output_path = output_path.with_suffix(".stats.json")
    ensure_private_output(output_path)
    ensure_private_output(stats_output_path)
    if output_path == source_path:
        raise SystemExit("Derived output manifest must differ from the source manifest")
    if not args.overwrite and (output_path.exists() or stats_output_path.exists()):
        raise SystemExit(
            f"Derived output already exists: {output_path} or {stats_output_path}; "
            "use --overwrite only when intentionally rebuilding it"
        )

    source_stats_path = source_path.with_suffix(".stats.json")
    source_manifest = read_json(source_path)
    source_stats = read_json(source_stats_path)
    source_stages = validate_source(source_manifest, source_stats, source_path)

    selected_stages: list[dict[str, Any]] = []
    category_tar_counts: dict[str, int] = {}
    for stage, (stage_name, categories) in zip(source_stages, EXPECTED_STAGES):
        selected_stage, selected_counts = validate_and_select_stage(stage, stage_name, categories)
        selected_stages.append(selected_stage)
        category_tar_counts.update(selected_counts)
        print(
            f"[stage] name={stage_name} source_tars={selected_stage['source_tar_count']} "
            f"selected_tars={selected_stage['tar_count']} samples={selected_stage['sample_count']} "
            f"selected_by_category={selected_counts}",
            flush=True,
        )

    total_tars = sum(int(stage["tar_count"]) for stage in selected_stages)
    total_samples = sum(int(stage["sample_count"]) for stage in selected_stages)
    source_digest = digest(source_path)
    manifest = {
        "schema_version": 2,
        "dataset_root": str(PUBLIC_ROOT),
        "public_root_mutation": "forbidden",
        "scan_mode": "derived_from_full",
        "seed": source_manifest.get("seed"),
        "sample_shuffle_buffer": source_manifest.get("sample_shuffle_buffer"),
        "schedule_policy": "stage_order_fixed_tar_order_shuffled_per_stage_sample_buffered_per_tar",
        "derived_from": {
            "source_manifest": str(source_path),
            "source_manifest_sha256": source_digest,
            "source_scan_mode": source_manifest.get("scan_mode"),
            "source_all_pairs_valid": source_stats.get("all_pairs_valid"),
            "selection_policy": "per_category_first_ceil_quarter_in_source_stage_random_order_then_preserve_source_stage_order",
            "selection_fraction": "ceil(N/4) per category",
        },
        "stages": selected_stages,
    }
    stats = {
        "schema_version": 2,
        "manifest_path": str(output_path),
        "dataset_root": str(PUBLIC_ROOT),
        "public_root_mutation": "forbidden",
        "scan_mode": "derived_from_full",
        "seed": source_manifest.get("seed"),
        "sample_shuffle_buffer": source_manifest.get("sample_shuffle_buffer"),
        "source_manifest": str(source_path),
        "source_manifest_sha256": source_digest,
        "source_all_pairs_valid": source_stats.get("all_pairs_valid"),
        "selection_policy": "per_category_first_ceil_quarter_in_source_stage_random_order_then_preserve_source_stage_order",
        "selection_fraction": "ceil(N/4) per category",
        "stage_order": [stage["name"] for stage in selected_stages],
        "stage_tar_counts": {stage["name"]: stage["tar_count"] for stage in selected_stages},
        "stage_sample_counts": {stage["name"]: stage["sample_count"] for stage in selected_stages},
        "category_tar_counts": category_tar_counts,
        "tar_count": total_tars,
        "sample_count": total_samples,
        "all_pairs_valid": True,
    }
    atomic_json_write(output_path, manifest)
    atomic_json_write(stats_output_path, stats)
    print(f"[manifest] wrote_private_manifest={output_path}")
    print(f"[stats] wrote_private_stats={stats_output_path}")
    print(f"[summary] selected_tar_count={total_tars} selected_sample_count={total_samples}")
    print("[result] status=PASS public_root_changed=false source_tar_rescan=false audio_decode=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
