"""Strict metadata-only validation for the derived ACAVCAPS quarter manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PUBLIC_ROOT = Path("/hpc_stor03/public/shared/data/raa/ACAVCAPS")
DEFAULT_MANIFEST = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/"
    "data/audio_swift/acavcaps_wds/acavcaps_wds_stage_schedule_quarter_ceil_seed20260723.json"
)
EXPECTED_STAGES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("stage1", ("00A", "0M0", "S00")),
    ("stage2", ("S0A", "SM0", "0MA")),
    ("stage3", ("SMA",)),
)
SOURCE_CATEGORY_TARS = {
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
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--stats", default="")
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON root must be an object: {path}")
    return payload


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fail(message: str) -> None:
    raise SystemExit(f"[quarter-preflight] FAIL {message}")


def main() -> int:
    args = parse_args()
    if min(args.world_size, args.per_device_batch_size, args.gradient_accumulation_steps) <= 0:
        fail("world_size, per_device_batch_size, and gradient_accumulation_steps must be positive")
    manifest_path = Path(args.manifest).expanduser().resolve()
    stats_path = Path(args.stats).expanduser().resolve() if args.stats else manifest_path.with_suffix(".stats.json")
    if not manifest_path.is_file() or not stats_path.is_file():
        fail(f"missing manifest/stats: manifest={manifest_path} stats={stats_path}")
    if PUBLIC_ROOT.resolve() in manifest_path.parents or PUBLIC_ROOT.resolve() in stats_path.parents:
        fail("private manifest/stats must not be written inside the public ACAVCAPS root")
    manifest, stats = load_json(manifest_path), load_json(stats_path)
    if Path(str(manifest.get("dataset_root", ""))).resolve() != PUBLIC_ROOT.resolve():
        fail(f"manifest dataset_root mismatch: {manifest.get('dataset_root')!r}")
    if stats.get("dataset_root") != str(PUBLIC_ROOT):
        fail(f"stats dataset_root mismatch: {stats.get('dataset_root')!r}")
    if manifest.get("public_root_mutation") != "forbidden" or stats.get("public_root_mutation") != "forbidden":
        fail("read-only public-root policy missing")
    if manifest.get("scan_mode") != "derived_from_full" or stats.get("scan_mode") != "derived_from_full":
        fail("expected a manifest derived from a fully validated source")
    if stats.get("all_pairs_valid") is not True:
        fail(f"inherited all_pairs_valid must be true, got {stats.get('all_pairs_valid')!r}")

    lineage = manifest.get("derived_from")
    source_path = Path(str(lineage.get("source_manifest", ""))).resolve() if isinstance(lineage, dict) else None
    if source_path is None or not source_path.is_file():
        fail(f"source full manifest missing: {source_path}")
    source_stats_path = source_path.with_suffix(".stats.json")
    if not source_stats_path.is_file():
        fail(f"source full stats missing: {source_stats_path}")
    source_manifest, source_stats = load_json(source_path), load_json(source_stats_path)
    actual_source_digest = sha256(source_path)
    if lineage.get("source_manifest_sha256") != actual_source_digest:
        fail("source manifest digest changed after quarter manifest derivation")
    if stats.get("source_manifest_sha256") != actual_source_digest:
        fail("stats source manifest digest mismatch")
    if source_manifest.get("scan_mode") != "full" or source_stats.get("all_pairs_valid") is not True:
        fail("source manifest/stats are no longer a full validated source")

    stages, source_stages = manifest.get("stages"), source_manifest.get("stages")
    expected_names = tuple(name for name, _ in EXPECTED_STAGES)
    if not isinstance(stages, list) or tuple(stage.get("name") for stage in stages) != expected_names:
        fail(f"derived stage order mismatch: {[stage.get('name') for stage in stages or []]!r}")
    if not isinstance(source_stages, list) or tuple(stage.get("name") for stage in source_stages) != expected_names:
        fail("source stage order mismatch")

    total_tars = 0
    total_samples = 0
    derived_stage_tars: dict[str, int] = {}
    derived_stage_samples: dict[str, int] = {}
    derived_categories: dict[str, int] = {}
    for stage, source_stage, (stage_name, categories) in zip(stages, source_stages, EXPECTED_STAGES):
        if tuple(stage.get("categories", [])) != categories:
            fail(f"{stage_name} category order mismatch: {stage.get('categories')!r}")
        source_tars = source_stage.get("tars")
        tars = stage.get("tars")
        if not isinstance(source_tars, list) or not isinstance(tars, list):
            fail(f"{stage_name} missing tars")
        source_by_path = {str(Path(str(entry.get("path", ""))).resolve()): (index, entry) for index, entry in enumerate(source_tars)}
        if len(source_by_path) != len(source_tars):
            fail(f"{stage_name} source contains duplicate paths")
        source_category_counts = {category: 0 for category in categories}
        for entry in source_tars:
            category = entry.get("category")
            if category in source_category_counts:
                source_category_counts[category] += 1
        last_source_index = -1
        selected_category_counts = {category: 0 for category in categories}
        sample_count = 0
        seen: set[str] = set()
        for output_index, entry in enumerate(tars):
            path = str(Path(str(entry.get("path", ""))).resolve())
            if path in seen:
                fail(f"{stage_name} duplicate selected path: {path}")
            seen.add(path)
            if path not in source_by_path:
                fail(f"{stage_name}[{output_index}] absent from source manifest: {path}")
            source_index, source_entry = source_by_path[path]
            if source_index <= last_source_index:
                fail(f"{stage_name} no longer preserves the source global randomized tar order")
            last_source_index = source_index
            if entry.get("source_order_index") != source_index or entry.get("order_index") != output_index:
                fail(f"{stage_name}[{output_index}] order provenance mismatch")
            if entry.get("category") != source_entry.get("category") or entry.get("category") not in selected_category_counts:
                fail(f"{stage_name}[{output_index}] category provenance mismatch")
            if entry.get("scan_status") != "scanned" or entry.get("valid") is not True:
                fail(f"{stage_name}[{output_index}] inherited validation is incomplete")
            json_count, flac_count = entry.get("json_count"), entry.get("flac_count")
            if not isinstance(json_count, int) or json_count <= 0 or json_count != flac_count:
                fail(f"{stage_name}[{output_index}] JSON/FLAC metadata mismatch")
            if not Path(path).is_file() or PUBLIC_ROOT.resolve() not in Path(path).parents:
                fail(f"{stage_name}[{output_index}] public tar missing/outside root: {path}")
            selected_category_counts[entry["category"]] += 1
            sample_count += json_count
        expected_selected_counts = {category: math.ceil(SOURCE_CATEGORY_TARS[category] / 4) for category in categories}
        if source_category_counts != {category: SOURCE_CATEGORY_TARS[category] for category in categories}:
            fail(f"{stage_name} source category counts changed: {source_category_counts!r}")
        if selected_category_counts != expected_selected_counts:
            fail(f"{stage_name} per-category ceil-quarter counts mismatch: {selected_category_counts!r}")
        if stage.get("selected_category_tar_counts") != selected_category_counts:
            fail(f"{stage_name} selected category stats mismatch")
        if stage.get("tar_count") != len(tars) or stage.get("sample_count") != sample_count:
            fail(f"{stage_name} stage totals mismatch")
        derived_stage_tars[stage_name] = len(tars)
        derived_stage_samples[stage_name] = sample_count
        derived_categories.update(selected_category_counts)
        total_tars += len(tars)
        total_samples += sample_count

    if total_tars != 271:
        fail(f"derived tar total must be 271 with ceil-quarter policy, got {total_tars}")
    expected_stats = {
        "stage_tar_counts": derived_stage_tars,
        "stage_sample_counts": derived_stage_samples,
        "category_tar_counts": derived_categories,
        "tar_count": total_tars,
        "sample_count": total_samples,
    }
    for field, expected in expected_stats.items():
        if stats.get(field) != expected:
            fail(f"stats {field} mismatch: actual={stats.get(field)!r} expected={expected!r}")

    global_batch = args.world_size * args.per_device_batch_size * args.gradient_accumulation_steps
    updates = math.ceil(total_samples / global_batch)
    print("========== ACAVCAPS WDS QUARTER MANIFEST PREFLIGHT ==========")
    print(f"[manifest] path={manifest_path}")
    print(f"[source] path={source_path} sha256={actual_source_digest}")
    print("[selection] per_category=ceil(N/4) source_order=randomized_full_manifest_order")
    print("[shuffle] stage_tar_order=preserved_source_random_order sample_order=webdataset_buffer_shuffle")
    print(f"[dataset] tar_count={total_tars} sample_count={total_samples}")
    print(f"[dataset] stage_tar_counts={derived_stage_tars}")
    print(f"[dataset] stage_sample_counts={derived_stage_samples}")
    print(f"[dataset] category_tar_counts={derived_categories}")
    print(
        f"[legacy_runtime] world_size={args.world_size} per_device_batch={args.per_device_batch_size} "
        f"accumulation={args.gradient_accumulation_steps} global_effective_batch={global_batch} updates_per_epoch={updates}"
    )
    print("[result] status=PASS metadata_only=true public_root_changed=false tar_rescan=false audio_decode=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
