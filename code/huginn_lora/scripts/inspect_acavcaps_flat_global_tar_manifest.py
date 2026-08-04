"""Strictly audit the flat all-stage ACAVCAPS tar-shuffle manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any


PUBLIC_ROOT = Path("/hpc_stor03/public/shared/data/raa/ACAVCAPS")
EXPECTED_STAGE_NAMES = ("stage1", "stage2", "stage3")
EXPECTED_TAR_COUNT = 1071
EXPECTED_SAMPLE_COUNT = 4_664_169


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--stats", default="")
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--expected_tar_count", type=int, default=EXPECTED_TAR_COUNT)
    parser.add_argument("--expected_sample_count", type=int, default=EXPECTED_SAMPLE_COUNT)
    parser.add_argument("--expected_buffer", type=int, default=512)
    parser.add_argument("--check_tar_files", action="store_true")
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(f"[flat-preflight] FAIL {message}")


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


def main() -> int:
    args = parse_args()
    if min(args.world_size, args.per_device_batch_size, args.gradient_accumulation_steps) <= 0:
        fail("world_size, per_device_batch_size, and gradient_accumulation_steps must be positive")
    manifest_path = Path(args.manifest).expanduser().resolve()
    stats_path = Path(args.stats).expanduser().resolve() if args.stats else manifest_path.with_suffix(".stats.json")
    if PUBLIC_ROOT.resolve() in manifest_path.parents or PUBLIC_ROOT.resolve() in stats_path.parents:
        fail("manifest/stats must be private, outside the public ACAVCAPS root")
    manifest = read_json(manifest_path)
    stats = read_json(stats_path)

    if Path(str(manifest.get("dataset_root", ""))).resolve() != PUBLIC_ROOT.resolve():
        fail(f"manifest dataset_root mismatch: {manifest.get('dataset_root')!r}")
    if stats.get("dataset_root") != str(PUBLIC_ROOT):
        fail(f"stats dataset_root mismatch: {stats.get('dataset_root')!r}")
    if manifest.get("public_root_mutation") != "forbidden" or stats.get("public_root_mutation") != "forbidden":
        fail("read-only public-root policy is missing")
    if manifest.get("scan_mode") != "derived_from_full" or stats.get("scan_mode") != "derived_from_full":
        fail("manifest/stats must be derived_from_full")
    if manifest.get("schedule_policy") != "global_tar_order_shuffle_all_stages_v1_per_tar_buffer_shuffle":
        fail(f"unexpected schedule policy: {manifest.get('schedule_policy')!r}")
    if stats.get("schedule_policy") != manifest.get("schedule_policy"):
        fail("stats schedule policy differs from manifest")
    if int(manifest.get("sample_shuffle_buffer", -1)) != args.expected_buffer:
        fail(
            f"sample shuffle buffer mismatch: expected={args.expected_buffer} "
            f"actual={manifest.get('sample_shuffle_buffer')!r}"
        )

    source_path = Path(str(manifest.get("source_manifest", ""))).expanduser().resolve()
    if not source_path.is_file():
        fail(f"source full manifest is missing: {source_path}")
    source_stats_path = source_path.with_suffix(".stats.json")
    if not source_stats_path.is_file():
        fail(f"source full stats are missing: {source_stats_path}")
    source_manifest = read_json(source_path)
    source_stats = read_json(source_stats_path)
    source_digest = sha256(source_path)
    if manifest.get("source_manifest_sha256") != source_digest or stats.get("source_manifest_sha256") != source_digest:
        fail("source manifest SHA-256 lineage mismatch")
    if source_manifest.get("scan_mode") != "full" or source_stats.get("all_pairs_valid") is not True:
        fail("source is no longer a fully validated full manifest")
    if tuple(manifest.get("source_stage_order", [])) != EXPECTED_STAGE_NAMES:
        fail(f"unexpected source stage order in flat manifest: {manifest.get('source_stage_order')!r}")

    source_entries: list[dict[str, Any]] = []
    source_stages = source_manifest.get("stages")
    if not isinstance(source_stages, list) or tuple(stage.get("name") for stage in source_stages) != EXPECTED_STAGE_NAMES:
        fail("source stage order is not stage1/stage2/stage3")
    for stage in source_stages:
        stage_name = str(stage["name"])
        tars = stage.get("tars")
        if not isinstance(tars, list):
            fail(f"source {stage_name} has no tar list")
        for source_order_index, entry in enumerate(tars):
            if not isinstance(entry, dict):
                fail(f"source {stage_name}[{source_order_index}] is not an object")
            if entry.get("scan_status") != "scanned" or entry.get("valid") is not True:
                fail(f"source {stage_name}[{source_order_index}] is not fully validated")
            source_entry = dict(entry)
            source_entry["source_stage"] = stage_name
            source_entry["source_stage_order_index"] = source_order_index
            source_entry["path"] = str(Path(str(entry.get("path", ""))).resolve())
            if PUBLIC_ROOT.resolve() not in Path(source_entry["path"]).parents:
                fail(f"source {stage_name}[{source_order_index}] tar is outside public root")
            if (
                not isinstance(entry.get("json_count"), int)
                or int(entry.get("json_count", 0)) <= 0
                or entry.get("json_count") != entry.get("flac_count")
            ):
                fail(f"source {stage_name}[{source_order_index}] JSON/FLAC counts are invalid")
            source_entries.append(source_entry)

    if len(source_entries) != args.expected_tar_count:
        fail(f"source tar count mismatch: expected={args.expected_tar_count} actual={len(source_entries)}")
    source_sample_count = sum(int(entry["json_count"]) for entry in source_entries)
    if source_sample_count != args.expected_sample_count:
        fail(f"source sample count mismatch: expected={args.expected_sample_count} actual={source_sample_count}")

    actual_tars = manifest.get("tars")
    if not isinstance(actual_tars, list):
        fail("flat manifest must contain one top-level tars list")
    if "stages" in manifest:
        fail("flat manifest must not contain a training stages list")
    if len(actual_tars) != args.expected_tar_count:
        fail(f"flat tar count mismatch: expected={args.expected_tar_count} actual={len(actual_tars)}")
    if manifest.get("tar_count") != len(actual_tars) or stats.get("tar_count") != len(actual_tars):
        fail("flat tar_count metadata mismatch")

    expected_entries = list(source_entries)
    random.Random(int(manifest["seed"])).shuffle(expected_entries)
    seen_paths: set[str] = set()
    total_samples = 0
    category_counts: Counter[str] = Counter()
    for order_index, (actual, expected) in enumerate(zip(actual_tars, expected_entries)):
        if not isinstance(actual, dict):
            fail(f"flat tars[{order_index}] is not an object")
        actual_path = str(Path(str(actual.get("path", ""))).resolve())
        expected_path = str(expected["path"])
        if actual_path != expected_path:
            fail(
                f"global tar permutation mismatch at order={order_index}: "
                f"actual={actual_path} expected={expected_path}"
            )
        if actual.get("order_index") != order_index:
            fail(f"order_index mismatch at {order_index}: {actual.get('order_index')!r}")
        for field in ("source_stage", "source_stage_order_index", "category", "json_count", "flac_count", "valid", "scan_status"):
            if actual.get(field) != expected.get(field):
                fail(
                    f"provenance mismatch at order={order_index} field={field}: "
                    f"actual={actual.get(field)!r} expected={expected.get(field)!r}"
                )
        if actual_path in seen_paths:
            fail(f"duplicate flat tar path: {actual_path}")
        seen_paths.add(actual_path)
        if args.check_tar_files:
            tar_path = Path(actual_path)
            if not tar_path.is_file() or PUBLIC_ROOT.resolve() not in tar_path.parents:
                fail(f"tar path is missing or outside public root: {tar_path}")
        total_samples += int(actual["json_count"])
        category_counts[str(actual["category"])] += 1

    if total_samples != args.expected_sample_count:
        fail(f"flat sample count mismatch: expected={args.expected_sample_count} actual={total_samples}")
    if manifest.get("sample_count") != total_samples or stats.get("sample_count") != total_samples:
        fail("flat sample_count metadata mismatch")
    if stats.get("category_tar_counts") != dict(sorted(category_counts.items())):
        fail("category tar counts mismatch")
    if stats.get("all_pairs_valid") is not True:
        fail("flat stats all_pairs_valid is not true")

    global_batch = args.world_size * args.per_device_batch_size * args.gradient_accumulation_steps
    updates = math.ceil(total_samples / global_batch)
    print("========== ACAVCAPS FLAT GLOBAL-TAR MANIFEST PREFLIGHT ==========")
    print(f"[manifest] path={manifest_path}")
    print(f"[source] path={source_path} sha256={source_digest}")
    print(f"[schedule] seed={manifest['seed']} policy={manifest['schedule_policy']}")
    print(f"[schedule] all_stages_flattened=true source_stage_order={EXPECTED_STAGE_NAMES}")
    print(f"[dataset] tar_count={len(actual_tars)} sample_count={total_samples}")
    print(f"[dataset] category_tar_counts={dict(sorted(category_counts.items()))}")
    print(
        f"[runtime] world_size={args.world_size} per_device_batch={args.per_device_batch_size} "
        f"accumulation={args.gradient_accumulation_steps} global_batch={global_batch} "
        f"nominal_updates={updates}"
    )
    print(
        f"[result] status=PASS metadata_only=true tar_order=global_all_stages "
        f"buffer={args.expected_buffer} public_root_changed=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
