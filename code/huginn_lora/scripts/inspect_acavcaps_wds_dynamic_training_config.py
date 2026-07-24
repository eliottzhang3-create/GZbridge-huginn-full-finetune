"""Validate the full ACAVCAPS manifest for the dynamic LoSATok training route.

This is metadata-only. It never decodes audio and never writes into the public
ACAVCAPS tree. The actual FLAC -> waveform decode remains in the Swift template
at training time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


DEFAULT_DATASET_ROOT = Path("/hpc_stor03/public/shared/data/raa/ACAVCAPS")
DEFAULT_MANIFEST = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/"
    "data/audio_swift/acavcaps_wds/acavcaps_wds_stage_schedule_full_seed20260723.json"
)
EXPECTED_STAGES = {
    "stage1": ("00A", "0M0", "S00"),
    "stage2": ("S0A", "SM0", "0MA"),
    "stage3": ("SMA",),
}
EXPECTED_TARS = {"stage1": 651, "stage2": 398, "stage3": 22}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--stats", default="")
    parser.add_argument("--dataset_root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--per_device_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(f"[config] FAIL {message}")


def main() -> int:
    args = parse_args()
    if min(
        args.world_size,
        args.per_device_batch_size,
        args.gradient_accumulation_steps,
        args.num_train_epochs,
    ) <= 0:
        fail("world_size, batch size, accumulation steps, and epochs must be positive")

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    stats_path = Path(args.stats).expanduser().resolve() if args.stats else manifest_path.with_suffix(".stats.json")
    if not manifest_path.is_file():
        fail(f"manifest does not exist: {manifest_path}")
    if not stats_path.is_file():
        fail(f"stats does not exist: {stats_path}")
    if not dataset_root.is_dir():
        fail(f"public dataset root does not exist: {dataset_root}")
    if dataset_root == manifest_path or dataset_root in manifest_path.parents:
        fail(f"manifest is inside the public dataset root: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(stats, dict):
        fail("manifest and stats must both be JSON objects")
    if Path(str(manifest.get("dataset_root", ""))).resolve() != dataset_root:
        fail(f"manifest dataset_root mismatch: {manifest.get('dataset_root')!r}")
    if stats.get("dataset_root") != str(dataset_root):
        fail(f"stats dataset_root mismatch: {stats.get('dataset_root')!r}")
    if manifest.get("scan_mode") != "full" or stats.get("scan_mode") != "full":
        fail(f"full scan required: manifest={manifest.get('scan_mode')!r} stats={stats.get('scan_mode')!r}")
    if manifest.get("public_root_mutation") != "forbidden" or stats.get("public_root_mutation") != "forbidden":
        fail("manifest/stats do not carry the public-root read-only policy")
    if stats.get("all_pairs_valid") is not True:
        fail(f"stats all_pairs_valid is not true: {stats.get('all_pairs_valid')!r}")

    stages = manifest.get("stages")
    if not isinstance(stages, list) or tuple(stage.get("name") for stage in stages) != tuple(EXPECTED_STAGES):
        fail(f"unexpected stage order: {[stage.get('name') for stage in stages or []]!r}")

    manifest_tar_count = 0
    manifest_sample_count = 0
    stage_sample_counts: dict[str, int] = {}
    for stage in stages:
        name = str(stage["name"])
        tars = stage.get("tars")
        if not isinstance(tars, list) or len(tars) != EXPECTED_TARS[name]:
            fail(f"{name} tar count mismatch: expected={EXPECTED_TARS[name]} actual={len(tars or [])}")
        if tuple(stage.get("categories", [])) != EXPECTED_STAGES[name]:
            fail(f"{name} category order mismatch: {stage.get('categories')!r}")
        stage_count = 0
        for index, entry in enumerate(tars):
            if not isinstance(entry, dict):
                fail(f"{name}[{index}] is not an object")
            tar_path = Path(str(entry.get("path", ""))).resolve()
            if dataset_root not in tar_path.parents or not tar_path.is_file():
                fail(f"{name}[{index}] tar is outside/missing: {tar_path}")
            if entry.get("scan_status") != "scanned" or entry.get("valid") is not True:
                fail(f"{name}[{index}] was not fully validated: scan_status={entry.get('scan_status')!r} valid={entry.get('valid')!r}")
            json_count = entry.get("json_count")
            flac_count = entry.get("flac_count")
            if not isinstance(json_count, int) or not isinstance(flac_count, int) or json_count != flac_count:
                fail(f"{name}[{index}] JSON/FLAC count mismatch: json={json_count!r} flac={flac_count!r}")
            stage_count += json_count
        if stage.get("sample_count") != stage_count:
            fail(f"{name} sample_count mismatch: manifest={stage.get('sample_count')!r} computed={stage_count}")
        stage_sample_counts[name] = stage_count
        manifest_tar_count += len(tars)
        manifest_sample_count += stage_count

    if stats.get("tar_count") != manifest_tar_count or stats.get("sample_count") != manifest_sample_count:
        fail(
            "stats totals mismatch: "
            f"stats=({stats.get('tar_count')},{stats.get('sample_count')}) "
            f"manifest=({manifest_tar_count},{manifest_sample_count})"
        )
    if stats.get("stage_tar_counts") != EXPECTED_TARS or stats.get("stage_sample_counts") != stage_sample_counts:
        fail(
            f"stats stage totals mismatch: tar_counts={stats.get('stage_tar_counts')!r} "
            f"sample_counts={stats.get('stage_sample_counts')!r} expected_samples={stage_sample_counts!r}"
        )

    if os.environ.get("ACAVCAPS_WDS_MAX_TARS_PER_STAGE", "").strip():
        fail("ACAVCAPS_WDS_MAX_TARS_PER_STAGE is set; formal training must consume all tars")
    if os.environ.get("HUGINN_LOSATOK_DYNAMIC_AUDIO_TOKENS", "").strip().lower() not in {"1", "true", "yes"}:
        fail("HUGINN_LOSATOK_DYNAMIC_AUDIO_TOKENS must be enabled for this route")

    global_batch = args.world_size * args.per_device_batch_size * args.gradient_accumulation_steps
    updates_per_epoch = math.ceil(manifest_sample_count / global_batch)
    max_steps = updates_per_epoch * args.num_train_epochs
    print("========== ACAVCAPS DYNAMIC LOSATOK TRAINING CONFIG INSPECT ==========")
    print(f"[manifest] path={manifest_path}")
    print(f"[stats] path={stats_path}")
    print(f"[dataset] public_root={dataset_root} mutation_policy=forbidden")
    print(f"[dataset] tar_count={manifest_tar_count} sample_count={manifest_sample_count}")
    print(f"[dataset] stage_sample_counts={stage_sample_counts}")
    print(f"[runtime] dynamic_audio_tokens=true max_audio_seconds=90 max_audio_tokens=375")
    print(f"[runtime] compressor_kernel=11 stride=6 adaptive_pool=false")
    print(f"[runtime] decode_policy=training_time_streaming_only offline_audio_decode=false")
    print(f"[distributed] world_size={args.world_size} per_device_batch={args.per_device_batch_size} accumulation={args.gradient_accumulation_steps}")
    print(f"[distributed] global_effective_batch={global_batch} updates_per_epoch={updates_per_epoch} max_steps={max_steps}")
    print("[result] status=PASS manifest=full stats=consistent tar_paths=valid all_pairs=valid public_root=read_only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
