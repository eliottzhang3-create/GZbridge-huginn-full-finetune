from __future__ import annotations

import argparse
import collections
import importlib.metadata
import inspect
import json
import os
import platform
import random
import sys
import tarfile
from pathlib import Path
from typing import Any


DEFAULT_DATASET_ROOT = Path("/hpc_stor03/public/shared/data/raa/ACAVCAPS")
DEFAULT_MANIFEST = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/"
    "data/audio_swift/acavcaps_wds/acavcaps_wds_stage_schedule.json"
)
TAR_SUFFIXES = (".tar", ".tar.gz", ".tgz")
STAGES = (
    ("stage1", ("00A", "0M0", "S00")),
    ("stage2", ("S0A", "SM0", "0MA")),
    ("stage3", ("SMA",)),
)


def header(title: str) -> None:
    print(f"========== {title} ==========", flush=True)


def is_tar_path(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in TAR_SUFFIXES)


def list_category_tars(dataset_root: Path, category: str) -> list[Path]:
    category_dir = dataset_root / category
    if not category_dir.is_dir():
        raise FileNotFoundError(f"Missing ACAVCAPS category directory: {category_dir}")
    paths = [path for path in category_dir.rglob("*") if path.is_file() and is_tar_path(path)]
    return sorted(paths, key=lambda path: path.relative_to(dataset_root).as_posix())


def json_caption_info(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"payload_type": type(payload).__name__, "long_present": False, "long_count": 0}
    value = payload.get("long")
    if isinstance(value, str):
        count = 1 if value.strip() else 0
    elif isinstance(value, list):
        count = sum(isinstance(item, str) and bool(item.strip()) for item in value)
    else:
        count = 0
    return {
        "payload_type": "dict",
        "long_present": "long" in payload,
        "long_count": count,
        "payload_keys": sorted(str(key) for key in payload),
    }


def scan_tar(tar_path: Path, preview_count: int) -> dict[str, Any]:
    json_names: list[str] = []
    flac_names: list[str] = []
    other_audio_names: list[str] = []
    invalid_json: list[dict[str, str]] = []
    invalid_caption: list[dict[str, str]] = []
    preview: list[dict[str, Any]] = []
    key_counter: collections.Counter[str] = collections.Counter()

    # r|* is intentionally sequential: this is a read-only inspection of a
    # compressed shard and does not seek or extract anything into the filesystem.
    with tarfile.open(tar_path, mode="r|*") as tar_obj:
        for member in tar_obj:
            if not member.isfile():
                continue
            lower_name = member.name.lower()
            if lower_name.endswith(".json"):
                json_names.append(member.name)
                extracted = tar_obj.extractfile(member)
                if extracted is None:
                    invalid_json.append({"member": member.name, "error": "extractfile returned None"})
                    continue
                try:
                    payload = json.loads(extracted.read().decode("utf-8"))
                except Exception as exc:  # noqa: BLE001 - report malformed public data, do not mutate it
                    invalid_json.append({"member": member.name, "error": f"{type(exc).__name__}: {exc}"})
                    continue
                info = json_caption_info(payload)
                if int(info.get("long_count", 0)) <= 0:
                    invalid_caption.append({"member": member.name, "error": "missing non-empty long caption"})
                for key in info.get("payload_keys", []):
                    key_counter[key] += 1
                if len(preview) < preview_count:
                    preview.append({"json_member": member.name, **info})
            elif lower_name.endswith(".flac"):
                flac_names.append(member.name)
            elif lower_name.endswith((".wav", ".mp3", ".ogg", ".m4a", ".aac")):
                other_audio_names.append(member.name)

    json_set = set(json_names)
    flac_set = set(flac_names)
    expected_flac = {f"{name[:-5]}.flac" for name in json_names}
    missing_flac = sorted(expected_flac - flac_set)
    orphan_flac = sorted(flac_set - expected_flac)
    duplicate_json = sorted(name for name, count in collections.Counter(json_names).items() if count > 1)
    duplicate_flac = sorted(name for name, count in collections.Counter(flac_names).items() if count > 1)

    return {
        "path": str(tar_path),
        "json_count": len(json_names),
        "flac_count": len(flac_names),
        "missing_flac_count": len(missing_flac),
        "orphan_flac_count": len(orphan_flac),
        "duplicate_json_count": len(duplicate_json),
        "duplicate_flac_count": len(duplicate_flac),
        "other_audio_count": len(other_audio_names),
        "invalid_json_count": len(invalid_json),
        "invalid_caption_count": len(invalid_caption),
        "missing_flac_preview": missing_flac[:5],
        "orphan_flac_preview": orphan_flac[:5],
        "invalid_json_preview": invalid_json[:5],
        "invalid_caption_preview": invalid_caption[:5],
        "preview": preview,
        "json_key_counts": dict(key_counter),
        "valid": not (missing_flac or duplicate_json or duplicate_flac or invalid_json or invalid_caption),
    }


def safe_manifest_path(manifest_path: Path, dataset_root: Path) -> None:
    root = dataset_root.resolve()
    output = manifest_path.resolve(strict=False)
    if output == root or root in output.parents:
        raise ValueError(
            "Refusing to write a manifest inside the public ACAVCAPS root; "
            f"dataset_root={root} output={output}"
        )


def progress_path_for(manifest_path: Path) -> Path:
    return manifest_path.with_suffix(".progress.json")


def write_progress(
    progress_path: Path,
    dataset_root: Path,
    args: argparse.Namespace,
    completed: dict[tuple[str, int, str], dict[str, Any]],
    *,
    status: str = "running",
) -> None:
    payload = {
        "schema_version": 1,
        "status": status,
        "dataset_root": str(dataset_root),
        "scan_mode": args.scan_mode,
        "seed": args.seed,
        "sample_shuffle_buffer": args.sample_shuffle_buffer,
        "completed_tars": [
            result for _, result in sorted(completed.items(), key=lambda item: item[0])
        ],
    }
    safe_manifest_path(progress_path, dataset_root)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = progress_path.with_name(f"{progress_path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, progress_path)


def load_progress(
    progress_path: Path,
    dataset_root: Path,
    args: argparse.Namespace,
) -> dict[tuple[str, int, str], dict[str, Any]]:
    if not args.resume or not progress_path.is_file():
        return {}
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    expected = {
        "dataset_root": str(dataset_root),
        "scan_mode": args.scan_mode,
        "seed": args.seed,
        "sample_shuffle_buffer": args.sample_shuffle_buffer,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"Progress checkpoint metadata mismatch for {key}: "
                f"stored={payload.get(key)!r} expected={value!r}; use a new manifest_out"
            )
    completed: dict[tuple[str, int, str], dict[str, Any]] = {}
    for result in payload.get("completed_tars", []):
        if not isinstance(result, dict):
            raise ValueError(f"Invalid completed tar entry in progress checkpoint: {result!r}")
        stage = str(result.get("stage", ""))
        order_index = result.get("order_index")
        path = str(result.get("path", ""))
        if not stage or not isinstance(order_index, int) or not path or result.get("valid") is None:
            raise ValueError(f"Invalid completed tar metadata in progress checkpoint: {result!r}")
        completed[(stage, order_index, path)] = result
    print(f"[resume] progress_path={progress_path} completed_tars={len(completed)}", flush=True)
    return completed


def inspect_webdataset(manifest: dict[str, Any], buffer_size: int, samples_per_tar: int) -> bool:
    header("WEBDATASET API")
    try:
        import torch
        import webdataset as wds
    except Exception as exc:  # noqa: BLE001 - report remote environment state
        print(f"[wds] import=FAIL {type(exc).__name__}: {exc}")
        return False

    try:
        version = importlib.metadata.version("webdataset")
    except Exception:
        version = getattr(wds, "__version__", "unknown")
    print(f"[wds] version={version}")
    print(f"[wds] module={getattr(wds, '__file__', '<unknown>')}")
    for name in ("WebDataset", "DataPipeline", "WebLoader"):
        value = getattr(wds, name, None)
        print(f"[wds] {name}={value}")
        if value is not None:
            try:
                print(f"[wds] {name}_signature={inspect.signature(value)}")
            except (TypeError, ValueError):
                pass
    web_dataset_cls = getattr(wds, "WebDataset", None)
    if web_dataset_cls is not None:
        try:
            print(f"[wds] WebDataset_is_torch_IterableDataset={issubclass(web_dataset_cls, torch.utils.data.IterableDataset)}")
        except TypeError:
            print("[wds] WebDataset_is_torch_IterableDataset=unknown")

    stage = manifest["stages"][0]
    if not stage["tars"]:
        print("[wds] no tar in stage1; probe skipped")
        return False
    tar_path = stage["tars"][0]["path"]
    print(f"[wds] probe_tar={tar_path}")
    try:
        dataset = wds.WebDataset(tar_path, shardshuffle=False)
        first = next(iter(dataset))
        print(f"[wds] raw_sample_keys={sorted(first.keys())}")
        print(f"[wds] raw_sample_types={{{', '.join(f'{k}:{type(v).__name__}' for k, v in first.items())}}}")
    except Exception as exc:  # noqa: BLE001
        print(f"[wds] raw_stream_probe=FAIL {type(exc).__name__}: {exc}")
        return False

    try:
        shuffled_dataset = wds.WebDataset(tar_path, shardshuffle=False)
        try:
            shuffled_dataset = shuffled_dataset.shuffle(buffer_size)
        except TypeError:
            shuffled_dataset = shuffled_dataset.shuffle(size=buffer_size)
        keys: list[str] = []
        for index, sample in enumerate(shuffled_dataset):
            keys.append(str(sample.get("__key__", "<no-key>")))
            if index + 1 >= samples_per_tar:
                break
        print(f"[wds] buffer_shuffle={buffer_size} sampled_keys={keys}")
    except Exception as exc:  # noqa: BLE001
        print(f"[wds] buffer_shuffle_probe=FAIL {type(exc).__name__}: {exc}")
        return False
    return True


def inspect_swift_iterable_support() -> bool:
    header("SWIFT ITERABLE DATASET COMPATIBILITY")
    try:
        import swift
        import transformers
        from torch.utils.data import IterableDataset
        from swift.pipelines.train.sft import SwiftSft
    except Exception as exc:  # noqa: BLE001
        print(f"[swift] import=FAIL {type(exc).__name__}: {exc}")
        return False

    print(f"[swift] version={getattr(swift, '__version__', 'unknown')}")
    print(f"[swift] module={swift.__file__}")
    print(f"[transformers] version={transformers.__version__}")
    print(f"[swift] SwiftSft={SwiftSft} module={inspect.getfile(SwiftSft)}")
    print(f"[swift] torch_IterableDataset={IterableDataset}")

    swift_root = Path(swift.__file__).resolve().parent
    keywords = ("IterableDataset", "get_train_dataloader", "load_dataset", "dataset_shuffle", "train_dataloader_shuffle")
    matches: list[str] = []
    for source_path in sorted(swift_root.rglob("*.py")):
        try:
            lines = source_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for line_number, line in enumerate(lines, start=1):
            if any(keyword in line for keyword in keywords):
                matches.append(f"{source_path}:{line_number}: {line.strip()}")
    print(f"[swift] source_matches={len(matches)}")
    for line in matches[:120]:
        print(f"[swift-match] {line}")

    try:
        source = inspect.getsource(SwiftSft)
        for line_number, line in enumerate(source.splitlines(), start=1):
            if any(keyword in line for keyword in keywords):
                print(f"[swift-sft-match] {line_number}: {line}")
    except (OSError, TypeError) as exc:
        print(f"[swift] SwiftSft_source=FAIL {type(exc).__name__}: {exc}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only ACAVCAPS tar inventory, private stage manifest, WebDataset probe, and Swift iterable probe."
    )
    parser.add_argument("--dataset_root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--manifest_out", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--sample_shuffle_buffer", type=int, default=512)
    parser.add_argument(
        "--scan_mode",
        choices=("inventory", "sampled", "full"),
        default="inventory",
        help=(
            "inventory only lists and schedules tar files; sampled scans the first N tars per stage; "
            "full scans every tar and is intentionally a long-running IO job"
        ),
    )
    parser.add_argument("--scan_tars_per_stage", type=int, default=2)
    parser.add_argument("--preview_per_tar", type=int, default=2)
    parser.add_argument("--wds_probe_samples", type=int, default=8)
    parser.add_argument("--print_each_tar", action="store_true")
    parser.add_argument("--skip_wds_probe", action="store_true")
    parser.add_argument("--skip_swift_probe", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume full/sampled scanning from the private .progress.json checkpoint if present.",
    )
    parser.add_argument(
        "--progress_interval_tars",
        type=int,
        default=10,
        help="Persist the private progress checkpoint after this many scanned tars.",
    )
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    args = parse_args()
    if args.sample_shuffle_buffer <= 0:
        raise ValueError("--sample_shuffle_buffer must be positive")
    if args.scan_tars_per_stage <= 0:
        raise ValueError("--scan_tars_per_stage must be positive")
    if args.progress_interval_tars <= 0:
        raise ValueError("--progress_interval_tars must be positive")
    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"ACAVCAPS public root does not exist: {dataset_root}")

    header("PREFLIGHT CONTEXT")
    print(f"[env] python={sys.version.split()[0]}")
    print(f"[env] platform={platform.platform()}")
    print(f"[dataset] root={dataset_root}")
    print("[dataset] policy=read_only_public_root")
    print(f"[schedule] seed={args.seed}")
    print(f"[schedule] sample_shuffle_buffer={args.sample_shuffle_buffer}")
    print(f"[schedule] scan_mode={args.scan_mode} scan_tars_per_stage={args.scan_tars_per_stage}")

    manifest_path = Path(args.manifest_out).resolve()
    safe_manifest_path(manifest_path, dataset_root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = progress_path_for(manifest_path)
    completed_results = load_progress(progress_path, dataset_root, args)
    print(f"[progress] checkpoint={progress_path} resume={args.resume} interval_tars={args.progress_interval_tars}")

    stages: list[dict[str, Any]] = []
    all_valid: bool | None = True if args.scan_mode != "inventory" else None
    scanned_tars_since_checkpoint = 0
    for stage_index, (stage_name, categories) in enumerate(STAGES):
        stage_tars: list[tuple[str, Path]] = []
        for category in categories:
            paths = list_category_tars(dataset_root, category)
            print(f"[inventory] category={category} tar_count={len(paths)}")
            stage_tars.extend((category, path) for path in paths)

        rng = random.Random(args.seed + stage_index)
        rng.shuffle(stage_tars)
        tar_entries: list[dict[str, Any]] = []
        stage_sample_count = 0
        scanned_tar_count = 0
        for order_index, (category, tar_path) in enumerate(stage_tars):
            should_scan = args.scan_mode == "full" or (
                args.scan_mode == "sampled" and order_index < args.scan_tars_per_stage
            )
            if should_scan:
                progress_key = (stage_name, order_index, str(tar_path))
                result = completed_results.get(progress_key)
                resumed = result is not None
                if result is None:
                    result = scan_tar(tar_path, args.preview_per_tar)
                    result["scan_status"] = "scanned"
                    result.update({"category": category, "stage": stage_name, "order_index": order_index})
                    completed_results[progress_key] = result
                    scanned_tars_since_checkpoint += 1
                else:
                    result = dict(result)
                stage_sample_count += int(result["json_count"])
                scanned_tar_count += 1
                if all_valid is None:
                    all_valid = bool(result["valid"])
                else:
                    all_valid = all_valid and bool(result["valid"])
                if not resumed and scanned_tars_since_checkpoint >= args.progress_interval_tars:
                    write_progress(progress_path, dataset_root, args, completed_results)
                    print(
                        f"[checkpoint] wrote={progress_path} completed_tars={len(completed_results)}",
                        flush=True,
                    )
                    scanned_tars_since_checkpoint = 0
            else:
                result = {
                    "path": str(tar_path),
                    "scan_status": "not_scanned",
                    "json_count": None,
                    "flac_count": None,
                    "missing_flac_count": None,
                    "orphan_flac_count": None,
                    "duplicate_json_count": None,
                    "duplicate_flac_count": None,
                    "other_audio_count": None,
                    "invalid_json_count": None,
                    "invalid_caption_count": None,
                    "preview": [],
                    "invalid_caption_preview": [],
                    "json_key_counts": {},
                    "valid": None,
                }
            result.update({"category": category, "stage": stage_name, "order_index": order_index})
            tar_entries.append(result)
            if should_scan and (
                args.print_each_tar
                or args.scan_mode != "full"
                or order_index < 3
                or (order_index + 1) % 25 == 0
                or order_index + 1 == len(stage_tars)
            ):
                print(
                    f"[tar] stage={stage_name} order={order_index} category={category} "
                    f"name={tar_path.name} json={result['json_count']} flac={result['flac_count']} "
                    f"missing_flac={result['missing_flac_count']} invalid_json={result['invalid_json_count']} "
                    f"invalid_caption={result['invalid_caption_count']}"
                )
            elif args.scan_mode == "full" and (order_index + 1) % 10 == 0:
                print(
                    f"[progress] stage={stage_name} scanned_tars={order_index + 1}/{len(stage_tars)} "
                    f"validated_tars={scanned_tar_count} resumed_total={len(completed_results)}",
                    flush=True,
                )

        stages.append(
            {
                "name": stage_name,
                "categories": list(categories),
                "seed": args.seed + stage_index,
                "tar_count": len(tar_entries),
                "scanned_tar_count": scanned_tar_count,
                "sample_count": stage_sample_count if scanned_tar_count == len(tar_entries) else None,
                "scanned_sample_count": stage_sample_count,
                "tars": tar_entries,
            }
        )
        print(
            f"[stage] {stage_name} tar_count={len(tar_entries)} scanned_tar_count={scanned_tar_count} "
            f"sample_count={stage_sample_count if scanned_tar_count == len(tar_entries) else 'unknown'}"
        )
        if args.scan_mode in {"full", "sampled"}:
            write_progress(progress_path, dataset_root, args, completed_results)
            print(f"[checkpoint] stage_complete={stage_name} completed_tars={len(completed_results)}", flush=True)

    manifest = {
        "schema_version": 1,
        "dataset_root": str(dataset_root),
        "public_root_mutation": "forbidden",
        "schedule_policy": "stage_order_fixed_tar_order_shuffled_per_stage_sample_buffered_per_tar",
        "seed": args.seed,
        "sample_shuffle_buffer": args.sample_shuffle_buffer,
        "scan_mode": args.scan_mode,
        "stages": stages,
    }
    total_samples = (
        sum(int(stage["sample_count"]) for stage in stages)
        if all(stage["sample_count"] is not None for stage in stages)
        else None
    )
    total_tars = sum(int(stage["tar_count"]) for stage in stages)
    print(
        f"[summary] tar_count={total_tars} sample_count={total_samples if total_samples is not None else 'unknown'} "
        f"all_pairs_valid={all_valid if all_valid is not None else 'not_scanned'}"
    )

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[manifest] wrote_private_manifest={manifest_path}")

    stats_path = manifest_path.with_suffix(".stats.json")
    safe_manifest_path(stats_path, dataset_root)
    stats = {
        "schema_version": 1,
        "manifest_path": str(manifest_path),
        "dataset_root": str(dataset_root),
        "public_root_mutation": "forbidden",
        "scan_mode": args.scan_mode,
        "seed": args.seed,
        "sample_shuffle_buffer": args.sample_shuffle_buffer,
        "stage_order": [stage["name"] for stage in stages],
        "stage_tar_counts": {stage["name"]: stage["tar_count"] for stage in stages},
        "stage_sample_counts": {stage["name"]: stage["sample_count"] for stage in stages},
        "tar_count": total_tars,
        "sample_count": total_samples,
        "all_pairs_valid": all_valid,
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[stats] wrote_private_stats={stats_path}")
    write_progress(progress_path, dataset_root, args, completed_results, status="complete")
    print(f"[progress] completed_checkpoint={progress_path}")

    wds_ok = True if args.skip_wds_probe else inspect_webdataset(manifest, args.sample_shuffle_buffer, args.wds_probe_samples)
    swift_ok = True if args.skip_swift_probe else inspect_swift_iterable_support()
    header("PREFLIGHT RESULT")
    print(f"[result] public_root_changed=false")
    print(f"[result] manifest_written=true path={manifest_path}")
    print(f"[result] stats_written=true path={stats_path}")
    pair_status = (
        "PASS" if args.scan_mode == "full" and all_valid
        else "FAIL" if all_valid is False
        else "PARTIAL" if args.scan_mode == "sampled"
        else "NOT_SCANNED"
    )
    print(f"[result] tar_pair_validation={pair_status}")
    print(f"[result] webdataset_probe={'PASS' if wds_ok else 'FAIL'}")
    print(f"[result] swift_iterable_probe={'PASS' if swift_ok else 'FAIL'}")
    return 0 if all_valid is not False and wds_ok and swift_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
