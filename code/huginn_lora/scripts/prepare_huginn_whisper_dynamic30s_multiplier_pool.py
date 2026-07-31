#!/usr/bin/env python3
"""Build the finite globally shuffled multiplier pool without decoding audio."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from array import array
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[3]
HUGINN_LORA_ROOT = REPO_ROOT / "code" / "huginn_lora"
if str(HUGINN_LORA_ROOT) not in sys.path:
    sys.path.insert(0, str(HUGINN_LORA_ROOT))

from data_pipeline.dynamic90s_mixture_rows import load_pool_registry  # noqa: E402
from data_pipeline.finite_multiplier_pool import (  # noqa: E402
    COMPONENT_ORDER,
    CONTRACT_VERSION,
    EXPECTED_MULTIPLIERS,
    EXPECTED_POOL_NAMES,
    EXPECTED_TASKS,
    GLOBAL_BATCH_SIZE,
    INDEX_FORMAT,
    POOL_ORDER,
    SAMPLER_VERSION,
)


DEFAULT_SOURCE_REGISTRY = (
    REPO_ROOT
    / "data/audio_swift/huginn_whisper_dynamic90s_multitask/v2_dynamic30s/pool_registry.json"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "data/audio_swift/huginn_whisper_dynamic30s_multiplier/v1_gigaspeech_m"
)
DEFAULT_SEED = 20260730
NOMINAL_SOURCE_HOURS = {
    "gigaspeech_m_asr": 1000.0,
    "audiocaps_v2_aac": 136.0,
    "clotho_v2_aac": 24.0,
    "wavcaps_audioset_aac": 300.0,
    "wavcaps_soundbible_aac": 4.5,
    "wavcaps_freesound_quarter_aac": 546.0,
}
SOURCE_POOL_FOR_COMPONENT = {
    "gigaspeech_m_asr": "gigaspeech_l_asr",
    "audiocaps_v2_aac": "audiocaps_v2_aac",
    "clotho_v2_aac": "clotho_v2_aac",
    "wavcaps_audioset_aac": "wavcaps_no_bbc_aac",
    "wavcaps_soundbible_aac": "wavcaps_no_bbc_aac",
    "wavcaps_freesound_quarter_aac": "wavcaps_no_bbc_aac",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-registry", type=Path, default=DEFAULT_SOURCE_REGISTRY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--progress-every", type=int, default=250000)
    parser.add_argument("--min-free-gib", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path, expected_count: int, progress_every: int) -> Iterator[tuple[int, dict[str, Any]]]:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                raise ValueError(f"Atomic manifest contains an empty line: path={path} index={index}")
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError(f"Atomic manifest record is not an object: path={path} index={index}")
            count += 1
            if count % progress_every == 0:
                print(f"[scan] manifest={path.name} records={count}", flush=True)
            yield index, record
    if count != expected_count:
        raise ValueError(
            f"Atomic manifest count mismatch: path={path} actual={count} expected={expected_count}"
        )


def positive_duration(record: dict[str, Any]) -> float | None:
    value = record.get("raw_duration_sec")
    if value is None or isinstance(value, bool):
        return None
    duration = float(value)
    return duration if duration > 0 else None


def aligned_quarter_count(source_count: int, fixed_expanded_count: int) -> tuple[int, int]:
    target = int(round(source_count / 4.0))
    required_modulus = (-fixed_expanded_count) % GLOBAL_BATCH_SIZE
    candidates = [
        value
        for value in range(required_modulus, source_count + 1, GLOBAL_BATCH_SIZE)
        if value > 0
    ]
    if not candidates:
        raise ValueError(
            f"Unable to align FreeSound quarter: source_count={source_count} fixed={fixed_expanded_count}"
        )
    selected = min(candidates, key=lambda value: (abs(value - target), value))
    if abs(selected - target) > GLOBAL_BATCH_SIZE // 2:
        raise AssertionError(f"FreeSound alignment moved too far: target={target} selected={selected}")
    return target, selected


def write_u64_atomic(path: Path, values: array) -> dict[str, Any]:
    if values.typecode != "Q" or values.itemsize != 8 or sys.byteorder != "little":
        raise RuntimeError("Multiplier index writer requires native little-endian uint64 arrays")
    temporary = path.with_name(f"{path.name}.tmp")
    digest = hashlib.sha256()
    view = memoryview(values).cast("B")
    with temporary.open("wb") as handle:
        for offset in range(0, len(view), 8 * 1024 * 1024):
            chunk = view[offset : offset + 8 * 1024 * 1024]
            handle.write(chunk)
            digest.update(chunk)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {
        "path": str(path.resolve()),
        "count": len(values),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "format": INDEX_FORMAT,
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def source_identity(entry: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(entry["manifest_path"]).resolve()
    index_path = Path(entry["index_path"]).resolve()
    identity = {
        "manifest_path": str(manifest_path),
        "manifest_size_bytes": manifest_path.stat().st_size,
        "index_path": str(index_path),
        "index_size_bytes": index_path.stat().st_size,
        "record_count": int(entry["record_count"]),
    }
    stats_path_value = entry.get("stats_path")
    if stats_path_value:
        stats_path = Path(stats_path_value).resolve()
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        identity.update(
            {
                "stats_path": str(stats_path),
                "manifest_sha256": stats.get("manifest_sha256"),
                "index_sha256": stats.get("index_sha256"),
            }
        )
    return identity


def main() -> None:
    args = parse_args()
    if args.seed < 0 or args.progress_every <= 0 or args.min_free_gib <= 0:
        raise ValueError("seed must be non-negative and progress/free-space values must be positive")
    source_registry_path = args.source_registry.expanduser().resolve()
    source_registry = load_pool_registry(source_registry_path)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(output_root).free / 1024**3
    if free_gib < args.min_free_gib:
        raise OSError(f"Insufficient free space: free_gib={free_gib:.3f} required={args.min_free_gib:.3f}")

    selections_dir = output_root / "selections"
    selections_dir.mkdir(parents=True, exist_ok=True)
    registry_path = output_root / "multiplier_pool_registry.json"
    report_path = output_root / "multiplier_pool_report.json"
    schedule_path = output_root / "global_schedule.idx"
    protected = [registry_path, report_path, schedule_path]
    if not args.overwrite and any(path.exists() for path in protected):
        raise FileExistsError(f"Refusing to overwrite multiplier artifacts: {[str(p) for p in protected if p.exists()]}")

    print("========== HUGINN WHISPER DYNAMIC30S MULTIPLIER POOL START ==========", flush=True)
    print("[scope] metadata_only=true audio_decode=false audio_copy=false", flush=True)
    print(f"[source] registry={source_registry_path}", flush=True)

    wav_entry = source_registry["pools"]["wavcaps_no_bbc_aac"]
    wav_manifest = Path(wav_entry["manifest_path"])
    wav_selections = {
        "AudioSet_SL": array("Q"),
        "SoundBible": array("Q"),
        "FreeSound": array("Q"),
    }
    wav_source_counts: Counter[str] = Counter()
    for record_index, record in iter_jsonl(
        wav_manifest,
        int(wav_entry["record_count"]),
        args.progress_every,
    ):
        source = str(record.get("source", ""))
        wav_source_counts[source] += 1
        if source not in wav_selections:
            raise ValueError(f"Unexpected WavCaps source in no-BBC pool: {source!r}")
        wav_selections[source].append(record_index)
    if set(wav_source_counts) != set(wav_selections):
        raise ValueError(f"WavCaps source coverage mismatch: {wav_source_counts}")

    giga_entry = source_registry["pools"]["gigaspeech_l_asr"]
    giga_manifest = Path(giga_entry["manifest_path"])
    giga_m_indices = array("Q")
    giga_m_raw_seconds = 0.0
    giga_m_effective_seconds = 0.0
    for record_index, record in iter_jsonl(
        giga_manifest,
        int(giga_entry["record_count"]),
        args.progress_every,
    ):
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        subsets = metadata.get("subsets") or []
        if "{M}" not in subsets:
            continue
        if record.get("task") != "ASR":
            raise ValueError(f"GigaSpeech-M record is not ASR: index={record_index}")
        duration = positive_duration(record)
        if duration is None:
            raise ValueError(f"GigaSpeech-M record lacks duration: index={record_index}")
        giga_m_indices.append(record_index)
        giga_m_raw_seconds += duration
        giga_m_effective_seconds += min(duration, 30.0)
    giga_m_raw_hours = giga_m_raw_seconds / 3600.0
    if not 850.0 <= giga_m_raw_hours <= 1150.0:
        raise ValueError(
            f"GigaSpeech {{M}} duration is inconsistent with the 1000h subset: {giga_m_raw_hours:.3f}h"
        )

    audiocaps_count = int(source_registry["pools"]["audiocaps_v2_aac"]["record_count"])
    clotho_count = int(source_registry["pools"]["clotho_v2_aac"]["record_count"])
    audioset_count = len(wav_selections["AudioSet_SL"])
    soundbible_count = len(wav_selections["SoundBible"])
    freesound_count = len(wav_selections["FreeSound"])
    fixed_expanded_count = (
        len(giga_m_indices)
        + 3 * audiocaps_count
        + 3 * clotho_count
        + 2 * audioset_count
        + 2 * soundbible_count
    )
    freesound_target_count, freesound_selected_count = aligned_quarter_count(
        freesound_count,
        fixed_expanded_count,
    )
    freesound_order = array("Q", wav_selections["FreeSound"])
    random.Random(args.seed ^ 0xF4EE50A1).shuffle(freesound_order)
    freesound_selected = array("Q", freesound_order[:freesound_selected_count])
    if len(set(freesound_selected)) != freesound_selected_count:
        raise AssertionError("FreeSound quarter selection contains duplicate base indices")

    selection_values = {
        "gigaspeech_m_asr": giga_m_indices,
        "wavcaps_audioset_aac": wav_selections["AudioSet_SL"],
        "wavcaps_soundbible_aac": wav_selections["SoundBible"],
        "wavcaps_freesound_quarter_aac": freesound_selected,
    }
    selection_artifacts: dict[str, dict[str, Any]] = {}
    for component_name, values in selection_values.items():
        selection_artifacts[component_name] = write_u64_atomic(
            selections_dir / f"{component_name}.idx",
            values,
        )

    selected_counts = {
        "gigaspeech_m_asr": len(giga_m_indices),
        "audiocaps_v2_aac": audiocaps_count,
        "clotho_v2_aac": clotho_count,
        "wavcaps_audioset_aac": audioset_count,
        "wavcaps_soundbible_aac": soundbible_count,
        "wavcaps_freesound_quarter_aac": freesound_selected_count,
    }
    source_entries = {
        name: source_registry["pools"][SOURCE_POOL_FOR_COMPONENT[name]]
        for name in COMPONENT_ORDER
    }
    components: dict[str, dict[str, Any]] = {}
    virtual_start = 0
    aggregate_offsets = {name: 0 for name in POOL_ORDER}
    for name in COMPONENT_ORDER:
        source_entry = source_entries[name]
        selected_count = selected_counts[name]
        multiplier = EXPECTED_MULTIPLIERS[name]
        expanded_count = selected_count * multiplier
        pool_name = EXPECTED_POOL_NAMES[name]
        components[name] = {
            "task": EXPECTED_TASKS[name],
            "pool_name": pool_name,
            "multiplier": multiplier,
            "base_record_count": int(source_entry["record_count"]),
            "selected_record_count": selected_count,
            "expanded_record_count": expanded_count,
            "virtual_start": virtual_start,
            "virtual_end": virtual_start + expanded_count,
            "aggregate_pool_offset": aggregate_offsets[pool_name],
            "manifest_path": str(Path(source_entry["manifest_path"]).resolve()),
            "index_path": str(Path(source_entry["index_path"]).resolve()),
            "selection_index_path": (
                selection_artifacts[name]["path"] if name in selection_artifacts else None
            ),
            "nominal_source_hours": NOMINAL_SOURCE_HOURS[name],
            "nominal_expanded_hours": NOMINAL_SOURCE_HOURS[name] * multiplier,
        }
        virtual_start += expanded_count
        aggregate_offsets[pool_name] += expanded_count

    total_records = virtual_start
    if total_records != fixed_expanded_count + freesound_selected_count or total_records % GLOBAL_BATCH_SIZE:
        raise AssertionError(
            f"Global-batch alignment failed: total={total_records} fixed={fixed_expanded_count} "
            f"freesound={freesound_selected_count}"
        )
    schedule = array("Q", range(total_records))
    random.Random(args.seed ^ 0x61B4A7E5).shuffle(schedule)
    schedule_artifact = write_u64_atomic(schedule_path, schedule)
    del schedule

    pools = {
        name: {
            "record_count": aggregate_offsets[name],
            "task": "ASR" if name == "gigaspeech_l_asr" else "AAC",
        }
        for name in POOL_ORDER
    }
    nominal_expanded_hours = {
        name: NOMINAL_SOURCE_HOURS[name] * EXPECTED_MULTIPLIERS[name]
        for name in COMPONENT_ORDER
    }
    registry = {
        "contract_version": CONTRACT_VERSION,
        "sampler_version": SAMPLER_VERSION,
        "statistics_version": "huginn_dynamic30s_multiplier_training_statistics_v1",
        "duration_policy": "retain_all_then_cap_at30s",
        "index_format": INDEX_FORMAT,
        "seed": args.seed,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "total_records": total_records,
        "max_steps": total_records // GLOBAL_BATCH_SIZE,
        "schedule_path": schedule_artifact["path"],
        "schedule_sha256": schedule_artifact["sha256"],
        "source_registry_path": str(source_registry_path),
        "source_registry_sha256": sha256_file(source_registry_path),
        "components": components,
        "pools": pools,
    }
    report = {
        "gate": "huginn_whisper_dynamic30s_multiplier_pool_preparation_v1",
        "validation_passed": True,
        "metadata_only": True,
        "audio_decode": False,
        "audio_copy": False,
        "contract_version": CONTRACT_VERSION,
        "sampler_version": SAMPLER_VERSION,
        "seed": args.seed,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "total_records": total_records,
        "max_steps": total_records // GLOBAL_BATCH_SIZE,
        "nominal_expanded_hours": nominal_expanded_hours,
        "nominal_total_expanded_hours": sum(nominal_expanded_hours.values()),
        "gigaspeech_m": {
            "record_count": len(giga_m_indices),
            "raw_metadata_hours": giga_m_raw_hours,
            "effective_hours_after_30s_cap": giga_m_effective_seconds / 3600.0,
        },
        "freesound_quarter": {
            "source_record_count": freesound_count,
            "unadjusted_quarter_record_count": freesound_target_count,
            "selected_record_count": freesound_selected_count,
            "selected_fraction": freesound_selected_count / freesound_count,
            "batch_alignment_adjustment_records": freesound_selected_count - freesound_target_count,
        },
        "wavcaps_source_counts": dict(wav_source_counts),
        "selection_artifacts": selection_artifacts,
        "schedule_artifact": schedule_artifact,
        "source_identities": {
            pool_name: source_identity(source_registry["pools"][pool_name])
            for pool_name in POOL_ORDER
        },
        "registry_path": str(registry_path),
    }
    write_json_atomic(registry_path, registry)
    write_json_atomic(report_path, report)
    print(
        f"[multiplier] total_records={total_records} max_steps={total_records // GLOBAL_BATCH_SIZE} "
        f"nominal_hours={sum(nominal_expanded_hours.values()):.3f}",
        flush=True,
    )
    print(
        f"[freesound] source={freesound_count} target={freesound_target_count} "
        f"selected={freesound_selected_count} adjustment={freesound_selected_count - freesound_target_count}",
        flush=True,
    )
    print(f"[multiplier] registry={registry_path}", flush=True)
    print(f"[multiplier] report={report_path}", flush=True)
    print("========== HUGINN WHISPER DYNAMIC30S MULTIPLIER POOL PREPARED ==========", flush=True)


if __name__ == "__main__":
    main()
