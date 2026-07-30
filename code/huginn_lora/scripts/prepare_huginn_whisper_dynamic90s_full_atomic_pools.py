"""Stream Huginn audio metadata pools to atomic JSONL.

Every eligible dataset record is retained. Source durations are copied only
when already present in metadata; missing WavCaps durations do not trigger
audio reads. Runtime decoding retains at most the first 30 seconds. Audio files
are never decoded, copied, converted, or exhaustively stat-checked here. Token
accounting is deferred to training-time statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

from inspect_huginn_whisper_dynamic90s_data_pools import iter_named_json_array, load_json_or_jsonl
from prepare_huginn_whisper_dynamic90s_atomic_pilot import (
    SCHEMA_VERSION,
    audio_format,
    clean_gigaspeech_text,
    extract_targets,
    extract_wavcaps_id,
    first_assistant_text,
    iter_jsonl,
    iter_wavcaps_metadata,
    load_contract,
    normalize_text,
    validate_atomic_record,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONTRACT = REPO_ROOT / "code/huginn_lora/configs/huginn_whisper_dynamic90s_data_contract_v1.json"
DEFAULT_INVENTORY = (
    REPO_ROOT / "data/audio_swift/huginn_whisper_dynamic90s_multitask/v1/audits/data_pool_inventory.json"
)
DEFAULT_PILOT_REPORT = (
    REPO_ROOT / "data/audio_swift/huginn_whisper_dynamic90s_multitask/v1/pilot/atomic_pilot_report.json"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/audio_swift/huginn_whisper_dynamic90s_multitask/v2_dynamic30s"
DEFAULT_AUDIOCAPS_MANIFEST = REPO_ROOT / "data/audio_swift/audiocaps_v2/audiocaps_v2_train_swift.jsonl"
DEFAULT_CLOTHO_ROOT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_caption_huginn")
DEFAULT_GIGASPEECH_ROOT = Path("/hpc_stor03/public/shared/data/asr/am/GigaSpeech")
POOL_ORDER = (
    "wavcaps_no_bbc_aac",
    "audiocaps_v2_aac",
    "clotho_v2_aac",
    "gigaspeech_l_asr",
)
POOL_WEIGHTS = {
    "wavcaps_no_bbc_aac": 0.36,
    "audiocaps_v2_aac": 0.18,
    "clotho_v2_aac": 0.06,
    "gigaspeech_l_asr": 0.40,
}
MAX_RETAINED_DURATION_SECONDS = 30.0
FALLBACK_EFFECTIVE_POOL_HOURS = {
    "audiocaps_v2_aac": 136.0,
    "clotho_v2_aac": 24.0,
}


def metadata_duration_seconds(record: dict[str, Any]) -> float | None:
    """Read a positive duration already present in source metadata."""
    candidates = [record]
    for key in ("metadata", "meta"):
        nested = record.get(key)
        if isinstance(nested, dict):
            candidates.append(nested)
    audio_metadata = record.get("audio")
    if isinstance(audio_metadata, dict):
        candidates.append(audio_metadata)
    for candidate in candidates:
        for key in ("duration", "duration_sec", "duration_secs", "duration_seconds", "length_seconds"):
            value = candidate.get(key)
            if value is None or isinstance(value, bool):
                continue
            try:
                if isinstance(value, str) and ":" in value:
                    parts = [float(part) for part in value.strip().split(":")]
                    if len(parts) == 3:
                        duration = parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
                    elif len(parts) == 2:
                        duration = parts[0] * 60.0 + parts[1]
                    else:
                        continue
                else:
                    duration = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(duration) and duration > 0:
                return duration
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--inventory_report", default=str(DEFAULT_INVENTORY))
    parser.add_argument("--pilot_report", default=str(DEFAULT_PILOT_REPORT))
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--audiocaps_manifest", default=str(DEFAULT_AUDIOCAPS_MANIFEST))
    parser.add_argument("--clotho_root", default=str(DEFAULT_CLOTHO_ROOT))
    parser.add_argument("--clotho_train_manifest", default="train_expand.json")
    parser.add_argument("--gigaspeech_root", default=str(DEFAULT_GIGASPEECH_ROOT))
    parser.add_argument("--gigaspeech_metadata", default="GigaSpeech.json")
    parser.add_argument("--progress_every", type=int, default=100000)
    parser.add_argument("--min_free_gib", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_passed_json(path: Path, passed_field: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required gate report is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get(passed_field):
        raise ValueError(f"Required gate has not passed: path={path} field={passed_field}")
    return payload


class AtomicPoolWriter:
    """Write JSONL and little-endian uint64 line offsets without early commit."""

    def __init__(self, pool_name: str, pools_dir: Path, overwrite: bool) -> None:
        self.pool_name = pool_name
        self.manifest_path = pools_dir / f"{pool_name}.jsonl"
        self.index_path = pools_dir / f"{pool_name}.idx"
        self.stats_path = pools_dir / f"{pool_name}.stats.json"
        self.manifest_tmp = self.manifest_path.with_name(f"{self.manifest_path.name}.tmp")
        self.index_tmp = self.index_path.with_name(f"{self.index_path.name}.tmp")
        self.stats_tmp = self.stats_path.with_name(f"{self.stats_path.name}.tmp")
        if not overwrite:
            existing = [path for path in (self.manifest_path, self.index_path, self.stats_path) if path.exists()]
            if existing:
                raise FileExistsError(f"Refusing to overwrite completed pool files: {existing}")
        self.manifest_handle = self.manifest_tmp.open("wb")
        self.index_handle = self.index_tmp.open("wb")
        self.manifest_hash = hashlib.sha256()
        self.index_hash = hashlib.sha256()
        self.offset = 0
        self.record_count = 0
        self.source_counts: Counter[str] = Counter()
        self.target_count_histogram: Counter[int] = Counter()
        self.raw_duration_seconds = 0.0
        self.effective_duration_seconds = 0.0
        self.duration_record_count = 0

    def write(self, record: dict[str, Any]) -> None:
        validate_atomic_record(record)
        line = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        offset_bytes = struct.pack("<Q", self.offset)
        self.index_handle.write(offset_bytes)
        self.manifest_handle.write(line)
        self.index_hash.update(offset_bytes)
        self.manifest_hash.update(line)
        self.offset += len(line)
        self.record_count += 1
        self.source_counts[record["source"]] += 1
        self.target_count_histogram[len(record["targets"])] += 1
        if "raw_duration_sec" in record:
            raw_duration = float(record["raw_duration_sec"])
            self.raw_duration_seconds += raw_duration
            self.effective_duration_seconds += min(raw_duration, MAX_RETAINED_DURATION_SECONDS)
            self.duration_record_count += 1

    def finish_temporary(self, extra_stats: dict[str, Any]) -> dict[str, Any]:
        self.manifest_handle.flush()
        self.index_handle.flush()
        os.fsync(self.manifest_handle.fileno())
        os.fsync(self.index_handle.fileno())
        self.manifest_handle.close()
        self.index_handle.close()
        if self.record_count == 0:
            raise ValueError(f"Pool {self.pool_name} emitted zero records")
        stats = {
            "pool": self.pool_name,
            "schema_version": SCHEMA_VERSION,
            "manifest_path": str(self.manifest_path),
            "index_path": str(self.index_path),
            "index_format": "little-endian uint64 JSONL byte offsets without header",
            "record_count": self.record_count,
            "manifest_size_bytes": self.offset,
            "index_size_bytes": self.record_count * 8,
            "manifest_sha256": self.manifest_hash.hexdigest(),
            "index_sha256": self.index_hash.hexdigest(),
            "source_counts": dict(sorted(self.source_counts.items())),
            "target_count_histogram": {
                str(key): value for key, value in sorted(self.target_count_histogram.items())
            },
            "raw_duration_hours_from_metadata": self.raw_duration_seconds / 3600.0,
            "effective_duration_hours_after_30s_cap": self.effective_duration_seconds / 3600.0,
            "duration_metadata_record_count": self.duration_record_count,
            "duration_metadata_complete": self.duration_record_count == self.record_count,
            "effective_audio_tokens_present": False,
            "audio_decode_performed": False,
            "audio_copy_performed": False,
            "full_audio_path_scan_performed": False,
            **extra_stats,
        }
        with self.stats_tmp.open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return stats

    def commit(self) -> None:
        os.replace(self.manifest_tmp, self.manifest_path)
        os.replace(self.index_tmp, self.index_path)
        os.replace(self.stats_tmp, self.stats_path)


def audiocaps_records(manifest: Path, counters: Counter[str]) -> Iterator[dict[str, Any]]:
    for source_record in iter_jsonl(manifest):
        counters["source_records"] += 1
        audios = source_record.get("audios")
        if not isinstance(audios, list) or len(audios) != 1 or not isinstance(audios[0], str):
            raise ValueError(f"Unexpected AudioCaps audio field: {audios!r}")
        path = Path(audios[0])
        caption = first_assistant_text(source_record.get("messages"))
        if not caption:
            raise ValueError("Verified AudioCaps record has no assistant caption")
        metadata = source_record.get("metadata") if isinstance(source_record.get("metadata"), dict) else {}
        sample_id = normalize_text(metadata.get("sample_id")) or path.stem
        record = {
            "schema_version": SCHEMA_VERSION,
            "uid": f"audiocaps_v2:{sample_id}",
            "dataset": "AudioCaps-v2",
            "source": "AudioCaps-v2",
            "task": "AAC",
            "split": "train",
            "audio": {"path": str(path), "format": audio_format(path)},
            "targets": [caption],
            "metadata": {
                "sample_id": sample_id,
                "youtube_id": normalize_text(metadata.get("youtube_id")),
                "audiocap_id": normalize_text(metadata.get("audiocap_id")),
            },
        }
        duration = metadata_duration_seconds(source_record)
        if duration is not None:
            record["raw_duration_sec"] = duration
            counters["duration_metadata_records"] += 1
        else:
            counters["duration_missing_records"] += 1
        counters["emitted_records"] += 1
        yield record


def wavcaps_records(pilot: dict[str, Any], counters: Counter[str]) -> Iterator[dict[str, Any]]:
    mapping_root = pilot["pools"]["wavcaps_no_bbc_aac"]["mapping"]
    if mapping_root.get("bbc_records") != 0:
        raise ValueError("Passed pilot does not prove zero BBC records")
    mapping = mapping_root["mapping"]
    expected_sources = {"FreeSound", "AudioSet_SL", "SoundBible"}
    if set(mapping) != expected_sources:
        raise ValueError(f"Unexpected WavCaps source mapping: {sorted(mapping)}")
    for source in ("FreeSound", "AudioSet_SL", "SoundBible"):
        source_mapping = mapping[source]
        audio_dir = Path(source_mapping["audio_dir"])
        metadata_path = Path(source_mapping["metadata_path"])
        suffix = str(source_mapping["audio_suffix"])
        for metadata_record in iter_wavcaps_metadata(metadata_path):
            counters["source_records"] += 1
            sample_id = extract_wavcaps_id(metadata_record, source)
            targets = extract_targets(metadata_record)
            path = audio_dir / f"{sample_id}{suffix}"
            duration = metadata_duration_seconds(metadata_record)
            if duration is None and source == "AudioSet_SL":
                # AudioSet clips are fixed 10-second excerpts; some WavCaps
                # metadata releases omit an explicit duration field.
                duration = 10.0
                counters["duration_assumed_audioset_10s_records"] += 1
            record = {
                "schema_version": SCHEMA_VERSION,
                "uid": f"wavcaps:{source}:{sample_id}",
                "dataset": "WavCaps",
                "source": source,
                "task": "AAC",
                "split": "train",
                "audio": {"path": str(path), "format": suffix.lstrip(".")},
                "targets": targets,
                "metadata": {"sample_id": sample_id, "metadata_path": str(metadata_path)},
            }
            if duration is not None:
                record["raw_duration_sec"] = duration
                counters["duration_available_records"] += 1
            else:
                counters["duration_missing_records"] += 1
            counters["emitted_records"] += 1
            yield record


def clotho_records(root: Path, manifest_name: str) -> Iterator[dict[str, Any]]:
    manifest = root / manifest_name
    records = load_json_or_jsonl(manifest)
    grouped: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Clotho manifest contains a non-object record")
        raw_path = record.get("audio_path") or record.get("audio")
        caption = normalize_text(record.get("caption", record.get("response")))
        if not isinstance(raw_path, str) or not raw_path.strip() or not caption:
            raise ValueError(f"Invalid Clotho record: {record}")
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        grouped[str(path)].add(caption)
    for rendered_path in sorted(grouped):
        path = Path(rendered_path)
        yield {
            "schema_version": SCHEMA_VERSION,
            "uid": f"clotho_v2:{path.stem}",
            "dataset": "Clotho-v2",
            "source": "Clotho-v2",
            "task": "AAC",
            "split": "train",
            "audio": {"path": str(path), "format": audio_format(path)},
            "targets": sorted(grouped[rendered_path]),
            "metadata": {
                "sample_id": path.stem,
                "target_selection": "one caption per scheduled training occurrence",
            },
        }


def gigaspeech_records(root: Path, metadata_name: str, counters: Counter[str]) -> Iterator[dict[str, Any]]:
    metadata_path = root / metadata_name
    for audio in iter_named_json_array(metadata_path, "audios"):
        counters["audio_objects"] += 1
        if not isinstance(audio, dict):
            raise ValueError("GigaSpeech audios array contains a non-object")
        raw_path = normalize_text(audio.get("path"))
        source = normalize_text(audio.get("source")).lower()
        segments = audio.get("segments")
        if not raw_path or not isinstance(segments, list):
            raise ValueError(f"Invalid GigaSpeech audio metadata: path={raw_path!r}")
        path = root / raw_path
        for segment in segments:
            counters["segments"] += 1
            if not isinstance(segment, dict) or "{L}" not in (segment.get("subsets") or []):
                continue
            counters["l_segments"] += 1
            sid = normalize_text(segment.get("sid"))
            raw_text = normalize_text(segment.get("text_tn"))
            cleaned_text = clean_gigaspeech_text(raw_text)
            if not sid or not cleaned_text:
                raise ValueError(f"Invalid GigaSpeech-L text: sid={sid!r} raw={raw_text!r}")
            begin = float(segment["begin_time"])
            end = float(segment["end_time"])
            if begin < 0 or end <= begin:
                raise ValueError(f"Invalid GigaSpeech-L bounds: sid={sid} begin={begin} end={end}")
            duration = end - begin
            counters["emitted_l_segments"] += 1
            yield {
                "schema_version": SCHEMA_VERSION,
                "uid": f"gigaspeech_l:{sid}",
                "dataset": "GigaSpeech",
                "source": source,
                "task": "ASR",
                "split": "L",
                "audio": {
                    "path": str(path),
                    "format": audio_format(path),
                    "start_sec": begin,
                    "end_sec": end,
                },
                "raw_duration_sec": duration,
                "targets": [cleaned_text],
                "metadata": {"sid": sid, "text_tn_raw": raw_text, "subsets": segment.get("subsets")},
            }


def stream_pool(
    writer: AtomicPoolWriter,
    records: Iterator[dict[str, Any]],
    progress_every: int,
    check_uids: bool,
) -> None:
    seen_uids: set[str] | None = set() if check_uids else None
    for record in records:
        if seen_uids is not None:
            uid = record["uid"]
            if uid in seen_uids:
                raise ValueError(f"Duplicate UID in {writer.pool_name}: {uid}")
            seen_uids.add(uid)
        writer.write(record)
        if writer.record_count % progress_every == 0:
            print(
                f"[full-pool] pool={writer.pool_name} records={writer.record_count} "
                f"size_gib={writer.offset / 1024**3:.3f}",
                flush=True,
            )


def write_json_temporary(path: Path, payload: dict[str, Any]) -> Path:
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return temporary


def main() -> None:
    args = parse_args()
    if args.progress_every <= 0 or args.min_free_gib <= 0:
        raise ValueError("progress_every and min_free_gib must be positive")
    contract = load_contract(Path(args.contract))
    inventory = load_passed_json(Path(args.inventory_report), "inspection_passed")
    pilot = load_passed_json(Path(args.pilot_report), "validation_passed")
    if inventory.get("blocking_issues"):
        raise ValueError(f"Inventory has blocking issues: {inventory['blocking_issues']}")
    if pilot.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Pilot schema mismatch: {pilot.get('schema_version')!r}")

    output_root = Path(args.output_root)
    pools_dir = output_root / "pools"
    pools_dir.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(output_root).free / 1024**3
    if free_gib < args.min_free_gib:
        raise OSError(
            f"Insufficient free space for full manifests: free_gib={free_gib:.3f} required={args.min_free_gib:.3f}"
        )
    registry_path = output_root / "pool_registry.json"
    report_path = output_root / "full_pool_report.json"
    if not args.overwrite:
        existing_top = [path for path in (registry_path, report_path) if path.exists()]
        if existing_top:
            raise FileExistsError(f"Refusing to overwrite completed full-pool files: {existing_top}")

    print("========== HUGINN WHISPER DYNAMIC30S FULL ATOMIC POOLS START ==========", flush=True)
    print("[scope] audio_decode=false audio_copy=false full_audio_scan=false token_accounting=false", flush=True)
    print(f"[preflight] free_gib={free_gib:.3f} required_free_gib={args.min_free_gib:.3f}", flush=True)
    writers = {name: AtomicPoolWriter(name, pools_dir, args.overwrite) for name in POOL_ORDER}
    pool_stats: dict[str, dict[str, Any]] = {}

    wavcaps_counters: Counter[str] = Counter()
    stream_pool(
        writers["wavcaps_no_bbc_aac"],
        wavcaps_records(pilot, wavcaps_counters),
        args.progress_every,
        check_uids=True,
    )
    pool_stats["wavcaps_no_bbc_aac"] = writers["wavcaps_no_bbc_aac"].finish_temporary(
        {
            "excluded_sources": ["BBC_Sound_Effects"],
            "bbc_record_count": 0,
            "duration_accounting": {
                "retain_all_records": True,
                "retained_cap_seconds": MAX_RETAINED_DURATION_SECONDS,
                **dict(wavcaps_counters),
            },
        }
    )
    if wavcaps_counters["source_records"] != wavcaps_counters["emitted_records"]:
        raise ValueError(
            "WavCaps metadata streaming did not emit every non-BBC record: "
            f"counters={dict(wavcaps_counters)} stats={pool_stats['wavcaps_no_bbc_aac']}"
        )
    print(f"[full-pool] pool=wavcaps_no_bbc_aac records={pool_stats['wavcaps_no_bbc_aac']['record_count']} ready=true", flush=True)
    print(
        "[duration] pool=wavcaps_no_bbc_aac "
        f"metadata_available={pool_stats['wavcaps_no_bbc_aac']['duration_metadata_record_count']} "
        f"metadata_missing={wavcaps_counters['duration_missing_records']} "
        "planning_hours=deferred",
        flush=True,
    )

    audiocaps_counters: Counter[str] = Counter()
    stream_pool(
        writers["audiocaps_v2_aac"],
        audiocaps_records(Path(args.audiocaps_manifest), audiocaps_counters),
        args.progress_every,
        check_uids=True,
    )
    pool_stats["audiocaps_v2_aac"] = writers["audiocaps_v2_aac"].finish_temporary(
        {
            "source_manifest": str(Path(args.audiocaps_manifest)),
            "duration_accounting": {
                "retain_all_records": True,
                "retained_cap_seconds": MAX_RETAINED_DURATION_SECONDS,
                **dict(audiocaps_counters),
            },
        }
    )
    print(f"[full-pool] pool=audiocaps_v2_aac records={pool_stats['audiocaps_v2_aac']['record_count']} ready=true", flush=True)

    stream_pool(
        writers["clotho_v2_aac"],
        clotho_records(Path(args.clotho_root), args.clotho_train_manifest),
        args.progress_every,
        check_uids=True,
    )
    clotho_expected = int(inventory["pools"]["clotho_v2_aac"]["grouped_audio_count"])
    if writers["clotho_v2_aac"].record_count != clotho_expected:
        raise ValueError(
            f"Clotho grouped count changed: expected={clotho_expected} actual={writers['clotho_v2_aac'].record_count}"
        )
    pool_stats["clotho_v2_aac"] = writers["clotho_v2_aac"].finish_temporary(
        {"target_selection": "one caption per scheduled training occurrence"}
    )
    print(f"[full-pool] pool=clotho_v2_aac records={pool_stats['clotho_v2_aac']['record_count']} ready=true", flush=True)

    giga_counters: Counter[str] = Counter()
    stream_pool(
        writers["gigaspeech_l_asr"],
        gigaspeech_records(Path(args.gigaspeech_root), args.gigaspeech_metadata, giga_counters),
        args.progress_every,
        check_uids=False,
    )
    giga_expected = int(inventory["pools"]["gigaspeech_l_asr"]["l_segment_count"])
    if (
        giga_counters["l_segments"] != giga_expected
        or writers["gigaspeech_l_asr"].record_count != giga_counters["emitted_l_segments"]
        or giga_counters["emitted_l_segments"] != giga_expected
    ):
        raise ValueError(
            "GigaSpeech-L count changed: "
            f"expected={giga_expected} emitted={writers['gigaspeech_l_asr'].record_count} "
            f"emitted_l_segments={giga_counters['emitted_l_segments']} "
            f"seen={giga_counters['l_segments']}"
        )
    pool_stats["gigaspeech_l_asr"] = writers["gigaspeech_l_asr"].finish_temporary(
        {"metadata_counters": dict(giga_counters), "duplicate_sid_audit": "passed by prerequisite inventory"}
    )
    print(f"[full-pool] pool=gigaspeech_l_asr records={pool_stats['gigaspeech_l_asr']['record_count']} ready=true", flush=True)
    print(
        "[duration] pool=gigaspeech_l_asr "
        "retained_all_segments=true "
        f"effective_hours_30s_cap={pool_stats['gigaspeech_l_asr']['effective_duration_hours_after_30s_cap']:.6f}",
        flush=True,
    )

    if not pool_stats["gigaspeech_l_asr"]["duration_metadata_complete"]:
        raise ValueError("GigaSpeech-L segment metadata must retain complete duration accounting")

    def planning_hours(pool_name: str) -> float | None:
        if pool_name in FALLBACK_EFFECTIVE_POOL_HOURS:
            return FALLBACK_EFFECTIVE_POOL_HOURS[pool_name]
        if pool_stats[pool_name]["duration_metadata_complete"]:
            return float(pool_stats[pool_name]["effective_duration_hours_after_30s_cap"])
        return None

    def planning_source(pool_name: str) -> str:
        if pool_name in FALLBACK_EFFECTIVE_POOL_HOURS:
            return "fixed_verified_pool_hours_all_samples_le_30s"
        if pool_stats[pool_name]["duration_metadata_complete"]:
            return "complete_source_metadata_after_30s_cap"
        return "deferred_missing_source_duration_metadata"

    registry = {
        "contract_version": contract.get("contract_version"),
        "schema_version": SCHEMA_VERSION,
        "duration_policy": "retain_all_then_cap_at30s",
        "index_format": "little-endian uint64 JSONL byte offsets without header",
        "sampling_weights": POOL_WEIGHTS,
        "per_record_token_accounting": False,
        "pools": {
            pool_name: {
                "manifest_path": pool_stats[pool_name]["manifest_path"],
                "index_path": pool_stats[pool_name]["index_path"],
                "stats_path": str(writers[pool_name].stats_path),
                "record_count": pool_stats[pool_name]["record_count"],
                "planning_effective_duration_hours": planning_hours(pool_name),
                "planning_duration_source": planning_source(pool_name),
                "global_weight": POOL_WEIGHTS[pool_name],
                "task": "ASR" if pool_name == "gigaspeech_l_asr" else "AAC",
            }
            for pool_name in POOL_ORDER
        },
    }
    report = {
        "gate": "huginn_whisper_dynamic30s_full_atomic_pools_v2",
        "validation_passed": True,
        "audio_decode": False,
        "audio_copy": False,
        "full_audio_path_scan": False,
        "token_accounting": False,
        "duration_policy": {
            "retain_all_records": True,
            "discard_above_seconds": None,
            "retain_at_most_seconds": MAX_RETAINED_DURATION_SECONDS,
            "audio_decode_for_eligibility": False,
        },
        "preflight_free_gib": free_gib,
        "required_free_gib": args.min_free_gib,
        "pool_stats": pool_stats,
    }
    registry_tmp = write_json_temporary(registry_path, registry)
    report_tmp = write_json_temporary(report_path, report)

    for writer in writers.values():
        writer.commit()
    os.replace(registry_tmp, registry_path)
    os.replace(report_tmp, report_path)
    print(f"[full-pool] registry={registry_path}", flush=True)
    print(f"[full-pool] report={report_path}", flush=True)
    print("========== HUGINN WHISPER DYNAMIC30S FULL ATOMIC POOLS PASSED ==========", flush=True)


if __name__ == "__main__":
    main()
