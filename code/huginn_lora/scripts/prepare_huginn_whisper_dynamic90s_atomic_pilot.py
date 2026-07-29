"""Prepare and validate small unified atomic-manifest pilots for four data pools.

This is a mapping gate, not full data preparation. It writes metadata-only JSONL
records into the private repository data directory and never decodes or copies audio.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

from inspect_huginn_whisper_dynamic90s_data_pools import (
    canonical_wavcaps_source,
    iter_named_json_array,
    iter_root_json_array,
    load_json_or_jsonl,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONTRACT = REPO_ROOT / "code/huginn_lora/configs/huginn_whisper_dynamic90s_data_contract_v1.json"
DEFAULT_INVENTORY = (
    REPO_ROOT
    / "data/audio_swift/huginn_whisper_dynamic90s_multitask/v1/audits/data_pool_inventory.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/audio_swift/huginn_whisper_dynamic90s_multitask/v1/pilot"
DEFAULT_AUDIOCAPS_MANIFEST = REPO_ROOT / "data/audio_swift/audiocaps_v2/audiocaps_v2_train_swift.jsonl"
DEFAULT_CLOTHO_ROOT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_caption_huginn")
DEFAULT_WAVCAPS_ROOT = Path("/hpc_stor03/public/shared/data/raa/WavCaps")
DEFAULT_GIGASPEECH_ROOT = Path("/hpc_stor03/public/shared/data/asr/am/GigaSpeech")
SCHEMA_VERSION = "huginn_whisper_dynamic90s_atomic_v1"
POOL_FILENAMES = {
    "audiocaps_v2_aac": "audiocaps_v2_aac.pilot.jsonl",
    "wavcaps_no_bbc_aac": "wavcaps_no_bbc_aac.pilot.jsonl",
    "clotho_v2_aac": "clotho_v2_aac.pilot.jsonl",
    "gigaspeech_l_asr": "gigaspeech_l_asr.pilot.jsonl",
}
PUNCTUATION_TAGS = {
    "<COMMA>": ",",
    "<PERIOD>": ".",
    "<QUESTIONMARK>": "?",
    "<EXCLAMATIONPOINT>": "!",
    "<COLON>": ":",
    "<SEMICOLON>": ";",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--inventory_report", default=str(DEFAULT_INVENTORY))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--records_per_pool", type=int, default=16)
    parser.add_argument("--audiocaps_manifest", default=str(DEFAULT_AUDIOCAPS_MANIFEST))
    parser.add_argument("--clotho_root", default=str(DEFAULT_CLOTHO_ROOT))
    parser.add_argument("--clotho_train_manifest", default="train_expand.json")
    parser.add_argument("--wavcaps_root", default=str(DEFAULT_WAVCAPS_ROOT))
    parser.add_argument("--gigaspeech_root", default=str(DEFAULT_GIGASPEECH_ROOT))
    parser.add_argument("--gigaspeech_metadata", default="GigaSpeech.json")
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def clean_gigaspeech_text(text: str) -> str:
    cleaned = normalize_text(text)
    for tag, punctuation in PUNCTUATION_TAGS.items():
        cleaned = cleaned.replace(tag, punctuation)
    cleaned = re.sub(r"<[^<>]+>", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return normalize_text(cleaned)


def audio_format(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if not suffix:
        raise ValueError(f"Audio path has no format suffix: {path}")
    return suffix


def load_contract(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Contract is missing: {path}")
    contract = json.loads(path.read_text(encoding="utf-8"))
    schema = contract.get("atomic_record_schema", {})
    if schema.get("required_fields") != [
        "schema_version",
        "uid",
        "dataset",
        "source",
        "task",
        "split",
        "audio",
        "targets",
        "metadata",
    ]:
        raise ValueError(f"Unexpected required atomic fields: {schema.get('required_fields')}")
    return contract


def validate_atomic_record(record: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "uid",
        "dataset",
        "source",
        "task",
        "split",
        "audio",
        "targets",
        "metadata",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"Atomic record is missing fields {missing}: {record}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unexpected schema version: {record['schema_version']!r}")
    if record["task"] not in {"AAC", "ASR"}:
        raise ValueError(f"Unexpected task: {record['task']!r}")
    if not isinstance(record["uid"], str) or not record["uid"].strip():
        raise ValueError("Atomic UID must be a non-empty string")
    audio = record["audio"]
    if not isinstance(audio, dict) or not normalize_text(audio.get("path")) or not normalize_text(audio.get("format")):
        raise ValueError(f"Invalid atomic audio reference: {audio!r}")
    targets = record["targets"]
    if not isinstance(targets, list) or not targets or any(not normalize_text(target) for target in targets):
        raise ValueError(f"Invalid atomic targets: {targets!r}")
    if not isinstance(record["metadata"], dict):
        raise ValueError("Atomic metadata must be an object")
    if "effective_audio_tokens" in record:
        raise ValueError("Pilot records must not precompute effective_audio_tokens")


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield record


def first_assistant_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "assistant":
            return normalize_text(message.get("content"))
    return ""


def build_audiocaps_pilot(manifest: Path, limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not manifest.is_file():
        raise FileNotFoundError(f"Verified AudioCaps Swift manifest is missing: {manifest}")
    output: list[dict[str, Any]] = []
    for source_record in iter_jsonl(manifest):
        audios = source_record.get("audios")
        if not isinstance(audios, list) or len(audios) != 1 or not isinstance(audios[0], str):
            raise ValueError(f"Unexpected AudioCaps audio field: {audios!r}")
        path = Path(audios[0])
        caption = first_assistant_text(source_record.get("messages"))
        if not caption:
            raise ValueError(f"AudioCaps record has no assistant caption: {source_record}")
        metadata = source_record.get("metadata") if isinstance(source_record.get("metadata"), dict) else {}
        sample_id = normalize_text(metadata.get("sample_id")) or path.stem
        output.append(
            {
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
        )
        if len(output) >= limit:
            break
    return output, {"source_manifest": str(manifest), "mapping": "verified Swift manifest -> atomic AAC"}


def inventory_wavcaps_sources(
    inventory: dict[str, Any], expected_root: Path
) -> tuple[dict[str, Path], dict[str, Path]]:
    pool = inventory["pools"]["wavcaps_no_bbc_aac"]
    if pool.get("excluded_sources") != ["BBC_Sound_Effects"]:
        raise ValueError("Inventory does not prove the exact BBC source exclusion")
    audio_dirs: dict[str, Path] = {}
    for report in pool.get("source_reports", {}).values():
        canonical = report.get("canonical_source")
        if canonical in {"FreeSound", "AudioSet_SL", "SoundBible"}:
            audio_dirs[canonical] = Path(report["path"])
    metadata_paths: dict[str, Path] = {}
    for report in pool.get("metadata_reports", []):
        canonical = report.get("canonical_source")
        if canonical in {"FreeSound", "AudioSet_SL", "SoundBible"}:
            metadata_paths[canonical] = Path(report["path"])
    required = {"FreeSound", "AudioSet_SL", "SoundBible"}
    if set(audio_dirs) != required or set(metadata_paths) != required:
        raise ValueError(f"Incomplete WavCaps mapping: audio={audio_dirs} metadata={metadata_paths}")
    resolved_root = expected_root.resolve()
    escaped = [
        str(path)
        for path in [*audio_dirs.values(), *metadata_paths.values()]
        if not path.resolve().is_relative_to(resolved_root)
    ]
    if escaped:
        raise ValueError(f"Inventory WavCaps paths escape the configured read-only root: {escaped}")
    return audio_dirs, metadata_paths


def iter_wavcaps_metadata(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        yield from iter_jsonl(path)
        return
    for record in iter_root_json_array(path):
        if not isinstance(record, dict):
            raise ValueError(f"WavCaps metadata contains non-object record: {path}")
        yield record


def extract_wavcaps_id(record: dict[str, Any], source: str) -> str:
    fields = ("key", "id") if source == "AudioSet_SL" else ("id", "key", "audio_id", "uid", "name")
    for field in fields:
        value = normalize_text(record.get(field))
        if value:
            return Path(value).stem
    raise ValueError(f"Cannot find WavCaps ID for source={source}: keys={sorted(record)}")


def extract_targets(record: dict[str, Any]) -> list[str]:
    for field in ("target", "caption", "captions", "description", "text", "sentence"):
        value = record.get(field)
        if isinstance(value, str) and normalize_text(value):
            return [normalize_text(value)]
        if isinstance(value, list):
            targets = [normalize_text(item) for item in value if normalize_text(item)]
            if targets:
                return list(dict.fromkeys(targets))
    raise ValueError(f"Cannot find caption target in metadata keys={sorted(record)}")


def infer_source_suffix(audio_dir: Path) -> str:
    for path in audio_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".wav", ".flac", ".opus"}:
            return path.suffix.lower()
    raise FileNotFoundError(f"No sample audio file found under {audio_dir}")


def build_wavcaps_pilot(
    inventory: dict[str, Any],
    wavcaps_root: Path,
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audio_dirs, metadata_paths = inventory_wavcaps_sources(inventory, wavcaps_root)
    sources = ("FreeSound", "AudioSet_SL", "SoundBible")
    base_quota, remainder = divmod(limit, len(sources))
    source_quotas = {
        source: base_quota + int(index < remainder) for index, source in enumerate(sources)
    }
    output: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    mapping: dict[str, Any] = {}
    for source in sources:
        audio_dir = audio_dirs[source]
        metadata_path = metadata_paths[source]
        suffix = infer_source_suffix(audio_dir)
        mapping[source] = {
            "audio_dir": str(audio_dir),
            "metadata_path": str(metadata_path),
            "audio_suffix": suffix,
        }
        for record in iter_wavcaps_metadata(metadata_path):
            sample_id = extract_wavcaps_id(record, source)
            targets = extract_targets(record)
            path = audio_dir / f"{sample_id}{suffix}"
            output.append(
                {
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
            )
            source_counts[source] += 1
            if source_counts[source] >= source_quotas[source]:
                break
    return output, {
        "source_quotas": source_quotas,
        "source_counts": dict(source_counts),
        "mapping": mapping,
        "bbc_records": 0,
    }


def build_clotho_pilot(root: Path, manifest_name: str, limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = root / manifest_name
    records = load_json_or_jsonl(manifest)
    grouped: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Clotho manifest contains a non-object record")
        raw_path = record.get("audio_path") or record.get("audio")
        caption = normalize_text(record.get("caption", record.get("response")))
        if not isinstance(raw_path, str) or not raw_path.strip() or not caption:
            raise ValueError(f"Invalid Clotho train record: {record}")
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        grouped[str(path)].add(caption)

    output: list[dict[str, Any]] = []
    multiplicity: Counter[int] = Counter()
    for rendered_path in sorted(grouped):
        path = Path(rendered_path)
        targets = sorted(grouped[rendered_path])
        multiplicity[len(targets)] += 1
        output.append(
            {
                "schema_version": SCHEMA_VERSION,
                "uid": f"clotho_v2:{path.stem}",
                "dataset": "Clotho-v2",
                "source": "Clotho-v2",
                "task": "AAC",
                "split": "train",
                "audio": {"path": str(path), "format": audio_format(path)},
                "targets": targets,
                "metadata": {
                    "sample_id": path.stem,
                    "target_selection": "one caption per scheduled training occurrence",
                },
            }
        )
        if len(output) >= limit:
            break
    return output, {"source_manifest": str(manifest), "pilot_caption_multiplicity": dict(multiplicity)}


def build_gigaspeech_pilot(
    root: Path,
    metadata_name: str,
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata_path = root / metadata_name
    output: list[dict[str, Any]] = []
    scanned_audio_objects = 0
    scanned_segments = 0
    placeholder_counts: Counter[str] = Counter()
    for audio in iter_named_json_array(metadata_path, "audios"):
        scanned_audio_objects += 1
        if not isinstance(audio, dict):
            continue
        raw_path = normalize_text(audio.get("path"))
        source = normalize_text(audio.get("source")).lower()
        path = root / raw_path
        segments = audio.get("segments")
        if not raw_path or not isinstance(segments, list):
            continue
        for segment in segments:
            scanned_segments += 1
            if not isinstance(segment, dict) or "{L}" not in (segment.get("subsets") or []):
                continue
            sid = normalize_text(segment.get("sid"))
            raw_text = normalize_text(segment.get("text_tn"))
            cleaned_text = clean_gigaspeech_text(raw_text)
            if not sid or not cleaned_text:
                raise ValueError(f"Invalid GigaSpeech-L segment: sid={sid!r} text={raw_text!r}")
            begin = float(segment["begin_time"])
            end = float(segment["end_time"])
            if begin < 0 or end <= begin:
                raise ValueError(f"Invalid GigaSpeech segment bounds: sid={sid} begin={begin} end={end}")
            for tag in re.findall(r"<[^<>]+>", raw_text):
                placeholder_counts[tag] += 1
            output.append(
                {
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
                    "raw_duration_sec": end - begin,
                    "targets": [cleaned_text],
                    "metadata": {"sid": sid, "text_tn_raw": raw_text, "subsets": segment.get("subsets")},
                }
            )
            if len(output) >= limit:
                return output, {
                    "metadata_path": str(metadata_path),
                    "scanned_audio_objects": scanned_audio_objects,
                    "scanned_segments": scanned_segments,
                    "placeholder_counts": dict(sorted(placeholder_counts.items())),
                }
    return output, {
        "metadata_path": str(metadata_path),
        "scanned_audio_objects": scanned_audio_objects,
        "scanned_segments": scanned_segments,
        "placeholder_counts": dict(sorted(placeholder_counts.items())),
    }


def validate_pool_records(pool_name: str, records: list[dict[str, Any]], expected_count: int) -> dict[str, Any]:
    if not records:
        raise ValueError(f"Pool {pool_name} produced no pilot records")
    if len(records) != expected_count:
        raise ValueError(f"Pool {pool_name} expected {expected_count} records, got {len(records)}")
    uid_counts: Counter[str] = Counter()
    missing_paths: list[str] = []
    source_counts: Counter[str] = Counter()
    target_count_histogram: Counter[int] = Counter()
    for record in records:
        validate_atomic_record(record)
        uid_counts[record["uid"]] += 1
        source_counts[record["source"]] += 1
        target_count_histogram[len(record["targets"])] += 1
        if not Path(record["audio"]["path"]).is_file():
            missing_paths.append(record["audio"]["path"])
    duplicates = sorted(uid for uid, count in uid_counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"Pool {pool_name} has duplicate pilot UIDs: {duplicates[:10]}")
    if missing_paths:
        raise FileNotFoundError(f"Pool {pool_name} pilot has missing audio paths: {missing_paths[:10]}")
    if pool_name == "wavcaps_no_bbc_aac" and "BBC_Sound_Effects" in source_counts:
        raise ValueError("BBC Sound Effects leaked into the WavCaps pilot")
    return {
        "record_count": len(records),
        "source_counts": dict(sorted(source_counts.items())),
        "target_count_histogram": {str(key): value for key, value in sorted(target_count_histogram.items())},
        "missing_audio_path_count": 0,
        "duplicate_uid_count": 0,
        "effective_audio_tokens_present": False,
    }


def main() -> None:
    args = parse_args()
    if args.records_per_pool < 3:
        raise ValueError("records_per_pool must be at least 3 so every retained WavCaps source is represented")
    contract = load_contract(Path(args.contract))
    inventory_path = Path(args.inventory_report)
    if not inventory_path.is_file():
        raise FileNotFoundError(f"Passed inventory report is missing: {inventory_path}")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if not inventory.get("inspection_passed") or inventory.get("blocking_issues"):
        raise ValueError("The metadata inventory has not passed cleanly")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("========== HUGINN WHISPER DYNAMIC90S ATOMIC PILOT START ==========", flush=True)
    print("[scope] full_manifests=false audio_decode=false audio_copy=false token_accounting=false", flush=True)
    print(f"[pilot] records_per_pool={args.records_per_pool} output_dir={output_dir}", flush=True)

    builders = {
        "audiocaps_v2_aac": lambda: build_audiocaps_pilot(
            Path(args.audiocaps_manifest), args.records_per_pool
        ),
        "wavcaps_no_bbc_aac": lambda: build_wavcaps_pilot(
            inventory, Path(args.wavcaps_root), args.records_per_pool
        ),
        "clotho_v2_aac": lambda: build_clotho_pilot(
            Path(args.clotho_root), args.clotho_train_manifest, args.records_per_pool
        ),
        "gigaspeech_l_asr": lambda: build_gigaspeech_pilot(
            Path(args.gigaspeech_root), args.gigaspeech_metadata, args.records_per_pool
        ),
    }
    report: dict[str, Any] = {
        "gate": "huginn_whisper_dynamic90s_atomic_pilot_v1",
        "contract_version": contract.get("contract_version"),
        "schema_version": SCHEMA_VERSION,
        "records_per_pool": args.records_per_pool,
        "full_manifests": False,
        "audio_decode": False,
        "audio_copy": False,
        "token_accounting": False,
        "pools": {},
    }
    registry: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "pools": {}}
    global_uids: set[str] = set()
    for pool_name, builder in builders.items():
        print(f"[pilot] pool={pool_name} building=true", flush=True)
        records, mapping = builder()
        validation = validate_pool_records(pool_name, records, args.records_per_pool)
        overlap = sorted(global_uids.intersection(record["uid"] for record in records))
        if overlap:
            raise ValueError(f"Cross-pool UID collision: {overlap[:10]}")
        global_uids.update(record["uid"] for record in records)
        output_path = output_dir / POOL_FILENAMES[pool_name]
        write_jsonl_atomic(output_path, records)
        report["pools"][pool_name] = {"output_path": str(output_path), "mapping": mapping, **validation}
        registry["pools"][pool_name] = {
            "path": str(output_path),
            "task": records[0]["task"],
            "record_count": len(records),
        }
        print(
            f"[pilot] pool={pool_name} records={len(records)} sources={validation['source_counts']} passed=true",
            flush=True,
        )

    report["global_unique_uid_count"] = len(global_uids)
    report["validation_passed"] = True
    report_path = output_dir / "atomic_pilot_report.json"
    registry_path = output_dir / "pool_registry.pilot.json"
    write_json_atomic(report_path, report)
    write_json_atomic(registry_path, registry)
    print(f"[pilot] report={report_path}", flush=True)
    print(f"[pilot] registry={registry_path}", flush=True)
    print("========== HUGINN WHISPER DYNAMIC90S ATOMIC PILOT PASSED ==========", flush=True)


if __name__ == "__main__":
    main()
