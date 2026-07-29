"""Lightweight metadata-only inventory for Huginn Whisper dynamic-90s data.

The gate never downloads, copies, converts, decodes, or scans every audio file.
It reads source metadata and checks only a small deterministic sample of audio
locations. Duration/token accounting is deferred to training-time statistics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONTRACT = (
    DEFAULT_REPO_ROOT
    / "code"
    / "huginn_lora"
    / "configs"
    / "huginn_whisper_dynamic90s_data_contract_v1.json"
)
DEFAULT_AUDIOCAPS_ROOT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/data/audiocaps_v2")
DEFAULT_WAVCAPS_ROOT = Path("/hpc_stor03/public/shared/data/raa/WavCaps")
DEFAULT_CLOTHO_ROOT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_caption_huginn")
DEFAULT_GIGASPEECH_ROOT = Path("/hpc_stor03/public/shared/data/asr/am/GigaSpeech")
DEFAULT_OUTPUT = (
    DEFAULT_REPO_ROOT
    / "data"
    / "audio_swift"
    / "huginn_whisper_dynamic90s_multitask"
    / "v1"
    / "audits"
    / "data_pool_inventory.json"
)
ATOMIC_SCHEMA_VERSION = "huginn_whisper_dynamic90s_atomic_v1"
AUDIO_SUFFIXES = {".wav", ".flac", ".opus", ".ogg", ".mp3", ".m4a"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--audiocaps_root", default=str(DEFAULT_AUDIOCAPS_ROOT))
    parser.add_argument("--audiocaps_split", default="train")
    parser.add_argument("--wavcaps_root", default=str(DEFAULT_WAVCAPS_ROOT))
    parser.add_argument("--clotho_root", default=str(DEFAULT_CLOTHO_ROOT))
    parser.add_argument("--clotho_train_manifest", default="train_expand.json")
    parser.add_argument("--gigaspeech_root", default=str(DEFAULT_GIGASPEECH_ROOT))
    parser.add_argument("--gigaspeech_metadata", default="GigaSpeech.json")
    parser.add_argument("--probe_count", type=int, default=4)
    parser.add_argument("--metadata_schema_records", type=int, default=20)
    parser.add_argument("--output_report", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
    temporary.replace(path)


def limited_append(mapping: dict[str, list[str]], key: str, value: str, limit: int = 10) -> None:
    examples = mapping.setdefault(key, [])
    if len(examples) < limit:
        examples.append(value)


def validate_contract(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Data contract not found: {path}")
    contract = json.loads(path.read_text(encoding="utf-8"))
    runtime = contract.get("audio_runtime_contract", {})
    expected = {
        "waveform_dtype": "float32",
        "channels": 1,
        "sample_rate_hz": 16000,
        "chunk_seconds": 30.0,
        "max_included_seconds": 90.0,
        "audio_token_duration_ms": 120,
    }
    mismatches = {
        key: {"expected": value, "actual": runtime.get(key)}
        for key, value in expected.items()
        if runtime.get(key) != value
    }
    required = contract.get("atomic_record_schema", {}).get("required_fields", [])
    optional = contract.get("atomic_record_schema", {}).get("optional_fields", [])
    if mismatches:
        raise ValueError(f"Data contract/runtime mismatch: {mismatches}")
    if "targets" not in required or "effective_audio_tokens" not in optional:
        raise ValueError(f"Atomic record schema is incomplete: required={required} optional={optional}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "contract_version": contract.get("contract_version"),
        "atomic_schema_version": ATOMIC_SCHEMA_VERSION,
        "validated": True,
        "note": "No per-record duration or token accounting is performed; training-time statistics will track it.",
    }


def check_sample_paths(paths: list[Path], limit: int) -> dict[str, Any]:
    checked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        rendered = str(path)
        if rendered in seen:
            continue
        seen.add(rendered)
        checked.append({"path": rendered, "exists": path.is_file(), "suffix": path.suffix.lower()})
        if len(checked) >= limit:
            break
    return {
        "checked_count": len(checked),
        "missing_count": sum(not item["exists"] for item in checked),
        "items": checked,
    }


def inspect_audiocaps(root: Path, split: str, probe_count: int) -> dict[str, Any]:
    csv_path = root / f"{split}.csv"
    audio_dir = root / split
    if not csv_path.is_file() or not audio_dir.is_dir():
        raise FileNotFoundError(f"AudioCaps layout missing: csv={csv_path} audio_dir={audio_dir}")

    errors: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    audio_ids: Counter[str] = Counter()
    sample_paths: list[Path] = []
    source_rows = 0
    valid_metadata_rows = 0
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])
        if "youtube_id" not in headers or "caption" not in headers:
            raise ValueError(f"Unexpected AudioCaps headers: {headers}")
        for row_number, row in enumerate(reader, start=2):
            source_rows += 1
            audio_id = str(row.get("youtube_id") or "").strip()
            caption = str(row.get("caption") or "").strip()
            if not audio_id:
                errors["empty_audio_id"] += 1
                limited_append(examples, "empty_audio_id", f"row={row_number}")
                continue
            if not caption:
                errors["empty_caption"] += 1
                limited_append(examples, "empty_caption", f"row={row_number} id={audio_id}")
                continue
            stem = Path(audio_id).stem
            if not stem.startswith("Y"):
                stem = f"Y{stem}"
            audio_ids[stem] += 1
            valid_metadata_rows += 1
            if len(sample_paths) < probe_count:
                sample_paths.append(audio_dir / f"{stem}.wav")

    return {
        "dataset": "AudioCaps-v2",
        "task": "AAC",
        "split_policy": "train",
        "root": str(root),
        "csv_path": str(csv_path),
        "source_row_count": source_rows,
        "valid_metadata_row_count": valid_metadata_rows,
        "metadata_error_counts": dict(sorted(errors.items())),
        "metadata_error_examples": examples,
        "unique_audio_id_count": len(audio_ids),
        "duplicate_audio_id_count": sum(count > 1 for count in audio_ids.values()),
        "audio_path_rule": str(audio_dir / "Y<youtube_id>.wav"),
        "sample_audio_locations": check_sample_paths(sample_paths, probe_count),
        "full_audio_scan_performed": False,
        "audio_decode_performed": False,
    }


def canonical_wavcaps_source(name: str) -> str:
    lowered = re.sub(r"[^a-z0-9]+", "", name.lower())
    if "bbc" in lowered:
        return "BBC_Sound_Effects"
    if "freesound" in lowered:
        return "FreeSound"
    if "soundbible" in lowered:
        return "SoundBible"
    if "audioset" in lowered:
        return "AudioSet_SL"
    return name


def iter_root_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[Any]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                raise ValueError(f"Empty JSON file: {path}")
            buffer += chunk
            stripped = buffer.lstrip()
            if stripped:
                if not stripped.startswith("["):
                    raise ValueError("top-level JSON value is not an array")
                buffer = stripped[1:]
                break
        while True:
            buffer = buffer.lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:]
                continue
            if buffer.startswith("]"):
                return
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError as exc:
                chunk = handle.read(chunk_size)
                if not chunk:
                    raise ValueError(f"Malformed JSON array in {path}: {exc}") from exc
                buffer += chunk
                continue
            yield value
            buffer = buffer[end:]


def observe_schema_record(
    record: Any,
    key_counts: Counter[str],
    examples: dict[str, list[str]],
) -> bool:
    if not isinstance(record, dict):
        return False
    for key, value in record.items():
        key_counts[key] += 1
        if len(examples.setdefault(key, [])) < 2:
            rendered = json.dumps(value, ensure_ascii=False)
            examples[key].append(rendered[:200] + ("..." if len(rendered) > 200 else ""))
    return True


def inspect_json_metadata_sample(path: Path, schema_limit: int) -> dict[str, Any]:
    key_counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    scanned = 0
    malformed = 0
    first_record: dict[str, Any] | None = None
    sampling_note = ""

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if first_record is None and isinstance(record, dict):
                    first_record = record
                malformed += int(not observe_schema_record(record, key_counts, examples))
                scanned += 1
                if scanned >= schema_limit:
                    break
        sampling_note = "Only the first metadata records were read; total rows were not counted."
    else:
        try:
            iterator = iter_root_json_array(path)
            for record in iterator:
                if first_record is None and isinstance(record, dict):
                    first_record = record
                malformed += int(not observe_schema_record(record, key_counts, examples))
                scanned += 1
                if scanned >= schema_limit:
                    break
            sampling_note = "Only the first records of the top-level JSON array were read."
        except ValueError as exc:
            with path.open("r", encoding="utf-8") as handle:
                preview = handle.read(4096)
            sampling_note = f"Schema sampling deferred: {exc}; first_4096_chars={preview!r}"

    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "scanned_record_count": scanned,
        "malformed_sample_count": malformed,
        "field_presence_counts": dict(sorted(key_counts.items())),
        "field_examples": {key: examples[key] for key in sorted(examples)},
        "first_record": first_record,
        "sampling_note": sampling_note,
    }


def find_first_audio_files(root: Path, limit: int) -> list[Path]:
    results: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
            results.append(path)
            if len(results) >= limit:
                break
    return results


def inspect_wavcaps(root: Path, schema_limit: int) -> dict[str, Any]:
    audio_root = root / "audio"
    metadata_root = root / "json"
    if not audio_root.is_dir() or not metadata_root.is_dir():
        raise FileNotFoundError(f"WavCaps layout missing: audio={audio_root} metadata={metadata_root}")

    source_reports: dict[str, dict[str, Any]] = {}
    for child in sorted(audio_root.iterdir(), key=lambda item: item.name.lower()):
        if not child.is_dir():
            continue
        canonical = canonical_wavcaps_source(child.name)
        sample_paths = find_first_audio_files(child, 1)
        source_reports[child.name] = {
            "canonical_source": canonical,
            "path": str(child),
            "training_eligible": canonical != "BBC_Sound_Effects",
            "exclusion_reason": "source-level BBC exclusion" if canonical == "BBC_Sound_Effects" else None,
            "sample_audio_locations": check_sample_paths(sample_paths, 1),
            "full_directory_scan_performed": False,
        }

    metadata_paths = sorted(
        path for path in metadata_root.rglob("*") if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}
    )
    metadata_reports = []
    for path in metadata_paths:
        report = inspect_json_metadata_sample(path, schema_limit)
        report["canonical_source"] = canonical_wavcaps_source(path.stem)
        report["training_eligible"] = report["canonical_source"] != "BBC_Sound_Effects"
        metadata_reports.append(report)

    discovered = sorted(
        {report["canonical_source"] for report in source_reports.values()}
        | {report["canonical_source"] for report in metadata_reports}
    )
    return {
        "dataset": "WavCaps",
        "task": "AAC",
        "root": str(root),
        "public_root_read_only": True,
        "source_reports": source_reports,
        "metadata_reports": metadata_reports,
        "discovered_canonical_sources": discovered,
        "required_eligible_sources": ["FreeSound", "AudioSet_SL", "SoundBible"],
        "excluded_sources": ["BBC_Sound_Effects"],
        "full_audio_scan_performed": False,
        "audio_decode_performed": False,
    }


def load_json_or_jsonl(path: Path) -> list[Any]:
    if path.suffix.lower() == ".jsonl":
        records: list[Any] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSONL at {path}:{line_number}: {exc}") from exc
        return records
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(payload).__name__}")
    return payload


def inspect_clotho(root: Path, manifest_name: str, probe_count: int) -> dict[str, Any]:
    manifest_path = root / manifest_name
    if not root.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError(f"Clotho train input missing: root={root} manifest={manifest_path}")
    if "train" not in manifest_path.name.lower():
        raise ValueError(f"Clotho manifest is not explicitly a train manifest: {manifest_path}")

    records = load_json_or_jsonl(manifest_path)
    errors: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    grouped_targets: dict[str, set[str]] = defaultdict(set)
    path_objects: dict[str, Path] = {}
    split_leakage_indicators = 0
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors["non_object_record"] += 1
            limited_append(examples, "non_object_record", f"index={index}")
            continue
        raw_audio = record.get("audio_path") or record.get("audio")
        caption = record.get("caption", record.get("response"))
        if not isinstance(raw_audio, str) or not raw_audio.strip():
            errors["missing_audio_path"] += 1
            limited_append(examples, "missing_audio_path", f"index={index}")
            continue
        if not isinstance(caption, str) or not caption.strip():
            errors["empty_caption"] += 1
            limited_append(examples, "empty_caption", f"index={index} audio={raw_audio}")
            continue
        path = Path(raw_audio)
        if not path.is_absolute():
            path = root / path
        rendered = str(path)
        lowered = rendered.replace("\\", "/").lower()
        if any(marker in lowered for marker in ("/test/", "/val/", "/validation/", "/evaluation/")):
            split_leakage_indicators += 1
        grouped_targets[rendered].add(caption.strip())
        path_objects[rendered] = path

    multiplicity = Counter(len(targets) for targets in grouped_targets.values())
    sample_paths = [path_objects[key] for key in sorted(path_objects)[:probe_count]]
    return {
        "dataset": "Clotho-v2",
        "task": "AAC",
        "root": str(root),
        "manifest_path": str(manifest_path),
        "split_policy": "train only",
        "source_record_count": len(records),
        "grouped_audio_count": len(grouped_targets),
        "caption_multiplicity_per_audio": {str(key): value for key, value in sorted(multiplicity.items())},
        "invalid_record_counts": dict(sorted(errors.items())),
        "invalid_record_examples": examples,
        "split_leakage_indicator_count": split_leakage_indicators,
        "runtime_target_policy": "select exactly one deterministic caption per scheduled training occurrence",
        "atomic_manifest_policy": "one grouped row per audio; do not expand one audio into five independent rows",
        "sample_audio_locations": check_sample_paths(sample_paths, probe_count),
        "full_audio_scan_performed": False,
        "audio_decode_performed": False,
    }


def iter_named_json_array(path: Path, key: str, chunk_size: int = 1024 * 1024) -> Iterator[Any]:
    """Stream objects from a named top-level JSON array without loading GigaSpeech.json."""
    decoder = json.JSONDecoder()
    key_pattern = re.compile(rf'"{re.escape(key)}"\s*:\s*\[')
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                raise ValueError(f"Could not find top-level array {key!r} in {path}")
            buffer += chunk
            match = key_pattern.search(buffer)
            if match is not None:
                buffer = buffer[match.end() :]
                break
            buffer = buffer[-max(4096, len(key) * 4) :]
        while True:
            buffer = buffer.lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:]
                continue
            if buffer.startswith("]"):
                return
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError as exc:
                chunk = handle.read(chunk_size)
                if not chunk:
                    raise ValueError(f"Incomplete or malformed {key!r} array in {path}: {exc}") from exc
                buffer += chunk
                continue
            yield value
            buffer = buffer[end:]


def inspect_gigaspeech(root: Path, metadata_name: str, probe_count: int) -> dict[str, Any]:
    metadata_path = root / metadata_name
    if not root.is_dir() or not metadata_path.is_file():
        raise FileNotFoundError(f"GigaSpeech layout missing: root={root} metadata={metadata_path}")

    audio_count = 0
    source_audio_counts: Counter[str] = Counter()
    total_segment_count = 0
    l_segment_count = 0
    l_segment_source_counts: Counter[str] = Counter()
    l_duration_seconds = 0.0
    invalid_counts: Counter[str] = Counter()
    invalid_examples: dict[str, list[str]] = {}
    placeholder_counts: Counter[str] = Counter()
    sid_counts: Counter[str] = Counter()
    parent_extensions: Counter[str] = Counter()
    observed_parent_paths: set[str] = set()
    sample_parent_paths: list[Path] = []
    first_l_segment: dict[str, Any] | None = None

    for audio_index, audio in enumerate(iter_named_json_array(metadata_path, "audios")):
        audio_count += 1
        if not isinstance(audio, dict):
            invalid_counts["non_object_audio"] += 1
            limited_append(invalid_examples, "non_object_audio", f"audio_index={audio_index}")
            continue
        source = str(audio.get("source") or "<missing>").strip().lower()
        source_audio_counts[source] += 1
        raw_path = str(audio.get("path") or "").strip()
        parent_duration = audio.get("duration")
        segments = audio.get("segments")
        if not isinstance(segments, list):
            invalid_counts["missing_segments_list"] += 1
            limited_append(invalid_examples, "missing_segments_list", f"audio_index={audio_index} path={raw_path}")
            continue
        total_segment_count += len(segments)
        audio_path = root / raw_path if raw_path else None

        for segment_index, segment in enumerate(segments):
            if not isinstance(segment, dict):
                invalid_counts["non_object_segment"] += 1
                limited_append(
                    invalid_examples,
                    "non_object_segment",
                    f"audio_index={audio_index} segment_index={segment_index}",
                )
                continue
            subsets = segment.get("subsets")
            if not isinstance(subsets, list) or "{L}" not in subsets:
                continue
            l_segment_count += 1
            l_segment_source_counts[source] += 1
            if audio_path is None:
                invalid_counts["empty_l_parent_audio_path"] += 1
                limited_append(invalid_examples, "empty_l_parent_audio_path", f"audio_index={audio_index}")
            else:
                rendered_path = str(audio_path)
                if rendered_path not in observed_parent_paths:
                    observed_parent_paths.add(rendered_path)
                    parent_extensions[audio_path.suffix.lower() or "<none>"] += 1
                    if len(sample_parent_paths) < probe_count:
                        sample_parent_paths.append(audio_path)

            sid = str(segment.get("sid") or "").strip()
            if sid:
                sid_counts[sid] += 1
            else:
                invalid_counts["empty_l_sid"] += 1
                limited_append(invalid_examples, "empty_l_sid", f"audio_index={audio_index}")
            try:
                begin = float(segment.get("begin_time"))
                end = float(segment.get("end_time"))
            except (TypeError, ValueError):
                invalid_counts["invalid_l_segment_time"] += 1
                limited_append(invalid_examples, "invalid_l_segment_time", f"sid={sid}")
                continue
            duration = end - begin
            if not math.isfinite(begin) or not math.isfinite(end) or begin < 0 or duration <= 0:
                invalid_counts["invalid_l_segment_time"] += 1
                limited_append(invalid_examples, "invalid_l_segment_time", f"sid={sid} begin={begin} end={end}")
                continue
            if isinstance(parent_duration, (int, float)) and end > float(parent_duration) + 0.05:
                invalid_counts["l_segment_exceeds_parent_duration"] += 1
                limited_append(
                    invalid_examples,
                    "l_segment_exceeds_parent_duration",
                    f"sid={sid} end={end} parent_duration={parent_duration}",
                )
            text_tn = str(segment.get("text_tn") or "").strip()
            if not text_tn:
                invalid_counts["empty_l_text_tn"] += 1
                limited_append(invalid_examples, "empty_l_text_tn", f"sid={sid}")
            for tag in re.findall(r"<[^<>]+>", text_tn):
                placeholder_counts[tag] += 1
            l_duration_seconds += duration
            if first_l_segment is None:
                first_l_segment = {
                    "sid": sid,
                    "source": source,
                    "audio_path": str(audio_path) if audio_path is not None else None,
                    "audio_format": audio.get("format"),
                    "begin_time": begin,
                    "end_time": end,
                    "duration_seconds": duration,
                    "subsets": subsets,
                    "text_tn_preview": text_tn[:240],
                }

    return {
        "dataset": "GigaSpeech",
        "task": "ASR",
        "root": str(root),
        "metadata_path": str(metadata_path),
        "public_root_read_only": True,
        "selection_policy": "include a segment iff its own subsets list contains {L}",
        "audio_object_count": audio_count,
        "source_audio_counts": dict(sorted(source_audio_counts.items())),
        "all_segment_count": total_segment_count,
        "l_segment_count": l_segment_count,
        "l_segment_source_counts": dict(sorted(l_segment_source_counts.items())),
        "l_raw_duration_hours_from_metadata": l_duration_seconds / 3600.0,
        "unique_l_sid_count": len(sid_counts),
        "duplicate_l_sid_count": sum(count > 1 for count in sid_counts.values()),
        "referenced_l_parent_audio_count": len(observed_parent_paths),
        "l_parent_audio_extension_counts": dict(sorted(parent_extensions.items())),
        "invalid_counts": dict(sorted(invalid_counts.items())),
        "invalid_examples": invalid_examples,
        "text_tn_placeholder_counts": dict(sorted(placeholder_counts.items())),
        "sample_audio_locations": check_sample_paths(sample_parent_paths, probe_count),
        "first_l_segment": first_l_segment,
        "full_audio_scan_performed": False,
        "audio_decode_performed": False,
        "token_accounting_performed": False,
    }


def build_blocking_issues(report: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    for pool_name, pool_report in report["pools"].items():
        if "inspection_error" in pool_report:
            issues.append(f"{pool_name} inspection failed: {pool_report['inspection_error']}")

    audiocaps = report["pools"]["audiocaps_v2_aac"]
    if "inspection_error" not in audiocaps:
        if audiocaps["valid_metadata_row_count"] == 0:
            issues.append("AudioCaps has no valid train metadata rows")
        if audiocaps["sample_audio_locations"]["missing_count"]:
            issues.append("AudioCaps sampled audio location is missing")

    wavcaps = report["pools"]["wavcaps_no_bbc_aac"]
    if "inspection_error" not in wavcaps:
        discovered = set(wavcaps["discovered_canonical_sources"])
        required = set(wavcaps["required_eligible_sources"])
        if missing := sorted(required - discovered):
            issues.append(f"WavCaps required sources were not discovered: {missing}")
        if "BBC_Sound_Effects" not in discovered:
            issues.append("WavCaps BBC source was not discovered, so source-level exclusion is not proven")
        for source_name, source_report in wavcaps["source_reports"].items():
            if source_report["training_eligible"] and source_report["sample_audio_locations"]["checked_count"] == 0:
                issues.append(f"WavCaps eligible source has no sampled audio location: {source_name}")

    clotho = report["pools"]["clotho_v2_aac"]
    if "inspection_error" not in clotho:
        if clotho["grouped_audio_count"] == 0:
            issues.append("Clotho train manifest has no grouped audio")
        if clotho["invalid_record_counts"]:
            issues.append(f"Clotho contains invalid metadata records: {clotho['invalid_record_counts']}")
        if clotho["split_leakage_indicator_count"]:
            issues.append("Clotho train manifest contains val/test/evaluation path indicators")
        if clotho["sample_audio_locations"]["missing_count"]:
            issues.append("Clotho sampled train audio location is missing")

    giga = report["pools"]["gigaspeech_l_asr"]
    if "inspection_error" not in giga:
        if giga["l_segment_count"] == 0:
            issues.append("GigaSpeech has no segment-level {L} records")
        if giga["duplicate_l_sid_count"]:
            issues.append(f"GigaSpeech has {giga['duplicate_l_sid_count']} duplicate L segment IDs")
        if giga["invalid_counts"]:
            issues.append(f"GigaSpeech contains invalid L metadata: {giga['invalid_counts']}")
        unexpected = sorted(ext for ext in giga["l_parent_audio_extension_counts"] if ext != ".opus")
        if unexpected:
            issues.append(f"GigaSpeech-L parent audio has unexpected extensions: {unexpected}")
        if giga["sample_audio_locations"]["missing_count"]:
            issues.append("GigaSpeech sampled parent Opus location is missing")
    return issues


def run_pool_inspection(pool_name: str, inspector: Any, *args: Any) -> dict[str, Any]:
    print(f"[inspect] pool={pool_name}", flush=True)
    try:
        result = inspector(*args)
    except Exception as exc:  # pragma: no cover - depends on remote data layout
        error = f"{type(exc).__name__}: {exc}"
        print(f"[inspect-error] pool={pool_name} error={error}", flush=True)
        return {"inspection_error": error, "traceback": traceback.format_exc()}
    print(f"[inspect] pool={pool_name} completed=true", flush=True)
    return result


def main() -> None:
    args = parse_args()
    if args.probe_count <= 0 or args.metadata_schema_records <= 0:
        raise ValueError("probe_count and metadata_schema_records must be positive")

    print("========== HUGINN WHISPER DYNAMIC90S METADATA INSPECT START ==========", flush=True)
    print("[scope] route=Huginn Whisper dynamic-90s only", flush=True)
    print("[scope] metadata_only=true source_roots_read_only=true", flush=True)
    print("[scope] downloads=0 copies=0 conversions=0 audio_decodes=0 full_audio_scans=0", flush=True)
    print("[scope] token_accounting=deferred_to_training_time_statistics", flush=True)

    report: dict[str, Any] = {
        "audit": "huginn_whisper_dynamic90s_metadata_inventory_v1",
        "access_contract": {
            "metadata_only": True,
            "downloads": 0,
            "copies": 0,
            "audio_conversions": 0,
            "audio_decodes": 0,
            "full_audio_directory_scans": 0,
            "per_record_token_accounting": False,
            "sample_audio_location_limit": args.probe_count,
            "wavcaps_sample_audio_location_limit_per_source": 1,
        },
        "contract": validate_contract(Path(args.contract)),
        "pools": {},
    }
    report["pools"]["audiocaps_v2_aac"] = run_pool_inspection(
        "audiocaps_v2_aac", inspect_audiocaps, Path(args.audiocaps_root), args.audiocaps_split, args.probe_count
    )
    report["pools"]["wavcaps_no_bbc_aac"] = run_pool_inspection(
        "wavcaps_no_bbc_aac",
        inspect_wavcaps,
        Path(args.wavcaps_root),
        args.metadata_schema_records,
    )
    report["pools"]["clotho_v2_aac"] = run_pool_inspection(
        "clotho_v2_aac", inspect_clotho, Path(args.clotho_root), args.clotho_train_manifest, args.probe_count
    )
    report["pools"]["gigaspeech_l_asr"] = run_pool_inspection(
        "gigaspeech_l_asr",
        inspect_gigaspeech,
        Path(args.gigaspeech_root),
        args.gigaspeech_metadata,
        args.probe_count,
    )

    report["blocking_issues"] = build_blocking_issues(report)
    report["inspection_passed"] = not report["blocking_issues"]
    output_report = Path(args.output_report)
    write_json_atomic(output_report, report)

    for pool_name, pool_report in report["pools"].items():
        if "inspection_error" in pool_report:
            summary = f"inspection_error={pool_report['inspection_error']}"
        elif pool_name == "audiocaps_v2_aac":
            summary = f"metadata_rows={pool_report['valid_metadata_row_count']}"
        elif pool_name == "wavcaps_no_bbc_aac":
            summary = f"sources={pool_report['discovered_canonical_sources']}"
        elif pool_name == "clotho_v2_aac":
            summary = f"source_records={pool_report['source_record_count']} grouped_audio={pool_report['grouped_audio_count']}"
        else:
            summary = (
                f"L_segments={pool_report['l_segment_count']} "
                f"metadata_hours={pool_report['l_raw_duration_hours_from_metadata']:.3f}"
            )
        print(f"[summary] pool={pool_name} {summary}", flush=True)
    print(f"[inspect] output_report={output_report}", flush=True)
    print(f"[inspect] blocking_issues={json.dumps(report['blocking_issues'], ensure_ascii=False)}", flush=True)
    if report["blocking_issues"]:
        raise SystemExit("Metadata inspection found blocking issues; inspect the saved report.")
    print("========== HUGINN WHISPER DYNAMIC90S METADATA INSPECT PASSED ==========", flush=True)


if __name__ == "__main__":
    main()
