"""Read-only inventory for the Huginn Whisper dynamic-90s multitask data pools.

This gate does not create training manifests, schedules, audio caches, or converted
audio. It verifies the four source pools and records enough schema/layout evidence
to implement the later canonical manifest builders without guessing remote paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import traceback
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator


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

CANONICAL_SAMPLE_RATE = 16000
CHUNK_SECONDS = 30.0
MAX_AUDIO_SECONDS = 90.0
WHISPER_MAX_FEATURE_FRAMES = 3000
WHISPER_FEATURE_HOP = 160
WHISPER_ENCODER_DOWNSAMPLE = 2
COMPRESSOR_KERNEL = 6
COMPRESSOR_STRIDE = 6
ATOMIC_SCHEMA_VERSION = "huginn_whisper_dynamic90s_atomic_v1"
GIGASPEECH_PUNCTUATION_TAGS = {
    "<COMMA>",
    "<PERIOD>",
    "<QUESTIONMARK>",
    "<EXCLAMATIONPOINT>",
    "<COLON>",
    "<SEMICOLON>",
}


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
    parser.add_argument("--probe_count", type=int, default=8)
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


def exact_dynamic_token_count(duration_seconds: float) -> int:
    """Mirror the current production 16-kHz duration planner without importing Swift."""
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError(f"duration_seconds must be finite and positive, got {duration_seconds!r}")
    total_samples = max(1, int(round(duration_seconds * CANONICAL_SAMPLE_RATE)))
    included_samples = min(total_samples, int(round(MAX_AUDIO_SECONDS * CANONICAL_SAMPLE_RATE)))
    chunk_samples = int(round(CHUNK_SECONDS * CANONICAL_SAMPLE_RATE))
    token_count = 0
    for start in range(0, included_samples, chunk_samples):
        chunk_size = min(start + chunk_samples, included_samples) - start
        feature_length = min(WHISPER_MAX_FEATURE_FRAMES, max(1, chunk_size // WHISPER_FEATURE_HOP))
        encoder_length = feature_length // WHISPER_ENCODER_DOWNSAMPLE
        if encoder_length >= COMPRESSOR_KERNEL:
            token_count += (encoder_length - COMPRESSOR_KERNEL) // COMPRESSOR_STRIDE + 1
    return token_count


def validate_contract(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Data contract not found: {path}")
    contract = json.loads(path.read_text(encoding="utf-8"))
    runtime = contract.get("audio_runtime_contract", {})
    expected = {
        "waveform_dtype": "float32",
        "channels": 1,
        "sample_rate_hz": CANONICAL_SAMPLE_RATE,
        "chunk_seconds": CHUNK_SECONDS,
        "max_included_seconds": MAX_AUDIO_SECONDS,
        "whisper_feature_hop_samples": WHISPER_FEATURE_HOP,
        "whisper_encoder_downsample": WHISPER_ENCODER_DOWNSAMPLE,
        "compressor_kernel": COMPRESSOR_KERNEL,
        "compressor_stride": COMPRESSOR_STRIDE,
        "audio_token_duration_ms": 120,
    }
    mismatches = {
        key: {"expected": value, "actual": runtime.get(key)}
        for key, value in expected.items()
        if runtime.get(key) != value
    }
    required = contract.get("atomic_record_schema", {}).get("required_fields", [])
    if mismatches:
        raise ValueError(f"Data contract/runtime mismatch: {mismatches}")
    if "targets" not in required or "effective_audio_tokens" not in required:
        raise ValueError(f"Atomic record schema is incomplete: required_fields={required}")
    token_landmarks = {
        rendered: exact_dynamic_token_count(seconds)
        for rendered, seconds in (("1s", 1.0), ("30s", 30.0), ("60s", 60.0), ("90s", 90.0), ("120s", 120.0))
    }
    if token_landmarks != {"1s": 8, "30s": 250, "60s": 500, "90s": 750, "120s": 750}:
        raise AssertionError(f"Unexpected dynamic token landmarks: {token_landmarks}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "contract_version": contract.get("contract_version"),
        "atomic_schema_version": ATOMIC_SCHEMA_VERSION,
        "runtime_normalization": {
            "dtype": "float32",
            "channels": 1,
            "sample_rate_hz": CANONICAL_SAMPLE_RATE,
        },
        "dynamic_token_landmarks": token_landmarks,
        "validated": True,
    }


def ffprobe_audio(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return {"path": str(path), "error": "ffprobe is unavailable"}
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_name,sample_rate,channels",
            "-of",
            "json",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return {
            "path": str(path),
            "error": result.stderr.strip() or f"ffprobe exited with {result.returncode}",
        }
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {"path": str(path), "error": f"ffprobe returned invalid JSON: {exc}"}
    return {"path": str(path), "ffprobe": payload}


def probe_paths(paths: Iterable[Path], limit: int) -> list[dict[str, Any]]:
    selected: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        rendered = str(path)
        if rendered in seen:
            continue
        seen.add(rendered)
        selected.append(path)
        if len(selected) >= limit:
            break
    return [ffprobe_audio(path) for path in selected]


def inspect_audiocaps(root: Path, split: str, probe_count: int) -> dict[str, Any]:
    csv_path = root / f"{split}.csv"
    audio_dir = root / split
    if not csv_path.is_file() or not audio_dir.is_dir():
        raise FileNotFoundError(f"AudioCaps layout missing: csv={csv_path} audio_dir={audio_dir}")

    errors: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    audio_paths: Counter[str] = Counter()
    valid_paths: list[Path] = []
    format_counts: Counter[str] = Counter()
    source_rows = 0
    total_duration = 0.0
    total_tokens = 0
    duration_over_90 = 0

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
            path = audio_dir / f"{stem}.wav"
            if not path.is_file():
                errors["missing_wav"] += 1
                limited_append(examples, "missing_wav", f"row={row_number} path={path}")
                continue
            try:
                with wave.open(str(path), "rb") as wav_file:
                    channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
                    sample_rate = wav_file.getframerate()
                    frames = wav_file.getnframes()
                    compression = wav_file.getcomptype()
                if channels <= 0 or sample_rate <= 0 or frames <= 0 or compression != "NONE":
                    raise ValueError(
                        f"channels={channels} rate={sample_rate} frames={frames} compression={compression}"
                    )
            except (OSError, ValueError, wave.Error) as exc:
                errors["unreadable_wav"] += 1
                limited_append(examples, "unreadable_wav", f"row={row_number} path={path} error={exc}")
                continue
            duration = frames / float(sample_rate)
            total_duration += duration
            total_tokens += exact_dynamic_token_count(duration)
            duration_over_90 += int(duration > MAX_AUDIO_SECONDS)
            format_counts[f"wav:pcm{sample_width * 8}:ch{channels}:sr{sample_rate}"] += 1
            audio_paths[str(path)] += 1
            valid_paths.append(path)

    probes = probe_paths(valid_paths, probe_count)
    return {
        "dataset": "AudioCaps-v2",
        "task": "AAC",
        "split_policy": "train",
        "root": str(root),
        "csv_path": str(csv_path),
        "source_row_count": source_rows,
        "valid_record_count": len(valid_paths),
        "excluded_record_count": sum(errors.values()),
        "excluded_record_counts": dict(sorted(errors.items())),
        "excluded_record_examples": examples,
        "unique_audio_path_count": len(audio_paths),
        "duplicate_audio_path_count": sum(count > 1 for count in audio_paths.values()),
        "source_format_counts": dict(sorted(format_counts.items())),
        "raw_duration_hours": total_duration / 3600.0,
        "effective_audio_token_count": total_tokens,
        "duration_over_90_seconds_count": duration_over_90,
        "audio_probes": probes,
        "probe_failure_count": sum("error" in probe for probe in probes),
        "runtime_target_policy": "one caption for each record",
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


def inspect_json_metadata(path: Path, schema_limit: int = 100) -> dict[str, Any]:
    key_counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    record_count = 0
    malformed_count = 0
    first_record: dict[str, Any] | None = None

    def observe(record: Any) -> None:
        nonlocal record_count, malformed_count, first_record
        if not isinstance(record, dict):
            malformed_count += 1
            return
        record_count += 1
        if first_record is None:
            first_record = record
        if record_count <= schema_limit:
            for key, value in record.items():
                key_counts[key] += 1
                if len(examples.setdefault(key, [])) < 2:
                    rendered = json.dumps(value, ensure_ascii=False)
                    examples[key].append(rendered[:200] + ("..." if len(rendered) > 200 else ""))

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    observe(json.loads(line))
                except json.JSONDecodeError:
                    malformed_count += 1
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidate_lists: list[list[Any]] = []
        if isinstance(payload, list):
            candidate_lists.append(payload)
        elif isinstance(payload, dict):
            candidate_lists.extend(value for value in payload.values() if isinstance(value, list))
        for records in candidate_lists:
            for record in records:
                observe(record)

    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "record_count": record_count,
        "malformed_record_count": malformed_count,
        "field_presence_in_schema_sample": dict(sorted(key_counts.items())),
        "field_examples": {key: examples[key] for key in sorted(examples)},
        "first_record": first_record,
    }


def inspect_wavcaps(root: Path, probe_count: int) -> dict[str, Any]:
    audio_root = root / "audio"
    metadata_root = root / "json"
    if not audio_root.is_dir() or not metadata_root.is_dir():
        raise FileNotFoundError(f"WavCaps layout missing: audio={audio_root} metadata={metadata_root}")

    source_reports: dict[str, dict[str, Any]] = {}
    canonical_source_audio_counts: Counter[str] = Counter()
    eligible_probe_paths: list[Path] = []
    for child in sorted(audio_root.iterdir(), key=lambda item: item.name.lower()):
        if not child.is_dir():
            continue
        extension_counts: Counter[str] = Counter()
        file_count = 0
        audio_file_count = 0
        first_paths: list[Path] = []
        for path in child.rglob("*"):
            if not path.is_file():
                continue
            file_count += 1
            extension_counts[path.suffix.lower() or "<none>"] += 1
            if path.suffix.lower() in {".wav", ".flac", ".opus", ".ogg", ".mp3", ".m4a"}:
                audio_file_count += 1
                if len(first_paths) < probe_count:
                    first_paths.append(path)
        canonical = canonical_wavcaps_source(child.name)
        canonical_source_audio_counts[canonical] += audio_file_count
        excluded = canonical == "BBC_Sound_Effects"
        source_reports[child.name] = {
            "canonical_source": canonical,
            "path": str(child),
            "file_count": file_count,
            "audio_file_count": audio_file_count,
            "extension_counts": dict(sorted(extension_counts.items())),
            "training_eligible": not excluded,
            "exclusion_reason": "source-level BBC exclusion" if excluded else None,
        }
        if not excluded:
            eligible_probe_paths.extend(first_paths)

    metadata_paths = sorted(
        path for path in metadata_root.rglob("*") if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}
    )
    metadata_reports = []
    for path in metadata_paths:
        report = inspect_json_metadata(path)
        report["canonical_source"] = canonical_wavcaps_source(path.stem)
        report["training_eligible"] = report["canonical_source"] != "BBC_Sound_Effects"
        metadata_reports.append(report)

    probes = probe_paths(eligible_probe_paths, probe_count)
    discovered_canonical_sources = sorted(
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
        "discovered_canonical_sources": discovered_canonical_sources,
        "canonical_source_audio_file_counts": dict(sorted(canonical_source_audio_counts.items())),
        "required_eligible_sources": ["FreeSound", "AudioSet_SL", "SoundBible"],
        "excluded_sources": ["BBC_Sound_Effects"],
        "audio_probes": probes,
        "probe_failure_count": sum("error" in probe for probe in probes),
        "duration_accounting": "deferred to canonical manifest preparation after metadata/path pairing is confirmed",
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

    missing_paths = [path for rendered, path in path_objects.items() if not path.is_file()]
    multiplicity = Counter(len(targets) for targets in grouped_targets.values())
    probes = probe_paths(
        (path_objects[rendered] for rendered in sorted(path_objects) if path_objects[rendered].is_file()),
        probe_count,
    )
    return {
        "dataset": "Clotho-v2",
        "task": "AAC",
        "root": str(root),
        "manifest_path": str(manifest_path),
        "split_policy": "train only",
        "source_record_count": len(records),
        "valid_source_record_count": sum(len(targets) for targets in grouped_targets.values()),
        "grouped_audio_count": len(grouped_targets),
        "caption_multiplicity_per_audio": {str(key): value for key, value in sorted(multiplicity.items())},
        "invalid_record_counts": dict(sorted(errors.items())),
        "invalid_record_examples": examples,
        "missing_audio_path_count": len(missing_paths),
        "missing_audio_path_examples": [str(path) for path in missing_paths[:10]],
        "split_leakage_indicator_count": split_leakage_indicators,
        "runtime_target_policy": "select exactly one deterministic caption per scheduled training occurrence",
        "atomic_manifest_policy": "one grouped row per audio; do not expand one audio into five independent rows",
        "audio_probes": probes,
        "probe_failure_count": sum("error" in probe for probe in probes),
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
    l_effective_tokens = 0
    duration_over_90 = 0
    zero_token_segments = 0
    invalid_counts: Counter[str] = Counter()
    invalid_examples: dict[str, list[str]] = {}
    placeholder_counts: Counter[str] = Counter()
    l_parent_extension_counts: Counter[str] = Counter()
    sid_counts: Counter[str] = Counter()
    referenced_audio_paths: set[str] = set()
    observed_l_parent_paths: set[str] = set()
    missing_audio_paths: set[str] = set()
    probe_candidates: list[Path] = []
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
        path_exists: bool | None = None
        if audio_path is not None:
            path_exists = audio_path.is_file()

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
            if audio_path is not None:
                rendered_path = str(audio_path)
                referenced_audio_paths.add(rendered_path)
                if rendered_path not in observed_l_parent_paths:
                    observed_l_parent_paths.add(rendered_path)
                    l_parent_extension_counts[audio_path.suffix.lower() or "<none>"] += 1
                if not path_exists:
                    missing_audio_paths.add(rendered_path)
            else:
                invalid_counts["empty_l_parent_audio_path"] += 1
                limited_append(
                    invalid_examples,
                    "empty_l_parent_audio_path",
                    f"audio_index={audio_index} segment_index={segment_index}",
                )
            sid = str(segment.get("sid") or "").strip()
            if sid:
                sid_counts[sid] += 1
            else:
                invalid_counts["empty_l_sid"] += 1
                limited_append(
                    invalid_examples,
                    "empty_l_sid",
                    f"audio_index={audio_index} segment_index={segment_index}",
                )
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
                limited_append(
                    invalid_examples,
                    "invalid_l_segment_time",
                    f"sid={sid} begin={begin} end={end}",
                )
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
            tokens = exact_dynamic_token_count(duration)
            l_effective_tokens += tokens
            zero_token_segments += int(tokens == 0)
            duration_over_90 += int(duration > MAX_AUDIO_SECONDS)
            if audio_path is not None and path_exists and len(probe_candidates) < probe_count:
                probe_candidates.append(audio_path)
            if first_l_segment is None:
                first_l_segment = {
                    "sid": sid,
                    "source": source,
                    "audio_path": str(audio_path) if audio_path is not None else None,
                    "audio_format": audio.get("format"),
                    "begin_time": begin,
                    "end_time": end,
                    "duration_seconds": duration,
                    "effective_audio_tokens": tokens,
                    "subsets": subsets,
                    "text_tn_preview": text_tn[:240],
                }

    encrypted_archives = sum(1 for path in (root / "audio").rglob("*.tgz.aes") if path.is_file())
    duplicate_sid_count = sum(count > 1 for count in sid_counts.values())
    probes = probe_paths(probe_candidates, probe_count)
    non_punctuation_tags = {
        tag: count for tag, count in placeholder_counts.items() if tag not in GIGASPEECH_PUNCTUATION_TAGS
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
        "l_raw_duration_hours": l_duration_seconds / 3600.0,
        "l_effective_audio_token_count": l_effective_tokens,
        "duration_over_90_seconds_count": duration_over_90,
        "zero_audio_token_segment_count": zero_token_segments,
        "unique_l_sid_count": len(sid_counts),
        "duplicate_l_sid_count": duplicate_sid_count,
        "referenced_parent_audio_count": len(referenced_audio_paths),
        "l_parent_audio_extension_counts": dict(sorted(l_parent_extension_counts.items())),
        "missing_parent_audio_count": len(missing_audio_paths),
        "missing_parent_audio_examples": sorted(missing_audio_paths)[:10],
        "invalid_counts": dict(sorted(invalid_counts.items())),
        "invalid_examples": invalid_examples,
        "text_tn_placeholder_counts": dict(sorted(placeholder_counts.items())),
        "non_punctuation_placeholder_counts": dict(sorted(non_punctuation_tags.items())),
        "transcript_cleanup_policy": "map punctuation placeholders, remove non-speech tags deliberately, normalize whitespace; implementation deferred",
        "encrypted_archive_count_ignored": encrypted_archives,
        "audio_decode_policy": "read extracted .opus only; ignore .tgz.aes; decode requested segment to mono 16-kHz float32 in memory",
        "first_l_segment": first_l_segment,
        "audio_probes": probes,
        "probe_failure_count": sum("error" in probe for probe in probes),
    }


def build_blocking_issues(report: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    for pool_name, pool_report in report["pools"].items():
        if "inspection_error" in pool_report:
            issues.append(f"{pool_name} inspection failed: {pool_report['inspection_error']}")

    audiocaps = report["pools"]["audiocaps_v2_aac"]
    if "inspection_error" not in audiocaps and audiocaps["valid_record_count"] == 0:
        issues.append("AudioCaps has no valid train records")
    if "inspection_error" not in audiocaps and audiocaps["probe_failure_count"]:
        issues.append("AudioCaps ffprobe sample failed")

    wavcaps = report["pools"]["wavcaps_no_bbc_aac"]
    if "inspection_error" not in wavcaps:
        discovered = set(wavcaps["discovered_canonical_sources"])
        required = set(wavcaps["required_eligible_sources"])
        if missing := sorted(required - discovered):
            issues.append(f"WavCaps required sources were not discovered: {missing}")
        source_audio_counts = wavcaps["canonical_source_audio_file_counts"]
        empty_sources = sorted(source for source in required if not source_audio_counts.get(source, 0))
        if empty_sources:
            issues.append(f"WavCaps required sources have no discovered audio files: {empty_sources}")
        if "BBC_Sound_Effects" not in discovered:
            issues.append("WavCaps BBC source was not discovered, so source-level exclusion is not yet proven")
        if wavcaps["probe_failure_count"]:
            issues.append("WavCaps ffprobe sample failed")

    clotho = report["pools"]["clotho_v2_aac"]
    if "inspection_error" not in clotho:
        if clotho["grouped_audio_count"] == 0:
            issues.append("Clotho train manifest has no grouped audio")
        if clotho["invalid_record_counts"]:
            issues.append(f"Clotho contains invalid records: {clotho['invalid_record_counts']}")
        if clotho["missing_audio_path_count"]:
            issues.append(f"Clotho has {clotho['missing_audio_path_count']} missing train audio paths")
        if clotho["split_leakage_indicator_count"]:
            issues.append("Clotho train manifest contains val/test/evaluation path indicators")
        if clotho["probe_failure_count"]:
            issues.append("Clotho ffprobe sample failed")

    giga = report["pools"]["gigaspeech_l_asr"]
    if "inspection_error" not in giga:
        if giga["l_segment_count"] == 0:
            issues.append("GigaSpeech has no segment-level {L} records")
        if giga["missing_parent_audio_count"]:
            issues.append(f"GigaSpeech has {giga['missing_parent_audio_count']} missing extracted parent audio files")
        if giga["duplicate_l_sid_count"]:
            issues.append(f"GigaSpeech has {giga['duplicate_l_sid_count']} duplicate L segment IDs")
        unexpected_extensions = sorted(
            extension for extension in giga["l_parent_audio_extension_counts"] if extension != ".opus"
        )
        if unexpected_extensions:
            issues.append(f"GigaSpeech-L parent audio has unexpected extensions: {unexpected_extensions}")
        if giga["invalid_counts"]:
            issues.append(f"GigaSpeech contains invalid L records: {giga['invalid_counts']}")
        if giga["probe_failure_count"]:
            issues.append("GigaSpeech ffprobe sample failed")
    return issues


def run_pool_inspection(
    pool_name: str,
    inspector: Any,
    *args: Any,
) -> dict[str, Any]:
    print(f"[inspect] pool={pool_name}", flush=True)
    try:
        result = inspector(*args)
    except Exception as exc:  # pragma: no cover - depends on remote data layout
        error = f"{type(exc).__name__}: {exc}"
        print(f"[inspect-error] pool={pool_name} error={error}", flush=True)
        return {
            "inspection_error": error,
            "traceback": traceback.format_exc(),
        }
    print(f"[inspect] pool={pool_name} completed=true", flush=True)
    return result


def main() -> None:
    args = parse_args()
    if args.probe_count <= 0:
        raise ValueError(f"probe_count must be positive, got {args.probe_count}")

    print("========== HUGINN WHISPER DYNAMIC90S DATA POOL INSPECT START ==========", flush=True)
    print("[scope] route=Huginn Whisper dynamic-90s only", flush=True)
    print("[scope] source_roots_read_only=true", flush=True)
    print("[scope] creates_training_manifest=false creates_audio_cache=false", flush=True)

    report: dict[str, Any] = {
        "audit": "huginn_whisper_dynamic90s_data_pool_inventory_v1",
        "contract": validate_contract(Path(args.contract)),
        "tools": {
            "ffmpeg": shutil.which("ffmpeg"),
            "ffprobe": shutil.which("ffprobe"),
        },
        "pools": {},
    }
    if report["tools"]["ffprobe"] is None:
        raise RuntimeError("ffprobe is required for source-format probes")

    report["pools"]["audiocaps_v2_aac"] = run_pool_inspection(
        "audiocaps_v2_aac",
        inspect_audiocaps,
        Path(args.audiocaps_root), args.audiocaps_split, args.probe_count
    )
    report["pools"]["wavcaps_no_bbc_aac"] = run_pool_inspection(
        "wavcaps_no_bbc_aac", inspect_wavcaps, Path(args.wavcaps_root), args.probe_count
    )
    report["pools"]["clotho_v2_aac"] = run_pool_inspection(
        "clotho_v2_aac",
        inspect_clotho,
        Path(args.clotho_root), args.clotho_train_manifest, args.probe_count
    )
    report["pools"]["gigaspeech_l_asr"] = run_pool_inspection(
        "gigaspeech_l_asr",
        inspect_gigaspeech,
        Path(args.gigaspeech_root), args.gigaspeech_metadata, args.probe_count
    )

    report["blocking_issues"] = build_blocking_issues(report)
    report["inspection_passed"] = not report["blocking_issues"]
    output_report = Path(args.output_report)
    write_json_atomic(output_report, report)

    for pool_name, pool_report in report["pools"].items():
        if "inspection_error" in pool_report:
            summary = f"inspection_error={pool_report['inspection_error']}"
        elif pool_name == "audiocaps_v2_aac":
            summary = (
                f"valid={pool_report['valid_record_count']} excluded={pool_report['excluded_record_count']} "
                f"hours={pool_report['raw_duration_hours']:.3f}"
            )
        elif pool_name == "wavcaps_no_bbc_aac":
            summary = f"sources={pool_report['discovered_canonical_sources']}"
        elif pool_name == "clotho_v2_aac":
            summary = (
                f"source_records={pool_report['source_record_count']} "
                f"grouped_audio={pool_report['grouped_audio_count']} "
                f"caption_multiplicity={pool_report['caption_multiplicity_per_audio']}"
            )
        else:
            summary = (
                f"L_segments={pool_report['l_segment_count']} "
                f"hours={pool_report['l_raw_duration_hours']:.3f} "
                f"missing_parent_audio={pool_report['missing_parent_audio_count']}"
            )
        print(f"[summary] pool={pool_name} {summary}", flush=True)
    print(f"[inspect] output_report={output_report}", flush=True)
    print(f"[inspect] blocking_issues={json.dumps(report['blocking_issues'], ensure_ascii=False)}", flush=True)
    if report["blocking_issues"]:
        raise SystemExit("Data pool inspection found blocking issues; inspect the report before manifest preparation.")
    print("========== HUGINN WHISPER DYNAMIC90S DATA POOL INSPECT PASSED ==========", flush=True)


if __name__ == "__main__":
    main()
