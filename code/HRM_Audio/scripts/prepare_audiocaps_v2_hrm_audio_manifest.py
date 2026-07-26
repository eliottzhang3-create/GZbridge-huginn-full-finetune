#!/usr/bin/env python3
"""Derive and audit the full HRM-audio AudioCaps-v2 manifest.

The source Huginn manifest is never modified.  The only record transformation is
removing its generic system message so the result matches the verified HRM direct
template contract: user -> assistant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import wave
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_RECORD_COUNT = 89_658
EXPECTED_SYSTEM_PROMPT = "You are a helpful assistant that can understand audio and describe it."
EXPECTED_USER_PROMPT = "Listen to the audio and describe it."
EXPECTED_SOURCE_ROLES = ("system", "user", "assistant")
EXPECTED_OUTPUT_ROLES = ("user", "assistant")
EXPECTED_TOP_LEVEL_KEYS = {"messages", "audios", "metadata"}
EXPECTED_WAV = {
    "channels": 1,
    "sample_width_bytes": 2,
    "sample_rate": 32_000,
    "compression": "NONE",
}
ACTIVE_STAGE = "initializing"
TEMPORARY_PATHS: list[Path] = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-stats", type=Path)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--expected-record-count", type=int, default=EXPECTED_RECORD_COUNT)
    parser.add_argument("--expected-system-prompt", default=EXPECTED_SYSTEM_PROMPT)
    parser.add_argument("--expected-user-prompt", default=EXPECTED_USER_PROMPT)
    parser.add_argument(
        "--skip-wav-header-verification",
        action="store_true",
        help="Trust the source stats instead of reopening every WAV. Not used by the formal preparation launcher.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing non-identical derived manifest. Identical outputs are accepted without this flag.",
    )
    return parser.parse_args()


def cleanup_temporary_paths() -> None:
    for path in TEMPORARY_PATHS:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def on_signal(signum, _frame) -> None:
    signal_name = signal.Signals(signum).name
    print(f"[hrm-manifest] received_signal={signal_name} active_stage={ACTIVE_STAGE}", flush=True)
    cleanup_temporary_paths()
    raise SystemExit(128 + signum)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def counter_to_json(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def require_nonempty_string(value: Any, *, field: str, line_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"Line {line_number} has an empty/non-string {field}: {value!r}")
    return value


def verify_wav(path: Path, *, line_number: int) -> tuple[int, int, int, str, int]:
    try:
        with wave.open(str(path), "rb") as handle:
            actual = {
                "channels": handle.getnchannels(),
                "sample_width_bytes": handle.getsampwidth(),
                "sample_rate": handle.getframerate(),
                "compression": handle.getcomptype(),
            }
            frame_count = handle.getnframes()
    except (OSError, EOFError, wave.Error) as exc:
        raise RuntimeError(f"Line {line_number} has an unreadable WAV: {path}: {exc}") from exc
    mismatches = {
        key: {"expected": expected, "actual": actual[key]}
        for key, expected in EXPECTED_WAV.items()
        if actual[key] != expected
    }
    if mismatches or frame_count <= 0:
        raise RuntimeError(
            f"Line {line_number} WAV contract mismatch: path={path} metadata={actual} "
            f"frames={frame_count} mismatches={mismatches}"
        )
    return (
        int(actual["channels"]),
        int(actual["sample_width_bytes"]),
        int(actual["sample_rate"]),
        str(actual["compression"]),
        int(frame_count),
    )


def load_and_validate_source_stats(path: Path, *, expected_record_count: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Source stats are missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "dataset": "audiocaps_v2",
        "split": "train",
        "record_count": expected_record_count,
        "audio_path_verification": "passed",
        "wav_readability_verification": "passed",
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Source AudioCaps stats mismatch: {mismatches}")
    unique_count = payload.get("unique_audio_path_count")
    if unique_count is not None and int(unique_count) != expected_record_count:
        raise RuntimeError(
            f"Source stats do not describe unique audio records: unique={unique_count} expected={expected_record_count}"
        )
    duplicate_count = payload.get("duplicate_audio_path_count")
    if duplicate_count is not None and int(duplicate_count) != 0:
        raise RuntimeError(f"Source stats contain duplicate audio paths: {duplicate_count}")
    return payload


def transform_record(
    record: Any,
    *,
    line_number: int,
    expected_system_prompt: str,
    expected_user_prompt: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(record, dict):
        raise RuntimeError(f"Line {line_number} is not a JSON object: {type(record)}")
    actual_top_level_keys = set(record)
    if actual_top_level_keys != EXPECTED_TOP_LEVEL_KEYS:
        raise RuntimeError(
            f"Line {line_number} top-level schema mismatch: "
            f"expected={sorted(EXPECTED_TOP_LEVEL_KEYS)} actual={sorted(actual_top_level_keys)}"
        )
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise RuntimeError(f"Line {line_number} must contain exactly three source messages: {messages}")
    roles = tuple(message.get("role") if isinstance(message, dict) else None for message in messages)
    if roles != EXPECTED_SOURCE_ROLES:
        raise RuntimeError(
            f"Line {line_number} source role sequence mismatch: expected={EXPECTED_SOURCE_ROLES} actual={roles}"
        )
    system_prompt = require_nonempty_string(messages[0].get("content"), field="system content", line_number=line_number)
    if system_prompt != expected_system_prompt:
        raise RuntimeError(
            f"Line {line_number} has an unexpected system prompt: expected={expected_system_prompt!r} "
            f"actual={system_prompt!r}"
        )
    user_prompt = require_nonempty_string(messages[1].get("content"), field="user content", line_number=line_number)
    if user_prompt != expected_user_prompt:
        raise RuntimeError(
            f"Line {line_number} has an unexpected user prompt: expected={expected_user_prompt!r} "
            f"actual={user_prompt!r}"
        )
    caption = require_nonempty_string(messages[2].get("content"), field="assistant content", line_number=line_number)

    audios = record.get("audios")
    if not isinstance(audios, list) or len(audios) != 1:
        raise RuntimeError(f"Line {line_number} must contain exactly one audio path: {audios}")
    audio_value = require_nonempty_string(audios[0], field="audio path", line_number=line_number)
    audio_path = Path(audio_value).expanduser()
    if not audio_path.is_absolute():
        raise RuntimeError(f"Line {line_number} audio path must be absolute: {audio_value}")
    if not audio_path.is_file():
        raise FileNotFoundError(f"Line {line_number} audio path is missing: {audio_path}")

    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError(f"Line {line_number} metadata must be an object: {metadata}")
    expected_metadata = {"dataset": "audiocaps_v2", "split": "train"}
    metadata_mismatches = {
        key: {"expected": expected, "actual": metadata.get(key)}
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    }
    if metadata_mismatches:
        raise RuntimeError(f"Line {line_number} metadata mismatch: {metadata_mismatches}")
    sample_id = require_nonempty_string(metadata.get("sample_id"), field="metadata.sample_id", line_number=line_number)

    output = dict(record)
    output["messages"] = [dict(messages[1]), dict(messages[2])]
    output_roles = tuple(message.get("role") for message in output["messages"])
    if output_roles != EXPECTED_OUTPUT_ROLES:
        raise RuntimeError(f"Line {line_number} output role sequence mismatch: {output_roles}")
    source_non_messages = {key: value for key, value in record.items() if key != "messages"}
    output_non_messages = {key: value for key, value in output.items() if key != "messages"}
    if source_non_messages != output_non_messages:
        raise RuntimeError(f"Line {line_number} transformation changed non-message fields")
    if output["messages"][0] != messages[1] or output["messages"][1] != messages[2]:
        raise RuntimeError(f"Line {line_number} transformation changed user/assistant messages")
    return output, {
        "audio_path": str(audio_path),
        "sample_id": sample_id,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "caption": caption,
        "top_level_keys": tuple(sorted(record)),
    }


def main() -> None:
    global ACTIVE_STAGE
    args = parse_args()
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    if args.expected_record_count <= 0:
        raise ValueError(f"expected-record-count must be positive, got {args.expected_record_count}")

    source_manifest = args.source_manifest.expanduser().resolve()
    source_stats_path = (
        args.source_stats.expanduser().resolve()
        if args.source_stats is not None
        else source_manifest.with_suffix(f"{source_manifest.suffix}.stats.json")
    )
    output_manifest = args.output_manifest.expanduser().resolve()
    output_stats_path = output_manifest.with_suffix(f"{output_manifest.suffix}.stats.json")
    if not source_manifest.is_file():
        raise FileNotFoundError(f"Source manifest is missing: {source_manifest}")
    if source_manifest == output_manifest:
        raise ValueError("Source and output manifests must be different paths")
    source_stats = load_and_validate_source_stats(
        source_stats_path,
        expected_record_count=args.expected_record_count,
    )
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    nonce = f"{os.getpid()}"
    temporary_manifest = output_manifest.with_name(f".{output_manifest.name}.{nonce}.tmp")
    temporary_stats = output_stats_path.with_name(f".{output_stats_path.name}.{nonce}.tmp")
    TEMPORARY_PATHS.extend((temporary_manifest, temporary_stats))

    print("========== PREPARE HRM AUDIO AUDIOCAPS V2 MANIFEST ==========", flush=True)
    print(f"[hrm-manifest] source={source_manifest}", flush=True)
    print(f"[hrm-manifest] source_stats={source_stats_path}", flush=True)
    print(f"[hrm-manifest] output={output_manifest}", flush=True)
    print(f"[hrm-manifest] output_stats={output_stats_path}", flush=True)
    print(f"[hrm-manifest] expected_record_count={args.expected_record_count}", flush=True)
    print(f"[hrm-manifest] verify_wav_headers={not args.skip_wav_header_verification}", flush=True)
    print("[hrm-manifest] transformation=remove_generic_system_message_only", flush=True)

    ACTIVE_STAGE = "hashing_source"
    source_stat_before = source_manifest.stat()
    source_stats_stat_before = source_stats_path.stat()
    source_sha256_before = sha256_file(source_manifest)
    source_stats_sha256_before = sha256_file(source_stats_path)

    record_count = 0
    source_role_counts: Counter[tuple[str, ...]] = Counter()
    output_role_counts: Counter[tuple[str, ...]] = Counter()
    system_prompt_counts: Counter[str] = Counter()
    user_prompt_counts: Counter[str] = Counter()
    top_level_key_counts: Counter[tuple[str, ...]] = Counter()
    wav_format_counts: Counter[tuple[int, int, int, str]] = Counter()
    audio_paths: set[str] = set()
    sample_ids: set[str] = set()
    duplicate_audio_examples: list[str] = []
    duplicate_sample_examples: list[str] = []
    total_caption_characters = 0
    min_caption_characters: int | None = None
    max_caption_characters = 0
    first_output_record: dict[str, Any] | None = None

    ACTIVE_STAGE = "streaming_transform_and_audit"
    try:
        with source_manifest.open("r", encoding="utf-8") as source_handle, temporary_manifest.open(
            "w", encoding="utf-8", newline="\n"
        ) as output_handle:
            for line_number, line in enumerate(source_handle, start=1):
                if not line.strip():
                    raise RuntimeError(f"Source manifest contains a blank line at {line_number}")
                try:
                    source_record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Invalid JSON at source line {line_number}: {exc}") from exc
                output_record, audit = transform_record(
                    source_record,
                    line_number=line_number,
                    expected_system_prompt=args.expected_system_prompt,
                    expected_user_prompt=args.expected_user_prompt,
                )
                audio_path = audit["audio_path"]
                sample_id = audit["sample_id"]
                if audio_path in audio_paths:
                    if len(duplicate_audio_examples) < 20:
                        duplicate_audio_examples.append(audio_path)
                else:
                    audio_paths.add(audio_path)
                if sample_id in sample_ids:
                    if len(duplicate_sample_examples) < 20:
                        duplicate_sample_examples.append(sample_id)
                else:
                    sample_ids.add(sample_id)
                if not args.skip_wav_header_verification:
                    channels, width, rate, compression, _ = verify_wav(
                        Path(audio_path),
                        line_number=line_number,
                    )
                    wav_format_counts[(channels, width, rate, compression)] += 1

                source_role_counts[EXPECTED_SOURCE_ROLES] += 1
                output_role_counts[EXPECTED_OUTPUT_ROLES] += 1
                system_prompt_counts[audit["system_prompt"]] += 1
                user_prompt_counts[audit["user_prompt"]] += 1
                top_level_key_counts[audit["top_level_keys"]] += 1
                caption_length = len(audit["caption"])
                total_caption_characters += caption_length
                min_caption_characters = (
                    caption_length
                    if min_caption_characters is None
                    else min(min_caption_characters, caption_length)
                )
                max_caption_characters = max(max_caption_characters, caption_length)
                record_count += 1
                if first_output_record is None:
                    first_output_record = output_record
                output_handle.write(json.dumps(output_record, ensure_ascii=False, separators=(",", ":")) + "\n")
            output_handle.flush()
            os.fsync(output_handle.fileno())

        if record_count != args.expected_record_count:
            raise RuntimeError(
                f"Derived record count mismatch: expected={args.expected_record_count} actual={record_count}"
            )
        if duplicate_audio_examples or len(audio_paths) != record_count:
            raise RuntimeError(
                f"Derived manifest contains duplicate audio paths: unique={len(audio_paths)} records={record_count} "
                f"examples={duplicate_audio_examples}"
            )
        if duplicate_sample_examples or len(sample_ids) != record_count:
            raise RuntimeError(
                f"Derived manifest contains duplicate sample IDs: unique={len(sample_ids)} records={record_count} "
                f"examples={duplicate_sample_examples}"
            )
        if first_output_record is None or min_caption_characters is None:
            raise RuntimeError("Derived manifest is unexpectedly empty")

        ACTIVE_STAGE = "hashing_and_committing_output"
        output_sha256 = sha256_file(temporary_manifest)
        output_bytes = temporary_manifest.stat().st_size
        source_stat_after = source_manifest.stat()
        source_stats_stat_after = source_stats_path.stat()
        source_sha256_after = sha256_file(source_manifest)
        source_stats_sha256_after = sha256_file(source_stats_path)
        source_unchanged = (
            source_sha256_before == source_sha256_after
            and source_stat_before.st_size == source_stat_after.st_size
            and source_stat_before.st_mtime_ns == source_stat_after.st_mtime_ns
        )
        if not source_unchanged:
            raise RuntimeError("Source manifest changed while the HRM view was being generated")
        source_stats_unchanged = (
            source_stats_sha256_before == source_stats_sha256_after
            and source_stats_stat_before.st_size == source_stats_stat_after.st_size
            and source_stats_stat_before.st_mtime_ns == source_stats_stat_after.st_mtime_ns
        )
        if not source_stats_unchanged:
            raise RuntimeError("Source stats changed while the HRM view was being generated")

        stats = {
            "schema_version": 1,
            "dataset": "audiocaps_v2",
            "split": "train",
            "route": "hrm_text_audio_whisper",
            "template_contract": "hrm_text_audio_direct_user_assistant",
            "transformation": "remove_generic_system_message_only",
            "source_manifest": str(source_manifest),
            "source_stats": str(source_stats_path),
            "source_manifest_bytes": int(source_stat_before.st_size),
            "source_manifest_sha256": source_sha256_before,
            "source_stats_sha256": source_stats_sha256_before,
            "source_stats_record_count": int(source_stats["record_count"]),
            "output_manifest": str(output_manifest),
            "output_manifest_bytes": output_bytes,
            "output_manifest_sha256": output_sha256,
            "record_count": record_count,
            "expected_record_count": args.expected_record_count,
            "unique_audio_path_count": len(audio_paths),
            "duplicate_audio_path_count": record_count - len(audio_paths),
            "unique_sample_id_count": len(sample_ids),
            "duplicate_sample_id_count": record_count - len(sample_ids),
            "source_role_sequences": counter_to_json(source_role_counts),
            "output_role_sequences": counter_to_json(output_role_counts),
            "source_system_prompt_counts": counter_to_json(system_prompt_counts),
            "user_prompt_counts": counter_to_json(user_prompt_counts),
            "top_level_key_sets": counter_to_json(top_level_key_counts),
            "caption_character_length": {
                "min": min_caption_characters,
                "max": max_caption_characters,
                "mean": total_caption_characters / record_count,
            },
            "audio_path_verification": "passed",
            "wav_header_verification": "skipped" if args.skip_wav_header_verification else "passed",
            "wav_format_counts": counter_to_json(wav_format_counts),
            "source_manifest_unchanged": True,
            "source_stats_unchanged": True,
            "all_non_message_fields_preserved": True,
            "user_messages_preserved": True,
            "assistant_messages_preserved": True,
            "audio_paths_preserved": True,
            "metadata_preserved": True,
            "removed_system_message_count": record_count,
            "first_output_record": first_output_record,
        }
        with temporary_stats.open("w", encoding="utf-8", newline="\n") as stats_handle:
            json.dump(stats, stats_handle, ensure_ascii=False, indent=2)
            stats_handle.write("\n")
            stats_handle.flush()
            os.fsync(stats_handle.fileno())

        existing_identical = False
        if output_manifest.exists() or output_stats_path.exists():
            if not output_manifest.is_file() or not output_stats_path.is_file():
                if not args.overwrite:
                    raise RuntimeError(
                        f"Existing output is incomplete: manifest={output_manifest.exists()} "
                        f"stats={output_stats_path.exists()}; inspect it or rerun with --overwrite"
                    )
            else:
                existing_stats = json.loads(output_stats_path.read_text(encoding="utf-8"))
                existing_identical = (
                    sha256_file(output_manifest) == output_sha256
                    and existing_stats.get("output_manifest_sha256") == output_sha256
                    and int(existing_stats.get("record_count", -1)) == record_count
                    and existing_stats.get("source_manifest_sha256") == source_sha256_before
                    and existing_stats.get("source_stats_sha256") == source_stats_sha256_before
                )
            if not existing_identical and not args.overwrite:
                raise FileExistsError(
                    "A non-identical HRM AudioCaps manifest already exists; inspect it or rerun with --overwrite: "
                    f"manifest={output_manifest} stats={output_stats_path}"
                )
        if existing_identical:
            temporary_manifest.unlink()
            temporary_stats.unlink()
            commit_status = "unchanged_identical_output"
        else:
            os.replace(temporary_manifest, output_manifest)
            os.replace(temporary_stats, output_stats_path)
            commit_status = "replaced" if args.overwrite else "created"
        ACTIVE_STAGE = "complete"
    finally:
        cleanup_temporary_paths()

    print("========== HRM AUDIO AUDIOCAPS V2 MANIFEST READY ==========", flush=True)
    print(f"[hrm-manifest] status=OK commit={commit_status}", flush=True)
    print(f"[hrm-manifest] record_count={record_count}", flush=True)
    print(f"[hrm-manifest] unique_audio_path_count={len(audio_paths)}", flush=True)
    print(f"[hrm-manifest] unique_sample_id_count={len(sample_ids)}", flush=True)
    print(f"[hrm-manifest] source_sha256={source_sha256_before}", flush=True)
    print(f"[hrm-manifest] source_stats_sha256={source_stats_sha256_before}", flush=True)
    print(f"[hrm-manifest] output_sha256={output_sha256}", flush=True)
    print(f"[hrm-manifest] output={output_manifest}", flush=True)
    print(f"[hrm-manifest] stats={output_stats_path}", flush=True)
    print(f"[hrm-manifest] first_record={json.dumps(first_output_record, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
