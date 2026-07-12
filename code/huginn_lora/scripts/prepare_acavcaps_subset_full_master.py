from __future__ import annotations

import argparse
import json
import os
import random
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

from prepare_acavcaps_swift_dataset import extract_texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge verified ACAVCAPS chunk manifests into one globally shuffled metadata-only master JSONL."
    )
    parser.add_argument("--manifest_dir", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--chunk_pattern", default="acavcaps_caption_long_formal_chunk_*.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected_chunk_count", type=int, default=None)
    parser.add_argument("--expected_record_count", type=int, default=None)
    return parser.parse_args()


def get_single_audio(record: dict, manifest_path: Path, line_number: int) -> dict:
    audios = record.get("audios") or []
    if len(audios) != 1 or not isinstance(audios[0], dict):
        raise ValueError(f"{manifest_path}:{line_number} must contain exactly one tar-backed audio entry")
    audio = audios[0]
    required_fields = ("tar_path", "audio_member", "json_member")
    missing_fields = [field for field in required_fields if not audio.get(field)]
    if missing_fields:
        raise ValueError(f"{manifest_path}:{line_number} missing audio fields: {missing_fields}")
    return audio


def get_single_assistant_content(record: dict, manifest_path: Path, line_number: int) -> str:
    assistants = [message.get("content") for message in record.get("messages", []) if message.get("role") == "assistant"]
    if len(assistants) != 1 or not isinstance(assistants[0], str) or not assistants[0].strip():
        raise ValueError(f"{manifest_path}:{line_number} must contain exactly one non-empty assistant caption")
    return assistants[0]


def verify_tar_pairs(tar_path: Path, expected_records: dict[str, dict]) -> None:
    seen_audio_members: set[str] = set()
    seen_json_members: set[str] = set()
    expected_audio_members = {str(record["audio_member"]) for record in expected_records.values()}

    with tarfile.open(tar_path, mode="r|*") as tar_obj:
        for member in tar_obj:
            if member.isfile() and member.name.endswith(".flac"):
                if member.name in expected_audio_members:
                    seen_audio_members.add(member.name)
                continue
            expected_record = expected_records.get(member.name)
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            if expected_record is None:
                continue

            extracted = tar_obj.extractfile(member)
            if extracted is None:
                raise FileNotFoundError(f"Unable to read expected JSON member {member.name} from {tar_path}")
            payload = json.loads(extracted.read().decode("utf-8"))
            text_field = str(expected_record["text_field"])
            expected_caption = str(expected_record["assistant_content"])
            source_captions = extract_texts(payload, text_field, text_index=0, expand_all_texts=False)
            if source_captions != [expected_caption]:
                raise ValueError(
                    f"Caption mismatch for {tar_path}:{member.name}. "
                    f"manifest={expected_caption!r} source={source_captions!r}"
                )
            seen_json_members.add(member.name)

    missing_json = sorted(set(expected_records) - seen_json_members)
    missing_audio = sorted(expected_audio_members - seen_audio_members)
    if missing_json or missing_audio:
        details = []
        if missing_json:
            details.append(f"missing_json={len(missing_json)} first={missing_json[:3]}")
        if missing_audio:
            details.append(f"missing_flac={len(missing_audio)} first={missing_audio[:3]}")
        raise FileNotFoundError(f"Tar pairing verification failed for {tar_path}: {'; '.join(details)}")


def main() -> None:
    args = parse_args()
    manifest_dir = Path(args.manifest_dir)
    output_manifest = Path(args.output_manifest)
    manifest_paths = sorted(manifest_dir.glob(args.chunk_pattern))
    if not manifest_paths:
        raise FileNotFoundError(f"No manifests matching {args.chunk_pattern!r} under {manifest_dir}")
    if args.expected_chunk_count is not None and len(manifest_paths) != args.expected_chunk_count:
        raise ValueError(f"Expected {args.expected_chunk_count} manifests, found {len(manifest_paths)}")

    records: list[dict] = []
    records_by_tar: dict[Path, dict[str, dict]] = defaultdict(dict)
    category_counts: Counter[str] = Counter()
    seen_audio_keys: set[tuple[str, str]] = set()

    print("========== ACAVCAPS MASTER PREP START ==========")
    print(f"[master] manifest_dir={manifest_dir}")
    print(f"[master] output_manifest={output_manifest}")
    print(f"[master] chunk_pattern={args.chunk_pattern}")
    print(f"[master] source_chunk_count={len(manifest_paths)}")
    print(f"[master] shuffle_seed={args.seed}")

    for manifest_path in manifest_paths:
        manifest_records = 0
        with manifest_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                audio = get_single_audio(record, manifest_path, line_number)
                assistant_content = get_single_assistant_content(record, manifest_path, line_number)
                tar_path = Path(str(audio["tar_path"]))
                audio_member = str(audio["audio_member"])
                json_member = str(audio["json_member"])
                expected_audio_member = f"{json_member[:-5]}.flac" if json_member.endswith(".json") else ""
                if audio_member != expected_audio_member:
                    raise ValueError(
                        f"{manifest_path}:{line_number} has mismatched JSON/FLAC names: "
                        f"json_member={json_member} audio_member={audio_member}"
                    )

                audio_key = (str(tar_path), audio_member)
                if audio_key in seen_audio_keys:
                    raise ValueError(f"Duplicate audio sample across source manifests: {tar_path}:{audio_member}")
                seen_audio_keys.add(audio_key)

                metadata = record.get("metadata") or {}
                category = str(metadata.get("source_category") or audio.get("source_category") or "unknown")
                text_field = str(metadata.get("text_field") or "long")
                records_by_tar[tar_path][json_member] = {
                    "audio_member": audio_member,
                    "assistant_content": assistant_content,
                    "text_field": text_field,
                }
                category_counts[category] += 1
                records.append(record)
                manifest_records += 1
        print(f"[master] source_manifest={manifest_path.name} records={manifest_records}")

    if args.expected_record_count is not None and len(records) != args.expected_record_count:
        raise ValueError(f"Expected {args.expected_record_count} records, found {len(records)}")

    print("========== ACAVCAPS TAR/CAPTION PAIR VERIFICATION ==========")
    for tar_path, expected_records in sorted(records_by_tar.items(), key=lambda item: str(item[0])):
        verify_tar_pairs(tar_path, expected_records)
        print(f"[master] verified_tar={tar_path.name} records={len(expected_records)}")

    random.Random(args.seed).shuffle(records)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_manifest = output_manifest.with_name(f"{output_manifest.name}.tmp")
    with tmp_output_manifest.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_output_manifest, output_manifest)

    stats_path = output_manifest.with_suffix(f"{output_manifest.suffix}.stats.json")
    stats = {
        "source_manifest_dir": str(manifest_dir),
        "source_chunk_count": len(manifest_paths),
        "record_count": len(records),
        "unique_tar_count": len(records_by_tar),
        "category_counts": dict(sorted(category_counts.items())),
        "shuffle_seed": args.seed,
        "audio_caption_pair_verification": "passed",
    }
    tmp_stats_path = stats_path.with_name(f"{stats_path.name}.tmp")
    with tmp_stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_stats_path, stats_path)

    print("========== ACAVCAPS MASTER PREP DONE ==========")
    print(f"[master] output_manifest={output_manifest}")
    print(f"[master] stats_path={stats_path}")
    print(f"[master] record_count={len(records)}")
    print(f"[master] unique_tar_count={len(records_by_tar)}")
    print(f"[master] category_counts={dict(sorted(category_counts.items()))}")
    print("[master] audio_caption_pair_verification=passed")


if __name__ == "__main__":
    main()
