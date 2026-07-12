from __future__ import annotations

import json
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_ACAVCAPS_ROOT = Path("/hpc_stor03/public/shared/data/raa/ACAVCAPS")
DEFAULT_CATEGORY_LIMITS = {
    "00A": 12,
    "0M0": 8,
    "S00": 10,
    "S0A": 12,
    "SMA": 8,
    "0MA": 3,
    "SM0": 3,
}

DEFAULT_CAPTION_SYSTEM = "You are a helpful assistant that can understand audio and describe it."
DEFAULT_CAPTION_USER = "Listen to the audio and describe it."


def get_all_category_limits(dataset_root: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for category_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        tar_count = len(sorted(category_dir.glob("*.tar.gz")))
        if tar_count == 0:
            continue
        result[category_dir.name] = tar_count
    if not result:
        raise ValueError(f"No ACAVCAPS tar directories found under {dataset_root}")
    return result


def parse_category_limits(spec: str | None) -> dict[str, int]:
    if spec is None or not spec.strip():
        return dict(DEFAULT_CATEGORY_LIMITS)
    if spec.strip().lower() in {"all", "__all__", "*"}:
        raise ValueError("Special category limit token requires dataset-aware resolution")

    result: dict[str, int] = {}
    for item in spec.split(","):
        piece = item.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(f"Invalid category limit entry: {piece}")
        category, value = piece.split("=", 1)
        category = category.strip()
        value = value.strip()
        result[category] = int(value)
    if not result:
        raise ValueError("No valid category limits parsed")
    return result


def resolve_category_limits(dataset_root: Path, spec: str | None) -> dict[str, int]:
    if spec is None or not spec.strip():
        return dict(DEFAULT_CATEGORY_LIMITS)
    if spec.strip().lower() in {"all", "__all__", "*"}:
        return get_all_category_limits(dataset_root)
    return parse_category_limits(spec)


def list_selected_tar_files(dataset_root: Path, category_limits: dict[str, int]) -> list[tuple[str, Path]]:
    selected: list[tuple[str, Path]] = []
    for category, limit in category_limits.items():
        category_dir = dataset_root / category
        if not category_dir.is_dir():
            raise FileNotFoundError(f"Category directory not found: {category_dir}")
        tar_files = sorted(category_dir.glob("*.tar.gz"))
        if len(tar_files) < limit:
            raise ValueError(
                f"Category {category} requested {limit} tar files, but only found {len(tar_files)} in {category_dir}"
            )
        for tar_path in tar_files[:limit]:
            selected.append((category, tar_path))
    return selected


def load_json_bytes_from_tar(tar_obj: tarfile.TarFile, member_name: str) -> dict[str, Any]:
    extracted = tar_obj.extractfile(member_name)
    if extracted is None:
        raise FileNotFoundError(f"Missing json member {member_name}")
    payload = extracted.read().decode("utf-8")
    return json.loads(payload)


def build_audio_member_from_json_name(json_member: str) -> str:
    if not json_member.endswith(".json"):
        raise ValueError(f"Expected json member, got {json_member}")
    return f"{json_member[:-5]}.flac"


def iter_tar_records(
    tar_path: Path,
    samples_per_tar: int | None = None,
    verify_audio_pairs: bool = False,
) -> list[dict[str, Any]]:
    if samples_per_tar is not None and samples_per_tar <= 0:
        raise ValueError(f"samples_per_tar must be positive when set, got {samples_per_tar}")
    if verify_audio_pairs and samples_per_tar is not None:
        raise ValueError("verify_audio_pairs requires samples_per_tar=None so the complete tar is scanned")

    # ACAVCAPS shards are gzip-compressed. Sequential mode avoids repeatedly
    # seeking through the compressed stream for JSON members selected by name.
    output: list[dict[str, Any]] = []
    audio_members: set[str] = set()
    with tarfile.open(tar_path, mode="r|*") as tar_obj:
        for member in tar_obj:
            if member.isfile() and member.name.endswith(".flac"):
                if verify_audio_pairs:
                    audio_members.add(member.name)
                continue
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            extracted = tar_obj.extractfile(member)
            if extracted is None:
                raise FileNotFoundError(f"Missing json member {member.name} in {tar_path}")
            payload = json.loads(extracted.read().decode("utf-8"))
            output.append(
                {
                    "json_member": member.name,
                    "audio_member": build_audio_member_from_json_name(member.name),
                    "payload": payload,
                }
            )
            if samples_per_tar is not None and len(output) >= samples_per_tar:
                break

    if verify_audio_pairs:
        missing_audio_members = [
            record["audio_member"] for record in output if record["audio_member"] not in audio_members
        ]
        if missing_audio_members:
            preview = ", ".join(missing_audio_members[:3])
            raise FileNotFoundError(
                f"Tar {tar_path} has {len(missing_audio_members)} JSON records without matching FLAC members. "
                f"Examples: {preview}"
            )
    return output


def maybe_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        output: list[str] = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    output.append(text)
        return output
    return []


def collect_schema_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    key_counter: Counter[str] = Counter()
    list_len_counter: Counter[str] = Counter()
    value_type_counter: Counter[str] = Counter()

    for record in records:
        payload = record["payload"]
        for key, value in payload.items():
            key_counter[key] += 1
            value_type_counter[f"{key}:{type(value).__name__}"] += 1
            if isinstance(value, list):
                list_len_counter[f"{key}:len={len(value)}"] += 1

    return {
        "key_counter": key_counter,
        "list_len_counter": list_len_counter,
        "value_type_counter": value_type_counter,
    }
