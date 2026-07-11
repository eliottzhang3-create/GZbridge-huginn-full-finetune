from __future__ import annotations

import argparse
import io
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path

from acavcaps_common import (
    DEFAULT_ACAVCAPS_ROOT,
    collect_schema_summary,
    iter_tar_records,
    list_selected_tar_files,
    parse_category_limits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect ACAVCAPS tar shards, json schema, and flac decode support.")
    parser.add_argument("--dataset_root", default=str(DEFAULT_ACAVCAPS_ROOT))
    parser.add_argument("--category_limits", default=None, help="Comma-separated form, e.g. 00A=12,0M0=8")
    parser.add_argument("--samples_per_tar", type=int, default=32)
    parser.add_argument("--preview_count", type=int, default=3)
    return parser.parse_args()


def print_header(title: str):
    print(f"========== {title} ==========")


def try_import_audio_backends():
    print_header("AUDIO BACKENDS")
    backends = {}
    for module_name in ("soundfile", "torchaudio"):
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"[backend] {module_name}=OK version={version}")
            backends[module_name] = module
        except Exception as exc:
            print(f"[backend] {module_name}=FAIL {type(exc).__name__}: {exc}")
    return backends


def try_decode_first_audio(selected_tar_files: list[tuple[str, Path]], backends: dict[str, object]):
    print_header("DECODE TEST")
    if not selected_tar_files:
        print("[decode] no selected tar files")
        return

    category, tar_path = selected_tar_files[0]
    records = iter_tar_records(tar_path, samples_per_tar=1)
    if not records:
        print(f"[decode] no records found in {tar_path}")
        return

    audio_member = records[0]["audio_member"]
    import tarfile

    with tarfile.open(tar_path, mode="r:*") as tar_obj:
        extracted = tar_obj.extractfile(audio_member)
        if extracted is None:
            print(f"[decode] missing audio member {audio_member}")
            return
        audio_bytes = extracted.read()

    print(f"[decode] category={category} tar={tar_path}")
    print(f"[decode] audio_member={audio_member} bytes={len(audio_bytes)}")

    if "soundfile" in backends:
        try:
            import soundfile as sf

            audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
            shape = getattr(audio, "shape", None)
            print(f"[decode] soundfile=OK sr={sr} shape={shape}")
        except Exception as exc:
            print(f"[decode] soundfile=FAIL {type(exc).__name__}: {exc}")

    if "torchaudio" in backends:
        try:
            import torchaudio

            waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
            print(f"[decode] torchaudio=OK sr={sr} shape={tuple(waveform.shape)}")
        except Exception as exc:
            print(f"[decode] torchaudio=FAIL {type(exc).__name__}: {exc}")


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    category_limits = parse_category_limits(args.category_limits)

    print_header("ENV")
    print(f"python={sys.version.split()[0]}")
    print(f"platform={platform.platform()}")
    print(f"dataset_root={dataset_root}")
    print(f"category_limits={category_limits}")
    print(f"samples_per_tar={args.samples_per_tar}")

    selected_tar_files = list_selected_tar_files(dataset_root, category_limits)
    print_header("SELECTED TARS")
    per_category = defaultdict(list)
    for category, tar_path in selected_tar_files:
        per_category[category].append(tar_path)
    for category, tar_paths in per_category.items():
        print(f"[selected] category={category} tar_count={len(tar_paths)} first_tar={tar_paths[0].name}")

    backends = try_import_audio_backends()
    try_decode_first_audio(selected_tar_files, backends)

    print_header("SCHEMA SUMMARY")
    total_records = 0
    category_record_counts: Counter[str] = Counter()
    merged_key_counter: Counter[str] = Counter()
    merged_list_len_counter: Counter[str] = Counter()
    merged_value_type_counter: Counter[str] = Counter()
    preview_printed = 0

    for category, tar_path in selected_tar_files:
        records = iter_tar_records(tar_path, samples_per_tar=args.samples_per_tar)
        total_records += len(records)
        category_record_counts[category] += len(records)
        schema = collect_schema_summary(records)
        merged_key_counter.update(schema["key_counter"])
        merged_list_len_counter.update(schema["list_len_counter"])
        merged_value_type_counter.update(schema["value_type_counter"])

        print(f"[tar] category={category} tar={tar_path.name} sampled_records={len(records)}")
        if records and preview_printed < args.preview_count:
            preview = records[0]
            print(f"[preview] tar={tar_path.name} json_member={preview['json_member']}")
            print(f"[preview] payload={preview['payload']}")
            preview_printed += 1

    print(f"[summary] total_sampled_records={total_records}")
    for category, count in category_record_counts.items():
        print(f"[summary] category={category} sampled_records={count}")

    print_header("JSON KEYS")
    for key, count in merged_key_counter.most_common():
        print(f"[json-key] {key} count={count}")

    print_header("JSON VALUE TYPES")
    for key, count in merged_value_type_counter.most_common():
        print(f"[json-type] {key} count={count}")

    print_header("JSON LIST LENGTHS")
    for key, count in merged_list_len_counter.most_common():
        print(f"[json-list] {key} count={count}")


if __name__ == "__main__":
    main()
