from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from acavcaps_common import (
    DEFAULT_ACAVCAPS_ROOT,
    DEFAULT_CAPTION_SYSTEM,
    DEFAULT_CAPTION_USER,
    iter_tar_records,
    list_selected_tar_files,
    maybe_text_list,
    parse_category_limits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ACAVCAPS tar shards into Swift multimodal JSONL.")
    parser.add_argument("--dataset_root", default=str(DEFAULT_ACAVCAPS_ROOT))
    parser.add_argument("--category_limits", default=None, help="Comma-separated form, e.g. 00A=12,0M0=8")
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--text_field", default="long")
    parser.add_argument("--text_index", type=int, default=0)
    parser.add_argument("--expand_all_texts", action="store_true")
    parser.add_argument("--samples_per_tar", type=int, default=None)
    parser.add_argument("--limit_total_records", type=int, default=None)
    return parser.parse_args()


def extract_texts(payload: dict, text_field: str, text_index: int, expand_all_texts: bool) -> list[str]:
    candidates = maybe_text_list(payload.get(text_field))
    if not candidates:
        return []
    if expand_all_texts:
        return candidates
    if text_index < 0 or text_index >= len(candidates):
        return [candidates[0]]
    return [candidates[text_index]]


def build_manifest_record(
    tar_path: Path,
    category: str,
    json_member: str,
    audio_member: str,
    assistant_content: str,
    text_field: str,
    payload: dict,
) -> dict:
    sample_id = Path(json_member).stem
    return {
        "messages": [
            {"role": "system", "content": DEFAULT_CAPTION_SYSTEM},
            {"role": "user", "content": DEFAULT_CAPTION_USER},
            {"role": "assistant", "content": assistant_content},
        ],
        "audios": [
            {
                "tar_path": str(tar_path),
                "audio_member": audio_member,
                "json_member": json_member,
                "audio_format": "flac",
                "source_category": category,
                "sample_id": sample_id,
            }
        ],
        "metadata": {
            "source_category": category,
            "sample_id": sample_id,
            "text_field": text_field,
            "raw_keys": sorted(payload.keys()),
        },
    }


def main():
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_manifest = Path(args.output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    category_limits = parse_category_limits(args.category_limits)
    tmp_output_manifest = output_manifest.with_name(f"{output_manifest.name}.tmp")

    total_manifest_records = 0
    audio_source_records = 0
    selected_tar_files = list_selected_tar_files(dataset_root, category_limits)
    first_manifest_record = None

    print("========== ACAVCAPS PREP START ==========")
    print(f"[manifest] dataset_root={dataset_root}")
    print(f"[manifest] output_manifest={output_manifest}")
    print(f"[manifest] tmp_output_manifest={tmp_output_manifest}")
    print(f"[manifest] category_limits={category_limits}")
    print(f"[manifest] text_field={args.text_field}")
    print(f"[manifest] text_index={args.text_index}")
    print(f"[manifest] expand_all_texts={args.expand_all_texts}")
    print(f"[manifest] samples_per_tar={args.samples_per_tar}")
    print(f"[manifest] limit_total_records={args.limit_total_records}")
    print(f"[manifest] selected_tar_count={len(selected_tar_files)}")

    with tmp_output_manifest.open("w", encoding="utf-8") as f:
        for category, tar_path in selected_tar_files:
            tar_source_records = 0
            tar_manifest_records = 0
            tar_records = iter_tar_records(tar_path, samples_per_tar=args.samples_per_tar)
            for record in tar_records:
                payload = record["payload"]
                texts = extract_texts(payload, args.text_field, args.text_index, args.expand_all_texts)
                if not texts:
                    continue
                audio_source_records += 1
                tar_source_records += 1
                for assistant_content in texts:
                    manifest_record = build_manifest_record(
                        tar_path=tar_path,
                        category=category,
                        json_member=record["json_member"],
                        audio_member=record["audio_member"],
                        assistant_content=assistant_content,
                        text_field=args.text_field,
                        payload=payload,
                    )
                    if first_manifest_record is None:
                        first_manifest_record = manifest_record
                    f.write(json.dumps(manifest_record, ensure_ascii=False) + "\n")
                    total_manifest_records += 1
                    tar_manifest_records += 1
                    if args.limit_total_records is not None and total_manifest_records >= args.limit_total_records:
                        break
                if args.limit_total_records is not None and total_manifest_records >= args.limit_total_records:
                    break
            f.flush()
            print(
                f"[manifest] category={category} tar={tar_path.name} "
                f"audio_source_records={tar_source_records} emitted_records={tar_manifest_records} "
                f"total_manifest_records={total_manifest_records}"
            )
            if args.limit_total_records is not None and total_manifest_records >= args.limit_total_records:
                break

    if total_manifest_records == 0:
        try:
            tmp_output_manifest.unlink()
        except FileNotFoundError:
            pass
        raise ValueError("No manifest records were generated")

    os.replace(tmp_output_manifest, output_manifest)

    print("========== ACAVCAPS SWIFT MANIFEST ==========")
    print(f"[manifest] dataset_root={dataset_root}")
    print(f"[manifest] output_manifest={output_manifest}")
    print(f"[manifest] category_limits={category_limits}")
    print(f"[manifest] text_field={args.text_field}")
    print(f"[manifest] text_index={args.text_index}")
    print(f"[manifest] expand_all_texts={args.expand_all_texts}")
    print(f"[manifest] samples_per_tar={args.samples_per_tar}")
    print(f"[manifest] audio_source_records={audio_source_records}")
    print(f"[manifest] total_manifest_records={total_manifest_records}")
    print("[manifest] first_record=")
    print(json.dumps(first_manifest_record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
