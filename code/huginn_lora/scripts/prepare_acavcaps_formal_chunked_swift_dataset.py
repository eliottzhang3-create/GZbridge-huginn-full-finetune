from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from acavcaps_common import (
    DEFAULT_ACAVCAPS_ROOT,
    iter_tar_records,
    list_selected_tar_files,
    parse_category_limits,
)
from prepare_acavcaps_swift_dataset import build_manifest_record, extract_texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ACAVCAPS formal manifests split into tar chunks.")
    parser.add_argument("--dataset_root", default=str(DEFAULT_ACAVCAPS_ROOT))
    parser.add_argument("--category_limits", default=None, help="Comma-separated form, e.g. 00A=12,0M0=8")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--text_field", default="long")
    parser.add_argument("--text_index", type=int, default=0)
    parser.add_argument("--expand_all_texts", action="store_true")
    parser.add_argument("--samples_per_tar", type=int, default=None)
    parser.add_argument("--chunk_size_tars", type=int, default=8)
    parser.add_argument("--start_chunk", type=int, default=None)
    parser.add_argument("--end_chunk", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def chunk_list(items: list[tuple[str, Path]], chunk_size: int) -> list[list[tuple[str, Path]]]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size_tars must be positive, got {chunk_size}")
    return [items[idx:idx + chunk_size] for idx in range(0, len(items), chunk_size)]


def summarize_existing_manifest(manifest_path: Path) -> tuple[int, int, dict | None]:
    manifest_records = 0
    audio_source_ids: set[str] = set()
    first_record = None
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if first_record is None:
                first_record = payload
            manifest_records += 1
            sample_id = payload.get("metadata", {}).get("sample_id")
            if sample_id:
                audio_source_ids.add(sample_id)
    return manifest_records, len(audio_source_ids), first_record


def main():
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    category_limits = parse_category_limits(args.category_limits)
    selected_tar_files = list_selected_tar_files(dataset_root, category_limits)
    chunks = chunk_list(selected_tar_files, args.chunk_size_tars)
    if not chunks:
        raise ValueError("No chunks were generated from the selected tar files")

    start_chunk = 0 if args.start_chunk is None else args.start_chunk
    end_chunk = len(chunks) - 1 if args.end_chunk is None else args.end_chunk
    if start_chunk < 0 or start_chunk >= len(chunks):
        raise ValueError(f"start_chunk out of range: {start_chunk}, valid range is 0..{len(chunks) - 1}")
    if end_chunk < 0 or end_chunk >= len(chunks):
        raise ValueError(f"end_chunk out of range: {end_chunk}, valid range is 0..{len(chunks) - 1}")
    if start_chunk > end_chunk:
        raise ValueError(f"start_chunk ({start_chunk}) must be <= end_chunk ({end_chunk})")

    print("========== ACAVCAPS FORMAL CHUNK PREP START ==========")
    print(f"[chunk] dataset_root={dataset_root}")
    print(f"[chunk] output_dir={output_dir}")
    print(f"[chunk] category_limits={category_limits}")
    print(f"[chunk] text_field={args.text_field}")
    print(f"[chunk] text_index={args.text_index}")
    print(f"[chunk] expand_all_texts={args.expand_all_texts}")
    print(f"[chunk] samples_per_tar={args.samples_per_tar}")
    print(f"[chunk] chunk_size_tars={args.chunk_size_tars}")
    print(f"[chunk] selected_tar_count={len(selected_tar_files)}")
    print(f"[chunk] chunk_count={len(chunks)}")
    print(f"[chunk] start_chunk={start_chunk}")
    print(f"[chunk] end_chunk={end_chunk}")
    print(f"[chunk] skip_existing={args.skip_existing}")

    index_records: list[dict] = []
    total_manifest_records = 0
    total_audio_source_records = 0

    for chunk_idx in range(start_chunk, end_chunk + 1):
        chunk_tar_files = chunks[chunk_idx]
        chunk_name = f"acavcaps_caption_long_formal_chunk_{chunk_idx:03d}.jsonl"
        output_manifest = output_dir / chunk_name
        tmp_output_manifest = output_manifest.with_name(f"{output_manifest.name}.tmp")
        categories_in_chunk = sorted({category for category, _ in chunk_tar_files})

        print(
            f"[chunk] idx={chunk_idx:03d} start "
            f"manifest_path={output_manifest} tar_count={len(chunk_tar_files)} "
            f"categories={categories_in_chunk}"
        )

        if args.skip_existing and output_manifest.exists() and output_manifest.stat().st_size > 0:
            chunk_manifest_records, chunk_audio_source_records, first_manifest_record = summarize_existing_manifest(
                output_manifest
            )
            total_manifest_records += chunk_manifest_records
            total_audio_source_records += chunk_audio_source_records
            index_records.append(
                {
                    "chunk_index": chunk_idx,
                    "manifest_path": str(output_manifest),
                    "tar_count": len(chunk_tar_files),
                    "categories": categories_in_chunk,
                    "audio_source_records": chunk_audio_source_records,
                    "manifest_records": chunk_manifest_records,
                    "first_record": first_manifest_record,
                    "skipped_existing": True,
                }
            )
            print(
                f"[chunk] idx={chunk_idx:03d} skip_existing manifest_path={output_manifest} "
                f"manifest_records={chunk_manifest_records} audio_source_records={chunk_audio_source_records}"
            )
            continue

        chunk_manifest_records = 0
        chunk_audio_source_records = 0
        first_manifest_record = None
        with tmp_output_manifest.open("w", encoding="utf-8") as f:
            for category, tar_path in chunk_tar_files:
                tar_source_records = 0
                tar_manifest_records = 0
                print(
                    f"[chunk] idx={chunk_idx:03d} tar_start "
                    f"category={category} tar={tar_path.name}"
                )
                tar_records = iter_tar_records(tar_path, samples_per_tar=args.samples_per_tar)
                for record in tar_records:
                    payload = record["payload"]
                    texts = extract_texts(payload, args.text_field, args.text_index, args.expand_all_texts)
                    if not texts:
                        continue
                    chunk_audio_source_records += 1
                    total_audio_source_records += 1
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
                        chunk_manifest_records += 1
                        total_manifest_records += 1
                        tar_manifest_records += 1
                f.flush()
                print(
                    f"[chunk] idx={chunk_idx:03d} category={category} tar={tar_path.name} "
                    f"audio_source_records={tar_source_records} emitted_records={tar_manifest_records} "
                    f"chunk_manifest_records={chunk_manifest_records}"
                )

        if chunk_manifest_records == 0:
            try:
                tmp_output_manifest.unlink()
            except FileNotFoundError:
                pass
            raise ValueError(f"Chunk {chunk_idx} produced zero records")

        os.replace(tmp_output_manifest, output_manifest)
        index_record = {
            "chunk_index": chunk_idx,
            "manifest_path": str(output_manifest),
            "tar_count": len(chunk_tar_files),
            "categories": categories_in_chunk,
            "audio_source_records": chunk_audio_source_records,
            "manifest_records": chunk_manifest_records,
            "first_record": first_manifest_record,
            "skipped_existing": False,
        }
        index_records.append(index_record)
        print(
            f"[chunk] idx={chunk_idx:03d} manifest_path={output_manifest} "
            f"tar_count={len(chunk_tar_files)} manifest_records={chunk_manifest_records}"
        )

    index_path = output_dir / "acavcaps_caption_long_formal_chunks_index.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index_records, f, ensure_ascii=False, indent=2)

    print("========== ACAVCAPS FORMAL CHUNK PREP DONE ==========")
    print(f"[chunk] output_dir={output_dir}")
    print(f"[chunk] index_path={index_path}")
    print(f"[chunk] chunk_count={len(index_records)}")
    print(f"[chunk] total_audio_source_records={total_audio_source_records}")
    print(f"[chunk] total_manifest_records={total_manifest_records}")


if __name__ == "__main__":
    main()
