#!/usr/bin/env python3
"""Split an existing unique BAT source manifest into N balanced shards.

This step does not reread the original QA JSON and does not touch AudioSet or
RIR files.  It is intentionally separate from deduplication so the number of
available accelerator cards can be chosen later.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

try:
    from .build_bat_unique_manifests import private_output, read_jsonl, split_sources, write_json, write_jsonl
except ImportError:  # Direct ``python path/to/script.py`` execution.
    from build_bat_unique_manifests import private_output, read_jsonl, split_sources, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.shard_count <= 0:
        raise ValueError("--shard-count must be positive")
    private_output(args.output_dir)
    if not args.source_manifest.is_file():
        raise FileNotFoundError(args.source_manifest)

    rows = read_jsonl(args.source_manifest)
    keys = [str(row.get("source_key")) for row in rows]
    if not rows:
        raise ValueError(f"Source manifest is empty: {args.source_manifest}")
    if len(keys) != len(set(keys)):
        raise ValueError("Source manifest contains duplicate source_key values")
    required = {
        "source_key",
        "audio_id",
        "reverb_id",
        "audio_id2",
        "reverb_id2",
        "source_shape",
        "estimated_render_source_count",
    }
    for index, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"Source manifest row {index} is missing fields: {missing}")

    shards = split_sources(rows, args.shard_count)
    shard_dir = args.output_dir / "source_shards"
    report_shards = []
    all_keys: list[str] = []
    for shard_id, shard_rows in enumerate(shards):
        path = shard_dir / f"shard-{shard_id:03d}-of-{args.shard_count:03d}.jsonl"
        count = write_jsonl(path, shard_rows)
        shard_keys = [str(row["source_key"]) for row in shard_rows]
        all_keys.extend(shard_keys)
        report_shards.append(
            {
                "shard_id": shard_id,
                "path": str(path),
                "source_tuple_count": count,
                "estimated_render_source_count": sum(
                    int(row["estimated_render_source_count"]) for row in shard_rows
                ),
                "source_key_sha256": hashlib.sha256(
                    ("\n".join(shard_keys) + "\n").encode("utf-8")
                ).hexdigest(),
            }
        )

    report = {
        "status": "ok",
        "source_manifest": str(args.source_manifest),
        "output_dir": str(args.output_dir),
        "shard_count": args.shard_count,
        "source_tuple_count": len(rows),
        "shard_source_tuple_count_total": len(all_keys),
        "shards_cover_each_source_once": len(all_keys) == len(set(all_keys)) == len(keys) and set(all_keys) == set(keys),
        "shards": report_shards,
        "contract": {
            "audio_processing_performed": False,
            "spatial_ast_processing_performed": False,
            "public_storage_written": False,
        },
    }
    if not report["shards_cover_each_source_once"]:
        report["status"] = "incomplete"
    report_path = args.output_dir / "source_shard_report.json"
    write_json(report_path, report)
    print(f"[input] source_manifest={args.source_manifest} rows={len(rows)}")
    print(f"[output] shards={args.shard_count} dir={shard_dir}")
    print(f"[report] {report_path}")
    print(f"[status] {report['status']}")
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
