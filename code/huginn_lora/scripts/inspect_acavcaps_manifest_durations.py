from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TIME_SUFFIX_PATTERN = re.compile(
    r"^.+_(?P<start_seconds>\d+)_(?P<start_fraction>\d+)_(?P<end_seconds>\d+)_(?P<end_fraction>\d+)\.flac$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect ACAVCAPS clip durations encoded in tar-backed manifest names.")
    parser.add_argument("--manifest_dir", required=True)
    parser.add_argument("--pattern", default="acavcaps_caption_long_formal_chunk_*.jsonl")
    return parser.parse_args()


def parse_duration_seconds(audio_member: str) -> float | None:
    match = TIME_SUFFIX_PATTERN.match(Path(audio_member).name)
    if match is None:
        return None
    start = float(f"{match['start_seconds']}.{match['start_fraction']}")
    end = float(f"{match['end_seconds']}.{match['end_fraction']}")
    duration = end - start
    return duration if duration >= 0 else None


def percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot calculate a percentile for an empty sequence")
    index = round((len(sorted_values) - 1) * fraction)
    return sorted_values[index]


def main() -> None:
    args = parse_args()
    manifest_dir = Path(args.manifest_dir)
    manifest_paths = sorted(manifest_dir.glob(args.pattern))
    if not manifest_paths:
        raise FileNotFoundError(f"No manifests matching {args.pattern!r} under {manifest_dir}")

    durations: list[float] = []
    unparsable_examples: list[str] = []
    records = 0
    over_30_seconds = 0

    print("========== ACAVCAPS MANIFEST DURATION INSPECT ==========")
    print(f"[duration] manifest_dir={manifest_dir}")
    print(f"[duration] pattern={args.pattern}")
    print(f"[duration] manifest_count={len(manifest_paths)}")

    for manifest_path in manifest_paths:
        manifest_records = 0
        manifest_durations: list[float] = []
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                audios = record.get("audios") or []
                if len(audios) != 1 or not isinstance(audios[0], dict):
                    raise ValueError(f"Expected one tar-backed audio entry in {manifest_path}")
                audio_member = str(audios[0].get("audio_member", ""))
                duration = parse_duration_seconds(audio_member)
                if duration is None:
                    if len(unparsable_examples) < 10:
                        unparsable_examples.append(f"{manifest_path.name}:{audio_member}")
                else:
                    durations.append(duration)
                    manifest_durations.append(duration)
                    if duration > 30.0:
                        over_30_seconds += 1
                manifest_records += 1
                records += 1

        if not manifest_durations:
            raise ValueError(f"No parseable durations found in {manifest_path}")
        print(
            f"[duration] manifest={manifest_path.name} records={manifest_records} "
            f"min_s={min(manifest_durations):.4f} max_s={max(manifest_durations):.4f}"
        )

    if not durations:
        raise ValueError("No parseable audio durations found")
    durations.sort()
    print("========== ACAVCAPS MANIFEST DURATION SUMMARY ==========")
    print(f"[duration] total_records={records}")
    print(f"[duration] parseable_records={len(durations)}")
    print(f"[duration] unparsable_records={records - len(durations)}")
    print(f"[duration] min_s={durations[0]:.4f}")
    print(f"[duration] p50_s={percentile(durations, 0.50):.4f}")
    print(f"[duration] p95_s={percentile(durations, 0.95):.4f}")
    print(f"[duration] max_s={durations[-1]:.4f}")
    print(f"[duration] over_30_seconds={over_30_seconds}")
    for example in unparsable_examples:
        print(f"[duration] unparsable_example={example}")


if __name__ == "__main__":
    main()
