"""Create and validate a one-record AudioCaps manifest for LoSATok Swift smoke training."""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_manifest", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--record_count", type=int, required=True)
    return parser.parse_args()


def load_records(path: Path, record_count: int) -> list[tuple[str, dict]]:
    if record_count <= 0:
        raise ValueError(f"record_count must be positive, got {record_count}")
    records: list[tuple[str, dict]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append((line.rstrip("\n"), json.loads(line)))
                if len(records) == record_count:
                    return records
    raise ValueError(f"Manifest has fewer than {record_count} records: {path}")


def main() -> None:
    args = parse_args()
    source_manifest = Path(args.source_manifest)
    output_manifest = Path(args.output_manifest)
    if not source_manifest.is_file() or source_manifest.stat().st_size == 0:
        raise FileNotFoundError(f"Source manifest is missing or empty: {source_manifest}")
    records = load_records(source_manifest, args.record_count)
    metadata: list[dict[str, object]] = []
    for _, record in records:
        audios = record.get("audios") or []
        if len(audios) != 1 or not isinstance(audios[0], str):
            raise ValueError("LoSATok smoke requires exactly one filesystem audio path per AudioCaps record")
        audio_path = Path(audios[0])
        if not audio_path.is_file():
            raise FileNotFoundError(f"Smoke audio is missing: {audio_path}")
        with wave.open(str(audio_path), "rb") as handle:
            channels = handle.getnchannels()
            sample_rate = handle.getframerate()
            frame_count = handle.getnframes()
            sample_width = handle.getsampwidth()
        if frame_count <= 0 or sample_rate <= 0:
            raise ValueError(f"Invalid smoke WAV metadata: frames={frame_count} sample_rate={sample_rate}")
        metadata.append({
            "audio_path": str(audio_path),
            "sample_rate": sample_rate,
            "channels": channels,
            "sample_width_bytes": sample_width,
            "frame_count": frame_count,
            "duration_seconds": frame_count / sample_rate,
        })
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_manifest.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(raw_record + "\n" for raw_record, _ in records), encoding="utf-8")
    temporary.replace(output_manifest)
    durations = [float(item["duration_seconds"]) for item in metadata]
    print("========== LOSATOK SMOKE DATA PREP ==========")
    print(f"[smoke] source_manifest={source_manifest}")
    print(f"[smoke] output_manifest={output_manifest}")
    print(f"[smoke] record_count={len(records)}")
    print(f"[smoke] duration_seconds_min={min(durations):.6f} max={max(durations):.6f}")
    for index, item in enumerate(metadata[:3]):
        print(f"[smoke] audio[{index}]={json.dumps(item, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
