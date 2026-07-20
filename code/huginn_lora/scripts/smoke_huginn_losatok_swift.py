"""Create and validate a one-record AudioCaps manifest for LoSATok Swift smoke training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torchaudio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_manifest", required=True)
    parser.add_argument("--output_manifest", required=True)
    return parser.parse_args()


def first_record(path: Path) -> tuple[str, dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return line.rstrip("\n"), json.loads(line)
    raise ValueError(f"Manifest has no records: {path}")


def main() -> None:
    args = parse_args()
    source_manifest = Path(args.source_manifest)
    output_manifest = Path(args.output_manifest)
    if not source_manifest.is_file() or source_manifest.stat().st_size == 0:
        raise FileNotFoundError(f"Source manifest is missing or empty: {source_manifest}")
    raw_record, record = first_record(source_manifest)
    audios = record.get("audios") or []
    if len(audios) != 1 or not isinstance(audios[0], str):
        raise ValueError("LoSATok smoke requires exactly one filesystem audio path in the first AudioCaps record")
    audio_path = Path(audios[0])
    if not audio_path.is_file():
        raise FileNotFoundError(f"Smoke audio is missing: {audio_path}")
    info = torchaudio.info(str(audio_path))
    if info.num_frames <= 0 or info.sample_rate <= 0:
        raise ValueError(f"Invalid smoke audio metadata: {info}")
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_manifest.with_suffix(".jsonl.tmp")
    temporary.write_text(raw_record + "\n", encoding="utf-8")
    temporary.replace(output_manifest)
    caption = record.get("messages", [])[-1].get("content", "") if record.get("messages") else ""
    print("========== LOSATOK SMOKE DATA PREP ==========")
    print(f"[smoke] source_manifest={source_manifest}")
    print(f"[smoke] output_manifest={output_manifest}")
    print(f"[smoke] audio_path={audio_path}")
    print(f"[smoke] source_sample_rate={info.sample_rate} channels={info.num_channels}")
    print(f"[smoke] source_frames={info.num_frames} duration_seconds={info.num_frames / info.sample_rate:.6f}")
    print(f"[smoke] caption={caption}")


if __name__ == "__main__":
    main()
