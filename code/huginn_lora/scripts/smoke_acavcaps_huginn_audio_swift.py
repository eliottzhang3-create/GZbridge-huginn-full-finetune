from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview ACAVCAPS Swift manifest and decode the first audio sample.")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to the ACAVCAPS Swift JSONL manifest. Defaults to the pilot manifest under data/audio_swift/acavcaps.",
    )
    return parser.parse_args()


def count_records(manifest_path: Path) -> int:
    with manifest_path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def load_first_record(manifest_path: Path) -> dict:
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return json.loads(line)
    raise ValueError(f"Manifest is empty: {manifest_path}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    manifest_path = (
        Path(args.manifest)
        if args.manifest is not None
        else repo_root / "data" / "audio_swift" / "acavcaps" / "acavcaps_caption_long_smoke_swift.jsonl"
    )
    if not manifest_path.exists():
        raise FileNotFoundError(f"ACAVCAPS smoke manifest is missing: {manifest_path}")
    if manifest_path.stat().st_size <= 0:
        raise ValueError(f"ACAVCAPS smoke manifest is empty: {manifest_path}")

    plugin_dir = repo_root / "code" / "huginn_lora" / "plugins"
    sys.path.insert(0, str(plugin_dir))
    plugin = importlib.import_module("huginn_audio_swift")

    first_record = load_first_record(manifest_path)
    record_count = count_records(manifest_path)
    audio_items = first_record.get("audios") or []
    if len(audio_items) != 1:
        raise ValueError(f"Expected exactly one audio item, got {len(audio_items)}")
    audio_item = audio_items[0]
    if not isinstance(audio_item, dict):
        raise TypeError(f"Expected tar-backed dict audio item, got {type(audio_item)}")
    if "tar_path" not in audio_item or "audio_member" not in audio_item:
        raise KeyError("Audio item must contain tar_path and audio_member")

    print("========== ACAVCAPS SMOKE PREVIEW ==========")
    print(f"[smoke] manifest={manifest_path}")
    print(f"[smoke] records={record_count}")
    print(f"[smoke] ffmpeg_path={plugin.get_ffmpeg_path()}")
    print("[smoke] first_record=")
    print(json.dumps(first_record, ensure_ascii=False, indent=2))

    waveform = plugin.load_audio_from_tar(
        Path(str(audio_item["tar_path"])),
        str(audio_item["audio_member"]),
        target_sr=plugin.DEFAULT_SAMPLE_RATE,
        max_audio_seconds=plugin.DEFAULT_MAX_AUDIO_SECONDS,
    )
    duration_s = float(waveform.shape[0]) / float(plugin.DEFAULT_SAMPLE_RATE)
    print("========== ACAVCAPS SMOKE AUDIO ==========")
    print(f"[smoke] waveform_shape={tuple(waveform.shape)}")
    print(f"[smoke] duration_s={duration_s:.4f}")
    print(f"[smoke] min={float(waveform.min()):.6f}")
    print(f"[smoke] max={float(waveform.max()):.6f}")
    print(f"[smoke] mean={float(waveform.mean()):.6f}")


if __name__ == "__main__":
    main()
