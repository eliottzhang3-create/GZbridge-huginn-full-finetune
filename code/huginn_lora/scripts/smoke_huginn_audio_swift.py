from __future__ import annotations

import json
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parents[3]
    manifest_path = repo_root / "data" / "audio_swift" / "clotho_aqa_tiny_train32_swift.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Swift smoke manifest is missing. Run prepare_huginn_audio_dataset.py first."
        )

    with manifest_path.open("r", encoding="utf-8") as f:
        first_record = json.loads(next(line for line in f if line.strip()))

    print(f"[audio-swift-smoke] manifest={manifest_path}")
    print("[audio-swift-smoke] first_record=")
    print(json.dumps(first_record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
