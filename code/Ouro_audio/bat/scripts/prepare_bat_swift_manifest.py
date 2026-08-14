#!/usr/bin/env python3
"""Convert one official BAT QA JSON into ms-swift standard JSONL.

The generated manifest remains private on the remote filesystem.  The audio
item is a canonical, fixed-schema BAT record rather than the arbitrary raw
QA dictionary.  This is important because Hugging Face Datasets infers Arrow
features for nested JSON objects: if ``audio_id2`` is absent in early
single-source rows and a string in later dual-source rows, it infers a null
feature and fails when the later string is encountered.

The registered BAT template still resolves AudioSet and binaural RIR
references at collation time, preserving the official lazy audio pipeline.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

PROMPT = (
    "Based on the audio you've heard, refer to the instruction and provide a response.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

RAW_TO_TYPE = {
    "CLASSIFICATION": "A",
    "DOA": "B",
    "MIXUP_SINGLE_CLASSIFICATION": "C",
    "MIXUP_SINGLE_DOA": "D",
    "MIXUP_DISTANCE_BOTH": "E",
    "MIXUP_DIRECTION": "E",
    "MIXUP_NONBINARY_DISTANCE": "E",
    "MIXUP_NONBINARY_SOURCE": "E",
    "MIXUP_NONBINARY_DIRECTION": "E",
}

STAGES = {
    "I": ("stage1-clsdoa", {"A", "B"}),
    "II": ("stage2-single", {"A", "B", "C", "D"}),
    "III": ("stage3-mixup", {"A", "B", "C", "D", "E"}),
}

CANONICAL_AUDIO_FIELDS = (
    "audio_id",
    "reverb_id",
    "audio_id2",
    "reverb_id2",
    "question",
    "answer",
    "question_type",
    "question_id",
)


def present(value: Any) -> bool:
    return value is not None and str(value).strip().lower() not in {"", "null", "none"}


def load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError(f"Expected a list or {{data: list}} in {path}")
    return records


def source_shape(row: dict[str, Any]) -> str:
    second_audio = present(row.get("audio_id2"))
    second_reverb = present(row.get("reverb_id2"))
    if second_audio != second_reverb:
        raise ValueError(f"Partial second-source pair at question_id={row.get('question_id')}")
    return "dual" if second_audio else "single"


def bat_type(row: dict[str, Any]) -> str:
    raw = re.sub(r"[^A-Z0-9]+", "_", str(row.get("question_type", "")).upper()).strip("_")
    if raw not in RAW_TO_TYPE:
        raise ValueError(f"Unmapped BAT question_type={raw!r}")
    return RAW_TO_TYPE[raw]


def canonical_audio_record(row: dict[str, Any]) -> dict[str, str]:
    """Return the fixed-schema record consumed by ``BATAudioRenderer``.

    All fields are present and string-typed.  Empty strings represent an
    absent second source; ``source_shape`` and the renderer's ``_present``
    helper interpret them as missing without creating a nullable Arrow field.
    """
    second_audio = row.get("audio_id2")
    second_reverb = row.get("reverb_id2")
    return {
        "audio_id": str(row["audio_id"]),
        "reverb_id": str(row["reverb_id"]),
        "audio_id2": "" if not present(second_audio) else str(second_audio),
        "reverb_id2": "" if not present(second_reverb) else str(second_reverb),
        "question": str(row["question"]),
        "answer": str(row["answer"]),
        "question_type": str(row["question_type"]),
        "question_id": str(row["question_id"]),
    }


def convert(row: dict[str, Any], stage: str) -> dict[str, Any]:
    required = ("audio_id", "reverb_id", "question", "answer", "question_type", "question_id")
    missing = [key for key in required if row.get(key) in (None, "")]
    if missing:
        raise ValueError(f"Missing {missing} at question_id={row.get('question_id')}")
    kind = bat_type(row)
    source = canonical_audio_record(row)
    return {
        "messages": [
            {"role": "user", "content": PROMPT.format(instruction=str(row["question"]))},
            {"role": "assistant", "content": str(row["answer"])},
        ],
        "audios": [source],
        "bat_stage": stage,
        "bat_type": kind,
        "question_id": source["question_id"],
        "question_type": source["question_type"],
        "source_shape": source_shape(row),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-json", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 means all records")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise ValueError(f"--limit must be non-negative, got {args.limit}")
    if str(args.output).replace("\\", "/").startswith("/hpc_stor03/public"):
        raise SystemExit(f"Refusing public output path: {args.output}")
    stage_name, expected_types = STAGES[args.stage]
    records = load_records(args.qa_json)
    random.Random(args.seed).shuffle(records)
    if args.limit > 0:
        records = records[:args.limit]
        if len(records) != args.limit:
            raise RuntimeError(
                f"Requested --limit={args.limit}, but source contains only {len(records)} records: {args.qa_json}"
            )
    converted = [convert(row, args.stage) for row in records]
    observed = {row["bat_type"] for row in converted}
    if not observed or not observed <= expected_types:
        raise RuntimeError(f"{stage_name} manifest has invalid types: observed={sorted(observed)} expected={sorted(expected_types)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in converted:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(args.output)
    print(f"[manifest] stage={args.stage} source={args.qa_json} records={len(converted)} output={args.output}")
    print(f"[manifest] bat_types={sorted(observed)}")


if __name__ == "__main__":
    main()
