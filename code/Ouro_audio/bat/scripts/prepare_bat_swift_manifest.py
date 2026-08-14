#!/usr/bin/env python3
"""Convert one official BAT QA JSON into ms-swift standard JSONL.

The generated manifest remains private on the remote filesystem.  The audio
item is deliberately the original BAT record rather than a copied waveform:
the registered BAT template resolves AudioSet and binaural RIR references at
collation time, preserving the official lazy audio pipeline.
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


def load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError(f"Expected a list or {{data: list}} in {path}")
    return records


def source_shape(row: dict[str, Any]) -> str:
    def present(value: Any) -> bool:
        return value is not None and str(value).strip().lower() not in {"", "null", "none"}

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


def convert(row: dict[str, Any], stage: str) -> dict[str, Any]:
    required = ("audio_id", "reverb_id", "question", "answer", "question_type", "question_id")
    missing = [key for key in required if row.get(key) in (None, "")]
    if missing:
        raise ValueError(f"Missing {missing} at question_id={row.get('question_id')}")
    kind = bat_type(row)
    return {
        "messages": [
            {"role": "user", "content": PROMPT.format(instruction=str(row["question"]))},
            {"role": "assistant", "content": str(row["answer"])},
        ],
        "audios": [row],
        "bat_stage": stage,
        "bat_type": kind,
        "question_id": row["question_id"],
        "question_type": row["question_type"],
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
