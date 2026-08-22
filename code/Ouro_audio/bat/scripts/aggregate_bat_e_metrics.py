#!/usr/bin/env python3
"""Aggregate the two independent BAT E metrics into the reported E average.

The online evaluator intentionally processes one official split per job.  This
small read-only utility combines the completed E-direction and E-distance
reports without re-running model generation or audio rendering.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction-report", type=Path, required=True)
    parser.add_argument("--distance-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_report(path: Path, expected_type: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object report: {path}")
    if payload.get("status") != "ok":
        raise ValueError(f"Report is not complete: {path} status={payload.get('status')!r}")
    if payload.get("official_bat_contract", {}).get("eval_type") != expected_type:
        raise ValueError(f"Unexpected eval type in {path}: expected={expected_type}")
    scoring = payload.get("scoring")
    if not isinstance(scoring, dict) or scoring.get("binary_accuracy") is None:
        raise ValueError(f"Missing binary_accuracy scoring in {path}")
    return payload


def main() -> int:
    args = parse_args()
    direction = load_report(args.direction_report, "E-direction")
    distance = load_report(args.distance_report, "E-distance")
    direction_score = float(direction["scoring"]["binary_accuracy"])
    distance_score = float(distance["scoring"]["binary_accuracy"])
    output = {
        "status": "ok",
        "metric": "E Avg",
        "direction_accuracy": direction_score,
        "direction_accuracy_percent": 100.0 * direction_score,
        "distance_accuracy": distance_score,
        "distance_accuracy_percent": 100.0 * distance_score,
        "average": (direction_score + distance_score) / 2.0,
        "average_percent": 50.0 * (direction_score + distance_score),
        "direction_record_count": direction["scoring"].get("record_count"),
        "distance_record_count": distance["scoring"].get("record_count"),
        "direction_invalid_prediction_count": direction["scoring"].get("invalid_prediction_count"),
        "distance_invalid_prediction_count": distance["scoring"].get("invalid_prediction_count"),
        "source_reports": {
            "direction": str(args.direction_report.resolve()),
            "distance": str(args.distance_report.resolve()),
        },
        "generation_contract": direction.get("generation", {}).get("contract"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"[report] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
