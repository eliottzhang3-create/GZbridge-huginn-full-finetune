#!/usr/bin/env python3
"""Create a small deterministic A+B/C+D+E manifest for the Stage-III resume smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records-per-group", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source_manifest.is_file():
        raise FileNotFoundError(args.source_manifest)
    if args.records_per_group <= 0:
        raise ValueError("--records-per-group must be positive")
    selected: dict[str, list[dict[str, Any]]] = {"AB": [], "CDE": []}
    with args.source_manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {args.source_manifest}:{line_number}")
            bat_type = str(row.get("bat_type", ""))
            group = "AB" if bat_type in {"A", "B"} else "CDE" if bat_type in {"C", "D", "E"} else None
            if group is None or len(selected[group]) >= args.records_per_group:
                if all(len(values) >= args.records_per_group for values in selected.values()):
                    break
                continue
            selected[group].append(row)
            if all(len(values) >= args.records_per_group for values in selected.values()):
                break
    missing = {key: len(values) for key, values in selected.items() if len(values) < args.records_per_group}
    if missing:
        raise RuntimeError(f"Source manifest does not contain enough records: {missing}")

    rows = selected["AB"] + selected["CDE"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(args.output)
    print(f"[smoke-manifest] output={args.output} records={len(rows)} AB={len(selected['AB'])} CDE={len(selected['CDE'])}")


if __name__ == "__main__":
    main()
