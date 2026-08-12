"""CPU-only Phase 1 audit for the remote OWL assets.

This script intentionally does not instantiate SAGE or allocate a CUDA
device. It inspects the SAGE checkpoint container and the BiDepth JSON files,
including the second-source fields that determine whether a record is
single-source or dual-source. It is meant to run on the remote login node.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SAGE = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/models/OWL/SAGE/finetuned.pth"
)
DEFAULT_BIDEPTH = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/data/BiDepth"
)
DEFAULT_OUTPUT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/owl/phase1_asset_audit.json"
)


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    return True


def _short(value: Any, limit: int = 180) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[: limit - 3] + "..."
    return value


def _load_records(path: Path) -> tuple[list[dict[str, Any]], str]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        records = payload
        container = "list"
    elif isinstance(payload, dict):
        container = "dict"
        candidate_keys = ["data", "records", "questions", "annotations", "items"]
        records = None
        for key in candidate_keys:
            value = payload.get(key)
            if isinstance(value, list):
                records = value
                container = f"dict[{key}]"
                break
        if records is None:
            list_values = [value for value in payload.values() if isinstance(value, list)]
            if len(list_values) == 1:
                records = list_values[0]
                container = "dict[唯一列表]"
            else:
                raise ValueError(
                    f"Cannot identify record list in {path}; top-level keys={list(payload)}"
                )
    else:
        raise TypeError(f"Unsupported JSON root in {path}: {type(payload).__name__}")

    invalid = [record for record in records if not isinstance(record, dict)]
    if invalid:
        raise TypeError(f"{path} contains {len(invalid)} non-object records")
    return records, container


def _record_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    keys = Counter(key for record in records for key in record)
    question_types = Counter(
        str(record.get("question_type", "<missing>")) for record in records
    )
    source_pairs = Counter()
    audio_id2_present = 0
    reverb_id2_present = 0
    both_second_ids_present = 0
    for record in records:
        audio2 = _is_present(record.get("audio_id2"))
        reverb2 = _is_present(record.get("reverb_id2"))
        audio_id2_present += int(audio2)
        reverb_id2_present += int(reverb2)
        both_second_ids_present += int(audio2 and reverb2)
        source_pairs["dual_if_audio_or_reverb_2"] += int(audio2 or reverb2)
        source_pairs["dual_if_both_2"] += int(audio2 and reverb2)
        source_pairs["single_if_both_2_missing"] += int(not audio2 and not reverb2)

    sample = records[0] if records else {}
    return {
        "count": len(records),
        "field_counts": dict(keys),
        "question_type_counts": dict(question_types),
        "audio_id2_present": audio_id2_present,
        "reverb_id2_present": reverb_id2_present,
        "both_second_ids_present": both_second_ids_present,
        "source_pair_heuristic_counts": dict(source_pairs),
        "first_record": {key: _short(value) for key, value in sample.items()},
    }


def _inspect_questions(root: Path) -> dict[str, Any]:
    questions_root = root / "owl-questions"
    if not questions_root.is_dir():
        raise FileNotFoundError(f"Missing question directory: {questions_root}")

    files = sorted(questions_root.glob("*/*.json"))
    if not files:
        raise FileNotFoundError(f"No stage JSON files found below {questions_root}")

    stages: dict[str, Any] = {}
    all_records: list[dict[str, Any]] = []
    for path in files:
        records, container = _load_records(path)
        stage = path.parent.name
        split = path.stem
        stages.setdefault(stage, {})[split] = {
            "path": str(path),
            "container": container,
            **_record_summary(records),
        }
        if split == "train":
            all_records.extend(records)

    return {
        "root": str(questions_root),
        "files": [str(path) for path in files],
        "stages": stages,
        "train_union_summary": _record_summary(all_records),
    }


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_sage(path: Path, hash_file: bool) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing SAGE checkpoint: {path}")

    report: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "size_gib": round(path.stat().st_size / (1024**3), 4),
    }
    if hash_file:
        print(f"[sage] computing sha256 for {path} ...", flush=True)
        report["sha256"] = _sha256(path)

    try:
        import torch
    except ImportError as exc:
        report["torch_import_error"] = repr(exc)
        return report

    print(f"[sage] loading checkpoint container on CPU: {path}", flush=True)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    report["container_type"] = type(checkpoint).__name__
    if isinstance(checkpoint, dict):
        report["container_keys"] = [str(key) for key in checkpoint.keys()]
        state = checkpoint.get("model")
        if isinstance(state, dict):
            report["state_dict_key_count"] = len(state)
            report["state_dict_preview"] = [
                {
                    "name": str(name),
                    "shape": list(value.shape) if hasattr(value, "shape") else None,
                    "dtype": str(value.dtype) if hasattr(value, "dtype") else None,
                }
                for name, value in list(state.items())[:20]
            ]
            report["state_dict_has_cls_tokens"] = any(
                "cls_tokens" in str(name) for name in state
            )
            report["state_dict_has_patch_embed"] = any(
                "patch_embed" in str(name) for name in state
            )
        else:
            report["model_state_type"] = type(state).__name__
    return report


def _print_stage_summary(question_report: dict[str, Any]) -> None:
    print("[questions] stage summary", flush=True)
    for stage, splits in question_report["stages"].items():
        for split, summary in splits.items():
            qtypes = summary["question_type_counts"]
            dual = summary["source_pair_heuristic_counts"].get(
                "dual_if_both_2", 0
            )
            single = summary["source_pair_heuristic_counts"].get(
                "single_if_both_2_missing", 0
            )
            print(
                f"  {stage}/{split}: count={summary['count']} "
                f"question_types={qtypes} second_ids(both)={dual} "
                f"second_ids(missing)={single}",
                flush=True,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sage-path", type=Path, default=DEFAULT_SAGE)
    parser.add_argument("--bidepth-root", type=Path, default=DEFAULT_BIDEPTH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--sha256",
        action="store_true",
        help="Compute the full SAGE checkpoint SHA256; this is CPU/I/O intensive.",
    )
    parser.add_argument(
        "--skip-sage",
        action="store_true",
        help="Only inspect BiDepth JSON files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("========== OWL PHASE 1 REMOTE ASSET AUDIT ==========")
    print(f"[python] version={sys.version.split()[0]} executable={sys.executable}")
    print(f"[sage] path={args.sage_path}")
    print(f"[bidepth] root={args.bidepth_root}")

    questions = _inspect_questions(args.bidepth_root)
    _print_stage_summary(questions)

    report: dict[str, Any] = {
        "status": "ok",
        "python": {"version": sys.version, "executable": sys.executable},
        "bidepth": questions,
        "interpretation_guardrails": [
            "Directory names stage1-clsdoa/stage2-single/stage3-mixup are recorded as dataset partitions, not assumed to equal OWL downstream curriculum stages.",
            "Single/dual status must be derived from actual second-source fields and official loader behavior.",
            "Question_type values are recorded verbatim and are not silently renamed to OWL Type I-IV.",
        ],
    }
    if args.skip_sage:
        report["sage"] = {"skipped": True}
    else:
        report["sage"] = _inspect_sage(args.sage_path, args.sha256)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"[report] {args.output}")


if __name__ == "__main__":
    main()
