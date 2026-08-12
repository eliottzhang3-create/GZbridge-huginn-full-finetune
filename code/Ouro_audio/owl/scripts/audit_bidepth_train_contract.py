"""Focused audit of the actual OWL Stage 1/2 training data contract.

This report deliberately ignores the duplicate val/test question.  It answers
the questions that determine whether training can start:

* what records are in Stage 1/2 train;
* whether each record is single-source or two-source according to the official
  ID fields;
* how many unique anechoic-audio and RIR references are required;
* whether every referenced extracted RIR exists and can be sampled;
* whether an external anechoic-audio root is present;
* what the official loader and launcher say about path construction, stage
  selection, convolution, and fixed-length processing.

It is read-only and does not load a GPU model.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from audit_bidepth_deep import (
    DEFAULT_BIDEPTH,
    _inspect_audio_roots,
    _load_records,
    _norm_path,
    _present,
)
from audit_bidepth_decompressed import (
    DEFAULT_OWL_SOURCE_ROOT,
    DEFAULT_REVERB_ROOT,
    _inspect_official_components,
    _load_npy,
    _resolve_reverb,
)


DEFAULT_OUTPUT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/owl/"
    "phase1_train_contract_audit.json"
)
TRAIN_STAGES = ("stage1-clsdoa", "stage2-single")
SOURCE_FIELDS = ("audio_id", "reverb_id", "audio_id2", "reverb_id2")
AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg", ".npy")


def _source_tuple(record: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        _norm_path(str(record.get(field))) if _present(record.get(field)) else ""
        for field in SOURCE_FIELDS
    )


def _audio_candidates(root: Path, audio_id: str) -> list[Path]:
    normalized = _norm_path(audio_id)
    candidates = [root / normalized]
    if Path(normalized).suffix:
        return candidates
    candidates.extend(root / f"{normalized}{suffix}" for suffix in AUDIO_SUFFIXES)
    return candidates


def _loader_context(text: str, patterns: tuple[str, ...], limit: int = 20) -> dict[str, Any]:
    lines = text.splitlines()
    result: dict[str, Any] = {}
    for pattern in patterns:
        hits: list[dict[str, Any]] = []
        regex = re.compile(pattern, flags=re.IGNORECASE)
        for line_number, line in enumerate(lines, start=1):
            if regex.search(line) and len(hits) < limit:
                start = max(1, line_number - 2)
                end = min(len(lines), line_number + 2)
                hits.append(
                    {
                        "line": line_number,
                        "text": line.strip(),
                        "context": [
                            {"line": number, "text": lines[number - 1].strip()}
                            for number in range(start, end + 1)
                        ],
                    }
                )
        result[pattern] = hits
    return result


def _ast_summary(text: str, path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path)}
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        return {**result, "ast_parse": "failed", "ast_error": repr(exc)}
    calls: Counter[str] = Counter()
    functions: list[str] = []
    classes: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls[node.func.id] += 1
            elif isinstance(node.func, ast.Attribute):
                calls[node.func.attr] += 1
    return {
        **result,
        "ast_parse": "ok",
        "functions": sorted(functions),
        "classes": sorted(classes),
        "calls": dict(calls),
    }


def _inspect_loader_contract(source_root: Path) -> dict[str, Any]:
    loader_path = source_root / "seld_cot/owl/dataset/spatial_audio_dataset.py"
    train_path = source_root / "seld_cot/owl/scripts/train.sh"
    result: dict[str, Any] = {
        "source_root": str(source_root),
        "loader_path": str(loader_path),
        "train_script_path": str(train_path),
        "status": "ok",
    }
    if not loader_path.is_file():
        result["status"] = "missing_loader"
        return result
    loader_text = loader_path.read_text(encoding="utf-8", errors="replace")
    result["loader"] = {
        "sha256": hashlib.sha256(loader_text.encode("utf-8")).hexdigest(),
        "line_count": len(loader_text.splitlines()),
        "ast": _ast_summary(loader_text, loader_path),
        "contexts": _loader_context(
            loader_text,
            (
                r"audio_id2?",
                r"reverb_id2?",
                r"anechoic_data_root",
                r"reverb_data_root",
                r"stage",
                r"np\.load|librosa|torchaudio|soundfile",
                r"convolve|fftconvolve|rir|impulse",
                r"fix_length|sample_rate|sr=|32000|32k",
                r"audio_path|audio_file|wav",
            ),
        ),
    }
    if train_path.is_file():
        train_text = train_path.read_text(encoding="utf-8", errors="replace")
        result["train_launcher"] = {
            "sha256": hashlib.sha256(train_text.encode("utf-8")).hexdigest(),
            "line_count": len(train_text.splitlines()),
            "contexts": _loader_context(
                train_text,
                (
                    r"stage=",
                    r"qa_data_root|reverb_data_root|anechoic_data_root",
                    r"max_words|fix_length_audio",
                    r"qformer_layers|freeze_encoder|freeze_llm",
                    r"num_epochs|total_steps|validation_interval",
                    r"batch_size_training|val_batch_size|num_workers",
                    r"use_peft|peft_method|lr|warmup_steps",
                ),
            ),
        }
    else:
        result["status"] = "missing_train_launcher"
    return result


def _record_summary(records: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    question_types = Counter()
    source_shapes = Counter()
    audio_ids: set[str] = set()
    audio_ids2: set[str] = set()
    reverb_ids: set[str] = set()
    reverb_ids2: set[str] = set()
    source_tuples: Counter[tuple[str, ...]] = Counter()
    question_ids: set[str] = set()
    answer_lengths: list[int] = []
    question_lengths: list[int] = []
    answer_markers = Counter()
    missing_fields = Counter()
    examples: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            missing_fields["non_dict_record"] += 1
            continue
        for field in SOURCE_FIELDS:
            if not _present(record.get(field)):
                missing_fields[field] += 1
        source = _source_tuple(record)
        has_second_audio = bool(source[2])
        has_second_reverb = bool(source[3])
        if has_second_audio and has_second_reverb:
            source_shapes["dual_both_second_ids"] += 1
        elif has_second_audio or has_second_reverb:
            source_shapes["partial_second_ids"] += 1
        else:
            source_shapes["single_no_second_ids"] += 1
        question_types[str(record.get("question_type", "<missing>"))] += 1
        if source[0]:
            audio_ids.add(source[0])
        if source[2]:
            audio_ids2.add(source[2])
        if source[1]:
            reverb_ids.add(source[1])
        if source[3]:
            reverb_ids2.add(source[3])
        source_tuples[source] += 1
        question_ids.add(str(record.get("question_id", "<missing>")))
        question = str(record.get("question", ""))
        answer = str(record.get("answer", ""))
        question_lengths.append(len(question))
        answer_lengths.append(len(answer))
        answer_lower = answer.lower()
        for marker in ("because", "therefore", "based on", "first", "then"):
            if marker in answer_lower:
                answer_markers[marker] += 1
        if len(examples) < 5:
            examples.append(
                {
                    "record_index": index,
                    "question_id": record.get("question_id"),
                    "question_type": record.get("question_type"),
                    "audio_id": record.get("audio_id"),
                    "reverb_id": record.get("reverb_id"),
                    "audio_id2": record.get("audio_id2"),
                    "reverb_id2": record.get("reverb_id2"),
                    "question": record.get("question"),
                    "answer": record.get("answer"),
                }
            )

    return {
        "stage": stage,
        "count": len(records),
        "question_type_counts": dict(question_types),
        "source_shape_counts": dict(source_shapes),
        "unique_primary_audio_id_count": len(audio_ids),
        "unique_second_audio_id_count": len(audio_ids2),
        "unique_primary_reverb_id_count": len(reverb_ids),
        "unique_second_reverb_id_count": len(reverb_ids2),
        "unique_audio_id_total_count": len(audio_ids | audio_ids2),
        "unique_reverb_id_total_count": len(reverb_ids | reverb_ids2),
        "unique_source_tuple_count": len(source_tuples),
        "source_tuple_reuse_extra_count": sum(
            count - 1 for count in source_tuples.values() if count > 1
        ),
        "unique_question_id_count": len(question_ids),
        "missing_or_empty_field_counts": dict(missing_fields),
        "answer_marker_counts": dict(answer_markers),
        "question_length_chars": {
            "min": min(question_lengths) if question_lengths else None,
            "max": max(question_lengths) if question_lengths else None,
            "mean": sum(question_lengths) / len(question_lengths)
            if question_lengths else None,
        },
        "answer_length_chars": {
            "min": min(answer_lengths) if answer_lengths else None,
            "max": max(answer_lengths) if answer_lengths else None,
            "mean": sum(answer_lengths) / len(answer_lengths)
            if answer_lengths else None,
        },
        "examples": examples,
    }


def _collect_refs(records: list[dict[str, Any]], fields: tuple[str, ...]) -> set[str]:
    refs: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        for field in fields:
            if _present(record.get(field)):
                refs.add(_norm_path(str(record[field])))
    return refs


def _audio_inventory(root: Path) -> list[str]:
    """Inspect likely audio locations without walking the extracted RIR tree."""

    if not root.is_dir():
        return []
    excluded = {"reverb_extracted", "owl-questions"}
    candidates: list[str] = []
    for child in sorted(root.iterdir()):
        if child.name in excluded or child.name == "reverb.tar.gz":
            continue
        if child.is_file() and child.suffix.lower() in AUDIO_SUFFIXES:
            candidates.append(str(child))
        elif child.is_dir():
            for path in child.rglob("*"):
                if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
                    candidates.append(str(path))
                    if len(candidates) >= 100:
                        return candidates
    return candidates[:100]


def _inspect_stage_assets(
    stage: str,
    records: list[dict[str, Any]],
    reverb_root: Path,
    audio_roots: list[Path],
    sample_limit: int,
) -> dict[str, Any]:
    audio_refs = _collect_refs(records, ("audio_id", "audio_id2"))
    reverb_refs = _collect_refs(records, ("reverb_id", "reverb_id2"))
    missing_reverb = sorted(ref for ref in reverb_refs if _resolve_reverb(reverb_root, ref) is None)
    reverb_samples: dict[str, Any] = {}
    for ref in sorted(reverb_refs)[:sample_limit]:
        path = _resolve_reverb(reverb_root, ref)
        reverb_samples[ref] = {"status": "missing"} if path is None else _load_npy(path)

    audio_report: dict[str, Any]
    if audio_roots:
        audio_report = _inspect_audio_roots(audio_roots, audio_refs)
    else:
        audio_report = {
            "status": "not_configured",
            "reference_count": len(audio_refs),
            "expected_reference_examples": sorted(audio_refs)[:20],
        }
    return {
        "stage": stage,
        "train_record_count": len(records),
        "unique_audio_reference_count": len(audio_refs),
        "unique_reverb_reference_count": len(reverb_refs),
        "reverb_reference_missing_count": len(missing_reverb),
        "reverb_reference_missing_examples": missing_reverb[:50],
        "reverb_reference_coverage": (
            (len(reverb_refs) - len(missing_reverb)) / len(reverb_refs)
            if reverb_refs else 1.0
        ),
        "reverb_samples": reverb_samples,
        "audio": audio_report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bidepth-root", type=Path, default=DEFAULT_BIDEPTH)
    parser.add_argument("--reverb-root", type=Path, default=DEFAULT_REVERB_ROOT)
    parser.add_argument("--owl-source-root", type=Path, default=DEFAULT_OWL_SOURCE_ROOT)
    parser.add_argument("--audio-root", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-rir-count", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("========== OWL STAGE 1/2 TRAIN CONTRACT AUDIT ==========")
    print(f"[python] version={sys.version.split()[0]} executable={sys.executable}")
    print(f"[bidepth] root={args.bidepth_root}")
    print(f"[reverb] root={args.reverb_root}")
    print(f"[owl] source_root={args.owl_source_root}")

    question_root = args.bidepth_root / "owl-questions"
    stages: dict[str, Any] = {}
    issues: list[str] = []
    for stage in TRAIN_STAGES:
        path = question_root / stage / "train.json"
        if not path.is_file():
            stages[stage] = {"status": "missing", "path": str(path)}
            issues.append(f"{stage}_train_missing")
            continue
        records, container = _load_records(path)
        stages[stage] = {
            "path": str(path),
            "container": container,
            "record_summary": _record_summary(records, stage),
            "assets": _inspect_stage_assets(
                stage,
                records,
                args.reverb_root,
                args.audio_root,
                args.sample_rir_count,
            ),
        }
        if stages[stage]["assets"]["reverb_reference_missing_count"]:
            issues.append(f"{stage}_reverb_reference_missing")

    source_inventory = {
        "root": str(args.bidepth_root),
        "top_level_entries": sorted(path.name for path in args.bidepth_root.iterdir())
        if args.bidepth_root.is_dir() else [],
        "audio_files_outside_archives": _audio_inventory(args.bidepth_root),
        "note": (
            "The presence of audio_id references in JSON is not the same as "
            "bundled audio waveforms. This inventory reports actual audio files "
            "under the BiDepth root."
        ),
    }
    report: dict[str, Any] = {
        "status": "ok",
        "python": {"version": sys.version, "executable": sys.executable},
        "scope": {
            "train_stages": list(TRAIN_STAGES),
            "validation_split_used_later": "val",
            "test_split_used_now": False,
        },
        "source_inventory": source_inventory,
        "stages": stages,
        "official_loader_contract": _inspect_loader_contract(args.owl_source_root),
        "official_component_summary": _inspect_official_components(args.owl_source_root),
        "interpretation": {
            "audio_id": (
                "A logical reference to an anechoic source clip. The loader is "
                "expected to resolve it below an external anechoic_data_root."
            ),
            "reverb_id": (
                "A relative reference to an MP3D/SoundSpaces room impulse response "
                "used to spatialize/convolve the source audio."
            ),
            "audio_id2_reverb_id2": (
                "When both second IDs are present, the record is a two-source/mixup "
                "sample; when both are empty, it is a single-source sample."
            ),
        },
        "audit_contract": {
            "read_only": True,
            "gpu_model_loaded": False,
            "val_test_duplicate_analysis_excluded": True,
        },
        "issues": issues,
    }
    report["status"] = "incomplete" if issues else "ok"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"[report] {args.output}")
    print(f"[status] {report['status']} issues={issues}")


if __name__ == "__main__":
    main()
