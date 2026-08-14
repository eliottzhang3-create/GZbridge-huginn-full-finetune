"""Read-only audit for the BAT language-model curriculum and data pipeline.

This audit intentionally does not train or load Spatial-AST.  It verifies the
contract needed before integrating the frozen encoder with Ouro:

* the cumulative BAT language-model stages I/II/III;
* raw question-type to BAT A-E mapping;
* single-source and two-source records;
* AudioSet and binaural-reverb reference coverage;
* representative waveform/RIR metadata when optional readers are available;
* the expected official loader, prompt, and Q-Former contracts.

Spatial-AST's own two-stage pre-training is not treated as a BAT curriculum
stage here.  The pretrained Spatial-AST checkpoint is intentionally not loaded.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SOURCE_FIELDS = ("audio_id", "reverb_id", "audio_id2", "reverb_id2")
STAGES = ("stage1-clsdoa", "stage2-single", "stage3-mixup")
PAPER_STAGE_TARGETS = {
    "stage1-clsdoa": {
        "paper_stage": "I",
        "types": ["A", "B"],
        "approx_cumulative_records": 278_000,
        "approx_incremental_records": 278_000,
    },
    "stage2-single": {
        "paper_stage": "II",
        "types": ["A", "B", "C", "D"],
        "approx_cumulative_records": 514_000,
        "approx_incremental_records": 236_000,
    },
    "stage3-mixup": {
        "paper_stage": "III",
        "types": ["A", "B", "C", "D", "E"],
        "approx_cumulative_records": 872_000,
        "approx_incremental_records": 358_000,
    },
}


def present(value: Any) -> bool:
    return value is not None and str(value).strip().lower() not in {"", "null", "none"}


def norm(value: Any) -> str:
    return str(value).replace("\\", "/").lstrip("./")


def load_records(path: Path) -> tuple[list[dict[str, Any]], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        records = payload
        container = "list"
    elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
        records = payload["data"]
        container = "dict[data]"
    else:
        raise ValueError(f"Unsupported JSON container: {path}")
    if not all(isinstance(item, dict) for item in records):
        raise ValueError(f"Non-object record found: {path}")
    return records, container


def canonical(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def source_shape(record: dict[str, Any]) -> str:
    second_audio = present(record.get("audio_id2"))
    second_reverb = present(record.get("reverb_id2"))
    if second_audio and second_reverb:
        return "dual"
    if second_audio or second_reverb:
        return "partial_second_ids"
    return "single"


def infer_bat_type(record: dict[str, Any]) -> tuple[str | None, str]:
    """Return (A-E, reason) without classifying from answer text alone."""

    raw = str(record.get("question_type", "")).strip().upper()
    compact = re.sub(r"[^A-Z0-9]+", "_", raw).strip("_")
    shape = source_shape(record)

    if shape == "single" and compact in {"CLASSIFICATION", "CLS", "DETECTION"}:
        return "A", "single-source classification/detection"
    if shape == "single" and compact in {"DOA", "DIRECTION", "DISTANCE_DIRECTION", "DOA_DP"}:
        return "B", "single-source DoA/distance"
    if shape == "dual":
        # BAT Type E contains binary and non-binary reasoning families.  Check
        # these markers before CLASSIFICATION/DIRECTION because names such as
        # MIXUP_NONBINARY_CLASSIFICATION contain those substrings too.
        explicit_reasoning_types = {
            "MIXUP_DISTANCE_BOTH",
            "MIXUP_DIRECTION",
            "MIXUP_NONBINARY_DISTANCE",
            "MIXUP_NONBINARY_SOURCE",
            "MIXUP_NONBINARY_DIRECTION",
        }
        if compact in explicit_reasoning_types:
            return "E", "two-source BAT reasoning question_type"
        reasoning_markers = ("BINARY", "NONBINARY", "REASON", "REASONING")
        if any(marker in compact for marker in reasoning_markers):
            return "E", "two-source reasoning family"
        if compact in {"MIXUP_SINGLE_CLASSIFICATION", "CLASSIFICATION", "CLS", "DETECTION", "MIXUP_CLASSIFICATION", "MIXUP_CLS", "MIXUP_DETECTION"}:
            return "C", "two-source target classification"
        if compact in {"MIXUP_SINGLE_DOA", "DOA", "DIRECTION", "DISTANCE_DIRECTION", "DOA_DP", "MIXUP_DOA", "MIXUP_DISTANCE_DIRECTION", "MIXUP_DOA_DP"}:
            return "D", "two-source target DoA/distance"

    return None, f"unmapped raw question_type={raw!r}, source_shape={shape!r}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_file_summary(path: Path) -> dict[str, Any]:
    item: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return item
    item["bytes"] = path.stat().st_size
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        item["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        item["line_count"] = len(text.splitlines())
        tree = ast.parse(text, filename=str(path))
        item["ast_parse"] = "ok"
        item["classes"] = sorted(node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
        item["functions"] = sorted(
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        item["ast_parse"] = "failed"
        item["error"] = repr(exc)
    return item


def audio_candidates(root: Path, audio_id: str) -> list[Path]:
    relative = norm(audio_id)
    path = root / relative
    if Path(relative).suffix:
        return [path]
    return [path.with_suffix(suffix) for suffix in (".wav", ".flac", ".mp3", ".ogg")]


def resolve_audio(root: Path, audio_id: str) -> Path | None:
    for candidate in audio_candidates(root, audio_id):
        if candidate.is_file():
            return candidate
    return None


def resolve_reverb(root: Path, reverb_id: str) -> Path | None:
    relative = norm(reverb_id)
    candidates = [
        root / "binaural" / relative,
        root / relative,
        root / "mp3d_reverb" / "binaural" / relative,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def collect_refs(records: list[dict[str, Any]], fields: tuple[str, ...]) -> set[str]:
    return {
        norm(record[field])
        for record in records
        for field in fields
        if present(record.get(field))
    }


def optional_audio_probe(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "status": "not_attempted"}
    try:
        import soundfile as sf

        info = sf.info(str(path))
        result.update(
            {
                "status": "ok",
                "samplerate": int(info.samplerate),
                "channels": int(info.channels),
                "frames": int(info.frames),
                "duration_seconds": float(info.frames / info.samplerate),
                "format": info.format,
                "subtype": info.subtype,
            }
        )
    except Exception as exc:
        result.update({"status": "unavailable_or_failed", "error": repr(exc)})
    return result


def optional_reverb_probe(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "status": "not_attempted"}
    try:
        import numpy as np

        array = np.load(path, allow_pickle=False)
        result.update(
            {
                "status": "ok",
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "finite": bool(np.isfinite(array).all()),
                "peak_abs": float(np.max(np.abs(array))) if array.size else 0.0,
            }
        )
    except Exception as exc:
        result.update({"status": "unavailable_or_failed", "error": repr(exc)})
    return result


def summarize_records(records: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    raw_types = Counter(str(record.get("question_type", "<missing>")) for record in records)
    shapes = Counter(source_shape(record) for record in records)
    inferred = Counter()
    unmapped: Counter[str] = Counter()
    source_tuples = Counter(tuple(norm(record.get(field)) if present(record.get(field)) else "" for field in SOURCE_FIELDS) for record in records)
    question_ids = Counter(str(record.get("question_id", "<missing>")) for record in records)
    answer_yes_no = Counter(str(record.get("answer", "")).strip().lower() for record in records)
    examples: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        inferred_type, reason = infer_bat_type(record)
        if inferred_type:
            inferred[inferred_type] += 1
        else:
            unmapped[reason] += 1
        if len(examples) < 8:
            examples.append(
                {
                    "record_index": index,
                    "question_id": record.get("question_id"),
                    "question_type": record.get("question_type"),
                    "inferred_bat_type": inferred_type,
                    "inference_reason": reason,
                    "source_shape": source_shape(record),
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
        "record_count": len(records),
        "raw_question_type_counts": dict(raw_types),
        "source_shape_counts": dict(shapes),
        "inferred_bat_type_counts": dict(inferred),
        "unmapped_type_count": sum(unmapped.values()),
        "unmapped_type_examples": dict(unmapped.most_common(20)),
        "unique_audio_reference_count": len(collect_refs(records, ("audio_id", "audio_id2"))),
        "unique_reverb_reference_count": len(collect_refs(records, ("reverb_id", "reverb_id2"))),
        "unique_source_tuple_count": len(source_tuples),
        "source_tuple_reuse_extra_count": sum(count - 1 for count in source_tuples.values() if count > 1),
        "duplicate_question_id_extra_count": sum(count - 1 for count in question_ids.values() if count > 1),
        "answer_exact_yes_count": answer_yes_no.get("yes", 0),
        "answer_exact_no_count": answer_yes_no.get("no", 0),
        "examples": examples,
        "paper_contract": PAPER_STAGE_TARGETS[stage],
    }


def stage_containment(previous: list[dict[str, Any]], current: list[dict[str, Any]]) -> dict[str, Any]:
    previous_counter = Counter(canonical(record) for record in previous)
    current_counter = Counter(canonical(record) for record in current)
    missing = previous_counter - current_counter
    extra = current_counter - previous_counter
    return {
        "previous_count": sum(previous_counter.values()),
        "current_count": sum(current_counter.values()),
        "previous_unique_count": len(previous_counter),
        "current_contains_previous_as_multiset": not missing,
        "missing_previous_record_count": sum(missing.values()),
        "current_additional_record_count": sum(extra.values()),
        "current_additional_unique_record_count": len(extra),
    }


def asset_coverage(records: list[dict[str, Any]], audio_root: Path, reverb_root: Path, sample_count: int) -> dict[str, Any]:
    audio_refs = sorted(collect_refs(records, ("audio_id", "audio_id2")))
    reverb_refs = sorted(collect_refs(records, ("reverb_id", "reverb_id2")))
    missing_audio = [ref for ref in audio_refs if resolve_audio(audio_root, ref) is None]
    missing_reverb = [ref for ref in reverb_refs if resolve_reverb(reverb_root, ref) is None]
    audio_samples = []
    for ref in audio_refs[:sample_count]:
        path = resolve_audio(audio_root, ref)
        if path is not None:
            audio_samples.append({"reference": ref, **optional_audio_probe(path)})
    reverb_samples = []
    for ref in reverb_refs[:sample_count]:
        path = resolve_reverb(reverb_root, ref)
        if path is not None:
            reverb_samples.append({"reference": ref, **optional_reverb_probe(path)})
    return {
        "audio_root": str(audio_root),
        "reverb_root": str(reverb_root),
        "audio_reference_count": len(audio_refs),
        "audio_matched_count": len(audio_refs) - len(missing_audio),
        "audio_missing_count": len(missing_audio),
        "audio_missing_examples": missing_audio[:50],
        "reverb_reference_count": len(reverb_refs),
        "reverb_matched_count": len(reverb_refs) - len(missing_reverb),
        "reverb_missing_count": len(missing_reverb),
        "reverb_missing_examples": missing_reverb[:50],
        "audio_samples": audio_samples,
        "reverb_samples": reverb_samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-root", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--reverb-root", type=Path, required=True)
    parser.add_argument("--spatial-ast-root", type=Path, default=None)
    parser.add_argument("--qformer-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if str(args.output).startswith("/hpc_stor03/public"):
        raise SystemExit(f"Refusing public output path: {args.output}")

    print("========== BAT PHASE 1 DATA/CURRICULUM CONTRACT AUDIT ==========")
    print(f"[python] version={sys.version.split()[0]} executable={sys.executable}")
    print(f"[qa] root={args.qa_root}")
    print(f"[audio] root={args.audio_root}")
    print(f"[reverb] root={args.reverb_root}")
    print("[scope] frozen Spatial-AST; no encoder training; BAT LM stages I/II/III only")

    issues: list[str] = []
    stage_records: dict[str, list[dict[str, Any]]] = {}
    stages: dict[str, Any] = {}
    for stage in STAGES:
        path = args.qa_root / stage / "train.json"
        if not path.is_file():
            stages[stage] = {"status": "missing", "path": str(path)}
            issues.append(f"{stage}_train_missing")
            continue
        try:
            records, container = load_records(path)
        except Exception as exc:
            stages[stage] = {"status": "invalid", "path": str(path), "error": repr(exc)}
            issues.append(f"{stage}_train_invalid")
            continue
        stage_records[stage] = records
        item = summarize_records(records, stage)
        item["path"] = str(path)
        item["container"] = container
        item["status"] = "ok"
        item["asset_coverage"] = asset_coverage(records, args.audio_root, args.reverb_root, args.sample_count)
        stages[stage] = item
        if item["unmapped_type_count"]:
            issues.append(f"{stage}_unmapped_question_types")
        if item["asset_coverage"]["audio_missing_count"]:
            issues.append(f"{stage}_audio_reference_missing")
        if item["asset_coverage"]["reverb_missing_count"]:
            issues.append(f"{stage}_reverb_reference_missing")

    containment: dict[str, Any] = {}
    if all(stage in stage_records for stage in STAGES):
        containment["stage1_to_stage2"] = stage_containment(stage_records[STAGES[0]], stage_records[STAGES[1]])
        containment["stage2_to_stage3"] = stage_containment(stage_records[STAGES[1]], stage_records[STAGES[2]])
        # The paper describes cumulative learning, but released JSON files may
        # be either cumulative partitions or incremental partitions.  Do not
        # reject a valid incremental release merely because its records do not
        # literally contain every previous record.

    evaluations: dict[str, Any] = {}
    eval_files = sorted(args.qa_root.glob("*/eval-*.json"))
    for path in eval_files:
        try:
            records, container = load_records(path)
            summary = summarize_records(records, path.parent.name)
            evaluations[str(path.relative_to(args.qa_root))] = {
                "path": str(path),
                "container": container,
                "summary": summary,
                "asset_coverage": asset_coverage(records, args.audio_root, args.reverb_root, args.sample_count),
            }
        except Exception as exc:
            evaluations[str(path.relative_to(args.qa_root))] = {"path": str(path), "status": "invalid", "error": repr(exc)}
            issues.append(f"evaluation_invalid:{path.name}")

    source_files = {}
    if args.spatial_ast_root is not None:
        source_files["spatial_ast.py"] = source_file_summary(args.spatial_ast_root / "spatial_ast.py")
        source_files["data/dataset.py"] = source_file_summary(args.spatial_ast_root / "data/dataset.py")
        if not source_files["spatial_ast.py"]["exists"]:
            issues.append("spatial_ast_source_missing")
    if args.qformer_path is not None:
        source_files["qformer_source"] = source_file_summary(args.qformer_path)

    cumulative_type_union: dict[str, Any] = {}
    accumulated_types: set[str] = set()
    for stage in STAGES:
        item = stages.get(stage, {})
        accumulated_types.update(item.get("inferred_bat_type_counts", {}).keys())
        expected = set(PAPER_STAGE_TARGETS[stage]["types"])
        cumulative_type_union[stage] = {
            "types_seen_in_this_file": sorted(item.get("inferred_bat_type_counts", {}).keys()),
            "cumulative_types_seen_through_this_file": sorted(accumulated_types),
            "expected_cumulative_types": sorted(expected),
            "expected_types_missing_from_cumulative_union": sorted(expected - accumulated_types),
            "cumulative_type_contract_satisfied": expected <= accumulated_types,
        }
        if expected - accumulated_types:
            issues.append(f"{stage}_expected_bat_types_missing_from_cumulative_union")

    report = {
        "status": "incomplete" if issues else "ok",
        "python": {"version": sys.version, "executable": sys.executable},
        "scope": {
            "encoder_training": False,
            "encoder_checkpoint_loaded": False,
            "bat_language_model_curriculum": ["I", "II", "III"],
            "spatial_ast_pretraining_stages": "out_of_scope",
            "paper_stage_targets_are_approximate": True,
        },
        "paths": {
            "qa_root": str(args.qa_root),
            "audio_root": str(args.audio_root),
            "reverb_root": str(args.reverb_root),
        },
        "stages": stages,
        "stage_containment": containment,
        "curriculum_type_union": cumulative_type_union,
        "evaluations": evaluations,
        "source_files": source_files,
        "loader_contract": {
            "audio_reference": "audio_id(+optional audio_id2) resolves under an external AudioSet root",
            "reverb_reference": "binaural/<reverb_id> under extracted mp3d_reverb root",
            "single_source": "audio_id2 and reverb_id2 are both absent",
            "dual_source": "audio_id2 and reverb_id2 are both present; each source is convolved separately and mixed",
            "audio_preprocess": ["read source audio", "resample to 32000 Hz", "RMS/loudness normalize", "convolve with RIR", "trim/pad to 10 seconds", "produce binaural waveform"],
            "prompt": "Based on the audio you've heard, refer to the instruction and provide a response.\\n\\n### Instruction:\\n{instruction}\\n\\n### Response:",
            "qformer": {"layers": 8, "query_len": 64, "encoder_dim": 768, "target_ouro_dim": 2048},
            "trainable_for_ouro_branch": ["Q-Former", "Q-Former output projection", "Ouro LoRA"],
            "frozen_for_ouro_branch": ["Spatial-AST", "Ouro backbone", "Ouro early-exit gate"],
        },
        "issues": issues,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[report] {args.output}")
    for stage in STAGES:
        item = stages.get(stage, {})
        if item.get("status") != "ok":
            print(f"[summary] {stage} status={item.get('status')}")
            continue
        coverage = item["asset_coverage"]
        print(
            f"[summary] {stage} records={item['record_count']} "
            f"raw_types={item['raw_question_type_counts']} "
            f"inferred={item['inferred_bat_type_counts']} "
            f"shapes={item['source_shape_counts']}"
        )
        print(
            f"[summary] {stage} audio={coverage['audio_matched_count']}/{coverage['audio_reference_count']} "
            f"reverb={coverage['reverb_matched_count']}/{coverage['reverb_reference_count']}"
        )
    print(f"[summary] evaluations={len(evaluations)}")
    print(f"[status] {report['status']} issues={issues}")
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
