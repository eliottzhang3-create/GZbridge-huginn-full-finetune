"""Audit BiDepth against the paper's semantic Type I-IV taxonomy.

The released OWL loader does not implement Type I-IV filtering.  It loads one
whole ``<qa_data_root>/<stage>/<split>.json`` file, so this script classifies
the records by their question/answer semantics and compares that result with
the directory name and the coarse ``question_type`` field.

This is deliberately a conservative audit, not a training-time filter.  Each
record receives a primary heuristic label plus evidence flags and examples.
The report makes it possible to decide whether the downloaded JSON files are
paper stages, cumulative stage files, or a different/revised release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from audit_bidepth_deep import DEFAULT_BIDEPTH, _load_records, _norm_path, _present
from output_safety import assert_private_output


DEFAULT_OUTPUT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/owl/"
    "phase1_paper_type_audit.json"
)

SOURCE_FIELDS = ("audio_id", "reverb_id", "audio_id2", "reverb_id2")
STAGES = ("stage1-clsdoa", "stage2-single", "stage3-mixup")
SPLITS = ("train", "val", "test")

YES_NO_RE = re.compile(r"^\s*(yes|no)\s*[.!]?\s*$", re.IGNORECASE)
POSITION_RE = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*o'?clock\b"
    r"|\b(?:up|down)\b.*\b\d+(?:\.\d+)?\b",
    re.IGNORECASE,
)
RELATIONAL_RE = re.compile(
    r"\b(left|right|front|back|above|below|overhead|closer|farther|nearer|distance)\b"
    r"|\b(compared|between|both|relative|than)\b",
    re.IGNORECASE,
)
COT_RE = re.compile(
    r"\b(therefore|because|since|this shows|which means|indicate|indicating|while)\b",
    re.IGNORECASE,
)
DOA_QUESTION_RE = re.compile(
    r"\b(direction|distance|where|origin|originating|located|bearing|range|spot|position|far)\b",
    re.IGNORECASE,
)
CLASSIFICATION_QUESTION_RE = re.compile(
    r"\b(classif|categor|sound events?|sounds? (?:are|can|do)|auditory events?|identify all|enumerate|list the types)\b",
    re.IGNORECASE,
)


def _source_shape(record: dict[str, Any]) -> str:
    has_audio2 = _present(record.get("audio_id2"))
    has_reverb2 = _present(record.get("reverb_id2"))
    if has_audio2 and has_reverb2:
        return "dual"
    if has_audio2 or has_reverb2:
        return "partial_second_ids"
    return "single"


def _type_evidence(record: dict[str, Any]) -> dict[str, Any]:
    question = str(record.get("question", ""))
    answer = str(record.get("answer", ""))
    question_lower = question.lower()
    answer_lower = answer.lower()
    source_shape = _source_shape(record)
    answer_yes_no = bool(YES_NO_RE.match(answer))
    has_position = bool(POSITION_RE.search(answer))
    has_cot = bool(COT_RE.search(answer))
    has_relation_question = bool(RELATIONAL_RE.search(question))
    has_doa_question = bool(DOA_QUESTION_RE.search(question))
    has_classification_question = bool(CLASSIFICATION_QUESTION_RE.search(question))

    # Type III is the strongest lexical case: dual source, relational query,
    # and a binary answer. Type IV is dual-source relational reasoning with an
    # explanatory answer rather than a bare Yes/No. Type I/II are separated by
    # event-label versus position/DoA answer patterns.
    if source_shape == "dual" and answer_yes_no and (
        has_relation_question or not has_position
    ):
        inferred = "Type_III"
    elif source_shape == "dual" and (has_cot or (has_relation_question and has_position)):
        inferred = "Type_IV"
    elif has_position or (has_doa_question and not has_classification_question):
        inferred = "Type_II"
    elif has_classification_question or not has_position:
        inferred = "Type_I"
    else:
        inferred = "Uncertain"

    return {
        "inferred_type": inferred,
        "source_shape": source_shape,
        "question_type_field": record.get("question_type"),
        "answer_is_bare_yes_no": answer_yes_no,
        "answer_has_position_pattern": has_position,
        "answer_has_cot_marker": has_cot,
        "question_has_relation_pattern": has_relation_question,
        "question_has_doa_pattern": has_doa_question,
        "question_has_classification_pattern": has_classification_question,
        "question_chars": len(question),
        "answer_chars": len(answer),
    }


def _canonical_record(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _record_example(record: dict[str, Any], evidence: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "record_index": index,
        "question_id": record.get("question_id"),
        "question_type_field": record.get("question_type"),
        "inferred_type": evidence["inferred_type"],
        "source_shape": evidence["source_shape"],
        "audio_id": record.get("audio_id"),
        "reverb_id": record.get("reverb_id"),
        "audio_id2": record.get("audio_id2"),
        "reverb_id2": record.get("reverb_id2"),
        "question": record.get("question"),
        "answer": record.get("answer"),
        "evidence": evidence,
    }


def _audit_file(path: Path, stage: str, split: str) -> dict[str, Any]:
    records, container = _load_records(path)
    inferred_counts: Counter[str] = Counter()
    matrix: Counter[tuple[str, str, str]] = Counter()
    shape_counts: Counter[str] = Counter()
    source_ids: dict[str, set[str]] = {field: set() for field in SOURCE_FIELDS}
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    uncertain_examples: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            inferred_counts["Invalid_record"] += 1
            continue
        evidence = _type_evidence(record)
        inferred = evidence["inferred_type"]
        source_shape = evidence["source_shape"]
        field_type = str(record.get("question_type", "<missing>"))
        inferred_counts[inferred] += 1
        shape_counts[source_shape] += 1
        matrix[(inferred, field_type, source_shape)] += 1
        for field in SOURCE_FIELDS:
            if _present(record.get(field)):
                source_ids[field].add(_norm_path(str(record[field])))
        if len(examples[inferred]) < 5:
            examples[inferred].append(_record_example(record, evidence, index))
        if inferred == "Uncertain" and len(uncertain_examples) < 20:
            uncertain_examples.append(_record_example(record, evidence, index))

    return {
        "path": str(path),
        "stage": stage,
        "split": split,
        "container": container,
        "record_count": len(records),
        "inferred_type_counts": dict(inferred_counts),
        "source_shape_counts": dict(shape_counts),
        "field_type_counts": dict(Counter(str(record.get("question_type", "<missing>")) for record in records if isinstance(record, dict))),
        "inferred_type_field_type_source_matrix": {
            f"{inferred}/{field_type}/{shape}": count
            for (inferred, field_type, shape), count in sorted(matrix.items())
        },
        "unique_source_id_counts": {field: len(values) for field, values in source_ids.items()},
        "examples_by_inferred_type": dict(examples),
        "uncertain_examples": uncertain_examples,
    }


def _stage_file_summary(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stage in STAGES:
        stage_dir = root / "owl-questions" / stage
        stage_result: dict[str, Any] = {"path": str(stage_dir), "files": {}}
        for split in SPLITS:
            path = stage_dir / f"{split}.json"
            if path.is_file():
                stage_result["files"][split] = _audit_file(path, stage, split)
        result[stage] = stage_result
    return result


def _semantic_expectations() -> dict[str, Any]:
    return {
        "Type_I": {
            "paper_meaning": "event detection",
            "expected_answers": "event/category labels",
            "expected_source_conditions": "single and dual",
        },
        "Type_II": {
            "paper_meaning": "absolute direction/azimuth/elevation/distance estimation",
            "expected_answers": "clock direction; up/down; distance",
            "expected_source_conditions": "single and dual",
        },
        "Type_III": {
            "paper_meaning": "relative spatial reasoning",
            "expected_answers": "binary Yes/No",
            "expected_source_conditions": "dual",
        },
        "Type_IV": {
            "paper_meaning": "spatial reasoning with explicit CoT",
            "expected_answers": "explanation plus final decision",
            "expected_source_conditions": "dual",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bidepth-root", type=Path, default=DEFAULT_BIDEPTH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stages", nargs="+", default=list(STAGES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert_private_output(args.output)
    print("========== OWL BIDEPTH PAPER TYPE I-IV AUDIT ==========")
    print(f"[python] version={sys.version.split()[0]} executable={sys.executable}")
    print(f"[bidepth] root={args.bidepth_root}")
    print(f"[scope] stages={args.stages} splits={SPLITS}")
    report: dict[str, Any] = {
        "status": "ok",
        "python": {"version": sys.version, "executable": sys.executable},
        "paper_type_expectations": _semantic_expectations(),
        "loader_contract": {
            "dataset_path": "qa_data_root/stage/<split>.json",
            "whole_json_loaded": True,
            "question_type_used_for_filtering": False,
            "question_type_used_for_inference_key": True,
        },
        "files": _stage_file_summary(args.bidepth_root),
        "audit_contract": {
            "read_only": True,
            "gpu_model_loaded": False,
            "heuristic_labels_are_not_training_filters": True,
        },
    }
    # Keep only requested stages while preserving the fixed schema.
    report["files"] = {
        stage: report["files"].get(stage, {"files": {}})
        for stage in args.stages
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"[report] {args.output}")
    for stage, stage_report in report["files"].items():
        for split, item in stage_report.get("files", {}).items():
            print(
                f"[summary] {stage}/{split} records={item['record_count']} "
                f"types={item['inferred_type_counts']} "
                f"shapes={item['source_shape_counts']}"
            )
    print(f"[status] {report['status']}")


if __name__ == "__main__":
    main()
