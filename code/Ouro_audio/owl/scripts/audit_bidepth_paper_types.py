"""Audit BiDepth's released partitions against the paper's Type I-IV taxonomy.

The released OWL loader does not implement Type I-IV filtering.  It loads one
whole ``<qa_data_root>/<stage>/<split>.json`` file, so this script classifies
the records by their partition and coarse fields.  It also records lexical
signals from questions/answers, but those signals are *not* treated as a
paper-Type classifier.

This distinction is important: the released ``stage3-mixup`` partition is the
paper's CoT/mixup partition, so every record in that partition must be counted
as Type IV for the curriculum audit even if its answer happens to contain
words such as ``left`` or ``front``.  Keyword matches are retained only as
diagnostic evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
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

YES_NO_RE = re.compile(r"^(yes|no)$", re.IGNORECASE)
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


def _normalized_answer(answer: Any) -> str:
    """Normalize harmless formatting while preserving answer semantics."""
    text = unicodedata.normalize("NFKC", str(answer or "")).strip().lower()
    text = text.strip(" \t\r\n\"'`([{<")
    text = text.strip(" \t\r\n\"'`.。!?！？;；:,，)]}>")
    return text


def _binary_answer_label(answer: Any) -> str | None:
    normalized = _normalized_answer(answer)
    if YES_NO_RE.fullmatch(normalized):
        return normalized.capitalize()
    return None


def _coarse_field_type(record: dict[str, Any]) -> str:
    return str(record.get("question_type", "<missing>")).strip().upper()


def _stable_field_mapping(field_type: str) -> str | None:
    """Map the released coarse task field to paper Type I/II."""
    return {
        "CLASSIFICATION": "Type_I",
        "DOA": "Type_II",
    }.get(field_type)


def _paper_assignment(stage: str, record: dict[str, Any]) -> tuple[str | None, str]:
    """Assign curriculum type using explicit release rules.

    Type III is identified by an exact Yes/No answer *and* two source IDs.
    Remaining Type I/II records use the stable released ``question_type``
    mapping.  Stage 3 is authoritative Type IV by partition contract.
    """
    shape = _source_shape(record)
    binary_label = _binary_answer_label(record.get("answer"))
    field_type = _coarse_field_type(record)
    if stage == "stage3-mixup":
        return "Type_IV", "stage3_partition_contract"
    if binary_label is not None and shape == "dual":
        return "Type_III", "dual_exact_yes_no_answer"
    mapped = _stable_field_mapping(field_type)
    if mapped is not None:
        if binary_label is not None and shape != "dual":
            return mapped, "single_source_yes_no_field_mapping_anomaly"
        return mapped, "stable_question_type_mapping"
    return None, "unknown_question_type"


def _type_evidence(record: dict[str, Any]) -> dict[str, Any]:
    question = str(record.get("question", ""))
    answer = str(record.get("answer", ""))
    question_lower = question.lower()
    answer_lower = answer.lower()
    source_shape = _source_shape(record)
    binary_label = _binary_answer_label(answer)
    answer_yes_no = binary_label is not None
    has_position = bool(POSITION_RE.search(answer))
    has_cot = bool(COT_RE.search(answer))
    has_relation_question = bool(RELATIONAL_RE.search(question))
    has_doa_question = bool(DOA_QUESTION_RE.search(question))
    has_classification_question = bool(CLASSIFICATION_QUESTION_RE.search(question))

    # These are diagnostic signals only.  They are intentionally not promoted
    # to the paper's Type I-IV labels: Type IV CoT answers routinely contain
    # direction/event words, which makes keyword classification unreliable.
    if source_shape == "dual" and answer_yes_no and (
        has_relation_question or not has_position
    ):
        heuristic_type = "Type_III_candidate"
    elif source_shape == "dual" and (has_cot or (has_relation_question and has_position)):
        heuristic_type = "Type_IV_candidate"
    elif has_position or (has_doa_question and not has_classification_question):
        heuristic_type = "Type_II_candidate"
    elif has_classification_question or not has_position:
        heuristic_type = "Type_I_candidate"
    else:
        heuristic_type = "Uncertain_candidate"

    return {
        "heuristic_type_candidate": heuristic_type,
        "source_shape": source_shape,
        "question_type_field": record.get("question_type"),
        "stable_field_type": _coarse_field_type(record),
        "stable_field_mapping": _stable_field_mapping(_coarse_field_type(record)),
        "binary_answer_label": binary_label,
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
        "heuristic_type_candidate": evidence["heuristic_type_candidate"],
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
    heuristic_counts: Counter[str] = Counter()
    matrix: Counter[tuple[str, str, str]] = Counter()
    authoritative_counts: Counter[str] = Counter()
    assignment_reasons: Counter[str] = Counter()
    shape_counts: Counter[str] = Counter()
    assignment_by_shape: Counter[tuple[str, str]] = Counter()
    assignment_by_field: Counter[tuple[str, str]] = Counter()
    binary_by_shape: Counter[tuple[str, str]] = Counter()
    binary_by_field: Counter[tuple[str, str]] = Counter()
    unknown_fields: Counter[str] = Counter()
    binary_single_source_count = 0
    source_ids: dict[str, set[str]] = {field: set() for field in SOURCE_FIELDS}
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    uncertain_examples: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            heuristic_counts["Invalid_record"] += 1
            continue
        evidence = _type_evidence(record)
        heuristic_type = evidence["heuristic_type_candidate"]
        source_shape = evidence["source_shape"]
        field_type = _coarse_field_type(record)
        heuristic_counts[heuristic_type] += 1
        authoritative_type, assignment_reason = _paper_assignment(stage, record)
        if authoritative_type is not None:
            authoritative_counts[authoritative_type] += 1
            assignment_by_shape[(authoritative_type, source_shape)] += 1
            assignment_by_field[(authoritative_type, field_type)] += 1
        assignment_reasons[assignment_reason] += 1
        binary_label = _binary_answer_label(record.get("answer"))
        if binary_label is not None:
            binary_by_shape[(binary_label, source_shape)] += 1
            binary_by_field[(binary_label, field_type)] += 1
            if source_shape != "dual":
                binary_single_source_count += 1
        if _stable_field_mapping(field_type) is None:
            unknown_fields[field_type] += 1
        shape_counts[source_shape] += 1
        matrix[(heuristic_type, field_type, source_shape)] += 1
        for field in SOURCE_FIELDS:
            if _present(record.get(field)):
                source_ids[field].add(_norm_path(str(record[field])))
        if len(examples[heuristic_type]) < 5:
            examples[heuristic_type].append(_record_example(record, evidence, index))
        if heuristic_type == "Uncertain_candidate" and len(uncertain_examples) < 20:
            uncertain_examples.append(_record_example(record, evidence, index))

    return {
        "path": str(path),
        "stage": stage,
        "split": split,
        "container": container,
        "record_count": len(records),
        "partition_contract": _partition_contract(stage),
        "authoritative_paper_type_counts": dict(authoritative_counts),
        "assignment_reason_counts": dict(assignment_reasons),
        "assignment_by_source_shape": {
            f"{paper_type}/{shape}": count
            for (paper_type, shape), count in sorted(assignment_by_shape.items())
        },
        "assignment_by_question_type_field": {
            f"{paper_type}/{field_type}": count
            for (paper_type, field_type), count in sorted(assignment_by_field.items())
        },
        "binary_yes_no_counts": {
            "total": sum(binary_by_shape.values()),
            "by_label_and_source_shape": {
                f"{label}/{shape}": count
                for (label, shape), count in sorted(binary_by_shape.items())
            },
            "by_label_and_question_type_field": {
                f"{label}/{field_type}": count
                for (label, field_type), count in sorted(binary_by_field.items())
            },
            "single_source_binary_count": binary_single_source_count,
            "dual_source_binary_count": sum(
                count for (label, shape), count in binary_by_shape.items() if shape == "dual"
            ),
        },
        "stable_field_mapping_validation": {
            "mapping": {
                "CLASSIFICATION": "Type_I",
                "DOA": "Type_II",
            },
            "unknown_question_type_field_counts": dict(unknown_fields),
            "known_question_type_field_total": sum(
                count for field_type, count in Counter(
                    _coarse_field_type(record)
                    for record in records
                    if isinstance(record, dict)
                ).items()
                if _stable_field_mapping(field_type) is not None
            ),
        },
        "heuristic_signal_counts_not_for_training": dict(heuristic_counts),
        "source_shape_counts": dict(shape_counts),
        "field_type_counts": dict(Counter(str(record.get("question_type", "<missing>")) for record in records if isinstance(record, dict))),
        "heuristic_signal_field_type_source_matrix": {
            f"{inferred}/{field_type}/{shape}": count
            for (inferred, field_type, shape), count in sorted(matrix.items())
        },
        "unique_source_id_counts": {field: len(values) for field, values in source_ids.items()},
        "examples_by_heuristic_signal": dict(examples),
        "uncertain_examples": uncertain_examples,
    }


def _authoritative_type(stage: str, record: dict[str, Any]) -> str | None:
    """Return only labels justified by the released partition contract.

    Stage 3 is a curriculum partition, not a bag of keyword-matched answers:
    its complete released split is the COT/mixup Type-IV data.  For the
    single-source cls/doa warmup, the coarse field has the stable mapping
    CLASSIFICATION -> Type I and DOA -> Type II.  For the current stage2 file,
    this mapping is applied record-by-record; the partition itself is not
    called "paper Stage 2" because it contains both single and dual records.
    """
    return _paper_assignment(stage, record)[0]


def _partition_contract(stage: str) -> dict[str, Any]:
    if stage == "stage1-clsdoa":
        return {
            "status": "authoritative_for_coarse_field_mapping",
            "paper_types": ["Type_I", "Type_II"],
            "mapping": {"CLASSIFICATION": "Type_I", "DOA": "Type_II"},
        }
    if stage == "stage3-mixup":
        return {
            "status": "authoritative_partition_label",
            "paper_types": ["Type_IV"],
            "mapping": "all records are treated as Type_IV for curriculum accounting",
        }
    return {
        "status": "mixed_release_partition_recordwise_mapping",
        "paper_types": ["Type_I", "Type_II", "Type_III_candidate_if_dual_exact_yes_no"],
        "mapping": {
            "dual_exact_yes_no_answer": "Type_III",
            "CLASSIFICATION": "Type_I",
            "DOA": "Type_II",
        },
        "reason": "stage2-single contains both single and dual records; its directory name is not treated as paper Stage 2",
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
    stage1_path = root / "owl-questions" / "stage1-clsdoa" / "train.json"
    stage2_path = root / "owl-questions" / "stage2-single" / "train.json"
    if stage1_path.is_file() and stage2_path.is_file():
        stage1_records, _ = _load_records(stage1_path)
        stage2_records, _ = _load_records(stage2_path)
        result["stage2-single"]["train_delta_after_stage1"] = _stage2_delta_summary(
            stage1_records, stage2_records
        )
        result["stage12_train_composition"] = _stage12_composition_summary(
            stage1_records, stage2_records
        )
    return result


def _stage2_delta_summary(
    stage1_records: list[dict[str, Any]], stage2_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Describe records newly appearing in stage2 relative to stage1.

    This is the useful audit for a cumulative release: it avoids assigning
    the whole 599,831-record stage2 file to paper Stage 2 when 330,714 records
    are already present in stage1.
    """
    stage1_counter = Counter(
        _canonical_record(record) for record in stage1_records if isinstance(record, dict)
    )
    stage2_counter = Counter(
        _canonical_record(record) for record in stage2_records if isinstance(record, dict)
    )
    exact_record_delta_counter = stage2_counter - stage1_counter
    stage1_source_tuples = {
        _source_tuple_key(record)
        for record in stage1_records
        if isinstance(record, dict)
    }
    source_tuple_delta_records = [
        record
        for record in stage2_records
        if isinstance(record, dict)
        and _source_tuple_key(record) not in stage1_source_tuples
    ]
    field_counts: Counter[str] = Counter()
    shape_counts: Counter[str] = Counter()
    paper_type_counts: Counter[str] = Counter()
    paper_type_shape_counts: Counter[tuple[str, str]] = Counter()
    yes_no_count = 0
    cot_signal_count = 0
    examples: list[dict[str, Any]] = []
    for encoded, count in exact_record_delta_counter.items():
        record = json.loads(encoded)
        evidence = _type_evidence(record)
        field_counts[str(record.get("question_type", "<missing>"))] += count
        shape_counts[evidence["source_shape"]] += count
        paper_type, _ = _paper_assignment("stage2-single", record)
        if paper_type is not None:
            paper_type_counts[paper_type] += count
            paper_type_shape_counts[(paper_type, evidence["source_shape"])] += count
        yes_no_count += int(evidence["answer_is_bare_yes_no"]) * count
        cot_signal_count += int(evidence["answer_has_cot_marker"]) * count
        if len(examples) < 10:
            examples.append(_record_example(record, evidence, -1))
    return {
        "stage1_record_count": len(stage1_records),
        "stage2_record_count": len(stage2_records),
        "exact_record_delta_count": sum(exact_record_delta_counter.values()),
        "stage2_contains_stage1_as_exact_record_multiset": not (Counter(
            _canonical_record(record) for record in stage1_records if isinstance(record, dict)
        ) - Counter(
            _canonical_record(record) for record in stage2_records if isinstance(record, dict)
        )),
        "stage1_source_tuple_count": len(stage1_source_tuples),
        "stage2_source_tuple_delta_record_count": len(source_tuple_delta_records),
        "stage2_source_tuple_delta_source_tuple_count": len({
            _source_tuple_key(record) for record in source_tuple_delta_records
        }),
        "stage2_source_tuple_delta_composition": _composition_for_records(
            source_tuple_delta_records, "stage2-single"
        ),
        "exact_record_delta_field_type_counts": dict(field_counts),
        "exact_record_delta_source_shape_counts": dict(shape_counts),
        "exact_record_delta_paper_type_counts": dict(paper_type_counts),
        "exact_record_delta_paper_type_source_shape_counts": {
            f"{paper_type}/{shape}": count
            for (paper_type, shape), count in sorted(paper_type_shape_counts.items())
        },
        "delta_bare_yes_no_count": yes_no_count,
        "delta_cot_lexical_signal_count_not_authoritative": cot_signal_count,
        "delta_examples": examples,
        "interpretation": "exact_record_delta and source_tuple_delta are different units. The source_tuple_delta is the relevant measure for detecting new acoustic source configurations.",
    }


def _source_tuple_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    return tuple(
        _norm_path(str(record.get(field))) if _present(record.get(field)) else "<none>"
        for field in SOURCE_FIELDS
    )


def _composition_for_records(records: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    record_counter: Counter[str] = Counter()
    tuple_counter: Counter[tuple[str, str, str, str]] = Counter()
    record_type_counts: Counter[str] = Counter()
    record_shape_counts: Counter[str] = Counter()
    tuple_shape_counts: Counter[str] = Counter()
    tuple_type_counts: Counter[str] = Counter()
    answer_by_shape: dict[str, Counter[str]] = defaultdict(Counter)
    answer_by_shape_and_field: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for record in records:
        if not isinstance(record, dict):
            continue
        record_counter[_canonical_record(record)] += 1
        tuple_key = _source_tuple_key(record)
        tuple_counter[tuple_key] += 1
        shape = _source_shape(record)
        record_shape_counts[shape] += 1
        tuple_shape_counts[shape] += 0
        answer = _normalized_answer(record.get("answer"))
        answer_by_shape[shape][answer] += 1
        answer_by_shape_and_field[(shape, _coarse_field_type(record))][answer] += 1
        paper_type, _ = _paper_assignment(stage, record)
        if paper_type is not None:
            record_type_counts[paper_type] += 1
            tuple_type_counts[paper_type] += 0
    # A source tuple can generate multiple questions.  For tuple-level counts,
    # assign a type only when all records sharing that tuple agree.
    tuple_records: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if isinstance(record, dict):
            tuple_records[_source_tuple_key(record)].append(record)
    tuple_type_conflicts = 0
    for tuple_key, tuple_items in tuple_records.items():
        shapes = {_source_shape(item) for item in tuple_items}
        tuple_shape_counts[next(iter(shapes)) if len(shapes) == 1 else "mixed"] += 1
        types = {_paper_assignment(stage, item)[0] for item in tuple_items}
        types.discard(None)
        if len(types) == 1:
            tuple_type_counts[next(iter(types))] += 1
        elif len(types) > 1:
            tuple_type_conflicts += 1
    return {
        "record_count": len([record for record in records if isinstance(record, dict)]),
        "unique_record_count": len(record_counter),
        "source_tuple_count": len(tuple_counter),
        "source_tuple_reuse_histogram": dict(Counter(tuple_counter.values())),
        "record_source_shape_counts": dict(record_shape_counts),
        "source_tuple_shape_counts": dict(tuple_shape_counts),
        "record_paper_type_counts": dict(record_type_counts),
        "source_tuple_paper_type_counts": dict(tuple_type_counts),
        "source_tuple_paper_type_conflict_count": tuple_type_conflicts,
        "answer_form_counts_by_source_shape": {
            shape: dict(counter.most_common(50))
            for shape, counter in sorted(answer_by_shape.items())
        },
        "answer_form_counts_by_source_shape_and_question_type": {
            f"{shape}/{field_type}": dict(counter.most_common(50))
            for (shape, field_type), counter in sorted(answer_by_shape_and_field.items())
        },
    }


def _stage12_composition_summary(
    stage1_records: list[dict[str, Any]], stage2_records: list[dict[str, Any]]
) -> dict[str, Any]:
    stage1_counter = Counter(
        _canonical_record(record) for record in stage1_records if isinstance(record, dict)
    )
    stage2_counter = Counter(
        _canonical_record(record) for record in stage2_records if isinstance(record, dict)
    )
    delta_records: list[dict[str, Any]] = []
    for encoded, count in (stage2_counter - stage1_counter).items():
        delta_records.extend([json.loads(encoded)] * count)
    union_records: list[dict[str, Any]] = []
    for encoded, count in (stage1_counter | stage2_counter).items():
        union_records.extend([json.loads(encoded)] * count)
    return {
        "units_note": "record_count is JSON QA records; source_tuple_count is unique audio/reverb source tuple count. Both are reported because paper dataset sizes may use different units.",
        "stage1_file": _composition_for_records(stage1_records, "stage1-clsdoa"),
        "stage2_file": _composition_for_records(stage2_records, "stage2-single"),
        "stage2_delta_after_stage1": _composition_for_records(delta_records, "stage2-single"),
        "stage1_union_stage2": _composition_for_records(union_records, "stage2-single"),
        "paper_target_reference": {
            "stage1": {
                "single_source_about": 270000,
                "dual_source_about": 270000,
                "types": ["Type_I", "Type_II"],
            },
            "stage2": {
                "new_dual_source_about": 300000,
                "type": "Type_III",
            },
            "cumulative_stage1_plus_stage2": {
                "single_source_about": 270000,
                "dual_source_about": 570000,
            },
            "comparison_warning": "Compare both JSON record counts and unique source-tuple counts; do not silently equate them.",
        },
    }


def _semantic_expectations() -> dict[str, Any]:
    return {
        "Type_I": {
            "paper_meaning": "event detection",
            "expected_answers": "event/category labels",
            "expected_source_conditions": "single and dual",
        },
        "released_partition_contract": {
            "stage1-clsdoa": {
                "paper_role": "single-source Type I/II warmup partition",
                "authoritative_mapping": {
                    "CLASSIFICATION": "Type_I",
                    "DOA": "Type_II",
                },
            },
            "stage2-single": {
                "paper_role": "mixed/cumulative release partition; do not map the whole file to paper Stage 2",
                "authoritative_mapping": None,
                "reason": "the observed file contains both single and dual source records and both CLASSIFICATION and DOA records",
            },
            "stage3-mixup": {
                "paper_role": "paper Stage 3 CoT/mixup partition",
                "authoritative_mapping": "Type_IV",
                "reason": "partition semantics override lexical answer-word heuristics",
            },
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
    all_file_reports = _stage_file_summary(args.bidepth_root)
    report: dict[str, Any] = {
        "status": "ok",
        "python": {"version": sys.version, "executable": sys.executable},
        "paper_type_expectations": _semantic_expectations(),
        "loader_contract": {
            "dataset_path": "qa_data_root/stage/<split>.json",
            "whole_json_loaded": True,
            "question_type_used_for_filtering": False,
            "question_type_used_for_inference_key": True,
            "paper_type_filter_in_official_loader": False,
        },
        "files": all_file_reports,
        "stage12_train_composition": all_file_reports.get("stage12_train_composition"),
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
                f"paper_types={item['authoritative_paper_type_counts']} "
                f"heuristic_signals={item['heuristic_signal_counts_not_for_training']} "
                f"shapes={item['source_shape_counts']}"
            )
    composition = report.get("stage12_train_composition")
    if composition:
        print(
            "[composition] stage1="
            f"{composition['stage1_file']['record_source_shape_counts']} "
            "stage2="
            f"{composition['stage2_file']['record_source_shape_counts']} "
            "stage2_delta="
            f"{composition['stage2_delta_after_stage1']['record_source_shape_counts']}"
        )
        print(
            "[composition] paper_types stage1="
            f"{composition['stage1_file']['record_paper_type_counts']} "
            "stage2="
            f"{composition['stage2_file']['record_paper_type_counts']} "
            "stage2_delta="
            f"{composition['stage2_delta_after_stage1']['record_paper_type_counts']}"
        )
    stage2_delta = report.get("files", {}).get("stage2-single", {}).get(
        "train_delta_after_stage1", {}
    )
    if stage2_delta:
        print(
            "[delta] exact_record_delta="
            f"{stage2_delta.get('exact_record_delta_count')} "
            "source_tuple_delta_records="
            f"{stage2_delta.get('stage2_source_tuple_delta_record_count')} "
            "source_tuple_delta_unique="
            f"{stage2_delta.get('stage2_source_tuple_delta_source_tuple_count')}"
        )
        print(
            "[delta] source_tuple_delta_types="
            f"{stage2_delta.get('stage2_source_tuple_delta_composition', {}).get('record_paper_type_counts', {})} "
            "shapes="
            f"{stage2_delta.get('stage2_source_tuple_delta_composition', {}).get('record_source_shape_counts', {})}"
        )
    print(f"[status] {report['status']}")


if __name__ == "__main__":
    main()
