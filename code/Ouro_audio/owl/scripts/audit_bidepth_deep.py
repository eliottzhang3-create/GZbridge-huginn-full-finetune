"""Deep, read-only audit of the remote OWL/BiDepth assets.

This audit deliberately does not extract or modify the dataset archive.  It
streams the tar.gz member table, compares every JSON reverb reference with
archive members, validates representative NPY payloads, checks dataset
partition invariants, and optionally checks source-audio roots and a local
checkout of the official OWL source.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_BIDEPTH = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/data/BiDepth"
)
DEFAULT_OUTPUT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/owl/"
    "phase1_deep_asset_audit.json"
)
DEFAULT_ARCHIVE_NAME = "reverb.tar.gz"

REQUIRED_FIELDS = (
    "audio_id",
    "reverb_id",
    "audio_id2",
    "reverb_id2",
    "question_id",
    "question_type",
    "question",
    "answer",
)
SOURCE_FIELDS = ("audio_id", "reverb_id", "audio_id2", "reverb_id2")
COT_MARKERS = (
    "because",
    "based on",
    "therefore",
    "hence",
    "relative to",
    "positioned",
    "at the back",
    "at the front",
)
SUPPORTED_AUDIO_EXTENSIONS = ("", ".wav", ".flac", ".mp3", ".ogg", ".npy")


def _norm_path(value: str) -> str:
    return value.replace("\\", "/").lstrip("./")


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    return True


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _safe_stats(values: list[float]) -> dict[str, Any]:
    finite = [value for value in values if math.isfinite(value)]
    return {
        "count": len(values),
        "finite_count": len(finite),
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
        "mean": sum(finite) / len(finite) if finite else None,
    }


def _load_records(path: Path) -> tuple[list[dict[str, Any]], str]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload, "list"
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported JSON root in {path}: {type(payload).__name__}")
    for key in ("data", "records", "questions", "annotations", "items"):
        if isinstance(payload.get(key), list):
            return payload[key], f"dict[{key}]"
    list_values = [value for value in payload.values() if isinstance(value, list)]
    if len(list_values) == 1:
        return list_values[0], "dict[single-list]"
    raise ValueError(f"Cannot identify records in {path}; keys={list(payload)}")


def _record_fingerprint(record: dict[str, Any]) -> str:
    fields = {key: record.get(key) for key in REQUIRED_FIELDS}
    payload = json.dumps(fields, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _record_summary(records: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    valid_records = [record for record in records if isinstance(record, dict)]
    invalid_record_count = len(records) - len(valid_records)
    field_counts = Counter(key for record in valid_records for key in record)
    missing_required = {
        key: sum(
            not isinstance(record, dict) or key not in record for record in records
        )
        for key in REQUIRED_FIELDS
    }
    null_counts = {
        key: sum(
            not isinstance(record, dict) or not _present(record.get(key))
            for record in records
        )
        for key in SOURCE_FIELDS
    }
    question_types = Counter(
        str(record.get("question_type", "<missing>")) for record in valid_records
    )
    source_shapes = Counter()
    audio_ids: set[str] = set()
    reverb_ids: set[str] = set()
    second_audio_ids: set[str] = set()
    second_reverb_ids: set[str] = set()
    fingerprints = Counter()
    question_ids = Counter()
    answer_lengths: list[int] = []
    question_lengths: list[int] = []
    cot_count = 0
    invalid_records: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            invalid_records.append({"index": index, "type": type(record).__name__})
            continue
        audio = record.get("audio_id")
        reverb = record.get("reverb_id")
        audio2 = record.get("audio_id2")
        reverb2 = record.get("reverb_id2")
        has_audio2 = _present(audio2)
        has_reverb2 = _present(reverb2)
        source_shapes["dual_both_second_ids"] += int(has_audio2 and has_reverb2)
        source_shapes["partial_second_ids"] += int(has_audio2 != has_reverb2)
        source_shapes["single_no_second_ids"] += int(not has_audio2 and not has_reverb2)
        if _present(audio):
            audio_ids.add(str(audio))
        if _present(reverb):
            reverb_ids.add(_norm_path(str(reverb)))
        if has_audio2:
            second_audio_ids.add(str(audio2))
        if has_reverb2:
            second_reverb_ids.add(_norm_path(str(reverb2)))
        fingerprints[_record_fingerprint(record)] += 1
        question_ids[str(record.get("question_id", "<missing>"))] += 1
        question = _as_text(record.get("question"))
        answer = _as_text(record.get("answer"))
        question_lengths.append(len(question))
        answer_lengths.append(len(answer))
        answer_lower = answer.lower()
        cot_count += int(any(marker in answer_lower for marker in COT_MARKERS))

    duplicate_fingerprints = sum(count - 1 for count in fingerprints.values() if count > 1)
    duplicate_question_ids = sum(count - 1 for count in question_ids.values() if count > 1)
    return {
        "path": str(path),
        "count": len(records),
        "container": None,
        "field_counts": dict(field_counts),
        "invalid_record_count": invalid_record_count,
        "missing_required_counts": missing_required,
        "null_or_empty_source_counts": null_counts,
        "question_type_counts": dict(question_types),
        "source_shape_counts": dict(source_shapes),
        "unique_primary_audio_ids": len(audio_ids),
        "unique_primary_reverb_ids": len(reverb_ids),
        "unique_second_audio_ids": len(second_audio_ids),
        "unique_second_reverb_ids": len(second_reverb_ids),
        "duplicate_exact_record_count": duplicate_fingerprints,
        "duplicate_question_id_count": duplicate_question_ids,
        "answer_with_cot_marker_count": cot_count,
        "answer_with_cot_marker_fraction": cot_count / len(records) if records else 0.0,
        "question_length_chars": _safe_stats([float(value) for value in question_lengths]),
        "answer_length_chars": _safe_stats([float(value) for value in answer_lengths]),
        "invalid_record_examples": invalid_records[:10],
        "first_record": records[0] if records and isinstance(records[0], dict) else None,
    }


def _inspect_questions(root: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    questions_root = root / "owl-questions"
    files = sorted(questions_root.glob("*/*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files below {questions_root}")
    stages: dict[str, Any] = {}
    records_by_split: dict[str, list[dict[str, Any]]] = {}
    all_records: list[dict[str, Any]] = []
    all_fingerprints: dict[str, list[str]] = defaultdict(list)
    for path in files:
        records, container = _load_records(path)
        summary = _record_summary(records, path)
        summary["container"] = container
        stage = path.parent.name
        split = path.stem
        stages.setdefault(stage, {})[split] = summary
        records_by_split[f"{stage}/{split}"] = records
        if split == "train":
            all_records.extend(records)
        for record in records:
            if isinstance(record, dict):
                all_fingerprints[_record_fingerprint(record)].append(f"{stage}/{split}")

    cross_split_duplicates = {
        fingerprint: sorted(set(locations))
        for fingerprint, locations in all_fingerprints.items()
        if len(set(locations)) > 1
    }
    invariants: dict[str, Any] = {}
    for stage, splits in stages.items():
        train = splits.get("train", {})
        shape_counts = train.get("source_shape_counts", {})
        types = train.get("question_type_counts", {})
        invariants[stage] = {
            "all_train_single": stage == "stage1-clsdoa"
            and shape_counts.get("single_no_second_ids", 0) == train.get("count", 0),
            "all_train_dual": stage == "stage3-mixup"
            and shape_counts.get("dual_both_second_ids", 0) == train.get("count", 0),
            "question_types": types,
        }

    return {
        "root": str(questions_root),
        "files": [str(path) for path in files],
        "stages": stages,
        "train_union_summary": _record_summary(all_records, questions_root / "<train-union>"),
        "cross_split_exact_duplicate_fingerprint_count": len(cross_split_duplicates),
        "cross_split_exact_duplicate_examples": dict(list(cross_split_duplicates.items())[:20]),
        "partition_invariants": invariants,
    }, records_by_split


def _reverb_candidates(reverb_id: str) -> list[str]:
    normalized = _norm_path(reverb_id)
    return [
        normalized,
        f"mp3d_reverb/{normalized}",
        f"mp3d_reverb/binaural/{normalized}",
    ]


def _collect_reverb_refs(records_by_split: dict[str, list[dict[str, Any]]]) -> set[str]:
    refs: set[str] = set()
    for records in records_by_split.values():
        for record in records:
            if not isinstance(record, dict):
                continue
            for field in ("reverb_id", "reverb_id2"):
                value = record.get(field)
                if _present(value):
                    refs.add(_norm_path(str(value)))
    return refs


def _collect_audio_refs(records_by_split: dict[str, list[dict[str, Any]]]) -> set[str]:
    refs: set[str] = set()
    for records in records_by_split.values():
        for record in records:
            if not isinstance(record, dict):
                continue
            for field in ("audio_id", "audio_id2"):
                value = record.get(field)
                if _present(value):
                    refs.add(_norm_path(str(value)))
    return refs


def _inspect_reverb_archive(
    archive_path: Path,
    reverb_refs: set[str],
    sample_limit: int,
) -> dict[str, Any]:
    if not archive_path.is_file():
        return {"status": "missing", "path": str(archive_path)}
    member_count = 0
    file_count = 0
    directory_count = 0
    suffix_counts: Counter[str] = Counter()
    prefix_counts: Counter[str] = Counter()
    matched_refs: dict[str, str] = {}
    sample_targets = set(sorted(reverb_refs)[:sample_limit])
    sample_payloads: dict[str, dict[str, Any]] = {}
    candidate_to_ref: dict[str, str] = {}
    for ref in reverb_refs:
        for candidate in _reverb_candidates(ref):
            candidate_to_ref.setdefault(candidate, ref)

    print(f"[reverb] streaming archive index: {archive_path}", flush=True)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            member_count += 1
            name = _norm_path(member.name)
            if member.isdir():
                directory_count += 1
                continue
            file_count += 1
            suffix_counts[Path(name).suffix.lower() or "<none>"] += 1
            prefix_counts[name.split("/", 1)[0]] += 1
            ref = candidate_to_ref.get(name)
            if ref is not None:
                matched_refs.setdefault(ref, name)
            target_ref = ref if ref in sample_targets else None
            if target_ref is not None and target_ref not in sample_payloads and name.endswith(".npy"):
                extracted = archive.extractfile(member)
                if extracted is None:
                    sample_payloads[target_ref] = {"status": "extract_failed", "member": name}
                else:
                    raw = extracted.read()
                    sample_payloads[target_ref] = _inspect_npy_bytes(raw, name)

    missing = sorted(reverb_refs - set(matched_refs))
    return {
        "status": "ok",
        "path": str(archive_path),
        "size_bytes": archive_path.stat().st_size,
        "member_count": member_count,
        "file_count": file_count,
        "directory_count": directory_count,
        "suffix_counts": dict(suffix_counts),
        "top_level_prefix_counts": dict(prefix_counts),
        "reverb_reference_count": len(reverb_refs),
        "reverb_reference_matched_count": len(matched_refs),
        "reverb_reference_missing_count": len(missing),
        "reverb_reference_coverage": len(matched_refs) / len(reverb_refs) if reverb_refs else 1.0,
        "missing_reverb_examples": missing[:50],
        "matched_reverb_examples": dict(list(sorted(matched_refs.items()))[:20]),
        "sample_npy_inspection": sample_payloads,
    }


def _inspect_npy_bytes(raw: bytes, member_name: str) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "ok", "member": member_name, "bytes": len(raw)}
    try:
        import numpy as np

        array = np.load(io.BytesIO(raw), allow_pickle=False)
        result.update(
            {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "ndim": int(array.ndim),
                "finite": bool(np.isfinite(array).all()) if np.issubdtype(array.dtype, np.number) else None,
                "numeric_stats": _safe_stats(array.astype("float64").reshape(-1).tolist())
                if np.issubdtype(array.dtype, np.number)
                else None,
            }
        )
    except Exception as exc:  # noqa: BLE001 - report the exact bad payload
        result.update({"status": "load_failed", "error": repr(exc)})
    return result


def _candidate_audio_paths(root: Path, audio_id: str) -> list[Path]:
    normalized = _norm_path(audio_id)
    return [root / f"{normalized}{extension}" for extension in SUPPORTED_AUDIO_EXTENSIONS]


def _inspect_audio_roots(audio_roots: list[Path], audio_refs: set[str]) -> dict[str, Any]:
    if not audio_roots:
        return {
            "status": "not_configured",
            "message": "No --audio-root was supplied; audio references remain unresolved.",
            "reference_count": len(audio_refs),
        }
    existing_roots = [root for root in audio_roots if root.is_dir()]
    missing_roots = [str(root) for root in audio_roots if not root.is_dir()]
    matched: dict[str, str] = {}
    for ref in sorted(audio_refs):
        for root in existing_roots:
            candidate = next((path for path in _candidate_audio_paths(root, ref) if path.is_file()), None)
            if candidate is not None:
                matched[ref] = str(candidate)
                break
    missing = sorted(audio_refs - set(matched))
    return {
        "status": "ok" if not missing and not missing_roots else "incomplete",
        "configured_roots": [str(root) for root in audio_roots],
        "missing_roots": missing_roots,
        "reference_count": len(audio_refs),
        "matched_count": len(matched),
        "missing_count": len(missing),
        "coverage": len(matched) / len(audio_refs) if audio_refs else 1.0,
        "missing_examples": missing[:50],
        "matched_examples": dict(list(sorted(matched.items()))[:20]),
    }


def _inspect_official_source(source_root: Path | None) -> dict[str, Any]:
    if source_root is None:
        return {"status": "not_configured"}
    if not source_root.is_dir():
        return {"status": "missing", "root": str(source_root)}
    candidates = sorted(source_root.rglob("spatial_audio_dataset.py"))
    train_scripts = sorted(source_root.rglob("train.sh"))
    result: dict[str, Any] = {
        "status": "ok" if candidates else "not_found",
        "root": str(source_root),
        "dataset_loader_candidates": [str(path) for path in candidates],
        "train_script_candidates": [str(path) for path in train_scripts],
    }
    loader_summaries = []
    for path in candidates[:10]:
        text = path.read_text(encoding="utf-8", errors="replace")
        loader_summaries.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "line_count": len(text.splitlines()),
                "mentions": {
                    token: text.count(token)
                    for token in (
                        "audio_id",
                        "audio_id2",
                        "reverb_id",
                        "reverb_id2",
                        "anechoic_data_root",
                        "reverb_data_root",
                        "stage1-clsdoa",
                        "stage2-single",
                        "stage3-mixup",
                    )
                },
            }
        )
    result["loader_summaries"] = loader_summaries
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bidepth-root", type=Path, default=DEFAULT_BIDEPTH)
    parser.add_argument("--reverb-archive", type=Path, default=None)
    parser.add_argument("--audio-root", type=Path, action="append", default=[])
    parser.add_argument("--owl-source-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-npy-count", type=int, default=12)
    parser.add_argument("--sha256", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("========== OWL PHASE 1 DEEP ASSET AUDIT ==========")
    print(f"[python] version={sys.version.split()[0]} executable={sys.executable}")
    print(f"[bidepth] root={args.bidepth_root}")

    questions, records_by_split = _inspect_questions(args.bidepth_root)
    reverb_refs = _collect_reverb_refs(records_by_split)
    audio_refs = _collect_audio_refs(records_by_split)
    archive_path = args.reverb_archive or (args.bidepth_root / DEFAULT_ARCHIVE_NAME)

    report: dict[str, Any] = {
        "status": "ok",
        "python": {"version": sys.version, "executable": sys.executable},
        "bidepth": questions,
        "references": {
            "unique_reverb_ids_all_splits": len(reverb_refs),
            "unique_audio_ids_all_splits": len(audio_refs),
            "reverb_examples": sorted(reverb_refs)[:20],
            "audio_examples": sorted(audio_refs)[:20],
        },
        "reverb_archive": _inspect_reverb_archive(archive_path, reverb_refs, args.sample_npy_count),
        "audio_roots": _inspect_audio_roots(args.audio_root, audio_refs),
        "official_source": _inspect_official_source(args.owl_source_root),
        "audit_contract": {
            "read_only": True,
            "archive_extracted": False,
            "gpu_model_loaded": False,
            "single_dual_inferred_from_second_ids": True,
        },
    }
    if args.sha256 and archive_path.is_file():
        print(f"[reverb] computing sha256: {archive_path}", flush=True)
        digest = hashlib.sha256()
        with archive_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        report["reverb_archive"]["sha256"] = digest.hexdigest()

    issues: list[str] = []
    if report["reverb_archive"].get("status") == "missing":
        issues.append("reverb_archive_missing")
    if report["reverb_archive"].get("reverb_reference_missing_count", 0):
        issues.append("reverb_reference_missing")
    if report["audio_roots"].get("status") != "ok":
        issues.append("audio_references_unresolved")
    if questions["cross_split_exact_duplicate_fingerprint_count"]:
        issues.append("cross_split_exact_duplicates_present")
    report["issues"] = issues
    report["status"] = "incomplete" if issues else "ok"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"[report] {args.output}")
    print(f"[status] {report['status']} issues={issues}")


if __name__ == "__main__":
    main()
