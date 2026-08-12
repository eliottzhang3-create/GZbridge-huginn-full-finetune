"""Read-only audit of extracted BiDepth RIR files and the official OWL loader.

This is the fast follow-up to ``audit_bidepth_deep.py``.  The compressed
archive remains the source of truth, but once it has been extracted we can
audit references with direct path lookups instead of rescanning a gzip stream
from byte zero on every run.

The report intentionally separates:
* hard integrity failures (missing files, unreadable NPY payloads, malformed
  official loader), and
* findings that need interpretation (source reuse across curriculum stages,
  cross-split reuse, or an audio root that was not supplied).

No model is loaded and no dataset file is modified.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from audit_bidepth_deep import (
    DEFAULT_BIDEPTH,
    _collect_audio_refs,
    _collect_reverb_refs,
    _inspect_questions,
    _norm_path,
    _present,
)


DEFAULT_REVERB_ROOT = DEFAULT_BIDEPTH / "reverb_extracted"
DEFAULT_OWL_SOURCE_ROOT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/OWL"
)
DEFAULT_OUTPUT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/owl/"
    "phase1_decompressed_asset_audit.json"
)

SOURCE_FIELDS = ("audio_id", "reverb_id", "audio_id2", "reverb_id2")
LOADER_TOKENS = (
    "audio_id",
    "audio_id2",
    "reverb_id",
    "reverb_id2",
    "stage",
    "anechoic_data_root",
    "reverb_data_root",
    "fix_length_audio",
    "sample_rate",
    "np.load",
    "torch.stft",
    "fftconvolve",
    "convolve",
    "qformer",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _safe_stats(values: Iterable[float]) -> dict[str, Any]:
    values = list(values)
    finite = [value for value in values if math.isfinite(value)]
    return {
        "count": len(values),
        "finite_count": len(finite),
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
        "mean": sum(finite) / len(finite) if finite else None,
    }


def _reverb_candidates(root: Path, reverb_id: str) -> list[Path]:
    """Resolve both dataset-style and archive-member-style references."""

    normalized = _norm_path(str(reverb_id))
    candidates = [
        root / normalized,
        root / "mp3d_reverb" / normalized,
        root / "mp3d_reverb" / "binaural" / normalized,
    ]
    if normalized.startswith("mp3d_reverb/"):
        without_prefix = normalized[len("mp3d_reverb/") :]
        candidates.extend(
            [
                root / without_prefix,
                root / "mp3d_reverb" / "binaural" / without_prefix,
            ]
        )
    if normalized.startswith("mp3d_reverb/binaural/"):
        without_prefix = normalized[len("mp3d_reverb/binaural/") :]
        candidates.append(root / "mp3d_reverb" / "binaural" / without_prefix)
    if Path(normalized).suffix == "":
        candidates.extend(path.with_suffix(".npy") for path in list(candidates))
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def _resolve_reverb(root: Path, ref: str) -> Path | None:
    for candidate in _reverb_candidates(root, ref):
        if candidate.is_file():
            return candidate
    return None


def _source_key(record: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        _norm_path(str(record.get(field))) if _present(record.get(field)) else ""
        for field in SOURCE_FIELDS
    )


def _canonical_record(record: dict[str, Any]) -> str:
    return json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest_strings(values: Iterable[str], *, sort_values: bool) -> str:
    import hashlib as _hashlib

    ordered = sorted(values) if sort_values else list(values)
    digest = _hashlib.sha256()
    for value in ordered:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _split_equality_report(
    records_by_split: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Compare val/test as ordered sequences and as record multisets."""

    result: dict[str, Any] = {}
    for stage in sorted({name.split("/", 1)[0] for name in records_by_split}):
        val_name = f"{stage}/val"
        test_name = f"{stage}/test"
        if val_name not in records_by_split or test_name not in records_by_split:
            continue
        val_records = [
            record for record in records_by_split[val_name] if isinstance(record, dict)
        ]
        test_records = [
            record for record in records_by_split[test_name] if isinstance(record, dict)
        ]
        val_keys = [_canonical_record(record) for record in val_records]
        test_keys = [_canonical_record(record) for record in test_records]
        val_counter = Counter(val_keys)
        test_counter = Counter(test_keys)
        intersection_counter = val_counter & test_counter
        only_val_counter = val_counter - test_counter
        only_test_counter = test_counter - val_counter
        result[stage] = {
            "val_count": len(val_records),
            "test_count": len(test_records),
            "val_unique_count": len(val_counter),
            "test_unique_count": len(test_counter),
            "val_internal_duplicate_extra_count": sum(
                count - 1 for count in val_counter.values() if count > 1
            ),
            "test_internal_duplicate_extra_count": sum(
                count - 1 for count in test_counter.values() if count > 1
            ),
            "multiset_intersection_count": sum(intersection_counter.values()),
            "unique_record_intersection_count": len(intersection_counter),
            "records_only_in_val_count": sum(only_val_counter.values()),
            "records_only_in_test_count": sum(only_test_counter.values()),
            "ordered_sequence_equal": val_keys == test_keys,
            "record_multiset_equal": val_counter == test_counter,
            "record_set_equal": set(val_keys) == set(test_keys),
            "val_ordered_sha256": _digest_strings(val_keys, sort_values=False),
            "test_ordered_sha256": _digest_strings(test_keys, sort_values=False),
            "val_canonical_set_sha256": _digest_strings(val_counter.keys(), sort_values=True),
            "test_canonical_set_sha256": _digest_strings(test_counter.keys(), sort_values=True),
        }
    return result


def _primary_key(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    return _norm_path(str(value)) if _present(value) else ""


def _overlap_report(
    records_by_split: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Report reuse without declaring curriculum reuse to be corruption."""

    key_builders = {
        "exact_record": lambda record: json.dumps(
            {key: record.get(key) for key in (
                "audio_id", "reverb_id", "audio_id2", "reverb_id2",
                "question_id", "question_type", "question", "answer",
            )},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "source_tuple": _source_key,
        "primary_audio_id": lambda record: _primary_key(record, "audio_id"),
        "primary_reverb_id": lambda record: _primary_key(record, "reverb_id"),
    }
    locations: dict[str, dict[Any, set[str]]] = {
        name: defaultdict(set) for name in key_builders
    }
    exact_record_locations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for split_name, records in records_by_split.items():
        for record_index, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            for name, builder in key_builders.items():
                key = builder(record)
                if key not in ("", ("", "", "", "")):
                    locations[name][key].add(split_name)
            exact_key = key_builders["exact_record"](record)
            exact_record_locations[exact_key].append(
                {
                    "split": split_name,
                    "record_index": record_index,
                    "question_id": record.get("question_id"),
                    "question_type": record.get("question_type"),
                    "source_tuple": list(_source_key(record)),
                }
            )

    same_stage: dict[str, Any] = {}
    cross_stage: dict[str, Any] = {}
    for key_name, by_key in locations.items():
        same_stage_count = 0
        cross_stage_count = 0
        examples: list[dict[str, Any]] = []
        for key, split_names in by_key.items():
            if len(split_names) < 2:
                continue
            stages = {split_name.split("/", 1)[0] for split_name in split_names}
            if len(stages) == 1:
                same_stage_count += 1
            else:
                cross_stage_count += 1
            if len(examples) < 20:
                examples.append({"key": str(key), "splits": sorted(split_names)})
        same_stage[key_name] = same_stage_count
        cross_stage[key_name] = cross_stage_count
        same_stage.setdefault("examples", {})[key_name] = examples
    exact_pair_counts: Counter[str] = Counter()
    exact_duplicate_examples: list[dict[str, Any]] = []
    for fingerprint, items in exact_record_locations.items():
        split_names = sorted({item["split"] for item in items})
        if len(split_names) < 2:
            continue
        for index, left in enumerate(split_names):
            for right in split_names[index + 1 :]:
                left_stage = left.split("/", 1)[0]
                right_stage = right.split("/", 1)[0]
                if left_stage == right_stage:
                    exact_pair_counts[f"{left} <-> {right}"] += 1
        stages = {split_name.split("/", 1)[0] for split_name in split_names}
        if len(stages) == 1 and len(exact_duplicate_examples) < 50:
            exact_duplicate_examples.append(
                {
                    "fingerprint": hashlib.sha1(fingerprint.encode("utf-8")).hexdigest(),
                    "locations": [
                        item for item in items
                        if item["split"].split("/", 1)[0] in stages
                    ],
                }
            )

    return {
        "same_stage_split_overlap_key_counts": same_stage,
        "cross_stage_overlap_key_counts": cross_stage,
        "same_stage_exact_record_duplicate_pair_counts": dict(exact_pair_counts),
        "same_stage_exact_record_duplicate_examples": exact_duplicate_examples,
        "interpretation": (
            "Exact record overlap within train/val/test is a hard leakage finding. "
            "Source tuple overlap across curriculum stages can be intentional, "
            "because one source may receive different questions or supervision."
        ),
    }


def _load_npy(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "status": "ok",
        "size_bytes": path.stat().st_size,
    }
    try:
        import numpy as np

        array = np.load(path, mmap_mode="r", allow_pickle=False)
        numeric = np.issubdtype(array.dtype, np.number)
        if numeric and array.size:
            finite_mask = np.isfinite(array)
            finite_values = array[finite_mask].astype("float64")
            stats = {
                "count": int(array.size),
                "finite_count": int(finite_values.size),
                "min": float(finite_values.min()) if finite_values.size else None,
                "max": float(finite_values.max()) if finite_values.size else None,
                "mean": float(finite_values.mean()) if finite_values.size else None,
                "std": float(finite_values.std()) if finite_values.size else None,
            }
            finite = bool(finite_values.size == array.size)
        else:
            stats = None
            finite = None if not numeric else True
        result.update(
            {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "ndim": int(array.ndim),
                "numeric": bool(numeric),
                "finite": finite,
                "stats": stats,
            }
        )
        if not numeric:
            result["status"] = "non_numeric"
    except Exception as exc:  # noqa: BLE001 - preserve exact payload failure
        result.update({"status": "load_failed", "error": repr(exc)})
    return result


def _representative_refs(records_by_split: dict[str, list[dict[str, Any]]], limit: int) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for split_name in sorted(records_by_split):
        records = records_by_split[split_name]
        for record in records:
            if not isinstance(record, dict):
                continue
            for field in ("reverb_id", "reverb_id2"):
                value = record.get(field)
                if not _present(value):
                    continue
                ref = _norm_path(str(value))
                if ref not in seen:
                    selected.append(ref)
                    seen.add(ref)
                if len(selected) >= limit:
                    return selected
    return selected


def _inspect_extracted_reverb(
    root: Path,
    records_by_split: dict[str, list[dict[str, Any]]],
    sample_limit: int,
) -> dict[str, Any]:
    started = time.monotonic()
    refs = sorted(_collect_reverb_refs(records_by_split))
    if not root.is_dir():
        return {"status": "missing", "root": str(root), "reference_count": len(refs)}

    by_split: dict[str, Any] = {}
    all_missing: set[str] = set()
    matched_paths: dict[str, str] = {}
    for split_name, records in sorted(records_by_split.items()):
        counters: Counter[str] = Counter()
        missing_examples: list[dict[str, Any]] = []
        for record_index, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            for field in ("reverb_id", "reverb_id2"):
                value = record.get(field)
                if not _present(value):
                    continue
                ref = _norm_path(str(value))
                counters["references"] += 1
                resolved = _resolve_reverb(root, ref)
                if resolved is None:
                    counters["missing"] += 1
                    all_missing.add(ref)
                    if len(missing_examples) < 20:
                        missing_examples.append(
                            {"record_index": record_index, "field": field, "ref": ref}
                        )
                else:
                    counters["matched"] += 1
                    matched_paths.setdefault(ref, str(resolved))
                    counters["npy"] += int(resolved.suffix.lower() == ".npy")
        by_split[split_name] = {
            **dict(counters),
            "coverage": counters["matched"] / counters["references"]
            if counters["references"] else 1.0,
            "missing_examples": missing_examples,
        }

    npy_files = list(root.rglob("*.npy"))
    room_dirs = {
        str(path.parent)
        for path in npy_files
        if path.parent.name and path.parent.parent.name
    }
    suffix_counts = Counter(path.suffix.lower() for path in root.rglob("*") if path.is_file())
    representative: dict[str, Any] = {}
    for ref in _representative_refs(records_by_split, sample_limit):
        resolved = _resolve_reverb(root, ref)
        representative[ref] = (
            {"status": "missing"} if resolved is None else _load_npy(resolved)
        )
    bad_samples = [
        ref for ref, item in representative.items() if item.get("status") != "ok"
    ]
    elapsed = time.monotonic() - started
    return {
        "status": "ok" if not all_missing and not bad_samples else "incomplete",
        "root": str(root),
        "scan_seconds": elapsed,
        "reference_count": len(refs),
        "unique_reference_matched_count": len(set(refs) - all_missing),
        "unique_reference_missing_count": len(all_missing),
        "unique_reference_coverage": (
            (len(refs) - len(all_missing)) / len(refs) if refs else 1.0
        ),
        "missing_reference_examples": sorted(all_missing)[:50],
        "matched_reference_examples": dict(list(sorted(matched_paths.items()))[:20]),
        "by_split": by_split,
        "filesystem": {
            "npy_file_count": len(npy_files),
            "room_directory_count": len(room_dirs),
            "suffix_counts": dict(suffix_counts),
            "example_files": [str(path) for path in sorted(npy_files)[:20]],
        },
        "representative_npy": representative,
    }


def _ast_names(tree: ast.AST) -> dict[str, list[str]]:
    functions: list[str] = []
    classes: list[str] = []
    calls: Counter[str] = Counter()
    imports: list[str] = []
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
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    return {
        "functions": sorted(functions),
        "classes": sorted(classes),
        "calls": dict(calls),
        "imports": sorted(set(imports)),
    }


def _context_lines(text: str, token: str, limit: int = 12) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if token.lower() in line.lower() and len(result) < limit:
            result.append({"line": line_number, "text": line.strip()})
    return result


def _inspect_official_loader(source_root: Path) -> dict[str, Any]:
    candidates = sorted(source_root.rglob("spatial_audio_dataset.py")) if source_root.is_dir() else []
    if not candidates:
        return {"status": "missing", "source_root": str(source_root)}
    path = candidates[0]
    text = path.read_text(encoding="utf-8", errors="replace")
    result: dict[str, Any] = {
        "status": "ok",
        "path": str(path),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "line_count": len(text.splitlines()),
        "byte_count": len(text.encode("utf-8")),
        "token_hits": {token: text.count(token) for token in LOADER_TOKENS},
        "token_context": {
            token: _context_lines(text, token) for token in LOADER_TOKENS if token in text
        },
    }
    try:
        tree = ast.parse(text, filename=str(path))
        result["ast"] = _ast_names(tree)
        result["ast_parse"] = "ok"
    except SyntaxError as exc:
        result["ast_parse"] = "failed"
        result["ast_error"] = repr(exc)
        result["status"] = "invalid"
        return result

    semantic_checks = {
        "has_primary_and_second_source_ids": all(
            token in text for token in ("audio_id", "audio_id2", "reverb_id", "reverb_id2")
        ),
        "has_stage_branching": bool(re.search(r"\bstage\b|stage1|stage2|stage3", text)),
        "loads_npy_or_npz": bool(re.search(r"np\.load|numpy\.load", text)),
        "performs_rir_convolution": bool(
            re.search(r"fftconvolve|convolve|irfft|stft", text, re.IGNORECASE)
        ),
        "handles_fixed_audio_length": "fix_length_audio" in text,
        "mentions_32khz_or_32000": bool(re.search(r"32\s*000|32k|32000", text, re.IGNORECASE)),
        "loader_has_dataset_class": any(
            name.endswith("Dataset") for name in result["ast"]["classes"]
        ),
    }
    result["semantic_checks"] = semantic_checks
    result["related_files"] = [str(item) for item in candidates]
    return result


def _inspect_official_components(source_root: Path) -> dict[str, Any]:
    """Hash and summarize the source files that define Stage 1/2 semantics."""

    relative_candidates = (
        "seld_cot/owl/dataset/spatial_audio_dataset.py",
        "seld_cot/owl/finetune_seld.py",
        "seld_cot/owl/model/slam_model_seld.py",
        "seld_cot/owl/scripts/train.sh",
        "src/slam_llm/models/SAGE/sage.py",
        "src/slam_llm/models/SAGE/vision_transformer.py",
        "src/slam_llm/models/projector.py",
        "src/slam_llm/models/slam_model.py",
        "src/slam_llm/models/encoder.py",
    )
    summaries: dict[str, Any] = {}
    missing: list[str] = []
    for relative in relative_candidates:
        path = source_root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        item: dict[str, Any] = {
            "path": str(path),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "line_count": len(text.splitlines()),
            "byte_count": len(text.encode("utf-8")),
            "key_hits": {
                token: text.lower().count(token.lower())
                for token in (
                    "stage",
                    "audio_id",
                    "reverb_id",
                    "qformer",
                    "query_len",
                    "freeze_encoder",
                    "freeze_llm",
                    "total_steps",
                    "num_epochs",
                    "fix_length_audio",
                    "target_frame",
                    "sample_rate",
                    "lora",
                )
            },
            "context": {
                token: _context_lines(text, token, limit=6)
                for token in (
                    "stage",
                    "qformer",
                    "freeze_encoder",
                    "freeze_llm",
                    "total_steps",
                    "query_len",
                    "target_frame",
                )
                if token.lower() in text.lower()
            },
        }
        try:
            tree = ast.parse(text, filename=str(path))
            item["ast_parse"] = "ok"
            item["ast"] = _ast_names(tree)
        except SyntaxError as exc:
            item["ast_parse"] = "not_applicable_or_failed"
            item["ast_error"] = repr(exc)
        summaries[relative] = item
    return {
        "status": "ok" if not missing else "incomplete",
        "source_root": str(source_root),
        "files": summaries,
        "missing_expected_files": missing,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bidepth-root", type=Path, default=DEFAULT_BIDEPTH)
    parser.add_argument("--reverb-root", type=Path, default=DEFAULT_REVERB_ROOT)
    parser.add_argument("--owl-source-root", type=Path, default=DEFAULT_OWL_SOURCE_ROOT)
    parser.add_argument("--audio-root", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-npy-count", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("========== OWL PHASE 1 DECOMPRESSED ASSET + LOADER AUDIT ==========")
    print(f"[python] version={sys.version.split()[0]} executable={sys.executable}")
    print(f"[bidepth] root={args.bidepth_root}")
    print(f"[reverb] extracted_root={args.reverb_root}")
    print(f"[owl] source_root={args.owl_source_root}")

    questions, records_by_split = _inspect_questions(args.bidepth_root)
    reverb_refs = _collect_reverb_refs(records_by_split)
    audio_refs = _collect_audio_refs(records_by_split)
    extracted = _inspect_extracted_reverb(
        args.reverb_root, records_by_split, args.sample_npy_count
    )
    loader = _inspect_official_loader(args.owl_source_root)

    report: dict[str, Any] = {
        "status": "ok",
        "python": {"version": sys.version, "executable": sys.executable},
        "bidepth": questions,
        "references": {
            "unique_reverb_ids_all_splits": len(reverb_refs),
            "unique_audio_ids_all_splits": len(audio_refs),
        },
        "extracted_reverb": extracted,
        "audio_roots": {
            "status": "not_configured",
            "message": (
                "Audio source roots are separate from the extracted RIR root. "
                "Pass --audio-root only when the anechoic source tree is available."
            ),
            "reference_count": len(audio_refs),
        },
        "overlap": _overlap_report(records_by_split),
        "split_equality": _split_equality_report(records_by_split),
        "official_loader": loader,
        "official_components": _inspect_official_components(args.owl_source_root),
        "audit_contract": {
            "read_only": True,
            "uses_extracted_rir_direct_lookups": True,
            "archive_extracted_by_audit": False,
            "gpu_model_loaded": False,
            "stage_semantics_inferred_from_official_source": True,
        },
    }
    if args.audio_root:
        from audit_bidepth_deep import _inspect_audio_roots

        report["audio_roots"] = _inspect_audio_roots(args.audio_root, audio_refs)

    issues: list[str] = []
    findings: list[str] = []
    if extracted.get("status") == "missing":
        issues.append("extracted_reverb_root_missing")
    elif extracted.get("unique_reference_missing_count", 0):
        issues.append("extracted_reverb_reference_missing")
    if extracted.get("representative_npy"):
        bad = [
            item for item in extracted["representative_npy"].values()
            if item.get("status") != "ok"
        ]
        if bad:
            issues.append("representative_npy_invalid")
    if loader.get("status") != "ok":
        issues.append("official_loader_unavailable_or_invalid")
    if report["official_components"].get("status") != "ok":
        issues.append("official_component_source_incomplete")
    if questions["same_stage_split_duplicate_fingerprint_count"]:
        issues.append("same_stage_exact_record_duplicates_present")
    for stage, equality in report["split_equality"].items():
        if equality.get("record_multiset_equal"):
            findings.append(f"{stage}_val_test_are_exactly_the_same_record_multiset")
    if report["overlap"]["same_stage_split_overlap_key_counts"]["source_tuple"]:
        findings.append("same_stage_source_tuple_reuse_present")
    if report["overlap"]["cross_stage_overlap_key_counts"]["source_tuple"]:
        findings.append("cross_stage_source_tuple_reuse_present_needs_curriculum_interpretation")
    if report["audio_roots"].get("status") != "ok":
        findings.append("audio_source_roots_not_yet_verified")
    report["issues"] = issues
    report["findings"] = findings
    report["status"] = "incomplete" if issues else "ok"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(report), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"[report] {args.output}")
    print(f"[status] {report['status']} issues={issues} findings={findings}")


if __name__ == "__main__":
    main()
