#!/usr/bin/env python3
"""Audit production Ouro-BAT token lengths without rendering real audio.

The production BAT template prepends 64 audio-prefix positions after the
normal Ouro chat encoding.  This audit imports that exact plugin/template and
uses a cached dummy waveform only to satisfy the template contract.  It never
opens AudioSet, reads an RIR, runs Spatial-AST, touches CUDA, or modifies the
manifest.  The resulting lengths therefore audit the text/token contract that
the training collator will see, while avoiding the expensive audio path.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any, Iterable

import torch
from transformers import AutoConfig


EXPECTED_AUDIO_TOKENS = 64
TEMPLATE_MAX_LENGTH = 512
MODEL_TYPE = "ouro_bat_spatial_ast"
TEMPLATE_TYPE = "ouro_bat_audio_prefix"
DEFAULT_MODEL_PATH = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/Ouro-1.4B")
DEFAULT_PLUGIN_PATH = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/"
    "code/Ouro_audio/plugins/ouro_bat_spatial_ast_swift.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--plugin-path", type=Path, default=DEFAULT_PLUGIN_PATH)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument(
        "--tail-records",
        type=int,
        default=650000,
        help="Audit only the final N physical JSONL records; use 0 to audit the full manifest.",
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=[128, 160, 192, 256, 512],
        help="Lengths for exceedance counts, e.g. 160 192 256.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def require_private_output(path: Path) -> None:
    normalized = str(path).replace("\\", "/")
    if normalized == "/hpc_stor03/public" or normalized.startswith("/hpc_stor03/public/"):
        raise ValueError(f"Refusing public audit output: {path}")


def import_plugin(path: Path):
    spec = importlib.util.spec_from_file_location("bat_token_length_plugin", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import plugin: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def manifest_physical_line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def iter_manifest(path: Path, start_line: int = 1) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line_number < start_line:
                continue
            if not line.strip():
                raise ValueError(f"Blank line in manifest at line {line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Manifest line {line_number} is not a JSON object")
            yield line_number, row


def as_int_list(value: Any) -> list[int]:
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    if not isinstance(value, list):
        raise TypeError(f"Expected list-like token field, got {type(value).__name__}")
    return [int(item) for item in value]


def record_preview(row: dict[str, Any], line_number: int, length: int) -> dict[str, Any]:
    audios = row.get("audios")
    source = audios[0] if isinstance(audios, list) and audios and isinstance(audios[0], dict) else {}
    return {
        "line_number": line_number,
        "length": length,
        "text_length_after_audio_prefix": length - EXPECTED_AUDIO_TOKENS,
        "question_id": source.get("question_id", row.get("question_id")),
        "question_type": source.get("question_type", row.get("question_type")),
        "bat_type": row.get("bat_type"),
        "source_shape": row.get("source_shape"),
        "audio_id": source.get("audio_id"),
        "reverb_id": source.get("reverb_id"),
    }


def percentile(sorted_values: list[int], fraction: float) -> int:
    if not sorted_values:
        return 0
    index = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * fraction))))
    return int(sorted_values[index])


def build_template(args: argparse.Namespace):
    plugin = import_plugin(resolve(args.plugin_path))
    if plugin.MODEL_TYPE != MODEL_TYPE or plugin.TEMPLATE_TYPE != TEMPLATE_TYPE:
        raise RuntimeError("BAT plugin registration constants do not match the production template")

    # build_processor loads only the local tokenizer/processor.  It does not
    # load Ouro weights and does not require CUDA.
    processor = plugin.build_processor(str(resolve(args.model_path)))
    # get_template() normally receives a processor produced by
    # get_model_processor(), which attaches these two metadata objects after
    # loading the model.  This audit intentionally skips model loading, so
    # provide the same minimal metadata contract explicitly.
    config = AutoConfig.from_pretrained(
        str(resolve(args.model_path)),
        trust_remote_code=True,
        local_files_only=True,
    )
    processor.model_info = SimpleNamespace(
        model_dir=str(resolve(args.model_path)),
        config=config,
        task_type="causal_lm",
        max_model_len=TEMPLATE_MAX_LENGTH,
    )
    processor.model_meta = SimpleNamespace(
        is_multimodal=True,
        template=TEMPLATE_TYPE,
        candidate_templates=[TEMPLATE_TYPE],
    )
    from swift import get_template

    template = get_template(
        template_type=TEMPLATE_TYPE,
        processor=processor,
        max_length=TEMPLATE_MAX_LENGTH,
        use_chat_template=True,
        padding_side="right",
        padding_free=False,
        template_backend="swift",
    )
    template.set_mode("train")

    # The plugin's _encode calls audio_renderer.load_item, but token lengths
    # do not depend on waveform values.  Return one cached tensor so this
    # audit exercises the exact token/prefix path without any audio I/O.
    dummy_waveform = torch.zeros((2, 320000), dtype=torch.float32)
    template.audio_renderer.load_item = lambda _record: dummy_waveform
    return template


def main() -> None:
    args = parse_args()
    manifest = resolve(args.manifest)
    model_path = resolve(args.model_path)
    plugin_path = resolve(args.plugin_path)
    output = resolve(args.output_report)
    require_private_output(output)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    if not plugin_path.is_file():
        raise FileNotFoundError(plugin_path)
    if args.progress_every <= 0:
        raise ValueError("progress-every must be positive")
    if args.tail_records < 0:
        raise ValueError("tail-records must be non-negative")
    thresholds = sorted({int(value) for value in args.thresholds})
    if any(value <= 0 for value in thresholds):
        raise ValueError("thresholds must be positive")

    total_manifest_lines = manifest_physical_line_count(manifest)
    if args.tail_records == 0:
        selection_start_line = 1
        selected_manifest_lines = total_manifest_lines
    else:
        selection_start_line = max(1, total_manifest_lines - args.tail_records + 1)
        selected_manifest_lines = total_manifest_lines - selection_start_line + 1

    print("========== BAT OURO PRODUCTION TOKEN-LENGTH AUDIT ==========")
    print(f"[manifest] {manifest}")
    print(f"[model] {model_path}")
    print(f"[plugin] {plugin_path}")
    print("[audio] bypassed: cached dummy waveform; no AudioSet/RIR/Spatial-AST access")
    print(f"[contract] audio_prefix_tokens={EXPECTED_AUDIO_TOKENS} template_max_length={TEMPLATE_MAX_LENGTH}")
    print(
        f"[selection] total_manifest_lines={total_manifest_lines} "
        f"start_line={selection_start_line} selected_tail_lines={selected_manifest_lines}"
    )

    template = build_template(args)
    lengths: list[int] = []
    length_counts: Counter[str] = Counter()
    threshold_counts = {str(value): 0 for value in thresholds}
    errors: list[dict[str, Any]] = []
    prefix_label_errors = 0
    alignment_errors = 0
    prefix_id_errors = 0
    encoding_error_count = 0
    max_records: list[dict[str, Any]] = []
    record_count = 0

    for line_number, row in iter_manifest(manifest, start_line=selection_start_line):
        record_count += 1
        try:
            encoded = template.encode(row)
            input_ids = as_int_list(encoded["input_ids"])
            labels = as_int_list(encoded["labels"])
            if len(input_ids) != len(labels):
                alignment_errors += 1
            if len(labels) >= EXPECTED_AUDIO_TOKENS and any(
                value != -100 for value in labels[:EXPECTED_AUDIO_TOKENS]
            ):
                prefix_label_errors += 1
            if len(input_ids) >= EXPECTED_AUDIO_TOKENS:
                prefix = input_ids[:EXPECTED_AUDIO_TOKENS]
                if len(set(prefix)) != 1:
                    prefix_id_errors += 1
            length = len(input_ids)
            lengths.append(length)
            length_counts[str(length)] += 1
            for threshold in thresholds:
                if length > threshold:
                    threshold_counts[str(threshold)] += 1
            preview = record_preview(row, line_number, length)
            if not max_records or length > int(max_records[0]["length"]):
                max_records = [preview]
            elif length == int(max_records[0]["length"]) and len(max_records) < 10:
                max_records.append(preview)
        except Exception as exc:  # Keep the audit diagnostic and continue.
            encoding_error_count += 1
            if len(errors) < 20:
                errors.append({"line_number": line_number, "error": f"{type(exc).__name__}: {exc}"})
        if record_count % args.progress_every == 0:
            print(f"[progress] records={record_count} encoded={len(lengths)} errors={len(errors)}", flush=True)

    sorted_lengths = sorted(lengths)
    issues: list[str] = []
    if not lengths:
        issues.append("no_records_encoded")
    if encoding_error_count:
        issues.append("record_encoding_errors")
    if alignment_errors:
        issues.append("input_label_alignment_errors")
    if prefix_label_errors:
        issues.append("audio_prefix_labels_not_ignored")
    if prefix_id_errors:
        issues.append("audio_prefix_ids_not_constant")

    observed_max_length = max(sorted_lengths) if sorted_lengths else 0
    recommended_padding_length = (
        int(math.ceil(observed_max_length * 1.05)) if observed_max_length else 0
    )
    warnings: list[str] = []
    if recommended_padding_length > TEMPLATE_MAX_LENGTH:
        warnings.append("recommended_padding_exceeds_template_max_length")

    report = {
        "status": "ok" if not issues else "incomplete",
        "manifest": str(manifest),
        "model_path": str(model_path),
        "plugin_path": str(plugin_path),
        "template_type": TEMPLATE_TYPE,
        "audio_rendering_bypassed": True,
        "record_count": record_count,
        "encoded_count": len(lengths),
        "encoding_error_count": encoding_error_count,
        "selection": {
            "mode": "tail" if args.tail_records else "full",
            "requested_tail_records": args.tail_records or None,
            "total_manifest_physical_lines": total_manifest_lines,
            "selection_start_line": selection_start_line,
            "selected_manifest_physical_lines": selected_manifest_lines,
        },
        "contract": {
            "audio_prefix_token_count": EXPECTED_AUDIO_TOKENS,
            "template_max_length": TEMPLATE_MAX_LENGTH,
            "prefix_labels_expected": -100,
            "input_label_alignment_error_count": alignment_errors,
            "prefix_label_error_count": prefix_label_errors,
            "prefix_id_error_count": prefix_id_errors,
        },
        "length_statistics": {
            "min": min(sorted_lengths) if sorted_lengths else 0,
            "max": observed_max_length,
            "mean": float(mean(sorted_lengths)) if sorted_lengths else 0.0,
            "p50": percentile(sorted_lengths, 0.50),
            "p90": percentile(sorted_lengths, 0.90),
            "p95": percentile(sorted_lengths, 0.95),
            "p99": percentile(sorted_lengths, 0.99),
            "p999": percentile(sorted_lengths, 0.999),
        },
        "recommended_padding": {
            "formula": "ceil(max_observed_length * 1.05)",
            "max_observed_length": observed_max_length,
            "margin_fraction": 0.05,
            "recommended_sequence_length": recommended_padding_length,
            "within_template_max_length": recommended_padding_length <= TEMPLATE_MAX_LENGTH,
        },
        "threshold_exceedance_counts": threshold_counts,
        "max_length_records": max_records,
        "length_histogram": dict(sorted(length_counts.items(), key=lambda item: int(item[0]))),
        "errors_preview": errors,
        "warnings": warnings,
        "issues": issues,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[summary] records={record_count} encoded={len(lengths)} max={report['length_statistics']['max']}")
    print(f"[summary] recommended_padding_length={recommended_padding_length}")
    print(f"[summary] thresholds={threshold_counts}")
    print(f"[report] {output}")
    print(f"[status] {report['status']} issues={issues[:10]}")
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
