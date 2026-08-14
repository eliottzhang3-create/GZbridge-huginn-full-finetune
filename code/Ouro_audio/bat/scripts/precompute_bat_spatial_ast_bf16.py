#!/usr/bin/env python3
"""Precompute one BAT source shard with frozen Spatial-AST.

For every unique source tuple in one JSONL shard this worker performs:

    AudioSet + binaural RIR -> [2, 320000] waveform
        -> Spatial-AST in FP32 -> [515, 768]
        -> BF16 CPU feature cache

The cache is written as chunked safetensors plus an index.  The worker is
single-process/single-GPU by design: multiple shards are launched as
independent scheduler jobs, so no DDP/NCCL communication is involved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

try:
    from ..models.spatial_ast_audio import (
        SPATIAL_AST_HIDDEN_SIZE,
        SPATIAL_AST_TOKEN_COUNT,
        BATAudioRenderer,
        SpatialASTAudioEncoder,
    )
    from .build_bat_unique_manifests import SOURCE_FIELDS, source_key
except ImportError:  # Direct ``python path/to/script.py`` execution.
    from bat.models.spatial_ast_audio import (
        SPATIAL_AST_HIDDEN_SIZE,
        SPATIAL_AST_TOKEN_COUNT,
        BATAudioRenderer,
        SpatialASTAudioEncoder,
    )
    from bat.scripts.build_bat_unique_manifests import SOURCE_FIELDS, source_key


def private_output(path: Path) -> None:
    normalized = str(path).replace("\\", "/")
    if normalized == "/hpc_stor03/public" or normalized.startswith("/hpc_stor03/public/"):
        raise ValueError(f"Refusing to write under read-only public storage: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank line in source manifest: {path}:{line_number}")
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Source manifest row is not an object: {path}:{line_number}")
            rows.append(item)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spatial-ast-root", type=Path, required=True)
    parser.add_argument("--spatial-ast-checkpoint", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--reverb-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows; intended for smoke tests")
    parser.add_argument("--max-errors", type=int, default=32)
    return parser.parse_args()


def validate_source_rows(rows: list[dict[str, Any]]) -> None:
    required = set(SOURCE_FIELDS) | {"source_key", "source_shape", "estimated_render_source_count"}
    seen: set[str] = set()
    for index, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"Source manifest row {index} missing fields: {missing}")
        key = str(row["source_key"])
        if key in seen:
            raise ValueError(f"Duplicate source_key in input shard at row {index}: {key}")
        seen.add(key)
        source = tuple(str(row.get(field, "")) for field in SOURCE_FIELDS)
        if source_key(source) != key:
            raise ValueError(f"source_key mismatch at row {index}: expected {source_key(source)} got {key}")
        if row["source_shape"] not in {"single", "dual"}:
            raise ValueError(f"Invalid source_shape at row {index}: {row['source_shape']!r}")
        expected_cost = 2 if row["source_shape"] == "dual" else 1
        if int(row["estimated_render_source_count"]) != expected_cost:
            raise ValueError(f"Invalid render cost at row {index}: {row['estimated_render_source_count']!r}")


def load_existing_index(index_path: Path, feature_root: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not index_path.is_file():
        return [], {}
    rows = read_jsonl(index_path)
    completed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        key = str(row.get("source_key"))
        if not key or key in completed:
            raise ValueError(f"Invalid or duplicate cache index source_key at row {index}: {key!r}")
        relative = Path(str(row.get("feature_file")))
        feature_file = feature_root / relative
        if not feature_file.is_file():
            raise FileNotFoundError(f"Cache index points to missing feature file: {feature_file}")
        if tuple(row.get("shape", [])) != (SPATIAL_AST_TOKEN_COUNT, SPATIAL_AST_HIDDEN_SIZE):
            raise ValueError(f"Unexpected cached feature shape in index for {key}: {row.get('shape')}")
        if row.get("dtype") != "bfloat16":
            raise ValueError(f"Unexpected cached feature dtype in index for {key}: {row.get('dtype')}")
        completed[key] = row
    return rows, completed


def save_feature_chunk(
    feature_root: Path,
    index_path: Path,
    index_rows: list[dict[str, Any]],
    source_keys: list[str],
    features: torch.Tensor,
    chunk_size: int,
) -> None:
    if features.ndim != 3 or tuple(features.shape[1:]) != (SPATIAL_AST_TOKEN_COUNT, SPATIAL_AST_HIDDEN_SIZE):
        raise ValueError(f"Unexpected feature chunk shape: {tuple(features.shape)}")
    if features.dtype != torch.bfloat16:
        raise ValueError(f"Expected BF16 feature chunk, got {features.dtype}")
    if len(source_keys) != features.shape[0]:
        raise ValueError("Feature chunk key count does not match tensor batch")
    part_id = len(index_rows) // chunk_size
    part_name = f"part-{part_id:05d}.safetensors"
    temporary = feature_root / (part_name + ".tmp")
    final = feature_root / part_name
    save_file({"features": features.contiguous()}, str(temporary))
    temporary.replace(final)

    for row, key in enumerate(source_keys):
        index_rows.append(
            {
                "source_key": key,
                "feature_file": part_name,
                "row": row,
                "shape": [SPATIAL_AST_TOKEN_COUNT, SPATIAL_AST_HIDDEN_SIZE],
                "dtype": "bfloat16",
            }
        )
    write_jsonl(index_path, index_rows)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.chunk_size <= 0 or args.max_errors <= 0:
        raise ValueError("batch-size, chunk-size and max-errors must be positive")
    if args.limit < 0:
        raise ValueError("limit must be non-negative")
    private_output(args.output_dir)
    if not args.source_manifest.is_file():
        raise FileNotFoundError(args.source_manifest)

    rows = read_jsonl(args.source_manifest)
    validate_source_rows(rows)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No source rows selected")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_root = args.output_dir / "features"
    feature_root.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / "index.jsonl"
    report_path = args.output_dir / "precompute_report.json"
    index_rows, completed = load_existing_index(index_path, feature_root)

    selected_keys = {str(row["source_key"]) for row in rows}
    unexpected_completed = sorted(set(completed) - selected_keys)
    if unexpected_completed:
        raise ValueError(
            f"Existing cache index contains {len(unexpected_completed)} keys not present in this source shard; "
            "use a separate output directory"
        )

    print("========== BAT SPATIAL-AST BF16 FEATURE PRECOMPUTE ==========")
    print(f"[source] {args.source_manifest} selected_rows={len(rows)}")
    print(f"[output] {args.output_dir}")
    print(f"[device] {device} name={torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'}")
    print(f"[config] batch_size={args.batch_size} chunk_size={args.chunk_size} output_dtype=bfloat16")
    print(f"[resume] existing={len(completed)} pending={len(rows) - len(completed)}")

    renderer = BATAudioRenderer(args.audio_root, args.reverb_root)
    encoder = SpatialASTAudioEncoder(args.spatial_ast_root, args.spatial_ast_checkpoint)
    encoder = encoder.to(device=device, dtype=torch.float32).eval()
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("Spatial-AST encoder unexpectedly has trainable parameters")

    pending_keys: list[str] = []
    pending_waveforms: list[torch.Tensor] = []
    chunk_keys: list[str] = []
    chunk_features: list[torch.Tensor] = []
    errors: list[dict[str, Any]] = []
    render_seconds = 0.0
    encode_seconds = 0.0
    new_count = 0
    skipped_count = 0
    start_time = time.perf_counter()

    def flush_feature_chunk() -> None:
        nonlocal new_count
        if not chunk_keys:
            return
        features = torch.cat(chunk_features, dim=0)
        save_feature_chunk(feature_root, index_path, index_rows, chunk_keys, features, args.chunk_size)
        new_count += len(chunk_keys)
        chunk_keys.clear()
        chunk_features.clear()

    def flush_batch() -> None:
        nonlocal render_seconds, encode_seconds
        if not pending_waveforms:
            return
        waveforms = torch.stack(pending_waveforms, dim=0).to(device=device, dtype=torch.float32, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        encode_start = time.perf_counter()
        with torch.inference_mode():
            tokens = encoder(waveforms)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        encode_seconds += time.perf_counter() - encode_start
        if tokens.dtype != torch.float32 or tuple(tokens.shape[1:]) != (SPATIAL_AST_TOKEN_COUNT, SPATIAL_AST_HIDDEN_SIZE):
            raise RuntimeError(f"Unexpected FP32 Spatial-AST output: dtype={tokens.dtype} shape={tuple(tokens.shape)}")
        if not torch.isfinite(tokens).all().item():
            raise RuntimeError("Spatial-AST produced non-finite tokens")
        bf16 = tokens.to(dtype=torch.bfloat16).cpu()
        chunk_keys.extend(pending_keys)
        chunk_features.append(bf16)
        pending_keys.clear()
        pending_waveforms.clear()
        if len(chunk_keys) >= args.chunk_size:
            flush_feature_chunk()

    for row_index, row in enumerate(rows):
        key = str(row["source_key"])
        if key in completed:
            skipped_count += 1
            continue
        try:
            render_start = time.perf_counter()
            waveform = renderer.render_record(row)
            render_seconds += time.perf_counter() - render_start
            if tuple(waveform.shape) != (2, 320000) or not torch.isfinite(waveform).all().item():
                raise RuntimeError(f"Invalid rendered waveform shape/finite status: {tuple(waveform.shape)}")
            pending_keys.append(key)
            pending_waveforms.append(waveform)
            if len(pending_waveforms) >= args.batch_size:
                flush_batch()
        except Exception as exc:
            errors.append({"row_index": row_index, "source_key": key, "error": repr(exc)})
            print(f"[error] row={row_index} source_key={key} error={exc}", file=sys.stderr)
            if len(errors) >= args.max_errors:
                break

    flush_batch()
    flush_feature_chunk()

    expected_keys = {str(row["source_key"]) for row in rows}
    cached_keys = set(completed) | {str(row["source_key"]) for row in index_rows}
    missing_keys = sorted(expected_keys - cached_keys)
    elapsed = time.perf_counter() - start_time
    status = "ok" if not errors and not missing_keys else "incomplete"
    report = {
        "status": status,
        "source_manifest": str(args.source_manifest),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "output_dir": str(args.output_dir),
        "device": {
            "requested": str(device),
            "name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        },
        "packages": {"torch": torch.__version__},
        "spatial_ast": {
            "source_root": str(args.spatial_ast_root),
            "checkpoint": str(args.spatial_ast_checkpoint),
            "checkpoint_bytes": args.spatial_ast_checkpoint.stat().st_size,
            "frozen": True,
            "inference_dtype": "float32",
            "output_shape": [SPATIAL_AST_TOKEN_COUNT, SPATIAL_AST_HIDDEN_SIZE],
            "cache_dtype": "bfloat16",
        },
        "counts": {
            "input_rows": len(rows),
            "existing_rows_skipped": skipped_count,
            "new_rows_written": new_count,
            "index_rows": len(index_rows),
            "missing_rows": len(missing_keys),
            "error_rows": len(errors),
        },
        "timing_seconds": {
            "wall": elapsed,
            "render": render_seconds,
            "spatial_ast": encode_seconds,
        },
        "memory": {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
        },
        "cache": {
            "index": str(index_path),
            "feature_root": str(feature_root),
            "feature_shape": [SPATIAL_AST_TOKEN_COUNT, SPATIAL_AST_HIDDEN_SIZE],
            "feature_dtype": "bfloat16",
            "index_keys_unique": len(index_rows) == len({str(row["source_key"]) for row in index_rows}),
        },
        "errors": errors,
        "missing_key_examples": missing_keys[:20],
        "contract": {
            "audio_root_read_only_input": str(args.audio_root),
            "reverb_root_read_only_or_private_input": str(args.reverb_root),
            "audio_augmentation": False,
            "audio_processing": [
                "AudioSet read",
                "resample to 32000 Hz",
                "RMS/loudness normalization",
                "binaural RIR convolution",
                "trim/pad to 10 seconds",
            ],
            "spatial_ast_gradients": False,
            "public_storage_written": False,
        },
    }
    write_json(report_path, report)
    print(f"[summary] status={status} input={len(rows)} skipped={skipped_count} new={new_count} missing={len(missing_keys)} errors={len(errors)}")
    print(f"[report] {report_path}")
    if status != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
