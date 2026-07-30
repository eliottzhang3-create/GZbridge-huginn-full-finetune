"""Validate the indexed mixture-to-Swift boundary and four real audio decoders."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
HUGINN_LORA_ROOT = REPO_ROOT / "code" / "huginn_lora"
if str(HUGINN_LORA_ROOT) not in sys.path:
    sys.path.insert(0, str(HUGINN_LORA_ROOT))

from data_pipeline.dynamic90s_mixture_rows import (  # noqa: E402
    SYSTEM_PROMPT,
    TASK_PROMPTS,
    iter_dynamic90s_mixture_rows,
    load_pool_registry,
    open_indexed_pools,
)
from data_pipeline.indexed_atomic_mixture import (  # noqa: E402
    POOL_ORDER,
    DeterministicHierarchicalMixture,
)


DEFAULT_REGISTRY = (
    REPO_ROOT
    / "data"
    / "audio_swift"
    / "huginn_whisper_dynamic90s_multitask"
    / "v2_dynamic30s"
    / "pool_registry.json"
)
DEFAULT_REPORT = (
    REPO_ROOT
    / "data"
    / "audio_swift"
    / "huginn_whisper_dynamic90s_multitask"
    / "v2_dynamic30s"
    / "real_data_chain_report.json"
)
DEFAULT_PLUGIN = (
    HUGINN_LORA_ROOT / "plugins" / "huginn_audio_whisper_dynamic90s_mixture_swift.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--plugin", default=str(DEFAULT_PLUGIN))
    parser.add_argument("--output_report", default=str(DEFAULT_REPORT))
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--resume_position", type=int, default=37)
    parser.add_argument("--resume_probe_rows", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_plugin(path: Path) -> Any:
    module_name = "huginn_audio_whisper_dynamic90s_mixture_swift"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Swift plugin: {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            if sys.modules.get(module_name) is module:
                sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    return module


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_swift_dataset(registry_path: Path):
    from swift.dataset import load_dataset

    try:
        return load_dataset(
            str(registry_path),
            split_dataset_ratio=0.0,
            shuffle=False,
            num_proc=1,
            streaming=True,
        )
    except TypeError as first_error:
        print(f"[swift] load_dataset_compat_retry={type(first_error).__name__}: {first_error}", flush=True)
        return load_dataset(
            str(registry_path),
            split_dataset_ratio=0.0,
            shuffle=False,
            num_proc=1,
        )


def normalize_audio_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"Audio item is not a mapping: {type(value).__name__}")
    return {
        "path": str(value.get("path", "")),
        "format": str(value.get("format", "")),
        "start_sec": None if value.get("start_sec") is None else float(value["start_sec"]),
        "end_sec": None if value.get("end_sec") is None else float(value["end_sec"]),
        "raw_duration_sec": (
            None if value.get("raw_duration_sec") is None else float(value["raw_duration_sec"])
        ),
    }


def validate_row(
    row: dict[str, Any],
    expected_position: int,
    planner: DeterministicHierarchicalMixture,
    pools: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    selection = planner.selection(expected_position)
    record = pools[selection.pool_name].record(selection.record_index)
    target_index = planner.target_index(expected_position, len(record["targets"]))
    messages = row.get("messages")
    audios = row.get("audios")
    metadata = row.get("metadata")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError(f"Position {expected_position} has invalid messages: {messages!r}")
    if not isinstance(audios, list) or len(audios) != 1:
        raise ValueError(f"Position {expected_position} has invalid audios: {audios!r}")
    if not isinstance(metadata, dict):
        raise ValueError(f"Position {expected_position} has invalid metadata: {metadata!r}")
    expected_metadata = {
        "global_position": expected_position,
        "pool_name": selection.pool_name,
        "record_index": selection.record_index,
        "target_index": target_index,
        "uid": record["uid"],
        "task": record["task"],
    }
    actual_metadata = {key: metadata.get(key) for key in expected_metadata}
    if actual_metadata != expected_metadata:
        raise ValueError(
            f"Position {expected_position} metadata mismatch: actual={actual_metadata} expected={expected_metadata}"
        )
    if messages[1].get("content") != TASK_PROMPTS[record["task"]]:
        raise ValueError(f"Position {expected_position} task prompt mismatch: {messages[1]!r}")
    if messages[2].get("content") != record["targets"][target_index]:
        raise ValueError(f"Position {expected_position} selected target mismatch")
    expected_audio = {
        "path": str(record["audio"]["path"]),
        "format": str(record["audio"].get("format", "")),
        "start_sec": (
            float(record["audio"]["start_sec"]) if record["audio"].get("start_sec") is not None else None
        ),
        "end_sec": (
            float(record["audio"]["end_sec"]) if record["audio"].get("end_sec") is not None else None
        ),
        "raw_duration_sec": (
            float(record["raw_duration_sec"]) if record.get("raw_duration_sec") is not None else None
        ),
    }
    actual_audio = normalize_audio_mapping(audios[0])
    if actual_audio != expected_audio:
        raise ValueError(
            f"Position {expected_position} audio mismatch: actual={actual_audio} expected={expected_audio}"
        )
    return selection.pool_name, actual_audio


def row_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    metadata = row["metadata"]
    return (
        int(metadata["global_position"]),
        str(metadata["pool_name"]),
        int(metadata["record_index"]),
        int(metadata["target_index"]),
        str(metadata["uid"]),
        str(row["messages"][-1]["content"]),
        tuple(normalize_audio_mapping(row["audios"][0]).items()),
    )


def main() -> None:
    args = parse_args()
    if args.rows <= 0 or args.resume_position < 0 or args.resume_probe_rows <= 0:
        raise ValueError("rows and resume_probe_rows must be positive; resume_position must be non-negative")
    if args.resume_position + args.resume_probe_rows > args.rows:
        raise ValueError("The baseline row window must contain the complete resume probe")

    registry_path = Path(args.registry).expanduser().resolve()
    plugin_path = Path(args.plugin).expanduser().resolve()
    report_path = Path(args.output_report).expanduser().resolve()
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite report without --overwrite: {report_path}")
    registry = load_pool_registry(registry_path)
    pool_sizes = {
        name: int(registry["pools"][name]["record_count"])
        for name in POOL_ORDER
    }
    planner = DeterministicHierarchicalMixture(pool_sizes=pool_sizes, seed=args.seed)

    os.environ["HUGINN_DYNAMIC90S_POOL_REGISTRY"] = str(registry_path)
    os.environ["HUGINN_DYNAMIC90S_MIXTURE_SEED"] = str(args.seed)
    os.environ["HUGINN_DYNAMIC90S_MIXTURE_START_POSITION"] = "0"
    os.environ["HUGINN_DYNAMIC90S_MIXTURE_MAX_SAMPLES"] = str(args.rows)

    print("========== HUGINN WHISPER DYNAMIC30S REAL DATA CHAIN START ==========", flush=True)
    print(
        f"[scope] rows={args.rows} real_audio_decodes={len(POOL_ORDER)} "
        "model_load=false whisper_load=false audio_copy=false opus_conversion=false",
        flush=True,
    )
    plugin = load_plugin(plugin_path)
    if plugin._BASE_PLUGIN.DEFAULT_SYSTEM_PROMPT != SYSTEM_PROMPT:
        raise RuntimeError(
            "Mixture renderer system prompt drifted from the dynamic-90s model plugin: "
            f"renderer={SYSTEM_PROMPT!r} model={plugin._BASE_PLUGIN.DEFAULT_SYSTEM_PROMPT!r}"
        )
    train_dataset, val_dataset = load_swift_dataset(registry_path)
    if val_dataset is not None:
        raise RuntimeError(f"split_dataset_ratio=0 produced a validation dataset: {type(val_dataset)}")

    baseline_rows: list[dict[str, Any]] = []
    first_audio_by_pool: dict[str, dict[str, Any]] = {}
    with ExitStack() as stack:
        pools = open_indexed_pools(registry, stack)
        for index, row in enumerate(train_dataset):
            if not isinstance(row, dict):
                raise TypeError(f"Swift row {index} has type {type(row).__name__}")
            pool_name, audio = validate_row(row, index, planner, pools)
            baseline_rows.append(row)
            first_audio_by_pool.setdefault(pool_name, audio)
        if len(baseline_rows) != args.rows:
            raise RuntimeError(f"Swift dataset row count mismatch: actual={len(baseline_rows)} expected={args.rows}")

    if set(first_audio_by_pool) != set(POOL_ORDER):
        raise RuntimeError(
            f"The deterministic probe did not cover all pools: observed={sorted(first_audio_by_pool)}"
        )
    print(f"[swift] rows={len(baseline_rows)} pools={sorted(first_audio_by_pool)} schema=pass", flush=True)

    # Rebuild the public dataset entry at a non-zero position. This is the data-layer
    # prerequisite for the later process-exit checkpoint/resume smoke.
    os.environ["HUGINN_DYNAMIC90S_MIXTURE_START_POSITION"] = str(args.resume_position)
    os.environ["HUGINN_DYNAMIC90S_MIXTURE_MAX_SAMPLES"] = str(args.resume_probe_rows)
    resumed_dataset = plugin.build_dataset(registry_path)
    resumed_rows = list(resumed_dataset)
    expected_rows = baseline_rows[args.resume_position : args.resume_position + args.resume_probe_rows]
    if [row_signature(row) for row in resumed_rows] != [row_signature(row) for row in expected_rows]:
        raise RuntimeError("Non-zero mixture start_position did not reproduce the baseline sample sequence")
    # Also verify the pure iterator, independently of Hugging Face/Swift wrapping.
    direct_rows = list(
        iter_dynamic90s_mixture_rows(
            registry_path=registry_path,
            seed=args.seed,
            start_position=args.resume_position,
            max_samples=args.resume_probe_rows,
        )
    )
    if [row_signature(row) for row in direct_rows] != [row_signature(row) for row in expected_rows]:
        raise RuntimeError("Pure indexed iterator resume sequence mismatch")
    print(
        f"[resume-data] start_position={args.resume_position} rows={args.resume_probe_rows} deterministic=pass",
        flush=True,
    )

    audio_probes: dict[str, dict[str, Any]] = {}
    for pool_name in POOL_ORDER:
        audio_reference = first_audio_by_pool[pool_name]
        waveform = plugin._BASE_PLUGIN.load_audio_item(
            audio_reference,
            target_sr=plugin._BASE_PLUGIN.DEFAULT_SAMPLE_RATE,
            max_audio_seconds=plugin._BASE_PLUGIN.DEFAULT_MAX_AUDIO_SECONDS,
        )
        if not isinstance(waveform, np.ndarray) or waveform.dtype != np.float32 or waveform.ndim != 1:
            raise TypeError(
                f"Decoded waveform contract failed for {pool_name}: "
                f"type={type(waveform).__name__} dtype={getattr(waveform, 'dtype', None)} "
                f"shape={getattr(waveform, 'shape', None)}"
            )
        if waveform.size <= 0 or not bool(np.isfinite(waveform).all()):
            raise RuntimeError(f"Decoded waveform is empty or non-finite for {pool_name}")
        max_samples = int(
            plugin._BASE_PLUGIN.DEFAULT_SAMPLE_RATE * plugin._BASE_PLUGIN.DEFAULT_MAX_AUDIO_SECONDS
        )
        if waveform.size > max_samples:
            raise RuntimeError(f"Decoded waveform exceeds 30 seconds for {pool_name}: samples={waveform.size}")
        if audio_reference["start_sec"] is not None:
            expected_seconds = min(
                float(audio_reference["end_sec"]) - float(audio_reference["start_sec"]),
                plugin._BASE_PLUGIN.DEFAULT_MAX_AUDIO_SECONDS,
            )
            actual_seconds = waveform.size / plugin._BASE_PLUGIN.DEFAULT_SAMPLE_RATE
            if not math.isclose(actual_seconds, expected_seconds, rel_tol=0.0, abs_tol=0.5):
                raise RuntimeError(
                    f"Segment decode duration mismatch for {pool_name}: "
                    f"actual={actual_seconds:.6f}s expected={expected_seconds:.6f}s"
                )
        audio_probes[pool_name] = {
            "path": audio_reference["path"],
            "segment": audio_reference["start_sec"] is not None,
            "decoded_samples": int(waveform.size),
            "decoded_seconds": waveform.size / plugin._BASE_PLUGIN.DEFAULT_SAMPLE_RATE,
        }
        print(
            f"[audio] pool={pool_name} seconds={audio_probes[pool_name]['decoded_seconds']:.3f} "
            f"segment_reference={audio_probes[pool_name]['segment']} decode=pass",
            flush=True,
        )

    report = {
        "gate": "huginn_whisper_dynamic30s_real_data_chain_v2",
        "validation_passed": True,
        "contract_version": registry.get("contract_version"),
        "duration_policy": "discard_gt90s_then_cap_at30s",
        "registry": str(registry_path),
        "seed": args.seed,
        "swift_rows_checked": len(baseline_rows),
        "covered_pools": sorted(first_audio_by_pool),
        "resume_probe": {
            "start_position": args.resume_position,
            "rows": args.resume_probe_rows,
            "deterministic": True,
        },
        "audio_probes": audio_probes,
        "scope": {
            "model_loaded": False,
            "whisper_loaded": False,
            "audio_files_decoded": len(audio_probes),
            "audio_copy": False,
            "materialized_opus_conversion": False,
            "token_accounting": False,
        },
    }
    write_json_atomic(report_path, report)
    print(f"[report] path={report_path}", flush=True)
    print("========== HUGINN WHISPER DYNAMIC30S REAL DATA CHAIN PASSED ==========", flush=True)


if __name__ == "__main__":
    main()
