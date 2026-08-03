#!/usr/bin/env python3
"""Read-only audit of a Huginn Whisper checkpoint before X-ARES evaluation.

This script intentionally does not instantiate the Huginn model and does not
load a full FSDP state dict.  It audits the checkpoint contract from DCP
metadata, trainer/runtime JSON files, the local model configuration, and a few
small representative tensors.  It is therefore safe to run before preparing
the separate X-ARES environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = REPO_ROOT / "code" / "huginn_lora" / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from inspect_huginn_whisper_dynamic90s_fsdp_checkpoints import (  # noqa: E402
    AUDIO_BOUNDARY_PARAMETER_NAMES,
    AUDIO_BOUNDARY_CHECKPOINT_ROLES,
    GROUP_NAMES,
    TRAINING_RUNTIME_CONTRACT_FILENAME,
    audio_boundary_key_identity,
    classify_key,
    classify_optimizer_key,
    key_aliases,
    load_json,
)


DEFAULT_MODEL_CONFIG = REPO_ROOT / "models" / "huginn-audio-whisper-dynamic90s-v1" / "config.json"
DEFAULT_CHECKPOINT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/"
    "huginn_whisper_dynamic30s_multiplier_single_epoch_fsdp4/run-20260731_084946/"
    "swift_output/v0-20260731-085036/checkpoint-15000"
)

EXPECTED_AUDIO_CONFIG = {
    "audio_encoder_hidden_size": 1280,
    "audio_compressor_kernel_size": 12,
    "audio_compressor_stride": 12,
    "audio_pooling_type": "conv1d_stride12_dynamic30s",
    "audio_dynamic_tokens": True,
    "audio_token_duration_ms": 240,
    "audio_reference_30s_token_count": 125,
    "audio_max_token_count": 125,
    "audio_chunk_seconds": 30.0,
    "audio_max_seconds": 30.0,
    "audio_feature_hop_length": 160,
    "audio_encoder_frame_rate": 50,
    "freeze_audio_encoder": False,
    "freeze_text_backbone": True,
    "use_audio_boundary_embeddings": True,
    "n_embd": 5280,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--expected-step", type=int, default=None)
    parser.add_argument("--expected-phase", default="multiplier_formal_checkpoint")
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument(
        "--skip-tensor-probes",
        action="store_true",
        help="Only inspect metadata/JSON; do not read representative DCP tensors.",
    )
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    name = path.resolve().name
    if not name.startswith("checkpoint-"):
        raise ValueError(f"Checkpoint directory must be named checkpoint-<step>: {path}")
    try:
        return int(name.split("-", 1)[1])
    except ValueError as exc:
        raise ValueError(f"Invalid checkpoint directory name: {path}") from exc


def metadata_shape_dtype(entry: Any) -> tuple[list[int], str]:
    shape = [int(value) for value in getattr(entry, "size", ())]
    dtype = str(getattr(getattr(entry, "properties", None), "dtype", None))
    return shape, dtype


def key_preview(keys: list[str], metadata_by_key: dict[str, Any], limit: int = 12) -> list[dict[str, Any]]:
    result = []
    for key in sorted(keys)[:limit]:
        shape, dtype = metadata_shape_dtype(metadata_by_key[key])
        result.append({"key": key, "shape": shape, "dtype": dtype})
    return result


def choose_probe_key(keys: list[str], preferred_tokens: tuple[str, ...]) -> str | None:
    candidates = [
        key for key in keys
        if all(token in key for token in preferred_tokens)
    ]
    if candidates:
        return sorted(candidates)[0]
    return sorted(keys)[0] if keys else None


def load_probe_tensor(model_dir: Path, key: str, entry: Any) -> dict[str, Any]:
    import torch.distributed.checkpoint as dcp

    shape, dtype_text = metadata_shape_dtype(entry)
    dtype = getattr(getattr(entry, "properties", None), "dtype", None)
    if not shape or not isinstance(dtype, torch.dtype):
        raise RuntimeError(f"DCP metadata lacks usable shape/dtype: key={key}")
    tensor = torch.empty(shape, dtype=dtype, device="cpu")
    dcp.load({key: tensor}, checkpoint_id=str(model_dir))
    finite = bool(torch.isfinite(tensor).all().item())
    if not finite:
        raise RuntimeError(f"Representative checkpoint tensor is non-finite: {key}")
    raw = tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()
    return {
        "key": key,
        "shape": shape,
        "dtype": dtype_text,
        "numel": int(tensor.numel()),
        "finite": finite,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def inspect_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    from torch.distributed.checkpoint import FileSystemReader

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
    actual_step = checkpoint_step(checkpoint)
    expected_step = actual_step if args.expected_step is None else int(args.expected_step)
    if actual_step != expected_step:
        raise RuntimeError(f"Checkpoint step mismatch: directory={actual_step} expected={expected_step}")

    model_dir = checkpoint / "pytorch_model_fsdp_0"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Full-model FSDP DCP directory is missing: {model_dir}")
    metadata = FileSystemReader(str(model_dir)).read_metadata()
    state_metadata = getattr(metadata, "state_dict_metadata", {})
    metadata_by_key = {str(key): value for key, value in state_metadata.items()}
    if not metadata_by_key:
        raise RuntimeError(f"FSDP model DCP metadata is empty: {model_dir}")

    grouped: dict[str, list[str]] = {name: [] for name in GROUP_NAMES}
    for key in metadata_by_key:
        grouped[classify_key(key)].append(key)
    counts = {name: len(keys) for name, keys in grouped.items()}
    if counts["audio_encoder"] <= 0:
        raise RuntimeError("Checkpoint contains no audio_encoder parameters")
    if counts["aligner"] <= 0:
        raise RuntimeError("Checkpoint contains no audio aligner parameters")
    if counts["huginn_base"] <= 0:
        raise RuntimeError("Checkpoint contains no Huginn base parameters")
    if counts["other"]:
        raise RuntimeError(f"Unexpected unclassified model parameters: {grouped['other'][:10]}")

    aligner_subgroups = {
        "temporal_compressor": sorted(
            key for key in grouped["aligner"] if any("temporal_compressor" in alias for alias in key_aliases(key))
        ),
        "audio_projector": sorted(
            key for key in grouped["aligner"] if any("audio_projector" in alias for alias in key_aliases(key))
        ),
        "audio_boundary_embeddings": sorted(
            key
            for key in grouped["aligner"]
            if any("audio_boundary_embeddings" in alias for alias in key_aliases(key))
        ),
    }
    for name, keys in aligner_subgroups.items():
        if not keys:
            raise RuntimeError(f"Checkpoint is missing aligner subgroup: {name}")

    boundary_keys = {
        name: {role: [] for role in AUDIO_BOUNDARY_CHECKPOINT_ROLES}
        for name in AUDIO_BOUNDARY_PARAMETER_NAMES
    }
    for key in grouped["aligner"]:
        identity = audio_boundary_key_identity(key)
        if identity is not None:
            boundary_name, role = identity
            boundary_keys[boundary_name][role].append(key)
    invalid_boundaries = {
        f"{name}:{role}": keys
        for name, role_keys in boundary_keys.items()
        for role, keys in role_keys.items()
        if len(keys) != 1
    }
    if invalid_boundaries:
        raise RuntimeError(f"Audio boundary checkpoint contract mismatch: {invalid_boundaries}")

    trainer_state = load_json(checkpoint / "trainer_state.json")
    if int(trainer_state.get("global_step", -1)) != expected_step:
        raise RuntimeError(
            f"trainer_state global_step mismatch: expected={expected_step} "
            f"actual={trainer_state.get('global_step')}"
        )
    scheduler_path = checkpoint / "scheduler.pt"
    if not scheduler_path.is_file():
        raise FileNotFoundError(f"Scheduler state is missing: {scheduler_path}")
    scheduler = torch.load(scheduler_path, map_location="cpu", weights_only=False)
    if int(scheduler.get("last_epoch", -1)) != expected_step:
        raise RuntimeError(
            f"Scheduler last_epoch mismatch: expected={expected_step} "
            f"actual={scheduler.get('last_epoch')}"
        )

    rng_files = sorted(checkpoint.glob("rng_state*.pth"))
    if len(rng_files) != args.world_size:
        raise RuntimeError(
            f"RNG file count mismatch: expected={args.world_size} actual={len(rng_files)} "
            f"files={rng_files}"
        )

    optimizer_dirs = []
    optimizer_counts: dict[str, dict[str, int]] = {}
    for candidate in sorted(checkpoint.iterdir()):
        if not candidate.is_dir() or candidate == model_dir or not (candidate / ".metadata").is_file():
            continue
        optimizer_metadata = FileSystemReader(str(candidate)).read_metadata()
        optimizer_state = getattr(optimizer_metadata, "state_dict_metadata", {})
        if not optimizer_state:
            continue
        optimizer_dirs.append(candidate)
        optimizer_groups = {name: [] for name in GROUP_NAMES}
        for raw_key in optimizer_state:
            key = str(raw_key)
            optimizer_groups[classify_optimizer_key(key)].append(key)
        optimizer_counts[candidate.name] = {
            name: len(keys) for name, keys in optimizer_groups.items()
        }
    if not optimizer_dirs:
        raise RuntimeError(f"No optimizer DCP directory found: {checkpoint}")

    runtime_path = checkpoint / TRAINING_RUNTIME_CONTRACT_FILENAME
    runtime_contract = load_json(runtime_path)
    if int(runtime_contract.get("global_step", -1)) != expected_step:
        raise RuntimeError(
            f"Runtime contract global_step mismatch: expected={expected_step} "
            f"actual={runtime_contract.get('global_step')}"
        )
    if int(runtime_contract.get("world_size", -1)) != args.world_size:
        raise RuntimeError(
            f"Runtime contract world_size mismatch: expected={args.world_size} "
            f"actual={runtime_contract.get('world_size')}"
        )
    if runtime_contract.get("phase") != args.expected_phase:
        raise RuntimeError(
            f"Runtime contract phase mismatch: expected={args.expected_phase} "
            f"actual={runtime_contract.get('phase')}"
        )
    if runtime_contract.get("audio") != {
        "maximum_seconds": 30.0,
        "token_duration_ms": 240,
        "maximum_content_tokens": 125,
        "trainable_boundary_tokens": ["audio_bos", "audio_eos"],
    }:
        raise RuntimeError(f"Runtime audio contract mismatch: {runtime_contract.get('audio')}")

    model_config_path = args.model_config.expanduser().resolve()
    model_config = load_json(model_config_path)
    config_mismatches = {
        key: {"expected": expected, "actual": model_config.get(key)}
        for key, expected in EXPECTED_AUDIO_CONFIG.items()
        if model_config.get(key) != expected
    }
    if config_mismatches:
        raise RuntimeError(f"Local model config is incompatible with X-ARES extraction: {config_mismatches}")

    probes = []
    if not args.skip_tensor_probes:
        probe_specs = (
            ("audio_encoder", ("weight",)),
            ("temporal_compressor", ("weight",)),
            ("audio_projector", ("weight",)),
            ("audio_bos", ("audio_bos",)),
            ("audio_eos", ("audio_eos",)),
        )
        for label, tokens in probe_specs:
            if label in grouped:
                keys = grouped[label]
            elif label in aligner_subgroups:
                keys = aligner_subgroups[label]
            else:
                keys = [
                    key for key in grouped["aligner"]
                    if any(label in alias for alias in key_aliases(key))
                ]
            key = choose_probe_key(keys, tokens)
            if key is None:
                raise RuntimeError(f"Unable to select representative tensor for {label}")
            probes.append({"label": label, **load_probe_tensor(model_dir, key, metadata_by_key[key])})

    optional_files = {
        name: {
            "path": str(checkpoint / name),
            "exists": (checkpoint / name).is_file(),
            "bytes": (checkpoint / name).stat().st_size if (checkpoint / name).is_file() else 0,
        }
        for name in (
            "audio_training_statistics.json",
            "multiplier_formal_training_plan.json",
            "huginn_training_runtime_contract.json",
        )
    }
    report = {
        "gate": "huginn_xares_checkpoint_readonly_inspect_v1",
        "validation_passed": True,
        "checkpoint": str(checkpoint),
        "step": expected_step,
        "world_size": args.world_size,
        "model_dcp": str(model_dir),
        "model_parameter_metadata_counts": counts,
        "aligner_subgroup_counts": {name: len(keys) for name, keys in aligner_subgroups.items()},
        "audio_boundary_keys": boundary_keys,
        "model_metadata_preview": {
            name: key_preview(grouped[name], metadata_by_key)
            for name in ("audio_encoder", "aligner", "lora", "huginn_base")
        },
        "optimizer_dcp_dirs": [str(path) for path in optimizer_dirs],
        "optimizer_metadata_counts": optimizer_counts,
        "trainer_state": {
            "global_step": int(trainer_state["global_step"]),
            "max_steps": int(trainer_state.get("max_steps", -1)),
        },
        "scheduler_last_epoch": int(scheduler["last_epoch"]),
        "rng_files": [str(path) for path in rng_files],
        "runtime_contract": runtime_contract,
        "local_model_config_path": str(model_config_path),
        "local_model_config_audio_contract": {
            key: model_config.get(key) for key in EXPECTED_AUDIO_CONFIG
        },
        "xares_representation_contract": {
            "input_sampling_rate": 16000,
            "output_shape": "[B, T_dynamic, 5280]",
            "output_dim": 5280,
            "hop_size_in_ms": 240,
            "maximum_seconds": 30.0,
            "maximum_content_tokens": 125,
            "boundary_tokens_included": False,
            "source_modules": [
                "audio_encoder.encoder",
                "audio_aligner.temporal_compressor",
                "audio_aligner.audio_projector",
            ],
        },
        "representative_tensor_probes": probes,
        "optional_checkpoint_files": optional_files,
    }
    return report


def main() -> None:
    args = parse_args()
    report = inspect_checkpoint(args)
    output = args.output_report.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[xares-checkpoint] checkpoint={report['checkpoint']}", flush=True)
    print(f"[xares-checkpoint] step={report['step']} model_counts={report['model_parameter_metadata_counts']}", flush=True)
    print(f"[xares-checkpoint] representation={report['xares_representation_contract']}", flush=True)
    print(f"[xares-checkpoint] report={output}", flush=True)
    print("========== HUGINN X-ARES CHECKPOINT READ-ONLY INSPECT PASSED ==========", flush=True)


if __name__ == "__main__":
    main()
