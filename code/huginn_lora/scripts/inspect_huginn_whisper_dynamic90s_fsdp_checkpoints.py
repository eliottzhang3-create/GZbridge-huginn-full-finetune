#!/usr/bin/env python3
"""Audit and compare full-model dynamic-90s Whisper FSDP checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


GROUP_NAMES = ("lora", "aligner", "audio_encoder", "huginn_base", "other")
ALIGNER_PREFIXES = (
    "audio_aligner",
    "temporal_compressor",
    "audio_projector",
    "audio_boundary_embeddings",
    "audio_bos",
    "audio_eos",
)
AUDIO_BOUNDARY_PARAMETER_NAMES = ("audio_bos", "audio_eos")
AUDIO_BOUNDARY_CHECKPOINT_ROLES = ("trainable_active", "frozen_original")
TRAINING_RUNTIME_CONTRACT_FILENAME = "huginn_training_runtime_contract.json"
EXPECTED_RESHARD_AFTER_FORWARD = {
    "WhisperEncoderFSDPUnit": True,
    "AudioAlignerFSDPUnit": True,
    "HuginnPreludeFSDPUnit": True,
    "HuginnRecurrentCoreFSDPUnit": False,
    "HuginnCodaFSDPUnit": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-checkpoint", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path, required=True)
    parser.add_argument("--save-step", type=int, default=4)
    parser.add_argument("--resume-step", type=int, default=6)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def key_aliases(key: str) -> set[str]:
    aliases = {key}
    changed = True
    while changed:
        changed = False
        for alias in list(aliases):
            for prefix in ("base_model.model.", "base_model.", "model.", "module."):
                if alias.startswith(prefix):
                    stripped = alias[len(prefix):]
                    if stripped not in aliases:
                        aliases.add(stripped)
                        changed = True
    normalized = set()
    for alias in aliases:
        normalized.add(alias)
        normalized.add(alias.replace(".modules_to_save.default.", "."))
        normalized.add(alias.replace(".original_module.", "."))
        normalized.add(alias.replace(".lora_A.default.", ".lora_A."))
        normalized.add(alias.replace(".lora_B.default.", ".lora_B."))
    return normalized


def classify_key(key: str) -> str:
    aliases = key_aliases(key)
    if any(".lora_A." in alias or ".lora_B." in alias for alias in aliases):
        return "lora"
    if any(alias.startswith(ALIGNER_PREFIXES) for alias in aliases):
        return "aligner"
    if any(alias.startswith("audio_encoder.") for alias in aliases):
        return "audio_encoder"
    if any(alias.startswith(("transformer.", "lm_head.")) for alias in aliases):
        return "huginn_base"
    return "other"


def audio_boundary_key_name(key: str) -> str | None:
    for alias in key_aliases(key):
        for boundary_name in AUDIO_BOUNDARY_PARAMETER_NAMES:
            if alias == boundary_name or alias.endswith(
                f"audio_boundary_embeddings.{boundary_name}"
            ):
                return boundary_name
    return None


def audio_boundary_key_identity(key: str) -> tuple[str, str] | None:
    boundary_name = audio_boundary_key_name(key)
    if boundary_name is None:
        return None
    if ".modules_to_save.default." in key:
        return boundary_name, "trainable_active"
    if ".original_module." in key:
        return boundary_name, "frozen_original"
    raise RuntimeError(f"Audio boundary checkpoint key has no PEFT ownership role: {key}")


def classify_optimizer_key(key: str) -> str:
    """Classify named optimizer-state entries without assuming one DCP layout."""
    normalized = key
    normalized = normalized.replace(".modules_to_save.default.", ".")
    normalized = normalized.replace(".original_module.", ".")
    normalized = normalized.replace(".lora_A.default.", ".lora_A.")
    normalized = normalized.replace(".lora_B.default.", ".lora_B.")
    if ".lora_A." in normalized or ".lora_B." in normalized:
        return "lora"
    if any(
        normalized.startswith(prefix)
        or f".{prefix}." in normalized
        for prefix in ALIGNER_PREFIXES
    ):
        return "aligner"
    if normalized.startswith("audio_encoder.") or ".audio_encoder." in normalized:
        return "audio_encoder"
    if (
        normalized.startswith(("transformer.", "lm_head."))
        or ".transformer." in normalized
        or ".lm_head." in normalized
    ):
        return "huginn_base"
    return "other"


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required checkpoint JSON is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint JSON is not an object: {path}")
    return payload


def inspect_checkpoint(
    path: Path,
    expected_step: int,
    world_size: int,
    expected_phase: str,
) -> dict[str, Any]:
    from torch.distributed.checkpoint import FileSystemReader

    checkpoint = path.resolve()
    model_dir = checkpoint / "pytorch_model_fsdp_0"
    if not checkpoint.is_dir() or not model_dir.is_dir():
        raise FileNotFoundError(f"FSDP checkpoint/model directory is missing: {checkpoint}")
    metadata = FileSystemReader(str(model_dir)).read_metadata()
    state_metadata = getattr(metadata, "state_dict_metadata", {})
    state_metadata_by_key = {str(key): value for key, value in state_metadata.items()}
    grouped: dict[str, list[str]] = {name: [] for name in GROUP_NAMES}
    for raw_key in state_metadata:
        key = str(raw_key)
        grouped[classify_key(key)].append(key)
    counts = {name: len(keys) for name, keys in grouped.items()}
    boundary_keys = {
        name: {role: [] for role in AUDIO_BOUNDARY_CHECKPOINT_ROLES}
        for name in AUDIO_BOUNDARY_PARAMETER_NAMES
    }
    for key in grouped["aligner"]:
        identity = audio_boundary_key_identity(key)
        if identity is not None:
            boundary_name, role = identity
            boundary_keys[boundary_name][role].append(key)
    if (
        counts["lora"] != 66
        or counts["aligner"] < 14
        or counts["audio_encoder"] <= 0
        or counts["huginn_base"] <= 0
        or counts["other"] != 0
    ):
        raise RuntimeError(
            f"Full-model checkpoint contract mismatch at {checkpoint}: actual={counts} "
            f"other_preview={grouped['other'][:10]}"
        )
    invalid_boundary_keys = {
        f"{name}:{role}": keys
        for name, role_keys in boundary_keys.items()
        for role, keys in role_keys.items()
        if len(keys) != 1
    }
    if invalid_boundary_keys:
        raise RuntimeError(
            f"Audio boundary checkpoint contract mismatch at {checkpoint}: "
            f"invalid={invalid_boundary_keys} all={boundary_keys}"
        )

    trainer_state = load_json(checkpoint / "trainer_state.json")
    if int(trainer_state.get("global_step", -1)) != expected_step:
        raise RuntimeError(
            f"Trainer global_step mismatch at {checkpoint}: "
            f"expected={expected_step} actual={trainer_state.get('global_step')}"
        )
    scheduler_path = checkpoint / "scheduler.pt"
    if not scheduler_path.is_file():
        raise FileNotFoundError(f"Scheduler state is missing: {scheduler_path}")
    scheduler_state = torch.load(scheduler_path, map_location="cpu", weights_only=False)
    scheduler_last_epoch = int(scheduler_state.get("last_epoch", -1))
    if scheduler_last_epoch != expected_step:
        raise RuntimeError(
            f"Scheduler last_epoch mismatch at {checkpoint}: expected={expected_step} actual={scheduler_last_epoch}"
        )

    rng_files = sorted(checkpoint.glob("rng_state*.pth"))
    if len(rng_files) != world_size:
        raise RuntimeError(
            f"Expected {world_size} per-rank RNG files at {checkpoint}, found {len(rng_files)}: {rng_files}"
        )
    optimizer_dcp_dirs = []
    optimizer_metadata_counts: dict[str, dict[str, int]] = {}
    for candidate in checkpoint.iterdir():
        if candidate == model_dir or not candidate.is_dir() or not (candidate / ".metadata").is_file():
            continue
        candidate_metadata = FileSystemReader(str(candidate)).read_metadata()
        candidate_state_metadata = getattr(candidate_metadata, "state_dict_metadata", {})
        if candidate_state_metadata:
            optimizer_dcp_dirs.append(candidate)
            optimizer_groups = {name: [] for name in GROUP_NAMES}
            for raw_key in candidate_state_metadata:
                key = str(raw_key)
                optimizer_groups[classify_optimizer_key(key)].append(key)
            optimizer_metadata_counts[candidate.name] = {
                name: len(keys) for name, keys in optimizer_groups.items()
            }
            print(
                f"[optimizer-metadata] checkpoint={checkpoint.name} dir={candidate.name} "
                f"counts={optimizer_metadata_counts[candidate.name]} "
                f"aligner_preview={optimizer_groups['aligner'][:4]} "
                f"audio_encoder_preview={optimizer_groups['audio_encoder'][:4]}"
            )
    if not optimizer_dcp_dirs:
        raise RuntimeError(f"No optimizer DCP state directory was found at {checkpoint}")

    runtime_contract = load_json(checkpoint / TRAINING_RUNTIME_CONTRACT_FILENAME)
    expected_checkpointing = {
        "fsdp_activation_checkpointing": True,
        "whisper_internal_gradient_checkpointing": False,
        "whisper_outer_activation_checkpointed": True,
        "double_checkpoint_candidate": False,
    }
    if (
        runtime_contract.get("gate") != "huginn_whisper_dynamic30s_training_runtime_contract_v1"
        or runtime_contract.get("phase") != expected_phase
        or int(runtime_contract.get("global_step", -1)) != expected_step
        or int(runtime_contract.get("world_size", -1)) != world_size
        or runtime_contract.get("checkpointing") != expected_checkpointing
        or runtime_contract.get("fsdp_reshard_after_forward") != EXPECTED_RESHARD_AFTER_FORWARD
        or runtime_contract.get("trainability")
        != {
            "whisper_encoder": True,
            "audio_aligner": True,
            "huginn_lora": True,
            "huginn_native_backbone": False,
        }
        or runtime_contract.get("learning_rates")
        != {
            "whisper_encoder": 1e-4,
            "audio_aligner": 1e-4,
            "huginn_lora": 1e-4,
        }
        or runtime_contract.get("lora")
        != {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.05,
            "tensor_count": 66,
            "target_module_count": 33,
            "direct_lora_layer_count": 33,
            "restricted_to_huginn_transformer": True,
        }
        or runtime_contract.get("audio")
        != {
            "maximum_seconds": 30.0,
            "token_duration_ms": 160,
            "maximum_content_tokens": 187,
            "trainable_boundary_tokens": ["audio_bos", "audio_eos"],
        }
        or runtime_contract.get("loss")
        != {
            "type": "shifted_next_token_prediction",
            "supervision": "assistant_response_only",
            "audio_prefix_labels": -100,
        }
    ):
        raise RuntimeError(
            f"Training runtime contract mismatch at {checkpoint}: {runtime_contract}"
        )

    print(
        f"[checkpoint] path={checkpoint} step={expected_step} model_counts={counts} "
        f"scheduler_last_epoch={scheduler_last_epoch} rng_files={len(rng_files)} "
        f"optimizer_dcp_dirs={[path.name for path in optimizer_dcp_dirs]}"
    )
    for group in ("lora", "aligner", "audio_encoder", "huginn_base"):
        preview = []
        for key in grouped[group][:8]:
            entry = state_metadata_by_key[key]
            preview.append({
                "key": key,
                "shape": tuple(int(value) for value in getattr(entry, "size", ())),
                "dtype": str(getattr(getattr(entry, "properties", None), "dtype", None)),
            })
        print(f"[model-metadata] checkpoint={checkpoint.name} group={group} preview={preview}")
    return {
        "path": str(checkpoint),
        "step": expected_step,
        "model_dir": str(model_dir),
        "state_metadata": state_metadata,
        "grouped_keys": grouped,
        "counts": counts,
        "audio_boundary_keys": boundary_keys,
        "scheduler_last_epoch": scheduler_last_epoch,
        "rng_files": [str(path) for path in rng_files],
        "optimizer_dcp_dirs": [str(path) for path in optimizer_dcp_dirs],
        "optimizer_metadata_counts": optimizer_metadata_counts,
        "training_runtime_contract": runtime_contract,
    }


def load_one_tensor(model_dir: str, key: str, metadata: Any) -> torch.Tensor:
    import torch.distributed.checkpoint as dcp

    shape = tuple(int(value) for value in getattr(metadata, "size", ()))
    dtype = getattr(getattr(metadata, "properties", None), "dtype", None)
    if not shape or not isinstance(dtype, torch.dtype):
        raise RuntimeError(f"DCP metadata lacks shape/dtype: key={key} shape={shape} dtype={dtype}")
    tensor = torch.empty(shape, dtype=dtype, device="cpu")
    dcp.load({key: tensor}, checkpoint_id=model_dir)
    if not bool(torch.isfinite(tensor).all().item()):
        raise RuntimeError(f"Checkpoint tensor contains non-finite values: {model_dir}:{key}")
    return tensor


def compare_model_states(saved: dict[str, Any], resumed: dict[str, Any]) -> dict[str, Any]:
    saved_metadata = saved["state_metadata"]
    resumed_metadata = resumed["state_metadata"]
    if set(saved_metadata) != set(resumed_metadata):
        raise RuntimeError(
            "Save/resume checkpoint model keys differ: "
            f"save_only={sorted(set(saved_metadata) - set(resumed_metadata))[:10]} "
            f"resume_only={sorted(set(resumed_metadata) - set(saved_metadata))[:10]}"
        )
    changed = {name: 0 for name in GROUP_NAMES}
    unchanged = {name: 0 for name in GROUP_NAMES}
    max_abs_delta = {name: 0.0 for name in GROUP_NAMES}
    dtypes: dict[str, set[str]] = {name: set() for name in GROUP_NAMES}
    boundary_changes = {
        name: {
            role: {"changed": 0, "unchanged": 0, "max_abs_delta": 0.0}
            for role in AUDIO_BOUNDARY_CHECKPOINT_ROLES
        }
        for name in AUDIO_BOUNDARY_PARAMETER_NAMES
    }
    for index, key in enumerate(sorted(saved_metadata), start=1):
        left = load_one_tensor(saved["model_dir"], str(key), saved_metadata[key])
        right = load_one_tensor(resumed["model_dir"], str(key), resumed_metadata[key])
        if left.shape != right.shape or left.dtype != right.dtype:
            raise RuntimeError(
                f"Checkpoint tensor metadata changed across resume: key={key} "
                f"left={left.shape}/{left.dtype} right={right.shape}/{right.dtype}"
            )
        group = classify_key(str(key))
        dtypes[group].add(str(left.dtype))
        boundary_identity = audio_boundary_key_identity(str(key))
        if torch.equal(left, right):
            unchanged[group] += 1
            if boundary_identity is not None:
                boundary_name, role = boundary_identity
                boundary_changes[boundary_name][role]["unchanged"] += 1
        else:
            changed[group] += 1
            tensor_delta = float((left.float() - right.float()).abs().max().item())
            max_abs_delta[group] = max(max_abs_delta[group], tensor_delta)
            if boundary_identity is not None:
                boundary_name, role = boundary_identity
                boundary_changes[boundary_name][role]["changed"] += 1
                boundary_changes[boundary_name][role]["max_abs_delta"] = tensor_delta
        del left, right
        if index == 1 or index % 20 == 0 or index == len(saved_metadata):
            print(f"[compare-progress] tensors={index}/{len(saved_metadata)}", flush=True)
    comparison_summary = {
        "changed": changed,
        "unchanged": unchanged,
        "max_abs_delta": max_abs_delta,
        "dtypes": {name: sorted(values) for name, values in dtypes.items()},
        "audio_boundary_changes": boundary_changes,
    }
    print(f"[model-change-summary] {comparison_summary}")
    if (
        changed["lora"] <= 0
        or changed["aligner"] <= 0
        or changed["audio_encoder"] <= 0
        or changed["huginn_base"] != 0
        or changed["other"] != 0
        or any(
            role_audits["trainable_active"]["changed"] != 1
            or role_audits["trainable_active"]["unchanged"] != 0
            or role_audits["trainable_active"]["max_abs_delta"] <= 0.0
            or role_audits["frozen_original"]["changed"] != 0
            or role_audits["frozen_original"]["unchanged"] != 1
            or role_audits["frozen_original"]["max_abs_delta"] != 0.0
            for role_audits in boundary_changes.values()
        )
    ):
        raise RuntimeError(
            "Cold-resume updates do not match Whisper+aligner+LoRA trainability: "
            f"{comparison_summary}"
        )
    print(f"[model-change] changed={changed} unchanged={unchanged} max_abs_delta={max_abs_delta}")
    return comparison_summary


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.save_step <= 0 or args.resume_step <= args.save_step or args.world_size != 4:
        raise ValueError("Expected 0 < save_step < resume_step and world_size=4")
    saved = inspect_checkpoint(args.save_checkpoint, args.save_step, args.world_size, "save")
    resumed = inspect_checkpoint(args.resume_checkpoint, args.resume_step, args.world_size, "resume")
    comparison = compare_model_states(saved, resumed)
    report = {
        "gate": "huginn_whisper_dynamic30s_full_model_fsdp4_checkpoint_content_v4",
        "validation_passed": True,
        "save_checkpoint": {key: value for key, value in saved.items() if key not in {"state_metadata", "grouped_keys"}},
        "resume_checkpoint": {
            key: value for key, value in resumed.items() if key not in {"state_metadata", "grouped_keys"}
        },
        "model_comparison": comparison,
    }
    write_json_atomic(args.output_report, report)
    print(f"[report] path={args.output_report}")
    print("========== HUGINN WHISPER DYNAMIC90S FSDP CHECKPOINT CONTENT PASSED ==========")


if __name__ == "__main__":
    main()
