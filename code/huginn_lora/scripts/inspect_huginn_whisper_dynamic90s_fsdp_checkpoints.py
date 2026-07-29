#!/usr/bin/env python3
"""Audit and compare dynamic-90s Whisper LoRA+aligner FSDP checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


EXPECTED_COUNTS = {"lora": 66, "aligner": 14, "other": 0}
ALIGNER_PREFIXES = (
    "audio_aligner",
    "temporal_compressor",
    "audio_projector",
    "audio_boundary_embeddings",
    "audio_bos",
    "audio_eos",
)


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
    return "other"


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required checkpoint JSON is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint JSON is not an object: {path}")
    return payload


def inspect_checkpoint(path: Path, expected_step: int, world_size: int) -> dict[str, Any]:
    from torch.distributed.checkpoint import FileSystemReader

    checkpoint = path.resolve()
    model_dir = checkpoint / "pytorch_model_fsdp_0"
    if not checkpoint.is_dir() or not model_dir.is_dir():
        raise FileNotFoundError(f"FSDP checkpoint/model directory is missing: {checkpoint}")
    metadata = FileSystemReader(str(model_dir)).read_metadata()
    state_metadata = getattr(metadata, "state_dict_metadata", {})
    grouped: dict[str, list[str]] = {"lora": [], "aligner": [], "other": []}
    for raw_key in state_metadata:
        key = str(raw_key)
        grouped[classify_key(key)].append(key)
    counts = {name: len(keys) for name, keys in grouped.items()}
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(
            f"Checkpoint model contract mismatch at {checkpoint}: expected={EXPECTED_COUNTS} actual={counts} "
            f"other_preview={grouped['other'][:10]}"
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
    for candidate in checkpoint.iterdir():
        if candidate == model_dir or not candidate.is_dir() or not (candidate / ".metadata").is_file():
            continue
        candidate_metadata = FileSystemReader(str(candidate)).read_metadata()
        if getattr(candidate_metadata, "state_dict_metadata", {}):
            optimizer_dcp_dirs.append(candidate)
    if not optimizer_dcp_dirs:
        raise RuntimeError(f"No optimizer DCP state directory was found at {checkpoint}")

    print(
        f"[checkpoint] path={checkpoint} step={expected_step} model_counts={counts} "
        f"scheduler_last_epoch={scheduler_last_epoch} rng_files={len(rng_files)} "
        f"optimizer_dcp_dirs={[path.name for path in optimizer_dcp_dirs]}"
    )
    return {
        "path": str(checkpoint),
        "step": expected_step,
        "model_dir": str(model_dir),
        "state_metadata": state_metadata,
        "grouped_keys": grouped,
        "counts": counts,
        "scheduler_last_epoch": scheduler_last_epoch,
        "rng_files": [str(path) for path in rng_files],
        "optimizer_dcp_dirs": [str(path) for path in optimizer_dcp_dirs],
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
    changed = {"lora": 0, "aligner": 0, "other": 0}
    unchanged = {"lora": 0, "aligner": 0, "other": 0}
    for index, key in enumerate(sorted(saved_metadata), start=1):
        left = load_one_tensor(saved["model_dir"], str(key), saved_metadata[key])
        right = load_one_tensor(resumed["model_dir"], str(key), resumed_metadata[key])
        if left.shape != right.shape or left.dtype != right.dtype:
            raise RuntimeError(
                f"Checkpoint tensor metadata changed across resume: key={key} "
                f"left={left.shape}/{left.dtype} right={right.shape}/{right.dtype}"
            )
        group = classify_key(str(key))
        if torch.equal(left, right):
            unchanged[group] += 1
        else:
            changed[group] += 1
        del left, right
        if index == 1 or index % 20 == 0 or index == len(saved_metadata):
            print(f"[compare-progress] tensors={index}/{len(saved_metadata)}", flush=True)
    if changed["lora"] <= 0 or changed["aligner"] <= 0 or changed["other"] != 0:
        raise RuntimeError(
            "Cold-resume updates did not change both required trainable groups: "
            f"changed={changed} unchanged={unchanged}"
        )
    print(f"[model-change] changed={changed} unchanged={unchanged}")
    return {"changed": changed, "unchanged": unchanged}


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
    saved = inspect_checkpoint(args.save_checkpoint, args.save_step, args.world_size)
    resumed = inspect_checkpoint(args.resume_checkpoint, args.resume_step, args.world_size)
    comparison = compare_model_states(saved, resumed)
    report = {
        "gate": "huginn_whisper_dynamic90s_fsdp4_checkpoint_content_v1",
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
