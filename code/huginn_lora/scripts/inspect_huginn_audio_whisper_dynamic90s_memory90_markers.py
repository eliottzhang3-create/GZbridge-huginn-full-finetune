#!/usr/bin/env python3
"""Validate the four-rank trainable-Whisper 90-second memory smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    return parser.parse_args()


def read_marker(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing or empty 90-second memory marker: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Memory marker is not an object: {path}")
    return payload


def validate_optimizer_groups(marker: dict, trainables: dict, rank: int) -> None:
    audits = marker.get("optimizer_group_audit")
    if not isinstance(audits, list) or not audits:
        raise AssertionError(f"Rank {rank} has no optimizer-group audit: {marker}")
    observed = {name: 0 for name in ("lora", "aligner", "audio_encoder", "huginn_base", "other")}
    for audit in audits:
        if abs(float(audit.get("learning_rate", -1.0)) - 1e-4) > 1e-12:
            raise AssertionError(f"Rank {rank} optimizer LR is not 1e-4: {audits}")
        counts = audit.get("parameter_counts")
        if not isinstance(counts, dict):
            raise AssertionError(f"Rank {rank} optimizer parameter counts are missing: {audits}")
        for name in observed:
            observed[name] += int(counts.get(name, 0))
    expected = {name: int(trainables.get(name, 0)) for name in observed}
    if observed != expected:
        raise AssertionError(
            f"Rank {rank} optimizer ownership mismatch: observed={observed} expected={expected}"
        )


def main() -> None:
    args = parse_args()
    devices: set[int] = set()
    max_allocated: list[float] = []
    max_reserved: list[float] = []
    for rank in range(4):
        fsdp = read_marker(args.audit_dir / f"fsdp-rank-{rank}.json")
        if (
            fsdp.get("kind") != "fsdp"
            or fsdp.get("stage") != "memory90"
            or fsdp.get("rank") != rank
            or fsdp.get("world_size") != 4
        ):
            raise AssertionError(f"Invalid memory90 FSDP marker for rank {rank}: {fsdp}")
        trainables = fsdp.get("trainable_tensors")
        if (
            not isinstance(trainables, dict)
            or int(trainables.get("lora", -1)) != 66
            or int(trainables.get("aligner", -1)) != 14
            or int(trainables.get("audio_encoder", 0)) <= 0
            or int(trainables.get("huginn_base", -1)) != 0
            or int(trainables.get("other", -1)) != 0
        ):
            raise AssertionError(f"Rank {rank} trainable split mismatch: {trainables}")
        if int(fsdp.get("dtensor_trainable_count", -1)) != sum(int(value) for value in trainables.values()):
            raise AssertionError(f"Rank {rank} DTensor trainable count mismatch: {fsdp}")
        units = fsdp.get("fsdp_units")
        if not isinstance(units, dict) or set(units) != {
            "WhisperEncoderFSDPUnit",
            "AudioAlignerFSDPUnit",
            "HuginnPreludeFSDPUnit",
            "HuginnRecurrentCoreFSDPUnit",
            "HuginnCodaFSDPUnit",
        }:
            raise AssertionError(f"Rank {rank} FSDP unit topology mismatch: {units}")
        whisper_unit = units["WhisperEncoderFSDPUnit"]
        if (
            int(whisper_unit.get("parameter_count", 0)) <= 0
            or whisper_unit.get("parameter_count") != whisper_unit.get("dtensor_parameter_count")
            or whisper_unit.get("parameter_count") != whisper_unit.get("trainable_parameter_count")
        ):
            raise AssertionError(f"Rank {rank} Whisper unit is not fully trainable/sharded: {whisper_unit}")
        if fsdp.get("valid_prefix_tokens") != [752, 752]:
            raise AssertionError(f"Rank {rank} did not receive two complete 90-second prefixes: {fsdp}")

        memory = read_marker(args.audit_dir / f"memory90-rank-{rank}.json")
        if (
            memory.get("kind") != "memory90"
            or memory.get("stage") != "memory90"
            or memory.get("rank") != rank
            or memory.get("world_size") != 4
            or memory.get("global_step") != 1
            or memory.get("per_device_train_batch_size") != 2
            or memory.get("gradient_accumulation_steps") != 4
            or memory.get("global_batch_size") != 32
            or memory.get("finite_loss_log_count") != 1
            or memory.get("finite_grad_norm_log_count") != 1
            or int(memory.get("optimizer_state_count", -1)) != sum(int(value) for value in trainables.values())
        ):
            raise AssertionError(f"Invalid memory90 runtime marker for rank {rank}: {memory}")
        gradients = memory.get("gradient_audit")
        if not isinstance(gradients, dict):
            raise AssertionError(f"Rank {rank} has no gradient audit: {memory}")
        for group in ("lora", "aligner", "audio_encoder"):
            if int(gradients.get(group, {}).get("nonzero_gradient_tensors", 0)) <= 0:
                raise AssertionError(f"Rank {rank} has no nonzero {group} gradients: {gradients}")
        for group in ("huginn_base", "other"):
            if int(gradients.get(group, {}).get("gradient_tensors", -1)) != 0:
                raise AssertionError(f"Rank {rank} observed forbidden {group} gradients: {gradients}")
        validate_optimizer_groups(memory, trainables, rank)

        devices.add(int(fsdp["cuda_device"]))
        max_allocated.append(float(memory["max_memory_allocated_gib"]))
        max_reserved.append(float(memory["max_memory_reserved_gib"]))
        print(
            f"[memory90-marker] rank={rank} cuda={fsdp['cuda_device']} "
            f"whisper_tensors={trainables['audio_encoder']} prefix_tokens={fsdp['valid_prefix_tokens']} "
            f"optimizer_states={memory['optimizer_state_count']} "
            f"max_allocated_gib={memory['max_memory_allocated_gib']:.3f} "
            f"max_reserved_gib={memory['max_memory_reserved_gib']:.3f} gradients={gradients}"
        )

    if devices != {0, 1, 2, 3}:
        raise AssertionError(f"Expected CUDA devices 0-3, observed {sorted(devices)}")
    print(
        "[memory90-summary] "
        f"max_allocated_gib={max(max_allocated):.3f} "
        f"max_reserved_gib={max(max_reserved):.3f} global_batch_size=32"
    )
    print("========== HUGINN WHISPER DYNAMIC90S MEMORY90 FSDP4 PASSED ==========")


if __name__ == "__main__":
    main()
