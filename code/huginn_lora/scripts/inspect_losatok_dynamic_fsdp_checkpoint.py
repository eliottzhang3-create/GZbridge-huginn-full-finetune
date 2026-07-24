#!/usr/bin/env python3
"""Read-only content audit for dynamic LoSATok FSDP2 model checkpoints.

This verifies the *model-weight* contents of a Swift DCP checkpoint.  It does
not build Huginn/LoSATok, allocate CUDA memory, decode audio, or change any
checkpoint file.  In particular, it distinguishes LoRA tensors from the
separately trainable audio aligner tensors required for faithful evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from generate_clotho_caption_samples_swift import ALIGNER_PREFIXES, candidate_target_keys


DEFAULT_RUN_ROOT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/"
    "huginn_losatok_dynamic90s_audiocaps_v2_e3_b4ga4_fsdp2/v0-20260723-054928"
)
FSDP_MODEL_DIR_NAME = "pytorch_model_fsdp_0"
TENSOR_FILE_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth"}
AUXILIARY_STATE_FILE_PREFIXES = ("rng_state", "scheduler", "optimizer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        dest="checkpoints",
        default=[],
        help="Checkpoint directory; pass once per checkpoint.",
    )
    parser.add_argument(
        "--require_complete",
        action="store_true",
        help="Exit nonzero unless every inspected checkpoint has exactly 66 LoRA and 20 aligner tensors.",
    )
    return parser.parse_args()


def classify_key(key: str) -> str:
    aliases = candidate_target_keys(key)
    if any(".lora_A." in alias or ".lora_B." in alias for alias in aliases):
        return "lora"
    if any(alias.startswith(ALIGNER_PREFIXES) for alias in aliases):
        return "aligner"
    return "other"


def metadata_entry_summary(key: str, metadata: Any) -> str:
    shape = getattr(metadata, "size", None)
    properties = getattr(metadata, "properties", None)
    dtype = getattr(properties, "dtype", None)
    return f"key={key} shape={tuple(shape) if shape is not None else None} dtype={dtype}"


def weight_sidecars(root: Path, *, exclude_root: Path | None = None) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TENSOR_FILE_SUFFIXES:
            continue
        if any(path.name.startswith(prefix) for prefix in AUXILIARY_STATE_FILE_PREFIXES):
            continue
        if exclude_root is not None and path.is_relative_to(exclude_root):
            continue
        files.append(path)
    return files


def inspect_checkpoint(checkpoint_dir: Path) -> dict[str, int]:
    from torch.distributed.checkpoint import FileSystemReader

    checkpoint_dir = checkpoint_dir.resolve()
    model_dir = checkpoint_dir / FSDP_MODEL_DIR_NAME
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"FSDP model directory does not exist: {model_dir}")

    metadata = FileSystemReader(str(model_dir)).read_metadata()
    state_metadata = getattr(metadata, "state_dict_metadata", {})
    if not state_metadata:
        raise RuntimeError(f"FSDP DCP metadata has no state entries: {model_dir}")

    grouped: dict[str, list[tuple[str, Any]]] = {"lora": [], "aligner": [], "other": []}
    for raw_key, entry_metadata in state_metadata.items():
        key = str(raw_key)
        grouped[classify_key(key)].append((key, entry_metadata))

    sidecars = weight_sidecars(checkpoint_dir)
    run_root_sidecars = weight_sidecars(checkpoint_dir.parent, exclude_root=checkpoint_dir)

    print("========== LOSATOK DYNAMIC FSDP2 CHECKPOINT CONTENT AUDIT ==========")
    print(f"[checkpoint] path={checkpoint_dir}")
    print(f"[dcp] model_dir={model_dir}")
    print(f"[dcp] metadata_tensor_count={len(state_metadata)}")
    for group in ("lora", "aligner", "other"):
        entries = grouped[group]
        print(f"[dcp] {group}_tensor_count={len(entries)}")
        for key, entry_metadata in entries[:8]:
            print(f"[dcp-{group}] {metadata_entry_summary(key, entry_metadata)}")
    print(f"[sidecar] checkpoint_weight_file_count={len(sidecars)}")
    for path in sidecars:
        print(f"[sidecar] path={path.relative_to(checkpoint_dir)} bytes={path.stat().st_size}")
    print(f"[sidecar] run_root_external_weight_file_count={len(run_root_sidecars)}")
    for path in run_root_sidecars:
        print(f"[sidecar-run-root] path={path.relative_to(checkpoint_dir.parent)} bytes={path.stat().st_size}")

    complete = len(grouped["lora"]) == 66 and len(grouped["aligner"]) == 20
    print(
        "[result] "
        f"status={'PASS' if complete else 'INCOMPLETE'} "
        f"expected_lora=66 actual_lora={len(grouped['lora'])} "
        f"expected_aligner=20 actual_aligner={len(grouped['aligner'])}"
    )
    return {group: len(entries) for group, entries in grouped.items()}


def main() -> None:
    args = parse_args()
    checkpoints = [Path(value) for value in args.checkpoints]
    if not checkpoints:
        checkpoints = [DEFAULT_RUN_ROOT / "checkpoint-2802", DEFAULT_RUN_ROOT / "checkpoint-5604"]
    totals = {"lora": 0, "aligner": 0, "other": 0}
    per_checkpoint_counts: list[dict[str, int]] = []
    for checkpoint in checkpoints:
        counts = inspect_checkpoint(checkpoint)
        per_checkpoint_counts.append(counts)
        for group, count in counts.items():
            totals[group] += count
    print(
        "[summary] "
        f"checkpoint_count={len(checkpoints)} total_lora={totals['lora']} "
        f"total_aligner={totals['aligner']} total_other={totals['other']}"
    )
    if args.require_complete and any(
        counts["lora"] != 66 or counts["aligner"] != 20
        for counts in per_checkpoint_counts
    ):
        raise SystemExit(
            "One or more inspected FSDP checkpoints are incomplete; "
            f"expected_per_checkpoint=(lora=66, aligner=20) totals={totals}"
        )


if __name__ == "__main__":
    main()
