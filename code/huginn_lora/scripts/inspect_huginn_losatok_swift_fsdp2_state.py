"""Inspect LoSATok Swift/PEFT state-dict buffers before FSDP2 preparation.

This is a read-only one-GPU diagnostic.  It deliberately does not enable FSDP,
does not call ``Trainer.train`` and does not save a checkpoint.  Its purpose is
to determine whether LoSATok BatchNorm buffers disappear on the raw model,
reappear after PEFT wrapping, or remain in the final Trainer model state dict.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import platform
import sys
import tempfile
from pathlib import Path

import torch


def load_shared_inspect_module(repo_root: Path):
    source_path = repo_root / "code" / "huginn_lora" / "scripts" / "inspect_huginn_audio_swift_trainables.py"
    spec = importlib.util.spec_from_file_location("huginn_swift_trainable_inspect_shared", source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load shared inspect helpers: {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_inspect_manifest(repo_root: Path) -> Path:
    source_manifest = repo_root / "data" / "audio_swift" / "audiocaps_v2" / "audiocaps_v2_train_swift.jsonl"
    runtime_dir = Path(tempfile.mkdtemp(prefix="huginn_losatok_fsdp2_state_inspect_"))
    inspect_manifest = runtime_dir / "audiocaps_v2_one_record.jsonl"
    if not source_manifest.is_file() or source_manifest.stat().st_size == 0:
        raise FileNotFoundError(f"AudioCaps manifest is missing or empty: {source_manifest}")
    with source_manifest.open(encoding="utf-8") as handle:
        first_record = next((line.rstrip("\n") for line in handle if line.strip()), None)
    if first_record is None:
        raise ValueError(f"AudioCaps manifest has no records: {source_manifest}")
    inspect_manifest.write_text(first_record + "\n", encoding="utf-8")
    print(f"[inspect] source_manifest={source_manifest}")
    print(f"[inspect] one_record_manifest={inspect_manifest}")
    return inspect_manifest


def build_argv(repo_root: Path) -> list[str]:
    manifest = build_inspect_manifest(repo_root)
    model = repo_root / "models" / "huginn-audio-losatok-v1"
    plugin = repo_root / "code" / "huginn_lora" / "plugins" / "huginn_losatok_swift.py"
    return [
        "--model", str(model),
        "--model_type", "huginn_losatok_raven",
        "--template", "huginn_losatok_text",
        "--external_plugins", str(plugin),
        "--dataset", str(manifest),
        "--max_length", "192",
        "--output_dir", str(repo_root / "outputs" / "huginn_losatok_fsdp2_state_inspect"),
        "--tuner_type", "lora_llm",
        "--freeze_aligner", "false",
        "--learning_rate", "1e-4",
        "--aligner_lr", "1e-4",
        "--lora_rank", "16",
        "--lora_alpha", "32",
        "--lora_dropout", "0.05",
        "--max_steps", "1",
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "1",
        "--logging_steps", "1",
        "--save_strategy", "no",
        "--dataloader_num_workers", "0",
        "--dataloader_pin_memory", "false",
        "--dataset_num_proc", "1",
        "--save_only_model", "false",
        "--report_to", "none",
        "--bf16", "true",
    ]


def resolve_module(model: torch.nn.Module, module_path: str) -> torch.nn.Module | None:
    if not module_path:
        return model
    try:
        return model.get_submodule(module_path)
    except (AttributeError, KeyError):
        return None


def print_buffer_inventory(model: torch.nn.Module, label: str) -> None:
    print(f"========== {label} BUFFERS ==========")
    found = False
    for module_path, module in model.named_modules():
        for buffer_name, buffer in module._buffers.items():
            if buffer_name != "num_batches_tracked" and buffer_name != "freqs_cis":
                continue
            found = True
            fqn = f"{module_path}.{buffer_name}" if module_path else buffer_name
            persistent = buffer_name not in module._non_persistent_buffers_set
            shape = None if buffer is None else tuple(buffer.shape)
            dtype = None if buffer is None else str(buffer.dtype)
            device = None if buffer is None else str(buffer.device)
            print(
                f"[buffer] fqn={fqn} type={type(module).__name__} value_none={buffer is None} "
                f"persistent={persistent} shape={shape} dtype={dtype} device={device}"
            )
    if not found:
        print("[buffer] no freqs_cis or num_batches_tracked buffers found")


def print_state_matches(model: torch.nn.Module, label: str) -> None:
    print(f"========== {label} STATE-DICT MATCHES ==========")
    state = model.state_dict()
    matches = [(name, value) for name, value in state.items() if "num_batches_tracked" in name or "freqs_cis" in name]
    if not matches:
        print("[state] no matching keys")
        return
    for name, value in matches:
        print(
            f"[state] key={name} type={type(value).__name__} shape={tuple(value.shape)} "
            f"dtype={value.dtype} device={value.device} has_device_mesh={hasattr(value, 'device_mesh')}"
        )


def print_model_chain(model: torch.nn.Module) -> None:
    print("========== MODEL CHAIN ==========")
    visited: set[int] = set()
    current: object | None = model
    depth = 0
    while isinstance(current, torch.nn.Module) and id(current) not in visited:
        visited.add(id(current))
        print(f"[chain] depth={depth} type={type(current)}")
        next_model = getattr(current, "model", None)
        if not isinstance(next_model, torch.nn.Module):
            next_model = getattr(current, "base_model", None)
        current = next_model
        depth += 1


class InspectSwiftSft:
    def __init__(self, argv: list[str]):
        from swift.pipelines.train.sft import SwiftSft

        class _InnerInspectSwiftSft(SwiftSft):
            def train(self, trainer):
                model = trainer.model
                print("========== INSPECT CONTEXT ==========")
                print(f"[inspect] python={sys.version.split()[0]}")
                print(f"[inspect] platform={platform.platform()}")
                print(f"[inspect] rank={os.environ.get('RANK', 'unset')}")
                print(f"[inspect] model_type={type(model)}")
                print_model_chain(model)
                print_buffer_inventory(model, "FINAL TRAINER MODEL")
                print_state_matches(model, "FINAL TRAINER MODEL")

                base = getattr(model, "base_model", None)
                if isinstance(base, torch.nn.Module):
                    print_buffer_inventory(base, "PEFT BASE MODEL")
                    print_state_matches(base, "PEFT BASE MODEL")
                    inner = getattr(base, "model", None)
                    if isinstance(inner, torch.nn.Module):
                        print_buffer_inventory(inner, "PEFT INNER MODEL")
                        print_state_matches(inner, "PEFT INNER MODEL")

                named_state = model.state_dict()
                print(f"[state] total_keys={len(named_state)}")
                print(
                    "[state] target_key_present="
                    f"{'base_model.model.audio_encoder.model.semantic_encoder.encoder.init_bn.num_batches_tracked' in named_state}"
                )
                print("========== INSPECT DONE ==========")
                return {"status": "inspected"}

        self.pipeline = _InnerInspectSwiftSft(argv)

    def main(self):
        return self.pipeline.main()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    plugin = repo_root / "code" / "huginn_lora" / "plugins" / "huginn_losatok_swift.py"
    os.environ.setdefault("HUGINN_AUDIO_FSDP2_NONPERSISTENT_ROPE", "1")
    os.environ.setdefault("HUGINN_LOSATOK_TRAIN_CHAIN_AUDIT", "1")
    print("========== HUGINN LOSATOK FSDP2 STATE INSPECT ==========")
    print(f"[inspect] plugin={plugin}")
    print(f"[inspect] plugin_sha256={hashlib.sha256(plugin.read_bytes()).hexdigest()}")
    print(f"[inspect] HUGINN_AUDIO_FSDP2_NONPERSISTENT_ROPE={os.environ.get('HUGINN_AUDIO_FSDP2_NONPERSISTENT_ROPE')}")
    print("[inspect] fsdp=disabled; trainer.train=overridden; checkpoint_save=disabled")
    InspectSwiftSft(build_argv(repo_root)).main()


if __name__ == "__main__":
    main()
