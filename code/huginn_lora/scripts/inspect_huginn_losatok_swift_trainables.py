"""Construct the LoSATok Swift LoRA route and inspect its final trainable split."""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path


def load_shared_inspect_module(repo_root: Path):
    source_path = repo_root / "code" / "huginn_lora" / "scripts" / "inspect_huginn_audio_swift_trainables.py"
    spec = importlib.util.spec_from_file_location("huginn_audio_swift_trainable_inspect", source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load shared Swift inspect helpers: {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_inspect_manifest(repo_root: Path) -> Path:
    source_manifest = repo_root / "data" / "audio_swift" / "audiocaps_v2" / "audiocaps_v2_train_swift.jsonl"
    runtime_dir = Path(tempfile.mkdtemp(prefix="huginn_losatok_swift_inspect_"))
    inspect_manifest = runtime_dir / "audiocaps_v2_one_record.jsonl"
    if not source_manifest.is_file() or source_manifest.stat().st_size == 0:
        raise FileNotFoundError(f"AudioCaps manifest is missing or empty: {source_manifest}")
    first_record = None
    with source_manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                first_record = line.rstrip("\n")
                break
    if first_record is None:
        raise ValueError(f"AudioCaps manifest has no records: {source_manifest}")
    temporary = inspect_manifest.with_suffix(".jsonl.tmp")
    temporary.write_text(first_record + "\n", encoding="utf-8")
    temporary.replace(inspect_manifest)
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
        "--output_dir", str(repo_root / "outputs" / "huginn_losatok_swift_inspect"),
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
        "--save_only_model", "true",
        "--report_to", "none",
        "--bf16", "true",
    ]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    shared = load_shared_inspect_module(repo_root)
    argv = build_argv(repo_root)
    print("========== HUGINN LOSATOK SWIFT TRAINABLE INSPECT ==========")
    print("[inspect] mode=lora_llm frozen_losatok aligner_trainable huginn_lora_trainable")
    print("[inspect] argv=" + " ".join(argv))
    shared.InspectSwiftSft(argv).main()


if __name__ == "__main__":
    main()
