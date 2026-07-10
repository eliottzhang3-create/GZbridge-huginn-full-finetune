from __future__ import annotations

import platform
import sys
from collections import defaultdict
from pathlib import Path

import torch


def classify_param(name: str) -> str:
    if "audio_encoder" in name:
        return "audio_encoder"
    if any(key in name for key in ("temporal_compressor", "audio_projector", "audio_bos", "audio_eos")):
        return "aligner"
    if "lora_" in name:
        return "llm_lora"
    if ".modules_to_save." in name:
        return "modules_to_save"
    if ".lm_head." in name or name.endswith("lm_head.weight") or name.endswith("lm_head.bias"):
        return "lm_head"
    if ".wte." in name or ".embed_tokens." in name or name.endswith("embedding.weight"):
        return "embedding"
    if "transformer" in name:
        return "llm_base_other"
    return "other"


def format_millions(num_params: int) -> str:
    return f"{num_params / 1_000_000:.4f}M"


def print_model_chain(model):
    print("========== MODEL CHAIN ==========")
    visited = set()
    current = model
    depth = 0
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        print(f"[model-chain] depth={depth} type={type(current)}")
        next_model = None
        if hasattr(current, "model"):
            next_model = getattr(current, "model")
        elif hasattr(current, "base_model"):
            next_model = getattr(current, "base_model")
        current = next_model if isinstance(next_model, torch.nn.Module) else None
        depth += 1


def print_grad_checkpointing_state(model):
    print("========== GRADIENT CHECKPOINTING ==========")
    for attr_name in ("is_gradient_checkpointing", "gradient_checkpointing", "supports_gradient_checkpointing"):
        if hasattr(model, attr_name):
            print(f"[gc] outer.{attr_name}={getattr(model, attr_name)}")
    base = getattr(model, "base_model", None)
    if hasattr(base, "model"):
        base = base.model
    if isinstance(base, torch.nn.Module):
        for attr_name in ("is_gradient_checkpointing", "gradient_checkpointing", "supports_gradient_checkpointing"):
            if hasattr(base, attr_name):
                print(f"[gc] base.{attr_name}={getattr(base, attr_name)}")
        if hasattr(base, "audio_encoder"):
            audio_encoder = getattr(base, "audio_encoder")
            for attr_name in ("gradient_checkpointing", "supports_gradient_checkpointing"):
                if hasattr(audio_encoder, attr_name):
                    print(f"[gc] audio_encoder.{attr_name}={getattr(audio_encoder, attr_name)}")


def summarize_parameters(model):
    total_params = 0
    trainable_params = 0
    trainable_by_group = defaultdict(int)
    trainable_entries: list[tuple[str, int, str, str]] = []
    frozen_examples: list[str] = []

    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
            group = classify_param(name)
            trainable_by_group[group] += param_count
            trainable_entries.append((name, param_count, str(param.dtype), str(param.device)))
        elif len(frozen_examples) < 20:
            frozen_examples.append(name)

    print("========== PARAMETER SUMMARY ==========")
    print(f"[params] total={total_params} ({format_millions(total_params)})")
    print(f"[params] trainable={trainable_params} ({format_millions(trainable_params)})")
    print(f"[params] frozen={total_params - trainable_params} ({format_millions(total_params - trainable_params)})")
    trainable_ratio = (trainable_params / total_params * 100.0) if total_params else 0.0
    print(f"[params] trainable_ratio={trainable_ratio:.4f}%")

    print("========== TRAINABLE GROUPS ==========")
    for group, count in sorted(trainable_by_group.items(), key=lambda item: item[1], reverse=True):
        print(f"[group] {group}: {count} ({format_millions(count)})")

    print("========== TOP TRAINABLE PARAMETERS ==========")
    for name, count, dtype, device in sorted(trainable_entries, key=lambda item: item[1], reverse=True)[:80]:
        print(f"[trainable] {name}: {count} ({format_millions(count)}) dtype={dtype} device={device}")

    print("========== FIRST FROZEN PARAMETERS ==========")
    for name in frozen_examples:
        print(f"[frozen] {name}")


def print_cuda_memory():
    print("========== CUDA MEMORY ==========")
    if not torch.cuda.is_available():
        print("[cuda] unavailable")
        return
    device = torch.cuda.current_device()
    print(f"[cuda] device={device} name={torch.cuda.get_device_name(device)}")
    print(f"[cuda] allocated_gb={torch.cuda.memory_allocated(device) / float(1024 ** 3):.4f}")
    print(f"[cuda] reserved_gb={torch.cuda.memory_reserved(device) / float(1024 ** 3):.4f}")
    print(f"[cuda] max_allocated_gb={torch.cuda.max_memory_allocated(device) / float(1024 ** 3):.4f}")
    print(f"[cuda] max_reserved_gb={torch.cuda.max_memory_reserved(device) / float(1024 ** 3):.4f}")


class InspectSwiftSft:
    def __init__(self, argv: list[str]):
        from swift.pipelines.train.sft import SwiftSft

        class _InnerInspectSwiftSft(SwiftSft):
            def train(self, trainer):
                print("========== INSPECT CONTEXT ==========")
                print(f"[inspect] python={sys.version.split()[0]}")
                print(f"[inspect] platform={platform.platform()}")
                print(f"[inspect] trainer_type={type(trainer)}")
                print(f"[inspect] model_type={type(trainer.model)}")
                print(f"[inspect] output_dir={self.args.output_dir}")
                print_model_chain(trainer.model)
                print_grad_checkpointing_state(trainer.model)
                summarize_parameters(trainer.model)
                print_cuda_memory()
                print("========== INSPECT DONE ==========")
                return {"status": "inspected"}

        self.pipeline = _InnerInspectSwiftSft(argv)

    def main(self):
        return self.pipeline.main()


def build_smoke_like_argv(repo_root: Path) -> list[str]:
    model_path = repo_root / "models" / "huginn-audio-whisper-v1"
    plugin_path = repo_root / "code" / "huginn_lora" / "plugins" / "huginn_audio_swift.py"
    swift_manifest = repo_root / "data" / "audio_swift" / "clotho_aqa_tiny_train32_swift.jsonl"
    output_dir = repo_root / "outputs" / "huginn_audio_swift_inspect"

    return [
        "--model",
        str(model_path),
        "--model_type",
        "huginn_audio_raven",
        "--template",
        "huginn_audio_text",
        "--external_plugins",
        str(plugin_path),
        "--dataset",
        str(swift_manifest),
        "--max_length",
        "192",
        "--output_dir",
        str(output_dir),
        "--tuner_type",
        "lora_llm",
        "--freeze_vit",
        "true",
        "--freeze_aligner",
        "false",
        "--learning_rate",
        "1e-4",
        "--aligner_lr",
        "1e-4",
        "--lora_rank",
        "16",
        "--lora_alpha",
        "32",
        "--lora_dropout",
        "0.05",
        "--max_steps",
        "4",
        "--per_device_train_batch_size",
        "2",
        "--gradient_accumulation_steps",
        "1",
        "--logging_steps",
        "1",
        "--save_steps",
        "4",
        "--save_total_limit",
        "2",
        "--dataloader_num_workers",
        "0",
        "--dataloader_pin_memory",
        "false",
        "--dataset_num_proc",
        "1",
        "--save_only_model",
        "true",
        "--report_to",
        "none",
        "--bf16",
        "true",
    ]


def main():
    repo_root = Path(__file__).resolve().parents[3]
    argv = build_smoke_like_argv(repo_root)
    print("========== SWIFT ARGV ==========")
    print(" ".join(argv))
    InspectSwiftSft(argv).main()


if __name__ == "__main__":
    main()
