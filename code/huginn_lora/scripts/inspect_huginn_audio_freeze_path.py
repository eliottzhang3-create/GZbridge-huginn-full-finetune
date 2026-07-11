from __future__ import annotations

import importlib.util
import platform
import sys
from collections import defaultdict
from pathlib import Path

import torch


def summarize(model: torch.nn.Module, stage: str):
    total = 0
    trainable = 0
    group_counts = defaultdict(int)
    audio_trainable = []

    for name, param in model.named_parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
            if "audio_encoder" in name:
                group_counts["audio_encoder"] += count
                if len(audio_trainable) < 40:
                    audio_trainable.append((name, count))
            elif any(key in name for key in ("temporal_compressor", "audio_projector", "audio_bos", "audio_eos")):
                group_counts["aligner"] += count
            elif "lora_" in name:
                group_counts["llm_lora"] += count
            elif "transformer" in name:
                group_counts["llm_base_other"] += count
            else:
                group_counts["other"] += count

    print(f"========== {stage} ==========")
    print(f"[stage] model_type={type(model)}")
    print(f"[stage] total={total}")
    print(f"[stage] trainable={trainable}")
    for key, value in sorted(group_counts.items(), key=lambda item: item[1], reverse=True):
        print(f"[stage] group[{key}]={value}")
    if audio_trainable:
        print("[stage] first_audio_encoder_trainables:")
        for name, count in audio_trainable:
            print(f"  - {name}: {count}")
    else:
        print("[stage] audio_encoder_trainables=0")


def load_plugin_module(repo_root: Path):
    plugin_path = repo_root / "code" / "huginn_lora" / "plugins" / "huginn_audio_swift.py"
    module_name = "huginn_audio_swift_inspect_plugin"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, plugin_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load plugin module from {plugin_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def inspect_direct_model(repo_root: Path):
    plugin = load_plugin_module(repo_root)

    model = plugin.build_huginn_audio_model(str(repo_root / "models" / "huginn-audio-whisper-v1"))
    summarize(model, "DIRECT_BUILD_MODEL")


def patch_peft_debug():
    try:
        import peft.mapping
    except ImportError:
        peft_mapping = None
    else:
        peft_mapping = peft.mapping

    try:
        import peft.mapping_func
    except ImportError:
        peft_mapping_func = None
    else:
        peft_mapping_func = peft.mapping_func

    def wrap_get_peft_model(module, module_name: str):
        if module is None or not hasattr(module, "get_peft_model"):
            return
        original = module.get_peft_model
        if getattr(original, "_huginn_audio_freeze_debug_wrapped", False):
            return

        def wrapped_get_peft_model(*args, **kwargs):
            model = args[0] if args else kwargs.get("model")
            if isinstance(model, torch.nn.Module):
                summarize(model, f"BEFORE_PEFT_WRAP[{module_name}]")
            wrapped_model = original(*args, **kwargs)
            if isinstance(wrapped_model, torch.nn.Module):
                summarize(wrapped_model, f"AFTER_PEFT_WRAP[{module_name}]")
            return wrapped_model

        wrapped_get_peft_model._huginn_audio_freeze_debug_wrapped = True  # type: ignore[attr-defined]
        module.get_peft_model = wrapped_get_peft_model

    wrap_get_peft_model(peft_mapping, "peft.mapping")
    wrap_get_peft_model(peft_mapping_func, "peft.mapping_func")


def build_swift_argv(repo_root: Path) -> list[str]:
    return [
        "--model",
        str(repo_root / "models" / "huginn-audio-whisper-v1"),
        "--model_type",
        "huginn_audio_raven",
        "--template",
        "huginn_audio_text",
        "--external_plugins",
        str(repo_root / "code" / "huginn_lora" / "plugins" / "huginn_audio_swift.py"),
        "--dataset",
        str(repo_root / "data" / "audio_swift" / "clotho_aqa_tiny_train32_swift.jsonl"),
        "--max_length",
        "192",
        "--output_dir",
        str(repo_root / "outputs" / "huginn_audio_swift_freeze_inspect"),
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


def inspect_swift_final(repo_root: Path):
    from swift.pipelines.train.sft import SwiftSft

    patch_peft_debug()

    class _InspectSwiftSft(SwiftSft):
        def train(self, trainer):
            summarize(trainer.model, "SWIFT_FINAL_TRAINER_MODEL")
            return {"status": "inspected"}

    _InspectSwiftSft(build_swift_argv(repo_root)).main()


def main():
    repo_root = Path(__file__).resolve().parents[3]
    print("========== ENV ==========")
    print(f"python={sys.version.split()[0]}")
    print(f"platform={platform.platform()}")
    print(f"repo_root={repo_root}")

    inspect_direct_model(repo_root)
    inspect_swift_final(repo_root)


if __name__ == "__main__":
    main()
