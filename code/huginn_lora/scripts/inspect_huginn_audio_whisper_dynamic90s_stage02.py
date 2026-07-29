#!/usr/bin/env python3
"""Validate the isolated Whisper dynamic-90s route through a real Swift batch.

This is a data-independent Stage 0-2 gate. It creates deterministic WAV files,
filters over-limit records before manifest construction, loads the real remote
Whisper-large and Huginn-0125 assets through the dynamic Swift plugin, audits
the real collator/prefix path for multiple durations, and performs one real
LoRA/aligner backward pass on a single GPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch


SAMPLE_RATE = 16_000
CONTRACT_CASES = (
    (0.10, 0, 1, False),
    (1.00, 8, 1, False),
    (15.00, 125, 1, False),
    (29.99, 249, 1, False),
    (30.00, 250, 1, False),
    (30.01, 250, 2, False),
    (45.00, 375, 2, False),
    (60.00, 500, 2, False),
    (75.00, 625, 3, False),
    (90.00, 750, 3, False),
    (91.00, 750, 3, False),
    (119.00, 750, 3, False),
    (120.00, 750, 3, False),
    (120.01, 0, 0, True),
)
TRAIN_DURATIONS = tuple(duration for duration, _, _, discarded in CONTRACT_CASES if not discarded)
EXPECTED_LORA_TENSOR_COUNT = 66
EXPECTED_ALIGNER_TENSOR_COUNT = 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[3]
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=repo_root / "outputs" / "huginn_audio_whisper_dynamic90s_stage02",
    )
    return parser.parse_args()


def import_plugin(plugin_path: Path):
    spec = importlib.util.spec_from_file_location(
        "huginn_audio_whisper_dynamic90s_stage02_plugin",
        plugin_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import dynamic plugin: {plugin_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_synthetic_wav(path: Path, duration_seconds: float) -> int:
    sample_count = int(round(duration_seconds * SAMPLE_RATE))
    sample_index = np.arange(sample_count, dtype=np.float64)
    waveform = 0.08 * np.sin(2.0 * math.pi * 220.0 * sample_index / SAMPLE_RATE)
    pcm = np.clip(np.rint(waveform * 32767.0), -32768, 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm.tobytes())
    return sample_count


def assert_contract(plugin: Any) -> dict[float, Any]:
    plans: dict[float, Any] = {}
    print("========== DYNAMIC LENGTH CONTRACT ==========")
    for duration, expected_tokens, expected_segments, expected_discarded in CONTRACT_CASES:
        sample_count = int(round(duration * SAMPLE_RATE))
        plan = plugin.plan_audio_for_whisper(sample_count, SAMPLE_RATE)
        plans[duration] = plan
        if bool(plan.discarded) != expected_discarded:
            raise AssertionError(
                f"discard mismatch for {duration}s: expected={expected_discarded} actual={plan.discarded}"
            )
        if plan.total_audio_tokens != expected_tokens:
            raise AssertionError(
                f"token mismatch for {duration}s: expected={expected_tokens} actual={plan.total_audio_tokens}"
            )
        if plan.segment_count != expected_segments:
            raise AssertionError(
                f"segment mismatch for {duration}s: expected={expected_segments} actual={plan.segment_count}"
            )
        if not plan.discarded and duration < 30.0 and plan.total_audio_tokens >= 250:
            raise AssertionError(f"Sub-30s audio was incorrectly expanded to 250 tokens: {duration}s")
        if 90.0 < duration <= 120.0 and plan.included_samples != 90 * SAMPLE_RATE:
            raise AssertionError(f"{duration}s must be truncated to exactly 90s")
        print(
            f"[contract] duration={duration:.2f}s discarded={plan.discarded} "
            f"segments={plan.segment_count} feature_lengths={plan.feature_lengths} "
            f"token_counts={plan.token_counts} total_audio_tokens={plan.total_audio_tokens}"
        )
    print("[contract] status=PASS dynamic_tokens=true token_duration_ms=120")
    return plans


def build_fixture(plugin: Any, work_dir: Path, plans: dict[float, Any]) -> Path:
    fixture_dir = work_dir / "fixture"
    manifest_path = fixture_dir / "dynamic90s_stage02.jsonl"
    discarded_records: list[dict[str, Any]] = []
    accepted_records: list[dict[str, Any]] = []
    fixture_dir.mkdir(parents=True, exist_ok=True)

    for duration, _, _, _ in CONTRACT_CASES:
        plan = plans[duration]
        wav_path = fixture_dir / f"duration_{duration:06.2f}s.wav"
        sample_count = write_synthetic_wav(wav_path, duration)
        if sample_count != plan.total_samples:
            raise AssertionError(f"Synthetic WAV sample-count mismatch for {duration}s")
        record = {
            "messages": [
                {
                    "role": "system",
                    "content": plugin.DEFAULT_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": "Listen to the synthetic audio and state that it is a validation tone.",
                },
                {
                    "role": "assistant",
                    "content": "This is a synthetic validation tone.",
                },
            ],
            "audios": [str(wav_path.resolve())],
            "metadata": {
                "dataset": "synthetic_dynamic90s_stage02",
                "duration_seconds": duration,
                "expected_audio_tokens": plan.total_audio_tokens,
                "expected_segments": plan.segment_count,
            },
        }
        if plan.discarded:
            discarded_records.append(record)
        else:
            accepted_records.append(record)

    if len(discarded_records) != 1 or discarded_records[0]["metadata"]["duration_seconds"] != 120.01:
        raise AssertionError(f"Unexpected discarded fixture records: {discarded_records}")
    if len(accepted_records) != len(TRAIN_DURATIONS):
        raise AssertionError(
            f"Accepted fixture count mismatch: expected={len(TRAIN_DURATIONS)} actual={len(accepted_records)}"
        )

    with manifest_path.open("w", encoding="utf-8") as output_file:
        for record in accepted_records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    (fixture_dir / "discarded_over_120s.json").write_text(
        json.dumps(discarded_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("========== SYNTHETIC FIXTURE ==========")
    print(f"[fixture] manifest={manifest_path}")
    print(f"[fixture] accepted={len(accepted_records)} discarded={len(discarded_records)}")
    print("[fixture] over_120s_filtered_before_swift=true")
    return manifest_path


def find_audio_model(model: torch.nn.Module) -> torch.nn.Module:
    candidates = [
        module
        for module in model.modules()
        if callable(getattr(type(module), "build_audio_prefix", None))
        and hasattr(module, "audio_encoder")
        and hasattr(module, "temporal_compressor")
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one dynamic audio base model, found {len(candidates)}")
    return candidates[0]


def compressed_tokens_from_feature_length(plugin: Any, feature_length: int) -> int:
    encoder_length = feature_length // plugin.WHISPER_ENCODER_DOWNSAMPLE
    if encoder_length < plugin.DYNAMIC_COMPRESSOR_KERNEL:
        return 0
    return (
        encoder_length - plugin.DYNAMIC_COMPRESSOR_KERNEL
    ) // plugin.DYNAMIC_COMPRESSOR_STRIDE + 1


def expected_prefix_lengths(plugin: Any, batch: dict[str, torch.Tensor]) -> list[int]:
    lengths = batch["audio_segment_feature_lengths"]
    segment_mask = batch["audio_segment_mask"].bool()
    expected: list[int] = []
    for sample_index in range(lengths.size(0)):
        audio_tokens = 0
        for segment_index in range(lengths.size(1)):
            if not bool(segment_mask[sample_index, segment_index].item()):
                continue
            audio_tokens += compressed_tokens_from_feature_length(
                plugin,
                int(lengths[sample_index, segment_index].item()),
            )
        expected.append(audio_tokens + 2)
    return expected


def audit_lora_configuration(model: torch.nn.Module) -> None:
    peft_configs = getattr(model, "peft_config", None)
    if not peft_configs:
        raise AssertionError("Final Swift model exposes no PEFT adapter configuration")
    for adapter_name, config in peft_configs.items():
        actual = {
            "rank": int(config.r),
            "alpha": float(config.lora_alpha),
            "dropout": float(config.lora_dropout),
        }
        expected = {"rank": 8, "alpha": 16.0, "dropout": 0.05}
        if actual != expected:
            raise AssertionError(
                f"PEFT config mismatch for adapter {adapter_name!r}: expected={expected} actual={actual}"
            )

    lora_parameters = [(name, parameter) for name, parameter in model.named_parameters() if "lora_" in name]
    if len(lora_parameters) != EXPECTED_LORA_TENSOR_COUNT:
        raise AssertionError(
            f"LoRA tensor count mismatch: expected={EXPECTED_LORA_TENSOR_COUNT} actual={len(lora_parameters)}"
        )
    for name, parameter in lora_parameters:
        if "lora_A" in name and parameter.ndim == 2 and parameter.shape[0] != 8:
            raise AssertionError(f"LoRA A rank is not 8: {name} shape={tuple(parameter.shape)}")
        if "lora_B" in name and parameter.ndim == 2 and parameter.shape[1] != 8:
            raise AssertionError(f"LoRA B rank is not 8: {name} shape={tuple(parameter.shape)}")

    target_modules = 0
    for module in model.modules():
        lora_a = getattr(module, "lora_A", None)
        if not lora_a:
            continue
        target_modules += 1
        alpha_values = getattr(module, "lora_alpha", {})
        if not alpha_values or any(float(value) != 16.0 for value in alpha_values.values()):
            raise AssertionError(f"Effective LoRA alpha is not 16: {alpha_values}")
        dropout_values = getattr(module, "lora_dropout", {})
        if not dropout_values:
            raise AssertionError("LoRA module exposes no effective dropout modules")
        for dropout in dropout_values.values():
            probability = float(getattr(dropout, "p", 0.0))
            if abs(probability - 0.05) > 1e-12:
                raise AssertionError(f"Effective LoRA dropout is not 0.05: {probability}")
    if target_modules * 2 != EXPECTED_LORA_TENSOR_COUNT:
        raise AssertionError(
            f"LoRA target module mismatch: modules={target_modules} tensors={EXPECTED_LORA_TENSOR_COUNT}"
        )
    print(
        f"[lora] tensors={len(lora_parameters)} target_modules={target_modules} "
        "rank=8 alpha=16 dropout=0.05"
    )


def audit_trainable_split(model: torch.nn.Module, audio_model: torch.nn.Module) -> None:
    aligner_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if any(
            key in name
            for key in ("temporal_compressor", "audio_projector", "audio_boundary_embeddings")
        )
    ]
    if len(aligner_parameters) != EXPECTED_ALIGNER_TENSOR_COUNT:
        raise AssertionError(
            f"Aligner tensor count mismatch: expected={EXPECTED_ALIGNER_TENSOR_COUNT} "
            f"actual={len(aligner_parameters)} names={[name for name, _ in aligner_parameters]}"
        )
    if any(not parameter.requires_grad for _, parameter in aligner_parameters):
        raise AssertionError("Every dynamic aligner tensor must be trainable")
    if any(parameter.requires_grad for parameter in audio_model.audio_encoder.parameters()):
        raise AssertionError("Whisper-large must remain frozen")
    if audio_model.audio_encoder.training:
        raise AssertionError("Frozen Whisper-large must remain in eval mode")

    unexpected_trainables = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_" in name or any(
            key in name
            for key in ("temporal_compressor", "audio_projector", "audio_boundary_embeddings")
        ):
            continue
        unexpected_trainables.append(name)
    if unexpected_trainables:
        raise AssertionError(f"Unexpected trainable base parameters: {unexpected_trainables[:20]}")
    print(
        f"[trainables] aligner_tensors={len(aligner_parameters)} "
        "whisper_trainable=0 huginn_base_trainable=0"
    )


def audit_real_prefix_batches(trainer: Any, plugin: Any, audio_model: torch.nn.Module) -> None:
    audio_model.eval()
    saw_dynamic_sub_250 = False
    saw_padding = False
    batch_count = 0
    for raw_batch in trainer.get_train_dataloader():
        batch = trainer._prepare_inputs(raw_batch)
        required = {
            "audio_input_features",
            "audio_segment_feature_lengths",
            "audio_segment_mask",
        }
        missing = required - batch.keys()
        if missing:
            raise AssertionError(f"Swift collator batch is missing dynamic audio fields: {sorted(missing)}")
        if batch["audio_input_features"].ndim != 4:
            raise AssertionError(
                f"Expected [B, segments, 80, frames], got {tuple(batch['audio_input_features'].shape)}"
            )
        with torch.no_grad():
            prefix, prefix_mask = audio_model.build_audio_prefix(
                batch["audio_input_features"],
                audio_segment_feature_lengths=batch["audio_segment_feature_lengths"],
                audio_segment_mask=batch["audio_segment_mask"],
            )
        expected_lengths = expected_prefix_lengths(plugin, batch)
        actual_lengths = [int(value) for value in prefix_mask.sum(dim=1).tolist()]
        if actual_lengths != expected_lengths:
            raise AssertionError(
                f"Real prefix mismatch: expected={expected_lengths} actual={actual_lengths}"
            )
        for prefix_length in actual_lengths:
            audio_token_count = prefix_length - 2
            if 0 <= audio_token_count < 250:
                saw_dynamic_sub_250 = True
        if any(length < prefix.size(1) for length in actual_lengths):
            saw_padding = True
            padded_values = prefix.masked_select(~prefix_mask.unsqueeze(-1))
            if padded_values.numel() and not bool(padded_values.eq(0).all().item()):
                raise AssertionError("Padded audio prefix embeddings must be exactly zero")
        print(
            f"[prefix-batch] index={batch_count} features={tuple(batch['audio_input_features'].shape)} "
            f"expected_prefix_lengths={expected_lengths} actual_prefix_lengths={actual_lengths}"
        )
        batch_count += 1
        del prefix, prefix_mask, batch
        torch.cuda.empty_cache()
    if not saw_dynamic_sub_250:
        raise AssertionError("No real sub-30s sample produced a dynamic token count below 250")
    if not saw_padding:
        raise AssertionError("No real mixed-length batch exercised prefix padding")
    print(f"[prefix-batch] batches={batch_count} dynamic_sub_250=true padding_exercised=true")


def audit_backward(trainer: Any, audio_model: torch.nn.Module) -> None:
    model = trainer.model
    model.train()
    if audio_model.audio_encoder.training:
        raise AssertionError("Whisper encoder re-entered train mode after model.train()")
    raw_batch = next(iter(trainer.get_train_dataloader()))
    batch = trainer._prepare_inputs(raw_batch)
    model.zero_grad(set_to_none=True)
    with trainer.compute_loss_context_manager():
        loss = trainer.compute_loss(model, batch)
    if not torch.is_tensor(loss) or not bool(torch.isfinite(loss).item()):
        raise AssertionError(f"Non-finite real Swift loss: {loss}")
    trainer.accelerator.backward(loss)

    grad_groups = {"lora": 0, "aligner": 0, "whisper": 0, "base": 0, "other": 0}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all().item()):
            raise AssertionError(f"Non-finite gradient: {name}")
        if "audio_encoder" in name:
            grad_groups["whisper"] += parameter.numel()
        elif "lora_" in name:
            grad_groups["lora"] += parameter.numel()
        elif any(
            key in name
            for key in ("temporal_compressor", "audio_projector", "audio_boundary_embeddings")
        ):
            grad_groups["aligner"] += parameter.numel()
        elif "transformer" in name or "lm_head" in name:
            grad_groups["base"] += parameter.numel()
        else:
            grad_groups["other"] += parameter.numel()
    if grad_groups["lora"] <= 0 or grad_groups["aligner"] <= 0:
        raise AssertionError(f"Missing LoRA/aligner gradients: {grad_groups}")
    if grad_groups["whisper"] or grad_groups["base"] or grad_groups["other"]:
        raise AssertionError(f"Frozen/unexpected parameters received gradients: {grad_groups}")

    prefix_mask = getattr(audio_model, "_last_audio_prefix_mask", None)
    combined_attention_mask = getattr(audio_model, "_last_audio_combined_attention_mask", None)
    full_labels = getattr(audio_model, "_last_dynamic90s_full_labels", None)
    if prefix_mask is None or combined_attention_mask is None or full_labels is None:
        raise AssertionError("Dynamic audit tensors were not captured by the real forward path")
    prefix_length = prefix_mask.size(1)
    if not torch.equal(combined_attention_mask[:, :prefix_length].bool(), prefix_mask.bool()):
        raise AssertionError("Combined attention mask does not preserve the real prefix mask")
    if not bool(full_labels[:, :prefix_length].eq(-100).all().item()):
        raise AssertionError("Audio prefix and padding labels must all be -100")
    has_audio_padding = prefix_mask.sum(dim=1).lt(prefix_length)
    if bool(has_audio_padding.any().item()) and not bool(
        full_labels[has_audio_padding, prefix_length].eq(-100).all().item()
    ):
        raise AssertionError("First text target after padded audio must be -100")
    print(
        f"[backward] loss={float(loss.detach()):.6f} grad_groups={grad_groups} "
        f"prefix_lengths={prefix_mask.sum(dim=1).tolist()} masks=PASS"
    )


def build_swift_argv(repo_root: Path, plugin: Any, manifest_path: Path, output_dir: Path) -> list[str]:
    return [
        "--model", str(plugin.AUDIO_MODEL_DIR),
        "--model_type", plugin.MODEL_TYPE,
        "--template", plugin.TEMPLATE_TYPE,
        "--external_plugins", str(Path(plugin.__file__).resolve()),
        "--dataset", str(manifest_path),
        "--dataset_shuffle", "false",
        "--train_dataloader_shuffle", "false",
        "--sortish_sampler", "false",
        "--group_by_length", "false",
        "--max_length", "192",
        "--output_dir", str(output_dir),
        "--tuner_type", "lora_llm",
        "--freeze_vit", "true",
        "--freeze_aligner", "false",
        "--learning_rate", "1e-4",
        "--aligner_lr", "1e-4",
        "--lora_rank", "8",
        "--lora_alpha", "16",
        "--lora_dropout", "0.05",
        "--max_steps", "1",
        "--per_device_train_batch_size", "2",
        "--gradient_accumulation_steps", "1",
        "--gradient_checkpointing", "false",
        "--logging_steps", "1",
        "--save_strategy", "no",
        "--dataloader_num_workers", "0",
        "--dataloader_pin_memory", "false",
        "--dataset_num_proc", "1",
        "--report_to", "none",
        "--bf16", "true",
    ]


def run_real_swift_gate(repo_root: Path, plugin: Any, manifest_path: Path, work_dir: Path) -> None:
    from swift.pipelines.train.sft import SwiftSft

    class Stage02SwiftSft(SwiftSft):
        def train(self, trainer):
            print("========== REAL SWIFT STAGE 0-2 GATE ==========")
            audio_model = find_audio_model(trainer.model)
            audit_lora_configuration(trainer.model)
            audit_trainable_split(trainer.model, audio_model)
            audit_real_prefix_batches(trainer, plugin, audio_model)
            audit_backward(trainer, audio_model)
            print("========== REAL SWIFT STAGE 0-2 PASSED ==========")
            return {"status": "passed"}

    argv = build_swift_argv(repo_root, plugin, manifest_path, work_dir / "swift_output")
    print("[swift-argv] " + " ".join(argv))
    Stage02SwiftSft(argv).main()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    plugin_path = repo_root / "code" / "huginn_lora" / "plugins" / "huginn_audio_whisper_dynamic90s_swift.py"
    if not torch.cuda.is_available():
        raise RuntimeError("The real Stage 0-2 gate requires one CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Stage 0-2 expects exactly one visible GPU, got {torch.cuda.device_count()}")
    print("========== ENVIRONMENT ==========")
    print(f"[env] python={sys.version.split()[0]} platform={platform.platform()}")
    print(f"[env] repo_root={repo_root}")
    print(f"[env] cuda={torch.cuda.get_device_name(0)}")
    print(f"[env] work_dir={args.work_dir}")
    plugin = import_plugin(plugin_path)
    plans = assert_contract(plugin)
    manifest_path = build_fixture(plugin, args.work_dir, plans)
    run_real_swift_gate(repo_root, plugin, manifest_path, args.work_dir)
    print("========== HUGINN WHISPER DYNAMIC90S STAGE 0-2 VALIDATION PASSED ==========")


if __name__ == "__main__":
    main()
