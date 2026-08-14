#!/usr/bin/env python3
"""Strict BAT one-stage ms-swift LoRA + Q-Former training smoke.

This is an engineering smoke, not the full AudioSet-20K run.  It uses a small
private JSONL manifest but keeps the BAT Stage-I optimizer, scheduler, batch,
epoch and freeze contracts.  It fails closed if Swift makes Spatial-AST,
Ouro's native parameters, or the early-exit gate trainable.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from bat.configs.training import BAT_TRAINING


MODEL_TYPE = "ouro_bat_spatial_ast"
TEMPLATE_TYPE = "ouro_bat_audio_prefix"
EXPECTED_STEPS = 4
GATE_MODULE = "early_exit_gate"
TARGET_MODULES = BAT_TRAINING.lora_target_modules
EXPECTED_OURO_LAYER_COUNT = 24
EXPECTED_QFORMER_STATE_TENSORS = 175


def package_version(name: str) -> str:
    try:
        return version(name)
    except Exception as exc:
        return f"<unavailable:{type(exc).__name__}>"


def require_environment() -> None:
    expected = {"ms-swift": "4.4.2", "transformers": "4.54.1", "peft": "0.18.1"}
    mismatches = {
        key: (value, package_version(key))
        for key, value in expected.items()
        if package_version(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Unexpected BAT Swift environment: {mismatches}")
    if not torch.cuda.is_available():
        raise RuntimeError("BAT LoRA smoke is GPU-only and must run in a submitted job")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--plugin-path", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def find_module(model: torch.nn.Module, class_name: str) -> torch.nn.Module:
    matches = [module for module in model.modules() if module.__class__.__name__ == class_name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {class_name}, found {len(matches)}")
    return matches[0]


def normalized_name(name: str) -> str:
    for prefix in ("base_model.model.", "base_model."):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name.replace(".base_layer.", ".")


def parameter_group_name(name: str) -> str:
    """Classify a parameter after normalizing PEFT wrapper prefixes."""
    normalized = normalized_name(name)
    if "lora_A" in name or "lora_B" in name:
        return "lora"
    if normalized.startswith("audio_qformer."):
        return "qformer"
    if normalized.startswith("spatial_ast_encoder."):
        return "spatial_ast"
    if GATE_MODULE in name:
        return "gate"
    if normalized.startswith(("model.", "lm_head.")):
        return "ouro_native"
    return "other"


def shape_tuple(value: Any) -> tuple[int, ...]:
    """Convert torch.Size/list/tuple shape payloads to one canonical form."""
    shape = getattr(value, "shape", value)
    if shape is None:
        raise TypeError(f"Cannot read a tensor shape from {type(value).__name__}")
    try:
        return tuple(int(item) for item in shape)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Invalid shape payload: {value!r}") from exc


def parameter_report(model: torch.nn.Module) -> dict[str, Any]:
    groups = {"qformer": 0, "lora": 0, "spatial_ast": 0, "gate": 0, "ouro_native": 0, "other": 0}
    trainable_names: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        trainable_names.append(name)
        groups[parameter_group_name(name)] += parameter.numel()
    unexpected = {key: value for key, value in groups.items() if key in {"spatial_ast", "gate", "ouro_native", "other"} and value}
    if unexpected:
        raise RuntimeError(f"Unexpected trainable BAT groups: {unexpected}")
    if groups["qformer"] <= 0 or groups["lora"] <= 0:
        raise RuntimeError(f"Q-Former or LoRA is not trainable: {groups}")
    return {
        "groups": groups,
        "trainable_parameter_count": sum(groups.values()),
        "trainable_name_count": len(trainable_names),
        "trainable_name_preview": trainable_names[:80],
    }


def adapter_config_report(path: Path) -> dict[str, Any]:
    """Validate the PEFT metadata saved alongside an adapter checkpoint."""
    if not path.is_file():
        raise RuntimeError(f"Missing adapter config: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise RuntimeError(f"Adapter config is not an object: {path}")
    target_modules = config.get("target_modules")
    if isinstance(target_modules, (set, tuple)):
        target_modules = list(target_modules)
    if not isinstance(target_modules, list):
        raise RuntimeError(f"Adapter target_modules is not a list: {target_modules!r}")
    target_set = {str(item) for item in target_modules}
    expected_set = set(TARGET_MODULES)
    if target_set != expected_set:
        raise RuntimeError(f"Unexpected adapter target_modules: saved={sorted(target_set)} expected={sorted(expected_set)}")
    if int(config.get("r", -1)) != BAT_TRAINING.lora_rank:
        raise RuntimeError(f"Unexpected adapter rank: saved={config.get('r')!r} expected={BAT_TRAINING.lora_rank}")
    if int(config.get("lora_alpha", -1)) != BAT_TRAINING.lora_alpha:
        raise RuntimeError(
            f"Unexpected adapter alpha: saved={config.get('lora_alpha')!r} expected={BAT_TRAINING.lora_alpha}"
        )
    if not math.isclose(float(config.get("lora_dropout", -1.0)), BAT_TRAINING.lora_dropout, rel_tol=0.0, abs_tol=1e-8):
        raise RuntimeError(
            f"Unexpected adapter dropout: saved={config.get('lora_dropout')!r} expected={BAT_TRAINING.lora_dropout}"
        )
    modules_to_save = config.get("modules_to_save") or []
    modules_to_save = {str(item) for item in modules_to_save}
    if "audio_qformer" not in modules_to_save:
        raise RuntimeError(f"Q-Former is missing from modules_to_save: {sorted(modules_to_save)}")
    return {
        "path": str(path),
        "r": int(config["r"]),
        "lora_alpha": int(config["lora_alpha"]),
        "lora_dropout": float(config.get("lora_dropout", -1.0)),
        "target_modules": sorted(target_set),
        "modules_to_save": sorted(modules_to_save),
    }


def gradient_report(model: torch.nn.Module) -> dict[str, Any]:
    """Audit the first backward without assuming every LoRA factor is nonzero."""
    groups = {"qformer": 0, "lora": 0, "spatial_ast": 0, "gate": 0, "ouro_native": 0, "other": 0}
    nonzero_groups = {key: 0 for key in groups}
    nonfinite: list[str] = []
    frozen_with_gradient: list[str] = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        group = parameter_group_name(name)
        groups[group] += 1
        gradient = parameter.grad.detach()
        if not bool(torch.isfinite(gradient).all().item()):
            nonfinite.append(name)
        elif bool((gradient.float().abs() > 0).any().item()):
            nonzero_groups[group] += 1
        if not parameter.requires_grad:
            frozen_with_gradient.append(name)
    if groups["qformer"] <= 0 or groups["lora"] <= 0:
        raise RuntimeError(f"Expected Q-Former and LoRA gradients, got {groups}")
    if nonzero_groups["qformer"] <= 0 or nonzero_groups["lora"] <= 0:
        raise RuntimeError(f"Expected nonzero Q-Former and LoRA gradients, got {nonzero_groups}")
    if any(groups[key] for key in ("spatial_ast", "gate", "ouro_native", "other")):
        raise RuntimeError(f"Frozen BAT components unexpectedly received gradients: {groups}")
    if nonfinite:
        raise RuntimeError(f"Non-finite gradients: {nonfinite[:10]}")
    if frozen_with_gradient:
        raise RuntimeError(f"Frozen parameters unexpectedly received gradients: {frozen_with_gradient[:10]}")
    return {
        "finite_gradient_parameter_counts": groups,
        "nonzero_gradient_parameter_counts": nonzero_groups,
        "nonfinite_names": nonfinite,
        "frozen_with_gradient_names": frozen_with_gradient,
    }


def lora_report(model: torch.nn.Module) -> dict[str, Any]:
    modules: list[str] = []
    ranks: set[int] = set()
    invalid: list[str] = []
    for name, module in model.named_modules():
        if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            continue
        modules.append(name)
        if name.rsplit(".", 1)[-1] not in TARGET_MODULES:
            invalid.append(name)
        values = getattr(module, "r", {})
        ranks.update(int(value) for value in values.values())
    expected = EXPECTED_OURO_LAYER_COUNT * len(TARGET_MODULES)
    if len(modules) != expected:
        raise RuntimeError(f"Expected {expected} Ouro LoRA modules, found {len(modules)}")
    if invalid or ranks != {BAT_TRAINING.lora_rank}:
        raise RuntimeError(f"Invalid LoRA contract: invalid={invalid[:10]} ranks={sorted(ranks)}")
    if any(GATE_MODULE in name or "audio_qformer" in name or "spatial_ast" in name for name in modules):
        raise RuntimeError("LoRA was injected outside the Ouro language model")
    return {
        "module_count": len(modules),
        "expected_module_count": expected,
        "target_modules": list(TARGET_MODULES),
        "rank": BAT_TRAINING.lora_rank,
        "alpha": BAT_TRAINING.lora_alpha,
        "dropout": BAT_TRAINING.lora_dropout,
        "module_preview": sorted(modules)[:24],
    }


def optimizer_report(trainer: Any, model: torch.nn.Module) -> dict[str, Any]:
    optimizer = trainer.optimizer
    if optimizer is None:
        raise RuntimeError("Swift did not create an optimizer")
    optimizer_params = {
        id(parameter): parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    trainable_params = {id(parameter): parameter for parameter in model.parameters() if parameter.requires_grad}
    if optimizer_params.keys() != trainable_params.keys():
        raise RuntimeError("Optimizer parameter set does not equal trainable parameter set")
    betas = sorted({tuple(float(value) for value in group.get("betas", ())) for group in optimizer.param_groups})
    weight_decays = sorted({float(group.get("weight_decay", 0.0)) for group in optimizer.param_groups})
    if betas != [(BAT_TRAINING.beta1, BAT_TRAINING.beta2)]:
        raise RuntimeError(f"Unexpected AdamW betas: {betas}")
    if any(value not in {0.0, BAT_TRAINING.weight_decay} for value in weight_decays):
        raise RuntimeError(f"Unexpected weight decay groups: {weight_decays}")
    learning_rates = sorted({float(group["lr"]) for group in optimizer.param_groups})
    scheduler = trainer.lr_scheduler
    scheduler_name = None if scheduler is None else f"{scheduler.__class__.__module__}.{scheduler.__class__.__name__}"
    scheduler_type = str(getattr(getattr(trainer, "args", None), "lr_scheduler_type", "")).lower()
    if scheduler is None or "cosine" not in scheduler_type:
        raise RuntimeError(f"Expected lr_scheduler_type=cosine, got type={scheduler_type} class={scheduler_name}")
    return {
        "class": f"{optimizer.__class__.__module__}.{optimizer.__class__.__name__}",
        "betas": betas,
        "weight_decay_groups": weight_decays,
        "learning_rates": learning_rates,
        "parameter_count": sum(parameter.numel() for parameter in optimizer_params.values()),
        "scheduler_class": scheduler_name,
        "scheduler_type": scheduler_type,
        "scheduler_last_epoch": int(scheduler.last_epoch),
    }


def checkpoint_path(output_dir: Path) -> Path:
    candidates = sorted(path for path in output_dir.rglob("checkpoint-2") if path.is_dir())
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one checkpoint-2 below {output_dir}, found {candidates}")
    return candidates[0]


def checkpoint_report(path: Path) -> dict[str, Any]:
    from safetensors import safe_open

    adapter = path / "adapter_model.safetensors"
    if not adapter.is_file():
        raise RuntimeError(f"Missing adapter checkpoint: {adapter}")
    with safe_open(str(adapter), framework="pt", device="cpu") as handle:
        keys = sorted(handle.keys())
    lora_keys = [key for key in keys if "lora_" in key]
    qformer_keys = [key for key in keys if "audio_qformer" in key]
    unexpected = [key for key in keys if key not in lora_keys and key not in qformer_keys]
    if not lora_keys or not qformer_keys or unexpected:
        raise RuntimeError(f"Unexpected BAT checkpoint keys: lora={len(lora_keys)} qformer={len(qformer_keys)} unexpected={unexpected[:10]}")
    expected_lora_keys = EXPECTED_OURO_LAYER_COUNT * len(TARGET_MODULES) * 2
    if len(lora_keys) != expected_lora_keys:
        raise RuntimeError(f"Unexpected LoRA tensor count: saved={len(lora_keys)} expected={expected_lora_keys}")
    lora_module_paths = {key.split(".lora_", 1)[0] for key in lora_keys if ".lora_" in key}
    invalid_lora_module_paths = [
        path for path in lora_module_paths if path.rsplit(".", 1)[-1] not in TARGET_MODULES
    ]
    if len(lora_module_paths) != EXPECTED_OURO_LAYER_COUNT * len(TARGET_MODULES) or invalid_lora_module_paths:
        raise RuntimeError(
            "Checkpoint LoRA module contract mismatch: "
            f"module_count={len(lora_module_paths)} invalid={invalid_lora_module_paths[:10]}"
        )
    if len(qformer_keys) != EXPECTED_QFORMER_STATE_TENSORS:
        raise RuntimeError(
            f"Unexpected Q-Former tensor count: saved={len(qformer_keys)} "
            f"expected={EXPECTED_QFORMER_STATE_TENSORS}"
        )
    adapter_config = adapter_config_report(path / "adapter_config.json")
    state_path = path / "trainer_state.json"
    if not state_path.is_file():
        raise RuntimeError(f"Missing trainer_state.json: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if int(state.get("global_step", -1)) != 2:
        raise RuntimeError(f"Expected checkpoint global_step=2, got {state.get('global_step')}")
    return {
        "path": str(path),
        "tensor_count": len(keys),
        "lora_tensor_count": len(lora_keys),
        "expected_lora_tensor_count": expected_lora_keys,
        "qformer_tensor_count": len(qformer_keys),
        "unexpected_tensor_count": len(unexpected),
        "global_step": 2,
        "adapter_config": adapter_config,
    }


def main() -> None:
    args = parse_args()
    BAT_TRAINING.validate()
    require_environment()
    for path in (args.model_path, args.plugin_path, args.dataset):
        if not path.expanduser().resolve().exists():
            raise FileNotFoundError(path)
    dataset_records = count_jsonl(args.dataset)
    if dataset_records != 2:
        raise RuntimeError(
            "BAT LoRA smoke requires exactly 2 JSONL records; "
            f"got {dataset_records} from {args.dataset}. "
            "Generate it with BAT_MANIFEST_LIMIT=2."
        )
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if str(args.output_report).replace("\\", "/").startswith("/hpc_stor03/public"):
        raise ValueError(f"Refusing public output path: {args.output_report}")

    from swift.pipelines.train.sft import SwiftSft

    # Two-record smoke: one batch per epoch, exactly the paper's Stage-I
    # epoch count and batch size.  Because this miniature schedule has only
    # two optimizer steps, both steps are warm-up steps; the full run uses the
    # same formula with the real Stage-I manifest length and then decays.
    schedule = BAT_TRAINING.schedule(dataset_size=2, world_size=1, gradient_accumulation_steps=1, stage_name="I")
    argv = [
        "--model", str(args.model_path), "--model_type", MODEL_TYPE, "--template", TEMPLATE_TYPE,
        "--external_plugins", str(args.plugin_path), "--dataset", str(args.dataset),
        "--split_dataset_ratio", "0", "--dataset_shuffle", "false", "--train_dataloader_shuffle", "false",
        "--sortish_sampler", "false", "--group_by_length", "false", "--max_length", "512",
        "--remove_unused_columns", "false",
        "--output_dir", str(args.output_dir), "--tuner_type", "lora", "--tuner_backend", "peft",
        "--target_modules", *TARGET_MODULES, "--modules_to_save", "audio_qformer",
        "--freeze_llm", "true", "--freeze_vit", "true", "--freeze_aligner", "false",
        "--lora_rank", str(BAT_TRAINING.lora_rank), "--lora_alpha", str(BAT_TRAINING.lora_alpha),
        "--lora_dropout", str(BAT_TRAINING.lora_dropout), "--learning_rate", str(BAT_TRAINING.learning_rate),
        "--lr_scheduler_type", "cosine", "--warmup_steps", str(schedule["warmup_steps"]),
        "--max_steps", str(schedule["total_steps"]), "--num_train_epochs", str(schedule["epochs"]),
        "--per_device_train_batch_size", str(BAT_TRAINING.per_device_batch_size), "--gradient_accumulation_steps", "1",
        "--gradient_checkpointing", "false", "--logging_steps", "1", "--save_strategy", "steps", "--save_steps", "2",
        "--save_total_limit", "1", "--save_only_model", "false", "--dataloader_num_workers", "0",
        "--dataloader_pin_memory", "false", "--dataset_num_proc", "1", "--lazy_tokenize", "false",
        "--load_from_cache_file", "false", "--loss_scale", "all", "--seed", "42", "--data_seed", "42",
        "--optim", "adamw_torch", "--adam_beta1", str(BAT_TRAINING.beta1), "--adam_beta2", str(BAT_TRAINING.beta2),
        "--weight_decay", str(BAT_TRAINING.weight_decay), "--attn_impl", "sdpa", "--bf16", "true", "--report_to", "none",
    ]

    class AuditedSwiftSft(SwiftSft):
        def train(self, trainer):
            model = trainer.model
            causal = find_module(model, "OuroForCausalLM")
            ouro = find_module(model, "OuroModel")
            if int(getattr(causal.config, "total_ut_steps", -1)) != EXPECTED_STEPS:
                raise RuntimeError("Ouro total_ut_steps is not 4")
            if float(getattr(causal, "early_exit_threshold", -1)) != 1.0:
                raise RuntimeError("Ouro early_exit_threshold is not frozen at 1.0")
            causal.config.use_cache = False
            ouro.config.use_cache = False
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
            model.train()
            parameters = parameter_report(model)
            lora = lora_report(model)
            trace = {
                "forward": 0, "backward": 0, "layer": 0, "gate": 0,
                "layer_backward": 0, "gate_backward": 0, "loss": None,
                "batch": None, "gradient_audit": None, "past_key_values_present": None,
            }
            handles: list[Any] = []
            first_layer = ouro.layers[0]
            handles.append(first_layer.register_forward_hook(lambda *_: trace.__setitem__("layer", trace["layer"] + 1)))
            handles.append(ouro.early_exit_gate.register_forward_hook(lambda *_: trace.__setitem__("gate", trace["gate"] + 1)))
            handles.append(first_layer.register_full_backward_hook(lambda *_: trace.__setitem__("layer_backward", trace["layer_backward"] + 1)))
            handles.append(ouro.early_exit_gate.register_full_backward_hook(lambda *_: trace.__setitem__("gate_backward", trace["gate_backward"] + 1)))
            original_compute_loss = trainer.compute_loss
            original_backward = trainer.accelerator.backward

            def compute_loss(actual_model, inputs, return_outputs=False, num_items_in_batch=None):
                trace["forward"] += 1
                result = original_compute_loss(actual_model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
                loss, outputs = result
                if trace["loss"] is None:
                    logits = outputs.logits
                    labels = inputs["labels"]
                    input_ids = inputs.get("input_ids")
                    attention_mask = inputs.get("attention_mask")
                    waveform = inputs.get("audio_waveforms")
                    if not torch.is_tensor(input_ids) or not torch.is_tensor(attention_mask):
                        raise RuntimeError("Real Swift batch is missing tensor input_ids or attention_mask")
                    if shape_tuple(input_ids) != shape_tuple(labels) or shape_tuple(attention_mask) != shape_tuple(input_ids):
                        raise RuntimeError(
                            "Input/label/attention shapes are not aligned: "
                            f"input_ids={shape_tuple(input_ids)} labels={shape_tuple(labels)} "
                            f"attention_mask={shape_tuple(attention_mask)}"
                        )
                    waveform_shape = shape_tuple(waveform) if torch.is_tensor(waveform) else None
                    if waveform_shape != (1, 2, 320000):
                        raise RuntimeError(f"Unexpected eager BAT waveform batch: {type(waveform).__name__} {waveform_shape}")
                    if not bool(torch.isfinite(waveform.float()).all().item()):
                        raise RuntimeError("Eager BAT waveform batch contains NaN or Inf")
                    if logits.ndim != 3 or labels.ndim != 2 or shape_tuple(logits)[:2] != shape_tuple(labels):
                        raise RuntimeError(f"Cannot audit full shifted CE: logits={tuple(logits.shape)} labels={tuple(labels.shape)}")
                    if not bool(torch.isfinite(logits.float()).all().item()):
                        raise RuntimeError("Ouro logits contain NaN or Inf")
                    if not bool(torch.isfinite(loss.detach().float()).all().item()):
                        raise RuntimeError("Trainer loss is NaN or Inf")
                    trace["past_key_values_present"] = getattr(outputs, "past_key_values", None) is not None
                    if trace["past_key_values_present"]:
                        raise RuntimeError("KV cache is unexpectedly enabled during BAT training smoke")
                    shifted_logits = logits[:, :-1].contiguous()
                    shifted_labels = labels[:, 1:].contiguous()
                    manual_loss = F.cross_entropy(
                        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
                        shifted_labels.reshape(-1),
                        ignore_index=-100,
                    )
                    trainer_value = float(loss.detach().float().cpu())
                    manual_value = float(manual_loss.detach().float().cpu())
                    if not math.isfinite(manual_value) or not math.isclose(manual_value, trainer_value, rel_tol=2e-3, abs_tol=2e-3):
                        raise RuntimeError(f"Trainer loss is not ordinary shifted CE: trainer={trainer_value} manual={manual_value}")
                    if not bool((labels[:, :BAT_TRAINING.audio_token_count] == -100).all().item()):
                        raise RuntimeError("Audio prefix labels are not fully masked")
                    trace["batch"] = {
                        "input_ids_shape": list(inputs["input_ids"].shape),
                        "labels_shape": list(labels.shape),
                        "attention_mask_shape": list(inputs["attention_mask"].shape),
                        "audio_waveforms_shape": list(waveform.shape),
                        "audio_prefix_label_ignore_count": int((labels[:, :BAT_TRAINING.audio_token_count] == -100).sum().item()),
                        "valid_shifted_target_count": int((shifted_labels != -100).sum().item()),
                        "manual_shifted_ce": manual_value,
                        "trainer_ce": trainer_value,
                        "shift_verified": True,
                    }
                    trace["loss"] = {"value": trainer_value, "logits_shape": list(logits.shape), "labels_shape": list(labels.shape), "audio_shape": list(inputs["audio_waveforms"].shape)}
                return result if return_outputs else loss

            def backward(loss, **kwargs):
                trace["backward"] += 1
                result = original_backward(loss, **kwargs)
                if trace["gradient_audit"] is None:
                    trace["gradient_audit"] = gradient_report(model)
                return result

            trainer.compute_loss = compute_loss
            trainer.accelerator.backward = backward
            started = time.perf_counter()
            try:
                result = super().train(trainer)
            finally:
                trainer.compute_loss = original_compute_loss
                trainer.accelerator.backward = original_backward
                for handle in handles:
                    handle.remove()
            if int(trainer.state.global_step) != int(schedule["total_steps"]):
                raise RuntimeError(f"Expected {schedule['total_steps']} optimizer steps, got {trainer.state.global_step}")
            if trace["forward"] != schedule["total_steps"] or trace["backward"] != schedule["total_steps"]:
                raise RuntimeError(f"Unexpected trainer forward/backward counts: {trace}")
            if trace["gradient_audit"] is None:
                raise RuntimeError("First backward gradient audit was not captured")
            if trace["layer"] != EXPECTED_STEPS * schedule["total_steps"] or trace["gate"] != EXPECTED_STEPS * schedule["total_steps"]:
                raise RuntimeError(f"Unexpected Ouro recurrent forward counts: {trace}")
            if trace["layer_backward"] != EXPECTED_STEPS * schedule["total_steps"]:
                raise RuntimeError(f"Unexpected Ouro recurrent backward count: {trace}")
            opt = optimizer_report(trainer, model)
            checkpoint = checkpoint_report(checkpoint_path(args.output_dir))
            report = {
                "status": "ok",
                "paper_contract": {
                    "sound_source": BAT_TRAINING.sound_source, "audio_normalization": BAT_TRAINING.audio_normalization,
                    "augmentation": BAT_TRAINING.augmentation, "weighted_sampling": BAT_TRAINING.weighted_sampling,
                    "optimizer": BAT_TRAINING.optimizer, "betas": [BAT_TRAINING.beta1, BAT_TRAINING.beta2],
                    "weight_decay": BAT_TRAINING.weight_decay, "learning_rate": BAT_TRAINING.learning_rate,
                    "scheduler": BAT_TRAINING.scheduler, "warmup_epochs": BAT_TRAINING.warmup_epochs,
                    "epoch_partitioning_factor": BAT_TRAINING.epoch_partitioning_factor, "stage": "I",
                    "stage_epochs": 2, "batch_size": BAT_TRAINING.per_device_batch_size,
                },
                "schedule": dict(schedule), "dataset_records": dataset_records, "argv": argv,
                "parameters": parameters, "lora": lora, "optimizer": opt,
                "forward_audit": {**trace, "expected_recurrent_steps": EXPECTED_STEPS, "use_cache": False},
                "checkpoint": checkpoint, "elapsed_seconds": time.perf_counter() - started,
                "trainer": {"class": f"{trainer.__class__.__module__}.{trainer.__class__.__name__}", "global_step": int(trainer.state.global_step), "log_history": trainer.state.log_history, "result_type": type(result).__name__},
            }
            write_json(args.output_report, report)
            print(f"[parameters] {json.dumps(parameters, ensure_ascii=False)}", flush=True)
            print(f"[lora] {json.dumps(lora, ensure_ascii=False)}", flush=True)
            print(f"[optimizer] {json.dumps(opt, ensure_ascii=False)}", flush=True)
            print(f"[forward] {json.dumps(trace, ensure_ascii=False)}", flush=True)
            print(f"[checkpoint] {json.dumps(checkpoint, ensure_ascii=False)}", flush=True)
            print(f"[report] {args.output_report}", flush=True)
            print("[status] ok", flush=True)
            return result

    print("========== BAT OURO LORA + Q-FORMER STRICT SMOKE ==========")
    print(f"[packages] ms-swift={package_version('ms-swift')} transformers={package_version('transformers')} torch={package_version('torch')}")
    print(f"[schedule] {json.dumps(schedule, ensure_ascii=False)}")
    print(f"[argv] {' '.join(argv)}")
    AuditedSwiftSft(argv).main()


if __name__ == "__main__":
    main()
