#!/usr/bin/env python3
"""Run and audit one ms-swift LoRA+gate update on text-only Ouro-1.4B.

This is deliberately a one-step engineering smoke, not a useful fine-tuning
run.  It uses the published Ouro forward path and ordinary causal-LM CE.  The
script audits the real Swift trainer batch, Ouro's four recurrent calls,
trainable parameter topology, optimizer membership, gradients, updates, and
the saved adapter checkpoint.
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


MODEL_TYPE = "ouro_text_native"
TEMPLATE_TYPE = "ouro_text_direct"
EXPECTED_STEPS = 4
EXPECTED_BATCH_SIZE = 3
LORA_RANK = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
LEARNING_RATE = 1e-4
TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
GATE_MODULE = "early_exit_gate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--plugin-path", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def package_version(name: str) -> str:
    try:
        return version(name)
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        return f"<unavailable: {type(exc).__name__}: {exc}>"


def require_environment() -> None:
    expected = {
        "ms-swift": "4.4.2",
        "transformers": "4.54.1",
        "peft": "0.18.1",
    }
    mismatches = {
        name: {"expected": target, "actual": package_version(name)}
        for name, target in expected.items()
        if package_version(name) != target
    }
    if mismatches:
        raise RuntimeError(f"Unexpected Ouro Swift environment: {mismatches}")
    if not torch.cuda.is_available():
        raise RuntimeError("This Ouro LoRA smoke is GPU-only and must run in a submitted job")


def find_unique_module(model: torch.nn.Module, class_name: str) -> torch.nn.Module:
    matches = [module for module in model.modules() if module.__class__.__name__ == class_name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {class_name}, found {len(matches)}")
    return matches[0]


def parameter_category(name: str, parameter: torch.nn.Parameter) -> str:
    if "lora_A" in name or "lora_B" in name:
        return "lora"
    if GATE_MODULE in name:
        return "gate"
    return "backbone" if not parameter.requires_grad else "unexpected"


def parameter_report(model: torch.nn.Module) -> dict[str, Any]:
    totals = {"all": 0, "trainable": 0, "frozen": 0, "lora": 0, "gate": 0, "unexpected": 0}
    trainable_names: list[str] = []
    trainable_dtypes: dict[str, int] = {}
    categories: dict[str, list[str]] = {key: [] for key in ("lora", "gate", "unexpected")}
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        totals["all"] += count
        if parameter.requires_grad:
            totals["trainable"] += count
            trainable_names.append(name)
            dtype = str(parameter.dtype)
            trainable_dtypes[dtype] = trainable_dtypes.get(dtype, 0) + count
            category = parameter_category(name, parameter)
            totals[category] += count
            categories.setdefault(category, []).append(name)
        else:
            totals["frozen"] += count
    if totals["unexpected"]:
        raise RuntimeError(f"Unexpected trainable backbone parameters: {categories['unexpected'][:40]}")
    if totals["lora"] <= 0:
        raise RuntimeError("No trainable LoRA parameters were found")
    if totals["gate"] <= 0:
        raise RuntimeError("No trainable Ouro early-exit gate parameters were found")
    if not trainable_names:
        raise RuntimeError("The PEFT model has no trainable parameters")
    return {
        **totals,
        "trainable_names": trainable_names,
        "trainable_name_count": len(trainable_names),
        "trainable_dtypes": trainable_dtypes,
        "category_name_counts": {key: len(value) for key, value in categories.items()},
        "category_name_preview": {key: value[:20] for key, value in categories.items()},
    }


def normalize_parameter_name(name: str) -> str:
    normalized = name
    for prefix in ("base_model.model.", "base_model."):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
    normalized = normalized.replace(".base_layer.", ".")
    return normalized


def gate_trainable_names(model: torch.nn.Module) -> list[str]:
    names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and GATE_MODULE in name and "lora_" not in name
    ]
    if not names:
        raise RuntimeError("The early-exit gate was not made trainable")
    if any("original_module" in name for name in names):
        raise RuntimeError(f"The frozen original gate copy is unexpectedly trainable: {names}")
    return names


def lora_module_report(model: torch.nn.Module) -> dict[str, Any]:
    modules: list[str] = []
    invalid: list[str] = []
    ranks: set[int] = set()
    for name, module in model.named_modules():
        if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            continue
        modules.append(name)
        suffix = name.rsplit(".", 1)[-1]
        if suffix not in TARGET_MODULES:
            invalid.append(name)
        configured_ranks = getattr(module, "r", {})
        if isinstance(configured_ranks, dict):
            ranks.update(int(value) for value in configured_ranks.values())
        else:
            ranks.add(int(configured_ranks))
    expected_count = 24 * len(TARGET_MODULES)
    if len(modules) != expected_count:
        raise RuntimeError(
            f"Unexpected Ouro LoRA module count: expected={expected_count} actual={len(modules)} "
            f"preview={modules[:20]}"
        )
    if invalid:
        raise RuntimeError(f"Unexpected LoRA target modules: {invalid[:40]}")
    if ranks != {LORA_RANK}:
        raise RuntimeError(f"Unexpected LoRA ranks: expected={LORA_RANK} actual={sorted(ranks)}")
    if any(GATE_MODULE in name for name in modules):
        raise RuntimeError("early_exit_gate unexpectedly received a LoRA adapter")
    return {
        "module_count": len(modules),
        "expected_module_count": expected_count,
        "target_modules": list(TARGET_MODULES),
        "ranks": sorted(ranks),
        "module_preview": sorted(modules)[:20],
    }


def select_update_probes(model: torch.nn.Module) -> dict[str, dict[str, Any]]:
    candidates = {
        "lora_B": [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and "lora_B" in name
        ],
        "gate": [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and GATE_MODULE in name and "weight" in name
        ],
        "frozen": [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if not parameter.requires_grad and GATE_MODULE not in name and "lora_" not in name
        ],
    }
    probes: dict[str, dict[str, Any]] = {}
    for category, entries in candidates.items():
        if not entries:
            raise RuntimeError(f"No parameter available for update probe category={category}")
        name, parameter = sorted(entries, key=lambda item: item[0])[0]
        probes[category] = {
            "name": name,
            "parameter": parameter,
            "before": parameter.detach().float().cpu().clone(),
        }
    return probes


def install_runtime_audits(model: torch.nn.Module) -> tuple[dict[str, Any], list[Any]]:
    ouro_model = find_unique_module(model, "OuroModel")
    causal_model = find_unique_module(model, "OuroForCausalLM")
    trace: dict[str, Any] = {
        "trainer_forward_calls": 0,
        "first_layer_calls": 0,
        "gate_calls": 0,
        "batch": None,
        "logits": None,
        "native_loss": None,
        "output_shape": None,
        "past_key_values_present": None,
        "gradient_records": {},
    }
    handles: list[Any] = []

    def pre_hook(_module: torch.nn.Module, _args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        trace["trainer_forward_calls"] += 1
        if trace["batch"] is not None:
            return
        input_ids = kwargs.get("input_ids")
        labels = kwargs.get("labels")
        attention_mask = kwargs.get("attention_mask")
        if not all(torch.is_tensor(value) for value in (input_ids, labels, attention_mask)):
            raise RuntimeError(
                "Swift trainer did not pass the expected causal-LM tensors: "
                f"keys={sorted(kwargs)}"
            )
        trace["batch"] = {
            "input_ids": input_ids.detach().cpu(),
            "labels": labels.detach().cpu(),
            "attention_mask": attention_mask.detach().cpu(),
            "use_cache": kwargs.get("use_cache"),
            "logits_to_keep": kwargs.get("logits_to_keep"),
        }

    def output_hook(_module: torch.nn.Module, _args: tuple[Any, ...], output: Any) -> None:
        if trace["logits"] is not None:
            return
        logits = getattr(output, "logits", None)
        if logits is None and isinstance(output, (tuple, list)) and output:
            logits = output[0]
        if not torch.is_tensor(logits):
            raise RuntimeError(f"Ouro forward did not expose logits; output_type={type(output)}")
        trace["logits"] = logits.detach().float().cpu()
        loss = getattr(output, "loss", None)
        trace["native_loss"] = None if loss is None else float(loss.detach().float().cpu().item())
        trace["output_shape"] = list(logits.shape)
        trace["past_key_values_present"] = getattr(output, "past_key_values", None) is not None

    def first_layer_hook(_module: torch.nn.Module, _args: tuple[Any, ...], _output: Any) -> None:
        trace["first_layer_calls"] += 1

    def gate_hook(_module: torch.nn.Module, _args: tuple[Any, ...], _output: Any) -> None:
        trace["gate_calls"] += 1

    def make_gradient_hook(parameter_name: str):
        def gradient_hook(gradient: torch.Tensor) -> torch.Tensor:
            norm = float(gradient.detach().float().norm().item())
            trace["gradient_records"][parameter_name] = {
                "grad_norm": norm,
                "numel": int(gradient.numel()),
                "finite": bool(torch.isfinite(gradient).all().item()),
            }
            return gradient

        return gradient_hook

    handles.append(model.register_forward_pre_hook(pre_hook, with_kwargs=True))
    handles.append(causal_model.register_forward_hook(output_hook))
    handles.append(ouro_model.layers[0].register_forward_hook(first_layer_hook))
    handles.append(ouro_model.early_exit_gate.register_forward_hook(gate_hook))
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            handles.append(parameter.register_hook(make_gradient_hook(name)))
    return trace, handles


def remove_handles(handles: list[Any]) -> None:
    for handle in handles:
        handle.remove()


def validate_batch_and_loss(trace: dict[str, Any]) -> dict[str, Any]:
    batch = trace.get("batch")
    logits = trace.get("logits")
    if batch is None or logits is None:
        raise RuntimeError("No real Swift training forward was captured")
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]
    if tuple(input_ids.shape) != (EXPECTED_BATCH_SIZE, input_ids.shape[1]):
        raise RuntimeError(f"Unexpected training batch size: shape={tuple(input_ids.shape)}")
    if labels.shape != input_ids.shape or attention_mask.shape != input_ids.shape:
        raise RuntimeError(
            "This smoke requires full-sequence causal labels without compact logits: "
            f"input={tuple(input_ids.shape)} labels={tuple(labels.shape)} attention={tuple(attention_mask.shape)}"
        )
    if logits.shape[:2] != input_ids.shape:
        raise RuntimeError(
            "Ouro logits and labels are not aligned before shifting: "
            f"logits={tuple(logits.shape)} input={tuple(input_ids.shape)}"
        )
    expected_targets = input_ids[:, 1:]
    actual_targets = labels[:, 1:]
    target_mask = attention_mask[:, 1:].bool()
    if not torch.equal(actual_targets[target_mask], expected_targets[target_mask]):
        raise RuntimeError("Swift labels are not the expected next-token targets")
    if bool((actual_targets[~target_mask] != -100).any().item()):
        raise RuntimeError("Padded next-token labels are not masked with -100")
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    manual_loss = torch.nn.functional.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=-100,
    )
    native_loss = trace.get("native_loss")
    if native_loss is None or not math.isfinite(native_loss):
        raise RuntimeError(f"Ouro native CE loss is missing or non-finite: {native_loss}")
    manual_value = float(manual_loss.item())
    if not math.isclose(manual_value, native_loss, rel_tol=2e-3, abs_tol=2e-3):
        raise RuntimeError(f"Native CE does not match explicit shifted CE: native={native_loss} manual={manual_value}")
    valid_targets = int((shifted_labels != -100).sum().item())
    if valid_targets <= 0:
        raise RuntimeError("The captured batch has no valid next-token targets")
    return {
        "input_shape": list(input_ids.shape),
        "labels_shape": list(labels.shape),
        "attention_shape": list(attention_mask.shape),
        "logits_shape": list(logits.shape),
        "valid_next_token_targets": valid_targets,
        "manual_shifted_ce": manual_value,
        "native_forward_loss": native_loss,
        "shift_verified": True,
        "use_cache_argument": batch.get("use_cache"),
        "logits_to_keep": batch.get("logits_to_keep"),
        "past_key_values_present": trace.get("past_key_values_present"),
    }


def inspect_optimizer(trainer: Any, model: torch.nn.Module) -> dict[str, Any]:
    if trainer.optimizer is None:
        raise RuntimeError("Swift trainer did not create an optimizer")
    optimizer_parameters = {
        id(parameter): parameter
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    }
    trainable_parameters = {id(parameter): parameter for parameter in model.parameters() if parameter.requires_grad}
    if set(optimizer_parameters) != set(trainable_parameters):
        raise RuntimeError(
            "Optimizer parameters differ from model trainables: "
            f"optimizer_only={len(set(optimizer_parameters) - set(trainable_parameters))} "
            f"trainable_only={len(set(trainable_parameters) - set(optimizer_parameters))}"
        )
    learning_rates = [float(group["lr"]) for group in trainer.optimizer.param_groups]
    if not learning_rates or any(not math.isfinite(value) or value <= 0 for value in learning_rates):
        raise RuntimeError(f"Invalid optimizer learning rates: {learning_rates}")
    return {
        "class": f"{trainer.optimizer.__class__.__module__}.{trainer.optimizer.__class__.__name__}",
        "parameter_tensor_count": len(optimizer_parameters),
        "parameter_count": sum(int(parameter.numel()) for parameter in optimizer_parameters.values()),
        "group_count": len(trainer.optimizer.param_groups),
        "learning_rates": learning_rates,
        "scheduler_class": None
        if trainer.lr_scheduler is None
        else f"{trainer.lr_scheduler.__class__.__module__}.{trainer.lr_scheduler.__class__.__name__}",
        "scheduler_last_epoch": None if trainer.lr_scheduler is None else int(trainer.lr_scheduler.last_epoch),
    }


def gradient_report(model: torch.nn.Module, gradient_records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    missing = [name for name in trainable_names if name not in gradient_records]
    nonfinite = [name for name, item in gradient_records.items() if not item["finite"]]
    if missing:
        raise RuntimeError(f"Trainable parameters without autograd-hook gradients: {missing[:40]}")
    if nonfinite:
        raise RuntimeError(f"Non-finite trainable gradients: {nonfinite[:40]}")
    trainable = [
        {"name": name, **gradient_records[name]}
        for name in trainable_names
    ]
    return {
        "trainable_tensor_count": len(trainable),
        "trainable_gradients": trainable[:80],
        "trainable_gradients_truncated": len(trainable) > 80,
        "missing_trainable_gradients": missing,
        "frozen_parameters_with_gradients": [],
        "capture_method": "autograd_parameter_hook_before_trainer_zero_grad",
    }


def checkpoint_report(checkpoint_dir: Path) -> dict[str, Any]:
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Expected checkpoint directory does not exist: {checkpoint_dir}")
    from safetensors import safe_open

    adapter_path = checkpoint_dir / "adapter_model.safetensors"
    if not adapter_path.is_file():
        raise RuntimeError(f"Missing adapter checkpoint: {adapter_path}")
    with safe_open(str(adapter_path), framework="pt", device="cpu") as handle:
        keys = sorted(handle.keys())
        dtype_counts: dict[str, int] = {}
        for key in keys:
            dtype = str(handle.get_tensor(key).dtype)
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    lora_keys = [key for key in keys if "lora_" in key]
    gate_keys = [key for key in keys if GATE_MODULE in key]
    unexpected = [key for key in keys if key not in lora_keys and key not in gate_keys]
    if not lora_keys:
        raise RuntimeError("Checkpoint contains no LoRA tensors")
    if len(gate_keys) < 2:
        raise RuntimeError(f"Checkpoint does not contain both early-exit gate tensors: {gate_keys}")
    if unexpected:
        raise RuntimeError(f"Checkpoint contains unexpected non-LoRA/non-gate tensors: {unexpected[:40]}")
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    if not trainer_state_path.is_file():
        raise RuntimeError(f"Missing trainer state: {trainer_state_path}")
    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    if int(trainer_state.get("global_step", -1)) != 1:
        raise RuntimeError(f"Checkpoint global_step is not 1: {trainer_state.get('global_step')}")
    return {
        "path": str(checkpoint_dir),
        "tensor_count": len(keys),
        "lora_tensor_count": len(lora_keys),
        "gate_tensor_count": len(gate_keys),
        "unexpected_tensor_count": len(unexpected),
        "key_preview": keys[:30],
        "dtype_counts": dtype_counts,
        "global_step": int(trainer_state["global_step"]),
    }


def main() -> None:
    args = parse_args()
    require_environment()
    model_path = args.model_path.expanduser().resolve()
    plugin_path = args.plugin_path.expanduser().resolve()
    dataset_path = args.dataset.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_report = args.output_report.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Ouro model directory does not exist: {model_path}")
    if not plugin_path.is_file():
        raise FileNotFoundError(f"Ouro Swift plugin does not exist: {plugin_path}")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Tiny Ouro dataset does not exist: {dataset_path}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")

    from swift.pipelines.train.sft import SwiftSft

    argv = [
        "--model", str(model_path),
        "--model_type", MODEL_TYPE,
        "--template", TEMPLATE_TYPE,
        "--external_plugins", str(plugin_path),
        "--dataset", str(dataset_path),
        "--split_dataset_ratio", "0",
        "--dataset_shuffle", "false",
        "--train_dataloader_shuffle", "false",
        "--sortish_sampler", "false",
        "--group_by_length", "false",
        "--max_length", "32",
        "--output_dir", str(output_dir),
        "--tuner_type", "lora",
        "--tuner_backend", "peft",
        "--target_modules", *TARGET_MODULES,
        "--modules_to_save", GATE_MODULE,
        "--lora_rank", str(LORA_RANK),
        "--lora_alpha", str(LORA_ALPHA),
        "--lora_dropout", str(LORA_DROPOUT),
        "--learning_rate", str(LEARNING_RATE),
        "--lr_scheduler_type", "constant",
        "--warmup_ratio", "0",
        "--max_steps", "1",
        "--num_train_epochs", "1",
        "--per_device_train_batch_size", str(EXPECTED_BATCH_SIZE),
        "--gradient_accumulation_steps", "1",
        "--gradient_checkpointing", "false",
        "--logging_steps", "1",
        "--save_strategy", "steps",
        "--save_steps", "1",
        "--save_total_limit", "1",
        "--save_only_model", "false",
        "--dataloader_num_workers", "0",
        "--dataloader_pin_memory", "false",
        "--dataset_num_proc", "1",
        "--lazy_tokenize", "false",
        "--load_from_cache_file", "false",
        "--loss_scale", "all",
        "--seed", "42",
        "--data_seed", "42",
        "--optim", "adamw_torch",
        "--attn_impl", "sdpa",
        "--bf16", "true",
        "--report_to", "none",
    ]

    class AuditedSwiftSft(SwiftSft):
        def train(self, trainer):
            model = trainer.model
            causal_model = find_unique_module(model, "OuroForCausalLM")
            ouro_model = find_unique_module(model, "OuroModel")
            if int(getattr(causal_model.config, "total_ut_steps", -1)) != EXPECTED_STEPS:
                raise RuntimeError(
                    f"Unexpected Ouro config total_ut_steps: {getattr(causal_model.config, 'total_ut_steps', None)}"
                )
            if int(getattr(ouro_model, "total_ut_steps", -1)) != EXPECTED_STEPS:
                raise RuntimeError(f"Unexpected Ouro runtime total_ut_steps: {ouro_model.total_ut_steps}")
            causal_model.config.use_cache = False
            ouro_model.config.use_cache = False
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
            if getattr(causal_model, "training", True) is False:
                model.train()

            parameters_before = parameter_report(model)
            gate_names = gate_trainable_names(model)
            lora_report = lora_module_report(model)
            probes = select_update_probes(model)
            trace, handles = install_runtime_audits(model)
            print("========== OURO SWIFT LORA PRE-TRAIN AUDIT ==========")
            print(f"[trainer] {trainer.__class__.__module__}.{trainer.__class__.__name__}", flush=True)
            print(f"[config] total_ut_steps={ouro_model.total_ut_steps} use_cache={causal_model.config.use_cache}", flush=True)
            print(f"[parameters] {json.dumps(parameters_before, ensure_ascii=False)}", flush=True)
            print(f"[gate] trainable_names={gate_names}", flush=True)
            print(f"[lora] {json.dumps(lora_report, ensure_ascii=False)}", flush=True)

            started = time.perf_counter()
            try:
                train_result = super().train(trainer)
            finally:
                remove_handles(handles)
            torch.cuda.synchronize()
            elapsed_seconds = time.perf_counter() - started

            if int(trainer.state.global_step) != 1:
                raise RuntimeError(f"Expected exactly one optimizer step, got {trainer.state.global_step}")
            if trace["trainer_forward_calls"] != 1:
                raise RuntimeError(
                    f"Expected exactly one real trainer forward, got {trace['trainer_forward_calls']}"
                )
            if trace["first_layer_calls"] != EXPECTED_STEPS:
                raise RuntimeError(
                    f"Expected first shared decoder layer to run {EXPECTED_STEPS} times, "
                    f"got {trace['first_layer_calls']}"
                )
            if trace["gate_calls"] != EXPECTED_STEPS:
                raise RuntimeError(
                    f"Expected early_exit_gate to run {EXPECTED_STEPS} times, got {trace['gate_calls']}"
                )
            if trace["past_key_values_present"]:
                raise RuntimeError("Training forward unexpectedly returned a KV cache")

            loss_report = validate_batch_and_loss(trace)
            optimizer_report = inspect_optimizer(trainer, model)
            gradient_report_value = gradient_report(model, trace["gradient_records"])

            update_report: dict[str, Any] = {}
            for category, probe in probes.items():
                parameter = probe["parameter"]
                delta = (parameter.detach().float().cpu() - probe["before"]).abs()
                max_abs_change = float(delta.max().item())
                update_report[category] = {
                    "name": probe["name"],
                    "max_abs_change": max_abs_change,
                }
                if category != "frozen" and max_abs_change <= 0:
                    raise RuntimeError(f"Expected {category} probe to update, observed no change")
                if category == "frozen" and max_abs_change != 0:
                    raise RuntimeError(f"Frozen probe changed: {probe['name']} delta={max_abs_change}")

            checkpoint_dir = output_dir / "checkpoint-1"
            checkpoint = checkpoint_report(checkpoint_dir)
            log_history = trainer.state.log_history
            finite_losses = [
                float(item["loss"])
                for item in log_history
                if "loss" in item and math.isfinite(float(item["loss"]))
            ]
            if not finite_losses:
                raise RuntimeError(f"No finite trainer loss was recorded: {log_history}")

            report = {
                "status": "ok",
                "policy": {
                    "loss": "ordinary_causal_lm_cross_entropy",
                    "entropy_or_kl": False,
                    "total_ut_steps": EXPECTED_STEPS,
                    "use_cache": False,
                    "backward_calls": 1,
                    "optimizer_steps": 1,
                },
                "packages": {
                    "ms-swift": package_version("ms-swift"),
                    "transformers": package_version("transformers"),
                    "peft": package_version("peft"),
                    "torch": package_version("torch"),
                },
                "model_path": str(model_path),
                "plugin_path": str(plugin_path),
                "dataset": str(dataset_path),
                "output_dir": str(output_dir),
                "argv": argv,
                "trainer": {
                    "class": f"{trainer.__class__.__module__}.{trainer.__class__.__name__}",
                    "global_step": int(trainer.state.global_step),
                    "elapsed_seconds": elapsed_seconds,
                    "log_history": log_history,
                    "finite_losses": finite_losses,
                    "train_result_type": f"{type(train_result).__module__}.{type(train_result).__name__}",
                },
                "parameters": parameters_before,
                "gate": {"trainable_names": gate_names},
                "lora": lora_report,
                "forward_audit": {
                    "trainer_forward_calls": trace["trainer_forward_calls"],
                    "first_shared_decoder_layer_calls": trace["first_layer_calls"],
                    "early_exit_gate_calls": trace["gate_calls"],
                    **loss_report,
                },
                "optimizer": optimizer_report,
                "gradients": gradient_report_value,
                "updates": update_report,
                "checkpoint": checkpoint,
                "cuda_memory": {
                    "device_index": torch.cuda.current_device(),
                    "device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                },
            }
            atomic_write_json(output_report, report)
            print("========== OURO SWIFT LORA POST-TRAIN AUDIT ==========")
            print(f"[loss] {json.dumps(loss_report, ensure_ascii=False)}", flush=True)
            print(f"[optimizer] {json.dumps(optimizer_report, ensure_ascii=False)}", flush=True)
            print(f"[gradients] trainable_tensors={gradient_report_value['trainable_tensor_count']}", flush=True)
            print(f"[updates] {json.dumps(update_report, ensure_ascii=False)}", flush=True)
            print(
                f"[checkpoint] path={checkpoint_dir} lora_tensors={checkpoint['lora_tensor_count']} "
                f"gate_tensors={checkpoint['gate_tensor_count']}",
                flush=True,
            )
            print(f"[result] status=OK output_report={output_report}", flush=True)
            return train_result

    print("========== OURO MS-SWIFT TEXT LORA ONE-STEP SMOKE ==========")
    print(f"[python] version={sys.version.split()[0]} executable={sys.executable}", flush=True)
    print(f"[torch] version={package_version('torch')} device={torch.cuda.get_device_name()}", flush=True)
    print(f"[model-path] {model_path}", flush=True)
    print(f"[plugin-path] {plugin_path}", flush=True)
    print(f"[dataset] {dataset_path}", flush=True)
    print(f"[output-dir] {output_dir}", flush=True)
    print(f"[output-report] {output_report}", flush=True)
    print(
        f"[policy] CE LoRA-rank={LORA_RANK} gate-trainable target_modules={list(TARGET_MODULES)} "
        f"steps={EXPECTED_STEPS} batch={EXPECTED_BATCH_SIZE} max_steps=1 use_cache=false",
        flush=True,
    )
    print("[argv] " + " ".join(argv), flush=True)
    AuditedSwiftSft(argv).main()


if __name__ == "__main__":
    main()
