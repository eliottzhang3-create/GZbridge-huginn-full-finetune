#!/usr/bin/env python3
"""Run one real SwiftSft/Trainer LoRA update on text-only HRM-Text."""

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


MODEL_TYPE = "hrm_text_native"
TEMPLATE_TYPE = "hrm_text_direct"
EXPECTED_BASE_PARAMETERS = 1_182_795_264
EXPECTED_LORA_MODULES = 256
EXPECTED_LORA_TENSORS = EXPECTED_LORA_MODULES * 2
EXPECTED_LORA_PARAMETERS_RANK8 = 8_257_536


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


def count_parameters(model: torch.nn.Module) -> dict[str, Any]:
    total = 0
    trainable = 0
    trainable_names: list[str] = []
    trainable_dtypes: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
            trainable_names.append(name)
            dtype = str(parameter.dtype)
            trainable_dtypes[dtype] = trainable_dtypes.get(dtype, 0) + count
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_names": trainable_names,
        "trainable_dtypes": trainable_dtypes,
    }


def find_unique_module(model: torch.nn.Module, class_name: str) -> torch.nn.Module:
    matches = [module for module in model.modules() if module.__class__.__name__ == class_name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {class_name}, found {len(matches)}")
    return matches[0]


def lora_module_report(model: torch.nn.Module) -> dict[str, Any]:
    names = [
        name
        for name, module in model.named_modules()
        if hasattr(module, "lora_A") and hasattr(module, "lora_B")
    ]
    groups = {
        "H": [name for name in names if ".H_module." in name],
        "L": [name for name in names if ".L_module." in name],
        "other": [name for name in names if ".H_module." not in name and ".L_module." not in name],
    }
    if len(names) != EXPECTED_LORA_MODULES:
        raise RuntimeError(f"Unexpected injected LoRA module count: expected={EXPECTED_LORA_MODULES} actual={len(names)}")
    if len(groups["H"]) != 128 or len(groups["L"]) != 128 or groups["other"]:
        raise RuntimeError(
            f"LoRA H/L split is wrong: H={len(groups['H'])} L={len(groups['L'])} other={groups['other'][:20]}"
        )
    if any("lm_head" in name for name in names):
        raise RuntimeError("Swift all-linear unexpectedly injected LoRA into lm_head")
    return {
        "count": len(names),
        "H_count": len(groups["H"]),
        "L_count": len(groups["L"]),
        "other_count": len(groups["other"]),
        "preview": names[:20],
    }


def select_update_probes(model: torch.nn.Module) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for stack_name, marker in (("H", ".H_module."), ("L", ".L_module.")):
        candidates = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if marker in name and "lora_B" in name and name.endswith("weight") and parameter.requires_grad
        ]
        if not candidates:
            raise RuntimeError(f"No trainable {stack_name}-stack lora_B parameter found")
        name, parameter = candidates[0]
        result[stack_name] = {
            "name": name,
            "parameter": parameter,
            "before": parameter.detach().float().cpu().clone(),
        }
    return result


def capture_training_forward(tokenizer: Any):
    captures: list[dict[str, Any]] = []

    def hook(_module, _args, kwargs):
        if captures:
            return
        required = ("input_ids", "attention_mask", "labels", "token_type_ids", "logits_to_keep")
        missing = [name for name in required if not torch.is_tensor(kwargs.get(name))]
        # For batch_size > 1 Swift represents logits_to_keep as a Python int.
        if "logits_to_keep" in missing and isinstance(kwargs.get("logits_to_keep"), int):
            missing.remove("logits_to_keep")
        if missing:
            raise RuntimeError(f"Trainer forward dropped required HRM tensors: {missing}; keys={sorted(kwargs)}")
        input_ids = kwargs["input_ids"].detach().cpu()
        attention_mask = kwargs["attention_mask"].detach().cpu()
        labels = kwargs["labels"].detach().cpu()
        token_type_ids = kwargs["token_type_ids"].detach().cpu()
        logits_to_keep_value = kwargs["logits_to_keep"]
        if torch.is_tensor(logits_to_keep_value):
            if logits_to_keep_value.ndim != 0:
                raise RuntimeError(
                    "This batch-size-2 smoke expects integer logits_to_keep, got "
                    f"tensor shape={tuple(logits_to_keep_value.shape)}"
                )
            logits_to_keep = int(logits_to_keep_value.item())
        else:
            logits_to_keep = int(logits_to_keep_value)
        if not (input_ids.shape == attention_mask.shape == token_type_ids.shape):
            raise RuntimeError(
                "Trainer batch shapes disagree: "
                f"input={tuple(input_ids.shape)} attention={tuple(attention_mask.shape)} "
                f"token_type={tuple(token_type_ids.shape)}"
            )
        if logits_to_keep <= 1 or labels.shape != (input_ids.shape[0], logits_to_keep):
            raise RuntimeError(
                "Swift logits_to_keep/compact-label shape mismatch: "
                f"input={tuple(input_ids.shape)} labels={tuple(labels.shape)} "
                f"logits_to_keep={logits_to_keep}"
            )

        im_end_id = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
        eos_id = int(tokenizer.eos_token_id)
        rows: list[dict[str, Any]] = []
        sequence_width = int(input_ids.shape[1])
        compact_start = sequence_width - logits_to_keep
        for row_index in range(input_ids.shape[0]):
            valid_length = int(attention_mask[row_index].sum().item())
            valid_ids = input_ids[row_index, :valid_length].tolist()
            valid_types = token_type_ids[row_index, :valid_length].tolist()
            response_positions = [index for index, token_type in enumerate(valid_types) if token_type == 0]
            if not response_positions:
                raise RuntimeError(f"Trainer row {row_index} has no causal response region")
            prefix_length = response_positions[0]
            if response_positions != list(range(prefix_length, valid_length)):
                raise RuntimeError(f"Trainer row {row_index} response token types are not contiguous")
            if prefix_length <= 0 or valid_ids[prefix_length - 1] != im_end_id:
                raise RuntimeError(f"Trainer row {row_index} first response is not predicted from <|im_end|>")
            if set(valid_types[:prefix_length]) != {1} or set(valid_types[prefix_length:]) != {0}:
                raise RuntimeError(f"Trainer row {row_index} PrefixLM token types are invalid")

            # Swift 4.4.2's prepare_logits_to_keep keeps only a common suffix
            # of labels for batch_size > 1. It adds one leading prediction
            # position before the earliest response boundary, then HRM returns
            # exactly the same suffix of logits. Reconstruct the full labels
            # from the PrefixLM boundary and prove that the compact labels are
            # the exact expected suffix.
            expected_full_labels = [-100] * sequence_width
            expected_full_labels[prefix_length:valid_length] = valid_ids[prefix_length:valid_length]
            expected_compact_labels = expected_full_labels[-logits_to_keep:]
            actual_compact_labels = labels[row_index].tolist()
            if actual_compact_labels != expected_compact_labels:
                raise RuntimeError(
                    f"Trainer row {row_index} compact labels are wrong: "
                    f"expected={expected_compact_labels} actual={actual_compact_labels}"
                )
            supervised_compact = [
                index for index, label in enumerate(actual_compact_labels) if label != -100
            ]
            if not supervised_compact or supervised_compact[0] <= 0:
                raise RuntimeError(
                    f"Trainer row {row_index} compact labels lack a leading prediction position: "
                    f"positions={supervised_compact}"
                )
            original_supervised_positions = [compact_start + index for index in supervised_compact]
            if original_supervised_positions != list(range(prefix_length, valid_length)):
                raise RuntimeError(
                    f"Trainer row {row_index} compact-label positions do not map to the full response: "
                    f"mapped={original_supervised_positions} response={list(range(prefix_length, valid_length))}"
                )
            first_prediction_position = compact_start + supervised_compact[0] - 1
            if first_prediction_position != prefix_length - 1:
                raise RuntimeError(
                    f"Trainer row {row_index} first compact logit does not predict the first response token"
                )
            last_supervised_compact = supervised_compact[-1]
            if actual_compact_labels[last_supervised_compact] != eos_id:
                raise RuntimeError(f"Trainer row {row_index} compact labels do not supervise EOS")
            if valid_length < input_ids.shape[1]:
                if not bool((attention_mask[row_index, valid_length:] == 0).all().item()):
                    raise RuntimeError(f"Trainer row {row_index} attention padding is not zero")
                if not bool((token_type_ids[row_index, valid_length:] == 0).all().item()):
                    raise RuntimeError(f"Trainer row {row_index} token-type padding is not zero")
            rows.append(
                {
                    "row": row_index,
                    "valid_length": valid_length,
                    "prefix_length": prefix_length,
                    "compact_start": compact_start,
                    "compact_label_length": logits_to_keep,
                    "supervised_targets": len(supervised_compact),
                    "supervised_compact_positions": supervised_compact,
                    "original_supervised_positions": original_supervised_positions,
                    "first_prediction_position": first_prediction_position,
                    "first_prediction_token": tokenizer.convert_ids_to_tokens(valid_ids[first_prediction_position]),
                    "first_target_token": tokenizer.convert_ids_to_tokens(valid_ids[prefix_length]),
                    "last_target_is_eos": actual_compact_labels[last_supervised_compact] == eos_id,
                    "decoded": tokenizer.decode(valid_ids, skip_special_tokens=False),
                }
            )
        captures.append(
            {
                "input_shape": list(input_ids.shape),
                "compact_labels_shape": list(labels.shape),
                "logits_to_keep": logits_to_keep,
                "compact_start": compact_start,
                "token_type_unique": [int(value) for value in torch.unique(token_type_ids).tolist()],
                "rows": rows,
                "batch": {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                },
            }
        )

    return captures, hook


def inspect_optimizer(trainer: Any, model: torch.nn.Module) -> dict[str, Any]:
    if trainer.optimizer is None:
        raise RuntimeError("Trainer did not create an optimizer")
    optimizer_parameters = {
        id(parameter): parameter
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    }
    trainable_parameters = {id(parameter): parameter for parameter in model.parameters() if parameter.requires_grad}
    if set(optimizer_parameters) != set(trainable_parameters):
        raise RuntimeError(
            "Trainer optimizer parameter set differs from model trainables: "
            f"optimizer_only={len(set(optimizer_parameters) - set(trainable_parameters))} "
            f"trainable_only={len(set(trainable_parameters) - set(optimizer_parameters))}"
        )
    optimizer_numel = sum(parameter.numel() for parameter in optimizer_parameters.values())
    current_lrs = [float(group["lr"]) for group in trainer.optimizer.param_groups]
    if optimizer_numel != EXPECTED_LORA_PARAMETERS_RANK8:
        raise RuntimeError(
            f"Optimizer parameter count mismatch: expected={EXPECTED_LORA_PARAMETERS_RANK8} actual={optimizer_numel}"
        )
    if not current_lrs or any(not math.isfinite(value) or value <= 0 for value in current_lrs):
        raise RuntimeError(f"Trainer optimizer has invalid learning rates: {current_lrs}")
    if trainer.lr_scheduler is None:
        raise RuntimeError("Trainer did not create a learning-rate scheduler")
    return {
        "class": f"{trainer.optimizer.__class__.__module__}.{trainer.optimizer.__class__.__name__}",
        "parameter_tensor_count": len(optimizer_parameters),
        "parameter_count": optimizer_numel,
        "group_count": len(trainer.optimizer.param_groups),
        "learning_rates": current_lrs,
        "scheduler_class": f"{trainer.lr_scheduler.__class__.__module__}.{trainer.lr_scheduler.__class__.__name__}",
        "scheduler_last_epoch": int(trainer.lr_scheduler.last_epoch),
    }


def inspect_checkpoint(checkpoint_dir: Path) -> dict[str, Any]:
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Trainer checkpoint not found: {checkpoint_dir}")
    from safetensors import safe_open

    adapter_path = checkpoint_dir / "adapter_model.safetensors"
    adapter_config_path = checkpoint_dir / "adapter_config.json"
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    required = (
        adapter_path,
        adapter_config_path,
        trainer_state_path,
        checkpoint_dir / "optimizer.pt",
        checkpoint_dir / "scheduler.pt",
        checkpoint_dir / "rng_state.pth",
    )
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Trainer checkpoint is incomplete: missing={missing}")
    with safe_open(str(adapter_path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        adapter_dtype_counts: dict[str, int] = {}
        for key in keys:
            dtype = str(handle.get_tensor(key).dtype)
            adapter_dtype_counts[dtype] = adapter_dtype_counts.get(dtype, 0) + 1
    non_lora = [key for key in keys if "lora_" not in key]
    h_keys = [key for key in keys if ".H_module." in key]
    l_keys = [key for key in keys if ".L_module." in key]
    if len(keys) != EXPECTED_LORA_TENSORS or len(h_keys) != 256 or len(l_keys) != 256 or non_lora:
        raise RuntimeError(
            f"Unexpected Trainer adapter contents: total={len(keys)} H={len(h_keys)} L={len(l_keys)} "
            f"non_lora={non_lora[:20]}"
        )
    state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    if int(state.get("global_step", -1)) != 1:
        raise RuntimeError(f"Checkpoint trainer_state global_step is not one: {state.get('global_step')}")
    files = sorted(
        [
            {
                "relative_path": str(path.relative_to(checkpoint_dir)),
                "size_bytes": path.stat().st_size,
            }
            for path in checkpoint_dir.rglob("*")
            if path.is_file()
        ],
        key=lambda item: item["relative_path"],
    )
    return {
        "path": str(checkpoint_dir),
        "files": files,
        "adapter_tensor_count": len(keys),
        "H_adapter_tensor_count": len(h_keys),
        "L_adapter_tensor_count": len(l_keys),
        "non_lora_tensor_count": len(non_lora),
        "adapter_key_preview": keys[:20],
        "adapter_dtype_counts": adapter_dtype_counts,
        "trainer_state_global_step": int(state["global_step"]),
    }


def compare_adapter_states(left_model: torch.nn.Module, right_model: torch.nn.Module) -> dict[str, Any]:
    from peft import get_peft_model_state_dict

    left = get_peft_model_state_dict(left_model)
    right = get_peft_model_state_dict(right_model)
    if set(left) != set(right):
        raise RuntimeError("Trainer model and reloaded adapter keys differ")
    max_abs_diff = 0.0
    dtype_mismatches: list[dict[str, str]] = []
    for key in left:
        if left[key].dtype != right[key].dtype:
            dtype_mismatches.append(
                {"key": key, "trained": str(left[key].dtype), "reloaded": str(right[key].dtype)}
            )
        difference = float(
            (left[key].detach().float().cpu() - right[key].detach().float().cpu()).abs().max().item()
        )
        max_abs_diff = max(max_abs_diff, difference)
    if dtype_mismatches:
        raise RuntimeError(f"Trainer/reloaded adapter dtype mismatch: {dtype_mismatches[:20]}")
    if max_abs_diff != 0.0:
        raise RuntimeError(f"Trainer model and reloaded adapter tensors differ: max_abs_diff={max_abs_diff}")
    dtype_counts: dict[str, int] = {}
    for tensor in left.values():
        dtype = str(tensor.dtype)
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    return {
        "tensor_count": len(left),
        "max_abs_diff": max_abs_diff,
        "dtype_counts": dtype_counts,
        "dtype_mismatches": dtype_mismatches,
    }


def compare_frozen_base_parameters(
    trained_hrm_model: torch.nn.Module,
    fresh_hrm_model: torch.nn.Module,
) -> dict[str, Any]:
    def normalize_trained_name(name: str) -> str:
        return name.replace(".base_layer.", ".")

    trained = {
        normalize_trained_name(name): parameter
        for name, parameter in trained_hrm_model.named_parameters()
        if "lora_" not in name
    }
    fresh = dict(fresh_hrm_model.named_parameters())
    if set(trained) != set(fresh):
        raise RuntimeError(
            "Trained/fresh frozen-base parameter keys differ: "
            f"trained_only={sorted(set(trained) - set(fresh))[:20]} "
            f"fresh_only={sorted(set(fresh) - set(trained))[:20]}"
        )
    dtype_mismatches: list[dict[str, str]] = []
    value_mismatches: list[str] = []
    for name in sorted(trained):
        if trained[name].dtype != fresh[name].dtype:
            dtype_mismatches.append(
                {"name": name, "trained": str(trained[name].dtype), "fresh": str(fresh[name].dtype)}
            )
        if not torch.equal(trained[name].detach(), fresh[name].detach()):
            value_mismatches.append(name)
    if dtype_mismatches or value_mismatches:
        raise RuntimeError(
            "Frozen HRM base changed or reloaded differently: "
            f"dtype_mismatches={dtype_mismatches[:20]} value_mismatches={value_mismatches[:20]}"
        )
    return {
        "parameter_tensor_count": len(trained),
        "dtype_mismatches": dtype_mismatches,
        "value_mismatches": value_mismatches,
        "exact": True,
    }


def compare_model_buffers(
    trained_hrm_model: torch.nn.Module,
    fresh_hrm_model: torch.nn.Module,
) -> dict[str, Any]:
    trained = dict(trained_hrm_model.named_buffers())
    fresh = dict(fresh_hrm_model.named_buffers())
    if set(trained) != set(fresh):
        raise RuntimeError(
            "Trained/fresh HRM buffer keys differ: "
            f"trained_only={sorted(set(trained) - set(fresh))[:20]} "
            f"fresh_only={sorted(set(fresh) - set(trained))[:20]}"
        )
    mismatches: list[dict[str, Any]] = []
    for name in sorted(trained):
        left = trained[name].detach()
        right = fresh[name].detach()
        if left.shape != right.shape or left.dtype != right.dtype or not torch.equal(left, right):
            mismatches.append(
                {
                    "name": name,
                    "trained_shape": list(left.shape),
                    "fresh_shape": list(right.shape),
                    "trained_dtype": str(left.dtype),
                    "fresh_dtype": str(right.dtype),
                }
            )
    if mismatches:
        raise RuntimeError(f"Trained/fresh HRM buffers differ: {mismatches[:20]}")
    return {"tensor_count": len(trained), "mismatches": mismatches, "exact": True}


def compare_runtime_semantics(
    trained_model: torch.nn.Module,
    reloaded_model: torch.nn.Module,
    trained_hrm_model: torch.nn.Module,
    fresh_hrm_model: torch.nn.Module,
) -> dict[str, Any]:
    config_fields = (
        "H_cycles",
        "L_cycles",
        "L_bp_cycles",
        "num_layers_per_stack",
        "prefix_lm",
        "embedding_scale",
        "attention_dropout",
        "_attn_implementation",
    )

    def config_report(model: torch.nn.Module) -> dict[str, Any]:
        report: dict[str, Any] = {}
        for field in config_fields:
            value = getattr(model.config, field, None)
            if isinstance(value, tuple):
                value = list(value)
            report[field] = value
        return report

    def peft_report(model: torch.nn.Module) -> dict[str, Any]:
        configs: dict[str, Any] = {}
        for adapter_name, config in sorted(model.peft_config.items()):
            target_modules = getattr(config, "target_modules", None)
            configs[adapter_name] = {
                "r": int(config.r),
                "lora_alpha": int(config.lora_alpha),
                "lora_dropout": float(config.lora_dropout),
                "bias": str(config.bias),
                "task_type": str(config.task_type),
                "inference_mode": bool(config.inference_mode),
                "use_dora": bool(getattr(config, "use_dora", False)),
                "use_rslora": bool(getattr(config, "use_rslora", False)),
                "target_modules": sorted(target_modules) if target_modules is not None else None,
            }
        modules: dict[str, Any] = {}
        for name, module in model.named_modules():
            if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
                continue
            modules[name] = {
                "active_adapters": list(module.active_adapters),
                "disable_adapters": bool(module.disable_adapters),
                "merged_adapters": list(module.merged_adapters),
                "scaling": {key: float(value) for key, value in sorted(module.scaling.items())},
            }
        return {"configs": configs, "modules": modules}

    trained_config = config_report(trained_hrm_model)
    fresh_config = config_report(fresh_hrm_model)
    if trained_config != fresh_config:
        raise RuntimeError(f"Trainer/reloaded HRM runtime config differs: {trained_config} vs {fresh_config}")
    trained_peft = peft_report(trained_model)
    reloaded_peft = peft_report(reloaded_model)
    if trained_peft != reloaded_peft:
        raise RuntimeError("Trainer/reloaded PEFT runtime semantics differ")
    return {"hrm_config": trained_config, "peft": trained_peft, "exact": True}


def logits_difference_report(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if left.shape != right.shape:
        raise RuntimeError(f"Logits shapes differ: left={tuple(left.shape)} right={tuple(right.shape)}")
    if not bool(torch.isfinite(left).all().item()) or not bool(torch.isfinite(right).all().item()):
        raise RuntimeError("Non-finite logits encountered during checkpoint reload audit")
    left_float = left.float()
    right_float = right.float()
    difference = (left_float - right_float).abs()
    scale = max(float(left_float.abs().max().item()), float(right_float.abs().max().item()), 1.0)
    cosine_similarity = float(
        torch.nn.functional.cosine_similarity(
            left_float.reshape(1, -1),
            right_float.reshape(1, -1),
            dim=1,
        ).item()
    )
    return {
        "shape": list(left.shape),
        "max_abs_diff": float(difference.max().item()),
        "mean_abs_diff": float(difference.mean().item()),
        "rmse": float(difference.square().mean().sqrt().item()),
        "left_max_abs": float(left_float.abs().max().item()),
        "right_max_abs": float(right_float.abs().max().item()),
        "max_abs_diff_over_scale": float(difference.max().item()) / scale,
        "mean_abs_diff_over_scale": float(difference.mean().item()) / scale,
        "cosine_similarity": cosine_similarity,
        "top1_agreement": float((left.argmax(dim=-1) == right.argmax(dim=-1)).float().mean().item()),
        "exact": bool(torch.equal(left, right)),
    }


def validate_bfloat16_cross_instance(report: dict[str, Any], *, name: str) -> dict[str, Any]:
    epsilon = float(torch.finfo(torch.bfloat16).eps)
    # Exact state equality is audited separately. This bound only guards the
    # expected numerical drift from executing the same BF16 graph through two
    # independently allocated CUDA module instances. Recurrent HRM applies
    # many GEMMs, so allow four BF16 epsilon at the output scale while still
    # requiring every token prediction to agree.
    max_relative_bound = 4.0 * epsilon
    mean_relative_bound = epsilon
    minimum_cosine_similarity = 1.0 - epsilon
    accepted = (
        report["max_abs_diff_over_scale"] <= max_relative_bound
        and report["mean_abs_diff_over_scale"] <= mean_relative_bound
        and report["cosine_similarity"] >= minimum_cosine_similarity
        and report["top1_agreement"] == 1.0
    )
    validation = {
        "name": name,
        "bf16_epsilon": epsilon,
        "max_relative_bound": max_relative_bound,
        "mean_relative_bound": mean_relative_bound,
        "minimum_cosine_similarity": minimum_cosine_similarity,
        "required_top1_agreement": 1.0,
        "accepted": accepted,
    }
    if not accepted:
        raise RuntimeError(f"{name} exceeds the BF16 cross-instance equivalence envelope: {report}")
    return validation


def recurrent_forward_trace(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    hrm_model = find_unique_module(model, "HrmTextForCausalLM")
    trace: list[dict[str, Any]] = []
    handles = []

    def make_hook(stack_name: str):
        def hook(_module, _args, output):
            trace.append(
                {
                    "stack": stack_name,
                    "output": output.detach().float().cpu().clone(),
                }
            )

        return hook

    handles.append(hrm_model.model.L_module.register_forward_hook(make_hook("L")))
    handles.append(hrm_model.model.H_module.register_forward_hook(make_hook("H")))
    try:
        with torch.inference_mode():
            logits = model(**batch, use_cache=False).logits.detach().clone()
    finally:
        for handle in handles:
            handle.remove()
    return logits, trace


def compare_recurrent_traces(
    trained_trace: list[dict[str, Any]],
    reloaded_trace: list[dict[str, Any]],
) -> dict[str, Any]:
    trained_sequence = [item["stack"] for item in trained_trace]
    reloaded_sequence = [item["stack"] for item in reloaded_trace]
    if trained_sequence != reloaded_sequence:
        raise RuntimeError(
            f"Trainer/reloaded recurrence sequences differ: {trained_sequence} vs {reloaded_sequence}"
        )
    calls: list[dict[str, Any]] = []
    first_mismatch = None
    for index, (trained_item, reloaded_item) in enumerate(zip(trained_trace, reloaded_trace)):
        report = logits_difference_report(trained_item["output"], reloaded_item["output"])
        call = {"index": index, "stack": trained_item["stack"], **report}
        calls.append(call)
        if first_mismatch is None and report["max_abs_diff"] > 1e-5:
            first_mismatch = call
    return {
        "sequence": trained_sequence,
        "calls": calls,
        "first_mismatch": first_mismatch,
        "exact": first_mismatch is None,
    }


def main() -> None:
    args = parse_args()
    expected_versions = {"ms-swift": "4.4.2", "transformers": "5.9.0", "peft": "0.18.1"}
    mismatches = {name: {"expected": expected, "actual": version(name)} for name, expected in expected_versions.items() if version(name) != expected}
    if mismatches:
        raise RuntimeError(f"Unexpected environment versions: {mismatches}")
    if not torch.cuda.is_available():
        raise RuntimeError("HRM Swift Trainer smoke requires CUDA")

    model_path = args.model_path.resolve()
    plugin_path = args.plugin_path.resolve()
    dataset_path = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    output_report = args.output_report.resolve()
    for path, description in ((model_path, "model"), (dataset_path, "dataset")):
        if not path.exists():
            raise FileNotFoundError(f"Missing {description} path: {path}")
    if not plugin_path.is_file():
        raise FileNotFoundError(f"Plugin not found: {plugin_path}")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists; refusing to overwrite: {output_dir}")

    from swift.pipelines.train.sft import SwiftSft
    from swift.tuners import Swift

    argv = [
        "--model", str(model_path),
        "--model_type", MODEL_TYPE,
        "--template", TEMPLATE_TYPE,
        "--external_plugins", str(plugin_path),
        "--dataset", str(dataset_path),
        "--split_dataset_ratio", "0",
        "--max_length", "128",
        "--output_dir", str(output_dir),
        "--tuner_type", "lora",
        "--tuner_backend", "peft",
        "--target_modules", "all-linear",
        "--lora_rank", "8",
        "--lora_alpha", "16",
        "--lora_dropout", "0",
        "--learning_rate", "1e-4",
        "--lr_scheduler_type", "constant",
        "--warmup_ratio", "0",
        "--max_steps", "1",
        "--per_device_train_batch_size", "2",
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
            hrm_model = find_unique_module(model, "HrmTextForCausalLM")
            tokenizer = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
            hrm_model.config.use_cache = False
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()

            parameter_report_before = count_parameters(model)
            expected_total = EXPECTED_BASE_PARAMETERS + EXPECTED_LORA_PARAMETERS_RANK8
            if parameter_report_before["total"] != expected_total:
                raise RuntimeError(
                    f"Unexpected PEFT model parameter count: expected={expected_total} "
                    f"actual={parameter_report_before['total']}"
                )
            invalid_trainables = [
                name for name in parameter_report_before["trainable_names"] if "lora_" not in name
            ]
            if invalid_trainables:
                raise RuntimeError(f"Swift Trainer left non-LoRA parameters trainable: {invalid_trainables[:40]}")
            if parameter_report_before["trainable"] != EXPECTED_LORA_PARAMETERS_RANK8:
                raise RuntimeError(
                    f"Unexpected Swift Trainer trainable count: expected={EXPECTED_LORA_PARAMETERS_RANK8} "
                    f"actual={parameter_report_before['trainable']}"
                )
            if (
                int(hrm_model.config.H_cycles) != 2
                or int(hrm_model.config.L_cycles) != 3
                or [int(value) for value in hrm_model.config.L_bp_cycles] != [0, 3]
            ):
                raise RuntimeError(
                    "Swift Trainer changed the official static-K=5 recurrence config: "
                    f"H={hrm_model.config.H_cycles} L={hrm_model.config.L_cycles} "
                    f"L_bp_cycles={hrm_model.config.L_bp_cycles}"
                )
            lora_report = lora_module_report(model)
            probes = select_update_probes(model)
            captures, forward_hook = capture_training_forward(tokenizer)
            # Trainer calls the outer PEFT model. PEFT may delegate to the
            # wrapped causal LM via ``base_model.forward(...)`` rather than
            # ``base_model(...)``; the former bypasses hooks registered on the
            # inner HrmTextForCausalLM. Attach the audit to the exact outer
            # module object that Trainer invokes so the real collated kwargs
            # are observed before PEFT forwards them to HRM.
            hook_handle = model.register_forward_pre_hook(forward_hook, with_kwargs=True)

            print("========== SWIFT TRAINER PRE-TRAIN AUDIT ==========", flush=True)
            print(f"[trainer] type={trainer.__class__.__module__}.{trainer.__class__.__name__}", flush=True)
            print(f"[trainer] output_dir={trainer.args.output_dir}", flush=True)
            print(
                f"[trainables] total={parameter_report_before['total']} "
                f"trainable={parameter_report_before['trainable']} frozen={parameter_report_before['frozen']} "
                f"dtypes={parameter_report_before['trainable_dtypes']}",
                flush=True,
            )
            print(f"[lora] {json.dumps(lora_report, ensure_ascii=False)}", flush=True)

            started = time.perf_counter()
            try:
                train_result = super().train(trainer)
            finally:
                hook_handle.remove()
            torch.cuda.synchronize()
            elapsed_seconds = time.perf_counter() - started

            if int(trainer.state.global_step) != 1:
                raise RuntimeError(f"Swift Trainer global_step mismatch: {trainer.state.global_step}")
            if len(captures) != 1:
                raise RuntimeError(f"Expected one captured Trainer forward batch, got {len(captures)}")
            optimizer_report = inspect_optimizer(trainer, model)

            update_report: dict[str, Any] = {}
            for stack_name in ("H", "L"):
                parameter = probes[stack_name]["parameter"]
                max_abs_change = float(
                    (parameter.detach().float().cpu() - probes[stack_name]["before"]).abs().max().item()
                )
                if max_abs_change <= 0:
                    raise RuntimeError(f"Swift Trainer did not update the {stack_name}-stack LoRA probe")
                update_report[stack_name] = {
                    "name": probes[stack_name]["name"],
                    "max_abs_change": max_abs_change,
                }

            checkpoint_dir = Path(trainer.args.output_dir) / "checkpoint-1"
            checkpoint_report = inspect_checkpoint(checkpoint_dir)

            captured_batch = {
                key: value.to(next(model.parameters()).device)
                for key, value in captures[0]["batch"].items()
            }
            model.eval()
            with torch.inference_mode():
                reference_logits = model(**captured_batch, use_cache=False).logits
                reference_repeat_logits = model(**captured_batch, use_cache=False).logits

            try:
                from swift import get_model_processor
            except ImportError:
                from swift.model import get_model_processor
            fresh_base, _ = get_model_processor(
                str(model_path),
                model_type=MODEL_TYPE,
                torch_dtype=torch.bfloat16,
                device_map={"": str(next(model.parameters()).device)},
                load_model=True,
                download_model=False,
                attn_impl="sdpa",
                model_kwargs={"local_files_only": True, "low_cpu_mem_usage": True},
            )
            if fresh_base is None:
                raise RuntimeError("Fresh Swift model load returned None during checkpoint restore")
            fresh_base.config.use_cache = False
            fresh_base.eval()
            base_state_report = compare_frozen_base_parameters(hrm_model, fresh_base)
            buffer_report = compare_model_buffers(hrm_model, fresh_base)
            # Match ms-swift's own generic-LoRA resume path exactly. The model
            # was constructed and saved through Swift.prepare_model/Trainer,
            # so restoring it through bare PEFT is not the framework contract
            # this smoke is intended to validate.
            reloaded_model = Swift.from_pretrained(
                fresh_base,
                str(checkpoint_dir),
                # Match TunerMixin's actual Trainer-resume contract. This also
                # preserves the FP32 adapter compute policy used by the model
                # that just completed the optimizer step.
                is_trainable=True,
            ).eval()
            state_report = compare_adapter_states(model, reloaded_model)
            runtime_report = compare_runtime_semantics(model, reloaded_model, hrm_model, fresh_base)
            with torch.inference_mode():
                reloaded_logits = reloaded_model(**captured_batch, use_cache=False).logits
                reloaded_repeat_logits = reloaded_model(**captured_batch, use_cache=False).logits
            default_sdpa_report = {
                "trained_self_repeat": logits_difference_report(reference_logits, reference_repeat_logits),
                "reloaded_self_repeat": logits_difference_report(reloaded_logits, reloaded_repeat_logits),
                "cross_instance": logits_difference_report(reference_logits, reloaded_logits),
            }

            # Pin attention to the math backend to separate SDPA backend choice
            # from the remaining cross-instance BF16 GEMM effects. Exact
            # checkpoint identity is already established above from parameters,
            # buffers, adapter state, and runtime semantics.
            from torch.nn.attention import SDPBackend, sdpa_kernel

            with sdpa_kernel([SDPBackend.MATH]):
                reference_math_logits, reference_trace = recurrent_forward_trace(model, captured_batch)
                reloaded_math_logits, reloaded_trace = recurrent_forward_trace(reloaded_model, captured_batch)
            math_sdpa_report = logits_difference_report(reference_math_logits, reloaded_math_logits)
            recurrence_reload_report = compare_recurrent_traces(reference_trace, reloaded_trace)
            if not default_sdpa_report["trained_self_repeat"]["exact"]:
                raise RuntimeError("Trained model is not self-deterministic under default SDPA")
            if not default_sdpa_report["reloaded_self_repeat"]["exact"]:
                raise RuntimeError("Reloaded model is not self-deterministic under default SDPA")
            default_validation = validate_bfloat16_cross_instance(
                default_sdpa_report["cross_instance"],
                name="default SDPA checkpoint reload",
            )
            math_validation = validate_bfloat16_cross_instance(
                math_sdpa_report,
                name="math SDPA checkpoint reload",
            )
            print(
                f"[checkpoint-reload-runtime] frozen_base_exact={base_state_report['exact']} "
                f"buffers_exact={buffer_report['exact']} runtime_exact={runtime_report['exact']}",
                flush=True,
            )
            print(
                f"[checkpoint-reload-default-sdpa] {json.dumps(default_sdpa_report, ensure_ascii=False)}",
                flush=True,
            )
            print(
                f"[checkpoint-reload-math-sdpa] logits={json.dumps(math_sdpa_report, ensure_ascii=False)} "
                f"recurrence_exact={recurrence_reload_report['exact']} "
                f"first_mismatch={json.dumps(recurrence_reload_report['first_mismatch'], ensure_ascii=False)}",
                flush=True,
            )
            log_history = trainer.state.log_history
            finite_losses = [
                float(item["loss"])
                for item in log_history
                if "loss" in item and math.isfinite(float(item["loss"]))
            ]
            if not finite_losses:
                raise RuntimeError(f"Trainer log history contains no finite loss: {log_history}")

            device_index = torch.cuda.current_device()
            report = {
                "status": "ok",
                "packages": {
                    "ms-swift": version("ms-swift"),
                    "transformers": version("transformers"),
                    "peft": version("peft"),
                    "torch": version("torch"),
                },
                "model_path": str(model_path),
                "plugin_path": str(plugin_path),
                "dataset": str(dataset_path),
                "requested_output_dir": str(output_dir),
                "trainer_output_dir": str(trainer.args.output_dir),
                "argv": argv,
                "trainer": {
                    "class": f"{trainer.__class__.__module__}.{trainer.__class__.__name__}",
                    "global_step": int(trainer.state.global_step),
                    "elapsed_seconds": elapsed_seconds,
                    "log_history": log_history,
                    "finite_losses": finite_losses,
                    "train_result_type": f"{type(train_result).__module__}.{type(train_result).__name__}",
                    "train_result_repr": repr(train_result)[:4000],
                },
                "parameters_before": parameter_report_before,
                "lora": lora_report,
                "captured_forward": {
                    key: value for key, value in captures[0].items() if key != "batch"
                },
                "optimizer": optimizer_report,
                "updates": update_report,
                "checkpoint": checkpoint_report,
                "checkpoint_reload": {
                    **state_report,
                    "frozen_base": base_state_report,
                    "buffers": buffer_report,
                    "runtime_semantics": runtime_report,
                    "default_sdpa": default_sdpa_report,
                    "default_sdpa_validation": default_validation,
                    "controlled_math_sdpa": math_sdpa_report,
                    "controlled_math_sdpa_validation": math_validation,
                    "controlled_math_recurrence": recurrence_reload_report,
                },
                "cuda_memory": {
                    "device_index": device_index,
                    "device_name": torch.cuda.get_device_name(device_index),
                    "allocated_gib": torch.cuda.memory_allocated(device_index) / (1024**3),
                    "reserved_gib": torch.cuda.memory_reserved(device_index) / (1024**3),
                    "peak_allocated_gib": torch.cuda.max_memory_allocated(device_index) / (1024**3),
                    "peak_reserved_gib": torch.cuda.max_memory_reserved(device_index) / (1024**3),
                },
            }
            atomic_write_json(output_report, report)
            print("========== SWIFT TRAINER POST-TRAIN AUDIT ==========", flush=True)
            print(f"[trainer] global_step={trainer.state.global_step} finite_losses={finite_losses}", flush=True)
            print(f"[optimizer] {json.dumps(optimizer_report, ensure_ascii=False)}", flush=True)
            print(f"[updates] {json.dumps(update_report, ensure_ascii=False)}", flush=True)
            print(
                f"[checkpoint] path={checkpoint_dir} tensors={checkpoint_report['adapter_tensor_count']} "
                f"H={checkpoint_report['H_adapter_tensor_count']} L={checkpoint_report['L_adapter_tensor_count']} "
                f"dtypes={checkpoint_report['adapter_dtype_counts']}",
                flush=True,
            )
            print(
                f"[checkpoint-reload] tensors={state_report['tensor_count']} "
                f"dtypes={state_report['dtype_counts']} state_max_abs_diff={state_report['max_abs_diff']} "
                f"frozen_base_exact={base_state_report['exact']} buffers_exact={buffer_report['exact']} "
                f"runtime_exact={runtime_report['exact']}",
                flush=True,
            )
            print(
                f"[checkpoint-reload-bf16-equivalence] "
                f"default_sdpa_accepted={default_validation['accepted']} "
                f"math_sdpa_accepted={math_validation['accepted']}",
                flush=True,
            )
            print(f"[checkpoint-reload-recurrence] sequence={recurrence_reload_report['sequence']}", flush=True)
            print(f"[memory] {json.dumps(report['cuda_memory'], ensure_ascii=False)}", flush=True)
            print(f"[result] status=OK output_report={output_report}", flush=True)
            return train_result

    print("========== HRM SWIFT TRAINER ONE-STEP SMOKE ==========", flush=True)
    print(f"[python] version={sys.version.split()[0]} executable={sys.executable}", flush=True)
    print(f"[model-path] {model_path}", flush=True)
    print(f"[plugin-path] {plugin_path}", flush=True)
    print(f"[dataset] {dataset_path}", flush=True)
    print(f"[output-dir] {output_dir}", flush=True)
    print(f"[output-report] {output_report}", flush=True)
    print("[policy] text-only tuner_type=lora target_modules=all-linear HRM-base=frozen", flush=True)
    print("[argv] " + " ".join(argv), flush=True)
    AuditedSwiftSft(argv).main()


if __name__ == "__main__":
    main()
