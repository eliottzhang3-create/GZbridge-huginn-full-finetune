#!/usr/bin/env python3
"""Audit HRM-Text Swift training semantics, recurrence gradients, and LoRA save/reload."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
import time
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import torch.nn.functional as F


DEFAULT_MODEL_PATH = "/hpc_stor03/sjtu_home/jinwei.zhang/models/HRM-text"
MODEL_TYPE = "hrm_text_native"
TEMPLATE_TYPE = "hrm_text_synth_cot"
EXPECTED_PARAMETER_COUNT = 1_182_795_264
EXPECTED_PROJECTION_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "self_attn.gate_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
UPSTREAM_SOURCES = {
    "paper": "https://arxiv.org/pdf/2605.20613",
    "bp_warmup_model": (
        "https://github.com/sapientinc/HRM-Text/blob/main/"
        "models/baselines/hrm_nocarry_bp_warmup.py"
    ),
    "pretrain_arch_config": "https://github.com/sapientinc/HRM-Text/blob/main/config/arch/net/hrm.yaml",
    "sft_config": "https://github.com/sapientinc/HRM-Text/blob/main/config/cfg_sft.yaml",
    "transformers_model": (
        "https://github.com/huggingface/transformers/blob/v5.9.0/"
        "src/transformers/models/hrm_text/modeling_hrm_text.py"
    ),
    "transformers_config": (
        "https://github.com/huggingface/transformers/blob/v5.9.0/"
        "src/transformers/models/hrm_text/configuration_hrm_text.py"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path(DEFAULT_MODEL_PATH))
    parser.add_argument("--plugin-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--adapter-output-dir", type=Path, required=True)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def import_plugin(path: Path) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(f"Plugin not found: {path}")
    module_name = "hrm_text_swift_training_audit"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def tokenizer_from_processor(processor: Any):
    return processor.tokenizer if hasattr(processor, "tokenizer") else processor


def as_int_list(value: Any, *, name: str) -> list[int]:
    if torch.is_tensor(value):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be list/tuple/tensor, got {type(value)}")
    return [int(item) for item in value]


def count_parameters(model: torch.nn.Module) -> tuple[int, int, Counter[str], Counter[str]]:
    total = 0
    trainable = 0
    dtypes: Counter[str] = Counter()
    devices: Counter[str] = Counter()
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        trainable += count if parameter.requires_grad else 0
        dtypes[str(parameter.dtype)] += count
        devices[str(parameter.device)] += count
    return total, trainable, dtypes, devices


def load_swift_model(get_model_processor, model_path: Path, device: torch.device):
    model, processor = get_model_processor(
        str(model_path),
        model_type=MODEL_TYPE,
        torch_dtype=torch.bfloat16,
        device_map={"": str(device)},
        load_model=True,
        download_model=False,
        attn_impl="sdpa",
        model_kwargs={"local_files_only": True, "low_cpu_mem_usage": True},
    )
    if model is None:
        raise RuntimeError("Swift get_model_processor(load_model=True) returned model=None")
    if model.__class__.__name__ != "HrmTextForCausalLM":
        raise RuntimeError(f"Unexpected model class: {model.__class__.__module__}.{model.__class__.__name__}")
    if not bool(getattr(model.config, "prefix_lm", False)):
        raise RuntimeError("Swift-loaded HRM config does not enable prefix_lm")
    if getattr(model.config, "_attn_implementation", None) != "sdpa":
        raise RuntimeError(f"Expected SDPA, got {getattr(model.config, '_attn_implementation', None)!r}")
    return model, processor


def build_training_batch(get_template, processor: Any, device: torch.device):
    template = get_template(
        template_type=TEMPLATE_TYPE,
        processor=processor,
        max_length=512,
        use_chat_template=True,
        padding_side="right",
        padding_free=False,
        template_backend="swift",
    )
    template.set_mode("train")
    samples = [
        {
            "messages": [
                {"role": "user", "content": "What is 1 + 1?"},
                {"role": "assistant", "content": "2."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is 2 + 3?"},
                {"role": "assistant", "content": "2 + 3 = 5."},
            ]
        },
    ]
    encoded_samples = [template.encode(sample) for sample in samples]
    batch = template.data_collator(encoded_samples)
    required = {"input_ids", "attention_mask", "labels", "token_type_ids"}
    missing = sorted(required - set(batch))
    if missing:
        raise RuntimeError(f"Swift collator dropped required HRM fields: {missing}; keys={sorted(batch)}")
    model_batch = {key: batch[key].to(device) for key in sorted(required)}
    return template, samples, encoded_samples, batch, model_batch


def audit_next_token_labels(
    *,
    tokenizer: Any,
    encoded_samples: list[dict[str, Any]],
    collated_batch: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    im_end_id = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    eos_id = int(tokenizer.eos_token_id)
    for row_index, encoded in enumerate(encoded_samples):
        input_ids = as_int_list(encoded["input_ids"], name="input_ids")
        labels = as_int_list(encoded["labels"], name="labels")
        token_type_ids = as_int_list(encoded["token_type_ids"], name="token_type_ids")
        if not (len(input_ids) == len(labels) == len(token_type_ids)):
            raise RuntimeError(f"Row {row_index} encoded lengths disagree")
        supervised_label_positions = [index for index, label in enumerate(labels) if label != -100]
        if not supervised_label_positions:
            raise RuntimeError(f"Row {row_index} has no supervised labels")
        first_label_position = supervised_label_positions[0]
        if supervised_label_positions != list(range(first_label_position, len(input_ids))):
            raise RuntimeError(
                f"Row {row_index} supervised labels must be one contiguous response+EOS suffix: "
                f"positions={supervised_label_positions}"
            )
        if labels[first_label_position:] != input_ids[first_label_position:]:
            raise RuntimeError(f"Row {row_index} response labels do not equal response input tokens")
        if input_ids[first_label_position - 1] != im_end_id:
            raise RuntimeError(
                f"Row {row_index} first response must be predicted from <|im_end|>: "
                f"previous_id={input_ids[first_label_position - 1]} im_end_id={im_end_id}"
            )
        if input_ids[-1] != eos_id or labels[-1] != eos_id:
            raise RuntimeError(f"Row {row_index} must end with supervised EOS={eos_id}")
        if set(token_type_ids[:first_label_position]) != {1}:
            raise RuntimeError(f"Row {row_index} prompt token_type_ids must be one")
        if set(token_type_ids[first_label_position:]) != {0}:
            raise RuntimeError(f"Row {row_index} response token_type_ids must be zero")

        # Causal-LM loss shifts labels left by one: logits[t] predicts labels[t+1].
        shifted_targets = labels[1:]
        supervised_prediction_positions = [
            prediction_index
            for prediction_index, target in enumerate(shifted_targets)
            if target != -100
        ]
        expected_prediction_positions = [position - 1 for position in supervised_label_positions]
        if supervised_prediction_positions != expected_prediction_positions:
            raise RuntimeError(
                f"Row {row_index} next-token target shift is wrong: "
                f"expected={expected_prediction_positions} actual={supervised_prediction_positions}"
            )
        first_prediction_position = supervised_prediction_positions[0]
        first_target_id = shifted_targets[first_prediction_position]
        if first_prediction_position != first_label_position - 1:
            raise RuntimeError(f"Row {row_index} first prediction position is not prefix_length-1")
        if first_target_id != input_ids[first_label_position]:
            raise RuntimeError(f"Row {row_index} first shifted target is not the first response token")

        valid_length = len(input_ids)
        padded_labels = collated_batch["labels"][row_index]
        padded_attention = collated_batch["attention_mask"][row_index]
        if int(padded_attention.sum().item()) != valid_length:
            raise RuntimeError(f"Row {row_index} attention-mask valid length mismatch")
        if valid_length < padded_labels.shape[0]:
            if not bool((padded_labels[valid_length:] == -100).all().item()):
                raise RuntimeError(f"Row {row_index} padding labels are not -100")

        reports.append(
            {
                "row": row_index,
                "sequence_length": len(input_ids),
                "prefix_length": first_label_position,
                "supervised_label_positions": supervised_label_positions,
                "supervised_prediction_positions_after_shift": supervised_prediction_positions,
                "first_prediction_position": first_prediction_position,
                "first_prediction_input_token": tokenizer.convert_ids_to_tokens(input_ids[first_prediction_position]),
                "first_target_id": first_target_id,
                "first_target_token": tokenizer.convert_ids_to_tokens(first_target_id),
                "last_target_is_eos": shifted_targets[-1] == eos_id,
                "decoded_input": tokenizer.decode(input_ids, skip_special_tokens=False),
            }
        )
    return reports


def discover_lora_targets(model: torch.nn.Module) -> list[str]:
    targets: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        in_hrm_stack = name.startswith("model.L_module.layers.") or name.startswith("model.H_module.layers.")
        if in_hrm_stack and name.endswith(EXPECTED_PROJECTION_SUFFIXES):
            targets.append(name)
    targets.sort()
    expected = int(model.config.num_layers_per_stack) * 2 * len(EXPECTED_PROJECTION_SUFFIXES)
    if len(targets) != expected:
        raise RuntimeError(f"Unexpected HRM LoRA target count: expected={expected} actual={len(targets)}")
    return targets


def count_injected_lora_modules(model: torch.nn.Module) -> int:
    return sum(1 for module in model.modules() if hasattr(module, "lora_A") and hasattr(module, "lora_B"))


def expected_recurrence_trace(config: Any, padded_l_bp_cycles: list[int]) -> list[dict[str, Any]]:
    expected: list[dict[str, Any]] = []
    layers = int(config.num_layers_per_stack)
    for high_cycle in range(int(config.H_cycles)):
        num_grad_l = int(padded_l_bp_cycles[high_cycle])
        grad_threshold = int(config.L_cycles) - num_grad_l
        for low_cycle in range(int(config.L_cycles)):
            expected.append(
                {
                    "stack": "L",
                    "high_cycle": high_cycle,
                    "low_cycle": low_cycle,
                    "cycle_offset": (high_cycle * (int(config.L_cycles) + 1) + low_cycle) * layers,
                    "grad_enabled": low_cycle >= grad_threshold,
                }
            )
        expected.append(
            {
                "stack": "H",
                "high_cycle": high_cycle,
                "low_cycle": None,
                "cycle_offset": (high_cycle * (int(config.L_cycles) + 1) + int(config.L_cycles)) * layers,
                "grad_enabled": True,
            }
        )
    return expected


def upstream_k_policy(config: Any, k: int) -> dict[str, Any]:
    h_cycles = int(config.H_cycles)
    l_cycles = int(config.L_cycles)
    h_bp_steps = min(h_cycles, k - 1)
    l_bp_steps = k - h_bp_steps
    sequence_length = h_cycles * (l_cycles + 1)
    grad_indices: list[int] = []
    for high_cycle in range(h_cycles):
        for low_cycle in range(l_cycles):
            global_low_index = high_cycle * l_cycles + low_cycle
            if global_low_index >= h_cycles * l_cycles - l_bp_steps:
                grad_indices.append(high_cycle * (l_cycles + 1) + low_cycle)
        if high_cycle >= h_cycles - h_bp_steps:
            grad_indices.append(high_cycle * (l_cycles + 1) + l_cycles)
    return {
        "K": k,
        "H_bp_steps": h_bp_steps,
        "L_bp_steps": l_bp_steps,
        "sequence_length": sequence_length,
        "grad_stack_indices": grad_indices,
    }


def install_recurrence_hooks(core_model: torch.nn.Module):
    trace: list[dict[str, Any]] = []
    pending: dict[str, list[int]] = {"L": [], "H": []}
    layer_counts: dict[str, Counter[int]] = {"L": Counter(), "H": Counter()}
    handles: list[Any] = []

    def make_stack_pre(stack_name: str):
        def hook(_module, args, kwargs):
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            index = len(trace)
            trace.append(
                {
                    "index": index,
                    "stack": stack_name,
                    "cycle_offset": int(kwargs.get("cycle_offset", -1)),
                    "grad_enabled": bool(torch.is_grad_enabled()),
                    "input_requires_grad": bool(getattr(hidden_states, "requires_grad", False)),
                    "output_requires_grad": None,
                    "output_grad_norm": None,
                    "_output": None,
                }
            )
            pending[stack_name].append(index)
        return hook

    def make_stack_post(stack_name: str):
        def hook(_module, _args, _kwargs, output):
            index = pending[stack_name].pop()
            trace[index]["output_requires_grad"] = bool(output.requires_grad)
            if output.requires_grad:
                output.retain_grad()
                trace[index]["_output"] = output
        return hook

    def make_layer_pre(stack_name: str, layer_index: int):
        def hook(_module, _args, _kwargs):
            layer_counts[stack_name][layer_index] += 1
        return hook

    for stack_name, stack in (("L", core_model.L_module), ("H", core_model.H_module)):
        handles.append(stack.register_forward_pre_hook(make_stack_pre(stack_name), with_kwargs=True))
        handles.append(stack.register_forward_hook(make_stack_post(stack_name), with_kwargs=True))
        for layer_index, layer in enumerate(stack.layers):
            handles.append(layer.register_forward_pre_hook(make_layer_pre(stack_name, layer_index), with_kwargs=True))
    return trace, layer_counts, handles


def validate_recurrence_trace(
    *,
    core_model: torch.nn.Module,
    trace: list[dict[str, Any]],
    layer_counts: dict[str, Counter[int]],
) -> dict[str, Any]:
    config = core_model.config
    raw_l_bp = [int(value) for value in config.L_bp_cycles]
    padded_l_bp = [int(value) for value in core_model.L_bp_cycles_padded]
    expected = expected_recurrence_trace(config, padded_l_bp)
    if len(trace) != len(expected):
        raise RuntimeError(f"Unexpected recurrent stack count: expected={len(expected)} actual={len(trace)}")

    for index, (actual, wanted) in enumerate(zip(trace, expected)):
        for key in ("stack", "cycle_offset", "grad_enabled"):
            if actual[key] != wanted[key]:
                raise RuntimeError(
                    f"Recurrent trace mismatch at call={index} key={key}: "
                    f"expected={wanted[key]!r} actual={actual[key]!r}"
                )
        if actual["output_requires_grad"] != wanted["grad_enabled"]:
            raise RuntimeError(
                f"Recurrent output grad-state mismatch at call={index}: "
                f"expected={wanted['grad_enabled']} actual={actual['output_requires_grad']}"
            )

    layers_per_stack = int(config.num_layers_per_stack)
    expected_layer_counts = {
        "L": int(config.H_cycles) * int(config.L_cycles),
        "H": int(config.H_cycles),
    }
    for stack_name in ("L", "H"):
        if set(layer_counts[stack_name]) != set(range(layers_per_stack)):
            raise RuntimeError(f"{stack_name} stack did not execute every physical layer")
        wrong = {
            index: count
            for index, count in layer_counts[stack_name].items()
            if count != expected_layer_counts[stack_name]
        }
        if wrong:
            raise RuntimeError(f"{stack_name} per-layer invocation counts are wrong: {wrong}")

    grad_indices = [item["index"] for item in trace if item["grad_enabled"]]
    runtime_k = len(grad_indices)
    expected_last_indices = list(range(len(trace) - runtime_k, len(trace)))
    if grad_indices != expected_last_indices:
        raise RuntimeError(
            f"Runtime gradient-enabled calls are not one trailing K-step suffix: "
            f"indices={grad_indices} expected={expected_last_indices}"
        )
    if raw_l_bp != [0, 3] or padded_l_bp != [0, 3] or runtime_k != 5:
        raise RuntimeError(
            f"Official HRM-Text-1B checkpoint must encode final static K=5: "
            f"raw_L_bp_cycles={raw_l_bp} padded={padded_l_bp} runtime_k={runtime_k}"
        )

    forward_signature = str(inspect.signature(core_model.forward))
    forward_source = inspect.getsource(type(core_model).forward)
    dynamic_warmup_fields = [
        name
        for name in ("bp_steps", "bp_min_steps", "bp_max_steps", "bp_warmup_ratio")
        if hasattr(config, name) or hasattr(core_model, name)
    ]
    dynamic_warmup_preserved = bool(dynamic_warmup_fields) or "compute_train_extra_args" in forward_source
    if dynamic_warmup_preserved:
        raise RuntimeError(
            "Transformers 5.9 HRM unexpectedly exposes a dynamic BP warmup mechanism; "
            f"re-audit required: fields={dynamic_warmup_fields}"
        )

    k2 = upstream_k_policy(config, 2)
    k5 = upstream_k_policy(config, 5)
    if k2["grad_stack_indices"] != [6, 7] or k5["grad_stack_indices"] != [3, 4, 5, 6, 7]:
        raise RuntimeError(f"Unexpected upstream K mapping: K2={k2} K5={k5}")
    if grad_indices != k5["grad_stack_indices"]:
        raise RuntimeError(f"Runtime static policy does not match upstream final K=5: runtime={grad_indices} K5={k5}")

    return {
        "H_cycles": int(config.H_cycles),
        "L_cycles": int(config.L_cycles),
        "num_layers_per_stack": layers_per_stack,
        "physical_parameter_layers_H_plus_L": layers_per_stack * 2,
        "stack_invocations": len(trace),
        "decoder_layer_invocations": {
            "L_total": sum(layer_counts["L"].values()),
            "H_total": sum(layer_counts["H"].values()),
            "combined": sum(layer_counts["L"].values()) + sum(layer_counts["H"].values()),
        },
        "raw_L_bp_cycles": raw_l_bp,
        "padded_L_bp_cycles": padded_l_bp,
        "runtime_static_K": runtime_k,
        "runtime_grad_stack_indices": grad_indices,
        "upstream_initial_K2": k2,
        "upstream_final_K5": k5,
        "dynamic_K2_to_K5_warmup_preserved": dynamic_warmup_preserved,
        "current_policy_interpretation": "static final K=5; matches upstream SFT, not upstream pretraining warmup",
        "forward_signature": forward_signature,
        "trace": [
            {key: value for key, value in item.items() if key != "_output"}
            | {
                "high_cycle": expected[index]["high_cycle"],
                "low_cycle": expected[index]["low_cycle"],
            }
            for index, item in enumerate(trace)
        ],
        "per_layer_call_counts": {
            stack_name: {str(index): count for index, count in sorted(counts.items())}
            for stack_name, counts in layer_counts.items()
        },
    }


def gradient_report(model: torch.nn.Module) -> dict[str, Any]:
    report: dict[str, dict[str, Any]] = {}
    base_grad_names: list[str] = []
    for stack_name, marker in (("L", ".L_module."), ("H", ".H_module.")):
        parameter_count = 0
        grad_present = 0
        grad_nonzero = 0
        grad_norm_sum = 0.0
        examples: list[dict[str, Any]] = []
        for name, parameter in model.named_parameters():
            if marker not in name or "lora_" not in name:
                continue
            parameter_count += 1
            if parameter.grad is not None:
                grad_present += 1
                if not bool(torch.isfinite(parameter.grad).all().item()):
                    raise RuntimeError(f"Non-finite LoRA gradient: {name}")
                norm = float(parameter.grad.float().norm().item())
                grad_norm_sum += norm
                if norm > 0:
                    grad_nonzero += 1
                if len(examples) < 8:
                    examples.append({"name": name, "grad_norm": norm})
        if parameter_count == 0 or grad_present == 0 or grad_nonzero == 0:
            raise RuntimeError(
                f"{stack_name}-stack LoRA gradient coverage failed: "
                f"parameters={parameter_count} present={grad_present} nonzero={grad_nonzero}"
            )
        report[stack_name] = {
            "lora_parameter_tensors": parameter_count,
            "grad_present_tensors": grad_present,
            "grad_nonzero_tensors": grad_nonzero,
            "grad_norm_sum": grad_norm_sum,
            "examples": examples,
        }

    for name, parameter in model.named_parameters():
        if "lora_" not in name and parameter.grad is not None:
            base_grad_names.append(name)
    if base_grad_names:
        raise RuntimeError(f"Frozen base parameters unexpectedly received gradients: {base_grad_names[:20]}")
    report["base"] = {"unexpected_gradient_names": base_grad_names}
    return report


def select_update_parameters(model: torch.nn.Module) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for stack_name, marker in (("L", ".L_module."), ("H", ".H_module.")):
        candidates = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if marker in name
            and "lora_" in name
            and parameter.grad is not None
            and float(parameter.grad.float().norm().item()) > 0
        ]
        if not candidates:
            raise RuntimeError(f"No nonzero-gradient {stack_name}-stack LoRA parameter available for update audit")
        selected[stack_name] = candidates[0][1]
        selected[f"{stack_name}_name"] = candidates[0][0]
    return selected


def compare_adapter_states(left_model: torch.nn.Module, right_model: torch.nn.Module) -> dict[str, Any]:
    from peft import get_peft_model_state_dict

    left = get_peft_model_state_dict(left_model)
    right = get_peft_model_state_dict(right_model)
    if set(left) != set(right):
        raise RuntimeError(
            f"Saved/reloaded adapter key mismatch: left_only={sorted(set(left) - set(right))[:20]} "
            f"right_only={sorted(set(right) - set(left))[:20]}"
        )
    max_abs_diff = 0.0
    mismatched: list[str] = []
    for key in sorted(left):
        left_tensor = left[key].detach().float().cpu()
        right_tensor = right[key].detach().float().cpu()
        if left_tensor.shape != right_tensor.shape:
            raise RuntimeError(f"Adapter tensor shape mismatch for {key}: {left_tensor.shape} vs {right_tensor.shape}")
        difference = float((left_tensor - right_tensor).abs().max().item()) if left_tensor.numel() else 0.0
        max_abs_diff = max(max_abs_diff, difference)
        if difference != 0.0:
            mismatched.append(key)
    if mismatched:
        raise RuntimeError(f"Saved/reloaded adapter tensors differ: keys={mismatched[:20]} max_abs_diff={max_abs_diff}")
    return {"tensor_count": len(left), "max_abs_diff": max_abs_diff, "mismatched_keys": mismatched}


def main() -> None:
    args = parse_args()
    if version("ms-swift") != "4.4.2" or version("transformers") != "5.9.0" or version("peft") != "0.18.1":
        raise RuntimeError(
            f"Unexpected environment: ms-swift={version('ms-swift')} transformers={version('transformers')} "
            f"peft={version('peft')}"
        )
    if args.lora_rank <= 0 or args.lora_alpha <= 0 or args.learning_rate <= 0:
        raise ValueError("LoRA rank/alpha and learning rate must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("HRM Swift training audit requires CUDA")

    try:
        from swift import get_model_processor, get_template
    except ImportError:
        from swift.model import get_model_processor
        from swift.template import get_template
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    model_path = args.model_path.resolve()
    plugin_path = args.plugin_path.resolve()
    output_report = args.output_report.resolve()
    adapter_output_dir = args.adapter_output_dir.resolve()
    required_files = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
    missing_files = [name for name in required_files if not (model_path / name).is_file()]
    if missing_files:
        raise FileNotFoundError(f"Incomplete HRM-Text snapshot at {model_path}: missing={missing_files}")
    if adapter_output_dir.exists():
        raise FileExistsError(f"Adapter output directory already exists; refusing to overwrite: {adapter_output_dir}")

    plugin = import_plugin(plugin_path)
    if getattr(plugin, "MODEL_TYPE", None) != MODEL_TYPE:
        raise RuntimeError(f"Plugin model type mismatch: {getattr(plugin, 'MODEL_TYPE', None)!r}")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"This audit requires a CUDA device, got {device}")
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    properties = torch.cuda.get_device_properties(device_index)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)

    print("========== HRM SWIFT TRAINING AUDIT ==========", flush=True)
    print(f"[python] version={sys.version.split()[0]} executable={sys.executable}", flush=True)
    print(
        f"[packages] ms-swift={version('ms-swift')} transformers={version('transformers')} "
        f"peft={version('peft')} torch={version('torch')}",
        flush=True,
    )
    print(f"[model-path] {model_path}", flush=True)
    print(f"[plugin-path] {plugin_path}", flush=True)
    print(f"[output-report] {output_report}", flush=True)
    print(f"[adapter-output-dir] {adapter_output_dir}", flush=True)
    print(
        f"[cuda] device={device} index={device_index} name={properties.name!r} "
        f"total_gib={properties.total_memory / (1024**3):.3f}",
        flush=True,
    )

    load_started = time.perf_counter()
    base_model, processor = load_swift_model(get_model_processor, model_path, device)
    base_model.train()
    torch.cuda.synchronize(device_index)
    load_seconds = time.perf_counter() - load_started
    tokenizer = tokenizer_from_processor(processor)

    base_total, base_trainable, base_dtypes, base_devices = count_parameters(base_model)
    if base_total != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"Unexpected base parameter count: expected={EXPECTED_PARAMETER_COUNT} actual={base_total}")
    if set(base_dtypes) != {"torch.bfloat16"} or set(base_devices) != {str(device)}:
        raise RuntimeError(f"Unexpected base placement: dtypes={dict(base_dtypes)} devices={dict(base_devices)}")
    print(
        f"[base-model] parameters={base_total} initially_trainable={base_trainable} "
        f"load_seconds={load_seconds:.6f}",
        flush=True,
    )

    template, samples, encoded_samples, cpu_batch, model_batch = build_training_batch(
        get_template, processor, device
    )
    label_reports = audit_next_token_labels(
        tokenizer=tokenizer,
        encoded_samples=encoded_samples,
        collated_batch=cpu_batch,
    )
    print("========== NEXT-TOKEN LABEL SHIFT ==========" , flush=True)
    for item in label_reports:
        print(f"[label-shift] {json.dumps(item, ensure_ascii=False)}", flush=True)

    target_modules = discover_lora_targets(base_model)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules=target_modules,
        bias="none",
    )
    core_model = base_model.model
    model = get_peft_model(base_model, lora_config)
    model.train()
    injected_count = count_injected_lora_modules(model)
    if injected_count != len(target_modules):
        raise RuntimeError(f"LoRA injection count mismatch: targets={len(target_modules)} injected={injected_count}")
    total_params, trainable_params, dtype_counts, device_counts = count_parameters(model)
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    invalid_trainables = [name for name in trainable_names if "lora_" not in name]
    if invalid_trainables:
        raise RuntimeError(f"Non-LoRA parameters remain trainable: {invalid_trainables[:40]}")
    if not trainable_names:
        raise RuntimeError("LoRA wrapping produced no trainable parameters")
    print(
        f"[lora] targets={len(target_modules)} injected={injected_count} rank={args.lora_rank} "
        f"alpha={args.lora_alpha} trainable_parameters={trainable_params}",
        flush=True,
    )
    print(f"[lora] target_preview={target_modules[:16]}", flush=True)

    if bool(getattr(model, "is_gradient_checkpointing", False)) or bool(
        getattr(core_model, "gradient_checkpointing", False)
    ):
        raise RuntimeError("Gradient checkpointing must be disabled for an unambiguous recurrence trace")

    trace, layer_counts, hook_handles = install_recurrence_hooks(core_model)
    model.zero_grad(set_to_none=True)
    forward_started = time.perf_counter()
    try:
        outputs = model(**model_batch, use_cache=False)
    finally:
        for handle in hook_handles:
            handle.remove()
    torch.cuda.synchronize(device_index)
    forward_seconds = time.perf_counter() - forward_started
    if outputs.loss is None or not bool(torch.isfinite(outputs.loss).item()):
        raise RuntimeError(f"Model loss is invalid: {outputs.loss}")
    if not bool(torch.isfinite(outputs.logits).all().item()):
        raise RuntimeError("Training logits contain NaN or Inf")

    recurrence_report = validate_recurrence_trace(
        core_model=core_model,
        trace=trace,
        layer_counts=layer_counts,
    )
    print("========== RECURRENT EXECUTION ==========" , flush=True)
    for item in recurrence_report["trace"]:
        print(f"[recurrent-call] {json.dumps(item, ensure_ascii=False)}", flush=True)
    print(
        f"[bp-policy] runtime_static_K={recurrence_report['runtime_static_K']} "
        f"grad_indices={recurrence_report['runtime_grad_stack_indices']} "
        f"dynamic_K2_to_K5_warmup_preserved={recurrence_report['dynamic_K2_to_K5_warmup_preserved']}",
        flush=True,
    )

    labels = model_batch["labels"]
    manual_shifted_loss = F.cross_entropy(
        outputs.logits[:, :-1, :].float().contiguous().view(-1, outputs.logits.shape[-1]),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )
    loss_abs_diff = float((outputs.loss.float() - manual_shifted_loss).abs().item())
    if loss_abs_diff > 1e-5:
        raise RuntimeError(
            f"HF loss does not match manual next-token shifted CE: "
            f"model={outputs.loss.item()} manual={manual_shifted_loss.item()} diff={loss_abs_diff}"
        )
    print(
        f"[loss] model={outputs.loss.item():.9f} manual_shifted_ce={manual_shifted_loss.item():.9f} "
        f"abs_diff={loss_abs_diff}",
        flush=True,
    )
    model_loss_value = float(outputs.loss.item())
    manual_shifted_loss_value = float(manual_shifted_loss.item())

    backward_started = time.perf_counter()
    outputs.loss.backward()
    torch.cuda.synchronize(device_index)
    backward_seconds = time.perf_counter() - backward_started
    for item in trace:
        output = item.get("_output")
        if item["grad_enabled"]:
            if output is None or output.grad is None:
                raise RuntimeError(f"Grad-enabled recurrent output has no retained gradient: call={item['index']}")
            if not bool(torch.isfinite(output.grad).all().item()):
                raise RuntimeError(f"Non-finite recurrent output gradient: call={item['index']}")
            grad_norm = float(output.grad.float().norm().item())
            if grad_norm <= 0:
                raise RuntimeError(f"Zero recurrent output gradient: call={item['index']}")
            item["output_grad_norm"] = grad_norm
        elif output is not None:
            raise RuntimeError(f"No-grad recurrent call unexpectedly retained a graph output: call={item['index']}")
    for report_item, trace_item in zip(recurrence_report["trace"], trace):
        report_item["output_grad_norm"] = trace_item["output_grad_norm"]

    gradients = gradient_report(model)
    print(
        f"[gradients] L_nonzero={gradients['L']['grad_nonzero_tensors']} "
        f"H_nonzero={gradients['H']['grad_nonzero_tensors']} base_unexpected=0",
        flush=True,
    )

    selected = select_update_parameters(model)
    before_update = {
        stack_name: selected[stack_name].detach().clone()
        for stack_name in ("L", "H")
    }
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.0,
    )
    optimizer.step()
    update_report: dict[str, Any] = {}
    for stack_name in ("L", "H"):
        parameter = selected[stack_name]
        max_abs_change = float((parameter.detach().float() - before_update[stack_name].float()).abs().max().item())
        if max_abs_change <= 0:
            raise RuntimeError(f"{stack_name}-stack selected LoRA parameter did not update")
        update_report[stack_name] = {
            "name": selected[f"{stack_name}_name"],
            "max_abs_change": max_abs_change,
        }
    optimizer.zero_grad(set_to_none=True)
    print(f"[optimizer-step] {json.dumps(update_report, ensure_ascii=False)}", flush=True)

    # Drop retained recurrent activations and optimizer state before loading a
    # second independent base-model instance for the adapter restore audit.
    for item in trace:
        item["_output"] = None
    del outputs, manual_shifted_loss, optimizer, before_update
    torch.cuda.empty_cache()

    adapter_output_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_output_dir, safe_serialization=True)
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(adapter_output_dir)
    required_adapter_files = ("adapter_config.json", "adapter_model.safetensors")
    missing_adapter_files = [name for name in required_adapter_files if not (adapter_output_dir / name).is_file()]
    if missing_adapter_files:
        raise RuntimeError(f"Adapter save is incomplete: missing={missing_adapter_files}")
    print(f"[adapter-save] path={adapter_output_dir}", flush=True)

    model.eval()
    with torch.inference_mode():
        reference_logits = model(**{key: value for key, value in model_batch.items() if key != "labels"}, use_cache=False).logits

    fresh_base, _fresh_processor = load_swift_model(get_model_processor, model_path, device)
    reloaded_model = PeftModel.from_pretrained(fresh_base, adapter_output_dir, is_trainable=False).eval()
    adapter_state_report = compare_adapter_states(model, reloaded_model)
    with torch.inference_mode():
        reloaded_logits = reloaded_model(
            **{key: value for key, value in model_batch.items() if key != "labels"}, use_cache=False
        ).logits
    reload_logits_max_abs_diff = float((reference_logits.float() - reloaded_logits.float()).abs().max().item())
    if reload_logits_max_abs_diff > 1e-5:
        raise RuntimeError(f"Saved/reloaded adapter logits differ: max_abs_diff={reload_logits_max_abs_diff}")
    print(
        f"[adapter-reload] tensors={adapter_state_report['tensor_count']} "
        f"state_max_abs_diff={adapter_state_report['max_abs_diff']} "
        f"logits_max_abs_diff={reload_logits_max_abs_diff}",
        flush=True,
    )

    report = {
        "status": "ok",
        "packages": {
            "ms-swift": version("ms-swift"),
            "transformers": version("transformers"),
            "peft": version("peft"),
            "torch": version("torch"),
        },
        "sources": UPSTREAM_SOURCES,
        "model_path": str(model_path),
        "plugin_path": str(plugin_path),
        "adapter_output_dir": str(adapter_output_dir),
        "model": {
            "base_parameter_count": base_total,
            "peft_total_parameter_count": total_params,
            "trainable_parameter_count": trainable_params,
            "dtype_counts": dict(dtype_counts),
            "device_counts": dict(device_counts),
            "load_seconds": load_seconds,
        },
        "batch": {
            "samples": samples,
            "shapes": {key: list(value.shape) for key, value in cpu_batch.items() if torch.is_tensor(value)},
            "next_token_label_audit": label_reports,
        },
        "lora": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": 0.0,
            "target_count": len(target_modules),
            "target_modules": target_modules,
            "injected_module_count": injected_count,
            "trainable_tensor_count": len(trainable_names),
            "trainable_parameter_count": trainable_params,
        },
        "loss": {
            "model_loss": model_loss_value,
            "manual_shifted_cross_entropy": manual_shifted_loss_value,
            "absolute_difference": loss_abs_diff,
        },
        "recurrence": recurrence_report,
        "gradients": gradients,
        "optimizer_update": update_report,
        "adapter_reload": {
            **adapter_state_report,
            "logits_max_abs_diff": reload_logits_max_abs_diff,
        },
        "timing": {
            "forward_seconds": forward_seconds,
            "backward_seconds": backward_seconds,
        },
        "cuda_memory": {
            "device_index": device_index,
            "device_name": properties.name,
            "allocated_gib": torch.cuda.memory_allocated(device_index) / (1024**3),
            "reserved_gib": torch.cuda.memory_reserved(device_index) / (1024**3),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device_index) / (1024**3),
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device_index) / (1024**3),
        },
    }
    atomic_write_json(output_report, report)
    print(f"[memory] {json.dumps(report['cuda_memory'], ensure_ascii=False)}", flush=True)
    print(f"[result] status=OK output_report={output_report}", flush=True)


if __name__ == "__main__":
    main()
