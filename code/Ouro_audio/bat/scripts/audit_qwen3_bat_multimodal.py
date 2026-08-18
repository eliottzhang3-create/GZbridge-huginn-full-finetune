#!/usr/bin/env python3
"""Audit one real BAT audio sample through Qwen3 multimodal forward/backward."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


os.environ.setdefault("BAT_AUDIO_AUDIT", "1")
MODEL_TYPE = "qwen3_bat_spatial_ast"
TEMPLATE_TYPE = "qwen3_bat_audio_prefix"
EXPECTED_AUDIO_TOKENS = 64
EXPECTED_SEQUENCE_LENGTH = 176
EXPECTED_HIDDEN_SIZE = 2560
EXPECTED_VOCAB_SIZE = 151936


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--plugin-path", type=Path, required=True)
    parser.add_argument("--qa-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def import_plugin(path: Path):
    spec = importlib.util.spec_from_file_location(
        "qwen3_bat_spatial_ast_audit_plugin", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import plugin: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise TypeError(f"Expected list data in {path}")
    return records


def find_record(qa_root: Path) -> dict[str, Any]:
    records = load_records(qa_root / "stage1-clsdoa" / "train.json")
    for record in records:
        if str(record.get("question_type", "")).upper() == "CLASSIFICATION":
            return record
    raise LookupError("No Stage-I CLASSIFICATION record found")


def as_long_batch(value: Any, device: torch.device) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.tensor(value, dtype=torch.long)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.to(device=device, dtype=torch.long)


def as_attention_batch(value: Any, fallback: torch.Tensor) -> torch.Tensor:
    if value is None:
        return torch.ones_like(fallback)
    tensor = value if torch.is_tensor(value) else torch.tensor(value, dtype=torch.long)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.to(device=fallback.device, dtype=torch.long)


def shape_tuple(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", value)
    if shape is None:
        raise TypeError(f"Cannot read shape from {type(value).__name__}")
    return tuple(int(item) for item in shape)


def parameter_groups(model: torch.nn.Module) -> dict[str, int]:
    groups = {"qformer": 0, "spatial_ast": 0, "qwen_native": 0, "other": 0}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("audio_qformer."):
            groups["qformer"] += parameter.numel()
        elif name.startswith("spatial_ast_encoder."):
            groups["spatial_ast"] += parameter.numel()
        elif name.startswith(("model.", "lm_head.")):
            groups["qwen_native"] += parameter.numel()
        else:
            groups["other"] += parameter.numel()
    return groups


def main() -> None:
    args = parse_args()
    if not args.output.is_absolute():
        raise ValueError(f"Output must be an absolute private path: {args.output}")
    if str(args.output).replace("\\", "/").startswith("/hpc_stor03/public"):
        raise ValueError(f"Refusing public output path: {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("This audit requires a submitted CUDA job")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"This audit is GPU-only, got {device}")
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    torch.cuda.set_device(device_index)

    if version("ms-swift") != "4.4.2":
        raise RuntimeError(f"Expected ms-swift==4.4.2, got {version('ms-swift')}")
    if version("transformers") != "4.54.1":
        raise RuntimeError(
            f"Expected transformers==4.54.1, got {version('transformers')}"
        )
    if version("peft") != "0.18.1":
        raise RuntimeError(f"Expected peft==0.18.1, got {version('peft')}")

    plugin = import_plugin(args.plugin_path.resolve())
    if plugin.MODEL_TYPE != MODEL_TYPE or plugin.TEMPLATE_TYPE != TEMPLATE_TYPE:
        raise RuntimeError("Qwen3 BAT plugin registration constants do not match audit")

    try:
        from swift import get_model_processor, get_template
    except ImportError:
        from swift.model import get_model_processor
        from swift.template import get_template

    print("========== QWEN3 BAT MULTIMODAL FORWARD/BACKWARD AUDIT ==========")
    print(f"[packages] ms-swift={version('ms-swift')} transformers={version('transformers')} torch={torch.__version__}")
    print(f"[model] {args.model_path}")
    print(f"[plugin] {args.plugin_path}")
    print(f"[device] {device} name={torch.cuda.get_device_name(device_index)}")

    model, processor = get_model_processor(
        str(args.model_path.resolve()),
        model_type=MODEL_TYPE,
        torch_dtype=torch.bfloat16,
        device_map={"": str(device)},
        load_model=True,
        download_model=False,
        attn_impl="sdpa",
        model_kwargs={"local_files_only": True, "low_cpu_mem_usage": True},
    )
    model.train()
    if model.__class__.__name__ != "Qwen3ForCausalLM":
        raise RuntimeError(
            f"Unexpected model class: {model.__class__.__module__}.{model.__class__.__name__}"
        )

    contract = getattr(model, "_qwen3_bat_audio_contract", None)
    if not isinstance(contract, dict):
        raise RuntimeError("Qwen3 BAT loader did not attach its audio contract")
    contract_checks = {
        "audio_token_count": contract.get("audio_token_count") == EXPECTED_AUDIO_TOKENS,
        "qwen3_hidden_size": contract.get("qwen3_hidden_size") == EXPECTED_HIDDEN_SIZE,
        "qformer_random": contract.get("qformer_initialization") == "random",
        "qformer_checkpoint_not_loaded": contract.get("qformer_checkpoint_loaded") is False,
        "spatial_ast_frozen": contract.get("encoder_trainable_parameters") == 0,
        "qwen_native_frozen": contract.get("qwen_native_trainable_parameters") == 0,
        "no_gate": contract.get("gate_present") is False,
        "training_cache_disabled": contract.get("use_cache") is False,
    }
    if not all(contract_checks.values()):
        raise RuntimeError(f"Invalid Qwen3 BAT contract: {contract} checks={contract_checks}")

    template = get_template(
        template_type=TEMPLATE_TYPE,
        processor=processor,
        max_length=512,
        use_chat_template=False,
        padding_side="right",
        padding_free=False,
        template_backend="swift",
    )
    template.set_mode("train")
    record = find_record(args.qa_root.resolve())
    encoded = template.encode(
        {
            "messages": [
                {"role": "user", "content": "Classify the sound."},
                {"role": "assistant", "content": str(record["answer"])},
            ],
            "audios": [record],
        }
    )
    input_ids = as_long_batch(encoded["input_ids"], device)
    labels = as_long_batch(encoded["labels"], device)
    attention_mask = as_attention_batch(encoded.get("attention_mask"), input_ids)
    waveform = encoded.get("audio_waveform")
    if waveform is None:
        raise RuntimeError(f"Template did not produce audio_waveform; keys={sorted(encoded)}")
    if shape_tuple(waveform) != (2, 320000):
        raise RuntimeError(f"Waveform shape mismatch: {shape_tuple(waveform)}")
    if not torch.isfinite(waveform.float()).all():
        raise RuntimeError("Template waveform contains NaN or Inf")
    waveform = waveform.unsqueeze(0).to(device=device, dtype=torch.float32)

    if shape_tuple(input_ids) != (1, EXPECTED_SEQUENCE_LENGTH):
        raise RuntimeError(
            f"Expected fixed input shape (1,{EXPECTED_SEQUENCE_LENGTH}), got {shape_tuple(input_ids)}"
        )
    if shape_tuple(labels) != shape_tuple(input_ids):
        raise RuntimeError(
            f"Input/label shape mismatch: input={shape_tuple(input_ids)} labels={shape_tuple(labels)}"
        )
    if shape_tuple(attention_mask) != shape_tuple(input_ids):
        raise RuntimeError(
            f"Input/attention shape mismatch: input={shape_tuple(input_ids)} "
            f"attention={shape_tuple(attention_mask)}"
        )
    if not torch.all(labels[:, :EXPECTED_AUDIO_TOKENS] == -100):
        raise RuntimeError("Audio prefix labels are not fully masked")
    if attention_mask[:, :EXPECTED_AUDIO_TOKENS].sum().item() != EXPECTED_AUDIO_TOKENS:
        raise RuntimeError("Audio prefix attention mask is not active")

    expected_waveform = model.audio_renderer.render_record(record).float().cpu()
    render_error = float(
        (waveform.detach().float().cpu()[0] - expected_waveform).abs().max().item()
    )
    if render_error > 1e-5:
        raise RuntimeError(
            f"Template waveform does not match model renderer: max_abs_error={render_error}"
        )

    qwen_layer = model.model.layers[0]
    layer_forward = 0
    layer_backward = 0

    def on_forward(_module, _args, _output):
        nonlocal layer_forward
        layer_forward += 1

    def on_backward(_module, _grad_input, _grad_output):
        nonlocal layer_backward
        layer_backward += 1

    handles = [
        qwen_layer.register_forward_hook(on_forward),
        qwen_layer.register_full_backward_hook(on_backward),
    ]
    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            audio_waveforms=waveform,
            use_cache=False,
        )
        logits = outputs.logits
        if shape_tuple(logits) != (1, EXPECTED_SEQUENCE_LENGTH, EXPECTED_VOCAB_SIZE):
            raise RuntimeError(
                f"Unexpected logits shape: {shape_tuple(logits)}"
            )
        if not torch.isfinite(logits.float()).all():
            raise RuntimeError("Qwen3 logits contain NaN or Inf")
        if getattr(outputs, "past_key_values", None) is not None:
            raise RuntimeError("KV cache is unexpectedly enabled during training audit")
        if outputs.loss is None or not torch.isfinite(outputs.loss.float()):
            raise RuntimeError("Qwen3 did not return a finite loss")

        shifted_logits = logits[:, :-1].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        manual_loss = F.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.shape[-1]),
            shifted_labels.reshape(-1),
            ignore_index=-100,
        )
        model_loss = float(outputs.loss.detach().float().cpu())
        manual_value = float(manual_loss.detach().float().cpu())
        if not math.isclose(model_loss, manual_value, rel_tol=2e-3, abs_tol=2e-3):
            raise RuntimeError(
                f"Qwen3 CE mismatch: model={model_loss} manual={manual_value}"
            )
        manual_loss.backward()
    finally:
        for handle in handles:
            handle.remove()

    qformer_grad_parameters = sum(
        int(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all().item()
            and (parameter.grad.detach().float().abs() > 0).any().item()
        )
        for parameter in model.audio_qformer.parameters()
        if parameter.requires_grad
    )
    qformer_trainable_parameters = sum(
        parameter.numel()
        for parameter in model.audio_qformer.parameters()
        if parameter.requires_grad
    )
    groups = parameter_groups(model)
    if groups["qformer"] <= 0 or any(groups[key] for key in ("spatial_ast", "qwen_native", "other")):
        raise RuntimeError(f"Unexpected pre-LoRA trainable groups: {groups}")
    if qformer_grad_parameters <= 0:
        raise RuntimeError("Q-Former did not receive finite nonzero gradients")
    frozen_with_gradient = [
        name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad and parameter.grad is not None
    ]
    if frozen_with_gradient:
        raise RuntimeError(
            f"Frozen parameters unexpectedly received gradients: {frozen_with_gradient[:10]}"
        )
    if layer_forward != 1 or layer_backward != 1:
        raise RuntimeError(
            f"Expected one Qwen3 layer forward/backward, got "
            f"forward={layer_forward} backward={layer_backward}"
        )
    wrapper_audit = getattr(model, "_qwen3_bat_last_audio_forward_audit", None)
    if not isinstance(wrapper_audit, dict) or not wrapper_audit.get("audio_prefix_replaced"):
        raise RuntimeError(f"Audio embedding replacement audit missing: {wrapper_audit}")

    report = {
        "status": "ok",
        "model_type": MODEL_TYPE,
        "template_type": TEMPLATE_TYPE,
        "contract": contract,
        "contract_checks": contract_checks,
        "template": {
            "input_ids_shape": list(input_ids.shape),
            "labels_shape": list(labels.shape),
            "attention_mask_shape": list(attention_mask.shape),
            "audio_prefix_tokens": EXPECTED_AUDIO_TOKENS,
            "audio_prefix_labels_all_ignore": True,
            "waveform_shape": list(waveform.shape),
            "renderer_max_abs_error": render_error,
        },
        "forward": {
            "logits_shape": list(logits.shape),
            "model_ce": model_loss,
            "manual_shifted_ce": manual_value,
            "shift_verified": True,
            "use_cache": False,
            "past_key_values_present": getattr(outputs, "past_key_values", None) is not None,
            "wrapper_audio_audit": wrapper_audit,
        },
        "layer_calls": {
            "expected_single_pass": 1,
            "qwen_layer_forward": layer_forward,
            "qwen_layer_backward": layer_backward,
        },
        "parameters": {
            "trainable_groups_before_lora": groups,
            "qformer_trainable_parameters": qformer_trainable_parameters,
            "qformer_finite_gradient_parameter_count": qformer_grad_parameters,
            "spatial_ast_trainable_parameters": contract.get("encoder_trainable_parameters"),
            "qwen_native_trainable_parameters": contract.get("qwen_native_trainable_parameters"),
        },
        "scope": {
            "lora_injected": False,
            "qwen_backbone_frozen": True,
            "spatial_ast_frozen": True,
            "qformer_trainable": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[template] input={list(input_ids.shape)} labels={list(labels.shape)} waveform={list(waveform.shape)}")
    print(f"[forward] logits={list(logits.shape)} loss={manual_value:.6f}")
    print(f"[layers] forward={layer_forward} backward={layer_backward}")
    print(f"[parameters] {json.dumps(report['parameters'], ensure_ascii=False)}")
    print(f"[report] {args.output}")
    print("[status] ok")


if __name__ == "__main__":
    main()
