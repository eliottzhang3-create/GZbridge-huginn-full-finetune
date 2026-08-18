#!/usr/bin/env python3
"""Strict single-card Qwen3 BAT multimodal LoRA + Q-Former smoke.

The smoke intentionally uses exactly two private JSONL records and two
optimizer steps.  It validates the real ms-swift/PEFT path rather than a
standalone hand-written forward:

* Spatial-AST is frozen;
* Q-Former is randomly initialized and trainable;
* Qwen3 native parameters are frozen;
* LoRA is present on Qwen3 q_proj/v_proj only;
* the audio prefix is 64 Q-Former tokens;
* training uses ordinary shifted next-token CE with KV cache disabled;
* the saved adapter contains LoRA and Q-Former weights only.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from bat.configs.training import BAT_TRAINING


MODEL_TYPE = "qwen3_bat_spatial_ast"
TEMPLATE_TYPE = "qwen3_bat_audio_prefix"
EXPECTED_QWEN3_LAYERS = 36
EXPECTED_AUDIO_TOKENS = 64
EXPECTED_SEQUENCE_LENGTH = 176
TARGET_MODULES = tuple(BAT_TRAINING.lora_target_modules)


def package_version(name: str) -> str:
    try:
        return version(name)
    except Exception as exc:
        return f"<unavailable:{type(exc).__name__}>"


def require_environment() -> None:
    expected = {"ms-swift": "4.4.2", "transformers": "4.54.1", "peft": "0.18.1"}
    mismatches = {
        key: (expected[key], package_version(key))
        for key in expected
        if package_version(key) != expected[key]
    }
    if mismatches:
        raise RuntimeError(f"Unexpected Qwen3 BAT environment: {mismatches}")
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3 BAT LoRA smoke is GPU-only and must run in a submitted job")


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
    normalized = normalized_name(name)
    if "lora_A" in name or "lora_B" in name:
        return "lora"
    if normalized.startswith("audio_qformer."):
        return "qformer"
    if normalized.startswith("spatial_ast_encoder."):
        return "spatial_ast"
    if normalized.startswith(("model.", "lm_head.", "embed_tokens.")):
        return "qwen_native"
    return "other"


def shape_tuple(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", value)
    if shape is None:
        raise TypeError(f"Cannot read shape from {type(value).__name__}")
    return tuple(int(item) for item in shape)


def parameter_report(model: torch.nn.Module) -> dict[str, Any]:
    all_groups = {"qformer": 0, "lora": 0, "spatial_ast": 0, "qwen_native": 0, "other": 0}
    trainable_groups = {key: 0 for key in all_groups}
    trainable_names: list[str] = []
    frozen_native = 0
    frozen_spatial_ast = 0
    for name, parameter in model.named_parameters():
        group = parameter_group_name(name)
        all_groups[group] += parameter.numel()
        if group == "qwen_native" and not parameter.requires_grad:
            frozen_native += parameter.numel()
        if group == "spatial_ast" and not parameter.requires_grad:
            frozen_spatial_ast += parameter.numel()
        if parameter.requires_grad:
            trainable_groups[group] += parameter.numel()
            trainable_names.append(name)
    unexpected = {
        key: value
        for key, value in trainable_groups.items()
        if key in {"spatial_ast", "qwen_native", "other"} and value
    }
    if unexpected:
        raise RuntimeError(f"Unexpected Qwen3 trainable groups: {unexpected}")
    if trainable_groups["qformer"] <= 0 or trainable_groups["lora"] <= 0:
        raise RuntimeError(f"Q-Former or LoRA is not trainable: {trainable_groups}")
    if frozen_native <= 0 or frozen_spatial_ast <= 0:
        raise RuntimeError(
            f"Backbone freeze audit is empty: frozen_native={frozen_native} frozen_spatial_ast={frozen_spatial_ast}"
        )
    return {
        "all_parameter_counts": all_groups,
        "trainable_parameter_counts": trainable_groups,
        "frozen_qwen_native_parameters": frozen_native,
        "frozen_spatial_ast_parameters": frozen_spatial_ast,
        "trainable_name_count": len(trainable_names),
        "trainable_name_preview": trainable_names[:80],
    }


def gradient_report(model: torch.nn.Module) -> dict[str, Any]:
    finite_counts = {key: 0 for key in ("qformer", "lora", "spatial_ast", "qwen_native", "other")}
    nonzero_counts = {key: 0 for key in finite_counts}
    nonfinite: list[str] = []
    frozen_with_gradient: list[str] = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        group = parameter_group_name(name)
        gradient = parameter.grad.detach()
        if not bool(torch.isfinite(gradient).all().item()):
            nonfinite.append(name)
        else:
            finite_counts[group] += 1
            if bool((gradient.float().abs() > 0).any().item()):
                nonzero_counts[group] += 1
        if not parameter.requires_grad:
            frozen_with_gradient.append(name)
    if finite_counts["qformer"] <= 0 or finite_counts["lora"] <= 0:
        raise RuntimeError(f"Expected Q-Former and LoRA gradients: {finite_counts}")
    if nonzero_counts["qformer"] <= 0 or nonzero_counts["lora"] <= 0:
        raise RuntimeError(f"Expected nonzero Q-Former and LoRA gradients: {nonzero_counts}")
    if any(finite_counts[key] for key in ("spatial_ast", "qwen_native", "other")):
        raise RuntimeError(f"Frozen native/audio component received gradients: {finite_counts}")
    if nonfinite:
        raise RuntimeError(f"Non-finite gradients: {nonfinite[:10]}")
    if frozen_with_gradient:
        raise RuntimeError(f"Frozen parameters received gradients: {frozen_with_gradient[:10]}")
    return {
        "finite_gradient_parameter_counts": finite_counts,
        "nonzero_gradient_parameter_counts": nonzero_counts,
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
        ranks.update(int(value) for value in getattr(module, "r", {}).values())
    expected = EXPECTED_QWEN3_LAYERS * len(TARGET_MODULES)
    if len(modules) != expected:
        raise RuntimeError(f"Expected {expected} Qwen3 LoRA modules, found {len(modules)}")
    if invalid or ranks != {BAT_TRAINING.lora_rank}:
        raise RuntimeError(f"Invalid Qwen3 LoRA contract: invalid={invalid[:10]} ranks={sorted(ranks)}")
    forbidden = ("audio_qformer", "spatial_ast_encoder", "lm_head", "embed_tokens")
    if any(any(part in name for part in forbidden) for name in modules):
        raise RuntimeError("LoRA was injected outside Qwen3 attention projections")
    return {
        "module_count": len(modules),
        "expected_module_count": expected,
        "target_modules": list(TARGET_MODULES),
        "rank": BAT_TRAINING.lora_rank,
        "alpha": BAT_TRAINING.lora_alpha,
        "dropout": BAT_TRAINING.lora_dropout,
        "module_preview": sorted(modules)[:24],
    }


def adapter_config_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Missing adapter config: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    saved_targets = {str(item) for item in (config.get("target_modules") or [])}
    if saved_targets != set(TARGET_MODULES):
        raise RuntimeError(f"Unexpected adapter targets: {sorted(saved_targets)}")
    if int(config.get("r", -1)) != BAT_TRAINING.lora_rank:
        raise RuntimeError(f"Unexpected adapter rank: {config.get('r')}")
    if int(config.get("lora_alpha", -1)) != BAT_TRAINING.lora_alpha:
        raise RuntimeError(f"Unexpected adapter alpha: {config.get('lora_alpha')}")
    if not math.isclose(float(config.get("lora_dropout", -1.0)), BAT_TRAINING.lora_dropout, abs_tol=1e-8):
        raise RuntimeError(f"Unexpected adapter dropout: {config.get('lora_dropout')}")
    modules_to_save = {str(item) for item in (config.get("modules_to_save") or [])}
    if "audio_qformer" not in modules_to_save:
        raise RuntimeError(f"audio_qformer missing from modules_to_save: {sorted(modules_to_save)}")
    return {
        "path": str(path),
        "target_modules": sorted(saved_targets),
        "r": int(config["r"]),
        "lora_alpha": int(config["lora_alpha"]),
        "lora_dropout": float(config["lora_dropout"]),
        "modules_to_save": sorted(modules_to_save),
    }


def optimizer_report(trainer: Any, model: torch.nn.Module) -> dict[str, Any]:
    optimizer = trainer.optimizer
    if optimizer is None:
        raise RuntimeError("Swift did not create an optimizer")
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    trainable_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if optimizer_ids != trainable_ids:
        raise RuntimeError("Optimizer parameter set does not equal trainable parameter set")
    betas = sorted({tuple(float(value) for value in group.get("betas", ())) for group in optimizer.param_groups})
    weight_decays = sorted({float(group.get("weight_decay", 0.0)) for group in optimizer.param_groups})
    if betas != [(BAT_TRAINING.beta1, BAT_TRAINING.beta2)]:
        raise RuntimeError(f"Unexpected AdamW betas: {betas}")
    if any(value not in {0.0, BAT_TRAINING.weight_decay} for value in weight_decays):
        raise RuntimeError(f"Unexpected weight decay groups: {weight_decays}")
    scheduler_type = str(getattr(getattr(trainer, "args", None), "lr_scheduler_type", "")).lower()
    if trainer.lr_scheduler is None or "cosine" not in scheduler_type:
        raise RuntimeError(f"Expected cosine scheduler, got {scheduler_type}")
    return {
        "class": f"{optimizer.__class__.__module__}.{optimizer.__class__.__name__}",
        "betas": betas,
        "weight_decay_groups": weight_decays,
        "learning_rates": sorted({float(group["lr"]) for group in optimizer.param_groups}),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "scheduler_class": f"{trainer.lr_scheduler.__class__.__module__}.{trainer.lr_scheduler.__class__.__name__}",
        "scheduler_type": scheduler_type,
        "scheduler_last_epoch": int(trainer.lr_scheduler.last_epoch),
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
        raise RuntimeError(f"Missing adapter model: {adapter}")
    with safe_open(str(adapter), framework="pt", device="cpu") as handle:
        keys = sorted(handle.keys())
    lora_keys = [key for key in keys if "lora_" in key]
    qformer_keys = [key for key in keys if "audio_qformer" in key]
    unexpected = [key for key in keys if key not in lora_keys and key not in qformer_keys]
    expected_lora_tensors = EXPECTED_QWEN3_LAYERS * len(TARGET_MODULES) * 2
    if len(lora_keys) != expected_lora_tensors or not qformer_keys or unexpected:
        raise RuntimeError(
            f"Unexpected Qwen3 adapter keys: lora={len(lora_keys)} expected={expected_lora_tensors} "
            f"qformer={len(qformer_keys)} unexpected={unexpected[:10]}"
        )
    lora_modules = {key.split(".lora_", 1)[0] for key in lora_keys if ".lora_" in key}
    if len(lora_modules) != EXPECTED_QWEN3_LAYERS * len(TARGET_MODULES):
        raise RuntimeError(f"Unexpected saved LoRA module count: {len(lora_modules)}")
    if any(path.rsplit(".", 1)[-1] not in TARGET_MODULES for path in lora_modules):
        raise RuntimeError("Saved LoRA contains a non-q_proj/v_proj module")
    state_path = path / "trainer_state.json"
    if not state_path.is_file():
        raise RuntimeError(f"Missing trainer state: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if int(state.get("global_step", -1)) != 2:
        raise RuntimeError(f"Expected checkpoint global_step=2, got {state.get('global_step')}")
    return {
        "path": str(path),
        "tensor_count": len(keys),
        "lora_tensor_count": len(lora_keys),
        "expected_lora_tensor_count": expected_lora_tensors,
        "qformer_tensor_count": len(qformer_keys),
        "unexpected_tensor_count": len(unexpected),
        "global_step": int(state["global_step"]),
        "adapter_config": adapter_config_report(path / "adapter_config.json"),
    }


def main() -> None:
    args = parse_args()
    BAT_TRAINING.validate()
    require_environment()
    for path in (args.model_path, args.plugin_path, args.dataset):
        if not path.expanduser().resolve().exists():
            raise FileNotFoundError(path)
    records = count_jsonl(args.dataset)
    if records != 2:
        raise RuntimeError(f"Qwen3 LoRA smoke requires exactly 2 JSONL records; got {records}")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if str(args.output_report).replace("\\", "/").startswith("/hpc_stor03/public"):
        raise ValueError(f"Refusing public output path: {args.output_report}")

    from swift.pipelines.train.sft import SwiftSft

    schedule = BAT_TRAINING.schedule(dataset_size=2, world_size=1, gradient_accumulation_steps=1, stage_name="I")
    argv = [
        "--model", str(args.model_path), "--model_type", MODEL_TYPE, "--template", TEMPLATE_TYPE,
        "--external_plugins", str(args.plugin_path), "--dataset", str(args.dataset),
        "--split_dataset_ratio", "0", "--dataset_shuffle", "false", "--train_dataloader_shuffle", "false",
        "--sortish_sampler", "false", "--group_by_length", "false", "--max_length", str(EXPECTED_SEQUENCE_LENGTH),
        "--remove_unused_columns", "false", "--output_dir", str(args.output_dir),
        "--tuner_type", "lora", "--tuner_backend", "peft", "--target_modules", *TARGET_MODULES,
        "--modules_to_save", "audio_qformer", "--freeze_llm", "true", "--freeze_vit", "true",
        "--freeze_aligner", "false", "--lora_rank", str(BAT_TRAINING.lora_rank),
        "--lora_alpha", str(BAT_TRAINING.lora_alpha), "--lora_dropout", str(BAT_TRAINING.lora_dropout),
        "--learning_rate", str(BAT_TRAINING.learning_rate), "--lr_scheduler_type", "cosine",
        "--warmup_steps", str(schedule["warmup_steps"]), "--max_steps", str(schedule["total_steps"]),
        "--num_train_epochs", str(schedule["epochs"]), "--per_device_train_batch_size", str(BAT_TRAINING.per_device_batch_size),
        "--gradient_accumulation_steps", "1", "--gradient_checkpointing", "false", "--logging_steps", "1",
        "--save_strategy", "steps", "--save_steps", "2", "--save_total_limit", "1", "--save_only_model", "false",
        "--dataloader_num_workers", "0", "--dataloader_pin_memory", "false", "--dataset_num_proc", "1",
        "--lazy_tokenize", "false", "--load_from_cache_file", "false", "--loss_scale", "all",
        "--seed", "42", "--data_seed", "42", "--optim", "adamw_torch", "--adam_beta1", str(BAT_TRAINING.beta1),
        "--adam_beta2", str(BAT_TRAINING.beta2), "--weight_decay", str(BAT_TRAINING.weight_decay),
        "--attn_impl", "sdpa", "--bf16", "true", "--report_to", "none",
    ]

    class AuditedSwiftSft(SwiftSft):
        def train(self, trainer):
            model = trainer.model
            causal = find_module(model, "Qwen3ForCausalLM")
            qformer = find_module(model, "BATQFormer")
            encoder = find_module(model, "SpatialASTAudioEncoder")
            if int(getattr(causal.config, "num_hidden_layers", -1)) != EXPECTED_QWEN3_LAYERS:
                raise RuntimeError("Unexpected Qwen3 layer count")
            if bool(getattr(causal.config, "use_cache", True)):
                raise RuntimeError("Qwen3 use_cache must be disabled during training")
            if any(parameter.requires_grad for parameter in encoder.parameters()):
                raise RuntimeError("Spatial-AST is unexpectedly trainable")
            if not any(parameter.requires_grad for parameter in qformer.parameters()):
                raise RuntimeError("Q-Former is unexpectedly frozen")
            model.train()
            encoder.eval()
            parameters = parameter_report(model)
            lora = lora_report(model)
            trace: dict[str, Any] = {
                "forward": 0, "backward": 0, "layer_forward": 0, "layer_backward": 0,
                "loss": None, "batch": None, "gradient_audit": None,
                "past_key_values_present": None, "audio_forward_audit": None,
            }
            handles: list[Any] = []
            layer0 = causal.model.layers[0]
            handles.append(layer0.register_forward_hook(lambda *_: trace.__setitem__("layer_forward", trace["layer_forward"] + 1)))
            handles.append(layer0.register_full_backward_hook(lambda *_: trace.__setitem__("layer_backward", trace["layer_backward"] + 1)))
            original_compute_loss = trainer.compute_loss
            original_backward = trainer.accelerator.backward

            def compute_loss(actual_model, inputs, return_outputs=False, num_items_in_batch=None):
                trace["forward"] += 1
                result = original_compute_loss(actual_model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
                loss, outputs = result
                if trace["loss"] is None:
                    logits = outputs.logits
                    labels = inputs["labels"]
                    input_ids = inputs["input_ids"]
                    attention_mask = inputs["attention_mask"]
                    waveform = inputs.get("audio_waveforms")
                    if shape_tuple(input_ids) != shape_tuple(labels) or shape_tuple(attention_mask) != shape_tuple(input_ids):
                        raise RuntimeError(
                            f"Input/label/mask mismatch: input={shape_tuple(input_ids)} labels={shape_tuple(labels)} mask={shape_tuple(attention_mask)}"
                        )
                    if shape_tuple(input_ids) != (2, EXPECTED_SEQUENCE_LENGTH):
                        raise RuntimeError(f"Unexpected fixed batch shape: {shape_tuple(input_ids)}")
                    if shape_tuple(waveform) != (2, 2, 320000):
                        raise RuntimeError(f"Unexpected waveform shape: {shape_tuple(waveform)}")
                    if not bool(torch.isfinite(waveform.float()).all().item()):
                        raise RuntimeError("Waveform contains NaN or Inf")
                    if logits.ndim != 3 or shape_tuple(logits)[:2] != shape_tuple(labels):
                        raise RuntimeError(f"Unexpected Qwen3 logits shape: {shape_tuple(logits)}")
                    if not bool(torch.isfinite(logits.float()).all().item()):
                        raise RuntimeError("Qwen3 logits contain NaN or Inf")
                    if not bool(torch.isfinite(loss.detach().float()).all().item()):
                        raise RuntimeError("Qwen3 loss is NaN or Inf")
                    trace["past_key_values_present"] = getattr(outputs, "past_key_values", None) is not None
                    if trace["past_key_values_present"]:
                        raise RuntimeError("KV cache is unexpectedly enabled in Qwen3 training")
                    if not bool((labels[:, :EXPECTED_AUDIO_TOKENS] == -100).all().item()):
                        raise RuntimeError("Audio prefix labels are not masked")
                    shifted_logits = logits[:, :-1].contiguous()
                    shifted_labels = labels[:, 1:].contiguous()
                    manual_loss = F.cross_entropy(
                        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
                        shifted_labels.reshape(-1), ignore_index=-100,
                    )
                    trainer_value = float(loss.detach().float().cpu())
                    manual_value = float(manual_loss.detach().float().cpu())
                    if not math.isclose(trainer_value, manual_value, rel_tol=2e-3, abs_tol=2e-3):
                        raise RuntimeError(f"Qwen3 Trainer CE mismatch: trainer={trainer_value} manual={manual_value}")
                    causal_audit = getattr(causal, "_qwen3_bat_last_audio_forward_audit", None)
                    if not causal_audit or not causal_audit.get("audio_prefix_replaced"):
                        raise RuntimeError("Qwen3 audio prefix replacement audit was not captured")
                    trace["audio_forward_audit"] = causal_audit
                    trace["batch"] = {
                        "input_ids_shape": list(input_ids.shape), "labels_shape": list(labels.shape),
                        "attention_mask_shape": list(attention_mask.shape), "audio_waveforms_shape": list(waveform.shape),
                        "audio_prefix_label_ignore_count": int((labels[:, :EXPECTED_AUDIO_TOKENS] == -100).sum().item()),
                        "padding_label_ignore_count": int((labels[:, EXPECTED_AUDIO_TOKENS:] == -100).sum().item()),
                        "valid_shifted_target_count": int((shifted_labels != -100).sum().item()),
                        "manual_shifted_ce": manual_value, "trainer_ce": trainer_value, "shift_verified": True,
                    }
                    trace["loss"] = {"value": trainer_value, "logits_shape": list(logits.shape), "labels_shape": list(labels.shape)}
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
                raise RuntimeError(f"Unexpected Qwen3 forward/backward counts: {trace}")
            if trace["layer_forward"] != schedule["total_steps"] or trace["layer_backward"] != schedule["total_steps"]:
                raise RuntimeError(f"Unexpected Qwen3 layer counts: {trace}")
            if trace["gradient_audit"] is None:
                raise RuntimeError("Gradient audit was not captured")
            optimizer = optimizer_report(trainer, model)
            checkpoint = checkpoint_report(checkpoint_path(args.output_dir))
            report = {
                "status": "ok",
                "model": {"path": str(args.model_path), "class": causal.__class__.__name__, "layers": EXPECTED_QWEN3_LAYERS},
                "audio_contract": getattr(causal, "_qwen3_bat_audio_contract", None),
                "schedule": dict(schedule), "dataset_records": records, "argv": argv,
                "parameters": parameters, "lora": lora, "optimizer": optimizer,
                "forward_audit": trace, "checkpoint": checkpoint,
                "elapsed_seconds": time.perf_counter() - started,
                "trainer": {"class": f"{trainer.__class__.__module__}.{trainer.__class__.__name__}", "global_step": int(trainer.state.global_step), "log_history": trainer.state.log_history},
            }
            write_json(args.output_report, report)
            print(f"[parameters] {json.dumps(parameters, ensure_ascii=False)}", flush=True)
            print(f"[lora] {json.dumps(lora, ensure_ascii=False)}", flush=True)
            print(f"[optimizer] {json.dumps(optimizer, ensure_ascii=False)}", flush=True)
            print(f"[forward] {json.dumps(trace, ensure_ascii=False)}", flush=True)
            print(f"[checkpoint] {json.dumps(checkpoint, ensure_ascii=False)}", flush=True)
            print(f"[report] {args.output_report}", flush=True)
            print("[status] ok", flush=True)
            return result

    print("========== QWEN3 BAT LORA + Q-FORMER STRICT SMOKE ==========")
    print(f"[packages] ms-swift={package_version('ms-swift')} transformers={package_version('transformers')} torch={package_version('torch')}")
    print(f"[schedule] {json.dumps(schedule, ensure_ascii=False)}")
    AuditedSwiftSft(argv).main()


if __name__ == "__main__":
    main()
