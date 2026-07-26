#!/usr/bin/env python3
"""Run and audit one real AudioCaps-v2 HRM audio Swift Trainer update."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import wave
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open

import inspect_hrm_audio_swift_trainability as trainability_audit
import inspect_hrm_swift_training as recurrence_audit
from smoke_hrm_swift_trainer import logits_difference_report


MODEL_TYPE = "hrm_text_audio_whisper"
TEMPLATE_TYPE = "hrm_text_audio"
EXPECTED_SOURCE_RECORDS = 89_658
EXPECTED_AUDIO_PREFIX = 34
EXPECTED_HRM_PARAMETERS = 1_182_795_264
EXPECTED_AUDIO_ENCODER_PARAMETERS = 636_784_640
EXPECTED_ALIGNER_PARAMETERS = 39_538_176
EXPECTED_ALIGNER_TENSORS = 20
EXPECTED_LORA_PARAMETERS = 8_257_536
EXPECTED_LORA_TENSORS = 512
EXPECTED_TOTAL_TRAINABLE = EXPECTED_ALIGNER_PARAMETERS + EXPECTED_LORA_PARAMETERS
ALIGNER_MARKERS = ("temporal_compressor.", "audio_projector.", "audio_boundary_embeddings.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrapper-model-path", type=Path, required=True)
    parser.add_argument("--plugin-path", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--reload-script", type=Path, required=True)
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def canonical_aligner_key(key: str) -> str | None:
    marker = next((candidate for candidate in ALIGNER_MARKERS if candidate in key), None)
    if marker is None:
        return None
    suffix = key[key.index(marker) :]
    return suffix.replace("original_module.", "").replace("modules_to_save.default.", "")


def canonical_adapter_key(key: str) -> str:
    for marker in ("H_module.", "L_module."):
        if marker in key:
            return key[key.index(marker) :].replace(".default.", ".")
    raise RuntimeError(f"Unexpected HRM adapter key: {key}")


def validate_source_record(record: dict[str, Any], *, line_number: int) -> dict[str, Any]:
    messages = record.get("messages")
    audios = record.get("audios")
    metadata = record.get("metadata")
    if not isinstance(messages, list) or [item.get("role") for item in messages] != ["system", "user", "assistant"]:
        raise RuntimeError(f"AudioCaps line {line_number} has unexpected messages: {messages}")
    if any(not isinstance(item.get("content"), str) or not item["content"].strip() for item in messages):
        raise RuntimeError(f"AudioCaps line {line_number} has an empty message")
    if not isinstance(audios, list) or len(audios) != 1:
        raise RuntimeError(f"AudioCaps line {line_number} must have exactly one audio path: {audios}")
    if not isinstance(metadata, dict) or metadata.get("dataset") != "audiocaps_v2" or metadata.get("split") != "train":
        raise RuntimeError(f"AudioCaps line {line_number} metadata mismatch: {metadata}")
    audio_path = Path(audios[0]).expanduser()
    if not audio_path.is_file():
        raise FileNotFoundError(f"AudioCaps smoke WAV is missing: {audio_path}")
    with wave.open(str(audio_path), "rb") as handle:
        wav = {
            "channels": handle.getnchannels(),
            "sample_width_bytes": handle.getsampwidth(),
            "sample_rate": handle.getframerate(),
            "frame_count": handle.getnframes(),
            "compression": handle.getcomptype(),
        }
    expected_wav = {"channels": 1, "sample_width_bytes": 2, "sample_rate": 32_000, "compression": "NONE"}
    mismatches = {key: {"expected": value, "actual": wav[key]} for key, value in expected_wav.items() if wav[key] != value}
    if mismatches or wav["frame_count"] <= 0:
        raise RuntimeError(f"AudioCaps line {line_number} WAV contract mismatch: {wav} mismatches={mismatches}")
    return {
        "line_number": line_number,
        "audio_path": str(audio_path),
        "sample_id": metadata.get("sample_id"),
        "system_prompt": messages[0]["content"],
        "user_prompt": messages[1]["content"],
        "caption": messages[-1]["content"],
        "wav": wav,
    }


def prepare_smoke_manifest(source_manifest: Path, run_dir: Path) -> tuple[Path, dict[str, Any]]:
    stats_path = source_manifest.with_suffix(f"{source_manifest.suffix}.stats.json")
    if not source_manifest.is_file() or not stats_path.is_file():
        raise FileNotFoundError(f"AudioCaps manifest/stats missing: manifest={source_manifest} stats={stats_path}")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    expected_stats = {
        "dataset": "audiocaps_v2",
        "split": "train",
        "record_count": EXPECTED_SOURCE_RECORDS,
        "audio_path_verification": "passed",
        "wav_readability_verification": "passed",
    }
    mismatches = {
        key: {"expected": expected, "actual": stats.get(key)}
        for key, expected in expected_stats.items()
        if stats.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"AudioCaps source stats mismatch: {mismatches}")

    selected: list[dict[str, Any]] = []
    selected_reports: list[dict[str, Any]] = []
    seen_audio: set[str] = set()
    with source_manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            record_report = validate_source_record(record, line_number=line_number)
            if record_report["audio_path"] in seen_audio:
                continue
            hrm_record = dict(record)
            hrm_record["messages"] = [dict(record["messages"][1]), dict(record["messages"][2])]
            selected.append(hrm_record)
            selected_reports.append(record_report)
            seen_audio.add(record_report["audio_path"])
            if len(selected) == 2:
                break
    if len(selected) != 2:
        raise RuntimeError(f"Unable to select two distinct AudioCaps records, found {len(selected)}")
    fixture_path = run_dir / "audiocaps_v2_first2_hrm_audio_smoke.jsonl"
    fixture_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in selected),
        encoding="utf-8",
    )
    return fixture_path, {
        "source_manifest": str(source_manifest),
        "source_stats": str(stats_path),
        "source_record_count": int(stats["record_count"]),
        "fixture_manifest": str(fixture_path),
        "fixture_records": selected_reports,
        "audio_user_caption_metadata_preserved": True,
        "source_system_prompt_removed_for_hrm_direct_template": True,
        "source_records_unchanged": True,
    }


def module_parameter_ids(module: torch.nn.Module | None) -> set[int]:
    return set() if module is None else {id(parameter) for parameter in module.parameters()}


def frozen_parameter_groups(model: torch.nn.Module, wrapper: torch.nn.Module) -> dict[str, list[tuple[str, torch.Tensor]]]:
    _, lora_ids = trainability_audit.lora_module_report(model)
    audio_ids = module_parameter_ids(wrapper.audio_encoder)
    hrm_ids = module_parameter_ids(wrapper.model) | module_parameter_ids(wrapper.lm_head)
    groups = {"audio_encoder": [], "hrm_base": []}
    for name, parameter in model.named_parameters():
        identity = id(parameter)
        if identity in audio_ids:
            groups["audio_encoder"].append((name, parameter))
        elif identity in hrm_ids and identity not in lora_ids:
            groups["hrm_base"].append((name, parameter))
    counts = {name: sum(parameter.numel() for _, parameter in entries) for name, entries in groups.items()}
    if counts != {
        "audio_encoder": EXPECTED_AUDIO_ENCODER_PARAMETERS,
        "hrm_base": EXPECTED_HRM_PARAMETERS,
    }:
        raise RuntimeError(f"Frozen parameter ownership mismatch: {counts}")
    if any(parameter.requires_grad for entries in groups.values() for _, parameter in entries):
        raise RuntimeError("A frozen Whisper/HRM parameter unexpectedly requires gradients")
    return groups


def update_hash_with_tensor(digest: Any, tensor: torch.Tensor, *, chunk_elements: int = 4_000_000) -> None:
    flat = tensor.detach().contiguous().view(-1)
    for start in range(0, flat.numel(), chunk_elements):
        chunk = flat[start : start + chunk_elements].cpu().contiguous()
        if chunk.dtype == torch.bfloat16:
            chunk = chunk.view(torch.uint16)
        digest.update(chunk.numpy().tobytes())


def parameter_group_digest(entries: list[tuple[str, torch.Tensor]]) -> dict[str, Any]:
    digest = hashlib.sha256()
    total = 0
    for name, parameter in sorted(entries, key=lambda item: item[0]):
        total += parameter.numel()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(parameter.shape)).encode("ascii"))
        digest.update(str(parameter.dtype).encode("ascii"))
        update_hash_with_tensor(digest, parameter)
    return {"sha256": digest.hexdigest(), "parameter_count": total, "tensor_count": len(entries)}


def snapshot_trainables(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def classify_trainable(name: str) -> str:
    if "lora_" in name:
        stack = "H" if ".H_module." in name else "L" if ".L_module." in name else "other"
        side = "B" if "lora_B" in name else "A" if "lora_A" in name else "other"
        return f"lora_{stack}_{side}"
    if "temporal_compressor." in name:
        return "temporal_compressor"
    if "audio_projector." in name:
        return "audio_projector"
    if "audio_boundary_embeddings." in name:
        return "audio_boundary_embeddings"
    return "other"


def install_gradient_audit(model: torch.nn.Module):
    records: dict[str, dict[str, Any]] = {}
    handles = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        def make_hook(parameter_name: str):
            def hook(gradient: torch.Tensor):
                finite = bool(torch.isfinite(gradient).all().item())
                norm = float(gradient.float().norm().item()) if finite else float("nan")
                records[parameter_name] = {
                    "finite": finite,
                    "norm": norm,
                    "max_abs": float(gradient.float().abs().max().item()) if finite else float("nan"),
                }
                return gradient
            return hook

        handles.append(parameter.register_hook(make_hook(name)))
    return records, handles


def summarize_gradients(model: torch.nn.Module, records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    missing = sorted(set(trainable_names) - set(records))
    nonfinite = sorted(name for name, values in records.items() if not values["finite"])
    if missing or nonfinite:
        raise RuntimeError(f"Gradient audit failed: missing={missing[:20]} nonfinite={nonfinite[:20]}")
    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"tensors": 0, "nonzero": 0, "norm_sum": 0.0})
    for name, values in records.items():
        group = classify_trainable(name)
        groups[group]["tensors"] += 1
        groups[group]["nonzero"] += int(values["norm"] > 0)
        groups[group]["norm_sum"] += values["norm"]
    required_nonzero = (
        "temporal_compressor",
        "audio_projector",
        "audio_boundary_embeddings",
        "lora_H_B",
        "lora_L_B",
    )
    failures = [group for group in required_nonzero if groups[group]["nonzero"] <= 0]
    if failures or groups.get("other", {}).get("tensors", 0):
        raise RuntimeError(f"Required gradient groups missing/nonzero failure={failures} groups={dict(groups)}")
    return {"tensor_count": len(records), "groups": dict(groups)}


def compare_trainable_updates(before: dict[str, torch.Tensor], model: torch.nn.Module) -> dict[str, Any]:
    after = {name: parameter.detach().cpu() for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(before) != set(after):
        raise RuntimeError("Trainable parameter names changed during Trainer update")
    groups: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"tensors": 0, "changed_tensors": 0, "max_abs_change": 0.0}
    )
    examples = []
    for name in sorted(before):
        difference = (before[name].float() - after[name].float()).abs()
        max_change = float(difference.max().item())
        group = classify_trainable(name)
        groups[group]["tensors"] += 1
        groups[group]["changed_tensors"] += int(max_change > 0)
        groups[group]["max_abs_change"] = max(groups[group]["max_abs_change"], max_change)
        if max_change > 0 and len(examples) < 20:
            examples.append({"name": name, "max_abs_change": max_change})
    required_changed = (
        "temporal_compressor",
        "audio_projector",
        "audio_boundary_embeddings",
        "lora_H_B",
        "lora_L_B",
    )
    failures = [group for group in required_changed if groups[group]["changed_tensors"] <= 0]
    if failures or groups.get("other", {}).get("tensors", 0):
        raise RuntimeError(f"Required trainable groups did not update: failures={failures} groups={dict(groups)}")
    return {"groups": dict(groups), "changed_examples": examples}


def capture_forward_audit(tokenizer: Any, wrapper: torch.nn.Module, outer_model: torch.nn.Module):
    outer_capture: dict[str, Any] = {}
    core_capture: dict[str, Any] = {}
    attention_capture: dict[str, Any] = {}

    def outer_pre(_module, _args, kwargs):
        if outer_capture:
            return
        required = ("input_ids", "attention_mask", "labels", "token_type_ids", "audio_input_features", "logits_to_keep")
        missing = [key for key in required if key not in kwargs]
        if missing:
            raise RuntimeError(f"Trainer forward dropped HRM audio fields: missing={missing} keys={sorted(kwargs)}")
        for key in required:
            value = kwargs[key]
            outer_capture[key] = value.detach().cpu().clone() if torch.is_tensor(value) else value

    def outer_post(_module, _args, _kwargs, output):
        if "logits" in outer_capture:
            return
        if output.loss is None or not torch.is_tensor(output.logits):
            raise RuntimeError("Trainer HRM audio output lacks loss/logits")
        outer_capture["loss"] = float(output.loss.detach().float().cpu().item())
        outer_capture["logits"] = output.logits.detach().cpu().clone()

    def core_pre(_module, _args, kwargs):
        if core_capture:
            return
        for key in ("inputs_embeds", "attention_mask", "token_type_ids"):
            value = kwargs.get(key)
            if not torch.is_tensor(value):
                raise RuntimeError(f"HRM core did not receive tensor {key}: keys={sorted(kwargs)}")
            core_capture[key] = value.detach().cpu().clone()

    def attention_pre(_module, args, kwargs):
        if attention_capture:
            return
        mask = kwargs.get("attention_mask")
        if not torch.is_tensor(mask):
            candidates = [
                value
                for value in args[1:]
                if torch.is_tensor(value) and value.ndim == 4
            ]
            if len(candidates) == 1:
                mask = candidates[0]
        if not torch.is_tensor(mask):
            raise RuntimeError(f"HRM attention did not receive an explicit PrefixLM mask: keys={sorted(kwargs)}")
        attention_capture["attention_mask"] = mask.detach().cpu().clone()

    handles = [
        outer_model.register_forward_pre_hook(outer_pre, with_kwargs=True),
        outer_model.register_forward_hook(outer_post, with_kwargs=True),
        wrapper.model.register_forward_pre_hook(core_pre, with_kwargs=True),
        wrapper.model.L_module.layers[0].self_attn.register_forward_pre_hook(attention_pre, with_kwargs=True),
    ]
    return outer_capture, core_capture, attention_capture, handles


def mask_allowed(mask: torch.Tensor, batch: int, query: int, key: int) -> bool:
    value = mask[batch, 0, query, key]
    if value.dtype == torch.bool:
        return bool(value.item())
    scalar = float(value.item())
    return math.isfinite(scalar) and scalar > -1e4


def audit_forward_semantics(
    tokenizer: Any,
    outer: dict[str, Any],
    core: dict[str, Any],
    attention: dict[str, Any],
    fixture_records: list[dict[str, Any]],
) -> dict[str, Any]:
    input_ids = outer["input_ids"]
    text_attention = outer["attention_mask"]
    labels = outer["labels"]
    text_types = outer["token_type_ids"]
    audio_features = outer["audio_input_features"]
    logits = outer["logits"]
    keep_value = outer["logits_to_keep"]
    keep = int(keep_value.item()) if torch.is_tensor(keep_value) else int(keep_value)
    if tuple(audio_features.shape) != (2, 80, 3000) or not bool(torch.isfinite(audio_features).all().item()):
        raise RuntimeError(f"Trainer audio feature batch is invalid: {tuple(audio_features.shape)}")
    if not (input_ids.shape == text_attention.shape == text_types.shape):
        raise RuntimeError("Trainer text input/attention/token_type shapes disagree")
    if labels.shape != (2, keep) or logits.shape[:2] != labels.shape:
        raise RuntimeError(
            f"Compact labels/logits mismatch: labels={tuple(labels.shape)} logits={tuple(logits.shape)} keep={keep}"
        )

    combined_types = core["token_type_ids"]
    combined_attention = core["attention_mask"]
    combined_embeds = core["inputs_embeds"]
    expected_combined_width = input_ids.shape[1] + EXPECTED_AUDIO_PREFIX
    if combined_types.shape != (2, expected_combined_width):
        raise RuntimeError(f"Combined PrefixLM token type shape mismatch: {tuple(combined_types.shape)}")
    if combined_attention.shape != (2, expected_combined_width):
        raise RuntimeError(f"Combined attention shape mismatch: {tuple(combined_attention.shape)}")
    if combined_embeds.shape[:2] != (2, expected_combined_width) or combined_embeds.shape[2] != 1536:
        raise RuntimeError(f"Combined embedding shape mismatch: {tuple(combined_embeds.shape)}")
    if not bool((combined_types[:, :EXPECTED_AUDIO_PREFIX] == 1).all().item()):
        raise RuntimeError("Audio prefix token types are not entirely PrefixLM ones")
    if not torch.equal(combined_types[:, EXPECTED_AUDIO_PREFIX:], text_types):
        raise RuntimeError("Combined text token types differ from the Swift-collated token types")
    if not bool((combined_attention[:, :EXPECTED_AUDIO_PREFIX] == 1).all().item()):
        raise RuntimeError("Audio prefix attention mask is not entirely valid")
    if not torch.equal(combined_attention[:, EXPECTED_AUDIO_PREFIX:], text_attention):
        raise RuntimeError("Combined text attention mask differs from the Swift-collated mask")

    im_end_id = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    eos_id = int(tokenizer.eos_token_id)
    compact_start = input_ids.shape[1] - keep
    rows = []
    actual_mask = attention["attention_mask"]
    if actual_mask.ndim != 4 or actual_mask.shape[0] != 2 or actual_mask.shape[-2:] != (
        expected_combined_width,
        expected_combined_width,
    ):
        raise RuntimeError(f"Unexpected native HRM PrefixLM attention mask shape: {tuple(actual_mask.shape)}")
    for row in range(2):
        valid_text = int(text_attention[row].sum().item())
        valid_ids = input_ids[row, :valid_text].tolist()
        valid_types = text_types[row, :valid_text].tolist()
        response_positions = [index for index, token_type in enumerate(valid_types) if token_type == 0]
        if not response_positions:
            raise RuntimeError(f"AudioCaps row {row} has no causal response region")
        prompt_length = response_positions[0]
        if response_positions != list(range(prompt_length, valid_text)):
            raise RuntimeError(f"AudioCaps row {row} response token types are not contiguous")
        if set(valid_types[:prompt_length]) != {1} or set(valid_types[prompt_length:]) != {0}:
            raise RuntimeError(f"AudioCaps row {row} PrefixLM/causal token types are invalid")
        if valid_ids[prompt_length - 1] != im_end_id:
            raise RuntimeError(f"AudioCaps row {row} response is not predicted from <|im_end|>")
        expected_full_labels = [-100] * input_ids.shape[1]
        expected_full_labels[prompt_length:valid_text] = valid_ids[prompt_length:valid_text]
        expected_compact = expected_full_labels[-keep:]
        actual_compact = labels[row].tolist()
        if actual_compact != expected_compact:
            raise RuntimeError(f"AudioCaps row {row} compact labels mismatch")
        supervised = [index for index, target in enumerate(actual_compact) if target != -100]
        if not supervised or supervised[0] <= 0 or actual_compact[supervised[-1]] != eos_id:
            raise RuntimeError(f"AudioCaps row {row} compact labels lack shifted-response/EOS supervision")
        full_supervised = [compact_start + index for index in supervised]
        if full_supervised != list(range(prompt_length, valid_text)):
            raise RuntimeError(f"AudioCaps row {row} compact labels do not map exactly to response tokens")
        first_prediction = compact_start + supervised[0] - 1
        if first_prediction != prompt_length - 1:
            raise RuntimeError(f"AudioCaps row {row} first response NTP position is wrong")

        def contains_subsequence(sequence: list[int], subsequence: list[int]) -> bool:
            if not subsequence or len(subsequence) > len(sequence):
                return False
            return any(
                sequence[start : start + len(subsequence)] == subsequence
                for start in range(len(sequence) - len(subsequence) + 1)
            )

        expected_record = fixture_records[row]
        user_tokens = tokenizer(expected_record["user_prompt"], add_special_tokens=False)["input_ids"]
        system_tokens = tokenizer(expected_record["system_prompt"], add_special_tokens=False)["input_ids"]
        caption_tokens = tokenizer(expected_record["caption"], add_special_tokens=False)["input_ids"]
        if not contains_subsequence(valid_ids[:prompt_length], user_tokens):
            raise RuntimeError(f"AudioCaps row {row} user prompt was lost or paired with the wrong audio")
        if not contains_subsequence(valid_ids[prompt_length:], caption_tokens):
            raise RuntimeError(f"AudioCaps row {row} caption was lost or paired with the wrong audio")
        system_in_prompt = contains_subsequence(valid_ids[:prompt_length], system_tokens)

        prefix_end = EXPECTED_AUDIO_PREFIX + prompt_length
        valid_combined = EXPECTED_AUDIO_PREFIX + valid_text
        for query in range(prefix_end):
            if not all(mask_allowed(actual_mask, row, query, key) for key in range(prefix_end)):
                raise RuntimeError(f"AudioCaps row {row} prefix block is not fully bidirectional")
            if prefix_end < valid_combined and mask_allowed(actual_mask, row, query, prefix_end):
                raise RuntimeError(f"AudioCaps row {row} prefix query can see causal response tokens")
            if valid_combined < expected_combined_width and mask_allowed(actual_mask, row, query, valid_combined):
                raise RuntimeError(f"AudioCaps row {row} prefix query can attend padding")
        for query in range(prefix_end, valid_combined):
            if not all(mask_allowed(actual_mask, row, query, key) for key in range(query + 1)):
                raise RuntimeError(f"AudioCaps row {row} response cannot attend its valid causal history")
            if query + 1 < valid_combined and mask_allowed(actual_mask, row, query, query + 1):
                raise RuntimeError(f"AudioCaps row {row} response can attend a future response token")
            if valid_combined < expected_combined_width and mask_allowed(actual_mask, row, query, valid_combined):
                raise RuntimeError(f"AudioCaps row {row} response can attend padding")
        rows.append(
            {
                "row": row,
                "valid_text_tokens": valid_text,
                "text_prompt_tokens": prompt_length,
                "combined_bidirectional_prefix_tokens": prefix_end,
                "causal_response_tokens": valid_text - prompt_length,
                "first_prediction_position_text": first_prediction,
                "first_prediction_position_combined": EXPECTED_AUDIO_PREFIX + first_prediction,
                "sample_id": expected_record["sample_id"],
                "audio_path": expected_record["audio_path"],
                "user_prompt_present": True,
                "caption_present": True,
                "source_system_prompt_present": system_in_prompt,
                "decoded_text": tokenizer.decode(valid_ids, skip_special_tokens=False),
            }
        )

    manual_loss = F.cross_entropy(
        logits[:, :-1, :].float().contiguous().view(-1, logits.shape[-1]),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )
    loss_diff = abs(float(manual_loss.item()) - float(outer["loss"]))
    if loss_diff > 1e-5:
        raise RuntimeError(
            f"Trainer loss is not shifted NTP cross entropy: model={outer['loss']} manual={manual_loss.item()}"
        )
    return {
        "audio_feature_shape": list(audio_features.shape),
        "text_shape": list(input_ids.shape),
        "combined_shape": list(combined_embeds.shape),
        "native_attention_mask_shape": list(actual_mask.shape),
        "audio_prefix_tokens": EXPECTED_AUDIO_PREFIX,
        "logits_to_keep": keep,
        "compact_labels_shape": list(labels.shape),
        "model_loss": float(outer["loss"]),
        "manual_shifted_cross_entropy": float(manual_loss.item()),
        "loss_absolute_difference": loss_diff,
        "rows": rows,
    }


def inspect_optimizer(trainer: Any, model: torch.nn.Module) -> dict[str, Any]:
    if trainer.optimizer is None or trainer.lr_scheduler is None:
        raise RuntimeError("Trainer optimizer/scheduler was not created")
    optimizer_parameters = {
        id(parameter): parameter
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    }
    trainable_parameters = {id(parameter): parameter for parameter in model.parameters() if parameter.requires_grad}
    if set(optimizer_parameters) != set(trainable_parameters):
        raise RuntimeError(
            "Optimizer/trainable parameter sets differ: "
            f"optimizer_only={len(set(optimizer_parameters) - set(trainable_parameters))} "
            f"trainable_only={len(set(trainable_parameters) - set(optimizer_parameters))}"
        )
    parameter_count = sum(parameter.numel() for parameter in optimizer_parameters.values())
    if parameter_count != EXPECTED_TOTAL_TRAINABLE:
        raise RuntimeError(f"Optimizer parameter count mismatch: {parameter_count}")
    learning_rates = [float(group["lr"]) for group in trainer.optimizer.param_groups]
    if not learning_rates or any(not math.isfinite(value) or value <= 0 for value in learning_rates):
        raise RuntimeError(f"Invalid optimizer learning rates: {learning_rates}")
    return {
        "class": f"{type(trainer.optimizer).__module__}.{type(trainer.optimizer).__name__}",
        "parameter_tensor_count": len(optimizer_parameters),
        "parameter_count": parameter_count,
        "group_count": len(trainer.optimizer.param_groups),
        "learning_rates": learning_rates,
        "scheduler": f"{type(trainer.lr_scheduler).__module__}.{type(trainer.lr_scheduler).__name__}",
        "scheduler_last_epoch": int(trainer.lr_scheduler.last_epoch),
    }


def read_safetensors(path: Path) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"Required checkpoint file is missing: {path}")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


def inspect_checkpoint(checkpoint: Path, model: torch.nn.Module, wrapper: torch.nn.Module) -> dict[str, Any]:
    required = (
        checkpoint / "adapter_model.safetensors",
        checkpoint / "adapter_config.json",
        checkpoint / "vit.safetensors",
        checkpoint / "trainer_state.json",
        checkpoint / "optimizer.pt",
        checkpoint / "scheduler.pt",
        checkpoint / "rng_state.pth",
    )
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"HRM audio checkpoint is incomplete: missing={missing}")
    adapter = read_safetensors(checkpoint / "adapter_model.safetensors")
    adapter_canonical = {canonical_adapter_key(key): tensor for key, tensor in adapter.items()}
    h_count = sum(key.startswith("H_module.") for key in adapter_canonical)
    l_count = sum(key.startswith("L_module.") for key in adapter_canonical)
    if len(adapter_canonical) != EXPECTED_LORA_TENSORS or h_count != 256 or l_count != 256:
        raise RuntimeError(
            f"Adapter checkpoint mismatch: total={len(adapter_canonical)} H={h_count} L={l_count}"
        )
    from peft import get_peft_model_state_dict

    runtime_adapter = {
        canonical_adapter_key(key): tensor.detach().cpu()
        for key, tensor in get_peft_model_state_dict(model).items()
    }
    if set(adapter_canonical) != set(runtime_adapter):
        raise RuntimeError("Saved/runtime LoRA keys differ")
    adapter_max_diff = max(
        float((adapter_canonical[key].float() - runtime_adapter[key].float()).abs().max().item())
        for key in adapter_canonical
    )
    if adapter_max_diff != 0.0:
        raise RuntimeError(f"Saved LoRA tensors differ from Trainer model: {adapter_max_diff}")

    aligner = read_safetensors(checkpoint / "vit.safetensors")
    invalid_aligner = [key for key in aligner if canonical_aligner_key(key) is None]
    aligner_canonical = {
        canonical_aligner_key(key): tensor for key, tensor in aligner.items() if canonical_aligner_key(key) is not None
    }
    runtime_aligner = {
        canonical_aligner_key(key): tensor.detach().cpu()
        for key, tensor in wrapper.state_dict().items()
        if canonical_aligner_key(key) is not None
    }
    if invalid_aligner or len(aligner_canonical) != EXPECTED_ALIGNER_TENSORS or set(aligner_canonical) != set(runtime_aligner):
        raise RuntimeError(
            f"Aligner sidecar mismatch: total={len(aligner_canonical)} invalid={invalid_aligner[:20]}"
        )
    aligner_max_diff = max(
        float((aligner_canonical[key].float() - runtime_aligner[key].float()).abs().max().item())
        for key in aligner_canonical
    )
    if aligner_max_diff != 0.0:
        raise RuntimeError(f"Saved aligner tensors differ from Trainer model: {aligner_max_diff}")
    trainer_state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    if int(trainer_state.get("global_step", -1)) != 1:
        raise RuntimeError(f"Checkpoint global_step mismatch: {trainer_state.get('global_step')}")
    files = [
        {"path": str(path.relative_to(checkpoint)), "bytes": path.stat().st_size}
        for path in sorted(checkpoint.rglob("*"))
        if path.is_file()
    ]
    return {
        "path": str(checkpoint),
        "files": files,
        "adapter": {
            "tensor_count": len(adapter_canonical),
            "H_tensor_count": h_count,
            "L_tensor_count": l_count,
            "max_abs_diff_vs_trainer": adapter_max_diff,
            "dtype_counts": dict(Counter(str(tensor.dtype) for tensor in adapter.values())),
        },
        "aligner": {
            "tensor_count": len(aligner_canonical),
            "max_abs_diff_vs_trainer": aligner_max_diff,
            "dtype_counts": dict(Counter(str(tensor.dtype) for tensor in aligner.values())),
            "canonical_keys": sorted(aligner_canonical),
        },
        "trainer_state_global_step": int(trainer_state["global_step"]),
    }


def build_controlled_prefix_batch(
    wrapper: torch.nn.Module,
    inference_batch: dict[str, Any],
    audio_prefix: torch.Tensor,
) -> dict[str, Any]:
    input_ids = inference_batch["input_ids"]
    attention_mask = inference_batch["attention_mask"]
    token_type_ids = inference_batch["token_type_ids"]
    text_embeds = wrapper.get_input_embeddings()(input_ids)
    audio_prefix = audio_prefix.to(device=text_embeds.device, dtype=text_embeds.dtype)
    if audio_prefix.shape != (input_ids.shape[0], EXPECTED_AUDIO_PREFIX, text_embeds.shape[-1]):
        raise RuntimeError(f"Controlled audio prefix shape mismatch: {tuple(audio_prefix.shape)}")
    prefix_attention = torch.ones(
        (input_ids.shape[0], EXPECTED_AUDIO_PREFIX),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    prefix_types = torch.ones(
        (input_ids.shape[0], EXPECTED_AUDIO_PREFIX),
        dtype=token_type_ids.dtype,
        device=token_type_ids.device,
    )
    logits_to_keep = inference_batch["logits_to_keep"]
    if torch.is_tensor(logits_to_keep):
        if logits_to_keep.ndim != 1 or logits_to_keep.dtype != torch.bool:
            raise RuntimeError(
                "Controlled logits_to_keep must be a one-dimensional boolean mask, "
                f"got shape={tuple(logits_to_keep.shape)} dtype={logits_to_keep.dtype}"
            )
        if logits_to_keep.numel() != input_ids.shape[1]:
            raise RuntimeError(
                "Controlled logits_to_keep/text length mismatch: "
                f"mask={logits_to_keep.numel()} text={input_ids.shape[1]}"
            )
        prefix_keep = torch.zeros(
            EXPECTED_AUDIO_PREFIX,
            dtype=torch.bool,
            device=logits_to_keep.device,
        )
        logits_to_keep = torch.cat([prefix_keep, logits_to_keep], dim=0)
    return {
        "inputs_embeds": torch.cat([audio_prefix, text_embeds], dim=1),
        "attention_mask": torch.cat([prefix_attention, attention_mask], dim=1),
        "token_type_ids": torch.cat([prefix_types, token_type_ids], dim=1),
        "logits_to_keep": logits_to_keep,
    }


def main() -> None:
    args = parse_args()
    expected_versions = {
        "ms-swift": "4.4.2",
        "transformers": "5.9.0",
        "torch": "2.11.0+cu128",
        "peft": "0.18.1",
    }
    versions = {name: version(name) for name in expected_versions}
    mismatches = {
        name: {"expected": expected_versions[name], "actual": actual}
        for name, actual in versions.items()
        if actual != expected_versions[name]
    }
    if mismatches:
        raise RuntimeError(f"Unexpected HRM audio Trainer environment: {mismatches}")
    if not torch.cuda.is_available():
        raise RuntimeError("HRM audio Trainer smoke requires CUDA")

    wrapper_model_path = args.wrapper_model_path.resolve()
    plugin_path = args.plugin_path.resolve()
    source_manifest = args.source_manifest.resolve()
    run_dir = args.run_dir.resolve()
    output_report = args.output_report.resolve()
    reload_script = args.reload_script.resolve()
    for path, description in (
        (wrapper_model_path, "wrapper model"),
        (plugin_path, "plugin"),
        (source_manifest, "AudioCaps manifest"),
        (reload_script, "fresh reload script"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {description}: {path}")
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists; refusing to overwrite: {run_dir}")
    run_dir.mkdir(parents=True)
    fixture_path, dataset_report = prepare_smoke_manifest(source_manifest, run_dir)
    swift_output_dir = run_dir / "swift_output"
    reference_payload_path = run_dir / "post_update_reference.pt"
    reload_report_path = run_dir / "fresh_reload_report.json"

    from swift.pipelines.train.sft import SwiftSft

    argv = [
        "--model", str(wrapper_model_path),
        "--model_type", MODEL_TYPE,
        "--template", TEMPLATE_TYPE,
        "--external_plugins", str(plugin_path),
        "--dataset", str(fixture_path),
        "--split_dataset_ratio", "0",
        "--dataset_shuffle", "false",
        "--train_dataloader_shuffle", "false",
        "--sortish_sampler", "false",
        "--group_by_length", "false",
        "--max_length", "192",
        "--output_dir", str(swift_output_dir),
        "--tuner_type", "lora_llm",
        "--tuner_backend", "peft",
        "--target_modules", "all-linear",
        "--freeze_llm", "true",
        "--freeze_vit", "true",
        "--freeze_aligner", "false",
        "--lora_rank", "8",
        "--lora_alpha", "16",
        "--lora_dropout", "0",
        "--learning_rate", "1e-4",
        "--aligner_lr", "1e-4",
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
            wrapper = trainability_audit.find_unique_module(model, "HrmTextAudioForConditionalGeneration")
            tokenizer = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
            wrapper.config.use_cache = False
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
            parameter_report_before = trainability_audit.audit_parameters(model, wrapper)
            frozen_groups = frozen_parameter_groups(model, wrapper)
            frozen_before = {name: parameter_group_digest(entries) for name, entries in frozen_groups.items()}
            trainable_before = snapshot_trainables(model)
            gradient_records, gradient_handles = install_gradient_audit(model)
            outer_capture, core_capture, attention_capture, forward_handles = capture_forward_audit(
                tokenizer, wrapper, model
            )
            recurrence_trace, layer_counts, recurrence_handles = recurrence_audit.install_recurrence_hooks(wrapper.model)

            print("========== HRM AUDIO SWIFT TRAINER PRE-TRAIN AUDIT ==========", flush=True)
            print(f"[trainer] type={type(trainer).__module__}.{type(trainer).__name__}", flush=True)
            print(f"[trainer] output_dir={trainer.args.output_dir}", flush=True)
            print(f"[dataset] {json.dumps(dataset_report, ensure_ascii=False)}", flush=True)
            print(
                f"[trainables] total={parameter_report_before['total']} "
                f"trainable={parameter_report_before['trainable']} groups={parameter_report_before['groups']}",
                flush=True,
            )
            print(f"[frozen-before] {frozen_before}", flush=True)

            torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
            started = time.perf_counter()
            try:
                train_result = super().train(trainer)
            finally:
                for handle in forward_handles + recurrence_handles + gradient_handles:
                    handle.remove()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if int(trainer.state.global_step) != 1:
                raise RuntimeError(f"Trainer global_step mismatch: {trainer.state.global_step}")
            if not outer_capture or not core_capture or not attention_capture:
                raise RuntimeError(
                    f"Incomplete real-forward capture: outer={bool(outer_capture)} core={bool(core_capture)} "
                    f"attention={bool(attention_capture)}"
                )

            forward_report = audit_forward_semantics(
                tokenizer,
                outer_capture,
                core_capture,
                attention_capture,
                dataset_report["fixture_records"],
            )
            recurrence_report = recurrence_audit.validate_recurrence_trace(
                core_model=wrapper.model,
                trace=recurrence_trace,
                layer_counts=layer_counts,
            )
            grad_enabled_outputs = [item for item in recurrence_trace if item["grad_enabled"]]
            missing_recurrent_grads = [
                item["index"]
                for item in grad_enabled_outputs
                if item.get("_output") is None or item["_output"].grad is None
            ]
            if missing_recurrent_grads:
                raise RuntimeError(f"Static-K recurrent outputs lack gradients: {missing_recurrent_grads}")
            gradient_report = summarize_gradients(model, gradient_records)
            update_report = compare_trainable_updates(trainable_before, model)
            optimizer_report = inspect_optimizer(trainer, model)
            parameter_report_after = trainability_audit.audit_parameters(model, wrapper)
            frozen_after = {name: parameter_group_digest(entries) for name, entries in frozen_groups.items()}
            if frozen_before != frozen_after:
                raise RuntimeError(f"Frozen Whisper/HRM weights changed: before={frozen_before} after={frozen_after}")

            checkpoint = Path(trainer.args.output_dir) / "checkpoint-1"
            checkpoint_report = inspect_checkpoint(checkpoint, model, wrapper)
            inference_batch = {
                key: value
                for key, value in outer_capture.items()
                if key in {"input_ids", "attention_mask", "token_type_ids", "audio_input_features", "logits_to_keep"}
            }
            model.eval()
            device = next(model.parameters()).device
            model_batch = {
                key: value.to(device=device) if torch.is_tensor(value) else value
                for key, value in inference_batch.items()
            }
            if torch.is_floating_point(model_batch["audio_input_features"]):
                model_batch["audio_input_features"] = model_batch["audio_input_features"].to(dtype=torch.bfloat16)
            with torch.inference_mode():
                reference_logits = model(**model_batch, use_cache=False).logits.detach().cpu()
                reference_repeat = model(**model_batch, use_cache=False).logits.detach().cpu()
                reference_audio_prefix = wrapper.build_audio_prefix(
                    model_batch["audio_input_features"]
                ).detach()
                reference_audio_prefix_repeat = wrapper.build_audio_prefix(
                    model_batch["audio_input_features"]
                ).detach()
                controlled_batch = build_controlled_prefix_batch(
                    wrapper,
                    model_batch,
                    reference_audio_prefix,
                )
                reference_controlled_logits = model(
                    **controlled_batch,
                    use_cache=False,
                ).logits.detach().cpu()
                reference_controlled_repeat = model(
                    **controlled_batch,
                    use_cache=False,
                ).logits.detach().cpu()
            self_repeat = logits_difference_report(reference_logits, reference_repeat)
            if not self_repeat["exact"]:
                raise RuntimeError(f"Post-update HRM audio model is not self-deterministic: {self_repeat}")
            audio_prefix_self_repeat = logits_difference_report(
                reference_audio_prefix.detach().cpu(),
                reference_audio_prefix_repeat.detach().cpu(),
            )
            if not audio_prefix_self_repeat["exact"]:
                raise RuntimeError(f"Post-update audio prefix is not self-deterministic: {audio_prefix_self_repeat}")
            controlled_self_repeat = logits_difference_report(
                reference_controlled_logits,
                reference_controlled_repeat,
            )
            if not controlled_self_repeat["exact"]:
                raise RuntimeError(f"Post-update controlled HRM path is not self-deterministic: {controlled_self_repeat}")
            torch.save(
                {
                    "batch": inference_batch,
                    "reference_logits": reference_logits,
                    "reference_audio_prefix": reference_audio_prefix.detach().cpu(),
                    "reference_controlled_logits": reference_controlled_logits,
                    "checkpoint": str(checkpoint),
                },
                reference_payload_path,
            )

            for item in recurrence_trace:
                item["_output"] = None

            memory_before_reload = {
                "device": torch.cuda.get_device_name(torch.cuda.current_device()),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
            }
            torch.cuda.empty_cache()
            reload_command = [
                sys.executable,
                "-u",
                str(reload_script),
                "--wrapper-model-path",
                str(wrapper_model_path),
                "--plugin-path",
                str(plugin_path),
                "--checkpoint",
                str(checkpoint),
                "--reference-payload",
                str(reference_payload_path),
                "--output-report",
                str(reload_report_path),
            ]
            reload_env = dict(os.environ)
            reload_env.update(
                {
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                }
            )
            print(f"[fresh-reload-command] {' '.join(reload_command)}", flush=True)
            subprocess.run(reload_command, cwd=str(wrapper_model_path.parents[1]), env=reload_env, check=True)
            fresh_reload_report = json.loads(reload_report_path.read_text(encoding="utf-8"))
            if fresh_reload_report.get("status") != "OK":
                raise RuntimeError(f"Fresh reload report failed: {fresh_reload_report}")

            finite_losses = [
                float(item["loss"])
                for item in trainer.state.log_history
                if "loss" in item and math.isfinite(float(item["loss"]))
            ]
            if not finite_losses:
                raise RuntimeError(f"Trainer produced no finite loss: {trainer.state.log_history}")
            report = {
                "status": "OK",
                "packages": versions,
                "dataset": dataset_report,
                "argv": argv,
                "trainer": {
                    "type": f"{type(trainer).__module__}.{type(trainer).__name__}",
                    "global_step": int(trainer.state.global_step),
                    "elapsed_seconds": elapsed,
                    "finite_losses": finite_losses,
                    "log_history": trainer.state.log_history,
                    "result_type": f"{type(train_result).__module__}.{type(train_result).__name__}",
                },
                "parameters_before": parameter_report_before,
                "parameters_after": parameter_report_after,
                "frozen_before": frozen_before,
                "frozen_after": frozen_after,
                "forward": forward_report,
                "recurrence": recurrence_report,
                "gradients": gradient_report,
                "updates": update_report,
                "optimizer": optimizer_report,
                "checkpoint": checkpoint_report,
                "post_update_self_repeat": self_repeat,
                "post_update_audio_prefix_self_repeat": audio_prefix_self_repeat,
                "post_update_controlled_self_repeat": controlled_self_repeat,
                "fresh_process_reload": fresh_reload_report,
                "memory_before_fresh_reload": memory_before_reload,
            }
            atomic_write_json(output_report, report)
            print("========== HRM AUDIO SWIFT TRAINER POST-TRAIN AUDIT ==========", flush=True)
            print(f"[trainer] global_step=1 finite_losses={finite_losses} elapsed_seconds={elapsed:.3f}", flush=True)
            print(f"[forward] {json.dumps(forward_report, ensure_ascii=False)}", flush=True)
            print(
                f"[recurrence] sequence={[item['stack'] for item in recurrence_report['trace']]} "
                f"static_K={recurrence_report['runtime_static_K']} "
                f"grad_indices={recurrence_report['runtime_grad_stack_indices']}",
                flush=True,
            )
            print(f"[gradients] {json.dumps(gradient_report, ensure_ascii=False)}", flush=True)
            print(f"[updates] {json.dumps(update_report, ensure_ascii=False)}", flush=True)
            print(f"[optimizer] {json.dumps(optimizer_report, ensure_ascii=False)}", flush=True)
            print(f"[frozen-exact] {frozen_before == frozen_after} hashes={frozen_after}", flush=True)
            print(f"[checkpoint] {json.dumps(checkpoint_report, ensure_ascii=False)}", flush=True)
            print(f"[fresh-process-reload] status={fresh_reload_report['status']}", flush=True)
            print(f"[memory] {memory_before_reload}", flush=True)
            print(f"[result] status=OK output_report={output_report}", flush=True)
            return train_result

    print("========== HRM AUDIO REAL AUDIOCAPS ONE-STEP TRAINER SMOKE ==========", flush=True)
    print(f"[python] version={sys.version.split()[0]} executable={sys.executable}", flush=True)
    print(f"[packages] {versions}", flush=True)
    print(f"[source-manifest] {source_manifest}", flush=True)
    print(f"[fixture] {fixture_path}", flush=True)
    print(f"[wrapper-model] {wrapper_model_path}", flush=True)
    print(f"[plugin] {plugin_path}", flush=True)
    print(f"[run-dir] {run_dir}", flush=True)
    print(f"[output-report] {output_report}", flush=True)
    print(
        "[policy] real AudioCaps-v2 B=2 GA=1 one update; frozen Whisper+HRM; "
        "trainable aligner+rank8 H/L LoRA; strict PrefixLM/NTP/save/fresh-reload audit",
        flush=True,
    )
    print("[argv] " + " ".join(argv), flush=True)
    AuditedSwiftSft(argv).main()


if __name__ == "__main__":
    main()
