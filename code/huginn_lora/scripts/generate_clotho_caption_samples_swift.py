#!/usr/bin/env python3
"""Generate deterministic Clotho caption samples with a Swift Huginn audio checkpoint."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import random
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any

import torch


DEFAULT_CHECKPOINT = (
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/"
    "huginn_audio_audiocaps_v2_train_e5_b8ga4_5090/v0-20260713-155848/checkpoint-8406"
)
DEFAULT_DATASET_DIR = "/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_caption_huginn"
DEFAULT_PLUGIN_PATH = (
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/"
    "code/huginn_lora/plugins/huginn_audio_swift.py"
)
ALIGNER_PREFIXES = (
    "temporal_compressor.",
    "audio_projector.",
    "audio_boundary_embeddings.",
    "audio_bos",
    "audio_eos",
)
SKIP_STATE_TOKENS = ("optimizer", "scheduler", "rng", "trainer_state", "training_args")
FSDP_MODEL_DIR_NAME = "pytorch_model_fsdp_0"
FULL_TRAINED_PREFIXES = (
    "transformer.",
    "lm_head.",
    "temporal_compressor.",
    "audio_projector.",
    "audio_boundary_embeddings.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--eval-manifest", default="test_expand.jsonl")
    parser.add_argument("--plugin-path", default=DEFAULT_PLUGIN_PATH)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=74)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--fsdp-export-dir",
        default=None,
        help="Deprecated compatibility option; FSDP checkpoints are restored tensor-by-tensor without an export.",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def import_plugin(plugin_path: str) -> ModuleType:
    path = Path(plugin_path)
    if not path.is_file():
        raise FileNotFoundError(f"Plugin not found: {path}")
    spec = importlib.util.spec_from_file_location("huginn_audio_caption_plugin", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import plugin from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def state_dict_from_file(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        payload = load_file(str(path), device="cpu")
    else:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
            payload = payload["state_dict"]
    if not isinstance(payload, dict):
        return {}
    return {key: value for key, value in payload.items() if isinstance(key, str) and torch.is_tensor(value)}


def candidate_target_keys(source_key: str) -> list[str]:
    candidates = {source_key}
    changed = True
    while changed:
        changed = False
        for key in list(candidates):
            for prefix in (
                "base_model.model.",
                "base_model.",
                "model.",
                "module.",
                "_fsdp_wrapped_module.",
            ):
                if key.startswith(prefix):
                    stripped = key[len(prefix):]
                    if stripped not in candidates:
                        candidates.add(stripped)
                        changed = True
    normalized = set()
    for key in candidates:
        normalized.add(key)
        normalized.add(key.replace(".modules_to_save.default.", "."))
        normalized.add(key.replace(".original_module.", "."))
    return list(normalized)


def load_aligner_state(model: torch.nn.Module, checkpoint_dir: Path) -> dict[str, Any]:
    target_state = model.state_dict()
    canonical_targets: dict[str, str] = {}
    for target_key in target_state:
        for candidate in candidate_target_keys(target_key):
            if candidate.startswith(ALIGNER_PREFIXES):
                canonical_targets.setdefault(candidate, target_key)

    selected: dict[str, torch.Tensor] = {}
    source_keys: list[str] = []
    for path in sorted(checkpoint_dir.rglob("*")):
        if not path.is_file() or path.suffix not in {".safetensors", ".bin", ".pt", ".pth"}:
            continue
        if any(token in path.name.lower() for token in SKIP_STATE_TOKENS):
            continue
        for source_key, tensor in state_dict_from_file(path).items():
            for target_key in candidate_target_keys(source_key):
                actual_target = canonical_targets.get(target_key)
                if actual_target is None or target_state[actual_target].shape != tensor.shape:
                    continue
                selected[actual_target] = tensor
                source_keys.append(source_key)
                break
    if not selected:
        raise RuntimeError(f"No aligner tensors could be recovered from checkpoint: {checkpoint_dir}")

    load_result = model.load_state_dict(selected, strict=False)
    boundary_targets = [key for key in selected if key.endswith((".audio_bos", ".audio_eos")) or key in {"audio_bos", "audio_eos"}]
    return {
        "loaded_aligner_tensor_count": len(selected),
        "source_key_preview": source_keys[:20],
        "restored_boundary_embeddings": boundary_targets,
        "missing_key_count": len(load_result.missing_keys),
        "unexpected_key_count": len(load_result.unexpected_keys),
    }


def is_fsdp_sharded_checkpoint(checkpoint_path: Path) -> bool:
    return (checkpoint_path / FSDP_MODEL_DIR_NAME).is_dir()


def infer_fsdp_lora_model(
    base_model: torch.nn.Module,
    state_metadata: dict[str, Any],
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Recreate the LoRA wrapper from FSDP DCP metadata for LoSATok eval.

    Swift's FSDP2 DCP checkpoint does not necessarily contain adapter_config.json.
    The model state still exposes lora_A/lora_B keys, so infer the exact target
    module paths and rank without loading any tensor payload first.
    """
    lora_keys = [str(key) for key in state_metadata if ".lora_A." in str(key) or ".lora_B." in str(key)]
    if not lora_keys:
        raise RuntimeError("LoSATok FSDP checkpoint contains no lora_A/lora_B metadata keys")

    module_names = set(dict(base_model.named_modules()))
    target_modules: set[str] = set()
    ranks: set[int] = set()
    lora_tensor_count = 0
    for source_key in lora_keys:
        metadata = state_metadata[source_key]
        shape_value = getattr(metadata, "size", None)
        shape = tuple(int(value) for value in shape_value) if shape_value is not None else ()
        if ".lora_A." in source_key:
            if not shape:
                raise RuntimeError(f"LoRA A metadata has no shape: {source_key}")
            ranks.add(int(shape[0]))
        for candidate in candidate_target_keys(source_key):
            if ".lora_A." in candidate:
                module_path = candidate.split(".lora_A.", 1)[0]
            elif ".lora_B." in candidate:
                module_path = candidate.split(".lora_B.", 1)[0]
            else:
                continue
            if module_path in module_names:
                target_modules.add(module_path)
                break
        lora_tensor_count += 1

    if not target_modules:
        raise RuntimeError("Unable to map FSDP LoRA metadata keys to base-model module paths")
    if len(ranks) != 1:
        raise RuntimeError(f"LoRA ranks are inconsistent in FSDP metadata: {sorted(ranks)}")

    from peft import LoraConfig, TaskType, get_peft_model

    rank = next(iter(ranks))
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.05,
        target_modules=sorted(target_modules),
        bias="none",
    )
    peft_model = get_peft_model(base_model, config)
    injected_targets = [name for name, _ in peft_model.named_modules() if "lora_A" in name]
    if len(injected_targets) != len(target_modules):
        raise RuntimeError(
            "Recreated LoRA target count does not match DCP metadata: "
            f"metadata_targets={len(target_modules)} injected_targets={len(injected_targets)}"
        )
    return peft_model, {
        "lora_metadata_key_count": lora_tensor_count,
        "lora_rank": rank,
        "lora_alpha": rank * 2,
        "target_module_count": len(target_modules),
        "target_module_preview": sorted(target_modules)[:20],
        "injected_lora_module_count": len(injected_targets),
    }


def load_full_fsdp_base_model(
    plugin: ModuleType,
    checkpoint_path: Path,
    device: torch.device,
    fsdp_export_dir: str | None = None,
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    source_dir = checkpoint_path / FSDP_MODEL_DIR_NAME
    if not source_dir.is_dir():
        raise FileNotFoundError(f"FSDP model checkpoint directory not found: {source_dir}")
    if fsdp_export_dir:
        print(f"[fsdp-stream] ignored_deprecated_export_dir={fsdp_export_dir}")

    try:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint import FileSystemReader
    except Exception as exc:
        raise RuntimeError("torch.distributed.checkpoint is required for FSDP streaming evaluation") from exc

    is_losatok = getattr(plugin, "MODEL_TYPE", None) == "huginn_losatok_raven"
    # The base model stays on CPU until every DCP tensor has been copied into it.
    # This avoids Accelerate's all-at-once merge, which exceeds the queue's 32G cap.
    metadata = FileSystemReader(str(source_dir)).read_metadata()
    state_metadata = getattr(metadata, "state_dict_metadata", {})
    if not state_metadata:
        raise RuntimeError(f"FSDP DCP metadata has no state entries: {source_dir}")

    base_model = (
        plugin.build_huginn_losatok_evaluation_model()
        if is_losatok else plugin.build_huginn_audio_model(str(plugin.AUDIO_MODEL_DIR))
    )
    peft_model = None
    lora_report: dict[str, Any] = {}
    restore_model: torch.nn.Module = base_model
    if is_losatok:
        peft_model, lora_report = infer_fsdp_lora_model(base_model, state_metadata)
        restore_model = peft_model
    target_state = restore_model.state_dict()

    restore_plan: list[tuple[str, str, tuple[int, ...], torch.dtype]] = []
    unmatched_source_keys: list[str] = []
    for source_key, tensor_metadata in state_metadata.items():
        source_key = str(source_key)
        shape_value = getattr(tensor_metadata, "size", None)
        properties = getattr(tensor_metadata, "properties", None)
        source_dtype = getattr(properties, "dtype", None)
        if shape_value is None or not isinstance(source_dtype, torch.dtype):
            continue
        shape = tuple(int(size) for size in shape_value)
        matched_target_key = None
        for target_key in candidate_target_keys(source_key):
            if target_key in target_state and tuple(target_state[target_key].shape) == shape:
                matched_target_key = target_key
                break
        if matched_target_key is None:
            unmatched_source_keys.append(source_key)
            continue
        restore_plan.append((source_key, matched_target_key, shape, source_dtype))
    if not restore_plan:
        raise RuntimeError(f"No FSDP DCP tensors matched the Huginn audio model: {source_dir}")

    print(f"[fsdp-stream] source_dir={source_dir}")
    print(
        f"[fsdp-stream] metadata_tensor_count={len(state_metadata)} "
        f"restore_tensor_count={len(restore_plan)} unmatched_source_count={len(unmatched_source_keys)}"
    )
    restored_target_keys: set[str] = set()
    source_key_preview: list[str] = []
    for index, (source_key, target_key, shape, source_dtype) in enumerate(restore_plan, start=1):
        # DCP reads only the requested key. Do not accumulate these temporary buffers.
        streamed_tensor = torch.empty(shape, dtype=source_dtype, device="cpu")
        dcp.load({source_key: streamed_tensor}, checkpoint_id=str(source_dir))
        with torch.no_grad():
            target_state[target_key].copy_(streamed_tensor.to(dtype=target_state[target_key].dtype))
        restored_target_keys.add(target_key)
        if len(source_key_preview) < 20:
            source_key_preview.append(source_key)
        del streamed_tensor
        if index == 1 or index % 25 == 0 or index == len(restore_plan):
            print(f"[fsdp-stream] restored_tensors={index}/{len(restore_plan)}", flush=True)

    normalized_restored_keys = {
        candidate for key in restored_target_keys for candidate in candidate_target_keys(key)
    }
    if is_losatok:
        base_target_keys = set(base_model.state_dict())
        missing_critical = [
            key
            for key in base_target_keys
            if key.startswith(ALIGNER_PREFIXES)
            and not any(candidate in normalized_restored_keys for candidate in candidate_target_keys(key))
        ]
        lora_target_keys = [key for key in target_state if ".lora_A." in key or ".lora_B." in key]
        missing_lora = [
            key
            for key in lora_target_keys
            if not any(candidate in normalized_restored_keys for candidate in candidate_target_keys(key))
        ]
        missing_critical.extend(missing_lora)
    else:
        missing_critical = [
            key for key in target_state if key.startswith(FULL_TRAINED_PREFIXES) and key not in restored_target_keys
        ]
    # Huginn ties lm_head.weight to transformer.wte.weight. Some FSDP state-dict
    # exports retain only one side of this alias, which is still a complete restore.
    if (
        "lm_head.weight" in missing_critical
        and "transformer.wte.weight" in restored_target_keys
        and bool(getattr(base_model.config, "tie_embeddings", False))
    ):
        missing_critical.remove("lm_head.weight")
    if missing_critical:
        raise RuntimeError(
            f"FSDP streaming restore did not recover critical trained tensors: {missing_critical[:20]} "
            f"(total={len(missing_critical)})"
        )
    missing_target_keys = [key for key in target_state if key not in restored_target_keys]
    if any(parameter.requires_grad for parameter in base_model.audio_encoder.parameters()):
        raise RuntimeError("Audio encoder unexpectedly became trainable during FSDP generation restore")

    restore_model.to(device=device, dtype=torch.bfloat16)
    restore_model.eval()
    model = restore_model.base_model.model if peft_model is not None else restore_model
    processor = (
        plugin.build_huginn_losatok_evaluation_processor()
        if is_losatok else plugin.build_huginn_audio_processor()
    )
    aligner_loaded = sum(
        1
        for key in base_model.state_dict()
        if key.startswith(ALIGNER_PREFIXES)
        and any(candidate in normalized_restored_keys for candidate in candidate_target_keys(key))
    )
    boundary_loaded = [
        key for key in base_model.state_dict()
        if (key.endswith((".audio_bos", ".audio_eos")) or key in {"audio_bos", "audio_eos"})
        and any(candidate in normalized_restored_keys for candidate in candidate_target_keys(key))
    ]
    return model, processor, {
        "checkpoint_dir": str(checkpoint_path),
        "checkpoint_format": "fsdp_sharded_streaming",
        "lora_restored": peft_model is not None,
        "lora_restore": lora_report,
        "full_state_restored": True,
        "fsdp_source_dir": str(source_dir),
        "dcp_metadata_tensor_count": len(state_metadata),
        "restored_tensor_count": len(restored_target_keys),
        "unmatched_source_key_count": len(unmatched_source_keys),
        "unmatched_source_key_preview": unmatched_source_keys[:20],
        "source_key_preview": source_key_preview,
        "missing_key_count": len(missing_target_keys),
        "unexpected_key_count": len(unmatched_source_keys),
        "audio_encoder_trainable_parameter_count": sum(
            parameter.numel() for parameter in base_model.audio_encoder.parameters() if parameter.requires_grad
        ),
        "aligner_restore": {
            "loaded_aligner_tensor_count": aligner_loaded,
            "restored_boundary_embeddings": boundary_loaded,
        },
    }


def load_generation_model(
    plugin: ModuleType,
    checkpoint_dir: str,
    device: torch.device,
    fsdp_export_dir: str | None = None,
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
    is_losatok = getattr(plugin, "MODEL_TYPE", None) == "huginn_losatok_raven"
    if is_fsdp_sharded_checkpoint(checkpoint_path):
        return load_full_fsdp_base_model(plugin, checkpoint_path, device, fsdp_export_dir)

    adapter_path = checkpoint_path / "adapter_model.safetensors"
    if not adapter_path.is_file():
        raise FileNotFoundError(f"LoRA adapter file not found: {adapter_path}")

    base_model = (
        plugin.build_huginn_losatok_evaluation_model()
        if is_losatok else plugin.build_huginn_audio_model(str(plugin.AUDIO_MODEL_DIR))
    )
    aligner_report = load_aligner_state(base_model, checkpoint_path)
    if any(parameter.requires_grad for parameter in base_model.audio_encoder.parameters()):
        raise RuntimeError("Audio encoder unexpectedly became trainable during generation restore")

    from peft import PeftModel

    peft_model = PeftModel.from_pretrained(base_model, str(checkpoint_path), is_trainable=False)
    peft_model.to(device=device, dtype=torch.bfloat16)
    peft_model.eval()
    # PEFT injects LoRA layers into this underlying model. Calling it directly
    # preserves LoRA while exposing the plugin-specific audio inputs.
    model = peft_model.base_model.model
    if not hasattr(model, "audio_encoder") or not hasattr(model, "audio_projector"):
        raise TypeError(f"Unexpected PEFT base model type: {type(model)}")
    model.eval()
    processor = (
        plugin.build_huginn_losatok_evaluation_processor()
        if is_losatok else plugin.build_huginn_audio_processor()
    )
    lora_tensor_count = len(state_dict_from_file(adapter_path))
    injected_lora_module_count = sum(1 for name, _ in model.named_modules() if "lora_A" in name)
    if injected_lora_module_count == 0:
        raise RuntimeError("LoRA restoration produced no injected lora_A modules in the generation model")
    return model, processor, {
        "checkpoint_dir": str(checkpoint_path),
        "checkpoint_format": "lora_adapter",
        "lora_restored": True,
        "lora_tensor_count": lora_tensor_count,
        "injected_lora_module_count": injected_lora_module_count,
        "audio_encoder_trainable_parameter_count": sum(
            parameter.numel() for parameter in base_model.audio_encoder.parameters() if parameter.requires_grad
        ),
        "aligner_restore": aligner_report,
    }


def as_text_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return []


def record_captions(record: dict[str, Any]) -> list[str]:
    captions: list[str] = []
    for key in ("references", "captions", "caption_list", "ref_captions", "caption", "text"):
        captions.extend(as_text_list(record.get(key)))
    if not captions and isinstance(record.get("messages"), list):
        captions.extend(
            message.get("content", "").strip()
            for message in record["messages"]
            if isinstance(message, dict)
            and message.get("role") == "assistant"
            and isinstance(message.get("content"), str)
            and message["content"].strip()
        )
    return list(dict.fromkeys(captions))


def load_clotho_groups(dataset_dir: str, manifest_name: str) -> list[tuple[Path, list[str]]]:
    root = Path(dataset_dir)
    manifest_path = root / manifest_name
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Clotho manifest not found: {manifest_path}")
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    grouped: dict[Path, list[str]] = defaultdict(list)
    for line_number, record in enumerate(records, start=1):
        raw_path = record.get("audio_path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"{manifest_path}:{line_number} has no audio_path")
        audio_path = Path(raw_path)
        if not audio_path.is_absolute():
            audio_path = root / audio_path
        captions = record_captions(record)
        if not captions:
            raise ValueError(f"{manifest_path}:{line_number} has no reference caption")
        grouped[audio_path].extend(captions)

    groups = [(audio_path, list(dict.fromkeys(captions))) for audio_path, captions in sorted(grouped.items())]
    for audio_path, _ in groups:
        if not audio_path.is_file():
            raise FileNotFoundError(f"Clotho audio file not found: {audio_path}")
    if not groups:
        raise ValueError(f"No audio groups found in {manifest_path}")
    return groups


def build_prompt(plugin: ModuleType) -> str:
    return (
        "<|begin_header|>system<|end_header|>\n\n"
        f"{plugin.DEFAULT_SYSTEM_PROMPT}<|end_turn|>"
        "<|begin_header|>user<|end_header|>\n\n"
        "Listen to the audio and describe it.<|end_turn|>"
        "<|begin_header|>Huginn<|end_header|>\n\n"
    )


def generate_caption(
    plugin: ModuleType,
    model: torch.nn.Module,
    processor: Any,
    audio_path: Path,
    max_new_tokens: int,
    device: torch.device,
) -> dict[str, Any]:
    tokenizer = processor.tokenizer
    is_losatok = getattr(plugin, "MODEL_TYPE", None) == "huginn_losatok_raven"
    if is_losatok:
        waveform = plugin.load_audio_16k(audio_path)
        sample_rate = plugin.DEFAULT_SAMPLE_RATE
        audio_inputs = {
            "audio_input_values": waveform.unsqueeze(0).to(device=device, dtype=torch.float32),
            "audio_attention_mask": torch.ones((1, waveform.numel()), device=device, dtype=torch.long),
        }
    else:
        feature_extractor = processor.feature_extractor
        sample_rate = int(getattr(feature_extractor, "sampling_rate", plugin.DEFAULT_SAMPLE_RATE))
        waveform = plugin.load_audio_file(audio_path, sample_rate, plugin.DEFAULT_MAX_AUDIO_SECONDS)
        features = feature_extractor([waveform], sampling_rate=sample_rate, return_tensors="pt")["input_features"]
        audio_inputs = {"audio_input_features": features.to(device=device, dtype=torch.bfloat16)}
    prompt = build_prompt(plugin)
    tokenized = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    stop_token_ids = {
        token_id
        for token_id in (getattr(tokenizer, "eos_token_id", None), getattr(model.config, "eos_token_id", None))
        if token_id is not None
    }
    # These are the Huginn generation stop tokens used by the base model.
    stop_token_ids.update({65504, 65505, 65508})

    with torch.inference_mode():
        # Audio is injected exactly once into this prefill. The base forward creates
        # a cache whose length includes the 34-token audio prefix and the text prompt.
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            **audio_inputs,
        )
        if outputs.logits is None or outputs.past_key_values is None:
            raise RuntimeError("Audio prefill did not return logits and a Huginn KV cache")
        cache = outputs.past_key_values
        prefill_cache_length = int(cache.get_seq_length())
        new_token_ids: list[int] = []
        stop_reason = "max_new_tokens"

        for _ in range(max_new_tokens):
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            token_id = int(next_token.item())
            if token_id in stop_token_ids:
                stop_reason = f"stop_token:{token_id}"
                break
            new_token_ids.append(token_id)
            if len(new_token_ids) == max_new_tokens:
                break

            # The next token's RoPE position is the actual cache length, including
            # audio prefix tokens, rather than the text-only position maintained by
            # Transformers' generic generate loop.
            cache_position = torch.tensor([cache.get_seq_length()], device=device, dtype=torch.long)
            outputs = model(
                input_ids=next_token,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
            )
            if outputs.logits is None or outputs.past_key_values is None:
                raise RuntimeError("Cached Huginn decode did not return logits and cache")
            cache = outputs.past_key_values

    new_token_tensor = torch.tensor(new_token_ids, device=device, dtype=torch.long)
    return {
        "prompt": prompt,
        "audio_seconds_after_truncation": len(waveform) / float(sample_rate),
        "text_prompt_token_count": int(input_ids.shape[1]),
        "audio_prefix_token_count": prefill_cache_length - int(input_ids.shape[1]),
        "prefill_cache_length": prefill_cache_length,
        "final_cache_length": int(cache.get_seq_length()),
        "stop_reason": stop_reason,
        "generated_token_count": len(new_token_ids),
        "generated_caption": tokenizer.decode(new_token_tensor, skip_special_tokens=True).strip(),
    }


def main() -> None:
    args = parse_args()
    if args.sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Huginn audio generation")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = load_clotho_groups(args.dataset_dir, args.eval_manifest)
    selected_indices = sorted(random.Random(args.seed).sample(range(len(groups)), min(args.sample_count, len(groups))))
    selected_groups = [groups[index] for index in selected_indices]

    print("========== HUGINN CLOTHO CAPTION GENERATION ==========")
    print(f"[config] checkpoint={args.checkpoint}")
    print(f"[config] dataset_dir={args.dataset_dir}")
    print(f"[config] eval_manifest={args.eval_manifest}")
    print(f"[config] available_audio_groups={len(groups)} selected_indices={selected_indices}")
    print(f"[config] max_new_tokens={args.max_new_tokens} generation_path=audio_manual_cache seed={args.seed}")
    plugin = import_plugin(args.plugin_path)
    model, processor, restore = load_generation_model(plugin, args.checkpoint, device, args.fsdp_export_dir)
    print(f"[restore] {json.dumps(restore, ensure_ascii=False)}")
    if not restore["aligner_restore"]["restored_boundary_embeddings"]:
        print("[warning] audio_bos/audio_eos were not found in the checkpoint; model initialization values are in use.")

    samples: list[dict[str, Any]] = []
    for sample_number, (audio_path, references) in enumerate(selected_groups, start=1):
        generated = generate_caption(
            plugin, model, processor, audio_path, args.max_new_tokens, device
        )
        sample = {
            "sample_number": sample_number,
            "audio_path": str(audio_path),
            "reference_count": len(references),
            "references": references,
            **generated,
        }
        samples.append(sample)
        print(f"========== SAMPLE {sample_number} ==========")
        print(f"[audio] path={audio_path}")
        print(f"[audio] seconds_after_truncation={generated['audio_seconds_after_truncation']:.3f}")
        print(
            "[generation] "
            f"prompt_tokens={generated['text_prompt_token_count']} "
            f"audio_prefix_tokens={generated['audio_prefix_token_count']} "
            f"cache={generated['prefill_cache_length']}->{generated['final_cache_length']} "
            f"stop_reason={generated['stop_reason']}"
        )
        print(f"[generation] token_count={generated['generated_token_count']}")
        print(f"[generation] caption={generated['generated_caption']}")
        for reference_number, reference in enumerate(references, start=1):
            print(f"[reference {reference_number}] {reference}")

    payload = {
        "checkpoint": args.checkpoint,
        "max_new_tokens": args.max_new_tokens,
        "generation_path": "audio_manual_cache",
        "restore": restore,
        "samples": samples,
    }
    output_path = output_dir / "clotho_caption_samples.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("========== HUGINN CLOTHO CAPTION GENERATION DONE ==========")
    print(f"[output] {output_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
