"""Swift registration for isolated Whisper-large dynamic-90s Huginn audio."""

from __future__ import annotations

import io
import json
import math
import os
import random
import shutil
import subprocess
import tarfile
import wave
from collections import OrderedDict
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from types import MethodType
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from swift.model import Model, ModelGroup, ModelLoader, ModelMeta, register_model

try:
    from swift.model import MultiModelKeys, register_model_arch
except ImportError:
    from swift.llm import MultiModelKeys, register_model_arch  # type: ignore

try:
    from swift.template import StdTemplateInputs, Template, TemplateMeta, register_template
except ImportError:
    from swift.llm import StdTemplateInputs, Template, TemplateMeta, register_template  # type: ignore

try:
    from swift.utils import Processor
except ImportError:
    Processor = Any  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[3]
AUDIO_MODEL_DIR = REPO_ROOT / "models" / "huginn-audio-whisper-dynamic90s-v1"
HUGINN_MODEL_DIR = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/huginn-0125")
WHISPER_MODEL_DIR = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/whisper-large")

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that can understand audio and respond accurately."
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_AUDIO_CHUNK_SECONDS = 30.0
DEFAULT_MAX_AUDIO_SECONDS = 90.0
WHISPER_MAX_FEATURE_FRAMES = 3000
WHISPER_FEATURE_HOP_LENGTH = 160
WHISPER_ENCODER_DOWNSAMPLE = 2
DYNAMIC_COMPRESSOR_KERNEL = 6
DYNAMIC_COMPRESSOR_STRIDE = 6
AUDIO_TOKEN_DURATION_MS = 120
DYNAMIC_LORA_DROPOUT = 0.05
_DERIVED_AUDIO_TOKEN_DURATION_MS = (
    WHISPER_FEATURE_HOP_LENGTH
    * WHISPER_ENCODER_DOWNSAMPLE
    * DYNAMIC_COMPRESSOR_STRIDE
    * 1000
    // DEFAULT_SAMPLE_RATE
)
if _DERIVED_AUDIO_TOKEN_DURATION_MS != AUDIO_TOKEN_DURATION_MS:
    raise RuntimeError(
        "Whisper dynamic token contract is inconsistent: "
        f"derived={_DERIVED_AUDIO_TOKEN_DURATION_MS}ms configured={AUDIO_TOKEN_DURATION_MS}ms"
    )
INIT_ALIGNER_CHECKPOINT_ENV = "HUGINN_AUDIO_DYNAMIC90S_INIT_ALIGNER_CHECKPOINT"
FORCE_ALIGNER_TRAINABLE_ENV = "HUGINN_AUDIO_DYNAMIC90S_FORCE_ALIGNER_TRAINABLE"
FSDP2_NONPERSISTENT_ROPE_ENV = "HUGINN_AUDIO_DYNAMIC90S_FSDP2_NONPERSISTENT_ROPE"
TRAIN_CHAIN_AUDIT_ENV = "HUGINN_AUDIO_DYNAMIC90S_TRAIN_CHAIN_AUDIT"
STAGE34_AUDIT_DIR_ENV = "HUGINN_AUDIO_DYNAMIC90S_STAGE34_AUDIT_DIR"
STAGE5_AUDIT_DIR_ENV = "HUGINN_AUDIO_DYNAMIC90S_STAGE5_AUDIT_DIR"
STAGE5_MAX_STEPS_ENV = "HUGINN_AUDIO_DYNAMIC90S_STAGE5_MAX_STEPS"
REALDATA_AUDIT_DIR_ENV = "HUGINN_AUDIO_DYNAMIC90S_REALDATA_AUDIT_DIR"
REALDATA_MAX_STEPS_ENV = "HUGINN_AUDIO_DYNAMIC90S_REALDATA_MAX_STEPS"
DATA_POSITION_AUDIT_DIR_ENV = "HUGINN_AUDIO_DYNAMIC90S_DATA_POSITION_AUDIT_DIR"
DATA_POSITION_AUDIT_PHASE_ENV = "HUGINN_AUDIO_DYNAMIC90S_DATA_POSITION_AUDIT_PHASE"
PEFT_ALIGNER_MODULES_TO_SAVE_ENV = "HUGINN_AUDIO_DYNAMIC90S_PEFT_ALIGNER_MODULES_TO_SAVE"
FSDP_SAVE_DEBUG_ENV = "HUGINN_AUDIO_DYNAMIC90S_FSDP_SAVE_DEBUG"
CHECKPOINT_AUDIT_DIR_ENV = "HUGINN_AUDIO_DYNAMIC90S_CHECKPOINT_AUDIT_DIR"
CHECKPOINT_PHASE_ENV = "HUGINN_AUDIO_DYNAMIC90S_CHECKPOINT_PHASE"
CHECKPOINT_START_STEP_ENV = "HUGINN_AUDIO_DYNAMIC90S_CHECKPOINT_START_STEP"
CHECKPOINT_END_STEP_ENV = "HUGINN_AUDIO_DYNAMIC90S_CHECKPOINT_END_STEP"
CHECKPOINT_LAUNCH_ID_ENV = "HUGINN_AUDIO_DYNAMIC90S_CHECKPOINT_LAUNCH_ID"
TRAINING_STATS_DIR_ENV = "HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_DIR"
TRAINING_STATS_RESUME_STATE_ENV = "HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_RESUME_STATE"
TRAINING_STATS_LOG_STEPS_ENV = "HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_LOG_STEPS"
TRAINING_STATS_CHECKPOINT_STEPS_ENV = "HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_CHECKPOINT_STEPS"
TRAINING_STATS_FORWARD_AUDIT_DIR_ENV = "HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_FORWARD_AUDIT_DIR"
TRAINING_STATS_PHASE_ENV = "HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_PHASE"
TRAINING_STATS_STATE_FILENAME = "audio_training_statistics.json"
TRAINING_STATS_VERSION = "huginn_dynamic90s_training_statistics_v1"
SAMPLER_VERSION = "deterministic_hierarchical_no_replacement_v2"
TRAINING_POOL_ORDER = (
    "wavcaps_no_bbc_aac",
    "audiocaps_v2_aac",
    "clotho_v2_aac",
    "gigaspeech_l_asr",
)
TRAINING_POOL_TO_ID = {name: index for index, name in enumerate(TRAINING_POOL_ORDER)}
ALIGNER_MODULES_TO_SAVE = (
    "temporal_compressor",
    "audio_projector",
    "audio_boundary_embeddings",
)
FSDP_UNIT_CLASS_NAMES = (
    "WhisperEncoderFSDPUnit",
    "AudioAlignerFSDPUnit",
    "HuginnPreludeFSDPUnit",
    "HuginnRecurrentCoreFSDPUnit",
    "HuginnCodaFSDPUnit",
)
FSDP_UNIT_EXPECTED_TRAINABLE_TENSORS = {
    "WhisperEncoderFSDPUnit": 0,
    "AudioAlignerFSDPUnit": 14,
    "HuginnPreludeFSDPUnit": 16,
    "HuginnRecurrentCoreFSDPUnit": 34,
    "HuginnCodaFSDPUnit": 16,
}


def get_tarfile_cache_limit() -> int:
    value = os.environ.get("HUGINN_AUDIO_TARFILE_CACHE_LIMIT", "4")
    try:
        cache_limit = int(value)
    except ValueError as exc:
        raise ValueError(f"HUGINN_AUDIO_TARFILE_CACHE_LIMIT must be an integer, got {value!r}") from exc
    if cache_limit <= 0:
        raise ValueError(f"HUGINN_AUDIO_TARFILE_CACHE_LIMIT must be positive, got {cache_limit}")
    return cache_limit


TARFILE_CACHE_LIMIT = get_tarfile_cache_limit()

ALIGNER_PREFIXES = (
    "audio_aligner",
    "temporal_compressor",
    "audio_projector",
    "audio_boundary_embeddings",
    "audio_bos",
    "audio_eos",
)


def normalize_parameter_name(name: str) -> str:
    normalized = name
    changed = True
    while changed:
        changed = False
        for prefix in ("base_model.model.", "base_model.", "model.", "module."):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                changed = True
    normalized = normalized.replace(".modules_to_save.default.", ".")
    normalized = normalized.replace(".original_module.", ".")
    return normalized


def is_aligner_parameter_name(name: str) -> bool:
    return normalize_parameter_name(name).startswith(ALIGNER_PREFIXES)

MODEL_TYPE = "huginn_audio_whisper_dynamic90s"
TEMPLATE_TYPE = "huginn_audio_whisper_dynamic90s"
MODEL_ARCH_NAME = "huginn_audio_whisper_dynamic90s"
_TARFILE_CACHE: "OrderedDict[str, tarfile.TarFile]" = OrderedDict()
print(f"[HuginnAudioSwift] tarfile_cache_limit={TARFILE_CACHE_LIMIT}")


def _requested(environment_name: str) -> bool:
    return os.environ.get(environment_name, "").strip().lower() in {"1", "true", "yes"}


def _training_tensor(kwargs: dict[str, Any], name: str) -> torch.Tensor | None:
    value = kwargs.pop(name, None)
    if value is not None and not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor when provided, got {type(value)}")
    return value


def _record_actual_training_samples(
    model: torch.nn.Module,
    global_positions: torch.Tensor | None,
    pool_ids: torch.Tensor | None,
    record_indices: torch.Tensor | None,
    pool_occurrence_indices: torch.Tensor | None,
    pool_epochs: torch.Tensor | None,
    effective_durations: torch.Tensor | None,
) -> None:
    fields = {
        "global_positions": global_positions,
        "pool_ids": pool_ids,
        "record_indices": record_indices,
        "pool_occurrence_indices": pool_occurrence_indices,
        "pool_epochs": pool_epochs,
        "effective_durations": effective_durations,
    }
    if all(value is None for value in fields.values()):
        return
    missing = [name for name, value in fields.items() if value is None]
    if missing:
        raise RuntimeError(f"Training statistics metadata is incomplete: missing={missing}")
    flattened = {name: value.detach().reshape(-1).cpu() for name, value in fields.items() if value is not None}
    batch_sizes = {tensor.numel() for tensor in flattened.values()}
    if len(batch_sizes) != 1 or next(iter(batch_sizes)) <= 0:
        raise RuntimeError(
            f"Training statistics metadata batch sizes differ: "
            f"{ {name: tensor.numel() for name, tensor in flattened.items()} }"
        )
    positions = [int(value) for value in flattened["global_positions"].tolist()]
    rendered_pool_ids = [int(value) for value in flattened["pool_ids"].tolist()]
    rendered_record_indices = [int(value) for value in flattened["record_indices"].tolist()]
    rendered_occurrences = [int(value) for value in flattened["pool_occurrence_indices"].tolist()]
    rendered_epochs = [int(value) for value in flattened["pool_epochs"].tolist()]
    rendered_durations = [float(value) for value in flattened["effective_durations"].tolist()]
    if any(position < 0 for position in positions):
        raise RuntimeError(f"Negative global training position: {positions}")
    if any(pool_id < 0 or pool_id >= len(TRAINING_POOL_ORDER) for pool_id in rendered_pool_ids):
        raise RuntimeError(f"Invalid training pool IDs: {rendered_pool_ids}")
    if any(value < 0 for value in rendered_record_indices + rendered_occurrences + rendered_epochs):
        raise RuntimeError("Training sampler provenance contains a negative index")
    if any(not math.isfinite(value) or value <= 0 or value > DEFAULT_MAX_AUDIO_SECONDS + 1e-3 for value in rendered_durations):
        raise RuntimeError(f"Invalid effective training durations: {rendered_durations}")

    if "_huginn_audio_training_sample_counts" not in vars(model):
        model._huginn_audio_training_sample_counts = [0 for _ in TRAINING_POOL_ORDER]
        model._huginn_audio_training_duration_seconds = [0.0 for _ in TRAINING_POOL_ORDER]
    for pool_id, duration in zip(rendered_pool_ids, rendered_durations):
        model._huginn_audio_training_sample_counts[pool_id] += 1
        model._huginn_audio_training_duration_seconds[pool_id] += duration

    audit_dir_value = os.environ.get(TRAINING_STATS_FORWARD_AUDIT_DIR_ENV, "").strip()
    if audit_dir_value:
        phase = os.environ.get(TRAINING_STATS_PHASE_ENV, "").strip()
        if not phase:
            raise RuntimeError(f"{TRAINING_STATS_PHASE_ENV} is required for forward statistics auditing")
        rank = int(os.environ.get("RANK", "0"))
        audit_dir = Path(audit_dir_value)
        audit_dir.mkdir(parents=True, exist_ok=True)
        path = audit_dir / f"forward-{phase}-rank-{rank}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            for position, pool_id, record_index, occurrence, epoch, duration in zip(
                positions,
                rendered_pool_ids,
                rendered_record_indices,
                rendered_occurrences,
                rendered_epochs,
                rendered_durations,
            ):
                payload = {
                    "phase": phase,
                    "rank": rank,
                    "global_position": position,
                    "pool_name": TRAINING_POOL_ORDER[pool_id],
                    "pool_id": pool_id,
                    "record_index": record_index,
                    "pool_occurrence_index": occurrence,
                    "pool_epoch": epoch,
                    "effective_duration_seconds": duration,
                }
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def patch_huginn_audio_shift_loss(model):
    if getattr(model, "_huginn_audio_shift_loss_patched", False):
        print("[HuginnAudioSwift] shift-loss patch already applied")
        return model

    original_forward = model.forward

    def forward_with_shift_loss(self, *args, **kwargs):
        training_global_positions = _training_tensor(kwargs, "audio_training_global_positions")
        training_pool_ids = _training_tensor(kwargs, "audio_training_pool_ids")
        training_record_indices = _training_tensor(kwargs, "audio_training_record_indices")
        training_pool_occurrence_indices = _training_tensor(
            kwargs,
            "audio_training_pool_occurrence_indices",
        )
        training_pool_epochs = _training_tensor(kwargs, "audio_training_pool_epochs")
        training_effective_durations = _training_tensor(
            kwargs,
            "audio_training_effective_duration_seconds",
        )
        if self.training and not getattr(self, "_huginn_audio_runtime_checkpoint_state_logged", False):
            rank = os.environ.get("RANK", "0")
            print(
                "[HuginnAudioSwift] runtime_checkpoint_state "
                f"rank={rank} model_gradient_checkpointing={getattr(self, 'gradient_checkpointing', None)}"
            )
            self._huginn_audio_runtime_checkpoint_state_logged = True

        labels = kwargs.get("labels")
        audio_input_features = kwargs.get("audio_input_features")
        past_key_values = kwargs.get("past_key_values")
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]

        if labels is None:
            return original_forward(*args, **kwargs)

        kwargs_no_labels = dict(kwargs)
        kwargs_no_labels["labels"] = None
        outputs = original_forward(*args, **kwargs_no_labels)
        logits = outputs.logits
        if logits is None:
            raise RuntimeError("Huginn audio forward returned logits=None; cannot recompute shifted loss.")

        full_labels = labels.to(logits.device)
        if audio_input_features is not None and past_key_values is None:
            prefix_len = logits.size(1) - labels.size(1)
            if prefix_len < 0:
                raise RuntimeError(
                    f"Unexpected negative audio prefix length: logits_len={logits.size(1)} labels_len={labels.size(1)}"
                )
            if prefix_len > 0:
                prefix_labels = torch.full(
                    (labels.size(0), prefix_len),
                    fill_value=-100,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                full_labels = torch.cat([prefix_labels, labels], dim=1).to(logits.device)
                prefix_mask = getattr(self, "_last_audio_prefix_mask", None)
                if prefix_mask is not None:
                    prefix_mask = prefix_mask.to(device=full_labels.device, dtype=torch.bool)
                    if prefix_mask.shape != (labels.size(0), prefix_len):
                        raise RuntimeError(
                            "Audio prefix mask shape mismatch: "
                            f"mask={tuple(prefix_mask.shape)} expected={(labels.size(0), prefix_len)}"
                        )
                    has_audio_padding = prefix_mask.sum(dim=1).lt(prefix_len)
                    if labels.size(1) > 0 and bool(has_audio_padding.any().item()):
                        full_labels[has_audio_padding, prefix_len] = -100
                        if bool(full_labels[has_audio_padding, prefix_len].ne(-100).any().item()):
                            raise RuntimeError(
                                "The first text target after an audio padding region must be masked with -100"
                            )

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = full_labels[:, 1:].contiguous()

        audit_requested = os.environ.get(TRAIN_CHAIN_AUDIT_ENV, "").strip().lower() in {"1", "true", "yes"}
        if audit_requested:
            self._last_dynamic90s_full_labels = full_labels.detach()
            self._last_dynamic90s_shift_labels = shift_labels.detach()
        if audit_requested and os.environ.get("RANK", "0") == "0" and not getattr(
            self, "_huginn_audio_shift_loss_audit_logged", False
        ):
            if shift_logits.shape[:2] != shift_labels.shape:
                raise RuntimeError(
                    "NTP shift shape mismatch: "
                    f"logits={tuple(shift_logits.shape)} labels={tuple(shift_labels.shape)}"
                )
            prefix_token_count = int(logits.size(1) - labels.size(1))
            if prefix_token_count < 0:
                raise RuntimeError(f"NTP prefix length is negative: {prefix_token_count}")
            if prefix_token_count and not bool((full_labels[:, :prefix_token_count] == -100).all().item()):
                raise RuntimeError("Audio prefix labels are not fully ignored by the NTP loss")
            supervised_token_count = int((shift_labels != -100).sum().item())
            if supervised_token_count <= 0:
                raise RuntimeError("NTP shift loss has no supervised target tokens")
            print(
                "[HuginnAudioSwift] train_chain_audit_ntp "
                f"text_input_ids={tuple(input_ids.shape) if torch.is_tensor(input_ids) else None} "
                f"audio_features={tuple(audio_input_features.shape) if torch.is_tensor(audio_input_features) else None} "
                f"logits={tuple(logits.shape)} prefix_tokens={prefix_token_count} "
                f"shift_logits={tuple(shift_logits.shape)} shift_labels={tuple(shift_labels.shape)} "
                f"supervised_tokens={supervised_token_count}"
            )
            self._huginn_audio_shift_loss_audit_logged = True

        if shift_labels.ne(-100).any():
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        else:
            loss = logits.new_tensor(0.0)

        if os.environ.get(STAGE5_AUDIT_DIR_ENV, "").strip() and not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"Stage 5 observed a non-finite raw training loss: {loss.detach()}")

        outputs.loss = loss
        if hasattr(outputs, "log_ppl"):
            outputs.log_ppl = loss.detach().clone()
        if self.training and audio_input_features is not None and past_key_values is None:
            _record_actual_training_samples(
                self,
                training_global_positions,
                training_pool_ids,
                training_record_indices,
                training_pool_occurrence_indices,
                training_pool_epochs,
                training_effective_durations,
            )
        return outputs

    model.forward = MethodType(forward_with_shift_loss, model)
    model._huginn_audio_shift_loss_patched = True
    print("[HuginnAudioSwift] applied shift-loss patch for multimodal SFT")
    return model


def checkpoint_key_aliases(key: str) -> list[str]:
    aliases = {key}
    changed = True
    while changed:
        changed = False
        for alias in list(aliases):
            for prefix in ("base_model.model.", "base_model.", "model.", "module."):
                if alias.startswith(prefix):
                    stripped = alias[len(prefix):]
                    if stripped not in aliases:
                        aliases.add(stripped)
                        changed = True
    normalized = set()
    for alias in aliases:
        normalized.add(alias)
        normalized.add(alias.replace(".modules_to_save.default.", "."))
        normalized.add(alias.replace(".original_module.", "."))
        normalized.add(alias.replace(".lora_A.default.", ".lora_A."))
        normalized.add(alias.replace(".lora_B.default.", ".lora_B."))
        if alias.startswith("audio_aligner."):
            normalized.add(alias[len("audio_aligner."):])
        elif alias.startswith(("temporal_compressor.", "audio_projector.", "audio_boundary_embeddings.")):
            normalized.add(f"audio_aligner.{alias}")
    return list(normalized)


def read_tensor_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return {key: handle.get_tensor(key) for key in handle.keys()}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint tensor file is not a state dict: {path}")
    return {key: value for key, value in payload.items() if isinstance(key, str) and torch.is_tensor(value)}


def load_initial_aligner_state(model: torch.nn.Module, checkpoint_dir: Path) -> dict[str, Any]:
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Initial aligner checkpoint directory does not exist: {checkpoint_dir}")

    target_state = model.state_dict()
    canonical_targets: dict[str, str] = {}
    for target_key in target_state:
        for alias in checkpoint_key_aliases(target_key):
            if alias.startswith(ALIGNER_PREFIXES):
                canonical_targets.setdefault(alias, target_key)

    preferred_file = checkpoint_dir / "vit.safetensors"
    state_paths = [preferred_file] if preferred_file.is_file() else sorted(
        path
        for path in checkpoint_dir.rglob("*")
        if path.is_file() and path.suffix in {".safetensors", ".bin", ".pt", ".pth"}
        and not any(token in path.name.lower() for token in ("adapter", "optimizer", "scheduler", "rng", "trainer_state"))
    )
    if not state_paths:
        raise FileNotFoundError(f"No aligner tensor file found in initial checkpoint: {checkpoint_dir}")

    selected: dict[str, torch.Tensor] = {}
    source_keys: list[str] = []
    for state_path in state_paths:
        for source_key, tensor in read_tensor_state_dict(state_path).items():
            for alias in checkpoint_key_aliases(source_key):
                target_key = canonical_targets.get(alias)
                if target_key is None or target_state[target_key].shape != tensor.shape:
                    continue
                selected[target_key] = tensor
                source_keys.append(source_key)
                break
    if not selected:
        raise RuntimeError(f"No aligner tensors could be restored from initial checkpoint: {checkpoint_dir}")

    load_result = model.load_state_dict(selected, strict=False)
    boundary_targets = [
        key
        for key in selected
        if key.endswith((".audio_bos", ".audio_eos")) or key in {"audio_bos", "audio_eos"}
    ]
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "loaded_aligner_tensor_count": len(selected),
        "restored_boundary_embeddings": boundary_targets,
        "source_key_preview": source_keys[:20],
        "missing_key_count": len(load_result.missing_keys),
        "unexpected_key_count": len(load_result.unexpected_keys),
    }


def force_audio_aligner_trainable(model: torch.nn.Module) -> None:
    if os.environ.get(FORCE_ALIGNER_TRAINABLE_ENV) != "1":
        return
    audio_model = next(
        (
            module
            for module in model.modules()
            if all(hasattr(module, name) for name in ("audio_encoder", "audio_aligner"))
        ),
        None,
    )
    if audio_model is None:
        raise RuntimeError("Unable to locate the Huginn audio model after PEFT adapter loading")

    audio_model.audio_aligner.requires_grad_(True)
    if any(parameter.requires_grad for parameter in audio_model.audio_encoder.parameters()):
        raise RuntimeError("audio_encoder became trainable while restoring the WavCaps adapter")

    aligner_trainable = sum(
        parameter.numel()
        for name, parameter in audio_model.named_parameters()
        if parameter.requires_grad and is_aligner_parameter_name(name)
    )
    print(f"[HuginnAudioSwift] forced_aligner_trainable_parameters={aligner_trainable}")


def _active_modules_to_save_names(wrapper: torch.nn.Module) -> list[str]:
    copies = getattr(wrapper, "modules_to_save", {})
    available = [str(name) for name in copies]
    active = getattr(wrapper, "active_adapter", None)
    if isinstance(active, str):
        requested = [active]
    elif isinstance(active, (list, tuple, set)):
        requested = [str(name) for name in active]
    else:
        requested = []
    selected = [name for name in requested if name in copies]
    if not selected and "default" in copies:
        selected = ["default"]
    if not selected and len(available) == 1:
        selected = available
    if not selected:
        raise RuntimeError(
            "Dynamic Whisper aligner ModulesToSaveWrapper has no selectable adapter copy: "
            f"available={available} active={active}"
        )
    return selected


def guard_peft_aligner_wrapper_trainability(wrapper: torch.nn.Module, *, module_name: str) -> None:
    """Train only PEFT's checkpoint-owned aligner copy, never its original branch."""
    if getattr(wrapper, "_huginn_whisper_dynamic90s_trainability_guarded", False):
        return
    if type(wrapper).__name__ != "ModulesToSaveWrapper":
        raise RuntimeError(
            "Expected PEFT ModulesToSaveWrapper for dynamic Whisper aligner, "
            f"got module={module_name} type={type(wrapper)}"
        )
    if not hasattr(wrapper, "original_module") or not hasattr(wrapper, "modules_to_save"):
        raise RuntimeError(f"PEFT ModulesToSaveWrapper lacks ownership fields: module={module_name}")

    def guarded_requires_grad_(self, requires_grad: bool = True):
        self.original_module.requires_grad_(False)
        for copy in self.modules_to_save.values():
            copy.requires_grad_(False)
        if requires_grad:
            for adapter_name in _active_modules_to_save_names(self):
                self.modules_to_save[adapter_name].requires_grad_(True)
        return self

    wrapper.requires_grad_ = MethodType(guarded_requires_grad_, wrapper)
    wrapper._huginn_whisper_dynamic90s_trainability_guarded = True
    wrapper.requires_grad_(True)
    original_trainable = sum(
        parameter.numel() for parameter in wrapper.original_module.parameters() if parameter.requires_grad
    )
    saved_trainable = sum(
        parameter.numel() for parameter in wrapper.modules_to_save.parameters() if parameter.requires_grad
    )
    if original_trainable or not saved_trainable:
        raise RuntimeError(
            "Failed to establish PEFT-owned dynamic Whisper aligner trainability: "
            f"module={module_name} original_trainable={original_trainable} saved_trainable={saved_trainable}"
        )


def patch_peft_constructor_modules_to_save() -> None:
    """Inject all 14 aligner tensors into PEFT's adapter-only FSDP state."""
    force_modules_to_save = _requested(PEFT_ALIGNER_MODULES_TO_SAVE_ENV)
    if not force_modules_to_save and not _requested(FSDP_SAVE_DEBUG_ENV):
        return
    try:
        from peft.peft_model import PeftModel
    except Exception as exc:
        raise RuntimeError("Unable to import PEFT for dynamic Whisper checkpoint ownership") from exc
    original = PeftModel.__init__
    if getattr(original, "_huginn_whisper_dynamic90s_constructor_patched", False):
        return

    def patched(self, *args, **kwargs):
        peft_config = kwargs.get("peft_config")
        if peft_config is None and len(args) >= 2:
            peft_config = args[1]
        if force_modules_to_save:
            if peft_config is None or not hasattr(peft_config, "modules_to_save"):
                raise RuntimeError(
                    "Dynamic Whisper checkpointing requires a PEFT config with modules_to_save; "
                    f"config_type={type(peft_config)}"
                )
            configured = list(getattr(peft_config, "modules_to_save", None) or [])
            peft_config.modules_to_save = list(dict.fromkeys([*configured, *ALIGNER_MODULES_TO_SAVE]))
            if os.environ.get("RANK", "0") == "0":
                print(
                    "[HuginnAudioDynamic90s] injected PEFT aligner modules_to_save "
                    f"existing={configured} final={peft_config.modules_to_save}"
                )
        result = original(self, *args, **kwargs)
        wrappers = {
            name: module
            for name, module in self.named_modules()
            if type(module).__name__ == "ModulesToSaveWrapper"
        }
        if force_modules_to_save:
            missing = [
                name
                for name in ALIGNER_MODULES_TO_SAVE
                if not any(wrapper_name.endswith(name) for wrapper_name in wrappers)
            ]
            if missing:
                raise RuntimeError(
                    "PEFT did not wrap every dynamic Whisper aligner module: "
                    f"missing={missing} wrappers={list(wrappers)}"
                )
            guarded = []
            for aligner_name in ALIGNER_MODULES_TO_SAVE:
                wrapper_name, wrapper = next(
                    (item for item in wrappers.items() if item[0].endswith(aligner_name)),
                    (None, None),
                )
                if wrapper_name is None or wrapper is None:
                    raise RuntimeError(f"Unable to locate PEFT wrapper for {aligner_name}")
                guard_peft_aligner_wrapper_trainability(wrapper, module_name=wrapper_name)
                guarded.append(wrapper_name)
            if os.environ.get("RANK", "0") == "0":
                print(f"[HuginnAudioDynamic90s] guarded PEFT aligner wrappers={guarded}")
        return result

    patched._huginn_whisper_dynamic90s_constructor_patched = True
    PeftModel.__init__ = patched
    print(
        "[HuginnAudioDynamic90s] installed PEFT modules-to-save constructor hook "
        f"enabled={force_modules_to_save}"
    )


def _is_dynamic_fsdp_dcp_checkpoint(model_id: Any) -> tuple[bool, Path | None]:
    checkpoint_dir = Path(model_id) if isinstance(model_id, (str, os.PathLike)) else None
    enabled = (
        _requested(PEFT_ALIGNER_MODULES_TO_SAVE_ENV)
        and checkpoint_dir is not None
        and (checkpoint_dir / "pytorch_model_fsdp_0").is_dir()
        and not (checkpoint_dir / "adapter_config.json").is_file()
    )
    return enabled, checkpoint_dir


def _fsdp_dcp_lora_resume_model(
    base_model: torch.nn.Module,
    checkpoint_dir: Path,
    *,
    adapter_name: str,
) -> torch.nn.Module:
    """Recreate the PEFT topology; Accelerate later restores DCP tensors/state."""
    source_dir = checkpoint_dir / "pytorch_model_fsdp_0"
    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from torch.distributed.checkpoint import FileSystemReader
    except Exception as exc:
        raise RuntimeError("Unable to import DCP/PEFT APIs for dynamic Whisper resume") from exc
    metadata = FileSystemReader(str(source_dir)).read_metadata()
    state_metadata = getattr(metadata, "state_dict_metadata", {})
    checkpoint_groups = _fsdp_save_state_groups(state_metadata)
    checkpoint_counts = {name: len(keys) for name, keys in checkpoint_groups.items()}
    expected_counts = {"lora": 66, "aligner": 14, "other": 0}
    if checkpoint_counts != expected_counts:
        raise RuntimeError(
            "Dynamic Whisper resume DCP does not match the adapter contract: "
            f"expected={expected_counts} actual={checkpoint_counts}"
        )
    lora_entries = [
        (str(key), entry)
        for key, entry in state_metadata.items()
        if ".lora_A." in str(key) or ".lora_B." in str(key)
    ]
    if not lora_entries:
        raise RuntimeError(f"Dynamic Whisper DCP has no LoRA metadata: {source_dir}")
    module_names = set(dict(base_model.named_modules()))
    target_modules: set[str] = set()
    ranks: set[int] = set()
    unresolved: list[str] = []
    for source_key, entry in lora_entries:
        if ".lora_A." in source_key:
            shape = tuple(int(value) for value in getattr(entry, "size", ()))
            if not shape:
                raise RuntimeError(f"LoRA-A DCP metadata has no shape: {source_key}")
            ranks.add(shape[0])
        matched = False
        for candidate in checkpoint_key_aliases(source_key):
            if ".lora_A." in candidate:
                module_path = candidate.split(".lora_A.", 1)[0]
            elif ".lora_B." in candidate:
                module_path = candidate.split(".lora_B.", 1)[0]
            else:
                continue
            if module_path in module_names:
                target_modules.add(module_path)
                matched = True
                break
        if not matched:
            unresolved.append(source_key)
    if unresolved or len(ranks) != 1 or len(target_modules) != 33:
        raise RuntimeError(
            "Unable to reconstruct dynamic Whisper LoRA topology from DCP: "
            f"unresolved={unresolved[:10]} ranks={sorted(ranks)} targets={len(target_modules)}"
        )
    rank = next(iter(ranks))
    if rank != 8:
        raise RuntimeError(f"Dynamic Whisper resume DCP LoRA rank must be 8, got {rank}")
    saved_args: dict[str, Any] = {}
    args_path = checkpoint_dir.parent / "args.json"
    if args_path.is_file():
        payload = json.loads(args_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            saved_args = payload
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=int(saved_args.get("lora_alpha", 16)),
        lora_dropout=float(saved_args.get("lora_dropout", DYNAMIC_LORA_DROPOUT)),
        target_modules=sorted(target_modules),
        bias="none",
        inference_mode=False,
    )
    restored = get_peft_model(base_model, config, adapter_name=adapter_name)
    if os.environ.get("RANK", "0") == "0":
        print(
            "[HuginnAudioDynamic90s] recreated PEFT topology from FSDP DCP "
            f"checkpoint={checkpoint_dir} target_count={len(target_modules)} "
            f"rank={rank} alpha={config.lora_alpha} dropout={config.lora_dropout}"
        )
    return restored


def patch_peft_adapter_restore() -> None:
    """PEFT freezes base modules on adapter load; re-enable our separately trained aligner."""
    if getattr(patch_peft_adapter_restore, "_patched", False):
        return
    try:
        from peft import PeftModel
    except ImportError:  # pragma: no cover - depends on Swift runtime
        return

    original_from_pretrained = PeftModel.from_pretrained

    @classmethod
    def from_pretrained_with_audio_aligner(cls, *args, **kwargs):
        base_model = args[0] if args else kwargs.get("model")
        model_id = args[1] if len(args) >= 2 else kwargs.get("model_id")
        is_dynamic_dcp, checkpoint_dir = _is_dynamic_fsdp_dcp_checkpoint(model_id)
        if is_dynamic_dcp:
            if checkpoint_dir is None or not isinstance(base_model, torch.nn.Module):
                raise RuntimeError(
                    "Dynamic Whisper DCP resume did not receive a valid base model/checkpoint: "
                    f"model_type={type(base_model)} checkpoint={checkpoint_dir}"
                )
            restored_model = _fsdp_dcp_lora_resume_model(
                base_model,
                checkpoint_dir,
                adapter_name=str(kwargs.get("adapter_name", "default")),
            )
        else:
            restored_model = original_from_pretrained(*args, **kwargs)
        force_audio_aligner_trainable(restored_model)
        return restored_model

    PeftModel.from_pretrained = from_pretrained_with_audio_aligner
    patch_peft_adapter_restore._patched = True
    print("[HuginnAudioSwift] installed PEFT adapter-restore aligner patch")


def patch_swift_lora_llm_fsdp_dcp_resume() -> None:
    """Bypass Swift's legacy vit.safetensors path for adapter-only DCP resume."""
    if not _requested(PEFT_ALIGNER_MODULES_TO_SAVE_ENV):
        return
    try:
        import swift.tuner_plugin.lora_llm as lora_llm_module
        from peft import PeftModel
    except Exception as exc:
        raise RuntimeError("Unable to import Swift lora_llm for dynamic Whisper DCP resume") from exc
    if getattr(lora_llm_module, "_huginn_whisper_dynamic90s_dcp_resume_patched", False):
        return
    candidates: list[tuple[str, type, Any]] = []
    for class_name, candidate in vars(lora_llm_module).items():
        if not isinstance(candidate, type):
            continue
        method = candidate.__dict__.get("from_pretrained")
        if method is None:
            continue
        function = method.__func__ if isinstance(method, (staticmethod, classmethod)) else method
        code = getattr(function, "__code__", None)
        if code is not None and "vit.safetensors" in repr(code.co_consts):
            candidates.append((str(class_name), candidate, method))
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected one Swift legacy vit.safetensors loader for dynamic Whisper DCP resume, "
            f"found={[name for name, _, _ in candidates]}"
        )
    class_name, tuner_class, descriptor = candidates[0]
    original = descriptor.__func__ if isinstance(descriptor, (staticmethod, classmethod)) else descriptor

    def maybe_restore(model, model_id, *args, **kwargs):
        is_dynamic_dcp, checkpoint_dir = _is_dynamic_fsdp_dcp_checkpoint(model_id)
        if not is_dynamic_dcp:
            return False, None
        if os.environ.get("RANK", "0") == "0":
            print(
                "[HuginnAudioDynamic90s] bypassing Swift legacy vit.safetensors reload "
                f"for FSDP DCP checkpoint={checkpoint_dir}"
            )
        return True, PeftModel.from_pretrained(model, model_id, *args, **kwargs)

    if isinstance(descriptor, staticmethod):
        def patched(model, model_id, *args, **kwargs):
            handled, result = maybe_restore(model, model_id, *args, **kwargs)
            return result if handled else original(model, model_id, *args, **kwargs)

        tuner_class.from_pretrained = staticmethod(patched)
        descriptor_kind = "staticmethod"
    elif isinstance(descriptor, classmethod):
        def patched(cls, model, model_id, *args, **kwargs):
            handled, result = maybe_restore(model, model_id, *args, **kwargs)
            return result if handled else original(cls, model, model_id, *args, **kwargs)

        tuner_class.from_pretrained = classmethod(patched)
        descriptor_kind = "classmethod"
    else:
        def patched(self, model, model_id, *args, **kwargs):
            handled, result = maybe_restore(model, model_id, *args, **kwargs)
            return result if handled else original(self, model, model_id, *args, **kwargs)

        tuner_class.from_pretrained = patched
        descriptor_kind = "instance_method"
    lora_llm_module._huginn_whisper_dynamic90s_dcp_resume_patched = True
    print(
        "[HuginnAudioDynamic90s] installed Swift DCP-resume patch "
        f"tuner_class={class_name} descriptor={descriptor_kind}"
    )


def _fsdp_save_state_groups(state_dict: dict[str, Any]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {"lora": [], "aligner": [], "other": []}
    for raw_key in state_dict:
        key = str(raw_key)
        aliases = checkpoint_key_aliases(key)
        if any(".lora_A." in alias or ".lora_B." in alias for alias in aliases):
            groups["lora"].append(key)
        elif any(alias.startswith(ALIGNER_PREFIXES) for alias in aliases):
            groups["aligner"].append(key)
        else:
            groups["other"].append(key)
    return groups


def patch_accelerate_fsdp2_save_state_audit() -> None:
    if not _requested(FSDP_SAVE_DEBUG_ENV):
        return
    try:
        from accelerate.utils import fsdp_utils
    except ImportError as exc:
        raise RuntimeError("Unable to import Accelerate FSDP save utilities") from exc
    original = fsdp_utils._get_model_state_dict
    if getattr(original, "_huginn_whisper_dynamic90s_save_audit_patched", False):
        return

    def patched(model, adapter_only=False, sd_options=None):
        state_dict = original(model, adapter_only=adapter_only, sd_options=sd_options)
        if os.environ.get("RANK", "0") == "0":
            groups = _fsdp_save_state_groups(state_dict)
            print(
                "[HuginnAudioDynamic90s] FSDP2 pre-DCP save-state audit "
                f"adapter_only={adapter_only} tensor_count={len(state_dict)} "
                f"lora={len(groups['lora'])} aligner={len(groups['aligner'])} "
                f"other={len(groups['other'])} aligner_preview={groups['aligner'][:8]}"
            )
        return state_dict

    patched._huginn_whisper_dynamic90s_save_audit_patched = True
    fsdp_utils._get_model_state_dict = patched
    print("[HuginnAudioDynamic90s] installed Accelerate FSDP2 save-state audit")


def patch_peft_dynamic90s_lora_dropout() -> None:
    """Force the requested dropout before PEFT creates LoRA layers.

    The installed ms-swift ``LoRALLMTuner`` does not forward its generic
    ``lora_dropout`` argument into ``peft.LoraConfig``. Patching the config
    constructor is earlier and more reliable than replacing already-created
    Identity modules, and it is scoped to jobs that explicitly load this
    isolated dynamic-90s plugin.
    """
    try:
        from peft import LoraConfig
    except ImportError:  # pragma: no cover - depends on Swift runtime
        return

    patch_marker = "_huginn_audio_dynamic90s_dropout_patch"
    existing = getattr(LoraConfig, patch_marker, None)
    if existing is not None:
        if float(existing) != DYNAMIC_LORA_DROPOUT:
            raise RuntimeError(
                "PEFT LoraConfig already has an incompatible dynamic-90s dropout patch: "
                f"existing={existing} requested={DYNAMIC_LORA_DROPOUT}"
            )
        return

    original_init = LoraConfig.__init__

    @wraps(original_init)
    def init_with_dynamic90s_dropout(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.lora_dropout = DYNAMIC_LORA_DROPOUT

    LoraConfig.__init__ = init_with_dynamic90s_dropout
    setattr(LoraConfig, patch_marker, DYNAMIC_LORA_DROPOUT)
    print(
        "[HuginnAudioDynamic90sSwift] patched PEFT LoraConfig "
        f"effective_lora_dropout={DYNAMIC_LORA_DROPOUT}"
    )


def classify_missing_keys(missing_keys: list[str]) -> dict[str, list[str]]:
    groups = {
        "audio_encoder": [],
        "aligner": [],
        "llm": [],
        "other": [],
    }
    for key in missing_keys:
        if key.startswith("audio_encoder."):
            groups["audio_encoder"].append(key)
        elif key.startswith(ALIGNER_PREFIXES):
            groups["aligner"].append(key)
        elif key.startswith("transformer.") or key.startswith("lm_head."):
            groups["llm"].append(key)
        else:
            groups["other"].append(key)
    return groups


def print_missing_key_summary(missing_keys: list[str], unexpected_keys: list[str]):
    groups = classify_missing_keys(missing_keys)
    print(f"[HuginnAudioSwift] backbone load missing={len(missing_keys)} unexpected={len(unexpected_keys)}")
    for group_name, keys in groups.items():
        print(f"[HuginnAudioSwift] missing_group[{group_name}]={len(keys)}")
        for key in keys[:5]:
            print(f"  - {key}")
    if unexpected_keys:
        print("[HuginnAudioSwift] first_unexpected_keys:")
        for key in unexpected_keys[:10]:
            print(f"  - {key}")


def resample_waveform(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return audio.astype(np.float32, copy=False)

    duration = audio.shape[0] / float(source_sr)
    target_length = max(1, int(round(duration * target_sr)))
    src_positions = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
    tgt_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    return np.interp(tgt_positions, src_positions, audio).astype(np.float32)


def normalize_audio_array(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    if audio.ndim == 2:
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            audio = audio.mean(axis=0)
        else:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32, copy=False)
    raise ValueError(f"Unsupported audio ndim={audio.ndim}")


@dataclass(frozen=True)
class WhisperAudioPlan:
    """Production duration plan shared by preprocessing and validation.

    Token counts are dynamic. A complete 120 ms block produces one audio
    token; shorter residual tails are not padded into an additional token.
    """

    total_samples: int
    included_samples: int
    chunk_ranges: tuple[tuple[int, int], ...]
    feature_lengths: tuple[int, ...]
    encoder_lengths: tuple[int, ...]
    token_counts: tuple[int, ...]

    @property
    def total_audio_tokens(self) -> int:
        return sum(self.token_counts)

    @property
    def segment_count(self) -> int:
        return len(self.chunk_ranges)


def plan_audio_for_whisper(
    total_samples: int,
    sample_rate: int,
    chunk_seconds: float = DEFAULT_AUDIO_CHUNK_SECONDS,
    max_audio_seconds: float = DEFAULT_MAX_AUDIO_SECONDS,
) -> WhisperAudioPlan:
    """Plan non-overlapping Whisper chunks, truncating every input to 90 seconds."""
    if total_samples <= 0:
        raise ValueError(f"total_samples must be positive, got {total_samples}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")

    chunk_samples = int(round(chunk_seconds * sample_rate))
    included_samples = min(total_samples, int(round(max_audio_seconds * sample_rate)))
    if chunk_samples <= 0 or included_samples <= 0:
        raise ValueError(
            f"Invalid audio split configuration: {chunk_samples=} {included_samples=}"
        )

    chunk_ranges: list[tuple[int, int]] = []
    feature_lengths: list[int] = []
    encoder_lengths: list[int] = []
    token_counts: list[int] = []
    for start in range(0, included_samples, chunk_samples):
        end = min(start + chunk_samples, included_samples)
        chunk_size = end - start
        if chunk_size <= 0:
            continue
        feature_length = min(
            WHISPER_MAX_FEATURE_FRAMES,
            max(1, chunk_size // WHISPER_FEATURE_HOP_LENGTH),
        )
        encoder_length = feature_length // WHISPER_ENCODER_DOWNSAMPLE
        token_count = (
            0
            if encoder_length < DYNAMIC_COMPRESSOR_KERNEL
            else (encoder_length - DYNAMIC_COMPRESSOR_KERNEL) // DYNAMIC_COMPRESSOR_STRIDE + 1
        )
        chunk_ranges.append((start, end))
        feature_lengths.append(feature_length)
        encoder_lengths.append(encoder_length)
        token_counts.append(token_count)

    return WhisperAudioPlan(
        total_samples=total_samples,
        included_samples=included_samples,
        chunk_ranges=tuple(chunk_ranges),
        feature_lengths=tuple(feature_lengths),
        encoder_lengths=tuple(encoder_lengths),
        token_counts=tuple(token_counts),
    )


def split_audio_for_whisper(
    audio: np.ndarray,
    sample_rate: int,
    chunk_seconds: float = DEFAULT_AUDIO_CHUNK_SECONDS,
    max_audio_seconds: float = DEFAULT_MAX_AUDIO_SECONDS,
) -> tuple[list[np.ndarray], list[int]]:
    """Split audio into Whisper windows and return true mel-frame lengths.

    Each returned waveform is at most one Whisper window. The feature extractor
    later pads every window to 3000 mel frames, while the returned lengths keep
    the padding out of the encoder attention and compressed token sequence.
    """
    if audio.ndim != 1:
        raise ValueError(f"Expected mono waveform with shape [samples], got {audio.shape}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if audio.size == 0:
        raise ValueError("Audio waveform is empty")

    plan = plan_audio_for_whisper(
        total_samples=int(audio.shape[0]),
        sample_rate=sample_rate,
        chunk_seconds=chunk_seconds,
        max_audio_seconds=max_audio_seconds,
    )

    chunks: list[np.ndarray] = []
    for start, end in plan.chunk_ranges:
        chunk = audio[start:end].astype(np.float32, copy=False)
        if chunk.size == 0:
            continue
        chunks.append(chunk)

    if not chunks:
        raise ValueError("Audio split produced no usable chunks")
    return chunks, list(plan.feature_lengths)


def trim_audio(audio: np.ndarray, target_sr: int, max_audio_seconds: float | None) -> np.ndarray:
    if max_audio_seconds is None:
        return audio.astype(np.float32, copy=False)
    max_samples = int(round(max_audio_seconds * target_sr))
    if audio.shape[0] > max_samples:
        audio = audio[:max_samples]
    return audio.astype(np.float32, copy=False)


def get_ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def load_wav_mono(path: Path, target_sr: int, max_audio_seconds: float | None) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        source_sr = wf.getframerate()
        num_frames = wf.getnframes()
        frames = wf.readframes(num_frames)

    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM wav is supported: {path}")

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if num_channels > 1:
        audio = audio.reshape(-1, num_channels).mean(axis=1)

    audio = resample_waveform(audio, source_sr, target_sr)
    return trim_audio(audio, target_sr, max_audio_seconds)


def _decode_audio_with_soundfile(source: str | io.BytesIO) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - depends on remote env
        raise RuntimeError("soundfile is not available") from exc

    audio, source_sr = sf.read(source, dtype="float32", always_2d=False)
    return normalize_audio_array(np.asarray(audio)), int(source_sr)


def _decode_audio_with_torchaudio(source: str | io.BytesIO) -> tuple[np.ndarray, int]:
    try:
        import torchaudio
    except ImportError as exc:  # pragma: no cover - depends on remote env
        raise RuntimeError("torchaudio is not available") from exc

    waveform, source_sr = torchaudio.load(source)
    audio = waveform.mean(dim=0).cpu().numpy().astype(np.float32, copy=False)
    return audio, int(source_sr)


def decode_audio_bytes(audio_bytes: bytes, source_label: str) -> tuple[np.ndarray, int]:
    errors: list[str] = []
    buffer = io.BytesIO(audio_bytes)
    for backend_name, backend in (
        ("soundfile", _decode_audio_with_soundfile),
        ("torchaudio", _decode_audio_with_torchaudio),
    ):
        try:
            buffer.seek(0)
            return backend(buffer)
        except Exception as exc:  # pragma: no cover - backend dependent
            errors.append(f"{backend_name}={type(exc).__name__}: {exc}")

    joined = "; ".join(errors) if errors else "no backend attempted"
    raise RuntimeError(f"Failed to decode audio bytes from {source_label}. Tried: {joined}")


def decode_audio_with_ffmpeg_bytes(audio_bytes: bytes, source_label: str, target_sr: int) -> np.ndarray:
    ffmpeg_path = get_ffmpeg_path()
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg is not available")

    cmd = [
        ffmpeg_path,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "pipe:1",
    ]
    result = subprocess.run(
        cmd,
        input=audio_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg decode failed for {source_label}: {stderr or 'unknown error'}")
    if not result.stdout:
        raise RuntimeError(f"ffmpeg decode produced empty output for {source_label}")
    return np.frombuffer(result.stdout, dtype=np.float32).astype(np.float32, copy=False)


def decode_audio_with_ffmpeg_file(path: Path, target_sr: int) -> np.ndarray:
    ffmpeg_path = get_ffmpeg_path()
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg is not available")

    cmd = [
        ffmpeg_path,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "pipe:1",
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg decode failed for {path}: {stderr or 'unknown error'}")
    if not result.stdout:
        raise RuntimeError(f"ffmpeg decode produced empty output for {path}")
    return np.frombuffer(result.stdout, dtype=np.float32).astype(np.float32, copy=False)


def decode_audio_segment_with_ffmpeg_file(
    path: Path,
    target_sr: int,
    start_sec: float,
    end_sec: float,
    max_audio_seconds: float | None,
) -> np.ndarray:
    """Decode one metadata-defined segment without materializing a converted file."""
    if not math.isfinite(start_sec) or not math.isfinite(end_sec) or start_sec < 0 or end_sec <= start_sec:
        raise ValueError(f"Invalid audio segment bounds for {path}: start={start_sec} end={end_sec}")
    duration = end_sec - start_sec
    if max_audio_seconds is not None:
        duration = min(duration, max_audio_seconds)
    ffmpeg_path = get_ffmpeg_path()
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg is not available")
    cmd = [
        ffmpeg_path,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_sec:.9f}",
        "-i",
        str(path),
        "-t",
        f"{duration:.9f}",
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "pipe:1",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(
            f"ffmpeg segment decode failed for {path}[{start_sec:.3f},{end_sec:.3f}]: "
            f"{stderr or 'unknown error'}"
        )
    if not result.stdout:
        raise RuntimeError(f"ffmpeg segment decode produced empty output for {path}")
    return np.frombuffer(result.stdout, dtype=np.float32).astype(np.float32, copy=False)


def load_audio_file(path: Path, target_sr: int, max_audio_seconds: float | None) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".wav":
        return load_wav_mono(path, target_sr, max_audio_seconds)
    if suffix in {".flac", ".ogg", ".mp3", ".m4a", ".opus"}:
        errors: list[str] = []
        for backend_name, backend in (
            ("soundfile", _decode_audio_with_soundfile),
            ("torchaudio", _decode_audio_with_torchaudio),
        ):
            try:
                audio, source_sr = backend(str(path))
                audio = resample_waveform(audio, source_sr, target_sr)
                return trim_audio(audio, target_sr, max_audio_seconds)
            except Exception as exc:  # pragma: no cover - backend dependent
                errors.append(f"{backend_name}={type(exc).__name__}: {exc}")
        try:
            audio = decode_audio_with_ffmpeg_file(path, target_sr)
            return trim_audio(audio, target_sr, max_audio_seconds)
        except Exception as exc:  # pragma: no cover - backend dependent
            errors.append(f"ffmpeg={type(exc).__name__}: {exc}")
        joined = "; ".join(errors)
        raise RuntimeError(f"Failed to decode audio file {path}. Tried: {joined}")
    raise ValueError(f"Unsupported audio suffix for {path}")


def get_cached_tarfile(tar_path: Path) -> tarfile.TarFile:
    cache_key = os.fspath(tar_path)
    cached = _TARFILE_CACHE.get(cache_key)
    if cached is not None:
        _TARFILE_CACHE.move_to_end(cache_key)
        return cached

    tar_obj = tarfile.open(cache_key, mode="r:*")
    _TARFILE_CACHE[cache_key] = tar_obj
    while len(_TARFILE_CACHE) > TARFILE_CACHE_LIMIT:
        _, old_tar = _TARFILE_CACHE.popitem(last=False)
        old_tar.close()
    return tar_obj


def load_audio_from_tar(
    tar_path: Path,
    member_name: str,
    target_sr: int,
    max_audio_seconds: float | None,
) -> np.ndarray:
    tar_obj = get_cached_tarfile(tar_path)
    extracted = tar_obj.extractfile(member_name)
    if extracted is None:
        raise FileNotFoundError(f"Member {member_name} not found in tar archive {tar_path}")
    audio_bytes = extracted.read()
    source_label = f"{tar_path}:{member_name}"
    try:
        audio, source_sr = decode_audio_bytes(audio_bytes, source_label)
        audio = resample_waveform(audio, source_sr, target_sr)
        return trim_audio(audio, target_sr, max_audio_seconds)
    except Exception as exc:
        ffmpeg_errors = [f"python_backends={type(exc).__name__}: {exc}"]
        try:
            audio = decode_audio_with_ffmpeg_bytes(audio_bytes, source_label, target_sr)
            return trim_audio(audio, target_sr, max_audio_seconds)
        except Exception as ffmpeg_exc:
            ffmpeg_errors.append(f"ffmpeg={type(ffmpeg_exc).__name__}: {ffmpeg_exc}")
            raise RuntimeError(f"Failed to decode tar audio {source_label}. Tried: {'; '.join(ffmpeg_errors)}")


def resolve_audio_path(audio_item: Any) -> Path:
    if isinstance(audio_item, str):
        return Path(audio_item)
    if isinstance(audio_item, dict):
        if "audio" in audio_item and isinstance(audio_item["audio"], str):
            return Path(audio_item["audio"])
        if "path" in audio_item and isinstance(audio_item["path"], str):
            return Path(audio_item["path"])
    raise TypeError(f"Unsupported audio source type: {type(audio_item)}")


def load_audio_item(
    audio_item: Any,
    target_sr: int,
    max_audio_seconds: float | None,
) -> np.ndarray:
    """Load whole-file, tar-backed, or metadata-segment audio into one canonical waveform."""
    if isinstance(audio_item, dict) and "tar_path" in audio_item and "audio_member" in audio_item:
        return load_audio_from_tar(
            Path(str(audio_item["tar_path"])),
            str(audio_item["audio_member"]),
            target_sr=target_sr,
            max_audio_seconds=max_audio_seconds,
        )
    audio_path = resolve_audio_path(audio_item)
    if isinstance(audio_item, dict) and (
        audio_item.get("start_sec") is not None or audio_item.get("end_sec") is not None
    ):
        if audio_item.get("start_sec") is None or audio_item.get("end_sec") is None:
            raise ValueError(f"Segment audio requires both start_sec and end_sec: {audio_item}")
        return decode_audio_segment_with_ffmpeg_file(
            audio_path,
            target_sr=target_sr,
            start_sec=float(audio_item["start_sec"]),
            end_sec=float(audio_item["end_sec"]),
            max_audio_seconds=max_audio_seconds,
        )
    return load_audio_file(audio_path, target_sr=target_sr, max_audio_seconds=max_audio_seconds)


def audit_consumed_audio_position(audio_item: Any) -> None:
    """Record the exact row entering template encoding for checkpoint gates."""
    audit_dir_value = os.environ.get(DATA_POSITION_AUDIT_DIR_ENV, "").strip()
    if not audit_dir_value:
        return
    phase = os.environ.get(DATA_POSITION_AUDIT_PHASE_ENV, "").strip()
    if not phase:
        raise RuntimeError(f"{DATA_POSITION_AUDIT_PHASE_ENV} is required when data-position audit is enabled")
    if not isinstance(audio_item, dict):
        raise TypeError(f"Data-position audit requires a dictionary audio item, got {type(audio_item)}")
    required_fields = (
        "global_position",
        "pool_name",
        "task",
        "uid",
        "record_index",
        "pool_occurrence_index",
        "pool_epoch",
        "pool_epoch_offset",
    )
    missing = [field for field in required_fields if audio_item.get(field) is None]
    if missing:
        raise RuntimeError(f"Data-position audit audio item is missing provenance fields: {missing}")
    rank = int(os.environ.get("RANK", "0"))
    payload = {
        "phase": phase,
        "rank": rank,
        "global_position": int(audio_item["global_position"]),
        "pool_name": str(audio_item["pool_name"]),
        "task": str(audio_item["task"]),
        "uid": str(audio_item["uid"]),
        "record_index": int(audio_item["record_index"]),
        "pool_occurrence_index": int(audio_item["pool_occurrence_index"]),
        "pool_epoch": int(audio_item["pool_epoch"]),
        "pool_epoch_offset": int(audio_item["pool_epoch_offset"]),
    }
    audit_dir = Path(audit_dir_value)
    audit_dir.mkdir(parents=True, exist_ok=True)
    path = audit_dir / f"data-{phase}-rank-{rank}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class HuginnAudioProcessor:
    def __init__(self, tokenizer, feature_extractor):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

    def __getattr__(self, item):
        return getattr(self.tokenizer, item)


def build_huginn_audio_processor() -> HuginnAudioProcessor:
    tokenizer = AutoTokenizer.from_pretrained(
        HUGINN_MODEL_DIR,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    audio_processor = AutoProcessor.from_pretrained(
        WHISPER_MODEL_DIR,
        trust_remote_code=True,
    )
    feature_extractor = getattr(audio_processor, "feature_extractor", audio_processor)
    return HuginnAudioProcessor(tokenizer, feature_extractor)


def enable_fsdp2_nonpersistent_rope_buffer(model: torch.nn.Module) -> None:
    """Keep deterministic RoPE frequencies out of Accelerate FSDP2's parameter reload.

    Accelerate 1.13's CPU-RAM-efficient FSDP2 loader assumes every ``state_dict``
    entry is a sharded DTensor. ``freqs_cis`` is a fixed buffer and therefore stays
    a normal Tensor. Marking it non-persistent lets Accelerate preserve and restore
    it through its dedicated non-persistent-buffer path instead.
    """
    requested = os.environ.get(FSDP2_NONPERSISTENT_ROPE_ENV, "").strip().lower()
    if requested not in {"1", "true", "yes"}:
        return
    if "freqs_cis" not in model._buffers:
        raise RuntimeError("FSDP2 compatibility requested but Huginn has no freqs_cis buffer")
    model._non_persistent_buffers_set.add("freqs_cis")
    print("[HuginnAudioSwift] FSDP2 compatibility: freqs_cis marked non-persistent")


def get_active_distributed_audit() -> tuple[Optional[str], Optional[str]]:
    configured = [
        ("stage34", os.environ.get(STAGE34_AUDIT_DIR_ENV, "").strip()),
        ("stage5", os.environ.get(STAGE5_AUDIT_DIR_ENV, "").strip()),
        ("realdata", os.environ.get(REALDATA_AUDIT_DIR_ENV, "").strip()),
        ("checkpoint", os.environ.get(CHECKPOINT_AUDIT_DIR_ENV, "").strip()),
    ]
    active = [(stage, path) for stage, path in configured if path]
    if len(active) > 1:
        raise RuntimeError(f"Only one distributed audit stage may be active: {active}")
    return active[0] if active else (None, None)


def _write_distributed_rank_marker(kind: str, payload: dict[str, Any]) -> None:
    _stage, audit_dir_value = get_active_distributed_audit()
    if audit_dir_value is None:
        return
    audit_dir = Path(audit_dir_value)
    audit_dir.mkdir(parents=True, exist_ok=True)
    rank = int(payload["rank"])
    marker_path = audit_dir / f"{kind}-rank-{rank}.json"
    temporary_path = marker_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_path, marker_path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_training_statistics_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Training statistics resume state is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Training statistics state is not an object: {path}")
    if payload.get("statistics_version") != TRAINING_STATS_VERSION:
        raise RuntimeError(f"Training statistics version mismatch: {payload.get('statistics_version')!r}")
    if payload.get("sampler_version") != SAMPLER_VERSION:
        raise RuntimeError(f"Training sampler version mismatch: {payload.get('sampler_version')!r}")
    pools = payload.get("pools")
    if not isinstance(pools, dict) or set(pools) != set(TRAINING_POOL_ORDER):
        raise RuntimeError(f"Training statistics pool set mismatch: {pools}")
    counts = [int(pools[name]["sample_count"]) for name in TRAINING_POOL_ORDER]
    durations = [float(pools[name]["effective_duration_seconds"]) for name in TRAINING_POOL_ORDER]
    if any(value < 0 for value in counts + durations):
        raise RuntimeError(f"Training statistics state contains negative values: {payload}")
    if sum(counts) != int(payload.get("total_samples", -1)):
        raise RuntimeError(f"Training statistics sample total is inconsistent: {payload}")
    if int(payload.get("next_global_position", -1)) != sum(counts):
        raise RuntimeError(f"Training statistics next position is inconsistent: {payload}")
    return payload


def patch_training_statistics_callback() -> None:
    """Aggregate actual forward-consumed samples and persist resumable statistics."""
    statistics_dir_value = os.environ.get(TRAINING_STATS_DIR_ENV, "").strip()
    if not statistics_dir_value:
        return
    from transformers import Trainer, TrainerCallback

    original_init = Trainer.__init__
    if getattr(original_init, "_huginn_audio_dynamic90s_training_stats_patched", False):
        return

    class Dynamic90sTrainingStatisticsCallback(TrainerCallback):
        _huginn_audio_dynamic90s_training_statistics_callback = True

        def __init__(self, tracked_model: torch.nn.Module):
            self.tracked_model = tracked_model
            self.statistics_dir = Path(statistics_dir_value).expanduser().resolve()
            self.statistics_dir.mkdir(parents=True, exist_ok=True)
            try:
                self.log_steps = int(os.environ.get(TRAINING_STATS_LOG_STEPS_ENV, "100"))
            except ValueError as exc:
                raise ValueError(f"{TRAINING_STATS_LOG_STEPS_ENV} must be an integer") from exc
            if self.log_steps <= 0:
                raise ValueError(f"{TRAINING_STATS_LOG_STEPS_ENV} must be positive")
            checkpoint_steps_value = os.environ.get(TRAINING_STATS_CHECKPOINT_STEPS_ENV, "").strip()
            try:
                self.checkpoint_steps = {
                    int(value.strip())
                    for value in checkpoint_steps_value.split(",")
                    if value.strip()
                }
            except ValueError as exc:
                raise ValueError(
                    f"{TRAINING_STATS_CHECKPOINT_STEPS_ENV} must be a comma-separated integer list"
                ) from exc
            if any(value <= 0 for value in self.checkpoint_steps):
                raise ValueError(f"{TRAINING_STATS_CHECKPOINT_STEPS_ENV} values must be positive")
            resume_state_value = os.environ.get(TRAINING_STATS_RESUME_STATE_ENV, "").strip()
            self.resume_state_path = Path(resume_state_value).expanduser().resolve() if resume_state_value else None
            self.base_state = (
                _load_training_statistics_state(self.resume_state_path)
                if self.resume_state_path is not None
                else None
            )
            self.base_counts = [
                int(self.base_state["pools"][name]["sample_count"]) if self.base_state else 0
                for name in TRAINING_POOL_ORDER
            ]
            self.base_durations = [
                float(self.base_state["pools"][name]["effective_duration_seconds"]) if self.base_state else 0.0
                for name in TRAINING_POOL_ORDER
            ]
            self.base_global_step = int(self.base_state.get("global_step", 0)) if self.base_state else 0
            self.sampler_seed = int(os.environ.get("HUGINN_DYNAMIC90S_MIXTURE_SEED", "20260730"))
            if self.base_state and int(self.base_state.get("sampler_seed", -1)) != self.sampler_seed:
                raise RuntimeError(
                    "Training statistics sampler seed differs from the resumed run: "
                    f"state={self.base_state.get('sampler_seed')} current={self.sampler_seed}"
                )
            registry_value = os.environ.get("HUGINN_DYNAMIC90S_POOL_REGISTRY", "").strip()
            if not registry_value:
                raise RuntimeError("HUGINN_DYNAMIC90S_POOL_REGISTRY is required for training statistics")
            registry = json.loads(Path(registry_value).expanduser().resolve().read_text(encoding="utf-8"))
            self.pool_sizes = {
                name: int(registry["pools"][name]["record_count"])
                for name in TRAINING_POOL_ORDER
            }
            if self.base_state:
                previous_sizes = {
                    name: int(self.base_state["pools"][name]["pool_size"])
                    for name in TRAINING_POOL_ORDER
                }
                if previous_sizes != self.pool_sizes:
                    raise RuntimeError(
                        f"Training statistics pool sizes changed across resume: "
                        f"state={previous_sizes} current={self.pool_sizes}"
                    )
            self.last_snapshot: dict[str, Any] | None = None

        @staticmethod
        def _identity() -> tuple[int, int]:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                raise RuntimeError("Dynamic-90s training statistics require torch.distributed")
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            if world_size != 4:
                raise RuntimeError(f"Dynamic-90s training statistics require world_size=4, got {world_size}")
            return rank, world_size

        def _owner(self) -> torch.nn.Module | None:
            owners = [
                module
                for module in self.tracked_model.modules()
                if "_huginn_audio_training_sample_counts" in vars(module)
            ]
            if len(owners) > 1:
                raise RuntimeError(f"Expected at most one direct training-statistics owner, found {len(owners)}")
            return owners[0] if owners else None

        def _snapshot(self, global_step: int, event: str) -> dict[str, Any]:
            rank, world_size = self._identity()
            owner = self._owner()
            local_counts = (
                list(owner._huginn_audio_training_sample_counts)
                if owner is not None
                else [0 for _ in TRAINING_POOL_ORDER]
            )
            local_durations = (
                list(owner._huginn_audio_training_duration_seconds)
                if owner is not None
                else [0.0 for _ in TRAINING_POOL_ORDER]
            )
            backend = str(torch.distributed.get_backend()).lower()
            device = torch.device("cuda", torch.cuda.current_device()) if "nccl" in backend else torch.device("cpu")
            reduced = torch.tensor(local_counts + local_durations, dtype=torch.float64, device=device)
            torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM)
            values = reduced.cpu().tolist()
            delta_counts = [int(round(value)) for value in values[: len(TRAINING_POOL_ORDER)]]
            delta_durations = [float(value) for value in values[len(TRAINING_POOL_ORDER):]]
            counts = [base + delta for base, delta in zip(self.base_counts, delta_counts)]
            durations = [base + delta for base, delta in zip(self.base_durations, delta_durations)]
            total_samples = sum(counts)
            total_duration = sum(durations)
            if total_samples <= 0 or total_duration <= 0:
                raise RuntimeError(
                    f"Training statistics snapshot is empty: counts={counts} durations={durations}"
                )
            pools: dict[str, dict[str, Any]] = {}
            for index, name in enumerate(TRAINING_POOL_ORDER):
                count = counts[index]
                duration = durations[index]
                pool_size = self.pool_sizes[name]
                pools[name] = {
                    "sample_count": count,
                    "sample_ratio": count / total_samples,
                    "effective_duration_seconds": duration,
                    "effective_duration_hours": duration / 3600.0,
                    "duration_ratio": duration / total_duration,
                    "pool_size": pool_size,
                    "completed_pool_epochs": count // pool_size,
                    "current_pool_epoch_offset": count % pool_size,
                }
            aac_samples = sum(counts[:3])
            aac_duration = sum(durations[:3])
            payload = {
                "statistics_version": TRAINING_STATS_VERSION,
                "sampler_version": SAMPLER_VERSION,
                "sampler_seed": self.sampler_seed,
                "event": event,
                "global_step": int(global_step),
                "world_size": world_size,
                "total_samples": total_samples,
                "next_global_position": total_samples,
                "total_effective_duration_seconds": total_duration,
                "total_effective_duration_hours": total_duration / 3600.0,
                "tasks": {
                    "AAC": {
                        "sample_count": aac_samples,
                        "sample_ratio": aac_samples / total_samples,
                        "effective_duration_seconds": aac_duration,
                        "duration_ratio": aac_duration / total_duration,
                    },
                    "ASR": {
                        "sample_count": counts[3],
                        "sample_ratio": counts[3] / total_samples,
                        "effective_duration_seconds": durations[3],
                        "duration_ratio": durations[3] / total_duration,
                    },
                },
                "pools": pools,
                "run_delta": {
                    "sample_counts": {
                        name: delta_counts[index]
                        for index, name in enumerate(TRAINING_POOL_ORDER)
                    },
                    "effective_duration_seconds": {
                        name: delta_durations[index]
                        for index, name in enumerate(TRAINING_POOL_ORDER)
                    },
                },
                "resume_state_path": str(self.resume_state_path) if self.resume_state_path else None,
                "rank0_writer": rank == 0,
            }
            self.last_snapshot = payload
            return payload

        def _emit(self, global_step: int, event: str) -> dict[str, Any]:
            payload = self._snapshot(global_step, event)
            rank, _world_size = self._identity()
            if rank == 0:
                _write_json_atomic(self.statistics_dir / "latest.json", payload)
                history_path = self.statistics_dir / "training_statistics.jsonl"
                with history_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                ratios = {name: round(payload["pools"][name]["sample_ratio"], 6) for name in TRAINING_POOL_ORDER}
                duration_ratios = {
                    name: round(payload["pools"][name]["duration_ratio"], 6)
                    for name in TRAINING_POOL_ORDER
                }
                print(
                    "[training-stats] "
                    f"event={event} global_step={global_step} samples={payload['total_samples']} "
                    f"hours={payload['total_effective_duration_hours']:.6f} "
                    f"sample_ratios={ratios} duration_ratios={duration_ratios}",
                    flush=True,
                )
            return payload

        @staticmethod
        def _checkpoint_dir(output_dir: str, global_step: int) -> Path:
            root = Path(output_dir)
            direct = root / f"checkpoint-{global_step}"
            if direct.is_dir():
                return direct
            matches = sorted(path for path in root.rglob(f"checkpoint-{global_step}") if path.is_dir())
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected one checkpoint-{global_step} below {root}, found {matches}"
                )
            return matches[0]

        def on_train_begin(self, args, state, control, **kwargs):
            del args, kwargs
            expected_start = int(os.environ.get("HUGINN_DYNAMIC90S_MIXTURE_START_POSITION", "0"))
            if sum(self.base_counts) != expected_start:
                raise RuntimeError(
                    "Training statistics resume position mismatch: "
                    f"state_samples={sum(self.base_counts)} dataset_start={expected_start}"
                )
            if int(state.global_step) != self.base_global_step:
                raise RuntimeError(
                    "Training statistics checkpoint step mismatch: "
                    f"state_step={self.base_global_step} trainer_step={state.global_step}"
                )
            return control

        def on_step_end(self, args, state, control, **kwargs):
            del args, kwargs
            if (
                int(state.global_step) % self.log_steps == 0
                or int(state.global_step) == int(state.max_steps)
                or int(state.global_step) in self.checkpoint_steps
            ):
                self._emit(int(state.global_step), "step")
            return control

        def on_save(self, args, state, control, **kwargs):
            del kwargs
            if self.last_snapshot is None or int(self.last_snapshot.get("global_step", -1)) != int(state.global_step):
                raise RuntimeError(
                    "Training statistics have no synchronized snapshot for checkpoint save; "
                    f"add step {state.global_step} to {TRAINING_STATS_CHECKPOINT_STEPS_ENV}"
                )
            payload = dict(self.last_snapshot)
            payload["event"] = "checkpoint"
            rank, _world_size = self._identity()
            if rank == 0:
                checkpoint_dir = self._checkpoint_dir(args.output_dir, int(state.global_step))
                _write_json_atomic(checkpoint_dir / TRAINING_STATS_STATE_FILENAME, payload)
            return control

        def on_train_end(self, args, state, control, **kwargs):
            del args, kwargs
            self._emit(int(state.global_step), "train_end")
            return control

    @wraps(original_init)
    def init_with_training_statistics(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not any(
            getattr(callback, "_huginn_audio_dynamic90s_training_statistics_callback", False)
            for callback in self.callback_handler.callbacks
        ):
            self.add_callback(Dynamic90sTrainingStatisticsCallback(self.model))

    init_with_training_statistics._huginn_audio_dynamic90s_training_stats_patched = True
    Trainer.__init__ = init_with_training_statistics
    print(
        "[HuginnAudioDynamic90s] installed cumulative training-statistics callback "
        f"dir={statistics_dir_value}"
    )


def audit_stage34_fsdp_rank(model: torch.nn.Module, prefix_mask: torch.Tensor) -> None:
    audit_stage, audit_dir_value = get_active_distributed_audit()
    if audit_dir_value is None:
        return
    if getattr(model, "_huginn_audio_dynamic90s_distributed_fsdp_audited", False):
        return
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError(f"{audit_stage} requires an initialized torch.distributed process group")

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    if world_size != 4:
        raise RuntimeError(f"{audit_stage} requires world_size=4, got {world_size}")

    def is_dtensor(parameter: torch.Tensor) -> bool:
        return all(hasattr(parameter, attribute) for attribute in ("device_mesh", "placements", "to_local"))

    trainable_tensors = {
        "lora": 0,
        "aligner": 0,
        "audio_encoder": 0,
        "huginn_base": 0,
        "other": 0,
    }
    dtensor_parameter_count = 0
    dtensor_trainable_count = 0
    for name, parameter in model.named_parameters():
        parameter_is_dtensor = is_dtensor(parameter)
        if parameter_is_dtensor:
            dtensor_parameter_count += 1
        if not parameter.requires_grad:
            continue
        if parameter_is_dtensor:
            dtensor_trainable_count += 1
        if "lora_" in name:
            normalized_name = normalize_parameter_name(name)
            if "audio_encoder" in normalized_name or is_aligner_parameter_name(normalized_name):
                raise RuntimeError(f"LoRA must not be attached to Whisper or the aligner: {name}")
            if not normalized_name.startswith("transformer."):
                raise RuntimeError(f"LoRA target is outside the Huginn transformer: {name}")
            group = "lora"
        elif is_aligner_parameter_name(name):
            group = "aligner"
        elif normalize_parameter_name(name).startswith("audio_encoder."):
            group = "audio_encoder"
        elif normalize_parameter_name(name).startswith(("transformer.", "lm_head.")):
            group = "huginn_base"
        else:
            group = "other"
        trainable_tensors[group] += 1

    expected_trainables = {
        "lora": 66,
        "aligner": 14,
        "audio_encoder": 0,
        "huginn_base": 0,
        "other": 0,
    }
    if trainable_tensors != expected_trainables:
        raise RuntimeError(
            f"{audit_stage} post-FSDP trainable split mismatch: expected={expected_trainables} "
            f"actual={trainable_tensors}"
        )
    total_trainable_tensors = sum(trainable_tensors.values())
    if dtensor_parameter_count <= 0 or dtensor_trainable_count != total_trainable_tensors:
        raise RuntimeError(
            f"{audit_stage} did not observe complete FSDP2 DTensor sharding: "
            f"dtensor_parameters={dtensor_parameter_count} "
            f"dtensor_trainables={dtensor_trainable_count} total_trainables={total_trainable_tensors}"
        )

    unit_audits: dict[str, dict[str, int]] = {}
    for expected_class_name in FSDP_UNIT_CLASS_NAMES:
        matching_units = [
            module
            for module in model.modules()
            if any(base.__name__ == expected_class_name for base in type(module).__mro__)
        ]
        if len(matching_units) != 1:
            raise RuntimeError(
                f"Expected exactly one {expected_class_name}, found {len(matching_units)}"
            )
        unit_parameters = list(matching_units[0].parameters())
        unit_dtensors = sum(is_dtensor(parameter) for parameter in unit_parameters)
        unit_trainables = sum(parameter.requires_grad for parameter in unit_parameters)
        if not unit_parameters or unit_dtensors != len(unit_parameters):
            raise RuntimeError(
                f"FSDP unit is not completely sharded: class={expected_class_name} "
                f"parameters={len(unit_parameters)} dtensors={unit_dtensors}"
            )
        expected_unit_trainables = FSDP_UNIT_EXPECTED_TRAINABLE_TENSORS[expected_class_name]
        if unit_trainables != expected_unit_trainables:
            raise RuntimeError(
                f"Unexpected trainable tensor count inside {expected_class_name}: "
                f"expected={expected_unit_trainables} actual={unit_trainables}"
            )
        unit_audits[expected_class_name] = {
            "parameter_count": len(unit_parameters),
            "dtensor_parameter_count": unit_dtensors,
            "trainable_parameter_count": unit_trainables,
        }

    payload = {
        "kind": "fsdp",
        "stage": audit_stage,
        "rank": rank,
        "world_size": world_size,
        "cuda_device": torch.cuda.current_device(),
        "dtensor_parameter_count": dtensor_parameter_count,
        "dtensor_trainable_count": dtensor_trainable_count,
        "trainable_tensors": trainable_tensors,
        "fsdp_units": unit_audits,
        "valid_prefix_tokens": [int(value) for value in prefix_mask.sum(dim=1).tolist()],
    }
    _write_distributed_rank_marker("fsdp", payload)
    print(
        f"[HuginnAudioDynamic90s] {audit_stage.upper()}_FSDP_RANK_AUDIT "
        f"rank={rank} world_size={world_size} dtensor_parameters={dtensor_parameter_count} "
        f"dtensor_trainables={dtensor_trainable_count} trainable_tensors={trainable_tensors} "
        f"fsdp_units={unit_audits} "
        f"valid_prefix_tokens={payload['valid_prefix_tokens']}",
        flush=True,
    )
    model._huginn_audio_dynamic90s_distributed_fsdp_audited = True


def patch_stage34_optimizer_step_callback() -> None:
    if not os.environ.get(STAGE34_AUDIT_DIR_ENV, "").strip():
        return
    from transformers import Trainer, TrainerCallback

    original_init = Trainer.__init__
    if getattr(original_init, "_huginn_audio_dynamic90s_stage34_patched", False):
        return

    class Stage34OptimizerStepCallback(TrainerCallback):
        _huginn_audio_dynamic90s_stage34_callback = True

        def on_step_end(self, args, state, control, **kwargs):
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                raise RuntimeError("Stage 3-4 optimizer callback requires initialized torch.distributed")
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            if world_size != 4:
                raise RuntimeError(f"Stage 3-4 optimizer callback requires world_size=4, got {world_size}")
            if int(state.global_step) != 1 or int(state.max_steps) != 1:
                raise RuntimeError(
                    "Stage 3-4 optimizer callback expected global_step=max_steps=1, "
                    f"got global_step={state.global_step} max_steps={state.max_steps}"
                )
            optimizer = kwargs.get("optimizer")
            payload = {
                "kind": "optimizer_step",
                "rank": rank,
                "world_size": world_size,
                "global_step": int(state.global_step),
                "max_steps": int(state.max_steps),
                "optimizer_type": type(optimizer).__name__ if optimizer is not None else None,
            }
            _write_distributed_rank_marker("optimizer-step", payload)
            print(
                "[HuginnAudioDynamic90s] STAGE34_OPTIMIZER_STEP_AUDIT "
                f"rank={rank} world_size={world_size} global_step={state.global_step} "
                f"max_steps={state.max_steps} optimizer={payload['optimizer_type']}",
                flush=True,
            )
            return control

    @wraps(original_init)
    def init_with_stage34_callback(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not any(
            getattr(callback, "_huginn_audio_dynamic90s_stage34_callback", False)
            for callback in self.callback_handler.callbacks
        ):
            self.add_callback(Stage34OptimizerStepCallback())

    init_with_stage34_callback._huginn_audio_dynamic90s_stage34_patched = True
    Trainer.__init__ = init_with_stage34_callback
    print("[HuginnAudioDynamic90s] installed Stage 3-4 optimizer-step callback")


def patch_stage5_stability_callback() -> None:
    if not os.environ.get(STAGE5_AUDIT_DIR_ENV, "").strip():
        return
    from transformers import Trainer, TrainerCallback

    try:
        expected_max_steps = int(os.environ.get(STAGE5_MAX_STEPS_ENV, "20"))
    except ValueError as exc:
        raise ValueError(f"{STAGE5_MAX_STEPS_ENV} must be an integer") from exc
    if expected_max_steps <= 1:
        raise ValueError(f"Stage 5 requires more than one optimizer step, got {expected_max_steps}")

    original_init = Trainer.__init__
    if getattr(original_init, "_huginn_audio_dynamic90s_stage5_patched", False):
        return

    class Stage5StabilityCallback(TrainerCallback):
        _huginn_audio_dynamic90s_stage5_callback = True

        def __init__(self):
            self.finite_loss_log_count = 0
            self.finite_grad_norm_log_count = 0
            self.optimizer_type = None

        def _distributed_identity(self) -> tuple[int, int]:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                raise RuntimeError("Stage 5 requires an initialized torch.distributed process group")
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            if world_size != 4:
                raise RuntimeError(f"Stage 5 requires world_size=4, got {world_size}")
            return rank, world_size

        def on_log(self, args, state, control, logs=None, **kwargs):
            del args, state, kwargs
            logs = logs or {}
            for key in ("loss", "grad_norm"):
                value = logs.get(key)
                if value is None:
                    continue
                numeric_value = float(value)
                if not math.isfinite(numeric_value):
                    raise RuntimeError(f"Stage 5 observed non-finite {key}: {numeric_value}")
                if key == "loss":
                    self.finite_loss_log_count += 1
                else:
                    self.finite_grad_norm_log_count += 1
            return control

        def on_step_end(self, args, state, control, **kwargs):
            del args
            rank, world_size = self._distributed_identity()
            if int(state.max_steps) != expected_max_steps:
                raise RuntimeError(
                    f"Stage 5 max_steps mismatch: expected={expected_max_steps} actual={state.max_steps}"
                )
            optimizer = kwargs.get("optimizer")
            if optimizer is not None:
                self.optimizer_type = type(optimizer).__name__
            if int(state.global_step) == expected_max_steps:
                payload = {
                    "kind": "optimizer_step",
                    "stage": "stage5",
                    "rank": rank,
                    "world_size": world_size,
                    "global_step": int(state.global_step),
                    "max_steps": int(state.max_steps),
                    "optimizer_type": self.optimizer_type,
                }
                _write_distributed_rank_marker("optimizer-step", payload)
            return control

        def on_train_end(self, args, state, control, **kwargs):
            del args, kwargs
            rank, world_size = self._distributed_identity()
            if int(state.global_step) != expected_max_steps or int(state.max_steps) != expected_max_steps:
                raise RuntimeError(
                    "Stage 5 did not complete every optimizer step: "
                    f"global_step={state.global_step} max_steps={state.max_steps} "
                    f"expected={expected_max_steps}"
                )
            if self.finite_loss_log_count != expected_max_steps:
                raise RuntimeError(
                    "Stage 5 did not audit one finite loss per optimizer step: "
                    f"expected={expected_max_steps} actual={self.finite_loss_log_count}"
                )
            if self.finite_grad_norm_log_count != expected_max_steps:
                raise RuntimeError(
                    "Stage 5 did not audit one finite gradient norm per optimizer step: "
                    f"expected={expected_max_steps} actual={self.finite_grad_norm_log_count}"
                )
            payload = {
                "kind": "stability",
                "stage": "stage5",
                "rank": rank,
                "world_size": world_size,
                "global_step": int(state.global_step),
                "max_steps": int(state.max_steps),
                "finite_loss_log_count": self.finite_loss_log_count,
                "finite_grad_norm_log_count": self.finite_grad_norm_log_count,
                "optimizer_type": self.optimizer_type,
            }
            _write_distributed_rank_marker("stability", payload)
            print(
                "[HuginnAudioDynamic90s] STAGE5_STABILITY_AUDIT "
                f"rank={rank} world_size={world_size} global_step={state.global_step} "
                f"finite_losses={self.finite_loss_log_count} "
                f"finite_grad_norms={self.finite_grad_norm_log_count} "
                f"optimizer={self.optimizer_type}",
                flush=True,
            )
            return control

    @wraps(original_init)
    def init_with_stage5_callback(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not any(
            getattr(callback, "_huginn_audio_dynamic90s_stage5_callback", False)
            for callback in self.callback_handler.callbacks
        ):
            self.add_callback(Stage5StabilityCallback())

    init_with_stage5_callback._huginn_audio_dynamic90s_stage5_patched = True
    Trainer.__init__ = init_with_stage5_callback
    print(
        "[HuginnAudioDynamic90s] installed Stage 5 stability callback "
        f"expected_max_steps={expected_max_steps}"
    )


def patch_realdata_stability_callback() -> None:
    """Audit a short FSDP4 run that consumes the real indexed mixture."""
    if not os.environ.get(REALDATA_AUDIT_DIR_ENV, "").strip():
        return
    from transformers import Trainer, TrainerCallback

    try:
        expected_max_steps = int(os.environ.get(REALDATA_MAX_STEPS_ENV, "8"))
    except ValueError as exc:
        raise ValueError(f"{REALDATA_MAX_STEPS_ENV} must be an integer") from exc
    if expected_max_steps <= 1:
        raise ValueError(f"Real-data stability gate requires more than one optimizer step, got {expected_max_steps}")

    original_init = Trainer.__init__
    if getattr(original_init, "_huginn_audio_dynamic90s_realdata_patched", False):
        return

    class RealDataStabilityCallback(TrainerCallback):
        _huginn_audio_dynamic90s_realdata_callback = True

        def __init__(self, tracked_model: torch.nn.Module):
            self.tracked_model = tracked_model
            self.finite_loss_log_count = 0
            self.finite_grad_norm_log_count = 0
            self.optimizer_type = None

        def _distributed_identity(self) -> tuple[int, int]:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                raise RuntimeError("Real-data gate requires an initialized torch.distributed process group")
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            if world_size != 4:
                raise RuntimeError(f"Real-data gate requires world_size=4, got {world_size}")
            return rank, world_size

        def _audio_statistics(self) -> dict[str, int]:
            statistics_owner_key = "_huginn_audio_realdata_sample_count"
            matching_modules = [
                module
                for module in self.tracked_model.modules()
                # PEFT and FSDP wrappers proxy unknown attributes to their
                # wrapped module. ``hasattr`` therefore sees the same counters
                # through several wrapper levels. Only the module whose own
                # __dict__ stores the counters is the real statistics owner.
                if statistics_owner_key in vars(module)
            ]
            if len(matching_modules) != 1:
                raise RuntimeError(
                    "Real-data gate expected exactly one direct owner of runtime audio statistics, "
                    f"found {len(matching_modules)}"
                )
            module = matching_modules[0]
            return {
                "audio_batch_count": int(module._huginn_audio_realdata_batch_count),
                "audio_sample_count": int(module._huginn_audio_realdata_sample_count),
                "realized_prefix_tokens": int(module._huginn_audio_realdata_prefix_tokens),
                "realized_audio_tokens": int(module._huginn_audio_realdata_audio_tokens),
                "min_audio_tokens": int(module._huginn_audio_realdata_min_audio_tokens),
                "max_audio_tokens": int(module._huginn_audio_realdata_max_audio_tokens),
            }

        def on_log(self, args, state, control, logs=None, **kwargs):
            del args, state, kwargs
            logs = logs or {}
            for key in ("loss", "grad_norm"):
                value = logs.get(key)
                if value is None:
                    continue
                numeric_value = float(value)
                if not math.isfinite(numeric_value):
                    raise RuntimeError(f"Real-data gate observed non-finite {key}: {numeric_value}")
                if key == "loss":
                    self.finite_loss_log_count += 1
                else:
                    self.finite_grad_norm_log_count += 1
            return control

        def on_step_end(self, args, state, control, **kwargs):
            del args
            rank, world_size = self._distributed_identity()
            if int(state.max_steps) != expected_max_steps:
                raise RuntimeError(
                    f"Real-data max_steps mismatch: expected={expected_max_steps} actual={state.max_steps}"
                )
            optimizer = kwargs.get("optimizer")
            if optimizer is not None:
                self.optimizer_type = type(optimizer).__name__
            if int(state.global_step) == expected_max_steps:
                payload = {
                    "kind": "optimizer_step",
                    "stage": "realdata",
                    "rank": rank,
                    "world_size": world_size,
                    "global_step": int(state.global_step),
                    "max_steps": int(state.max_steps),
                    "optimizer_type": self.optimizer_type,
                }
                _write_distributed_rank_marker("optimizer-step", payload)
            return control

        def on_train_end(self, args, state, control, **kwargs):
            del args, kwargs
            rank, world_size = self._distributed_identity()
            if int(state.global_step) != expected_max_steps or int(state.max_steps) != expected_max_steps:
                raise RuntimeError(
                    "Real-data gate did not complete every optimizer step: "
                    f"global_step={state.global_step} max_steps={state.max_steps} expected={expected_max_steps}"
                )
            if self.finite_loss_log_count != expected_max_steps:
                raise RuntimeError(
                    "Real-data gate did not audit one finite loss per optimizer step: "
                    f"expected={expected_max_steps} actual={self.finite_loss_log_count}"
                )
            if self.finite_grad_norm_log_count != expected_max_steps:
                raise RuntimeError(
                    "Real-data gate did not audit one finite gradient norm per optimizer step: "
                    f"expected={expected_max_steps} actual={self.finite_grad_norm_log_count}"
                )
            audio_statistics = self._audio_statistics()
            if audio_statistics["audio_batch_count"] != expected_max_steps:
                raise RuntimeError(
                    "Real-data gate audio batch count mismatch: "
                    f"expected={expected_max_steps} actual={audio_statistics['audio_batch_count']}"
                )
            if audio_statistics["audio_sample_count"] != expected_max_steps:
                raise RuntimeError(
                    "Real-data gate expects local batch size one on every step: "
                    f"expected_samples={expected_max_steps} actual={audio_statistics['audio_sample_count']}"
                )
            if (
                audio_statistics["realized_audio_tokens"] <= 0
                or audio_statistics["min_audio_tokens"] < 0
                or audio_statistics["max_audio_tokens"] > 750
                or audio_statistics["min_audio_tokens"] > audio_statistics["max_audio_tokens"]
            ):
                raise RuntimeError(f"Invalid real-data dynamic audio-token statistics: {audio_statistics}")
            payload = {
                "kind": "realdata_stability",
                "stage": "realdata",
                "rank": rank,
                "world_size": world_size,
                "global_step": int(state.global_step),
                "max_steps": int(state.max_steps),
                "finite_loss_log_count": self.finite_loss_log_count,
                "finite_grad_norm_log_count": self.finite_grad_norm_log_count,
                "optimizer_type": self.optimizer_type,
                **audio_statistics,
            }
            _write_distributed_rank_marker("realdata-stability", payload)
            print(
                "[HuginnAudioDynamic90s] REALDATA_STABILITY_AUDIT "
                f"rank={rank} world_size={world_size} global_step={state.global_step} "
                f"finite_losses={self.finite_loss_log_count} "
                f"finite_grad_norms={self.finite_grad_norm_log_count} "
                f"audio_samples={audio_statistics['audio_sample_count']} "
                f"audio_tokens={audio_statistics['realized_audio_tokens']} "
                f"audio_token_range=[{audio_statistics['min_audio_tokens']},"
                f"{audio_statistics['max_audio_tokens']}] optimizer={self.optimizer_type}",
                flush=True,
            )
            return control

    @wraps(original_init)
    def init_with_realdata_callback(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not any(
            getattr(callback, "_huginn_audio_dynamic90s_realdata_callback", False)
            for callback in self.callback_handler.callbacks
        ):
            self.add_callback(RealDataStabilityCallback(self.model))

    init_with_realdata_callback._huginn_audio_dynamic90s_realdata_patched = True
    Trainer.__init__ = init_with_realdata_callback
    print(
        "[HuginnAudioDynamic90s] installed real-data stability callback "
        f"expected_max_steps={expected_max_steps}"
    )


def patch_checkpoint_resume_callback() -> None:
    """Audit state restoration and new updates in each cold-start phase."""
    audit_dir = os.environ.get(CHECKPOINT_AUDIT_DIR_ENV, "").strip()
    if not audit_dir:
        return
    from transformers import Trainer, TrainerCallback

    phase = os.environ.get(CHECKPOINT_PHASE_ENV, "").strip()
    if phase not in {"save", "resume"}:
        raise ValueError(f"{CHECKPOINT_PHASE_ENV} must be save or resume, got {phase!r}")
    try:
        expected_start_step = int(os.environ[CHECKPOINT_START_STEP_ENV])
        expected_end_step = int(os.environ[CHECKPOINT_END_STEP_ENV])
    except (KeyError, ValueError) as exc:
        raise ValueError("Checkpoint audit start/end steps must be configured as integers") from exc
    expected_updates = expected_end_step - expected_start_step
    launch_id = os.environ.get(CHECKPOINT_LAUNCH_ID_ENV, "").strip()
    if not launch_id:
        raise ValueError(f"{CHECKPOINT_LAUNCH_ID_ENV} must identify the fresh process launch")
    if expected_start_step < 0 or expected_updates <= 0:
        raise ValueError(
            f"Invalid checkpoint audit step window: start={expected_start_step} end={expected_end_step}"
        )

    original_init = Trainer.__init__
    if getattr(original_init, "_huginn_audio_dynamic90s_checkpoint_patched", False):
        return

    class CheckpointResumeCallback(TrainerCallback):
        _huginn_audio_dynamic90s_checkpoint_callback = True

        def __init__(self, tracked_model: torch.nn.Module):
            self.tracked_model = tracked_model
            self.finite_loss_log_count = 0
            self.finite_grad_norm_log_count = 0
            self.optimizer_type = None
            self.observed_start_step = None

        @staticmethod
        def _identity() -> tuple[int, int]:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                raise RuntimeError("Checkpoint smoke requires initialized torch.distributed")
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            if world_size != 4:
                raise RuntimeError(f"Checkpoint smoke requires world_size=4, got {world_size}")
            return rank, world_size

        @staticmethod
        def _optimizer_steps(optimizer: Any) -> list[int]:
            inner = getattr(optimizer, "optimizer", optimizer)
            values: list[int] = []
            for state in getattr(inner, "state", {}).values():
                if not isinstance(state, dict) or "step" not in state:
                    continue
                step = state["step"]
                if torch.is_tensor(step):
                    if step.numel() == 1:
                        values.append(int(step.item()))
                elif isinstance(step, (int, float)):
                    values.append(int(step))
            return values

        def _audio_statistics(self) -> dict[str, int]:
            owners = [
                module
                for module in self.tracked_model.modules()
                if "_huginn_audio_realdata_sample_count" in vars(module)
            ]
            if len(owners) != 1:
                raise RuntimeError(
                    "Checkpoint smoke expected one direct runtime-statistics owner, "
                    f"found {len(owners)}"
                )
            module = owners[0]
            return {
                "audio_batch_count": int(module._huginn_audio_realdata_batch_count),
                "audio_sample_count": int(module._huginn_audio_realdata_sample_count),
                "realized_prefix_tokens": int(module._huginn_audio_realdata_prefix_tokens),
                "realized_audio_tokens": int(module._huginn_audio_realdata_audio_tokens),
                "min_audio_tokens": int(module._huginn_audio_realdata_min_audio_tokens),
                "max_audio_tokens": int(module._huginn_audio_realdata_max_audio_tokens),
            }

        def on_train_begin(self, args, state, control, **kwargs):
            del args
            rank, world_size = self._identity()
            self.observed_start_step = int(state.global_step)
            if self.observed_start_step != expected_start_step:
                raise RuntimeError(
                    f"Checkpoint {phase} phase start step mismatch: "
                    f"expected={expected_start_step} actual={self.observed_start_step}"
                )
            optimizer = kwargs.get("optimizer")
            scheduler = kwargs.get("lr_scheduler")
            self.optimizer_type = type(optimizer).__name__ if optimizer is not None else None
            optimizer_steps = self._optimizer_steps(optimizer) if optimizer is not None else []
            scheduler_last_epoch = int(getattr(scheduler, "last_epoch", -999)) if scheduler is not None else None
            learning_rates = [float(group["lr"]) for group in optimizer.param_groups] if optimizer is not None else []
            if phase == "resume":
                if not optimizer_steps or min(optimizer_steps) != expected_start_step or max(optimizer_steps) != expected_start_step:
                    raise RuntimeError(
                        "Resume optimizer state was not restored to the checkpoint step: "
                        f"expected={expected_start_step} observed={optimizer_steps[:20]}"
                    )
                if scheduler_last_epoch != expected_start_step:
                    raise RuntimeError(
                        "Resume scheduler state mismatch: "
                        f"expected_last_epoch={expected_start_step} actual={scheduler_last_epoch}"
                    )
                if not learning_rates or any(value <= 0 for value in learning_rates):
                    raise RuntimeError(f"Resume phase restored non-positive learning rates: {learning_rates}")
            payload = {
                "kind": "checkpoint_start",
                "stage": "checkpoint",
                "phase": phase,
                "rank": rank,
                "world_size": world_size,
                "pid": os.getpid(),
                "launch_id": launch_id,
                "global_step": self.observed_start_step,
                "optimizer_type": self.optimizer_type,
                "optimizer_step_min": min(optimizer_steps) if optimizer_steps else None,
                "optimizer_step_max": max(optimizer_steps) if optimizer_steps else None,
                "optimizer_state_count": len(optimizer_steps),
                "scheduler_last_epoch": scheduler_last_epoch,
                "learning_rates": learning_rates,
            }
            _write_distributed_rank_marker("checkpoint-start", payload)
            print(
                "[HuginnAudioDynamic90s] CHECKPOINT_START_AUDIT "
                f"phase={phase} rank={rank} global_step={self.observed_start_step} "
                f"optimizer_steps=[{payload['optimizer_step_min']},{payload['optimizer_step_max']}] "
                f"scheduler_last_epoch={scheduler_last_epoch} learning_rates={learning_rates}",
                flush=True,
            )
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            del args, state, kwargs
            for key in ("loss", "grad_norm"):
                value = (logs or {}).get(key)
                if value is None:
                    continue
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise RuntimeError(f"Checkpoint {phase} phase observed non-finite {key}: {numeric}")
                if key == "loss":
                    self.finite_loss_log_count += 1
                else:
                    self.finite_grad_norm_log_count += 1
            return control

        def on_train_end(self, args, state, control, **kwargs):
            del args, kwargs
            rank, world_size = self._identity()
            if int(state.global_step) != expected_end_step:
                raise RuntimeError(
                    f"Checkpoint {phase} phase end step mismatch: "
                    f"expected={expected_end_step} actual={state.global_step}"
                )
            if self.finite_loss_log_count != expected_updates or self.finite_grad_norm_log_count != expected_updates:
                raise RuntimeError(
                    f"Checkpoint {phase} finite-log count mismatch: updates={expected_updates} "
                    f"losses={self.finite_loss_log_count} grad_norms={self.finite_grad_norm_log_count}"
                )
            audio = self._audio_statistics()
            if audio["audio_batch_count"] != expected_updates or audio["audio_sample_count"] != expected_updates:
                raise RuntimeError(
                    f"Checkpoint {phase} runtime audio count mismatch: expected={expected_updates} actual={audio}"
                )
            payload = {
                "kind": "checkpoint_end",
                "stage": "checkpoint",
                "phase": phase,
                "rank": rank,
                "world_size": world_size,
                "start_global_step": self.observed_start_step,
                "global_step": int(state.global_step),
                "new_optimizer_steps": expected_updates,
                "finite_loss_log_count": self.finite_loss_log_count,
                "finite_grad_norm_log_count": self.finite_grad_norm_log_count,
                "optimizer_type": self.optimizer_type,
                **audio,
            }
            _write_distributed_rank_marker("checkpoint-end", payload)
            print(
                "[HuginnAudioDynamic90s] CHECKPOINT_END_AUDIT "
                f"phase={phase} rank={rank} step_window=[{self.observed_start_step},{state.global_step}] "
                f"finite_losses={self.finite_loss_log_count} finite_grad_norms={self.finite_grad_norm_log_count} "
                f"audio_samples={audio['audio_sample_count']} audio_tokens={audio['realized_audio_tokens']}",
                flush=True,
            )
            return control

    @wraps(original_init)
    def init_with_checkpoint_callback(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not any(
            getattr(callback, "_huginn_audio_dynamic90s_checkpoint_callback", False)
            for callback in self.callback_handler.callbacks
        ):
            self.add_callback(CheckpointResumeCallback(self.model))

    init_with_checkpoint_callback._huginn_audio_dynamic90s_checkpoint_patched = True
    Trainer.__init__ = init_with_checkpoint_callback
    print(
        "[HuginnAudioDynamic90s] installed checkpoint-resume callback "
        f"phase={phase} step_window=[{expected_start_step},{expected_end_step}]"
    )


def patch_checkpoint_rng_restore_audit() -> None:
    """Verify Trainer restored the exact per-rank RNG payload from checkpoint."""
    if not os.environ.get(CHECKPOINT_AUDIT_DIR_ENV, "").strip():
        return
    if os.environ.get(CHECKPOINT_PHASE_ENV, "").strip() != "resume":
        return
    from transformers import Trainer

    original = Trainer._load_rng_state
    if getattr(original, "_huginn_audio_dynamic90s_rng_audit_patched", False):
        return

    def numpy_rng_equal(left: Any, right: Any) -> bool:
        return (
            isinstance(left, tuple)
            and isinstance(right, tuple)
            and len(left) == len(right) == 5
            and left[0] == right[0]
            and np.array_equal(left[1], right[1])
            and left[2:] == right[2:]
        )

    @wraps(original)
    def patched(self, resume_from_checkpoint):
        original(self, resume_from_checkpoint)
        checkpoint_dir = Path(resume_from_checkpoint)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        candidates = [checkpoint_dir / f"rng_state_{rank}.pth", checkpoint_dir / "rng_state.pth"]
        rng_path = next((path for path in candidates if path.is_file()), None)
        if rng_path is None:
            raise FileNotFoundError(f"No RNG state file for rank {rank} in {checkpoint_dir}")
        saved = torch.load(rng_path, map_location="cpu", weights_only=False)
        saved_cpu = saved.get("cpu")
        checks = {
            "python": saved.get("python") == random.getstate(),
            "numpy": numpy_rng_equal(saved.get("numpy"), np.random.get_state()),
            "cpu": torch.is_tensor(saved_cpu) and torch.equal(saved_cpu, torch.random.get_rng_state()),
        }
        if "cuda" in saved:
            current_cuda = torch.cuda.random.get_rng_state_all()
            saved_cuda = saved["cuda"]
            checks["cuda"] = (
                isinstance(saved_cuda, (list, tuple))
                and len(saved_cuda) == len(current_cuda)
                and all(torch.equal(left.cpu(), right.cpu()) for left, right in zip(saved_cuda, current_cuda))
            )
        if not all(checks.values()):
            raise RuntimeError(
                f"Checkpoint RNG restoration mismatch for rank {rank}: path={rng_path} checks={checks}"
            )
        payload = {
            "kind": "rng_restore",
            "stage": "checkpoint",
            "phase": "resume",
            "rank": rank,
            "world_size": world_size,
            "rng_path": str(rng_path),
            "checks": checks,
        }
        _write_distributed_rank_marker("rng-restore", payload)
        print(
            "[HuginnAudioDynamic90s] CHECKPOINT_RNG_RESTORE_AUDIT "
            f"rank={rank} path={rng_path} checks={checks}",
            flush=True,
        )

    patched._huginn_audio_dynamic90s_rng_audit_patched = True
    Trainer._load_rng_state = patched
    print("[HuginnAudioDynamic90s] installed checkpoint RNG-restore audit")


def patch_huginn_audio_train_chain_audit(model: torch.nn.Module) -> None:
    """Log one actual audio-prefix pass for the distributed stability smoke."""
    requested = os.environ.get(TRAIN_CHAIN_AUDIT_ENV, "").strip().lower()
    if requested not in {"1", "true", "yes"}:
        return
    if getattr(model, "_huginn_audio_train_chain_audit_patched", False):
        return
    original_build_audio_prefix = model.build_audio_prefix

    def audited_build_audio_prefix(
        self,
        audio_input_features,
        audio_attention_mask=None,
        audio_segment_feature_lengths=None,
        audio_segment_mask=None,
    ):
        prefix, prefix_mask = original_build_audio_prefix(
            audio_input_features,
            audio_attention_mask,
            audio_segment_feature_lengths,
            audio_segment_mask,
        )
        if self.training:
            audit_stage34_fsdp_rank(self, prefix_mask)
            audit_stage, _audit_dir = get_active_distributed_audit()
            if audit_stage in {"realdata", "checkpoint"}:
                boundary_tokens = int(self.audio_bos is not None) + int(self.audio_eos is not None)
                valid_prefix_tokens = [int(value) for value in prefix_mask.sum(dim=1).tolist()]
                valid_audio_tokens = [value - boundary_tokens for value in valid_prefix_tokens]
                if not valid_audio_tokens or any(value < 0 or value > 750 for value in valid_audio_tokens):
                    raise RuntimeError(
                        "Real-data runtime audio-token count is outside [0, 750]: "
                        f"prefix_tokens={valid_prefix_tokens} boundary_tokens={boundary_tokens}"
                    )
                if not hasattr(self, "_huginn_audio_realdata_sample_count"):
                    self._huginn_audio_realdata_batch_count = 0
                    self._huginn_audio_realdata_sample_count = 0
                    self._huginn_audio_realdata_prefix_tokens = 0
                    self._huginn_audio_realdata_audio_tokens = 0
                    self._huginn_audio_realdata_min_audio_tokens = min(valid_audio_tokens)
                    self._huginn_audio_realdata_max_audio_tokens = max(valid_audio_tokens)
                self._huginn_audio_realdata_batch_count += 1
                self._huginn_audio_realdata_sample_count += len(valid_audio_tokens)
                self._huginn_audio_realdata_prefix_tokens += sum(valid_prefix_tokens)
                self._huginn_audio_realdata_audio_tokens += sum(valid_audio_tokens)
                self._huginn_audio_realdata_min_audio_tokens = min(
                    self._huginn_audio_realdata_min_audio_tokens,
                    min(valid_audio_tokens),
                )
                self._huginn_audio_realdata_max_audio_tokens = max(
                    self._huginn_audio_realdata_max_audio_tokens,
                    max(valid_audio_tokens),
                )
        if self.training and os.environ.get("RANK", "0") == "0" and not getattr(
            self, "_huginn_audio_prefix_audit_logged", False
        ):
            audio_encoder_trainable = sum(
                parameter.numel() for parameter in self.audio_encoder.parameters() if parameter.requires_grad
            )
            boundary_tokens = int(self.audio_bos is not None) + int(self.audio_eos is not None)
            valid_prefix_tokens = prefix_mask.sum(dim=1).tolist()
            max_audio_tokens = int(getattr(self.config, "audio_max_token_count", 750))
            if any(tokens < boundary_tokens or tokens - boundary_tokens > max_audio_tokens for tokens in valid_prefix_tokens):
                raise RuntimeError(
                    "Audio prefix token count exceeds dynamic bounds: "
                    f"valid_prefix_tokens={valid_prefix_tokens} max_audio_tokens={max_audio_tokens}"
                )
            if audio_encoder_trainable != 0:
                raise RuntimeError(
                    f"Audio encoder must remain frozen, found {audio_encoder_trainable} trainable local parameters"
                )
            print(
                "[HuginnAudioSwift] train_chain_audit_audio "
                f"audio_features={tuple(audio_input_features.shape)} audio_prefix={tuple(prefix.shape)} "
                f"valid_prefix_tokens={valid_prefix_tokens} max_audio_tokens={max_audio_tokens} "
                f"boundary_tokens={boundary_tokens} audio_encoder_trainable={audio_encoder_trainable}"
            )
            self._huginn_audio_prefix_audit_logged = True
        return prefix, prefix_mask

    model.build_audio_prefix = MethodType(audited_build_audio_prefix, model)
    model._huginn_audio_train_chain_audit_patched = True
    print("[HuginnAudioSwift] installed train-chain audit hook")


def print_train_chain_parameter_audit(model: torch.nn.Module) -> None:
    """Assert the requested full-tuning split before Swift wraps the model with FSDP."""
    requested = os.environ.get(TRAIN_CHAIN_AUDIT_ENV, "").strip().lower()
    if requested not in {"1", "true", "yes"}:
        return
    groups = {"audio_encoder": 0, "aligner": 0, "huginn_backbone": 0, "other": 0}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        normalized_name = normalize_parameter_name(name)
        if normalized_name.startswith("audio_encoder."):
            groups["audio_encoder"] += parameter.numel()
        elif is_aligner_parameter_name(name):
            groups["aligner"] += parameter.numel()
        elif normalized_name.startswith(("transformer.", "lm_head.")):
            groups["huginn_backbone"] += parameter.numel()
        else:
            groups["other"] += parameter.numel()
    if groups["audio_encoder"] != 0:
        raise RuntimeError(f"Audio encoder must be frozen before FSDP wrapping: {groups}")
    if groups["aligner"] <= 0 or groups["huginn_backbone"] <= 0:
        raise RuntimeError(f"Full-tuning parameter split is incomplete before FSDP wrapping: {groups}")
    print(f"[HuginnAudioSwift] train_chain_audit_parameters={groups}")


def build_huginn_audio_model(model_dir: str):
    whisper_config = AutoConfig.from_pretrained(
        WHISPER_MODEL_DIR,
        trust_remote_code=True,
    )
    config = AutoConfig.from_pretrained(
        model_dir,
        trust_remote_code=True,
    )
    config.audio_encoder_name = str(WHISPER_MODEL_DIR)
    config.audio_encoder_hidden_size = int(getattr(whisper_config, "d_model", 1280))
    config.audio_dynamic_tokens = True
    config.audio_token_duration_ms = AUDIO_TOKEN_DURATION_MS
    config.audio_reference_30s_token_count = 250
    config.audio_max_token_count = 750
    config.audio_chunk_seconds = DEFAULT_AUDIO_CHUNK_SECONDS
    config.audio_max_seconds = DEFAULT_MAX_AUDIO_SECONDS
    config.audio_compressor_kernel_size = DYNAMIC_COMPRESSOR_KERNEL
    config.audio_compressor_stride = DYNAMIC_COMPRESSOR_STRIDE
    config.freeze_audio_encoder = True
    config.freeze_text_backbone = False

    model = AutoModelForCausalLM.from_config(
        config,
        trust_remote_code=True,
    )
    if not hasattr(model, "load_huginn_backbone_from_pretrained"):
        raise AttributeError("Audio Huginn model is missing load_huginn_backbone_from_pretrained")

    enable_fsdp2_nonpersistent_rope_buffer(model)
    patch_huginn_audio_train_chain_audit(model)

    load_result = model.load_huginn_backbone_from_pretrained(
        str(HUGINN_MODEL_DIR),
        torch_dtype=torch.float32,
    )
    print_missing_key_summary(load_result.missing_keys, load_result.unexpected_keys)
    print_train_chain_parameter_audit(model)
    initial_aligner_checkpoint = os.environ.get(INIT_ALIGNER_CHECKPOINT_ENV)
    if initial_aligner_checkpoint:
        aligner_report = load_initial_aligner_state(model, Path(initial_aligner_checkpoint))
        print(f"[HuginnAudioSwift] initial_aligner_restore={aligner_report}")
        if not aligner_report["restored_boundary_embeddings"]:
            print(
                "[HuginnAudioSwift] warning: initial checkpoint lacks audio_bos/audio_eos; "
                "newly initialized boundary embeddings will be trained and saved by this run."
            )
    return patch_huginn_audio_shift_loss(model)


class HuginnAudioTemplate(Template):
    use_model = False
    support_padding_free = False

    def init_processor(self, processor: Processor):
        super().init_processor(processor)
        self.audio_feature_extractor = processor.feature_extractor
        self.audio_sampling_rate = int(
            getattr(self.audio_feature_extractor, "sampling_rate", DEFAULT_SAMPLE_RATE)
        )

    def replace_tag(self, media_type: str, index: int, inputs: StdTemplateInputs):
        if media_type == "audio":
            return []
        return super().replace_tag(media_type, index, inputs)

    def _load_audio_item(self, audio_item: Any) -> np.ndarray:
        return load_audio_item(
            audio_item,
            target_sr=self.audio_sampling_rate,
            max_audio_seconds=DEFAULT_MAX_AUDIO_SECONDS,
        )

    def _encode(self, inputs: StdTemplateInputs) -> dict[str, Any]:
        encoded = super()._encode(inputs)
        if not getattr(inputs, "audios", None):
            return encoded
        if len(inputs.audios) != 1:
            raise ValueError("Huginn audio Swift template currently supports exactly one audio clip per sample.")

        audit_consumed_audio_position(inputs.audios[0])
        audio_item = inputs.audios[0]
        waveform = self._load_audio_item(audio_item)
        audio_chunks, audio_feature_lengths = split_audio_for_whisper(
            waveform,
            sample_rate=self.audio_sampling_rate,
        )
        media_inputs = self.audio_feature_extractor(
            audio_chunks,
            sampling_rate=self.audio_sampling_rate,
            padding="max_length",
            truncation=True,
            max_length=int(getattr(self.audio_feature_extractor, "n_samples", 480000)),
            return_tensors="pt",
        )
        # Whisper is currently frozen in FP32. Preserve the feature extractor's
        # FP32 log-mel values instead of quantizing them to the LLM BF16 dtype.
        # The model still performs a defensive dtype/device match immediately
        # before every encoder call.
        media_inputs["input_features"] = media_inputs["input_features"].float()
        encoded["audio_input_features"] = media_inputs["input_features"]
        encoded["audio_segment_feature_lengths"] = torch.tensor(audio_feature_lengths, dtype=torch.long)
        encoded["audio_segment_mask"] = torch.ones(len(audio_feature_lengths), dtype=torch.bool)
        sampler_fields = (
            "global_position",
            "pool_name",
            "record_index",
            "pool_occurrence_index",
            "pool_epoch",
        )
        if isinstance(audio_item, dict) and any(audio_item.get(field) is not None for field in sampler_fields):
            missing = [field for field in sampler_fields if audio_item.get(field) is None]
            if missing:
                raise RuntimeError(f"Audio sampler provenance is incomplete: missing={missing}")
            pool_name = str(audio_item["pool_name"])
            if pool_name not in TRAINING_POOL_TO_ID:
                raise RuntimeError(f"Unknown training pool in audio provenance: {pool_name!r}")
            effective_duration_seconds = len(waveform) / float(self.audio_sampling_rate)
            encoded["audio_training_global_positions"] = torch.tensor(
                int(audio_item["global_position"]),
                dtype=torch.long,
            )
            encoded["audio_training_pool_ids"] = torch.tensor(TRAINING_POOL_TO_ID[pool_name], dtype=torch.long)
            encoded["audio_training_record_indices"] = torch.tensor(
                int(audio_item["record_index"]),
                dtype=torch.long,
            )
            encoded["audio_training_pool_occurrence_indices"] = torch.tensor(
                int(audio_item["pool_occurrence_index"]),
                dtype=torch.long,
            )
            encoded["audio_training_pool_epochs"] = torch.tensor(
                int(audio_item["pool_epoch"]),
                dtype=torch.long,
            )
            encoded["audio_training_effective_duration_seconds"] = torch.tensor(
                effective_duration_seconds,
                dtype=torch.float64,
            )
        return encoded

    def _data_collator_mm_data(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        audio_items = [item for item in batch if "audio_input_features" in item]
        if not audio_items:
            return {}
        if len(audio_items) != len(batch):
            raise RuntimeError("A batch mixes records with and without audio")

        feature_batches = [item["audio_input_features"] for item in audio_items]
        segment_lengths = [item["audio_segment_feature_lengths"] for item in audio_items]
        segment_masks = [item["audio_segment_mask"] for item in audio_items]
        if any(features.ndim != 3 for features in feature_batches):
            raise RuntimeError(
                "Each encoded audio_input_features item must have shape [segments, 80, frames]"
            )
        max_segments = max(int(features.shape[0]) for features in feature_batches)
        mel_bins = int(feature_batches[0].shape[1])
        max_feature_frames = max(int(features.shape[2]) for features in feature_batches)
        feature_dtype = feature_batches[0].dtype
        padded_features = torch.zeros(
            (len(batch), max_segments, mel_bins, max_feature_frames),
            dtype=feature_dtype,
        )
        padded_lengths = torch.zeros((len(batch), max_segments), dtype=torch.long)
        padded_segment_mask = torch.zeros((len(batch), max_segments), dtype=torch.bool)
        for index, (features, lengths, masks) in enumerate(zip(feature_batches, segment_lengths, segment_masks)):
            segment_count = int(features.shape[0])
            padded_features[index, :segment_count, :, : features.shape[2]] = features
            padded_lengths[index, :segment_count] = lengths
            padded_segment_mask[index, :segment_count] = masks
        collated = {
            "audio_input_features": padded_features,
            "audio_segment_feature_lengths": padded_lengths,
            "audio_segment_mask": padded_segment_mask,
        }
        training_keys = (
            "audio_training_global_positions",
            "audio_training_pool_ids",
            "audio_training_record_indices",
            "audio_training_pool_occurrence_indices",
            "audio_training_pool_epochs",
            "audio_training_effective_duration_seconds",
        )
        items_with_training_metadata = [
            all(key in item for key in training_keys)
            for item in audio_items
        ]
        if any(items_with_training_metadata) and not all(items_with_training_metadata):
            raise RuntimeError("A batch mixes audio records with and without training statistics provenance")
        if all(items_with_training_metadata):
            for key in training_keys:
                collated[key] = torch.stack([item[key] for item in audio_items], dim=0)
        return collated


class HuginnAudioLoader(ModelLoader):
    def get_config(self, model_dir: str):
        whisper_config = AutoConfig.from_pretrained(
            WHISPER_MODEL_DIR,
            trust_remote_code=True,
        )
        config = AutoConfig.from_pretrained(
            model_dir,
            trust_remote_code=True,
        )
        config.audio_encoder_name = str(WHISPER_MODEL_DIR)
        config.audio_encoder_hidden_size = int(getattr(whisper_config, "d_model", 1280))
        config.audio_dynamic_tokens = True
        config.audio_token_duration_ms = AUDIO_TOKEN_DURATION_MS
        config.audio_reference_30s_token_count = 250
        config.audio_max_token_count = 750
        config.audio_chunk_seconds = DEFAULT_AUDIO_CHUNK_SECONDS
        config.audio_max_seconds = DEFAULT_MAX_AUDIO_SECONDS
        config.audio_compressor_kernel_size = DYNAMIC_COMPRESSOR_KERNEL
        config.audio_compressor_stride = DYNAMIC_COMPRESSOR_STRIDE
        config.freeze_audio_encoder = True
        config.freeze_text_backbone = False
        print(f"[HuginnAudioSwift] config.audio_encoder_name={config.audio_encoder_name}")
        print(f"[HuginnAudioSwift] config.audio_encoder_hidden_size={config.audio_encoder_hidden_size}")
        print(f"[HuginnAudioSwift] config.audio_dynamic_tokens={config.audio_dynamic_tokens}")
        print(f"[HuginnAudioSwift] config.audio_max_token_count={config.audio_max_token_count}")
        print(f"[HuginnAudioSwift] config.audio_chunk_seconds={config.audio_chunk_seconds}")
        print(f"[HuginnAudioSwift] config.audio_max_seconds={config.audio_max_seconds}")
        return config

    def get_processor(self, model_dir: str, config):
        del model_dir, config
        processor = build_huginn_audio_processor()
        print(f"[HuginnAudioSwift] tokenizer_type={type(processor.tokenizer)}")
        print(f"[HuginnAudioSwift] feature_extractor_type={type(processor.feature_extractor)}")
        return processor

    def get_model(self, model_dir: str, config, processor, model_kwargs):
        del config, processor, model_kwargs
        model = build_huginn_audio_model(model_dir)
        print(f"[HuginnAudioSwift] model_type={type(model)}")
        return model


def register_huginn_audio_model_arch():
    multi_model_kwargs = {
        "language_model": ["transformer", "lm_head"],
        # During checkpoint runs PEFT wraps these three children, so Swift's
        # unfreeze calls must reach their guarded wrapper methods. Preserve the
        # already-passed whole-aligner registration in every ordinary run.
        "aligner": (
            list(ALIGNER_MODULES_TO_SAVE)
            if _requested(PEFT_ALIGNER_MODULES_TO_SAVE_ENV)
            else ["audio_aligner"]
        ),
        "generator": ["audio_encoder"],
    }
    try:
        multi_model_keys = MultiModelKeys(
            arch_name=MODEL_ARCH_NAME,
            **multi_model_kwargs,
        )
        print("[HuginnAudioSwift] registered model arch using MultiModelKeys(arch_name=...)")
    except TypeError as exc:
        if "arch_name" not in str(exc):
            raise
        try:
            multi_model_keys = MultiModelKeys(
                model_arch=MODEL_ARCH_NAME,
                **multi_model_kwargs,
            )
            print("[HuginnAudioSwift] registered model arch using MultiModelKeys(model_arch=...)")
        except TypeError as inner_exc:
            if "model_arch" not in str(inner_exc):
                raise
            print("[HuginnAudioSwift] MultiModelKeys lacks keyword arch field; retrying positional model arch registration")
            multi_model_keys = MultiModelKeys(
                MODEL_ARCH_NAME,
                **multi_model_kwargs,
            )
    try:
        register_model_arch(multi_model_keys)
    except ValueError as exc:
        duplicate_msg = f"The `{MODEL_ARCH_NAME}` has already been registered"
        if duplicate_msg not in str(exc):
            raise
        print(f"[HuginnAudioSwift] model arch `{MODEL_ARCH_NAME}` already registered; skip duplicate registration")


register_huginn_audio_model_arch()
patch_peft_dynamic90s_lora_dropout()
patch_peft_constructor_modules_to_save()
patch_peft_adapter_restore()
patch_swift_lora_llm_fsdp_dcp_resume()
patch_accelerate_fsdp2_save_state_audit()
patch_stage34_optimizer_step_callback()
patch_stage5_stability_callback()
patch_realdata_stability_callback()
patch_training_statistics_callback()
patch_checkpoint_resume_callback()
patch_checkpoint_rng_restore_audit()

register_model(
    ModelMeta(
        MODEL_TYPE,
        [
            ModelGroup(
                [
                    Model("huginn-audio-whisper-dynamic90s-v1", str(AUDIO_MODEL_DIR)),
                ]
            ),
        ],
        HuginnAudioLoader,
        template=TEMPLATE_TYPE,
        model_arch=MODEL_ARCH_NAME,
        architectures=["HuginnAudioForConditionalGeneration"],
        is_multimodal=True,
        requires=["transformers>=4.53.3"],
        tags=["huginn", "audio"],
    ),
    exist_ok=True,
)

register_template(
    TemplateMeta(
        template_type=TEMPLATE_TYPE,
        template_cls=HuginnAudioTemplate,
        prefix=[],
        system_prefix=["<|begin_header|>system<|end_header|>\n\n{{SYSTEM}}<|end_turn|>"],
        prompt=[
            "<|begin_header|>user<|end_header|>\n\n{{QUERY}}<|end_turn|>"
            "<|begin_header|>Huginn<|end_header|>\n\n"
        ],
        chat_sep=None,
        auto_add_bos=True,
        default_system=DEFAULT_SYSTEM_PROMPT,
        stop_words=[["eos_token_id"]],
    ),
    exist_ok=True,
)
