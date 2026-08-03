#!/usr/bin/env python3
"""Huginn Whisper audio-encoder adapter for X-ARES.

The wrapper restores only the Whisper encoder and audio aligner tensors from a
full-model FSDP DCP checkpoint.  It deliberately does not execute Huginn's
recurrent backbone or LoRA modules.  The X-ARES representation is the
projected dynamic audio-token sequence, excluding audio BOS/EOS boundaries.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from torch import nn


DEFAULT_CHECKPOINT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/"
    "huginn_whisper_dynamic30s_multiplier_single_epoch_fsdp4/run-20260731_084946/"
    "swift_output/v0-20260731-085036/checkpoint-20000"
)
DEFAULT_PLUGIN_PATH = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/"
    "code/huginn_lora/plugins/huginn_audio_whisper_dynamic90s_swift.py"
)
DEFAULT_MODEL_DIR = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/"
    "models/huginn-audio-whisper-dynamic90s-v1"
)
FSDP_MODEL_DIR_NAME = "pytorch_model_fsdp_0"
MAX_AUDIO_SECONDS = 30.0
SAMPLE_RATE = 16000
COMPRESSOR_KERNEL = 12
COMPRESSOR_STRIDE = 12
OUTPUT_DIM = 5280
TOKEN_DURATION_MS = 240


def _load_module(path: Path, name: str) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(f"Required Python module does not exist: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate_keys(key: str) -> set[str]:
    aliases = {key}
    changed = True
    while changed:
        changed = False
        for alias in list(aliases):
            for prefix in (
                "base_model.model.",
                "base_model.",
                "model.",
                "module.",
                "_fsdp_wrapped_module.",
            ):
                if alias.startswith(prefix):
                    stripped = alias[len(prefix):]
                    if stripped not in aliases:
                        aliases.add(stripped)
                        changed = True
    normalized = set(aliases)
    for alias in list(aliases):
        normalized.add(alias.replace(".modules_to_save.default.", "."))
        normalized.add(alias.replace(".original_module.", "."))
        normalized.add(alias.replace(".lora_A.default.", ".lora_A."))
        normalized.add(alias.replace(".lora_B.default.", ".lora_B."))
    for alias in list(normalized):
        if alias.startswith("audio_aligner."):
            normalized.add(alias[len("audio_aligner."):])
        elif alias.startswith(("temporal_compressor.", "audio_projector.", "audio_boundary_embeddings.")):
            normalized.add(f"audio_aligner.{alias}")
    return normalized


def _is_audio_target(key: str) -> bool:
    return key.startswith(("audio_encoder.", "audio_aligner."))


def _source_priority(key: str) -> tuple[int, int, str]:
    # Prefer the active PEFT modules_to_save copy for the aligner.  The
    # original_module copy is retained in the checkpoint for contract/audit
    # purposes but is not the trained active copy.
    if ".modules_to_save.default." in key:
        role = 0
    elif ".original_module." in key:
        role = 2
    else:
        role = 1
    return role, len(key), key


def _select_audio_restore_plan(model: nn.Module, state_metadata: dict[Any, Any]) -> list[tuple[str, str, tuple[int, ...], torch.dtype]]:
    target_state = model.state_dict()
    target_aliases: dict[str, list[str]] = {}
    for target_key in target_state:
        if not _is_audio_target(target_key):
            continue
        for alias in _candidate_keys(target_key):
            target_aliases.setdefault(alias, []).append(target_key)

    selected: dict[str, tuple[str, tuple[int, ...], torch.dtype]] = {}
    for raw_source_key, metadata in state_metadata.items():
        source_key = str(raw_source_key)
        shape_value = getattr(metadata, "size", None)
        properties = getattr(metadata, "properties", None)
        source_dtype = getattr(properties, "dtype", None)
        if shape_value is None or not isinstance(source_dtype, torch.dtype):
            continue
        shape = tuple(int(value) for value in shape_value)
        matches: list[str] = []
        for alias in _candidate_keys(source_key):
            for target_key in target_aliases.get(alias, []):
                if tuple(target_state[target_key].shape) == shape:
                    matches.append(target_key)
        for target_key in sorted(set(matches)):
            previous = selected.get(target_key)
            candidate = (source_key, shape, source_dtype)
            if previous is None or _source_priority(source_key) < _source_priority(previous[0]):
                selected[target_key] = candidate

    relevant_target_keys = [
        key for key, tensor in target_state.items()
        if _is_audio_target(key) and torch.is_tensor(tensor)
    ]
    missing = [key for key in relevant_target_keys if key not in selected]
    if missing:
        raise RuntimeError(
            "Checkpoint does not provide a complete Whisper+aligner restore: "
            f"missing_count={len(missing)} preview={missing[:20]}"
        )
    return [
        (source_key, target_key, shape, source_dtype)
        for target_key, (source_key, shape, source_dtype) in sorted(selected.items())
    ]


def _restore_audio_and_aligner(model: nn.Module, checkpoint: Path) -> dict[str, Any]:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader

    model_dir = checkpoint / FSDP_MODEL_DIR_NAME
    if not model_dir.is_dir():
        raise FileNotFoundError(f"FSDP model DCP directory is missing: {model_dir}")
    metadata = FileSystemReader(str(model_dir)).read_metadata()
    state_metadata = getattr(metadata, "state_dict_metadata", {})
    if not state_metadata:
        raise RuntimeError(f"FSDP model DCP metadata is empty: {model_dir}")
    plan = _select_audio_restore_plan(model, state_metadata)
    target_state = model.state_dict()
    restored = []
    for index, (source_key, target_key, shape, source_dtype) in enumerate(plan, start=1):
        streamed = torch.empty(shape, dtype=source_dtype, device="cpu")
        dcp.load({source_key: streamed}, checkpoint_id=str(model_dir))
        with torch.no_grad():
            target_state[target_key].copy_(streamed.to(dtype=target_state[target_key].dtype))
        restored.append({"source_key": source_key, "target_key": target_key, "shape": list(shape)})
        del streamed
        if index == 1 or index % 25 == 0 or index == len(plan):
            print(f"[xares-restore] audio_aligner_tensors={index}/{len(plan)}", flush=True)
    return {
        "checkpoint": str(checkpoint),
        "model_dcp": str(model_dir),
        "dcp_metadata_tensor_count": len(state_metadata),
        "restored_tensor_count": len(restored),
        "restored_tensor_preview": restored[:20],
    }


class HuginnWhisperXaresEncoder(nn.Module):
    """Post-aligner frame encoder consumed by X-ARES."""

    output_dim = OUTPUT_DIM
    sampling_rate = SAMPLE_RATE
    hop_size_in_ms = TOKEN_DURATION_MS

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        plugin_path: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.checkpoint_path = Path(
            checkpoint or os.environ.get("HUGINN_XARES_CHECKPOINT", str(DEFAULT_CHECKPOINT))
        ).expanduser().resolve()
        self.plugin_path = Path(
            plugin_path or os.environ.get("HUGINN_XARES_PLUGIN_PATH", str(DEFAULT_PLUGIN_PATH))
        ).expanduser().resolve()
        requested_device = device or os.environ.get("HUGINN_XARES_DEVICE", "cuda:0")
        self.device = torch.device(requested_device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested CUDA device but CUDA is unavailable: {self.device}")
        self.plugin = _load_module(self.plugin_path, "huginn_xares_dynamic30s_plugin")
        self.model = self.plugin.build_huginn_audio_model(str(DEFAULT_MODEL_DIR))
        self.processor = self.plugin.build_huginn_audio_processor()
        self.restore_report = _restore_audio_and_aligner(self.model, self.checkpoint_path)
        self.model.requires_grad_(False)
        self.model.eval()
        if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
            self.compute_dtype = torch.bfloat16
        elif self.device.type == "cuda":
            self.compute_dtype = torch.float16
        else:
            self.compute_dtype = torch.float32
        self.model.to(device=self.device, dtype=self.compute_dtype)
        self.model.eval()
        print(
            f"[xares-wrapper] checkpoint={self.checkpoint_path} device={self.device} "
            f"dtype={self.compute_dtype} output_dim={self.output_dim} hop_ms={self.hop_size_in_ms}",
            flush=True,
        )

    @staticmethod
    def _expected_token_count(feature_length: int) -> int:
        encoder_length = feature_length // 2
        if encoder_length < COMPRESSOR_KERNEL:
            return 0
        return (encoder_length - COMPRESSOR_KERNEL) // COMPRESSOR_STRIDE + 1

    def _encode_one(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform_np = waveform.detach().to(dtype=torch.float32, device="cpu").numpy()
        chunks, feature_lengths = self.plugin.split_audio_for_whisper(
            waveform_np,
            sample_rate=self.sampling_rate,
        )
        if len(chunks) != 1 or len(feature_lengths) != 1:
            raise RuntimeError(
                f"Expected exactly one dynamic30s chunk: chunks={len(chunks)} lengths={feature_lengths}"
            )
        feature_extractor = self.processor.feature_extractor
        inputs = feature_extractor(
            chunks,
            sampling_rate=self.sampling_rate,
            padding="max_length",
            truncation=True,
            max_length=int(getattr(feature_extractor, "n_samples", 480000)),
            return_tensors="pt",
        )
        features = inputs["input_features"].to(device=self.device, dtype=self.compute_dtype)
        feature_length = int(feature_lengths[0])
        feature_mask = (
            torch.arange(features.shape[-1], device=self.device).unsqueeze(0) < feature_length
        ).to(dtype=torch.long)
        with torch.inference_mode():
            encoded = self.model.audio_encoder(
                input_features=features,
                attention_mask=feature_mask,
                return_dict=True,
            ).last_hidden_state
            encoded = encoded.to(device=self.device, dtype=self.compute_dtype)
            projected = self.model.audio_projector(self.model.temporal_compressor(encoded))
        token_count = min(
            projected.shape[1],
            self._expected_token_count(feature_length),
        )
        if token_count <= 0:
            raise RuntimeError(
                f"Audio produced no X-ARES frame tokens: samples={waveform.numel()} feature_length={feature_length}"
            )
        return projected[0, :token_count].float().cpu()

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(audio):
            audio = torch.as_tensor(audio, dtype=torch.float32)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        if audio.ndim != 2:
            raise ValueError(f"X-ARES audio must have shape [B,T] or [T], got {tuple(audio.shape)}")
        outputs = [self._encode_one(row) for row in audio]
        lengths = {int(output.shape[0]) for output in outputs}
        if len(lengths) != 1:
            raise ValueError(
                "The X-ARES wrapper requires equal-length inputs inside one call; "
                "use batch_size_encode=1 for variable-length VoxCeleb1 audio."
            )
        return torch.stack(outputs, dim=0)


__all__ = ["HuginnWhisperXaresEncoder"]
