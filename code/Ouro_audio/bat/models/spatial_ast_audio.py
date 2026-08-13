"""BAT Spatial-AST audio encoder and Q-Former adapters.

This module contains the reusable, production-side version of the chain that
was previously validated by ``audit_bat_spatial_ast_audio.py``:

    AudioSet + binaural RIR -> [B, 2, 320000]
        -> frozen Spatial-AST token sequence [B, 515, 768]
        -> trainable BAT Q-Former output [B, 64, 2048]

The official Spatial-AST repository exposes task heads from ``forward``.  BAT
uses the final transformer sequence instead, so the adapter intentionally
reuses the official preprocessing/modules and calls ``forward_features_mask``.
No official source checkout or checkpoint is modified.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SAMPLE_RATE = 32_000
TARGET_SAMPLES = 10 * SAMPLE_RATE
SPATIAL_AST_TOKEN_COUNT = 515
SPATIAL_AST_HIDDEN_SIZE = 768
BAT_QUERY_COUNT = 64
OURO_HIDDEN_SIZE = 2048


def _present(value: Any) -> bool:
    return value is not None and str(value).strip().lower() not in {"", "none", "null"}


def _import_source_module(name: str, path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import source file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_spatial_ast_module(source_root: Path):
    source_root = source_root.expanduser().resolve()
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    return _import_source_module("bat_runtime_spatial_ast", source_root / "spatial_ast.py")


def _load_spatial_ast_checkpoint(model: nn.Module, checkpoint_path: Path) -> dict[str, Any]:
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected Spatial-AST checkpoint root: {type(payload).__name__}")
    state = payload.get("model", payload.get("state_dict"))
    if not isinstance(state, dict):
        raise TypeError(f"Spatial-AST checkpoint has no model/state_dict mapping: {checkpoint_path}")

    candidates: list[tuple[str, dict[str, Any]]] = [("exact", state)]
    for prefix in ("module.", "model.", "encoder."):
        if state and all(str(key).startswith(prefix) for key in state):
            candidates.append((f"strip:{prefix}", {str(key)[len(prefix):]: value for key, value in state.items()}))

    attempts: list[dict[str, Any]] = []
    for candidate_name, candidate in candidates:
        try:
            result = model.load_state_dict(candidate, strict=True)
            attempts.append({
                "candidate": candidate_name,
                "status": "ok",
                "missing": list(result.missing_keys),
                "unexpected": list(result.unexpected_keys),
            })
            return {
                "path": str(checkpoint_path),
                "strict_loaded": True,
                "selected_candidate": candidate_name,
                "state_dict_key_count": len(state),
                "attempts": attempts,
            }
        except RuntimeError as exc:
            missing, unexpected = model.load_state_dict(candidate, strict=False)
            attempts.append({
                "candidate": candidate_name,
                "status": "failed",
                "error": repr(exc),
                "missing_count": len(missing),
                "unexpected_count": len(unexpected),
            })
    raise RuntimeError(f"Spatial-AST strict checkpoint load failed: {attempts}")


def _token_forward_from_binaural(model: nn.Module, waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim != 3 or waveform.shape[1] != 2:
        raise ValueError(f"Expected binaural waveform [B,2,T], got {tuple(waveform.shape)}")
    batch, channels, samples = waveform.shape
    flattened = waveform.reshape(batch * channels, samples)
    real, imag = model.spectrogram_extractor(flattened)
    log_mel = model.logmel_extractor(torch.sqrt(real**2 + imag**2)).reshape(batch, channels, -1, 128)
    log_mel = model.bn(log_mel)
    ipd = torch.atan2(imag[1::2], real[1::2]) - torch.atan2(imag[::2], real[::2])
    spatial = torch.matmul(
        torch.cat([torch.cos(ipd), torch.sin(ipd)], dim=1),
        model.logmel_extractor.melW,
    )
    features = torch.cat([log_mel, spatial], dim=1)
    if features.shape[2] < model.target_frame:
        features = F.interpolate(
            features,
            (model.target_frame, features.shape[3]),
            mode="bicubic",
            align_corners=True,
        )
    patch_tokens = model.patch_embed(model.conv_downsample(features))
    return model.forward_features_mask(patch_tokens, mask_t_prob=0.0, mask_f_prob=0.0)


class SpatialASTAudioEncoder(nn.Module):
    """Frozen official Spatial-AST sequence encoder."""

    def __init__(self, source_root: str | Path, checkpoint_path: str | Path):
        super().__init__()
        source_root = Path(source_root).expanduser().resolve()
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        module = _load_spatial_ast_module(source_root)
        model = module.build_AST(num_classes=355, num_cls_tokens=3)
        self.load_report = _load_spatial_ast_checkpoint(model, checkpoint_path)
        model.eval().requires_grad_(False)
        self.encoder = model
        self.source_root = str(source_root)
        self.checkpoint_path = str(checkpoint_path)

        if sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad) != 0:
            raise RuntimeError("Spatial-AST encoder must be fully frozen")

    def train(self, mode: bool = True):
        # BatchNorm/dropout behavior must remain inference mode even when the
        # enclosing Ouro wrapper enters training mode.
        super().train(False)
        self.encoder.eval()
        return self

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3 or waveform.shape[1] != 2 or waveform.shape[2] != TARGET_SAMPLES:
            raise ValueError(
                "Spatial-AST expects fixed BAT binaural waveforms [B,2,320000], "
                f"got {tuple(waveform.shape)}"
            )
        # The encoder is deliberately outside the autograd graph. Gradients
        # must still flow through the following Q-Former because its input can
        # be a non-requires-grad tensor.
        with torch.no_grad():
            tokens = _token_forward_from_binaural(self.encoder, waveform.float())
        if tuple(tokens.shape[1:]) != (SPATIAL_AST_TOKEN_COUNT, SPATIAL_AST_HIDDEN_SIZE):
            raise RuntimeError(f"Unexpected Spatial-AST token shape: {tuple(tokens.shape)}")
        if not torch.isfinite(tokens).all():
            raise RuntimeError("Spatial-AST produced non-finite tokens")
        return tokens


class BATQFormer(nn.Module):
    """Official BAT/SLAM-LLM Q-Former projector, trainable by default."""

    def __init__(
        self,
        source_path: str | Path,
        encoder_dim: int = SPATIAL_AST_HIDDEN_SIZE,
        llm_dim: int = OURO_HIDDEN_SIZE,
        layers: int = 8,
        query_len: int = BAT_QUERY_COUNT,
    ):
        super().__init__()
        source_path = Path(source_path).expanduser().resolve()
        module = _import_source_module("bat_runtime_qformer", source_path)

        class Config:
            def __init__(self):
                self.encoder_dim = encoder_dim
                self.llm_dim = llm_dim
                self.qformer_layers = layers
                self.query_len = query_len

            def get(self, key: str, default: Any = None):
                return getattr(self, key, default)

        self.projector = module.EncoderProjectorQFormer(Config())
        self.source_path = str(source_path)
        self.encoder_dim = int(encoder_dim)
        self.llm_dim = int(llm_dim)
        self.query_len = int(query_len)
        self.layers = int(layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.encoder_dim:
            raise ValueError(f"Expected Spatial-AST tokens [B,S,{self.encoder_dim}], got {tuple(tokens.shape)}")
        attention = torch.ones(tokens.shape[:2], dtype=torch.long, device=tokens.device)
        projected = self.projector(tokens, attention)
        expected = (tokens.shape[0], self.query_len, self.llm_dim)
        if tuple(projected.shape) != expected:
            raise RuntimeError(f"Unexpected Q-Former output shape: got={tuple(projected.shape)} expected={expected}")
        if not torch.isfinite(projected).all():
            raise RuntimeError("Q-Former produced non-finite audio embeddings")
        return projected


class BATAudioRenderer:
    """Official BAT waveform rendering contract for real dataset records."""

    def __init__(self, audio_root: str | Path, reverb_root: str | Path):
        self.audio_root = Path(audio_root).expanduser().resolve()
        self.reverb_root = Path(reverb_root).expanduser().resolve()

    @staticmethod
    def _resolve_audio(root: Path, audio_id: str) -> Path:
        relative = str(audio_id).replace("\\", "/").lstrip("./")
        candidate = root / relative
        candidates = [candidate] if candidate.suffix else [candidate.with_suffix(suffix) for suffix in (".wav", ".flac", ".mp3", ".ogg")]
        for path in candidates:
            if path.is_file():
                return path
        raise FileNotFoundError(f"AudioSet reference not found: root={root} audio_id={audio_id}")

    @staticmethod
    def _resolve_reverb(root: Path, reverb_id: str) -> Path:
        relative = str(reverb_id).replace("\\", "/").lstrip("./")
        for path in (root / "binaural" / relative, root / relative):
            if path.is_file():
                return path
        raise FileNotFoundError(f"Binaural RIR reference not found: root={root} reverb_id={reverb_id}")

    @staticmethod
    def _load_source(path: Path) -> tuple[np.ndarray, int]:
        import soundfile as sf

        value, sample_rate = sf.read(str(path), always_2d=False, dtype="float32")
        value = np.asarray(value, dtype=np.float32)
        if value.ndim == 2:
            value = value[:, 0]
        if value.ndim != 1:
            raise ValueError(f"Expected mono/first-channel AudioSet waveform, got {value.shape}: {path}")
        return value, int(sample_rate)

    @staticmethod
    def _normalize(value: np.ndarray) -> np.ndarray:
        rms = float(np.sqrt(np.mean(value.astype(np.float64) ** 2))) if value.size else 0.0
        if rms == 0.0:
            return value.astype(np.float32, copy=False)
        return (value * (10.0 ** ((-14.0 - 20.0 * math.log10(rms)) / 20.0))).astype(np.float32)

    @staticmethod
    def _resample(value: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate == SAMPLE_RATE:
            return value.astype(np.float32, copy=False)
        from scipy import signal

        divisor = math.gcd(int(sample_rate), SAMPLE_RATE)
        return signal.resample_poly(value, SAMPLE_RATE // divisor, sample_rate // divisor).astype(np.float32)

    @staticmethod
    def _crop_or_pad(value: np.ndarray) -> np.ndarray:
        output = np.zeros((2, TARGET_SAMPLES), dtype=np.float32)
        output[:, : min(TARGET_SAMPLES, value.shape[-1])] = value[:, :TARGET_SAMPLES]
        return output

    def _render_one(self, audio_id: str, reverb_id: str) -> torch.Tensor:
        from scipy import signal

        audio, sample_rate = self._load_source(self._resolve_audio(self.audio_root, audio_id))
        audio = self._normalize(self._resample(audio, sample_rate))[None, :]
        rir = np.asarray(np.load(self._resolve_reverb(self.reverb_root, reverb_id), allow_pickle=False), dtype=np.float32)
        if rir.ndim == 1:
            rir = rir[None, :]
        if rir.ndim != 2 or rir.shape[0] != 2:
            raise ValueError(f"Expected binaural RIR [2,L], got {rir.shape}")
        rendered = signal.fftconvolve(audio, rir, mode="full")
        return torch.from_numpy(self._crop_or_pad(rendered)).float()

    def render_record(self, record: dict[str, Any]) -> torch.Tensor:
        first = self._render_one(str(record["audio_id"]), str(record["reverb_id"]))
        if _present(record.get("audio_id2")) and _present(record.get("reverb_id2")):
            second = self._render_one(str(record["audio_id2"]), str(record["reverb_id2"]))
            first = (first + second) / 2.0
        return first

    def load_item(self, item: Any) -> torch.Tensor:
        """Load either a rendered tensor/NPY or a BAT QA record dictionary."""
        if torch.is_tensor(item):
            waveform = item.detach().float().cpu()
        elif isinstance(item, dict) and torch.is_tensor(item.get("waveform")):
            waveform = item["waveform"].detach().float().cpu()
        elif isinstance(item, dict) and item.get("waveform_path"):
            waveform = torch.from_numpy(np.asarray(np.load(item["waveform_path"], allow_pickle=False))).float()
        elif isinstance(item, dict) and _present(item.get("audio_id")) and _present(item.get("reverb_id")):
            waveform = self.render_record(item)
        else:
            raise TypeError(
                "BAT audio item must be a [2,320000] tensor, waveform_path, "
                "or a record containing audio_id/reverb_id"
            )
        if waveform.ndim != 2 or waveform.shape[0] != 2:
            raise ValueError(f"Expected rendered binaural waveform [2,T], got {tuple(waveform.shape)}")
        return torch.from_numpy(self._crop_or_pad(waveform.numpy())).float()
