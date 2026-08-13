"""Read-only GPU audit for the BAT Spatial-AST audio contract.

The audit validates the complete pre-LLM path without training anything:

    AudioSet mono waveform + binaural RIR
        -> BAT/SLAM-LLM waveform loader
        -> 2-channel, 32 kHz, 10-second waveform
        -> Spatial-AST token sequence [B, 515, 768]
        -> BAT Q-Former [B, 64, 2048]

The official Spatial-AST repository exposes a task-head ``forward`` that
accepts waveform and RIR separately and returns four prediction heads.  BAT's
LLM path consumes the transformer sequence instead.  We therefore test both:

* the official forward, with a forward hook capturing its 515 hidden tokens;
* a token-level adapter that reuses the official preprocessing/modules after
  the BAT loader has already rendered one or two sources into binaural audio.

No checkpoint, public AudioSet file, or source repository is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SAMPLE_RATE = 32_000
TARGET_SAMPLES = 10 * SAMPLE_RATE
EXPECTED_TOKEN_COUNT = 515
EXPECTED_HIDDEN_SIZE = 768


def install_spatial_ast_compat(source_root: Path) -> dict[str, Any]:
    """Install only the legacy helper symbols Spatial-AST imports.

    The official repository pins timm==0.3.2, but the shared swift_ouro
    environment intentionally uses a newer Torch stack and does not have timm
    installed.  Spatial-AST's source uses only ``to_2tuple`` and
    ``trunc_normal_`` from timm in its top-level module; its local
    ``utils/vision_transformer.py`` additionally imports ``DropPath``.
    Keeping these tiny compatibility definitions local avoids installing an
    obsolete timm package into the Ouro environment.
    """
    import types

    timm_module = types.ModuleType("timm")
    models_module = types.ModuleType("timm.models")
    layers_module = types.ModuleType("timm.models.layers")

    def to_2tuple(value: Any) -> tuple[Any, Any]:
        if isinstance(value, (tuple, list)):
            if len(value) != 2:
                raise ValueError(f"Expected a 2-tuple value, got {value!r}")
            return tuple(value)
        return (value, value)

    def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0) -> torch.Tensor:
        # Use the current Torch implementation, which is the numerical
        # equivalent needed by the old timm call sites.
        return torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)

    class DropPath(nn.Module):
        def __init__(self, drop_prob: float = 0.0) -> None:
            super().__init__()
            self.drop_prob = float(drop_prob)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            if self.drop_prob == 0.0 or not self.training:
                return value
            keep_prob = 1.0 - self.drop_prob
            shape = (value.shape[0],) + (1,) * (value.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=value.dtype, device=value.device)
            return value.div(keep_prob) * random_tensor.floor()

    layers_module.to_2tuple = to_2tuple
    layers_module.trunc_normal_ = trunc_normal_
    layers_module.DropPath = DropPath
    models_module.layers = layers_module
    timm_module.models = models_module
    sys.modules.setdefault("timm", timm_module)
    sys.modules.setdefault("timm.models", models_module)
    sys.modules.setdefault("timm.models.layers", layers_module)
    return {
        "mode": "local_legacy_timm_compat",
        "source_root": str(source_root),
        "symbols": ["to_2tuple", "trunc_normal_", "DropPath"],
        "official_timm_requirement": "0.3.2",
        "installed_timm_package": False,
    }


def install_librosa_compat(source_root: Path) -> dict[str, Any]:
    """Provide the librosa filter-bank API needed by official ``utils/stft.py``.

    The official Spatial-AST STFT helper imports ``librosa`` only to build
    ``librosa.filters.mel``.  Installing an old librosa into the shared
    Torch-2.11 environment is unnecessary and can introduce NumPy/numba
    compatibility problems, so use a small NumPy implementation of the
    librosa Slaney mel filter bank when librosa is unavailable.
    """
    try:
        import librosa  # noqa: F401

        return {
            "mode": "installed_librosa",
            "source_root": str(source_root),
            "installed_librosa_package": True,
        }
    except Exception as exc:  # noqa: BLE001 - fallback is deliberate
        import types

        def hz_to_mel(frequencies: np.ndarray) -> np.ndarray:
            frequencies = np.asarray(frequencies, dtype=np.float64)
            f_sp = 200.0 / 3.0
            mels = (frequencies - 0.0) / f_sp
            min_log_hz = 1000.0
            min_log_mel = (min_log_hz - 0.0) / f_sp
            logstep = math.log(6.4) / 27.0
            log_t = frequencies >= min_log_hz
            mels = mels.copy()
            mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
            return mels

        def mel_to_hz(mels: np.ndarray) -> np.ndarray:
            mels = np.asarray(mels, dtype=np.float64)
            f_sp = 200.0 / 3.0
            freqs = 0.0 + f_sp * mels
            min_log_hz = 1000.0
            min_log_mel = (min_log_hz - 0.0) / f_sp
            logstep = math.log(6.4) / 27.0
            log_t = mels >= min_log_mel
            freqs = freqs.copy()
            freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
            return freqs

        def mel_filter_bank(
            *,
            sr: int,
            n_fft: int,
            n_mels: int = 128,
            fmin: float = 0.0,
            fmax: float | None = None,
            htk: bool = False,
            norm: str | float | None = "slaney",
            dtype: Any = np.float32,
        ) -> np.ndarray:
            if htk:
                hz_to_mel_fn = lambda value: 2595.0 * np.log10(1.0 + value / 700.0)
                mel_to_hz_fn = lambda value: 700.0 * (10.0 ** (value / 2595.0) - 1.0)
            else:
                hz_to_mel_fn = hz_to_mel
                mel_to_hz_fn = mel_to_hz
            if fmax is None:
                fmax = float(sr) / 2.0
            if not 0.0 <= fmin <= fmax <= float(sr) / 2.0:
                raise ValueError(f"Invalid mel frequency range: fmin={fmin}, fmax={fmax}, sr={sr}")
            fft_freqs = np.linspace(0.0, float(sr) / 2.0, int(1 + n_fft // 2))
            mel_frequencies = mel_to_hz_fn(
                np.linspace(hz_to_mel_fn(np.asarray([fmin]))[0], hz_to_mel_fn(np.asarray([fmax]))[0], n_mels + 2)
            )
            fdiff = np.diff(mel_frequencies)
            ramps = np.subtract.outer(mel_frequencies, fft_freqs)
            weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=np.float64)
            for index in range(n_mels):
                lower = -ramps[index] / fdiff[index]
                upper = ramps[index + 2] / fdiff[index + 1]
                weights[index] = np.maximum(0.0, np.minimum(lower, upper))
            if norm == "slaney":
                enorm = 2.0 / (mel_frequencies[2 : n_mels + 2] - mel_frequencies[:n_mels])
                weights *= enorm[:, None]
            elif norm not in (None, "slaney"):
                raise ValueError(f"Fallback mel filter only supports norm=None/'slaney', got {norm!r}")
            return weights.astype(dtype, copy=False)

        librosa_module = types.ModuleType("librosa")
        filters_module = types.ModuleType("librosa.filters")
        filters_module.mel = mel_filter_bank
        librosa_module.filters = filters_module
        sys.modules.setdefault("librosa", librosa_module)
        sys.modules.setdefault("librosa.filters", filters_module)
        return {
            "mode": "local_librosa_mel_compat",
            "source_root": str(source_root),
            "installed_librosa_package": False,
            "import_error": repr(exc),
            "api": "librosa.filters.mel",
            "mel_contract": {"sr": 32000, "n_fft": 1024, "n_mels": 128, "fmin": 50, "fmax": 14000, "norm": "slaney", "htk": False},
        }


def private_output(path: Path) -> None:
    text = str(path).replace("\\", "/")
    if text.startswith("/hpc_stor03/public"):
        raise ValueError(f"Refusing to write report under public storage: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_stats(value: torch.Tensor) -> dict[str, Any]:
    detached = value.detach()
    numeric = detached.float()
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": int(detached.numel()),
        "finite": bool(torch.isfinite(numeric).all().item()),
        "min": float(numeric.min().item()),
        "max": float(numeric.max().item()),
        "mean": float(numeric.mean().item()),
        "std": float(numeric.std(unbiased=False).item()),
    }


def model_parameter_report(module: nn.Module) -> dict[str, Any]:
    named = list(module.named_parameters())
    trainable = [(name, parameter) for name, parameter in named if parameter.requires_grad]
    frozen = [(name, parameter) for name, parameter in named if not parameter.requires_grad]
    return {
        "all": sum(parameter.numel() for _, parameter in named),
        "trainable": sum(parameter.numel() for _, parameter in trainable),
        "frozen": sum(parameter.numel() for _, parameter in frozen),
        "trainable_name_count": len(trainable),
        "frozen_name_count": len(frozen),
        "trainable_name_preview": [name for name, _ in trainable[:20]],
        "dtype_set": sorted({str(parameter.dtype) for _, parameter in named}),
    }


def load_json_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise TypeError(f"Expected a list of object records: {path}")
    return records


def present(value: Any) -> bool:
    return value is not None and str(value).strip().lower() not in {"", "none", "null"}


def resolve_audio(root: Path, audio_id: str) -> Path | None:
    relative = str(audio_id).replace("\\", "/").lstrip("./")
    candidate = root / relative
    candidates = [candidate] if candidate.suffix else [candidate.with_suffix(suffix) for suffix in (".wav", ".flac", ".mp3", ".ogg")]
    return next((path for path in candidates if path.is_file()), None)


def resolve_reverb(root: Path, reverb_id: str) -> Path | None:
    relative = str(reverb_id).replace("\\", "/").lstrip("./")
    candidates = [root / "binaural" / relative, root / relative]
    return next((path for path in candidates if path.is_file()), None)


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    waveform, sample_rate = sf.read(str(path), always_2d=False, dtype="float32")
    array = np.asarray(waveform, dtype=np.float32)
    if array.ndim == 2:
        # Match the official BAT loader: use the first channel of an AudioSet
        # file before rendering it through the spatial RIR.
        array = array[:, 0]
    if array.ndim != 1:
        raise ValueError(f"Expected mono/first-channel waveform, got {array.shape} from {path}")
    return array, int(sample_rate)


def normalize_audio(array: np.ndarray, target_dbfs: float = -14.0) -> np.ndarray:
    rms = float(np.sqrt(np.mean(array.astype(np.float64) ** 2))) if array.size else 0.0
    if rms == 0.0:
        return array.astype(np.float32, copy=False)
    current_dbfs = 20.0 * math.log10(rms)
    gain = 10.0 ** ((target_dbfs - current_dbfs) / 20.0)
    return (array * gain).astype(np.float32, copy=False)


def resample_audio(array: np.ndarray, source_rate: int) -> np.ndarray:
    if source_rate == SAMPLE_RATE:
        return array.astype(np.float32, copy=False)
    from scipy import signal

    divisor = math.gcd(int(source_rate), SAMPLE_RATE)
    return signal.resample_poly(array, SAMPLE_RATE // divisor, source_rate // divisor).astype(np.float32)


def load_rir(path: Path) -> np.ndarray:
    array = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[0] != 2:
        raise ValueError(f"Expected binaural RIR [2,L], got {array.shape} from {path}")
    if not np.isfinite(array).all():
        raise ValueError(f"RIR contains non-finite values: {path}")
    return array


def crop_or_pad(waveform: np.ndarray, target_samples: int = TARGET_SAMPLES) -> np.ndarray:
    if waveform.shape[-1] < target_samples:
        padded = np.zeros((waveform.shape[0], target_samples), dtype=np.float32)
        padded[:, : waveform.shape[-1]] = waveform
        return padded
    return waveform[:, :target_samples].astype(np.float32, copy=False)


def convolve_source(audio: np.ndarray, sample_rate: int, rir: np.ndarray) -> np.ndarray:
    from scipy import signal

    audio = resample_audio(audio, sample_rate)
    audio = normalize_audio(audio, -14.0)
    source = audio[None, :]
    rendered = signal.fftconvolve(source, rir, mode="full")
    if rendered.ndim != 2 or rendered.shape[0] != 2:
        raise ValueError(f"Unexpected rendered shape {rendered.shape}")
    return crop_or_pad(rendered)


def render_record(record: dict[str, Any], audio_root: Path, reverb_root: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    audio_id = str(record["audio_id"])
    reverb_id = str(record["reverb_id"])
    audio_path = resolve_audio(audio_root, audio_id)
    reverb_path = resolve_reverb(reverb_root, reverb_id)
    if audio_path is None or reverb_path is None:
        raise FileNotFoundError(f"Cannot resolve primary assets audio={audio_id!r} rir={reverb_id!r}")
    audio, sample_rate = load_audio(audio_path)
    rir = load_rir(reverb_path)
    rendered = convolve_source(audio, sample_rate, rir)
    source_info: dict[str, Any] = {
        "audio_id": audio_id,
        "audio_path": str(audio_path),
        "audio_sha256": sha256_file(audio_path),
        "audio_sample_rate_original": sample_rate,
        "audio_samples_original": int(audio.shape[-1]),
        "reverb_id": reverb_id,
        "reverb_path": str(reverb_path),
        "reverb_sha256": sha256_file(reverb_path),
        "reverb_shape": list(rir.shape),
    }
    if present(record.get("audio_id2")) and present(record.get("reverb_id2")):
        audio_id2 = str(record["audio_id2"])
        reverb_id2 = str(record["reverb_id2"])
        audio_path2 = resolve_audio(audio_root, audio_id2)
        reverb_path2 = resolve_reverb(reverb_root, reverb_id2)
        if audio_path2 is None or reverb_path2 is None:
            raise FileNotFoundError(f"Cannot resolve second assets audio={audio_id2!r} rir={reverb_id2!r}")
        audio2, sample_rate2 = load_audio(audio_path2)
        rir2 = load_rir(reverb_path2)
        rendered2 = convolve_source(audio2, sample_rate2, rir2)
        rendered = (rendered + rendered2) / 2.0
        source_info.update(
            {
                "audio_id2": audio_id2,
                "audio_path2": str(audio_path2),
                "audio_sha256_2": sha256_file(audio_path2),
                "audio_sample_rate_original_2": sample_rate2,
                "audio_samples_original_2": int(audio2.shape[-1]),
                "reverb_id2": reverb_id2,
                "reverb_path2": str(reverb_path2),
                "reverb_sha256_2": sha256_file(reverb_path2),
                "reverb_shape_2": list(rir2.shape),
            }
        )
        source_info["source_shape"] = "dual"
    else:
        source_info["source_shape"] = "single"
    return torch.from_numpy(rendered).float(), source_info


def select_samples(qa_root: Path) -> list[tuple[str, dict[str, Any]]]:
    requests = [
        ("A", "stage1-clsdoa", "CLASSIFICATION", False),
        ("B", "stage1-clsdoa", "DOA", False),
        ("C", "stage2-single", "MIXUP_SINGLE_CLASSIFICATION", True),
        ("D", "stage2-single", "MIXUP_SINGLE_DOA", True),
        ("E", "stage3-mixup", "MIXUP_DIRECTION", True),
    ]
    selected: list[tuple[str, dict[str, Any]]] = []
    for label, stage, question_type, dual in requests:
        records = load_json_records(qa_root / stage / "train.json")
        record = next(
            (
                item
                for item in records
                if str(item.get("question_type", "")).upper() == question_type
                and (present(item.get("audio_id2")) and present(item.get("reverb_id2"))) == dual
            ),
            None,
        )
        if record is None:
            raise LookupError(f"No representative {label} record for {stage}/{question_type}, dual={dual}")
        selected.append((label, record))
    return selected


def import_module_from_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_spatial_ast(source_root: Path) -> tuple[nn.Module, dict[str, Any]]:
    source_path = source_root / "spatial_ast.py"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    compat = install_spatial_ast_compat(source_root)
    librosa_compat = install_librosa_compat(source_root)
    module = import_module_from_file("bat_official_spatial_ast", source_path)
    model = module.build_AST(num_classes=355, num_cls_tokens=3)
    return model, {
        "path": str(source_path),
        "sha256": sha256_file(source_path),
        "dependency_compat": compat,
        "librosa_compat": librosa_compat,
    }


def checkpoint_state(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unexpected checkpoint root: {type(checkpoint).__name__}")
    state = checkpoint.get("model", checkpoint.get("state_dict"))
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint has no model/state_dict mapping; keys={list(checkpoint)[:20]}")
    return checkpoint, state


def state_candidates(state: dict[str, torch.Tensor]) -> list[tuple[str, dict[str, torch.Tensor]]]:
    candidates: list[tuple[str, dict[str, torch.Tensor]]] = [("exact", state)]
    for prefix in ("module.", "model.", "encoder."):
        if state and all(str(key).startswith(prefix) for key in state):
            candidates.append((f"strip:{prefix}", {str(key)[len(prefix):]: value for key, value in state.items()}))
    return candidates


def load_spatial_checkpoint(model: nn.Module, path: Path) -> dict[str, Any]:
    checkpoint, state = checkpoint_state(path)
    attempts: list[dict[str, Any]] = []
    loaded = False
    selected = ""
    for name, candidate in state_candidates(state):
        try:
            result = model.load_state_dict(candidate, strict=True)
            attempts.append({"candidate": name, "status": "ok", "missing": list(result.missing_keys), "unexpected": list(result.unexpected_keys)})
            loaded = True
            selected = name
            break
        except RuntimeError as exc:
            missing, unexpected = model.load_state_dict(candidate, strict=False)
            attempts.append({"candidate": name, "status": "failed", "error": repr(exc), "missing_count": len(missing), "unexpected_count": len(unexpected), "missing_preview": list(missing)[:12], "unexpected_preview": list(unexpected)[:12]})
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "container_keys": [str(key) for key in checkpoint.keys()],
        "state_dict_key_count": len(state),
        "state_dict_key_preview": [str(key) for key in list(state)[:20]],
        "strict_loaded": loaded,
        "selected_candidate": selected,
        "attempts": attempts,
    }


def official_forward_with_tokens(model: nn.Module, waveform: torch.Tensor, reverb: torch.Tensor) -> tuple[Any, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"forward_features_mask returned {type(output).__name__}")
        captured["tokens"] = output

    # ``forward_features_mask`` is a Python method, not an nn.Module.  The
    # output of the final Transformer block is exactly the sequence consumed
    # by the official forward heads, so hook the final block instead.
    if not hasattr(model, "blocks") or len(model.blocks) == 0:
        raise AttributeError("Spatial-AST model has no Transformer blocks")
    handle = model.blocks[-1].register_forward_hook(hook)
    try:
        outputs = model(waveform, reverb, mask_t_prob=0.0, mask_f_prob=0.0)
    finally:
        handle.remove()
    if "tokens" not in captured:
        raise RuntimeError("Spatial-AST token hook did not capture forward_features_mask output")
    return outputs, captured["tokens"]


def token_forward_from_binaural(model: nn.Module, waveform: torch.Tensor) -> torch.Tensor:
    """Reuse official Spatial-AST modules after BAT has rendered binaural audio."""
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
        features = F.interpolate(features, (model.target_frame, features.shape[3]), mode="bicubic", align_corners=True)
    features = model.conv_downsample(features)
    patch_tokens = model.patch_embed(features)
    return model.forward_features_mask(patch_tokens, mask_t_prob=0.0, mask_f_prob=0.0)


def build_qformer(source_path: Path, encoder_dim: int = 768, llm_dim: int = 2048, layers: int = 8, query_len: int = 64) -> tuple[nn.Module, dict[str, Any]]:
    module = import_module_from_file("bat_audited_qformer", source_path)

    class Config:
        def __init__(self) -> None:
            self.encoder_dim = encoder_dim
            self.llm_dim = llm_dim
            self.qformer_layers = layers
            self.query_len = query_len

        def get(self, key: str, default: Any = None) -> Any:
            return getattr(self, key, default)

    projector = module.EncoderProjectorQFormer(Config())
    return projector, {"path": str(source_path), "sha256": sha256_file(source_path)}


def compare_tensors(first: torch.Tensor, second: torch.Tensor) -> dict[str, Any]:
    a = first.float()
    b = second.float()
    cosine = F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1)
    return {
        "max_abs_difference": float((a - b).abs().max().item()),
        "mean_abs_difference": float((a - b).abs().mean().item()),
        "cosine_similarity": float(cosine.mean().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial-ast-root", type=Path, required=True)
    parser.add_argument("--spatial-ast-checkpoint", type=Path, required=True)
    parser.add_argument("--qa-root", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--reverb-root", type=Path, required=True)
    parser.add_argument("--qformer-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    private_output(args.output)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    print("========== BAT SPATIAL-AST AUDIO LINK AUDIT ==========")
    print(f"[python] version={sys.version.split()[0]} executable={sys.executable}")
    print(f"[torch] version={torch.__version__} cuda={torch.cuda.is_available()}")
    print(f"[device] {device} name={torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'}")
    print(f"[spatial-ast] source={args.spatial_ast_root}")
    print(f"[checkpoint] {args.spatial_ast_checkpoint}")
    print(f"[qa] {args.qa_root}")
    print(f"[audio] {args.audio_root} (read-only input)")
    print(f"[reverb] {args.reverb_root}")

    issues: list[str] = []
    samples = select_samples(args.qa_root)
    model, source_contract = build_spatial_ast(args.spatial_ast_root)
    checkpoint_report = load_spatial_checkpoint(model, args.spatial_ast_checkpoint)
    if not checkpoint_report["strict_loaded"]:
        issues.append("spatial_ast_checkpoint_strict_load_failed")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    parameter_report = model_parameter_report(model)
    if parameter_report["trainable"] != 0:
        issues.append("spatial_ast_not_fully_frozen")

    qformer, qformer_contract = build_qformer(args.qformer_source)
    qformer.to(device).eval()
    for parameter in qformer.parameters():
        parameter.requires_grad = True

    sample_reports: list[dict[str, Any]] = []
    token_inputs: list[torch.Tensor] = []
    for label, record in samples:
        waveform, source_info = render_record(record, args.audio_root, args.reverb_root)
        waveform_batch = waveform.unsqueeze(0).to(device)
        report: dict[str, Any] = {
            "label": label,
            "question_id": record.get("question_id"),
            "question_type": record.get("question_type"),
            "source_shape": source_info["source_shape"],
            "source_info": source_info,
            "rendered_waveform": tensor_stats(waveform_batch),
        }
        with torch.inference_mode():
            tokens = token_forward_from_binaural(model, waveform_batch)
        report["token_forward"] = tensor_stats(tokens)
        report["token_contract"] = {
            "sequence_length_ok": list(tokens.shape[1:]) == [EXPECTED_TOKEN_COUNT, EXPECTED_HIDDEN_SIZE],
            "encoder_frozen": parameter_report["trainable"] == 0,
        }
        if list(tokens.shape[1:]) != [EXPECTED_TOKEN_COUNT, EXPECTED_HIDDEN_SIZE]:
            issues.append(f"{label}_token_shape_mismatch")
        if not bool(torch.isfinite(tokens.float()).all().item()):
            issues.append(f"{label}_token_nonfinite")
        if label == "A":
            # For a single source, compare the BAT-rendered waveform path with
            # the official two-input Spatial-AST forward path.
            raw_audio, original_sr = load_audio(Path(source_info["audio_path"]))
            raw_audio = normalize_audio(resample_audio(raw_audio, original_sr), -14.0)
            raw_audio = crop_or_pad(raw_audio[None, :])
            rir = load_rir(Path(source_info["reverb_path"]))
            raw_tensor = torch.from_numpy(raw_audio).unsqueeze(0).to(device)
            rir_tensor = torch.from_numpy(rir).unsqueeze(0).to(device)
            with torch.inference_mode():
                official_heads, official_tokens = official_forward_with_tokens(model, raw_tensor, rir_tensor)
            report["official_forward_heads"] = [tensor_stats(output) for output in official_heads]
            report["official_token_forward"] = tensor_stats(official_tokens)
            report["official_vs_loader_adapter_tokens"] = compare_tensors(official_tokens, tokens)
            if list(official_tokens.shape[1:]) != [EXPECTED_TOKEN_COUNT, EXPECTED_HIDDEN_SIZE]:
                issues.append("official_token_shape_mismatch")
        token_inputs.append(tokens)
        sample_reports.append(report)

    token_batch = torch.cat(token_inputs, dim=0)
    attention_mask = torch.ones(token_batch.shape[:2], dtype=torch.long, device=device)
    with torch.inference_mode():
        projected = qformer(token_batch, attention_mask)
    qformer_report = {
        "source": qformer_contract,
        "input": tensor_stats(token_batch),
        "attention_mask": tensor_stats(attention_mask),
        "output": tensor_stats(projected),
        "parameters": model_parameter_report(qformer),
        "output_contract": list(projected.shape[1:]) == [64, 2048],
    }
    if list(projected.shape[1:]) != [64, 2048]:
        issues.append("qformer_output_shape_mismatch")
    if not bool(torch.isfinite(projected.float()).all().item()):
        issues.append("qformer_output_nonfinite")

    peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
    report = {
        "status": "ok" if not issues else "incomplete",
        "scope": {
            "training": False,
            "spatial_ast_training": False,
            "qformer_training": False,
            "ouro_loaded": False,
            "public_audio_written": False,
        },
        "environment": {
            "python": {"version": sys.version, "executable": sys.executable},
            "torch": {"version": torch.__version__, "cuda_available": torch.cuda.is_available()},
            "device": {"requested": str(device), "name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu", "peak_allocated_bytes": int(peak_allocated), "peak_reserved_bytes": int(peak_reserved)},
        },
        "paths": {
            "spatial_ast_root": str(args.spatial_ast_root),
            "spatial_ast_checkpoint": str(args.spatial_ast_checkpoint),
            "qa_root": str(args.qa_root),
            "audio_root_read_only": str(args.audio_root),
            "reverb_root": str(args.reverb_root),
            "qformer_source": str(args.qformer_source),
            "output": str(args.output),
        },
        "source_contract": source_contract,
        "spatial_ast_checkpoint": checkpoint_report,
        "spatial_ast_parameters": parameter_report,
        "samples": sample_reports,
        "qformer": qformer_report,
        "loader_contract": {
            "audio_channel_policy": "AudioSet first channel if source file is multi-channel",
            "resample_hz": SAMPLE_RATE,
            "normalization": "RMS target -14 dBFS",
            "rir_convolution": "full FFT convolution per source, then crop/pad to 10 seconds",
            "dual_source_mix": "render each source separately and average the two binaural waveforms",
            "spatial_ast_official_input": "waveforms [B,1,320000] plus binaural RIR [B,2,L]",
            "bat_llm_input": "rendered binaural waveform [B,2,320000]",
            "spatial_ast_token_contract": "[B,515,768] including 3 class tokens and 512 patch tokens",
            "qformer_contract": "[B,515,768] -> [B,64,2048]",
        },
        "issues": issues,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[samples] count={len(sample_reports)} labels={[item['label'] for item in sample_reports]}")
    print(f"[spatial-ast] params={parameter_report} strict_loaded={checkpoint_report['strict_loaded']}")
    print(f"[qformer] input={list(token_batch.shape)} output={list(projected.shape)}")
    print(f"[memory] peak_allocated={peak_allocated} peak_reserved={peak_reserved}")
    print(f"[report] {args.output}")
    print(f"[status] {report['status']} issues={issues}")


if __name__ == "__main__":
    main()
