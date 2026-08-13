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
    module = import_module_from_file("bat_official_spatial_ast", source_path)
    model = module.build_AST(num_classes=355, num_cls_tokens=3)
    return model, {"path": str(source_path), "sha256": sha256_file(source_path)}


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
