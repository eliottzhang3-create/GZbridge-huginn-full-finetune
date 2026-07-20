#!/usr/bin/env python3
"""Verify a locally mirrored LoSATok + MiDasheng encoder on one remote GPU."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_LOSATOK_ROOT = "/hpc_stor03/sjtu_home/jinwei.zhang/models/LoSATok"
DEFAULT_CODE_DIR = (
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/"
    "code/huginn_lora/LosatokCode"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--losatok-root", default=DEFAULT_LOSATOK_ROOT)
    parser.add_argument("--code-dir", default=DEFAULT_CODE_DIR)
    parser.add_argument("--checkpoint-name", default="losatok_kl1e-3.pth")
    parser.add_argument("--input-wav", default=None)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def package_report() -> dict[str, dict[str, str | bool]]:
    result: dict[str, dict[str, str | bool]] = {}
    for package_name in ("torch", "transformers", "yaml", "einops", "librosa", "torchaudio", "soundfile"):
        spec = importlib.util.find_spec(package_name)
        entry: dict[str, str | bool] = {"available": spec is not None}
        if spec is not None:
            try:
                module = __import__(package_name)
                entry["version"] = str(getattr(module, "__version__", "unknown"))
            except Exception as exc:  # pragma: no cover - remote dependency dependent
                entry["import_error"] = f"{type(exc).__name__}: {exc}"
        result[package_name] = entry
    return result


def read_pcm_wav(path: Path) -> tuple[torch.Tensor, int]:
    with wave.open(str(path), "rb") as handle:
        channel_count = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frame_count = handle.getnframes()
        compression = handle.getcomptype()
        raw = handle.readframes(frame_count)
    if compression != "NONE":
        raise ValueError(f"Expected uncompressed WAV, got compression={compression!r}: {path}")
    if sample_width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width={sample_width} bytes: {path}")
    if samples.size % channel_count != 0:
        raise ValueError(f"Invalid PCM frame layout: {path}")
    samples = samples.reshape(-1, channel_count).mean(axis=1)
    return torch.from_numpy(samples.copy()).unsqueeze(0), int(sample_rate)


def build_local_semantic_encoder_class(midasheng_dir: Path):
    """Use local MiDasheng without the official source's Hugging Face network ID."""
    import torch.nn as nn
    from transformers import AutoModelForCausalLM

    import semantic_bottleneck

    class LocalSemanticEncoder(nn.Module):
        def __init__(self, high_dim: int = 1280, low_dim: int = 128, hidden_dim: int = 512):
            super().__init__()
            # The full 7B causal LM is needed only transiently to obtain audio_encoder.
            # bf16 plus low_cpu_mem_usage keeps that transient load below the 32G job cap.
            dashenglm = AutoModelForCausalLM.from_pretrained(
                str(midasheng_dir),
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            if not hasattr(dashenglm, "audio_encoder"):
                raise AttributeError("Local MiDasheng model has no audio_encoder attribute")
            self.encoder = dashenglm.audio_encoder
            del dashenglm
            gc.collect()
            self.bottleneck = semantic_bottleneck.SemanticBottleneck(high_dim, low_dim, hidden_dim)
            self.freeze_encoder()

        def freeze_encoder(self) -> None:
            self.encoder.eval()
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        def forward(self, audio: torch.Tensor):
            encoder_dtype = next(self.encoder.parameters()).dtype
            with torch.no_grad():
                embeddings = self.encoder(audio.to(dtype=encoder_dtype))[0].detach().float()
            low, reconstructed = self.bottleneck(embeddings)
            loss_dict = semantic_bottleneck.semantic_bottleneck_loss(embeddings, reconstructed, low)
            return embeddings, low, reconstructed, loss_dict

    return LocalSemanticEncoder


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if not torch.cuda.is_available():
        raise RuntimeError("LoSATok inspect requires CUDA")

    root = Path(args.losatok_root)
    code_dir = Path(args.code_dir)
    ckpt_dir = root / "ckpts"
    midasheng_dir = root / "midashenglm"
    semantic_checkpoint = ckpt_dir / "semantic_encoder.pth"
    losatok_checkpoint = ckpt_dir / args.checkpoint_name
    config_path = code_dir / "config" / "16k_16k_25Hz_losatok.yml"
    input_wav = Path(args.input_wav) if args.input_wav else code_dir / "example" / "en.wav"
    required_paths = (root, code_dir, midasheng_dir, semantic_checkpoint, losatok_checkpoint, config_path, input_wav)
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing LoSATok paths: {missing_paths}")

    packages = package_report()
    print("========== LOSATOK REMOTE ENCODER INSPECT ==========")
    print(f"[env] python={sys.version.split()[0]} executable={sys.executable}")
    print(f"[env] cuda_device={args.device} cuda_name={torch.cuda.get_device_name(args.device)}")
    print(f"[paths] losatok_root={root}")
    print(f"[paths] code_dir={code_dir}")
    print(f"[paths] midasheng_dir={midasheng_dir}")
    print(f"[paths] semantic_checkpoint={semantic_checkpoint} bytes={semantic_checkpoint.stat().st_size}")
    print(f"[paths] losatok_checkpoint={losatok_checkpoint} bytes={losatok_checkpoint.stat().st_size}")
    print(f"[paths] input_wav={input_wav}")
    print(f"[packages] {json.dumps(packages, ensure_ascii=False)}")

    required_modules = ("yaml", "einops")
    unavailable = [name for name in required_modules if not packages[name].get("available")]
    if unavailable:
        raise RuntimeError(f"Required LoSATok Python modules are unavailable: {unavailable}")

    sys.path.insert(0, str(code_dir))
    import yaml
    import losatok

    # AudioVAE resolves SemanticEncoder through the losatok module global at runtime.
    losatok.SemanticEncoder = build_local_semantic_encoder_class(midasheng_dir)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    audio_vae_config = {
        key.split("AudioVAE.", 1)[1]: value
        for key, value in config.items()
        if key.startswith("AudioVAE.")
    }
    audio_vae_config["semantic_encoder_path"] = str(semantic_checkpoint)

    print("[stage] build=AudioVAE_with_local_midasheng", flush=True)
    model = losatok.AudioVAE(**audio_vae_config)
    print("[stage] load=losatok_checkpoint", flush=True)
    state = torch.load(losatok_checkpoint, map_location="cpu", weights_only=False)
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unexpected LoSATok checkpoint payload: {type(state_dict)}")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"LoSATok checkpoint mismatch: missing={missing_keys[:20]} unexpected={unexpected_keys[:20]}"
        )
    model.eval().to(args.device)

    waveform, sample_rate = read_pcm_wav(input_wav)
    duration_seconds = waveform.shape[-1] / float(sample_rate)
    if sample_rate != 16000:
        raise ValueError(f"Expected the inspect WAV to be 16 kHz, got {sample_rate} Hz: {input_wav}")
    print(
        f"[audio] sample_rate={sample_rate} waveform_shape={tuple(waveform.shape)} "
        f"duration_seconds={duration_seconds:.6f}",
        flush=True,
    )

    with torch.inference_mode():
        embeddings = model.encoder_forward(waveform.to(args.device))
        mu = model.fc_mu(embeddings["unified_emb_low"])
    token_count = int(mu.shape[1])
    token_rate_hz = token_count / duration_seconds
    semantic_encoder_trainable = sum(
        parameter.numel() for parameter in model.semantic_encoder.encoder.parameters() if parameter.requires_grad
    )
    report = {
        "losatok_root": str(root),
        "code_dir": str(code_dir),
        "midasheng_dir": str(midasheng_dir),
        "checkpoint": str(losatok_checkpoint),
        "input_wav": str(input_wav),
        "input_sample_rate": sample_rate,
        "input_duration_seconds": duration_seconds,
        "token_count": token_count,
        "token_rate_hz": token_rate_hz,
        "feature_shapes": {key: list(value.shape) for key, value in embeddings.items()} | {"mu": list(mu.shape)},
        "semantic_encoder_trainable_parameter_count": semantic_encoder_trainable,
        "model_trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "model_total_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint_missing_keys": missing_keys,
        "checkpoint_unexpected_keys": unexpected_keys,
        "gpu_memory_allocated_gib": torch.cuda.memory_allocated(args.device) / (1024**3),
        "gpu_memory_reserved_gib": torch.cuda.memory_reserved(args.device) / (1024**3),
    }
    if mu.shape[-1] != 128 or embeddings["unified_emb"].shape[-1] != 1280:
        raise RuntimeError(f"Unexpected LoSATok dimensions: {report['feature_shapes']}")
    if not 20.0 <= token_rate_hz <= 30.0:
        raise RuntimeError(f"Expected approximately 25 Hz tokens, got {token_rate_hz:.4f} Hz")
    if semantic_encoder_trainable != 0:
        raise RuntimeError("Local MiDasheng audio encoder is unexpectedly trainable")

    output_report = Path(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = output_report.with_name(f"{output_report.name}.tmp")
    temporary_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_report.replace(output_report)
    print(f"[result] {json.dumps(report, ensure_ascii=False)}")
    print(f"[result] output_report={output_report}")
    print("========== LOSATOK REMOTE ENCODER INSPECT PASSED ==========")


if __name__ == "__main__":
    main()
