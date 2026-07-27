"""Profile Dynamic-90s LoSATok MACs on one ordinary (non-FSDP) GPU.

The script measures operator MACs only.  It intentionally does not read the
ACAVCAPS WebDataset, load a DCP checkpoint, initialize distributed training,
or save any model state.  The official LoSATok stack and the Huginn backbone
are loaded so that the traced shapes and recurrence path match the current
route.  Three audio durations are profiled by default because the dynamic
compressor has length-dependent compute.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "huginn-audio-losatok-v1"
DEFAULT_PLUGIN = REPO_ROOT / "code" / "huginn_lora" / "plugins" / "huginn_losatok_swift.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--plugin", default=str(DEFAULT_PLUGIN))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--audio_seconds", type=float, nargs="+", default=[5.0, 30.0, 90.0])
    parser.add_argument("--text_tokens", type=int, default=64)
    parser.add_argument("--no_grad_steps", type=int, default=24)
    parser.add_argument("--grad_steps", type=int, default=8)
    parser.add_argument("--with_lora", action="store_true", default=True)
    parser.add_argument("--without_lora", action="store_false", dest="with_lora")
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def load_plugin(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"LoSATok plugin does not exist: {path}")
    spec = importlib.util.spec_from_file_location("losatok_dynamic_macs_plugin", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import LoSATok plugin: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AudioEncoderOutput(nn.Module):
    """Adapt the official encoder's list output to one traceable tensor."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, audio_values: torch.Tensor, audio_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(audio_values, audio_mask)
        return outputs[0]


class ProjectorPath(nn.Module):
    def __init__(self, compressor: nn.Module, projector: nn.Module):
        super().__init__()
        self.compressor = compressor
        self.projector = projector

    def forward(self, encoder_tokens: torch.Tensor) -> torch.Tensor:
        return self.projector(self.compressor(encoder_tokens))


class TextForward(nn.Module):
    def __init__(self, model: nn.Module, num_steps: tuple[int, int]):
        super().__init__()
        self.model = model
        self.num_steps = num_steps

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_steps=self.num_steps,
        )
        return outputs.logits


class MultimodalForward(nn.Module):
    def __init__(self, model: nn.Module, num_steps: tuple[int, int]):
        super().__init__()
        self.model = model
        self.num_steps = num_steps

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_values: torch.Tensor,
        audio_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_input_values=audio_values,
            audio_attention_mask=audio_mask,
            num_steps=self.num_steps,
        )
        return outputs.logits


def profile_macs(profile_macs_fn: Callable[..., Any], model: nn.Module, args: tuple[Any, ...]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        with torch.no_grad():
            value = profile_macs_fn(model, args=args)
        return {
            "status": "PASS",
            "macs": int(value) if isinstance(value, (int, float)) else repr(value),
            "elapsed_seconds": time.perf_counter() - started,
        }
    except Exception as exc:  # noqa: BLE001 - retain partial profiling evidence
        return {
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": time.perf_counter() - started,
        }


def inject_lora(model: nn.Module) -> tuple[nn.Module, dict[str, Any]]:
    from peft import LoraConfig, TaskType, get_peft_model

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=r"^(transformer(?=\.).*\.(fc|adapter|Wqkv|proj))$",
        bias="none",
        inference_mode=False,
    )
    wrapped = get_peft_model(model, config)
    targets = sum(
        1
        for name, _ in wrapped.named_modules()
        if name.endswith(".lora_A.default") or name == "lora_A.default"
    )
    trainable = sum(parameter.numel() for parameter in wrapped.parameters() if parameter.requires_grad)
    return wrapped, {
        "enabled": True,
        "target_count": targets,
        "trainable_parameter_count": trainable,
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
    }


def main() -> int:
    args = parse_args()
    if not args.audio_seconds or any(seconds <= 0 or seconds > 90 for seconds in args.audio_seconds):
        raise ValueError("audio_seconds must be in (0, 90]")
    if args.text_tokens <= 0 or args.no_grad_steps < 0 or args.grad_steps <= 0:
        raise ValueError("text_tokens must be positive, no_grad_steps >= 0, grad_steps > 0")

    os.environ["HUGINN_LOSATOK_DYNAMIC_AUDIO_TOKENS"] = "1"
    os.environ["HUGINN_AUDIO_FSDP2_NONPERSISTENT_ROPE"] = "0"
    os.environ.pop("HUGINN_LOSATOK_INIT_FSDP_DCP_CHECKPOINT", None)
    os.environ.pop("HUGINN_LOSATOK_INIT_ALIGNER_CHECKPOINT", None)
    os.environ.pop("HUGINN_LOSATOK_PEFT_ALIGNER_MODULES_TO_SAVE", None)

    model_dir = Path(args.model_dir).expanduser().resolve()
    plugin_path = Path(args.plugin).expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is unavailable for the requested profiling device")
    device = torch.device(args.device)
    torch.cuda.set_device(device) if device.type == "cuda" else None

    import torchprofile

    profile_macs_fn = getattr(torchprofile, "profile_macs", None)
    if not callable(profile_macs_fn):
        raise RuntimeError("torchprofile.profile_macs is unavailable")

    print("========== LOSATOK DYNAMIC90S MACS PROFILE ==========")
    print(f"[context] python={sys.version.split()[0]} platform={platform.platform()}")
    print(f"[context] torch={torch.__version__} device={device} model_dir={model_dir}")
    print(f"[context] torchprofile={getattr(torchprofile, '__version__', '<unknown>')} module={torchprofile.__file__}")
    print(
        f"[config] audio_seconds={list(args.audio_seconds)} text_tokens={args.text_tokens} "
        f"recurrence_no_grad={args.no_grad_steps} recurrence_grad={args.grad_steps} with_lora={args.with_lora}"
    )

    plugin = load_plugin(plugin_path)
    print("========== MODEL LOAD ==========")
    model = plugin.build_model(str(model_dir))
    model = model.to(device=device)
    model.eval()

    lora_report: dict[str, Any] = {"enabled": False}
    if args.with_lora:
        try:
            model, lora_report = inject_lora(model)
            model = model.to(device=device)
            model.eval()
        except Exception as exc:  # noqa: BLE001 - profile base model if PEFT tracing setup fails
            lora_report = {
                "enabled": False,
                "requested": True,
                "status": "FAIL",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            print(f"[lora] injection failed; continuing with base model: {lora_report}")
    print(f"[lora] report={json.dumps(lora_report, sort_keys=True)}")

    config = model.get_base_model().config if hasattr(model, "get_base_model") else model.config
    vocab_size = int(getattr(config, "padded_vocab_size", getattr(config, "vocab_size", 65536)))
    pad_id = int(getattr(config, "pad_token_id", 65509))
    bos_id = int(getattr(config, "bos_token_id", 65504))
    input_ids = torch.full((1, args.text_tokens), bos_id, dtype=torch.long, device=device)
    text_mask = torch.ones_like(input_ids)
    num_steps = (args.no_grad_steps, args.grad_steps)

    results: list[dict[str, Any]] = []
    audio_encoder_wrapper = AudioEncoderOutput(model.get_base_model().audio_encoder if hasattr(model, "get_base_model") else model.audio_encoder).to(device)
    projector_path = ProjectorPath(
        model.get_base_model().temporal_compressor if hasattr(model, "get_base_model") else model.temporal_compressor,
        model.get_base_model().audio_projector if hasattr(model, "get_base_model") else model.audio_projector,
    ).to(device)
    text_wrapper = TextForward(model, num_steps).to(device).eval()
    multimodal_wrapper = MultimodalForward(model, num_steps).to(device).eval()

    print("========== TEXT-ONLY BASELINE ==========")
    text_result = profile_macs(profile_macs_fn, text_wrapper.eval(), (input_ids, text_mask))
    text_result.update({"component": "huginn_text_forward", "text_tokens": args.text_tokens, "recurrence": list(num_steps)})
    print(json.dumps(text_result, sort_keys=True))
    results.append(text_result)

    for seconds in args.audio_seconds:
        samples = int(round(seconds * 16000))
        audio_values = torch.zeros((1, samples), dtype=torch.float32, device=device)
        audio_values[:, : min(samples, 1600)] = torch.randn((1, min(samples, 1600)), device=device) * 0.01
        audio_mask = torch.ones_like(audio_values, dtype=torch.long)
        label = f"{seconds:g}s"
        print(f"========== AUDIO PROFILE {label} ==========")

        with torch.no_grad():
            encoded_values = audio_encoder_wrapper(audio_values, audio_mask)
        encoded_tokens = int(encoded_values.size(1))
        aligner_dtype = next(
            (model.get_base_model() if hasattr(model, "get_base_model") else model).temporal_compressor.parameters()
        ).dtype
        encoded_for_aligner = encoded_values.to(dtype=aligner_dtype)
        with torch.no_grad():
            compressed_values = projector_path.compressor(encoded_for_aligner)
        compressed_tokens = int(compressed_values.size(1))
        print(f"[shape] seconds={seconds:g} encoder_tokens={encoded_tokens} compressed_tokens={compressed_tokens}")

        encoder_result = profile_macs(profile_macs_fn, audio_encoder_wrapper.eval(), (audio_values, audio_mask))
        encoder_result.update({"component": "losatok_encoder", "audio_seconds": seconds, "audio_samples": samples, "encoder_tokens": encoded_tokens})
        print(json.dumps(encoder_result, sort_keys=True))
        results.append(encoder_result)

        aligner_result = profile_macs(profile_macs_fn, projector_path.eval(), (encoded_for_aligner,))
        aligner_result.update({"component": "temporal_compressor_plus_projector", "audio_seconds": seconds, "encoder_tokens": encoded_tokens, "compressed_tokens": compressed_tokens})
        print(json.dumps(aligner_result, sort_keys=True))
        results.append(aligner_result)

        multimodal_result = profile_macs(
            profile_macs_fn,
            multimodal_wrapper.eval(),
            (input_ids, text_mask, audio_values, audio_mask),
        )
        multimodal_result.update({
            "component": "full_multimodal_forward",
            "audio_seconds": seconds,
            "audio_samples": samples,
            "text_tokens": args.text_tokens,
            "encoder_tokens": encoded_tokens,
            "compressed_tokens": compressed_tokens,
            "recurrence": list(num_steps),
        })
        print(json.dumps(multimodal_result, sort_keys=True))
        results.append(multimodal_result)

        del audio_values, audio_mask, encoded_values, encoded_for_aligner, compressed_values
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = {
        "status": "PASS" if all(item["status"] == "PASS" for item in results) else "PARTIAL",
        "model_dir": str(model_dir),
        "plugin": str(plugin_path),
        "torch": torch.__version__,
        "torchprofile": getattr(torchprofile, "__version__", "<unknown>"),
        "device": str(device),
        "dynamic_audio_policy": {"max_seconds": 90, "kernel": 11, "stride": 6, "max_audio_tokens": 375},
        "lora": lora_report,
        "results": results,
        "no_checkpoint_load": True,
        "no_acavcaps_tar_scan": True,
    }
    output_path = Path(args.output_json).expanduser().resolve() if args.output_json else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[output] path={output_path}")
    print("========== MACS PROFILE RESULT ==========")
    print(f"[result] status={output['status']} profile_count={len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
