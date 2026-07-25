"""Register the independent HRM-Text Whisper audio wrapper in ms-swift 4.4.2."""

from __future__ import annotations

import importlib.util
import os
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
from transformers import AutoTokenizer, WhisperFeatureExtractor

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
    from swift.utils import Processor, to_float_dtype
except ImportError:
    Processor = Any  # type: ignore

    def to_float_dtype(data: Any, dtype: torch.dtype | None):
        if dtype is None:
            return data
        if torch.is_tensor(data):
            return data.to(dtype=dtype) if torch.is_floating_point(data) else data
        if isinstance(data, dict):
            return {key: to_float_dtype(value, dtype) for key, value in data.items()}
        return data


MODEL_TYPE = "hrm_text_audio_whisper"
TEMPLATE_TYPE = "hrm_text_audio"
MODEL_ARCH_NAME = "hrm_text_audio_whisper"

REPO_ROOT = Path(__file__).resolve().parents[3]
WRAPPER_MODEL_DIR = REPO_ROOT / "models" / "hrm-text-audio-v1"
HRM_MODEL_DIR = Path(
    os.environ.get("HRM_TEXT_MODEL_PATH", "/hpc_stor03/sjtu_home/jinwei.zhang/models/HRM-text")
).expanduser()
WHISPER_MODEL_DIR = Path(
    os.environ.get("HRM_AUDIO_WHISPER_MODEL_PATH", "/hpc_stor03/sjtu_home/jinwei.zhang/models/whisper-large")
).expanduser()

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MAX_AUDIO_SECONDS = 30.0
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
DIRECT_CONDITION = "<|object_ref_start|>"


def _import_wrapper_package():
    package_name = "hrm_text_audio_v1_swift_plugin"
    existing = sys.modules.get(package_name)
    if existing is not None:
        return existing
    init_path = WRAPPER_MODEL_DIR / "__init__.py"
    if not init_path.is_file():
        raise FileNotFoundError(f"HRM audio wrapper package is missing: {init_path}")
    spec = importlib.util.spec_from_file_location(
        package_name,
        init_path,
        submodule_search_locations=[str(WRAPPER_MODEL_DIR)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import HRM audio wrapper from {WRAPPER_MODEL_DIR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module


WRAPPER_PACKAGE = _import_wrapper_package()


def _to_int_list(value: Any, *, name: str) -> list[int]:
    if torch.is_tensor(value):
        if value.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional before collation, got {tuple(value.shape)}")
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list/tuple/tensor, got {type(value)}")
    return [int(item) for item in value]


def _apply_prefix_lm_token_types(encoded: dict[str, Any]) -> dict[str, Any]:
    input_ids = _to_int_list(encoded.get("input_ids"), name="input_ids")
    labels_value = encoded.get("labels")
    if labels_value is None:
        prefix_length = len(input_ids)
    else:
        labels = _to_int_list(labels_value, name="labels")
        if len(labels) != len(input_ids):
            raise RuntimeError(
                f"HRM audio template input/label length mismatch: input={len(input_ids)} labels={len(labels)}"
            )
        supervised = [index for index, label in enumerate(labels) if label != -100]
        prefix_length = supervised[0] if supervised else len(input_ids)
        if any(label != -100 for label in labels[:prefix_length]):
            raise RuntimeError("HRM audio prompt labels must be -100 before the response boundary")
    if prefix_length <= 0 or prefix_length > len(input_ids):
        raise RuntimeError(
            f"Invalid HRM audio PrefixLM boundary: prefix={prefix_length} sequence={len(input_ids)}"
        )
    encoded["token_type_ids"] = [1] * prefix_length + [0] * (len(input_ids) - prefix_length)
    return encoded


def _as_mono_waveform(value: Any) -> torch.Tensor:
    waveform = torch.as_tensor(value)
    if not torch.is_floating_point(waveform):
        waveform = waveform.float()
    else:
        waveform = waveform.to(dtype=torch.float32)
    if waveform.ndim == 1:
        return waveform
    if waveform.ndim == 2:
        if waveform.shape[0] <= 8:
            return waveform.mean(dim=0)
        if waveform.shape[1] <= 8:
            return waveform.mean(dim=1)
    raise ValueError(f"Audio waveform must be mono or channel-first/channel-last rank two, got {tuple(waveform.shape)}")


def _load_audio_item(audio_item: Any) -> tuple[torch.Tensor, int]:
    if isinstance(audio_item, dict):
        if "array" in audio_item:
            sampling_rate = int(audio_item.get("sampling_rate", DEFAULT_SAMPLE_RATE))
            return _as_mono_waveform(audio_item["array"]), sampling_rate
        for key in ("path", "audio"):
            if key in audio_item:
                return _load_audio_item(audio_item[key])
        raise TypeError(f"Unsupported HRM audio mapping keys: {sorted(audio_item)}")
    if isinstance(audio_item, (str, os.PathLike)):
        path = Path(audio_item).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Audio file is missing: {path}")
        if path.suffix.lower() == ".wav":
            with wave.open(str(path), "rb") as handle:
                channels = handle.getnchannels()
                sample_width = handle.getsampwidth()
                sampling_rate = handle.getframerate()
                frame_count = handle.getnframes()
                compression = handle.getcomptype()
                payload = handle.readframes(frame_count)
            if compression != "NONE":
                raise ValueError(f"Compressed WAV is unsupported: {path} compression={compression}")
            if sample_width == 1:
                samples = (np.frombuffer(payload, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            elif sample_width == 2:
                samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
            elif sample_width == 4:
                samples = np.frombuffer(payload, dtype="<i4").astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"Unsupported PCM WAV sample width: {sample_width} bytes")
            if channels <= 0 or samples.size % channels != 0:
                raise ValueError(f"Invalid WAV channel layout: channels={channels} samples={samples.size}")
            samples = samples.reshape(-1, channels).mean(axis=1)
            return torch.from_numpy(samples.copy()), int(sampling_rate)
        waveform, sampling_rate = torchaudio.load(str(path))
        return _as_mono_waveform(waveform), int(sampling_rate)
    if isinstance(audio_item, (np.ndarray, torch.Tensor, list, tuple)):
        return _as_mono_waveform(audio_item), DEFAULT_SAMPLE_RATE
    raise TypeError(f"Unsupported HRM audio source type: {type(audio_item)}")


def load_audio_16k(audio_item: Any) -> np.ndarray:
    waveform, sampling_rate = _load_audio_item(audio_item)
    if sampling_rate <= 0:
        raise ValueError(f"Audio sampling rate must be positive, got {sampling_rate}")
    if sampling_rate != DEFAULT_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sampling_rate, DEFAULT_SAMPLE_RATE)
    max_samples = int(DEFAULT_SAMPLE_RATE * DEFAULT_MAX_AUDIO_SECONDS)
    waveform = waveform[:max_samples].contiguous()
    if waveform.numel() == 0:
        raise ValueError("Audio waveform is empty")
    if not bool(torch.isfinite(waveform).all().item()):
        raise ValueError("Audio waveform contains NaN or Inf")
    return waveform.cpu().numpy().astype(np.float32, copy=False)


class HrmTextAudioProcessor:
    def __init__(self, tokenizer, feature_extractor):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

    def __getattr__(self, item):
        return getattr(self.tokenizer, item)


def build_hrm_audio_processor() -> HrmTextAudioProcessor:
    tokenizer = AutoTokenizer.from_pretrained(HRM_MODEL_DIR, local_files_only=True, use_fast=True)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_MODEL_DIR, local_files_only=True)
    if int(feature_extractor.sampling_rate) != DEFAULT_SAMPLE_RATE:
        raise RuntimeError(f"Whisper sampling rate mismatch: {feature_extractor.sampling_rate}")
    if int(feature_extractor.feature_size) != 80:
        raise RuntimeError(f"Whisper feature size mismatch: {feature_extractor.feature_size}")
    return HrmTextAudioProcessor(tokenizer, feature_extractor)


class HrmTextAudioTemplate(Template):
    use_model = False
    support_padding_free = False

    def init_processor(self, processor: Processor):
        super().init_processor(processor)
        self.audio_feature_extractor = processor.feature_extractor
        self.audio_sampling_rate = int(self.audio_feature_extractor.sampling_rate)

    def replace_tag(self, media_type: str, index: int, inputs: StdTemplateInputs):
        if media_type == "audio":
            return []
        return super().replace_tag(media_type, index, inputs)

    def _encode(self, inputs: StdTemplateInputs) -> dict[str, Any]:
        encoded = _apply_prefix_lm_token_types(super()._encode(inputs))
        audios = getattr(inputs, "audios", None)
        if not audios:
            return encoded
        if len(audios) != 1:
            raise ValueError("HRM audio template currently supports exactly one audio clip per sample")
        waveform = load_audio_16k(audios[0])
        media_inputs = self.audio_feature_extractor(
            waveform,
            sampling_rate=self.audio_sampling_rate,
            return_tensors="pt",
            padding="max_length",
            max_length=int(self.audio_sampling_rate * DEFAULT_MAX_AUDIO_SECONDS),
            truncation=True,
        )
        input_features = media_inputs["input_features"]
        if tuple(input_features.shape) != (1, 80, 3000):
            raise RuntimeError(f"Unexpected Whisper input feature shape: {tuple(input_features.shape)}")
        target_dtype = getattr(getattr(self, "model_info", None), "torch_dtype", None)
        encoded["audio_input_features"] = to_float_dtype(input_features[0], target_dtype)
        return encoded

    def _data_collator_mm_data(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        present = ["audio_input_features" in item for item in batch]
        if not any(present):
            return {}
        if not all(present):
            raise RuntimeError("HRM audio batches cannot mix samples with and without audio features")
        features = torch.stack([item["audio_input_features"] for item in batch], dim=0)
        if features.ndim != 3 or tuple(features.shape[1:]) != (80, 3000):
            raise RuntimeError(f"Collated Whisper features have invalid shape: {tuple(features.shape)}")
        return {"audio_input_features": features}


class HrmTextAudioLoader(ModelLoader):
    def get_config(self, model_dir: str):
        config = WRAPPER_PACKAGE.HrmTextAudioConfig.from_pretrained(model_dir, local_files_only=True)
        config.base_model_name_or_path = str(HRM_MODEL_DIR)
        config.audio_encoder_name = str(WHISPER_MODEL_DIR)
        config.freeze_audio_encoder = True
        config.freeze_text_backbone = True
        return config

    def get_processor(self, model_dir: str, config):
        del model_dir, config
        processor = build_hrm_audio_processor()
        print(
            f"[HrmTextAudioSwift] tokenizer={type(processor.tokenizer)} "
            f"feature_extractor={type(processor.feature_extractor)}",
            flush=True,
        )
        return processor

    def get_model(self, model_dir: str, config, processor, model_kwargs):
        del model_dir, processor
        model_kwargs = dict(model_kwargs or {})
        dtype = model_kwargs.pop("dtype", model_kwargs.pop("torch_dtype", torch.bfloat16))
        if dtype is None:
            dtype = torch.bfloat16
        device_map = model_kwargs.pop("device_map", None)
        local_files_only = bool(model_kwargs.pop("local_files_only", True))
        low_cpu_mem_usage = bool(model_kwargs.pop("low_cpu_mem_usage", True))
        attn_implementation = model_kwargs.pop(
            "attn_implementation",
            getattr(config, "_attn_implementation", "sdpa") or "sdpa",
        )
        model_kwargs.pop("trust_remote_code", None)
        if model_kwargs:
            raise RuntimeError(f"Unsupported HRM audio Swift model kwargs: {sorted(model_kwargs)}")
        model = WRAPPER_PACKAGE.HrmTextAudioForConditionalGeneration.from_hrm_text_pretrained(
            HRM_MODEL_DIR,
            audio_encoder_path=WHISPER_MODEL_DIR,
            config=config,
            dtype=dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
            local_files_only=local_files_only,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
        print(f"[HrmTextAudioSwift] model={type(model)}", flush=True)
        return model


def register_hrm_audio_model_arch() -> None:
    grouped_modules = {
        "language_model": ["model", "lm_head"],
        "aligner": ["temporal_compressor", "audio_projector", "audio_boundary_embeddings"],
        "generator": ["audio_encoder"],
    }
    try:
        model_keys = MultiModelKeys(arch_name=MODEL_ARCH_NAME, **grouped_modules)
    except TypeError as exc:
        if "arch_name" not in str(exc):
            raise
        try:
            model_keys = MultiModelKeys(model_arch=MODEL_ARCH_NAME, **grouped_modules)
        except TypeError as inner_exc:
            if "model_arch" not in str(inner_exc):
                raise
            model_keys = MultiModelKeys(MODEL_ARCH_NAME, **grouped_modules)
    try:
        register_model_arch(model_keys)
    except ValueError as exc:
        if f"The `{MODEL_ARCH_NAME}` has already been registered" not in str(exc):
            raise


register_hrm_audio_model_arch()
register_model(
    ModelMeta(
        model_type=MODEL_TYPE,
        model_groups=[ModelGroup(models=[Model(model_path=str(WRAPPER_MODEL_DIR))])],
        loader=HrmTextAudioLoader,
        template=TEMPLATE_TYPE,
        model_arch=MODEL_ARCH_NAME,
        architectures=["HrmTextAudioForConditionalGeneration"],
        torch_dtype=torch.bfloat16,
        is_multimodal=True,
        requires=["transformers==5.9.0"],
        tags=["hrm", "audio", "whisper", "prefix-lm"],
    ),
    exist_ok=True,
)
register_template(
    TemplateMeta(
        template_type=TEMPLATE_TYPE,
        template_cls=HrmTextAudioTemplate,
        prefix=[],
        prompt=[f"{IM_START}{DIRECT_CONDITION}{{{{QUERY}}}}{IM_END}"],
        chat_sep=None,
        suffix=[["eos_token_id"]],
        auto_add_bos=False,
        stop_words=[],
    ),
    exist_ok=True,
)
print(
    f"[HrmTextAudioSwift] registered model_type={MODEL_TYPE} template={TEMPLATE_TYPE} "
    f"model_arch={MODEL_ARCH_NAME} wrapper={WRAPPER_MODEL_DIR}",
    flush=True,
)


__all__ = [
    "HrmTextAudioLoader",
    "HrmTextAudioProcessor",
    "HrmTextAudioTemplate",
    "MODEL_ARCH_NAME",
    "MODEL_TYPE",
    "TEMPLATE_TYPE",
    "build_hrm_audio_processor",
    "load_audio_16k",
]
