from __future__ import annotations

import io
import os
import tarfile
import wave
from collections import OrderedDict
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
    from swift.utils import Processor, to_float_dtype
except ImportError:
    Processor = Any  # type: ignore

    def to_float_dtype(data: Any, dtype: torch.dtype | None):
        if dtype is None:
            return data
        if torch.is_tensor(data):
            return data.to(dtype=dtype) if torch.is_floating_point(data) else data
        if isinstance(data, dict):
            return {k: to_float_dtype(v, dtype) for k, v in data.items()}
        return data


REPO_ROOT = Path(__file__).resolve().parents[3]
AUDIO_MODEL_DIR = REPO_ROOT / "models" / "huginn-audio-whisper-v1"
HUGINN_MODEL_DIR = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/huginn-0125")
WHISPER_MODEL_DIR = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/whisper-large")

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that can understand audio and respond accurately."
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MAX_AUDIO_SECONDS = 30.0
TARFILE_CACHE_LIMIT = 4

ALIGNER_PREFIXES = (
    "temporal_compressor",
    "audio_projector",
    "audio_bos",
    "audio_eos",
)

MODEL_ARCH_NAME = "huginn_audio_whisper"
_TARFILE_CACHE: "OrderedDict[str, tarfile.TarFile]" = OrderedDict()


def patch_huginn_audio_shift_loss(model):
    if getattr(model, "_huginn_audio_shift_loss_patched", False):
        print("[HuginnAudioSwift] shift-loss patch already applied")
        return model

    original_forward = model.forward

    def forward_with_shift_loss(self, *args, **kwargs):
        labels = kwargs.get("labels")
        audio_input_features = kwargs.get("audio_input_features")
        past_key_values = kwargs.get("past_key_values")

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

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = full_labels[:, 1:].contiguous()

        if shift_labels.ne(-100).any():
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        else:
            loss = logits.new_tensor(0.0)

        outputs.loss = loss
        if hasattr(outputs, "log_ppl"):
            outputs.log_ppl = loss.detach().clone()
        return outputs

    model.forward = MethodType(forward_with_shift_loss, model)
    model._huginn_audio_shift_loss_patched = True
    print("[HuginnAudioSwift] applied shift-loss patch for multimodal SFT")
    return model


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


def trim_audio(audio: np.ndarray, target_sr: int, max_audio_seconds: float) -> np.ndarray:
    max_samples = int(round(max_audio_seconds * target_sr))
    if audio.shape[0] > max_samples:
        audio = audio[:max_samples]
    return audio.astype(np.float32, copy=False)


def load_wav_mono(path: Path, target_sr: int, max_audio_seconds: float) -> np.ndarray:
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


def load_audio_file(path: Path, target_sr: int, max_audio_seconds: float) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".wav":
        return load_wav_mono(path, target_sr, max_audio_seconds)
    if suffix in {".flac", ".ogg", ".mp3", ".m4a"}:
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
    max_audio_seconds: float,
) -> np.ndarray:
    tar_obj = get_cached_tarfile(tar_path)
    extracted = tar_obj.extractfile(member_name)
    if extracted is None:
        raise FileNotFoundError(f"Member {member_name} not found in tar archive {tar_path}")
    audio_bytes = extracted.read()
    audio, source_sr = decode_audio_bytes(audio_bytes, f"{tar_path}:{member_name}")
    audio = resample_waveform(audio, source_sr, target_sr)
    return trim_audio(audio, target_sr, max_audio_seconds)


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
    config.freeze_audio_encoder = True
    config.freeze_text_backbone = False

    model = AutoModelForCausalLM.from_config(
        config,
        trust_remote_code=True,
    )
    if not hasattr(model, "load_huginn_backbone_from_pretrained"):
        raise AttributeError("Audio Huginn model is missing load_huginn_backbone_from_pretrained")

    load_result = model.load_huginn_backbone_from_pretrained(
        str(HUGINN_MODEL_DIR),
        torch_dtype=torch.float32,
    )
    print_missing_key_summary(load_result.missing_keys, load_result.unexpected_keys)
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

    def _resolve_audio_path(self, audio_item: Any) -> Path:
        if isinstance(audio_item, str):
            return Path(audio_item)
        if isinstance(audio_item, dict):
            if "audio" in audio_item and isinstance(audio_item["audio"], str):
                return Path(audio_item["audio"])
            if "path" in audio_item and isinstance(audio_item["path"], str):
                return Path(audio_item["path"])
        raise TypeError(f"Unsupported audio source type: {type(audio_item)}")

    def _load_audio_item(self, audio_item: Any) -> np.ndarray:
        if isinstance(audio_item, dict) and "tar_path" in audio_item and "audio_member" in audio_item:
            tar_path = Path(str(audio_item["tar_path"]))
            audio_member = str(audio_item["audio_member"])
            return load_audio_from_tar(
                tar_path,
                audio_member,
                target_sr=self.audio_sampling_rate,
                max_audio_seconds=DEFAULT_MAX_AUDIO_SECONDS,
            )

        audio_path = self._resolve_audio_path(audio_item)
        return load_audio_file(
            audio_path,
            target_sr=self.audio_sampling_rate,
            max_audio_seconds=DEFAULT_MAX_AUDIO_SECONDS,
        )

    def _encode(self, inputs: StdTemplateInputs) -> dict[str, Any]:
        encoded = super()._encode(inputs)
        if not getattr(inputs, "audios", None):
            return encoded
        if len(inputs.audios) != 1:
            raise ValueError("Huginn audio Swift template currently supports exactly one audio clip per sample.")

        waveform = self._load_audio_item(inputs.audios[0])
        media_inputs = self.audio_feature_extractor(
            [waveform],
            sampling_rate=self.audio_sampling_rate,
            return_tensors="pt",
        )
        target_dtype = getattr(getattr(self, "model_info", None), "torch_dtype", None)
        media_inputs = to_float_dtype(media_inputs, target_dtype)
        encoded["audio_input_features"] = media_inputs["input_features"][0]
        return encoded

    def _data_collator_mm_data(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        audio_input_features = [item["audio_input_features"] for item in batch if "audio_input_features" in item]
        if not audio_input_features:
            return {}
        return {
            "audio_input_features": torch.stack(audio_input_features, dim=0),
        }


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
        config.freeze_audio_encoder = True
        config.freeze_text_backbone = False
        print(f"[HuginnAudioSwift] config.audio_encoder_name={config.audio_encoder_name}")
        print(f"[HuginnAudioSwift] config.audio_encoder_hidden_size={config.audio_encoder_hidden_size}")
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
        "aligner": ["temporal_compressor", "audio_projector", "audio_bos", "audio_eos"],
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

register_model(
    ModelMeta(
        "huginn_audio_raven",
        [
            ModelGroup(
                [
                    Model("huginn-audio-whisper-v1", str(AUDIO_MODEL_DIR)),
                ]
            ),
        ],
        HuginnAudioLoader,
        template="huginn_audio_text",
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
        template_type="huginn_audio_text",
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
