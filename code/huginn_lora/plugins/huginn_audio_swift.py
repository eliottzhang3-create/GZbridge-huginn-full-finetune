from __future__ import annotations

import json
import wave
from pathlib import Path
from types import MethodType
from typing import Any, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from swift.model import Model, ModelGroup, ModelLoader, ModelMeta, register_model
from swift.template import TemplateMeta, register_template

REPO_ROOT = Path(__file__).resolve().parents[3]
AUDIO_MODEL_DIR = REPO_ROOT / "models" / "huginn-audio-whisper-v1"
HUGINN_MODEL_DIR = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/huginn-0125")
WHISPER_MODEL_DIR = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/whisper-large")

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that answers questions about audio."
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MAX_AUDIO_SECONDS = 30.0
DEFAULT_MAX_LENGTH = 192


def patch_huginn_audio_shift_loss(model):
    if getattr(model, "_huginn_audio_shift_loss_patched", False):
        print("[HuginnAudioLoader] shift-loss patch already applied")
        return model

    original_forward = model.forward

    def forward_with_shift_loss(self, *args, **kwargs):
        labels = kwargs.get("labels", None)
        audio_input_features = kwargs.get("audio_input_features", None)
        past_key_values = kwargs.get("past_key_values", None)

        if labels is None:
            return original_forward(*args, **kwargs)

        kwargs_no_labels = dict(kwargs)
        kwargs_no_labels["labels"] = None

        output = original_forward(*args, **kwargs_no_labels)
        logits = output.logits
        if logits is None:
            raise RuntimeError(
                "Huginn audio forward returned logits=None; cannot recompute shifted loss."
            )

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

        valid_mask = shift_labels.ne(-100)
        if not valid_mask.any():
            loss = logits.new_tensor(0.0)
        else:
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        output.loss = loss
        if hasattr(output, "log_ppl"):
            output.log_ppl = loss.detach().clone()

        return output

    model.forward = MethodType(forward_with_shift_loss, model)
    model._huginn_audio_shift_loss_patched = True
    print("[HuginnAudioLoader] applied shift-loss patch for audio SFT")
    return model


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
    max_samples = int(round(max_audio_seconds * target_sr))
    if audio.shape[0] > max_samples:
        audio = audio[:max_samples]
    return audio


def normalize_audio_record(record: dict[str, Any], dataset_dir: Optional[Path] = None) -> dict[str, str]:
    if "question" in record and "answer" in record and "audio_path" in record:
        query = f"Listen to the audio and answer the question.\nQuestion: {record['question']}"
        response = record["answer"]
        audio_path = record["audio_path"]
    elif "query" in record and "response" in record and "audio" in record:
        query = record["query"]
        response = record["response"]
        audio_path = record["audio"]
    else:
        raise KeyError(
            "Audio SFT record must contain either question/answer/audio_path or query/response/audio."
        )

    audio_path = Path(audio_path)
    if dataset_dir is not None and not audio_path.is_absolute():
        audio_path = dataset_dir / audio_path

    return {
        "query": str(query),
        "response": str(response),
        "audio_path": str(audio_path),
    }


def load_jsonl_records(manifest_path: Path, max_records: Optional[int] = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if max_records is not None and len(records) >= max_records:
                break
    if not records:
        raise ValueError(f"No records found in {manifest_path}")
    return records


class HuginnAudioProcessor:
    def __init__(self, tokenizer, feature_extractor):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

    def __getattr__(self, item):
        return getattr(self.tokenizer, item)

    def build_conversations(
        self,
        samples: Iterable[dict[str, str]],
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> list[list[dict[str, str]]]:
        conversations = []
        for sample in samples:
            conversations.append(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": sample["query"]},
                    {"role": "Huginn", "content": sample["response"]},
                ]
            )
        return conversations

    def collate_audio_sft_batch(
        self,
        samples: list[dict[str, str]],
        max_length: int = DEFAULT_MAX_LENGTH,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        max_audio_seconds: float = DEFAULT_MAX_AUDIO_SECONDS,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> dict[str, Any]:
        conversations = self.build_conversations(samples, system_prompt=system_prompt)
        chat_encoding = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
            padding="longest",
            max_length=max_length + 1,
            return_tensors="pt",
            return_dict=True,
            truncation=True,
        )

        input_ids = chat_encoding["input_ids"][:, :-1]
        assistant_masks = chat_encoding["assistant_masks"].bool()
        attention_mask = chat_encoding["attention_mask"]
        labels = torch.where(
            assistant_masks[:, 1:] & attention_mask[:, 1:].bool(),
            chat_encoding["input_ids"][:, 1:],
            torch.full_like(chat_encoding["input_ids"][:, 1:], -100),
        )

        waveforms = [
            load_wav_mono(Path(sample["audio_path"]), sample_rate, max_audio_seconds) for sample in samples
        ]
        audio_inputs = self.feature_extractor(
            waveforms,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask[:, :-1],
            "audio_input_features": audio_inputs["input_features"],
            "audio_paths": [sample["audio_path"] for sample in samples],
            "queries": [sample["query"] for sample in samples],
        }


def summarize_trainable_parameter_groups(model) -> dict[str, list[str]]:
    groups = {
        "audio_encoder": [],
        "aligner": [],
        "llm": [],
        "other_trainable": [],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("audio_encoder."):
            groups["audio_encoder"].append(name)
        elif (
            name.startswith("temporal_compressor.")
            or name.startswith("audio_projector.")
            or name.startswith("audio_bos")
            or name.startswith("audio_eos")
        ):
            groups["aligner"].append(name)
        elif name.startswith("transformer.") or name.startswith("lm_head."):
            groups["llm"].append(name)
        else:
            groups["other_trainable"].append(name)
    return groups


def get_huginn_backbone_lora_target_modules(model) -> list[str]:
    target_modules: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if name == "lm_head" or name.startswith("transformer."):
            target_modules.append(name)
    return sorted(set(target_modules))


def build_huginn_audio_model(model_dir: str, model_kwargs: Optional[dict[str, Any]] = None):
    model_kwargs = dict(model_kwargs or {})
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
    config.freeze_text_backbone = True

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
    print(
        f"[HuginnAudioLoader] backbone load missing={len(load_result.missing_keys)} "
        f"unexpected={len(load_result.unexpected_keys)}"
    )

    model = patch_huginn_audio_shift_loss(model)
    model._huginn_audio_lora_target_modules = get_huginn_backbone_lora_target_modules(model)
    model._huginn_audio_trainable_groups = summarize_trainable_parameter_groups(model)
    return model


def build_huginn_audio_processor():
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


class HuginnAudioLoader(ModelLoader):
    def get_config(self, model_dir: str):
        print(f"[HuginnAudioLoader] get_config: {model_dir}")
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
        config.freeze_text_backbone = True
        print(f"[HuginnAudioLoader] config type = {type(config)}")
        print(f"[HuginnAudioLoader] config.audio_encoder_name = {config.audio_encoder_name}")
        print(f"[HuginnAudioLoader] config.audio_encoder_hidden_size = {config.audio_encoder_hidden_size}")
        return config

    def get_processor(self, model_dir: str, config):
        del model_dir, config
        print("[HuginnAudioLoader] get_processor")
        processor = build_huginn_audio_processor()
        print(f"[HuginnAudioLoader] tokenizer type = {type(processor.tokenizer)}")
        print(f"[HuginnAudioLoader] feature_extractor type = {type(processor.feature_extractor)}")
        return processor

    def get_model(self, model_dir: str, config, processor, model_kwargs):
        del config, processor
        print(f"[HuginnAudioLoader] get_model: {model_dir}")
        model = build_huginn_audio_model(model_dir, model_kwargs=model_kwargs)
        print(f"[HuginnAudioLoader] model type = {type(model)}")
        print(
            f"[HuginnAudioLoader] huginn-only lora target modules count = "
            f"{len(model._huginn_audio_lora_target_modules)}"
        )
        return model


def create_huginn_audio_model_and_processor(
    model_dir: str = str(AUDIO_MODEL_DIR),
    model_kwargs: Optional[dict[str, Any]] = None,
):
    model = build_huginn_audio_model(model_dir, model_kwargs=model_kwargs)
    processor = build_huginn_audio_processor()
    return model, processor


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
        architectures=["HuginnAudioForConditionalGeneration"],
        requires=["transformers>=4.53.3"],
        tags=["huginn", "audio"],
    ),
    exist_ok=True,
)

register_template(
    TemplateMeta(
        template_type="huginn_audio_text",
        prefix=[],
        prompt=[
            "<|begin_header|>user<|end_header|>\n\n"
            "{{QUERY}}"
            "<|end_turn|>"
            "<|begin_header|>Huginn<|end_header|>\n\n"
        ],
        chat_sep=None,
        auto_add_bos=True,
        stop_words=[["eos_token_id"]],
    ),
    exist_ok=True,
)
