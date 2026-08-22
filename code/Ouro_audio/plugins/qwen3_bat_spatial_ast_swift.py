"""ms-swift registration for BAT Spatial-AST -> Q-Former -> Qwen3.

The Qwen3 branch deliberately has its own registration and forward wrapper.
It reuses only the validated BAT audio renderer and Q-Former implementation.

Trainable contract:
* Spatial-AST: frozen FP32 encoder.
* Q-Former: random initialization, trainable, 64 queries.
* Qwen3 native model: frozen.
* Qwen3 LoRA: added later by ms-swift/PEFT to q_proj and v_proj only.
* KV cache: disabled for training.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from swift.model import Model, ModelGroup, ModelLoader, ModelMeta, register_model

try:
    from swift.model import MultiModelKeys, register_model_arch
except ImportError:  # pragma: no cover
    from swift.llm import MultiModelKeys, register_model_arch  # type: ignore

try:
    from swift.template import StdTemplateInputs, Template, TemplateMeta, register_template
except ImportError:  # pragma: no cover
    from swift.llm import StdTemplateInputs, Template, TemplateMeta, register_template  # type: ignore

try:
    from swift.utils import Processor
except ImportError:  # pragma: no cover
    Processor = Any  # type: ignore


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

MODEL_TYPE = "qwen3_bat_spatial_ast"
TEMPLATE_TYPE = "qwen3_bat_audio_prefix"
MODEL_ARCH = "qwen3_bat_spatial_ast_arch"

DEFAULT_MODEL_DIR = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/models/Qwen3-4B-Base"
)
DEFAULT_SPATIAL_AST_ROOT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/Spatial-AST"
)
DEFAULT_SPATIAL_AST_CHECKPOINT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/models/BAT/SpatialAST/finetuned.pth"
)
DEFAULT_QFORMER_SOURCE = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/OWL/src/slam_llm/models/projector.py"
)
DEFAULT_AUDIO_ROOT = Path("/hpc_stor03/public/shared/data/raa/AudioSet")
DEFAULT_REVERB_ROOT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/SpatialSoundQA/mp3d_reverb"
)

AUDIO_TOKEN_COUNT = 64
SPATIAL_AST_HIDDEN_SIZE = 768
QWEN3_HIDDEN_SIZE = 2560
EXPECTED_QWEN3_LAYERS = 36
EXPECTED_QWEN3_VOCAB_SIZE = 151936
DEFAULT_TRAIN_SEQUENCE_LENGTH = 176


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
AUDIO_AUDIT_ENABLED = os.environ.get("BAT_AUDIO_AUDIT", "0") == "1"


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


class Qwen3BATProcessor:
    """Tokenizer-compatible processor used by the custom BAT template."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __getattr__(self, name: str) -> Any:
        return getattr(self.tokenizer, name)


def build_processor(model_dir: str | Path) -> Qwen3BATProcessor:
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return Qwen3BATProcessor(tokenizer)


def _register_architecture() -> None:
    kwargs = {
        "language_model": ["model", "lm_head"],
        "aligner": ["audio_qformer"],
        "generator": ["spatial_ast_encoder"],
    }
    attempts = (
        {"arch_name": MODEL_ARCH},
        {"model_arch": MODEL_ARCH},
        {},
    )
    registered = False
    last_error: Exception | None = None
    for arch_kwargs in attempts:
        try:
            metadata = (
                MultiModelKeys(**arch_kwargs, **kwargs)
                if arch_kwargs
                else MultiModelKeys(MODEL_ARCH, **kwargs)
            )
            register_model_arch(metadata)
            registered = True
            break
        except ValueError as exc:
            if "already been registered" in str(exc):
                registered = True
                break
            last_error = exc
        except TypeError as exc:
            last_error = exc
    if not registered:
        raise RuntimeError(
            f"Unable to register Qwen3 BAT model architecture: {last_error}"
        )


def _parameter_device(model: nn.Module) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _prefix_labels(labels: torch.Tensor, count: int) -> torch.Tensor:
    prefix = torch.full(
        (labels.shape[0], count),
        fill_value=-100,
        dtype=labels.dtype,
        device=labels.device,
    )
    return torch.cat([prefix, labels], dim=1)


def _install_audio_forward(model: nn.Module) -> None:
    if getattr(model, "_qwen3_bat_audio_forward_installed", False):
        return
    original_forward = model.forward

    def forward_with_bat_audio(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        audio_waveforms: torch.Tensor | None = None,
        bat_audio_records: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ):
        if AUDIO_AUDIT_ENABLED and bat_audio_records is not None:
            self._qwen3_bat_last_audio_records = bat_audio_records

        if audio_waveforms is None:
            return original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                **kwargs,
            )

        if inputs_embeds is not None:
            raise ValueError(
                "Qwen3 BAT audio forward does not accept caller-provided inputs_embeds"
            )
        if input_ids is None or input_ids.ndim != 2:
            raise ValueError(
                f"Expected input_ids [B,T], got {getattr(input_ids, 'shape', None)}"
            )

        # Generation after the first prefill call carries a cache and should
        # use normal Qwen3 input_ids without rebuilding the audio prefix.
        if kwargs.get("past_key_values") is not None:
            return original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=None,
                labels=labels,
                **kwargs,
            )

        if (
            not torch.is_tensor(audio_waveforms)
            or audio_waveforms.ndim != 3
            or audio_waveforms.shape[0] != input_ids.shape[0]
            or audio_waveforms.shape[1] != 2
            or audio_waveforms.shape[2] != 320000
        ):
            raise ValueError(
                "Expected audio_waveforms [B,2,320000] matching input_ids, "
                f"got audio={getattr(audio_waveforms, 'shape', None)} "
                f"input_ids={tuple(input_ids.shape)}"
            )
        if input_ids.shape[1] < AUDIO_TOKEN_COUNT:
            raise ValueError(
                f"Input sequence is shorter than audio prefix: {tuple(input_ids.shape)}"
            )

        audio_embeddings = self.audio_qformer(
            self.spatial_ast_encoder(audio_waveforms)
        )
        expected_audio_shape = (input_ids.shape[0], AUDIO_TOKEN_COUNT, QWEN3_HIDDEN_SIZE)
        if tuple(audio_embeddings.shape) != expected_audio_shape:
            raise RuntimeError(
                f"Unexpected Qwen3 audio embedding shape: "
                f"got={tuple(audio_embeddings.shape)} expected={expected_audio_shape}"
            )

        text_ids = input_ids[:, AUDIO_TOKEN_COUNT:]
        text_embeddings = self.get_input_embeddings()(text_ids)
        audio_embeddings = audio_embeddings.to(
            device=text_embeddings.device,
            dtype=text_embeddings.dtype,
        )
        merged_embeddings = torch.cat([audio_embeddings, text_embeddings], dim=1)
        if tuple(merged_embeddings.shape[:2]) != tuple(input_ids.shape):
            raise RuntimeError(
                "Audio prefix replacement changed sequence width: "
                f"input_ids={tuple(input_ids.shape)} "
                f"inputs_embeds={tuple(merged_embeddings.shape)}"
            )

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if tuple(attention_mask.shape) != tuple(input_ids.shape):
            raise ValueError(
                f"Attention mask is incompatible with input_ids: "
                f"attention={tuple(attention_mask.shape)} input={tuple(input_ids.shape)}"
            )

        call_kwargs = dict(kwargs)
        call_kwargs["input_ids"] = None
        call_kwargs["inputs_embeds"] = merged_embeddings
        call_kwargs["attention_mask"] = attention_mask
        call_kwargs["labels"] = labels
        call_kwargs.setdefault("use_cache", False)

        if labels is not None:
            if labels.ndim != 2 or labels.shape[0] != input_ids.shape[0]:
                raise ValueError(f"Expected labels [B,T], got {tuple(labels.shape)}")
            if labels.shape[1] == input_ids.shape[1] - AUDIO_TOKEN_COUNT:
                call_kwargs["labels"] = _prefix_labels(labels, AUDIO_TOKEN_COUNT)
            elif labels.shape[1] != input_ids.shape[1]:
                raise ValueError(
                    "Qwen3 BAT labels must be text-width or full prefixed width: "
                    f"labels={tuple(labels.shape)} input_ids={tuple(input_ids.shape)}"
                )

        for name in ("position_ids", "cache_position"):
            value = call_kwargs.get(name)
            if value is not None and torch.is_tensor(value):
                if value.ndim >= 1 and value.shape[-1] != merged_embeddings.shape[1]:
                    call_kwargs.pop(name, None)

        if AUDIO_AUDIT_ENABLED:
            self._qwen3_bat_last_audio_forward_audit = {
                "input_ids_shape": list(input_ids.shape),
                "audio_embeddings_shape": list(audio_embeddings.shape),
                "text_embeddings_shape": list(text_embeddings.shape),
                "inputs_embeds_shape": list(merged_embeddings.shape),
                "audio_prefix_replaced": True,
                "audio_prefix_token_count": AUDIO_TOKEN_COUNT,
            }
        return original_forward(**call_kwargs)

    model.forward = MethodType(forward_with_bat_audio, model)
    model._qwen3_bat_audio_forward_installed = True


def _attach_bat_audio_modules(model: nn.Module) -> dict[str, Any]:
    from bat.models.spatial_ast_audio import (
        BATAudioRenderer,
        BATQFormer,
        SpatialASTAudioEncoder,
    )

    config = model.config
    actual_summary = {
        "model_type": getattr(config, "model_type", None),
        "hidden_size": int(getattr(config, "hidden_size", -1)),
        "num_hidden_layers": int(getattr(config, "num_hidden_layers", -1)),
        "num_attention_heads": int(getattr(config, "num_attention_heads", -1)),
        "num_key_value_heads": int(getattr(config, "num_key_value_heads", -1)),
        "vocab_size": int(getattr(config, "vocab_size", -1)),
    }
    expected_summary = {
        "model_type": "qwen3",
        "hidden_size": QWEN3_HIDDEN_SIZE,
        "num_hidden_layers": EXPECTED_QWEN3_LAYERS,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": EXPECTED_QWEN3_VOCAB_SIZE,
    }
    if actual_summary != expected_summary:
        raise RuntimeError(
            f"Unexpected Qwen3 model config: "
            f"expected={expected_summary} actual={actual_summary}"
        )

    model.requires_grad_(False)
    model.spatial_ast_encoder = SpatialASTAudioEncoder(
        env_path("BAT_SPATIAL_AST_CODE_ROOT", DEFAULT_SPATIAL_AST_ROOT),
        env_path("BAT_SPATIAL_AST_CHECKPOINT", DEFAULT_SPATIAL_AST_CHECKPOINT),
    )
    model.audio_qformer = BATQFormer(
        env_path("BAT_QFORMER_SOURCE", DEFAULT_QFORMER_SOURCE),
        encoder_dim=SPATIAL_AST_HIDDEN_SIZE,
        llm_dim=QWEN3_HIDDEN_SIZE,
        layers=8,
        query_len=AUDIO_TOKEN_COUNT,
    )
    model.audio_renderer = BATAudioRenderer(
        env_path("BAT_AUDIO_ROOT", DEFAULT_AUDIO_ROOT),
        env_path("BAT_REVERB_ROOT", DEFAULT_REVERB_ROOT),
    )

    device = _parameter_device(model)
    model.spatial_ast_encoder.to(device).requires_grad_(False).eval()
    model.audio_qformer.to(device).requires_grad_(True).train()

    model.config.use_cache = False
    if hasattr(model, "model") and hasattr(model.model, "config"):
        model.model.config.use_cache = False
    _install_audio_forward(model)

    return {
        "model_type": "qwen3",
        "model_hidden_size": QWEN3_HIDDEN_SIZE,
        "model_layers": EXPECTED_QWEN3_LAYERS,
        "model_attention_heads": 32,
        "model_key_value_heads": 8,
        "model_vocab_size": EXPECTED_QWEN3_VOCAB_SIZE,
        "spatial_ast_root": str(env_path("BAT_SPATIAL_AST_CODE_ROOT", DEFAULT_SPATIAL_AST_ROOT)),
        "spatial_ast_checkpoint": str(env_path("BAT_SPATIAL_AST_CHECKPOINT", DEFAULT_SPATIAL_AST_CHECKPOINT)),
        "qformer_source": str(env_path("BAT_QFORMER_SOURCE", DEFAULT_QFORMER_SOURCE)),
        "qformer_initialization": "random",
        "qformer_checkpoint_loaded": False,
        "audio_root_read_only": str(env_path("BAT_AUDIO_ROOT", DEFAULT_AUDIO_ROOT)),
        "reverb_root": str(env_path("BAT_REVERB_ROOT", DEFAULT_REVERB_ROOT)),
        "rir_sample_rate": 32000,
        "rir_target_seconds": 2.0,
        "rir_target_samples": 64000,
        "rir_length_policy": "crop_or_zero_pad_before_convolution",
        "audio_token_count": AUDIO_TOKEN_COUNT,
        "spatial_ast_hidden_size": SPATIAL_AST_HIDDEN_SIZE,
        "qwen3_hidden_size": QWEN3_HIDDEN_SIZE,
        "encoder_trainable_parameters": sum(
            p.numel() for p in model.spatial_ast_encoder.parameters() if p.requires_grad
        ),
        "qformer_trainable_parameters": sum(
            p.numel() for p in model.audio_qformer.parameters() if p.requires_grad
        ),
        "qwen_native_trainable_parameters": sum(
            p.numel()
            for name, p in model.named_parameters()
            if p.requires_grad
            and not name.startswith(("audio_qformer.", "spatial_ast_encoder."))
        ),
        "gate_present": any(
            "early_exit_gate" in name for name, _ in model.named_parameters()
        ),
        "use_cache": False,
    }


class Qwen3BATLoader(ModelLoader):
    def get_processor(self, model_dir: str, config: Any) -> Processor:
        del config
        return build_processor(model_dir)

    def get_model(
        self,
        model_dir: str,
        config: Any,
        processor: Any,
        model_kwargs: dict[str, Any],
    ):
        model = super().get_model(model_dir, config, processor, model_kwargs)
        report = _attach_bat_audio_modules(model)
        model._qwen3_bat_audio_contract = report
        print(f"[Qwen3BATSwift] audio_contract={report}", flush=True)
        return model


class Qwen3BATTemplate(Template):
    use_model = False
    support_padding_free = False

    def init_processor(self, processor: Processor) -> None:
        super().init_processor(processor)
        from bat.models.spatial_ast_audio import BATAudioRenderer

        self.audio_renderer = BATAudioRenderer(
            env_path("BAT_AUDIO_ROOT", DEFAULT_AUDIO_ROOT),
            env_path("BAT_REVERB_ROOT", DEFAULT_REVERB_ROOT),
        )
        self.audio_token_count = AUDIO_TOKEN_COUNT
        self.fixed_sequence_length = env_bool("BAT_FIXED_SEQUENCE_LENGTH", True)
        self.train_sequence_length = int(
            os.environ.get("BAT_MAX_SEQUENCE_LENGTH", str(DEFAULT_TRAIN_SEQUENCE_LENGTH))
        )
        if self.train_sequence_length <= self.audio_token_count:
            raise ValueError(
                f"BAT_MAX_SEQUENCE_LENGTH must be greater than {self.audio_token_count}"
            )

    def replace_tag(self, media_type: str, index: int, inputs: StdTemplateInputs):
        if media_type == "audio":
            return []
        return super().replace_tag(media_type, index, inputs)

    def _encode(self, inputs: StdTemplateInputs) -> dict[str, Any]:
        encoded = super()._encode(inputs)
        audios = getattr(inputs, "audios", None) or []
        if len(audios) != 1:
            raise ValueError(
                f"Qwen3 BAT template requires exactly one audio record, got {len(audios)}"
            )
        tokenizer = (
            self.processor.tokenizer
            if hasattr(self.processor, "tokenizer")
            else self.processor
        )
        dummy_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        if dummy_id is None:
            raise ValueError("Qwen3 tokenizer has neither pad_token_id nor eos_token_id")

        input_ids = encoded["input_ids"]
        if torch.is_tensor(input_ids):
            input_ids = input_ids.tolist()
        input_ids = [int(value) for value in input_ids]
        labels = encoded.get("labels")
        if torch.is_tensor(labels):
            labels = labels.tolist()
        if labels is not None:
            labels = [int(value) for value in labels]
        if labels is not None and len(labels) != len(input_ids):
            raise ValueError(
                f"Text input/label widths differ: {len(input_ids)} vs {len(labels)}"
            )

        training_mode = getattr(self, "mode", "train") == "train"
        valid_text_length = len(input_ids)
        pad_count = 0
        if training_mode and self.fixed_sequence_length:
            text_budget = self.train_sequence_length - self.audio_token_count
            if len(input_ids) > text_budget:
                input_ids = input_ids[:text_budget]
                if labels is not None:
                    labels = labels[:text_budget]
            valid_text_length = len(input_ids)
            pad_count = text_budget - valid_text_length
            input_ids = input_ids + [int(dummy_id)] * pad_count
            if labels is not None:
                labels = labels + [-100] * pad_count
            encoded["attention_mask"] = (
                [1] * (self.audio_token_count + valid_text_length)
                + [0] * pad_count
            )
        elif training_mode:
            text_budget = self.train_sequence_length - self.audio_token_count
            if len(input_ids) > text_budget:
                input_ids = input_ids[:text_budget]
                if labels is not None:
                    labels = labels[:text_budget]
            valid_text_length = len(input_ids)
            encoded["attention_mask"] = [1] * (self.audio_token_count + valid_text_length)

        waveform = self.audio_renderer.load_item(audios[0])
        encoded["input_ids"] = [int(dummy_id)] * self.audio_token_count + list(input_ids)
        if labels is not None:
            encoded["labels"] = [-100] * self.audio_token_count + list(labels)
        encoded["audio_waveform"] = waveform
        encoded["bat_text_contract"] = {
            "training_mode": training_mode,
            "fixed_sequence_length": bool(self.fixed_sequence_length),
            "natural_text_length": len(input_ids),
            "audio_prefix_tokens": self.audio_token_count,
            "pre_collation_sequence_length": len(encoded["input_ids"]),
        }
        if AUDIO_AUDIT_ENABLED:
            encoded["bat_audio_record"] = dict(audios[0])
            encoded["bat_text_contract"].update({
                "training_mode": training_mode,
                "valid_text_length": valid_text_length,
                "padding_count": pad_count,
                "total_sequence_length": len(encoded["input_ids"]),
            })
        return encoded

    def _data_collator_mm_data(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        waveforms = [item.get("audio_waveform") for item in batch]
        if any(value is None for value in waveforms):
            raise ValueError("Every Qwen3 BAT sample must contain audio_waveform")
        payload = {
            "audio_waveforms": torch.stack(
                [
                    value if torch.is_tensor(value) else torch.as_tensor(value)
                    for value in waveforms
                ],
                dim=0,
            ).float()
        }
        if AUDIO_AUDIT_ENABLED:
            records = [item.get("bat_audio_record") for item in batch]
            if any(record is None for record in records):
                raise RuntimeError("BAT audio audit metadata was lost before collation")
            payload["bat_audio_records"] = records
        return payload


_register_architecture()

register_model(
    ModelMeta(
        model_type=MODEL_TYPE,
        model_groups=[
            ModelGroup(
                models=[
                    Model(
                        model_path=str(
                            env_path("QWEN3_MODEL_PATH", DEFAULT_MODEL_DIR)
                        )
                    )
                ]
            )
        ],
        loader=Qwen3BATLoader,
        template=TEMPLATE_TYPE,
        model_arch=MODEL_ARCH,
        architectures=["Qwen3ForCausalLM"],
        is_multimodal=True,
        torch_dtype=torch.bfloat16,
        requires=["transformers>=4.51.0", "soundfile", "scipy", "timm", "librosa"],
        tags=["qwen3", "qwen3-4b-base", "bat", "spatial-audio", "audio"],
    ),
    exist_ok=True,
)

register_template(
    TemplateMeta(
        template_type=TEMPLATE_TYPE,
        template_cls=Qwen3BATTemplate,
        prefix=[],
        prompt=["{{QUERY}}"],
        chat_sep=None,
        suffix=[["eos_token_id"]],
        auto_add_bos=False,
        stop_words=[],
    ),
    exist_ok=True,
)

print(
    f"[Qwen3BATSwift] registered model_type={MODEL_TYPE} "
    f"template={TEMPLATE_TYPE} model_arch={MODEL_ARCH} "
    f"audio_tokens={AUDIO_TOKEN_COUNT} hidden={QWEN3_HIDDEN_SIZE}",
    flush=True,
)
