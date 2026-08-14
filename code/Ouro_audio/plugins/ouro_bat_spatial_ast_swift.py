"""ms-swift registration for the BAT Spatial-AST -> Q-Former -> Ouro path.

Model contract
--------------
* Ouro-1.4B native backbone: frozen, with rank-8 LoRA added by Swift/PEFT.
* Ouro early-exit gate: frozen; ``total_ut_steps=4`` and threshold ``1.0``.
* Spatial-AST: official pretrained encoder, fully frozen.
* BAT Q-Former: official eight-layer, 64-query projector, trainable.
* Audio representation: exactly 64 x 2048 embeddings.

The first integration uses a fixed audio-prefix representation.  The Swift
template prepends 64 harmless pad/eos token ids and masks their labels with
``-100``.  The model replaces those 64 token embeddings with Q-Former output.
This keeps input_ids, labels, attention_mask, and logits_to_keep aligned while
avoiding an unverified vocabulary extension for a new ``<audio>`` token.
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
except ImportError:  # pragma: no cover - compatibility with older Swift layouts
    from swift.llm import MultiModelKeys, register_model_arch  # type: ignore

try:
    from swift.template import StdTemplateInputs, Template, TemplateMeta, register_template
except ImportError:  # pragma: no cover - compatibility with older Swift layouts
    from swift.llm import StdTemplateInputs, Template, TemplateMeta, register_template  # type: ignore

try:
    from swift.utils import Processor, to_float_dtype
except ImportError:  # pragma: no cover - compatibility with older Swift layouts
    Processor = Any  # type: ignore

    def to_float_dtype(value: Any, dtype: torch.dtype | None):
        if dtype is None:
            return value
        if torch.is_tensor(value) and torch.is_floating_point(value):
            return value.to(dtype=dtype)
        return value


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

MODEL_TYPE = "ouro_bat_spatial_ast"
TEMPLATE_TYPE = "ouro_bat_audio_prefix"
MODEL_ARCH = "ouro_bat_spatial_ast_arch"
DEFAULT_MODEL_DIR = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/Ouro-1.4B")
DEFAULT_SPATIAL_AST_ROOT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/code/Spatial-AST")
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
OURO_HIDDEN_SIZE = 2048
EXPECTED_UT_STEPS = 4
EXPECTED_EARLY_EXIT_THRESHOLD = 1.0
AUDIO_AUDIT_ENABLED = os.environ.get("BAT_AUDIO_AUDIT", "0") == "1"


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


class OuroBATProcessor:
    """Tokenizer-compatible processor carrying the BAT audio renderer."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __getattr__(self, name: str) -> Any:
        return getattr(self.tokenizer, name)


def build_processor(model_dir: str | Path) -> OuroBATProcessor:
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return OuroBATProcessor(tokenizer)


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
            if arch_kwargs:
                metadata = MultiModelKeys(**arch_kwargs, **kwargs)
            else:
                metadata = MultiModelKeys(MODEL_ARCH, **kwargs)
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
        raise RuntimeError(f"Unable to register Ouro BAT model architecture: {last_error}")


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
    if getattr(model, "_ouro_bat_audio_forward_installed", False):
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
            self._ouro_bat_last_audio_records = bat_audio_records
        if audio_waveforms is None:
            return original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                **kwargs,
            )

        if inputs_embeds is not None:
            raise ValueError("Ouro BAT audio forward does not accept caller-provided inputs_embeds")
        if input_ids is None:
            raise ValueError("Ouro BAT audio forward requires input_ids")
        if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
            raise ValueError(f"Expected input_ids [B,T], got {type(input_ids).__name__} {getattr(input_ids, 'shape', None)}")

        # During generation, the first call has no cache and consumes the full
        # audio-prefixed sequence. Later calls retain the KV cache and must
        # process only newly generated text tokens; passing audio again would
        # duplicate the prefix.
        past_key_values = kwargs.get("past_key_values")
        if past_key_values is not None:
            return original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=None,
                labels=labels,
                **kwargs,
            )

        waveforms = audio_waveforms
        if not torch.is_tensor(waveforms):
            raise TypeError(f"audio_waveforms must be a tensor, got {type(waveforms).__name__}")
        if waveforms.ndim != 3 or waveforms.shape[0] != input_ids.shape[0]:
            raise ValueError(
                "Expected audio_waveforms [B,2,320000] with batch matching input_ids, "
                f"got audio={tuple(waveforms.shape)} input_ids={tuple(input_ids.shape)}"
            )

        audio_embeddings = self.audio_qformer(self.spatial_ast_encoder(waveforms))
        if tuple(audio_embeddings.shape[1:]) != (AUDIO_TOKEN_COUNT, OURO_HIDDEN_SIZE):
            raise RuntimeError(f"Unexpected BAT audio embedding shape: {tuple(audio_embeddings.shape)}")

        # The template carries 64 dummy ids so Swift can pad/collate a normal
        # integer token sequence. Those ids are placeholders only; replace
        # exactly that prefix rather than appending another 64 positions.
        if input_ids.shape[1] < AUDIO_TOKEN_COUNT:
            raise ValueError(
                f"Input sequence is shorter than the required audio prefix: {tuple(input_ids.shape)}"
            )
        placeholder_ids = input_ids[:, :AUDIO_TOKEN_COUNT]
        text_ids = input_ids[:, AUDIO_TOKEN_COUNT:]
        embed_tokens = self.get_input_embeddings()
        text_embeddings = embed_tokens(text_ids)
        audio_embeddings = audio_embeddings.to(device=text_embeddings.device, dtype=text_embeddings.dtype)
        inputs_embeds = torch.cat([audio_embeddings, text_embeddings], dim=1)
        if tuple(inputs_embeds.shape[:2]) != tuple(input_ids.shape):
            raise RuntimeError(
                "Audio prefix replacement changed sequence width: "
                f"input_ids={tuple(input_ids.shape)} inputs_embeds={tuple(inputs_embeds.shape)}"
            )
        if AUDIO_AUDIT_ENABLED:
            self._ouro_bat_last_audio_forward_audit = {
                "input_ids_shape": list(input_ids.shape),
                "audio_embeddings_shape": list(audio_embeddings.shape),
                "text_embeddings_shape": list(text_embeddings.shape),
                "inputs_embeds_shape": list(inputs_embeds.shape),
                "audio_prefix_replaced": True,
                "audio_prefix_token_count": AUDIO_TOKEN_COUNT,
            }
        call_kwargs = dict(kwargs)
        call_kwargs["inputs_embeds"] = inputs_embeds
        call_kwargs["input_ids"] = None

        if labels is not None:
            if labels.ndim != 2 or labels.shape[0] != input_ids.shape[0]:
                raise ValueError(f"Expected labels [B,T], got {tuple(labels.shape)}")
            if labels.shape[1] == input_ids.shape[1] - AUDIO_TOKEN_COUNT:
                labels = _prefix_labels(labels, AUDIO_TOKEN_COUNT)
            elif labels.shape[1] != inputs_embeds.shape[1]:
                raise ValueError(
                    "BAT labels must be text-width or full audio-prefixed width: "
                    f"labels={tuple(labels.shape)} input_ids={tuple(input_ids.shape)} "
                    f"audio_prefix={AUDIO_TOKEN_COUNT}"
                )

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if attention_mask.shape[1] != inputs_embeds.shape[1]:
            raise ValueError(
                f"Attention mask width is incompatible with audio prefix: {tuple(attention_mask.shape)}"
            )

        for name in ("position_ids", "cache_position"):
            value = kwargs.get(name)
            if value is not None and torch.is_tensor(value):
                expected_width = inputs_embeds.shape[1]
                if value.ndim >= 1 and value.shape[-1] != expected_width:
                    call_kwargs.pop(name, None)

        # The prefix is part of the real sequence during training. Returning
        # the full sequence also keeps Swift's logits_to_keep/labels contract
        # explicit and auditable.
        call_kwargs["attention_mask"] = attention_mask
        call_kwargs["labels"] = labels
        call_kwargs.setdefault("use_cache", False)
        # The original model must receive only one input representation. When
        # the caller passed input_ids positionally, remove that positional
        # argument because kwargs now carries input_ids=None and inputs_embeds.
        del placeholder_ids
        return original_forward(**call_kwargs)

    model.forward = MethodType(forward_with_bat_audio, model)
    model._ouro_bat_audio_forward_installed = True


def _attach_bat_audio_modules(model: nn.Module) -> dict[str, Any]:
    from compat.ouro_cache import patch_ouro_cache
    from bat.models.spatial_ast_audio import BATQFormer, SpatialASTAudioEncoder, BATAudioRenderer

    patch_report = patch_ouro_cache(model)
    spatial_ast_root = env_path("BAT_SPATIAL_AST_CODE_ROOT", DEFAULT_SPATIAL_AST_ROOT)
    spatial_ast_checkpoint = env_path("BAT_SPATIAL_AST_CHECKPOINT", DEFAULT_SPATIAL_AST_CHECKPOINT)
    qformer_source = env_path("BAT_QFORMER_SOURCE", DEFAULT_QFORMER_SOURCE)
    audio_root = env_path("BAT_AUDIO_ROOT", DEFAULT_AUDIO_ROOT)
    reverb_root = env_path("BAT_REVERB_ROOT", DEFAULT_REVERB_ROOT)

    model.requires_grad_(False)
    model.spatial_ast_encoder = SpatialASTAudioEncoder(spatial_ast_root, spatial_ast_checkpoint)
    model.audio_qformer = BATQFormer(
        qformer_source,
        encoder_dim=SPATIAL_AST_HIDDEN_SIZE,
        llm_dim=OURO_HIDDEN_SIZE,
        layers=8,
        query_len=AUDIO_TOKEN_COUNT,
    )
    model.audio_renderer = BATAudioRenderer(audio_root, reverb_root)

    device = _parameter_device(model)
    model.spatial_ast_encoder.to(device)
    model.audio_qformer.to(device)
    model.spatial_ast_encoder.requires_grad_(False).eval()
    model.audio_qformer.requires_grad_(True).train()

    total_ut_steps = int(getattr(model.config, "total_ut_steps", -1))
    if total_ut_steps != EXPECTED_UT_STEPS:
        raise RuntimeError(f"Expected Ouro total_ut_steps={EXPECTED_UT_STEPS}, got {total_ut_steps}")
    threshold = getattr(model, "early_exit_threshold", None)
    if threshold is None or float(threshold) != EXPECTED_EARLY_EXIT_THRESHOLD:
        raise RuntimeError(
            f"Expected frozen early_exit_threshold={EXPECTED_EARLY_EXIT_THRESHOLD}, got {threshold}"
        )
    if any(parameter.requires_grad for name, parameter in model.named_parameters() if "early_exit_gate" in name):
        raise RuntimeError("Ouro early_exit_gate must remain frozen in BAT training")
    model.config.use_cache = False
    if hasattr(model, "model") and hasattr(model.model, "config"):
        model.model.config.use_cache = False

    _install_audio_forward(model)
    return {
        "cache_patch": patch_report,
        "spatial_ast_root": str(spatial_ast_root),
        "spatial_ast_checkpoint": str(spatial_ast_checkpoint),
        "qformer_source": str(qformer_source),
        "qformer_initialization": "random",
        "qformer_checkpoint_loaded": False,
        "audio_root_read_only": str(audio_root),
        "reverb_root": str(reverb_root),
        "audio_token_count": AUDIO_TOKEN_COUNT,
        "spatial_ast_hidden_size": SPATIAL_AST_HIDDEN_SIZE,
        "ouro_hidden_size": OURO_HIDDEN_SIZE,
        "total_ut_steps": total_ut_steps,
        "early_exit_threshold": float(threshold),
        "encoder_trainable_parameters": sum(p.numel() for p in model.spatial_ast_encoder.parameters() if p.requires_grad),
        "qformer_trainable_parameters": sum(p.numel() for p in model.audio_qformer.parameters() if p.requires_grad),
        "gate_trainable_parameters": sum(p.numel() for n, p in model.named_parameters() if "early_exit_gate" in n and p.requires_grad),
    }


class OuroBATLoader(ModelLoader):
    def get_processor(self, model_dir: str, config: Any) -> Processor:
        del config
        return build_processor(model_dir)

    def get_model(self, model_dir: str, config: Any, processor: Any, model_kwargs: dict[str, Any]):
        model = super().get_model(model_dir, config, processor, model_kwargs)
        report = _attach_bat_audio_modules(model)
        model._ouro_bat_audio_contract = report
        print(f"[OuroBATSwift] audio_contract={report}", flush=True)
        return model


class OuroBATTemplate(Template):
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

    def replace_tag(self, media_type: str, index: int, inputs: StdTemplateInputs):
        if media_type == "audio":
            # Audio is represented by a fixed prefix, not a tokenizer token.
            return []
        return super().replace_tag(media_type, index, inputs)

    def _encode(self, inputs: StdTemplateInputs) -> dict[str, Any]:
        encoded = super()._encode(inputs)
        audios = getattr(inputs, "audios", None) or []
        if len(audios) != 1:
            raise ValueError(f"Ouro BAT template requires exactly one audio record, got {len(audios)}")
        waveform = self.audio_renderer.load_item(audios[0])
        tokenizer = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
        dummy_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if dummy_id is None:
            raise ValueError("Ouro tokenizer has neither pad_token_id nor eos_token_id")

        input_ids = encoded["input_ids"]
        if torch.is_tensor(input_ids):
            input_ids = input_ids.tolist()
        input_ids = [int(value) for value in input_ids]
        labels = encoded.get("labels")
        if torch.is_tensor(labels):
            labels = labels.tolist()
        if labels is not None:
            labels = [int(value) for value in labels]
        prefix_ids = [int(dummy_id)] * self.audio_token_count
        encoded["input_ids"] = prefix_ids + list(input_ids)
        if labels is not None:
            encoded["labels"] = [-100] * self.audio_token_count + list(labels)
        encoded["audio_waveform"] = waveform
        if AUDIO_AUDIT_ENABLED:
            # Preserve only the source metadata needed by the smoke audit.
            # The model forward consumes this field before calling Ouro.
            encoded["bat_audio_record"] = dict(audios[0])
        return encoded

    def _data_collator_mm_data(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        waveforms = [item.get("audio_waveform") for item in batch]
        if any(value is None for value in waveforms):
            raise ValueError("Every Ouro BAT sample must contain audio_waveform")
        stacked = torch.stack([value if torch.is_tensor(value) else torch.as_tensor(value) for value in waveforms], dim=0)
        payload = {"audio_waveforms": stacked.float()}
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
        model_groups=[ModelGroup(models=[Model(model_path=str(env_path("OURO_MODEL_PATH", DEFAULT_MODEL_DIR)))])],
        loader=OuroBATLoader,
        template=TEMPLATE_TYPE,
        model_arch=MODEL_ARCH,
        architectures=["OuroForCausalLM"],
        is_multimodal=True,
        torch_dtype=torch.bfloat16,
        requires=["transformers==4.54.1", "soundfile", "scipy", "timm", "librosa"],
        tags=["ouro", "bat", "spatial-audio", "audio"],
    ),
    exist_ok=True,
)

register_template(
    TemplateMeta(
        template_type=TEMPLATE_TYPE,
        template_cls=OuroBATTemplate,
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
    f"[OuroBATSwift] registered model_type={MODEL_TYPE} template={TEMPLATE_TYPE} "
    f"model_arch={MODEL_ARCH} audio_tokens={AUDIO_TOKEN_COUNT}",
    flush=True,
)
