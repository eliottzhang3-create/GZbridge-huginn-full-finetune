"""Whisper audio-prefix wrapper around the native Transformers HRM-Text model."""

from __future__ import annotations

import gc
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import HrmTextConfig, HrmTextForCausalLM, WhisperModel

from .configuration_hrm_text_audio import HrmTextAudioConfig


class TrainableTemporalCompressor(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        target_token_count: int,
        intermediate_size: int,
        kernel_size: int,
        stride: int,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.target_token_count = target_token_count
        self.stride = stride
        self.gate_proj = nn.Conv1d(
            hidden_size,
            intermediate_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.up_proj = nn.Conv1d(
            hidden_size,
            intermediate_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.down_proj = nn.Conv1d(intermediate_size, hidden_size, kernel_size=1)
        self.shortcut_pool = nn.AvgPool1d(kernel_size=stride, stride=stride, ceil_mode=True)
        self.shortcut_proj = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.output_pool = nn.AdaptiveAvgPool1d(target_token_count)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.transpose(1, 2)
        gated = self.down_proj(self.up_proj(hidden_states) * torch.sigmoid(self.gate_proj(hidden_states)))
        shortcut = self.shortcut_proj(self.shortcut_pool(hidden_states))
        if gated.shape[-1] != shortcut.shape[-1]:
            common_length = min(gated.shape[-1], shortcut.shape[-1])
            gated = gated[..., :common_length]
            shortcut = shortcut[..., :common_length]
        return self.output_pool(gated + shortcut).transpose(1, 2)


class AudioProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.w1 = nn.Linear(input_dim, hidden_dim)
        self.w2 = nn.Linear(input_dim, hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.input_norm(hidden_states)
        hidden_states = self.w1(hidden_states) * F.silu(self.w2(hidden_states))
        return self.output_norm(self.c_proj(hidden_states))


class AudioBoundaryEmbeddings(nn.Module):
    def __init__(self, hidden_size: int, init_std: float):
        super().__init__()
        self.audio_bos = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.audio_eos = nn.Parameter(torch.empty(1, 1, hidden_size))
        nn.init.normal_(self.audio_bos, mean=0.0, std=init_std)
        nn.init.normal_(self.audio_eos, mean=0.0, std=init_std)


def _normalized_loading_info(loading_info: dict[str, Any]) -> dict[str, list[str]]:
    return {
        key: sorted(str(item) for item in loading_info.get(key, []))
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    }


class HrmTextAudioForConditionalGeneration(HrmTextForCausalLM):
    config_class = HrmTextAudioConfig

    def __init__(self, config: HrmTextAudioConfig):
        super().__init__(config)
        self.config = config
        self._initialize_audio_modules()
        self._freeze_requested_modules()

    def _initialize_audio_modules(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.audio_encoder: nn.Module | None = None
        self.temporal_compressor = TrainableTemporalCompressor(
            hidden_size=int(self.config.audio_encoder_hidden_size),
            target_token_count=int(self.config.audio_target_token_count),
            intermediate_size=int(self.config.audio_compressor_intermediate_size),
            kernel_size=int(self.config.audio_compressor_kernel_size),
            stride=int(self.config.audio_compressor_stride),
        )
        self.audio_projector = AudioProjector(
            input_dim=int(self.config.audio_encoder_hidden_size),
            hidden_dim=int(self.config.audio_projector_hidden_size),
            output_dim=int(self.config.hidden_size),
        )
        self.audio_boundary_embeddings = (
            AudioBoundaryEmbeddings(int(self.config.hidden_size), float(self.config.initializer_range))
            if bool(self.config.use_audio_boundary_embeddings)
            else None
        )
        self._reset_aligner_parameters()
        if device is not None or dtype is not None:
            move_kwargs: dict[str, Any] = {}
            if device is not None:
                move_kwargs["device"] = device
            if dtype is not None:
                move_kwargs["dtype"] = dtype
            self.temporal_compressor.to(**move_kwargs)
            self.audio_projector.to(**move_kwargs)
            if self.audio_boundary_embeddings is not None:
                self.audio_boundary_embeddings.to(**move_kwargs)

    @property
    def audio_bos(self) -> torch.Tensor | None:
        if self.audio_boundary_embeddings is None:
            return None
        return self.audio_boundary_embeddings.audio_bos

    @property
    def audio_eos(self) -> torch.Tensor | None:
        if self.audio_boundary_embeddings is None:
            return None
        return self.audio_boundary_embeddings.audio_eos

    @property
    def audio_prefix_length(self) -> int:
        boundary_count = int(self.audio_bos is not None) + int(self.audio_eos is not None)
        return int(self.config.audio_target_token_count) + boundary_count

    def _reset_aligner_parameters(self) -> None:
        init_std = float(self.config.initializer_range)
        for module in (self.temporal_compressor, self.audio_projector):
            for submodule in module.modules():
                if isinstance(submodule, (nn.Conv1d, nn.Linear)):
                    nn.init.normal_(submodule.weight, mean=0.0, std=init_std)
                    if submodule.bias is not None:
                        nn.init.zeros_(submodule.bias)
                elif isinstance(submodule, nn.LayerNorm):
                    nn.init.ones_(submodule.weight)
                    nn.init.zeros_(submodule.bias)
        if self.audio_boundary_embeddings is not None:
            nn.init.normal_(self.audio_boundary_embeddings.audio_bos, mean=0.0, std=init_std)
            nn.init.normal_(self.audio_boundary_embeddings.audio_eos, mean=0.0, std=init_std)

    def _freeze_requested_modules(self) -> None:
        if bool(self.config.freeze_text_backbone):
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            for parameter in self.lm_head.parameters():
                parameter.requires_grad = False
        if self.audio_encoder is not None and bool(self.config.freeze_audio_encoder):
            for parameter in self.audio_encoder.parameters():
                parameter.requires_grad = False
            self.audio_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.audio_encoder is not None and bool(self.config.freeze_audio_encoder):
            self.audio_encoder.eval()
        return self

    @classmethod
    def from_hrm_text_pretrained(
        cls,
        hrm_model_path: str | Path,
        *,
        audio_encoder_path: str | Path,
        config: HrmTextAudioConfig | None = None,
        dtype: torch.dtype = torch.bfloat16,
        device_map: dict[str, str] | None = None,
        attn_implementation: str = "sdpa",
        local_files_only: bool = True,
        low_cpu_mem_usage: bool = True,
    ) -> "HrmTextAudioForConditionalGeneration":
        hrm_model_path = str(Path(hrm_model_path).expanduser())
        audio_encoder_path = str(Path(audio_encoder_path).expanduser())
        base_config = HrmTextConfig.from_pretrained(hrm_model_path, local_files_only=local_files_only)
        if config is None:
            config_values = base_config.to_dict()
            for key in ("model_type", "architectures", "auto_map"):
                config_values.pop(key, None)
            config = HrmTextAudioConfig(
                base_model_name_or_path=hrm_model_path,
                audio_encoder_name=audio_encoder_path,
                **config_values,
            )
        else:
            config.base_model_name_or_path = hrm_model_path
            config.audio_encoder_name = audio_encoder_path

        critical_fields = (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_layers_per_stack",
            "num_attention_heads",
            "head_dim",
            "H_cycles",
            "L_cycles",
            "L_bp_cycles",
            "prefix_lm",
            "max_position_embeddings",
        )
        mismatches = {
            name: {"wrapper": getattr(config, name, None), "base": getattr(base_config, name, None)}
            for name in critical_fields
            if getattr(config, name, None) != getattr(base_config, name, None)
        }
        if mismatches:
            raise RuntimeError(f"HRM audio/base config mismatch: {mismatches}")

        # The published HRM checkpoint stores fused gqkv/gate_up tensors. Its
        # official Transformers conversion is selected from the native
        # ``hrm_text`` config model type, so load natively before upgrading the
        # same instance to this subclass. Loading directly with the custom
        # audio model type would bypass that conversion and randomly initialize
        # the complete H/L backbone.
        model, loading_info = HrmTextForCausalLM.from_pretrained(
            hrm_model_path,
            config=base_config,
            local_files_only=local_files_only,
            dtype=dtype,
            attn_implementation=attn_implementation,
            low_cpu_mem_usage=low_cpu_mem_usage,
            device_map=device_map,
            output_loading_info=True,
        )
        normalized_hrm_info = _normalized_loading_info(loading_info)
        if any(normalized_hrm_info.values()):
            raise RuntimeError(f"Native HRM checkpoint did not load strictly: {normalized_hrm_info}")

        runtime_config = model.config
        model.__class__ = cls
        model.config = config
        model.config.name_or_path = hrm_model_path
        for attribute in ("_attn_implementation", "_attn_implementation_internal"):
            if hasattr(runtime_config, attribute):
                setattr(model.config, attribute, getattr(runtime_config, attribute))
        reference_parameter = next(model.parameters())
        model._initialize_audio_modules(
            device=reference_parameter.device,
            dtype=reference_parameter.dtype,
        )

        whisper, whisper_loading_info = WhisperModel.from_pretrained(
            audio_encoder_path,
            local_files_only=local_files_only,
            dtype=dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            device_map=device_map,
            output_loading_info=True,
        )
        normalized_whisper_info = _normalized_loading_info(whisper_loading_info)
        if any(normalized_whisper_info.values()):
            raise RuntimeError(f"Unexpected Whisper checkpoint loading issues: {normalized_whisper_info}")
        if int(whisper.config.d_model) != int(config.audio_encoder_hidden_size):
            raise RuntimeError(
                f"Whisper hidden size mismatch: config={config.audio_encoder_hidden_size} "
                f"checkpoint={whisper.config.d_model}"
            )
        if int(whisper.config.num_mel_bins) != int(config.audio_feature_size):
            raise RuntimeError(
                f"Whisper mel-bin mismatch: config={config.audio_feature_size} "
                f"checkpoint={whisper.config.num_mel_bins}"
            )
        model.audio_encoder = whisper.encoder
        del whisper
        gc.collect()
        model._freeze_requested_modules()
        model._hrm_base_loading_info = normalized_hrm_info
        model._whisper_loading_info = normalized_whisper_info
        return model

    def build_audio_prefix(
        self,
        audio_input_features: torch.Tensor,
        audio_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.audio_encoder is None:
            raise RuntimeError("Whisper encoder is not loaded; use from_hrm_text_pretrained")
        if audio_input_features.ndim != 3:
            raise ValueError(
                "audio_input_features must have shape [batch, feature_size, frames], got "
                f"{tuple(audio_input_features.shape)}"
            )
        if audio_input_features.shape[1] != int(self.config.audio_feature_size):
            raise ValueError(
                f"Expected {self.config.audio_feature_size} mel bins, got {audio_input_features.shape[1]}"
            )
        if audio_attention_mask is not None and audio_attention_mask.shape[0] != audio_input_features.shape[0]:
            raise ValueError("audio_attention_mask batch size differs from audio_input_features")

        encoder_context = torch.no_grad() if bool(self.config.freeze_audio_encoder) else nullcontext()
        with encoder_context:
            audio_hidden = self.audio_encoder(
                input_features=audio_input_features,
                return_dict=True,
            ).last_hidden_state
        aligner_parameter = next(self.temporal_compressor.parameters())
        audio_hidden = audio_hidden.to(device=aligner_parameter.device, dtype=aligner_parameter.dtype)
        audio_embeds = self.audio_projector(self.temporal_compressor(audio_hidden))
        chunks = []
        if self.audio_bos is not None:
            chunks.append(self.audio_bos.expand(audio_embeds.shape[0], -1, -1))
        chunks.append(audio_embeds)
        if self.audio_eos is not None:
            chunks.append(self.audio_eos.expand(audio_embeds.shape[0], -1, -1))
        prefix = torch.cat(chunks, dim=1)
        expected_shape = (audio_embeds.shape[0], self.audio_prefix_length, int(self.config.hidden_size))
        if tuple(prefix.shape) != expected_shape:
            raise RuntimeError(f"Audio prefix shape mismatch: expected={expected_shape} actual={tuple(prefix.shape)}")
        return prefix

    @staticmethod
    def _cache_is_initialized(past_key_values: Any) -> bool:
        if past_key_values is None:
            return False
        get_seq_length = getattr(past_key_values, "get_seq_length", None)
        if callable(get_seq_length):
            return int(get_seq_length()) > 0
        initialized = getattr(past_key_values, "is_initialized", None)
        if initialized is not None:
            return bool(initialized)
        try:
            return len(past_key_values) > 0
        except TypeError:
            return True

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values=None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        audio_input_features: torch.Tensor | None = None,
        audio_attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        if cache_position is not None:
            kwargs["cache_position"] = cache_position
        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if audio_input_features is not None:
            model_inputs["audio_input_features"] = audio_input_features
        if audio_attention_mask is not None:
            model_inputs["audio_attention_mask"] = audio_attention_mask

        if audio_input_features is None:
            return model_inputs

        cache_initialized = self._cache_is_initialized(past_key_values)
        current_input = model_inputs.get("input_ids")
        if current_input is None:
            current_input = model_inputs.get("inputs_embeds")
        if current_input is None:
            raise RuntimeError("Generation prepare produced neither input_ids nor inputs_embeds")
        batch_size, sequence_length = current_input.shape[:2]

        if not cache_initialized:
            # Generic generation derives text-only positions before the wrapper
            # inserts its 34-token prefix. Let native HRM rebuild position_ids,
            # and expand legacy cache_position when that argument is present.
            model_inputs.pop("position_ids", None)
            if "cache_position" in model_inputs:
                model_inputs["cache_position"] = torch.arange(
                    self.audio_prefix_length + sequence_length,
                    dtype=torch.long,
                    device=current_input.device,
                )
            return model_inputs

        cache_length = int(past_key_values.get_seq_length())
        positions = torch.arange(
            cache_length,
            cache_length + sequence_length,
            dtype=torch.long,
            device=current_input.device,
        )
        model_inputs["position_ids"] = positions.unsqueeze(0).expand(batch_size, -1)
        if "cache_position" in model_inputs:
            model_inputs["cache_position"] = positions

        # Generated response tokens are causal, even though generic generation
        # would otherwise repeat the prompt's final PrefixLM type id (1).
        model_inputs["token_type_ids"] = torch.zeros(
            (batch_size, sequence_length),
            dtype=torch.long,
            device=current_input.device,
        )

        generation_attention = model_inputs.get("attention_mask")
        if generation_attention is not None and generation_attention.ndim == 2:
            expected_combined_length = cache_length + sequence_length
            if generation_attention.shape[1] + self.audio_prefix_length == expected_combined_length:
                prefix_attention = torch.ones(
                    (batch_size, self.audio_prefix_length),
                    dtype=generation_attention.dtype,
                    device=generation_attention.device,
                )
                generation_attention = torch.cat([prefix_attention, generation_attention], dim=1)
            if generation_attention.shape[1] != expected_combined_length:
                raise RuntimeError(
                    "Audio generation attention length mismatch after cache alignment: "
                    f"attention={generation_attention.shape[1]} expected={expected_combined_length} "
                    f"cache={cache_length} current={sequence_length}"
                )
            model_inputs["attention_mask"] = generation_attention
        return model_inputs

    def _prepare_audio_labels(
        self,
        labels: torch.Tensor | None,
        *,
        text_length: int,
        prefix_length: int,
        logits_to_keep: int | torch.Tensor,
    ) -> torch.Tensor | None:
        if labels is None:
            return None
        if labels.ndim != 2:
            raise ValueError(f"labels must be rank two, got shape={tuple(labels.shape)}")
        if torch.is_tensor(logits_to_keep):
            if logits_to_keep.ndim != 1 or logits_to_keep.dtype != torch.bool:
                raise ValueError(
                    "Tensor logits_to_keep must be Swift's one-dimensional boolean position mask, got "
                    f"shape={tuple(logits_to_keep.shape)} dtype={logits_to_keep.dtype}"
                )
            if logits_to_keep.numel() != text_length:
                raise ValueError(
                    f"Tensor logits_to_keep length must match text length: "
                    f"mask={logits_to_keep.numel()} text={text_length}"
                )
            selected = int(logits_to_keep.sum().item())
            if labels.shape[1] != selected:
                raise ValueError(
                    f"Tensor-selected compact labels mismatch: labels={labels.shape[1]} selected={selected}"
                )
            return labels
        keep = int(logits_to_keep)
        if keep > 0:
            if labels.shape[1] != keep:
                raise ValueError(
                    f"Compact labels must match logits_to_keep: labels={labels.shape[1]} keep={keep}"
                )
            return labels
        if labels.shape[1] == text_length + prefix_length:
            return labels
        if labels.shape[1] != text_length:
            raise ValueError(
                "Full labels must match the text or combined sequence length: "
                f"labels={labels.shape[1]} text={text_length} prefix={prefix_length}"
            )
        prefix_labels = torch.full(
            (labels.shape[0], prefix_length),
            -100,
            dtype=labels.dtype,
            device=labels.device,
        )
        return torch.cat([prefix_labels, labels], dim=1)

    @staticmethod
    def _shift_logits_to_keep_for_audio_prefix(
        logits_to_keep: int | torch.Tensor,
        *,
        text_length: int,
        prefix_length: int,
        device: torch.device,
    ) -> int | torch.Tensor:
        if not torch.is_tensor(logits_to_keep):
            return int(logits_to_keep)
        if logits_to_keep.ndim != 1 or logits_to_keep.dtype != torch.bool:
            raise ValueError("Only one-dimensional boolean tensor logits_to_keep is supported")
        if logits_to_keep.numel() != text_length:
            raise ValueError(
                f"Tensor logits_to_keep length must match text length: "
                f"mask={logits_to_keep.numel()} text={text_length}"
            )
        prefix_mask = torch.zeros(prefix_length, dtype=torch.bool, device=device)
        return torch.cat([prefix_mask, logits_to_keep.to(device=device)], dim=0)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        token_type_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        audio_input_features: torch.Tensor | None = None,
        audio_attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        audio_prefill = audio_input_features is not None and not self._cache_is_initialized(past_key_values)
        if not audio_prefill:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

        if position_ids is not None:
            raise ValueError("Do not pass position_ids on audio prefill; HRM must rebuild positions after prefix insertion")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Audio prefill requires exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            text_embeds = self.get_input_embeddings()(input_ids)
            text_batch, text_length = input_ids.shape
        else:
            text_embeds = inputs_embeds
            text_batch, text_length = inputs_embeds.shape[:2]
        if token_type_ids is None:
            raise ValueError("Audio prefill requires PrefixLM token_type_ids for the text sequence")
        if tuple(token_type_ids.shape) != (text_batch, text_length):
            raise ValueError(
                f"token_type_ids shape must match text sequence: {tuple(token_type_ids.shape)} "
                f"vs {(text_batch, text_length)}"
            )
        if attention_mask is None:
            attention_mask = torch.ones(
                (text_batch, text_length),
                dtype=torch.long,
                device=text_embeds.device,
            )
        if tuple(attention_mask.shape) != (text_batch, text_length):
            raise ValueError(
                f"attention_mask shape must match text sequence: {tuple(attention_mask.shape)} "
                f"vs {(text_batch, text_length)}"
            )

        audio_prefix = self.build_audio_prefix(audio_input_features, audio_attention_mask)
        audio_prefix = audio_prefix.to(device=text_embeds.device, dtype=text_embeds.dtype)
        prefix_length = audio_prefix.shape[1]
        combined_embeds = torch.cat([audio_prefix, text_embeds], dim=1)
        prefix_attention = torch.ones(
            (text_batch, prefix_length),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        combined_attention = torch.cat([prefix_attention, attention_mask], dim=1)
        prefix_token_types = torch.ones(
            (text_batch, prefix_length),
            dtype=token_type_ids.dtype,
            device=token_type_ids.device,
        )
        combined_token_types = torch.cat([prefix_token_types, token_type_ids], dim=1)
        combined_labels = self._prepare_audio_labels(
            labels,
            text_length=text_length,
            prefix_length=prefix_length,
            logits_to_keep=logits_to_keep,
        )
        combined_logits_to_keep = self._shift_logits_to_keep_for_audio_prefix(
            logits_to_keep,
            text_length=text_length,
            prefix_length=prefix_length,
            device=combined_embeds.device,
        )
        return super().forward(
            input_ids=None,
            attention_mask=combined_attention,
            position_ids=None,
            past_key_values=past_key_values,
            token_type_ids=combined_token_types,
            inputs_embeds=combined_embeds,
            labels=combined_labels,
            use_cache=use_cache,
            logits_to_keep=combined_logits_to_keep,
            **kwargs,
        )


__all__ = [
    "AudioBoundaryEmbeddings",
    "AudioProjector",
    "HrmTextAudioForConditionalGeneration",
    "TrainableTemporalCompressor",
]
