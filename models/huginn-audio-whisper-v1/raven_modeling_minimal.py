"""Audio experiment wrapper around the original Huginn backbone."""

from __future__ import annotations

import gc
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from transformers import WhisperModel

from ._base import CausalLMOutputRecurrentLatents, RavenForCausalLM
from .raven_config_minimal import HuginnAudioConfig


class TrainableTemporalCompressor(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        target_token_count: int,
        intermediate_size: int | None = None,
        kernel_size: int = 7,
        stride: int = 12,
    ):
        super().__init__()
        intermediate_size = intermediate_size or hidden_size * 2
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        gate = torch.sigmoid(self.gate_proj(x))
        value = self.up_proj(x)
        x_conv = self.down_proj(value * gate)

        x_shortcut = self.shortcut_proj(self.shortcut_pool(x))

        if x_conv.size(-1) != x_shortcut.size(-1):
            target_len = min(x_conv.size(-1), x_shortcut.size(-1))
            x_conv = x_conv[..., :target_len]
            x_shortcut = x_shortcut[..., :target_len]

        x = x_conv + x_shortcut
        x = self.output_pool(x)
        return x.transpose(1, 2)


class AudioProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.w1 = nn.Linear(input_dim, hidden_dim)
        self.w2 = nn.Linear(input_dim, hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        a1 = self.w1(x)
        a2 = self.w2(x)
        x = a1 * F.silu(a2)
        x = self.c_proj(x)
        return self.output_norm(x)


class AudioBoundaryEmbeddings(nn.Module):
    """Keep boundary parameters in a named module so Swift saves them with the aligner."""

    def __init__(self, hidden_size: int, init_std: float):
        super().__init__()
        self.audio_bos = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.audio_eos = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.normal_(self.audio_bos, mean=0.0, std=init_std)
        nn.init.normal_(self.audio_eos, mean=0.0, std=init_std)


class HuginnAudioForConditionalGeneration(RavenForCausalLM):
    config_class = HuginnAudioConfig

    def __init__(self, config: HuginnAudioConfig):
        super().__init__(config)
        self.config = config

        whisper = WhisperModel.from_pretrained(config.audio_encoder_name)
        self.audio_encoder = whisper.encoder
        del whisper

        self.temporal_compressor = TrainableTemporalCompressor(
            hidden_size=config.audio_encoder_hidden_size,
            target_token_count=config.audio_target_token_count,
            intermediate_size=config.audio_compressor_intermediate_size,
            kernel_size=config.audio_compressor_kernel_size,
            stride=config.audio_compressor_stride,
        )
        self.audio_projector = AudioProjector(
            input_dim=config.audio_encoder_hidden_size,
            hidden_dim=config.audio_projector_hidden_size,
            output_dim=config.n_embd,
        )

        self.audio_boundary_embeddings = (
            AudioBoundaryEmbeddings(config.n_embd, config.init_values["std"])
            if config.use_audio_boundary_embeddings
            else None
        )

        self._freeze_requested_modules()

    @property
    def audio_bos(self):
        if self.audio_boundary_embeddings is None:
            return None
        return self.audio_boundary_embeddings.audio_bos

    @property
    def audio_eos(self):
        if self.audio_boundary_embeddings is None:
            return None
        return self.audio_boundary_embeddings.audio_eos

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Accept any legacy checkpoint that stored the boundary parameters at model root.
        for name in ("audio_bos", "audio_eos"):
            legacy_key = f"{prefix}{name}"
            current_key = f"{prefix}audio_boundary_embeddings.{name}"
            if legacy_key in state_dict and current_key not in state_dict:
                state_dict[current_key] = state_dict.pop(legacy_key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _freeze_requested_modules(self):
        if self.config.freeze_text_backbone:
            for param in self.transformer.parameters():
                param.requires_grad = False
            for param in self.lm_head.parameters():
                param.requires_grad = False

        if self.config.freeze_audio_encoder:
            for param in self.audio_encoder.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def load_huginn_backbone_from_pretrained(
        self,
        base_model_name_or_path: str,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        base_model = RavenForCausalLM.from_pretrained(
            base_model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        backbone_state = base_model.state_dict()
        if "freqs_cis" in self._non_persistent_buffers_set:
            # FSDP2 reconstructs this deterministic RoPE table separately. Do not
            # report the legacy persistent entry from the Huginn checkpoint as unexpected.
            backbone_state.pop("freqs_cis", None)
        incompatible = self.load_state_dict(backbone_state, strict=False)
        del base_model
        gc.collect()
        self._freeze_requested_modules()
        return incompatible

    def build_audio_prefix(
        self,
        audio_input_features: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del audio_attention_mask
        encoder_outputs = self.audio_encoder(
            input_features=audio_input_features,
            return_dict=True,
        )
        audio_hidden = encoder_outputs.last_hidden_state
        audio_hidden = self.temporal_compressor(audio_hidden)
        audio_embeds = self.audio_projector(audio_hidden)

        prefix_chunks = []
        if self.audio_bos is not None:
            prefix_chunks.append(self.audio_bos.expand(audio_embeds.size(0), -1, -1))
        prefix_chunks.append(audio_embeds)
        if self.audio_eos is not None:
            prefix_chunks.append(self.audio_eos.expand(audio_embeds.size(0), -1, -1))
        return torch.cat(prefix_chunks, dim=1)

    def trainable_parameter_summary(self):
        trainable = []
        frozen = []
        for name, param in self.named_parameters():
            (trainable if param.requires_grad else frozen).append(name)
        return {"trainable": trainable, "frozen_count": len(frozen), "trainable_count": len(trainable)}

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        cache_lookup_strategy: str = "full",
        audio_input_features: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Expose custom audio inputs to Transformers generation validation."""
        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            cache_lookup_strategy=cache_lookup_strategy,
            **kwargs,
        )
        if audio_input_features is not None:
            model_inputs["audio_input_features"] = audio_input_features
        if audio_attention_mask is not None:
            model_inputs["audio_attention_mask"] = audio_attention_mask
        return model_inputs

    def forward(
        self,
        input_ids: torch.Tensor,
        input_embeds: Optional[torch.Tensor] = None,
        input_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        num_steps: Optional[torch.Tensor] = None,
        past_key_values=None,
        output_details: dict = {
            "return_logits": True,
            "return_latents": True,
            "return_head": False,
            "return_stats": False,
        },
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        init_scale: float = 1.0,
        audio_input_features: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputRecurrentLatents:
        model_input_ids = input_ids
        model_labels = labels
        model_attention_mask = attention_mask

        if audio_input_features is not None and past_key_values is None:
            if input_embeds is not None:
                raise ValueError("Pass either input_embeds or audio_input_features, not both.")

            text_embeds = self.transformer.wte(input_ids)  # type: ignore[attr-defined]
            audio_prefix = self.build_audio_prefix(audio_input_features, audio_attention_mask)
            input_embeds = torch.cat([audio_prefix.to(text_embeds.dtype), text_embeds], dim=1)

            prefix_len = audio_prefix.shape[1]
            prefix_ids = torch.full(
                (input_ids.size(0), prefix_len),
                fill_value=self.config.pad_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            model_input_ids = torch.cat([prefix_ids, input_ids], dim=1)

            if labels is not None:
                prefix_labels = torch.full(
                    (labels.size(0), prefix_len),
                    fill_value=-100,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                model_labels = torch.cat([prefix_labels, labels], dim=1)

            if attention_mask is not None:
                prefix_mask = torch.ones(
                    (attention_mask.size(0), prefix_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                model_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        return super().forward(
            input_ids=model_input_ids,
            input_embeds=input_embeds,
            input_states=input_states,
            attention_mask=model_attention_mask,
            position_ids=position_ids,
            labels=model_labels,
            num_steps=num_steps,
            past_key_values=past_key_values,
            output_details=output_details,
            use_cache=use_cache,
            cache_position=cache_position,
            init_scale=init_scale,
            audio_input_features=audio_input_features,
            audio_attention_mask=audio_attention_mask,
            **kwargs,
        )
