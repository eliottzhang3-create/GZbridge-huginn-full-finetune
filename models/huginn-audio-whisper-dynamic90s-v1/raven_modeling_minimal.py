"""Isolated Whisper-large dynamic-90s wrapper around Huginn-0125."""

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
        kernel_size: int = 6,
        stride: int = 6,
    ):
        super().__init__()
        if kernel_size <= 0 or stride <= 0:
            raise ValueError(f"kernel_size and stride must be positive, got {kernel_size=} {stride=}")

        self.kernel_size = kernel_size
        self.stride = stride
        self.downsample = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected compressor input with shape [batch, time, hidden], got {tuple(x.shape)}")
        if x.size(1) < self.kernel_size:
            raise ValueError(
                "Audio encoder sequence is shorter than the compressor kernel: "
                f"time={x.size(1)} kernel_size={self.kernel_size}"
            )

        # Whisper-large emits approximately 20 ms per encoder frame.  A
        # kernel/stride of 6 therefore produces one audio token per 120 ms.
        x = x.transpose(1, 2)
        x = self.downsample(x)
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
            self.audio_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_audio_encoder:
            # ``Module.train`` recursively switches the frozen Whisper encoder
            # back to train mode. Keep its dropout/statistics behavior frozen.
            self.audio_encoder.eval()
        return self

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
        audio_segment_feature_lengths: Optional[torch.Tensor] = None,
        audio_segment_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del audio_attention_mask

        if audio_input_features.ndim == 3:
            # Backward-compatible single-segment input: [B, 80, 3000].
            audio_input_features = audio_input_features.unsqueeze(1)
        elif audio_input_features.ndim != 4:
            raise ValueError(
                "audio_input_features must have shape [B, 80, frames] or "
                f"[B, segments, 80, frames], got {tuple(audio_input_features.shape)}"
            )

        batch_size, segment_count, _, feature_frame_count = audio_input_features.shape
        if audio_segment_feature_lengths is None:
            audio_segment_feature_lengths = torch.full(
                (batch_size, segment_count),
                fill_value=feature_frame_count,
                dtype=torch.long,
                device=audio_input_features.device,
            )
        else:
            if tuple(audio_segment_feature_lengths.shape) != (batch_size, segment_count):
                raise ValueError(
                    "audio_segment_feature_lengths must have shape "
                    f"[{batch_size}, {segment_count}], got {tuple(audio_segment_feature_lengths.shape)}"
                )
            audio_segment_feature_lengths = audio_segment_feature_lengths.to(
                device=audio_input_features.device,
                dtype=torch.long,
            )

        if audio_segment_mask is None:
            audio_segment_mask = audio_segment_feature_lengths.gt(0)
        else:
            if tuple(audio_segment_mask.shape) != (batch_size, segment_count):
                raise ValueError(
                    "audio_segment_mask must have shape "
                    f"[{batch_size}, {segment_count}], got {tuple(audio_segment_mask.shape)}"
                )
            audio_segment_mask = audio_segment_mask.to(
                device=audio_input_features.device,
                dtype=torch.bool,
            )

        audio_segment_feature_lengths = audio_segment_feature_lengths.clamp(
            min=0,
            max=feature_frame_count,
        )
        audio_segment_mask = audio_segment_mask & audio_segment_feature_lengths.gt(0)

        per_sample_audio_tokens: list[torch.Tensor] = []
        audio_encoder_parameter = next(self.audio_encoder.parameters())
        audio_encoder_dtype = audio_encoder_parameter.dtype
        audio_encoder_device = audio_encoder_parameter.device
        aligner_dtype = next(self.temporal_compressor.parameters()).dtype
        compressor_kernel = int(self.temporal_compressor.kernel_size)
        max_audio_token_count = int(getattr(self.config, "audio_max_token_count", 750))

        for sample_index in range(batch_size):
            sample_tokens: list[torch.Tensor] = []
            for segment_index in range(segment_count):
                if not bool(audio_segment_mask[sample_index, segment_index].item()):
                    continue

                valid_feature_frames = int(audio_segment_feature_lengths[sample_index, segment_index].item())
                if valid_feature_frames <= 0:
                    continue

                # Keep Whisper's native 3000-frame input shape. The feature
                # extractor pads every segment to that shape; the attention
                # mask prevents padded mel frames from participating in the
                # encoder, and the valid encoder length below removes padded
                # outputs from the compressor.
                segment_features = audio_input_features[
                    sample_index,
                    segment_index,
                    :,
                ].unsqueeze(0).to(
                    device=audio_encoder_device,
                    dtype=audio_encoder_dtype,
                )
                feature_mask = torch.zeros(
                    (1, segment_features.size(-1)),
                    dtype=torch.long,
                    device=audio_encoder_device,
                )
                feature_mask[:, :valid_feature_frames] = 1

                with torch.no_grad():
                    encoder_outputs = self.audio_encoder(
                        input_features=segment_features,
                        attention_mask=feature_mask,
                        return_dict=True,
                    )
                audio_hidden = encoder_outputs.last_hidden_state
                # Keep only complete 20 ms encoder frames so the stride-6
                # compressor emits complete 120 ms tokens. Any residual tail
                # below 120 ms is intentionally dropped.
                valid_encoder_frames = min(audio_hidden.size(1), valid_feature_frames // 2)
                audio_hidden = audio_hidden[:, :valid_encoder_frames]
                if audio_hidden.size(1) < compressor_kernel:
                    continue

                audio_hidden = audio_hidden.to(dtype=aligner_dtype)
                compressed = self.temporal_compressor(audio_hidden)
                projected = self.audio_projector(compressed).squeeze(0)
                sample_tokens.append(projected)

            if sample_tokens:
                audio_tokens = torch.cat(sample_tokens, dim=0)
            else:
                audio_tokens = torch.empty(
                    (0, self.config.n_embd),
                    device=audio_input_features.device,
                    dtype=aligner_dtype,
                )

            if audio_tokens.size(0) > max_audio_token_count:
                raise ValueError(
                    "Audio prefix exceeds configured maximum token count: "
                    f"tokens={audio_tokens.size(0)} max={max_audio_token_count}"
                )
            per_sample_audio_tokens.append(audio_tokens)

        boundary_tokens = int(self.audio_bos is not None) + int(self.audio_eos is not None)
        max_prefix_length = max((tokens.size(0) + boundary_tokens for tokens in per_sample_audio_tokens), default=0)
        prefix = audio_input_features.new_zeros(
            (batch_size, max_prefix_length, self.config.n_embd),
            dtype=aligner_dtype,
        )
        prefix_mask = torch.zeros(
            (batch_size, max_prefix_length),
            dtype=torch.bool,
            device=audio_input_features.device,
        )

        for sample_index, audio_tokens in enumerate(per_sample_audio_tokens):
            chunks = []
            if self.audio_bos is not None:
                chunks.append(self.audio_bos.expand(1, -1, -1).squeeze(0).to(dtype=aligner_dtype))
            chunks.append(audio_tokens)
            if self.audio_eos is not None:
                chunks.append(self.audio_eos.expand(1, -1, -1).squeeze(0).to(dtype=aligner_dtype))
            sample_prefix = torch.cat(chunks, dim=0) if chunks else audio_tokens
            prefix_length = sample_prefix.size(0)
            if prefix_length:
                prefix[sample_index, :prefix_length] = sample_prefix
                prefix_mask[sample_index, :prefix_length] = True

        return prefix, prefix_mask

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
        audio_segment_feature_lengths: Optional[torch.Tensor] = None,
        audio_segment_mask: Optional[torch.Tensor] = None,
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
        if audio_segment_feature_lengths is not None:
            model_inputs["audio_segment_feature_lengths"] = audio_segment_feature_lengths
        if audio_segment_mask is not None:
            model_inputs["audio_segment_mask"] = audio_segment_mask
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
        audio_segment_feature_lengths: Optional[torch.Tensor] = None,
        audio_segment_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputRecurrentLatents:
        model_input_ids = input_ids
        model_labels = labels
        model_attention_mask = attention_mask
        self._last_audio_prefix_mask = None

        if audio_input_features is not None and past_key_values is None:
            if input_embeds is not None:
                raise ValueError("Pass either input_embeds or audio_input_features, not both.")

            text_embeds = self.transformer.wte(input_ids)  # type: ignore[attr-defined]
            audio_prefix, prefix_mask = self.build_audio_prefix(
                audio_input_features,
                audio_attention_mask,
                audio_segment_feature_lengths,
                audio_segment_mask,
            )
            self._last_audio_prefix_mask = prefix_mask
            input_embeds = torch.cat([audio_prefix.to(text_embeds.dtype), text_embeds], dim=1)

            prefix_len = audio_prefix.shape[1]
            prefix_ids = torch.full(
                (input_ids.size(0), prefix_len),
                fill_value=self.config.pad_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            # Huginn's compiled mask treats pad_token_id as invalid even when
            # input_embeds are supplied. Valid audio positions therefore need
            # a non-pad placeholder id; only batch padding stays as pad.
            prefix_ids[prefix_mask.to(device=input_ids.device)] = self.config.bos_token_id
            model_input_ids = torch.cat([prefix_ids, input_ids], dim=1)

            if labels is not None:
                prefix_labels = torch.full(
                    (labels.size(0), prefix_len),
                    fill_value=-100,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                text_labels = labels.clone()
                has_audio_padding = prefix_mask.sum(dim=1).lt(prefix_len)
                if text_labels.size(1) > 0 and bool(has_audio_padding.any().item()):
                    # The first text target after a padded audio prefix would
                    # otherwise be predicted from a masked padding position.
                    # Omit only that invalid transition; later text targets
                    # remain normally supervised.
                    text_labels[has_audio_padding.to(device=text_labels.device), 0] = -100
                model_labels = torch.cat([prefix_labels, text_labels], dim=1)

            if attention_mask is not None:
                model_attention_mask = torch.cat(
                    [prefix_mask.to(device=attention_mask.device, dtype=attention_mask.dtype), attention_mask],
                    dim=1,
                )
                self._last_audio_combined_attention_mask = model_attention_mask.detach()

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
