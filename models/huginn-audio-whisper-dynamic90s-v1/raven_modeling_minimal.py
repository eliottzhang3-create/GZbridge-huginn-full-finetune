"""Isolated Whisper-large dynamic-90s wrapper around Huginn-0125."""

from __future__ import annotations

import gc
from contextlib import nullcontext
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

    def forward(self, _reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # PEFT's AuxiliaryTrainingWrapper requires one positional input even
        # though these learned boundary embeddings are input-independent.
        return self.audio_bos * 1.0, self.audio_eos * 1.0


class WhisperEncoderFSDPUnit(nn.Module):
    """One callable FSDP unit containing the complete Whisper encoder."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    @property
    def gradient_checkpointing(self) -> bool:
        return bool(getattr(self.encoder, "gradient_checkpointing", False))

    @property
    def is_gradient_checkpointing(self) -> bool:
        return bool(getattr(self.encoder, "is_gradient_checkpointing", self.gradient_checkpointing))

    def gradient_checkpointing_enable(self, *args, **kwargs):
        return self.encoder.gradient_checkpointing_enable(*args, **kwargs)

    def gradient_checkpointing_disable(self):
        return self.encoder.gradient_checkpointing_disable()

    def enable_input_require_grads(self):
        # Whisper is full-tuned and receives floating log-mel tensors rather
        # than frozen embedding outputs. Non-reentrant checkpointing does not
        # require those inputs to have requires_grad=True. This method exists
        # so Swift reaches gradient_checkpointing_enable on the wrapped tower.
        return None

    def disable_input_require_grads(self):
        return None

    def forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)


class AudioAlignerFSDPUnit(nn.Module):
    """One callable FSDP unit containing every trainable audio aligner tensor."""

    def __init__(self, config: HuginnAudioConfig):
        super().__init__()
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

    def forward(
        self,
        audio_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        projected = self.audio_projector(self.temporal_compressor(audio_hidden))
        if self.audio_boundary_embeddings is None:
            return projected, None, None

        # Calling the module is required for PEFT's ModulesToSaveWrapper to
        # route execution through its active trainable copy. It also keeps both
        # parameters materialized inside this FSDP unit's wrapped forward.
        audio_bos, audio_eos = self.audio_boundary_embeddings(audio_hidden)
        return projected, audio_bos, audio_eos


class _HuginnBlockFSDPUnit(nn.Module):
    """Store blocks under their historical numeric state-dict paths."""

    def __init__(self, blocks: list[nn.Module]):
        super().__init__()
        if not blocks:
            raise ValueError(f"{type(self).__name__} requires at least one SandwichBlock")
        self._block_count = len(blocks)
        for index, block in enumerate(blocks):
            self.add_module(str(index), block)

    def __len__(self) -> int:
        return self._block_count

    def __getitem__(self, index: int) -> nn.Module:
        if index < 0:
            index += self._block_count
        if index < 0 or index >= self._block_count:
            raise IndexError(index)
        return self._modules[str(index)]

    def _blocks(self):
        for index in range(self._block_count):
            yield self[index]


class HuginnPreludeFSDPUnit(_HuginnBlockFSDPUnit):
    """Execute both prelude SandwichBlocks through one FSDP forward."""

    def __iter__(self):
        # Raven's inherited forward iterates ``transformer.prelude``. Yielding
        # this callable container once ensures its FSDP hooks are not bypassed.
        yield self

    def forward(self, x, freqs_cis, block_idx, mask=None, past_key_values=None):
        initial_block_idx = block_idx
        for offset, block in enumerate(self._blocks()):
            x = block(x, freqs_cis, initial_block_idx + offset, mask, past_key_values)
        if self._block_count > 1:
            # The inherited Raven loop owns block_idx. Update its CPU scalar
            # only after all block calls so the next recurrent index is exact.
            block_idx.add_(self._block_count - 1)
        return x


class HuginnCodaFSDPUnit(_HuginnBlockFSDPUnit):
    """Execute both coda SandwichBlocks through one FSDP forward."""

    def __iter__(self):
        yield self

    def forward(self, x, freqs_cis, block_idx, mask=None, past_key_values=None):
        initial_block_idx = block_idx
        for offset, block in enumerate(self._blocks()):
            x = block(x, freqs_cis, initial_block_idx - offset, mask, past_key_values)
        if self._block_count > 1:
            block_idx.sub_(self._block_count - 1)
        return x


class HuginnRecurrentCoreFSDPUnit(_HuginnBlockFSDPUnit):
    """One recurrent unit: native adapter plus all four reused core blocks."""

    def __init__(self, adapter: nn.Module, blocks: list[nn.Module]):
        super().__init__(blocks)
        self.adapter = adapter

    def __iter__(self):
        # Compatibility for code that inspects the physical recurrent blocks.
        return self._blocks()

    @property
    def norm_4(self):
        return self[-1].norm_4

    def forward(self, x, input_embeds, freqs_cis, block_idx, mask=None, past_key_values=None):
        adapter_in = torch.cat([x, input_embeds.to(x.device)], dim=-1)
        x = self.adapter(adapter_in)
        for block in self._blocks():
            block_idx = block_idx + 1
            x = block(x, freqs_cis, block_idx, mask, past_key_values)
        return x, block_idx


class HuginnAudioForConditionalGeneration(RavenForCausalLM):
    config_class = HuginnAudioConfig
    _no_split_modules = [
        "WhisperEncoderFSDPUnit",
        "AudioAlignerFSDPUnit",
        "HuginnPreludeFSDPUnit",
        "HuginnRecurrentCoreFSDPUnit",
        "HuginnCodaFSDPUnit",
    ]

    def __init__(self, config: HuginnAudioConfig):
        super().__init__(config)
        self.config = config

        prelude_blocks = list(self.transformer.prelude)
        recurrent_blocks = list(self.transformer.core_block)
        coda_blocks = list(self.transformer.coda)
        recurrent_adapter = self.transformer.adapter
        del self.transformer["adapter"]
        self.transformer["prelude"] = HuginnPreludeFSDPUnit(prelude_blocks)
        self.transformer["core_block"] = HuginnRecurrentCoreFSDPUnit(
            recurrent_adapter,
            recurrent_blocks,
        )
        self.transformer["coda"] = HuginnCodaFSDPUnit(coda_blocks)

        whisper = WhisperModel.from_pretrained(config.audio_encoder_name)
        self.audio_encoder = WhisperEncoderFSDPUnit(whisper.encoder)
        del whisper

        self.audio_aligner = AudioAlignerFSDPUnit(config)

        self._freeze_requested_modules()

    @property
    def temporal_compressor(self):
        return self.audio_aligner.temporal_compressor

    @property
    def audio_projector(self):
        return self.audio_aligner.audio_projector

    @property
    def audio_boundary_embeddings(self):
        return self.audio_aligner.audio_boundary_embeddings

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
        # Accept legacy root-level and pre-grouping aligner checkpoint paths.
        for name in ("audio_bos", "audio_eos"):
            current_key = f"{prefix}audio_aligner.audio_boundary_embeddings.{name}"
            for legacy_key in (
                f"{prefix}{name}",
                f"{prefix}audio_boundary_embeddings.{name}",
            ):
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
        # The recurrent adapter now lives inside the single recurrent FSDP
        # unit. All physical block keys deliberately retain their old paths.
        for key in list(backbone_state):
            if key.startswith("transformer.adapter."):
                new_key = key.replace(
                    "transformer.adapter.",
                    "transformer.core_block.adapter.",
                    1,
                )
                backbone_state[new_key] = backbone_state.pop(key)
        incompatible = self.load_state_dict(backbone_state, strict=False)
        del base_model
        gc.collect()
        self._freeze_requested_modules()
        return incompatible

    def core_block_forward(
        self,
        x,
        input_embeds,
        freqs_cis,
        mask,
        past_key_values,
        block_idx: torch.Tensor,
        current_step,
    ):
        """Run adapter + four physical recurrent blocks as one callable unit."""
        block_idx = block_idx.detach().clone()
        # Legacy activation-stat logging is intentionally disabled. Keep the
        # recurrent computation and noise path unchanged.
        # self._debug_activation_stats("core_in", x, current_step)
        x = self._maybe_inject_noise(x, current_step)
        # self._debug_activation_stats("after_noise", x, current_step)
        x, block_idx = self.transformer.core_block(
            x,
            input_embeds,
            freqs_cis,
            block_idx,
            mask,
            past_key_values,
        )
        # self._debug_activation_stats("core_unit_out", x, current_step)
        return x, block_idx

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

        audio_encoder_parameter = next(self.audio_encoder.parameters())
        audio_encoder_dtype = audio_encoder_parameter.dtype
        audio_encoder_device = audio_encoder_parameter.device
        aligner_parameter = next(self.audio_aligner.parameters())
        aligner_dtype = aligner_parameter.dtype
        aligner_device = aligner_parameter.device
        compressor_kernel = int(self.temporal_compressor.kernel_size)
        compressor_stride = int(self.temporal_compressor.stride)
        max_audio_token_count = int(getattr(self.config, "audio_max_token_count", 750))

        valid_positions = audio_segment_mask.nonzero(as_tuple=False)
        if valid_positions.size(0) == 0:
            raise ValueError("Every audio batch must contain at least one valid segment")

        # Flatten all local segments so each large FSDP unit is entered exactly
        # once per model forward, independent of whether samples use 1, 2, or 3
        # Whisper chunks.
        segment_features = audio_input_features[audio_segment_mask].to(
            device=audio_encoder_device,
            dtype=audio_encoder_dtype,
        )
        valid_feature_lengths = audio_segment_feature_lengths[audio_segment_mask].to(
            device=audio_encoder_device,
        )
        feature_mask = (
            torch.arange(feature_frame_count, device=audio_encoder_device).unsqueeze(0)
            < valid_feature_lengths.unsqueeze(1)
        ).to(dtype=torch.long)

        encoder_context = torch.no_grad() if self.config.freeze_audio_encoder else nullcontext()
        with encoder_context:
            encoder_outputs = self.audio_encoder(
                input_features=segment_features,
                attention_mask=feature_mask,
                return_dict=True,
            )
        audio_hidden = encoder_outputs.last_hidden_state.to(
            device=aligner_device,
            dtype=aligner_dtype,
        )
        projected_segments, audio_bos, audio_eos = self.audio_aligner(audio_hidden)

        per_sample_segments: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
        for flat_index, position in enumerate(valid_positions.tolist()):
            sample_index, _segment_index = position
            valid_feature_frames = int(valid_feature_lengths[flat_index].item())
            valid_encoder_frames = min(audio_hidden.size(1), valid_feature_frames // 2)
            if valid_encoder_frames < compressor_kernel:
                token_count = 0
            else:
                token_count = (valid_encoder_frames - compressor_kernel) // compressor_stride + 1
            if token_count > 0:
                per_sample_segments[sample_index].append(projected_segments[flat_index, :token_count])

        per_sample_audio_tokens: list[torch.Tensor] = []
        for sample_segments in per_sample_segments:
            if sample_segments:
                audio_tokens = torch.cat(sample_segments, dim=0)
            else:
                audio_tokens = projected_segments.new_empty((0, self.config.n_embd))
            if audio_tokens.size(0) > max_audio_token_count:
                raise ValueError(
                    "Audio prefix exceeds configured maximum token count: "
                    f"tokens={audio_tokens.size(0)} max={max_audio_token_count}"
                )
            per_sample_audio_tokens.append(audio_tokens)

        boundary_tokens = int(audio_bos is not None) + int(audio_eos is not None)
        max_prefix_length = max((tokens.size(0) + boundary_tokens for tokens in per_sample_audio_tokens), default=0)
        prefix = projected_segments.new_zeros((batch_size, max_prefix_length, self.config.n_embd))
        prefix_mask = torch.zeros(
            (batch_size, max_prefix_length),
            dtype=torch.bool,
            device=projected_segments.device,
        )

        for sample_index, audio_tokens in enumerate(per_sample_audio_tokens):
            chunks = []
            if audio_bos is not None:
                chunks.append(audio_bos.squeeze(0))
            chunks.append(audio_tokens)
            if audio_eos is not None:
                chunks.append(audio_eos.squeeze(0))
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
