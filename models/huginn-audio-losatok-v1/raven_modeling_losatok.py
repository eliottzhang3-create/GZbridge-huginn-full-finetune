"""Huginn model that prepends frozen LoSATok unified audio embeddings."""

from __future__ import annotations

import gc
import importlib
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM

from ._base import CausalLMOutputRecurrentLatents, RavenForCausalLM
from .raven_config_losatok import HuginnLoSATokConfig


class TrainableTemporalCompressor(nn.Module):
    def __init__(self, hidden_size: int, target_token_count: int, intermediate_size: int, kernel_size: int, stride: int):
        super().__init__()
        padding = kernel_size // 2
        self.gate_proj = nn.Conv1d(hidden_size, intermediate_size, kernel_size, stride, padding)
        self.up_proj = nn.Conv1d(hidden_size, intermediate_size, kernel_size, stride, padding)
        self.down_proj = nn.Conv1d(intermediate_size, hidden_size, kernel_size=1)
        self.shortcut_pool = nn.AvgPool1d(kernel_size=stride, stride=stride, ceil_mode=True)
        self.shortcut_proj = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.output_pool = nn.AdaptiveAvgPool1d(target_token_count)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values.transpose(1, 2)
        gated = self.down_proj(torch.sigmoid(self.gate_proj(values)) * self.up_proj(values))
        shortcut = self.shortcut_proj(self.shortcut_pool(values))
        length = min(gated.size(-1), shortcut.size(-1))
        return self.output_pool(gated[..., :length] + shortcut[..., :length]).transpose(1, 2)


class AudioProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.w1 = nn.Linear(input_dim, hidden_dim)
        self.w2 = nn.Linear(input_dim, hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.input_norm(values)
        return self.output_norm(self.c_proj(self.w1(values) * F.silu(self.w2(values))))


class AudioBoundaryEmbeddings(nn.Module):
    def __init__(self, hidden_size: int, init_std: float):
        super().__init__()
        self.audio_bos = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.audio_eos = nn.Parameter(torch.empty(1, 1, hidden_size))
        nn.init.normal_(self.audio_bos, std=init_std)
        nn.init.normal_(self.audio_eos, std=init_std)


def _build_local_semantic_encoder(midasheng_dir: Path):
    semantic_bottleneck = importlib.import_module("semantic_bottleneck")

    class LocalSemanticEncoder(nn.Module):
        def __init__(self, high_dim: int = 1280, low_dim: int = 128, hidden_dim: int = 512):
            super().__init__()
            dasheng = AutoModelForCausalLM.from_pretrained(
                str(midasheng_dir), trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
            self.encoder = dasheng.audio_encoder
            del dasheng
            gc.collect()
            self.bottleneck = semantic_bottleneck.SemanticBottleneck(high_dim, low_dim, hidden_dim)
            self.encoder.requires_grad_(False)
            self.encoder.eval()

        def forward(self, waveform: torch.Tensor):
            encoder_dtype = next(self.encoder.parameters()).dtype
            with torch.no_grad():
                high = self.encoder(waveform.to(dtype=encoder_dtype))[0].detach().float()
            bottleneck_dtype = next(self.bottleneck.parameters()).dtype
            low, reconstructed = self.bottleneck(high.to(dtype=bottleneck_dtype))
            loss = semantic_bottleneck.semantic_bottleneck_loss(high, reconstructed, low)
            return high, low, reconstructed, loss

    return LocalSemanticEncoder


class FrozenLoSATokEncoder(nn.Module):
    """Loads the official LoSATok stack locally and exposes variable-length unified tokens."""

    def __init__(self, config: HuginnLoSATokConfig):
        super().__init__()
        root = Path(config.losatok_root)
        code_dir = Path(config.losatok_code_dir)
        checkpoint = root / "ckpts" / config.losatok_checkpoint_name
        semantic_checkpoint = root / "ckpts" / "semantic_encoder.pth"
        midasheng_dir = root / "midashenglm"
        yaml_path = code_dir / "config" / "16k_16k_25Hz_losatok.yml"
        for path in (code_dir, checkpoint, semantic_checkpoint, midasheng_dir, yaml_path):
            if not path.exists():
                raise FileNotFoundError(f"LoSATok required path is missing: {path}")
        if str(code_dir) not in sys.path:
            sys.path.insert(0, str(code_dir))
        yaml = importlib.import_module("yaml")
        losatok = importlib.import_module("losatok")

        losatok.SemanticEncoder = _build_local_semantic_encoder(midasheng_dir)
        yaml_config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        vae_config = {
            key.split("AudioVAE.", 1)[1]: value
            for key, value in yaml_config.items()
            if key.startswith("AudioVAE.")
        }
        vae_config["semantic_encoder_path"] = str(semantic_checkpoint)
        self.model = losatok.AudioVAE(**vae_config)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"LoSATok checkpoint mismatch: missing={missing[:10]} unexpected={unexpected[:10]}")
        self.output_key = config.losatok_output_key
        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    def _apply(self, fn):
        """Move the frozen official stack with its parent, retaining its load dtypes.

        Swift casts the complete Huginn model to BF16.  The official LoSATok load
        is intentionally mixed precision (local MiDasheng starts in BF16 while
        other modules retain their own load dtype), so only its device follows
        the parent. The trainable compressor/projector perform the later BF16
        conversion at the encoder-output boundary.
        """
        probe = fn(torch.empty(0))
        self.model.to(device=probe.device)
        return self

    @torch.no_grad()
    def forward(self, audio_input_values: torch.Tensor, audio_attention_mask: torch.Tensor) -> list[torch.Tensor]:
        if audio_input_values.ndim != 2 or audio_attention_mask.shape != audio_input_values.shape:
            raise ValueError("LoSATok audio values and attention mask must both have shape [batch, samples]")
        device = next(self.model.parameters()).device
        outputs: list[torch.Tensor] = []
        for waveform, mask in zip(audio_input_values, audio_attention_mask):
            length = int(mask.sum().item())
            if length <= 0:
                raise ValueError("LoSATok received an empty audio waveform")
            encoded = self.model.encoder_forward(waveform[:length].unsqueeze(0).to(device=device, dtype=torch.float32))
            values = encoded.get(self.output_key)
            if values is None:
                raise KeyError(f"LoSATok output key {self.output_key!r} is unavailable: {sorted(encoded)}")
            outputs.append(values.detach().float())
        return outputs


class HuginnLoSATokForConditionalGeneration(RavenForCausalLM):
    config_class = HuginnLoSATokConfig

    def __init__(self, config: HuginnLoSATokConfig):
        super().__init__(config)
        self.config = config
        self.audio_encoder = FrozenLoSATokEncoder(config)
        self.temporal_compressor = TrainableTemporalCompressor(
            config.audio_encoder_hidden_size,
            config.audio_target_token_count,
            config.audio_compressor_intermediate_size,
            config.audio_compressor_kernel_size,
            config.audio_compressor_stride,
        )
        self.audio_projector = AudioProjector(
            config.audio_encoder_hidden_size, config.audio_projector_hidden_size, config.n_embd)
        self.audio_boundary_embeddings = (
            AudioBoundaryEmbeddings(config.n_embd, config.init_values["std"])
            if config.use_audio_boundary_embeddings else None
        )
        self._freeze_requested_modules()

    @property
    def audio_bos(self):
        return None if self.audio_boundary_embeddings is None else self.audio_boundary_embeddings.audio_bos

    @property
    def audio_eos(self):
        return None if self.audio_boundary_embeddings is None else self.audio_boundary_embeddings.audio_eos

    def _freeze_requested_modules(self) -> None:
        if self.config.freeze_text_backbone:
            self.transformer.requires_grad_(False)
            self.lm_head.requires_grad_(False)
        if self.config.freeze_audio_encoder:
            self.audio_encoder.requires_grad_(False)
            self.audio_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.audio_encoder.eval()
        return self

    @torch.no_grad()
    def load_huginn_backbone_from_pretrained(self, base_model_name_or_path: str, torch_dtype: Optional[torch.dtype] = None):
        base_model = RavenForCausalLM.from_pretrained(
            base_model_name_or_path, trust_remote_code=True, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
        result = self.load_state_dict(base_model.state_dict(), strict=False)
        del base_model
        gc.collect()
        self._freeze_requested_modules()
        return result

    def build_audio_prefix(self, audio_input_values: torch.Tensor, audio_attention_mask: torch.Tensor) -> torch.Tensor:
        token_sequences = self.audio_encoder(audio_input_values, audio_attention_mask)
        aligner_dtype = next(self.temporal_compressor.parameters()).dtype
        projected = [
            self.audio_projector(self.temporal_compressor(tokens.to(dtype=aligner_dtype)))
            for tokens in token_sequences
        ]
        audio_embeds = torch.cat(projected, dim=0)
        chunks = []
        if self.audio_bos is not None:
            chunks.append(self.audio_bos.expand(audio_embeds.size(0), -1, -1))
        chunks.append(audio_embeds)
        if self.audio_eos is not None:
            chunks.append(self.audio_eos.expand(audio_embeds.size(0), -1, -1))
        return torch.cat(chunks, dim=1)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None,
                                      cache_position=None, audio_input_values=None, audio_attention_mask=None, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids, past_key_values=past_key_values, attention_mask=attention_mask,
            inputs_embeds=inputs_embeds, cache_position=cache_position, **kwargs)
        if audio_input_values is not None:
            model_inputs["audio_input_values"] = audio_input_values
        if audio_attention_mask is not None:
            model_inputs["audio_attention_mask"] = audio_attention_mask
        return model_inputs

    def forward(self, input_ids: torch.Tensor, input_embeds: Optional[torch.Tensor] = None,
                input_states: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None,
                num_steps: Optional[torch.Tensor] = None, past_key_values=None, output_details: dict | None = None,
                use_cache: bool = False, cache_position: Optional[torch.Tensor] = None, init_scale: float = 1.0,
                audio_input_values: Optional[torch.Tensor] = None, audio_attention_mask: Optional[torch.Tensor] = None,
                **kwargs) -> CausalLMOutputRecurrentLatents:
        model_ids, model_labels, model_mask = input_ids, labels, attention_mask
        if audio_input_values is not None and past_key_values is None:
            if input_embeds is not None or audio_attention_mask is None:
                raise ValueError("LoSATok prefill requires audio values/mask and no precomputed input_embeds")
            text_embeds = self.transformer.wte(input_ids)
            prefix = self.build_audio_prefix(audio_input_values, audio_attention_mask)
            input_embeds = torch.cat([prefix.to(text_embeds.dtype), text_embeds], dim=1)
            prefix_length = prefix.size(1)
            model_ids = torch.cat([
                torch.full((input_ids.size(0), prefix_length), self.config.pad_token_id, dtype=input_ids.dtype, device=input_ids.device),
                input_ids,
            ], dim=1)
            if labels is not None:
                model_labels = torch.cat([
                    torch.full((labels.size(0), prefix_length), -100, dtype=labels.dtype, device=labels.device), labels,
                ], dim=1)
            if attention_mask is not None:
                model_mask = torch.cat([
                    torch.ones((attention_mask.size(0), prefix_length), dtype=attention_mask.dtype, device=attention_mask.device),
                    attention_mask,
                ], dim=1)
        if output_details is None:
            output_details = {
                "return_logits": True,
                "return_latents": True,
                "return_head": False,
                "return_stats": False,
            }
        return super().forward(
            input_ids=model_ids, input_embeds=input_embeds, input_states=input_states, attention_mask=model_mask,
            position_ids=position_ids, labels=model_labels, num_steps=num_steps, past_key_values=past_key_values,
            output_details=output_details, use_cache=use_cache, cache_position=cache_position, init_scale=init_scale, **kwargs)
