"""Configuration for the independent HRM-Text Whisper audio experiment."""

from __future__ import annotations

from transformers import HrmTextConfig


class HrmTextAudioConfig(HrmTextConfig):
    model_type = "hrm_text_audio_whisper_v1"

    def __init__(
        self,
        base_model_name_or_path: str = "/hpc_stor03/sjtu_home/jinwei.zhang/models/HRM-text",
        audio_encoder_name: str = "/hpc_stor03/sjtu_home/jinwei.zhang/models/whisper-large",
        audio_encoder_hidden_size: int = 1280,
        audio_feature_size: int = 80,
        audio_sample_rate: int = 16000,
        audio_max_seconds: float = 30.0,
        audio_target_token_count: int = 32,
        audio_compressor_intermediate_size: int = 1536,
        audio_compressor_kernel_size: int = 7,
        audio_compressor_stride: int = 12,
        audio_projector_hidden_size: int = 2048,
        freeze_audio_encoder: bool = True,
        freeze_text_backbone: bool = True,
        use_audio_boundary_embeddings: bool = True,
        **kwargs,
    ):
        self.base_model_name_or_path = base_model_name_or_path
        self.audio_encoder_name = audio_encoder_name
        self.audio_encoder_hidden_size = audio_encoder_hidden_size
        self.audio_feature_size = audio_feature_size
        self.audio_sample_rate = audio_sample_rate
        self.audio_max_seconds = audio_max_seconds
        self.audio_target_token_count = audio_target_token_count
        self.audio_compressor_intermediate_size = audio_compressor_intermediate_size
        self.audio_compressor_kernel_size = audio_compressor_kernel_size
        self.audio_compressor_stride = audio_compressor_stride
        self.audio_projector_hidden_size = audio_projector_hidden_size
        self.freeze_audio_encoder = freeze_audio_encoder
        self.freeze_text_backbone = freeze_text_backbone
        self.use_audio_boundary_embeddings = use_audio_boundary_embeddings
        super().__init__(**kwargs)

        positive_int_fields = (
            "audio_encoder_hidden_size",
            "audio_feature_size",
            "audio_sample_rate",
            "audio_target_token_count",
            "audio_compressor_intermediate_size",
            "audio_compressor_kernel_size",
            "audio_compressor_stride",
            "audio_projector_hidden_size",
        )
        invalid = {name: getattr(self, name) for name in positive_int_fields if int(getattr(self, name)) <= 0}
        if invalid:
            raise ValueError(f"HRM audio config fields must be positive: {invalid}")
        if float(self.audio_max_seconds) <= 0:
            raise ValueError(f"audio_max_seconds must be positive, got {self.audio_max_seconds}")
        if int(self.audio_encoder_hidden_size) != 1280:
            raise ValueError(
                "The first HRM audio route is pinned to Whisper-large d_model=1280, got "
                f"{self.audio_encoder_hidden_size}"
            )
        if int(self.hidden_size) != 1536:
            raise ValueError(f"HRM-Text-1B hidden_size must be 1536, got {self.hidden_size}")


__all__ = ["HrmTextAudioConfig"]
