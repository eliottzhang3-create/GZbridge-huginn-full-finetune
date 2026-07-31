"""Configuration for the isolated Whisper-large dynamic single-chunk branch."""

from ._base import RavenConfig


class HuginnAudioConfig(RavenConfig):
    model_type = "huginn_audio_raven_whisper_dynamic90s_v1"

    def __init__(
        self,
        audio_encoder_name: str = "/hpc_stor03/sjtu_home/jinwei.zhang/models/whisper-large",
        audio_encoder_hidden_size: int = 1280,
        audio_pooling_type: str = "conv1d_stride12_dynamic30s",
        audio_dynamic_tokens: bool = True,
        audio_token_duration_ms: int = 240,
        audio_reference_30s_token_count: int = 125,
        audio_max_token_count: int = 125,
        audio_chunk_seconds: float = 30.0,
        audio_max_seconds: float = 30.0,
        audio_feature_hop_length: int = 160,
        audio_encoder_frame_rate: int = 50,
        audio_compressor_kernel_size: int = 12,
        audio_compressor_stride: int = 12,
        audio_projector_hidden_size: int = 2048,
        freeze_audio_encoder: bool = False,
        freeze_text_backbone: bool = True,
        use_audio_boundary_embeddings: bool = True,
        **kwargs,
    ):
        self.audio_encoder_name = audio_encoder_name
        self.audio_encoder_hidden_size = audio_encoder_hidden_size
        self.audio_pooling_type = audio_pooling_type
        self.audio_dynamic_tokens = audio_dynamic_tokens
        self.audio_token_duration_ms = audio_token_duration_ms
        self.audio_reference_30s_token_count = audio_reference_30s_token_count
        self.audio_max_token_count = audio_max_token_count
        self.audio_chunk_seconds = audio_chunk_seconds
        self.audio_max_seconds = audio_max_seconds
        self.audio_feature_hop_length = audio_feature_hop_length
        self.audio_encoder_frame_rate = audio_encoder_frame_rate
        self.audio_compressor_kernel_size = audio_compressor_kernel_size
        self.audio_compressor_stride = audio_compressor_stride
        self.audio_projector_hidden_size = audio_projector_hidden_size
        self.freeze_audio_encoder = freeze_audio_encoder
        self.freeze_text_backbone = freeze_text_backbone
        self.use_audio_boundary_embeddings = use_audio_boundary_embeddings
        super().__init__(**kwargs)
