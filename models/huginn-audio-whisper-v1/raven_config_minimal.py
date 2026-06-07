"""Configuration for the Huginn audio experiment branch."""

from ._base import RavenConfig


class HuginnAudioConfig(RavenConfig):
    model_type = "huginn_audio_raven_whisper_v1"

    def __init__(
        self,
        audio_encoder_name: str = "/hpc_stor03/sjtu_home/jinwei.zhang/models/whisper-small",
        audio_encoder_hidden_size: int = 768,
        audio_pooling_type: str = "conv1d_avg",
        audio_target_token_count: int = 32,
        audio_projector_hidden_size: int = 2048,
        freeze_audio_encoder: bool = True,
        freeze_text_backbone: bool = True,
        use_audio_boundary_embeddings: bool = True,
        **kwargs,
    ):
        self.audio_encoder_name = audio_encoder_name
        self.audio_encoder_hidden_size = audio_encoder_hidden_size
        self.audio_pooling_type = audio_pooling_type
        self.audio_target_token_count = audio_target_token_count
        self.audio_projector_hidden_size = audio_projector_hidden_size
        self.freeze_audio_encoder = freeze_audio_encoder
        self.freeze_text_backbone = freeze_text_backbone
        self.use_audio_boundary_embeddings = use_audio_boundary_embeddings
        super().__init__(**kwargs)
