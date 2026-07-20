"""Configuration for the frozen-LoSATok Huginn audio route."""

from ._base import RavenConfig


class HuginnLoSATokConfig(RavenConfig):
    model_type = "huginn_audio_raven_losatok_v1"

    def __init__(
        self,
        losatok_root: str = "/hpc_stor03/sjtu_home/jinwei.zhang/models/LoSATok",
        losatok_code_dir: str = (
            "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/"
            "code/huginn_lora/LosatokCode"
        ),
        losatok_checkpoint_name: str = "losatok_kl1e-3.pth",
        losatok_output_key: str = "unified_emb",
        audio_encoder_hidden_size: int = 1280,
        audio_sample_rate: int = 16000,
        audio_target_token_count: int = 32,
        audio_compressor_intermediate_size: int = 1536,
        audio_compressor_kernel_size: int = 7,
        audio_compressor_stride: int = 4,
        audio_projector_hidden_size: int = 2048,
        freeze_audio_encoder: bool = True,
        freeze_text_backbone: bool = True,
        use_audio_boundary_embeddings: bool = True,
        **kwargs,
    ):
        self.losatok_root = losatok_root
        self.losatok_code_dir = losatok_code_dir
        self.losatok_checkpoint_name = losatok_checkpoint_name
        self.losatok_output_key = losatok_output_key
        self.audio_encoder_hidden_size = audio_encoder_hidden_size
        self.audio_sample_rate = audio_sample_rate
        self.audio_target_token_count = audio_target_token_count
        self.audio_compressor_intermediate_size = audio_compressor_intermediate_size
        self.audio_compressor_kernel_size = audio_compressor_kernel_size
        self.audio_compressor_stride = audio_compressor_stride
        self.audio_projector_hidden_size = audio_projector_hidden_size
        self.freeze_audio_encoder = freeze_audio_encoder
        self.freeze_text_backbone = freeze_text_backbone
        self.use_audio_boundary_embeddings = use_audio_boundary_embeddings
        super().__init__(**kwargs)
