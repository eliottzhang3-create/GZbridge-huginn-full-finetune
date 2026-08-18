"""Register Qwen3-4B-Base as an isolated text model in ms-swift.

This is intentionally separate from the Ouro registrations. Qwen3 is
supported natively by Transformers and does not need the Ouro cache patch or
trust_remote_code model files.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from swift.model import Model, ModelGroup, ModelLoader, ModelMeta, register_model
from swift.template import TemplateMeta, register_template


MODEL_TYPE = "qwen3_text_base"
TEMPLATE_TYPE = "qwen3_text_direct"
DEFAULT_MODEL_DIR = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/Qwen3-4B-Base")
MODEL_DIR = Path(os.environ.get("QWEN3_MODEL_PATH", str(DEFAULT_MODEL_DIR))).expanduser()


class Qwen3TextLoader(ModelLoader):
    """Keep an explicit loader class for stable plugin identity and auditing."""


def _register_template() -> None:
    # The first registration gate deliberately uses a direct causal-LM prompt
    # instead of an Instruct/chat template.
    register_template(
        TemplateMeta(
            template_type=TEMPLATE_TYPE,
            prefix=[],
            prompt=["{{QUERY}}"],
            chat_sep=None,
            suffix=[["eos_token_id"]],
            auto_add_bos=False,
            stop_words=[],
        ),
        exist_ok=True,
    )


def _register_model() -> None:
    register_model(
        ModelMeta(
            model_type=MODEL_TYPE,
            model_groups=[ModelGroup(models=[Model(model_path=str(MODEL_DIR))])],
            loader=Qwen3TextLoader,
            template=TEMPLATE_TYPE,
            architectures=["Qwen3ForCausalLM"],
            torch_dtype=torch.bfloat16,
            is_multimodal=False,
            requires=["transformers>=4.51.0"],
            tags=["qwen3", "qwen3-4b-base", "text", "baseline"],
        ),
        exist_ok=True,
    )


_register_template()
_register_model()

print(
    f"[Qwen3TextSwift] registered model_type={MODEL_TYPE} "
    f"template={TEMPLATE_TYPE} model_path={MODEL_DIR}",
    flush=True,
)
