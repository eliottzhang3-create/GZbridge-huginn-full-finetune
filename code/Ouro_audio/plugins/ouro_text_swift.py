"""Register the text-only Ouro-1.4B baseline in ms-swift 4.4.2.

This plugin intentionally does not add Whisper, an aligner, or multimodal
inputs. It is the first registration gate before the Ouro audio wrapper is
introduced.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch

from swift.model import Model, ModelGroup, ModelLoader, ModelMeta, register_model
from swift.template import TemplateMeta, register_template


MODEL_TYPE = "ouro_text_native"
TEMPLATE_TYPE = "ouro_text_direct"
DEFAULT_MODEL_DIR = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/Ouro-1.4B")
MODEL_DIR = Path(os.environ.get("OURO_MODEL_PATH", str(DEFAULT_MODEL_DIR))).expanduser()
PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))


class OuroTextLoader(ModelLoader):
    """Load Ouro through ms-swift and install its cache compatibility patch.

    The patch must be installed immediately after ``from_pretrained`` and
    before any Swift inference/evaluation path calls the model with
    ``use_cache=True``. Keeping it in the loader also makes the normal
    ``swift infer`` and future ``swift sft`` entry points consistent with the
    dedicated audit scripts.
    """

    def get_model(self, model_dir: str, config: Any, processor: Any, model_kwargs: dict):
        model = super().get_model(model_dir, config, processor, model_kwargs)
        from compat.ouro_cache import patch_ouro_cache

        patch_ouro_cache(model)
        return model


def _register_template() -> None:
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
            loader=OuroTextLoader,
            template=TEMPLATE_TYPE,
            architectures=["OuroForCausalLM"],
            torch_dtype=torch.bfloat16,
            is_multimodal=False,
            requires=["transformers==4.54.1"],
            tags=["ouro", "text", "looped-language-model"],
        ),
        exist_ok=True,
    )


_register_template()
_register_model()

print(
    f"[OuroTextSwift] registered model_type={MODEL_TYPE} "
    f"template={TEMPLATE_TYPE} model_path={MODEL_DIR}",
    flush=True,
)
