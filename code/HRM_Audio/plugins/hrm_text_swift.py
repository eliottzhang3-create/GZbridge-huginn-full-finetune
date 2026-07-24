"""Register native HRM-Text and PrefixLM-aware text templates in ms-swift 4.4.2."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from swift.model import Model, ModelGroup, ModelMeta, register_model
from swift.template import Template, TemplateMeta, register_template


MODEL_TYPE = "hrm_text_native"
DIRECT_TEMPLATE_TYPE = "hrm_text_direct"
SYNTH_COT_TEMPLATE_TYPE = "hrm_text_synth_cot"
DEFAULT_MODEL_DIR = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/HRM-text")
MODEL_DIR = Path(os.environ.get("HRM_TEXT_MODEL_PATH", str(DEFAULT_MODEL_DIR))).expanduser()

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
DIRECT_CONDITION = "<|object_ref_start|>"
SYNTH_COT_CONDITION = "<|quad_end|><|object_ref_end|>"


def _to_int_list(value: Any, *, name: str) -> list[int]:
    if torch.is_tensor(value):
        if value.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional before collation, got shape={tuple(value.shape)}")
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list/tuple/tensor, got {type(value)}")
    return [int(item) for item in value]


class HrmTextPrefixLMTemplate(Template):
    """Add the prompt/response PrefixLM boundary expected by HRM-Text."""

    support_padding_free = False

    def _encode(self, inputs):
        encoded = super()._encode(inputs)
        input_ids = _to_int_list(encoded.get("input_ids"), name="input_ids")
        labels_value = encoded.get("labels")

        if labels_value is None:
            # Inference prefill: the complete prompt is one bidirectional block.
            prefix_length = len(input_ids)
        else:
            labels = _to_int_list(labels_value, name="labels")
            if len(labels) != len(input_ids):
                raise RuntimeError(
                    f"HRM template input/label length mismatch: input_ids={len(input_ids)} labels={len(labels)}"
                )
            supervised_positions = [index for index, label in enumerate(labels) if label != -100]
            # Query-only encoding may still contain an all--100 labels list in
            # some Swift modes. It has the same mask semantics as inference.
            prefix_length = supervised_positions[0] if supervised_positions else len(input_ids)
            if any(label != -100 for label in labels[:prefix_length]):
                raise RuntimeError("HRM prompt labels must all be -100 before the first response token")

        if prefix_length <= 0 or prefix_length > len(input_ids):
            raise RuntimeError(
                f"Invalid HRM PrefixLM boundary: prefix_length={prefix_length} sequence_length={len(input_ids)}"
            )

        # Prefix tokens (including <|im_end|>) are bidirectional. Response and
        # EOS tokens are causal. Later ignored response labels, if any, remain
        # causal because the mask uses one boundary rather than labels per token.
        encoded["token_type_ids"] = [1] * prefix_length + [0] * (len(input_ids) - prefix_length)
        return encoded


def _register_templates() -> None:
    register_template(
        TemplateMeta(
            template_type=DIRECT_TEMPLATE_TYPE,
            template_cls=HrmTextPrefixLMTemplate,
            prefix=[],
            prompt=[f"{IM_START}{DIRECT_CONDITION}{{{{QUERY}}}}{IM_END}"],
            chat_sep=None,
            suffix=[["eos_token_id"]],
            auto_add_bos=False,
            stop_words=[],
        ),
        exist_ok=True,
    )
    register_template(
        TemplateMeta(
            template_type=SYNTH_COT_TEMPLATE_TYPE,
            template_cls=HrmTextPrefixLMTemplate,
            prefix=[],
            prompt=[f"{IM_START}{SYNTH_COT_CONDITION}{{{{QUERY}}}}{IM_END}"],
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
            template=DIRECT_TEMPLATE_TYPE,
            architectures=["HrmTextForCausalLM"],
            torch_dtype=torch.bfloat16,
            is_multimodal=False,
            requires=["transformers==5.9.0"],
            tags=["hrm", "text", "prefix-lm"],
        ),
        exist_ok=True,
    )


_register_templates()
_register_model()
print(
    "[HrmTextSwift] registered "
    f"model_type={MODEL_TYPE} templates={[DIRECT_TEMPLATE_TYPE, SYNTH_COT_TEMPLATE_TYPE]} "
    f"model_path={MODEL_DIR}",
    flush=True,
)
