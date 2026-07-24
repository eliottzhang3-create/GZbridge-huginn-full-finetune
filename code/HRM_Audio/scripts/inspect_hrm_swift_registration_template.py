#!/usr/bin/env python3
"""Audit HRM Swift registration and PrefixLM template encoding without loading model weights."""

from __future__ import annotations

import argparse
import copy
import importlib
import importlib.util
import inspect
import json
import sys
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path
from types import ModuleType
from typing import Any

import torch


DEFAULT_MODEL_PATH = "/hpc_stor03/sjtu_home/jinwei.zhang/models/HRM-text"
MODEL_TYPE = "hrm_text_native"
DIRECT_TEMPLATE_TYPE = "hrm_text_direct"
SYNTH_COT_TEMPLATE_TYPE = "hrm_text_synth_cot"
TEMPLATE_SPECS = {
    DIRECT_TEMPLATE_TYPE: {
        "condition": "<|object_ref_start|>",
        "question": "What is 1 + 1?",
        "response": "2.",
    },
    SYNTH_COT_TEMPLATE_TYPE: {
        "condition": "<|quad_end|><|object_ref_end|>",
        "question": "What is 2 + 3?",
        "response": "2 + 3 = 5.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path(DEFAULT_MODEL_PATH))
    parser.add_argument("--plugin-path", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def import_plugin(path: Path, module_name: str) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(f"Plugin not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def discover_registry(
    modules: list[Any],
    required_key_hint: str,
    preferred_attribute_name: str,
) -> tuple[str, Mapping[Any, Any]]:
    candidates: list[tuple[str, Mapping[Any, Any]]] = []
    for module in modules:
        for attribute_name, value in vars(module).items():
            if "MAPPING" in attribute_name.upper() and isinstance(value, Mapping):
                candidates.append((f"{module.__name__}.{attribute_name}", value))
    if not candidates:
        raise RuntimeError(f"No Swift mapping registry found for hint={required_key_hint!r}")
    preferred = [
        (name, mapping)
        for name, mapping in candidates
        if name.rsplit(".", 1)[-1].upper() == preferred_attribute_name.upper()
    ]
    preferred_exact = [(name, mapping) for name, mapping in preferred if required_key_hint in mapping]
    if preferred_exact:
        return preferred_exact[0]
    if preferred:
        return preferred[0]
    exact = [(name, mapping) for name, mapping in candidates if required_key_hint in mapping]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        exact.sort(key=lambda item: len(item[1]), reverse=True)
        return exact[0]
    candidates.sort(key=lambda item: len(item[1]), reverse=True)
    return candidates[0]


def as_int_list(value: Any, name: str) -> list[int]:
    if torch.is_tensor(value):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be list/tuple/tensor, got {type(value)}")
    return [int(item) for item in value]


def tokenizer_from_processor(processor: Any):
    return processor.tokenizer if hasattr(processor, "tokenizer") else processor


def build_template(get_template, processor: Any, template_type: str):
    return get_template(
        template_type=template_type,
        processor=processor,
        max_length=512,
        use_chat_template=False,
        padding_side="right",
        padding_free=False,
        template_backend="swift",
    )


def audit_encoded(
    *,
    template: Any,
    tokenizer: Any,
    template_type: str,
    mode: str,
    question: str,
    response: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = [{"role": "user", "content": question}]
    if response is not None:
        messages.append({"role": "assistant", "content": response})
    sample = {"messages": messages}

    template.set_mode(mode)
    encoded = template.encode(sample)
    input_ids = as_int_list(encoded["input_ids"], "input_ids")
    token_type_ids = as_int_list(encoded.get("token_type_ids"), "token_type_ids")
    labels_value = encoded.get("labels")
    labels = as_int_list(labels_value, "labels") if labels_value is not None else None

    if len(token_type_ids) != len(input_ids):
        raise RuntimeError(
            f"{template_type}/{mode} token_type_ids length mismatch: "
            f"input={len(input_ids)} token_type={len(token_type_ids)}"
        )
    if labels is not None and len(labels) != len(input_ids):
        raise RuntimeError(
            f"{template_type}/{mode} labels length mismatch: input={len(input_ids)} labels={len(labels)}"
        )

    condition = TEMPLATE_SPECS[template_type]["condition"]
    expected_prompt = f"<|im_start|>{condition}{question}<|im_end|>"
    expected_prompt_ids = tokenizer(expected_prompt, add_special_tokens=False)["input_ids"]
    expected_prompt_ids = as_int_list(expected_prompt_ids, "expected_prompt_ids")
    if input_ids[: len(expected_prompt_ids)] != expected_prompt_ids:
        raise RuntimeError(
            f"{template_type}/{mode} prompt token mismatch: "
            f"expected={expected_prompt_ids} actual_prefix={input_ids[:len(expected_prompt_ids)]}"
        )

    im_start_id = int(tokenizer.convert_tokens_to_ids("<|im_start|>"))
    im_end_id = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    if input_ids[0] != im_start_id or input_ids.count(im_start_id) != 1:
        raise RuntimeError(f"{template_type}/{mode} must contain exactly one leading <|im_start|>")
    if input_ids[len(expected_prompt_ids) - 1] != im_end_id:
        raise RuntimeError(f"{template_type}/{mode} prompt must end with <|im_end|>")

    transitions = [index for index in range(1, len(token_type_ids)) if token_type_ids[index] != token_type_ids[index - 1]]
    supervised_positions = [] if labels is None else [index for index, label in enumerate(labels) if label != -100]

    if response is None:
        if set(token_type_ids) != {1}:
            raise RuntimeError(f"{template_type}/infer requires all token_type_ids=1, got {token_type_ids}")
        if labels is not None and supervised_positions:
            raise RuntimeError(f"{template_type}/infer unexpectedly has supervised labels: {supervised_positions}")
        if input_ids != expected_prompt_ids:
            raise RuntimeError(f"{template_type}/infer contains tokens outside the official prompt")
        prefix_length = len(input_ids)
    else:
        if not supervised_positions:
            raise RuntimeError(f"{template_type}/train has no supervised response tokens")
        prefix_length = supervised_positions[0]
        if prefix_length != len(expected_prompt_ids):
            raise RuntimeError(
                f"{template_type}/train response boundary mismatch: "
                f"expected={len(expected_prompt_ids)} actual={prefix_length}"
            )
        if transitions != [prefix_length]:
            raise RuntimeError(
                f"{template_type}/train requires one 1->0 transition at {prefix_length}, got {transitions}"
            )
        if set(token_type_ids[:prefix_length]) != {1} or set(token_type_ids[prefix_length:]) != {0}:
            raise RuntimeError(f"{template_type}/train PrefixLM mask is invalid: {token_type_ids}")
        if any(label != -100 for label in labels[:prefix_length]):
            raise RuntimeError(f"{template_type}/train prompt labels are not fully ignored")
        if input_ids[-1] != tokenizer.eos_token_id or labels[-1] != tokenizer.eos_token_id:
            raise RuntimeError(
                f"{template_type}/train sequence must end with supervised EOS: "
                f"input={input_ids[-1]} label={labels[-1]} eos={tokenizer.eos_token_id}"
            )

    token_rows: list[dict[str, Any]] = []
    for index, input_id in enumerate(input_ids):
        label = None if labels is None else labels[index]
        token_rows.append(
            {
                "index": index,
                "input_id": input_id,
                "input_token": tokenizer.convert_ids_to_tokens(input_id),
                "label": label,
                "label_token": None if label is None else ("<IGNORED>" if label == -100 else tokenizer.convert_ids_to_tokens(label)),
                "token_type_id": token_type_ids[index],
                "region": "prefix" if index < prefix_length else "response",
            }
        )

    print(f"\n========== {template_type} {mode.upper()} ==========", flush=True)
    print(f"[sample] {json.dumps(sample, ensure_ascii=False)}", flush=True)
    print(f"[decoded-input] {template.safe_decode(input_ids)}", flush=True)
    if labels is not None:
        print(f"[decoded-labels] {template.safe_decode(labels)}", flush=True)
    print(
        f"[boundary] sequence={len(input_ids)} prefix={prefix_length} "
        f"supervised={len(supervised_positions)} transitions={transitions}",
        flush=True,
    )
    for row in token_rows:
        print(
            f"[token] i={row['index']:03d} input={row['input_token']!r} id={row['input_id']} "
            f"label={row['label_token']!r} type={row['token_type_id']} region={row['region']}",
            flush=True,
        )

    report = {
        "template_type": template_type,
        "mode": mode,
        "sample": sample,
        "expected_prompt": expected_prompt,
        "input_ids": input_ids,
        "labels": labels,
        "token_type_ids": token_type_ids,
        "sequence_length": len(input_ids),
        "prefix_length": prefix_length,
        "supervised_positions": supervised_positions,
        "transitions": transitions,
        "decoded_input": template.safe_decode(input_ids),
        "decoded_labels": template.safe_decode(labels) if labels is not None else None,
        "token_rows": token_rows,
    }
    return encoded, report


def audit_collator(template: Any, encoded_samples: list[dict[str, Any]]) -> dict[str, Any]:
    original_lengths = [len(as_int_list(item["input_ids"], "input_ids")) for item in encoded_samples]
    batch = template.data_collator(copy.deepcopy(encoded_samples))
    required = {"input_ids", "attention_mask", "labels", "token_type_ids"}
    missing = sorted(required - set(batch))
    if missing:
        raise RuntimeError(f"Swift collator dropped HRM fields: missing={missing}, keys={sorted(batch)}")

    batch_size, padded_length = batch["input_ids"].shape
    if batch_size != len(encoded_samples) or padded_length != max(original_lengths):
        raise RuntimeError(
            f"Unexpected collated shape: batch={tuple(batch['input_ids'].shape)} lengths={original_lengths}"
        )
    for row_index, original_length in enumerate(original_lengths):
        expected_types = as_int_list(encoded_samples[row_index]["token_type_ids"], "token_type_ids")
        actual_types = batch["token_type_ids"][row_index, :original_length].tolist()
        if actual_types != expected_types:
            raise RuntimeError(
                f"Collator changed token_type_ids for row={row_index}: expected={expected_types} actual={actual_types}"
            )
        if original_length < padded_length:
            if not bool((batch["attention_mask"][row_index, original_length:] == 0).all().item()):
                raise RuntimeError(f"Collator attention padding is not zero for row={row_index}")
            if not bool((batch["labels"][row_index, original_length:] == -100).all().item()):
                raise RuntimeError(f"Collator label padding is not -100 for row={row_index}")
            if not bool((batch["token_type_ids"][row_index, original_length:] == 0).all().item()):
                raise RuntimeError(f"Collator token_type_ids padding is not zero for row={row_index}")

    report = {
        "keys": sorted(batch),
        "original_lengths": original_lengths,
        "shapes": {key: list(value.shape) for key, value in batch.items() if torch.is_tensor(value)},
        "input_ids": batch["input_ids"].tolist(),
        "attention_mask": batch["attention_mask"].tolist(),
        "labels": batch["labels"].tolist(),
        "token_type_ids": batch["token_type_ids"].tolist(),
    }
    print(f"\n[collator] {json.dumps(report, ensure_ascii=False)}", flush=True)
    return report


def main() -> None:
    args = parse_args()
    if version("ms-swift") != "4.4.2" or version("transformers") != "5.9.0":
        raise RuntimeError(
            f"Unexpected environment: ms-swift={version('ms-swift')} transformers={version('transformers')}"
        )

    import swift.model as swift_model
    import swift.template as swift_template

    try:
        from swift import get_model_processor, get_template
    except ImportError:
        from swift.model import get_model_processor
        from swift.template import get_template

    model_register_module = importlib.import_module("swift.model.register")
    template_register_module = importlib.import_module("swift.template.register")
    model_registry_name, model_registry = discover_registry(
        [swift_model, model_register_module],
        required_key_hint=MODEL_TYPE,
        preferred_attribute_name="MODEL_MAPPING",
    )
    template_registry_name, template_registry = discover_registry(
        [swift_template, template_register_module],
        required_key_hint=DIRECT_TEMPLATE_TYPE,
        preferred_attribute_name="TEMPLATE_MAPPING",
    )
    sizes_before = {"model": len(model_registry), "template": len(template_registry)}

    plugin_path = args.plugin_path.resolve()
    plugin = import_plugin(plugin_path, "hrm_text_swift_registration_audit")
    if plugin.MODEL_TYPE != MODEL_TYPE:
        raise RuntimeError(f"Plugin model type mismatch: {plugin.MODEL_TYPE}")
    if plugin.DIRECT_TEMPLATE_TYPE != DIRECT_TEMPLATE_TYPE or plugin.SYNTH_COT_TEMPLATE_TYPE != SYNTH_COT_TEMPLATE_TYPE:
        raise RuntimeError("Plugin template type constants do not match the audit contract")

    # Rediscover after registration in case the public module exposed a proxy.
    model_registry_name, model_registry = discover_registry(
        [swift_model, model_register_module],
        required_key_hint=MODEL_TYPE,
        preferred_attribute_name="MODEL_MAPPING",
    )
    template_registry_name, template_registry = discover_registry(
        [swift_template, template_register_module],
        required_key_hint=DIRECT_TEMPLATE_TYPE,
        preferred_attribute_name="TEMPLATE_MAPPING",
    )
    sizes_after_first_import = {"model": len(model_registry), "template": len(template_registry)}
    expected_sizes_after_first_import = {
        "model": sizes_before["model"] + 1,
        "template": sizes_before["template"] + 2,
    }
    if sizes_after_first_import != expected_sizes_after_first_import:
        raise RuntimeError(
            "Unexpected registry size delta after first plugin import: "
            f"before={sizes_before} expected={expected_sizes_after_first_import} actual={sizes_after_first_import}"
        )
    if MODEL_TYPE not in model_registry:
        raise RuntimeError(f"Model registration missing from {model_registry_name}: {MODEL_TYPE}")
    for template_type in TEMPLATE_SPECS:
        if template_type not in template_registry:
            raise RuntimeError(f"Template registration missing from {template_registry_name}: {template_type}")

    expected_prompts = {
        DIRECT_TEMPLATE_TYPE: ["<|im_start|><|object_ref_start|>{{QUERY}}<|im_end|>"],
        SYNTH_COT_TEMPLATE_TYPE: [
            "<|im_start|><|quad_end|><|object_ref_end|>{{QUERY}}<|im_end|>"
        ],
    }
    registered_template_meta: dict[str, Any] = {}
    for template_type, expected_prompt in expected_prompts.items():
        template_meta = template_registry[template_type]
        if template_meta.prompt != expected_prompt:
            raise RuntimeError(
                f"Unexpected registered prompt for {template_type}: "
                f"expected={expected_prompt} actual={template_meta.prompt}"
            )
        if template_meta.prefix != [] or template_meta.chat_sep is not None:
            raise RuntimeError(f"{template_type} must be prefix-free and single-turn")
        if template_meta.suffix != [["eos_token_id"]] or template_meta.auto_add_bos:
            raise RuntimeError(
                f"{template_type} EOS/BOS metadata mismatch: "
                f"suffix={template_meta.suffix} auto_add_bos={template_meta.auto_add_bos}"
            )
        if template_meta.template_cls.__name__ != "HrmTextPrefixLMTemplate":
            raise RuntimeError(f"Unexpected template class for {template_type}: {template_meta.template_cls}")
        if getattr(template_meta.template_cls, "support_padding_free", None) is not False:
            raise RuntimeError(f"{template_type} must explicitly disable padding-free mode")
        registered_template_meta[template_type] = {
            "prompt": template_meta.prompt,
            "prefix": template_meta.prefix,
            "chat_sep": template_meta.chat_sep,
            "suffix": template_meta.suffix,
            "auto_add_bos": template_meta.auto_add_bos,
            "template_cls": f"{template_meta.template_cls.__module__}.{template_meta.template_cls.__name__}",
            "support_padding_free": template_meta.template_cls.support_padding_free,
        }

    import_plugin(plugin_path, "hrm_text_swift_registration_audit_second_import")
    sizes_after_second_import = {"model": len(model_registry), "template": len(template_registry)}
    if sizes_after_second_import != sizes_after_first_import:
        raise RuntimeError(
            "Plugin registration is not idempotent: "
            f"first={sizes_after_first_import} second={sizes_after_second_import}"
        )

    model_meta = model_registry[MODEL_TYPE]
    if model_meta.loader.__name__ != "ModelLoader":
        raise RuntimeError(f"HRM registration must use default ModelLoader, got {model_meta.loader}")
    if model_meta.template != DIRECT_TEMPLATE_TYPE:
        raise RuntimeError(f"Unexpected default HRM template: {model_meta.template}")
    if model_meta.architectures != ["HrmTextForCausalLM"]:
        raise RuntimeError(f"Unexpected HRM architectures: {model_meta.architectures}")
    if model_meta.is_multimodal:
        raise RuntimeError("Text-only HRM registration is unexpectedly multimodal")
    registered_model_paths = [
        model.model_path
        for group in model_meta.model_groups
        for model in group.models
        if model.model_path is not None
    ]
    if str(args.model_path.resolve()) not in [str(Path(path).resolve()) for path in registered_model_paths]:
        raise RuntimeError(
            f"Registered HRM local path mismatch: expected={args.model_path.resolve()} actual={registered_model_paths}"
        )

    print("========== HRM SWIFT REGISTRATION + TEMPLATE AUDIT ==========", flush=True)
    print(f"[python] version={sys.version.split()[0]} executable={sys.executable}", flush=True)
    print(f"[plugin] path={plugin_path}", flush=True)
    print(f"[registry] model={model_registry_name} template={template_registry_name}", flush=True)
    print(
        f"[registry-sizes] before={sizes_before} first={sizes_after_first_import} "
        f"second={sizes_after_second_import}",
        flush=True,
    )
    print(f"[get_model_processor] signature={inspect.signature(get_model_processor)}", flush=True)
    print(f"[get_template] signature={inspect.signature(get_template)}", flush=True)

    model_path = args.model_path.resolve()
    model, processor = get_model_processor(
        str(model_path),
        model_type=MODEL_TYPE,
        load_model=False,
        download_model=False,
    )
    if model is not None:
        raise RuntimeError("load_model=False unexpectedly returned a model instance")
    tokenizer = tokenizer_from_processor(processor)
    if tokenizer.eos_token != "<|box_end|>" or tokenizer.eos_token_id != 11:
        raise RuntimeError(f"Unexpected HRM EOS: token={tokenizer.eos_token!r} id={tokenizer.eos_token_id}")
    print(
        f"[processor] type={type(processor)} tokenizer={type(tokenizer)} "
        f"vocab={len(tokenizer)} eos={tokenizer.eos_token!r}/{tokenizer.eos_token_id}",
        flush=True,
    )

    encoding_reports: list[dict[str, Any]] = []
    train_encoded_for_collator: list[dict[str, Any]] = []
    direct_train_template = None
    for template_type, spec in TEMPLATE_SPECS.items():
        infer_template = build_template(get_template, processor, template_type)
        _, infer_report = audit_encoded(
            template=infer_template,
            tokenizer=tokenizer,
            template_type=template_type,
            mode="infer",
            question=spec["question"],
            response=None,
        )
        encoding_reports.append(infer_report)

        train_template = build_template(get_template, processor, template_type)
        train_encoded, train_report = audit_encoded(
            template=train_template,
            tokenizer=tokenizer,
            template_type=template_type,
            mode="train",
            question=spec["question"],
            response=spec["response"],
        )
        encoding_reports.append(train_report)
        if template_type == DIRECT_TEMPLATE_TYPE:
            direct_train_template = train_template
            train_encoded_for_collator.append(train_encoded)

    if direct_train_template is None:
        raise RuntimeError("Direct HRM train template was not created")
    second_encoded, second_report = audit_encoded(
        template=direct_train_template,
        tokenizer=tokenizer,
        template_type=DIRECT_TEMPLATE_TYPE,
        mode="train",
        question="State the result of ten plus twenty in one short sentence.",
        response="Ten plus twenty equals thirty.",
    )
    encoding_reports.append(second_report)
    train_encoded_for_collator.append(second_encoded)
    collator_report = audit_collator(direct_train_template, train_encoded_for_collator)

    report = {
        "status": "ok",
        "packages": {
            "ms-swift": version("ms-swift"),
            "transformers": version("transformers"),
            "torch": version("torch"),
        },
        "plugin_path": str(plugin_path),
        "model_path": str(model_path),
        "registries": {
            "model_registry": model_registry_name,
            "template_registry": template_registry_name,
            "sizes_before": sizes_before,
            "sizes_after_first_import": sizes_after_first_import,
            "sizes_after_second_import": sizes_after_second_import,
            "idempotent": sizes_after_first_import == sizes_after_second_import,
        },
        "model_meta": {
            "model_type": model_meta.model_type,
            "loader": f"{model_meta.loader.__module__}.{model_meta.loader.__name__}",
            "template": model_meta.template,
            "architectures": model_meta.architectures,
            "torch_dtype": str(model_meta.torch_dtype),
            "is_multimodal": model_meta.is_multimodal,
            "requires": model_meta.requires,
            "tags": model_meta.tags,
            "registered_model_paths": registered_model_paths,
        },
        "template_meta": registered_template_meta,
        "processor": {
            "type": f"{type(processor).__module__}.{type(processor).__name__}",
            "tokenizer_type": f"{type(tokenizer).__module__}.{type(tokenizer).__name__}",
            "vocab_size": len(tokenizer),
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
        },
        "encodings": encoding_reports,
        "collator": collator_report,
    }
    atomic_write_json(args.output_report, report)
    print(f"\n[result] status=OK output_report={args.output_report}", flush=True)


if __name__ == "__main__":
    main()
