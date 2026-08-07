#!/usr/bin/env python3
"""Load the text-only Ouro registration through ms-swift and generate text."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path
from types import ModuleType
from typing import Any

import torch


MODEL_TYPE = "ouro_text_native"
TEMPLATE_TYPE = "ouro_text_direct"
DEFAULT_MODEL_PATH = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/Ouro-1.4B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--plugin-path", type=Path, required=True)
    parser.add_argument("--question", default="The future of artificial intelligence is")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def import_plugin(path: Path) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(f"Plugin not found: {path}")
    module_name = "ouro_text_swift_inference_audit"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def tokenizer_from_processor(processor: Any) -> Any:
    return processor.tokenizer if hasattr(processor, "tokenizer") else processor


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def main() -> None:
    args = parse_args()
    if version("ms-swift") != "4.4.2":
        raise RuntimeError(f"Expected ms-swift==4.4.2, got {version('ms-swift')}")
    if version("transformers") != "4.54.1":
        raise RuntimeError(f"Expected transformers==4.54.1, got {version('transformers')}")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if not args.question.strip():
        raise ValueError("--question must not be empty")
    if not torch.cuda.is_available():
        raise RuntimeError("Ouro Swift inference requires CUDA and must run in a submitted GPU job")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"This audit requires CUDA, got {device}")
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_device(device_index)

    model_path = args.model_path.expanduser().resolve()
    plugin_path = args.plugin_path.expanduser().resolve()
    required_files = (
        "config.json",
        "model.safetensors",
        "configuration_ouro.py",
        "modeling_ouro.py",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    missing = [name for name in required_files if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete Ouro snapshot at {model_path}: missing={missing}")

    plugin = import_plugin(plugin_path)
    if getattr(plugin, "MODEL_TYPE", None) != MODEL_TYPE:
        raise RuntimeError(f"Plugin model type mismatch: expected={MODEL_TYPE} actual={getattr(plugin, 'MODEL_TYPE', None)!r}")
    if getattr(plugin, "TEMPLATE_TYPE", None) != TEMPLATE_TYPE:
        raise RuntimeError(
            f"Plugin template mismatch: expected={TEMPLATE_TYPE} actual={getattr(plugin, 'TEMPLATE_TYPE', None)!r}"
        )

    try:
        from swift import get_model_processor, get_template
    except ImportError:
        from swift.model import get_model_processor
        from swift.template import get_template

    print("========== OURO MS-SWIFT TEXT INFERENCE ==========", flush=True)
    print(f"[packages] ms-swift={version('ms-swift')} transformers={version('transformers')} torch={version('torch')}", flush=True)
    print(f"[model-path] {model_path}", flush=True)
    print(f"[plugin-path] {plugin_path}", flush=True)
    print(f"[settings] model_type={MODEL_TYPE} template={TEMPLATE_TYPE} device={device}", flush=True)

    load_started = time.perf_counter()
    model, processor = get_model_processor(
        str(model_path),
        model_type=MODEL_TYPE,
        torch_dtype=torch.bfloat16,
        device_map={"": str(device)},
        load_model=True,
        download_model=False,
        attn_impl="sdpa",
        model_kwargs={"local_files_only": True, "low_cpu_mem_usage": True, "trust_remote_code": True},
    )
    if model is None:
        raise RuntimeError("ms-swift get_model_processor returned model=None")
    model.eval()
    torch.cuda.synchronize(device_index)
    load_seconds = time.perf_counter() - load_started

    if model.__class__.__name__ != "OuroForCausalLM":
        raise RuntimeError(f"Unexpected Swift model class: {model.__class__.__module__}.{model.__class__.__name__}")
    config = model.config
    config_summary = {
        "model_type": getattr(config, "model_type", None),
        "hidden_size": int(getattr(config, "hidden_size", -1)),
        "num_hidden_layers": int(getattr(config, "num_hidden_layers", -1)),
        "total_ut_steps": int(getattr(config, "total_ut_steps", -1)),
    }
    expected_config = {"model_type": "ouro", "hidden_size": 2048, "num_hidden_layers": 24, "total_ut_steps": 4}
    if config_summary != expected_config:
        raise RuntimeError(f"Unexpected Swift-loaded Ouro config: expected={expected_config} actual={config_summary}")

    tokenizer = tokenizer_from_processor(processor)
    template = get_template(
        template_type=TEMPLATE_TYPE,
        processor=processor,
        max_length=512,
        use_chat_template=True,
        padding_side="right",
        padding_free=False,
        template_backend="swift",
    )
    template.set_mode("transformers")
    encoded = template.encode({"messages": [{"role": "user", "content": args.question.strip()}]})
    input_ids = encoded.get("input_ids")
    if input_ids is None:
        raise RuntimeError(f"Swift template did not produce input_ids: keys={sorted(encoded)}")
    if torch.is_tensor(input_ids):
        input_ids = input_ids.tolist()
    input_ids = [int(item) for item in input_ids]
    if not input_ids:
        raise RuntimeError("Swift template produced an empty prompt")

    model_inputs = {
        "input_ids": torch.tensor([input_ids], dtype=torch.long, device=device),
        "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long, device=device),
    }
    print(f"[prompt] tokens={len(input_ids)} ids={input_ids}", flush=True)

    with torch.inference_mode():
        prefill = model(**model_inputs, use_cache=False, logits_to_keep=1)
    if tuple(prefill.logits.shape) != (1, 1, int(config.vocab_size)):
        raise RuntimeError(f"Unexpected Swift prefill logits shape: {tuple(prefill.logits.shape)}")
    if not bool(torch.isfinite(prefill.logits).all().item()):
        raise RuntimeError("Swift prefill logits contain NaN or Inf")

    generation_started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=config.eos_token_id,
            pad_token_id=getattr(config, "pad_token_id", None) or tokenizer.eos_token_id,
        )
    torch.cuda.synchronize(device_index)
    generation_seconds = time.perf_counter() - generation_started
    if output_ids.shape[1] <= len(input_ids):
        raise RuntimeError("Swift generation produced no new tokens")
    generated_ids = output_ids[0, len(input_ids):]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"[generate] new_tokens={generated_ids.numel()} seconds={generation_seconds:.3f}", flush=True)
    print(f"[generate] text={generated_text!r}", flush=True)

    report = {
        "status": "ok",
        "packages": {
            "ms-swift": version("ms-swift"),
            "transformers": version("transformers"),
            "torch": version("torch"),
        },
        "model_path": str(model_path),
        "plugin_path": str(plugin_path),
        "model_type": MODEL_TYPE,
        "template_type": TEMPLATE_TYPE,
        "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "parameter_count": count_parameters(model),
        "config": config_summary,
        "prompt": args.question.strip(),
        "prompt_length": len(input_ids),
        "prefill_logits_shape": list(prefill.logits.shape),
        "generated_token_count": int(generated_ids.numel()),
        "generated_text": generated_text,
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
    }
    atomic_write_json(args.output_report, report)
    print(f"[result] status=OK output_report={args.output_report}", flush=True)


if __name__ == "__main__":
    main()
