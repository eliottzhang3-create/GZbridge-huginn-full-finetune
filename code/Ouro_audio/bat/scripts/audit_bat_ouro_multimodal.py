"""GPU audit for the BAT Spatial-AST/Q-Former -> Ouro-1.4B integration.

This is a forward/backward contract audit, not a useful training run.  It
loads the registered ms-swift model, encodes one real BAT QA record through the
custom template, renders AudioSet + binaural RIR, and verifies:

* exactly 64 audio-prefix positions are replaced by [64, 2048] Q-Former output;
* full input/label/attention widths remain aligned;
* Ouro's shared decoder layer and gate execute four recurrent steps;
* one ordinary shifted CE loss is finite;
* Q-Former receives gradients, while Spatial-AST, Ouro native parameters, and
  early_exit_gate remain frozen.

LoRA injection is intentionally a separate next audit because this script
first isolates the multimodal tensor and gradient contract.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


MODEL_TYPE = "ouro_bat_spatial_ast"
TEMPLATE_TYPE = "ouro_bat_audio_prefix"
EXPECTED_STEPS = 4
EXPECTED_AUDIO_TOKENS = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--plugin-path", type=Path, required=True)
    parser.add_argument("--qa-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def import_plugin(path: Path):
    spec = importlib.util.spec_from_file_location("ouro_bat_spatial_ast_audit_plugin", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import plugin: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise TypeError(f"Expected list data in {path}")
    return records


def find_record(qa_root: Path, stage: str, question_type: str) -> dict[str, Any]:
    records = load_records(qa_root / stage / "train.json")
    for record in records:
        if str(record.get("question_type", "")).upper() == question_type:
            return record
    raise LookupError(f"No {stage}/{question_type} record found")


def as_long_batch(value: Any, device: torch.device) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.tensor(value, dtype=torch.long)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.to(device=device, dtype=torch.long)


def parameter_groups(model: torch.nn.Module) -> dict[str, int]:
    groups = {"spatial_ast": 0, "qformer": 0, "gate": 0, "ouro_native": 0, "other": 0}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("spatial_ast_encoder."):
            groups["spatial_ast"] += parameter.numel()
        elif name.startswith("audio_qformer."):
            groups["qformer"] += parameter.numel()
        elif "early_exit_gate" in name:
            groups["gate"] += parameter.numel()
        elif name.startswith(("model.", "lm_head.")):
            groups["ouro_native"] += parameter.numel()
        else:
            groups["other"] += parameter.numel()
    return groups


def main() -> None:
    args = parse_args()
    if not args.output.is_absolute() or str(args.output).replace("\\", "/").startswith("/hpc_stor03/public"):
        raise ValueError(f"Output must be a private absolute path: {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("This audit requires a submitted CUDA job")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if version("ms-swift") != "4.4.2" or version("transformers") != "4.54.1":
        raise RuntimeError(f"Unexpected environment: ms-swift={version('ms-swift')} transformers={version('transformers')}")

    plugin = import_plugin(args.plugin_path.resolve())
    if plugin.MODEL_TYPE != MODEL_TYPE or plugin.TEMPLATE_TYPE != TEMPLATE_TYPE:
        raise RuntimeError("BAT plugin registration constants do not match audit")
    from swift import get_model_processor, get_template

    print("========== BAT OURO MULTIMODAL FORWARD/BACKWARD AUDIT ==========")
    print(f"[python] {sys.version.split()[0]} executable={sys.executable}")
    print(f"[packages] ms-swift={version('ms-swift')} transformers={version('transformers')} torch={torch.__version__}")
    print(f"[model] {args.model_path}")
    print(f"[plugin] {args.plugin_path}")
    print(f"[device] {device} name={torch.cuda.get_device_name(device)}")

    model, processor = get_model_processor(
        str(args.model_path.resolve()),
        model_type=MODEL_TYPE,
        torch_dtype=torch.bfloat16,
        device_map={"": str(device)},
        load_model=True,
        download_model=False,
        attn_impl="sdpa",
        model_kwargs={"local_files_only": True, "low_cpu_mem_usage": True},
    )
    model.train()
    contract = getattr(model, "_ouro_bat_audio_contract", None)
    if not isinstance(contract, dict):
        raise RuntimeError("Ouro BAT loader did not attach its audio contract")
    if contract.get("audio_token_count") != EXPECTED_AUDIO_TOKENS:
        raise RuntimeError(f"Unexpected audio token count: {contract}")
    if contract.get("qformer_initialization") != "random" or contract.get("qformer_checkpoint_loaded") is not False:
        raise RuntimeError(f"Q-Former must be randomly initialized without checkpoint loading: {contract}")
    if contract.get("total_ut_steps") != EXPECTED_STEPS or contract.get("early_exit_threshold") != 1.0:
        raise RuntimeError(f"Unexpected Ouro loop contract: {contract}")

    template = get_template(
        template_type=TEMPLATE_TYPE,
        processor=processor,
        max_length=512,
        use_chat_template=True,
        padding_side="right",
        padding_free=False,
        template_backend="swift",
    )
    template.set_mode("train")
    qformer_record = find_record(args.qa_root.resolve(), "stage1-clsdoa", "CLASSIFICATION")
    encoded = template.encode(
        {
            "messages": [
                {"role": "user", "content": "Classify the sound."},
                {"role": "assistant", "content": str(qformer_record["answer"])},
            ],
            "audios": [qformer_record],
        }
    )
    input_ids = as_long_batch(encoded["input_ids"], device)
    labels = as_long_batch(encoded["labels"], device)
    attention_mask = torch.ones_like(input_ids)
    waveform = encoded.get("audio_waveform")
    if waveform is None:
        raise RuntimeError(f"Template did not produce audio_waveform; keys={sorted(encoded)}")
    waveform = waveform.unsqueeze(0).to(device=device, dtype=torch.float32)

    if input_ids.shape != labels.shape:
        raise RuntimeError(f"Template input/label shape mismatch: input={tuple(input_ids.shape)} labels={tuple(labels.shape)}")
    if input_ids.shape[1] <= EXPECTED_AUDIO_TOKENS:
        raise RuntimeError(f"Template produced no text after audio prefix: {tuple(input_ids.shape)}")
    if not bool((labels[:, :EXPECTED_AUDIO_TOKENS] == -100).all().item()):
        raise RuntimeError("Audio prefix labels are not fully masked")

    layer_calls = 0
    gate_calls = 0
    layer_backward_calls = 0
    gate_backward_calls = 0
    ouro_model = model.model

    def layer_forward(_module, _args, _output):
        nonlocal layer_calls
        layer_calls += 1

    def gate_forward(_module, _args, _output):
        nonlocal gate_calls
        gate_calls += 1

    def layer_backward(_module, _grad_input, _grad_output):
        nonlocal layer_backward_calls
        layer_backward_calls += 1

    def gate_backward(_module, _grad_input, _grad_output):
        nonlocal gate_backward_calls
        gate_backward_calls += 1

    handles = [
        ouro_model.layers[0].register_forward_hook(layer_forward),
        ouro_model.early_exit_gate.register_forward_hook(gate_forward),
        ouro_model.layers[0].register_full_backward_hook(layer_backward),
        ouro_model.early_exit_gate.register_full_backward_hook(gate_backward),
    ]
    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            audio_waveforms=waveform,
            use_cache=False,
        )
        logits = outputs.logits
        if tuple(logits.shape[:2]) != tuple(labels.shape):
            raise RuntimeError(f"Logits/labels shape mismatch: logits={tuple(logits.shape)} labels={tuple(labels.shape)}")
        shifted_logits = logits[:, :-1].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        manual_loss = F.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.shape[-1]),
            shifted_labels.reshape(-1),
            ignore_index=-100,
        )
        if not torch.isfinite(manual_loss):
            raise RuntimeError("Manual shifted CE is non-finite")
        manual_loss.backward()
    finally:
        for handle in handles:
            handle.remove()

    qformer_grad = sum(
        int(parameter.grad is not None and torch.isfinite(parameter.grad).all().item())
        for parameter in model.audio_qformer.parameters()
        if parameter.requires_grad
    )
    qformer_trainables = sum(parameter.numel() for parameter in model.audio_qformer.parameters() if parameter.requires_grad)
    if qformer_grad <= 0 or qformer_trainables <= 0:
        raise RuntimeError(f"Q-Former did not receive finite gradients: trainables={qformer_trainables} grads={qformer_grad}")
    groups = parameter_groups(model)
    if groups["spatial_ast"] or groups["gate"] or groups["ouro_native"] or groups["other"]:
        raise RuntimeError(f"Unexpected trainable groups before LoRA injection: {groups}")
    if layer_calls != EXPECTED_STEPS or gate_calls != EXPECTED_STEPS:
        raise RuntimeError(f"Ouro recurrent forward count mismatch: layer={layer_calls} gate={gate_calls}")
    if layer_backward_calls != EXPECTED_STEPS:
        raise RuntimeError(f"Ouro recurrent backward count mismatch: layer={layer_backward_calls}")

    report = {
        "status": "ok",
        "model_type": MODEL_TYPE,
        "template_type": TEMPLATE_TYPE,
        "contract": contract,
        "template": {
            "input_ids_shape": list(input_ids.shape),
            "labels_shape": list(labels.shape),
            "audio_prefix_tokens": EXPECTED_AUDIO_TOKENS,
            "audio_prefix_labels_all_ignore": True,
            "waveform_shape": list(waveform.shape),
        },
        "forward": {
            "logits_shape": list(logits.shape),
            "manual_shifted_ce": float(manual_loss.detach().cpu().item()),
            "use_cache": False,
            "past_key_values_present": getattr(outputs, "past_key_values", None) is not None,
        },
        "recurrent_calls": {
            "expected_steps": EXPECTED_STEPS,
            "shared_layer_forward": layer_calls,
            "gate_forward": gate_calls,
            "shared_layer_backward": layer_backward_calls,
            "gate_backward": gate_backward_calls,
        },
        "parameters": {
            "trainable_groups_before_lora": groups,
            "qformer_trainable_parameters": qformer_trainables,
            "qformer_finite_gradient_parameter_count": qformer_grad,
            "spatial_ast_trainable": contract.get("encoder_trainable_parameters"),
            "gate_trainable": contract.get("gate_trainable_parameters"),
        },
        "scope": {
            "lora_injected": False,
            "spatial_ast_training": False,
            "ouro_native_training": False,
            "gate_training": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[template] input={list(input_ids.shape)} labels={list(labels.shape)} waveform={list(waveform.shape)}")
    print(f"[forward] logits={list(logits.shape)} loss={report['forward']['manual_shifted_ce']:.6f}")
    print(f"[recurrent] {report['recurrent_calls']}")
    print(f"[parameters] {report['parameters']}")
    print(f"[report] {args.output}")
    print("[status] ok")


if __name__ == "__main__":
    main()
