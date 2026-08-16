#!/usr/bin/env python3
"""Profile the real BAT Spatial-AST -> Q-Former -> Ouro training path.

This is deliberately a profiler, not a training job:

* it loads the same registered Ouro BAT model and injects the same LoRA;
* it reads real BAT manifest records and renders AudioSet/RIR through the
  production template in DataLoader workers;
* it runs forward and backward to measure the actual four-step Ouro graph and
  DDP gradient synchronization;
* it never creates an optimizer and never calls ``optimizer.step``;
* it writes no checkpoint and does not update any model parameter.

The report separates steady-state data wait, host-to-device transfer,
Spatial-AST, Q-Former, Ouro forward, backward main-stream time, and DDP
communication-hook time.  The DDP hook performs the same averaged all-reduce
as the default reducer while recording each gradient bucket's completion.  A
step-level wall timer is also retained because bucket all-reduces can overlap
with autograd and therefore their durations must not simply be summed.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

from bat.configs.training import BAT_TRAINING


MODEL_TYPE = "ouro_bat_spatial_ast"
TEMPLATE_TYPE = "ouro_bat_audio_prefix"
EXPECTED_WORLD_SIZE = 8
EXPECTED_AUDIO_TOKENS = 64
EXPECTED_SPATIAL_AST_SHAPE = (515, 768)
EXPECTED_QFORMER_SHAPE = (64, 2048)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--plugin-path", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--local-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--measure-ddp-communication",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use an asynchronous DDP communication hook to measure gradient buckets.",
    )
    parser.add_argument(
        "--torch-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compile only OuroForCausalLM.model, the four-cycle Transformer core.",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
        help="torch.compile mode used by the isolated benchmark.",
    )
    parser.add_argument(
        "--compile-dynamic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow dynamic sequence shapes in torch.compile.",
    )
    parser.add_argument(
        "--attention-profile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Profile one real forward/backward and identify the selected SDPA backend.",
    )
    parser.add_argument("--expected-world-size", type=int, default=EXPECTED_WORLD_SIZE)
    return parser.parse_args()


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def initialize_distributed(expected_world_size: int) -> tuple[int, int]:
    current_world = world_size()
    current_rank = rank()
    if current_world != expected_world_size:
        raise RuntimeError(
            f"Profiler world-size mismatch: environment={current_world} expected={expected_world_size}"
        )
    if current_world > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed BAT profiling requires CUDA")
        current_local_rank = local_rank()
        if current_local_rank < 0 or current_local_rank >= current_world:
            raise RuntimeError(f"Invalid LOCAL_RANK={current_local_rank}")
        torch.cuda.set_device(current_local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    else:
        current_local_rank = 0
        torch.cuda.set_device(0)
    return current_rank, current_local_rank


def import_plugin(path: Path):
    spec = importlib.util.spec_from_file_location("bat_profile_ouro_plugin", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import plugin: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_manifest_rows(path: Path, start_index: int, count: int) -> tuple[list[dict[str, Any]], float]:
    if start_index < 0 or count <= 0:
        raise ValueError("start-index must be non-negative and count must be positive")
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < start_index:
                continue
            if len(rows) >= count:
                break
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Manifest row {index} is not an object")
            rows.append(value)
    if len(rows) != count:
        raise RuntimeError(
            f"Manifest does not contain enough rows: requested={count} start={start_index} got={len(rows)}"
        )
    return rows, time.perf_counter() - started


class EncodedBATDataset(Dataset):
    """Lazy production-template encoding for DataLoader profiling."""

    def __init__(self, rows: list[dict[str, Any]], template: Any):
        self.rows = rows
        self.template = template

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.template.encode(self.rows[index])


def to_list(value: Any) -> list[int]:
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    if not isinstance(value, list):
        raise TypeError(f"Expected token list, got {type(value).__name__}")
    return [int(item) for item in value]


def make_collator(template: Any, pad_token_id: int):
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        if not batch:
            raise RuntimeError("Profiler received an empty batch")
        input_rows = [to_list(item["input_ids"]) for item in batch]
        label_rows = [to_list(item["labels"]) for item in batch]
        if any(len(a) != len(b) for a, b in zip(input_rows, label_rows)):
            raise RuntimeError("Template input_ids/labels are not aligned")
        max_length = max(len(row) for row in input_rows)
        input_ids = torch.full((len(batch), max_length), pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), max_length), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_length), dtype=torch.long)
        for index, (input_row, label_row) in enumerate(zip(input_rows, label_rows)):
            length = len(input_row)
            input_ids[index, :length] = torch.tensor(input_row, dtype=torch.long)
            labels[index, :length] = torch.tensor(label_row, dtype=torch.long)
            attention_mask[index, :length] = 1
        payload = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        payload.update(template._data_collator_mm_data(batch))
        records = [item.get("bat_audio_record") for item in batch]
        if all(record is not None for record in records):
            payload["bat_audio_records"] = records
        return payload

    return collate


def source_summary(record: Any) -> dict[str, Any] | None:
    """Keep only provenance fields needed to diagnose a slow audio batch."""
    if not isinstance(record, dict):
        return None
    return {
        key: str(record.get(key, ""))
        for key in ("audio_id", "reverb_id", "audio_id2", "reverb_id2", "question_id", "question_type")
    }


class CudaRegionTimer:
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def begin(self) -> None:
        self.start.record()

    def finish(self) -> None:
        self.end.record()

    def milliseconds(self) -> float:
        return float(self.start.elapsed_time(self.end))


class DDPCommunicationProfiler:
    """Record real DDP bucket all-reduces without waiting in the hook.

    DDP communication hooks replace the reducer's default hook, so this hook
    must preserve the default semantics: all-reduce the bucket and divide by
    world size.  The asynchronous Future is returned directly, preserving
    communication/compute overlap instead of serializing every bucket with a
    blocking ``wait``.
    """

    def __init__(self, world_size: int):
        self.world_size = int(world_size)
        self.current_step: int | None = None
        self._records: list[dict[str, Any]] = []
        self._pending_callbacks = 0

    def begin_step(self, step_index: int) -> None:
        if self._pending_callbacks:
            raise RuntimeError("Previous DDP communication callbacks are still pending")
        self.current_step = int(step_index)
        self._records = []

    def hook(self, _state: Any, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
        launched_at = time.perf_counter()
        try:
            bucket_index = int(bucket.index())
        except Exception:
            bucket_index = -1
        bucket_numel = int(bucket.buffer().numel())
        self._pending_callbacks += 1
        work = dist.all_reduce(bucket.buffer(), async_op=True)
        future = work.get_future()

        def complete(fut: Any):
            # The Future callback runs after the asynchronous collective has
            # completed.  We deliberately use CPU timestamps here.  A CUDA
            # event recorded from this CPU Future callback is associated with
            # the current compute stream, not necessarily NCCL's stream, and
            # can incorrectly include unrelated autograd work.
            completed_at = time.perf_counter()
            record = {
                "step_index": self.current_step,
                "bucket_index": bucket_index,
                "bucket_numel": bucket_numel,
                "launched_at": launched_at,
                "completed_at": completed_at,
            }
            self._records.append(record)
            try:
                tensor = fut.value()[0]
                tensor.div_(self.world_size)
                return tensor
            finally:
                self._pending_callbacks -= 1

        return future.then(complete)

    def finish_step(self) -> dict[str, Any]:
        # The caller synchronizes the CUDA device before asking for these
        # values.  Keep this method defensive because a failed collective
        # should result in an explicit audit error, not a misleading zero.
        deadline = time.perf_counter() + 5.0
        while self._pending_callbacks and time.perf_counter() < deadline:
            time.sleep(0.001)
        if self._pending_callbacks:
            raise RuntimeError(
                f"Timed out waiting for DDP communication callbacks: pending={self._pending_callbacks}"
            )
        records = list(self._records)
        bucket_details: list[dict[str, Any]] = []
        for record in records:
            bucket_details.append({
                "step_index": record["step_index"],
                "bucket_index": record["bucket_index"],
                "bucket_numel": record["bucket_numel"],
                "cpu_completion_seconds": float(record["completed_at"] - record["launched_at"]),
            })
        cpu_starts = [float(record["launched_at"]) for record in records]
        cpu_ends = [float(record["completed_at"]) for record in records]
        cpu_span = max(cpu_ends) - min(cpu_starts) if records else 0.0
        cpu_latencies = [item["cpu_completion_seconds"] for item in bucket_details]
        return {
            "enabled": True,
            "bucket_count": len(bucket_details),
            "bucket_latency_sum_seconds": float(sum(cpu_latencies)) if cpu_latencies else 0.0,
            "bucket_span_seconds": float(cpu_span),
            "bucket_details": bucket_details,
        }


class AttentionKernelProfiler:
    """Collect only the SDPA-related events from one real CUDA step.

    The generic ``aten::scaled_dot_product_attention`` event is not enough to
    prove which backend was selected.  We therefore inspect both the operator
    events and their backend-specific descendants, such as
    ``_scaled_dot_product_flash_attention`` and
    ``_scaled_dot_product_efficient_attention``.  If only the generic event is
    visible, the audit remains incomplete instead of claiming Flash Attention.
    """

    def __init__(self):
        self._profiler = None

    def __enter__(self):
        self._profiler = torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
        )
        self._profiler.__enter__()
        return self

    @staticmethod
    def _seconds(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value) / 1_000_000.0
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _backend(name: str) -> str | None:
        lowered = name.lower()
        if "flash_attention" in lowered or "flashattention" in lowered:
            return "flash"
        if "efficient_attention" in lowered or "efficientattention" in lowered:
            return "efficient"
        if "scaled_dot_product_attention_math" in lowered or "sdpa_math" in lowered:
            return "math"
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        if self._profiler is not None:
            self._profiler.__exit__(exc_type, exc_value, traceback)

    def report(self) -> dict[str, Any]:
        if self._profiler is None:
            raise RuntimeError("Attention profiler was not started")

        operator_events: list[dict[str, Any]] = []
        backend_events: list[dict[str, Any]] = []
        seen_operator: set[tuple[str, int]] = set()
        seen_backend: set[tuple[str, int]] = set()

        def consume(event: Any, default_count: int = 0) -> None:
            name = str(getattr(event, "key", ""))
            lowered = name.lower()
            is_operator = "scaled_dot_product" in lowered
            backend = self._backend(name)
            if not is_operator and backend is None:
                return
            count = int(getattr(event, "count", default_count) or default_count)
            row = {
                "name": name,
                "count": count,
                "cpu_total_seconds": self._seconds(getattr(event, "cpu_time_total", None)),
                "cuda_total_seconds": self._seconds(getattr(event, "self_cuda_time_total", None)),
            }
            key = (name, count)
            if is_operator and key not in seen_operator:
                seen_operator.add(key)
                operator_events.append(row)
            if backend is not None and key not in seen_backend:
                seen_backend.add(key)
                row["backend"] = backend
                backend_events.append(row)

        for event in self._profiler.key_averages():
            consume(event)
        # Depending on the PyTorch profiler build, backend CUDA kernels may be
        # exposed only as individual FunctionEvents rather than grouped key
        # averages.  Inspect both views so a kernel is not missed.
        for event in self._profiler.events():
            consume(event, default_count=1)

        backends = sorted({str(item["backend"]) for item in backend_events})
        if "flash" in backends:
            selected_backend = "flash"
        elif "efficient" in backends:
            selected_backend = "efficient"
        elif "math" in backends:
            selected_backend = "math"
        elif operator_events:
            selected_backend = "generic_sdpa_unresolved"
        else:
            selected_backend = "not_observed"

        return {
            "status": "ok" if selected_backend in {"flash", "efficient", "math"} else "incomplete",
            "selected_backend": selected_backend,
            "runtime_sdp_flags": {
                "flash_sdp_enabled": bool(getattr(torch.backends.cuda, "flash_sdp_enabled", lambda: False)()),
                "mem_efficient_sdp_enabled": bool(getattr(torch.backends.cuda, "mem_efficient_sdp_enabled", lambda: False)()),
                "math_sdp_enabled": bool(getattr(torch.backends.cuda, "math_sdp_enabled", lambda: False)()),
                "device_capability": list(torch.cuda.get_device_capability()),
                "torch_version": torch.__version__,
            },
            "backend_events": backend_events,
            "operator_events": operator_events,
            "backend_event_count": len(backend_events),
            "operator_event_count": len(operator_events),
        }


def install_module_timer(module: torch.nn.Module) -> tuple[CudaRegionTimer, list[Any]]:
    timer = CudaRegionTimer()

    def before(_module: torch.nn.Module, _inputs: tuple[Any, ...]) -> None:
        timer.begin()

    def after(_module: torch.nn.Module, _inputs: tuple[Any, ...], _output: Any) -> None:
        timer.finish()

    return timer, [module.register_forward_pre_hook(before), module.register_forward_hook(after)]


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, DistributedDataParallel):
        model = model.module
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()
    return model


def find_causal_model(model: torch.nn.Module) -> torch.nn.Module:
    base = unwrap_model(model)
    if base.__class__.__name__ == "OuroForCausalLM":
        return base
    for module in base.modules():
        if module.__class__.__name__ == "OuroForCausalLM":
            return module
    raise RuntimeError("Unable to locate OuroForCausalLM after LoRA/DDP wrapping")


def inject_lora(model: torch.nn.Module) -> torch.nn.Module:
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=BAT_TRAINING.lora_rank,
        lora_alpha=BAT_TRAINING.lora_alpha,
        lora_dropout=BAT_TRAINING.lora_dropout,
        target_modules=list(BAT_TRAINING.lora_target_modules),
        modules_to_save=["audio_qformer"],
    )
    return get_peft_model(model, config)


def parameter_update_probe(model: torch.nn.Module) -> dict[str, Any]:
    before: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            before[name] = parameter.detach().cpu().clone()
    return {"count": len(before), "before": before}


def finish_parameter_update_probe(model: torch.nn.Module, probe: dict[str, Any]) -> dict[str, Any]:
    max_difference = 0.0
    changed: list[str] = []
    before = probe["before"]
    for name, old in before.items():
        current = dict(model.named_parameters())[name].detach().cpu()
        difference = float((current.float() - old.float()).abs().max().item())
        max_difference = max(max_difference, difference)
        if difference != 0.0:
            changed.append(name)
    return {
        "trainable_parameter_tensor_count": int(probe["count"]),
        "parameters_changed": bool(changed),
        "changed_parameter_preview": changed[:10],
        "max_abs_difference": max_difference,
        "optimizer_step_called": False,
    }


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p95": 0.0}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
        return float(ordered[index])

    return {
        "count": len(values),
        "mean": float(sum(values) / len(values)),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
    }


def snapshot_dynamo_counters() -> dict[str, Any] | None:
    try:
        from torch._dynamo.utils import counters
    except Exception:
        return None
    snapshot: dict[str, Any] = {}
    for group, values in counters.items():
        if hasattr(values, "items"):
            snapshot[str(group)] = {str(key): int(value) for key, value in values.items()}
        else:
            snapshot[str(group)] = int(values)
    return snapshot


def summarize_dynamo_counters(snapshot: dict[str, Any] | None) -> dict[str, int]:
    """Extract cumulative compile counters in a stable, JSON-safe shape."""
    if not snapshot:
        return {
            "unique_graphs": 0,
            "graph_break_count": 0,
            "calls_captured": 0,
        }
    stats = snapshot.get("stats", {})
    graph_breaks = snapshot.get("graph_break", {})
    return {
        "unique_graphs": int(stats.get("unique_graphs", 0)),
        "graph_break_count": int(sum(graph_breaks.values())) if hasattr(graph_breaks, "values") else 0,
        "calls_captured": int(stats.get("calls_captured", 0)),
    }


def compile_ouro_transformer_core(
    causal: torch.nn.Module,
    mode: str,
    dynamic: bool,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Compile only the Ouro recurrent Transformer core.

    The registered BAT model has this structure after LoRA injection:

        PeftModel -> OuroForCausalLM -> model -> 24 Transformer layers

    The audio renderer, frozen Spatial-AST, trainable Q-Former, audio-prefix
    replacement, LM head, and Peft outer wrapper remain eager.  Assigning the
    compiled module back to ``OuroForCausalLM.model`` makes the real causal-LM
    forward use the compiled four-cycle Transformer without compiling the
    entire multimodal Python path.
    """
    transformer_core = getattr(causal, "model", None)
    if not isinstance(transformer_core, torch.nn.Module):
        raise RuntimeError("Unable to locate OuroForCausalLM.model Transformer core")
    original_class = type(transformer_core).__name__
    compiled_core = torch.compile(
        transformer_core,
        backend="inductor",
        mode=mode,
        dynamic=dynamic,
        fullgraph=False,
    )
    causal.model = compiled_core
    return compiled_core, {
        "target": "OuroForCausalLM.model",
        "target_class_before_compile": original_class,
        "target_class_after_compile": type(compiled_core).__name__,
        "outer_multimodal_model_compiled": False,
        "spatial_ast_compiled": False,
        "qformer_compiled": False,
        "audio_renderer_compiled": False,
    }


def gather_reports(report: dict[str, Any], current_world: int) -> list[dict[str, Any]]:
    if current_world == 1:
        return [report]
    gathered: list[dict[str, Any] | None] = [None] * current_world
    dist.all_gather_object(gathered, report)
    return [item for item in gathered if item is not None]


def main() -> None:
    args = parse_args()
    BAT_TRAINING.validate()
    if args.steps <= 0 or args.warmup_steps < 0 or args.local_batch_size <= 0:
        raise ValueError("steps, local-batch-size must be positive and warmup-steps must be non-negative")
    if args.num_workers < 0 or args.prefetch_factor <= 0:
        raise ValueError("num-workers must be non-negative and prefetch-factor must be positive")
    output = args.output_report.expanduser().resolve()
    if str(output).replace("\\", "/").startswith("/hpc_stor03/public"):
        raise ValueError(f"Profiler output must be private: {output}")
    for path in (args.model_path, args.plugin_path, args.dataset):
        if not path.expanduser().resolve().exists():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("BAT profiling requires a submitted CUDA job")

    current_world = world_size()
    if args.attention_profile and current_world != 1:
        raise ValueError("Attention kernel audit must run single-card; use the DDP benchmark separately")
    current_rank, current_local_rank = initialize_distributed(args.expected_world_size)
    device = torch.device(f"cuda:{current_local_rank}")
    total_records = (args.warmup_steps + args.steps) * current_world * args.local_batch_size
    all_rows, manifest_read_seconds = load_manifest_rows(
        args.dataset.resolve(), args.start_index, total_records
    )
    rank_rows: list[dict[str, Any]] = []
    block = current_world * args.local_batch_size
    for step in range(args.warmup_steps + args.steps):
        start = step * block + current_rank * args.local_batch_size
        rank_rows.extend(all_rows[start : start + args.local_batch_size])

    # Keep source provenance in the profiler batches.  This does not alter the
    # model path; it only lets a later report identify a slow AudioSet/RIR
    # lookup instead of exposing only an aggregate 80-second stall.
    os.environ.setdefault("BAT_AUDIO_AUDIT", "1")
    plugin = import_plugin(args.plugin_path.resolve())
    if plugin.MODEL_TYPE != MODEL_TYPE or plugin.TEMPLATE_TYPE != TEMPLATE_TYPE:
        raise RuntimeError("BAT plugin registration constants do not match profiler")
    from swift import get_model_processor, get_template

    if current_rank == 0:
        print("========== BAT OURO PURE PIPELINE PROFILING ==========")
        print(f"[world] world_size={current_world} local_batch={args.local_batch_size} device={device}")
        print(f"[dataset] path={args.dataset} start={args.start_index} records={total_records}")
        print(f"[dataloader] workers={args.num_workers} prefetch_factor={args.prefetch_factor} persistent={args.persistent_workers}")

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
    model = inject_lora(model)
    model.train()
    causal = find_causal_model(model)
    if int(getattr(causal.config, "total_ut_steps", -1)) != 4:
        raise RuntimeError("Profiler requires Ouro total_ut_steps=4")
    if float(getattr(causal, "early_exit_threshold", -1.0)) != 1.0:
        raise RuntimeError("Profiler requires early_exit_threshold=1.0")
    if any(parameter.requires_grad for name, parameter in causal.named_parameters() if "early_exit_gate" in name):
        raise RuntimeError("Ouro early_exit_gate must remain frozen during profiling")
    causal.config.use_cache = False
    if hasattr(causal, "model") and hasattr(causal.model, "config"):
        causal.model.config.use_cache = False

    compile_report: dict[str, Any] = {
        "requested": bool(args.torch_compile),
        "enabled": False,
        "mode": args.compile_mode,
        "dynamic": bool(args.compile_dynamic),
        "wrapper_class": None,
        "repro_after_aot_disabled": False,
        "inductor_graph_repro_disabled": False,
        "dynamo_optimize_ddp_disabled": False,
    }
    if args.torch_compile:
        try:
            # If Inductor compilation fails, torch 2.11 may enter its AOT
            # compiler-repro path and invoke ``nvcc --version``.  The cluster
            # image can contain nvcc without granting execute permission,
            # which masks the real compiler error with PermissionError.  This
            # benchmark must report the real compile failure and must not
            # generate a large repro tree as a side effect.
            # PyTorch 2.11 uses TORCHDYNAMO_REPRO_AFTER (not the older
            # REPRO_AFTER_AOT spelling) to select the AOT repro wrapper.
            os.environ.pop("TORCHDYNAMO_REPRO_AFTER", None)
            os.environ.pop("TORCHDYNAMO_REPRO_AFTER_AOT", None)
            os.environ["TORCHDYNAMO_REPRO_LEVEL"] = "0"
            from torch import _dynamo
            from torch._dynamo.repro import after_aot

            _dynamo.reset()
            dynamo_config = getattr(_dynamo, "config", None)
            if dynamo_config is not None:
                # The benchmark compiles only OuroForCausalLM.model and then
                # wraps the complete multimodal model in ordinary DDP below.
                # PyTorch's Dynamo DDP optimizer is a different compilation
                # path: it partitions the already-compiled submodule and
                # assumes every FX graph output is a Tensor node.  Ouro's
                # native recurrent return structure contains auxiliary
                # Python values/lists, so that path can fail in AOTAutograd
                # with ``AttributeError: 'float' object has no attribute
                # 'meta'``.  Disable only that optimizer; DDP itself and its
                # gradient all-reduce remain enabled and are still measured.
                if hasattr(dynamo_config, "optimize_ddp"):
                    dynamo_config.optimize_ddp = False
                    compile_report["dynamo_optimize_ddp_disabled"] = True
                if hasattr(dynamo_config, "repro_after"):
                    dynamo_config.repro_after = None
                if hasattr(dynamo_config, "repro_after_aot"):
                    dynamo_config.repro_after_aot = False
                if hasattr(dynamo_config, "repro_level"):
                    dynamo_config.repro_level = 0
            compile_report["repro_after_aot_disabled"] = True

            # PyTorch 2.11's Inductor path can call save_graph_repro directly
            # while preparing every FX graph, even when repro_after is None.
            # That helper probes nvcc and is unusable in this cluster image.
            # Replace only this diagnostic helper; compilation itself remains
            # enabled and any real Inductor error is still raised normally.
            after_aot.save_graph_repro = lambda *args, **kwargs: None
            compile_report["inductor_graph_repro_disabled"] = True
        except Exception:
            pass
    if args.torch_compile:
        compile_started = time.perf_counter()
        try:
            compiled_core, target_report = compile_ouro_transformer_core(
                causal,
                mode=args.compile_mode,
                dynamic=args.compile_dynamic,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Ouro Transformer-core torch.compile setup failed: {type(exc).__name__}: {exc}"
            ) from exc
        compile_report.update({
            "enabled": True,
            "wrapper_class": type(compiled_core).__name__,
            **target_report,
            "setup_seconds": time.perf_counter() - compile_started,
        })

    communication_profiler: DDPCommunicationProfiler | None = None
    if current_world > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[current_local_rank],
            output_device=current_local_rank,
            find_unused_parameters=False,
        )
        if args.measure_ddp_communication:
            communication_profiler = DDPCommunicationProfiler(current_world)
            model.register_comm_hook(communication_profiler, communication_profiler.hook)

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
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        raise RuntimeError("Ouro tokenizer has no pad/eos token for profiling collation")

    microprofile: list[float] = []
    for row in rank_rows[: min(2, len(rank_rows))]:
        started = time.perf_counter()
        encoded = template.encode(row)
        elapsed = time.perf_counter() - started
        if "audio_waveform" not in encoded:
            raise RuntimeError("Production template did not render audio_waveform")
        microprofile.append(elapsed)

    dataset = EncodedBATDataset(rank_rows, template)
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": args.local_batch_size,
        "shuffle": False,
        "drop_last": True,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "collate_fn": make_collator(template, int(pad_token_id)),
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["persistent_workers"] = args.persistent_workers
    loader = DataLoader(**loader_kwargs)

    # Module hooks can force graph breaks in torch.compile.  The compile
    # benchmark therefore measures the complete forward/backward path, while
    # the eager profiler keeps the detailed Spatial-AST/Q-Former timers.
    base_for_hooks = causal
    if args.torch_compile:
        spatial_timer, spatial_handles = CudaRegionTimer(), []
        qformer_timer, qformer_handles = CudaRegionTimer(), []
    else:
        spatial_timer, spatial_handles = install_module_timer(base_for_hooks.spatial_ast_encoder)
        qformer_timer, qformer_handles = install_module_timer(base_for_hooks.audio_qformer)
    trainable_probe = parameter_update_probe(model)
    data_wait_values: list[float] = []
    h2d_values: list[float] = []
    forward_values: list[float] = []
    backward_values: list[float] = []
    backward_cuda_event_values: list[float] = []
    ddp_bucket_latency_sum_values: list[float] = []
    ddp_bucket_span_values: list[float] = []
    backward_compute_approx_values: list[float] = []
    spatial_values: list[float] = []
    qformer_values: list[float] = []
    step_values: list[float] = []
    losses: list[float] = []
    step_details: list[dict[str, Any]] = []
    attention_report: dict[str, Any] | None = None
    compile_first_step_seconds: float | None = None
    compile_step_counters: list[dict[str, Any]] = []
    iterator = iter(loader)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for step_index in range(args.warmup_steps + args.steps):
            step_started = time.perf_counter()
            data_started = time.perf_counter()
            batch = next(iterator)
            data_wait = time.perf_counter() - data_started
            if not torch.is_tensor(batch.get("audio_waveforms")):
                raise RuntimeError("Profiler batch is missing audio_waveforms")

            h2d_started = time.perf_counter()
            inputs = {
                key: value.to(device=device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            torch.cuda.synchronize(device)
            h2d_seconds = time.perf_counter() - h2d_started

            attention_context = None
            if args.attention_profile and step_index == args.warmup_steps:
                attention_context = AttentionKernelProfiler()
                attention_context.__enter__()
            try:
                if args.torch_compile and current_rank == 0:
                    print(f"[compile] entering_forward step={step_index}", flush=True)
                forward_timer = CudaRegionTimer()
                forward_timer.begin()
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=inputs["labels"],
                    audio_waveforms=inputs["audio_waveforms"],
                    use_cache=False,
                )
                forward_timer.finish()
                loss = outputs.loss
                if loss is None or not torch.isfinite(loss.detach().float()):
                    raise RuntimeError("Profiler observed a non-finite or missing loss")

                backward_timer = CudaRegionTimer()
                backward_wall_started = time.perf_counter()
                if communication_profiler is not None:
                    communication_profiler.begin_step(step_index)
                backward_timer.begin()
                loss.backward()
                backward_timer.finish()
                torch.cuda.synchronize(device)
                backward_plus_ddp_wall = time.perf_counter() - backward_wall_started
            finally:
                if attention_context is not None:
                    attention_context.__exit__(None, None, None)
                    attention_report = attention_context.report()
            if args.torch_compile and step_index == 0:
                compile_first_step_seconds = time.perf_counter() - step_started
            if args.torch_compile:
                cumulative = summarize_dynamo_counters(snapshot_dynamo_counters())
                previous = (
                    compile_step_counters[-1]["cumulative"]
                    if compile_step_counters
                    else {"unique_graphs": 0, "graph_break_count": 0, "calls_captured": 0}
                )
                delta = {
                    key: int(cumulative[key]) - int(previous.get(key, 0))
                    for key in cumulative
                }
                compile_step_counters.append({
                    "step_index": int(step_index),
                    "cumulative": cumulative,
                    "delta_from_previous_step": delta,
                })
                if current_rank == 0:
                    print(
                        f"[compile] completed_step={step_index} "
                        f"unique_graphs_total={cumulative['unique_graphs']} "
                        f"unique_graphs_delta={delta['unique_graphs']}",
                        flush=True,
                    )
            communication = (
                communication_profiler.finish_step()
                if communication_profiler is not None
                else {
                    "enabled": False,
                    "bucket_count": 0,
                    "bucket_latency_sum_seconds": 0.0,
                    "bucket_span_seconds": 0.0,
                    "bucket_details": [],
                }
            )
            if current_world > 1 and args.measure_ddp_communication and communication["bucket_count"] <= 0:
                raise RuntimeError(
                    "DDP communication hook observed no gradient buckets; "
                    "the backward/all-reduce split is invalid"
                )

            forward_ms = forward_timer.milliseconds()
            backward_ms = backward_timer.milliseconds()
            if step_index >= args.warmup_steps:
                source_records = [source_summary(item) for item in batch.get("bat_audio_records", [])]
                data_wait_values.append(data_wait)
                h2d_values.append(h2d_seconds)
                forward_values.append(forward_ms / 1000.0)
                backward_values.append(backward_plus_ddp_wall)
                backward_cuda_event_values.append(backward_ms / 1000.0)
                ddp_bucket_latency_sum_values.append(float(communication["bucket_latency_sum_seconds"]))
                ddp_bucket_span_values.append(float(communication["bucket_span_seconds"]))
                backward_compute_approx_values.append(
                    max(0.0, backward_plus_ddp_wall - float(communication["bucket_span_seconds"]))
                )
                if args.torch_compile:
                    spatial_values.append(0.0)
                    qformer_values.append(0.0)
                else:
                    spatial_values.append(spatial_timer.milliseconds() / 1000.0)
                    qformer_values.append(qformer_timer.milliseconds() / 1000.0)
                step_values.append(time.perf_counter() - step_started)
                losses.append(float(loss.detach().float().cpu().item()))
                step_details.append(
                    {
                        "measured_step_index": step_index - args.warmup_steps,
                        "data_wait_seconds": data_wait,
                        "host_to_device_seconds": h2d_seconds,
                        "spatial_ast_seconds": 0.0 if args.torch_compile else spatial_timer.milliseconds() / 1000.0,
                        "qformer_seconds": 0.0 if args.torch_compile else qformer_timer.milliseconds() / 1000.0,
                        "ouro_forward_seconds": forward_ms / 1000.0,
                        "backward_plus_ddp_allreduce_seconds": backward_plus_ddp_wall,
                        "backward_cuda_event_seconds": backward_ms / 1000.0,
                        "ddp_allreduce_bucket_latency_sum_seconds": float(communication["bucket_latency_sum_seconds"]),
                        "ddp_allreduce_bucket_span_seconds": float(communication["bucket_span_seconds"]),
                        "backward_compute_excluding_ddp_approx_seconds": max(
                            0.0,
                            backward_plus_ddp_wall - float(communication["bucket_span_seconds"]),
                        ),
                        "ddp_allreduce_bucket_count": int(communication["bucket_count"]),
                        "ddp_allreduce_buckets": communication["bucket_details"],
                        "step_wall_seconds": time.perf_counter() - step_started,
                        "loss": float(loss.detach().float().cpu().item()),
                        "sources": source_records,
                    }
                )

            model.zero_grad(set_to_none=True)
            if current_world > 1:
                dist.barrier()
    finally:
        for handle in spatial_handles + qformer_handles:
            handle.remove()

    parameter_probe_report = finish_parameter_update_probe(model, trainable_probe)
    if args.torch_compile:
        dynamo_counters = snapshot_dynamo_counters()
        compile_report["dynamo_counters"] = dynamo_counters
        stats = (dynamo_counters or {}).get("stats", {})
        graph_breaks = (dynamo_counters or {}).get("graph_break", {})
        compile_report["unique_graphs"] = int(stats.get("unique_graphs", 0))
        compile_report["graph_break_count"] = int(sum(graph_breaks.values()))
        compile_report["compilation_observed"] = compile_report["unique_graphs"] > 0
        compile_report["step_counters"] = compile_step_counters
    local_report = {
        "rank": current_rank,
        "local_rank": current_local_rank,
        "world_size": current_world,
        "manifest_read_seconds": manifest_read_seconds,
        "microprofile_template_encode_seconds": summarize(microprofile),
        "timings_seconds": {
            "data_wait_and_collate": summarize(data_wait_values),
            "host_to_device": summarize(h2d_values),
            "spatial_ast": summarize(spatial_values),
            "qformer": summarize(qformer_values),
            "ouro_forward": summarize(forward_values),
            "backward_plus_ddp_allreduce": summarize(backward_values),
            "backward_cuda_event_main_stream": summarize(backward_cuda_event_values),
            "ddp_allreduce_bucket_latency_sum": summarize(ddp_bucket_latency_sum_values),
            "ddp_allreduce_bucket_span": summarize(ddp_bucket_span_values),
            "backward_compute_excluding_ddp_approx": summarize(backward_compute_approx_values),
            "step_wall_from_data_to_sync": summarize(step_values),
        },
        "loss": summarize(losses),
        "slowest_steps": sorted(
            step_details, key=lambda item: float(item["step_wall_seconds"]), reverse=True
        )[:5],
        "step_details": step_details,
        "memory": {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "parameter_update_probe": parameter_probe_report,
        "compile": compile_report | {
            "first_step_wall_seconds": compile_first_step_seconds,
            "detailed_module_timers_disabled": bool(args.torch_compile),
        },
        "attention_kernel_audit": attention_report,
        "contracts": {
            "audio_token_count": EXPECTED_AUDIO_TOKENS,
            "spatial_ast_output": list(EXPECTED_SPATIAL_AST_SHAPE),
            "qformer_output": list(EXPECTED_QFORMER_SHAPE),
            "total_ut_steps": int(getattr(causal.config, "total_ut_steps", -1)),
            "early_exit_threshold": float(getattr(causal, "early_exit_threshold", -1.0)),
            "use_cache": False,
            "loRA_injected": True,
            "optimizer_step_called": False,
        },
    }
    reports = gather_reports(local_report, current_world)
    if current_rank == 0:
        output.parent.mkdir(parents=True, exist_ok=True)
        issues: list[str] = []
        if args.attention_profile and (attention_report is None or attention_report.get("status") != "ok"):
            issues.append("attention_kernel_not_identified")
        if args.torch_compile and local_report["compile"].get("compilation_observed") is False:
            issues.append("torch_compile_no_graph_observed")
        report = {
            "status": "ok" if not issues else "incomplete",
            "issues": issues,
            "scope": "pure_profile_no_optimizer_update",
            "model_path": str(args.model_path.resolve()),
            "plugin_path": str(args.plugin_path.resolve()),
            "dataset": str(args.dataset.resolve()),
            "start_index": args.start_index,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "torch_compile": bool(args.torch_compile),
            "compile_mode": args.compile_mode,
            "compile_dynamic": bool(args.compile_dynamic),
            "attention_profile": bool(args.attention_profile),
            "compile_report": local_report["compile"],
            "attention_kernel_audit": local_report["attention_kernel_audit"],
            "local_batch_size": args.local_batch_size,
            "global_batch_size": args.local_batch_size * current_world,
            "dataloader": {
                "num_workers": args.num_workers,
                "prefetch_factor": args.prefetch_factor,
                "persistent_workers": args.persistent_workers,
                "pin_memory": True,
                "shuffle": False,
            },
            "environment": {
                "python": sys.version,
                "torch": torch.__version__,
                "cuda_device": torch.cuda.get_device_name(current_local_rank),
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
                "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            },
            "ddp": {
                "enabled": current_world > 1,
                "backend": dist.get_backend() if current_world > 1 else None,
                "communication_hook_enabled": communication_profiler is not None,
                "communication_hook_semantics": "async all_reduce + divide by world_size",
                "backward_plus_ddp_wall_timer": "loss.backward() through CUDA synchronization",
                "bucket_latency_sum_caveat": "sum of Future callback latencies can overcount overlapping buckets",
                "bucket_span_caveat": "CPU Future callback span estimates the overlapping communication window",
            },
            "rank_reports": reports,
        }
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[report] {output}", flush=True)
        print(f"[ranks] {len(reports)}", flush=True)
        print(f"[rank0 timings] {json.dumps(local_report['timings_seconds'], ensure_ascii=False)}", flush=True)
        print(
            f"[rank0 ddp split] backward_plus_ddp_wall={local_report['timings_seconds']['backward_plus_ddp_allreduce']} "
            f"backward_cuda={local_report['timings_seconds']['backward_cuda_event_main_stream']} "
            f"allreduce_bucket_span={local_report['timings_seconds']['ddp_allreduce_bucket_span']} "
            f"backward_excluding_ddp_approx={local_report['timings_seconds']['backward_compute_excluding_ddp_approx']}",
            flush=True,
        )
        print(f"[rank0 slowest_steps] {json.dumps(local_report['slowest_steps'], ensure_ascii=False)}", flush=True)
        print(f"[rank0 memory] {json.dumps(local_report['memory'], ensure_ascii=False)}", flush=True)
        print(f"[update_probe] {json.dumps(parameter_probe_report, ensure_ascii=False)}", flush=True)
        print(f"[compile] {json.dumps(local_report['compile'], ensure_ascii=False)}", flush=True)
        if attention_report is not None:
            print(f"[attention] {json.dumps(attention_report, ensure_ascii=False)}", flush=True)
        print(
            f"========== BAT OURO PURE PIPELINE PROFILING {report['status'].upper()} ==========",
            flush=True,
        )

    if current_world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
