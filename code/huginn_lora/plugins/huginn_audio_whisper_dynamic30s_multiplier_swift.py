"""Swift registration for the finite globally shuffled multiplier pool."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
HUGINN_LORA_ROOT = REPO_ROOT / "code" / "huginn_lora"
BASE_PLUGIN_PATH = Path(__file__).with_name("huginn_audio_whisper_dynamic90s_swift.py")
DEFAULT_REGISTRY = (
    REPO_ROOT
    / "data/audio_swift/huginn_whisper_dynamic30s_multiplier/v1_gigaspeech_m"
    / "multiplier_pool_registry.json"
)
REGISTRY_ENV = "HUGINN_MULTIPLIER_POOL_REGISTRY"
START_POSITION_ENV = "HUGINN_MULTIPLIER_START_POSITION"
MAX_SAMPLES_ENV = "HUGINN_MULTIPLIER_MAX_SAMPLES"
RESUME_STATE_ENV = "HUGINN_AUDIO_DYNAMIC90S_TRAINING_STATS_RESUME_STATE"
BASE_START_POSITION_ENV = "HUGINN_DYNAMIC90S_MIXTURE_START_POSITION"
DATASET_NAME = "huginn_whisper_dynamic30s_multiplier"
FORMAL_PLAN_ENV = "HUGINN_MULTIPLIER_FORMAL_PLAN_PATH"
FORMAL_PLAN_FILENAME = "multiplier_formal_training_plan.json"

if str(HUGINN_LORA_ROOT) not in sys.path:
    sys.path.insert(0, str(HUGINN_LORA_ROOT))

from data_pipeline.finite_multiplier_pool import (  # noqa: E402
    POOL_ORDER,
    SAMPLER_VERSION,
    STATISTICS_VERSION,
    FiniteMultiplierPool,
    iter_multiplier_rows,
    load_multiplier_registry,
)


def _synchronize_base_plugin_environment() -> None:
    # The reused statistics callback still reads the historical mixture name.
    # Keep the finite multiplier position authoritative for both fresh and resumed runs.
    start_position = os.environ.get(START_POSITION_ENV, "0").strip() or "0"
    if int(start_position) < 0:
        raise ValueError(f"{START_POSITION_ENV} must be non-negative, got {start_position}")
    os.environ[BASE_START_POSITION_ENV] = start_position


_synchronize_base_plugin_environment()


def _load_base_plugin() -> Any:
    module_name = "huginn_audio_whisper_dynamic90s_base"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, BASE_PLUGIN_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Huginn dynamic-30s base plugin: {BASE_PLUGIN_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if sys.modules.get(module_name) is module:
            sys.modules.pop(module_name, None)
        raise
    return module


_BASE_PLUGIN = _load_base_plugin()
# Base training statistics are generic over the existing four aggregate pool
# names. This route keeps those names but gives the finite schedule its own
# checkpoint contract versions.
_BASE_PLUGIN.SAMPLER_VERSION = SAMPLER_VERSION
_BASE_PLUGIN.TRAINING_STATS_VERSION = STATISTICS_VERSION


def _registry_path(dataset_meta: Any | None = None) -> Path:
    configured = os.environ.get(REGISTRY_ENV, "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    metadata_path = getattr(dataset_meta, "dataset_path", None)
    if metadata_path:
        return Path(str(metadata_path)).expanduser().resolve()
    return DEFAULT_REGISTRY


def _integer_environment(name: str, default: int, minimum: int) -> int:
    raw = os.environ.get(name, "").strip()
    value = int(raw) if raw else int(default)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _start_position() -> int:
    return _integer_environment(START_POSITION_ENV, 0, 0)


def _max_samples() -> int | None:
    raw = os.environ.get(MAX_SAMPLES_ENV, "").strip()
    if not raw:
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{MAX_SAMPLES_ENV} must be positive when set, got {value}")
    return value


def _validate_resume_state(registry_path: Path, start_position: int) -> None:
    state_value = os.environ.get(RESUME_STATE_ENV, "").strip()
    if not state_value:
        if start_position != 0:
            raise RuntimeError(
                f"A nonzero multiplier start position requires {RESUME_STATE_ENV}: {start_position}"
            )
        return
    state_path = Path(state_value).expanduser().resolve()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    registry = load_multiplier_registry(registry_path)
    if (
        state.get("statistics_version") != STATISTICS_VERSION
        or state.get("sampler_version") != SAMPLER_VERSION
        or int(state.get("sampler_seed", -1)) != int(registry["seed"])
        or int(state.get("next_global_position", -1)) != start_position
        or int(state.get("total_samples", -1)) != start_position
    ):
        raise RuntimeError(
            f"Multiplier resume state contract mismatch: state={state_path} start={start_position}"
        )
    pools = state.get("pools")
    if not isinstance(pools, dict) or tuple(pools) != POOL_ORDER:
        raise RuntimeError(f"Multiplier resume pool set mismatch: {pools}")
    with FiniteMultiplierPool(registry_path) as schedule:
        expected_counts = schedule.pool_counts_before(start_position)
    actual_counts = {name: int(pools[name]["sample_count"]) for name in POOL_ORDER}
    expected_sizes = {name: int(registry["pools"][name]["record_count"]) for name in POOL_ORDER}
    actual_sizes = {name: int(pools[name]["pool_size"]) for name in POOL_ORDER}
    if actual_counts != expected_counts or actual_sizes != expected_sizes:
        raise RuntimeError(
            "Multiplier resume schedule differs from cumulative statistics: "
            f"actual_counts={actual_counts} expected_counts={expected_counts} "
            f"actual_sizes={actual_sizes} expected_sizes={expected_sizes}"
        )


def _install_multiplier_data_position_audit() -> None:
    def audit_multiplier_position(audio_item: Any) -> None:
        audit_dir_value = os.environ.get(_BASE_PLUGIN.DATA_POSITION_AUDIT_DIR_ENV, "").strip()
        if not audit_dir_value:
            return
        phase = os.environ.get(_BASE_PLUGIN.DATA_POSITION_AUDIT_PHASE_ENV, "").strip()
        if not phase or not isinstance(audio_item, dict):
            raise RuntimeError("Multiplier data-position audit requires a phase and dictionary audio item")
        required = (
            "global_position",
            "pool_name",
            "task",
            "uid",
            "record_index",
            "pool_occurrence_index",
            "pool_epoch",
            "pool_epoch_offset",
            "component_name",
            "replica_id",
            "schedule_slot",
        )
        missing = [name for name in required if audio_item.get(name) is None]
        if missing:
            raise RuntimeError(f"Multiplier audio provenance is incomplete: missing={missing}")
        rank = int(os.environ.get("RANK", "0"))
        payload = {
            "phase": phase,
            "rank": rank,
            **{
                name: (
                    str(audio_item[name])
                    if name in {"pool_name", "task", "uid", "component_name"}
                    else int(audio_item[name])
                )
                for name in required
            },
        }
        audit_dir = Path(audit_dir_value)
        audit_dir.mkdir(parents=True, exist_ok=True)
        path = audit_dir / f"data-{phase}-rank-{rank}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    _BASE_PLUGIN.audit_consumed_audio_position = audit_multiplier_position


_install_multiplier_data_position_audit()


def _install_multiplier_formal_callback() -> None:
    plan_value = os.environ.get(FORMAL_PLAN_ENV, "").strip()
    if not plan_value:
        return
    from transformers import Trainer, TrainerCallback

    original_init = Trainer.__init__
    if getattr(original_init, "_huginn_multiplier_formal_patched", False):
        return

    class MultiplierFormalCallback(TrainerCallback):
        _huginn_multiplier_formal_callback = True

        def __init__(self, tracked_model):
            self.tracked_model = tracked_model
            self.plan_path = Path(plan_value).expanduser().resolve()
            self.plan = json.loads(self.plan_path.read_text(encoding="utf-8"))
            if self.plan.get("plan_version") != "huginn_dynamic30s_multiplier_single_epoch_plan_v1":
                raise RuntimeError(f"Invalid multiplier formal plan: {self.plan}")
            self.gradient_audited = False
            self.finite_loss_logs = 0
            self.finite_grad_norm_logs = 0

        @staticmethod
        def _identity() -> tuple[int, int]:
            torch = _BASE_PLUGIN.torch
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                raise RuntimeError("Multiplier formal callback requires torch.distributed")
            return torch.distributed.get_rank(), torch.distributed.get_world_size()

        @staticmethod
        def _checkpoint_dir(output_dir: str, step: int) -> Path:
            direct = Path(output_dir) / f"checkpoint-{step}"
            if direct.is_dir():
                return direct
            matches = sorted(Path(output_dir).glob(f"*/checkpoint-{step}"))
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Expected one multiplier checkpoint-{step} below {output_dir}, found {matches}"
                )
            return matches[0]

        def on_train_begin(self, args, state, control, **kwargs):
            rank, world_size = self._identity()
            del rank
            global_batch = (
                world_size
                * int(args.per_device_train_batch_size)
                * int(args.gradient_accumulation_steps)
            )
            expected = {
                "sampler_version": SAMPLER_VERSION,
                "world_size": 4,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 4,
                "global_batch_size": 32,
                "total_records": int(state.max_steps) * global_batch,
                "max_steps": int(state.max_steps),
            }
            mismatches = {
                key: {"actual": self.plan.get(key), "expected": value}
                for key, value in expected.items()
                if self.plan.get(key) != value
            }
            checkpoint_steps = [int(value) for value in self.plan.get("checkpoint_steps", [])]
            if (
                mismatches
                or world_size != 4
                or global_batch != 32
                or int(state.global_step) not in {0, *checkpoint_steps[:-1]}
                or checkpoint_steps[-1] != int(state.max_steps)
            ):
                raise RuntimeError(
                    f"Multiplier formal frozen plan mismatch: mismatches={mismatches} "
                    f"state_step={state.global_step} checkpoints={checkpoint_steps}"
                )
            _BASE_PLUGIN._audit_optimizer_parameter_groups(
                self.tracked_model,
                kwargs.get("optimizer"),
                context="Multiplier formal training",
                allow_scheduled_learning_rate=True,
            )
            _BASE_PLUGIN._audit_lora_runtime_configuration(
                self.tracked_model,
                context="Multiplier formal training",
            )
            _BASE_PLUGIN._audit_formal_fsdp_topology(self.tracked_model)
            whisper_internal = _BASE_PLUGIN._whisper_gradient_checkpoint_modules(self.tracked_model)
            wrappers = _BASE_PLUGIN._audit_activation_checkpoint_wrappers(self.tracked_model)
            outer_whisper = [item for item in wrappers if item["contains_whisper_encoder"]]
            if (
                bool(getattr(args, "vit_gradient_checkpointing", False))
                or whisper_internal
                or len(outer_whisper) != 1
                or not outer_whisper[0]["path"].endswith("audio_encoder.encoder")
            ):
                raise RuntimeError(
                    f"Multiplier formal Whisper checkpointing mismatch: "
                    f"internal={whisper_internal} outer={outer_whisper}"
                )
            for class_name in _BASE_PLUGIN.FSDP_UNIT_CLASS_NAMES:
                matching = [
                    module
                    for module in self.tracked_model.modules()
                    if class_name in _BASE_PLUGIN._module_mro_names(module)
                ]
                if len(matching) != 1:
                    raise RuntimeError(
                        f"Multiplier formal expected one {class_name}, found {len(matching)}"
                    )
                _BASE_PLUGIN._assert_fsdp_reshard_state(
                    matching[0],
                    expected=class_name != "HuginnRecurrentCoreFSDPUnit",
                    context=f"Multiplier formal {class_name}",
                )
            print(
                f"[multiplier-formal-audit] plan={self.plan_path} max_steps={state.max_steps} "
                f"global_batch={global_batch} checkpoints={checkpoint_steps}",
                flush=True,
            )
            return control

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            del args, state, kwargs
            if not self.gradient_audited:
                gradients = _BASE_PLUGIN._audit_local_trainable_gradients(
                    self.tracked_model,
                    context="Multiplier formal first optimizer update",
                )
                print(f"[multiplier-formal-audit] first_update_gradients={gradients}", flush=True)
                self.gradient_audited = True
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            del args, state, kwargs
            for key in ("loss", "grad_norm"):
                if not logs or key not in logs:
                    continue
                value = float(logs[key])
                if not math.isfinite(value):
                    raise RuntimeError(f"Multiplier formal logged non-finite {key}: {value}")
                if key == "loss":
                    self.finite_loss_logs += 1
                else:
                    self.finite_grad_norm_logs += 1
            return control

        def on_step_end(self, args, state, control, **kwargs):
            del args, kwargs
            if int(state.global_step) == int(state.max_steps):
                control.should_save = True
            return control

        def on_save(self, args, state, control, **kwargs):
            del kwargs
            step = int(state.global_step)
            checkpoint_steps = [int(value) for value in self.plan["checkpoint_steps"]]
            if step not in checkpoint_steps:
                raise RuntimeError(
                    f"Multiplier formal attempted undeclared checkpoint save at {step}: {checkpoint_steps}"
                )
            rank, world_size = self._identity()
            lora_audit = _BASE_PLUGIN._audit_lora_runtime_configuration(
                self.tracked_model,
                context=f"Multiplier formal checkpoint-{step}",
            )
            role = "final" if step == int(self.plan["max_steps"]) else "scheduled"
            contract = _BASE_PLUGIN._build_dynamic30s_training_runtime_contract(
                phase=("multiplier_formal_final" if role == "final" else "multiplier_formal_checkpoint"),
                global_step=step,
                world_size=world_size,
                lora_runtime_audit=lora_audit,
                formal_training={
                    "checkpoint_role": role,
                    "checkpoint_step": step,
                    "plan_version": self.plan["plan_version"],
                    "step_policy": self.plan["step_policy"],
                    "sampler_version": self.plan["sampler_version"],
                    "sampler_seed": int(self.plan["seed"]),
                    "max_steps": int(self.plan["max_steps"]),
                    "total_scheduled_samples": int(self.plan["total_records"]),
                    "global_batch_size": int(self.plan["global_batch_size"]),
                    "checkpoint_steps": checkpoint_steps,
                    "schedule_sha256": self.plan["schedule_sha256"],
                },
            )
            if rank == 0:
                checkpoint_dir = self._checkpoint_dir(args.output_dir, step)
                _BASE_PLUGIN._write_json_atomic(
                    checkpoint_dir / _BASE_PLUGIN.TRAINING_RUNTIME_CONTRACT_FILENAME,
                    contract,
                )
                _BASE_PLUGIN._write_json_atomic(
                    checkpoint_dir / FORMAL_PLAN_FILENAME,
                    self.plan,
                )
                print(
                    f"[multiplier-formal-checkpoint] role={role} step={step} path={checkpoint_dir}",
                    flush=True,
                )
            return control

        def on_train_end(self, args, state, control, **kwargs):
            del args, kwargs
            if (
                int(state.global_step) != int(self.plan["max_steps"])
                or not self.gradient_audited
                or self.finite_loss_logs <= 0
                or self.finite_grad_norm_logs <= 0
            ):
                raise RuntimeError(
                    f"Multiplier formal run ended without complete audits: step={state.global_step} "
                    f"gradients={self.gradient_audited} losses={self.finite_loss_logs} "
                    f"grad_norms={self.finite_grad_norm_logs}"
                )
            return control

    def init_with_multiplier_formal(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not any(
            getattr(callback, "_huginn_multiplier_formal_callback", False)
            for callback in self.callback_handler.callbacks
        ):
            self.add_callback(MultiplierFormalCallback(self.model))

    init_with_multiplier_formal._huginn_multiplier_formal_patched = True
    Trainer.__init__ = init_with_multiplier_formal


_install_multiplier_formal_callback()


def build_dataset(registry_path: str | Path):
    from datasets import IterableDataset

    resolved = Path(registry_path).expanduser().resolve()
    registry = load_multiplier_registry(resolved)
    start_position = _start_position()
    if start_position > int(registry["total_records"]):
        raise ValueError(
            f"Multiplier start position exceeds the finite epoch: "
            f"start={start_position} total={registry['total_records']}"
        )
    _validate_resume_state(resolved, start_position)
    configured_max = _max_samples()
    remaining = int(registry["total_records"]) - start_position
    max_samples = remaining if configured_max is None else min(remaining, configured_max)
    if max_samples <= 0:
        raise ValueError(f"Multiplier dataset has no remaining samples at position {start_position}")
    return IterableDataset.from_generator(
        iter_multiplier_rows,
        gen_kwargs={
            "registry_path": str(resolved),
            "start_position": start_position,
            "max_samples": max_samples,
        },
    )


def _register_dataset() -> None:
    try:
        from swift.dataset import register_dataset
    except ImportError:
        from swift.dataset.register import register_dataset  # type: ignore
    try:
        from swift.dataset.register import DatasetMeta
    except ImportError:
        from swift.llm import DatasetMeta  # type: ignore
    from swift.dataset.loader import DatasetLoader

    registry = _registry_path()

    class HuginnMultiplierLoader(DatasetLoader):
        def load(self, dataset_syntax=None, dataset_meta=None, *, use_hf=None):
            del dataset_syntax, use_hf
            path = _registry_path(dataset_meta)
            dataset = build_dataset(path)
            print(
                "[HuginnMultiplier] loaded finite IterableDataset "
                f"registry={path} start={_start_position()} max_samples={_max_samples() or 'remaining'} "
                f"sampler={SAMPLER_VERSION}",
                flush=True,
            )
            return dataset

    metadata = DatasetMeta(
        dataset_path=str(registry),
        dataset_name=DATASET_NAME,
        loader=HuginnMultiplierLoader,
    )
    try:
        register_dataset(metadata, exist_ok=True)
    except TypeError as exc:
        if "exist_ok" not in str(exc):
            raise
        register_dataset(metadata)
    print(f"[HuginnMultiplier] registered dataset path={registry} name={DATASET_NAME}", flush=True)


_register_dataset()
