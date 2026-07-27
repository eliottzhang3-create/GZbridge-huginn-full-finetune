"""Read-only preflight for torchprofile and the current Dynamic-90s route.

This script deliberately does not load LoSATok/Huginn weights, read ACAVCAPS
tar files, initialize a process group, or start training.  It validates the
installed torchprofile API, a tiny supported operator graph, torch.profiler
availability, and the shell-level Dynamic-90s/FSDP2 configuration used by the
quarter-ACAVCAPS training script.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRAINING_SCRIPT = (
    REPO_ROOT / "code" / "huginn_lora" / "scripts"
    / "train_acavcaps_wds_huginn_losatok_dynamic90s_quarter_fsdp2_5090.sh"
)
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "huginn-audio-losatok-v1"
DEFAULT_PLUGIN = REPO_ROOT / "code" / "huginn_lora" / "plugins" / "huginn_losatok_swift.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training_script", default=str(DEFAULT_TRAINING_SCRIPT))
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--plugin", default=str(DEFAULT_PLUGIN))
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on every missing expected configuration marker instead of reporting warnings.",
    )
    return parser.parse_args()


def version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "<not-installed>"


def first_installed_version(*packages: str) -> str:
    for package in packages:
        value = version(package)
        if value != "<not-installed>":
            return f"{package}=={value}"
    return "<not-installed>"


def safe_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [safe_json(item) for item in value]
    return repr(value)


def load_plugin(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"LoSATok plugin does not exist: {path}")
    spec = importlib.util.spec_from_file_location("torchprofile_losatok_plugin", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import LoSATok plugin: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inspect_torchprofile() -> dict[str, Any]:
    import torch
    import torch.nn as nn
    import torchprofile

    profile_macs = getattr(torchprofile, "profile_macs", None)
    if not callable(profile_macs):
        raise RuntimeError("torchprofile.profile_macs is unavailable")
    profile_operators = getattr(torchprofile, "profile_operators", None)

    result: dict[str, Any] = {
        "module": str(getattr(torchprofile, "__file__", "<unknown>")),
        "distribution_version": version("torchprofile"),
        "module_version": getattr(torchprofile, "__version__", "<no __version__>"),
        "profile_macs_signature": str(inspect.signature(profile_macs)),
        "profile_operators_available": callable(profile_operators),
        "profile_operators_signature": str(inspect.signature(profile_operators))
        if callable(profile_operators)
        else None,
    }

    toy = nn.Sequential(
        nn.Conv1d(4, 8, kernel_size=3, padding=1),
        nn.SiLU(),
        nn.Conv1d(8, 2, kernel_size=1),
    ).eval()
    toy_input = torch.randn(2, 4, 16)
    with torch.no_grad():
        toy_output = toy(toy_input)
    result["toy_output_shape"] = list(toy_output.shape)
    try:
        with torch.no_grad():
            toy_macs = profile_macs(toy, args=(toy_input,))
        result["toy_profile_macs"] = safe_json(toy_macs)
        result["toy_profile_macs_status"] = "PASS"
    except Exception as exc:  # noqa: BLE001 - this is the compatibility report
        result["toy_profile_macs_status"] = "FAIL"
        result["toy_profile_macs_error"] = f"{type(exc).__name__}: {exc}"

    if callable(profile_operators):
        try:
            with torch.no_grad():
                operators = profile_operators(toy, args=(toy_input,))
            result["toy_profile_operators"] = safe_json(operators)
            result["toy_profile_operators_status"] = "PASS"
        except Exception as exc:  # noqa: BLE001 - optional API report
            result["toy_profile_operators_status"] = "FAIL"
            result["toy_profile_operators_error"] = f"{type(exc).__name__}: {exc}"

    return result


def inspect_torch_profiler() -> dict[str, Any]:
    import torch

    profiler = getattr(torch, "profiler", None)
    result: dict[str, Any] = {"available": profiler is not None}
    if profiler is None:
        return result
    activities = getattr(profiler, "ProfilerActivity", None)
    result["cpu_activity"] = getattr(activities, "CPU", None) is not None
    result["cuda_activity"] = getattr(activities, "CUDA", None) is not None
    result["profile_callable"] = callable(getattr(profiler, "profile", None))
    result["schedule_callable"] = callable(getattr(profiler, "schedule", None))
    return result


def inspect_runtime_config(model_dir: Path, plugin_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"Model config does not exist: {model_dir / 'config.json'}")

    # configure_audio_compressor is environment-controlled.  Enable only for
    # this inspection so the model's runtime dynamic settings are observable.
    previous = os.environ.get("HUGINN_LOSATOK_DYNAMIC_AUDIO_TOKENS")
    os.environ["HUGINN_LOSATOK_DYNAMIC_AUDIO_TOKENS"] = "1"
    try:
        plugin = load_plugin(plugin_path)
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
        configure = getattr(plugin, "configure_audio_compressor", None)
        if not callable(configure):
            raise RuntimeError("LoSATok plugin does not expose configure_audio_compressor")
        config = configure(config)
        result["runtime_config"] = {
            "audio_dynamic_tokens": bool(config.audio_dynamic_tokens),
            "audio_max_token_count": int(config.audio_max_token_count),
            "audio_compressor_kernel_size": int(config.audio_compressor_kernel_size),
            "audio_compressor_stride": int(config.audio_compressor_stride),
            "audio_target_token_count": int(config.audio_target_token_count),
            "audio_encoder_hidden_size": int(config.audio_encoder_hidden_size),
            "audio_projector_hidden_size": int(config.audio_projector_hidden_size),
            "n_embd": int(config.n_embd),
            "block_size": int(config.block_size),
        }
    finally:
        if previous is None:
            os.environ.pop("HUGINN_LOSATOK_DYNAMIC_AUDIO_TOKENS", None)
        else:
            os.environ["HUGINN_LOSATOK_DYNAMIC_AUDIO_TOKENS"] = previous

    runtime = result["runtime_config"]
    expected = {
        "audio_dynamic_tokens": True,
        "audio_max_token_count": 375,
        "audio_compressor_kernel_size": 11,
        "audio_compressor_stride": 6,
        "audio_target_token_count": 32,
    }
    result["runtime_config_checks"] = {
        key: {"expected": value, "actual": runtime[key], "pass": runtime[key] == value}
        for key, value in expected.items()
    }
    return result


def inspect_training_script(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Training script does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    markers = {
        "dynamic_env": "export HUGINN_LOSATOK_DYNAMIC_AUDIO_TOKENS=1",
        "world_size": "WORLD_SIZE=2",
        "micro_batch": "MICRO_BATCH_SIZE=4",
        "gradient_accumulation": "GRADIENT_ACCUMULATION_STEPS=4",
        "buffer_size": 'ACAVCAPS_WDS_BUFFER_SIZE:-512',
        "fsdp_full_shard": '"fsdp":"full_shard auto_wrap"',
        "fsdp_version_2": '"fsdp_version":2',
        "reshard_after_forward": '"reshard_after_forward":true',
        "sharded_state_dict": '"state_dict_type":"SHARDED_STATE_DICT"',
        "streaming": "--streaming true",
        "dataset_shuffle_false": "--dataset_shuffle false",
        "dataloader_shuffle_false": "--train_dataloader_shuffle false",
        "text_max_length": "--max_length 192",
        "dataloader_workers_zero": "--dataloader_num_workers 0",
        "pin_memory_false": "--dataloader_pin_memory false",
        "save_only_model_false": "--save_only_model false",
        "checkpoint_audit": "--require_complete",
    }
    checks = {name: marker in text for name, marker in markers.items()}
    # The script's comments and log line are useful evidence but must not be
    # mistaken for executable configuration checks.
    result = {
        "path": str(path),
        "sha256": __import__("hashlib").sha256(text.encode("utf-8")).hexdigest(),
        "line_count": len(text.splitlines()),
        "checks": checks,
        "missing_checks": [name for name, passed in checks.items() if not passed],
    }
    batch_match = re.search(
        r"WORLD_SIZE=(\d+).*?MICRO_BATCH_SIZE=(\d+).*?GRADIENT_ACCUMULATION_STEPS=(\d+)",
        text,
        flags=re.S,
    )
    if batch_match:
        world, micro, accumulation = (int(value) for value in batch_match.groups())
        result["derived_batch"] = {
            "world_size": world,
            "micro_batch": micro,
            "gradient_accumulation": accumulation,
            "global_effective_batch": world * micro * accumulation,
        }
    return result


def main() -> int:
    args = parse_args()
    print("========== TORCHPROFILE / DYNAMIC FSDP2 PREFLIGHT ==========")
    print(f"[context] python={sys.version.split()[0]} platform={platform.platform()}")
    print(f"[context] repo_root={REPO_ROOT}")

    import torch

    print(
        f"[torch] version={torch.__version__} cuda={torch.version.cuda} "
        f"cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}"
    )
    print(
        f"[packages] torchprofile={version('torchprofile')} accelerate={version('accelerate')} "
        f"transformers={version('transformers')} swift={first_installed_version('ms-swift', 'swift')} "
        f"webdataset={version('webdataset')}"
    )

    torchprofile_result = inspect_torchprofile()
    print("========== TORCHPROFILE API ==========")
    print(json.dumps(torchprofile_result, ensure_ascii=False, sort_keys=True, indent=2))

    profiler_result = inspect_torch_profiler()
    print("========== TORCH PROFILER API ==========")
    print(json.dumps(profiler_result, ensure_ascii=False, sort_keys=True, indent=2))

    runtime_result = inspect_runtime_config(Path(args.model_dir).expanduser().resolve(), Path(args.plugin).expanduser().resolve())
    print("========== DYNAMIC LOSATOK RUNTIME CONFIG ==========")
    print(json.dumps(runtime_result, ensure_ascii=False, sort_keys=True, indent=2))

    script_result = inspect_training_script(Path(args.training_script).expanduser().resolve())
    print("========== CURRENT TRAINING SCRIPT CONFIG ==========")
    print(json.dumps(script_result, ensure_ascii=False, sort_keys=True, indent=2))

    hard_failures: list[str] = []
    if torchprofile_result.get("toy_profile_macs_status") != "PASS":
        hard_failures.append("torchprofile profile_macs toy graph")
    for name, check in runtime_result.get("runtime_config_checks", {}).items():
        if not check["pass"]:
            hard_failures.append(f"runtime config {name}")
    if script_result.get("missing_checks"):
        hard_failures.extend(f"training script marker {name}" for name in script_result["missing_checks"])
    if not profiler_result.get("available") or not profiler_result.get("profile_callable"):
        hard_failures.append("torch.profiler API")

    status = "PASS" if not hard_failures else "FAIL"
    print("========== PREFLIGHT RESULT ==========")
    print(f"[result] status={status}")
    if hard_failures:
        print(f"[result] failures={hard_failures}")
        if args.strict:
            return 1
    print("[result] no_model_load=true no_checkpoint_load=true no_tar_scan=true training_started=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
