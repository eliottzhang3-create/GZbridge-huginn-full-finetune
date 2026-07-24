"""Swift registration for frozen LoSATok + trainable Huginn audio alignment.

LoSATok receives an individual unpadded 16 kHz waveform for every item.  Its
official encoder does not consume an attention mask, so the template keeps the
real waveform lengths and the model slices each padded batch item before LoSATok
is called.  This prevents padded zeros from changing the audio representation.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import traceback
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from swift.model import Model, ModelGroup, ModelLoader, ModelMeta, register_model

try:
    from swift.model import MultiModelKeys, register_model_arch
except ImportError:
    from swift.llm import MultiModelKeys, register_model_arch  # type: ignore

try:
    from swift.template import StdTemplateInputs, Template, TemplateMeta, register_template
except ImportError:
    from swift.llm import StdTemplateInputs, Template, TemplateMeta, register_template  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[3]
AUDIO_MODEL_DIR = REPO_ROOT / "models" / "huginn-audio-losatok-v1"
HUGINN_MODEL_DIR = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/huginn-0125")
LOSATOK_ROOT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/LoSATok")
LOSATOK_CODE_DIR = REPO_ROOT / "code" / "huginn_lora" / "LosatokCode"

MODEL_TYPE = "huginn_losatok_raven"
MODEL_ARCH_NAME = "huginn_audio_losatok"
TEMPLATE_TYPE = "huginn_losatok_text"
DEFAULT_SAMPLE_RATE = 16000
DYNAMIC_AUDIO_TOKENS_ENV = "HUGINN_LOSATOK_DYNAMIC_AUDIO_TOKENS"
DYNAMIC_AUDIO_TOKENS_ENABLED = os.environ.get(DYNAMIC_AUDIO_TOKENS_ENV, "").strip().lower() in {"1", "true", "yes"}
DEFAULT_MAX_AUDIO_SECONDS = 90.0 if DYNAMIC_AUDIO_TOKENS_ENABLED else 30.0
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that can understand audio and respond accurately."
ALIGNER_PREFIXES = ("temporal_compressor", "audio_projector", "audio_boundary_embeddings")
INIT_ALIGNER_CHECKPOINT_ENV = "HUGINN_LOSATOK_INIT_ALIGNER_CHECKPOINT"
FORCE_ALIGNER_TRAINABLE_ENV = "HUGINN_LOSATOK_FORCE_ALIGNER_TRAINABLE"
FSDP2_NONPERSISTENT_ROPE_ENV = "HUGINN_AUDIO_FSDP2_NONPERSISTENT_ROPE"
TRAIN_CHAIN_AUDIT_ENV = "HUGINN_LOSATOK_TRAIN_CHAIN_AUDIT"
FSDP_SAVE_DEBUG_ENV = "HUGINN_LOSATOK_FSDP_SAVE_DEBUG"
PEFT_ALIGNER_MODULES_TO_SAVE_ENV = "HUGINN_LOSATOK_PEFT_ALIGNER_MODULES_TO_SAVE"
DYNAMIC_ALIGNER_TRAINABLE_PARAMETERS = 62_953_248
HUGINN_LORA_TRAINABLE_PARAMETERS = 12_541_440


def _requested(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


def configure_audio_compressor(config):
    """Enable the new variable-length 90-second LoSATok alignment route."""
    if _requested(DYNAMIC_AUDIO_TOKENS_ENV):
        config.audio_dynamic_tokens = True
        config.audio_max_token_count = 375
        config.audio_compressor_kernel_size = 11
        config.audio_compressor_stride = 6
    return config


def enable_fsdp2_nonpersistent_rope_buffer(model: torch.nn.Module) -> None:
    """Make Huginn and the frozen LoSATok stack compatible with Accelerate FSDP2.

    With ``cpu_ram_efficient_loading=True``, Accelerate 1.13's FSDP2 loader
    reconstructs the model from ``model.state_dict()`` and expects every
    persistent entry to be a sharded DTensor. Huginn's ``freqs_cis`` and any
    buffers inside the dynamically loaded official LoSATok stack are ordinary
    Tensors rather than sharded parameters, so they do not have ``device_mesh``.
    Keeping all non-empty buffers non-persistent removes them from that
    state-dict reload and lets Accelerate preserve and re-register them through
    its dedicated non-persistent-buffer path instead.

    The behavior is opt-in so evaluation and non-FSDP2 runs retain the normal
    persistent-buffer semantics. The four-GPU FSDP2 smoke script enables it
    through ``HUGINN_AUDIO_FSDP2_NONPERSISTENT_ROPE=1``.
    """
    requested = os.environ.get(FSDP2_NONPERSISTENT_ROPE_ENV, "").strip().lower()
    if requested not in {"1", "true", "yes"}:
        return
    if "freqs_cis" not in model._buffers:
        raise RuntimeError("FSDP2 compatibility requested but Huginn has no freqs_cis buffer")
    marked = []
    nonpersistent_batchnorm_counters = []
    for module_name, module in model.named_modules():
        for buffer_name, buffer in module._buffers.items():
            if buffer is None:
                continue
            module._non_persistent_buffers_set.add(buffer_name)
            marked.append(f"{module_name}.{buffer_name}" if module_name else buffer_name)
        if (
            isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            and module._buffers.get("num_batches_tracked") is not None
        ):
            # Keep the registered tensor so ordinary recursive load_state_dict
            # calls can access BatchNorm.num_batches_tracked. The FSDP2 loader
            # patch below filters only its transient state-dict key.
            nonpersistent_batchnorm_counters.append(
                f"{module_name}.num_batches_tracked" if module_name else "num_batches_tracked"
            )
    print(
        "[HuginnLoSATokSwift] FSDP2 compatibility: marked non-persistent buffers "
        f"count={len(marked)} names={marked} "
        f"nonpersistent_batchnorm_counters={nonpersistent_batchnorm_counters}"
    )


def patch_accelerate_fsdp2_state_dict_key_alignment() -> None:
    """Filter stale BatchNorm counters from both FSDP2 state-dict branches.

    Accelerate 1.13 pairs ``full_sd.items()`` with
    ``meta_sharded_sd.values()`` by position on rank 0, while non-main ranks
    iterate ``meta_sharded_sd`` directly. FSDP can re-expose a non-persistent
    ``num_batches_tracked`` buffer, and its metadata-free ``sharded_sd`` also
    makes PyTorch BatchNorm's legacy loader synthesize that counter again. The
    underlying module then rejects the synthesized key because the buffer is
    non-persistent. Filter the counter from every FSDP-generated state dict and
    temporarily disable BatchNorm running-stat tracking only during the load so
    the legacy compatibility path cannot recreate it.

    LoSATok is permanently frozen and forced to eval mode, so this training-only
    BatchNorm counter has no effect on encoder outputs. Running means and
    variances remain untouched. The patch is opt-in and only installed for the
    existing Huginn FSDP2 compatibility environment.
    """
    requested = os.environ.get(FSDP2_NONPERSISTENT_ROPE_ENV, "").strip().lower()
    if requested not in {"1", "true", "yes"}:
        return
    try:
        from accelerate.utils import fsdp_utils
    except ImportError:
        return
    original = fsdp_utils.fsdp2_load_full_state_dict
    if getattr(original, "_huginn_key_alignment_patched", False):
        return

    def patched(accelerator, model, full_sd, cpu_offload=False):
        def is_stale_batchnorm_counter(name: str) -> bool:
            return name == "num_batches_tracked" or name.endswith(".num_batches_tracked")

        removed_full_keys = sorted(name for name in full_sd if is_stale_batchnorm_counter(name))
        if removed_full_keys:
            full_sd = full_sd.copy()
            for name in removed_full_keys:
                full_sd.pop(name, None)

        removed_meta_keys: set[str] = set()

        def filter_state_dict(module, state_dict, prefix, local_metadata):
            del module, prefix, local_metadata
            stale_keys = [name for name in state_dict if is_stale_batchnorm_counter(name)]
            for name in stale_keys:
                state_dict.pop(name, None)
                removed_meta_keys.add(name)

        batchnorm_tracking_states = []
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                batchnorm_tracking_states.append((module, module.track_running_stats))
                module.track_running_stats = False

        hook = model.register_state_dict_post_hook(filter_state_dict)
        try:
            return original(accelerator, model, full_sd, cpu_offload=cpu_offload)
        finally:
            hook.remove()
            for module, track_running_stats in batchnorm_tracking_states:
                module.track_running_stats = track_running_stats
            if removed_full_keys or removed_meta_keys:
                print(
                    "[HuginnLoSATokSwift] FSDP2 removed stale BatchNorm counters "
                    f"rank={os.environ.get('RANK', '0')} full_sd={removed_full_keys} "
                    f"meta_sharded_sd={sorted(removed_meta_keys)}"
                )
            print(
                "[HuginnLoSATokSwift] FSDP2 BatchNorm load guard "
                f"rank={os.environ.get('RANK', '0')} modules={len(batchnorm_tracking_states)} restored=true"
            )

    patched._huginn_key_alignment_patched = True
    fsdp_utils.fsdp2_load_full_state_dict = patched
    print("[HuginnLoSATokSwift] installed Accelerate FSDP2 state-dict key-alignment patch")


def _fsdp_save_state_groups(state_dict: dict[str, Any]) -> dict[str, list[str]]:
    """Classify the exact state entries returned to Accelerate's FSDP writer.

    This function deliberately inspects keys only.  It does not materialize,
    clone, move, or otherwise alter any sharded tensor in a checkpoint save.
    """
    groups: dict[str, list[str]] = {"lora": [], "aligner": [], "other": []}
    for raw_key in state_dict:
        key = str(raw_key)
        aliases = checkpoint_key_aliases(key)
        if any(".lora_A." in alias or ".lora_B." in alias for alias in aliases):
            groups["lora"].append(key)
        elif any(alias.startswith(ALIGNER_PREFIXES) for alias in aliases):
            groups["aligner"].append(key)
        else:
            groups["other"].append(key)
    return groups


def patch_accelerate_fsdp2_save_state_audit() -> None:
    """Opt-in trace of the state dict *before* Accelerate writes FSDP DCP.

    The current failure can arise in only two places: PEFT may omit the
    aligner from the state dict it gives Accelerate, or DCP may omit an entry
    that it receives.  Accelerate's DCP writer is deterministic and serializes
    precisely the mapping returned by ``_get_model_state_dict``.  Logging that
    mapping therefore distinguishes the two cases without changing the save.

    Enabled only for the dedicated debug smoke via
    ``HUGINN_LOSATOK_FSDP_SAVE_DEBUG=1``.  It must never be set for a formal
    run merely to avoid unnecessary diagnostic output.
    """
    if not _requested(FSDP_SAVE_DEBUG_ENV):
        return
    try:
        from accelerate.utils import fsdp_utils
    except ImportError:
        return
    original = fsdp_utils._get_model_state_dict
    if getattr(original, "_huginn_losatok_save_audit_patched", False):
        return

    def patched(model, adapter_only=False, sd_options=None):
        state_dict = original(model, adapter_only=adapter_only, sd_options=sd_options)
        if os.environ.get("RANK", "0") == "0":
            groups = _fsdp_save_state_groups(state_dict)
            caller_frames = []
            for frame in traceback.extract_stack(limit=48):
                normalized_path = frame.filename.replace("\\", "/")
                if any(
                    marker in normalized_path
                    for marker in ("/swift/", "/transformers/", "/accelerate/", "/peft/")
                ):
                    caller_frames.append(f"{normalized_path}:{frame.lineno}:{frame.name}")
            print(
                "[HuginnLoSATokSwift] FSDP2 pre-DCP save-state audit "
                f"adapter_only={adapter_only} model_type={type(model)} tensor_count={len(state_dict)} "
                f"lora={len(groups['lora'])} aligner={len(groups['aligner'])} "
                f"other={len(groups['other'])} "
                f"aligner_preview={groups['aligner'][:8]} "
                f"other_preview={groups['other'][:8]} caller_frames={caller_frames[-12:]}"
            )
        return state_dict

    patched._huginn_losatok_save_audit_patched = True
    fsdp_utils._get_model_state_dict = patched
    print("[HuginnLoSATokSwift] installed Accelerate FSDP2 pre-DCP save-state audit")


def _decode_with_ffmpeg(path: Path, target_sr: int) -> torch.Tensor:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is unavailable and torchaudio failed to decode the audio file")
    command = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(path),
        "-f", "f32le", "-ac", "1", "-ar", str(target_sr), "pipe:1",
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {path}: {result.stderr.decode(errors='replace')}")
    values = np.frombuffer(result.stdout, dtype=np.float32).copy()
    if values.size == 0:
        raise RuntimeError(f"ffmpeg decoded no samples from {path}")
    return torch.from_numpy(values)


def load_audio_16k(path: Path) -> torch.Tensor:
    """Decode to mono 16 kHz and retain the configured leading audio window."""
    if not path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {path}")
    try:
        audio_load = getattr(torchaudio, "load", None)
        if not callable(audio_load):
            raise RuntimeError("installed torchaudio exposes no top-level load API")
        waveform, source_sr = audio_load(str(path))
        waveform = waveform.mean(dim=0)
        if source_sr != DEFAULT_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, source_sr, DEFAULT_SAMPLE_RATE)
    except Exception as torchaudio_error:
        try:
            waveform = _decode_with_ffmpeg(path, DEFAULT_SAMPLE_RATE)
        except Exception as ffmpeg_error:
            raise RuntimeError(
                f"Unable to decode {path}; torchaudio={torchaudio_error}; ffmpeg={ffmpeg_error}"
            ) from ffmpeg_error
    max_samples = int(DEFAULT_SAMPLE_RATE * DEFAULT_MAX_AUDIO_SECONDS)
    waveform = waveform[:max_samples].contiguous().to(dtype=torch.float32)
    if waveform.numel() == 0:
        raise ValueError(f"Audio file decoded to an empty waveform: {path}")
    return waveform


def decode_audio_bytes_16k(audio_bytes: bytes, source_label: str) -> torch.Tensor:
    """Decode MMAU's embedded media bytes through ffmpeg into the LoSATok input form."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to decode embedded MMAU audio")
    command = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
        "-f", "f32le", "-ac", "1", "-ar", str(DEFAULT_SAMPLE_RATE), "pipe:1",
    ]
    result = subprocess.run(command, input=audio_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to decode embedded audio {source_label}: {result.stderr.decode(errors='replace')}")
    waveform = torch.from_numpy(np.frombuffer(result.stdout, dtype=np.float32).copy())
    waveform = waveform[:int(DEFAULT_SAMPLE_RATE * DEFAULT_MAX_AUDIO_SECONDS)].contiguous()
    if waveform.numel() == 0:
        raise ValueError(f"Embedded audio decoded to an empty waveform: {source_label}")
    return waveform


class HuginnLoSATokProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.sampling_rate = DEFAULT_SAMPLE_RATE

    def __getattr__(self, name: str):
        return getattr(self.tokenizer, name)


def build_processor() -> HuginnLoSATokProcessor:
    tokenizer = AutoTokenizer.from_pretrained(HUGINN_MODEL_DIR, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return HuginnLoSATokProcessor(tokenizer)


def classify_missing_keys(missing_keys: list[str]) -> dict[str, list[str]]:
    groups = {"audio_encoder": [], "aligner": [], "llm": [], "other": []}
    for key in missing_keys:
        if key.startswith("audio_encoder."):
            groups["audio_encoder"].append(key)
        elif key.startswith(ALIGNER_PREFIXES):
            groups["aligner"].append(key)
        elif key.startswith(("transformer.", "lm_head.")):
            groups["llm"].append(key)
        else:
            groups["other"].append(key)
    return groups


def print_missing_summary(missing_keys: list[str], unexpected_keys: list[str]) -> None:
    print(f"[HuginnLoSATokSwift] backbone_load missing={len(missing_keys)} unexpected={len(unexpected_keys)}")
    for group, keys in classify_missing_keys(missing_keys).items():
        print(f"[HuginnLoSATokSwift] missing_group[{group}]={len(keys)}")
        for key in keys[:5]:
            print(f"  - {key}")


def checkpoint_key_aliases(key: str) -> list[str]:
    """Normalize PEFT/Trainer wrappers around a tensor name."""
    aliases = {key}
    changed = True
    while changed:
        changed = False
        for alias in list(aliases):
            for prefix in ("base_model.model.", "base_model.", "model.", "module."):
                if alias.startswith(prefix):
                    stripped = alias[len(prefix):]
                    if stripped not in aliases:
                        aliases.add(stripped)
                        changed = True
    normalized = set()
    for alias in aliases:
        normalized.add(alias)
        normalized.add(alias.replace(".modules_to_save.default.", "."))
        normalized.add(alias.replace(".original_module.", "."))
    return list(normalized)


def read_tensor_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return {key: handle.get_tensor(key) for key in handle.keys()}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint tensor file is not a state dict: {path}")
    return {key: value for key, value in payload.items() if isinstance(key, str) and torch.is_tensor(value)}


def load_initial_aligner_state(model: torch.nn.Module, checkpoint_dir: Path) -> dict[str, Any]:
    """Restore the separately trained aligner before PEFT loads the LoRA adapter."""
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Initial LoSATok aligner checkpoint does not exist: {checkpoint_dir}")
    state_path = checkpoint_dir / "vit.safetensors"
    if not state_path.is_file():
        raise FileNotFoundError(f"Initial LoSATok checkpoint has no vit.safetensors: {checkpoint_dir}")

    target_state = model.state_dict()
    canonical_targets: dict[str, str] = {}
    expected_targets: set[str] = set()
    for target_key in target_state:
        for alias in checkpoint_key_aliases(target_key):
            if alias.startswith(ALIGNER_PREFIXES):
                canonical_targets.setdefault(alias, target_key)
                expected_targets.add(target_key)

    selected: dict[str, torch.Tensor] = {}
    source_keys: list[str] = []
    for source_key, tensor in read_tensor_state_dict(state_path).items():
        for alias in checkpoint_key_aliases(source_key):
            target_key = canonical_targets.get(alias)
            if target_key is None or target_state[target_key].shape != tensor.shape:
                continue
            selected[target_key] = tensor
            source_keys.append(source_key)
            break

    missing_targets = sorted(expected_targets - set(selected))
    if missing_targets:
        raise RuntimeError(
            "Initial LoSATok checkpoint did not restore every aligner tensor; "
            f"missing={missing_targets}"
        )
    boundary_targets = [
        key for key in selected
        if key.endswith((".audio_bos", ".audio_eos")) or key in {"audio_bos", "audio_eos"}
    ]
    if len(boundary_targets) != 2:
        raise RuntimeError(
            "Initial LoSATok checkpoint is missing audio boundary embeddings; "
            f"restored={boundary_targets}"
        )

    load_result = model.load_state_dict(selected, strict=False)
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "loaded_aligner_tensor_count": len(selected),
        "restored_boundary_embeddings": boundary_targets,
        "source_key_preview": source_keys[:20],
        "missing_key_count": len(load_result.missing_keys),
        "unexpected_key_count": len(load_result.unexpected_keys),
    }


def force_aligner_trainable(model: torch.nn.Module) -> None:
    if not _requested(FORCE_ALIGNER_TRAINABLE_ENV):
        return
    audio_model = next(
        (module for module in model.modules() if all(hasattr(module, attr) for attr in (
            "audio_encoder", "temporal_compressor", "audio_projector"))),
        None,
    )
    if audio_model is None:
        raise RuntimeError("Unable to find Huginn LoSATok base model after adapter restoration")
    for name in ALIGNER_PREFIXES:
        module = getattr(audio_model, name, None)
        if module is not None:
            module.requires_grad_(True)
    frozen_count = sum(parameter.numel() for parameter in audio_model.audio_encoder.parameters() if parameter.requires_grad)
    if frozen_count:
        raise RuntimeError(f"LoSATok must remain frozen after adapter restore, found {frozen_count} trainable params")
    trainable = sum(
        parameter.numel() for name, parameter in audio_model.named_parameters()
        if parameter.requires_grad and name.startswith(ALIGNER_PREFIXES)
    )
    print(f"[HuginnLoSATokSwift] restored_aligner_trainable_parameters={trainable}")


def _fsdp_dcp_lora_resume_model(
    base_model: torch.nn.Module,
    checkpoint_dir: Path,
    *,
    adapter_name: str,
) -> torch.nn.Module:
    """Recreate a PEFT topology from an adapter-only FSDP2 DCP checkpoint.

    Swift 4.1.3 treats ``resume_from_checkpoint`` as an adapter directory
    during its early model-preparation phase.  That is valid for legacy
    LoRA checkpoints, which contain ``adapter_config.json``.  A Transformers
    4.53 / Accelerate 1.13 FSDP2 adapter-only checkpoint instead stores the
    adapter tensors in ``pytorch_model_fsdp_0`` DCP shards and intentionally
    has no standalone PEFT files.  The later Trainer/Accelerate resume path
    still knows how to restore those DCP shards, but it first needs this same
    PEFT module topology to exist.

    Infer the exact LoRA targets and rank from DCP metadata only.  No model
    tensor is loaded here; Accelerate remains the sole owner of DCP model and
    optimizer/RNG restoration.  The constructor hook adds the three aligner
    modules-to-save wrappers for the dynamic route.
    """
    source_dir = checkpoint_dir / "pytorch_model_fsdp_0"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Dynamic LoSATok FSDP DCP directory is missing: {source_dir}")
    try:
        from torch.distributed.checkpoint import FileSystemReader
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as exc:
        raise RuntimeError(
            "Unable to import the FSDP-DCP/PEFT APIs required for dynamic LoSATok resume"
        ) from exc

    metadata = FileSystemReader(str(source_dir)).read_metadata()
    state_metadata = getattr(metadata, "state_dict_metadata", {})
    lora_entries = [
        (str(key), entry) for key, entry in state_metadata.items()
        if ".lora_A." in str(key) or ".lora_B." in str(key)
    ]
    if not lora_entries:
        raise RuntimeError(f"Dynamic LoSATok FSDP DCP has no LoRA metadata entries: {source_dir}")

    module_names = set(dict(base_model.named_modules()))
    target_modules: set[str] = set()
    ranks: set[int] = set()
    unresolved: list[str] = []
    for source_key, entry in lora_entries:
        if ".lora_A." in source_key:
            shape_value = getattr(entry, "size", None)
            shape = tuple(int(value) for value in shape_value) if shape_value is not None else ()
            if not shape:
                raise RuntimeError(f"Dynamic LoSATok DCP LoRA-A metadata has no shape: {source_key}")
            ranks.add(int(shape[0]))
        matched = False
        for candidate in checkpoint_key_aliases(source_key):
            if ".lora_A." in candidate:
                module_path = candidate.split(".lora_A.", 1)[0]
            elif ".lora_B." in candidate:
                module_path = candidate.split(".lora_B.", 1)[0]
            else:
                continue
            if module_path in module_names:
                target_modules.add(module_path)
                matched = True
                break
        if not matched:
            unresolved.append(source_key)
    if unresolved:
        raise RuntimeError(
            "Unable to map every dynamic LoSATok FSDP DCP LoRA entry onto the fresh base model: "
            f"unresolved={unresolved[:10]} count={len(unresolved)}"
        )
    if len(ranks) != 1:
        raise RuntimeError(f"Dynamic LoSATok DCP LoRA ranks are inconsistent: {sorted(ranks)}")

    rank = next(iter(ranks))
    saved_args: dict[str, Any] = {}
    args_path = checkpoint_dir.parent / "args.json"
    if args_path.is_file():
        try:
            payload = json.loads(args_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                saved_args = payload
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unable to read dynamic LoSATok checkpoint args: {args_path}") from exc
    lora_alpha = int(saved_args.get("lora_alpha", rank * 2))
    lora_dropout = float(saved_args.get("lora_dropout", 0.05))
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=sorted(target_modules),
        bias="none",
        inference_mode=False,
    )
    restored = get_peft_model(base_model, config, adapter_name=adapter_name)
    injected_target_count = sum(
        1 for name, _ in restored.named_modules()
        if name.endswith(".lora_A.default") or name == "lora_A.default"
    )
    if injected_target_count != len(target_modules):
        raise RuntimeError(
            "Dynamic LoSATok FSDP resume recreated an unexpected LoRA topology: "
            f"metadata_targets={len(target_modules)} injected_targets={injected_target_count}"
        )
    if os.environ.get("RANK", "0") == "0":
        print(
            "[HuginnLoSATokSwift] recreated PEFT topology from dynamic FSDP DCP for Trainer resume "
            f"checkpoint={checkpoint_dir} lora_metadata_tensors={len(lora_entries)} "
            f"target_count={len(target_modules)} rank={rank} alpha={lora_alpha} dropout={lora_dropout} "
            f"aligner_modules_to_save={list(ALIGNER_PREFIXES)}"
        )
    return restored


def audit_modules_to_save_topology(model: torch.nn.Module, *, stage: str = "first_forward") -> None:
    """Opt-in, key-only inspection of PEFT's aligner wrapping after FSDP setup."""
    if not _requested(FSDP_SAVE_DEBUG_ENV) or getattr(
        model, "_huginn_losatok_modules_to_save_audited", False
    ):
        return

    peft_configs: list[dict[str, Any]] = []
    wrappers: list[dict[str, Any]] = []
    aligner_modules: list[dict[str, Any]] = []
    for module_name, module in model.named_modules():
        config = getattr(module, "peft_config", None)
        if isinstance(config, dict):
            for adapter_name, adapter_config in config.items():
                configured = getattr(adapter_config, "modules_to_save", None)
                if configured is not None:
                    peft_configs.append({
                        "module": module_name or "<root>",
                        "adapter": str(adapter_name),
                        "modules_to_save": list(configured),
                    })
        if type(module).__name__ == "ModulesToSaveWrapper":
            wrappers.append({
                "module": module_name,
                "module_key": getattr(module, "module_key", None),
                "adapters": sorted(str(name) for name in getattr(module, "modules_to_save", {})),
            })
        if module_name.split(".")[-1] in ALIGNER_PREFIXES:
            aligner_modules.append({
                "module": module_name,
                "type": f"{type(module).__module__}.{type(module).__name__}",
                "parameter_count": sum(parameter.numel() for parameter in module.parameters()),
                "trainable_parameter_count": sum(
                    parameter.numel() for parameter in module.parameters() if parameter.requires_grad
                ),
            })
    if os.environ.get("RANK", "0") == "0":
        print(
            "[HuginnLoSATokSwift] PEFT modules-to-save topology audit "
            f"stage={stage} model_type={type(model)} peft_configs={peft_configs} "
            f"wrappers={wrappers} aligner_modules={aligner_modules}"
        )
    model._huginn_losatok_modules_to_save_audited = True


def _active_modules_to_save_names(wrapper: torch.nn.Module) -> list[str]:
    """Return PEFT adapter copies selected by a ModulesToSaveWrapper.

    PEFT 0.18 normally uses the ``default`` adapter here.  Keep this helper
    tolerant of its string/list API so the narrowly-scoped protection below
    remains correct if an explicit adapter is selected during restoration.
    """
    copies = getattr(wrapper, "modules_to_save", {})
    available = [str(name) for name in copies]
    active = getattr(wrapper, "active_adapter", None)
    if isinstance(active, str):
        requested = [active]
    elif isinstance(active, (list, tuple, set)):
        requested = [str(name) for name in active]
    else:
        requested = []
    selected = [name for name in requested if name in copies]
    if not selected and "default" in copies:
        selected = ["default"]
    if not selected and len(available) == 1:
        selected = available
    if not selected:
        raise RuntimeError(
            "Dynamic LoSATok aligner ModulesToSaveWrapper has no selectable adapter copy: "
            f"available={available} active={active}"
        )
    return selected


def guard_peft_aligner_wrapper_trainability(wrapper: torch.nn.Module, *, module_name: str) -> None:
    """Ensure Swift can only unfreeze the PEFT-owned aligner copy.

    ``ModulesToSaveWrapper`` contains both ``original_module`` and a deep-copy
    in ``modules_to_save.<adapter>``.  Swift's generic ``freeze_aligner``
    handling calls ``requires_grad_(True)`` on the wrapper, which otherwise
    recursively unfreezes both copies.  That doubles optimizer parameters and
    violates PEFT's adapter-only checkpoint ownership.  Replacing *only this
    wrapper instance's* method preserves Swift's intended API while directing
    its unfreeze request exclusively to PEFT's selected saved copy.
    """
    if getattr(wrapper, "_huginn_losatok_trainability_guarded", False):
        return
    if type(wrapper).__name__ != "ModulesToSaveWrapper":
        raise RuntimeError(
            "Expected PEFT ModulesToSaveWrapper for dynamic LoSATok aligner, "
            f"got module={module_name} type={type(wrapper)}"
        )
    if not hasattr(wrapper, "original_module") or not hasattr(wrapper, "modules_to_save"):
        raise RuntimeError(
            "PEFT ModulesToSaveWrapper lacks expected ownership fields for dynamic LoSATok aligner: "
            f"module={module_name}"
        )

    def guarded_requires_grad_(self, requires_grad: bool = True):
        # Never optimize the base-model branch.  PEFT's adapter-only state
        # dict intentionally owns only modules_to_save.<active_adapter>.
        self.original_module.requires_grad_(False)
        for copy in self.modules_to_save.values():
            copy.requires_grad_(False)
        if requires_grad:
            for adapter_name in _active_modules_to_save_names(self):
                self.modules_to_save[adapter_name].requires_grad_(True)
        return self

    wrapper.requires_grad_ = MethodType(guarded_requires_grad_, wrapper)
    wrapper._huginn_losatok_trainability_guarded = True
    # Establish the correct topology immediately, before Trainer constructs
    # its optimizer.  Later Swift --freeze_aligner false calls are intercepted
    # by the wrapper method above and retain this same ownership split.
    wrapper.requires_grad_(True)
    original_trainable = sum(parameter.numel() for parameter in wrapper.original_module.parameters()
                             if parameter.requires_grad)
    saved_trainable = sum(parameter.numel() for parameter in wrapper.modules_to_save.parameters()
                          if parameter.requires_grad)
    if original_trainable or not saved_trainable:
        raise RuntimeError(
            "Failed to establish PEFT-owned dynamic LoSATok aligner trainability: "
            f"module={module_name} original_trainable={original_trainable} "
            f"saved_trainable={saved_trainable}"
        )


def patch_swift_prepare_model_audit() -> None:
    """Trace PEFT wrapping immediately after Swift creates the adapter model.

    The forward audit observes the inner Huginn model after FSDP setup.  This
    opt-in hook observes the object returned by ``Swift.prepare_model`` before
    FSDP preparation, which distinguishes a PEFT wrapping failure from a later
    FSDP transformation.  It is diagnostic only and returns Swift's original
    model object unchanged.
    """
    if not _requested(FSDP_SAVE_DEBUG_ENV):
        return
    try:
        from swift.tuners.base import Swift
    except Exception as exc:
        print(
            "[HuginnLoSATokSwift] unable to install Swift.prepare_model PEFT wrapping audit "
            f"error_type={type(exc).__name__} error={exc}"
        )
        return
    descriptor = Swift.__dict__.get("prepare_model")
    original = Swift.prepare_model
    if getattr(original, "_huginn_losatok_prepare_audit_patched", False):
        return

    def run_original_and_audit(*args, **kwargs):
        config = args[1] if len(args) > 1 else kwargs.get("config")
        configured_modules = getattr(config, "modules_to_save", None)
        result = original(*args, **kwargs)
        if os.environ.get("RANK", "0") == "0":
            print(
                "[HuginnLoSATokSwift] PEFT prepare-model audit "
                f"config_type={type(config)} configured_modules_to_save={configured_modules} "
                f"result_type={type(result)}"
            )
        audit_modules_to_save_topology(result, stage="immediately_after_swift_prepare_model")
        return result

    if isinstance(descriptor, staticmethod):
        run_original_and_audit._huginn_losatok_prepare_audit_patched = True
        Swift.prepare_model = staticmethod(run_original_and_audit)
    elif isinstance(descriptor, classmethod):
        def classmethod_wrapper(cls, *args, **kwargs):
            del cls
            return run_original_and_audit(*args, **kwargs)

        classmethod_wrapper._huginn_losatok_prepare_audit_patched = True
        Swift.prepare_model = classmethod(classmethod_wrapper)
    else:
        run_original_and_audit._huginn_losatok_prepare_audit_patched = True
        Swift.prepare_model = run_original_and_audit
    print("[HuginnLoSATokSwift] installed Swift.prepare_model PEFT wrapping audit")


def patch_peft_constructor_modules_to_save() -> None:
    """Make dynamic-aligner weights part of PEFT adapter state before wrapping.

    Transformers 4.53 requests FSDP ``adapter_only=True`` checkpoints.  In
    Accelerate 1.13 this calls PEFT ``get_peft_model_state_dict``, which saves
    LoRA plus only modules represented by ``ModulesToSaveWrapper``.  Swift's
    CLI accepts our ``--modules_to_save`` list, but the observed PEFT
    ``LoraConfig`` arrives with that field reset to ``None``.  The only safe
    repair is to restore the three aligner module names *before*
    ``PeftModel.__init__`` injects adapters, so PEFT creates normal wrappers
    and its standard save/load pair handles the aligner automatically.

    The behavior is explicit and opt-in for the dynamic route.  The same hook
    additionally prints constructor topology when the dedicated save-debug
    environment is enabled.
    """
    force_modules_to_save = _requested(PEFT_ALIGNER_MODULES_TO_SAVE_ENV)
    debug = _requested(FSDP_SAVE_DEBUG_ENV)
    if not force_modules_to_save and not debug:
        return
    try:
        from peft.peft_model import PeftModel
    except Exception as exc:
        print(
            "[HuginnLoSATokSwift] unable to install PEFT constructor modules-to-save hook "
            f"error_type={type(exc).__name__} error={exc}"
        )
        return
    original = PeftModel.__init__
    if getattr(original, "_huginn_losatok_constructor_audit_patched", False):
        return

    def patched(self, *args, **kwargs):
        peft_config = kwargs.get("peft_config")
        if peft_config is None and len(args) >= 2:
            # PeftModel.__init__(self, model, peft_config, adapter_name, ...)
            peft_config = args[1]
        if force_modules_to_save:
            if peft_config is None or not hasattr(peft_config, "modules_to_save"):
                raise RuntimeError(
                    "Dynamic LoSATok requires a PEFT config with modules_to_save before adapter construction; "
                    f"config_type={type(peft_config)}"
                )
            configured = list(getattr(peft_config, "modules_to_save", None) or [])
            merged = list(dict.fromkeys([*configured, *ALIGNER_PREFIXES]))
            peft_config.modules_to_save = merged
            if os.environ.get("RANK", "0") == "0":
                print(
                    "[HuginnLoSATokSwift] injected PEFT aligner modules_to_save before constructor "
                    f"existing={configured} final={merged}"
                )
        result = original(self, *args, **kwargs)
        wrappers = {
            name: module for name, module in self.named_modules()
            if type(module).__name__ == "ModulesToSaveWrapper"
        }
        if force_modules_to_save:
            missing = [
                name for name in ALIGNER_PREFIXES
                if not any(wrapper_name.endswith(name) for wrapper_name in wrappers)
            ]
            if missing:
                raise RuntimeError(
                    "PEFT did not wrap every required dynamic LoSATok aligner module: "
                    f"missing={missing} wrappers={list(wrappers)} "
                    f"config_modules_to_save={getattr(peft_config, 'modules_to_save', None)}"
                )
            guarded_wrappers: list[dict[str, Any]] = []
            for aligner_name in ALIGNER_PREFIXES:
                wrapper_name, wrapper = next(
                    (item for item in wrappers.items() if item[0].endswith(aligner_name)),
                    (None, None),
                )
                if wrapper is None or wrapper_name is None:
                    # ``missing`` above already makes this unreachable; retain
                    # an explicit error to keep this ownership repair fail-fast.
                    raise RuntimeError(
                        "Unable to locate required PEFT wrapper after dynamic LoSATok validation: "
                        f"aligner={aligner_name} wrappers={list(wrappers)}"
                    )
                guard_peft_aligner_wrapper_trainability(wrapper, module_name=wrapper_name)
                guarded_wrappers.append({
                    "module": wrapper_name,
                    "active_copies": _active_modules_to_save_names(wrapper),
                    "original_trainable": sum(
                        parameter.numel() for parameter in wrapper.original_module.parameters()
                        if parameter.requires_grad
                    ),
                    "saved_trainable": sum(
                        parameter.numel() for parameter in wrapper.modules_to_save.parameters()
                        if parameter.requires_grad
                    ),
                })
            if os.environ.get("RANK", "0") == "0":
                print(
                    "[HuginnLoSATokSwift] installed PEFT aligner wrapper trainability guards "
                    f"wrappers={guarded_wrappers}"
                )
        if debug and os.environ.get("RANK", "0") == "0":
            print(
                "[HuginnLoSATokSwift] PEFT constructor audit "
                f"model_type={type(self)} peft_configs={getattr(self, 'peft_config', None)}"
            )
        if debug:
            audit_modules_to_save_topology(self, stage="immediately_after_peft_constructor")
        return result

    patched._huginn_losatok_constructor_audit_patched = True
    PeftModel.__init__ = patched
    print(
        "[HuginnLoSATokSwift] installed PEFT constructor modules-to-save hook "
        f"force_aligner_modules_to_save={force_modules_to_save} debug={debug}"
    )


def patch_peft_adapter_restore() -> None:
    if getattr(patch_peft_adapter_restore, "_patched", False):
        return
    try:
        from peft import PeftModel
    except ImportError:
        return
    original = PeftModel.from_pretrained

    @classmethod
    def patched(cls, *args, **kwargs):
        # PeftModel.from_pretrained(model, model_id, ...) receives the
        # checkpoint from Swift's resume_from_checkpoint path.  Intercept only
        # our opt-in dynamic FSDP DCP format; ordinary legacy PEFT adapter
        # directories must retain PEFT's original load behavior.
        base_model = args[0] if args else kwargs.get("model")
        model_id = args[1] if len(args) >= 2 else kwargs.get("model_id")
        checkpoint_dir = Path(model_id) if isinstance(model_id, (str, os.PathLike)) else None
        is_dynamic_fsdp_dcp = (
            DYNAMIC_AUDIO_TOKENS_ENABLED
            and _requested(PEFT_ALIGNER_MODULES_TO_SAVE_ENV)
            and checkpoint_dir is not None
            and (checkpoint_dir / "pytorch_model_fsdp_0").is_dir()
            and not (checkpoint_dir / "adapter_config.json").is_file()
        )
        if is_dynamic_fsdp_dcp:
            if not isinstance(base_model, torch.nn.Module):
                raise RuntimeError(
                    "Dynamic LoSATok FSDP resume did not receive a base torch model for PEFT reconstruction: "
                    f"model_type={type(base_model)}"
                )
            adapter_name = str(kwargs.get("adapter_name", "default"))
            restored = _fsdp_dcp_lora_resume_model(
                base_model,
                checkpoint_dir,
                adapter_name=adapter_name,
            )
        else:
            restored = original(*args, **kwargs)
        force_aligner_trainable(restored)
        return restored

    PeftModel.from_pretrained = patched
    patch_peft_adapter_restore._patched = True
    print("[HuginnLoSATokSwift] installed PEFT adapter-restore aligner patch")


def audit_final_trainable_split(model: torch.nn.Module) -> dict[str, int]:
    """Verify the actual post-PEFT/FSDP trainable topology on first forward."""
    groups = {
        "losatok_encoder": 0,
        "aligner": 0,
        "huginn_lora": 0,
        "huginn_base": 0,
        "other": 0,
    }
    trainable_names: dict[str, list[str]] = {name: [] for name in groups}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parts = set(name.split("."))
        if "audio_encoder" in parts:
            group = "losatok_encoder"
        elif any(prefix in parts for prefix in ALIGNER_PREFIXES):
            group = "aligner"
        elif "lora_" in name:
            group = "huginn_lora"
        elif "transformer" in parts or "lm_head" in parts:
            group = "huginn_base"
        else:
            group = "other"
        groups[group] += parameter.numel()
        if len(trainable_names[group]) < 20:
            trainable_names[group].append(name)

    invalid = {
        "losatok_encoder": groups["losatok_encoder"],
        "huginn_base": groups["huginn_base"],
        "other": groups["other"],
    }
    if any(invalid.values()):
        raise RuntimeError(
            "Final trainable topology contains forbidden parameters: "
            f"groups={groups} names={trainable_names}"
        )
    if groups["aligner"] <= 0 or groups["huginn_lora"] <= 0:
        raise RuntimeError(
            "Final trainable topology is missing aligner or Huginn LoRA parameters: "
            f"groups={groups} names={trainable_names}"
        )
    if DYNAMIC_AUDIO_TOKENS_ENABLED:
        expected = {
            "aligner": DYNAMIC_ALIGNER_TRAINABLE_PARAMETERS,
            "huginn_lora": HUGINN_LORA_TRAINABLE_PARAMETERS,
        }
        actual = {name: groups[name] for name in expected}
        if actual != expected:
            raise RuntimeError(
                "Dynamic LoSATok post-PEFT/FSDP trainable counts changed unexpectedly: "
                f"expected={expected} actual={actual} names={trainable_names}"
            )
    if model.audio_bos is None or model.audio_eos is None:
        raise RuntimeError("Audio BOS/EOS boundary embeddings are required for LoSATok training")
    if not model.audio_bos.requires_grad or not model.audio_eos.requires_grad:
        raise RuntimeError("Audio BOS/EOS boundary embeddings must remain trainable")

    if os.environ.get("RANK", "0") == "0":
        print(
            "[HuginnLoSATokSwift] final_trainable_split "
            f"groups={groups} total={sum(groups.values())} "
            "policy=aligner_plus_huginn_lora_only audio_bos_eos_trainable=true"
        )
    return groups


def patch_shift_loss(model: torch.nn.Module) -> torch.nn.Module:
    if getattr(model, "_huginn_losatok_shift_loss_patched", False):
        return model
    original = model.forward

    def forward_with_shift_loss(self, *args, **kwargs):
        labels = kwargs.get("labels")
        audio_values = kwargs.get("audio_input_values")
        past_key_values = kwargs.get("past_key_values")
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if labels is None:
            return original(*args, **kwargs)
        if _requested(TRAIN_CHAIN_AUDIT_ENV) and not getattr(
            self, "_huginn_losatok_final_split_audited", False
        ):
            audit_final_trainable_split(self)
            audit_modules_to_save_topology(self, stage="first_forward_inner_model")
            self._huginn_losatok_final_split_audited = True
        without_labels = dict(kwargs)
        without_labels["labels"] = None
        outputs = original(*args, **without_labels)
        logits = outputs.logits
        if logits is None:
            raise RuntimeError("LoSATok Huginn forward returned logits=None")
        full_labels = labels.to(logits.device)
        if audio_values is not None and past_key_values is None:
            prefix_length = logits.size(1) - labels.size(1)
            if prefix_length <= 0:
                raise RuntimeError(f"Invalid LoSATok prefix length: {prefix_length}")
            prefix_labels = torch.full(
                (labels.size(0), prefix_length), -100, dtype=labels.dtype, device=labels.device)
            full_labels = torch.cat([prefix_labels, labels], dim=1).to(logits.device)
        if full_labels.shape != logits.shape[:2]:
            raise RuntimeError(
                f"NTP full-sequence mismatch: logits={tuple(logits.shape)} labels={tuple(full_labels.shape)}"
            )
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = full_labels[:, 1:].contiguous()
        if shift_logits.shape[:2] != shift_labels.shape:
            raise RuntimeError(f"NTP shift mismatch: logits={tuple(shift_logits.shape)} labels={tuple(shift_labels.shape)}")
        if not shift_labels.ne(-100).any():
            loss = logits.new_tensor(0.0)
        else:
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
        if _requested(TRAIN_CHAIN_AUDIT_ENV) and os.environ.get("RANK", "0") == "0" and not getattr(
            self, "_huginn_losatok_ntp_audited", False):
            prefix_tokens = int(logits.size(1) - labels.size(1))
            if not bool((full_labels[:, :prefix_tokens] == -100).all().item()):
                raise RuntimeError("LoSATok audio prefix labels are not masked")
            text_supervision = labels.ne(-100)
            if not bool(text_supervision.any(dim=1).all().item()):
                raise RuntimeError("Every LoSATok SFT sample must contain at least one supervised text token")
            first_supervised_text = text_supervision.to(torch.int64).argmax(dim=1)
            if bool(first_supervised_text.eq(0).any().item()):
                raise RuntimeError(
                    "The first text position must be prompt-masked so dynamic audio padding cannot predict a target"
                )
            print(
                "[HuginnLoSATokSwift] train_chain_audit_ntp "
                f"text_input_ids={tuple(input_ids.shape) if torch.is_tensor(input_ids) else None} "
                f"audio_values={tuple(audio_values.shape) if torch.is_tensor(audio_values) else None} "
                f"logits={tuple(logits.shape)} prefix_tokens={prefix_tokens} "
                f"supervised_tokens={int((shift_labels != -100).sum().item())} "
                f"first_supervised_text_positions={first_supervised_text.tolist()} "
                "loss_alignment=logits[t]_predict_labels[t+1] prefix_labels=-100"
            )
            self._huginn_losatok_ntp_audited = True
        outputs.loss = loss
        if hasattr(outputs, "log_ppl"):
            outputs.log_ppl = loss.detach().clone()
        return outputs

    model.forward = MethodType(forward_with_shift_loss, model)
    model._huginn_losatok_shift_loss_patched = True
    print("[HuginnLoSATokSwift] applied shift-loss patch for multimodal SFT")
    return model


def audit_model_split(model: torch.nn.Module) -> None:
    groups = {"losatok_encoder": 0, "aligner": 0, "huginn": 0, "other": 0}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("audio_encoder."):
            groups["losatok_encoder"] += parameter.numel()
        elif name.startswith(ALIGNER_PREFIXES):
            groups["aligner"] += parameter.numel()
        elif name.startswith(("transformer.", "lm_head.")):
            groups["huginn"] += parameter.numel()
        else:
            groups["other"] += parameter.numel()
    if groups["losatok_encoder"]:
        raise RuntimeError(f"LoSATok must be frozen before Swift wrapping: {groups}")
    print(f"[HuginnLoSATokSwift] parameter_split_before_tuner={groups}")


def build_model(model_dir: str) -> torch.nn.Module:
    config = configure_audio_compressor(AutoConfig.from_pretrained(model_dir, trust_remote_code=True))
    config.losatok_root = str(LOSATOK_ROOT)
    config.losatok_code_dir = str(LOSATOK_CODE_DIR)
    config.freeze_audio_encoder = True
    config.freeze_text_backbone = False
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    result = model.load_huginn_backbone_from_pretrained(str(HUGINN_MODEL_DIR), torch_dtype=torch.float32)
    print_missing_summary(result.missing_keys, result.unexpected_keys)
    model.audio_encoder.requires_grad_(False)
    model.audio_encoder.eval()
    initial_aligner_checkpoint = os.environ.get(INIT_ALIGNER_CHECKPOINT_ENV)
    if initial_aligner_checkpoint:
        aligner_report = load_initial_aligner_state(model, Path(initial_aligner_checkpoint))
        print(f"[HuginnLoSATokSwift] initial_aligner_restore={aligner_report}")
    enable_fsdp2_nonpersistent_rope_buffer(model)
    patch_accelerate_fsdp2_state_dict_key_alignment()
    patch_accelerate_fsdp2_save_state_audit()
    audit_model_split(model)
    return patch_shift_loss(model)


def build_huginn_losatok_evaluation_model() -> torch.nn.Module:
    """Public evaluation entrypoint used by retrieval, generation, and MMAU scripts."""
    return build_model(str(AUDIO_MODEL_DIR))


def build_huginn_losatok_evaluation_processor() -> HuginnLoSATokProcessor:
    return build_processor()


class HuginnLoSATokTemplate(Template):
    use_model = False
    support_padding_free = False

    def init_processor(self, processor: HuginnLoSATokProcessor):
        super().init_processor(processor)
        self.audio_sampling_rate = processor.sampling_rate

    def replace_tag(self, media_type: str, index: int, inputs: StdTemplateInputs):
        if media_type == "audio":
            return []
        return super().replace_tag(media_type, index, inputs)

    @staticmethod
    def _path(audio_item: Any) -> Path:
        if isinstance(audio_item, str):
            return Path(audio_item)
        if isinstance(audio_item, dict):
            for key in ("audio", "path"):
                if isinstance(audio_item.get(key), str):
                    return Path(audio_item[key])
        raise TypeError(f"Unsupported LoSATok audio item: {type(audio_item)}")

    @staticmethod
    def _audio_bytes(audio_item: Any) -> bytes | None:
        if not isinstance(audio_item, dict) or "audio_bytes" not in audio_item:
            return None
        value = audio_item["audio_bytes"]
        if isinstance(value, bytes):
            return value
        if isinstance(value, (bytearray, memoryview)):
            return bytes(value)
        raise TypeError(f"Unsupported LoSATok embedded audio type: {type(value)}")

    def _encode(self, inputs: StdTemplateInputs) -> dict[str, Any]:
        encoded = super()._encode(inputs)
        if not getattr(inputs, "audios", None):
            return encoded
        if len(inputs.audios) != 1:
            raise ValueError("Huginn LoSATok template requires exactly one audio clip per record")
        audio_bytes = self._audio_bytes(inputs.audios[0])
        if audio_bytes is not None:
            encoded["audio_input_values"] = decode_audio_bytes_16k(audio_bytes, "Swift dataset audio_bytes")
        else:
            encoded["audio_input_values"] = load_audio_16k(self._path(inputs.audios[0]))
        return encoded

    def _data_collator_mm_data(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        waveforms = [item["audio_input_values"] for item in batch if "audio_input_values" in item]
        if not waveforms:
            return {}
        if len(waveforms) != len(batch):
            raise RuntimeError("A batch mixes records with and without audio")
        max_length = max(waveform.numel() for waveform in waveforms)
        values = torch.zeros((len(waveforms), max_length), dtype=torch.float32)
        mask = torch.zeros((len(waveforms), max_length), dtype=torch.long)
        for index, waveform in enumerate(waveforms):
            values[index, :waveform.numel()] = waveform
            mask[index, :waveform.numel()] = 1
        return {"audio_input_values": values, "audio_attention_mask": mask}


class HuginnLoSATokLoader(ModelLoader):
    def get_config(self, model_dir: str):
        config = configure_audio_compressor(AutoConfig.from_pretrained(model_dir, trust_remote_code=True))
        config.losatok_root = str(LOSATOK_ROOT)
        config.losatok_code_dir = str(LOSATOK_CODE_DIR)
        config.freeze_audio_encoder = True
        config.freeze_text_backbone = False
        print(f"[HuginnLoSATokSwift] config.losatok_root={config.losatok_root}")
        print(f"[HuginnLoSATokSwift] config.audio_sample_rate={config.audio_sample_rate}")
        print(f"[HuginnLoSATokSwift] config.audio_dynamic_tokens={config.audio_dynamic_tokens}")
        print(f"[HuginnLoSATokSwift] config.audio_max_token_count={config.audio_max_token_count}")
        print(f"[HuginnLoSATokSwift] config.audio_max_seconds={DEFAULT_MAX_AUDIO_SECONDS}")
        print(f"[HuginnLoSATokSwift] config.audio_compressor_kernel_size={config.audio_compressor_kernel_size}")
        print(f"[HuginnLoSATokSwift] config.audio_compressor_stride={config.audio_compressor_stride}")
        return config

    def get_processor(self, model_dir: str, config):
        del model_dir, config
        processor = build_processor()
        print(f"[HuginnLoSATokSwift] tokenizer_type={type(processor.tokenizer)}")
        return processor

    def get_model(self, model_dir: str, config, processor, model_kwargs):
        del config, processor, model_kwargs
        model = build_model(model_dir)
        print(f"[HuginnLoSATokSwift] model_type={type(model)}")
        return model


def register_huginn_losatok_arch() -> None:
    keys = {
        "language_model": ["transformer", "lm_head"],
        "aligner": ["temporal_compressor", "audio_projector", "audio_boundary_embeddings"],
        "generator": ["audio_encoder"],
    }
    try:
        model_keys = MultiModelKeys(arch_name=MODEL_ARCH_NAME, **keys)
    except TypeError:
        try:
            model_keys = MultiModelKeys(model_arch=MODEL_ARCH_NAME, **keys)
        except TypeError:
            model_keys = MultiModelKeys(MODEL_ARCH_NAME, **keys)
    try:
        register_model_arch(model_keys)
        print("[HuginnLoSATokSwift] registered model architecture")
    except ValueError as error:
        if f"The `{MODEL_ARCH_NAME}` has already been registered" not in str(error):
            raise
        print("[HuginnLoSATokSwift] model architecture already registered")


register_huginn_losatok_arch()
patch_peft_adapter_restore()
patch_swift_prepare_model_audit()
patch_peft_constructor_modules_to_save()
register_model(
    ModelMeta(
        MODEL_TYPE,
        [ModelGroup([Model("huginn-audio-losatok-v1", str(AUDIO_MODEL_DIR))])],
        HuginnLoSATokLoader,
        template=TEMPLATE_TYPE,
        model_arch=MODEL_ARCH_NAME,
        architectures=["HuginnLoSATokForConditionalGeneration"],
        is_multimodal=True,
        requires=["transformers>=4.53.3", "torchaudio"],
        tags=["huginn", "audio", "losatok"],
    ),
    exist_ok=True,
)
register_template(
    TemplateMeta(
        template_type=TEMPLATE_TYPE,
        template_cls=HuginnLoSATokTemplate,
        prefix=[],
        system_prefix=["<|begin_header|>system<|end_header|>\n\n{{SYSTEM}}<|end_turn|>"],
        prompt=[
            "<|begin_header|>user<|end_header|>\n\n{{QUERY}}<|end_turn|>"
            "<|begin_header|>Huginn<|end_header|>\n\n"
        ],
        chat_sep=None,
        auto_add_bos=True,
        default_system=DEFAULT_SYSTEM_PROMPT,
        stop_words=[["eos_token_id"]],
    ),
    exist_ok=True,
)
