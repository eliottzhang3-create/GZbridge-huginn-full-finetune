"""Build the Huginn audio model through Swift full/FSDP2 and inspect its final split."""

from __future__ import annotations

import dataclasses
import platform
import sys
from collections import defaultdict
from pathlib import Path

import torch


def classify_parameter(name: str) -> str:
    if 'audio_encoder' in name:
        return 'audio_encoder'
    if any(key in name for key in ('temporal_compressor', 'audio_projector', 'audio_boundary_embeddings')):
        return 'aligner'
    if 'lora_' in name:
        return 'lora'
    if 'transformer' in name or 'lm_head' in name:
        return 'llm_base'
    return 'other'


def distributed_context() -> tuple[int, int, bool]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size(), True
    return 0, 1, False


def collect_summary(model: torch.nn.Module) -> dict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            counts[classify_parameter(name)] += parameter.numel()
    return dict(counts)


def print_argument_support() -> None:
    from swift import __version__ as swift_version
    from swift.arguments.sft_args import SftArguments

    requested = ('fsdp', 'tuner_type', 'freeze_llm', 'freeze_vit', 'freeze_aligner',
                 'gradient_checkpointing', 'gradient_checkpointing_kwargs')
    fields = {field.name: field for field in dataclasses.fields(SftArguments)}
    rank, _, _ = distributed_context()
    if rank == 0:
        print('========== SWIFT FULL FSDP ARGUMENT SUPPORT ==========', flush=True)
        print(f'[swift] version={swift_version}', flush=True)
        for name in requested:
            field = fields.get(name)
            if field is None:
                print(f'[swift-arg] name={name} present=false', flush=True)
            else:
                default = '<missing>' if field.default is dataclasses.MISSING else repr(field.default)
                print(f'[swift-arg] name={name} present=true default={default} type={field.type}', flush=True)


def print_final_summary(model: torch.nn.Module) -> None:
    rank, world_size, distributed = distributed_context()
    groups = ('audio_encoder', 'aligner', 'lora', 'llm_base', 'other')
    local_counts = collect_summary(model)
    device = torch.device('cuda', torch.cuda.current_device())
    local_tensor = torch.tensor([local_counts.get(group, 0) for group in groups], device=device, dtype=torch.long)
    global_tensor = local_tensor.clone()
    failure_flag = torch.zeros(1, device=device, dtype=torch.int32)
    if distributed:
        torch.distributed.all_reduce(global_tensor, op=torch.distributed.ReduceOp.SUM)

    fsdp_module_count = sum(1 for module in model.modules() if 'fsdp' in type(module).__name__.lower())
    print('========== SWIFT FULL FSDP RANK SUMMARY ==========', flush=True)
    print(f'[rank] rank={rank} world_size={world_size} model_type={type(model)} fsdp_module_count={fsdp_module_count}', flush=True)
    print(f'[rank] local_trainable_groups={dict(zip(groups, local_tensor.tolist()))}', flush=True)
    print(f'[rank] cuda_device={torch.cuda.current_device()} name={torch.cuda.get_device_name()}', flush=True)
    print(f'[rank] cuda_allocated_gb={torch.cuda.memory_allocated() / (1024 ** 3):.4f}', flush=True)
    print(f'[rank] cuda_reserved_gb={torch.cuda.memory_reserved() / (1024 ** 3):.4f}', flush=True)

    if rank == 0:
        global_counts = dict(zip(groups, global_tensor.tolist()))
        failures = []
        if global_counts['audio_encoder'] != 0:
            failures.append('audio_encoder has trainable parameters')
        if global_counts['aligner'] == 0:
            failures.append('aligner has no trainable parameters')
        if global_counts['llm_base'] == 0:
            failures.append('Huginn base has no trainable parameters')
        if global_counts['lora'] != 0:
            failures.append('full tuning unexpectedly contains LoRA parameters')
        if fsdp_module_count == 0:
            failures.append('no FSDP module was detected on rank 0')
        print('========== SWIFT FULL FSDP GLOBAL SUMMARY ==========', flush=True)
        print(f'[global] summed_trainable_groups={global_counts}', flush=True)
        print(f'[global] lora_expected_zero={global_counts["lora"] == 0}', flush=True)
        print(f'[global] audio_encoder_expected_zero={global_counts["audio_encoder"] == 0}', flush=True)
        print(f'[global] aligner_expected_positive={global_counts["aligner"] > 0}', flush=True)
        print(f'[global] llm_base_expected_positive={global_counts["llm_base"] > 0}', flush=True)
        if failures:
            print(f'[global] status=FAIL failures={failures}', flush=True)
            failure_flag.fill_(1)
        else:
            print('[global] status=PASS', flush=True)
    if distributed:
        torch.distributed.broadcast(failure_flag, src=0)
    if failure_flag.item():
        raise RuntimeError('Swift full/FSDP parameter inspection failed; consult rank-0 global summary')


def build_argv(repo_root: Path) -> list[str]:
    return [
        '--model', str(repo_root / 'models' / 'huginn-audio-whisper-v1'),
        '--model_type', 'huginn_audio_raven',
        '--template', 'huginn_audio_text',
        '--external_plugins', str(repo_root / 'code' / 'huginn_lora' / 'plugins' / 'huginn_audio_swift.py'),
        '--dataset', str(repo_root / 'data' / 'audio_swift' / 'wavcaps_audioset' / 'wavcaps_audioset_sl_train_swift.jsonl'),
        '--max_length', '192',
        '--output_dir', str(repo_root / 'outputs' / 'huginn_audio_full_fsdp7_inspect'),
        '--tuner_type', 'full',
        '--freeze_llm', 'false',
        '--freeze_vit', 'true',
        '--freeze_aligner', 'false',
        '--fsdp', 'fsdp2',
        '--learning_rate', '1e-5',
        '--aligner_lr', '1e-4',
        '--gradient_checkpointing', 'true',
        '--gradient_checkpointing_kwargs', '{"use_reentrant": false}',
        '--max_steps', '1',
        '--per_device_train_batch_size', '1',
        '--gradient_accumulation_steps', '4',
        '--logging_steps', '1',
        '--save_strategy', 'no',
        '--dataloader_num_workers', '0',
        '--dataloader_pin_memory', 'false',
        '--dataset_num_proc', '1',
        '--report_to', 'none',
        '--bf16', 'true',
    ]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    rank, world_size, _ = distributed_context()
    if rank == 0:
        print('========== HUGINN AUDIO SWIFT FULL FSDP7 INSPECT ==========', flush=True)
        print(f'[env] python={sys.version.split()[0]} platform={platform.platform()}', flush=True)
        print(f'[env] repo_root={repo_root}', flush=True)
        print(f'[env] requested_world_size=7 actual_world_size={world_size}', flush=True)
    print_argument_support()

    from swift.pipelines.train.sft import SwiftSft

    class InspectSwiftSft(SwiftSft):
        def train(self, trainer):
            print_final_summary(trainer.model)
            return {'status': 'inspected'}

    InspectSwiftSft(build_argv(repo_root)).main()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    if rank == 0:
        print('========== HUGINN AUDIO SWIFT FULL FSDP7 INSPECT DONE ==========', flush=True)


if __name__ == '__main__':
    main()
