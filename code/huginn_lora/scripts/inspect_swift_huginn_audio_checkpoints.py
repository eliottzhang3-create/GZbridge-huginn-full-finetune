"""Inspect Swift/PEFT audio checkpoints before retrieval evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


DEFAULT_CHECKPOINTS = [
    '/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/'
    'outputs/huginn_audio_audiocaps_v2_train_e5_b8ga4_5090/v0-20260713-155848/checkpoint-5604',
    '/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/'
    'outputs/huginn_audio_audiocaps_v2_train_e5_b8ga4_5090/v0-20260713-155848/checkpoint-8406',
]
SKIP_TORCH_FILES = ('optimizer', 'scheduler', 'rng', 'trainer_state', 'training_args')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', action='append', default=None, help='Repeat for each checkpoint directory.')
    parser.add_argument('--output_report', required=True)
    parser.add_argument('--expected_lora_tensor_count', type=int, default=None)
    parser.add_argument('--expected_aligner_tensor_count', type=int, default=None)
    parser.add_argument('--require_boundary_embeddings', action='store_true')
    return parser.parse_args()


def classify_key(key: str) -> str:
    if 'audio_encoder.' in key:
        return 'audio_encoder'
    if any(name in key for name in ('temporal_compressor.', 'audio_projector.', 'audio_bos', 'audio_eos')):
        return 'aligner'
    if 'lora_' in key:
        return 'lora'
    if key.startswith(('transformer.', 'lm_head.', 'base_model.model.transformer.', 'base_model.model.lm_head.')):
        return 'llm'
    return 'other'


def inspect_tensor_file(path: Path) -> dict[str, Any]:
    if path.suffix == '.safetensors':
        from safetensors import safe_open

        with safe_open(str(path), framework='pt', device='cpu') as handle:
            keys = list(handle.keys())
    elif path.suffix in {'.bin', '.pt', '.pth'}:
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(payload, dict) and isinstance(payload.get('state_dict'), dict):
            payload = payload['state_dict']
        if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
            return {'path': str(path), 'kind': 'non_tensor_or_non_state_dict'}
        keys = [key for key, value in payload.items() if torch.is_tensor(value)]
    else:
        return {'path': str(path), 'kind': 'unsupported'}

    groups: dict[str, int] = {}
    for key in keys:
        group = classify_key(key)
        groups[group] = groups.get(group, 0) + 1
    normalized_keys = {key.split('base_model.model.', 1)[-1] for key in keys}
    return {
        'path': str(path),
        'kind': 'tensor_state_dict',
        'tensor_key_count': len(keys),
        'group_counts': groups,
        'boundary_embeddings_present': {
            'audio_bos': 'audio_boundary_embeddings.audio_bos' in normalized_keys,
            'audio_eos': 'audio_boundary_embeddings.audio_eos' in normalized_keys,
        },
        'key_preview': keys[:20],
    }


def inspect_checkpoint(checkpoint_dir: Path) -> dict[str, Any]:
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f'Checkpoint directory not found: {checkpoint_dir}')

    files = sorted(path for path in checkpoint_dir.rglob('*') if path.is_file())
    entries = [
        {
            'relative_path': str(path.relative_to(checkpoint_dir)),
            'size_bytes': path.stat().st_size,
        }
        for path in files
    ]
    tensor_reports = []
    json_reports: dict[str, Any] = {}
    for path in files:
        lower_name = path.name.lower()
        if path.suffix in {'.safetensors', '.bin', '.pt', '.pth'} and not any(token in lower_name for token in SKIP_TORCH_FILES):
            try:
                tensor_reports.append(inspect_tensor_file(path))
            except Exception as exc:  # pragma: no cover - remote checkpoint dependent
                tensor_reports.append({'path': str(path), 'error': f'{type(exc).__name__}: {exc}'})
        if path.suffix == '.json' and path.name in {'adapter_config.json', 'trainer_state.json', 'config.json'}:
            try:
                json_reports[str(path.relative_to(checkpoint_dir))] = json.loads(path.read_text(encoding='utf-8'))
            except Exception as exc:  # pragma: no cover - remote checkpoint dependent
                json_reports[str(path.relative_to(checkpoint_dir))] = {'error': f'{type(exc).__name__}: {exc}'}

    return {
        'checkpoint_dir': str(checkpoint_dir),
        'file_count': len(entries),
        'files': entries,
        'tensor_reports': tensor_reports,
        'json_reports': json_reports,
    }


def validate_checkpoint_report(
    report: dict[str, Any],
    expected_lora_tensor_count: int | None,
    expected_aligner_tensor_count: int | None,
    require_boundary_embeddings: bool,
) -> list[str]:
    group_counts: dict[str, int] = {}
    boundary_present = {'audio_bos': False, 'audio_eos': False}
    for tensor_report in report['tensor_reports']:
        for group, count in tensor_report.get('group_counts', {}).items():
            group_counts[group] = group_counts.get(group, 0) + count
        for name, present in tensor_report.get('boundary_embeddings_present', {}).items():
            boundary_present[name] = boundary_present[name] or bool(present)

    failures = []
    if expected_lora_tensor_count is not None and group_counts.get('lora', 0) != expected_lora_tensor_count:
        failures.append(
            f"expected {expected_lora_tensor_count} LoRA tensors, found {group_counts.get('lora', 0)}"
        )
    if expected_aligner_tensor_count is not None and group_counts.get('aligner', 0) != expected_aligner_tensor_count:
        failures.append(
            f"expected {expected_aligner_tensor_count} aligner tensors, found {group_counts.get('aligner', 0)}"
        )
    if require_boundary_embeddings and not all(boundary_present.values()):
        failures.append(f"missing boundary embeddings: {boundary_present}")
    return failures


def main() -> None:
    args = parse_args()
    checkpoints = args.checkpoint or DEFAULT_CHECKPOINTS
    reports = []
    print('========== SWIFT HUGINN AUDIO CHECKPOINT INSPECT ==========')
    for checkpoint in checkpoints:
        report = inspect_checkpoint(Path(checkpoint))
        reports.append(report)
        print(f"[checkpoint] path={report['checkpoint_dir']} file_count={report['file_count']}")
        for tensor_report in report['tensor_reports']:
            print(f'[checkpoint] tensor_report={json.dumps(tensor_report, ensure_ascii=False)}')
        print(f"[checkpoint] json_files={list(report['json_reports'])}")

    validation_failures = []
    if (
        args.expected_lora_tensor_count is not None
        or args.expected_aligner_tensor_count is not None
        or args.require_boundary_embeddings
    ):
        for report in reports:
            failures = validate_checkpoint_report(
                report,
                args.expected_lora_tensor_count,
                args.expected_aligner_tensor_count,
                args.require_boundary_embeddings,
            )
            if failures:
                validation_failures.append({'checkpoint': report['checkpoint_dir'], 'failures': failures})
            else:
                print(f"[checkpoint] validation=passed path={report['checkpoint_dir']}")

    output_report = Path(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output_report.with_name(f'{output_report.name}.tmp')
    tmp_output.write_text(
        json.dumps({'checkpoints': reports, 'validation_failures': validation_failures}, ensure_ascii=False, indent=2)
        + '\n',
        encoding='utf-8',
    )
    tmp_output.replace(output_report)
    print(f'[checkpoint] output_report={output_report}')
    if validation_failures:
        raise SystemExit(f'Checkpoint validation failed: {validation_failures}')


if __name__ == '__main__':
    main()
