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
    return {
        'path': str(path),
        'kind': 'tensor_state_dict',
        'tensor_key_count': len(keys),
        'group_counts': groups,
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

    output_report = Path(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output_report.with_name(f'{output_report.name}.tmp')
    tmp_output.write_text(json.dumps({'checkpoints': reports}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp_output.replace(output_report)
    print(f'[checkpoint] output_report={output_report}')


if __name__ == '__main__':
    main()
