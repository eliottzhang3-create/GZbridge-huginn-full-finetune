"""Build a verified metadata-only Swift JSONL manifest for WavCaps AudioSet."""

from __future__ import annotations

import argparse
import json
import os
import signal
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_DATASET_ROOT = Path('/hpc_stor03/public/shared/data/raa/WavCaps')
DEFAULT_AUDIO_SUBDIR = 'AudioSet_SL_flac'
DEFAULT_METADATA_FILE = 'json/AudioSet_SL.jsonl'
DEFAULT_SYSTEM = 'You are a helpful assistant that can understand audio and describe it.'
DEFAULT_USER = 'Listen to the audio and describe it.'
ACTIVE_STAGE = 'initializing'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_root', default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument('--audio_subdir', default=DEFAULT_AUDIO_SUBDIR)
    parser.add_argument('--metadata_file', default=DEFAULT_METADATA_FILE)
    parser.add_argument('--output_manifest', required=True)
    parser.add_argument('--limit_records', type=int, default=None)
    parser.add_argument('--invalid_row_policy', choices=('error', 'skip'), default='skip')
    return parser.parse_args()


def on_signal(signum, _frame) -> None:
    signal_name = signal.Signals(signum).name
    print(f'[manifest] received_signal={signal_name} active_stage={ACTIVE_STAGE}', flush=True)
    raise SystemExit(128 + signum)


def build_manifest_record(audio_path: Path, record: dict[str, Any]) -> dict[str, Any]:
    sample_id = str(record['key']).strip()
    caption = str(record['target']).strip()
    return {
        'messages': [
            {'role': 'system', 'content': DEFAULT_SYSTEM},
            {'role': 'user', 'content': DEFAULT_USER},
            {'role': 'assistant', 'content': caption},
        ],
        'audios': [str(audio_path)],
        'metadata': {
            'dataset': 'wavcaps',
            'subset': 'AudioSet_SL',
            'sample_id': sample_id,
            'source_metadata_path': str(record.get('source', '')),
            'prompt': str(record.get('prompt', '')),
            'target_len': record.get('target_len'),
            'source_len': record.get('source_len'),
            'text_type': str(record.get('text-type', '')),
            'task_type': str(record.get('task-type', '')),
            'audio_language': str(record.get('audio_language', '')),
            'text_language': str(record.get('text_language', '')),
        },
    }


def main() -> None:
    global ACTIVE_STAGE
    args = parse_args()
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    if args.limit_records is not None and args.limit_records <= 0:
        raise ValueError(f'limit_records must be positive when set, got {args.limit_records}')

    dataset_root = Path(args.dataset_root)
    audio_dir = dataset_root / 'audio' / args.audio_subdir
    metadata_path = dataset_root / args.metadata_file
    output_manifest = Path(args.output_manifest)
    if not audio_dir.is_dir():
        raise FileNotFoundError(f'AudioSet FLAC directory not found: {audio_dir}')
    if not metadata_path.is_file():
        raise FileNotFoundError(f'AudioSet JSONL metadata file not found: {metadata_path}')

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp_manifest = output_manifest.with_name(f'{output_manifest.name}.tmp')
    stats_path = output_manifest.with_suffix(f'{output_manifest.suffix}.stats.json')
    tmp_stats = stats_path.with_name(f'{stats_path.name}.tmp')

    print('========== WAVCAPS AUDIOSET MANIFEST PREP START ==========')
    print(f'[manifest] dataset_root={dataset_root}')
    print(f'[manifest] metadata_path={metadata_path}')
    print(f'[manifest] audio_dir={audio_dir}')
    print(f'[manifest] output_manifest={output_manifest}')
    print(f'[manifest] limit_records={args.limit_records}')
    print(f'[manifest] invalid_row_policy={args.invalid_row_policy}')
    print('[manifest] id_field=key')
    print('[manifest] caption_field=target')
    print('[manifest] audio_path_rule=<audio_dir>/<key>.flac')
    print('[manifest] source_field_is_metadata_only=true')

    source_record_count = 0
    emitted_record_count = 0
    error_counts: Counter[str] = Counter()
    error_examples: dict[str, list[str]] = {}
    audio_path_counts: Counter[str] = Counter()
    metadata_key_counts: Counter[str] = Counter()
    first_record: dict[str, Any] | None = None

    def record_error(kind: str, detail: str) -> None:
        error_counts[kind] += 1
        error_examples.setdefault(kind, [])
        if len(error_examples[kind]) < 10:
            error_examples[kind].append(detail)

    ACTIVE_STAGE = 'streaming_jsonl_and_verifying_audio_paths'
    with metadata_path.open('r', encoding='utf-8') as input_file, tmp_manifest.open('w', encoding='utf-8') as output_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                record_error('empty_metadata_line', f'line={line_number}')
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                record_error('malformed_jsonl', f'line={line_number} error={exc}')
                continue
            if not isinstance(record, dict):
                record_error('non_object_metadata', f'line={line_number} type={type(record).__name__}')
                continue

            source_record_count += 1
            sample_id = str(record.get('key') or '').strip()
            caption = str(record.get('target') or '').strip()
            if not sample_id:
                record_error('empty_key', f'line={line_number}')
                continue
            if not caption:
                record_error('empty_target', f'line={line_number} key={sample_id}')
                continue

            audio_path = audio_dir / f'{sample_id}.flac'
            if not audio_path.is_file():
                record_error('missing_flac', f'line={line_number} key={sample_id} path={audio_path}')
                continue

            manifest_record = build_manifest_record(audio_path, record)
            output_file.write(json.dumps(manifest_record, ensure_ascii=False) + '\n')
            if first_record is None:
                first_record = manifest_record
            emitted_record_count += 1
            audio_path_counts[str(audio_path)] += 1
            metadata_key_counts[sample_id] += 1
            if args.limit_records is not None and emitted_record_count >= args.limit_records:
                break

        output_file.flush()
        os.fsync(output_file.fileno())

    if error_counts and args.invalid_row_policy == 'error':
        raise ValueError(f'WavCaps validation failed: {dict(sorted(error_counts.items()))}')
    if emitted_record_count == 0:
        raise ValueError('No WavCaps AudioSet records were emitted')

    ACTIVE_STAGE = 'writing_stats'
    duplicate_audio_path_count = sum(count > 1 for count in audio_path_counts.values())
    duplicate_metadata_key_count = sum(count > 1 for count in metadata_key_counts.values())
    stats = {
        'dataset': 'wavcaps',
        'subset': 'AudioSet_SL',
        'dataset_root': str(dataset_root),
        'metadata_path': str(metadata_path),
        'audio_dir': str(audio_dir),
        'id_field': 'key',
        'caption_field': 'target',
        'audio_suffix': '.flac',
        'source_record_count': source_record_count,
        'record_count': emitted_record_count,
        'excluded_row_count': sum(error_counts.values()),
        'excluded_row_counts': dict(sorted(error_counts.items())),
        'excluded_row_examples': error_examples,
        'invalid_row_policy': args.invalid_row_policy,
        'unique_audio_path_count': len(audio_path_counts),
        'duplicate_audio_path_count': duplicate_audio_path_count,
        'duplicate_metadata_key_count': duplicate_metadata_key_count,
        'audio_path_verification': 'passed',
        'metadata_pairing_verification': 'passed',
        'limit_records': args.limit_records,
    }
    with tmp_stats.open('w', encoding='utf-8') as output_file:
        json.dump(stats, output_file, ensure_ascii=False, indent=2)
        output_file.write('\n')
        output_file.flush()
        os.fsync(output_file.fileno())

    os.replace(tmp_manifest, output_manifest)
    os.replace(tmp_stats, stats_path)
    ACTIVE_STAGE = 'complete'
    print('========== WAVCAPS AUDIOSET MANIFEST PREP DONE ==========')
    print(f'[manifest] output_manifest={output_manifest}')
    print(f'[manifest] stats_path={stats_path}')
    print(f'[manifest] source_record_count={source_record_count}')
    print(f'[manifest] record_count={emitted_record_count}')
    print(f'[manifest] excluded_row_count={sum(error_counts.values())}')
    print(f'[manifest] excluded_row_counts={dict(sorted(error_counts.items()))}')
    print(f'[manifest] unique_audio_path_count={len(audio_path_counts)}')
    print(f'[manifest] duplicate_audio_path_count={duplicate_audio_path_count}')
    print(f'[manifest] duplicate_metadata_key_count={duplicate_metadata_key_count}')
    print('[manifest] audio_path_verification=passed')
    print('[manifest] metadata_pairing_verification=passed')
    print(f'[manifest] first_record={json.dumps(first_record, ensure_ascii=False)}')


if __name__ == '__main__':
    main()
