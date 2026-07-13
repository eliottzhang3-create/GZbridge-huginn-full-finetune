from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import wave
from collections import Counter
from pathlib import Path


DEFAULT_DATASET_ROOT = Path('/hpc_stor03/sjtu_home/jinwei.zhang/data/audiocaps_v2')
DEFAULT_SYSTEM = 'You are a helpful assistant that can understand audio and describe it.'
DEFAULT_USER = 'Listen to the audio and describe it.'
ACTIVE_STAGE = 'initializing'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a verified local-WAV Swift JSONL manifest for AudioCaps v2.')
    parser.add_argument('--dataset_root', default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument('--split', default='train', choices=('train', 'val', 'test'))
    parser.add_argument('--output_manifest', required=True)
    parser.add_argument('--audio_id_column', default='youtube_id')
    parser.add_argument('--caption_column', default='caption')
    parser.add_argument('--audio_filename_prefix', default='Y')
    parser.add_argument('--limit_records', type=int, default=None)
    parser.add_argument(
        '--invalid_row_policy',
        choices=('error', 'skip'),
        default='skip',
        help='Whether malformed CSV rows, missing WAV files, and unreadable WAV files stop preparation or are excluded.',
    )
    return parser.parse_args()


def on_signal(signum, _frame) -> None:
    signal_name = signal.Signals(signum).name
    print(f'[manifest] received_signal={signal_name} active_stage={ACTIVE_STAGE}', flush=True)
    raise SystemExit(128 + signum)


def audio_path_for_id(audio_dir: Path, raw_id: str, prefix: str) -> Path:
    stem = Path(raw_id.strip()).stem
    if not stem:
        raise ValueError('Encountered an empty audio identifier')
    if prefix and not stem.startswith(prefix):
        stem = f'{prefix}{stem}'
    return audio_dir / f'{stem}.wav'


def verify_wav_readable(path: Path) -> dict[str, int | str]:
    with wave.open(str(path), 'rb') as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        compression = wf.getcomptype()
        frame_count = wf.getnframes()
    if compression != 'NONE':
        raise ValueError(f'Unsupported compressed WAV {path}: compression={compression}')
    if sample_width != 2:
        raise ValueError(f'Current plugin requires 16-bit PCM WAV: {path}, sample_width_bytes={sample_width}')
    if channels <= 0 or sample_rate <= 0 or frame_count <= 0:
        raise ValueError(
            f'Invalid WAV metadata for {path}: channels={channels} sample_rate={sample_rate} frames={frame_count}'
        )
    return {
        'channels': channels,
        'sample_width_bytes': sample_width,
        'sample_rate': sample_rate,
        'compression_type': compression,
    }


def build_manifest_record(audio_path: Path, caption: str, row: dict[str, str], split: str) -> dict:
    sample_id = audio_path.stem
    return {
        'messages': [
            {'role': 'system', 'content': DEFAULT_SYSTEM},
            {'role': 'user', 'content': DEFAULT_USER},
            {'role': 'assistant', 'content': caption},
        ],
        'audios': [str(audio_path)],
        'metadata': {
            'dataset': 'audiocaps_v2',
            'split': split,
            'sample_id': sample_id,
            'youtube_id': row.get('youtube_id', ''),
            'audiocap_id': row.get('audiocap_id', ''),
        },
    }


def main() -> None:
    global ACTIVE_STAGE
    args = parse_args()
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    dataset_root = Path(args.dataset_root)
    csv_path = dataset_root / f'{args.split}.csv'
    audio_dir = dataset_root / args.split
    output_manifest = Path(args.output_manifest)
    if not csv_path.is_file():
        raise FileNotFoundError(f'CSV not found: {csv_path}')
    if not audio_dir.is_dir():
        raise FileNotFoundError(f'Audio directory not found: {audio_dir}')
    if args.limit_records is not None and args.limit_records <= 0:
        raise ValueError(f'limit_records must be positive when set, got {args.limit_records}')

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp_manifest = output_manifest.with_name(f'{output_manifest.name}.tmp')
    stats_path = output_manifest.with_suffix(f'{output_manifest.suffix}.stats.json')
    tmp_stats = stats_path.with_name(f'{stats_path.name}.tmp')

    print('========== AUDIOCAPS V2 MANIFEST PREP START ==========')
    print(f'[manifest] dataset_root={dataset_root}')
    print(f'[manifest] split={args.split}')
    print(f'[manifest] csv_path={csv_path}')
    print(f'[manifest] audio_dir={audio_dir}')
    print(f'[manifest] output_manifest={output_manifest}')
    print(f'[manifest] audio_id_column={args.audio_id_column}')
    print(f'[manifest] caption_column={args.caption_column}')
    print(f'[manifest] audio_filename_prefix={args.audio_filename_prefix!r}')
    print(f'[manifest] limit_records={args.limit_records}')
    print(f'[manifest] invalid_row_policy={args.invalid_row_policy}')

    validated_rows: list[tuple[Path, str, dict[str, str]]] = []
    source_csv_row_count = 0
    unique_audio_paths: set[str] = set()
    audio_path_counts: Counter[str] = Counter()
    wav_format_counts: Counter[tuple[int | str, ...]] = Counter()
    validation_error_counts: Counter[str] = Counter()
    validation_error_examples: dict[str, list[str]] = {}

    def record_validation_error(kind: str, detail: str) -> None:
        validation_error_counts[kind] += 1
        validation_error_examples.setdefault(kind, [])
        if len(validation_error_examples[kind]) < 10:
            validation_error_examples[kind].append(detail)

    ACTIVE_STAGE = 'reading_csv_and_verifying_wav_files'
    with csv_path.open('r', encoding='utf-8-sig', newline='') as input_file:
        reader = csv.DictReader(input_file)
        headers = list(reader.fieldnames or [])
        missing_columns = [name for name in (args.audio_id_column, args.caption_column) if name not in headers]
        if missing_columns:
            raise ValueError(f'CSV missing required columns {missing_columns}; headers={headers}')

        for row_number, row in enumerate(reader, start=2):
            source_csv_row_count += 1
            raw_id = (row.get(args.audio_id_column) or '').strip()
            caption = (row.get(args.caption_column) or '').strip()
            if not raw_id:
                record_validation_error('empty_audio_id', f'row={row_number}')
                continue
            if not caption:
                record_validation_error('empty_caption', f'row={row_number} audio_id={raw_id}')
                continue

            audio_path = audio_path_for_id(audio_dir, raw_id, args.audio_filename_prefix)
            if not audio_path.is_file():
                record_validation_error('missing_wav', f'row={row_number} path={audio_path}')
                continue
            try:
                wav_metadata = verify_wav_readable(audio_path)
            except (OSError, ValueError, wave.Error) as exc:
                record_validation_error('unreadable_wav', f'row={row_number} path={audio_path} error={exc}')
                continue
            wav_format_counts[
                (
                    wav_metadata['channels'],
                    wav_metadata['sample_width_bytes'],
                    wav_metadata['sample_rate'],
                    wav_metadata['compression_type'],
                )
            ] += 1
            validated_rows.append((audio_path, caption, row))
            unique_audio_paths.add(str(audio_path))
            audio_path_counts[str(audio_path)] += 1

            if args.limit_records is not None and len(validated_rows) >= args.limit_records:
                break

    if validation_error_counts:
        for kind, count in sorted(validation_error_counts.items()):
            print(f'[manifest] validation_error[{kind}]={count} examples={validation_error_examples[kind]}')
        if args.invalid_row_policy == 'error':
            raise ValueError('AudioCaps validation failed; no manifest was committed')
        print(f'[manifest] excluded_invalid_rows={sum(validation_error_counts.values())}')

    record_count = len(validated_rows)
    if record_count == 0:
        raise ValueError('No AudioCaps records were emitted')

    ACTIVE_STAGE = 'writing_manifest'
    first_record = None
    with tmp_manifest.open('w', encoding='utf-8') as output_file:
        for audio_path, caption, row in validated_rows:
            manifest_record = build_manifest_record(audio_path, caption, row, args.split)
            if first_record is None:
                first_record = manifest_record
            output_file.write(json.dumps(manifest_record, ensure_ascii=False) + '\n')
        output_file.flush()
        os.fsync(output_file.fileno())

    ACTIVE_STAGE = 'writing_manifest_and_stats'
    stats = {
        'dataset': 'audiocaps_v2',
        'split': args.split,
        'dataset_root': str(dataset_root),
        'csv_path': str(csv_path),
        'audio_dir': str(audio_dir),
        'audio_id_column': args.audio_id_column,
        'caption_column': args.caption_column,
        'audio_filename_prefix': args.audio_filename_prefix,
        'source_csv_row_count': source_csv_row_count,
        'record_count': record_count,
        'excluded_row_count': sum(validation_error_counts.values()),
        'excluded_row_counts': dict(sorted(validation_error_counts.items())),
        'excluded_row_examples': validation_error_examples,
        'invalid_row_policy': args.invalid_row_policy,
        'unique_audio_path_count': len(unique_audio_paths),
        'duplicate_audio_path_count': sum(count > 1 for count in audio_path_counts.values()),
        'audio_path_verification': 'passed',
        'wav_readability_verification': 'passed',
        'wav_format_counts': {str(key): value for key, value in sorted(wav_format_counts.items())},
        'limit_records': args.limit_records,
    }
    with tmp_stats.open('w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_manifest, output_manifest)
    os.replace(tmp_stats, stats_path)
    ACTIVE_STAGE = 'complete'
    print('========== AUDIOCAPS V2 MANIFEST PREP DONE ==========')
    print(f'[manifest] output_manifest={output_manifest}')
    print(f'[manifest] stats_path={stats_path}')
    print(f'[manifest] record_count={record_count}')
    print(f'[manifest] source_csv_row_count={source_csv_row_count}')
    print(f'[manifest] excluded_row_count={sum(validation_error_counts.values())}')
    print(f'[manifest] excluded_row_counts={dict(sorted(validation_error_counts.items()))}')
    print(f'[manifest] unique_audio_path_count={len(unique_audio_paths)}')
    print(f'[manifest] duplicate_audio_path_count={sum(count > 1 for count in audio_path_counts.values())}')
    print('[manifest] audio_path_verification=passed')
    print('[manifest] wav_readability_verification=passed')
    print(f'[manifest] wav_format_counts={stats["wav_format_counts"]}')
    print(f'[manifest] first_record={json.dumps(first_record, ensure_ascii=False)}')


if __name__ == '__main__':
    main()
