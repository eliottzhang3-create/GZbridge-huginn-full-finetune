from __future__ import annotations

import argparse
import csv
import json
import random
import wave
from collections import Counter
from dataclasses import fields
from pathlib import Path
from typing import Any


DEFAULT_DATASET_ROOT = Path('/hpc_stor03/sjtu_home/jinwei.zhang/data/audiocaps_v2')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Inspect AudioCaps v2 CSV/WAV layout before manifest generation.')
    parser.add_argument('--dataset_root', default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument('--split', default='train', choices=('train', 'val', 'test'))
    parser.add_argument('--audio_id_column', default='youtube_id')
    parser.add_argument('--caption_column', default='caption')
    parser.add_argument('--audio_filename_prefix', default='Y')
    parser.add_argument('--wave_samples', type=int, default=10)
    parser.add_argument('--output_report', required=True)
    return parser.parse_args()


def audio_path_for_id(audio_dir: Path, raw_id: str, prefix: str) -> Path:
    stem = Path(raw_id.strip()).stem
    if not stem:
        raise ValueError('Encountered an empty audio identifier')
    if prefix and not stem.startswith(prefix):
        stem = f'{prefix}{stem}'
    return audio_dir / f'{stem}.wav'


def read_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f'CSV has no header: {csv_path}')
        return list(reader.fieldnames), list(reader)


def inspect_wave(path: Path) -> dict[str, Any]:
    with wave.open(str(path), 'rb') as wf:
        return {
            'path': str(path),
            'channels': wf.getnchannels(),
            'sample_width_bytes': wf.getsampwidth(),
            'sample_rate': wf.getframerate(),
            'frame_count': wf.getnframes(),
            'compression_type': wf.getcomptype(),
            'compression_name': wf.getcompname(),
            'duration_seconds': wf.getnframes() / float(wf.getframerate()),
        }


def inspect_swift_fields() -> dict[str, Any]:
    try:
        from swift.arguments.sft_args import SftArguments
    except Exception as exc:  # pragma: no cover - depends on remote runtime
        return {'import_error': f'{type(exc).__name__}: {exc}'}

    available = {field.name: field.default for field in fields(SftArguments)}
    requested = ('num_train_epochs', 'save_strategy', 'save_steps', 'save_total_limit')
    return {
        name: {'present': name in available, 'default': repr(available.get(name))}
        for name in requested
    }


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    csv_path = dataset_root / f'{args.split}.csv'
    audio_dir = dataset_root / args.split
    if not csv_path.is_file():
        raise FileNotFoundError(f'CSV not found: {csv_path}')
    if not audio_dir.is_dir():
        raise FileNotFoundError(f'Audio directory not found: {audio_dir}')

    headers, rows = read_rows(csv_path)
    required = (args.audio_id_column, args.caption_column)
    missing_columns = [column for column in required if column not in headers]
    if missing_columns:
        raise ValueError(f'CSV missing required columns {missing_columns}; headers={headers}')

    missing_audio: list[str] = []
    empty_captions = 0
    audio_paths: list[Path] = []
    caption_counter: Counter[str] = Counter()
    for row_number, row in enumerate(rows, start=2):
        raw_id = (row.get(args.audio_id_column) or '').strip()
        caption = (row.get(args.caption_column) or '').strip()
        if not raw_id:
            raise ValueError(f'Empty {args.audio_id_column} at CSV row {row_number}')
        if not caption:
            empty_captions += 1
        else:
            caption_counter[caption] += 1
        audio_path = audio_path_for_id(audio_dir, raw_id, args.audio_filename_prefix)
        audio_paths.append(audio_path)
        if not audio_path.is_file():
            missing_audio.append(str(audio_path))

    existing_audio = [path for path in audio_paths if path.is_file()]
    rng = random.Random(42)
    sample_paths = existing_audio[: min(5, len(existing_audio))]
    remaining = existing_audio[len(sample_paths):]
    if remaining and args.wave_samples > len(sample_paths):
        sample_paths.extend(rng.sample(remaining, k=min(args.wave_samples - len(sample_paths), len(remaining))))
    wave_samples = [inspect_wave(path) for path in sample_paths]

    print('========== AUDIOCAPS V2 DATASET INSPECT ==========')
    print(f'[inspect] dataset_root={dataset_root}')
    print(f'[inspect] split={args.split}')
    print(f'[inspect] csv_path={csv_path}')
    print(f'[inspect] audio_dir={audio_dir}')
    print(f'[inspect] csv_headers={headers}')
    print(f'[inspect] csv_rows={len(rows)}')
    print(f'[inspect] unique_audio_paths={len(set(audio_paths))}')
    print(f'[inspect] missing_audio_paths={len(missing_audio)}')
    print(f'[inspect] empty_captions={empty_captions}')
    print(f'[inspect] duplicate_nonempty_captions={sum(count > 1 for count in caption_counter.values())}')
    print(f'[inspect] first_rows={json.dumps(rows[:3], ensure_ascii=False)}')
    for sample in wave_samples:
        print(f'[inspect] wave_sample={json.dumps(sample, ensure_ascii=False)}')
    print(f'[inspect] swift_sft_fields={json.dumps(inspect_swift_fields(), ensure_ascii=False)}')

    report = {
        'dataset_root': str(dataset_root),
        'split': args.split,
        'csv_path': str(csv_path),
        'audio_dir': str(audio_dir),
        'csv_headers': headers,
        'csv_rows': len(rows),
        'audio_id_column': args.audio_id_column,
        'caption_column': args.caption_column,
        'audio_filename_prefix': args.audio_filename_prefix,
        'unique_audio_paths': len(set(audio_paths)),
        'missing_audio_paths': len(missing_audio),
        'missing_audio_examples': missing_audio[:10],
        'empty_captions': empty_captions,
        'wave_samples': wave_samples,
        'swift_sft_fields': inspect_swift_fields(),
    }
    output_report = Path(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    tmp_report = output_report.with_name(f'{output_report.name}.tmp')
    with tmp_report.open('w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    tmp_report.replace(output_report)
    print(f'[inspect] output_report={output_report}')

    if missing_audio or empty_captions:
        raise SystemExit('AudioCaps layout inspection failed; see counts above before manifest generation.')


if __name__ == '__main__':
    main()
