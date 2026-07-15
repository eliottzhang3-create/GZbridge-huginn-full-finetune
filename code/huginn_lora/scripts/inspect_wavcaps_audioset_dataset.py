"""Inspect the public WavCaps AudioSet metadata and FLAC layout without modifying it."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_DATASET_ROOT = Path('/hpc_stor03/public/shared/data/raa/WavCaps')
DEFAULT_AUDIO_SUBDIR = 'AudioSet_SL_flac'
LIKELY_CAPTION_FIELDS = ('caption', 'captions', 'description', 'text', 'sentence')
LIKELY_ID_FIELDS = ('id', 'audio_id', 'youtube_id', 'uid', 'name')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_root', default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument('--audio_subdir', default=DEFAULT_AUDIO_SUBDIR)
    parser.add_argument('--metadata_root', default=None)
    parser.add_argument('--metadata_schema_records', type=int, default=100)
    parser.add_argument('--flac_samples', type=int, default=8)
    parser.add_argument('--output_report', required=True)
    return parser.parse_args()


def json_record_lists(payload: Any) -> dict[str, list[dict[str, Any]]]:
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return {'<root>': payload}
    if not isinstance(payload, dict):
        return {}
    return {
        key: value
        for key, value in payload.items()
        if isinstance(value, list) and all(isinstance(item, dict) for item in value)
    }


def json_value_preview(value: Any, limit: int = 180) -> str:
    rendered = json.dumps(value, ensure_ascii=False)
    return rendered if len(rendered) <= limit else f'{rendered[:limit]}...'


def inspect_record_list(records: list[dict[str, Any]], scan_limit: int) -> dict[str, Any]:
    scanned = records[: min(len(records), scan_limit)]
    key_counts: Counter[str] = Counter()
    value_examples: dict[str, list[str]] = {}
    for record in scanned:
        for key, value in record.items():
            key_counts[key] += 1
            value_examples.setdefault(key, [])
            if len(value_examples[key]) < 2:
                value_examples[key].append(json_value_preview(value))

    return {
        'record_count': len(records),
        'scanned_record_count': len(scanned),
        'field_presence_counts': dict(sorted(key_counts.items())),
        'field_examples': {key: value_examples[key] for key in sorted(value_examples)},
        'likely_caption_fields_present': [field for field in LIKELY_CAPTION_FIELDS if key_counts[field]],
        'likely_id_fields_present': [field for field in LIKELY_ID_FIELDS if key_counts[field]],
        'first_record': records[0] if records else None,
    }


def ffprobe_flac(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which('ffprobe')
    if ffprobe is None:
        return {'path': str(path), 'error': 'ffprobe is not available'}
    result = subprocess.run(
        [
            ffprobe,
            '-v', 'error',
            '-show_entries', 'format=duration:stream=codec_name,sample_rate,channels',
            '-of', 'json',
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return {
            'path': str(path),
            'error': result.stderr.strip() or f'ffprobe failed with exit code {result.returncode}',
        }
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {'path': str(path), 'error': f'ffprobe emitted invalid JSON: {exc}'}
    return {'path': str(path), 'ffprobe': payload}


def main() -> None:
    args = parse_args()
    if args.metadata_schema_records <= 0 or args.flac_samples <= 0:
        raise ValueError('metadata_schema_records and flac_samples must be positive')

    dataset_root = Path(args.dataset_root)
    audio_dir = dataset_root / 'audio' / args.audio_subdir
    metadata_root = Path(args.metadata_root) if args.metadata_root else dataset_root / 'json'
    if not dataset_root.is_dir():
        raise FileNotFoundError(f'WavCaps dataset root not found: {dataset_root}')
    if not audio_dir.is_dir():
        raise FileNotFoundError(f'AudioSet FLAC directory not found: {audio_dir}')
    if not metadata_root.is_dir():
        raise FileNotFoundError(f'WavCaps metadata directory not found: {metadata_root}')

    json_paths = sorted(path for path in metadata_root.rglob('*.json') if path.is_file())
    audio_paths = sorted(path for path in audio_dir.glob('*.flac') if path.is_file())
    audioset_json_paths = [path for path in json_paths if 'audioset' in str(path).lower()]

    print('========== WAVCAPS AUDIOSET DATASET INSPECT START ==========')
    print(f'[inspect] dataset_root={dataset_root}')
    print(f'[inspect] metadata_root={metadata_root}')
    print(f'[inspect] audio_dir={audio_dir}')
    print(f'[inspect] metadata_json_count={len(json_paths)}')
    print(f'[inspect] audioset_metadata_candidate_count={len(audioset_json_paths)}')
    print(f'[inspect] audio_flac_count={len(audio_paths)}')
    print(f'[inspect] ffprobe={shutil.which("ffprobe")}')

    metadata_reports = []
    for json_path in audioset_json_paths:
        relative_path = str(json_path.relative_to(dataset_root))
        try:
            payload = json.loads(json_path.read_text(encoding='utf-8'))
            record_lists = json_record_lists(payload)
            list_reports = {
                name: inspect_record_list(records, args.metadata_schema_records)
                for name, records in record_lists.items()
            }
            report = {
                'path': str(json_path),
                'relative_path': relative_path,
                'size_bytes': json_path.stat().st_size,
                'top_level_type': type(payload).__name__,
                'top_level_keys': sorted(payload) if isinstance(payload, dict) else None,
                'record_lists': list_reports,
            }
        except Exception as exc:  # pragma: no cover - remote data dependent
            report = {
                'path': str(json_path),
                'relative_path': relative_path,
                'error': f'{type(exc).__name__}: {exc}',
            }
        metadata_reports.append(report)
        print(f'[metadata] report={json.dumps(report, ensure_ascii=False)}')

    if not audioset_json_paths:
        print('[metadata] no AudioSet-named JSON was found; inspect the JSON listing in the report.')
    for path in json_paths:
        print(f'[metadata-file] relative_path={path.relative_to(dataset_root)} size_bytes={path.stat().st_size}')

    sample_paths = audio_paths[: args.flac_samples]
    flac_reports = [ffprobe_flac(path) for path in sample_paths]
    for report in flac_reports:
        print(f'[flac] sample={json.dumps(report, ensure_ascii=False)}')

    output_report = Path(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    report = {
        'dataset_root': str(dataset_root),
        'metadata_root': str(metadata_root),
        'audio_dir': str(audio_dir),
        'metadata_json_count': len(json_paths),
        'metadata_files': [
            {'relative_path': str(path.relative_to(dataset_root)), 'size_bytes': path.stat().st_size}
            for path in json_paths
        ],
        'audioset_metadata_reports': metadata_reports,
        'audio_flac_count': len(audio_paths),
        'flac_samples': flac_reports,
        'ffprobe_path': shutil.which('ffprobe'),
    }
    tmp_output = output_report.with_name(f'{output_report.name}.tmp')
    tmp_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp_output.replace(output_report)

    print('========== WAVCAPS AUDIOSET DATASET INSPECT DONE ==========')
    print(f'[inspect] output_report={output_report}')
    if not audioset_json_paths:
        raise SystemExit('No AudioSet metadata candidate was found; do not generate a training manifest yet.')
    if not audio_paths:
        raise SystemExit('No AudioSet FLAC files were found; do not generate a training manifest yet.')


if __name__ == '__main__':
    main()
