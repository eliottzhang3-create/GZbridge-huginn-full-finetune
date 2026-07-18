"""Merge verified Swift caption manifests without copying their audio files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


CAPTION_SYSTEM = 'You are a helpful assistant that can understand audio and describe it.'
CAPTION_USER = 'Listen to the audio and describe it.'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--source',
        action='append',
        required=True,
        help='Repeat NAME=MANIFEST.jsonl. Source records are appended in the supplied order.',
    )
    parser.add_argument('--output_manifest', required=True)
    return parser.parse_args()


def parse_source_spec(spec: str) -> tuple[str, Path]:
    if '=' not in spec:
        raise ValueError(f'Source must use NAME=PATH syntax, got: {spec}')
    name, raw_path = spec.split('=', 1)
    name = name.strip()
    path = Path(raw_path).expanduser()
    if not name or not path:
        raise ValueError(f'Invalid source specification: {spec}')
    if not path.is_file():
        raise FileNotFoundError(f'Source manifest not found: {path}')
    return name, path


def validate_caption_record(record: Any, source_name: str, line_number: int) -> tuple[str, str]:
    if not isinstance(record, dict):
        raise TypeError(f'{source_name} line {line_number} is not a JSON object')
    messages = record.get('messages')
    audios = record.get('audios')
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError(f'{source_name} line {line_number} must contain exactly three messages')
    expected = (("system", CAPTION_SYSTEM), ("user", CAPTION_USER))
    for index, (role, content) in enumerate(expected):
        message = messages[index]
        if not isinstance(message, dict) or message.get('role') != role or message.get('content') != content:
            raise ValueError(f'{source_name} line {line_number} has an incompatible caption prompt')
    assistant = messages[2]
    if not isinstance(assistant, dict) or assistant.get('role') != 'assistant':
        raise ValueError(f'{source_name} line {line_number} has no assistant caption')
    caption = assistant.get('content')
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError(f'{source_name} line {line_number} has an empty caption')
    if not isinstance(audios, list) or len(audios) != 1 or not isinstance(audios[0], str) or not audios[0].strip():
        raise ValueError(f'{source_name} line {line_number} must contain exactly one audio path')
    audio_path = Path(audios[0]).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f'{source_name} line {line_number} references missing audio: {audio_path}')
    metadata = record.get('metadata')
    if isinstance(metadata, dict) and 'source_target' in metadata and metadata['source_target'] != caption.strip():
        raise ValueError(f'{source_name} line {line_number} source_target does not match assistant caption')
    return str(audio_path), caption.strip()


def main() -> None:
    args = parse_args()
    sources = [parse_source_spec(spec) for spec in args.source]
    source_names = [name for name, _ in sources]
    if len(source_names) != len(set(source_names)):
        raise ValueError(f'Duplicate source names are not allowed: {source_names}')

    output_manifest = Path(args.output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp_manifest = output_manifest.with_name(f'{output_manifest.name}.tmp')
    stats_path = output_manifest.with_suffix(f'{output_manifest.suffix}.stats.json')
    tmp_stats = stats_path.with_name(f'{stats_path.name}.tmp')

    source_record_counts: Counter[str] = Counter()
    source_audio_counts: dict[str, set[str]] = {name: set() for name, _ in sources}
    extension_counts: Counter[str] = Counter()
    audio_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    digest = hashlib.sha256()
    first_records: dict[str, dict[str, Any]] = {}
    mixture_index = 0

    print('========== AUDIO CAPTION MIXTURE PREP START ==========', flush=True)
    print(f'[mixture] output_manifest={output_manifest}', flush=True)
    print(f'[mixture] sources={[f"{name}={path}" for name, path in sources]}', flush=True)

    with tmp_manifest.open('w', encoding='utf-8') as output_file:
        for source_name, source_path in sources:
            with source_path.open('r', encoding='utf-8') as source_file:
                for line_number, line in enumerate(source_file, start=1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    audio_path, caption = validate_caption_record(record, source_name, line_number)
                    metadata = record.get('metadata')
                    metadata = dict(metadata) if isinstance(metadata, dict) else {}
                    metadata.update(
                        {
                            'mixture_source': source_name,
                            'mixture_source_manifest': str(source_path),
                            'mixture_source_line': line_number,
                            'mixture_index': mixture_index,
                        }
                    )
                    record['audios'] = [audio_path]
                    record['messages'][2]['content'] = caption
                    record['metadata'] = metadata
                    output_file.write(json.dumps(record, ensure_ascii=False) + '\n')

                    source_record_counts[source_name] += 1
                    source_audio_counts[source_name].add(audio_path)
                    extension_counts[Path(audio_path).suffix.lower() or '<none>'] += 1
                    audio_counts[audio_path] += 1
                    pair_counts[(audio_path, caption)] += 1
                    digest.update(f'{source_name}\0{audio_path}\0{caption}\n'.encode('utf-8'))
                    if source_name not in first_records:
                        first_records[source_name] = record
                    mixture_index += 1
            print(
                f"[mixture] source={source_name} records={source_record_counts[source_name]} "
                f"unique_audio_paths={len(source_audio_counts[source_name])}",
                flush=True,
            )
        output_file.flush()
        os.fsync(output_file.fileno())

    if mixture_index == 0:
        raise ValueError('No records were emitted into the mixture')
    stats = {
        'dataset': 'audio_caption_mixture',
        'record_count': mixture_index,
        'source_manifests': {name: str(path) for name, path in sources},
        'source_record_counts': dict(source_record_counts),
        'source_unique_audio_path_counts': {name: len(paths) for name, paths in source_audio_counts.items()},
        'unique_audio_path_count': len(audio_counts),
        'duplicate_audio_path_count': sum(count > 1 for count in audio_counts.values()),
        'duplicate_audio_caption_pair_count': sum(count > 1 for count in pair_counts.values()),
        'audio_extension_counts': dict(sorted(extension_counts.items())),
        'audio_path_verification': 'passed',
        'caption_prompt_verification': 'passed',
        'audio_caption_pair_verification': 'passed',
        'ordered_pair_sha256': digest.hexdigest(),
        'first_records': first_records,
        'shuffle_policy': 'not_applied_during_preparation; training script controls dataset/DataLoader shuffle',
    }
    tmp_stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp_manifest, output_manifest)
    os.replace(tmp_stats, stats_path)
    print('========== AUDIO CAPTION MIXTURE PREP DONE ==========', flush=True)
    print(f'[mixture] output_manifest={output_manifest}', flush=True)
    print(f'[mixture] stats_path={stats_path}', flush=True)
    print(f'[mixture] record_count={mixture_index}', flush=True)
    print(f'[mixture] source_record_counts={dict(source_record_counts)}', flush=True)
    print('[mixture] audio_path_verification=passed', flush=True)
    print('[mixture] caption_prompt_verification=passed', flush=True)
    print('[mixture] audio_caption_pair_verification=passed', flush=True)


if __name__ == '__main__':
    main()
