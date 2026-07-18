"""Inspect Clotho caption/AQA training inputs before a LoRA continuation run."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_CAPTION_ROOT = Path('/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_caption_huginn')
DEFAULT_AQA_ROOT = Path('/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_aqa_huginn')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--caption_root', default=str(DEFAULT_CAPTION_ROOT))
    parser.add_argument('--caption_manifest', default='train_expand.json')
    parser.add_argument('--aqa_root', default=str(DEFAULT_AQA_ROOT))
    parser.add_argument('--aqa_manifest', default='train.jsonl')
    parser.add_argument('--audio_probe_count', type=int, default=8)
    parser.add_argument('--output_report', required=True)
    return parser.parse_args()


def load_records(path: Path) -> tuple[list[Any], list[str]]:
    with path.open('r', encoding='utf-8') as handle:
        if path.suffix.lower() == '.json':
            payload = json.load(handle)
            if not isinstance(payload, list):
                raise ValueError(f'Expected a JSON list in {path}')
            return payload, []
        records: list[Any] = []
        malformed_lines: list[str] = []
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                malformed_lines.append(f'line={line_number} error={exc}')
        return records, malformed_lines


def resolve_audio_path(dataset_root: Path, record: dict[str, Any]) -> Path:
    raw_path = record.get('audio_path') or record.get('audio')
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError('missing audio_path/audio')
    path = Path(raw_path)
    if not path.is_absolute():
        path = dataset_root / path
    return path.resolve()


def extract_target(record: dict[str, Any], task: str) -> tuple[str, str | None]:
    if task == 'caption':
        target = record.get('caption', record.get('response'))
        if not isinstance(target, str) or not target.strip():
            raise ValueError('missing or empty caption/response')
        return target.strip(), None

    if 'question' in record and 'answer' in record:
        question, target = record['question'], record['answer']
    elif 'query' in record and 'response' in record:
        question, target = record['query'], record['response']
    else:
        raise ValueError('missing question/answer or query/response')
    if not isinstance(question, str) or not question.strip() or not isinstance(target, str) or not target.strip():
        raise ValueError('empty AQA question or answer')
    return target.strip(), question.strip()


def probe_audio(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which('ffprobe')
    if ffprobe is None:
        return {'path': str(path), 'error': 'ffprobe is unavailable'}
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
        return {'path': str(path), 'error': result.stderr.strip() or f'ffprobe exit={result.returncode}'}
    try:
        return {'path': str(path), 'ffprobe': json.loads(result.stdout)}
    except json.JSONDecodeError as exc:
        return {'path': str(path), 'error': f'ffprobe returned invalid JSON: {exc}'}


def inspect_dataset(dataset_root: Path, manifest_name: str, task: str, probe_count: int) -> dict[str, Any]:
    manifest_path = dataset_root / manifest_name
    if not dataset_root.is_dir():
        raise FileNotFoundError(f'Dataset root not found: {dataset_root}')
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Training manifest not found: {manifest_path}')

    records, malformed_lines = load_records(manifest_path)
    error_counts: Counter[str] = Counter()
    error_examples: dict[str, list[str]] = {}
    extension_counts: Counter[str] = Counter()
    audio_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    valid_audio_paths: list[Path] = []
    first_valid_record = None

    def record_error(kind: str, detail: str) -> None:
        error_counts[kind] += 1
        error_examples.setdefault(kind, [])
        if len(error_examples[kind]) < 10:
            error_examples[kind].append(detail)

    for detail in malformed_lines:
        record_error('malformed_jsonl', detail)

    for record_index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            record_error('non_object_record', f'record={record_index} type={type(record).__name__}')
            continue
        try:
            audio_path = resolve_audio_path(dataset_root, record)
            target, question = extract_target(record, task)
        except (TypeError, ValueError) as exc:
            record_error('invalid_fields', f'record={record_index} error={exc}')
            continue
        if not audio_path.is_file():
            record_error('missing_audio', f'record={record_index} path={audio_path}')
            continue

        rendered_audio = str(audio_path)
        extension_counts[audio_path.suffix.lower() or '<none>'] += 1
        audio_counts[rendered_audio] += 1
        pair_counts[(rendered_audio, target)] += 1
        valid_audio_paths.append(audio_path)
        if first_valid_record is None:
            first_valid_record = {
                'audio_path': rendered_audio,
                'target': target,
                'question': question,
                'source_record_keys': sorted(record),
            }

    unique_audio_paths = sorted(set(valid_audio_paths), key=str)
    probes = [probe_audio(path) for path in unique_audio_paths[:probe_count]]
    probe_failures = [probe for probe in probes if 'error' in probe]
    return {
        'dataset_root': str(dataset_root),
        'manifest_path': str(manifest_path),
        'task': task,
        'source_record_count': len(records),
        'valid_record_count': len(valid_audio_paths),
        'invalid_record_count': sum(error_counts.values()),
        'invalid_record_counts': dict(sorted(error_counts.items())),
        'invalid_record_examples': error_examples,
        'unique_audio_path_count': len(unique_audio_paths),
        'duplicate_audio_path_count': sum(count > 1 for count in audio_counts.values()),
        'duplicate_audio_caption_pair_count': sum(count > 1 for count in pair_counts.values()),
        'audio_extension_counts': dict(sorted(extension_counts.items())),
        'audio_probe_count': len(probes),
        'audio_probe_failures': probe_failures,
        'audio_probes': probes,
        'first_valid_record': first_valid_record,
        'validation_passed': not error_counts and not probe_failures and bool(valid_audio_paths),
    }


def main() -> None:
    args = parse_args()
    if args.audio_probe_count <= 0:
        raise ValueError(f'audio_probe_count must be positive, got {args.audio_probe_count}')

    print('========== CLOTHO CONTINUATION INPUT INSPECT ==========', flush=True)
    caption_report = inspect_dataset(
        Path(args.caption_root), args.caption_manifest, 'caption', args.audio_probe_count
    )
    aqa_report = inspect_dataset(Path(args.aqa_root), args.aqa_manifest, 'aqa', args.audio_probe_count)
    for name, report in (('caption', caption_report), ('aqa', aqa_report)):
        print(
            f"[inspect] dataset={name} manifest={report['manifest_path']} "
            f"source_records={report['source_record_count']} valid_records={report['valid_record_count']} "
            f"invalid_records={report['invalid_record_count']} unique_audio_paths={report['unique_audio_path_count']}",
            flush=True,
        )
        print(f"[inspect] dataset={name} invalid_counts={report['invalid_record_counts']}", flush=True)
        print(f"[inspect] dataset={name} audio_extensions={report['audio_extension_counts']}", flush=True)
        print(f"[inspect] dataset={name} audio_probe_failures={len(report['audio_probe_failures'])}", flush=True)
        print(f"[inspect] dataset={name} first_valid_record={json.dumps(report['first_valid_record'], ensure_ascii=False)}", flush=True)

    report = {'caption': caption_report, 'aqa': aqa_report}
    output_report = Path(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output_report.with_name(f'{output_report.name}.tmp')
    tmp_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp_output.replace(output_report)
    print(f'[inspect] output_report={output_report}', flush=True)
    if not caption_report['validation_passed'] or not aqa_report['validation_passed']:
        raise SystemExit('Clotho continuation input inspection failed; do not prepare manifests yet.')
    print('========== CLOTHO CONTINUATION INPUT INSPECT PASSED ==========', flush=True)


if __name__ == '__main__':
    main()
