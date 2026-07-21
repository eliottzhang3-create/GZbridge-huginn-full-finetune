#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1

DATASET_ROOT="${CLOTHOAQA_DATASET_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_aqa_huginn}"
SOURCE_MANIFEST="${CLOTHOAQA_SOURCE_MANIFEST:-train.jsonl}"
OUTPUT_MANIFEST="${CLOTHOAQA_TRAIN_MANIFEST:-$REPO_ROOT/data/audio_swift/clotho_aqa/clotho_aqa_train_swift.jsonl}"
STATS_PATH="$OUTPUT_MANIFEST.stats.json"
TMP_MANIFEST="$OUTPUT_MANIFEST.tmp"

if [ ! -f "$DATASET_ROOT/$SOURCE_MANIFEST" ]; then
  echo "ClothoAQA source manifest is missing: $DATASET_ROOT/$SOURCE_MANIFEST" >&2
  exit 1
fi
mkdir -p "$(dirname "$OUTPUT_MANIFEST")"

echo "========== PREPARE CLOTHOAQA LOSATOK SWIFT MANIFEST =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "dataset_root=$DATASET_ROOT"
echo "source_manifest=$SOURCE_MANIFEST"
echo "output_manifest=$OUTPUT_MANIFEST"
echo "tmp_manifest=$TMP_MANIFEST"
echo "stats_path=$STATS_PATH"
echo "audio_policy=absolute_paths_verified_no_audio_copy"

# This validates every referenced path before committing the converted JSONL.
python -u code/huginn_lora/scripts/prepare_huginn_audio_dataset.py \
  --dataset_dir "$DATASET_ROOT" \
  --input_manifest "$SOURCE_MANIFEST" \
  --output_manifest "$TMP_MANIFEST" \
  --task aqa \
  --dataset_name clotho_aqa \
  --verify_audio_paths

python - "$TMP_MANIFEST" "$OUTPUT_MANIFEST" "$STATS_PATH" "$DATASET_ROOT" "$SOURCE_MANIFEST" <<'PY'
import json
import os
import sys
from collections import Counter
from pathlib import Path

manifest_path = Path(sys.argv[1])
output_manifest = Path(sys.argv[2])
stats_path = Path(sys.argv[3])
dataset_root = Path(sys.argv[4])
source_manifest = Path(sys.argv[5])
tmp_stats_path = stats_path.with_name(f'{stats_path.name}.tmp')
required_system = 'You are a helpful assistant that can understand audio and answer questions about it.'
records = 0
audio_paths = set()
extension_counts = Counter()
first_record = None

with manifest_path.open(encoding='utf-8') as handle:
    for line_number, line in enumerate(handle, start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        messages = record.get('messages')
        audios = record.get('audios')
        if not isinstance(messages, list) or len(messages) != 3:
            raise SystemExit(f'Invalid converted messages at line {line_number}')
        roles = [message.get('role') if isinstance(message, dict) else None for message in messages]
        if roles != ['system', 'user', 'assistant']:
            raise SystemExit(f'Invalid converted roles at line {line_number}: {roles}')
        if messages[0].get('content') != required_system:
            raise SystemExit(f'Unexpected ClothoAQA system prompt at line {line_number}')
        if not isinstance(messages[1].get('content'), str) or not messages[1]['content'].strip():
            raise SystemExit(f'Empty ClothoAQA question at line {line_number}')
        if not isinstance(messages[2].get('content'), str) or not messages[2]['content'].strip():
            raise SystemExit(f'Empty ClothoAQA answer at line {line_number}')
        if not isinstance(audios, list) or len(audios) != 1 or not isinstance(audios[0], str):
            raise SystemExit(f'Invalid audio field at line {line_number}')
        audio_path = Path(audios[0])
        if not audio_path.is_absolute() or not audio_path.is_file():
            raise SystemExit(f'Missing/non-absolute audio at line {line_number}: {audio_path}')
        records += 1
        audio_paths.add(str(audio_path))
        extension_counts[audio_path.suffix.lower() or '<none>'] += 1
        if first_record is None:
            first_record = record

if records == 0:
    raise SystemExit('Converted ClothoAQA manifest is empty')
payload = {
    'dataset': 'clotho_aqa',
    'dataset_root': str(dataset_root),
    'source_manifest': str(dataset_root / source_manifest),
    'output_manifest': str(output_manifest),
    'record_count': records,
    'unique_audio_path_count': len(audio_paths),
    'audio_extension_counts': dict(sorted(extension_counts.items())),
    'audio_path_verification': 'passed',
    'aqa_prompt_verification': 'passed',
    'first_record': first_record,
}
with tmp_stats_path.open('w', encoding='utf-8') as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
    handle.write('\n')
    handle.flush()
    os.fsync(handle.fileno())
os.replace(manifest_path, output_manifest)
os.replace(tmp_stats_path, stats_path)
print(f'[manifest] record_count={records}')
print(f'[manifest] unique_audio_path_count={len(audio_paths)}')
print(f'[manifest] audio_extension_counts={dict(sorted(extension_counts.items()))}')
print('[manifest] audio_path_verification=passed')
print('[manifest] aqa_prompt_verification=passed')
print(f'[manifest] stats_path={stats_path}')
PY

echo "========== PREPARE CLOTHOAQA LOSATOK SWIFT MANIFEST DONE =========="
