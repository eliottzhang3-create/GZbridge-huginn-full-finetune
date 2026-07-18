#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1

AUDIOCAPS_MANIFEST="${AUDIOCAPS_TRAIN_MANIFEST:-$REPO_ROOT/data/audio_swift/audiocaps_v2/audiocaps_v2_train_swift.jsonl}"
AUDIOCAPS_STATS="$AUDIOCAPS_MANIFEST.stats.json"
CLOTHO_CAPTION_ROOT="${CLOTHO_CAPTION_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_caption_huginn}"
CLOTHO_CAPTION_MANIFEST="${CLOTHO_CAPTION_MANIFEST:-train_expand.json}"
ARTIFACT_DIR="${AUDIOCAPS_CLOTHO_ARTIFACT_DIR:-$REPO_ROOT/data/audio_swift/continuation}"
CLOTHO_SWIFT_MANIFEST="${CLOTHO_CAPTION_SWIFT_MANIFEST:-$ARTIFACT_DIR/clotho_v2_caption_train_swift.jsonl}"
MIXTURE_MANIFEST="${AUDIOCAPS_CLOTHO_MIXTURE_MANIFEST:-$ARTIFACT_DIR/audiocaps_v2_clotho_v2_caption_train_swift.jsonl}"

if [ ! -s "$AUDIOCAPS_MANIFEST" ] || [ ! -s "$AUDIOCAPS_STATS" ]; then
  echo "AudioCaps manifest or stats is missing: manifest=$AUDIOCAPS_MANIFEST stats=$AUDIOCAPS_STATS" >&2
  exit 1
fi
if [ ! -f "$CLOTHO_CAPTION_ROOT/$CLOTHO_CAPTION_MANIFEST" ]; then
  echo "Clotho caption manifest is missing: $CLOTHO_CAPTION_ROOT/$CLOTHO_CAPTION_MANIFEST" >&2
  exit 1
fi
mkdir -p "$ARTIFACT_DIR"

python - "$AUDIOCAPS_STATS" <<'PY'
import json
import sys

with open(sys.argv[1], encoding='utf-8') as handle:
    stats = json.load(handle)
if stats.get('dataset') != 'audiocaps_v2' or stats.get('split') != 'train':
    raise SystemExit(f"Unexpected AudioCaps stats: dataset={stats.get('dataset')!r} split={stats.get('split')!r}")
if stats.get('audio_path_verification') != 'passed' or stats.get('wav_readability_verification') != 'passed':
    raise SystemExit('AudioCaps source manifest is not fully verified')
if not isinstance(stats.get('record_count'), int) or stats['record_count'] <= 0:
    raise SystemExit(f"Unexpected AudioCaps record_count: {stats.get('record_count')!r}")
print(f"[mixture] verified_audiocaps_records={stats['record_count']}")
PY

echo "========== PREPARE AUDIOCAPS + CLOTHO V2 CAPTION MIXTURE =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "audiocaps_manifest=$AUDIOCAPS_MANIFEST"
echo "clotho_caption_root=$CLOTHO_CAPTION_ROOT"
echo "clotho_caption_manifest=$CLOTHO_CAPTION_MANIFEST"
echo "clotho_swift_manifest=$CLOTHO_SWIFT_MANIFEST"
echo "mixture_manifest=$MIXTURE_MANIFEST"
echo "mixture_policy=full_concatenation_no_ratio_control_training_shuffle_enabled_later"

python -u code/huginn_lora/scripts/prepare_huginn_audio_dataset.py \
  --dataset_dir "$CLOTHO_CAPTION_ROOT" \
  --input_manifest "$CLOTHO_CAPTION_MANIFEST" \
  --output_manifest "$CLOTHO_SWIFT_MANIFEST" \
  --task caption \
  --dataset_name clotho_v2_caption \
  --verify_audio_paths

python -u code/huginn_lora/scripts/prepare_audio_caption_mixture.py \
  --source "audiocaps_v2=$AUDIOCAPS_MANIFEST" \
  --source "clotho_v2_caption=$CLOTHO_SWIFT_MANIFEST" \
  --output_manifest "$MIXTURE_MANIFEST"

python - "$MIXTURE_MANIFEST.stats.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding='utf-8') as handle:
    stats = json.load(handle)
required = ('audio_path_verification', 'caption_prompt_verification', 'audio_caption_pair_verification')
failed = {name: stats.get(name) for name in required if stats.get(name) != 'passed'}
if failed:
    raise SystemExit(f'Mixture verification failed: {failed}')
source_counts = stats.get('source_record_counts', {})
if not source_counts.get('audiocaps_v2') or not source_counts.get('clotho_v2_caption'):
    raise SystemExit(f'Mixture source counts are incomplete: {source_counts}')
print(f"[mixture] verified_record_count={stats.get('record_count')}")
print(f"[mixture] verified_source_record_counts={source_counts}")
PY

echo "========== PREPARE AUDIOCAPS + CLOTHO V2 CAPTION MIXTURE DONE =========="
