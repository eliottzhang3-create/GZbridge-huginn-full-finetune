#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_HRM"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1

SOURCE_MANIFEST="${HRM_AUDIOCAPS_SOURCE_MANIFEST:-$REPO_ROOT/data/audio_swift/audiocaps_v2/audiocaps_v2_train_swift.jsonl}"
SOURCE_STATS="${HRM_AUDIOCAPS_SOURCE_STATS:-$SOURCE_MANIFEST.stats.json}"
OUTPUT_MANIFEST="${HRM_AUDIOCAPS_TRAIN_MANIFEST:-$REPO_ROOT/data/audio_swift/audiocaps_v2/audiocaps_v2_train_hrm_audio.jsonl}"
EXPECTED_RECORD_COUNT="${HRM_AUDIOCAPS_EXPECTED_RECORD_COUNT:-89658}"
OVERWRITE="${HRM_AUDIOCAPS_MANIFEST_OVERWRITE:-false}"

OVERWRITE_ARGS=()
case "$OVERWRITE" in
  true|TRUE|1|yes|YES) OVERWRITE_ARGS=(--overwrite) ;;
  false|FALSE|0|no|NO) ;;
  *) echo "HRM_AUDIOCAPS_MANIFEST_OVERWRITE must be true/false, got: $OVERWRITE" >&2; exit 1 ;;
esac

echo "========== PREPARE HRM AUDIO AUDIOCAPS V2 TRAIN MANIFEST =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "PYTHON=$(which python)"
echo "SOURCE_MANIFEST=$SOURCE_MANIFEST"
echo "SOURCE_STATS=$SOURCE_STATS"
echo "OUTPUT_MANIFEST=$OUTPUT_MANIFEST"
echo "OUTPUT_STATS=$OUTPUT_MANIFEST.stats.json"
echo "EXPECTED_RECORD_COUNT=$EXPECTED_RECORD_COUNT"
echo "VERIFY_WAV_HEADERS=true"
echo "OVERWRITE=$OVERWRITE"
echo "TRANSFORMATION=system-user-assistant_to_user-assistant"
echo "SOURCE_IS_READ_ONLY=true"

python -u code/HRM_Audio/scripts/prepare_audiocaps_v2_hrm_audio_manifest.py \
  --source-manifest "$SOURCE_MANIFEST" \
  --source-stats "$SOURCE_STATS" \
  --output-manifest "$OUTPUT_MANIFEST" \
  --expected-record-count "$EXPECTED_RECORD_COUNT" \
  "${OVERWRITE_ARGS[@]}"
