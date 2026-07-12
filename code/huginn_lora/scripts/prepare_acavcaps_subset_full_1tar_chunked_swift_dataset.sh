#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Keep this dataset variant separate from the all-shard 4-tar/256-sample route.
export FORMAL_CHUNK_DIR="${FORMAL_CHUNK_DIR:-$REPO_ROOT/data/audio_swift/acavcaps/subset_56_full_1tar_chunks}"
export FORMAL_CATEGORY_LIMITS="${FORMAL_CATEGORY_LIMITS:-00A=12,0M0=8,S00=10,S0A=12,SMA=8,0MA=3,SM0=3}"
export FORMAL_CHUNK_SIZE_TARS="${FORMAL_CHUNK_SIZE_TARS:-1}"
export FORMAL_SKIP_EXISTING="${FORMAL_SKIP_EXISTING:-1}"

# This variant must not silently inherit a cap from an earlier submission shell.
if [ -n "${FORMAL_SAMPLES_PER_TAR:-}" ]; then
  echo "FORMAL_SAMPLES_PER_TAR must be unset for the full-per-tar subset route" >&2
  exit 2
fi
unset FORMAL_SAMPLES_PER_TAR

# An unset cap means each selected tar is read sequentially to EOF.
bash "$SCRIPT_DIR/prepare_acavcaps_formal_chunked_swift_dataset.sh"
