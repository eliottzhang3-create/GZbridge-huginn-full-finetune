#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export FORMAL_CHUNK_DIR="${FORMAL_CHUNK_DIR:-$REPO_ROOT/data/audio_swift/acavcaps/formal_chunks_all_4tar_256}"
export FORMAL_CATEGORY_LIMITS="${FORMAL_CATEGORY_LIMITS:-ALL}"
export FORMAL_CHUNK_SIZE_TARS="${FORMAL_CHUNK_SIZE_TARS:-4}"
export FORMAL_SKIP_EXISTING="${FORMAL_SKIP_EXISTING:-1}"

export FORMAL_SAMPLES_PER_TAR="${FORMAL_SAMPLES_PER_TAR:-256}"

bash "$SCRIPT_DIR/prepare_acavcaps_formal_chunked_swift_dataset.sh"
