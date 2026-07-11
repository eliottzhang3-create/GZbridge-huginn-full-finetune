#!/bin/bash
set -euo pipefail

export FORMAL_CHUNK_SIZE_TARS="${FORMAL_CHUNK_SIZE_TARS:-4}"
export FORMAL_SAMPLES_PER_TAR="${FORMAL_SAMPLES_PER_TAR:-64}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
bash "$SCRIPT_DIR/prepare_acavcaps_formal_chunked_swift_dataset.sh"
