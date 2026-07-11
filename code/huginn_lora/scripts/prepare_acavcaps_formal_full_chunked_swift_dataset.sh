#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export FORMAL_CHUNK_DIR="${FORMAL_CHUNK_DIR:-$REPO_ROOT/data/audio_swift/acavcaps/formal_chunks_full}"
export FORMAL_CHUNK_SIZE_TARS="${FORMAL_CHUNK_SIZE_TARS:-1}"
export FORMAL_SKIP_EXISTING="${FORMAL_SKIP_EXISTING:-1}"

# Formal mode means full tar coverage, so keep samples_per_tar unset unless the
# caller explicitly overrides it for emergency debugging.
if [ "${FORMAL_SAMPLES_PER_TAR+x}" = "x" ] && [ -n "${FORMAL_SAMPLES_PER_TAR}" ]; then
  export FORMAL_SAMPLES_PER_TAR
else
  unset FORMAL_SAMPLES_PER_TAR || true
fi

bash "$SCRIPT_DIR/prepare_acavcaps_formal_chunked_swift_dataset.sh"
