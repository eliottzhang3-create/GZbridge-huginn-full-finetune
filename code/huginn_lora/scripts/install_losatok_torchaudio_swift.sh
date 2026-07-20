#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

EXPECTED_TORCH_VERSION="2.11.0+cu128"
EXPECTED_WHEEL_NAME="torchaudio-2.11.0+cu128-cp310-cp310-manylinux_2_28_x86_64.whl"
WHEEL_DIR="${LOSATOK_WHEEL_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/models/LoSATok/wheels}"
WHEEL_PATH="${LOSATOK_TORCHAUDIO_WHEEL:-$WHEEL_DIR/$EXPECTED_WHEEL_NAME}"

echo "========== INSTALL LOSATOK TORCHAUDIO INTO SWIFT ENV =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "python=$(command -v python)"
echo "wheel_path=$WHEEL_PATH"

if [ ! -f "$WHEEL_PATH" ]; then
  echo "Expected compatible torchaudio wheel was not found: $WHEEL_PATH" >&2
  echo "wheel_dir_listing:" >&2
  ls -lh "$WHEEL_DIR" >&2 || true
  exit 1
fi

python - "$EXPECTED_TORCH_VERSION" <<'PY'
import sys
import torch

expected = sys.argv[1]
print(f'[before] torch={torch.__version__} cuda={torch.version.cuda}')
if torch.__version__ != expected:
    raise SystemExit(f'Refuse to modify unexpected Torch environment: expected={expected} actual={torch.__version__}')
PY

# The direct local wheel plus --no-index/--no-deps guarantees pip cannot alter
# the already validated Swift Torch or Transformers installation.
python -m pip install --no-index --no-deps --force-reinstall "$WHEEL_PATH"

python - "$EXPECTED_TORCH_VERSION" <<'PY'
import sys
import torch
import torchaudio

expected_torch = sys.argv[1]
print(f'[after] torch={torch.__version__} cuda={torch.version.cuda}')
print(f'[after] torchaudio={torchaudio.__version__}')
if torch.__version__ != expected_torch:
    raise SystemExit(f'Torch changed unexpectedly: expected={expected_torch} actual={torch.__version__}')
if not torchaudio.__version__.startswith('2.11.0'):
    raise SystemExit(f'Unexpected torchaudio version: {torchaudio.__version__}')

waveform = torch.zeros(1, 16000)
mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=400,
    hop_length=160,
    n_mels=128,
)(waveform)
db = torchaudio.transforms.AmplitudeToDB(top_db=120)(mel)
if tuple(mel.shape) != (1, 128, 101) or not torch.isfinite(db).all():
    raise SystemExit(f'torchaudio transform smoke failed: mel_shape={tuple(mel.shape)}')
print(f'[after] mel_shape={tuple(mel.shape)} db_finite=True')
print('========== LOSATOK TORCHAUDIO INSTALL PASSED ==========')
PY
