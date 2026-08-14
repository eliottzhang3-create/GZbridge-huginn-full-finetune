#!/usr/bin/env bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_ouro"
REPO_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/code/Ouro_audio:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

AUDIO_ROOT="${BAT_AUDIO_ROOT:-/hpc_stor03/public/shared/data/raa/AudioSet}"
REVERB_ROOT="${BAT_REVERB_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/SpatialSoundQA/mp3d_reverb}"
STAGE1="${BAT_STAGE1_MANIFEST:?Set BAT_STAGE1_MANIFEST}"
STAGE2="${BAT_STAGE2_MANIFEST:?Set BAT_STAGE2_MANIFEST}"
STAGE3="${BAT_STAGE3_MANIFEST:?Set BAT_STAGE3_MANIFEST}"
REPORT_ROOT="${BAT_MANIFEST_AUDIT_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/bat/manifest_audits}"
mkdir -p "$REPORT_ROOT"
case "$REPORT_ROOT" in /hpc_stor03/public|/hpc_stor03/public/*) echo "Refusing public output" >&2; exit 2;; esac

python -u code/Ouro_audio/bat/scripts/audit_bat_swift_manifest.py \
  --manifest "$STAGE1" --stage I --expected-count 278784 \
  --audio-root "$AUDIO_ROOT" --reverb-root "$REVERB_ROOT" \
  --output-report "$REPORT_ROOT/stage1_manifest_audit.json"
python -u code/Ouro_audio/bat/scripts/audit_bat_swift_manifest.py \
  --manifest "$STAGE2" --stage II --expected-count 514784 \
  --audio-root "$AUDIO_ROOT" --reverb-root "$REVERB_ROOT" \
  --output-report "$REPORT_ROOT/stage2_manifest_audit.json"
python -u code/Ouro_audio/bat/scripts/audit_bat_swift_manifest.py \
  --manifest "$STAGE3" --stage III --expected-count 872312 \
  --audio-root "$AUDIO_ROOT" --reverb-root "$REVERB_ROOT" \
  --output-report "$REPORT_ROOT/stage3_manifest_audit.json"
echo "========== BAT STAGE-I/II/III MANIFEST AUDIT PASSED =========="
