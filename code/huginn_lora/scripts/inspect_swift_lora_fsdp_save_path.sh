#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SWIFT_ROOT="$(python -c 'import pathlib, swift; print(pathlib.Path(swift.__file__).resolve().parent)')"

echo "========== SWIFT LORA/FSDP CHECKPOINT SAVE-PATH INSPECT =========="
echo "[swift] version=$(python -c 'import swift; print(swift.__version__)')"
echo "[swift] root=$SWIFT_ROOT"

echo "========== CANDIDATE SOURCE FILES =========="
rg -l --glob '*.py' 'get_peft_model_state_dict|modules_to_save|save_pretrained|def save_model|def _save|FSDP' \
  "$SWIFT_ROOT" | sort | sed -n '1,120p'

SEARCH_DIRS=()
for directory in trainers tuners arguments llm; do
  if [ -d "$SWIFT_ROOT/$directory" ]; then
    SEARCH_DIRS+=("$SWIFT_ROOT/$directory")
  fi
done
if [ "${#SEARCH_DIRS[@]}" -eq 0 ]; then
  echo "No expected Swift source directories exist below: $SWIFT_ROOT" >&2
  exit 1
fi

echo "========== SAVE / PEFT / MODULES-TO-SAVE MATCHES =========="
rg -n -C 4 --glob '*.py' \
  'get_peft_model_state_dict|modules_to_save|save_pretrained|def save_model|def _save|FSDP' \
  "${SEARCH_DIRS[@]}" | sed -n '1,700p'

echo "========== SFT ARGUMENT MATCHES =========="
rg -n -C 3 --glob '*.py' 'modules_to_save|tuner_type|freeze_aligner|fsdp|save_only_model' \
  "${SEARCH_DIRS[@]}" | sed -n '1,500p'

echo "[result] status=PASS source_inspection=complete"
