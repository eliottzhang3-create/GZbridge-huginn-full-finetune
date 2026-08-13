#!/usr/bin/env bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
if [[ -f "$USER_CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx 'swift_ouro'; then
    conda activate swift_ouro
  fi
fi

REPO_ROOT="/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

OUTPUT="${BAT_QFORMER_OUTPUT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/bat/qformer_contract_audit.json}"
SOURCE="${BAT_QFORMER_SOURCE:-/hpc_stor03/sjtu_home/jinwei.zhang/code/OWL/src/slam_llm/models/projector.py}"
case "$OUTPUT" in
  /hpc_stor03/public|/hpc_stor03/public/*)
    echo "Refusing public output path: $OUTPUT" >&2
    exit 2
    ;;
esac

python -u code/Ouro_audio/bat/scripts/audit_bat_qformer_contract.py \
  --encoder-dim "${BAT_QFORMER_ENCODER_DIM:-768}" \
  --llm-dim "${BAT_QFORMER_LLM_DIM:-2048}" \
  --layers "${BAT_QFORMER_LAYERS:-8}" \
  --query-len "${BAT_QFORMER_QUERY_LEN:-64}" \
  --batch-size "${BAT_QFORMER_BATCH_SIZE:-2}" \
  --encoder-seq-len "${BAT_QFORMER_ENCODER_SEQ_LEN:-515}" \
  --qformer-source "$SOURCE" \
  --output "$OUTPUT"
