#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/inference.sh
#   bash scripts/inference.sh IHI0050H_amba_chi_architecture_spec
#
# Qwen model answers are written to:
#   data/evaluation/base_model/<SPEC_NAME>/preds.json
#
# Note: scripts/evaluation/eval_base_model.py also runs the OpenAI LLM judge
# after Qwen inference, so export OPENAI_API_KEY if you want scores.json too.

SPEC_NAME="${1:-IHI0050H_amba_chi_architecture_spec}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-4B-Instruct}"
DEVICE="${DEVICE:-cuda}"

QA_DIR="data/datasets/amba_document/eval_qa/${SPEC_NAME}"
MATERIALS_DIR="data/datasets/amba_document/processed/${SPEC_NAME}/materials"
CHUNKS_DIR="data/datasets/amba_document/processed/${SPEC_NAME}/chunks"
OUTPUT_DIR="data/evaluation/base_model/${SPEC_NAME}"

if [[ ! -d "${QA_DIR}" ]]; then
  echo "QA directory not found: ${QA_DIR}" >&2
  exit 1
fi

if [[ ! -d "${MATERIALS_DIR}" ]]; then
  echo "Materials directory not found: ${MATERIALS_DIR}" >&2
  exit 1
fi

if [[ ! -d "${CHUNKS_DIR}" ]]; then
  echo "Chunks directory not found: ${CHUNKS_DIR}" >&2
  exit 1
fi

echo "Running Qwen inference/eval"
echo "  spec:       ${SPEC_NAME}"
echo "  model:      ${MODEL_NAME}"
echo "  device:     ${DEVICE}"
echo "  qa_dir:     ${QA_DIR}"
echo "  output_dir: ${OUTPUT_DIR}"

PYTHONPATH=src python scripts/evaluation/eval_base_model.py \
  qa_dir="${QA_DIR}" \
  materials_dir="${MATERIALS_DIR}" \
  chunks_dir="${CHUNKS_DIR}" \
  output_dir="${OUTPUT_DIR}" \
  model.params.model_name="${MODEL_NAME}" \
  model.params.device="${DEVICE}"

echo "Done."
echo "Qwen outputs: ${OUTPUT_DIR}/preds.json"
echo "Judge scores: ${OUTPUT_DIR}/scores.json"
