#!/usr/bin/env bash
# =====================================================================
# Model 12: RGCK-Net — CV-fold ensemble runner for R011 (output/014/)
#
# Loads the 5 R011 CV best.pt checkpoints, averages sigmoid probabilities
# on the canonical FIW test split, and reports ensemble metrics.
#
# Wall-clock: ~30 min (5 × ~6 min inference, no training).
#
# Usage:
#   bash models/12_rgck_net/AMD/cv_ensemble_r011.sh
# =====================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${MODEL_ROOT}/../.." && pwd)"

PYTHON="${PYTHON:-${MODEL_ROOT}/.venv/bin/python}"
CV_DIR="${CV_DIR:-${MODEL_ROOT}/output/014}"
OUT_DIR="${OUT_DIR:-${CV_DIR}/ensemble}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/datasets/FIW}"
ALIGNED_ROOT="${ALIGNED_ROOT:-${PROJECT_ROOT}/datasets/FIW_aligned}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
GPU_ID="${GPU_ID:-0}"

mkdir -p "${OUT_DIR}"

echo "============================================"
echo "M12 RGCK-Net — CV-fold ensemble for R011"
echo "============================================"
echo "CV dir:       ${CV_DIR}"
echo "Out dir:      ${OUT_DIR}"
echo "Aligned root: ${ALIGNED_ROOT}"
echo "Batch size:   ${BATCH_SIZE}"
echo "ROCm device:  ${GPU_ID}"
echo "Python:       ${PYTHON}"
echo "============================================"

export PYTHONPATH="${PROJECT_ROOT}/models:${PROJECT_ROOT}/models/shared:${PYTHONPATH:-}"

"${PYTHON}" "${SCRIPT_DIR}/cv_ensemble.py" \
    --cv_dir       "${CV_DIR}" \
    --out_dir      "${OUT_DIR}" \
    --dataset      fiw \
    --data_root    "${DATA_ROOT}" \
    --aligned_root "${ALIGNED_ROOT}" \
    --batch_size   "${BATCH_SIZE}" \
    --num_workers  "${NUM_WORKERS}" \
    --rocm_device  "${GPU_ID}" \
    2>&1 | tee "${OUT_DIR}/ensemble.log"

echo "============================================"
echo "Ensemble complete. Results in ${OUT_DIR}/"
echo "============================================"
