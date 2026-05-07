#!/bin/bash
# Chained: test+evaluate R004 best.pt, then launch R005 (full unfreeze, M02-style regime).
# Stops R005 launch if test or evaluate fails (pipefail + exit codes).
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${MODEL_ROOT}/../.." && pwd)"
PYTHON="${MODEL_ROOT}/.venv/bin/python"
R004_DIR="${MODEL_ROOT}/output/004"

export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-FAST}"
export HSA_FORCE_FINE_GRAIN_PCIE="${HSA_FORCE_FINE_GRAIN_PCIE:-1}"
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
export PYTHONPATH="${PROJECT_ROOT}/models:${PROJECT_ROOT}/models/shared:${PYTHONPATH}"

cd "${SCRIPT_DIR}"

echo "============================================"
echo "[1/3] Testing R004 best.pt (ep 4 checkpoint)"
echo "============================================"
"${PYTHON}" test.py \
    --checkpoint "${R004_DIR}/checkpoints/best.pt" \
    --dataset fiw \
    --batch_size 4 \
    --output_dir "${R004_DIR}/results" \
    --num_workers 4 \
    --rocm_device 0 \
    2>&1 | tee "${R004_DIR}/logs/test.log"

echo ""
echo "============================================"
echo "[2/3] Evaluating R004 best.pt"
echo "============================================"
"${PYTHON}" evaluate.py \
    --checkpoint "${R004_DIR}/checkpoints/best.pt" \
    --dataset fiw \
    --batch_size 4 \
    --output_dir "${R004_DIR}/results" \
    --num_workers 4 \
    --rocm_device 0 \
    --full_analysis \
    2>&1 | tee "${R004_DIR}/logs/evaluate.log"

echo ""
echo "============================================"
echo "[3/3] R004 done — launching R005 (full unfreeze, M02-style)"
echo "============================================"
echo ""

SKIP_INSTALL=1 EPOCHS=20 PATIENCE=10 NUM_WORKERS=4 \
    UNFREEZE_BACKBONE_BLOCKS=12 BACKBONE_LR_FACTOR=1.0 \
    LEARNING_RATE=5e-6 WARMUP_EPOCHS=5 DROPOUT=0.2 \
    bash "${SCRIPT_DIR}/run_pipeline.sh"
