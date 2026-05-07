#!/bin/bash
# Chained: wait for R005 eval to finish, then run option E (per-relation
# thresholds on R001 best.pt) followed by option A launch (R006 contrastive).
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${MODEL_ROOT}/../.." && pwd)"
PYTHON="${MODEL_ROOT}/.venv/bin/python"
R001_DIR="${MODEL_ROOT}/output/001"
R005_DIR="${MODEL_ROOT}/output/005"

# Wait for R005 eval to finish (per_relation.json is written last)
echo "Waiting for R005 test+eval to finish..."
until [ -f "${R005_DIR}/results/per_relation.json" ]; do
    sleep 60
done
echo "R005 eval done."
echo ""

export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-FAST}"
export HSA_FORCE_FINE_GRAIN_PCIE="${HSA_FORCE_FINE_GRAIN_PCIE:-1}"
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
export PYTHONPATH="${PROJECT_ROOT}/models:${PROJECT_ROOT}/models/shared:${PYTHONPATH}"

echo "============================================"
echo "[E] Per-relation thresholds on R001 best.pt"
echo "============================================"
"${PYTHON}" "${PROJECT_ROOT}/tools/per_relation_thresholds.py" \
    --model 05 \
    --checkpoint "${R001_DIR}/checkpoints/best.pt" \
    --dataset fiw \
    --batch_size 4 \
    --num_workers 4 \
    --output_dir "${R001_DIR}/results" \
    2>&1 | tee "${R001_DIR}/logs/per_relation_thresholds.log"

echo ""
echo "============================================"
echo "[A] Launching R006: contrastive loss only"
echo "============================================"

SKIP_INSTALL=1 EPOCHS=20 PATIENCE=10 NUM_WORKERS=4 \
    LOSS=contrastive RELATION_LOSS_WEIGHT=0 \
    bash "${SCRIPT_DIR}/run_pipeline.sh"
