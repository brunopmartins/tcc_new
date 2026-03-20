#!/bin/bash
# =============================================================================
# Model 01: Age Synthesis Comparison — AMD ROCm pipeline runner
# Local host execution (no Docker).
#
# Run from anywhere in the project:
#   bash models/01_age_synthesis_comparison/AMD/run_pipeline.sh
#
# Or from within the model root or AMD/ directory:
#   bash AMD/run_pipeline.sh
#   bash run_pipeline.sh
#
# Override any setting via environment variable:
#   EPOCHS=50 bash AMD/run_pipeline.sh
# =============================================================================
set -eo pipefail

# ---------- resolve paths from script location --------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"    # .../AMD/
MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"                   # .../01_age_synthesis_comparison/
PROJECT_ROOT="$(cd "${MODEL_ROOT}/../.." && pwd)"              # .../tcc_new/

# ---------- Python interpreter (project venv) ---------------------------------
PYTHON="${MODEL_ROOT}/.venv/bin/python"
if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: Virtual environment not found at ${MODEL_ROOT}/.venv"
    echo "  Create it with:"
    echo "    python3 -m venv ${MODEL_ROOT}/.venv"
    echo "    ${MODEL_ROOT}/.venv/bin/pip install -r <requirements>"
    exit 1
fi

# ---------- ROCm environment (must be set before Python imports torch) --------
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-FAST}"
export HSA_FORCE_FINE_GRAIN_PCIE="${HSA_FORCE_FINE_GRAIN_PCIE:-1}"
export HIP_VISIBLE_DEVICES="${GPU_ID:-0}"
export ROCR_VISIBLE_DEVICES="${GPU_ID:-0}"

# ---------- render group check (AMD GPU requires membership) ------------------
if ! id -Gn 2>/dev/null | tr ' ' '\n' | grep -qx render; then
    echo "WARNING: Current user is not in the 'render' group."
    echo "  AMD GPU (/dev/kfd) may be inaccessible — torch.cuda.is_available() may return False."
    echo "  Fix: sudo usermod -aG render,video \$(whoami)  then log out and back in."
fi

# ---------- SAM pretrained weights (auto-download if missing) -----------------
SAM_WEIGHTS="${MODEL_ROOT}/SAM/pretrained_models/sam_ffhq_aging.pt"
if [ ! -f "${SAM_WEIGHTS}" ]; then
    echo "SAM weights not found at ${SAM_WEIGHTS}"
    echo "Downloading sam_ffhq_aging.pt (~2.2 GB) from Google Drive..."
    mkdir -p "$(dirname "${SAM_WEIGHTS}")"
    "${MODEL_ROOT}/.venv/bin/pip" install -q gdown 2>/dev/null
    "${MODEL_ROOT}/.venv/bin/gdown" \
        "https://drive.google.com/uc?id=1XyumF6_fdAxFmxpFcmPf-q84LU_22EMC" \
        -O "${SAM_WEIGHTS}"
    if [ ! -f "${SAM_WEIGHTS}" ]; then
        echo "ERROR: Failed to download SAM weights."
        echo "  Download manually from: https://drive.google.com/file/d/1XyumF6_fdAxFmxpFcmPf-q84LU_22EMC"
        echo "  Place at: ${SAM_WEIGHTS}"
        exit 1
    fi
    echo "SAM weights downloaded successfully."
else
    echo "SAM weights found: ${SAM_WEIGHTS}"
fi

# ---------- all settings with best defaults -----------------------------------
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"   # SAM (StyleGAN2 1024) needs ~7.7 GB; batch>8 OOMs on 12 GB GPU
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
TRAIN_DATASET="${TRAIN_DATASET:-kinface}"
BACKBONE="${BACKBONE:-resnet50}"
AGGREGATION="${AGGREGATION:-attention}"
LOSS="${LOSS:-bce}"
USE_AGE_SYNTHESIS="${USE_AGE_SYNTHESIS:-1}"
SEED="${SEED:-42}"
ROCM_DEVICE="${GPU_ID:-0}"

# ---------- numbered output folder --------------------------------------------
OUTPUT_BASE="${MODEL_ROOT}/output"
mkdir -p "${OUTPUT_BASE}"
RUN_ID=1
while [ -d "${OUTPUT_BASE}/$(printf '%03d' ${RUN_ID})" ]; do
    RUN_ID=$((RUN_ID + 1))
done
RUN_LABEL="$(printf '%03d' ${RUN_ID})"
RUN_DIR="${OUTPUT_BASE}/${RUN_LABEL}"
CKPT_DIR="${RUN_DIR}/checkpoints"
RESULTS_DIR="${RUN_DIR}/results"
LOGS_DIR="${RUN_DIR}/logs"
mkdir -p "${CKPT_DIR}" "${RESULTS_DIR}" "${LOGS_DIR}"

echo '============================================'
echo 'Model 01: Age Synthesis Comparison'
echo 'Platform: AMD ROCm (local)'
echo '============================================'
echo "Run ID:        ${RUN_LABEL}"
echo "Output:        ${RUN_DIR}"
echo "Epochs:        ${EPOCHS}"
echo "Batch size:    ${BATCH_SIZE}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Weight decay:  ${WEIGHT_DECAY}"
echo "Backbone:      ${BACKBONE}"
echo "Aggregation:   ${AGGREGATION}"
echo "Loss:          ${LOSS}"
echo "Dataset:       ${TRAIN_DATASET}"
echo "Seed:          ${SEED}"
echo "ROCm device:   ${ROCM_DEVICE}"
echo "Python:        ${PYTHON}"
echo "Age synthesis: $([ "${USE_AGE_SYNTHESIS}" = "1" ] && echo enabled || echo disabled)"
echo '============================================'

# ---------- argument arrays (bash arrays avoid all word-splitting issues) -----
TRAIN_ARGS=(
    --train_dataset  "${TRAIN_DATASET}"
    --test_dataset   "${TRAIN_DATASET}"
    --epochs         "${EPOCHS}"
    --batch_size     "${BATCH_SIZE}"
    --lr             "${LEARNING_RATE}"
    --weight_decay   "${WEIGHT_DECAY}"
    --backbone       "${BACKBONE}"
    --aggregation    "${AGGREGATION}"
    --loss           "${LOSS}"
    --seed           "${SEED}"
    --rocm_device    "${ROCM_DEVICE}"
    --checkpoint_dir "${CKPT_DIR}"
)
[ "${USE_AGE_SYNTHESIS}" = "1" ] && TRAIN_ARGS+=(--use_age_synthesis)

TEST_ARGS=(
    --checkpoint   "${CKPT_DIR}/best.pt"
    --dataset      "${TRAIN_DATASET}"
    --batch_size   "${BATCH_SIZE}"
    --output_dir   "${RESULTS_DIR}"
    --rocm_device  "${ROCM_DEVICE}"
    --save_predictions
)

EVAL_ARGS=(
    --checkpoint   "${CKPT_DIR}/best.pt"
    --dataset      "${TRAIN_DATASET}"
    --output_dir   "${RESULTS_DIR}"
    --rocm_device  "${ROCM_DEVICE}"
    --full_analysis
    --ablation
)

# ---------- cd into AMD/ so Python relative imports resolve correctly ---------
cd "${SCRIPT_DIR}"

# ---------- pipeline ----------------------------------------------------------
echo ''
echo '[1/3] Training...'
"${PYTHON}" train.py "${TRAIN_ARGS[@]}" 2>&1 | tee "${LOGS_DIR}/train.log"

echo ''
echo '[2/3] Testing...'
"${PYTHON}" test.py "${TEST_ARGS[@]}" 2>&1 | tee "${LOGS_DIR}/test.log"

echo ''
echo '[3/3] Evaluating...'
"${PYTHON}" evaluate.py "${EVAL_ARGS[@]}" 2>&1 | tee "${LOGS_DIR}/evaluate.log"

echo ''
echo '============================================'
echo "Model 01 — Run ${RUN_LABEL} completed!"
echo "Results: ${RUN_DIR}"
echo '============================================'
