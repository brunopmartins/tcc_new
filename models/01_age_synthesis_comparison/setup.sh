#!/bin/bash
# =============================================================================
# Model 01: Age Synthesis Comparison — Local setup & run script
# =============================================================================
# Runs the full pipeline (train → test → evaluate) directly on the host
# machine without Docker. Autodetects AMD ROCm or NVIDIA CUDA.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# With custom parameters:
#   EPOCHS=50 BATCH_SIZE=16 ./setup.sh
#
# Environment variables:
#   EPOCHS          Number of training epochs       (default: 100)
#   BATCH_SIZE      Training batch size              (default: 32)
#   LEARNING_RATE   Learning rate                    (default: 1e-4)
#   TRAIN_DATASET   Dataset to train on              (default: kinface)
#   GPU_ID          GPU device index                 (default: 0)
#   USE_AGE_SYNTH   Enable age synthesis (1/0)       (default: 0)
#   SKIP_INSTALL    Skip pip install step (1/0)      (default: 0)
# =============================================================================
set -e

# ---------- resolve project paths -------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_DIR="${SCRIPT_DIR}"
SHARED_DIR="${PROJECT_ROOT}/models/shared"

# ---------- configurable parameters -----------------------------------------
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
TRAIN_DATASET="${TRAIN_DATASET:-kinface}"
GPU_ID="${GPU_ID:-0}"
USE_AGE_SYNTH="${USE_AGE_SYNTH:-0}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

# ---------- output directories ----------------------------------------------
OUTPUT_DIR="${MODEL_DIR}/output"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"
RESULTS_DIR="${OUTPUT_DIR}/results"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${CHECKPOINT_DIR}" "${RESULTS_DIR}" "${LOG_DIR}"

# ---------- detect GPU platform ---------------------------------------------
detect_platform() {
    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        # Check whether this is ROCm (HIP) or CUDA
        if python3 -c "import torch; print(torch.version.hip)" 2>/dev/null | grep -q "^[0-9]"; then
            echo "rocm"
        else
            echo "cuda"
        fi
    else
        echo "cpu"
    fi
}

PLATFORM=$(detect_platform)

echo ''
echo '============================================================'
echo '  Model 01: Age Synthesis + All-vs-All Comparison'
echo '============================================================'
echo "  Platform:       ${PLATFORM}"
echo "  Epochs:         ${EPOCHS}"
echo "  Batch Size:     ${BATCH_SIZE}"
echo "  Learning Rate:  ${LEARNING_RATE}"
echo "  Dataset:        ${TRAIN_DATASET}"
echo "  GPU ID:         ${GPU_ID}"
echo "  Age Synthesis:  $([ "${USE_AGE_SYNTH}" = "1" ] && echo 'enabled' || echo 'disabled')"
echo "  Project Root:   ${PROJECT_ROOT}"
echo '============================================================'
echo ''

# ---------- install dependencies (optional) ---------------------------------
if [ "${SKIP_INSTALL}" != "1" ]; then
    echo '[0/3] Installing Python dependencies...'
    pip install --quiet --upgrade pip setuptools wheel
    pip install --quiet \
        torch torchvision torchaudio \
        timm'>=0.9.0' \
        'numpy>=1.24.0' \
        'pandas>=2.0.0' \
        'scikit-learn>=1.3.0' \
        'Pillow>=10.0.0' \
        'tqdm>=4.65.0' \
        'matplotlib>=3.7.0' \
        'seaborn>=0.12.0'
    echo '    Dependencies installed.'
    echo ''
fi

# ---------- set PYTHONPATH --------------------------------------------------
export PYTHONPATH="${PROJECT_ROOT}/models:${SHARED_DIR}:${PYTHONPATH}"

# ---------- platform-specific env vars --------------------------------------
if [ "${PLATFORM}" = "rocm" ]; then
    export MIOPEN_FIND_MODE=FAST
    export HSA_FORCE_FINE_GRAIN_PCIE=1
    export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"
    PLATFORM_DIR="${MODEL_DIR}/AMD"
    EXTRA_ARGS="--rocm_device ${GPU_ID}"
elif [ "${PLATFORM}" = "cuda" ]; then
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    PLATFORM_DIR="${MODEL_DIR}/Nvidia"
    EXTRA_ARGS=""
else
    echo "WARNING: No GPU detected — running on CPU (this will be slow)"
    PLATFORM_DIR="${MODEL_DIR}/Nvidia"
    EXTRA_ARGS=""
fi

# ---------- age synthesis flag ----------------------------------------------
AGE_FLAG=""
if [ "${USE_AGE_SYNTH}" = "1" ]; then
    AGE_FLAG="--use_age_synthesis"
fi

# ---------- run pipeline ----------------------------------------------------
cd "${PLATFORM_DIR}"

echo '[1/3] Training...'
python train.py \
    --train_dataset "${TRAIN_DATASET}" \
    --test_dataset kinface \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LEARNING_RATE}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    ${AGE_FLAG} \
    ${EXTRA_ARGS} \
    2>&1 | tee "${LOG_DIR}/train.log"

echo ''
echo '[2/3] Testing...'
python test.py \
    --checkpoint "${CHECKPOINT_DIR}/best.pt" \
    --dataset kinface \
    --output_dir "${RESULTS_DIR}" \
    --save_predictions \
    ${EXTRA_ARGS} \
    2>&1 | tee "${LOG_DIR}/test.log"

echo ''
echo '[3/3] Evaluating...'
python evaluate.py \
    --checkpoint "${CHECKPOINT_DIR}/best.pt" \
    --dataset kinface \
    --output_dir "${RESULTS_DIR}" \
    --full_analysis \
    --ablation \
    ${EXTRA_ARGS} \
    2>&1 | tee "${LOG_DIR}/evaluate.log"

echo ''
echo '============================================================'
echo '  Model 01 completed!'
echo "  Results:     ${RESULTS_DIR}"
echo "  Checkpoints: ${CHECKPOINT_DIR}"
echo "  Logs:        ${LOG_DIR}"
echo '============================================================'
