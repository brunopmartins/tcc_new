#!/bin/bash
# =============================================================================
# Model 01: Age Synthesis Comparison — Local setup & run script
# =============================================================================
# Runs the full pipeline (train → test → evaluate) directly on the host
# machine without Docker. Autodetects AMD ROCm or NVIDIA CUDA.
# Creates an isolated virtualenv at .venv/ inside this directory.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# With custom parameters:
#   EPOCHS=50 BATCH_SIZE=16 ./setup.sh
#
# Skip the install step after the first run:
#   SKIP_INSTALL=1 ./setup.sh
#
# Environment variables:
#   EPOCHS          Number of training epochs       (default: 100)
#   BATCH_SIZE      Training batch size              (default: 32)
#   LEARNING_RATE   Learning rate                    (default: 1e-4)
#   TRAIN_DATASET   Dataset to train on              (default: kinface)
#   GPU_ID          GPU device index                 (default: 0)
#   USE_AGE_SYNTH   Enable age synthesis (1/0)       (default: 0)
#   SKIP_INSTALL    Skip venv creation/pip install   (default: 0)
# =============================================================================
set -e

# ---------- resolve project paths -------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_DIR="${SCRIPT_DIR}"
SHARED_DIR="${PROJECT_ROOT}/models/shared"
VENV_DIR="${MODEL_DIR}/.venv"
PYTHON="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"

# ---------- configurable parameters -----------------------------------------
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
TRAIN_DATASET="${TRAIN_DATASET:-kinface}"
GPU_ID="${GPU_ID:-0}"
USE_AGE_SYNTH="${USE_AGE_SYNTH:-0}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

# ---------- output directories — wipe and recreate on every run -------------
OUTPUT_DIR="${MODEL_DIR}/output"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"
RESULTS_DIR="${OUTPUT_DIR}/results"
LOG_DIR="${OUTPUT_DIR}/logs"
rm -rf "${OUTPUT_DIR}"
mkdir -p "${CHECKPOINT_DIR}" "${RESULTS_DIR}" "${LOG_DIR}"

# ---------- detect GPU platform from hardware (before torch is installed) ---
detect_platform() {
    # AMD ROCm: /dev/kfd is the compute device node
    if [ -e /dev/kfd ] && ls /dev/dri/renderD* 2>/dev/null | head -1 >/dev/null; then
        echo "rocm"
        return
    fi
    # NVIDIA: nvidia-smi present and returns a GPU
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q "GPU"; then
        echo "cuda"
        return
    fi
    echo "cpu"
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
echo "  Virtualenv:     ${VENV_DIR}"
echo '============================================================'
echo ''

# ---------- create / update virtualenv and install dependencies -------------
if [ "${SKIP_INSTALL}" != "1" ]; then
    echo '[0/3] Setting up Python virtualenv and installing dependencies...'

    # Create venv if it doesn't exist
    if [ ! -x "${PYTHON}" ]; then
        echo "    Creating virtualenv at ${VENV_DIR} ..."
        python3 -m venv "${VENV_DIR}"
    fi

    # Upgrade pip inside the venv (no system-package restrictions apply here)
    "${PIP}" install --quiet --upgrade pip setuptools wheel

    # Install PyTorch — platform-specific wheel index
    case "${PLATFORM}" in
        rocm)
            echo "    Installing PyTorch with ROCm 5.7 support..."
            "${PIP}" install --quiet \
                torch torchvision torchaudio \
                --index-url https://download.pytorch.org/whl/rocm5.7
            ;;
        cuda)
            echo "    Installing PyTorch with CUDA 12.1 support..."
            "${PIP}" install --quiet \
                torch torchvision torchaudio \
                --index-url https://download.pytorch.org/whl/cu121
            ;;
        *)
            echo "    Installing PyTorch (CPU only)..."
            "${PIP}" install --quiet torch torchvision torchaudio
            ;;
    esac

    # Install remaining dependencies
    "${PIP}" install --quiet \
        'timm>=0.9.0' \
        'numpy>=1.24.0' \
        'pandas>=2.0.0' \
        'scikit-learn>=1.3.0' \
        'Pillow>=10.0.0' \
        'tqdm>=4.65.0' \
        'matplotlib>=3.7.0' \
        'seaborn>=0.12.0'

    echo "    Done. Run with SKIP_INSTALL=1 to skip this step next time."
    echo ''
fi

# Verify the venv exists before proceeding
if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: virtualenv not found at ${VENV_DIR}. Run without SKIP_INSTALL=1 first."
    exit 1
fi

# ---------- set PYTHONPATH --------------------------------------------------
export PYTHONPATH="${PROJECT_ROOT}/models:${SHARED_DIR}:${PYTHONPATH}"

# ---------- platform-specific env vars and script directory -----------------
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
"${PYTHON}" train.py \
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
"${PYTHON}" test.py \
    --checkpoint "${CHECKPOINT_DIR}/best.pt" \
    --dataset kinface \
    --output_dir "${RESULTS_DIR}" \
    --save_predictions \
    ${EXTRA_ARGS} \
    2>&1 | tee "${LOG_DIR}/test.log"

echo ''
echo '[3/3] Evaluating...'
"${PYTHON}" evaluate.py \
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
