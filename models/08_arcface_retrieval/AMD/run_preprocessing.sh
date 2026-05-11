#!/bin/bash
# Run MTCNN alignment + 112x112 cropping over the FIW dataset, once.
# Output is cached under models/08_arcface_retrieval/data/fiw_aligned/.
# Subsequent training runs read from there.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${MODEL_ROOT}/../.." && pwd)"
PYTHON="${MODEL_ROOT}/.venv/bin/python"

DATASET="${DATASET:-fiw}"
SRC_ROOT="${SRC_ROOT:-${PROJECT_ROOT}/datasets/FIW}"
DST_ROOT="${DST_ROOT:-${MODEL_ROOT}/data/${DATASET}_aligned}"
GPU_ID="${GPU_ID:-0}"
USE_CPU="${USE_CPU:-0}"
LIMIT="${LIMIT:-}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-FAST}"
export HIP_VISIBLE_DEVICES="${GPU_ID}"
export ROCR_VISIBLE_DEVICES="${GPU_ID}"

if [ "${SKIP_INSTALL}" != "1" ]; then
    if [ ! -x "${PYTHON}" ]; then
        echo "  Creating virtualenv at ${MODEL_ROOT}/.venv ..."
        python3 -m venv "${MODEL_ROOT}/.venv"
        "${MODEL_ROOT}/.venv/bin/pip" install --quiet --upgrade pip setuptools wheel
    fi
    echo "[setup] Installing preprocessing dependencies..."
    "${MODEL_ROOT}/.venv/bin/pip" install --quiet \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/rocm5.7
    "${MODEL_ROOT}/.venv/bin/pip" install --quiet \
        facenet-pytorch opencv-python-headless numpy pillow tqdm
fi

if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: virtualenv missing"; exit 1
fi

echo '============================================'
echo 'Model 08 — MTCNN preprocessing'
echo '============================================'
echo "Dataset:    ${DATASET}"
echo "Source:     ${SRC_ROOT}"
echo "Destination:${DST_ROOT}"
echo "Device:     $([ "${USE_CPU}" = "1" ] && echo CPU || echo GPU${GPU_ID})"
echo '============================================'

ARGS=(
    --dataset "${DATASET}"
    --src-root "${SRC_ROOT}"
    --dst-root "${DST_ROOT}"
    --rocm-device "${GPU_ID}"
)
[ "${USE_CPU}" = "1" ] && ARGS+=(--use-cpu)
[ -n "${LIMIT}" ]      && ARGS+=(--limit "${LIMIT}")

"${PYTHON}" "${MODEL_ROOT}/preprocessing/align_faces.py" "${ARGS[@]}"
