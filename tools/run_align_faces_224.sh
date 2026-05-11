#!/bin/bash
# Project-wide wrapper: MTCNN alignment + 224x224 cropping over the FIW dataset.
# Writes to datasets/FIW_aligned_224/ by default. Resumable (skips existing).
#
# Reuses model 08's virtualenv because that's where facenet-pytorch is already
# installed (same dependency set as the 112 version of this script).
#
# Environment overrides:
#   SRC_ROOT     source root (default: $PROJECT_ROOT/datasets/FIW)
#   DST_ROOT     destination (default: $PROJECT_ROOT/datasets/FIW_aligned_224)
#   OUTPUT_SIZE  square output size in px (default: 224)
#   DATASET      'fiw' or 'kinface' (default: fiw)
#   GPU_ID       ROCm/CUDA device index (default: 0)
#   USE_CPU      '1' to force CPU (slower; useful if training is using the GPU)
#   LIMIT        smoke-test cap on number of images (default: unbounded)
#   PYTHON       override Python interpreter
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default to model 08's venv (already has facenet-pytorch + torch + opencv).
DEFAULT_PYTHON="${PROJECT_ROOT}/models/08_arcface_retrieval/.venv/bin/python"
PYTHON="${PYTHON:-${DEFAULT_PYTHON}}"

DATASET="${DATASET:-fiw}"
SRC_ROOT="${SRC_ROOT:-${PROJECT_ROOT}/datasets/FIW}"
DST_ROOT="${DST_ROOT:-${PROJECT_ROOT}/datasets/FIW_aligned_224}"
OUTPUT_SIZE="${OUTPUT_SIZE:-224}"
GPU_ID="${GPU_ID:-0}"
USE_CPU="${USE_CPU:-0}"
LIMIT="${LIMIT:-}"

export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-FAST}"
export HIP_VISIBLE_DEVICES="${GPU_ID}"
export ROCR_VISIBLE_DEVICES="${GPU_ID}"

if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: Python interpreter not found at ${PYTHON}"
    echo "Either set PYTHON=<path> or first build model 08's venv via:"
    echo "  ${PROJECT_ROOT}/models/08_arcface_retrieval/AMD/run_preprocessing.sh"
    exit 1
fi

echo '============================================'
echo "Project-wide MTCNN alignment (${OUTPUT_SIZE}×${OUTPUT_SIZE})"
echo '============================================'
echo "Dataset:    ${DATASET}"
echo "Source:     ${SRC_ROOT}"
echo "Destination:${DST_ROOT}"
echo "Output:     ${OUTPUT_SIZE}×${OUTPUT_SIZE}"
echo "Device:     $([ "${USE_CPU}" = "1" ] && echo CPU || echo GPU${GPU_ID})"
echo "Python:     ${PYTHON}"
echo '============================================'

ARGS=(
    --dataset "${DATASET}"
    --src-root "${SRC_ROOT}"
    --dst-root "${DST_ROOT}"
    --output-size "${OUTPUT_SIZE}"
    --rocm-device "${GPU_ID}"
)
[ "${USE_CPU}" = "1" ] && ARGS+=(--use-cpu)
[ -n "${LIMIT}" ]      && ARGS+=(--limit "${LIMIT}")

"${PYTHON}" "${SCRIPT_DIR}/align_faces_224.py" "${ARGS[@]}"
