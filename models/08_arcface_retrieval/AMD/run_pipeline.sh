#!/bin/bash
# =============================================================================
# Model 08 — ArcFace + Retrieval-Augmented Kinship — AMD ROCm pipeline runner
#
# End-to-end local runner: venv setup → train → test → evaluate. The encoder
# is frozen ArcFace IResNet-100 (weights must already be downloaded — see
# README). Input data must be pre-aligned MTCNN crops; run run_preprocessing.sh
# first.
#
# Usage:
#   bash models/08_arcface_retrieval/AMD/run_pipeline.sh
#
# Override settings via env vars (defaults shown):
#   EPOCHS                20
#   BATCH_SIZE            16          (ArcFace 112×112 is small, can fit bigger)
#   GRAD_ACCUM            2           (effective batch = 32)
#   LEARNING_RATE         1e-4
#   WEIGHT_DECAY          1e-4
#   WARMUP_EPOCHS         3
#   MIN_LR                1e-6
#   SCHEDULER             cosine
#   TRAIN_DATASET         fiw
#   ARCFACE_WEIGHTS       <model_root>/weights/arcface_r100.pth
#   ARCFACE_ARCH          r100        (or r50)
#   IMG_SIZE              112
#   RETRIEVAL_K           32
#   RETRIEVAL_ATTN_LAYERS 2
#   RETRIEVAL_ATTN_HEADS  4
#   EMBEDDING_DIM         512
#   DROPOUT               0.1
#   LOSS                  combined
#   CONTRASTIVE_WEIGHT    0.3
#   TEMPERATURE           0.1
#   RELATION_SET          (auto = TRAIN_DATASET)
#   RELATION_LOSS_WEIGHT  0.15
#   MAX_GALLERY           200000
#   STORE_GALLERY_ON_CPU  0
#   GALLERY_REFRESH_EVERY 0
#   NUM_WORKERS           4
#   PATIENCE              10
#   MAX_GRAD_NORM         1.0
#   DISABLE_AMP           0
#   GPU_ID                0
#   SEED                  42
#   SKIP_INSTALL          0
#   ALIGNED_ROOT          (path to pre-aligned face crops; convention env var)
#   DATA_ROOT             (path to dataset root w/ track-I CSVs; default = original FIW)
# =============================================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${MODEL_ROOT}/../.." && pwd)"

PYTHON="${MODEL_ROOT}/.venv/bin/python"
PIP="${MODEL_ROOT}/.venv/bin/pip"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
MIN_LR="${MIN_LR:-1e-6}"
SCHEDULER="${SCHEDULER:-cosine}"
TRAIN_DATASET="${TRAIN_DATASET:-fiw}"
ARCFACE_WEIGHTS="${ARCFACE_WEIGHTS:-${MODEL_ROOT}/weights/arcface_r100.pth}"
ARCFACE_ARCH="${ARCFACE_ARCH:-r100}"
IMG_SIZE="${IMG_SIZE:-112}"
RETRIEVAL_K="${RETRIEVAL_K:-32}"
RETRIEVAL_ATTN_LAYERS="${RETRIEVAL_ATTN_LAYERS:-2}"
RETRIEVAL_ATTN_HEADS="${RETRIEVAL_ATTN_HEADS:-4}"
EMBEDDING_DIM="${EMBEDDING_DIM:-512}"
DROPOUT="${DROPOUT:-0.1}"
LOSS="${LOSS:-combined}"
CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-0.3}"
TEMPERATURE="${TEMPERATURE:-0.1}"
RELATION_SET="${RELATION_SET:-}"
RELATION_LOSS_WEIGHT="${RELATION_LOSS_WEIGHT:-0.15}"
MAX_GALLERY="${MAX_GALLERY:-200000}"
STORE_GALLERY_ON_CPU="${STORE_GALLERY_ON_CPU:-0}"
GALLERY_REFRESH_EVERY="${GALLERY_REFRESH_EVERY:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PATIENCE="${PATIENCE:-10}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
DISABLE_AMP="${DISABLE_AMP:-0}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
ALIGNED_ROOT="${ALIGNED_ROOT:-}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/datasets/FIW}"

# ---------- ROCm render group --------------------------------------------------
if [ -e /dev/kfd ] && [ -z "${_ROCM_SETUP_COMPLETE}" ]; then
    if ! id -Gn 2>/dev/null | tr ' ' '\n' | grep -qx render; then
        echo "  [ROCm] Adding $(whoami) to render+video groups..."
        sudo usermod -aG render,video "$(whoami)"
    fi
    export EPOCHS BATCH_SIZE GRAD_ACCUM LEARNING_RATE WEIGHT_DECAY WARMUP_EPOCHS MIN_LR SCHEDULER
    export TRAIN_DATASET ARCFACE_WEIGHTS ARCFACE_ARCH IMG_SIZE
    export RETRIEVAL_K RETRIEVAL_ATTN_LAYERS RETRIEVAL_ATTN_HEADS EMBEDDING_DIM DROPOUT
    export LOSS CONTRASTIVE_WEIGHT TEMPERATURE RELATION_SET RELATION_LOSS_WEIGHT
    export MAX_GALLERY STORE_GALLERY_ON_CPU GALLERY_REFRESH_EVERY
    export NUM_WORKERS PATIENCE MAX_GRAD_NORM DISABLE_AMP
    export GPU_ID SEED SKIP_INSTALL DATA_ROOT ALIGNED_ROOT
    export _ROCM_SETUP_COMPLETE=1
    SELF="$(readlink -f "${BASH_SOURCE[0]}")"
    echo "  [ROCm] Restarting script with render group active..."
    exec sg render -c "bash '${SELF}'"
fi

export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-FAST}"
export HSA_FORCE_FINE_GRAIN_PCIE="${HSA_FORCE_FINE_GRAIN_PCIE:-1}"
export HIP_VISIBLE_DEVICES="${GPU_ID}"
export ROCR_VISIBLE_DEVICES="${GPU_ID}"

# ---------- venv ---------------------------------------------------------------
if [ "${SKIP_INSTALL}" != "1" ]; then
    echo '[setup] Installing dependencies into .venv ...'
    if [ ! -x "${PYTHON}" ]; then
        echo "  Creating virtualenv at ${MODEL_ROOT}/.venv ..."
        python3 -m venv "${MODEL_ROOT}/.venv"
    fi
    "${PIP}" install --quiet --upgrade pip setuptools wheel
    echo "  Installing PyTorch with ROCm 5.7 support..."
    "${PIP}" install --quiet \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/rocm5.7
    "${PIP}" install --quiet \
        'numpy>=1.24.0' 'pandas>=2.0.0' 'scikit-learn>=1.3.0' \
        'Pillow>=10.0.0' 'tqdm>=4.65.0' 'matplotlib>=3.7.0'
fi

if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: virtualenv not found at ${MODEL_ROOT}/.venv"; exit 1
fi

# ---------- preconditions ------------------------------------------------------
if [ ! -f "${ARCFACE_WEIGHTS}" ]; then
    echo "ERROR: ArcFace weights not found at ${ARCFACE_WEIGHTS}"
    echo ""
    echo "Download from one of these sources:"
    echo "  https://github.com/deepinsight/insightface/releases (model zoo)"
    echo "  https://huggingface.co/deepinsight/insightface"
    echo ""
    echo "Save as: ${ARCFACE_WEIGHTS}"
    exit 1
fi

if [ ! -d "${DATA_ROOT}" ]; then
    echo "ERROR: dataset root not found at ${DATA_ROOT}"
    echo "Set DATA_ROOT to the FIW directory containing track-I/ CSVs."
    exit 1
fi

if [ -n "${ALIGNED_ROOT}" ] && [ ! -d "${ALIGNED_ROOT}" ]; then
    echo "ERROR: ALIGNED_ROOT set to ${ALIGNED_ROOT} but directory does not exist"
    exit 1
fi

# ---------- numbered output ----------------------------------------------------
OUTPUT_BASE="${MODEL_ROOT}/output"
mkdir -p "${OUTPUT_BASE}"
RUN_ID=0
for d in "${OUTPUT_BASE}"/*; do
    [ -d "${d}" ] || continue
    name="$(basename "${d}")"
    case "${name}" in ''|*[!0-9]*) continue ;; esac
    n=$((10#${name}))
    [ "${n}" -gt "${RUN_ID}" ] && RUN_ID="${n}"
done
RUN_ID=$((RUN_ID + 1))
RUN_LABEL="$(printf '%03d' ${RUN_ID})"
RUN_DIR="${OUTPUT_BASE}/${RUN_LABEL}"
CKPT_DIR="${RUN_DIR}/checkpoints"
RESULTS_DIR="${RUN_DIR}/results"
LOGS_DIR="${RUN_DIR}/logs"
mkdir -p "${CKPT_DIR}" "${RESULTS_DIR}" "${LOGS_DIR}"

echo '============================================'
echo 'Model 08: ArcFace + Retrieval-Augmented Kinship'
echo 'Platform: AMD ROCm (local)'
echo '============================================'
echo "Run ID:            ${RUN_LABEL}"
echo "Output:            ${RUN_DIR}"
echo "Epochs:            ${EPOCHS}"
echo "Batch size:        ${BATCH_SIZE}  (grad_accum ${GRAD_ACCUM} = eff $((BATCH_SIZE * GRAD_ACCUM)))"
echo "Learning rate:     ${LEARNING_RATE}"
echo "Backbone:          ArcFace IResNet-${ARCFACE_ARCH#r} (frozen)"
echo "ArcFace weights:   ${ARCFACE_WEIGHTS}"
echo "Img size:          ${IMG_SIZE}"
echo "Retrieval K:       ${RETRIEVAL_K}"
echo "Attn layers/heads: ${RETRIEVAL_ATTN_LAYERS}/${RETRIEVAL_ATTN_HEADS}"
echo "Embedding dim:     ${EMBEDDING_DIM}"
echo "Loss:              ${LOSS}"
echo "Contrastive w:     ${CONTRASTIVE_WEIGHT}"
echo "Relation w:        ${RELATION_LOSS_WEIGHT}"
echo "Patience:          ${PATIENCE}"
echo "AMP:               $([ "${DISABLE_AMP}" = "1" ] && echo off || echo on)"
echo "Dataset:           ${TRAIN_DATASET}"
echo "Data root:         ${DATA_ROOT}"
echo "Aligned root:      ${ALIGNED_ROOT:-(none — using DATA_ROOT images)}"
echo "Seed:              ${SEED}"
echo "ROCm device:       ${GPU_ID}"
echo "Python:            ${PYTHON}"
echo '============================================'

export PYTHONPATH="${PROJECT_ROOT}/models:${PROJECT_ROOT}/models/shared:${PYTHONPATH}"

TRAIN_ARGS=(
    --train_dataset            "${TRAIN_DATASET}"
    --data_root                "${DATA_ROOT}"
    --epochs                   "${EPOCHS}"
    --batch_size               "${BATCH_SIZE}"
    --gradient_accumulation    "${GRAD_ACCUM}"
    --lr                       "${LEARNING_RATE}"
    --weight_decay             "${WEIGHT_DECAY}"
    --warmup_epochs            "${WARMUP_EPOCHS}"
    --min_lr                   "${MIN_LR}"
    --scheduler                "${SCHEDULER}"
    --arcface_weights          "${ARCFACE_WEIGHTS}"
    --arcface_arch             "${ARCFACE_ARCH}"
    --img_size                 "${IMG_SIZE}"
    --retrieval_k              "${RETRIEVAL_K}"
    --retrieval_attn_layers    "${RETRIEVAL_ATTN_LAYERS}"
    --retrieval_attn_heads     "${RETRIEVAL_ATTN_HEADS}"
    --embedding_dim            "${EMBEDDING_DIM}"
    --dropout                  "${DROPOUT}"
    --loss                     "${LOSS}"
    --contrastive_weight       "${CONTRASTIVE_WEIGHT}"
    --temperature              "${TEMPERATURE}"
    --relation_loss_weight     "${RELATION_LOSS_WEIGHT}"
    --max_gallery              "${MAX_GALLERY}"
    --gallery_refresh_every    "${GALLERY_REFRESH_EVERY}"
    --num_workers              "${NUM_WORKERS}"
    --patience                 "${PATIENCE}"
    --max_grad_norm            "${MAX_GRAD_NORM}"
    --seed                     "${SEED}"
    --rocm_device              "${GPU_ID}"
    --checkpoint_dir           "${CKPT_DIR}"
)
[ -n "${RELATION_SET}" ]            && TRAIN_ARGS+=(--relation_set "${RELATION_SET}")
[ "${STORE_GALLERY_ON_CPU}" = "1" ] && TRAIN_ARGS+=(--store_gallery_on_cpu)
[ "${DISABLE_AMP}" = "1" ]          && TRAIN_ARGS+=(--disable_amp)
[ -n "${ALIGNED_ROOT}" ]            && TRAIN_ARGS+=(--aligned_root "${ALIGNED_ROOT}")

TEST_ARGS=(
    --checkpoint   "${CKPT_DIR}/best.pt"
    --dataset      "${TRAIN_DATASET}"
    --data_root    "${DATA_ROOT}"
    --batch_size   "${BATCH_SIZE}"
    --output_dir   "${RESULTS_DIR}"
    --num_workers  "${NUM_WORKERS}"
    --rocm_device  "${GPU_ID}"
)
[ -n "${ALIGNED_ROOT}" ] && TEST_ARGS+=(--aligned_root "${ALIGNED_ROOT}")

EVAL_ARGS=(
    --checkpoint   "${CKPT_DIR}/best.pt"
    --dataset      "${TRAIN_DATASET}"
    --data_root    "${DATA_ROOT}"
    --batch_size   "${BATCH_SIZE}"
    --output_dir   "${RESULTS_DIR}"
    --num_workers  "${NUM_WORKERS}"
    --rocm_device  "${GPU_ID}"
    --full_analysis
)
[ -n "${ALIGNED_ROOT}" ] && EVAL_ARGS+=(--aligned_root "${ALIGNED_ROOT}")

cd "${SCRIPT_DIR}"

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
echo "Model 08 — Run ${RUN_LABEL} completed"
echo "Results:     ${RESULTS_DIR}"
echo "Checkpoints: ${CKPT_DIR}"
echo "Logs:        ${LOGS_DIR}"
echo '============================================'
