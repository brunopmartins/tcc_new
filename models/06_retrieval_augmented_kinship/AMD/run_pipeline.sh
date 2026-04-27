#!/bin/bash
# =============================================================================
# Model 06 — Retrieval-Augmented Kinship — AMD ROCm pipeline runner
#
# End-to-end local runner: venv setup → train → test → evaluate. Encoder
# is frozen by default; the only large memory structure is the retrieval
# gallery, which can be offloaded to CPU with STORE_GALLERY_ON_CPU=1.
#
# Usage:
#   bash models/06_retrieval_augmented_kinship/AMD/run_pipeline.sh
#
# Override settings via env vars (defaults shown):
#   EPOCHS                60
#   BATCH_SIZE            8
#   GRAD_ACCUM            4
#   LEARNING_RATE         1e-4
#   WEIGHT_DECAY          1e-4
#   WARMUP_EPOCHS         3
#   MIN_LR                1e-6
#   SCHEDULER             cosine
#   TRAIN_DATASET         fiw
#   BACKBONE              vit_base_patch16_224
#   IMG_SIZE              224
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
#   PATIENCE              20
#   MAX_GRAD_NORM         1.0
#   DISABLE_AMP           0
#   NO_GRAD_CKPT          0
#   FREEZE_BACKBONE       1
#   GPU_ID                0
#   SEED                  42
#   SKIP_INSTALL          0
# =============================================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${MODEL_ROOT}/../.." && pwd)"

PYTHON="${MODEL_ROOT}/.venv/bin/python"
PIP="${MODEL_ROOT}/.venv/bin/pip"

EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
MIN_LR="${MIN_LR:-1e-6}"
SCHEDULER="${SCHEDULER:-cosine}"
TRAIN_DATASET="${TRAIN_DATASET:-fiw}"
BACKBONE="${BACKBONE:-vit_base_patch16_224}"
IMG_SIZE="${IMG_SIZE:-224}"
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
PATIENCE="${PATIENCE:-20}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
DISABLE_AMP="${DISABLE_AMP:-0}"
NO_GRAD_CKPT="${NO_GRAD_CKPT:-0}"
FREEZE_BACKBONE="${FREEZE_BACKBONE:-1}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

# ---------- ROCm render group --------------------------------------------------
if [ -e /dev/kfd ] && [ -z "${_ROCM_SETUP_COMPLETE}" ]; then
    if ! id -Gn 2>/dev/null | tr ' ' '\n' | grep -qx render; then
        echo "  [ROCm] Adding $(whoami) to render+video groups for /dev/kfd access..."
        sudo usermod -aG render,video "$(whoami)"
    fi
    export EPOCHS BATCH_SIZE GRAD_ACCUM LEARNING_RATE WEIGHT_DECAY WARMUP_EPOCHS MIN_LR SCHEDULER
    export TRAIN_DATASET BACKBONE IMG_SIZE
    export RETRIEVAL_K RETRIEVAL_ATTN_LAYERS RETRIEVAL_ATTN_HEADS EMBEDDING_DIM DROPOUT
    export LOSS CONTRASTIVE_WEIGHT TEMPERATURE RELATION_SET RELATION_LOSS_WEIGHT
    export MAX_GALLERY STORE_GALLERY_ON_CPU GALLERY_REFRESH_EVERY
    export NUM_WORKERS PATIENCE MAX_GRAD_NORM DISABLE_AMP NO_GRAD_CKPT FREEZE_BACKBONE
    export GPU_ID SEED SKIP_INSTALL
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

if ! id -Gn 2>/dev/null | tr ' ' '\n' | grep -qx render; then
    echo "WARNING: Not in the 'render' group — GPU may not be detected."
fi

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
        'timm>=0.9.12' \
        'numpy>=1.24.0' \
        'pandas>=2.0.0' \
        'scikit-learn>=1.3.0' \
        'Pillow>=10.0.0' \
        'tqdm>=4.65.0' \
        'matplotlib>=3.7.0'
    echo "  Done."
    echo ''
fi

if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: virtualenv not found at ${MODEL_ROOT}/.venv"
    exit 1
fi

# ---------- numbered output ----------------------------------------------------
OUTPUT_BASE="${MODEL_ROOT}/output"
mkdir -p "${OUTPUT_BASE}"
RUN_ID=0
for existing_dir in "${OUTPUT_BASE}"/*; do
    [ -d "${existing_dir}" ] || continue
    dir_name="$(basename "${existing_dir}")"
    case "${dir_name}" in
        ''|*[!0-9]*) continue ;;
    esac
    dir_num=$((10#${dir_name}))
    [ "${dir_num}" -gt "${RUN_ID}" ] && RUN_ID="${dir_num}"
done
RUN_ID=$((RUN_ID + 1))
RUN_LABEL="$(printf '%03d' ${RUN_ID})"
RUN_DIR="${OUTPUT_BASE}/${RUN_LABEL}"
CKPT_DIR="${RUN_DIR}/checkpoints"
RESULTS_DIR="${RUN_DIR}/results"
LOGS_DIR="${RUN_DIR}/logs"
mkdir -p "${CKPT_DIR}" "${RESULTS_DIR}" "${LOGS_DIR}"

echo '============================================'
echo 'Model 06: Retrieval-Augmented Kinship'
echo 'Platform: AMD ROCm (local)'
echo '============================================'
echo "Run ID:            ${RUN_LABEL}"
echo "Output:            ${RUN_DIR}"
echo "Epochs:            ${EPOCHS}"
echo "Batch size:        ${BATCH_SIZE}"
echo "Grad accum:        ${GRAD_ACCUM}  (effective batch = $((BATCH_SIZE * GRAD_ACCUM)))"
echo "Learning rate:     ${LEARNING_RATE}"
echo "Backbone:          ${BACKBONE}  (frozen=${FREEZE_BACKBONE})"
echo "Retrieval K:       ${RETRIEVAL_K}"
echo "Attn layers/heads: ${RETRIEVAL_ATTN_LAYERS}/${RETRIEVAL_ATTN_HEADS}"
echo "Embedding dim:     ${EMBEDDING_DIM}"
echo "Loss:              ${LOSS}"
echo "Relation set:      ${RELATION_SET:-auto (=${TRAIN_DATASET})}"
echo "Relation weight:   ${RELATION_LOSS_WEIGHT}"
echo "Gallery cap:       ${MAX_GALLERY}"
echo "Gallery on CPU:    ${STORE_GALLERY_ON_CPU}"
echo "Refresh every:     ${GALLERY_REFRESH_EVERY}"
echo "AMP:               $([ "${DISABLE_AMP}" = "1" ] && echo off || echo on)"
echo "Dataset:           ${TRAIN_DATASET}"
echo "Seed:              ${SEED}"
echo "ROCm device:       ${GPU_ID}"
echo "Python:            ${PYTHON}"
echo '============================================'

export PYTHONPATH="${PROJECT_ROOT}/models:${PROJECT_ROOT}/models/shared:${PYTHONPATH}"

TRAIN_ARGS=(
    --train_dataset             "${TRAIN_DATASET}"
    --epochs                    "${EPOCHS}"
    --batch_size                "${BATCH_SIZE}"
    --gradient_accumulation     "${GRAD_ACCUM}"
    --lr                        "${LEARNING_RATE}"
    --weight_decay              "${WEIGHT_DECAY}"
    --warmup_epochs             "${WARMUP_EPOCHS}"
    --min_lr                    "${MIN_LR}"
    --scheduler                 "${SCHEDULER}"
    --backbone_name             "${BACKBONE}"
    --img_size                  "${IMG_SIZE}"
    --retrieval_k               "${RETRIEVAL_K}"
    --retrieval_attn_layers     "${RETRIEVAL_ATTN_LAYERS}"
    --retrieval_attn_heads      "${RETRIEVAL_ATTN_HEADS}"
    --embedding_dim             "${EMBEDDING_DIM}"
    --dropout                   "${DROPOUT}"
    --loss                      "${LOSS}"
    --contrastive_weight        "${CONTRASTIVE_WEIGHT}"
    --temperature               "${TEMPERATURE}"
    --relation_loss_weight      "${RELATION_LOSS_WEIGHT}"
    --max_gallery               "${MAX_GALLERY}"
    --gallery_refresh_every     "${GALLERY_REFRESH_EVERY}"
    --num_workers               "${NUM_WORKERS}"
    --patience                  "${PATIENCE}"
    --max_grad_norm             "${MAX_GRAD_NORM}"
    --seed                      "${SEED}"
    --rocm_device               "${GPU_ID}"
    --checkpoint_dir            "${CKPT_DIR}"
)
[ -n "${RELATION_SET}" ]             && TRAIN_ARGS+=(--relation_set "${RELATION_SET}")
[ "${STORE_GALLERY_ON_CPU}" = "1" ]  && TRAIN_ARGS+=(--store_gallery_on_cpu)
[ "${DISABLE_AMP}" = "1" ]           && TRAIN_ARGS+=(--disable_amp)
[ "${NO_GRAD_CKPT}" = "1" ]          && TRAIN_ARGS+=(--no_grad_ckpt)
[ "${FREEZE_BACKBONE}" = "0" ]       && TRAIN_ARGS+=(--no_freeze_backbone)

TEST_ARGS=(
    --checkpoint  "${CKPT_DIR}/best.pt"
    --dataset     "${TRAIN_DATASET}"
    --batch_size  "${BATCH_SIZE}"
    --output_dir  "${RESULTS_DIR}"
    --num_workers "${NUM_WORKERS}"
    --rocm_device "${GPU_ID}"
)

EVAL_ARGS=(
    --checkpoint   "${CKPT_DIR}/best.pt"
    --dataset      "${TRAIN_DATASET}"
    --batch_size   "${BATCH_SIZE}"
    --output_dir   "${RESULTS_DIR}"
    --num_workers  "${NUM_WORKERS}"
    --rocm_device  "${GPU_ID}"
    --full_analysis
)

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
echo "Model 06 — Run ${RUN_LABEL} completed"
echo "Results:     ${RESULTS_DIR}"
echo "Checkpoints: ${CKPT_DIR}"
echo "Logs:        ${LOGS_DIR}"
echo '============================================'
