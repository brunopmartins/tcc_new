#!/usr/bin/env bash
# =====================================================================
# Model 13: LGKT-Net — AMD ROCm pipeline runner (R001 recipe)
#
# Usage:
#   bash models/13_landmark_graph_kinship_transformer/AMD/run_pipeline.sh
#
# Common env overrides:
#   ALIGNED_ROOT             Path to FIW_aligned 224×224 (required)
#   BATCH_SIZE               Batch size (default 16)
#   GRAD_ACCUM               Gradient accumulation (default 2)
#   FREEZE_BACKBONE          1 = freeze AdaFace (default)
#   UNFREEZE_LAST_STAGE      1 = Phase 2 — unfreeze body[46:49] + output_layer
#   LEARNING_RATE            default 1e-5
#   NUM_GRAPH_LAYERS         default 2
#   NUM_HEADS                default 4
#   DROPOUT                  default 0.2
#   ROI_OUTPUT_SIZE          default 3
#   RELATION_AUX_WEIGHT      default 0.0 (off in R001)
#   TRAIN_NEGATIVE_STRATEGY  random | relation_matched (default random)
#   HARD_NEGATIVE_RATIO      default 0.0
#   FOLD, NUM_FOLDS          K-fold CV (default off)
#   RUN_OVERRIDE             reuse an existing run id (used by cv_runner)
#   SKIP_INSTALL=1           Skip pip install
# =====================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${MODEL_ROOT}/../.." && pwd)"

PYTHON="${MODEL_ROOT}/.venv/bin/python"

TRAIN_DATASET="${TRAIN_DATASET:-fiw}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/datasets/FIW}"
ALIGNED_ROOT="${ALIGNED_ROOT:-${PROJECT_ROOT}/datasets/FIW_aligned}"
ADAFACE_WEIGHTS="${ADAFACE_WEIGHTS:-${PROJECT_ROOT}/models/12_rgck_net/weights/adaface_ir101_webface4m.pth}"

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
SCHEDULER="${SCHEDULER:-cosine}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
MIN_LR="${MIN_LR:-1e-6}"

IMG_SIZE="${IMG_SIZE:-224}"
EMBEDDING_DIM="${EMBEDDING_DIM:-512}"
NUM_HEADS="${NUM_HEADS:-4}"
NUM_GRAPH_LAYERS="${NUM_GRAPH_LAYERS:-2}"
GATE_HIDDEN="${GATE_HIDDEN:-128}"
CLASSIFIER_HIDDEN="${CLASSIFIER_HIDDEN:-512}"
DROPOUT="${DROPOUT:-0.2}"
ROI_OUTPUT_SIZE="${ROI_OUTPUT_SIZE:-3}"

FREEZE_BACKBONE="${FREEZE_BACKBONE:-1}"
UNFREEZE_LAST_STAGE="${UNFREEZE_LAST_STAGE:-0}"

RELATION_AUX_WEIGHT="${RELATION_AUX_WEIGHT:-0.0}"
RELATION_AUX_BALANCED="${RELATION_AUX_BALANCED:-1}"

DIFFERENTIAL_LR="${DIFFERENTIAL_LR:-0}"
LR_STAGE4="${LR_STAGE4:-5e-6}"
LR_OUTPUT_LAYER="${LR_OUTPUT_LAYER:-5e-6}"
LR_HEAD="${LR_HEAD:-2e-5}"

NEGATIVE_RATIO="${NEGATIVE_RATIO:-1.0}"
EVAL_NEGATIVE_RATIO="${EVAL_NEGATIVE_RATIO:-1.0}"
TRAIN_NEGATIVE_STRATEGY="${TRAIN_NEGATIVE_STRATEGY:-random}"
EVAL_NEGATIVE_STRATEGY="${EVAL_NEGATIVE_STRATEGY:-random}"
HARD_NEGATIVE_RATIO="${HARD_NEGATIVE_RATIO:-0.0}"

NUM_WORKERS="${NUM_WORKERS:-4}"
PATIENCE="${PATIENCE:-50}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

FOLD="${FOLD:-}"
NUM_FOLDS="${NUM_FOLDS:-5}"
RUN_OVERRIDE="${RUN_OVERRIDE:-}"

OUTPUT_BASE="${MODEL_ROOT}/output"
mkdir -p "${OUTPUT_BASE}"
if [ -n "${RUN_OVERRIDE}" ]; then
    RUN_LABEL="${RUN_OVERRIDE}"
    RUN_DIR="${OUTPUT_BASE}/${RUN_LABEL}"
    mkdir -p "${RUN_DIR}"
else
    RUN_ID=1
    while [ -d "${OUTPUT_BASE}/$(printf '%03d' ${RUN_ID})" ]; do
        RUN_ID=$((RUN_ID + 1))
    done
    RUN_LABEL="$(printf '%03d' ${RUN_ID})"
    RUN_DIR="${OUTPUT_BASE}/${RUN_LABEL}"
fi

if [ -n "${FOLD}" ]; then
    SUBRUN_DIR="${RUN_DIR}/fold_${FOLD}"
else
    SUBRUN_DIR="${RUN_DIR}"
fi
CKPT_DIR="${SUBRUN_DIR}/checkpoints"
RESULTS_DIR="${SUBRUN_DIR}/results"
LOGS_DIR="${SUBRUN_DIR}/logs"
mkdir -p "${CKPT_DIR}" "${RESULTS_DIR}" "${LOGS_DIR}"

if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: virtualenv python not found at ${PYTHON}"
    echo "       Link from M12 venv: ln -s ../12_rgck_net/.venv ${MODEL_ROOT}/.venv"
    exit 1
fi
if [ ! -f "${ADAFACE_WEIGHTS}" ]; then
    echo "ERROR: AdaFace weights not found at ${ADAFACE_WEIGHTS}"
    exit 1
fi
if [ ! -d "${DATA_ROOT}" ]; then
    echo "ERROR: DATA_ROOT not found at ${DATA_ROOT}"; exit 1
fi
if [ ! -d "${ALIGNED_ROOT}" ]; then
    echo "ERROR: ALIGNED_ROOT not found at ${ALIGNED_ROOT}"; exit 1
fi

echo '============================================'
echo 'Model 13: LGKT-Net (Landmark Graph Kinship Transformer)'
echo '============================================'
echo "Run ID:              ${RUN_LABEL}"
echo "Run dir:             ${SUBRUN_DIR}"
echo "Aligned root:        ${ALIGNED_ROOT}"
echo "AdaFace weights:     ${ADAFACE_WEIGHTS}"
echo "Backbone frozen:     ${FREEZE_BACKBONE} (unfreeze_last_stage=${UNFREEZE_LAST_STAGE})"
echo "Graph layers/heads:  ${NUM_GRAPH_LAYERS} × ${NUM_HEADS}"
echo "ROI output size:     ${ROI_OUTPUT_SIZE}"
echo "Dropout:             ${DROPOUT}"
echo "Batch / grad accum:  ${BATCH_SIZE} × ${GRAD_ACCUM} (eff $((BATCH_SIZE * GRAD_ACCUM)))"
echo "Learning rate:       ${LEARNING_RATE}"
echo "Weight decay:        ${WEIGHT_DECAY}"
echo "Scheduler:           ${SCHEDULER} warmup=${WARMUP_EPOCHS} min_lr=${MIN_LR}"
echo "Relation aux λ:      ${RELATION_AUX_WEIGHT}"
echo "Differential LR:     ${DIFFERENTIAL_LR}"
echo "Negative ratio:      ${NEGATIVE_RATIO}"
echo "Train neg strategy:  ${TRAIN_NEGATIVE_STRATEGY} (hard ratio: ${HARD_NEGATIVE_RATIO})"
echo "Eval neg strategy:   ${EVAL_NEGATIVE_STRATEGY}"
echo "Patience:            ${PATIENCE}"
echo "Dataset:             ${TRAIN_DATASET}"
echo "Seed:                ${SEED}"
if [ -n "${FOLD}" ]; then
    echo "K-fold CV:           fold ${FOLD}/${NUM_FOLDS}"
fi
echo "ROCm device:         ${GPU_ID}"
echo "Python:              ${PYTHON}"
echo '============================================'

export PYTHONPATH="${PROJECT_ROOT}/models:${PROJECT_ROOT}/models/shared:${PYTHONPATH:-}"

TRAIN_ARGS=(
    --train_dataset           "${TRAIN_DATASET}"
    --test_dataset            "${TRAIN_DATASET}"
    --data_root               "${DATA_ROOT}"
    --aligned_root            "${ALIGNED_ROOT}"
    --epochs                  "${EPOCHS}"
    --batch_size              "${BATCH_SIZE}"
    --gradient_accumulation   "${GRAD_ACCUM}"
    --lr                      "${LEARNING_RATE}"
    --weight_decay            "${WEIGHT_DECAY}"
    --scheduler               "${SCHEDULER}"
    --warmup_epochs           "${WARMUP_EPOCHS}"
    --min_lr                  "${MIN_LR}"
    --adaface_weights         "${ADAFACE_WEIGHTS}"
    --img_size                "${IMG_SIZE}"
    --embedding_dim           "${EMBEDDING_DIM}"
    --num_heads               "${NUM_HEADS}"
    --num_graph_layers        "${NUM_GRAPH_LAYERS}"
    --gate_hidden             "${GATE_HIDDEN}"
    --classifier_hidden       "${CLASSIFIER_HIDDEN}"
    --dropout                 "${DROPOUT}"
    --roi_output_size         "${ROI_OUTPUT_SIZE}"
    --negative_ratio          "${NEGATIVE_RATIO}"
    --eval_negative_ratio     "${EVAL_NEGATIVE_RATIO}"
    --train_negative_strategy "${TRAIN_NEGATIVE_STRATEGY}"
    --eval_negative_strategy  "${EVAL_NEGATIVE_STRATEGY}"
    --hard_negative_ratio     "${HARD_NEGATIVE_RATIO}"
    --num_workers             "${NUM_WORKERS}"
    --patience                "${PATIENCE}"
    --max_grad_norm           "${MAX_GRAD_NORM}"
    --relation_aux_weight     "${RELATION_AUX_WEIGHT}"
    --seed                    "${SEED}"
    --rocm_device             "${GPU_ID}"
    --checkpoint_dir          "${CKPT_DIR}"
)
[ "${FREEZE_BACKBONE}" != "1" ] && TRAIN_ARGS+=(--unfreeze_backbone)
[ "${UNFREEZE_LAST_STAGE}" = "1" ] && TRAIN_ARGS+=(--unfreeze_last_stage)
[ "${RELATION_AUX_BALANCED}" != "1" ] && TRAIN_ARGS+=(--relation_aux_unbalanced)
if [ "${DIFFERENTIAL_LR}" = "1" ]; then
    TRAIN_ARGS+=(--differential_lr --lr_stage4 "${LR_STAGE4}" --lr_output_layer "${LR_OUTPUT_LAYER}" --lr_head "${LR_HEAD}")
fi
[ -n "${FOLD}" ] && TRAIN_ARGS+=(--fold "${FOLD}" --num_folds "${NUM_FOLDS}")

TEST_ARGS=(
    --checkpoint   "${CKPT_DIR}/best.pt"
    --dataset      "${TRAIN_DATASET}"
    --data_root    "${DATA_ROOT}"
    --aligned_root "${ALIGNED_ROOT}"
    --batch_size   "${BATCH_SIZE}"
    --num_workers  "${NUM_WORKERS}"
    --output_dir   "${RESULTS_DIR}"
    --rocm_device  "${GPU_ID}"
    --adaface_weights "${ADAFACE_WEIGHTS}"
)

cd "${SCRIPT_DIR}"

echo ''
echo '[1/2] Training...'
"${PYTHON}" train.py "${TRAIN_ARGS[@]}" 2>&1 | tee "${LOGS_DIR}/train.log"

echo ''
echo '[2/2] Testing...'
"${PYTHON}" test.py "${TEST_ARGS[@]}" 2>&1 | tee "${LOGS_DIR}/test.log"

echo ''
echo '============================================'
echo "Model 13 — Run ${RUN_LABEL} completed!"
echo '============================================'
