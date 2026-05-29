#!/usr/bin/env bash
# =====================================================================
# Model 12: RGCK-Net — AMD ROCm pipeline runner (Phase 1 recipe)
#
# Usage:
#   bash models/12_rgck_net/AMD/run_pipeline.sh
#
# Common env overrides:
#   ALIGNED_ROOT             Path to FIW_aligned 224×224 (required)
#   BATCH_SIZE               Batch size (default 8)
#   GRAD_ACCUM               Gradient accumulation (default 4)
#   FREEZE_BACKBONE          1 = freeze AdaFace (default), 0 = full FT (Phase 3)
#   UNFREEZE_LAST_STAGE      1 = Phase 2 — unfreeze body[46:49] + output_layer (effective when FREEZE_BACKBONE=1)
#   TRAIN_NEGATIVE_STRATEGY  random | relation_matched  (default random)
#   EVAL_NEGATIVE_STRATEGY   random | relation_matched  (default random)
#   NEGATIVE_RATIO           default 1.0
#   LEARNING_RATE            default 1e-4 (head-only LR per proposta)
#   WEIGHT_DECAY             default 1e-4
#   DROPOUT                  default 0.2
#   CROSS_ATTN_HEADS         default 4
#   CROSS_ATTN_LAYERS        default 1
#   PATIENCE                 default 50
#   NUM_WORKERS              default 4
#   SEED                     default 42
#   SKIP_INSTALL=1           Skip pip install (used when re-running)
# =====================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${MODEL_ROOT}/../.." && pwd)"

PYTHON="${MODEL_ROOT}/.venv/bin/python"

# Defaults
TRAIN_DATASET="${TRAIN_DATASET:-fiw}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/datasets/FIW}"
ALIGNED_ROOT="${ALIGNED_ROOT:-${PROJECT_ROOT}/datasets/FIW_aligned}"
ADAFACE_WEIGHTS="${ADAFACE_WEIGHTS:-${MODEL_ROOT}/weights/adaface_ir101_webface4m.pth}"

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
SCHEDULER="${SCHEDULER:-cosine}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
MIN_LR="${MIN_LR:-1e-6}"

IMG_SIZE="${IMG_SIZE:-224}"
EMBEDDING_DIM="${EMBEDDING_DIM:-512}"
CROSS_ATTN_HEADS="${CROSS_ATTN_HEADS:-4}"
CROSS_ATTN_LAYERS="${CROSS_ATTN_LAYERS:-1}"
GATE_HIDDEN="${GATE_HIDDEN:-128}"
CLASSIFIER_HIDDEN="${CLASSIFIER_HIDDEN:-512}"
DROPOUT="${DROPOUT:-0.2}"
FREEZE_BACKBONE="${FREEZE_BACKBONE:-1}"        # Phase 1 default: frozen
UNFREEZE_LAST_STAGE="${UNFREEZE_LAST_STAGE:-0}" # Phase 2: set to 1 to unfreeze body[46:49] + output_layer
UNFREEZE_EXTRA_STAGE3_TAIL="${UNFREEZE_EXTRA_STAGE3_TAIL:-0}"  # R012: 1 = additionally unfreeze body[43:46] on top of body[46:49] + output_layer
CONSISTENCY_WEIGHT="${CONSISTENCY_WEIGHT:-0.0}"                # R012: cosine consistency between fusion_AB and fusion_BA (requires SYMMETRIC_FORWARD=1)
SUPCON_WEIGHT="${SUPCON_WEIGHT:-0.0}"             # Phase 4: weight λ for supervised-contrastive aux (0.0 = disabled)
SUPCON_MARGIN="${SUPCON_MARGIN:-0.3}"             # margin for negative pairs in supcon term
RELATION_AUX_WEIGHT="${RELATION_AUX_WEIGHT:-0.0}" # Phase 5: weight λ for relation-type CE aux on positives (0.0 = disabled)
RELATION_AUX_BALANCED="${RELATION_AUX_BALANCED:-1}" # 1 = inverse-freq class weights (default), 0 = uniform CE
SYMMETRIC_FORWARD="${SYMMETRIC_FORWARD:-0}"        # R006: 1 = process each pair in both (A,B) and (B,A) orders (Option-B BCE)
DIFFERENTIAL_LR="${DIFFERENTIAL_LR:-0}"            # R007: 1 = per-group LRs (overrides LEARNING_RATE)
LR_STAGE4="${LR_STAGE4:-5e-6}"                     # R007: LR for backbone body[46:49] (only if DIFFERENTIAL_LR=1)
LR_OUTPUT_LAYER="${LR_OUTPUT_LAYER:-5e-6}"         # R007: LR for backbone output_layer
LR_HEAD="${LR_HEAD:-2e-5}"                         # R007: LR for cross_region + gate + classifier + relation_head
L2SP_WEIGHT="${L2SP_WEIGHT:-0.0}"                  # R008: λ for L2-SP penalty on unfrozen backbone (stage 4 + output_layer)
COMPARISON_ONLY_FUSION="${COMPARISON_ONLY_FUSION:-0}"  # R009: 1 = drop gA, gB from classifier input
HARD_NEGATIVE_RATIO="${HARD_NEGATIVE_RATIO:-0.0}"      # R011: fraction of role-matched (hard) train negatives (0 = pure random, 1 = pure hard)

NEGATIVE_RATIO="${NEGATIVE_RATIO:-1.0}"
EVAL_NEGATIVE_RATIO="${EVAL_NEGATIVE_RATIO:-1.0}"
TRAIN_NEGATIVE_STRATEGY="${TRAIN_NEGATIVE_STRATEGY:-random}"
EVAL_NEGATIVE_STRATEGY="${EVAL_NEGATIVE_STRATEGY:-random}"

NUM_WORKERS="${NUM_WORKERS:-4}"
PATIENCE="${PATIENCE:-50}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

# K-fold cross-validation. When FOLD is set, the run output is nested under
# fold_${FOLD}/ inside the existing run dir (set via RUN_OVERRIDE) or a fresh
# run id. NUM_FOLDS controls the partition cardinality.
FOLD="${FOLD:-}"
NUM_FOLDS="${NUM_FOLDS:-5}"
RUN_OVERRIDE="${RUN_OVERRIDE:-}"

# Determine next run id
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

# Sanity: venv + python
if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: virtualenv python not found at ${PYTHON}"
    echo "       Run: ln -s ../02_vit_facor_crossattn/.venv ${MODEL_ROOT}/.venv"
    exit 1
fi

# Sanity: AdaFace weights
if [ ! -f "${ADAFACE_WEIGHTS}" ]; then
    echo "ERROR: AdaFace weights not found at ${ADAFACE_WEIGHTS}"
    exit 1
fi

if [ ! -d "${DATA_ROOT}" ]; then
    echo "ERROR: DATA_ROOT not found at ${DATA_ROOT}"
    exit 1
fi
if [ ! -d "${ALIGNED_ROOT}" ]; then
    echo "ERROR: ALIGNED_ROOT not found at ${ALIGNED_ROOT}"
    exit 1
fi

echo '============================================'
echo 'Model 12: RGCK-Net (Region-Guided Cross Kinship)'
echo '============================================'
echo "Run ID:              ${RUN_LABEL}"
echo "Run dir:             ${RUN_DIR}"
echo "Aligned root:        ${ALIGNED_ROOT}"
echo "AdaFace weights:     ${ADAFACE_WEIGHTS}"
echo "Backbone frozen:     ${FREEZE_BACKBONE} (unfreeze_last_stage=${UNFREEZE_LAST_STAGE}, extra_stage3_tail=${UNFREEZE_EXTRA_STAGE3_TAIL})"
echo "Consistency weight:  ${CONSISTENCY_WEIGHT}"
echo "Cross-attn:          ${CROSS_ATTN_LAYERS} layer × ${CROSS_ATTN_HEADS} heads"
echo "Dropout:             ${DROPOUT}"
echo "Batch / grad accum:  ${BATCH_SIZE} × ${GRAD_ACCUM} (eff $((BATCH_SIZE * GRAD_ACCUM)))"
echo "Learning rate:       ${LEARNING_RATE}"
echo "Weight decay:        ${WEIGHT_DECAY}"
echo "Scheduler:           ${SCHEDULER} warmup=${WARMUP_EPOCHS} min_lr=${MIN_LR}"
echo "Negative ratio:      ${NEGATIVE_RATIO}"
echo "Train neg strategy:  ${TRAIN_NEGATIVE_STRATEGY}"
echo "Eval neg strategy:   ${EVAL_NEGATIVE_STRATEGY}"
echo "Patience:            ${PATIENCE}"
echo "Dataset:             ${TRAIN_DATASET}"
echo "Data root:           ${DATA_ROOT}"
echo "Seed:                ${SEED}"
if [ -n "${FOLD}" ]; then
    echo "K-fold CV:           fold ${FOLD}/${NUM_FOLDS}  (output -> ${SUBRUN_DIR})"
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
    --cross_attn_heads        "${CROSS_ATTN_HEADS}"
    --cross_attn_layers       "${CROSS_ATTN_LAYERS}"
    --gate_hidden             "${GATE_HIDDEN}"
    --classifier_hidden       "${CLASSIFIER_HIDDEN}"
    --dropout                 "${DROPOUT}"
    --negative_ratio          "${NEGATIVE_RATIO}"
    --eval_negative_ratio     "${EVAL_NEGATIVE_RATIO}"
    --train_negative_strategy "${TRAIN_NEGATIVE_STRATEGY}"
    --eval_negative_strategy  "${EVAL_NEGATIVE_STRATEGY}"
    --num_workers             "${NUM_WORKERS}"
    --patience                "${PATIENCE}"
    --max_grad_norm           "${MAX_GRAD_NORM}"
    --supcon_weight           "${SUPCON_WEIGHT}"
    --supcon_margin           "${SUPCON_MARGIN}"
    --relation_aux_weight     "${RELATION_AUX_WEIGHT}"
    --seed                    "${SEED}"
    --rocm_device             "${GPU_ID}"
    --checkpoint_dir          "${CKPT_DIR}"
)
if [ "${FREEZE_BACKBONE}" != "1" ]; then
    TRAIN_ARGS+=(--unfreeze_backbone)
fi
if [ "${UNFREEZE_LAST_STAGE}" = "1" ]; then
    TRAIN_ARGS+=(--unfreeze_last_stage)
fi
if [ "${UNFREEZE_EXTRA_STAGE3_TAIL}" = "1" ]; then
    TRAIN_ARGS+=(--unfreeze_extra_stage3_tail)
fi
TRAIN_ARGS+=(--consistency_weight "${CONSISTENCY_WEIGHT}")
if [ "${RELATION_AUX_BALANCED}" != "1" ]; then
    TRAIN_ARGS+=(--relation_aux_unbalanced)
fi
if [ "${SYMMETRIC_FORWARD}" = "1" ]; then
    TRAIN_ARGS+=(--symmetric_forward)
fi
if [ "${DIFFERENTIAL_LR}" = "1" ]; then
    TRAIN_ARGS+=(--differential_lr --lr_stage4 "${LR_STAGE4}" --lr_output_layer "${LR_OUTPUT_LAYER}" --lr_head "${LR_HEAD}")
fi
TRAIN_ARGS+=(--l2sp_weight "${L2SP_WEIGHT}")
if [ "${COMPARISON_ONLY_FUSION}" = "1" ]; then
    TRAIN_ARGS+=(--comparison_only_fusion)
fi
TRAIN_ARGS+=(--hard_negative_ratio "${HARD_NEGATIVE_RATIO}")
if [ -n "${FOLD}" ]; then
    TRAIN_ARGS+=(--fold "${FOLD}" --num_folds "${NUM_FOLDS}")
fi

TEST_ARGS=(
    --checkpoint   "${CKPT_DIR}/best.pt"
    --dataset      "${TRAIN_DATASET}"
    --data_root    "${DATA_ROOT}"
    --aligned_root "${ALIGNED_ROOT}"
    --batch_size   "${BATCH_SIZE}"
    --num_workers  "${NUM_WORKERS}"
    --output_dir   "${RESULTS_DIR}"
    --rocm_device  "${GPU_ID}"
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
echo "Model 12 — Run ${RUN_LABEL} completed!"
echo '============================================'
