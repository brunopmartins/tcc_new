#!/bin/bash
# =============================================================================
# Model 02: ViT + FaCoR Cross-Attention — AMD ROCm pipeline runner
# Local host execution (no Docker). Sets up the virtualenv on first run.
#
# Run from anywhere in the project:
#   bash models/02_vit_facor_crossattn/AMD/run_pipeline.sh
#
# Or from within the model root or AMD/ directory:
#   bash AMD/run_pipeline.sh
#   ./AMD/run_pipeline.sh
#
# Override any setting via environment variable:
#   EPOCHS=50 bash AMD/run_pipeline.sh
#   LOSS=bce BATCH_SIZE=16 bash AMD/run_pipeline.sh
#
# Skip pip install after first run:
#   SKIP_INSTALL=1 bash AMD/run_pipeline.sh
#
# Environment variables:
#   EPOCHS            Training epochs              (default: 100)
#   BATCH_SIZE        Batch size                   (default: 32)
#   LEARNING_RATE     Learning rate                (default: 1e-4)
#   WEIGHT_DECAY      L2 regularisation            (default: 1e-5)
#   SCHEDULER         cosine | plateau | none      (default: cosine)
#   WARMUP_EPOCHS     Warmup epochs                (default: 5)
#   MIN_LR            Minimum LR                   (default: 1e-7)
#   TRAIN_DATASET     kinface | fiw                (default: kinface)
#   VIT_MODEL         ViT backbone variant         (default: vit_base_patch16_224)
#   CROSS_ATTN_LAYERS Number of cross-attn layers  (default: 2)
#   CROSS_ATTN_HEADS  Number of attention heads    (default: 8)
#   LOSS              bce | contrastive | cosine_contrastive | relation_guided  (default: cosine_contrastive)
#   TEMPERATURE       Contrastive temperature      (default: 0.07)
#   MARGIN            Contrastive margin           (default: 0.5)
#   NEGATIVE_RATIO    Train negatives / positive   (default: 1.0)
#   EVAL_NEGATIVE_RATIO Eval negatives / positive  (default: 1.0)
#   TRAIN_NEGATIVE_STRATEGY random | relation_matched (default: random)
#   EVAL_NEGATIVE_STRATEGY random | relation_matched  (default: random)
#   NUM_WORKERS       Dataloader workers           (default: 4)
#   DROPOUT           Model dropout                (default: 0.1)
#   FREEZE_VIT        1 = freeze ViT backbone      (default: 0)
#   UNFREEZE_AFTER_EPOCH Epoch to start opening ViT blocks (default: 0 = disabled)
#   UNFREEZE_LAST_VIT_BLOCKS Number of final ViT blocks to unfreeze (default: 0)
#   USE_CLASSIFIER_HEAD 1 = add BCE classifier head (default: 0)
#   GPU_ID            GPU device index             (default: 0)
#   SEED              Random seed                  (default: 42)
#   SKIP_INSTALL      1 = skip venv/pip setup      (default: 0)
# =============================================================================
set -eo pipefail

# ---------- resolve paths from script location --------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"    # .../AMD/
MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"                   # .../02_vit_facor_crossattn/
PROJECT_ROOT="$(cd "${MODEL_ROOT}/../.." && pwd)"              # .../tcc_new/

PYTHON="${MODEL_ROOT}/.venv/bin/python"
PIP="${MODEL_ROOT}/.venv/bin/pip"

# ---------- configurable parameters -------------------------------------------
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
SCHEDULER="${SCHEDULER:-cosine}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
MIN_LR="${MIN_LR:-1e-7}"
TRAIN_DATASET="${TRAIN_DATASET:-kinface}"
VIT_MODEL="${VIT_MODEL:-vit_base_patch16_224}"
CROSS_ATTN_LAYERS="${CROSS_ATTN_LAYERS:-2}"
CROSS_ATTN_HEADS="${CROSS_ATTN_HEADS:-8}"
LOSS="${LOSS:-cosine_contrastive}"
TEMPERATURE="${TEMPERATURE:-0.3}"
MARGIN="${MARGIN:-0.5}"
NEGATIVE_RATIO="${NEGATIVE_RATIO:-1.0}"
EVAL_NEGATIVE_RATIO="${EVAL_NEGATIVE_RATIO:-1.0}"
TRAIN_NEGATIVE_STRATEGY="${TRAIN_NEGATIVE_STRATEGY:-random}"
EVAL_NEGATIVE_STRATEGY="${EVAL_NEGATIVE_STRATEGY:-random}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DROPOUT="${DROPOUT:-0.1}"
FREEZE_VIT="${FREEZE_VIT:-0}"
UNFREEZE_AFTER_EPOCH="${UNFREEZE_AFTER_EPOCH:-0}"
UNFREEZE_LAST_VIT_BLOCKS="${UNFREEZE_LAST_VIT_BLOCKS:-0}"
USE_CLASSIFIER_HEAD="${USE_CLASSIFIER_HEAD:-0}"
PATIENCE="${PATIENCE:-50}"
CV_FOLDS="${CV_FOLDS:-0}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

# ---------- ROCm: ensure render group access to /dev/kfd ----------------------
# Re-exec via 'sg render' so the group is active without requiring logout.
if [ -e /dev/kfd ] && [ -z "${_ROCM_SETUP_COMPLETE}" ]; then
    if ! id -Gn 2>/dev/null | tr ' ' '\n' | grep -qx render; then
        echo "  [ROCm] Adding $(whoami) to render+video groups for /dev/kfd access..."
        sudo usermod -aG render,video "$(whoami)"
    fi
    export EPOCHS BATCH_SIZE LEARNING_RATE WEIGHT_DECAY SCHEDULER WARMUP_EPOCHS MIN_LR
    export TRAIN_DATASET VIT_MODEL CROSS_ATTN_LAYERS CROSS_ATTN_HEADS LOSS TEMPERATURE MARGIN
    export NEGATIVE_RATIO EVAL_NEGATIVE_RATIO TRAIN_NEGATIVE_STRATEGY EVAL_NEGATIVE_STRATEGY NUM_WORKERS DROPOUT
    export FREEZE_VIT UNFREEZE_AFTER_EPOCH UNFREEZE_LAST_VIT_BLOCKS USE_CLASSIFIER_HEAD PATIENCE CV_FOLDS GPU_ID SEED SKIP_INSTALL
    export _ROCM_SETUP_COMPLETE=1
    SELF="$(readlink -f "${BASH_SOURCE[0]}")"
    echo "  [ROCm] Restarting script with render group active..."
    exec sg render -c "bash '${SELF}'"
fi

# ---------- ROCm environment (set before Python/torch init) -------------------
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-FAST}"
export HSA_FORCE_FINE_GRAIN_PCIE="${HSA_FORCE_FINE_GRAIN_PCIE:-1}"
export HIP_VISIBLE_DEVICES="${GPU_ID}"
export ROCR_VISIBLE_DEVICES="${GPU_ID}"

# ---------- render group warning (soft, non-fatal) ----------------------------
if ! id -Gn 2>/dev/null | tr ' ' '\n' | grep -qx render; then
    echo "WARNING: Not in the 'render' group — GPU may not be detected."
    echo "  Fix: sudo usermod -aG render,video \$(whoami) && log out/in"
fi

# ---------- virtualenv setup --------------------------------------------------
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
        'timm>=0.9.0' \
        'numpy>=1.24.0' \
        'pandas>=2.0.0' \
        'scikit-learn>=1.3.0' \
        'Pillow>=10.0.0' \
        'tqdm>=4.65.0' \
        'matplotlib>=3.7.0' \
        'seaborn>=0.12.0'

    echo "  Done. Run with SKIP_INSTALL=1 to skip this step next time."
    echo ''
fi

if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: virtualenv not found at ${MODEL_ROOT}/.venv"
    echo "  Run without SKIP_INSTALL=1 to create it."
    exit 1
fi

# ---------- numbered output folder --------------------------------------------
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
    if [ "${dir_num}" -gt "${RUN_ID}" ]; then
        RUN_ID="${dir_num}"
    fi
done
RUN_ID=$((RUN_ID + 1))
RUN_LABEL="$(printf '%03d' ${RUN_ID})"
RUN_DIR="${OUTPUT_BASE}/${RUN_LABEL}"
CKPT_DIR="${RUN_DIR}/checkpoints"
RESULTS_DIR="${RUN_DIR}/results"
LOGS_DIR="${RUN_DIR}/logs"
mkdir -p "${CKPT_DIR}" "${RESULTS_DIR}" "${LOGS_DIR}"

echo '============================================'
echo 'Model 02: ViT + FaCoR Cross-Attention'
echo 'Platform: AMD ROCm (local)'
echo '============================================'
echo "Run ID:            ${RUN_LABEL}"
echo "Output:            ${RUN_DIR}"
echo "Epochs:            ${EPOCHS}"
echo "Batch size:        ${BATCH_SIZE}"
echo "Learning rate:     ${LEARNING_RATE}"
echo "Weight decay:      ${WEIGHT_DECAY}"
echo "Scheduler:         ${SCHEDULER}"
echo "Warmup epochs:     ${WARMUP_EPOCHS}"
echo "Min LR:            ${MIN_LR}"
echo "ViT model:         ${VIT_MODEL}"
echo "Cross-attn layers: ${CROSS_ATTN_LAYERS}"
echo "Cross-attn heads:  ${CROSS_ATTN_HEADS}"
echo "Loss:              ${LOSS}"
echo "Temperature:       ${TEMPERATURE}"
echo "Margin:            ${MARGIN}"
echo "Negative ratio:    ${NEGATIVE_RATIO}"
echo "Eval neg ratio:    ${EVAL_NEGATIVE_RATIO}"
echo "Train neg strat:   ${TRAIN_NEGATIVE_STRATEGY}"
echo "Eval neg strat:    ${EVAL_NEGATIVE_STRATEGY}"
echo "Workers:           ${NUM_WORKERS}"
echo "Dropout:           ${DROPOUT}"
echo "Freeze ViT:        $([ "${FREEZE_VIT}" = "1" ] && echo yes || echo no)"
echo "Unfreeze epoch:    ${UNFREEZE_AFTER_EPOCH}"
echo "Unfreeze blocks:   ${UNFREEZE_LAST_VIT_BLOCKS}"
echo "Classifier head:   $([ "${USE_CLASSIFIER_HEAD}" = "1" ] && echo yes || echo no)"
echo "Patience:          ${PATIENCE}"
echo "CV folds:          $([ "${CV_FOLDS}" -gt 0 ] 2>/dev/null && echo "${CV_FOLDS}" || echo "disabled")"
echo "Dataset:           ${TRAIN_DATASET}"
echo "Seed:              ${SEED}"
echo "ROCm device:       ${GPU_ID}"
echo "Python:            ${PYTHON}"
echo '============================================'

# ---------- set PYTHONPATH ----------------------------------------------------
export PYTHONPATH="${PROJECT_ROOT}/models:${PROJECT_ROOT}/models/shared:${PYTHONPATH}"

# ---------- argument arrays ---------------------------------------------------
# Shared training args (used by both train.py and train_cv.py)
SHARED_TRAIN_ARGS=(
    --train_dataset     "${TRAIN_DATASET}"
    --epochs            "${EPOCHS}"
    --batch_size        "${BATCH_SIZE}"
    --lr                "${LEARNING_RATE}"
    --weight_decay      "${WEIGHT_DECAY}"
    --scheduler         "${SCHEDULER}"
    --warmup_epochs     "${WARMUP_EPOCHS}"
    --min_lr            "${MIN_LR}"
    --vit_model         "${VIT_MODEL}"
    --cross_attn_layers "${CROSS_ATTN_LAYERS}"
    --cross_attn_heads  "${CROSS_ATTN_HEADS}"
    --dropout           "${DROPOUT}"
    --unfreeze_after_epoch "${UNFREEZE_AFTER_EPOCH}"
    --unfreeze_last_vit_blocks "${UNFREEZE_LAST_VIT_BLOCKS}"
    --loss              "${LOSS}"
    --temperature       "${TEMPERATURE}"
    --margin            "${MARGIN}"
    --negative_ratio    "${NEGATIVE_RATIO}"
    --eval_negative_ratio "${EVAL_NEGATIVE_RATIO}"
    --train_negative_strategy "${TRAIN_NEGATIVE_STRATEGY}"
    --eval_negative_strategy "${EVAL_NEGATIVE_STRATEGY}"
    --num_workers       "${NUM_WORKERS}"
    --patience          "${PATIENCE}"
    --seed              "${SEED}"
    --rocm_device       "${GPU_ID}"
    --checkpoint_dir    "${CKPT_DIR}"
)
[ "${FREEZE_VIT}" = "1" ] && SHARED_TRAIN_ARGS+=(--freeze_vit)
[ "${USE_CLASSIFIER_HEAD}" = "1" ] && SHARED_TRAIN_ARGS+=(--use_classifier_head)

TEST_ARGS=(
    --checkpoint  "${CKPT_DIR}/best.pt"
    --dataset     "${TRAIN_DATASET}"
    --batch_size  "${BATCH_SIZE}"
    --output_dir  "${RESULTS_DIR}"
    --rocm_device "${GPU_ID}"
)

EVAL_ARGS=(
    --checkpoint  "${CKPT_DIR}/best.pt"
    --dataset     "${TRAIN_DATASET}"
    --batch_size  "${BATCH_SIZE}"
    --output_dir  "${RESULTS_DIR}"
    --rocm_device "${GPU_ID}"
    --full_analysis
    --visualize_attention
)

# ---------- cd into AMD/ so relative imports resolve --------------------------
cd "${SCRIPT_DIR}"

# ---------- pipeline ----------------------------------------------------------
if [ "${CV_FOLDS:-0}" -gt 0 ] 2>/dev/null; then
    # ── Cross-validation mode ──────────────────────────────────────────────
    echo ''
    echo "[1/1] ${CV_FOLDS}-fold cross-validation..."
    "${PYTHON}" train_cv.py \
        "${SHARED_TRAIN_ARGS[@]}" \
        --n_folds      "${CV_FOLDS}" \
        --results_dir  "${RESULTS_DIR}" \
        2>&1 | tee "${LOGS_DIR}/train.log"

    echo ''
    echo '============================================'
    echo "Model 02 — Run ${RUN_LABEL} completed (${CV_FOLDS}-fold CV)!"
    echo "CV summary:  ${RESULTS_DIR}/cv_summary.json"
    echo "Checkpoints: ${CKPT_DIR}/fold_*/"
    echo "Log:         ${LOGS_DIR}/train.log"
    echo '============================================'
else
    # ── Standard single-run mode ───────────────────────────────────────────
    echo ''
    echo '[1/3] Training...'
    "${PYTHON}" train.py \
        "${SHARED_TRAIN_ARGS[@]}" \
        --test_dataset "${TRAIN_DATASET}" \
        2>&1 | tee "${LOGS_DIR}/train.log"

    echo ''
    echo '[2/3] Testing...'
    "${PYTHON}" test.py "${TEST_ARGS[@]}" 2>&1 | tee "${LOGS_DIR}/test.log"

    echo ''
    echo '[3/3] Evaluating...'
    "${PYTHON}" evaluate.py "${EVAL_ARGS[@]}" 2>&1 | tee "${LOGS_DIR}/evaluate.log"

    echo ''
    echo '============================================'
    echo "Model 02 — Run ${RUN_LABEL} completed!"
    echo "Results:     ${RESULTS_DIR}"
    echo "Checkpoints: ${CKPT_DIR}"
    echo "Logs:        ${LOGS_DIR}"
    echo '============================================'
fi
