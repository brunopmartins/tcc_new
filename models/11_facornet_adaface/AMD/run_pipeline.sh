#!/bin/bash
# =============================================================================
# Model 11: AdaFace + FaCoR Cross-Attention — AMD ROCm pipeline runner
#
# Mirrors M02's M02 recipe (best so far: Test ROC AUC 0.850) but swaps the
# backbone for AdaFace IR-101 pretrained on WebFace4M (cvlface release).
#
# Usage:
#   bash models/11_facornet_adaface/AMD/run_pipeline.sh
#
# Environment variables:
#   EPOCHS                   Training epochs              (default: 100)
#   BATCH_SIZE               Batch size                   (default: 8)
#   GRAD_ACCUM               Grad accumulation steps      (default: 4 → eff 32)
#   LEARNING_RATE            Peak LR                      (default: 5e-6, M02 tuned)
#   WEIGHT_DECAY             L2                           (default: 1e-5)
#   SCHEDULER                cosine | plateau | none      (default: cosine)
#   WARMUP_EPOCHS                                         (default: 5)
#   MIN_LR                                                (default: 1e-7)
#   TRAIN_DATASET            kinface | fiw                (default: fiw)
#   ADAFACE_WEIGHTS          Path to AdaFace .pth         (default: weights/adaface_ir101_webface4m.pth)
#   IMG_SIZE                                              (default: 112)
#   CROSS_ATTN_LAYERS                                     (default: 2)
#   CROSS_ATTN_HEADS                                      (default: 8)
#   DROPOUT                                               (default: 0.2, M02 tuned)
#   FREEZE_BACKBONE          1 = freeze backbone          (default: 0 — full FT)
#   NO_POSITIONAL_EMBEDDING  1 = drop 49-token PE         (default: 0)
#   NO_GLOBAL_EMBEDDING      1 = drop AdaFace pool token  (default: 0)
#   USE_CLASSIFIER_HEAD      1 = BCE classifier head      (default: 0)
#   USE_MULTISTAGE           1 = multi-stage cross-attn (M09 arch)  (default: 0 — top-only M10 arch)
#   CROSS_ATTN_STAGES        Stages for multistage         (default: 3,4)
#   CROSS_ATTN_LAYERS_PER_STAGE  Layers per stage          (default: 1)
#   LOSS                     bce | contrastive | cosine_contrastive | relation_guided  (default: cosine_contrastive)
#   TEMPERATURE                                           (default: 0.3)
#   MARGIN                                                (default: 0.3, M02 tuned)
#   NEGATIVE_RATIO                                        (default: 1.0)
#   EVAL_NEGATIVE_RATIO                                   (default: 1.0)
#   TRAIN_NEGATIVE_STRATEGY  random | relation_matched    (default: random)
#   EVAL_NEGATIVE_STRATEGY   random | relation_matched    (default: random)
#   NUM_WORKERS                                           (default: 4)
#   PATIENCE                                              (default: 50)
#   MAX_GRAD_NORM                                         (default: 1.0)
#   GPU_ID                                                (default: 0)
#   SEED                                                  (default: 42)
#   SKIP_INSTALL             1 = skip venv/pip setup      (default: 0)
#   ALIGNED_ROOT             Path to MTCNN-aligned FIW crops  (default: empty)
#   DATA_ROOT                Dataset root w/ track-I CSVs (default: <project>/datasets/FIW)
#   AGE_AUGMENT_ROOT         Path to SAM-aged variants    (default: empty — off)
#   AGE_TARGET_AGES          Comma-separated ages         (default: 8,25,70)
#   AGE_ORIGINAL_WEIGHT      Weight for original face     (default: 0.5)
# =============================================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${MODEL_ROOT}/../.." && pwd)"

PYTHON="${MODEL_ROOT}/.venv/bin/python"
PIP="${MODEL_ROOT}/.venv/bin/pip"

# ---------- configurable parameters -------------------------------------------
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
SCHEDULER="${SCHEDULER:-cosine}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
MIN_LR="${MIN_LR:-1e-7}"
TRAIN_DATASET="${TRAIN_DATASET:-fiw}"
ADAFACE_WEIGHTS="${ADAFACE_WEIGHTS:-${MODEL_ROOT}/weights/adaface_ir101_webface4m.pth}"
IMG_SIZE="${IMG_SIZE:-112}"
CROSS_ATTN_LAYERS="${CROSS_ATTN_LAYERS:-2}"
CROSS_ATTN_HEADS="${CROSS_ATTN_HEADS:-8}"
DROPOUT="${DROPOUT:-0.2}"
FREEZE_BACKBONE="${FREEZE_BACKBONE:-0}"
NO_POSITIONAL_EMBEDDING="${NO_POSITIONAL_EMBEDDING:-0}"
NO_GLOBAL_EMBEDDING="${NO_GLOBAL_EMBEDDING:-0}"
USE_CLASSIFIER_HEAD="${USE_CLASSIFIER_HEAD:-0}"
# M11 architecture variant
USE_MULTISTAGE="${USE_MULTISTAGE:-0}"
CROSS_ATTN_STAGES="${CROSS_ATTN_STAGES:-3,4}"
CROSS_ATTN_LAYERS_PER_STAGE="${CROSS_ATTN_LAYERS_PER_STAGE:-1}"
# FaCoRNet recipe defaults differ from M10:
#   * Loss: relation_guided (FaCoR-inspired attention-driven dynamic temperature)
#   * Temperature: 0.07 (FaCoRNet base; vs M10 cosine_contrastive's 0.3)
#   * Negative sampling: relation_matched on both train and eval (hard negatives)
LOSS="${LOSS:-relation_guided}"
TEMPERATURE="${TEMPERATURE:-0.07}"
MARGIN="${MARGIN:-0.3}"
NEGATIVE_RATIO="${NEGATIVE_RATIO:-1.0}"
EVAL_NEGATIVE_RATIO="${EVAL_NEGATIVE_RATIO:-1.0}"
TRAIN_NEGATIVE_STRATEGY="${TRAIN_NEGATIVE_STRATEGY:-relation_matched}"
EVAL_NEGATIVE_STRATEGY="${EVAL_NEGATIVE_STRATEGY:-relation_matched}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PATIENCE="${PATIENCE:-50}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
ALIGNED_ROOT="${ALIGNED_ROOT:-}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/datasets/FIW}"
AGE_AUGMENT_ROOT="${AGE_AUGMENT_ROOT:-}"
AGE_TARGET_AGES="${AGE_TARGET_AGES:-8,25,70}"
AGE_ORIGINAL_WEIGHT="${AGE_ORIGINAL_WEIGHT:-0.5}"

# ---------- ROCm: render group fix --------------------------------------------
if [ -e /dev/kfd ] && [ -z "${_ROCM_SETUP_COMPLETE}" ]; then
    if ! id -Gn 2>/dev/null | tr ' ' '\n' | grep -qx render; then
        echo "  [ROCm] Adding $(whoami) to render+video groups for /dev/kfd access..."
        sudo usermod -aG render,video "$(whoami)"
    fi
    export EPOCHS BATCH_SIZE GRAD_ACCUM LEARNING_RATE WEIGHT_DECAY SCHEDULER WARMUP_EPOCHS MIN_LR
    export TRAIN_DATASET ADAFACE_WEIGHTS IMG_SIZE CROSS_ATTN_LAYERS CROSS_ATTN_HEADS DROPOUT
    export FREEZE_BACKBONE NO_POSITIONAL_EMBEDDING NO_GLOBAL_EMBEDDING USE_CLASSIFIER_HEAD
    export LOSS TEMPERATURE MARGIN NEGATIVE_RATIO EVAL_NEGATIVE_RATIO
    export TRAIN_NEGATIVE_STRATEGY EVAL_NEGATIVE_STRATEGY NUM_WORKERS PATIENCE MAX_GRAD_NORM
    export GPU_ID SEED SKIP_INSTALL ALIGNED_ROOT DATA_ROOT
    export AGE_AUGMENT_ROOT AGE_TARGET_AGES AGE_ORIGINAL_WEIGHT
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
    echo "  Fix: sudo usermod -aG render,video \$(whoami) && log out/in"
fi

# ---------- venv --------------------------------------------------------------
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
        'Pillow>=10.0.0' 'tqdm>=4.65.0' 'matplotlib>=3.7.0' \
        'huggingface_hub>=0.20.0'

    echo "  Done. Run with SKIP_INSTALL=1 to skip this step next time."
fi

if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: virtualenv not found at ${MODEL_ROOT}/.venv"
    echo "  Run without SKIP_INSTALL=1 to create it."
    exit 1
fi

# ---------- preconditions -----------------------------------------------------
if [ ! -f "${ADAFACE_WEIGHTS}" ]; then
    echo "ERROR: AdaFace weights not found at ${ADAFACE_WEIGHTS}"
    echo ""
    echo "Download from:"
    echo "  huggingface-cli download minchul/cvlface_adaface_ir101_webface4m \\"
    echo "      pretrained_model/model.pt --local-dir ${MODEL_ROOT}/weights/_hf_cache"
    echo "  mv ${MODEL_ROOT}/weights/_hf_cache/pretrained_model/model.pt \\"
    echo "     ${ADAFACE_WEIGHTS}"
    echo ""
    echo "See ${MODEL_ROOT}/weights/README.md for details and fallback options."
    exit 1
fi

if [ ! -d "${DATA_ROOT}" ]; then
    echo "ERROR: dataset root not found at ${DATA_ROOT}"
    exit 1
fi

if [ -n "${ALIGNED_ROOT}" ] && [ ! -d "${ALIGNED_ROOT}" ]; then
    echo "ERROR: ALIGNED_ROOT set to ${ALIGNED_ROOT} but directory does not exist"
    exit 1
fi

if [ -n "${AGE_AUGMENT_ROOT}" ] && [ ! -d "${AGE_AUGMENT_ROOT}" ]; then
    echo "ERROR: AGE_AUGMENT_ROOT set to ${AGE_AUGMENT_ROOT} but directory does not exist"
    echo "  Generate variants first with tools/sam_age_augment.py"
    exit 1
fi

# ---------- numbered output ---------------------------------------------------
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
echo 'Model 11: AdaFace + FaCoR Cross-Attention'
echo 'Platform: AMD ROCm (local)'
echo '============================================'
echo "Run ID:            ${RUN_LABEL}"
echo "Output:            ${RUN_DIR}"
echo "Epochs:            ${EPOCHS}"
echo "Batch size:        ${BATCH_SIZE}  (grad_accum ${GRAD_ACCUM} = eff $((BATCH_SIZE * GRAD_ACCUM)))"
echo "Learning rate:     ${LEARNING_RATE}"
echo "Weight decay:      ${WEIGHT_DECAY}"
echo "Scheduler:         ${SCHEDULER}"
echo "Warmup epochs:     ${WARMUP_EPOCHS}"
echo "Min LR:            ${MIN_LR}"
echo "Backbone:          AdaFace IR-101 (WebFace4M)"
echo "AdaFace weights:   ${ADAFACE_WEIGHTS}"
echo "Img size:          ${IMG_SIZE}"
echo "Cross-attn layers: ${CROSS_ATTN_LAYERS}"
echo "Cross-attn heads:  ${CROSS_ATTN_HEADS}"
echo "Dropout:           ${DROPOUT}"
echo "Loss:              ${LOSS}"
echo "Temperature:       ${TEMPERATURE}"
echo "Margin:            ${MARGIN}"
echo "Freeze backbone:   $([ "${FREEZE_BACKBONE}" = "1" ] && echo yes || echo no)"
echo "Pos embed:         $([ "${NO_POSITIONAL_EMBEDDING}" = "1" ] && echo off || echo on)"
echo "Global embed:      $([ "${NO_GLOBAL_EMBEDDING}" = "1" ] && echo off || echo on)"
echo "Classifier head:   $([ "${USE_CLASSIFIER_HEAD}" = "1" ] && echo yes || echo no)"
echo "Negative ratio:    ${NEGATIVE_RATIO}"
echo "Eval neg ratio:    ${EVAL_NEGATIVE_RATIO}"
echo "Train neg strat:   ${TRAIN_NEGATIVE_STRATEGY}"
echo "Eval neg strat:    ${EVAL_NEGATIVE_STRATEGY}"
echo "Workers:           ${NUM_WORKERS}"
echo "Patience:          ${PATIENCE}"
echo "Dataset:           ${TRAIN_DATASET}"
echo "Data root:         ${DATA_ROOT}"
echo "Aligned root:      ${ALIGNED_ROOT:-(none — using DATA_ROOT images)}"
echo "Age augment root:  ${AGE_AUGMENT_ROOT:-(none — no SAM age ensemble)}"
if [ -n "${AGE_AUGMENT_ROOT}" ]; then
echo "Age target ages:   ${AGE_TARGET_AGES}"
echo "Age original w:    ${AGE_ORIGINAL_WEIGHT}"
fi
echo "Seed:              ${SEED}"
echo "ROCm device:       ${GPU_ID}"
echo "Python:            ${PYTHON}"
echo '============================================'

export PYTHONPATH="${PROJECT_ROOT}/models:${PROJECT_ROOT}/models/shared:${PYTHONPATH}"

TRAIN_ARGS=(
    --train_dataset            "${TRAIN_DATASET}"
    --test_dataset             "${TRAIN_DATASET}"
    --data_root                "${DATA_ROOT}"
    --epochs                   "${EPOCHS}"
    --batch_size               "${BATCH_SIZE}"
    --gradient_accumulation    "${GRAD_ACCUM}"
    --lr                       "${LEARNING_RATE}"
    --weight_decay             "${WEIGHT_DECAY}"
    --scheduler                "${SCHEDULER}"
    --warmup_epochs            "${WARMUP_EPOCHS}"
    --min_lr                   "${MIN_LR}"
    --adaface_weights          "${ADAFACE_WEIGHTS}"
    --img_size                 "${IMG_SIZE}"
    --cross_attn_layers        "${CROSS_ATTN_LAYERS}"
    --cross_attn_heads         "${CROSS_ATTN_HEADS}"
    --dropout                  "${DROPOUT}"
    --loss                     "${LOSS}"
    --temperature              "${TEMPERATURE}"
    --margin                   "${MARGIN}"
    --negative_ratio           "${NEGATIVE_RATIO}"
    --eval_negative_ratio      "${EVAL_NEGATIVE_RATIO}"
    --train_negative_strategy  "${TRAIN_NEGATIVE_STRATEGY}"
    --eval_negative_strategy   "${EVAL_NEGATIVE_STRATEGY}"
    --num_workers              "${NUM_WORKERS}"
    --patience                 "${PATIENCE}"
    --max_grad_norm            "${MAX_GRAD_NORM}"
    --seed                     "${SEED}"
    --rocm_device              "${GPU_ID}"
    --checkpoint_dir           "${CKPT_DIR}"
)
[ "${FREEZE_BACKBONE}" = "1" ]         && TRAIN_ARGS+=(--freeze_backbone)
[ "${NO_POSITIONAL_EMBEDDING}" = "1" ] && TRAIN_ARGS+=(--no_positional_embedding)
[ "${NO_GLOBAL_EMBEDDING}" = "1" ]     && TRAIN_ARGS+=(--no_global_embedding)
[ "${USE_CLASSIFIER_HEAD}" = "1" ]     && TRAIN_ARGS+=(--use_classifier_head)
[ "${USE_MULTISTAGE}" = "1" ]          && TRAIN_ARGS+=(--use_multistage --cross_attn_stages "${CROSS_ATTN_STAGES}" --cross_attn_layers_per_stage "${CROSS_ATTN_LAYERS_PER_STAGE}")
[ -n "${ALIGNED_ROOT}" ]               && TRAIN_ARGS+=(--aligned_root "${ALIGNED_ROOT}")
if [ -n "${AGE_AUGMENT_ROOT}" ]; then
    TRAIN_ARGS+=(
        --age_augment_root    "${AGE_AUGMENT_ROOT}"
        --age_target_ages     "${AGE_TARGET_AGES}"
        --age_original_weight "${AGE_ORIGINAL_WEIGHT}"
    )
fi

TEST_ARGS=(
    --checkpoint  "${CKPT_DIR}/best.pt"
    --dataset     "${TRAIN_DATASET}"
    --data_root   "${DATA_ROOT}"
    --batch_size  "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --output_dir  "${RESULTS_DIR}"
    --rocm_device "${GPU_ID}"
)
[ -n "${ALIGNED_ROOT}" ] && TEST_ARGS+=(--aligned_root "${ALIGNED_ROOT}")
if [ -n "${AGE_AUGMENT_ROOT}" ]; then
    TEST_ARGS+=(
        --age_augment_root "${AGE_AUGMENT_ROOT}"
        --age_target_ages  "${AGE_TARGET_AGES}"
    )
fi

EVAL_ARGS=(
    --checkpoint  "${CKPT_DIR}/best.pt"
    --dataset     "${TRAIN_DATASET}"
    --data_root   "${DATA_ROOT}"
    --batch_size  "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --output_dir  "${RESULTS_DIR}"
    --rocm_device "${GPU_ID}"
    --full_analysis
    --visualize_attention
)
[ -n "${ALIGNED_ROOT}" ] && EVAL_ARGS+=(--aligned_root "${ALIGNED_ROOT}")
if [ -n "${AGE_AUGMENT_ROOT}" ]; then
    EVAL_ARGS+=(
        --age_augment_root "${AGE_AUGMENT_ROOT}"
        --age_target_ages  "${AGE_TARGET_AGES}"
    )
fi

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
echo "Model 11 — Run ${RUN_LABEL} completed!"
echo "Results:     ${RESULTS_DIR}"
echo "Checkpoints: ${CKPT_DIR}"
echo "Logs:        ${LOGS_DIR}"
echo '============================================'
