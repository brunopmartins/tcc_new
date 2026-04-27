#!/bin/bash
# =============================================================================
# Model 05 — DINOv2 + LoRA + Differential FaCoR — AMD ROCm pipeline runner
#
# Local host execution (no Docker). Creates its own virtualenv under the
# model's folder and runs train → test → evaluate in a numbered output dir.
#
# Run from anywhere in the project:
#   bash models/05_dinov2_lora_diffattn/AMD/run_pipeline.sh
#
# Override any setting via environment variable, e.g.:
#   EPOCHS=60 TRAIN_DATASET=kinface \
#       bash models/05_dinov2_lora_diffattn/AMD/run_pipeline.sh
#
# For 12 GB VRAM AMD, defaults are already OOM-safe. If you still hit OOM:
#   BATCH_SIZE=2 GRAD_ACCUM=16 bash .../run_pipeline.sh
#
# Skip pip install after first run:
#   SKIP_INSTALL=1 bash .../run_pipeline.sh
#
# Environment variables (defaults shown):
#   EPOCHS                80
#   BATCH_SIZE            4           # keep small — VRAM-bound
#   GRAD_ACCUM            8           # effective batch = BATCH_SIZE * this
#   LEARNING_RATE         3e-4        # LoRA tolerates high LR
#   WEIGHT_DECAY          1e-4
#   WARMUP_EPOCHS         3
#   MIN_LR                1e-6
#   SCHEDULER             cosine
#   TRAIN_DATASET         fiw
#   BACKBONE              vit_base_patch14_dinov2.lvd142m
#   IMG_SIZE              224
#   LORA_RANK             8
#   LORA_ALPHA            16
#   LORA_DROPOUT          0.0
#   CROSS_ATTN_LAYERS     2
#   CROSS_ATTN_HEADS      8
#   DROPOUT               0.1
#   EMBEDDING_DIM         512
#   LOSS                  combined
#   CONTRASTIVE_WEIGHT    0.5
#   TEMPERATURE           0.1
#   RELATION_SET          (auto = TRAIN_DATASET)
#   RELATION_LOSS_WEIGHT  0.2
#   NEGATIVE_RATIO        (dataset default)
#   EVAL_NEGATIVE_RATIO   (dataset default)
#   TRAIN_NEG_STRATEGY    (dataset default)
#   EVAL_NEG_STRATEGY     (dataset default)
#   NUM_WORKERS           4
#   PATIENCE              25
#   MAX_GRAD_NORM         1.0
#   DISABLE_AMP           0
#   NO_GRAD_CKPT          0
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

EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
MIN_LR="${MIN_LR:-1e-6}"
SCHEDULER="${SCHEDULER:-cosine}"
TRAIN_DATASET="${TRAIN_DATASET:-fiw}"
BACKBONE="${BACKBONE:-vit_base_patch14_dinov2.lvd142m}"
IMG_SIZE="${IMG_SIZE:-224}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
CROSS_ATTN_LAYERS="${CROSS_ATTN_LAYERS:-2}"
CROSS_ATTN_HEADS="${CROSS_ATTN_HEADS:-8}"
DROPOUT="${DROPOUT:-0.1}"
EMBEDDING_DIM="${EMBEDDING_DIM:-512}"
LOSS="${LOSS:-combined}"
CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-0.5}"
TEMPERATURE="${TEMPERATURE:-0.1}"
RELATION_SET="${RELATION_SET:-}"
RELATION_LOSS_WEIGHT="${RELATION_LOSS_WEIGHT:-0.2}"
NEGATIVE_RATIO="${NEGATIVE_RATIO:-}"
EVAL_NEGATIVE_RATIO="${EVAL_NEGATIVE_RATIO:-}"
TRAIN_NEG_STRATEGY="${TRAIN_NEG_STRATEGY:-}"
EVAL_NEG_STRATEGY="${EVAL_NEG_STRATEGY:-}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PATIENCE="${PATIENCE:-25}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
DISABLE_AMP="${DISABLE_AMP:-0}"
NO_GRAD_CKPT="${NO_GRAD_CKPT:-0}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

# ---------- ROCm: ensure render group access to /dev/kfd ----------------------
if [ -e /dev/kfd ] && [ -z "${_ROCM_SETUP_COMPLETE}" ]; then
    if ! id -Gn 2>/dev/null | tr ' ' '\n' | grep -qx render; then
        echo "  [ROCm] Adding $(whoami) to render+video groups for /dev/kfd access..."
        sudo usermod -aG render,video "$(whoami)"
    fi
    export EPOCHS BATCH_SIZE GRAD_ACCUM LEARNING_RATE WEIGHT_DECAY WARMUP_EPOCHS MIN_LR SCHEDULER
    export TRAIN_DATASET BACKBONE IMG_SIZE LORA_RANK LORA_ALPHA LORA_DROPOUT
    export CROSS_ATTN_LAYERS CROSS_ATTN_HEADS DROPOUT EMBEDDING_DIM
    export LOSS CONTRASTIVE_WEIGHT TEMPERATURE RELATION_SET RELATION_LOSS_WEIGHT
    export NEGATIVE_RATIO EVAL_NEGATIVE_RATIO TRAIN_NEG_STRATEGY EVAL_NEG_STRATEGY
    export NUM_WORKERS PATIENCE MAX_GRAD_NORM DISABLE_AMP NO_GRAD_CKPT GPU_ID SEED SKIP_INSTALL
    export _ROCM_SETUP_COMPLETE=1
    SELF="$(readlink -f "${BASH_SOURCE[0]}")"
    echo "  [ROCm] Restarting script with render group active..."
    exec sg render -c "bash '${SELF}'"
fi

# ---------- ROCm environment (before torch loads) -----------------------------
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-FAST}"
export HSA_FORCE_FINE_GRAIN_PCIE="${HSA_FORCE_FINE_GRAIN_PCIE:-1}"
export HIP_VISIBLE_DEVICES="${GPU_ID}"
export ROCR_VISIBLE_DEVICES="${GPU_ID}"

if ! id -Gn 2>/dev/null | tr ' ' '\n' | grep -qx render; then
    echo "WARNING: Not in the 'render' group — GPU may not be detected."
    echo "  Fix: sudo usermod -aG render,video \$(whoami) && log out/in"
fi

# ---------- virtualenv --------------------------------------------------------
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

    echo "  Done. Run with SKIP_INSTALL=1 to skip this step next time."
    echo ''
fi

if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: virtualenv not found at ${MODEL_ROOT}/.venv"
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
echo 'Model 05: DINOv2 + LoRA + DiffAttn'
echo 'Platform: AMD ROCm (local)'
echo '============================================'
echo "Run ID:            ${RUN_LABEL}"
echo "Output:            ${RUN_DIR}"
echo "Epochs:            ${EPOCHS}"
echo "Batch size:        ${BATCH_SIZE}"
echo "Grad accum:        ${GRAD_ACCUM}  (effective batch = $((BATCH_SIZE * GRAD_ACCUM)))"
echo "Learning rate:     ${LEARNING_RATE}"
echo "Weight decay:      ${WEIGHT_DECAY}"
echo "Scheduler:         ${SCHEDULER}"
echo "Warmup epochs:     ${WARMUP_EPOCHS}"
echo "Min LR:            ${MIN_LR}"
echo "Backbone:          ${BACKBONE}"
echo "Image size:        ${IMG_SIZE}"
echo "LoRA rank / alpha: ${LORA_RANK} / ${LORA_ALPHA}"
echo "LoRA dropout:      ${LORA_DROPOUT}"
echo "Cross-attn:        ${CROSS_ATTN_LAYERS} layers, ${CROSS_ATTN_HEADS} heads"
echo "Dropout:           ${DROPOUT}"
echo "Embedding dim:     ${EMBEDDING_DIM}"
echo "Loss:              ${LOSS}"
echo "Contrastive w:     ${CONTRASTIVE_WEIGHT}"
echo "Temperature:       ${TEMPERATURE}"
echo "Relation set:      ${RELATION_SET:-auto (=${TRAIN_DATASET})}"
echo "Relation weight:   ${RELATION_LOSS_WEIGHT}"
echo "Patience:          ${PATIENCE}"
echo "AMP:               $([ "${DISABLE_AMP}" = "1" ] && echo off || echo on)"
echo "Grad checkpoint:   $([ "${NO_GRAD_CKPT}" = "1" ] && echo off || echo on)"
echo "Dataset:           ${TRAIN_DATASET}"
echo "Seed:              ${SEED}"
echo "ROCm device:       ${GPU_ID}"
echo "Python:            ${PYTHON}"
echo '============================================'

export PYTHONPATH="${PROJECT_ROOT}/models:${PROJECT_ROOT}/models/shared:${PYTHONPATH}"

TRAIN_ARGS=(
    --train_dataset            "${TRAIN_DATASET}"
    --epochs                   "${EPOCHS}"
    --batch_size               "${BATCH_SIZE}"
    --gradient_accumulation    "${GRAD_ACCUM}"
    --lr                       "${LEARNING_RATE}"
    --weight_decay             "${WEIGHT_DECAY}"
    --warmup_epochs            "${WARMUP_EPOCHS}"
    --min_lr                   "${MIN_LR}"
    --scheduler                "${SCHEDULER}"
    --backbone_name            "${BACKBONE}"
    --img_size                 "${IMG_SIZE}"
    --lora_rank                "${LORA_RANK}"
    --lora_alpha               "${LORA_ALPHA}"
    --lora_dropout             "${LORA_DROPOUT}"
    --cross_attn_layers        "${CROSS_ATTN_LAYERS}"
    --cross_attn_heads         "${CROSS_ATTN_HEADS}"
    --dropout                  "${DROPOUT}"
    --embedding_dim            "${EMBEDDING_DIM}"
    --loss                     "${LOSS}"
    --contrastive_weight       "${CONTRASTIVE_WEIGHT}"
    --temperature              "${TEMPERATURE}"
    --relation_loss_weight     "${RELATION_LOSS_WEIGHT}"
    --num_workers              "${NUM_WORKERS}"
    --patience                 "${PATIENCE}"
    --max_grad_norm            "${MAX_GRAD_NORM}"
    --seed                     "${SEED}"
    --rocm_device              "${GPU_ID}"
    --checkpoint_dir           "${CKPT_DIR}"
)
[ -n "${RELATION_SET}" ]          && TRAIN_ARGS+=(--relation_set "${RELATION_SET}")
[ -n "${NEGATIVE_RATIO}" ]        && TRAIN_ARGS+=(--negative_ratio "${NEGATIVE_RATIO}")
[ -n "${EVAL_NEGATIVE_RATIO}" ]   && TRAIN_ARGS+=(--eval_negative_ratio "${EVAL_NEGATIVE_RATIO}")
[ -n "${TRAIN_NEG_STRATEGY}" ]    && TRAIN_ARGS+=(--train_negative_strategy "${TRAIN_NEG_STRATEGY}")
[ -n "${EVAL_NEG_STRATEGY}" ]     && TRAIN_ARGS+=(--eval_negative_strategy "${EVAL_NEG_STRATEGY}")
[ "${DISABLE_AMP}" = "1" ]        && TRAIN_ARGS+=(--disable_amp)
[ "${NO_GRAD_CKPT}" = "1" ]       && TRAIN_ARGS+=(--no_grad_ckpt)

TEST_ARGS=(
    --checkpoint   "${CKPT_DIR}/best.pt"
    --dataset      "${TRAIN_DATASET}"
    --batch_size   "${BATCH_SIZE}"
    --output_dir   "${RESULTS_DIR}"
    --num_workers  "${NUM_WORKERS}"
    --rocm_device  "${GPU_ID}"
)

EVAL_ARGS=(
    --checkpoint    "${CKPT_DIR}/best.pt"
    --dataset       "${TRAIN_DATASET}"
    --batch_size    "${BATCH_SIZE}"
    --output_dir    "${RESULTS_DIR}"
    --num_workers   "${NUM_WORKERS}"
    --rocm_device   "${GPU_ID}"
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
echo "Model 05 — Run ${RUN_LABEL} completed"
echo "Results:     ${RESULTS_DIR}"
echo "Checkpoints: ${CKPT_DIR}"
echo "Logs:        ${LOGS_DIR}"
echo '============================================'
