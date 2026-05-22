#!/usr/bin/env bash
# =====================================================================
# Model 02: ViT-FaCoR — 5-fold CV runner for the R031 recipe.
#
# Each fold takes ~6.5-7 h training (ViT-B/16 full FT, 100M params) +
# 5-10 min test + 15 min thermal cooldown. Total ~36 h continuous.
#
# Output layout:
#   output/033/fold_{0..4}/  — R031 recipe (full FT, LR 5e-6, dropout 0.2,
#                              neg 2:1 relation_matched, contrastive margin 0.3)
#
# Usage:
#   bash models/02_vit_facor_crossattn/AMD/cv_runner.sh
# =====================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${MODEL_ROOT}/../.." && pwd)"

COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-900}"  # 15 min thermal cooldown
NUM_FOLDS=5
RUN_ID="${RUN_ID:-033}"
RUNNER_LOG="/tmp/m02_cv_runner.log"

run_fold() {
    local fold="$1"
    local label="r031_fold${fold}"
    local results_file="${MODEL_ROOT}/output/${RUN_ID}/fold_${fold}/results/test_metrics_rocm.txt"

    if [ -f "${results_file}" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP ${label}: results exist at ${results_file}" | tee -a "${RUNNER_LOG}"
        return 0
    fi

    local started=$(date '+%Y-%m-%d %H:%M:%S')
    echo "==========================================" | tee -a "${RUNNER_LOG}"
    echo "[${started}] START ${label}" | tee -a "${RUNNER_LOG}"
    echo "==========================================" | tee -a "${RUNNER_LOG}"

    # R031 recipe (see run-review/run-031.md):
    #   ViT-B/16 ImageNet (no freeze, full FT)
    #   LR 5e-6 peak, warmup 5, cosine to 1e-7
    #   Dropout 0.2, contrastive loss with margin 0.3 + T 0.3
    #   Negative ratio 2:1 (train) + 1:1 (eval), relation_matched train negs
    #   Batch 32, 50 max epochs, patience 20
    SKIP_INSTALL=1 \
        ALIGNED_ROOT="${PROJECT_ROOT}/datasets/FIW_aligned" \
        TRAIN_DATASET=fiw \
        EPOCHS=50 \
        BATCH_SIZE=32 \
        LEARNING_RATE=5e-6 \
        WEIGHT_DECAY=1e-5 \
        WARMUP_EPOCHS=5 \
        MIN_LR=1e-7 \
        SCHEDULER=cosine \
        VIT_MODEL=vit_base_patch16_224 \
        CROSS_ATTN_LAYERS=2 \
        CROSS_ATTN_HEADS=8 \
        LOSS=contrastive \
        TEMPERATURE=0.3 \
        MARGIN=0.3 \
        DROPOUT=0.2 \
        FREEZE_VIT=0 \
        UNFREEZE_AFTER_EPOCH=0 \
        UNFREEZE_LAST_VIT_BLOCKS=0 \
        USE_CLASSIFIER_HEAD=0 \
        NEGATIVE_RATIO=2.0 \
        EVAL_NEGATIVE_RATIO=1.0 \
        TRAIN_NEGATIVE_STRATEGY=relation_matched \
        EVAL_NEGATIVE_STRATEGY=random \
        NUM_WORKERS=4 \
        PATIENCE=20 \
        SEED=42 \
        RUN_OVERRIDE="${RUN_ID}" \
        FOLD="${fold}" \
        NUM_FOLDS="${NUM_FOLDS}" \
        bash "${SCRIPT_DIR}/run_pipeline.sh" \
            > "/tmp/m02_cv_${label}.log" 2>&1
    local rc=$?

    local finished=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${finished}] END   ${label}  rc=${rc}" | tee -a "${RUNNER_LOG}"

    if [ "${rc}" -ne 0 ]; then
        echo "  WARNING: fold ${label} exited with rc=${rc}. Continuing." | tee -a "${RUNNER_LOG}"
    fi
}

cooldown() {
    local secs="$1"
    local label="$2"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] cooldown ${secs}s after ${label}" | tee -a "${RUNNER_LOG}"
    sleep "${secs}"
}

main() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] M02 R031 CV runner START" | tee -a "${RUNNER_LOG}"
    echo "  R031 -> output/${RUN_ID}/fold_{0..4}" | tee -a "${RUNNER_LOG}"
    echo "  cooldown between folds: ${COOLDOWN_SECONDS}s" | tee -a "${RUNNER_LOG}"
    echo "  Note: folds with existing results are skipped." | tee -a "${RUNNER_LOG}"

    # Determine if any fold already has results so we know to cooldown
    # before the first run that actually executes (GPU possibly warm).
    local some_done=0
    for k in 0 1 2 3 4; do
        [ -f "${MODEL_ROOT}/output/${RUN_ID}/fold_${k}/results/test_metrics_rocm.txt" ] && some_done=1
    done

    local first_unfinished=1
    for k in 0 1 2 3 4; do
        local results_file="${MODEL_ROOT}/output/${RUN_ID}/fold_${k}/results/test_metrics_rocm.txt"
        if [ -f "${results_file}" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP r031_fold${k}: results exist" | tee -a "${RUNNER_LOG}"
            continue
        fi

        if [ "${first_unfinished}" = "1" ]; then
            if [ "${some_done}" = "1" ]; then
                cooldown "${COOLDOWN_SECONDS}" "GPU warm from prior session"
            fi
            first_unfinished=0
        else
            cooldown "${COOLDOWN_SECONDS}" "previous fold completed"
        fi

        run_fold "${k}"
    done

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] M02 R031 CV runner DONE" | tee -a "${RUNNER_LOG}"
}

main "$@"
