#!/usr/bin/env bash
# =====================================================================
# Model 12: RGCK-Net — 5-fold CV runner for R011 (NEW HEADLINE recipe)
#
# R011 = R010 stack + role-matched hard negatives at 30% mix.
# Each fold ≈ 6 h training + 7 min test + 15 min thermal cooldown.
# Total wall-clock: ≈ 33 h continuous.
#
# Output layout:
#   output/014/fold_{0..4}/  — R011 (R010 + relation_matched fixed + HNR=0.30)
#
# Usage:
#   bash models/12_rgck_net/AMD/cv_runner_r011.sh
# =====================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${MODEL_ROOT}/../.." && pwd)"

COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-900}"
NUM_FOLDS=5
RUN_ID="${RUN_ID:-014}"
RUNNER_LOG="/tmp/r011_cv_runner.log"

# R010 stack + R011 negative-sampling delta:
#   DIFFERENTIAL_LR + LR_STAGE4 / LR_OUTPUT_LAYER / LR_HEAD  ← R007
#   COMPARISON_ONLY_FUSION=1                                  ← R009
#   TRAIN_NEGATIVE_STRATEGY=relation_matched + HARD_NEGATIVE_RATIO=0.30  ← R011
# Everything else inherits from the inner cv schedule (R006 base).
R011_EXTRA="DIFFERENTIAL_LR=1 LR_STAGE4=5e-6 LR_OUTPUT_LAYER=5e-6 LR_HEAD=2e-5 COMPARISON_ONLY_FUSION=1 TRAIN_NEGATIVE_STRATEGY=relation_matched HARD_NEGATIVE_RATIO=0.30"

run_fold() {
    local fold="$1"
    local label="r011_fold${fold}"
    local results_file="${MODEL_ROOT}/output/${RUN_ID}/fold_${fold}/results/test_metrics_rocm.txt"

    if [ -f "${results_file}" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP ${label}: results exist" | tee -a "${RUNNER_LOG}"
        return 0
    fi

    local started=$(date '+%Y-%m-%d %H:%M:%S')
    echo "==========================================" | tee -a "${RUNNER_LOG}"
    echo "[${started}] START ${label}" | tee -a "${RUNNER_LOG}"
    echo "==========================================" | tee -a "${RUNNER_LOG}"

    env ${R011_EXTRA} \
        SKIP_INSTALL=1 \
        ALIGNED_ROOT="${PROJECT_ROOT}/datasets/FIW_aligned" \
        BATCH_SIZE=8 \
        GRAD_ACCUM=4 \
        UNFREEZE_LAST_STAGE=1 \
        LEARNING_RATE=1e-5 \
        RELATION_AUX_WEIGHT=0.05 \
        SYMMETRIC_FORWARD=1 \
        NUM_WORKERS=4 \
        SEED=42 \
        RUN_OVERRIDE="${RUN_ID}" \
        FOLD="${fold}" \
        NUM_FOLDS="${NUM_FOLDS}" \
        bash "${SCRIPT_DIR}/run_pipeline.sh" \
            > "/tmp/r011_cv_${label}.log" 2>&1
    local rc=$?

    local finished=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${finished}] END   ${label}  rc=${rc}" | tee -a "${RUNNER_LOG}"
    [ "${rc}" -ne 0 ] && echo "  WARNING: fold ${label} exited with rc=${rc}. Continuing." | tee -a "${RUNNER_LOG}"
}

cooldown() {
    local secs="$1"
    local label="$2"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] cooldown ${secs}s after ${label}" | tee -a "${RUNNER_LOG}"
    sleep "${secs}"
}

main() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] R011 CV runner START" | tee -a "${RUNNER_LOG}"
    echo "  R011 -> output/${RUN_ID}/fold_{0..4}" | tee -a "${RUNNER_LOG}"
    echo "  cooldown between folds: ${COOLDOWN_SECONDS}s" | tee -a "${RUNNER_LOG}"
    echo "  Note: folds with existing results are skipped." | tee -a "${RUNNER_LOG}"

    local some_done=0
    for k in 0 1 2 3 4; do
        [ -f "${MODEL_ROOT}/output/${RUN_ID}/fold_${k}/results/test_metrics_rocm.txt" ] && some_done=1
    done

    local first_unfinished=1
    for k in 0 1 2 3 4; do
        local results_file="${MODEL_ROOT}/output/${RUN_ID}/fold_${k}/results/test_metrics_rocm.txt"
        if [ -f "${results_file}" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP r011_fold${k}: results exist" | tee -a "${RUNNER_LOG}"
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

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] R011 CV runner DONE" | tee -a "${RUNNER_LOG}"
}

main "$@"
