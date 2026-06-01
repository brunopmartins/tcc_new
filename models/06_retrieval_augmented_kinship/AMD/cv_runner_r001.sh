#!/usr/bin/env bash
# =====================================================================
# Model 06: Retrieval-Augmented Kinship — 5-fold CV runner for R001
#
# R001 = ViT-base + frozen backbone + K=32 retrieval + 60 epochs early
# stop. Single-run produced Test AUC 0.776, best Val AUC 0.836 at ep 8.
# This runner repeats the same recipe over 5 family-disjoint folds for
# proper CV reporting.
#
# Output layout:
#   output/<run>/fold_{0..4}/ — same R001 recipe per fold
#
# Wall-clock estimate: 5 folds × ~6h training + 5 × ~7 min test
#                      + 4 × 15 min thermal cooldown = ~33h continuous.
# =====================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${MODEL_ROOT}/../.." && pwd)"

COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-900}"
NUM_FOLDS=5
if [ -z "${RUN_ID:-}" ]; then
    next=1
    while [ -d "${MODEL_ROOT}/output/$(printf '%03d' ${next})" ]; do
        next=$((next + 1))
    done
    RUN_ID="$(printf '%03d' ${next})"
fi
RUNNER_LOG="/tmp/m06_r001_cv_runner.log"

R001_EXTRA="EPOCHS=20 PATIENCE=10"

run_fold() {
    local fold="$1"
    local label="r001_fold${fold}"
    local results_file="${MODEL_ROOT}/output/${RUN_ID}/fold_${fold}/results/test_metrics_rocm.json"

    if [ -f "${results_file}" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP ${label}: results exist" | tee -a "${RUNNER_LOG}"
        return 0
    fi

    local started=$(date '+%Y-%m-%d %H:%M:%S')
    echo "==========================================" | tee -a "${RUNNER_LOG}"
    echo "[${started}] START ${label}" | tee -a "${RUNNER_LOG}"
    echo "==========================================" | tee -a "${RUNNER_LOG}"

    env ${R001_EXTRA} \
        SKIP_INSTALL=1 \
        NUM_WORKERS=4 \
        SEED=42 \
        RUN_OVERRIDE="${RUN_ID}" \
        FOLD="${fold}" \
        NUM_FOLDS="${NUM_FOLDS}" \
        bash "${SCRIPT_DIR}/run_pipeline.sh" \
            > "/tmp/m06_r001_cv_${label}.log" 2>&1
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
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] M06 R001 CV runner START" | tee -a "${RUNNER_LOG}"
    echo "  R001 -> output/${RUN_ID}/fold_{0..4}" | tee -a "${RUNNER_LOG}"
    echo "  cooldown between folds: ${COOLDOWN_SECONDS}s" | tee -a "${RUNNER_LOG}"

    local some_done=0
    for k in 0 1 2 3 4; do
        [ -f "${MODEL_ROOT}/output/${RUN_ID}/fold_${k}/results/test_metrics_rocm.json" ] && some_done=1
    done

    local first_unfinished=1
    for k in 0 1 2 3 4; do
        local results_file="${MODEL_ROOT}/output/${RUN_ID}/fold_${k}/results/test_metrics_rocm.json"
        if [ -f "${results_file}" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP r001_fold${k}: results exist" | tee -a "${RUNNER_LOG}"
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

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] M06 R001 CV runner DONE" | tee -a "${RUNNER_LOG}"
}

main "$@"
