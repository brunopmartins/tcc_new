#!/usr/bin/env bash
# =====================================================================
# Model 12: RGCK-Net — 5-fold CV runner for R012
#
# R012 = R011 stack (R006 symmetric forward + R007 differential LR +
#   R009 comparison-only fusion + role-matched hard negatives at 30%)
#   PLUS:
#     - cosine-consistency loss between fusion_AB and fusion_BA (λ=0.05)
#     - extended stage-3 tail unfreeze (body[43:46]) on top of
#       body[46:49] + output_layer  -> ~34.6 M trainable (49.24 %)
#
# Single-run (output/016): Test AUC 0.8813, AP 0.8636,
#   TAR@FAR=0.001 0.0878 (best single-run strict-FAR in the project),
#   val->test gap -0.026. This CV checks whether that strict-FAR gain
#   is real or an upper-tail draw (cf. R011, whose single-run 0.8825
#   collapsed to a CV mean of 0.8761 +/- 0.0029).
#
# Each fold ~= 6 h training + ~5 min test + 15 min thermal cooldown.
# Total wall-clock: ~= 33-35 h continuous.
#
# Output layout:
#   output/017/fold_{0..4}/  — R012
#
# Usage:
#   bash models/12_rgck_net/AMD/cv_runner_r012.sh
# =====================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${MODEL_ROOT}/../.." && pwd)"

COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-900}"
INITIAL_COOLDOWN="${INITIAL_COOLDOWN:-900}"   # GPU is warm from the R012 single-run (output/016)
NUM_FOLDS=5
RUN_ID="${RUN_ID:-017}"
RUNNER_LOG="/tmp/r012_cv_runner.log"

# R011 stack + R012 delta:
#   DIFFERENTIAL_LR + LR_STAGE4 / LR_OUTPUT_LAYER / LR_HEAD             <- R007
#   COMPARISON_ONLY_FUSION=1                                           <- R009
#   TRAIN_NEGATIVE_STRATEGY=relation_matched + HARD_NEGATIVE_RATIO=0.30 <- R011
#   CONSISTENCY_WEIGHT=0.05 + UNFREEZE_EXTRA_STAGE3_TAIL=1             <- R012
# Everything else inherits from the inner cv schedule (R006 base).
R012_EXTRA="DIFFERENTIAL_LR=1 LR_STAGE4=5e-6 LR_OUTPUT_LAYER=5e-6 LR_HEAD=2e-5 COMPARISON_ONLY_FUSION=1 TRAIN_NEGATIVE_STRATEGY=relation_matched HARD_NEGATIVE_RATIO=0.30 CONSISTENCY_WEIGHT=0.05 UNFREEZE_EXTRA_STAGE3_TAIL=1"

run_fold() {
    local fold="$1"
    local label="r012_fold${fold}"
    local results_file="${MODEL_ROOT}/output/${RUN_ID}/fold_${fold}/results/test_metrics_rocm.txt"

    if [ -f "${results_file}" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP ${label}: results exist" | tee -a "${RUNNER_LOG}"
        return 0
    fi

    local started=$(date '+%Y-%m-%d %H:%M:%S')
    echo "==========================================" | tee -a "${RUNNER_LOG}"
    echo "[${started}] START ${label}" | tee -a "${RUNNER_LOG}"
    echo "==========================================" | tee -a "${RUNNER_LOG}"

    env ${R012_EXTRA} \
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
            > "/tmp/r012_cv_${label}.log" 2>&1
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
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] R012 CV runner START" | tee -a "${RUNNER_LOG}"
    echo "  R012 -> output/${RUN_ID}/fold_{0..4}" | tee -a "${RUNNER_LOG}"
    echo "  cooldown between folds: ${COOLDOWN_SECONDS}s; initial: ${INITIAL_COOLDOWN}s" | tee -a "${RUNNER_LOG}"
    echo "  Note: folds with existing results are skipped." | tee -a "${RUNNER_LOG}"

    local first_unfinished=1
    for k in 0 1 2 3 4; do
        local results_file="${MODEL_ROOT}/output/${RUN_ID}/fold_${k}/results/test_metrics_rocm.txt"
        if [ -f "${results_file}" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP r012_fold${k}: results exist" | tee -a "${RUNNER_LOG}"
            continue
        fi

        if [ "${first_unfinished}" = "1" ]; then
            cooldown "${INITIAL_COOLDOWN}" "GPU warm from R012 single-run (output/016)"
            first_unfinished=0
        else
            cooldown "${COOLDOWN_SECONDS}" "previous fold completed"
        fi

        run_fold "${k}"
    done

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] R012 CV runner DONE" | tee -a "${RUNNER_LOG}"
}

main "$@"
