#!/usr/bin/env bash
# =====================================================================
# Model 12: RGCK-Net — 5-fold CV runner (R006 then R010, 10 folds total)
#
# Each fold ≈ 5.7 h training + 7 min test + 15 min thermal cooldown.
# Total wall-clock: ≈ 60-65 h continuous.
#
# Output layout:
#   output/011/fold_{0..4}/  — R006 (Phase 5 + symmetric forward)
#   output/012/fold_{0..4}/  — R010 (R006 + diff-LR + cmp-only fusion)
#
# Usage:
#   bash models/12_rgck_net/AMD/run_cv_5fold.sh
# =====================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${MODEL_ROOT}/../.." && pwd)"

COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-900}"  # 15 min thermal cooldown
NUM_FOLDS=5

run_fold() {
    local recipe="$1"
    local run_id="$2"
    local fold="$3"
    local extra_env="$4"

    local label="${recipe}_fold${fold}"
    local results_file="${MODEL_ROOT}/output/${run_id}/fold_${fold}/results/test_metrics_rocm.txt"
    if [ -f "${results_file}" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP ${label}: results exist at ${results_file}" | tee -a "/tmp/cv_runner.log"
        return 0
    fi
    local started=$(date '+%Y-%m-%d %H:%M:%S')
    echo "==========================================" | tee -a "/tmp/cv_runner.log"
    echo "[${started}] START ${label}" | tee -a "/tmp/cv_runner.log"
    echo "==========================================" | tee -a "/tmp/cv_runner.log"

    # Use `env` so the variable-assignment prefixes from ${extra_env} are
    # parsed at runtime by env (which accepts VAR=VAL args), instead of being
    # interpreted by bash as shell-prefix assignments (which must be literal
    # in the source and don't accept expansion of VAR=VAL tokens).
    env ${extra_env} \
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
        RUN_OVERRIDE="${run_id}" \
        FOLD="${fold}" \
        NUM_FOLDS="${NUM_FOLDS}" \
        bash "${SCRIPT_DIR}/run_pipeline.sh" \
            > "/tmp/cv_${label}.log" 2>&1
    local rc=$?

    local finished=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${finished}] END   ${label}  rc=${rc}" | tee -a "/tmp/cv_runner.log"

    if [ "${rc}" -ne 0 ]; then
        echo "  WARNING: fold ${label} exited with rc=${rc}. Continuing." | tee -a "/tmp/cv_runner.log"
    fi
}

cooldown() {
    local secs="$1"
    local label="$2"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] cooldown ${secs}s after ${label}" | tee -a "/tmp/cv_runner.log"
    sleep "${secs}"
}

fold_was_run() {
    # Returns 0 if a fold was actually executed this invocation (not skipped),
    # so the caller can decide whether to cooldown before the next one.
    local recipe="$1"
    local run_id="$2"
    local fold="$3"
    local results_file="${MODEL_ROOT}/output/${run_id}/fold_${fold}/results/test_metrics_rocm.txt"
    # Refresh check: file exists AFTER run_fold, so combine "results exist" with
    # "we entered the body". Simplest: just check existence; caller skips
    # cooldown if it doesn't exist (which means run_fold failed).
    [ -f "${results_file}" ]
}

main() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] CV runner START" | tee -a "/tmp/cv_runner.log"
    echo "  R006 -> output/011/fold_{0..4}" | tee -a "/tmp/cv_runner.log"
    echo "  R010 -> output/012/fold_{0..4}" | tee -a "/tmp/cv_runner.log"
    echo "  cooldown between folds: ${COOLDOWN_SECONDS}s" | tee -a "/tmp/cv_runner.log"
    echo "  Note: folds with existing results are skipped." | tee -a "/tmp/cv_runner.log"

    local R010_EXTRA="DIFFERENTIAL_LR=1 LR_STAGE4=5e-6 LR_OUTPUT_LAYER=5e-6 LR_HEAD=2e-5 COMPARISON_ONLY_FUSION=1"

    # Full schedule: 10 folds total (R006 ×5, then R010 ×5).
    # Cooldown rule: any fold that runs is preceded by a 15-min cooldown,
    # except the very first executed fold in a clean (no existing results)
    # invocation. Skipped folds incur no cooldown — they're just metadata.
    local first_unfinished=1
    local schedule=("r006:011:0:" "r006:011:1:" "r006:011:2:" "r006:011:3:" "r006:011:4:"
                    "r010:012:0:${R010_EXTRA}" "r010:012:1:${R010_EXTRA}" "r010:012:2:${R010_EXTRA}" "r010:012:3:${R010_EXTRA}" "r010:012:4:${R010_EXTRA}")

    # Pre-pass: determine if any fold already has results, which means GPU
    # was used recently and we should cooldown before the first run too.
    local some_done=0
    for entry in "${schedule[@]}"; do
        IFS=':' read -r _ run_id fold _ <<<"${entry}"
        [ -f "${MODEL_ROOT}/output/${run_id}/fold_${fold}/results/test_metrics_rocm.txt" ] && some_done=1
    done

    for entry in "${schedule[@]}"; do
        IFS=':' read -r recipe run_id fold extra <<<"${entry}"

        local results_file="${MODEL_ROOT}/output/${run_id}/fold_${fold}/results/test_metrics_rocm.txt"
        if [ -f "${results_file}" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP ${recipe}_fold${fold}: results exist" | tee -a "/tmp/cv_runner.log"
            continue
        fi

        # First unfinished fold: cooldown only if some other fold was done
        # before this invocation (GPU recently warm).
        if [ "${first_unfinished}" = "1" ]; then
            if [ "${some_done}" = "1" ]; then
                cooldown "${COOLDOWN_SECONDS}" "GPU warm from prior session"
            fi
            first_unfinished=0
        else
            cooldown "${COOLDOWN_SECONDS}" "previous fold completed"
        fi

        run_fold "${recipe}" "${run_id}" "${fold}" "${extra}"
    done

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] CV runner DONE" | tee -a "/tmp/cv_runner.log"
}

main "$@"
