#!/usr/bin/env bash
# =====================================================================
# Sequential orchestrator (M14 R001 already done; .venv symlinks fixed).
# Starts M15 IMMEDIATELY, then 10-min cooldown between runs.
#   1. M15 R001 (hi-res ROI @224)            -> models/15_rgck_hires/output/001
#   2. M16 R001 (family-adversarial DANN)    -> models/16_fa_rgck/output/001
#   3. M12 R013 (ROI @112, batch 32)         -> models/12_rgck_net/output/018
# One GPU job at a time. Each run_pipeline trains (SAFEGUARD) + tests.
# Usage: nohup bash tools/orchestrate_m15_m16_r013_go.sh > /tmp/orch_go.log 2>&1 &
# =====================================================================
set -uo pipefail
ROOT="/home/bruno/Desktop/tcc_new"
COOLDOWN="${COOLDOWN:-600}"
LOG="/tmp/orch_go.log"
ts() { date '+%Y-%m-%d %H:%M:%S'; }
run_step() {  # name script result logf
    echo "[$(ts)] >>> $1 starting" | tee -a "$LOG"
    bash "$2" > "$4" 2>&1; local rc=$?
    echo "[$(ts)] <<< $1 finished rc=${rc} (result: $([ -f "$3" ] && echo ok || echo MISSING))" | tee -a "$LOG"
}

echo "[$(ts)] GO orchestrator start (M15 now -> M16 -> R013, ${COOLDOWN}s gaps)" | tee -a "$LOG"
run_step "M15 R001 (hi-res ROI @224)" "${ROOT}/models/15_rgck_hires/AMD/run_m15_r001.sh" \
         "${ROOT}/models/15_rgck_hires/output/001/results/test_metrics_rocm.txt" /tmp/m15_r001_run.log
sleep "${COOLDOWN}"
run_step "M16 R001 (family-adversarial)" "${ROOT}/models/16_fa_rgck/AMD/run_m16_r001.sh" \
         "${ROOT}/models/16_fa_rgck/output/001/results/test_metrics_rocm.txt" /tmp/m16_r001_run.log
sleep "${COOLDOWN}"
run_step "M12 R013 (ROI @112, batch 32)" "${ROOT}/models/12_rgck_net/AMD/run_r013.sh" \
         "${ROOT}/models/12_rgck_net/output/018/results/test_metrics_rocm.txt" /tmp/r013_run.log
echo "[$(ts)] ORCHESTRATOR DONE" | tee -a "$LOG"
