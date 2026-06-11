#!/usr/bin/env bash
# =====================================================================
# Sequential orchestrator (queued after M14 R001, already running):
#   0. WAIT for M14 R001 to finish (its test_metrics file appears)
#   1. M15 R001 (hi-res ROI-Align @224, R011 recipe) -> models/15_rgck_hires/output/001
#   2. M16 R001 (family-adversarial DANN, R011 recipe) -> models/16_fa_rgck/output/001
#   3. M12 R013 (ROI-Align @112, batch 32)            -> models/12_rgck_net/output/018
#
# One GPU job at a time. Each run_pipeline trains (SAFEGUARD auto-stop) + tests.
# (M14 R002 dropped — M15 is the canonical hi-res-ROI experiment.)
#
# Usage: nohup bash tools/orchestrate_m15_m16_r013.sh > /tmp/orch_m15_m16_r013.log 2>&1 &
# =====================================================================
set -uo pipefail
ROOT="/home/bruno/Desktop/tcc_new"
COOLDOWN="${COOLDOWN:-900}"
LOG="/tmp/orch_m15_m16_r013.log"
R001_RES="${ROOT}/models/14_rgck_lora/output/001/results/test_metrics_rocm.txt"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

run_step() {  # name, script, result_file
    local name="$1" script="$2" res="$3" logf="$4"
    echo "[$(ts)] >>> ${name} starting" | tee -a "$LOG"
    bash "${script}" > "${logf}" 2>&1
    echo "[$(ts)] <<< ${name} finished rc=$? (result: $([ -f "$res" ] && echo ok || echo MISSING))" | tee -a "$LOG"
}

echo "[$(ts)] armed; waiting for M14 R001 to finish..." | tee -a "$LOG"
until [ -f "$R001_RES" ]; do sleep 120; done
echo "[$(ts)] M14 R001 done." | tee -a "$LOG"

sleep "${COOLDOWN}"
run_step "M15 R001 (hi-res ROI @224)" "${ROOT}/models/15_rgck_hires/AMD/run_m15_r001.sh" \
         "${ROOT}/models/15_rgck_hires/output/001/results/test_metrics_rocm.txt" /tmp/m15_r001_run.log

sleep "${COOLDOWN}"
run_step "M16 R001 (family-adversarial)" "${ROOT}/models/16_fa_rgck/AMD/run_m16_r001.sh" \
         "${ROOT}/models/16_fa_rgck/output/001/results/test_metrics_rocm.txt" /tmp/m16_r001_run.log

sleep "${COOLDOWN}"
run_step "M12 R013 (ROI @112, batch 32)" "${ROOT}/models/12_rgck_net/AMD/run_r013.sh" \
         "${ROOT}/models/12_rgck_net/output/018/results/test_metrics_rocm.txt" /tmp/r013_run.log

echo "[$(ts)] ORCHESTRATOR DONE" | tee -a "$LOG"
