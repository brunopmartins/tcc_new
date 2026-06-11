#!/usr/bin/env bash
# M16 R002 (fixed-λ family-invariance) FIRST, then M15@160 (hi-res ROI). 10-min gaps.
set -uo pipefail
ROOT="/home/bruno/Desktop/tcc_new"; COOLDOWN="${COOLDOWN:-600}"; LOG="/tmp/orch_m16r002_m15.log"
ts(){ date '+%Y-%m-%d %H:%M:%S'; }
run(){ echo "[$(ts)] >>> $1 starting" | tee -a "$LOG"; bash "$2" > "$4" 2>&1; echo "[$(ts)] <<< $1 finished rc=$? (result: $([ -f "$3" ] && echo ok || echo MISSING))" | tee -a "$LOG"; }
echo "[$(ts)] GO: M16 R002 -> M15@160" | tee -a "$LOG"
run "M16 R002 (family-invariance, fixed λ)" "${ROOT}/models/16_fa_rgck/AMD/run_m16_r002.sh" \
    "${ROOT}/models/16_fa_rgck/output/002/results/test_metrics_rocm.txt" /tmp/m16_r002_run.log
sleep "${COOLDOWN}"
run "M15 R001 (hi-res ROI @160)" "${ROOT}/models/15_rgck_hires/AMD/run_m15_r001.sh" \
    "${ROOT}/models/15_rgck_hires/output/001/results/test_metrics_rocm.txt" /tmp/m15_160_run.log
echo "[$(ts)] DONE" | tee -a "$LOG"
