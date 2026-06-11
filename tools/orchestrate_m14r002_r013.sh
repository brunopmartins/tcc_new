#!/usr/bin/env bash
# =====================================================================
# Sequential orchestrator (queued after M14 R001, which is already running):
#   0. WAIT for M14 R001 to finish (its test_metrics file appears)
#   1. M14 R002 (LoRA + ROI-Align @ 224)  -> models/14_rgck_lora/output/002/
#   2. cooldown
#   3. M12 R013 (ROI-Align @ 112, batch 32) -> models/12_rgck_net/output/018/
#
# One GPU job at a time. Each run_pipeline trains (SAFEGUARD auto-stop) + tests.
#
# Usage: nohup bash tools/orchestrate_m14r002_r013.sh > /tmp/orch_m14r002_r013.log 2>&1 &
# =====================================================================
set -uo pipefail
ROOT="/home/bruno/Desktop/tcc_new"
COOLDOWN="${COOLDOWN:-900}"
LOG="/tmp/orch_m14r002_r013.log"
R001_RES="${ROOT}/models/14_rgck_lora/output/001/results/test_metrics_rocm.txt"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] orchestrator armed; waiting for M14 R001 to finish..." | tee -a "$LOG"
until [ -f "$R001_RES" ]; do sleep 120; done
echo "[$(ts)] M14 R001 done. cooldown ${COOLDOWN}s before R002." | tee -a "$LOG"
sleep "${COOLDOWN}"

echo "[$(ts)] >>> M14 R002 (LoRA + ROI-Align @224) starting" | tee -a "$LOG"
bash "${ROOT}/models/14_rgck_lora/AMD/run_m14_r002.sh" > /tmp/m14_r002_run.log 2>&1
echo "[$(ts)] <<< M14 R002 finished rc=$?" | tee -a "$LOG"

echo "[$(ts)] cooldown ${COOLDOWN}s before R013" | tee -a "$LOG"
sleep "${COOLDOWN}"

echo "[$(ts)] >>> M12 R013 (ROI-Align @112) starting" | tee -a "$LOG"
bash "${ROOT}/models/12_rgck_net/AMD/run_r013.sh" > /tmp/r013_run.log 2>&1
echo "[$(ts)] <<< M12 R013 finished rc=$?" | tee -a "$LOG"

echo "[$(ts)] ORCHESTRATOR DONE" | tee -a "$LOG"
