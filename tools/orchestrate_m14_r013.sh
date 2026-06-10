#!/usr/bin/env bash
# =====================================================================
# Sequential orchestrator (reordered after R013 was found ~4.6x slower):
#   1. M14 R001 (LoRA)        -> models/14_rgck_lora/output/001/   (~7 h, high-EV)
#   2. cooldown
#   3. M12 R013 (ROI-Align)   -> models/12_rgck_net/output/018/    (~27 h, slow)
#
# Each run_pipeline trains (auto-stops via the below-peak SAFEGUARD ~ep14-16)
# and then tests on the fixed RFIW Track-I set. Strictly ONE GPU job at a time.
#
# Usage: nohup bash tools/orchestrate_m14_r013.sh > /tmp/orch_m14_r013.log 2>&1 &
# =====================================================================
set -uo pipefail
ROOT="/home/bruno/Desktop/tcc_new"
COOLDOWN="${COOLDOWN:-900}"
LOG="/tmp/orch_m14_r013.log"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] ORCHESTRATOR START (M14 -> R013)" | tee -a "$LOG"

echo "[$(ts)] >>> M14 R001 (LoRA) starting" | tee -a "$LOG"
bash "${ROOT}/models/14_rgck_lora/AMD/run_m14_r001.sh" > /tmp/m14_r001_run.log 2>&1
echo "[$(ts)] <<< M14 R001 finished rc=$?" | tee -a "$LOG"

echo "[$(ts)] cooldown ${COOLDOWN}s before R013" | tee -a "$LOG"
sleep "${COOLDOWN}"

echo "[$(ts)] >>> M12 R013 (ROI-Align) starting" | tee -a "$LOG"
bash "${ROOT}/models/12_rgck_net/AMD/run_r013.sh" > /tmp/r013_run.log 2>&1
echo "[$(ts)] <<< M12 R013 finished rc=$?" | tee -a "$LOG"

echo "[$(ts)] ORCHESTRATOR DONE" | tee -a "$LOG"
