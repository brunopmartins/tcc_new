#!/usr/bin/env bash
# =====================================================================
# Sequential orchestrator: M12 R013 (ROI-Align)  ->  M14 R001 (LoRA)
#
# Each run_pipeline trains (auto-stops via the below-peak SAFEGUARD ~ep14-16)
# and then tests on the fixed RFIW Track-I set. Fully unattended.
#
#   1. M12 R013  -> models/12_rgck_net/output/018/   (ROI-Align tokenizer)
#   2. cooldown (thermal headroom for the RX 6750 XT)
#   3. M14 R001  -> models/14_rgck_lora/output/001/  (LoRA backbone)
#
# Usage:  nohup bash tools/orchestrate_r013_m14.sh > /tmp/orch_r013_m14.log 2>&1 &
# =====================================================================
set -uo pipefail
ROOT="/home/bruno/Desktop/tcc_new"
COOLDOWN="${COOLDOWN:-900}"
LOG="/tmp/orch_r013_m14.log"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] ORCHESTRATOR START" | tee -a "$LOG"

echo "[$(ts)] >>> M12 R013 (ROI-Align) starting" | tee -a "$LOG"
bash "${ROOT}/models/12_rgck_net/AMD/run_r013.sh" > /tmp/r013_run.log 2>&1
echo "[$(ts)] <<< M12 R013 finished rc=$?" | tee -a "$LOG"

echo "[$(ts)] cooldown ${COOLDOWN}s before M14" | tee -a "$LOG"
sleep "${COOLDOWN}"

echo "[$(ts)] >>> M14 R001 (LoRA) starting" | tee -a "$LOG"
bash "${ROOT}/models/14_rgck_lora/AMD/run_m14_r001.sh" > /tmp/m14_r001_run.log 2>&1
echo "[$(ts)] <<< M14 R001 finished rc=$?" | tee -a "$LOG"

echo "[$(ts)] ORCHESTRATOR DONE" | tee -a "$LOG"
