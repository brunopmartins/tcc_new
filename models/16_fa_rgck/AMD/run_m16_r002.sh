#!/usr/bin/env bash
# =====================================================================
# Model 16: FA-RGCK (Family-Adversarial RGCK-Net) — R002 launch wrapper
#
# R002 = the SAME recipe as R001 (M12 R011 head + family-invariance DANN
# branch) with three fixes that make it an actual test of the hypothesis
# the R001 run never reached (see models/16_fa_rgck/run-review/run-001.md
# caveats):
#
#   1. DANN λ ramp DECOUPLED from --epochs and made SHORT.
#      R001 set the ramp length to --epochs (=100) but the run safeguard-
#      stopped at ~ep14 and the best-val checkpoint (ep3) saw λ ≈ 0.15 →
#      effective adversarial weight ≈ 0.015 — essentially R011 with a 1.5%
#      nudge. R002 sets DANN_MAX_EPOCHS=6 with a gentler γ=5 so λ climbs
#      gradually but is already substantial by the typical val peak (ep3-4).
#      Actual per-epoch λ the trainer prints (λ=0 on epoch 1, ramps after):
#        ep1 λ=0.000 | ep2 0.394 | ep3 0.682 | ep4 0.848 | ep5 0.931 | ep6+ ≈0.99
#      → the SELECTED best-val checkpoint actually experiences the treatment
#      (λ≈0.68-0.85 vs R001's ≈0.15).
#
#   2. Base family weight raised 0.1 → 0.3 (N=1 at 0.1 was inconclusive;
#      0.3 is the first real step of the planned sweep).
#
#   3. Per-term loss logging (added in train.py): each epoch prints
#        [M16] λ=… | base(BCE+rel)=… | CE_family(raw)=… | weighted_fam=…
#      so the "did it memorise families?" question (R001 caveat 2) is
#      answerable from the log — watch whether base(BCE+rel) collapses to
#      ~0.01 (memorised anyway) or stays high ~0.27 (invariance is biting).
#
# Everything else is verbatim R001 / R011: symmetric forward, cmp-only
# fusion, relation aux 0.05, role-matched hard negs 30%, single LR 1e-5,
# IMG_SIZE 224, batch 8 × grad-accum 4, seed 42.
#
# Decision rules (from run-001.md, now testable at real strength):
#   - Test AUC ≥ 0.884                  → beats R011 ensemble as single model.
#   - 0.876 ≤ AUC < 0.884 AND gap < R011's (−0.021) → mechanism works; CV.
#   - base(BCE+rel) high + gap NOT tighter + AUC flat → ceiling is
#     representational/data-bound (the R001 conclusion, now actually tested).
#
# VRAM: identical to R001 (~31.3M trainable). 12 GB OK at batch 8 × accum 4.
# If OOM: BATCH_SIZE=4 GRAD_ACCUM=8.
#
# Usage:
#   bash models/16_fa_rgck/AMD/run_m16_r002.sh
# Sweep the pressure without editing this file:
#   FAMILY_ADV_WEIGHT=0.5 bash models/16_fa_rgck/AMD/run_m16_r002.sh
#   DANN_MAX_EPOCHS=10 DANN_GAMMA=10 bash models/16_fa_rgck/AMD/run_m16_r002.sh
# =====================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SKIP_INSTALL=1 \
ALIGNED_ROOT="${ALIGNED_ROOT:-/home/bruno/Desktop/tcc_new/datasets/FIW_aligned}" \
BATCH_SIZE="${BATCH_SIZE:-8}" \
GRAD_ACCUM="${GRAD_ACCUM:-4}" \
IMG_SIZE=224 \
UNFREEZE_LAST_STAGE=1 \
LEARNING_RATE=1e-5 \
RELATION_AUX_WEIGHT=0.05 \
SYMMETRIC_FORWARD=1 \
COMPARISON_ONLY_FUSION=1 \
TRAIN_NEGATIVE_STRATEGY=relation_matched \
HARD_NEGATIVE_RATIO=0.30 \
FAMILY_ADV_WEIGHT="${FAMILY_ADV_WEIGHT:-0.3}" \
DANN_GAMMA="${DANN_GAMMA:-5.0}" \
DANN_MAX_EPOCHS="${DANN_MAX_EPOCHS:-6}" \
ADV_HIDDEN=256 \
NUM_WORKERS=4 SEED=42 \
bash "${SCRIPT_DIR}/run_pipeline.sh"
