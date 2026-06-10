#!/usr/bin/env bash
# =====================================================================
# Model 16: FA-RGCK (Family-Adversarial RGCK-Net) — R001 launch wrapper
#
# M16 R001 = the M12 R011 data+head recipe (symmetric forward +
# comparison-only fusion + relation-type aux 0.05 + role-matched hard
# negatives 30%) + a FAMILY-ADVERSARIAL objective: a gradient-reversal
# discriminator that tries to predict each face's FIW family, driving the
# backbone to family-invariant features (domain generalisation, each
# family = a domain).
#
# Why: every M00-M15 model is capped by generalisation to unseen families
# (the val→test gap; R012 showed memorisation by ~epoch 3). No model
# attacks this DIRECTLY. FA-RGCK does — the DANN λ schedule ramps the
# invariance pressure 0→1, so it starts at the M12 R011 solution (λ=0 ≡
# R011, high floor) and increases family-invariance as training proceeds.
#
# Single LR (no differential LR): the family adversary adds params under a
# new top-level module, and the differential-LR grouping resolves the
# backbone by module reference — single LR keeps the optimizer simple and
# robust. (Differential LR is a later variant once name-routing is adapted.)
#
# VRAM: ~31.3 M trainable (= R011 31.0 M + 0.28 M adversary), backbone same
# as M12. Tuned for 12 GB (AMD RX 6750 XT). If OOM: BATCH_SIZE=4 GRAD_ACCUM=8.
#
# Usage:
#   bash models/16_fa_rgck/AMD/run_m16_r001.sh
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
FAMILY_ADV_WEIGHT="${FAMILY_ADV_WEIGHT:-0.1}" \
DANN_GAMMA="${DANN_GAMMA:-10.0}" \
ADV_HIDDEN=256 \
NUM_WORKERS=4 SEED=42 \
bash "${SCRIPT_DIR}/run_pipeline.sh"
