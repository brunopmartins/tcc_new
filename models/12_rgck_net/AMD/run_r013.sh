#!/usr/bin/env bash
# =====================================================================
# Model 12: RGCK-Net — R013 launch wrapper
#
# R013 = R012 recipe + ROI-Align region tokenizer (single change vs R012).
#
#   ROI_ALIGN_TOKENIZER=1 replaces the FixedPartition tokenizer (crop each
#   anatomical box, squash to 112×112, re-run AdaFace 5× — which distorts
#   thin strips like the eye box and feeds AdaFace out-of-distribution crops)
#   with a single conv-body forward + ROI-Align pooling on the shared 7×7
#   feature map. Region tokens become undistorted, in-distribution, and stay
#   in AdaFace's embedding space (shared output_layer); the global token is
#   the exact standard AdaFace embedding. Cost: 1 backbone pass instead of 5.
#
# Motivation (measured on the R012 checkpoint, full 13 425-pair test):
#   - AUC(model)=0.8813 vs AUC(global cosine)=0.8649 → the whole regional +
#     classifier head adds only +0.0164 over a plain global cosine.
#   - mean gate weight of the eyes region = 0.21 (most distorted crop, most
#     distrusted), vs nose 0.85 / jaw 0.76 / mouth 0.75 / global 0.63.
#   ROI-Align removes the distortion so the regions can carry real signal.
#
# Single change vs R012: ONLY the tokenizer. Everything else identical, so
# any Test-AUC delta is attributable to the tokenizer (cf. the ±0.008
# negative-sampler noise floor). Embedding-aware hard negatives are the
# SEPARATE R014 follow-up.
#
# Usage:
#   bash models/12_rgck_net/AMD/run_r013.sh
# =====================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ROI-Align runs the backbone ONCE on BATCH_SIZE images (vs FixedPartition's
# BATCH_SIZE×5 region crops), so at BATCH_SIZE=8 the backbone ran at batch 8 and
# badly underutilised the GPU (~1.2 it/s). ROI-Align uses ~5× less backbone
# memory per sample, so we raise the batch to restore occupancy. BATCH_SIZE=32 ×
# GRAD_ACCUM=1 keeps the effective batch at 32 (same recipe) while running the
# backbone at batch 32 ≈ FixedPartition's effective 40 → speed recovers.
SKIP_INSTALL=1 \
ALIGNED_ROOT="${ALIGNED_ROOT:-/home/bruno/Desktop/tcc_new/datasets/FIW_aligned}" \
BATCH_SIZE="${BATCH_SIZE:-32}" \
GRAD_ACCUM="${GRAD_ACCUM:-1}" \
UNFREEZE_LAST_STAGE=1 \
UNFREEZE_EXTRA_STAGE3_TAIL=1 \
LEARNING_RATE=1e-5 \
RELATION_AUX_WEIGHT=0.05 \
SYMMETRIC_FORWARD=1 \
DIFFERENTIAL_LR=1 \
LR_STAGE4=5e-6 \
LR_OUTPUT_LAYER=5e-6 \
LR_HEAD=2e-5 \
COMPARISON_ONLY_FUSION=1 \
TRAIN_NEGATIVE_STRATEGY=relation_matched \
HARD_NEGATIVE_RATIO=0.30 \
CONSISTENCY_WEIGHT=0.05 \
ROI_ALIGN_TOKENIZER=1 \
NUM_WORKERS=4 SEED=42 \
bash "${SCRIPT_DIR}/run_pipeline.sh"
