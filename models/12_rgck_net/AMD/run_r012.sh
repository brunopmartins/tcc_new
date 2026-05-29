#!/usr/bin/env bash
# =====================================================================
# Model 12: RGCK-Net — R012 launch wrapper
#
# R012 = R011 stack +
#   - cosine-consistency loss between fusion_AB and fusion_BA (λ=0.05).
#     Forces the pair representation to be order-invariant at the fusion
#     level, not only at the BCE-output level (R006 Option-B already
#     forces order-invariance at the logit; this is a strictly stronger
#     constraint that propagates back through the cross-region adapter).
#   - additionally unfreeze body[43:46] (stage-3 tail). Gives the
#     backbone more capacity (~3 BasicBlockIR × 256-ch units) to satisfy
#     the consistency constraint without collapsing the fusion features.
#
# Hypothesis: the AUC ceiling at 0.876 ± 0.003 (R006/R010/R011 CV) is
# partly because cross_region(A,B) and cross_region(B,A) still produce
# meaningfully different fusion vectors even with symmetric BCE — the
# average is forced symmetric, but each direction is free to memorise
# direction-specific shortcuts that cancel only on average. Adding an
# explicit consistency penalty forbids that.
#
# Single-run on the canonical train/val/test split (no CV yet — gated
# on R012 hitting ≥ 0.880 Test AUC, or any meaningful CV-grade lift
# in TAR@FAR=0.001 above R011's 0.068 ± 0.015).
#
# Usage:
#   bash models/12_rgck_net/AMD/run_r012.sh
# =====================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SKIP_INSTALL=1 \
ALIGNED_ROOT="${ALIGNED_ROOT:-/home/bruno/Desktop/tcc_new/datasets/FIW_aligned}" \
BATCH_SIZE=8 \
GRAD_ACCUM=4 \
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
NUM_WORKERS=4 SEED=42 \
bash "${SCRIPT_DIR}/run_pipeline.sh"
