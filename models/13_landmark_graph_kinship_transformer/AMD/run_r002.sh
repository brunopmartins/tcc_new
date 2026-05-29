#!/usr/bin/env bash
# =====================================================================
# Model 13: LGKT-Net — R002 launch wrapper
#
# R002 = R001 stack +
#   - stage-3 feature map (14×14×256, projected to 512)
#   - comparison-only SymmetricPairPooler (global node excluded from
#     pooling, retained as graph context)
#   - relation aux head, balanced CE, λ=0.05
#   - UNFREEZE_LAST_STAGE=1 unfreezes body[43:46] (last 3 stage-3 blocks)
#
# Single-run on the canonical train/val/test split (no CV yet — gated on
# R002 hitting >= 0.870 Test AUC per the M13 line-restart criteria).
#
# Usage:
#   bash models/13_landmark_graph_kinship_transformer/AMD/run_r002.sh
# =====================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SKIP_INSTALL=1 \
ALIGNED_ROOT="${ALIGNED_ROOT:-/home/bruno/Desktop/tcc_new/datasets/FIW_aligned}" \
UNFREEZE_LAST_STAGE=1 \
LEARNING_RATE=1e-5 \
BATCH_SIZE=16 GRAD_ACCUM=2 \
DROPOUT=0.2 \
FEATURE_STAGE=stage3 \
COMPARISON_ONLY_POOLING=1 \
RELATION_AUX_WEIGHT=0.05 \
RELATION_AUX_BALANCED=1 \
NUM_WORKERS=4 SEED=42 \
bash "${SCRIPT_DIR}/run_pipeline.sh"
