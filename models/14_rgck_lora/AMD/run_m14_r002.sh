#!/usr/bin/env bash
# =====================================================================
# Model 14: RGCK-Net + LoRA — R002 launch wrapper
#
# M14 R002 = R001 recipe (LoRA r16 on stage4+output_layer, symmetric +
# comparison-only + relation aux + role-matched hard negs 30%) PLUS:
#   - ROI-Align tokenizer (ROI_ALIGN_TOKENIZER=1)
#   - backbone fed at 224 (BACKBONE_INPUT_SIZE=224) -> 14×14 feature map,
#     giving the region ROI-Align a finer grid; the global token is
#     adaptive-pooled 14×14 -> 7×7 for AdaFace's fixed head.
#
# Note: FIW source faces are ~110px, so 224 is upsampled — the gain is the
# finer feature grid for region localisation, not new pixel detail.
#
# Batch: 224 input is ~4× the activation memory of 112, so BATCH_SIZE=8 ×
# GRAD_ACCUM=4 (eff 32, same recipe) for VRAM safety on the 12 GB card.
# Bump BATCH_SIZE if VRAM allows (improves ROI-Align backbone occupancy).
#
# Usage:
#   bash models/14_rgck_lora/AMD/run_m14_r002.sh
# =====================================================================

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SKIP_INSTALL=1 \
ALIGNED_ROOT="${ALIGNED_ROOT:-/home/bruno/Desktop/tcc_new/datasets/FIW_aligned}" \
BATCH_SIZE="${BATCH_SIZE:-8}" \
GRAD_ACCUM="${GRAD_ACCUM:-4}" \
FREEZE_BACKBONE=1 \
UNFREEZE_LAST_STAGE=0 \
LEARNING_RATE=1e-4 \
RELATION_AUX_WEIGHT=0.05 \
SYMMETRIC_FORWARD=1 \
COMPARISON_ONLY_FUSION=1 \
ROI_ALIGN_TOKENIZER=1 \
BACKBONE_INPUT_SIZE=224 \
TRAIN_NEGATIVE_STRATEGY=relation_matched \
HARD_NEGATIVE_RATIO=0.30 \
LORA_RANK=16 \
LORA_ALPHA=16 \
NUM_WORKERS=4 SEED=42 \
bash "${SCRIPT_DIR}/run_pipeline.sh"
