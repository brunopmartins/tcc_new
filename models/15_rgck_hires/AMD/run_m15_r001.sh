#!/usr/bin/env bash
# =====================================================================
# Model 15: RGCK-Net hi-res ROI-Align — R001 launch wrapper
#
# M15 R001 = the M12 R011 champion recipe (symmetric forward +
# comparison-only fusion + relation-type aux 0.05 + role-matched hard
# negatives 30% + differential LR + stage-4 unfreeze) with the region
# tokenizer swapped for the HIGH-RESOLUTION ROI-Align tokenizer.
#
# Why: M12 squashes each region to 112×112 (out-of-distribution crops);
# R013's ROI-Align fixed that but still ran the body at 112 → a coarse
# 7×7 map. The 112 limit comes only from AdaFace's FC head; the conv
# body is resolution-agnostic. M15 runs the body at the native 224
# (→ 14×14 map, 4× the region detail) and ROI-Aligns each region to the
# 7×7 grid the FC needs — so the FC works at any resolution and each
# region token carries real spatial detail.
#
# Only the tokenizer changes vs M12 R011, so the delta isolates
# "hi-res in-distribution region tokens" vs "112 resize-crop tokens".
#
# global_token_mode=exact keeps the global token identical to M12/B0
# (genuine AdaFace embedding of the 112 face), so the comparison is
# clean: only the 4 anatomical regions get finer.
#
# VRAM: tuned for a 12 GB card (AMD RX 6750 XT). If you hit OOM, drop to
#   BATCH_SIZE=4 GRAD_ACCUM=8 (same effective batch 32).
#
# Usage:
#   bash models/15_rgck_hires/AMD/run_m15_r001.sh
# =====================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SKIP_INSTALL=1 \
ALIGNED_ROOT="${ALIGNED_ROOT:-/home/bruno/Desktop/tcc_new/datasets/FIW_aligned}" \
BATCH_SIZE="${BATCH_SIZE:-8}" \
GRAD_ACCUM="${GRAD_ACCUM:-4}" \
IMG_SIZE=224 \
UNFREEZE_LAST_STAGE=1 \
RELATION_AUX_WEIGHT=0.05 \
SYMMETRIC_FORWARD=1 \
DIFFERENTIAL_LR=1 \
LR_STAGE4=5e-6 \
LR_OUTPUT_LAYER=5e-6 \
LR_HEAD=2e-5 \
COMPARISON_ONLY_FUSION=1 \
TRAIN_NEGATIVE_STRATEGY=relation_matched \
HARD_NEGATIVE_RATIO=0.30 \
GLOBAL_TOKEN_MODE=exact \
NUM_WORKERS=4 SEED=42 \
bash "${SCRIPT_DIR}/run_pipeline.sh"
