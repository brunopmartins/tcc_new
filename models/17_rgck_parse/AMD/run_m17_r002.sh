#!/usr/bin/env bash
# =====================================================================
# Model 17: PG-RGCK — R002 launch wrapper (finer feature map @ 224)
#
# R002 = R001 (parsing-guided per-image boxes, M12 R011 recipe) but with the
# conv body at the native IMG_SIZE=224 → a 14×14 feature map (4× the spatial
# detail of R001's 160→10×10). Rationale: adaptive per-face boxes only pay off
# if the feature map is fine enough to resolve the per-region differences they
# capture; at 10×10 a region spans 1-3 cells (the M13/R013 coarse-ROI problem).
# R002 gives the adaptive geometry room to bite. Costs more VRAM/time.
#
# Same box cache as R001 (boxes live in the 224 frame; src_coord_size=224
# normalises, so the SAME .npz works at any IMG_SIZE — no re-parsing).
#
# NOTE: the contour-aware "mask-weighted ROI" variant (#2 from the design
# discussion) is a FUTURE code step (weight the 7×7 ROI by the soft parse mask
# before the FC) — NOT wired here. R001/R002 are both the bbox mechanism (#1).
#
# VRAM: 224 is heavier than R001's 160. If OOM: BATCH_SIZE=4 GRAD_ACCUM=8, or
# fall back to IMG_SIZE=192.
#
# Usage:
#   bash models/17_rgck_parse/AMD/run_m17_r002.sh
# =====================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ALIGNED_ROOT="${ALIGNED_ROOT:-/home/bruno/Desktop/tcc_new/datasets/FIW_aligned}"
REGION_BOX_CACHE="${REGION_BOX_CACHE:-/home/bruno/Desktop/tcc_new/datasets/fiw_region_boxes.npz}"

if [ ! -f "${REGION_BOX_CACHE}" ]; then
    echo "ERROR: region-box cache not found: ${REGION_BOX_CACHE}"
    echo "       Build it first — see models/17_rgck_parse/AMD/run_m17_r001.sh header."
    exit 1
fi

SKIP_INSTALL=1 \
ALIGNED_ROOT="${ALIGNED_ROOT}" \
REGION_BOX_CACHE="${REGION_BOX_CACHE}" \
BATCH_SIZE="${BATCH_SIZE:-8}" \
GRAD_ACCUM="${GRAD_ACCUM:-4}" \
IMG_SIZE="${IMG_SIZE:-224}" \
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
