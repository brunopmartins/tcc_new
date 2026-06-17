#!/usr/bin/env bash
# =====================================================================
# Model 17: PG-RGCK (parsing-guided regions) — R001 launch wrapper
#
# M17 R001 = the SAME recipe as M15 R001 (= M12 R011 champion: symmetric
# forward + comparison-only fusion + relation aux 0.05 + role-matched hard
# negatives 30% + differential LR + stage-4 unfreeze + hi-res ROI-Align)
# with ONE change: the four anatomical region boxes (eyes, nose, mouth, jaw)
# are PER-IMAGE, derived offline by a face-parsing network, instead of the
# fixed DEFAULT_REGIONS_224 rectangles. So the delta vs M15 R001 isolates
# exactly "adaptive per-face geometry" vs "fixed boxes" — the open question
# M13 (canonical landmarks) could not answer.
#
# PREREQUISITE — build the box cache once (offline, CPU is fine):
#   git clone https://github.com/zllrunning/face-parsing.PyTorch  /path/bisenet
#   # download its weights 79999_iter.pth
#   python tools/parse_faces_boxes.py --backend bisenet \
#       --bisenet-repo /path/bisenet --weights /path/79999_iter.pth \
#       --aligned-root "$ALIGNED_ROOT" --out "$REGION_BOX_CACHE"
#   # sanity first: --backend dummy writes the fixed boxes → M17 ≡ M15.
#
# geometric aug is auto-disabled when REGION_BOX_CACHE is set (flip/rotation
# would desync the cached boxes); the eval path passes the same boxes, so
# there is no train/eval geometry mismatch.
#
# IMG_SIZE matches M15 R001 (160) so the comparison is clean. Boxes live in
# the 224 frame regardless (src_coord_size=224 normalises), so the parser is
# always run on the 224 aligned faces.
#
# VRAM: identical to M15 R001 (parser is offline; 0 extra trainable params).
# If OOM: BATCH_SIZE=4 GRAD_ACCUM=8.
#
# Usage:
#   bash models/17_rgck_parse/AMD/run_m17_r001.sh
#   REGION_BOX_CACHE=/path/fiw_region_boxes.npz bash .../run_m17_r001.sh
# =====================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ALIGNED_ROOT="${ALIGNED_ROOT:-/home/bruno/Desktop/tcc_new/datasets/FIW_aligned}"
REGION_BOX_CACHE="${REGION_BOX_CACHE:-/home/bruno/Desktop/tcc_new/datasets/fiw_region_boxes.npz}"

if [ ! -f "${REGION_BOX_CACHE}" ]; then
    echo "ERROR: region-box cache not found: ${REGION_BOX_CACHE}"
    echo "       Build it first (see the header of this script):"
    echo "         python tools/parse_faces_boxes.py --backend bisenet \\"
    echo "           --bisenet-repo <repo> --weights <79999_iter.pth> \\"
    echo "           --aligned-root ${ALIGNED_ROOT} --out ${REGION_BOX_CACHE}"
    echo "       Or --backend dummy for a fixed-box (≡ M15) sanity cache."
    exit 1
fi

SKIP_INSTALL=1 \
ALIGNED_ROOT="${ALIGNED_ROOT}" \
REGION_BOX_CACHE="${REGION_BOX_CACHE}" \
BATCH_SIZE="${BATCH_SIZE:-8}" \
GRAD_ACCUM="${GRAD_ACCUM:-4}" \
IMG_SIZE="${IMG_SIZE:-160}" \
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
