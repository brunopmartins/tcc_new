#!/bin/bash
# =============================================================================
# Model 07 — AdaFace + Stacking — AMD ROCm pipeline runner
#
# STATUS: SCAFFOLD — train.py/test.py/evaluate.py are stubs.
# See ../IMPLEMENTATION_PLAN.md.
#
# When implemented, this script will mirror models/05_dinov2_lora_diffattn/AMD/run_pipeline.sh
# with these key differences:
#   - ADAFACE_ARCH=ir50 (or ir100)
#   - ADAFACE_PRETRAINED_PATH=... (path to AdaFace weights)
#   - IMG_SIZE=112 (AdaFace native resolution)
#   - USE_ALIGNED_CROPS=1 (requires running tools/align_fiw_dataset.py first)
#   - LOSS=arcface_pair MARGIN=0.3 SCALE=64
#
# Quick start once implemented:
#   bash models/07_adaface_stacked/AMD/run_pipeline.sh
# =============================================================================

echo 'Model 07 is in scaffold phase.'
echo 'See models/07_adaface_stacked/IMPLEMENTATION_PLAN.md before running.'
exit 1
