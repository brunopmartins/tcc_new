#!/usr/bin/env python3
"""
Model 07 — AdaFace + Stacking — AMD ROCm training (SCAFFOLD).

STATUS: stub. See ../IMPLEMENTATION_PLAN.md Phase 1.4 (R001) and 2.3 (R002).

When implemented, mirror models/05_dinov2_lora_diffattn/AMD/train.py with
these adaptations:
  - Import AdaFaceStackedKinship (instead of DINOv2LoRAKinship)
  - Use ArcFacePairLoss (instead of BCE+contrastive combined)
  - Default IMG_SIZE=112 (AdaFace native)
  - Support --use_aligned_crops flag (loads from FIW_aligned/)
  - Support --use_tta flag (test-time augmentation in validation phase)
  - Keep the multi-LR optimizer pattern from M05 for partial unfreeze
"""
import sys

if __name__ == "__main__":
    print("Model 07 train.py is a scaffold. Implement per IMPLEMENTATION_PLAN.md.")
    sys.exit(1)
