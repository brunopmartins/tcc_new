# Run 14 - Local / GPU / KinFaceW-I / Manual stop at epoch 2 / Simpler head + more dropout

**Date:** 2026-03-20
**GPU:** AMD Radeon RX 6750 XT (11.98 GB VRAM, gfx1031, ROCm 5.7)
**Status:** MANUALLY STOPPED after epoch 2 - best checkpoint recovered manually

---

## Context

Run 13 suggested the optimizer and loss were no longer the main blockers. The next hypothesis was that the trainable head was still too expressive. Run 14 simplified the head while keeping the stronger BCE + 4:1-negative training setup.

### Changes implemented since Run 13

1. **Reduced cross-attention depth from 2 layers to 1** to cut head capacity.
2. **Raised dropout from 0.1 to 0.2** to regularize the smaller head more aggressively.
3. **Kept BCE, 4:1 train negatives, 1:1 evaluation, frozen ViT, and LR=1e-5** so the architectural simplification was isolated.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Method | Local (`AMD/run_pipeline.sh` + `.venv`) |
| Device | GPU - cuda:0 (gfx1031, ROCm 5.7, PyTorch 2.3.1+rocm5.7) |
| Dataset | KinFaceW-I |
| Split | 70/15/15 -> 1865 train pairs / 158 val / 162 test |
| Epochs | 2 completed / 100 planned |
| Batch size | 32 |
| Learning rate | 1e-5 |
| Scheduler | none (warmup 1 epoch) |
| Loss | bce |
| Temperature | 0.3 |
| Train negative ratio | 4.0 |
| Eval negative ratio | 1.0 |
| ViT model | vit_base_patch16_224 |
| Cross-attn layers | 1 |
| Cross-attn heads | 8 |
| Dropout | 0.2 |
| Freeze ViT | Yes |
| Trainable params | 7.8M |
| AMP | Enabled (ROCm AMP) |

---

## Training Progress

| Epoch | Train Loss | Val Acc | Val AUC | LR | Notes |
|-------|-----------|---------|---------|-----|-------|
| **1** | **0.8038** | **0.5063** | **0.8023** | **1.00e-05** | **Best checkpoint saved** |
| 2 | 0.7023 | 0.6646 | 0.7949 | 1.00e-05 | Regression |

The run was stopped manually because it showed the same pattern as earlier experiments: best AUC at the very start, then immediate regression. The saved best checkpoint was recovered manually with the shared validation-threshold protocol.

---

## Results

### Validation-selected threshold

| Metric | Value |
|--------|-------|
| Threshold | **0.650** |
| Val Accuracy | 72.15% |
| Val F1 | 0.732 |
| Val ROC-AUC | 0.807 |

### Test metrics (recovered protocol)

| Metric | Value |
|--------|-------|
| Accuracy | **62.96%** |
| Precision | 63.64% |
| Recall | 60.49% |
| F1 | 0.620 |
| ROC-AUC | 0.666 |
| TAR@FAR=0.001 | 0.00% |
| TAR@FAR=0.01 | 12.35% |
| TAR@FAR=0.1 | 35.80% |

### Comparison with Run 13

| Metric | Run 13 | Run 14 |
|--------|--------|--------|
| Accuracy | **62.96%** | **62.96%** |
| F1 | **0.667** | 0.620 |
| ROC-AUC | **0.676** | 0.666 |
| Negative accuracy | 51.85% | **65.43%** |

Run 14 substantially improved non-kin rejection, but the gain came from sacrificing too much recall.

---

## Analysis

### What improved

The simpler head did exactly what it was supposed to do on the negative class. Test negative accuracy jumped to 65.4%, which is the best specificity seen in these new runs.

### What still failed

The model over-corrected. Recall fell from 74.1% in run 13 to 60.5%, which pulled F1 and ROC-AUC down even though accuracy stayed flat.

### Likely root cause

The direction is right, but **the base ViT backbone plus simplified head still does not find a stable balance between kin recall and non-kin rejection**. The model is now less biased toward positives, but not enough overall to beat the earlier baseline.

---

## Next Step

Keep the simpler head idea, but test whether the backbone itself is oversized for KinFaceW-I:

1. switch from `vit_base_patch16_224` to `vit_small_patch16_224`
2. keep BCE, 4:1 train negatives, 1-layer head, and dropout 0.2
