# Run 13 - Local / GPU / KinFaceW-I / Early stop at epoch 12 / Frozen ViT / BCE + 4:1 negatives

**Date:** 2026-03-20
**GPU:** AMD Radeon RX 6750 XT (11.98 GB VRAM, gfx1031, ROCm 5.7)
**Status:** COMPLETED - early stopped at epoch 12

---

## Context

Run 12 made the frozen-head optimization much more stable, but cosine contrastive loss still produced weak non-kin rejection. Run 13 kept the stable optimizer settings and changed the supervision signal plus class pressure.

### Changes implemented since Run 12

1. **Switched loss from `cosine_contrastive` to `bce`** to train the decision boundary directly.
2. **Raised train negative ratio from 2:1 to 4:1** to make non-kin rejection more important during training.
3. **Kept evaluation at 1:1 negatives** so the test protocol stayed comparable.
4. **Kept the frozen ViT, low LR, and no scheduler** because those were the most stable parts of run 12.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Method | Local (`AMD/run_pipeline.sh` + `.venv`) |
| Device | GPU - cuda:0 (gfx1031, ROCm 5.7, PyTorch 2.3.1+rocm5.7) |
| Dataset | KinFaceW-I |
| Split | 70/15/15 -> 1865 train pairs / 158 val / 162 test |
| Epochs | 12 completed / 100 planned |
| Batch size | 32 |
| Learning rate | 1e-5 |
| Scheduler | none (warmup 1 epoch) |
| Loss | bce |
| Temperature | 0.3 |
| Train negative ratio | 4.0 |
| Eval negative ratio | 1.0 |
| ViT model | vit_base_patch16_224 |
| Cross-attn layers | 2 |
| Cross-attn heads | 8 |
| Freeze ViT | Yes |
| Trainable params | 14.9M |
| AMP | Enabled (ROCm AMP) |

---

## Training Progress

| Epoch | Train Loss | Val Acc | Val AUC | LR | Notes |
|-------|-----------|---------|---------|-----|-------|
| 1 | 0.8113 | 0.5823 | 0.8050 | 1.00e-05 | Strong start |
| **2** | **0.6714** | **0.7278** | **0.8079** | **1.00e-05** | **Best checkpoint saved** |
| 3 | 0.6155 | 0.6962 | 0.8045 | 1.00e-05 | Small regression |
| 5 | 0.5494 | 0.5696 | 0.7813 | 1.00e-05 | Overfitting begins |
| 10 | 0.5242 | 0.5506 | 0.7605 | 1.00e-05 | Plateau/regression |
| 12 | 0.5239 | 0.5506 | 0.7530 | 1.00e-05 | Early stopping |

---

## Results

### Validation-selected threshold

| Metric | Value |
|--------|-------|
| Threshold | **0.450** |
| Val Accuracy | 72.15% |
| Val F1 | 0.758 |
| Val ROC-AUC | 0.808 |

### Test metrics (shared protocol)

| Metric | Value |
|--------|-------|
| Accuracy | **62.96%** |
| Precision | 60.61% |
| Recall | 74.07% |
| F1 | 0.667 |
| ROC-AUC | 0.676 |
| TAR@FAR=0.001 | 0.00% |
| TAR@FAR=0.01 | 0.00% |
| TAR@FAR=0.1 | 24.69% |

### Comparison with Run 12

| Metric | Run 12 | Run 13 |
|--------|--------|--------|
| Accuracy | 59.26% | **62.96%** |
| F1 | 0.656 | **0.667** |
| ROC-AUC | 0.643 | **0.676** |

Run 13 is the best of the new frozen-backbone experiments so far, but it still does not beat run 09 (`66.05% acc / 0.715 F1 / 0.767 AUC`) or the README CNN+FaCoR target.

---

## Analysis

### What improved

The BCE + 4:1 setup improved the negative-class behavior. Test negative accuracy rose from 40.7% in run 12 to 51.9% here, and all headline test metrics improved.

### What still failed

The improvement was not large enough. Even after harder negatives, the model still sits well below the best earlier ViT run and far below the CNN+FaCoR README benchmark.

### Likely root cause

The main failure mode now looks less like optimizer instability and more like **excess head capacity relative to the dataset size**. The 2-layer cross-attention stack may still be flexible enough to overfit kin-favoring shortcuts.

---

## Next Step

Keep the stronger supervision setup:

1. frozen ViT
2. BCE
3. 4:1 train negatives

Simplify the trainable head:

1. reduce cross-attention layers from 2 to 1
2. raise dropout for stronger regularization
