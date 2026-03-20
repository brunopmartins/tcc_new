# Run 12 - Local / GPU / KinFaceW-I / Early stop at epoch 13 / Frozen ViT / Lower LR

**Date:** 2026-03-20
**GPU:** AMD Radeon RX 6750 XT (11.98 GB VRAM, gfx1031, ROCm 5.7)
**Status:** COMPLETED - early stopped at epoch 13

---

## Context

Run 11 confirmed that freezing the ViT backbone was directionally correct, but `LR=1e-4` was still too aggressive for the cross-attention/projection stack. Run 12 kept the frozen-backbone setup and changed only the optimizer behavior to see whether the immediate epoch-1 collapse could be avoided.

### Changes implemented since Run 11

1. **Reduced LR from `1e-4` to `1e-5`** to stop the rapid head drift seen in run 11.
2. **Removed the plateau scheduler** and used `scheduler=none` with a minimal 1-epoch warmup.
3. **Kept the ViT frozen** (`FREEZE_VIT=1`) because that was still the best explanation for the run-10 threshold collapse.
4. **Kept train negatives at 2:1 and evaluation at 1:1** so only optimization changed.
5. **Reduced patience to 10** because the goal was to quickly test whether the gentler optimizer stabilized validation AUC.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Method | Local (`AMD/run_pipeline.sh` + `.venv`) |
| Device | GPU - cuda:0 (gfx1031, ROCm 5.7, PyTorch 2.3.1+rocm5.7) |
| Dataset | KinFaceW-I |
| Split | 70/15/15 -> 1119 train / 158 val / 162 test |
| Epochs | 13 completed / 100 planned |
| Batch size | 32 |
| Learning rate | 1e-5 |
| Scheduler | none (warmup 1 epoch) |
| Loss | cosine_contrastive |
| Temperature | 0.3 |
| Train negative ratio | 2.0 |
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
| 1 | 4.0929 | 0.5000 | 0.7776 | 1.00e-05 | Initial checkpoint |
| 2 | 3.9706 | 0.5000 | 0.8000 | 1.00e-05 | Clear improvement |
| **3** | **3.7060** | **0.5000** | **0.8048** | **1.00e-05** | **Best checkpoint saved** |
| 4 | 3.2849 | 0.5000 | 0.7845 | 1.00e-05 | Regression starts |
| 10 | 2.4013 | 0.5000 | 0.7462 | 1.00e-05 | Continued drift |
| 13 | 2.3138 | 0.5000 | 0.7318 | 1.00e-05 | Early stopping |

The online validation accuracy stayed at 0.5000 because training logs use the default threshold during the epoch loop. Cross-run comparison below uses the shared validation-threshold protocol saved at the end of training.

---

## Results

### Validation-selected threshold

| Metric | Value |
|--------|-------|
| Threshold | **0.800** |
| Val Accuracy | 71.52% |
| Val F1 | 0.749 |
| Val ROC-AUC | 0.805 |

### Test metrics (shared protocol)

| Metric | Value |
|--------|-------|
| Accuracy | **59.26%** |
| Precision | 56.76% |
| Recall | 77.78% |
| F1 | 0.656 |
| ROC-AUC | 0.643 |
| TAR@FAR=0.001 | 0.00% |
| TAR@FAR=0.01 | 7.41% |
| TAR@FAR=0.1 | 19.75% |

### Comparison with Run 11

| Metric | Run 11 | Run 12 |
|--------|--------|--------|
| Accuracy | 57.41% | **59.26%** |
| F1 | **0.670** | 0.656 |
| ROC-AUC | **0.692** | 0.643 |

Run 12 improved validation stability and slightly improved test accuracy, but it regressed on the more reliable ranking metrics (F1 and ROC-AUC).

---

## Analysis

### What improved

The lower LR fixed the immediate head destruction from run 11. Validation AUC now improved from epoch 1 to epoch 3 instead of peaking immediately and collapsing.

### What still failed

The model still generalized poorly to non-kin pairs. Test negative accuracy remained only 40.7%, so the gentler optimizer did not solve the core recall bias.

### Likely root cause

The frozen-backbone setup seems workable, but **cosine contrastive loss still pulls the head toward overly permissive kin predictions** even when optimization is stable. Validation AUC improved, but the decision boundary on test remained weak.

---

## Next Step

Keep the parts that helped:

1. frozen ViT
2. low LR
3. fair 1:1 evaluation

Change the supervision and class pressure:

1. switch to `bce`
2. increase train negatives from 2:1 to 4:1
