# Run 21 - Local / GPU / KinFaceW-I / Manual stop during epoch 12 / Relation-matched training negatives

**Date:** 2026-03-20
**GPU:** AMD Radeon RX 6750 XT (11.98 GB VRAM, gfx1031, ROCm 5.7)
**Status:** MANUALLY STOPPED during epoch 12 - best checkpoint recovered manually

---

## Context

Run 20 showed that shrinking the head was the wrong direction. The next step was to keep the run 19 backbone and loss recipe, but make the training negatives more informative instead of random.

### Changes implemented since Run 20

1. **Reverted to the stronger run 19 architecture** by restoring 2 cross-attention layers and dropout `0.1`.
2. **Kept supervised contrastive loss with margin `0.3`** because that was still the best-performing objective.
3. **Changed training negative sampling from `random` to `relation_matched`** so non-kin impostors come from the same KinFace relation type.
4. **Kept evaluation negatives random at 1:1** to preserve comparability with earlier runs.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Method | Local (`AMD/run_pipeline.sh` + `.venv`) |
| Device | GPU - cuda:0 (gfx1031, ROCm 5.7, PyTorch 2.3.1+rocm5.7) |
| Dataset | KinFaceW-I |
| Split | 70/15/15 -> 1865 train pairs / 158 val / 162 test |
| Epochs | 11 completed / 100 planned |
| Batch size | 32 |
| Learning rate | 1e-5 |
| Scheduler | none (warmup 1 epoch) |
| Loss | contrastive |
| Margin | 0.3 |
| Train negative ratio | 4.0 |
| Train negative strategy | relation_matched |
| Eval negative ratio | 1.0 |
| Eval negative strategy | random |
| ViT model | vit_base_patch16_224 |
| Cross-attn layers | 2 |
| Cross-attn heads | 8 |
| Dropout | 0.1 |
| Freeze ViT | Yes |
| Classifier head | No |
| Trainable params | 14.91M |
| AMP | Enabled (ROCm AMP) |

---

## Training Progress

| Epoch | Train Loss | Val Acc | Val AUC | LR | Notes |
|-------|-----------|---------|---------|-----|-------|
| 1 | 0.0078 | 0.5000 | 0.7071 | 1.00e-05 | Slightly below run 19 start |
| 2 | 0.0072 | 0.5000 | 0.7273 | 1.00e-05 | Good early climb |
| 3 | 0.0072 | 0.5000 | 0.7399 | 1.00e-05 | Still improving |
| 4 | 0.0069 | 0.5000 | 0.7529 | 1.00e-05 | Stable upward trend |
| 5 | 0.0066 | 0.5000 | 0.7635 | 1.00e-05 | Better than run 20 branch |
| 6 | 0.0066 | 0.5000 | 0.7712 | 1.00e-05 | Competitive but below run 19 |
| 7 | 0.0065 | 0.5000 | 0.7790 | 1.00e-05 | Continued gain |
| 8 | 0.0063 | 0.5000 | 0.7877 | 1.00e-05 | Strong checkpoint |
| 9 | 0.0062 | 0.5000 | 0.8010 | 1.00e-05 | Crossed 0.80 AUC |
| **10** | **0.0061** | **0.5000** | **0.8087** | **1.00e-05** | **Best checkpoint saved** |
| 11 | 0.0061 | 0.5000 | 0.8085 | 1.00e-05 | Flat vs epoch 10 |

The run was stopped in epoch 12 because epoch 10 was already the best checkpoint and the validation curve had started to flatten.

---

## Results

### Validation-selected threshold

| Metric | Value |
|--------|-------|
| Threshold | **0.900** |
| Val Accuracy | 74.68% |
| Val F1 | 0.785 |
| Val ROC-AUC | 0.849 |

### Test metrics (recovered protocol)

| Metric | Value |
|--------|-------|
| Accuracy | **68.52%** |
| Precision | 63.39% |
| Recall | 87.65% |
| F1 | **0.736** |
| ROC-AUC | 0.784 |
| TAR@FAR=0.001 | 0.00% |
| TAR@FAR=0.01 | 0.00% |
| TAR@FAR=0.1 | 33.33% |

### Comparison with Run 19

| Metric | Run 19 | Run 21 |
|--------|--------|--------|
| Accuracy | 67.28% | **68.52%** |
| F1 | 0.720 | **0.736** |
| ROC-AUC | **0.787** | 0.784 |

### Comparison with Run 20

| Metric | Run 20 | Run 21 |
|--------|--------|--------|
| Accuracy | 53.70% | **68.52%** |
| F1 | 0.684 | **0.736** |
| ROC-AUC | 0.692 | **0.784** |

Run 21 is the new best local run for **accuracy and F1**, but not for ROC-AUC.

---

## Analysis

### What improved

Relation-matched training negatives helped. The model kept the strong kin-pair recall from run 19 while pushing the thresholded operating point slightly further in the right direction.

### What did not improve enough

The remaining weakness is still non-kin rejection. Negative-class accuracy on test stayed below 50%, so the model is still too willing to call impostor pairs kin.

### Main takeaway

This was a useful step forward, but it did not fully solve the precision problem. The idea of harder negatives looks valid, yet `4:1` relation-matched negatives were not strong enough to lift AUC over run 19.

---

## Next Step

Keep the run 21 recipe and increase training negative pressure from `4:1` to `6:1` while preserving relation-matched sampling. That should test whether the same direction can improve precision rather than only recall.
