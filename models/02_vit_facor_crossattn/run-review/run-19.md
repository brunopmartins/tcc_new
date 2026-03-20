# Run 19 - Local / GPU / KinFaceW-I / Manual stop during epoch 9 / Lower contrastive margin

**Date:** 2026-03-20
**GPU:** AMD Radeon RX 6750 XT (11.98 GB VRAM, gfx1031, ROCm 5.7)
**Status:** MANUALLY STOPPED during epoch 9 - best checkpoint recovered manually

---

## Context

Run 18 established that the label-aware contrastive fix was the first real breakthrough of the batch. The next question was how hard the loss should push non-kin pairs apart. Run 19 kept the same architecture and only relaxed the contrastive margin.

### Changes implemented since Run 18

1. **Reduced contrastive margin from `0.5` to `0.3`** to make the negative-pair constraint less aggressive.
2. **Kept the supervised contrastive loss** because run 18 showed it was far stronger than the BCE branch.
3. **Kept the 2-layer head and dropout 0.1** so the margin change stayed isolated.
4. **Kept frozen ViT, LR=1e-5, 4:1 train negatives, and 1:1 evaluation** for direct comparability.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Method | Local (`AMD/run_pipeline.sh` + `.venv`) |
| Device | GPU - cuda:0 (gfx1031, ROCm 5.7, PyTorch 2.3.1+rocm5.7) |
| Dataset | KinFaceW-I |
| Split | 70/15/15 -> 1865 train pairs / 158 val / 162 test |
| Epochs | 8 completed / 100 planned |
| Batch size | 32 |
| Learning rate | 1e-5 |
| Scheduler | none (warmup 1 epoch) |
| Loss | contrastive |
| Margin | 0.3 |
| Train negative ratio | 4.0 |
| Eval negative ratio | 1.0 |
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
| 1 | 0.0078 | 0.5000 | 0.7166 | 1.00e-05 | Better than run 18 start |
| 2 | 0.0071 | 0.5000 | 0.7406 | 1.00e-05 | Strong improvement |
| 3 | 0.0072 | 0.5000 | 0.7609 | 1.00e-05 | Still climbing |
| 4 | 0.0068 | 0.5000 | 0.7760 | 1.00e-05 | Best so far in batch |
| 5 | 0.0065 | 0.5000 | 0.7891 | 1.00e-05 | Continued gain |
| 6 | 0.0065 | 0.5000 | 0.8047 | 1.00e-05 | Crossed 0.80 AUC |
| 7 | 0.0064 | 0.5000 | 0.8141 | 1.00e-05 | Strong checkpoint |
| **8** | **0.0063** | **0.5000** | **0.8213** | **1.00e-05** | **Best checkpoint saved** |

The run was stopped in epoch 9 because epoch 8 already gave a clear new high-water mark for this direction. The saved best checkpoint was recovered and evaluated with the shared validation-threshold protocol.

---

## Results

### Validation-selected threshold

| Metric | Value |
|--------|-------|
| Threshold | **0.900** |
| Val Accuracy | 71.52% |
| Val F1 | 0.757 |
| Val ROC-AUC | 0.808 |

### Test metrics (recovered protocol)

| Metric | Value |
|--------|-------|
| Accuracy | **67.28%** |
| Precision | 62.96% |
| Recall | 83.95% |
| F1 | 0.720 |
| ROC-AUC | 0.787 |
| TAR@FAR=0.001 | 0.00% |
| TAR@FAR=0.01 | 14.81% |
| TAR@FAR=0.1 | 39.51% |

### Comparison with Run 18

| Metric | Run 18 | Run 19 |
|--------|--------|--------|
| Accuracy | 66.05% | **67.28%** |
| F1 | 0.658 | **0.720** |
| ROC-AUC | 0.749 | **0.787** |

### Comparison with previous best local ViT run

| Metric | Run 09 | Run 19 |
|--------|--------|--------|
| Accuracy | 66.05% | **67.28%** |
| F1 | 0.715 | **0.720** |
| ROC-AUC | 0.767 | **0.787** |

Run 19 is the first run in this later batch that clearly beats the earlier local best from run 09 on all three headline metrics.

---

## Analysis

### What improved

Lowering the margin worked. It kept the strong ranking signal from run 18 while restoring much more recall, which pushed both F1 and ROC-AUC up.

### Why this matters

This is the first run since the early experiments that looks like a genuine step forward rather than just a different tradeoff. It is still below the README benchmark, but it is now the strongest local result in the repo.

### Remaining gap

The model still trails the README CNN+FaCoR target (`76.5% acc / 0.75 F1 / 0.83 AUC`). The remaining gap looks mostly like non-kin rejection: recall is very strong here, but precision is still not high enough.

---

## Next Step

Now that the supervised contrastive direction works, test whether the remaining error comes from excess head capacity:

1. keep margin `0.3`
2. simplify the head from 2 layers to 1
3. raise dropout back to 0.2
