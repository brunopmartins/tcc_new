# Run 22 - Local / GPU / KinFaceW-I / Manual stop during epoch 10 / Higher negative ratio on relation-matched branch

**Date:** 2026-03-20
**GPU:** AMD Radeon RX 6750 XT (11.98 GB VRAM, gfx1031, ROCm 5.7)
**Status:** MANUALLY STOPPED during epoch 10 - best checkpoint recovered manually

---

## Context

Run 21 showed that relation-matched negatives were a real improvement, but the model still failed to reject enough non-kin pairs. The next experiment kept the same structure and increased the amount of negative pressure.

### Changes implemented since Run 21

1. **Increased training negative ratio from `4:1` to `6:1`** to push non-kin separation harder.
2. **Kept relation-matched training negatives** because run 21 showed that the strategy itself was helpful.
3. **Kept evaluation negatives random at 1:1** so test numbers stayed comparable with the earlier runs.
4. **Kept the rest of the run 21 recipe fixed**: frozen ViT, LR=1e-5, 2-layer head, dropout `0.1`, contrastive loss, margin `0.3`.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Method | Local (`AMD/run_pipeline.sh` + `.venv`) |
| Device | GPU - cuda:0 (gfx1031, ROCm 5.7, PyTorch 2.3.1+rocm5.7) |
| Dataset | KinFaceW-I |
| Split | 70/15/15 -> 2611 train pairs / 158 val / 162 test |
| Epochs | 9 completed / 100 planned |
| Batch size | 32 |
| Learning rate | 1e-5 |
| Scheduler | none (warmup 1 epoch) |
| Loss | contrastive |
| Margin | 0.3 |
| Train negative ratio | 6.0 |
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
| 1 | 0.0059 | 0.5000 | 0.7250 | 1.00e-05 | Better start than run 21 |
| 2 | 0.0056 | 0.5000 | 0.7427 | 1.00e-05 | Still ahead of run 21 |
| 3 | 0.0054 | 0.5000 | 0.7558 | 1.00e-05 | Stronger early curve |
| 4 | 0.0053 | 0.5000 | 0.7669 | 1.00e-05 | Continued gain |
| 5 | 0.0052 | 0.5000 | 0.7765 | 1.00e-05 | Stable improvement |
| 6 | 0.0050 | 0.5000 | 0.7829 | 1.00e-05 | Good but still below run 19 |
| 7 | 0.0049 | 0.5000 | 0.7893 | 1.00e-05 | Best before brief flattening |
| 8 | 0.0048 | 0.5000 | 0.7882 | 1.00e-05 | Small plateau |
| **9** | **0.0047** | **0.5000** | **0.7954** | **1.00e-05** | **Best checkpoint saved** |

The run was stopped in epoch 10 after epoch 9 became the new best checkpoint for this branch and the curve no longer looked likely to overtake the previous baseline quickly.

---

## Results

### Validation-selected threshold

| Metric | Value |
|--------|-------|
| Threshold | **0.900** |
| Val Accuracy | 78.48% |
| Val F1 | 0.795 |
| Val ROC-AUC | 0.860 |

### Test metrics (recovered protocol)

| Metric | Value |
|--------|-------|
| Accuracy | 67.90% |
| Precision | **64.95%** |
| Recall | 77.78% |
| F1 | 0.708 |
| ROC-AUC | 0.757 |
| TAR@FAR=0.001 | 0.00% |
| TAR@FAR=0.01 | 9.88% |
| TAR@FAR=0.1 | 29.63% |

### Comparison with Run 21

| Metric | Run 21 | Run 22 |
|--------|--------|--------|
| Accuracy | **68.52%** | 67.90% |
| Precision | 63.39% | **64.95%** |
| Recall | **87.65%** | 77.78% |
| F1 | **0.736** | 0.708 |
| ROC-AUC | **0.784** | 0.757 |

### Comparison with Run 19

| Metric | Run 19 | Run 22 |
|--------|--------|--------|
| Accuracy | 67.28% | **67.90%** |
| Precision | 62.96% | **64.95%** |
| Recall | **83.95%** | 77.78% |
| F1 | **0.720** | 0.708 |
| ROC-AUC | **0.787** | 0.757 |

---

## Analysis

### What improved

The higher negative ratio did improve rejection behavior. Test precision increased and negative-class accuracy rose to 58.02%, which is the clearest sign yet that the model can be pushed away from its all-kin bias.

### What got worse

The price was too high. Recall dropped sharply, F1 fell below run 21 and even below run 19, and ROC-AUC regressed noticeably. The branch became more conservative, but not more reliable overall.

### Main takeaway

`6:1` relation-matched negatives overcorrected the run 21 behavior. The model started rejecting more impostors, but it also rejected too many true kin pairs. That means the direction is useful, but the ratio is too aggressive in its current form.

---

## Next Step

Do **not** keep pushing the ratio upward blindly. The better direction now is to stay near run 21 and improve negative quality more selectively:

1. keep relation-matched negatives
2. return to `4:1`
3. add targeted hard negatives or a curriculum instead of globally raising the ratio
