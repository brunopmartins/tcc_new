# Run 17 - Local / GPU / KinFaceW-I / Manual stop during epoch 2 / BCE classifier head with more capacity

**Date:** 2026-03-20
**GPU:** AMD Radeon RX 6750 XT (11.98 GB VRAM, gfx1031, ROCm 5.7)
**Status:** MANUALLY STOPPED during epoch 2 - best checkpoint recovered manually

---

## Context

Run 16 showed that simply fixing the BCE classifier path was not enough. The next hypothesis was that the repaired classifier head had been made too small, so run 17 restored the larger trainable head while keeping the new BCE path active.

### Changes implemented since Run 16

1. **Raised cross-attention depth from 1 layer back to 2** to restore head capacity.
2. **Reduced dropout from 0.2 to 0.1** so the larger head could fit the repaired BCE objective more easily.
3. **Kept the BCE classifier head enabled** because run 16 confirmed that path needed to stay fixed.
4. **Kept frozen ViT, LR=1e-5, 4:1 train negatives, and 1:1 evaluation** so only head capacity changed.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Method | Local (`AMD/run_pipeline.sh` + `.venv`) |
| Device | GPU - cuda:0 (gfx1031, ROCm 5.7, PyTorch 2.3.1+rocm5.7) |
| Dataset | KinFaceW-I |
| Split | 70/15/15 -> 1865 train pairs / 158 val / 162 test |
| Epochs | 1 completed / 100 planned |
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
| Dropout | 0.1 |
| Freeze ViT | Yes |
| Classifier head | Yes |
| Trainable params | 15.43M |
| AMP | Enabled (ROCm AMP) |

---

## Training Progress

| Epoch | Train Loss | Val Acc | Val AUC | LR | Notes |
|-------|-----------|---------|---------|-----|-------|
| **1** | **0.6683** | **0.5000** | **0.5690** | **1.00e-05** | **Best checkpoint saved** |

The run was stopped in epoch 2 because the repaired-BCE direction was still visibly weak. The best checkpoint from epoch 1 was recovered and evaluated with the shared validation-threshold protocol.

---

## Results

### Validation-selected threshold

| Metric | Value |
|--------|-------|
| Threshold | **0.100** |
| Val Accuracy | 50.00% |
| Val F1 | 0.667 |
| Val ROC-AUC | 0.637 |

### Test metrics (recovered protocol)

| Metric | Value |
|--------|-------|
| Accuracy | **50.00%** |
| Precision | 50.00% |
| Recall | 100.00% |
| F1 | 0.667 |
| ROC-AUC | 0.665 |
| TAR@FAR=0.001 | 0.00% |
| TAR@FAR=0.01 | 1.23% |
| TAR@FAR=0.1 | 29.63% |

### Comparison with Run 16

| Metric | Run 16 | Run 17 |
|--------|--------|--------|
| Accuracy | 50.00% | 50.00% |
| F1 | 0.667 | 0.667 |
| ROC-AUC | 0.542 | **0.665** |

---

## Analysis

### What improved

The larger head did recover some ranking signal. ROC-AUC improved substantially over run 16.

### What still failed

The operating point was still completely degenerate. Validation again selected a threshold of `0.10`, and the model still behaved like an all-kin classifier at test time.

### Likely root cause

At this point the core issue no longer looked like classifier capacity. The more important problem was that the current supervision strategy still was not teaching non-kin separation in a stable, label-aware way.

---

## Next Step

Stop spending budget on the repaired BCE direction and switch to a truly label-aware metric-learning loss:

1. replace BCE with supervised contrastive loss
2. keep the stable frozen-backbone setup
3. keep 4:1 train negatives and 1:1 evaluation
