# Run 18 - Local / GPU / KinFaceW-I / Manual stop during epoch 5 / First supervised contrastive run

**Date:** 2026-03-20
**GPU:** AMD Radeon RX 6750 XT (11.98 GB VRAM, gfx1031, ROCm 5.7)
**Status:** MANUALLY STOPPED after epoch 4 - best checkpoint recovered manually

---

## Context

Runs 16 and 17 showed that fixing the BCE head was not enough. The bigger discovery was that the old `cosine_contrastive` path ignored labels entirely. Run 18 was the first run after replacing that with a genuinely supervised contrastive loss.

### Changes implemented since Run 17

1. **Switched loss from `bce` to supervised `contrastive`** so positive and negative pairs finally affected the loss differently.
2. **Disabled the classifier head again** because this experiment returned to metric learning rather than BCE scoring.
3. **Kept the 2-layer head and dropout 0.1** from run 17 because the goal was to isolate the loss-function fix.
4. **Kept frozen ViT, LR=1e-5, 4:1 train negatives, and 1:1 evaluation** to preserve the stable training setup.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Method | Local (`AMD/run_pipeline.sh` + `.venv`) |
| Device | GPU - cuda:0 (gfx1031, ROCm 5.7, PyTorch 2.3.1+rocm5.7) |
| Dataset | KinFaceW-I |
| Split | 70/15/15 -> 1865 train pairs / 158 val / 162 test |
| Epochs | 4 completed / 100 planned |
| Batch size | 32 |
| Learning rate | 1e-5 |
| Scheduler | none (warmup 1 epoch) |
| Loss | contrastive |
| Margin | 0.5 |
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
| 1 | 0.0193 | 0.5000 | 0.7089 | 1.00e-05 | Stronger start than BCE |
| 2 | 0.0184 | 0.5000 | 0.7242 | 1.00e-05 | Clear improvement |
| 3 | 0.0186 | 0.5000 | 0.7340 | 1.00e-05 | Still climbing |
| **4** | **0.0178** | **0.5000** | **0.7454** | **1.00e-05** | **Best checkpoint saved** |

The run was stopped early because it had already shown the key result: the supervised contrastive fix was materially better than the repaired BCE branch. The saved checkpoint was then evaluated with the shared validation-threshold protocol.

---

## Results

### Validation-selected threshold

| Metric | Value |
|--------|-------|
| Threshold | **0.850** |
| Val Accuracy | 77.22% |
| Val F1 | 0.775 |
| Val ROC-AUC | 0.840 |

### Test metrics (recovered protocol)

| Metric | Value |
|--------|-------|
| Accuracy | **66.05%** |
| Precision | 66.25% |
| Recall | 65.43% |
| F1 | 0.658 |
| ROC-AUC | 0.749 |
| TAR@FAR=0.001 | 0.00% |
| TAR@FAR=0.01 | 6.17% |
| TAR@FAR=0.1 | 32.10% |

### Comparison with Run 17

| Metric | Run 17 | Run 18 |
|--------|--------|--------|
| Accuracy | 50.00% | **66.05%** |
| F1 | 0.667 | 0.658 |
| ROC-AUC | 0.665 | **0.749** |

---

## Analysis

### What improved

This was the first run in the batch that clearly escaped the all-kin collapse. Accuracy jumped back into the same range as the older better runs, and ROC-AUC improved sharply.

### What still failed

The run became more balanced, but not yet better than run 09 overall. It matched run 09 on accuracy range, but still trailed it on F1 and AUC.

### Likely root cause

The supervised loss fix was the right direction, but the margin was probably still too strict for this frozen-backbone setup. The model gained structure, yet it still left recall and ranking headroom on the table.

---

## Next Step

Keep the new supervised contrastive path and tune the margin:

1. lower the margin from `0.5` to `0.3`
2. keep the same 2-layer head
3. keep the same frozen-backbone training recipe
