# Run 23 - Local / GPU / KinFaceW-I / 20 epochs / Freeze-then-partial-unfreeze on relation-matched branch

**Date:** 2026-03-20
**GPU:** AMD Radeon RX 6750 XT (11.98 GB VRAM, gfx1031, ROCm 5.7)
**Status:** COMPLETED

---

## Context

Run 22 confirmed that simply pushing the negative ratio upward was too blunt. The next step was to keep the stronger run 21 data recipe, but add controlled ViT adaptation instead of leaving the backbone fully frozen for the whole run.

### Changes implemented since Run 22

1. **Kept the stronger relation-matched branch at `4:1` negatives** instead of the over-aggressive `6:1` from run 22.
2. **Added a freeze-then-unfreeze schedule**: start with `--freeze_vit`, then unfreeze the **last 2 ViT blocks after epoch 4**.
3. **Kept the successful run 21 core recipe fixed**: contrastive loss, margin `0.3`, 2 cross-attention layers, dropout `0.1`, eval negatives `1:1` random.
4. **Patched the AMD training pipeline** so the runner can continue run numbering after output cleanup and the trainer can change ViT trainability mid-run.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Method | Local (`AMD/run_pipeline.sh` + `.venv`) |
| Device | GPU - cuda:0 (gfx1031, ROCm 5.7, PyTorch 2.3.1+rocm5.7) |
| Dataset | KinFaceW-I |
| Split | 70/15/15 -> 1865 train pairs / 158 val / 162 test |
| Epochs | 20 / 20 |
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
| Freeze ViT | Yes, then partial unfreeze |
| Unfreeze schedule | last 2 ViT blocks after epoch 4 |
| Classifier head | No |
| Trainable params | 14.91M frozen -> 29.08M after unfreeze |
| AMP | Enabled (ROCm AMP) |

---

## Training Progress

| Epoch | Train Loss | Val Acc | Val AUC | LR | Notes |
|-------|-----------|---------|---------|-----|-------|
| 1 | 0.0078 | 0.5000 | 0.6853 | 1.00e-05 | Slow frozen start |
| 4 | 0.0069 | 0.5000 | 0.7451 | 1.00e-05 | Stable warmup branch |
| **5** | **0.0065** | **0.5000** | **0.7888** | **1.00e-05** | **First epoch after unfreezing last 2 blocks** |
| 8 | 0.0059 | 0.5000 | 0.8369 | 1.00e-05 | Clear improvement over fully frozen branch |
| 10 | 0.0055 | 0.5000 | 0.8513 | 1.00e-05 | Stronger than any previous local ViT checkpoint |
| 16 | 0.0046 | 0.5000 | 0.8611 | 1.00e-05 | Still improving late |
| **19** | **0.0043** | **0.5000** | **0.8683** | **1.00e-05** | **Best checkpoint saved** |
| 20 | 0.0042 | 0.5000 | 0.8659 | 1.00e-05 | Slight dip after best epoch |

Unlike the older unfrozen branch, this run did **not** collapse after the warmup stage. The partial-unfreeze schedule kept improving validation AUC all the way to epoch 19.

---

## Results

### Validation-selected threshold

| Metric | Value |
|--------|-------|
| Threshold | **0.900** |
| Val Accuracy | 78.48% |
| Val F1 | **0.800** |
| Val ROC-AUC | **0.868** |

### Test metrics (shared protocol)

| Metric | Value |
|--------|-------|
| Accuracy | **74.07%** |
| Precision | **71.91%** |
| Recall | 79.01% |
| F1 | **0.753** |
| ROC-AUC | **0.861** |
| TAR@FAR=0.001 | 0.00% |
| TAR@FAR=0.01 | 17.28% |
| TAR@FAR=0.1 | **65.43%** |

### Comparison with Run 21

| Metric | Run 21 | Run 23 |
|--------|--------|--------|
| Accuracy | 68.52% | **74.07%** |
| Precision | 63.39% | **71.91%** |
| Recall | **87.65%** | 79.01% |
| F1 | 0.736 | **0.753** |
| ROC-AUC | 0.784 | **0.861** |
| Negative-class accuracy | 49.38% | **69.14%** |

### Comparison with README CNN baseline

| Metric | CNN + FaCoR | Run 23 | Delta |
|--------|-------------|--------|-------|
| Accuracy | 76.5% | 74.07% | -2.43% |
| F1 | 0.75 | **0.753** | **+0.003** |
| AUC | 0.83 | **0.861** | **+0.031** |

Run 23 is the first local ViT run that beats the CNN baseline on **F1** and **AUC**, but it still falls slightly short on **accuracy**.

---

## Analysis

### What improved

The controlled unfreeze worked. Opening only the last 2 ViT blocks after the frozen warmup preserved the stable relation-matched behavior from run 21 while letting the backbone adapt enough to separate impostor pairs much better. Precision, negative-class accuracy, TAR@FAR, and ROC-AUC all jumped together.

### What still limits the run

Accuracy is still a few samples short of the README CNN target. The remaining gap is mostly non-kin rejection: test negative-class accuracy reached 69.14%, which is much better than before, but still not high enough to pull overall accuracy above 76.5%.

### Main takeaway

This is the new best ViT branch by a wide margin. The repo’s earlier conclusion that the ViT backbone was "too plastic" was true for full fine-tuning, but **not** for a delayed partial unfreeze. That means the right direction is now clear:

1. keep the freeze-then-unfreeze schedule
2. keep relation-matched negatives
3. make only small, targeted changes to negative pressure from here

---

## Next Step

Keep the run 23 recipe and increase training negatives **slightly** to `5:1` instead of jumping back to `6:1`. The aim is to push negative-class accuracy just enough to close the remaining accuracy gap without sacrificing the new AUC/F1 gains.
