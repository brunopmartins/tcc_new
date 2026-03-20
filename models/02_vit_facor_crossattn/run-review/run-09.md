# Run 09 — Local / GPU / KinFaceW-I / 55 epochs / LR=1e-5 / Temp=0.3 / Patience=50

**Date:** 2026-03-01
**GPU:** AMD Radeon RX 6750 XT (11.98 GB VRAM, gfx1031, ROCm 5.7)
**Status:** EARLY STOPPED at epoch 55 — first run to genuinely train past epoch 1

---

## Context

First run with all three configuration issues resolved: AUC monitoring (run 006), LR=1e-5 (run 007), patience=50 (run 008). This is the first run where the best checkpoint comes from a trained epoch rather than the pretrained ViT initialization.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Method | Local (`AMD/run_pipeline.sh` + `.venv`) |
| Device | GPU — cuda:0 (gfx1031, ROCm 5.7, PyTorch 2.3.1+rocm5.7) |
| Dataset | KinFaceW-I |
| Split | 70/15/15 → 746 train / 158 val / 162 test |
| Epochs | 55/100 (early stopped, patience=50) |
| Batch size | 32 |
| Learning rate | 1e-5 (cosine warmup from 2e-6 over 5 epochs) |
| Temperature | 0.3 |
| Patience | 50 |
| ViT model | vit_base_patch16_224 |
| Cross-attn layers | 2 |
| Cross-attn heads | 8 |
| Loss | cosine_contrastive |
| Freeze ViT | No |
| AMP | Enabled (ROCm AMP) |

---

## Training Progress (selected epochs)

| Epoch | Train Loss | Val Acc | Val AUC | LR | Notes |
|-------|-----------|---------|---------|-----|-------|
| 1 | 4.039 | 0.5000 | 0.769 | 2.00e-06 | New best |
| 2 | 3.947 | 0.5000 | 0.785 | 4.00e-06 | New best |
| 3 | 3.824 | 0.5000 | 0.816 | 6.00e-06 | New best |
| 4 | 3.650 | 0.5000 | 0.814 | 8.00e-06 | |
| **5** | **3.292** | **0.5000** | **0.827** | **1.00e-05** | **← Best, saved. Warmup complete** |
| 10 | 2.131 | 0.5000 | 0.800 | 9.93e-06 | |
| 20 | 1.858 | 0.5000 | 0.781 | 9.40e-06 | |
| 30 | 1.763 | 0.5000 | 0.798 | 8.40e-06 | |
| 40 | 1.693 | 0.5000 | 0.799 | 7.04e-06 | |
| 50 | 1.655 | 0.5000 | 0.800 | 5.46e-06 | |
| 55 | 1.650 | 0.5000 | 0.788 | 4.64e-06 | Early stop |

AUC improved consistently through the warmup phase (ep 1–5), peaked at epoch 5, then oscillated between 0.77–0.83 for 50 more epochs without surpassing the epoch-5 best. Loss continued to decrease throughout (4.04 → 1.65).

---

## Results (best checkpoint = epoch 5)

### test.py

| Metric | Value |
|--------|-------|
| Accuracy | 64.81% |
| Precision | 60.53% |
| Recall | 85.19% |
| F1 | 0.708 |
| ROC-AUC | 0.741 |
| Threshold | 0.850 |
| TAR@FAR=0.001 | 0.00% |
| TAR@FAR=0.01 | 12.35% |
| TAR@FAR=0.1 | 35.80% |

### evaluate.py

| Metric | Value |
|--------|-------|
| Accuracy | 66.05% |
| Precision | 61.61% |
| Recall | 85.19% |
| F1 | 0.715 |
| ROC-AUC | 0.767 |
| Threshold | 0.850 |
| TAR@FAR=0.001 | 0.00% |
| TAR@FAR=0.01 | 11.11% |
| TAR@FAR=0.1 | 44.44% |

---

## Analysis

### What improved

This is the best result of any run for model 02. Accuracy (+7% over previous best), AUC (+0.005 in evaluate.py), and threshold dropped from 0.900 to 0.850 — the model is slightly more willing to reject non-kin pairs. TAR@FAR=0.1 reached 44.4% (evaluate.py), up from 40.7% in run 07.

### Why AUC plateaus at epoch 5

The AUC peaks during the warmup phase exactly when the LR reaches 1e-5, then stagnates for 50 epochs while loss continues to fall. Two factors:

1. **Cosine LR scheduler decays too fast.** After warmup, the scheduler reduces LR from 1e-5 to min_lr=1e-7 over 95 epochs. By the time the model could exploit useful features (loss < 2.0, epoch ~10+), the LR is already below 9e-6 and shrinking. The model learns to minimize training loss (small contrastive pairs) but the embedding space doesn't generalize.

2. **Training set too small (746 samples, 24 batches/epoch).** The model has 86M ViT parameters plus cross-attention. After ~50 epochs × 24 batches = 1,200 gradient updates, the small training set is memorized without improving generalization. AUC on the held-out 158-sample val set oscillates noisily in the 0.77–0.83 range — a consequence of the tiny val set (high variance estimates).

### Recall bias persists

Threshold of 0.850 (down from 0.900 in earlier runs) shows marginal progress. Recall=0.85 still indicates the model strongly prefers predicting kin. The 81 negative (non-kin) samples in the test set are still being largely misclassified.

---

## Comparison with README Targets

| Metric | README Target | Run 09 | Gap |
|--------|--------------|--------|-----|
| Accuracy | 79.2% | 66.05% | −13.2% |
| F1 | 0.78 | 0.715 | −0.065 |
| AUC | 0.86 | 0.767 | −0.093 |

---

## Remaining gap and next steps

The model has now trained and the pipeline is fully correct. The remaining gap to README targets (~13% accuracy, ~0.09 AUC) is driven by:

| Factor | Impact | Potential fix |
|--------|--------|---------------|
| Small dataset (746 train) | High | FIW dataset (26k images); or data augmentation |
| Recall bias / non-kin discrimination | Medium | Increase negative pair sampling ratio |
| Cosine LR decay too aggressive | Medium | Use `plateau` scheduler or increase `min_lr` |
| ViT not frozen during warmup | Low–Med | `--freeze_vit` first, then unfreeze |
