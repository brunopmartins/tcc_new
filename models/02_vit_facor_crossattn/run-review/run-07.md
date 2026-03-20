# Run 07 — Local / GPU / KinFaceW-I / 100 epochs / AUC monitoring — EARLY STOPPED (LR too high)

**Date:** 2026-03-01
**GPU:** AMD Radeon RX 6750 XT (11.98 GB VRAM, gfx1031, ROCm 5.7)
**Status:** EARLY STOPPED at epoch 16 — LR=1e-4 destroys pretrained ViT features

---

## Context

First full training run with the AUC-based early stopping fix in place. The pipeline ran all 100 intended epochs but stopped at epoch 16 (patience=15) because Val AUC peaked at epoch 1 and degraded monotonically as the learning rate ramped up through the warmup phase.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Method | Local (`AMD/run_pipeline.sh` + `.venv`) |
| Device | GPU — cuda:0 (gfx1031, ROCm 5.7, PyTorch 2.3.1+rocm5.7) |
| Dataset | KinFaceW-I |
| Split | 70/15/15 → 746 train / 158 val / 162 test |
| Epochs | 16/100 (early stopped) |
| Batch size | 32 |
| Learning rate | **1e-4** (cosine warmup from 2e-5 over 5 epochs) |
| Temperature | **0.07** |
| ViT model | vit_base_patch16_224 |
| Cross-attn layers | 2 |
| Cross-attn heads | 8 |
| Loss | cosine_contrastive |
| Freeze ViT | No |
| AMP | Enabled (ROCm AMP) |

---

## Training Progress

| Epoch | Train Loss | Val Acc | Val AUC | LR | Notes |
|-------|-----------|---------|---------|-----|-------|
| 1 | 3.9972 | 0.5000 | **0.8721** | 2.00e-05 | Best — saved |
| 2 | 2.6383 | 0.5000 | 0.8527 | 4.00e-05 | |
| 3 | 0.9041 | 0.5000 | 0.8415 | 6.00e-05 | |
| 4 | 0.3985 | 0.5000 | 0.7978 | 8.00e-05 | ← AUC drops sharply as LR hits 8e-5 |
| 5 | 0.3406 | 0.5000 | 0.7858 | 1.00e-04 | ← Full LR reached, warmup ends |
| 6 | 0.3256 | 0.5000 | 0.7898 | 1.00e-04 | |
| 7 | 0.2774 | 0.5000 | 0.7555 | 9.99e-05 | |
| 8 | 0.2580 | 0.5000 | 0.7733 | 9.98e-05 | |
| 9 | 0.2468 | 0.5000 | 0.7324 | 9.96e-05 | |
| 10 | 0.2730 | 0.5000 | 0.6909 | 9.93e-05 | |
| 11 | 0.2423 | 0.5000 | 0.7133 | 9.90e-05 | |
| 12 | 0.1660 | 0.5000 | 0.7478 | 9.87e-05 | |
| 13 | 0.1485 | 0.5000 | 0.6834 | 9.83e-05 | |
| 14 | 0.1669 | 0.5000 | 0.6576 | 9.78e-05 | |
| 15 | 0.1503 | 0.5000 | 0.7210 | 9.73e-05 | |
| 16 | 0.1813 | 0.5000 | 0.7058 | 9.67e-05 | Early stop (patience=15) |

Val Acc is always 0.5000 — the model still predicts everything as kin (threshold=0.5, all similarities above 0.5). AUC is the only meaningful signal.

---

## Results

### test.py (best checkpoint = epoch 1)

| Metric | Value |
|--------|-------|
| Accuracy | 58.64% |
| Precision | 55.30% |
| Recall | 90.12% |
| F1 | 0.685 |
| ROC-AUC | 0.726 |
| Threshold | 0.900 |

### evaluate.py (best checkpoint = epoch 1)

| Metric | Value |
|--------|-------|
| Accuracy | 57.41% |
| Precision | 54.48% |
| Recall | 90.12% |
| F1 | 0.679 |
| ROC-AUC | 0.735 |
| Threshold | 0.900 |

---

## Root Cause

**LR=1e-4 is too high for ViT fine-tuning.** The pretrained `vit_base_patch16_224` backbone encodes powerful ImageNet-21K features at initialization. With LR=1e-4 and temperature=0.07 (very aggressive contrastive loss), the cosine contrastive loss reshapes the embedding space faster than the projection head and cross-attention layers can learn useful kinship structure:

1. **Epoch 1 (LR=2e-05, warmup):** ViT features largely intact → AUC=0.872 from pretrained representations alone.
2. **Epochs 2–4 (LR ramps 4e-5 → 8e-5):** Loss drops from 4.0 → 1.0 → 0.4, but AUC drops from 0.872 → 0.798. The loss optimizing on only 746 training samples causes overfitting of the contrastive pairs, destroying generalizable features.
3. **Epoch 5+ (LR=1e-04, full):** Loss plateaus (0.33–0.18) as the model memorizes the small training set. AUC oscillates between 0.66–0.75, never recovering.

**Temperature=0.07 compounds this** — it produces very sharp probability distributions in the contrastive loss (InfoNCE), making gradients aggressive and the update step large relative to the feature space.

### Secondary issue
The `best.pt` checkpoint saved at epoch 1 (the warmup epoch) reflects essentially zero training — the model is just the pretrained ViT + randomly initialized projection/cross-attention layers. This explains why test accuracy (58.64%) is barely above the 54.94% from 1-epoch runs.

---

## Fix Applied

Changed defaults in `AMD/run_pipeline.sh` and `AMD/train.py`:

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| Learning rate | 1e-4 | **1e-5** | Standard ViT fine-tuning range (1e-5 to 5e-5) |
| Temperature | 0.07 | **0.3** | Less aggressive contrastive loss; larger margin allows gradual feature adaptation |

With LR=1e-5, the LR ramp during warmup stays in the range 2e-6 to 1e-5 — slow enough for pretrained ViT features to survive while the projection head and cross-attention layers converge first.

---

## Comparison with README Targets

| Metric | README Target | Run 07 (best ckpt) |
|--------|--------------|-------------------|
| Accuracy | 79.2% | 58.64% |
| F1 | 0.78 | 0.685 |
| AUC | 0.86 | 0.726 |

Gap remains large. Best checkpoint from epoch 1 provides no trained model — results reflect pretrained ViT features only. A full convergent training run is needed.

---

## Lessons

- ViT fine-tuning requires LR ≤ 1e-5; using 1e-4 (standard for CNNs) wipes pretrained features within a few epochs.
- With small datasets (746 training samples), aggressive contrastive temperatures (0.07) overfit quickly and destroy generalizable structure.
- Saving the best model by AUC is correct, but when AUC peaks at epoch 1 due to purely pretrained features, the "best" checkpoint is essentially an untrained model.
- The README explicitly recommends freezing ViT first; this should be the default strategy.
