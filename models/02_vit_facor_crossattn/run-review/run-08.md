# Run 08 — Local / GPU / KinFaceW-I / 100 epochs / LR=1e-5 / Temp=0.3 — EARLY STOPPED (patience too short)

**Date:** 2026-03-01
**GPU:** AMD Radeon RX 6750 XT (11.98 GB VRAM, gfx1031, ROCm 5.7)
**Status:** EARLY STOPPED at epoch 16 — patience=15 too short for slow ViT fine-tuning at LR=1e-5

---

## Context

First run with corrected LR (1e-5) and temperature (0.3). These fixed the feature destruction observed in Run 07. However, patience=15 (shared config default) still fired too early — with LR=1e-5 the model needs ~50-80 epochs to converge, and loss was still actively decreasing at epoch 16.

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
| Learning rate | **1e-5** |
| Temperature | **0.3** |
| Patience | 15 (shared default — too short) |
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
| 1 | 4.049 | 0.5000 | **0.8252** | 2.00e-06 | Best — saved |
| 2 | 3.940 | 0.5000 | 0.8229 | 4.00e-06 | |
| 3 | 3.822 | 0.5000 | 0.7722 | 6.00e-06 | |
| 4 | 3.612 | 0.5000 | 0.8122 | 8.00e-06 | |
| 5 | 3.254 | 0.5000 | 0.8052 | 1.00e-05 | ← Full LR reached |
| 6 | 2.830 | 0.5000 | 0.8135 | 1.00e-05 | |
| 7 | 2.530 | 0.5000 | 0.7806 | 9.99e-06 | |
| 8 | 2.317 | 0.5000 | 0.7757 | 9.98e-06 | |
| 9 | 2.226 | 0.5000 | 0.8020 | 9.96e-06 | |
| 10 | 2.153 | 0.5000 | 0.7834 | 9.93e-06 | |
| 11 | 2.110 | 0.5000 | 0.7790 | 9.90e-06 | |
| 12 | 2.075 | 0.5000 | 0.7875 | 9.87e-06 | |
| 13 | 2.048 | 0.5000 | 0.7984 | 9.83e-06 | |
| 14 | 2.021 | 0.5000 | 0.7840 | 9.78e-06 | |
| 15 | 1.985 | 0.5000 | 0.7984 | 9.73e-06 | |
| 16 | 1.925 | 0.5000 | 0.8120 | 9.68e-06 | Early stop |

Compare with Run 07 at epoch 5: loss=0.34 (rapid collapse). Here epoch 5: loss=3.25 — the LR fix is working. The model is learning gradually as intended, but hasn't converged yet.

---

## Results (best checkpoint = epoch 1 — zero real training)

### test.py

| Metric | Value |
|--------|-------|
| Accuracy | 56.79% |
| Precision | 53.79% |
| Recall | 96.30% |
| F1 | 0.690 |
| ROC-AUC | 0.650 |
| Threshold | 0.800 |

### evaluate.py

| Metric | Value |
|--------|-------|
| Accuracy | 59.88% |
| Precision | 55.71% |
| Recall | 96.30% |
| F1 | 0.706 |
| ROC-AUC | 0.724 |
| Threshold | 0.800 |

---

## Root Cause

**patience=15 with LR=1e-5 is structurally mismatched.** Val AUC at epoch 1 (0.8252) reflects pretrained ViT features with zero training. Loss at epoch 16 (1.925, steadily falling from 4.049) shows the model is still mid-training — not converged, not even close. With 24 batches per epoch and LR=1e-5, convergence requires approximately 50-80 epochs minimum.

The AUC oscillates in the 0.77-0.82 range throughout. Once the embedding space begins to separate (loss < ~1.0), AUC should improve consistently. That inflection point was not reached before early stopping.

---

## Fix Applied

Added `--patience` argument to `AMD/train.py` (default 50) and `PATIENCE` variable to `run_pipeline.sh` (default 50). Also exposed as an env-var override: `PATIENCE=80 bash AMD/run_pipeline.sh`.

With patience=50, early stopping will only fire if AUC doesn't improve over any 50-epoch window — giving the model enough time to work through the slow convergence phase at LR=1e-5.

---

## Comparison with README Targets

| Metric | README Target | Run 08 |
|--------|--------------|--------|
| Accuracy | 79.2% | 56.79% |
| F1 | 0.78 | 0.690 |
| AUC | 0.86 | 0.724 |

No trained model has been produced yet — all best checkpoints are from epoch 1. Run 09 (patience=50) is expected to be the first run where the model trains past the pretrained ViT baseline.
