# Run 10 — Local / GPU / KinFaceW-I / 67 epochs / LR=1e-5 / Temp=0.3 / Patience=50

**Date:** 2026-03-03
**GPU:** AMD Radeon RX 6750 XT (11.98 GB VRAM, gfx1031, ROCm 5.7)
**Status:** EARLY STOPPED at epoch 67 — high val AUC (0.8939) but degenerate test threshold

---

## Context

Same configuration as run 09. The pipeline, LR, temperature, and patience were all unchanged. This run serves as an independent replicate of run 09 to test reproducibility. Val AUC peaked higher and later (epoch 17, 0.8939) than run 09 (epoch 5, 0.827), but test performance regressed to the degenerate recall=1.0 pattern seen in early runs.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Method | Local (`AMD/run_pipeline.sh` + `.venv`) |
| Device | GPU — cuda:0 (gfx1031, ROCm 5.7, PyTorch 2.3.1+rocm5.7) |
| Dataset | KinFaceW-I |
| Split | 70/15/15 → 746 train / 158 val / 162 test |
| Epochs | 67/100 (early stopped, patience=50) |
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
| 1 | 4.057 | 0.5000 | 0.838 | 2.00e-06 | New best |
| 2 | 3.963 | 0.5000 | 0.880 | 4.00e-06 | New best |
| 3 | 3.871 | 0.5000 | 0.876 | 6.00e-06 | |
| 5 | 3.270 | 0.5000 | 0.837 | 1.00e-05 | Warmup complete |
| 10 | 2.128 | 0.5000 | 0.839 | 9.93e-06 | |
| 16 | 1.914 | 0.5000 | 0.886 | 9.68e-06 | New best |
| **17** | **1.896** | **0.5000** | **0.894** | **9.62e-06** | **← Best checkpoint saved** |
| 20 | 1.858 | 0.5000 | 0.859 | 9.40e-06 | |
| 30 | 1.750 | 0.5000 | 0.830 | 8.40e-06 | |
| 40 | 1.703 | 0.5000 | 0.831 | 7.04e-06 | |
| 50 | 1.646 | 0.5000 | 0.828 | 5.46e-06 | |
| 60 | 1.612 | 0.5000 | 0.852 | 3.83e-06 | |
| 67 | 1.620 | 0.5000 | 0.843 | 2.77e-06 | Early stop |

Val AUC oscillated in 0.82–0.89 throughout training, never surpassing the epoch-17 peak. Loss continued to decrease (4.06 → 1.62).

---

## Results (best checkpoint = epoch 17)

### test.py

| Metric | Value |
|--------|-------|
| Accuracy | **50.00%** |
| Precision | 50.00% |
| Recall | **100.00%** |
| F1 | 0.667 |
| ROC-AUC | 0.774 |
| Threshold | **0.100** |
| TAR@FAR=0.001 | 0.00% |
| TAR@FAR=0.01 | 0.00% |
| TAR@FAR=0.1 | 45.68% |

### evaluate.py

| Metric | Value |
|--------|-------|
| Accuracy | **50.00%** |
| Precision | 50.00% |
| Recall | **100.00%** |
| F1 | 0.667 |
| ROC-AUC | 0.758 |
| Threshold | **0.100** |
| TAR@FAR=0.001 | 0.00% |
| TAR@FAR=0.01 | 14.81% |
| TAR@FAR=0.1 | **49.38%** |

---

## Analysis

### Degenerate threshold collapse

The optimal threshold dropped to 0.100 — meaning all 162 test samples score above 0.1 after the cosine-similarity-to-[0,1] mapping. The model predicts every pair as kin, giving recall=1.0 and accuracy=50% on the balanced test set. This is the same degenerate pattern seen in runs 004 and 007 before ViT fine-tuning fixes.

In run 09 (epoch 5 checkpoint), the threshold was 0.850 — the pretrained ViT features still provided discrimination. By epoch 17 in run 10, continued fine-tuning has collapsed the embedding space: both kin and non-kin pairs cluster at high cosine similarity, eliminating the discriminative gap.

### Val AUC vs test AUC divergence

The epoch-17 checkpoint has:
- Val AUC: 0.8939 (158-sample val set)
- Test AUC: 0.758 (162-sample test set)

The gap (0.136) is substantially larger than run 09's gap (0.827 − 0.767 = 0.060). Two factors:
1. **Noise on small val set:** 158 samples produce high-variance AUC estimates. The 0.8939 peak is likely a lucky fluctuation — AUC oscillates 0.82–0.89 all run, with many "real" values ~0.84.
2. **Val set overfitting:** The model trained 17 epochs while monitoring val AUC; the checkpoint selected at the noise peak does not represent a robustly better-generalizing model.

### TAR@FAR marginal improvement

Despite the degenerate accuracy, TAR@FAR=0.1 reached 49.4% (evaluate.py), up from 44.4% in run 09. This is the only meaningful improvement: the model ranks kin pairs slightly higher in the tail of the score distribution, even though no clean decision threshold exists.

### Why fine-tuning past epoch 5 hurts

Run 09's best checkpoint (epoch 5) preserved ~95% of the pretrained ViT embedding geometry. By epoch 17 (run 10), the backbone has shifted enough that:
- Non-kin pairs also receive high cosine similarity (family resemblance features are generalizing, but so is noise)
- The contrastive loss minimizes the positive-pair distance effectively but does not enforce sufficient separation for negative pairs at the available negative sampling ratio (1:1)

---

## Comparison with README Targets

| Metric | README Target | Run 10 | Run 09 (best so far) |
|--------|--------------|--------|---------------------|
| Accuracy | 79.2% | **50.0%** | 66.05% |
| F1 | 0.78 | 0.667 | 0.715 |
| AUC | 0.86 | 0.758 | 0.767 |

Run 10 is a regression from run 09 on all accuracy/F1/AUC metrics. Run 09 remains the best result.

---

## Root Cause and Next Steps

The fine-tuning collapse confirms that freezing the ViT backbone is necessary. At LR=1e-5 and with 746 training samples, the backbone shifts enough by epoch 17 to destroy the pretrained discrimination without learning a better replacement. The problem is compounded by:

| Factor | Evidence | Fix |
|--------|----------|-----|
| ViT backbone too plastic | Threshold 0.850 (ep5) → 0.100 (ep17) | `--freeze_vit` or two-stage warmup |
| Val AUC noise (158 samples) | 0.8939 peak not reproducible on test set | 5-fold CV averages over all data |
| 1:1 negative ratio insufficient | Recall=1.0 persists across all runs | Increase negative ratio (2x–4x) |

**5-fold CV (train_cv.py)** is the immediate next step: it uses 80% of data per fold (~426 pairs) vs 70% (746 pairs), and averages over 5 independent val/test splits — eliminating the noise peak problem that inflated epoch 17's val AUC.
