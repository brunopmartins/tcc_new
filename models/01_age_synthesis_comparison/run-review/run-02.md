# Run 02 — Local / CPU / KinFaceW-I (proper split)

**Date:** 2026-02-18
**GPU:** AMD Radeon RX 6750 XT (available but not detected)

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Method | Local (`setup.sh` + `.venv`) |
| Device | **CPU** — ROCm detection failed (user not in `render` group) |
| Dataset | KinFaceW-I only |
| Split | Fixed: 70/15/15 → 746 train / 158 val / 162 test |
| Epochs | 37 of 100 (early stop, patience=15) |
| Batch size | 32 |
| Learning rate | 1e-4 (cosine warmup) |
| Weight decay | 1e-5 |
| Backbone | ResNet-50 |
| Age synthesis | Disabled |
| AMP | Disabled (CPU) |
| Loss | BCE |
| Training speed | ~147 s/epoch |
| Total time | ~91 min |

---

## Training Curve

| Epoch | Notes |
|-------|-------|
| 1–22 | Steady improvement |
| **22** | **Best Val Acc: 66.46%** ← checkpoint saved |
| 22–37 | No improvement for 15 epochs |
| 37 | Early stopping triggered (patience=15) |

---

## Results

| Source | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | Samples |
|--------|-----------|----------|-----------|--------|-----|---------|---------|
| train.py (end-of-training) | 0.50 | 71.60% | 67.68% | 82.72% | 74.44% | 76.68% | 162 |
| test.py (optimal threshold) | 0.55 | 68.52% | 64.15% | 83.95% | 72.73% | 68.19% | 162 |
| evaluate.py | 0.50 | 70.37% | 65.42% | 86.42% | 74.47% | 76.09% | 162 |

### Per-Relation (test.py, threshold=0.55)

| Relation | Accuracy | F1 | Samples |
|----------|---------|-----|---------|
| fd — father/daughter | 88.00% | 93.62% | 25 |
| fs — father/son | 84.00% | 91.30% | 25 |
| md — mother/daughter | 85.71% | 92.31% | 14 |
| ms — mother/son | 76.47% | 86.67% | 17 |
| negative (non-kin) | **53.09%** | 0.00% | 81 |

### TAR @ FAR

| FAR | TAR |
|-----|-----|
| 0.001 | 0.00% |
| 0.01 | 2.47% |
| 0.1 | 11.11% |

---

## Flaws

1. **GPU not used.** `torch.cuda.is_available()` returned False because user `bruno` was not in the `render` group — `/dev/kfd` was inaccessible. All training ran on CPU at 147 s/epoch (13× slower than GPU).

2. **Age synthesis disabled.** No SAM pretrained weights available; `--use_age_synthesis` not passed.

3. **Early stop at epoch 37.** Only 37 of 100 epochs ran. The model may not have converged — accuracy was still improving slowly up to epoch 22.

4. **Recall bias persists.** Non-kin accuracy of 53.09% (barely above chance). High recall (84–86%) masks weak discriminability.

5. **TAR@FAR near zero.** The model cannot operate at any realistic biometric operating point.

6. **Small test set.** 162 samples (81 kin + 81 non-kin) — high variance in metrics. Per-relation groups have as few as 14 samples (md), making accuracy estimates unreliable.

7. **`UndefinedMetricWarning` spam.** 5 warnings per epoch from per-relation AUC computation where relation groups in small batches contain only one class. Harmless but clutters output.

## Notes

This is the first honest evaluation using a proper held-out test set. The 68.52% accuracy (vs 74.25% in Run 1) is the more reliable number — the Run 1 gap was largely due to data leakage.
