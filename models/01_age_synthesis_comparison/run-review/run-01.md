# Run 01 — Docker / GPU / KinFaceW-I+II (split bug)

**Date:** 2026-02-18
**GPU:** AMD Radeon RX 6750 XT (11.98 GB VRAM)

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Method | Docker (`docker-compose.amd.yml`) |
| Device | GPU — cuda:0 (gfx1031, ROCm 5.7, PyTorch 2.0.1+gita61a294) |
| Dataset | KinFaceW-I + KinFaceW-II (combined, ~930 pairs) |
| Split | **Bug: all splits loaded the same data** — no true held-out test |
| Epochs | 50 of 100 (no early stopping triggered) |
| Batch size | 16 |
| Learning rate | 1e-4 (cosine warmup) |
| Weight decay | 1e-5 |
| Backbone | ResNet-50 |
| Age synthesis | Disabled |
| AMP | Enabled (ROCm AMP) |
| Loss | BCE |
| Training speed | ~11 s/epoch |
| Total time | ~9.5 min |

---

## Training Curve

| Epoch | Train Loss | Val Acc | Val AUC | LR |
|-------|-----------|---------|---------|-----|
| 1 | 0.6937 | 56.82% | 0.5453 | 2.00e-05 |
| 5 | 0.6679 | 50.97% | 0.6029 | 1.00e-04 |
| 8 | 0.6148 | 68.61% | 0.7293 | 9.89e-05 |
| 10 | 0.5953 | 70.67% | 0.7177 | 9.70e-05 |
| 15 | 0.5313 | 73.27% | 0.7700 | 8.83e-05 |
| 20 | 0.4973 | 74.68% | 0.7887 | 7.50e-05 |
| 25 | 0.4501 | 75.32% | 0.7925 | 5.87e-05 |
| 30 | 0.4132 | 75.65% | 0.8095 | 4.14e-05 |
| 35 | 0.3831 | 76.62% | 0.8043 | 2.51e-05 |
| **38** | **0.4123** | **76.84%** | **0.8157** | 1.66e-05 |
| 50 | 0.3947 | 76.52% | 0.8211 | 1.00e-07 |

Best checkpoint at epoch 38 (Val Acc 76.84%). Loss dropped 43% (0.694 → 0.395).

---

## Results

| Source | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | Samples |
|--------|-----------|----------|-----------|--------|-----|---------|---------|
| train.py (end-of-training) | 0.50 | 75.70% | 70.96% | 98.12% | 82.36% | 79.88% | ~922 |
| test.py (optimal threshold) | 0.60 | 74.25% | 70.17% | 95.31% | 80.83% | 74.96% | 936 |
| evaluate.py | 0.50 | 72.22% | 68.21% | 96.99% | 80.09% | 77.29% | 925 |

### Per-Relation (test.py, threshold=0.60)

| Relation | Accuracy | F1 | Samples |
|----------|---------|-----|---------|
| fd — father/daughter | 96.27% | 98.10% | 134 |
| fs — father/son | 96.15% | 98.04% | 156 |
| md — mother/daughter | 93.70% | 96.75% | 127 |
| ms — mother/son | 94.83% | 97.35% | 116 |
| negative (non-kin) | **46.40%** | 0.00% | 403 |

### TAR @ FAR

| FAR | TAR |
|-----|-----|
| 0.001 | 0.00% |
| 0.01 | 2.81% |
| 0.1 | 18.57% |

---

## Flaws

1. **Dataset split bug.** All three splits (train/val/test) loaded the same data. Validation and test metrics were computed on training samples. Results are optimistic and unreliable — the model was partially graded on its own training set.

2. **Extreme recall bias.** Recall=95–98% with non-kin accuracy of only 46.4%. The model predicts "kin" for almost every pair. The threshold of 0.60 was needed just to bring it down from worse.

3. **Age synthesis disabled.** The core contribution of the model (all-vs-all cross-age comparison) was never activated. The 9-comparison matrix degenerates to a single comparison.

4. **KinFaceW-I and KinFaceW-II combined without protocol.** The standard benchmark evaluates each dataset separately. Combining them inflates the sample count but violates evaluation protocol.

5. **TAR@FAR near zero.** The model cannot operate at any realistic biometric operating point.

6. **Docker-only.** Not reproducible without Docker and the specific ROCm image (`rocm/pytorch:rocm5.7_ubuntu22.04_py3.10_pytorch_2.0.1`).
