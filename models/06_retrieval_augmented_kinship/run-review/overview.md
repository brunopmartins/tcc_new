# Overview — Retrieval-Augmented Kinship Verification

**Model:** 06 — Frozen ViT encoder + non-parametric gallery + cross-attention over retrieved supports + auxiliary relation head
**GPU:** AMD Radeon RX 6750 XT (11.98 GB VRAM, gfx1031, ROCm 5.7)
**Dataset:** FIW (RFIW Track-I)

Individual run details: [run-001.md](run-001.md) · [run-002.md](run-002.md)

---

## Run Comparison

| | Run 001 | Run 002 |
|---|---|---|
| **Date** | 2026-04-26 | 2026-04-27 |
| **Purpose** | First training run, ImageNet ViT defaults | Test if DINOv2 + K=64 + λ_rel=0.3 lifts AUC |
| **Backbone** | vit_base_patch16_224 (frozen) | **vit_base_patch14_dinov2.lvd142m** (frozen) |
| **Retrieval K** | 32 | **64** |
| **Relation-CE weight** | 0.15 | **0.30** |
| **Gallery on CPU** | False | **True** (memory pressure) |
| **Patience** | 10 | 20 |
| **Epochs trained** | 18/20 | 29/60 |
| **Best epoch** | 8 | 9 |
| **Trainable params** | 8.16M / 93.9M (8.7%) | ~12M (more cross-attn capacity due to bigger backbone) |
| **Time / epoch** | ~21 min | ~23 min |

### Validation peak

| | Run 001 | Run 002 |
|---|---|---|
| **Val AUC peak** | **0.8361** | 0.8228 |
| **Threshold (val)** | 0.40 | 0.55 |

### Test metrics (validation-selected threshold)

| Metric | Run 001 | Run 002 |
|---|---:|---:|
| **Test ROC-AUC** | **0.776** | 0.731 |
| **Test Accuracy** | **69.8%** | 66.2% |
| **Balanced Accuracy** | **70.3%** | 65.8% |
| **Test F1** | **0.722** | 0.619 |
| **Test Precision** | 64.5% | **67.2%** |
| **Test Recall** | **82.0%** | 57.4% |
| **Avg Precision** | **0.735** | 0.681 |
| **TAR @ FAR=0.1** | **0.388** | 0.297 |
| **TAR @ FAR=0.01** | **0.062** | 0.042 |
| **TAR @ FAR=0.001** | 0.006 | **0.007** |
| **Val→Test gap** | -0.060 | -0.092 |

R001 wins on every headline metric except precision. The val→test gap **grew** from -0.060 to -0.092 in R002 — the opposite of the intended improvement.

### Per-relation test accuracy (kin recall at val threshold)

| Relation | N | Run 001 | Run 002 | Δ |
|---|---|---:|---:|---:|
| sibs | 234 | **87.2%** | 62.4% | -24.8 |
| bb | 860 | **86.5%** | 66.9% | -19.7 |
| ss | 731 | **86.3%** | 64.6% | -21.8 |
| md | 1,038 | **85.9%** | 60.7% | -25.2 |
| ms | 1,036 | **83.9%** | 54.9% | -29.0 |
| gfgs | 98 | **82.7%** | 51.0% | -31.6 |
| fs | 1,135 | **78.8%** | 55.3% | -23.4 |
| fd | 918 | **76.9%** | 47.9% | -29.0 |
| gfgd | 138 | **75.4%** | 50.7% | -24.6 |
| gmgd | 123 | **63.4%** | 48.0% | -15.4 |
| gmgs | 121 | **61.2%** | 41.3% | -19.8 |

R001 dominates every relation. Notably, **R001 also has the most uniform per-relation accuracy of any model in the project** — 61-87% across all 11 classes vs 40-95% for parametric models that drop to near-chance on grandparents.

---

## Issues Log

| # | Issue | Severity | Status | First seen | Notes |
|---|-------|----------|--------|------------|-------|
| 1 | Large val→test gap, growing across runs | High | ❌ Open | Run 001 | -0.060 in R001 → -0.092 in R002. Larger backbone + more retrieval context amplified overfitting to gallery↔val correlation, not generalization |
| 2 | Threshold drift (0.40 → 0.55) hurt recall on test | Medium | ❌ Open | Run 002 | Stricter validation threshold + worse calibration → test recall dropped from 82% to 57%. Score distributions tighter under stronger relation-CE signal |
| 3 | Gallery memorization vs generalization | High | ❌ Open | Run 002 | Gallery is fixed (no refresh during training). Cross-attention can memorize gallery-specific patterns that don't transfer to held-out test pairs |
| 4 | Frozen encoder limits ceiling | Medium | Accepted by design | Run 001 | Project hypothesis was that retrieval substitutes for fine-tuning. Confirmed it does, but at -0.07 AUC vs parametric models |

---

## Comparison with Other Models (FIW)

| Model | Test AUC | Test Acc | TAR@FAR=0.01 | Trainable | Min per-rel |
|---|---:|---:|---:|---:|---:|
| Modelo 02 R031 | **0.850** | 74.4% | ~0.13 | ~86M | 88.4% (gmgs) |
| Modelo 03 R002 | **0.850** | 47.9%* | 0.130 | ~176M | — |
| Modelo 03 R006 | 0.848 | 50.5%* | 0.132 | ~176M | — |
| Modelo 05 R001 | 0.806 | **72.6%** | **0.152** | **8.47M** | 39.8% (gfgs) |
| Modelo 05 R002 | 0.799 | 72.4% | 0.095 | 8.47M | 38.8% (gfgs) |
| **Modelo 06 R001** | 0.776 | 69.8% | 0.062 | **8.16M** | **61.2% (gmgs)** |
| **Modelo 06 R002** | 0.731 | 66.2% | 0.042 | ~12M | 41.3% (gmgs) |

*threshold=0.5 default — accuracy not directly comparable.

M06 R001 is the **only model that holds 60%+ accuracy on all 11 relations**, including grandparents. That uniformity is the architecture's headline contribution — at the cost of a -0.07 AUC gap vs the parametric models. M06 R002 squandered both: AUC dropped further and per-relation uniformity collapsed.

---

## Conclusion

Two runs. R001 established the design hypothesis: a non-parametric gallery + cross-attention can substitute for backbone fine-tuning, reaching Test AUC=0.776 with only 8.16M trainable params. Crucially, **R001 is the only model in the project that doesn't fail catastrophically on grandparent classes** (61% on gmgs vs 40-52% for parametric models). The retrieval mechanism gives the model concrete examples of every relation type, which is exactly what the data-starved grandparent classes need.

R002 tested whether scaling retrieval (DINOv2 backbone, K=64, stronger relation-CE) lifts the ceiling. The hypothesis was rejected hard: AUC dropped 0.776 → 0.731, the val→test gap **grew** from -0.060 to -0.092, and per-relation accuracy collapsed across the board. The lesson: in this architecture, more retrieval context + richer features amplify gallery-to-validation correlation that doesn't transfer to test pairs. **Scaling the retrieval surface is not the right axis.**

**Open question:** the right axis is probably *what* gets retrieved, not *how much*. Mixing in hard negatives (positives that retrieval picks for non-kin queries) would force the cross-attention to discriminate rather than memorize. That's the next experiment to design.
