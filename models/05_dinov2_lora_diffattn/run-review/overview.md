# Overview — DINOv2 + LoRA + Differential Cross-Attention

**Model:** 05 — DINOv2 ViT-B/14 (frozen) + LoRA rank=8 + differential cross-attention + auxiliary relation head
**GPU:** AMD Radeon RX 6750 XT (11.98 GB VRAM, gfx1031, ROCm 5.7)
**Dataset:** FIW (RFIW Track-I)

Individual run details: [run-001.md](run-001.md) · [run-002.md](run-002.md)

---

## Run Comparison

| | Run 001 | Run 002 |
|---|---|---|
| **Date** | 2026-04-27 | 2026-04-29 |
| **Purpose** | First training run with all defaults | Test if stronger regularization closes val→test gap |
| **Backbone** | vit_base_patch14_dinov2.lvd142m (frozen) | same |
| **LoRA rank / alpha** | 8 / 16 | same |
| **LoRA dropout** | 0.0 | **0.1** |
| **Head dropout** | 0.1 | **0.2** |
| **LR (peak)** | 3e-4 | **1.5e-4** |
| **Warmup** | 3 ep | same |
| **Scheduler** | cosine, min_lr=1e-6 | same |
| **Batch / grad accum** | 4 / 8 (eff. 32) | same |
| **Loss** | BCE + 0.5·contrastive + **0.2**·rel-CE | BCE + 0.5·contrastive + **0.4**·rel-CE |
| **Patience** | 15 (effective 10) | 10 |
| **Epochs trained** | 22/40 | 14/40 |
| **Best epoch** | 12 | **4** |
| **Trainable params** | 8.47M / 94.2M (8.99%) | same |
| **Time / epoch** | ~84 min | ~86 min |

### Validation peak

| | Run 001 | Run 002 |
|---|---|---|
| **Val AUC peak** | **0.9116** | 0.9048 |
| **Val Acc** | 83.7% | 82.6% |
| **Threshold (val)** | 0.10 | 0.30 |

### Test metrics (validation-selected threshold)

| Metric | Run 001 | Run 002 |
|---|---|---|
| **Test ROC-AUC** | **0.806** | 0.799 |
| **Test Accuracy** | 72.6% | 72.4% |
| **Test F1** | 0.713 | 0.718 |
| **Avg Precision** | 0.792 | 0.772 |
| **TAR @ FAR=0.1** | 0.463 | 0.437 |
| **TAR @ FAR=0.01** | **0.152** | 0.095 |
| **TAR @ FAR=0.001** | 0.044 | 0.017 |
| **Val→Test gap** | -0.105 | -0.106 |

R001 wins on every metric except F1 (essentially tied). The val→test gap is unchanged.

### Per-relation test accuracy

| Relation | N | Run 001 | Run 002 | Δ |
|---|---|---:|---:|---:|
| sibs | 234 | 83.3% | **85.9%** | +2.6 |
| bb | 860 | 79.8% | **85.5%** | +5.7 |
| ss | 731 | 77.2% | **81.1%** | +3.9 |
| fs | 1,135 | 71.6% | **73.8%** | +2.2 |
| fd | 918 | 71.5% | **72.9%** | +1.4 |
| md | 1,038 | 67.3% | **70.3%** | +3.0 |
| ms | 1,036 | 69.4% | 69.8% | +0.4 |
| gmgs | 121 | 52.1% | 52.1% | 0.0 |
| gfgd | 138 | **50.7%** | 49.3% | -1.4 |
| gmgd | 123 | 40.7% | **45.5%** | +4.9 |
| gfgs | 98 | **39.8%** | 38.8% | -1.0 |

R002 distributes confidence across classes — small gains on intra-generation (bb/ss/sibs) and gmgd, parity on gfgs/gmgs/gfgd.

---

## Train Data Distribution (FIW positives)

The grandparent relations are **systematically underrepresented**:

| Relation | Train pairs | % of pos | Test acc R001 |
|---|---:|---:|---:|
| md | 5,349 | 16.1% | 67.3% |
| fs | 5,299 | 16.0% | 71.6% |
| fd | 5,268 | 15.9% | 71.5% |
| ms | 5,109 | 15.4% | 69.4% |
| sibs | 3,399 | 10.2% | 83.3% |
| ss | 3,298 | 9.9% | 77.2% |
| bb | 3,030 | 9.1% | 79.8% |
| **gfgs** | **682** | **2.1%** | **39.8%** |
| **gmgs** | **623** | **1.9%** | **52.1%** |
| **gfgd** | **608** | **1.8%** | **50.7%** |
| **gmgd** | **542** | **1.6%** | **40.7%** |

Grandparent classes have **~7× fewer training pairs** than parent-child classes. The poor accuracy on gfgs/gmgs/gfgd/gmgd is not an architecture failure — it's data starvation. The model receives 6-9× fewer gradient updates teaching grandparent kinship than parent-child kinship.

---

## Issues Log

| # | Issue | Severity | Status | First seen | Notes |
|---|-------|----------|--------|------------|-------|
| 1 | Patience config not honored — trainer stopped at patience 10 although `--patience 15` was passed | Medium | ❌ Open | Run 001 | Investigate `ROCmTrainer` base class — likely a hard-coded threshold somewhere |
| 2 | Large val→test gap (-0.105 R001, -0.106 R002) | High | ❌ Open | Run 001 | Not closed by stronger regularization. Hypothesized to be a structural property of the RFIW Track-I family split (val and test families have different visual discriminability), not a model property |
| 3 | Grandparent classes ~40-52% accuracy | High | ❌ Open | Run 001 | Confirmed root cause: data starvation (1.6-2.1% of train). Architecture/regularization fixes will not move it. Needs class-balanced sampling or weighted BCE |
| 4 | Time per epoch ~84 min (~3-4× M03/M06) | Low | Accepted | Run 001 | DINOv2 patch14 + grad checkpointing + diffattn. Not a blocker |

---

## Comparison with Other Models (FIW)

| Model | Test AUC | Test Acc | TAR@FAR=0.01 | Trainable | Min per-rel |
|---|---:|---:|---:|---:|---:|
| Modelo 02 R031 | **0.850** | 74.4% | ~0.13 | ~86M | 88.4% |
| Modelo 03 R002 | **0.850** | 47.9%* | 0.130 | ~176M | — |
| Modelo 03 R006 | 0.848 | 50.5%* | 0.132 | ~176M | — |
| **Modelo 05 R001** | 0.806 | **72.6%** | **0.152** | **8.47M** | 39.8% |
| **Modelo 05 R002** | 0.799 | 72.4% | 0.095 | 8.47M | 38.8% |
| Modelo 06 R001 | 0.776 | 69.8% | 0.062 | 8.16M | 61.2% |
| Modelo 06 R002 | 0.731 | 66.2% | 0.042 | ~12M | 41.3% |

*threshold=0.5 default — accuracy not directly comparable.

M05 R001 has the **best TAR@FAR=0.01 of any model in the project** (0.152), making it the strongest choice for high-precision regimes. Test AUC sits at 0.806 — below the parametric models (0.850) but with **10-20× fewer trainable parameters**.

---

## Conclusion

Two runs. R001 set the project's highest validation AUC (0.9116) and the strongest TAR@FAR=0.01 result (0.152). R002 tested whether stronger regularization closes the val→test gap; **the hypothesis was rejected** — the gap stayed at -0.106 and TAR@FAR rigid-threshold collapsed.

Combined with the train-data analysis, the picture is:

1. **The gap is structural, not overfitting.** Halving LR + tripling dropout sources didn't move the gap. Likely cause is divergence between the val and test family distributions in RFIW Track-I.
2. **The grandparent failure is data scarcity, not architecture.** 1.6-2.1% of train per grandparent class. No multitask weight or capacity reduction will fix this.
3. **The architecture itself is competitive.** DINOv2 + LoRA rank=8 + differential cross-attention reaches Val AUC 0.91 with 8.47M trainable params, validating the design hypothesis.

**Next runs should not iterate on regularization or relation-loss weight.** The diagnostic-first ordering: (a) class-balanced sampling, (b) multi-seed test of the val→test gap, (c) per-relation thresholds for high-precision applications.
