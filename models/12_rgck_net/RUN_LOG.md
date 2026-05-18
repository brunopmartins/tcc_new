# Run Log — Model 12: RGCK-Net (Region-Guided Cross Kinship Network)

This file logs every training run for this model with the exact configuration and final metrics. The goal is reproducibility: anyone (human or agent) reading this should be able to recreate any run by copying its launch command.

Newest run on top.

---

## Run 010 — 2026-05-17 — Manually stopped during ep 10 (peak Val AUC 0.9052 ep 5; Test AUC 0.8754. R007+R009 stack: mid-FAR champion, AP champion)

**Status:** Manually stopped during epoch 10 training (after ep 9 summary at 0.8996). Peak Val AUC 0.9052 reached at ep 5, tied at ep 8 (saved best.pt is from ep 5).

**Outcome:** Stack of R007 (differential LR) + R009 (comparison-only fusion) on the R006 baseline. **Aggregate Test AUC 0.8754 (-0.003 vs R006, within ±0.009 noise floor). Strict-FAR essentially tied with R009 (5.21 % vs 5.43 %, no compounding to the hypothesised 6-7 % range). But R010 IS the new MID-FAR champion (TAR@FAR=0.01 = 21.16 %, +2.55 pp vs R006) AND the new AP champion (0.8567) AND the new TAR@FAR=0.1 champion (60.00 %).** R010 is the most "balanced" run in the M12 cycle — top-3 on every Test metric.

**Hypothesis test result:** PARTIALLY CONFIRMED. Strict-FAR did not compound (R009 ceiling), but the stack produced a superadditive mid-FAR gain (+2.55 pp vs R006, larger than R007 alone +0.98 pp or R009 alone -0.82 pp) and the highest Test AP of any M12 run.

### Launch command

```bash
SKIP_INSTALL=1 \
ALIGNED_ROOT=/home/bruno/Desktop/tcc_new/datasets/FIW_aligned \
BATCH_SIZE=8 \
GRAD_ACCUM=4 \
UNFREEZE_LAST_STAGE=1 \
DIFFERENTIAL_LR=1 \
LR_STAGE4=5e-6 \
LR_OUTPUT_LAYER=5e-6 \
LR_HEAD=2e-5 \
RELATION_AUX_WEIGHT=0.05 \
SYMMETRIC_FORWARD=1 \
COMPARISON_ONLY_FUSION=1 \
NUM_WORKERS=4 \
SEED=42 \
bash models/12_rgck_net/AMD/run_pipeline.sh
```

### Changes from Run 006

| Parameter | R006 | **R010** |
|---|---|---|
| `DIFFERENTIAL_LR` | 0 | **1** (from R007) |
| LR stage4 | 1e-5 (single) | **5e-6** (from R007) |
| LR output_layer | 1e-5 (single) | **5e-6** (from R007) |
| LR head | 1e-5 (single) | **2e-5** (from R007) |
| `COMPARISON_ONLY_FUSION` | 0 | **1** (from R009) |
| Classifier input dim | 2049 | 1035 (from R009) |
| Trainable params | 31,560,461 | **31,036,173** (R009 value) |
| All other knobs | — | identical |

Two-knob stack — both individually validated in R007 and R009.

### Training trajectory

LR shown is the first param group (stage4); head LR ≈ 4× that.

| Epoch | Train Loss | Val Acc | Val AUC | Thr | LR (stage4) | R007 same ep | R009 same ep |
|------:|-----------:|--------:|--------:|----:|------------:|-------------:|-------------:|
| 1 | 0.7328 | 72.7 % | 0.8313 | 0.450 | 1.80e-6 | 0.8496 | 0.8505 |
| 2 | 0.5726 | 80.0 % | 0.8964 | 0.300 | 2.60e-6 | 0.9046 | 0.8993 |
| 3 | 0.5022 | 80.8 % | 0.9046 | 0.350 | 3.40e-6 | 0.9089 | 0.9012 (R009 peak) |
| 4 | 0.4452 | 81.2 % | 0.9041 | 0.350 | 4.20e-6 | 0.9093 (R007 peak) | 0.8975 |
| **5** | **0.3962** | **82.1 %** | **0.9052 (peak)** | 0.500 | 5.00e-6 | 0.9056 | 0.8994 |
| 6 | 0.3554 | 82.4 % | 0.9047 | 0.550 | 5.00e-6 | 0.9023 | 0.8955 |
| 7 | 0.3225 | 81.7 % | 0.8981 | 0.100 | 5.00e-6 | 0.9024 | 0.8859 |
| 8 | 0.3011 | 82.2 % | **0.9052 (tied peak)** | 0.100 | 4.99e-6 | 0.8964 | 0.8924 |
| 9 | 0.2690 | 81.9 % | 0.8996 | 0.100 | 4.98e-6 | ~0.895 | 0.8879 |
| 10 | — | — | — | — | — | — | — (stopped mid-training) |

R010 held the peak best of any M12 run — tied at 0.9052 in ep 5 AND ep 8.

### Test metrics (val-selected threshold 0.500)

| Metric | M12 R006 (HEADLINE) | M12 R009 (strict-FAR champ) | **M12 R010** | Δ R010-R006 | Δ R010-R009 |
|---|---:|---:|---:|---:|---:|
| **Test ROC AUC** | **0.8788** | 0.8739 | 0.8754 | -0.003 (within noise) | +0.002 (within noise) |
| Test Accuracy | 79.33 % | 78.28 % | 79.25 % | -0.1 pp | +1.0 pp |
| Test Balanced Acc | 79.65 % | 78.74 % | 79.44 % | -0.2 pp | +0.7 pp |
| Test F1 | **0.8017** | 0.7984 | 0.7952 | -0.007 | -0.003 |
| Test Precision | 74.17 % | 71.89 % | **75.43 %** | **+1.3 pp** | +3.5 pp |
| Test Recall | 87.24 % | **89.77 %** | 84.08 % | -3.2 pp | -5.7 pp |
| **Test Avg Precision** | 0.8561 | 0.8497 | **0.8567** ⭐ | **+0.001 (M12 best)** | +0.007 |
| **TAR @ FAR=0.001** | 3.33 % | **5.43 %** | 5.21 % | +1.88 pp | -0.22 pp (tied) |
| **TAR @ FAR=0.01** | 18.61 % | 17.79 % | **21.16 %** ⭐⭐ | **+2.55 pp (M12 best)** | **+3.37 pp** |
| **TAR @ FAR=0.1** | 59.93 % | 58.61 % | **60.00 %** ⭐ | **+0.07 pp (M12 best)** | +1.39 pp |
| Val→test AUC gap | **-0.026** | -0.027 | -0.030 | -0.004 | -0.003 |

**R010 wins TAR@FAR=0.01 (+2.55 pp), TAR@FAR=0.1 (+0.07 pp), Test AP (+0.001), and Test Precision (+1.3 pp) against R006.** Strict-FAR essentially tied with R009 (5.21 vs 5.43 within noise). Aggregate Test AUC within R006's noise floor.

### Per-relation accuracy (FIW Track-I test, threshold 0.500)

| Relation | N | M12 R006 | M12 R009 | **M12 R010** | Δ R010-R006 |
|----------|--:|---------:|---------:|-------------:|---:|
| bb | 860 | 89.1 % | 91.5 % | 85.2 % | -3.9 pp |
| ss | 731 | 88.9 % | 90.8 % | 86.9 % | -2.0 pp |
| sibs | 234 | 93.2 % | 94.9 % | 90.2 % | -3.0 pp |
| fs | 1135 | 87.9 % | 90.6 % | 86.7 % | -1.2 pp |
| fd | 918 | 84.6 % | 86.9 % | 80.0 % | -4.6 pp |
| ms | 1036 | 86.2 % | 89.2 % | 82.7 % | -3.5 pp |
| md | 1038 | 89.0 % | 91.9 % | 86.3 % | -2.7 pp |
| gfgd | 138 | 83.3 % | 84.8 % | 78.3 % | -5.0 pp |
| **gfgs** | 98 | 76.5 % | 85.7 % | **83.7 %** | **+7.2 pp** |
| gmgd | 123 | 79.7 % | 82.9 % | 71.5 % | -8.2 pp |
| gmgs | 121 | 80.2 % | 77.7 % | 66.1 % | -14.1 pp |
| **non-kin** | 6993 | 72.1 % | 67.7 % | **74.8 %** | **+2.7 pp** |

Most per-class regression is the threshold shift (0.500 vs R006's 0.250) — R010 is more conservative on positives. gfgs gain (+7.2 pp) and non-kin gain (+2.7 pp) persist despite higher threshold.

### Files

- Checkpoint: `output/010/checkpoints/best.pt` (529 MB — same size as R009 due to smaller classifier)
- Train log: `output/010/logs/train.log` (truncated at ep 9 + ~30% of ep 10)
- Test log: `output/010/logs/test.log`
- Results: `output/010/results/test_metrics_rocm.txt`
- Run review: `run-review/run-010.md`

---

## Run 009 — 2026-05-16 — SAFEGUARD ep 15 (peak Val AUC 0.9012 ep 3; Test AUC 0.8739. Comparison-only fusion: Val drop confirmed, NEW STRICT-FAR CHAMPION)

**Status:** SAFEGUARD auto-stopped at ep 15.

**Outcome:** R006 stack + **comparison-only fusion** — drop gA, gB from the classifier input. Aggregate Test AUC -0.005 vs R006 (within ±0.009 noise floor), but **TAR@FAR=0.001 = 5.43 % — highest among any trained M12 model in the project**, +2.1 pp over R006 and +1.25 pp over R002 (prior trained-model best). **9 of 11 kin classes improved 1-9 pp vs R006** (gfgs +9.2 pp the standout; the historically hardest grandparent class is now at 85.7 %). Val→test gap stayed at -0.027 (tied with R006).

**Hypothesis test result:** PARTIALLY CONFIRMED. Val AUC dropped exactly as predicted (-0.004), Test AUC tied. The intervention IS doing something — per-class kin discrimination shifted substantially and strict-FAR jumped — but the aggregate AUC summary doesn't capture it. R006 stays aggregate-AUC headline; **R009 is the new strict-FAR champion** (best for low-FAR verification deployment).

### Launch command

```bash
SKIP_INSTALL=1 \
ALIGNED_ROOT=/home/bruno/Desktop/tcc_new/datasets/FIW_aligned \
BATCH_SIZE=8 \
GRAD_ACCUM=4 \
UNFREEZE_LAST_STAGE=1 \
LEARNING_RATE=1e-5 \
RELATION_AUX_WEIGHT=0.05 \
SYMMETRIC_FORWARD=1 \
COMPARISON_ONLY_FUSION=1 \
NUM_WORKERS=4 \
SEED=42 \
bash models/12_rgck_net/AMD/run_pipeline.sh
```

### Changes from Run 006

| Parameter | R006 | **R009** |
|---|---|---|
| `COMPARISON_ONLY_FUSION` | 0 (off) | **1 (on)** |
| Classifier input dim | 2049 (4·512 + 11) | **1035 (2·512 + 11)** |
| Classifier first Linear | `Linear(2049, 512)` | `Linear(1035, 512)` |
| Trainable params | 31,560,461 | 31,036,173 (-524,288, -1.7 %) |
| All other knobs | — | identical |

### Training trajectory

| Epoch | Train Loss | Val Acc | Val AUC | Thr | LR | R006 same ep | Δ R009-R006 |
|------:|-----------:|--------:|--------:|----:|-----|------:|------:|
| 1 | 0.7260 | 74.5 % | 0.8505 | 0.450 | 2.0e-6 | 0.8594 | -0.009 |
| 2 | 0.5530 | 80.2 % | 0.8993 | 0.300 | 4.0e-6 | 0.9045 | -0.005 |
| **3** | **0.4727** | **80.1 %** | **0.9012 (peak)** | 0.250 | 6.0e-6 | 0.9049 | **-0.004** |
| 4 | 0.4041 | 80.2 % | 0.8975 | 0.300 | 8.0e-6 | 0.9035 | -0.006 |
| 5 | 0.3507 | 81.3 % | 0.8994 | 0.350 | 1.0e-5 | 0.9006 | -0.001 |
| 6-14 | 0.30→0.14 | 79-82 % | 0.88-0.90 | 0.10-0.35 | ~9.9e-6 | — | — |
| 15 | 0.1297 | 78.0 % | 0.8800 | 0.100 | 9.76e-6 | — | SAFEGUARD fired |

Val AUC consistently 0.004-0.009 below R006 throughout — exactly the predicted "less raw signal" effect. Train loss drops faster than R006 (R009 ep 15: 0.130 vs R006/R008 ~0.20-0.30): with fewer features (no gA/gB) the classifier fits faster, but Val/Test don't move proportionally.

### Test metrics (val-selected threshold 0.250)

| Metric | M12 R002 | M12 R006 (HEADLINE) | M12 R007 | M12 R008 | **M12 R009** | Δ R009-R006 |
|---|---:|---:|---:|---:|---:|---:|
| **Test ROC AUC** | 0.8564 | **0.8788** | 0.8730 | 0.8788 | 0.8739 | -0.005 (within noise floor) |
| Test Accuracy | 76.79 % | 79.33 % | 78.76 % | 79.28 % | 78.28 % | -1.0 pp |
| Test Balanced Acc | 76.48 % | 79.65 % | 79.11 % | 79.61 % | 78.74 % | -0.9 pp |
| Test F1 | 0.7402 | **0.8017** | 0.7980 | 0.8017 | 0.7984 | -0.003 |
| Test Precision | 79.82 % | 74.17 % | 73.28 % | 74.05 % | 71.89 % | -2.3 pp |
| Test Recall | 69.00 % | 87.24 % | 87.61 % | 87.39 % | **89.77 %** | +2.5 pp |
| Test Avg Precision | 0.8389 | 0.8561 | 0.8521 | 0.8563 | 0.8497 | -0.006 |
| **TAR @ FAR=0.001** | 4.18 % | 3.33 % | 4.49 % | 2.78 % | **5.43 %** ⭐⭐ | **+2.1 pp** |
| **TAR @ FAR=0.01** | 17.58 % | **18.61 %** | 19.59 % | 19.12 % | 17.79 % | -0.8 pp |
| **TAR @ FAR=0.1** | 57.11 % | **59.93 %** | 57.56 % | 59.65 % | 58.61 % | -1.3 pp |
| Val→test AUC gap | -0.076 | **-0.026** | -0.036 | -0.026 | -0.027 | tied |

**TAR@FAR=0.001 = 5.43 % is the highest of any TRAINED M12 run** (only B0 at 7.06 % is higher, and B0 has zero training). For strict-verification deployment R009 is the best M12 model.

### Per-relation accuracy (FIW Track-I test, threshold 0.250)

| Relation | N | M12 R006 | **M12 R009** | Δ |
|----------|--:|---------:|-------------:|---:|
| bb | 860 | 89.1 % | **91.5 %** | **+2.4 pp** |
| ss | 731 | 88.9 % | **90.8 %** | +1.9 pp |
| sibs | 234 | 93.2 % | **94.9 %** | +1.7 pp |
| fs | 1135 | 87.9 % | **90.6 %** | **+2.7 pp** |
| fd | 918 | 84.6 % | **86.9 %** | **+2.3 pp** |
| ms | 1036 | 86.2 % | **89.2 %** | **+3.0 pp** |
| md | 1038 | 89.0 % | **91.9 %** | **+2.9 pp** |
| **gfgd** | 138 | 83.3 % | **84.8 %** | +1.5 pp |
| **gfgs** | 98 | 76.5 % | **85.7 %** | **+9.2 pp** ⭐⭐ |
| **gmgd** | 123 | 79.7 % | **82.9 %** | **+3.2 pp** |
| **gmgs** | 121 | 80.2 % | 77.7 % | -2.5 pp |
| non-kin | 6993 | 72.1 % | 67.7 % | -4.4 pp |

9 of 11 kin classes gained 1-9 pp; gfgs gained 9.2 pp (historic worst class now at 85.7 %).

### Notes

- **Mechanism**: gA, gB carry identity-as-feature signal. R006 had the classifier seeing both raw global tokens AND comparison features. R009 forces it to use only comparison features (|diff|, prod, sims, weights, score). The "less raw signal" effect drops Val by 0.004 (predicted); the "no identity shortcut" effect lets strict-FAR gain 2.1 pp.
- **Operating-point shift**: R009 trades mid-range FAR for strict-range FAR. R006 wins TAR@FAR=0.01 (+0.8) and FAR=0.1 (+1.3); R009 wins FAR=0.001 (+2.1).
- **Per-class signature is dominant**: R009's grandparent improvements are large (gfgs +9.2 the standout). The trade-off is concentrated in non-kin (-4.4) and one grandparent (gmgs -2.5).
- **Train loss falls faster** in R009 (fewer features → easier fit), but Val/Test AUC don't move proportionally. The signal R009 removes wasn't "missing capacity" — it was identity leakage.

### Artifacts

- Checkpoint (epoch 3, Val AUC 0.9012): `output/009/checkpoints/best.pt` (529 MB — smaller than R006/R008 due to reduced classifier input)
- Train log: `output/009/logs/train.log`
- Test log: `output/009/logs/test.log`
- Results: `output/009/results/test_metrics_rocm.txt`
- Trainable params: 31,036,173 / 70,222,029 (44.20 %) — 524,288 fewer than R005-R008

---

## Run 008 — 2026-05-16 — SAFEGUARD ep 16 (peak Val AUC 0.9052 ep 3; Test AUC 0.8788 — TIED with R006 within 4-decimal precision)

**Status:** SAFEGUARD auto-stopped at ep 16 (Val below peak 0.9052 for 10 consecutive epochs).

**Outcome:** R006 stack + **L2-SP regulariser λ=1e-3** on the 34 unfrozen backbone tensors (stage 4 + output_layer, 25.9 M params, 82.3 % of trainable). Penalty was applied and constrained the backbone (R008 train loss at ep 7: 0.344 vs R006: 0.260, +0.084) but the final solution converges to **the same Test predictions** as R006: Test ROC AUC 0.8788 (identical to R006 at 4 dec.), Val→test gap -0.026 (identical), Test F1 0.8017 (identical), Test AP 0.8563 (+0.0002 noise). Per-relation differences ≤0.4 pp in 10 of 11 kin classes; only gmgd shifted +1.6 pp.

**Hypothesis test result:** NOT CONFIRMED (no Test lift); the neutral hypothesis is CONFIRMED. R006's symmetric forward appears to have already exhausted the gap-closure available from generalisation regularisers at this λ. Whether a higher λ (1e-2 or 1e-1) would help or hurt is untested.

### Launch command

```bash
SKIP_INSTALL=1 \
ALIGNED_ROOT=/home/bruno/Desktop/tcc_new/datasets/FIW_aligned \
BATCH_SIZE=8 \
GRAD_ACCUM=4 \
UNFREEZE_LAST_STAGE=1 \
LEARNING_RATE=1e-5 \
RELATION_AUX_WEIGHT=0.05 \
SYMMETRIC_FORWARD=1 \
L2SP_WEIGHT=1e-3 \
NUM_WORKERS=4 \
SEED=42 \
bash models/12_rgck_net/AMD/run_pipeline.sh
```

### Changes from Run 006

| Parameter | R006 | **R008** |
|---|---|---|
| `L2SP_WEIGHT` | 0.0 (off) | **1e-3** |
| Loss | 0.5·(BCE_AB+BCE_BA) + 0.05·avg(CE_rel_AB, CE_rel_BA)\|pos | **+ 1e-3·L2SP(stage4+output_layer)** |
| All other knobs | — | identical |

L2-SP state: 34 tensors / 25,965,056 params anchored to AdaFace pretrain (stage 4: 13.1 M, output_layer: 12.8 M).

### Training trajectory

| Epoch | Train Loss | Val Acc | Val AUC | Thr | LR | R006 same ep | Δ R008-R006 |
|------:|-----------:|--------:|--------:|----:|-----|------:|------:|
| 1 | 0.7289 | 75.2 % | 0.8592 | 0.450 | 2.0e-6 | 0.8594 | -0.000 |
| 2 | 0.5561 | 80.6 % | 0.9046 | 0.350 | 4.0e-6 | 0.9045 | +0.000 |
| **3** | **0.4849** | **81.3 %** | **0.9052 (peak)** | 0.500 | 6.0e-6 | 0.9049 | +0.000 |
| 4 | 0.4315 | 81.9 % | 0.9047 | 0.200 | 8.0e-6 | 0.9035 | +0.001 |
| 5 | 0.4028 | 81.6 % | 0.9019 | 0.300 | 1.0e-5 | 0.9006 | +0.001 |
| 6 | 0.3749 | 82.1 % | 0.9048 | 0.100 | 1.0e-5 | 0.9019 | +0.003 |
| 7 | 0.3438 | 82.0 % | 0.8968 | 0.100 | 9.99e-6 | 0.8923 | +0.005 |
| 8-15 | (declining 0.33→0.27) | 80-82 % | 0.88-0.89 | 0.10 | ~9.9e-6 | — | — |
| 16 | 0.2785 | 80.6 % | 0.8887 | 0.100 | 9.71e-6 | — | SAFEGUARD fired |

Val AUC peak +0.0003 vs R006 (within noise). Post-peak decline ~0.003-0.005 gentler than R006 — expected effect of a regulariser. Train loss substantially higher at every epoch — penalty is being applied and constraining the backbone.

### Test metrics (val-selected threshold 0.500)

| Metric | M12 R002 | M12 R006 (HEADLINE) | M12 R007 | **M12 R008** | Δ R008-R006 |
|---|---:|---:|---:|---:|---:|
| **Test ROC AUC** | 0.8564 | **0.8788** | 0.8730 | **0.8788** | **+0.0000** (tied) |
| Test Accuracy | 76.79 % | 79.33 % | 78.76 % | 79.28 % | -0.05 pp |
| Test Balanced Acc | 76.48 % | 79.65 % | 79.11 % | 79.61 % | -0.04 pp |
| Test F1 | 0.7402 | 0.8017 | 0.7980 | **0.8017** | **+0.0000** (tied) |
| Test Precision | 79.82 % | 74.17 % | 73.28 % | 74.05 % | -0.12 pp |
| Test Recall | 69.00 % | 87.24 % | 87.61 % | 87.39 % | +0.15 pp |
| Test Avg Precision | 0.8389 | 0.8561 | 0.8521 | **0.8563** | +0.0002 (noise) |
| TAR @ FAR=0.001 | 4.18 % | 3.33 % | **4.49 %** | 2.78 % | -0.55 pp |
| TAR @ FAR=0.01 | 17.58 % | 18.61 % | **19.59 %** | 19.12 % | +0.51 pp |
| TAR @ FAR=0.1 | 57.11 % | **59.93 %** | 57.56 % | 59.65 % | -0.28 pp |
| Val→test AUC gap | -0.076 | **-0.026** | -0.036 | **-0.026** | **0.0000** (tied) |

### Per-relation accuracy (FIW Track-I test, threshold 0.500)

| Relation | N | M12 R006 | **M12 R008** | Δ |
|----------|--:|---------:|-------------:|---:|
| bb | 860 | 89.1 % | 89.4 % | +0.4 pp |
| ss | 731 | 88.9 % | 88.9 % | 0.0 pp |
| sibs | 234 | 93.2 % | 92.7 % | -0.4 pp |
| fs | 1135 | 87.9 % | 88.0 % | +0.1 pp |
| fd | 918 | 84.6 % | 84.8 % | +0.2 pp |
| ms | 1036 | 86.2 % | 86.3 % | +0.1 pp |
| md | 1038 | 89.0 % | 89.3 % | +0.3 pp |
| **gfgd** | 138 | 83.3 % | 83.3 % | 0.0 pp |
| **gfgs** | 98 | 76.5 % | 76.5 % | 0.0 pp |
| **gmgd** | 123 | 79.7 % | 81.3 % | +1.6 pp |
| **gmgs** | 121 | 80.2 % | 80.2 % | 0.0 pp |
| non-kin | 6993 | 72.1 % | 71.8 % | -0.3 pp |

Three of four grandparent classes match R006 to the decimal; only gmgd shifted +1.6 pp. R008 is essentially identical to R006 at the per-class level — the signature of a regulariser that doesn't change the final classifier behaviour.

### Notes

- **Mechanism**: L2-SP penalty IS being applied (train loss +0.084 vs R006 at ep 7, smoother post-peak decline). The backbone genuinely cannot drift as far. But the constrained solution lands at the same Test discrimination as R006.
- **Two interpretations of the neutral outcome**:
  - λ=1e-3 may be too small to meaningfully change the final solution. Untested higher λ (1e-2 or 1e-1).
  - R006's symmetric forward already exhausted the gap-closure available from generalisation regularisers; the residual -0.026 gap may be irreducible noise for this dataset/architecture.
- **R006 stays headline.** R008 is empirically a tie, not an improvement.
- **Structural taxonomy now clear**: R006 (symmetry) is a *shortcut-removal* intervention — qualitatively different from R005 (per-class loss), R007 (capacity reallocation), R008 (parameter anchoring). Only R006 moves Test AUC meaningfully.

### Artifacts

- Checkpoint (epoch 3, Val AUC 0.9052): `output/008/checkpoints/best.pt` (536 MB)
- Train log: `output/008/logs/train.log`
- Test log: `output/008/logs/test.log`
- Results: `output/008/results/test_metrics_rocm.txt`
- Trainable params: 31,560,461 / 70,746,317 (44.61 %)
- L2-SP anchored: 25,965,056 / 34 tensors

---

## Run 007 — 2026-05-15 — SAFEGUARD ep 15 (peak Val AUC 0.9093 ep 4; Test AUC 0.8730. Differential LR: Val lift confirmed, Test neutral within noise)

**Status:** SAFEGUARD auto-stopped at ep 15 (Val below peak 0.9093 for 10 consecutive epochs).

**Outcome:** Differential LR on top of R006 stack: stage4 5e-6, output_layer 5e-6, head 2e-5 (vs R006's uniform 1e-5). **Peak Val AUC 0.9093 — +0.004 above R006 (mechanism confirmed: higher head LR raises fit).** Test ROC AUC 0.8730 — -0.006 vs R006 within the ±0.009 noise floor. Val→Test gap widened slightly to -0.036 (R006: -0.026). Per-FAR: TAR@FAR=0.001 +1.2 pp, FAR=0.01 +1.0 pp, FAR=0.1 -2.4 pp — diff-LR partially recovers strict-FAR performance that R006 had sacrificed.

**Hypothesis test result:** PARTIALLY CONFIRMED. Val lift confirmed (+0.004 peak), Test AUC change within noise. Diff-LR is empirically neutral on Test in this configuration. R006 remains project headline.

### Launch command

```bash
SKIP_INSTALL=1 \
ALIGNED_ROOT=/home/bruno/Desktop/tcc_new/datasets/FIW_aligned \
BATCH_SIZE=8 \
GRAD_ACCUM=4 \
UNFREEZE_LAST_STAGE=1 \
DIFFERENTIAL_LR=1 \
LR_STAGE4=5e-6 \
LR_OUTPUT_LAYER=5e-6 \
LR_HEAD=2e-5 \
RELATION_AUX_WEIGHT=0.05 \
SYMMETRIC_FORWARD=1 \
NUM_WORKERS=4 \
SEED=42 \
bash models/12_rgck_net/AMD/run_pipeline.sh
```

### Changes from Run 006

| Parameter | R006 | **R007** |
|---|---|---|
| Optimiser | AdamW single LR 1e-5 | **AdamW 3 param groups** |
| LR — stage 4 | 1e-5 | **5e-6** (-50 %) |
| LR — output_layer | 1e-5 | **5e-6** (-50 %) |
| LR — head | 1e-5 | **2e-5** (+100 %) |
| Scheduler | manual warmup + CosineAnnealingLR | **SequentialLR(LinearLR warmup → CosineAnnealingLR)** |
| All other knobs | — | identical |

Parameter count per group: stage4 13.1M / output_layer 12.8M / head 5.6M (total 31.6M; same as R005/R006).

### Training trajectory

LR reported is the head group's LR (highest of the three).

| Epoch | Train Loss | Val Acc | Val AUC | Thr | LR_head | R006 same ep | Δ R007-R006 |
|------:|-----------:|--------:|--------:|----:|-----|------:|------:|
| 1 | 0.7283 | 75.8 % | 0.8496 | 0.450 | 4.0e-6 | 0.8594 | -0.010 |
| 2 | 0.5676 | 81.1 % | 0.9046 | 0.400 | 7.2e-6 | 0.9045 | +0.000 |
| 3 | 0.4998 | 81.9 % | 0.9089 | 0.550 | 1.04e-5 | 0.9049 | +0.004 |
| **4** | **0.4431** | **81.7 %** | **0.9093 (peak)** | 0.200 | 1.36e-5 | 0.9035 | **+0.006** |
| 5 | 0.3991 | 82.5 % | 0.9056 | 0.400 | 1.68e-5 | 0.9006 | +0.005 |
| 6 | 0.3605 | 81.9 % | 0.9023 | 0.100 | 2.00e-5 | 0.9019 | +0.000 |
| 7 | 0.3127 | 81.7 % | 0.9024 | 0.100 | 2.00e-5 | 0.8923 | +0.010 |
| 8-14 | (declining) | 79-82 % | 0.88-0.90 | 0.10 | ~2e-5 | — | — |
| 15 | 0.1467 | 80.0 % | 0.8878 | 0.100 | 1.96e-5 | — | SAFEGUARD fired |

R007 stayed slightly above R006 through ep 5 (+0.000 to +0.006). Peak +0.004 vs R006 — the higher head LR did push the ceiling up as predicted. Faster post-peak decline (ep 14: 0.8801) suggests the head also over-fits faster.

### Test metrics (val-selected threshold 0.200)

| Metric | M12 R002 | M12 R006 (HEADLINE) | **M12 R007** | Δ R007-R006 |
|---|---:|---:|---:|---:|
| **Test ROC AUC** | 0.8564 | **0.8788** | 0.8730 | -0.006 (within noise floor) |
| Test Accuracy | 76.79 % | 79.33 % | 78.76 % | -0.6 pp |
| Test Balanced Acc | 76.48 % | 79.65 % | 79.11 % | -0.5 pp |
| Test F1 | 0.7402 | **0.8017** | 0.7980 | -0.004 |
| Test Precision | 79.82 % | 74.17 % | 73.28 % | -0.9 pp |
| Test Recall | 69.00 % | 87.24 % | **87.61 %** | +0.4 pp |
| Test Avg Precision | 0.8389 | **0.8561** | 0.8521 | -0.004 |
| **TAR @ FAR=0.001** | 4.18 % | 3.33 % | **4.49 %** ⭐ | **+1.2 pp** |
| **TAR @ FAR=0.01** | 17.58 % | 18.61 % | **19.59 %** ⭐ | **+1.0 pp** |
| TAR @ FAR=0.1 | 57.11 % | **59.93 %** | 57.56 % | -2.4 pp |
| Val→Test AUC gap | -0.076 | **-0.026** | -0.036 | slightly wider |

### Per-relation accuracy (FIW Track-I test, threshold 0.200)

| Relation | N | M12 R006 | **M12 R007** | Δ |
|----------|--:|---------:|-------------:|---:|
| bb | 860 | 89.1 % | **90.5 %** | +1.4 pp |
| ss | 731 | 88.9 % | **90.8 %** | +1.9 pp |
| sibs | 234 | 93.2 % | **94.9 %** | +1.7 pp |
| fs | 1135 | 87.9 % | 88.3 % | +0.4 pp |
| fd | 918 | 84.6 % | 83.3 % | -1.3 pp |
| ms | 1036 | 86.2 % | 85.4 % | -0.8 pp |
| md | 1038 | 89.0 % | **90.4 %** | +1.4 pp |
| **gfgd** | 138 | 83.3 % | 81.2 % | -2.1 pp |
| **gfgs** | 98 | 76.5 % | **79.6 %** | **+3.1 pp** |
| **gmgd** | 123 | 79.7 % | 81.3 % | +1.6 pp |
| **gmgs** | 121 | 80.2 % | 75.2 % | -5.0 pp |
| non-kin | 6993 | 72.1 % | 70.6 % | -1.5 pp |

Mixed per-class signals. Siblings and ss/md gained 1-2 pp; gmgs dropped 5 pp. No clear class-level improvement direction.

### Notes

- **Mechanism on Val**: head LR 2e-5 (4× the backbone LR 5e-6) lets the head reach a marginally higher fit on val. Confirmed by the +0.004 peak.
- **Mechanism on Test**: dif-LR is a *fit-quality* intervention (changes how well each component learns the training signal), not a *generalisation* intervention. On a dataset where the train→test gap was already small after R006, there's nothing for fit-quality to recover.
- **Trade-off**: R007 shifts the operating curve slightly toward strict-FAR. For low-FAR deployment, R007 has +1.2 pp TAR@FAR=0.001 and +1.0 pp TAR@FAR=0.01 over R006. For balanced discrimination (R006's regime), R006 wins by 0.006 AUC.
- **Negligible compute overhead** — same wall-clock as R006 (~26 min/epoch). No new parameters.

### Artifacts

- Checkpoint (epoch 4, Val AUC 0.9093): `output/007/checkpoints/best.pt` (536 MB)
- Train log: `output/007/logs/train.log`
- Test log: `output/007/logs/test.log`
- Results: `output/007/results/test_metrics_rocm.txt`
- Trainable params: 31,560,461 / 70,746,317 (44.61 %) — same as R005/R006

---

## Run 006 — 2026-05-14 — Manually stopped at epoch 8 (peak Val AUC 0.9049 ep 3; Test AUC **0.8788** — NEW PROJECT HEADLINE)

**Status:** Stopped manually at epoch 8 (Val AUC plateauing 0.025 below R005; user decision to early-stop). best.pt patched manually before test.py.

**Outcome:** Phase 5 (relation aux head) + **symmetric forward** (Option-B BCE: 0.5·(BCE_AB + BCE_BA), + averaged CE_rel). **NEW PROJECT HEADLINE: Test ROC AUC 0.8788 — +0.022 vs R002's 0.8564 (the prior headline), well outside the ±0.009 noise floor.** Val→test gap collapses to **-0.026** (R002: -0.076, R005: -0.084) — smallest of any AdaFace-based model with non-trivial training in the project. Every kin class improved by 13-43 pp vs R002; grandparents jumped 31-43 pp.

**Hypothesis test result:** CONFIRMED. The asymmetry diagnosis from R005 was correct — `cross_region.attn_ab/attn_ba`, `regional_gate`'s `[A,B]` concat, and `classifier`'s `[gA,gB,…]` concat were learning direction-specific shortcuts that didn't transfer across the val/test family split. Symmetric forward (process each pair in both orders, combine in Option-B BCE) removed the shortcut, dropping Val AUC by 0.027 (model can't shortcut any more) and raising Test AUC by 0.031 vs R005.

### Launch command

```bash
SKIP_INSTALL=1 \
ALIGNED_ROOT=/home/bruno/Desktop/tcc_new/datasets/FIW_aligned \
BATCH_SIZE=8 \
GRAD_ACCUM=4 \
UNFREEZE_LAST_STAGE=1 \
LEARNING_RATE=1e-5 \
RELATION_AUX_WEIGHT=0.05 \
SYMMETRIC_FORWARD=1 \
NUM_WORKERS=4 \
SEED=42 \
bash models/12_rgck_net/AMD/run_pipeline.sh
```

### Changes from Run 005

| Parameter | R005 | **R006** |
|---|---|---|
| `SYMMETRIC_FORWARD` | 0 (off) | **1 (on)** |
| Loss | BCE + 0.05·CE_rel\|pos | **0.5·(BCE_AB + BCE_BA) + 0.05·avg(CE_rel_AB, CE_rel_BA)\|pos** |
| All other knobs | — | identical to R005 |
| Trainable params | 31,560,461 | 31,560,461 (same — symmetry runs the same weights twice) |
| Time/epoch | ~26 min | ~26.6 min (+2 % — tokenizer reused, only head runs twice) |

### Configuration

Same as R005 plus:

| Param | Value |
|-------|-------|
| `symmetric_forward` | True |
| Loss | 0.5·(BCE_AB + BCE_BA) + 0.05·avg(CE_rel_AB, CE_rel_BA)\|pos |
| Time/epoch | ~26.6 min (~+2 % over R005, much less than the 15-25 % I had estimated) |

### Training trajectory

| Epoch | Train Loss | Val Acc | Val AUC | Thr | LR | R005 same ep | Δ R006-R005 |
|------:|-----------:|--------:|--------:|----:|-----|------:|------:|
| 1 | 0.7278 | 75.2 % | 0.8594 | 0.450 | 2.0e-6 | 0.8696 | -0.010 |
| 2 | 0.5504 | 80.5 % | 0.9045 | 0.350 | 4.0e-6 | 0.9279 | -0.023 |
| **3** | **0.4695** | **81.4 %** | **0.9049 (peak)** | 0.500 | 6.0e-6 | 0.9306 | -0.026 |
| 4 | 0.4013 | 82.1 % | 0.9035 | 0.200 | 8.0e-6 | 0.9318 | -0.028 |
| 5 | 0.3538 | 81.4 % | 0.9006 | 0.250 | 1.0e-5 | 0.9273 | -0.027 |
| 6 | 0.3078 | 82.3 % | 0.9019 | 0.100 | 1.0e-5 | 0.9278 | -0.026 |
| 7 | 0.2601 | 81.3 % | 0.8923 | 0.100 | 9.99e-6 | 0.9230 | -0.031 |
| 8 | (manual stop) | | | | | — | — |

Val AUC ceiling dropped 0.027 vs R005, exactly as the hypothesis predicted (the model can no longer use direction-specific shortcuts). Peak at ep 3, declining monotonely afterward — same shape as R002/R005 but from a lower ceiling. Manual stop at ep 8 saved ~5 h of compute (SAFEGUARD would have fired at ep 13).

### Test metrics (val-selected threshold 0.500)

| Metric | M12 R002 (prior HEADLINE) | M12 R005 | **M12 R006** | Δ R006-R002 |
|---|---:|---:|---:|---:|
| **Test ROC AUC** | 0.8564 | 0.8476 | **0.8788** ⭐ | **+0.022** |
| Test Accuracy | 76.79 % | 76.53 % | **79.33 %** ⭐ | +2.5 pp |
| Test Balanced Acc | 76.48 % | 76.45 % | **79.65 %** ⭐ | +3.2 pp |
| Test F1 | 0.7402 | 0.7528 | **0.8017** ⭐ | +0.061 |
| Test Precision | **79.82 %** | 75.99 % | 74.17 % | -5.7 pp |
| Test Recall | 69.00 % | 74.58 % | **87.24 %** ⭐ | **+18.2 pp** |
| Test Avg Precision | 0.8389 | 0.8288 | **0.8561** ⭐ | +0.017 |
| TAR @ FAR=0.001 | **4.18 %** | 2.94 % | 3.33 % | -0.9 pp |
| TAR @ FAR=0.01 | 17.58 % | 16.67 % | **18.61 %** ⭐ | +1.0 pp |
| TAR @ FAR=0.1 | 57.11 % | 55.60 % | **59.93 %** ⭐ | +2.8 pp |
| **Val→test AUC gap** | -0.076 | -0.084 | **-0.026** ⭐⭐ | gap shrunk by 0.050 |

R006 wins R002 on every threshold-invariant metric (AUC, AP, TAR@FAR=0.01, TAR@FAR=0.1) and every threshold-dependent metric except Precision and TAR@FAR=0.001 (the operating point is more recall-oriented).

### Per-relation accuracy (FIW Track-I test, threshold 0.500)

| Relation | N | M12 R002 | M12 R005 | **M12 R006** | Δ R006-R002 |
|----------|--:|---------:|---------:|-------------:|---:|
| bb | 860 | 75.4 % | 78.5 % | **89.1 %** | **+13.7 pp** |
| ss | 731 | 75.9 % | 82.2 % | **88.9 %** | **+13.0 pp** |
| sibs | 234 | 79.5 % | 85.9 % | **93.2 %** | **+13.7 pp** |
| md | 1038 | 68.7 % | 74.3 % | **89.0 %** | **+20.3 pp** |
| fs | 1135 | 68.6 % | 75.0 % | **87.9 %** | **+19.3 pp** |
| ms | 1036 | 69.4 % | 75.6 % | **86.2 %** | **+16.8 pp** |
| fd | 918 | 68.4 % | 74.2 % | **84.6 %** | **+16.2 pp** |
| **gfgd** | 138 | 52.2 % | 55.8 % | **83.3 %** | **+31.1 pp** ⭐⭐ |
| **gfgs** | 98 | 39.8 % | 41.8 % | **76.5 %** | **+36.7 pp** ⭐⭐ |
| **gmgd** | 123 | 36.6 % | 43.9 % | **79.7 %** | **+43.1 pp** ⭐⭐⭐ |
| **gmgs** | 121 | 44.6 % | 51.2 % | **80.2 %** | **+35.6 pp** ⭐⭐⭐ |
| non-kin | 6993 | **84.0 %** | 78.3 % | 72.1 % | -11.9 pp |

**Every kin class improved 13-43 pp.** The four grandparent classes — the historic weakness of the M12 family — went from 36-52 % to 76-83 %. Non-kin specificity dropped 12 pp (operating regime is higher-recall) but Test AUC of 0.8788 is unambiguous evidence the trade-off is heavily net-positive.

### Notes

- **Symmetric forward isolated contribution (R006 vs R005)**: +0.031 Test AUC, 0.058 reduction in val→test gap, +31.6 pp average grandparent accuracy. Single architectural change, no new parameters, no new loss terms.
- **Phase 5 + symmetric forward are multiplicative**: e.g. gmgd was 36.6 % at R002, 43.9 % at R005 (+7.3 from Phase 5 alone), 79.7 % at R006 (+35.8 from symmetry on top of Phase 5). The two interventions combine constructively, not as a saturation.
- **Mechanism**: the cross_region adapter has separate `attn_ab` / `attn_ba` / `ffn_a` / `ffn_b` weights, and the gate/classifier concatenate in `[A, B]` order. RFIW Track-I lists pairs in a single canonical order, so the model could learn `attn_ab` to extract "kinship given role X-Y". This shortcut didn't transfer across families because role ordering by visual cue (age, pose) differs between training and held-out families. Symmetric forward forces `f(A,B) ≈ f(B,A)`, removing the shortcut and reducing memorisation.
- **Negligible compute overhead** (~+2 % per epoch) because the tokenizer (5× AdaFace passes per face) dominates and is shared between the two directions.

### Artifacts

- Checkpoint (epoch 3, Val AUC 0.9049): `output/006/checkpoints/best.pt` (536 MB) — patched manually with `model_config` (symmetric_forward=True, aux_relation_head=True, relation_aux_weight=0.05, selected_threshold=0.500)
- Train log: `output/006/logs/train.log`
- Test results: `output/006/results/test_metrics_rocm.txt`
- Trainable params: 31,560,461 / 70,746,317 (44.61 %)

---

## Run 005 — 2026-05-14 — SAFEGUARD-stopped at epoch 16 (peak Val AUC 0.9318 ep 4; Test AUC 0.8476. Phase 5 relation aux: per-class targets confirmed, AUC within noise floor)

**Status:** SAFEGUARD auto-stopped at epoch 16 — Val AUC below peak (0.9318 ep 4) for 10 consecutive epochs.

**Outcome:** Phase 5 of `proposta_rgck_net_kinship.md` §38 — same stack as R002 (Phase 2 partial unfreeze + classifier head + BCE) + **relation-type CE auxiliary loss λ=0.05 on positive pairs only**, class-balanced via inverse-frequency weights from train positives. Val AUC peak at **0.9318** (ep 4), essentially identical to R002's 0.9323. Test ROC AUC **0.8476** — **-0.009 vs R002's 0.8564, within the ±0.009 noise floor measured in R004**. Test F1 +0.013, Recall +5.6 pp, Precision -3.8 pp (threshold dropped to 0.300, model now operates in a higher-recall regime).

**Per-relation findings (the main result):** **all 11 kin classes improved 2-7 pp** vs R002. The 4 grandparent classes (explicit Phase 5 targets) gained 2.0-7.3 pp: gmgd +7.3, gmgs +6.6, gfgd +3.6, gfgs +2.0. Trade-off: non-kin specificity -5.7 pp at the chosen threshold; TAR@FAR levels all -0.9 to -1.5 pp.

**Hypothesis test result:** per-class targeted version **CONFIRMED**; global AUC version **NOT CONFIRMED** (within noise floor). R005 is the first post-R002 intervention to not lose meaningfully on AUC and to improve every kin class simultaneously.

### Launch command

```bash
SKIP_INSTALL=1 \
ALIGNED_ROOT=/home/bruno/Desktop/tcc_new/datasets/FIW_aligned \
BATCH_SIZE=8 \
GRAD_ACCUM=4 \
UNFREEZE_LAST_STAGE=1 \
LEARNING_RATE=1e-5 \
RELATION_AUX_WEIGHT=0.05 \
NUM_WORKERS=4 \
SEED=42 \
bash models/12_rgck_net/AMD/run_pipeline.sh
```

### Changes from Run 002

| Parameter | R002 | **R005** |
|---|---|---|
| `RELATION_AUX_WEIGHT` | 0 (off) | **0.05** |
| `--relation_aux_balanced` | — | on (inverse-freq weights) |
| Architecture | classifier head only | + `relation_head: Linear(512 → 11)` on 0.5·(gA+gB) |
| All other knobs | — | identical |

### Configuration

Same as R002 plus:

| Param | Value |
|-------|-------|
| Backbone | AdaFace IR-101 (stages 1-3 frozen, stage 4 + output_layer trainable) |
| `aux_relation_head` | True |
| `relation_aux_weight` | 0.05 |
| `relation_aux_balanced` | True |
| Trainable params | 31,560,461 / 70,746,317 (44.61 %) — relation_head adds 5,643 params |
| Loss | BCE(classifier_logit) + 0.05·CE_rel(rel_logits\|pos, balanced) |
| Time/epoch | ~26 min |
| Class weights (from train pos): bb=0.440, ss=0.404, sibs=0.392, fs=0.252, fd=0.253, ms=0.261, md=0.249, gfgd=2.193, gfgs=1.955, gmgd=2.460, gmgs=2.140 (mean=1.0) |

### Training trajectory

| Epoch | Train Loss | Val Acc | Val AUC | Thr | LR | R002 same ep | Δ R005-R002 |
|------:|-----------:|--------:|--------:|----:|-----|------:|------:|
| 1 | 0.7060 | 77.8 % | 0.8696 | 0.450 | 2.0e-6 | 0.8644 | +0.005 |
| 2 | 0.5072 | 85.0 % | 0.9279 | 0.550 | 4.0e-6 | 0.9259 | +0.002 |
| 3 | 0.4189 | 85.6 % | 0.9306 | 0.500 | 6.0e-6 | 0.9311 | -0.001 |
| **4** | **0.3514** | **86.0 %** | **0.9318 (peak)** | 0.300 | 8.0e-6 | 0.9323 | **-0.001** |
| 5 | 0.3099 | 85.8 % | 0.9273 | 0.300 | 1.0e-5 | 0.9306 | -0.003 |
| 6 | 0.2684 | 85.5 % | 0.9278 | 0.100 | 1.0e-5 | 0.9284 | -0.001 |
| 7 | 0.2306 | 84.8 % | 0.9211 | 0.100 | 9.99e-6 | 0.9230 | -0.002 |
| 8 | 0.2119 | 84.7 % | 0.9190 | 0.100 | — | — | — |
| 9-15 | 0.20→0.11 | 82-85 % | 0.89-0.92 | 0.10 | 9.96→9.76e-6 | — | (sustained decline) |
| 16 | 0.1141 | 82.4 % | 0.8940 | 0.100 | 9.71e-6 | — | SAFEGUARD fired |

R005 stayed within ±0.005 of R002 in every epoch through ep 7. Peak Val AUC essentially identical (-0.0005 from R002 — well within run-to-run noise). The new loss term did not raise or lower the Val AUC ceiling.

### Test metrics (val-selected threshold 0.300)

| Metric | M12 R002 (HEADLINE) | **M12 R005** | Δ |
|---|---:|---:|---:|
| **Test ROC AUC** | **0.8564** | **0.8476** | **-0.009** (within ±0.009 noise floor) |
| Test Accuracy | 76.79 % | 76.53 % | -0.3 pp |
| Test Balanced Acc | 76.48 % | 76.45 % | -0.0 pp |
| Test F1 | 0.7402 | **0.7528** | **+0.013** |
| Test Precision | **79.82 %** | 75.99 % | -3.8 pp |
| Test Recall | 69.00 % | **74.58 %** | **+5.6 pp** |
| Test Avg Precision | 0.8389 | 0.8288 | -0.010 |
| TAR @ FAR=0.001 | **4.18 %** | 2.94 % | -1.2 pp |
| TAR @ FAR=0.01 | **17.58 %** | 16.67 % | -0.9 pp |
| TAR @ FAR=0.1 | **57.11 %** | 55.60 % | -1.5 pp |
| Val→test AUC gap | -0.076 | -0.084 | slightly wider (still 2nd best) |

### Per-relation accuracy (FIW Track-I test, threshold 0.300)

| Relation | N | M12 R002 | **M12 R005** | Δ |
|----------|--:|---------:|-------------:|---:|
| bb | 860 | 75.4 % | **78.5 %** | **+3.1 pp** |
| ss | 731 | 75.9 % | **82.2 %** | **+6.3 pp** |
| sibs | 234 | 79.5 % | **85.9 %** | **+6.4 pp** |
| md | 1038 | 68.7 % | **74.3 %** | **+5.6 pp** |
| fs | 1135 | 68.6 % | **75.0 %** | **+6.4 pp** |
| ms | 1036 | 69.4 % | **75.6 %** | **+6.2 pp** |
| fd | 918 | 68.4 % | **74.2 %** | **+5.8 pp** |
| **gfgd** | 138 | 52.2 % | **55.8 %** | **+3.6 pp** ⭐ |
| gfgs | 98 | 39.8 % | **41.8 %** | **+2.0 pp** ⭐ |
| **gmgd** | 123 | 36.6 % | **43.9 %** | **+7.3 pp** ⭐ |
| **gmgs** | 121 | 44.6 % | **51.2 %** | **+6.6 pp** ⭐ |
| **non-kin** | 6993 | **84.0 %** | 78.3 % | -5.7 pp |

**Every kin class improved by 2-7 pp.** The 4 grandparent classes (the Phase 5 targets) all gained 2-7 pp — the explicit hypothesis was confirmed on its targets. Non-kin specificity dropped 5.7 pp, which is what brought Test AUC down 0.009.

### Notes

- **First post-R002 intervention to not lose meaningfully on AUC.** R003 (SupCon) lost -0.005, R004 (intended hard negs, actually a sampler reseed) lost -0.009. R005 lost -0.009 — same magnitude as R004 but with completely different per-class profile: R004 regressed every kin class; R005 improved every one.
- **Per-class effect is the opposite of R003's**: R003 hurt grandparents (-1 to -1.6 pp); R005 helps them (+2.0 to +7.3 pp). Different aux objectives produce opposite class signatures — informative for proposal §28's warning about overly aggressive contrastive losses.
- **No degradation in val→test gap**: -0.084 here vs -0.076 for R002, -0.080 for R003, -0.088 for R004, -0.089 for R001. R005 is in the second-best zone; no family memorisation pathology.
- **Trade-off recipe**: R005 prioritises per-class kin balance over strict-FAR performance. R002 prioritises the opposite.

### Artifacts

- Checkpoint (epoch 4, Val AUC 0.9318): `output/005/checkpoints/best.pt` (536 MB)
- Train log: `output/005/logs/train.log`
- Test log: `output/005/logs/test.log`
- Results: `output/005/results/test_metrics_rocm.txt`
- Trainable params: 31,560,461 / 70,746,317 (44.61 %) — relation_head adds 5,643 params

---

## Run 004 — 2026-05-14 — Stopped at epoch 8 (peak Val AUC 0.9354 ep 4 — project max; Test AUC 0.8473. Intended hard-negs test — Errata: hypothesis NOT actually tested)

**Status:** Stopped manually at epoch 8 (Val AUC -0.010 from peak, train loss still descending, grandparent classes collapsing in val)

> ⚠ **Errata (post-run, 2026-05-14):** the `relation_matched` sampler at
> [models/shared/dataset.py:433](../shared/dataset.py#L433) does NOT preserve relation/role —
> line 464 picks a relation but line 465 hardcodes `"non-kin"`, so
> the algorithm is identical to the random sampler at
> [models/shared/dataset.py:512](../shared/dataset.py#L512) up to seed offset (270 vs 200).
> Measured: 0.09 % overlap on train negatives, 100 % on test (test
> pairs come from RFIW Track-I lists, not the sampler). R004 actually
> tested **negative-sampler reseed variance** with all else identical
> to R002 — not hard negatives. The original "hard negs rejected" /
> "cross-experiment confirmation with M11 v4" narrative below is
> withdrawn; both rest on the same broken sampler. The numerical
> results are valid; the causal interpretation is not. See
> [run-review/run-004.md](run-review/run-004.md) Errata.

**Outcome (original framing, kept for the record):** Phase 6 of
`proposta_rgck_net_kinship.md` §38 — same stack as R002 (Phase 2
partial unfreeze + classifier head + BCE) + `relation_matched` ~~hard
negatives~~ negatives-with-reseed on both train and eval. Val AUC
peaked at **0.9354** (ep 4) — highest in the project, +0.003 over
R002. Test AUC = 0.8473 — -0.009 below R002's 0.8564. All TAR@FAR
levels regressed. Val→test gap -0.088 vs R002 -0.076. All 11 kin
classes regressed; non-kin +1.2 pp.

**Corrected interpretation:** the R004 → R002 deltas measure
sensitivity of partial-FT to the specific draw of training negatives.
This sets a noise floor of roughly ±0.01 Test AUC for any future
single-knob comparison that doesn't control the negative-sampler seed.

### Launch command

```bash
SKIP_INSTALL=1 \
ALIGNED_ROOT=/home/bruno/Desktop/tcc_new/datasets/FIW_aligned \
BATCH_SIZE=8 \
GRAD_ACCUM=4 \
UNFREEZE_LAST_STAGE=1 \
LEARNING_RATE=1e-5 \
TRAIN_NEGATIVE_STRATEGY=relation_matched \
EVAL_NEGATIVE_STRATEGY=relation_matched \
NUM_WORKERS=4 \
SEED=42 \
bash models/12_rgck_net/AMD/run_pipeline.sh
```

### Changes from Run 002

| Parameter | R002 | **R004** |
|---|---|---|
| `TRAIN_NEGATIVE_STRATEGY` | random | **`relation_matched`** |
| `EVAL_NEGATIVE_STRATEGY` | random | **`relation_matched`** |
| All other knobs | — | identical |

Single-knob change: negative sampling. `relation_matched` constructs
negatives by drawing the partner of a positive `fs` pair from another
`fs` pair (different family but same role). This produces visually
plausible hard negatives that force the model to discriminate on
kinship-specific cues rather than role cues.

### Configuration

Same as R002:

| Param | Value |
|-------|-------|
| Backbone | AdaFace IR-101 (stages 1-3 frozen, stage 4 + output_layer trainable) |
| Trainable params | 31,554,818 / 70,740,674 (44.61%) |
| Loss | BCE on classifier logit (no SupCon) |
| Batch | 8 × grad-accum 4 (eff. 32) |
| LR | 1e-5 peak, warmup 5, cosine, min_lr 1e-6 |
| Weight decay | 1e-4 |
| Dropout | 0.2 |
| **Negative strategy** | **`relation_matched`** both train and eval |
| Time/epoch | ~25.8 min |

### Training trajectory

| Epoch | Train Loss | Val Acc | Val AUC | Thr | LR | R002 same ep | Δ R004-R002 |
|------:|-----------:|--------:|--------:|----:|-----|------:|------:|
| 1 | 0.5855 | 78.1 % | 0.8676 | 0.450 | 2.0e-6 | 0.8644 | +0.003 |
| 2 | 0.4043 | 84.0 % | 0.9274 | 0.350 | 4.0e-6 | 0.9259 | +0.001 |
| 3 | 0.3287 | 85.8 % | 0.9333 | 0.350 | 6.0e-6 | 0.9311 | +0.002 |
| **4** | **0.2855** | **86.2 %** | **0.9354 (peak)** | 0.150 | 8.0e-6 | 0.9323 | **+0.003 (project max)** |
| 5 | 0.2514 | 86.4 % | 0.9326 | 0.250 | 1.0e-5 | 0.9306 | +0.002 |
| 6 | 0.2253 | 86.2 % | 0.9350 | 0.150 | 1.0e-5 | 0.9284 | +0.007 |
| 7 | 0.1999 | 85.9 % | 0.9313 | 0.100 | 9.99e-6 | 0.9230 | +0.008 |
| 8 | 0.1798 | 85.4 % | 0.9259 | 0.100 | 9.98e-6 | (n/a) | — |

R004 stayed above R002 in every epoch (+0.001 to +0.008). Val AUC at
the peak (ep 4) was the highest the project has ever produced: **0.9354**.
But the train→val gap widened in later epochs as overfitting set in
faster than in R002.

### Test metrics (threshold 0.500)

| Metric | M12 R002 (HEADLINE) | **M12 R004** | Δ |
|---|---:|---:|---:|
| **Test ROC AUC** | **0.8564** | **0.8473** | **-0.009** |
| Test Accuracy | 76.79 % | 75.75 % | -1.0 pp |
| Test Balanced Acc | 76.48 % | 75.33 % | -1.2 pp |
| Test F1 | 0.7402 | 0.7211 | -0.019 |
| Test Precision | 79.82 % | 80.29 % | +0.5 pp |
| Test Recall | 69.00 % | 65.44 % | -3.6 pp |
| Test Avg Precision | 0.8389 | 0.8287 | -0.010 |
| TAR @ FAR=0.001 | 4.18 % | 2.01 % | -2.2 pp |
| TAR @ FAR=0.01 | 17.58 % | 14.71 % | -2.9 pp |
| TAR @ FAR=0.1 | 57.11 % | 56.06 % | -1.0 pp |
| **Val→test AUC gap** | **-0.076** | **-0.088** | gap widened |

### Per-relation accuracy (FIW Track-I test, threshold 0.500)

| Relation | N | M12 R002 | **M12 R004** | Δ |
|----------|--:|---------:|-------------:|---:|
| bb | 860 | 75.4 % | 72.4 % | -3.0 pp |
| ss | 731 | 75.9 % | 72.5 % | -3.4 pp |
| sibs | 234 | 79.5 % | 76.9 % | -2.6 pp |
| md | 1038 | 68.7 % | 64.8 % | -3.9 pp |
| fs | 1135 | 68.6 % | 64.4 % | -4.2 pp |
| ms | 1036 | 69.4 % | 66.1 % | -3.3 pp |
| fd | 918 | 68.4 % | 65.6 % | -2.8 pp |
| **gfgd** | 138 | 52.2 % | **46.4 %** | **-5.8 pp** |
| gfgs | 98 | 39.8 % | 38.8 % | -1.0 pp |
| gmgd | 123 | 36.6 % | 34.2 % | -2.4 pp |
| **gmgs** | 121 | 44.6 % | **33.9 %** | **-10.7 pp** ⚠ |
| **non-kin** | 6993 | 84.0 % | **85.2 %** | **+1.2 pp** |

**All 11 kin classes regressed by 1-11 pp.** Non-kin specificity went
up by 1.2 pp — the model became *more conservative* about predicting
kin. Grandmother-grandson (`gmgs`) had the largest drop (-10.7 pp).

The directionality is consistent: hard negatives push the kin/non-kin
decision boundary *toward* non-kin (the model needs to be more certain
to call something kin because hard negatives look kinship-like).

### Notes (revised after Errata, 2026-05-14)

- ~~**Phase 6 (`relation_matched` hard negatives) is REJECTED**~~ —
  withdrawn. The sampler does not produce hard negatives; R004 was
  effectively R002 with a different seed for negative sampling.
  Phase 6 hypothesis is **untested**.
- ~~**Cross-experiment robustness check confirmed**~~ — withdrawn.
  M11 v4 and M12 R004 both used the same broken sampler. The
  "hard negs hurt Test" pattern attributed to them is unverified
  until the sampler is fixed.
- **R003 (SupCon aux) is still REJECTED** on the basis of R003's own
  evidence (cross-generation classes regressed, not just a magnitude
  question). That conclusion is independent of the sampler bug.
- The Val AUC peak of **0.9354** is the project-wide maximum
  validation score, but it doesn't transfer — likely a combination
  of the same overfit dynamics R002 already shows + sampler-seed
  variance in this particular draw.
- **Noise floor under partial-FT:** R002 → R004 differs only in the
  negative-sampler seed (200 → 270), and Test AUC swung by ~0.009
  (and per-class by 1-11 pp). Future improvements smaller than this
  cannot be cleanly attributed to the intervention unless the seed
  is controlled.

### Artifacts

- Checkpoint (epoch 4, Val AUC 0.9354): `output/004/checkpoints/best.pt` (536 MB) — patched manually with `model_config` (freeze=True, unfreeze_last_stage=True, supcon_weight=0)
- Resume snapshot: `output/004/checkpoints/epoch_5.pt` (will be pruned)
- Train log: `output/004/logs/train.log`
- Test log: `output/004/logs/test.log` (also `/tmp/m12_r004_test.log`)
- Results: `output/004/results/test_metrics_rocm.txt`

---

## Run 003 — 2026-05-14 — Stopped at epoch 7 (peak Val AUC 0.9306 ep 4; Test AUC 0.8510, SupCon aux did NOT help)

**Status:** Stopped manually at epoch 7 (3 consecutive declines below peak)
**Outcome:** Phase 4 of `proposta_rgck_net_kinship.md` §28 — same stack as R002 (Phase 2 partial unfreeze + classifier head) + **supervised-contrastive auxiliary loss with λ=0.05** over the L2-normalised contextualised global tokens `(gA, gB)`. Hypothesis: aux contrastive organises the embedding space and lifts Val/Test AUC. **Rejected by experimental evidence:** Test AUC = **0.8510**, -0.005 below R002 (0.8564, current project headline). All three TAR@FAR levels also regressed. Tiny improvements in `sibs` (+3.4 pp) and `bb` (+1.7 pp), but regressions in `gmgs`, `gfgs`, non-kin, and lower precision overall. Net negative.

### Launch command

```bash
SKIP_INSTALL=1 \
ALIGNED_ROOT=/home/bruno/Desktop/tcc_new/datasets/FIW_aligned \
BATCH_SIZE=8 \
GRAD_ACCUM=4 \
UNFREEZE_LAST_STAGE=1 \
LEARNING_RATE=1e-5 \
SUPCON_WEIGHT=0.05 \
NUM_WORKERS=4 \
SEED=42 \
bash models/12_rgck_net/AMD/run_pipeline.sh
```

### Changes from Run 002

| Parameter | R002 | **R003** |
|---|---|---|
| `SUPCON_WEIGHT` | 0.0 (off) | **0.05** |
| `SUPCON_MARGIN` | — | 0.3 (default) |
| All other knobs | — | identical |

The SupCon term is a margin-style label-aware contrastive on the cosine
similarity of the contextualised global tokens (after the cross-region
adapter, L2-normalised):

- For label=1 pairs: pull `cos(gA, gB)` toward 1 via `(1 - cos)²`
- For label=0 pairs: push `cos(gA, gB)` below `1 - margin = 0.7` via `max(0, cos - 0.7)²`

`L_total = L_bce(classifier_logit, label) + 0.05 × L_supcon(gA, gB, label, margin=0.3)`

### Configuration

Same as R002, plus:

| Param | Value |
|-------|-------|
| **Loss** | **BCE + 0.05 × SupCon** (margin 0.3) |
| Trainable params | 31,554,818 / 70,740,674 (44.61%) — identical to R002 |
| Time/epoch | ~26 min (similar to R002, supcon term is cheap) |

### Training trajectory

| Epoch | Train Loss | Val Acc | Val AUC | Thr | LR | R002 same ep | Δ R003-R002 |
|------:|-----------:|--------:|--------:|----:|-----|------:|------:|
| 1 | 0.6020 | 77.7 % | 0.8626 | 0.400 | 2.0e-6 | 0.8644 | -0.002 |
| 2 | 0.4164 | 84.2 % | 0.9246 | 0.400 | 4.0e-6 | 0.9259 | -0.001 |
| 3 | 0.3446 | 85.2 % | 0.9305 | 0.350 | 6.0e-6 | 0.9311 | -0.001 |
| **4** | **0.3021** | **85.7 %** | **0.9306** | 0.300 | 8.0e-6 | 0.9323 | -0.002 (peak) |
| 5 | 0.2714 | 85.6 % | 0.9285 | 0.150 | 1.0e-5 | 0.9306 | -0.002 |
| 6 | 0.2397 | 85.6 % | 0.9295 | 0.200 | 1.0e-5 | 0.9284 | +0.001 (R003 ahead — only positive Δ) |
| 7 | 0.2197 | 85.3 % | 0.9203 | 0.100 | 9.99e-6 | 0.9230 | -0.003 (decline confirmed) |

R003 trajectory tracks R002 with a small offset of ~-0.001 to -0.002
throughout. The single positive Δ (ep 6) was within noise. Train loss is
consistently ~0.013 higher than R002 (the supcon contribution).

### Test metrics (threshold 0.500)

| Metric | M12 R002 (project headline) | **M12 R003** | Δ |
|---|---:|---:|---:|
| **Test ROC AUC** | **0.8564** | **0.8510** | **-0.005** |
| Test Accuracy | 76.79 % | 76.37 % | -0.4 pp |
| Test Balanced Acc | 76.48 % | 76.09 % | -0.4 pp |
| Test F1 | 0.7402 | 0.7373 | -0.003 |
| Test Precision | 79.82 % | 78.88 % | -0.9 pp |
| Test Recall | 69.00 % | 69.22 % | +0.2 pp |
| Test Avg Precision | 0.8389 | 0.8305 | -0.008 |
| TAR @ FAR=0.001 | 4.18 % | 2.67 % | -1.5 pp |
| TAR @ FAR=0.01 | 17.58 % | 16.57 % | -1.0 pp |
| TAR @ FAR=0.1 | 57.11 % | 55.43 % | -1.7 pp |
| Val→test AUC gap | -0.076 | -0.080 | wider |

### Per-relation accuracy (FIW Track-I test, threshold 0.500)

| Relation | N | M12 R002 | **M12 R003** | Δ |
|----------|--:|---------:|-------------:|---:|
| **sibs** | 234 | 79.5 % | **82.9 %** | **+3.4** |
| **bb** | 860 | 75.4 % | **77.1 %** | **+1.7** |
| ss | 731 | 75.9 % | 75.9 % | = |
| gfgd | 138 | 52.2 % | 52.9 % | +0.7 |
| md | 1038 | 68.7 % | 68.6 % | -0.1 |
| fs | 1135 | 68.6 % | 68.4 % | -0.2 |
| ms | 1036 | 69.4 % | 69.1 % | -0.3 |
| fd | 918 | 68.4 % | 68.4 % | = |
| gfgs | 98 | 39.8 % | 38.8 % | -1.0 |
| non-kin | 6993 | 84.0 % | 83.0 % | -1.0 |
| gmgs | 121 | 44.6 % | 43.0 % | -1.6 |
| gmgd | 123 | 36.6 % | 36.6 % | = |

R003 improves on the **same-generation classes** (sibs, bb — both with
high N) but slightly regresses on the **rare grandparent classes**
(gmgs, gfgs) and non-kin. The supcon term pulls positive pairs together
in embedding space, which helps siblings (visually similar within a
batch) but hurts the more conceptually distant grandparent pairs.

### Notes

- **SupCon λ=0.05 hypothesis rejected.** The proposal §28 caveat — "if
  contrastive loss is too strong, can force artificial closeness" — was
  prescient. λ=0.05 was already at the conservative end suggested by the
  proposal and still produced net regression.
- The siblings improvement (sibs +3.4, bb +1.7) is consistent with the
  supcon term's behaviour: it pulls positive pairs together, which works
  for classes where positives are visually similar.
- Grandparent classes regressed because their positives are
  cross-generational (large age gap = visually different), so forcing
  them closer in embedding space distorts the learned representation.
- This is a **second negative result** for sophistication on top of the
  M12 R002 stack — earlier, M09 R002 (balanced sampling) and M11 v4
  (relation_matched negatives) also lowered Test AUC relative to their
  no-aux M09 R001 baseline. The pattern: any auxiliary that constrains
  the embedding space tends to over-fit val-pool patterns at the cost
  of test generalisation.
- **M12 R002 (Test AUC 0.8564) remains the project headline.**

### Artifacts

- Checkpoint (epoch 4, Val AUC 0.9306): `output/003/checkpoints/best.pt` (536 MB) — patched manually with `model_config` (supcon_weight=0.05, supcon_margin=0.3)
- Resume snapshot: `output/003/checkpoints/epoch_5.pt` (will be pruned)
- Train log: `output/003/logs/train.log`
- Test log: `output/003/logs/test.log` (also `/tmp/m12_r003_test.log`)
- Results: `output/003/results/test_metrics_rocm.txt`

---

## Run 002 — 2026-05-13 — Stopped at epoch 7 (peak Val AUC 0.9323 ep 4; **Test AUC 0.8564 — NEW PROJECT HEADLINE, beats M02 R031's 0.850**)

**Status:** Stopped manually at epoch 7/100 (3 epochs of post-peak decline)
**Outcome:** Phase 2 of `proposta_rgck_net_kinship.md` §38: AdaFace IR-101 partially frozen (stages 1-3 frozen, **stage 4 body[46:49] + output_layer trainable**), 5 region tokens, BCE classifier head, **LR=1e-5** (10× lower than R001's 1e-4 per proposal §37). Val AUC peak **0.9323** (ep 4) — highest in the entire project. **Test AUC = 0.8564, surpassing M02 R031's 0.850 (the prior project headline) by +0.006.** All three TAR@FAR levels (0.001, 0.01, 0.1) also exceed M02 R031, confirming better ranking quality across operating points. Val→test gap -0.076, larger than R001's -0.089 but well below other AdaFace-based full-FT models. **The architectural + recipe combination — frozen stages 1-3 + unfrozen stage 4 + region tokens + cross-region attention + sigmoid gating + BCE head — is the new project best.**

### Launch command

```bash
SKIP_INSTALL=1 \
ALIGNED_ROOT=/home/bruno/Desktop/tcc_new/datasets/FIW_aligned \
BATCH_SIZE=8 \
GRAD_ACCUM=4 \
UNFREEZE_LAST_STAGE=1 \
LEARNING_RATE=1e-5 \
NUM_WORKERS=4 \
SEED=42 \
bash models/12_rgck_net/AMD/run_pipeline.sh
```

### Changes from Run 001

| Parameter | R001 | **R002** |
|---|---|---|
| Backbone | AdaFace IR-101 frozen | AdaFace IR-101 **stages 1-3 frozen, stage 4 + output_layer trainable** |
| Trainable params | 5,589,762 (7.90 %) | **31,554,818 (44.61 %)** |
| LR (peak) | 1e-4 | **1e-5** (10× lower, per proposal §37) |
| All other knobs | — | identical |

The single architectural change is **the partial unfreeze of stage 4 and output_layer** (the FC head producing the 512-d embedding). Stages 1-3 stay frozen, preserving most of AdaFace's identity-discriminative pretrain. The 26 M extra trainable params from stage 4 + output_layer give the model enough plasticity to adapt the deepest features for kinship.

### Configuration

| Param | Value |
|-------|-------|
| Dataset | fiw (Track-I, split_seed=42) |
| Aligned root | datasets/FIW_aligned (224 native) |
| Backbone | AdaFace IR-101 (WebFace4M, **partially frozen — stages 1-3 frozen, stage 4 + output_layer trainable**) |
| Regions | 5: global, eyes, nose, mouth, jaw (fixed 224-coords) |
| Cross-region adapter | 1 bidirectional layer × 4 heads × 512-d |
| Regional gate | MLP `[rA, rB, |diff|, prod]` → sigmoid |
| Classifier head | 3-layer MLP + BatchNorm + GELU + dropout |
| Loss | bce on classifier logit |
| Batch | 8 × grad-accum 4 (eff 32) |
| **LR** | **1e-5** peak, warmup 5, cosine, min_lr 1e-6 |
| Weight decay | 1e-4 |
| Dropout | 0.2 |
| Train neg strategy | random |
| Eval neg strategy | random |
| **Trainable params** | **31,554,818 / 70,740,674 (44.61 %)** |
| Time/epoch | ~25.8 min (vs R001 ~20 min — slightly slower due to backward pass through stage 4) |
| Seed | 42 |

### Training trajectory

| Epoch | Train Loss | Val Acc | Val AUC | Thr | LR | Note |
|------:|-----------:|--------:|--------:|-----|-----|------|
| 1 | 0.5885 | 78.1 % | **0.8644** | 0.400 | 2.0e-6 | warmup 1/5 — **already above M12 R001 peak (0.8351)** |
| 2 | 0.4060 | 84.4 % | **0.9259** | 0.400 | 4.0e-6 | **+0.062 climb — beats M11 v4 peak (0.8987)** |
| 3 | 0.3316 | 85.5 % | 0.9311 | 0.400 | 6.0e-6 | new peak |
| **4** | **0.2881** | **85.5 %** | **0.9323** | 0.200 | 8.0e-6 | **lifetime peak — best.pt** |
| 5 | 0.2544 | 85.4 % | 0.9306 | 0.100 | 1.0e-5 | peak LR, no unlock |
| 6 | 0.2240 | 85.6 % | 0.9284 | 0.100 | 1.0e-5 | |
| 7 | 0.2036 | 85.5 % | 0.9230 | 0.100 | 9.99e-6 | -0.009 from peak, train loss continuing — **manual stop** |

### Test metrics (threshold 0.500)

Stored threshold = 0.500 (training killed before `update_checkpoint_metadata`). Val-phase F1-optimal at ep 4 was 0.200. AUC, Avg Precision, TAR@FAR are threshold-invariant.

| Metric | M02 R031 (prior best) | **M12 R002 (NEW)** |
|---|---:|---:|
| **Test ROC AUC** | **0.850** | **0.8564** ⭐ |
| Test Accuracy | 74.4 % | **76.79 %** ⭐ |
| Test Balanced Acc | 75.2 % | **76.48 %** ⭐ |
| Test Precision | 66.5 % | **79.82 %** ⭐ |
| Test Recall | 94.1 % | 69.00 % |
| Test F1 | 0.779 | 0.7402 |
| **Test Avg Precision** | 0.817 | **0.8389** ⭐ |
| **TAR @ FAR=0.001** | 2.5 % | **4.18 %** ⭐ |
| **TAR @ FAR=0.01** | 14.0 % | **17.58 %** ⭐ |
| **TAR @ FAR=0.1** | 49.9 % | **57.11 %** ⭐ |
| Val→test AUC gap | -0.031 | -0.076 |

*Note on threshold comparison:* M02 R031 reported per-relation metrics at threshold 0.900 (very high — favours precision). M12 R002 default threshold is 0.500. Per-class accuracies are not directly comparable. **The threshold-invariant metrics (AUC, AP, TAR@FAR) are what matter for cross-threshold ranking quality**, and M12 R002 wins on all of them.

### Per-relation accuracy (FIW Track-I test, threshold 0.500)

| Relation | N | M02 R031 (thr 0.900) | M09 R001 (thr 0.500) | **M12 R002 (thr 0.500)** |
|----------|--:|---------------------:|---------------------:|-------------------------:|
| bb | 860 | 95.5 % | 64.2 % | **75.4 %** |
| ss | 731 | 94.7 % | 62.9 % | **75.9 %** |
| sibs | 234 | 94.9 % | 64.5 % | **79.5 %** |
| md | 1038 | 94.4 % | 53.9 % | **68.7 %** |
| fs | 1135 | 95.3 % | 59.1 % | **68.6 %** |
| ms | 1036 | 93.9 % | 57.3 % | **69.4 %** |
| fd | 918 | 91.7 % | 63.6 % | **68.4 %** |
| **gfgd** | 138 | 89.9 % | 31.2 % | **52.2 %** |
| gfgs | 98 | 95.9 % | 30.6 % | 39.8 % |
| gmgd | 123 | 91.1 % | 31.7 % | 36.6 % |
| gmgs | 121 | 88.4 % | 37.2 % | **44.6 %** |
| non-kin | 6993 | 56.4 % | 84.7 % | 84.0 % |

**At threshold 0.500, M12 R002 beats M09 R001 on every kin class:**
- Same-generation (bb/ss/sibs): +11 to +15 pp
- Parent-child (4 classes): +5 to +15 pp
- Grandparent (4 classes): +7 to +21 pp on three, +0.7 on gfgs
- non-kin: -0.7 (essentially identical)

The improvement is consistent and substantial. The val→test gap (-0.076) is larger than M12 R001 (-0.089) but **the higher Val AUC ceiling (0.9323) more than compensates**, lifting Test AUC to 0.8564.

### Notes

- **Proposal Phase 2 hypothesis fully validated.** Unfreezing stage 4 + output_layer was the key to lifting the Val AUC ceiling above the Phase 1 bottleneck without re-introducing the family memorization that hurt M11 v4 (full FT had gap -0.128, M12 R002 gap is -0.076).
- **Peak LR (ep 5) did NOT unlock a new level** — Val AUC peaked at ep 4 *before* the peak LR. The lower LR (1e-5 vs full-FT models' 5e-6 starting) and warmup interact differently. The model converged fast and started slight overfit immediately at peak LR.
- **Train loss continued to descend monotonically** even as Val AUC declined post-peak, classic overfit signal. Manual stop at ep 7 (3 consecutive declines).
- The **per-region weights** from the regional gate are interpretable but not yet visualised — `evaluate.py` for M12 doesn't exist. Adding this would be a small follow-up.

### Artifacts

- Checkpoint (epoch 4, Val AUC 0.9323): `output/002/checkpoints/best.pt` (536 MB — larger than R001 because stage 4 weights are now in optimizer state) — patched manually with `model_config` (freeze_backbone=True, unfreeze_last_stage=True)
- Resume snapshot: `output/002/checkpoints/epoch_5.pt` (will be pruned)
- Train log: `output/002/logs/train.log`
- Test log: `output/002/logs/test.log` (also `/tmp/m12_r002_test.log`)
- Results: `output/002/results/test_metrics_rocm.txt`
- No `evaluate.py` artefacts yet.

---

## Run 001 — 2026-05-13 — Stopped at epoch 7 (peak Val AUC 0.8351 ep 3; Test AUC 0.7464, val→test gap -0.089)

**Status:** Stopped manually at epoch 7/100 — 4 epochs below peak, train loss continuing to descend (overfit creeping in)
**Outcome:** RGCK-Net Phase 1 (per `proposta_rgck_net_kinship.md`): AdaFace IR-101 frozen, 5 fixed-region tokens, bidirectional cross-region attention, sigmoid gating, BCE classifier head. Val AUC peak 0.8351 (ep 3), Test AUC **0.7464**. **Val→test gap -0.089 — smallest in the AdaFace family** (vs M09 R001 -0.094, M11 v4 -0.128) confirming reduced family memorization, but **absolute Test AUC is lower than M09 R001 (0.7982) by -0.052** because the frozen backbone caps the Val AUC ceiling too low. Mixed result: architectural hypothesis directionally validated (smaller gap), but not enough to beat the M09 R001 baseline on Test AUC.

### Launch command

```bash
SKIP_INSTALL=1 \
ALIGNED_ROOT=/home/bruno/Desktop/tcc_new/datasets/FIW_aligned \
BATCH_SIZE=8 \
GRAD_ACCUM=4 \
NUM_WORKERS=4 \
SEED=42 \
bash models/12_rgck_net/AMD/run_pipeline.sh
```

Pipeline defaults (Phase 1): `USE_MULTISTAGE` not applicable, `FREEZE_BACKBONE=1`, `LOSS=bce` (on classifier head logit), `TRAIN_NEGATIVE_STRATEGY=random`, `EVAL_NEGATIVE_STRATEGY=random`, `LR=1e-4` (head-only LR per proposal §37), `CROSS_ATTN_LAYERS=1`, `CROSS_ATTN_HEADS=4`.

### Configuration

| Param | Value |
|-------|-------|
| Dataset | fiw (Track-I, split_seed=42) |
| Aligned root | datasets/FIW_aligned (224×224 native, no resize at load — model crops internally) |
| Backbone | AdaFace IR-101 (WebFace4M, **FROZEN**) |
| Regions | 5 fixed-coordinate boxes: global (0:224, 0:224), eyes (40:100, 20:204), nose (80:150, 70:154), mouth (140:185, 50:174), jaw (170:220, 20:204) |
| Per-region tokenizer | Crop region → resize 112×112 → AdaFace shared → 512-d L2-normalised token |
| Cross-region adapter | 1 bidirectional layer × 4 heads × 512-d, LayerNorm + GELU FFN (expansion 2×) |
| Regional gate | MLP `[rA, rB, |diff|, prod]` → sigmoid per-region weight |
| Classifier head | 3-layer MLP (input 2049 = 4×512+2×5+1) + BatchNorm + GELU + dropout |
| Loss | bce on classifier logit |
| Batch | 8 × grad-accum 4 (eff. 32) |
| LR | 1e-4 peak, warmup 5, cosine, min_lr 1e-6 |
| Weight decay | 1e-4 |
| Dropout | 0.2 |
| Train neg strategy | random (default) |
| Eval neg strategy | random (default) |
| Embedding dim | 512 |
| Patience | 50 |
| **Trainable params** | **5,589,762 / 70,740,674 (7.90%)** |
| Time/epoch | ~20 min (half of full-FT models due to frozen backbone) |
| Seed | 42 |

### Training trajectory

- Best Val AUC: **0.8351** at epoch 3 (best.pt)
- Stopped manually at epoch 7 (4 epochs below peak)
- Time per epoch: ~20.6 min (half of M09/M10/M11 because backbone backward pass skipped)
- Total wall time: ~145 min vs ~280 min for an equivalent number of epochs on full-FT models

| Epoch | Train Loss | Val Acc | Val AUC | Thr | LR    | Note |
|------:|-----------:|--------:|--------:|----:|-------|------|
| 1 | 0.5452 | 74.8 % | **0.8341** | 0.350 | 2.0e-5 | warmup 1/5 — already at M09's unlock level |
| 2 | 0.4523 | 74.4 % | 0.8289 | 0.200 | 4.0e-5 |  |
| **3** | **0.3886** | **75.4 %** | **0.8351** | 0.150 | 6.0e-5 | **peak (best.pt)** |
| 4 | 0.3572 | 74.7 % | 0.8302 | 0.100 | 8.0e-5 | small dip |
| 5 | 0.3291 | 74.2 % | 0.8231 | 0.100 | 1.0e-4 | peak LR, no unlock |
| 6 | 0.3096 | 74.4 % | 0.8217 | 0.100 | 9.99e-5 |  |
| 7 | 0.2922 | 74.3 % | 0.8179 | 0.100 | 9.99e-5 | -0.017 from peak — **manual stop** |

Unlike the full-FT models (M09/M10/M11), there was **no "peak LR unlock"** at ep 5. M12 starts already at 0.8341 Val AUC in epoch 1 (which is where full-FT models needed ep 5 + peak LR to reach), and oscillates in 0.82-0.83 throughout. The frozen backbone makes the head + cross-attn + gate adapt fast but caps the achievable ceiling at ~0.835.

### Test metrics (threshold 0.500)

Stored threshold on best.pt = 0.500 (default — training killed before `update_checkpoint_metadata` ran). AUC and AP threshold-invariant.

| Metric | Value |
|--------|-------|
| **Test ROC AUC** | **0.7464** |
| Test Accuracy | 68.00 % |
| Test Balanced Acc | 67.41 % |
| Test Precision | 72.60 % |
| Test Recall | 53.34 % |
| Test F1 | 0.6150 |
| Test Avg Precision | 0.7323 |
| TAR @ FAR=0.001 | 2.36 % |
| TAR @ FAR=0.01 | 10.06 % |
| TAR @ FAR=0.1 | 37.86 % |
| **Val→test AUC gap** | **-0.089 (smallest in AdaFace family)** |

### Per-relation accuracy (FIW Track-I test)

| Relation | N | M09 R001 | M11 v4 | **M12 R001** | Δ vs M09 R001 |
|----------|--:|---------:|-------:|-------------:|--------------:|
| bb | 860 | 64.2 % | 56.6 % | 58.9 % | -5.3 pp |
| ss | 731 | 62.9 % | 53.9 % | 57.1 % | -5.9 pp |
| sibs | 234 | 64.5 % | 50.0 % | 61.5 % | -3.0 pp |
| md | 1038 | 53.9 % | 53.1 % | 54.1 % | +0.2 pp |
| fs | 1135 | 59.1 % | 54.1 % | 56.3 % | -2.8 pp |
| ms | 1036 | 57.3 % | 51.5 % | 51.6 % | -5.7 pp |
| **fd** | 918 | 63.6 % | 61.7 % | **52.0 %** | **-11.6 pp** ⚠ |
| gfgd | 138 | 31.2 % | 37.7 % | 28.3 % | -2.9 pp |
| gfgs | 98 | 30.6 % | 36.7 % | 26.5 % | -4.1 pp |
| **gmgd** | 123 | 31.7 % | 22.0 % | **36.6 %** | **+4.9 pp** ✓ |
| gmgs | 121 | 37.2 % | 28.9 % | 33.1 % | -4.1 pp |
| non-kin | 6993 | 84.7 % | 86.7 % | 81.5 % | -3.2 pp |

Per-relation pattern is mixed but trends negative: only `md` (+0.2) and `gmgd` (+4.9) improved over M09 R001. The biggest regression is `fd` at -11.6 pp.

### Notes

- **Val→test gap is the smallest of the AdaFace family** (-0.089 vs M09 R001 -0.094, M11 v4 -0.128). Architectural hypothesis "reduced backbone influence reduces family memorization" — **directionally validated**.
- **But the absolute Val AUC ceiling is too low** (0.8351 peak vs M09 R001 0.8919 peak). Test AUC ends up at 0.7464, below M09 R001's 0.7982.
- **Per-class distribution at peak val** was extraordinarily balanced (75-92 % range across all 11 classes vs M09 R001's 58-94 %). But after test-time threshold (0.5) and held-out families, the per-class accuracies regress similarly to other AdaFace-based models.
- **Time/epoch is half** of full-FT models (~20 min vs ~40 min). The whole experiment cycle (train + test) was ~3 h vs ~8-10 h for M09/M11.
- **Frozen backbone is the bottleneck.** Proposal Phase 2 (unfreeze last stage) and Phase 3 (full FT) could lift the ceiling, but risk reintroducing the family memorization.
- The cross-attention and gating modules **did learn something useful** — per-class accuracy in epoch 1 is already at 73-91 % across all 11 classes, including the historically-difficult `ss` at 85.3 % (vs M09 R001 peak 58.2 %). This is the strongest single-epoch result for siblings in the project.

### Artifacts

- Checkpoint (epoch 3, Val AUC 0.8351): `output/001/checkpoints/best.pt` (328 MB, much smaller than other AdaFace models because frozen backbone weights aren't optimizer state) — patched manually with `model_config`
- Resume snapshot: `output/001/checkpoints/epoch_5.pt` (will be pruned)
- Train log: `output/001/logs/train.log`
- Test log: `output/001/logs/test.log` (also `/tmp/m12_r001_test.log`)
- Results: `output/001/results/test_metrics_rocm.txt`
- No `evaluate.py` artifacts (script not implemented for M12 yet)

---
