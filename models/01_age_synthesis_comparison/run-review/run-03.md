# Run 03 — Local / GPU / KinFaceW-I / Age Synthesis ON (degraded)

**Date:** 2026-02-18
**GPU:** AMD Radeon RX 6750 XT (11.98 GB VRAM) — first GPU run on local host

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Method | Local (`AMD/run_pipeline.sh` + `.venv`) |
| Device | GPU — cuda:0 (gfx1031, ROCm 5.7, PyTorch 2.3.1+rocm5.7) |
| Dataset | KinFaceW-I only |
| Split | 70/15/15 → 746 train / 158 val / 162 test |
| Epochs | 26 of 100 (early stop, patience=15) |
| Batch size | 32 |
| Learning rate | 1e-4 (cosine warmup from 2e-5 over 5 epochs) |
| Weight decay | 1e-5 |
| Backbone | ResNet-50 |
| Age synthesis | Enabled — but **no SAM weights** → identity fallback |
| Aggregation | Attention |
| AMP | Enabled (ROCm AMP) |
| Loss | BCE with pos_weight=1.0 (balanced dataset) |
| Training speed | 36.8 s/epoch (epoch 1, MIOpen tuning) / ~22 s/epoch (subsequent) |
| GPU fix | `sudo usermod -aG render,video bruno` → user now in `render` group |
| Output folder | `output/002/` (on-disk label — actual run count: 3) |

---

## Training Curve

| Epoch | Train Loss | Val Acc | Val AUC | LR | Notes |
|-------|-----------|---------|---------|-----|-------|
| 1 | 0.6931 | 50.00% | 0.4161 | 2.00e-05 | Warmup |
| 2 | 0.6931 | 51.27% | 0.5256 | 4.00e-05 | |
| 4 | 0.6923 | 51.90% | 0.5731 | 8.00e-05 | |
| 5 | 0.6909 | 53.80% | 0.5502 | 1.00e-04 | Full LR reached |
| 6 | 0.6873 | 56.33% | 0.5773 | 1.00e-04 | |
| 9 | 0.6571 | 61.39% | 0.6259 | 9.96e-05 | |
| **11** | **0.6292** | **64.56%** | **0.6235** | 9.90e-05 | **Best checkpoint** |
| 22 | 0.5286 | 60.76% | 0.6424 | 9.23e-05 | |
| 26 | 0.4919 | 57.59% | 0.6132 | 8.84e-05 | Early stop triggered |

---

## Results

| Source | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | Samples |
|--------|-----------|----------|-----------|--------|-----|---------|---------|
| train.py (end-of-training) | 0.50 | 61.11% | 58.18% | 79.01% | 67.02% | 68.28% | 162 |
| test.py (optimal threshold) | 0.50 | 64.81% | 61.32% | 80.25% | 69.52% | 66.44% | 162 |
| evaluate.py | 0.50 | 61.11% | 58.04% | 80.25% | 67.36% | 67.26% | 162 |

### Per-Relation (test.py, threshold=0.50)

| Relation | Accuracy | F1 | Samples |
|----------|---------|-----|---------|
| fd — father/daughter | 88.00% | 93.62% | 25 |
| fs — father/son | 88.00% | 93.62% | 25 |
| md — mother/daughter | 85.71% | 92.31% | 14 |
| ms — mother/son | **52.94%** | 69.23% | 17 |
| negative (non-kin) | **49.38%** | 0.00% | 81 |

### TAR @ FAR

| FAR | TAR |
|-----|-----|
| 0.001 | 0.00% |
| 0.01 | 0.00% |
| 0.1 | 16.05% |

### Ablation Study (aggregation method, evaluate.py)

| Method | Accuracy | F1 | ROC-AUC |
|--------|---------|-----|---------|
| attention | 61.11% | 67.36% | 67.26% |
| max | 61.11% | 67.36% | 67.26% |
| mean | 61.11% | 67.36% | 67.26% |

> All three methods produce **identical results** — see Flaw #1 below.

---

## Flaws

### 1. Age synthesis is structurally disabled (critical)
Age synthesis is flagged as enabled (`--use_age_synthesis`), but no SAM pretrained weights exist at `SAM/pretrained_models/sam_ffhq_aging.pt`. The `AgeEncoder` detects this (`_initialized=False`) and returns the input image unchanged. As a result:
- Each person is represented by 3 identical copies (no actual age variants)
- The 9-comparison matrix contains 9 equal scores (3×3 of the same pair)
- The `AgeAggregator` learns a constant offset, not age-sensitive weighting
- **The ablation study is meaningless** — attention, max, and mean over 9 equal values are identical (confirmed: all three give exactly Acc=61.11%, F1=67.36%, AUC=67.26%)

The core architectural contribution of this model has never been exercised in any run.

### 2. Worse than Run 2 despite GPU
Overall accuracy dropped from 68.52% (Run 2, CPU) to 64.81% (Run 3, GPU). Enabling age synthesis without weights adds a degenerate 9-comparison path that is harder to train than the direct 1-comparison baseline. The model must now learn to ignore 8 redundant identical comparisons, which adds noise without signal.

### 3. Extreme non-kin bias
Non-kin accuracy is 49.38% — essentially random. The model labels ~80% of all pairs as "kin" regardless of actual relationship. This is worse than Run 2 (53.09%) despite GPU training. The degenerate comparison matrix (Flaw #1) prevents the model from learning discriminative features between kin and non-kin.

### 4. Early stop at epoch 26, best at epoch 11
The best validation accuracy was reached at epoch 11 (64.56%). After that, validation accuracy *declined* while training loss continued to drop — a clear sign of overfitting on the training set. Patience=15 triggered early stop at epoch 26. With such a small dataset (746 train), the model memorizes rather than generalizes.

### 5. ms relation near chance (52.94%)
Mother/son pairs are the hardest relation and are barely above chance. Without age synthesis generating cross-age variants of mother and son faces, the model cannot bridge the appearance gap introduced by gender and age differences.

### 6. GPU underutilization (~22 s/epoch for 24 batches)
For 24 batches of 32 images (768 samples per epoch), ~22 s/epoch gives ~35 samples/s throughput. This is low for a 12 GB GPU with ResNet-50. The bottleneck is the 9× forward pass overhead from the (degenerate) age synthesis path — even with identity encoders, the model runs 9 pairwise comparisons per sample. MIOpen kernel tuning dominated the first epoch (36.8 s).

### 7. Warning spam (130 warnings over 26 epochs)
`UndefinedMetricWarning: Only one class present in y_true` fires 5× per epoch during per-relation validation AUC computation. Each relation's validation batch often contains only kin pairs. This is harmless but generates 130 warnings in the training log, obscuring real issues.

### 8. TAR@FAR effectively zero
FAR=0.001: 0%, FAR=0.01: 0%, FAR=0.1: 16.05%. The model cannot be deployed at any biometric operating point. This is expected without the age synthesis working, but confirms the current model provides no practical verification capability.

---

## What Worked

- **GPU detection fixed** — `sudo usermod -aG render,video bruno` resolved `/dev/kfd` access. ROCm 5.7 with 1 GPU (AMD Radeon RX 6750 XT) detected correctly.
- **ROCm AMP enabled** — Mixed precision training running on GPU.
- **Pipeline ran end-to-end** — train → test → evaluate completed without errors.
- **Numbered output folders** — Results saved to `output/002/` without overwriting previous runs.
- **Checkpoint architecture saved** — `model_config` stored in `best.pt`; test.py and evaluate.py reconstruct the exact architecture automatically.

---

## Priority Fix

Download SAM pretrained weights (`sam_ffhq_aging.pt`, ~1 GB) and place at:
```
SAM/pretrained_models/sam_ffhq_aging.pt
```

This single change activates the actual architectural contribution and should:
- Produce genuinely different age variants per person
- Make the 9-comparison matrix meaningful (age-gap-sensitive scores)
- Make the ablation study informative
- Likely improve accuracy by the +4% predicted in the README
