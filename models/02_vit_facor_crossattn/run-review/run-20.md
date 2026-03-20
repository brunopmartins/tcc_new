# Run 20 - Local / GPU / KinFaceW-I / Manual stop during epoch 7 / Capacity-control ablation after run 19

**Date:** 2026-03-20
**GPU:** AMD Radeon RX 6750 XT (11.98 GB VRAM, gfx1031, ROCm 5.7)
**Status:** MANUALLY STOPPED during epoch 7 - best checkpoint recovered manually

---

## Context

Run 19 finally produced the best local result in the repo, but it still leaned hard toward high recall. The follow-up question was whether the remaining errors came from head overcapacity. Run 20 kept the winning contrastive margin and simplified the head.

### Changes implemented since Run 19

1. **Reduced cross-attention depth from 2 layers to 1** to cut trainable capacity.
2. **Raised dropout from 0.1 to 0.2** to regularize the smaller head more strongly.
3. **Kept supervised contrastive loss with margin `0.3`** because that was the main run 19 improvement.
4. **Kept frozen ViT, LR=1e-5, 4:1 train negatives, and 1:1 evaluation** so the only intended change was capacity control.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Method | Local (`AMD/run_pipeline.sh` + `.venv`) |
| Device | GPU - cuda:0 (gfx1031, ROCm 5.7, PyTorch 2.3.1+rocm5.7) |
| Dataset | KinFaceW-I |
| Split | 70/15/15 -> 1865 train pairs / 158 val / 162 test |
| Epochs | 6 completed / 100 planned |
| Batch size | 32 |
| Learning rate | 1e-5 |
| Scheduler | none (warmup 1 epoch) |
| Loss | contrastive |
| Margin | 0.3 |
| Train negative ratio | 4.0 |
| Eval negative ratio | 1.0 |
| ViT model | vit_base_patch16_224 |
| Cross-attn layers | 1 |
| Cross-attn heads | 8 |
| Dropout | 0.2 |
| Freeze ViT | Yes |
| Classifier head | No |
| Trainable params | 7.82M |
| AMP | Enabled (ROCm AMP) |

---

## Training Progress

| Epoch | Train Loss | Val Acc | Val AUC | LR | Notes |
|-------|-----------|---------|---------|-----|-------|
| 1 | 0.0125 | 0.5000 | 0.7129 | 1.00e-05 | Good start, but below run 19 |
| 2 | 0.0076 | 0.5000 | 0.7151 | 1.00e-05 | Minor gain |
| 3 | 0.0075 | 0.5000 | 0.7202 | 1.00e-05 | Still improving |
| 4 | 0.0072 | 0.5000 | 0.7297 | 1.00e-05 | Better, but slower |
| 5 | 0.0069 | 0.5000 | 0.7411 | 1.00e-05 | Still behind run 19 |
| **6** | **0.0072** | **0.5000** | **0.7480** | **1.00e-05** | **Best checkpoint saved** |

The run was stopped in epoch 7 because the simplified-head branch was improving too slowly and remained well behind run 19.

---

## Results

### Validation-selected threshold

| Metric | Value |
|--------|-------|
| Threshold | **0.900** |
| Val Accuracy | 53.16% |
| Val F1 | 0.678 |
| Val ROC-AUC | 0.786 |

### Test metrics (recovered protocol)

| Metric | Value |
|--------|-------|
| Accuracy | **53.70%** |
| Precision | 51.92% |
| Recall | 100.00% |
| F1 | 0.684 |
| ROC-AUC | 0.692 |
| TAR@FAR=0.001 | 0.00% |
| TAR@FAR=0.01 | 0.00% |
| TAR@FAR=0.1 | 33.33% |

### Comparison with Run 19

| Metric | Run 19 | Run 20 |
|--------|--------|--------|
| Accuracy | **67.28%** | 53.70% |
| F1 | **0.720** | 0.684 |
| ROC-AUC | **0.787** | 0.692 |

---

## Analysis

### What changed

The smaller head did make the model lighter and faster, but it gave up too much discrimination. The validation-selected threshold again pushed the model toward a very recall-heavy regime.

### What failed

This ablation lost the stronger balance that run 19 had found. It drifted back toward an all-kin operating point and gave away most of the AUC gain from the lower-margin contrastive setup.

### Final conclusion after runs 16-20

The important finding from this batch is not run 20 itself, but the path that led to run 19:

1. fixing the BCE classifier path alone was not enough
2. switching to a supervised contrastive loss was the real turning point
3. lowering the contrastive margin to `0.3` produced the best local result so far

---

## Next Step

Run 19 should now be treated as the primary checkpoint for this model line. If more budget is available later, the next experiments should stay near the run 19 recipe rather than the run 20 simplification.
