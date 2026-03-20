# Run 11 — Local / GPU / KinFaceW-I / Manual stop at epoch 12 / Frozen ViT / Harder train negatives

**Date:** 2026-03-20
**GPU:** AMD Radeon RX 6750 XT (11.98 GB VRAM, gfx1031, ROCm 5.7)
**Status:** MANUALLY STOPPED after epoch 11 completed and epoch 12 started — no recovery after epoch-1 AUC peak

---

## Context

Run 10 showed that continued fine-tuning collapses the embedding space: the threshold fell to 0.100 and the model predicted every test pair as kin. Run 11 was the first attempt to directly address that failure mode while keeping evaluation fair.

### Changes implemented since Run 10

1. **Froze the ViT backbone** (`FREEZE_VIT=1`) to stop backbone drift.
2. **Raised the training negative ratio to 2:1** to improve non-kin rejection.
3. **Kept evaluation negative ratio fixed at 1:1** by patching the shared loader so train difficulty can change without changing the validation/test protocol.
4. **Switched scheduler from cosine to plateau** to avoid the aggressive post-warmup LR decay seen in earlier runs.
5. **Raised LR to 1e-4** because only the cross-attention/projection layers remained trainable.
6. **Reduced patience to 20** to save run budget if the frozen setup failed early.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Method | Local (`AMD/run_pipeline.sh` + `.venv`) |
| Device | GPU — cuda:0 (gfx1031, ROCm 5.7, PyTorch 2.3.1+rocm5.7) |
| Dataset | KinFaceW-I |
| Split | 70/15/15 → 1119 train / 158 val / 162 test |
| Epochs | 11 completed / 100 planned |
| Batch size | 32 |
| Learning rate | 1e-4 (warmup 5 epochs) |
| Scheduler | plateau |
| Loss | cosine_contrastive |
| Temperature | 0.3 |
| Train negative ratio | 2.0 |
| Eval negative ratio | 1.0 |
| ViT model | vit_base_patch16_224 |
| Cross-attn layers | 2 |
| Cross-attn heads | 8 |
| Freeze ViT | Yes |
| Trainable params | 14.9M |
| AMP | Enabled (ROCm AMP) |

---

## Training Progress

| Epoch | Train Loss | Val Acc | Val AUC | LR | Notes |
|-------|-----------|---------|---------|-----|-------|
| **1** | **4.0431** | **0.5000** | **0.7774** | **2.00e-05** | **Best checkpoint saved** |
| 2 | 3.3078 | 0.5000 | 0.7182 | 4.00e-05 | Immediate regression |
| 3 | 2.5900 | 0.5000 | 0.6980 | 6.00e-05 | |
| 4 | 2.3550 | 0.5000 | 0.6847 | 8.00e-05 | |
| 5 | 2.2166 | 0.5000 | 0.6863 | 1.00e-04 | Warmup complete |
| 6 | 2.1474 | 0.5000 | 0.6794 | 1.00e-04 | |
| 7 | 2.0575 | 0.5000 | 0.6826 | 1.00e-04 | |
| 8 | 1.9631 | 0.5000 | 0.6762 | 1.00e-04 | |
| 9 | 1.9364 | 0.5000 | 0.6718 | 1.00e-04 | |
| 10 | 1.8817 | 0.5000 | 0.6618 | 1.00e-04 | |
| 11 | 1.8425 | 0.5000 | 0.6869 | 1.00e-04 | Manual stop after clear plateau/regression |

The run was stopped manually instead of waiting for patience=20 because the model never recovered the epoch-1 AUC peak and continuing would have spent time without improving the checkpoint.

---

## Results

Because the run was interrupted before `train.py` could write protocol metadata, the saved `best.pt` checkpoint was recovered manually with the shared validation-threshold protocol after training stopped.

### Validation-selected threshold

| Metric | Value |
|--------|-------|
| Threshold | **0.800** |
| Val Accuracy | 68.99% |
| Val F1 | 0.754 |
| Val ROC-AUC | 0.811 |

### Test metrics (recovered protocol)

| Metric | Value |
|--------|-------|
| Accuracy | **57.41%** |
| Precision | 54.69% |
| Recall | 86.42% |
| F1 | 0.670 |
| ROC-AUC | 0.692 |
| TAR@FAR=0.001 | 0.00% |
| TAR@FAR=0.01 | 6.17% |
| TAR@FAR=0.1 | 19.75% |

### Comparison with previous best

| Metric | Run 09 (best so far) | Run 11 |
|--------|----------------------|--------|
| Accuracy | **66.05%** | 57.41% |
| F1 | **0.715** | 0.670 |
| ROC-AUC | **0.767** | 0.692 |

Run 11 is a regression from run 09 across all headline metrics.

---

## Analysis

### What improved

The severe threshold collapse from run 10 was reduced. With validation threshold recovery, the selected threshold moved to 0.800 instead of 0.100, and the model no longer predicted every test pair as kin.

### What still failed

Non-kin rejection remained weak. Test recall stayed high (0.864), but negative-pair accuracy was only 28.4%, so the extra train negatives were not enough to create a clean separation boundary.

### Likely root cause

The frozen-backbone idea itself looks reasonable, but **LR=1e-4 is still too aggressive for the randomly initialized cross-attention/projection stack**. The evidence is that:

1. Validation AUC peaked immediately at epoch 1.
2. Every later epoch regressed despite steadily falling training loss.
3. Plateau scheduling never had a chance to rescue the run before the head had already overfit away from the pretrained geometry.

---

## Next Step

Keep the successful parts of run 11:

1. `--freeze_vit`
2. fair evaluation at 1:1 negatives
3. harder train negatives

Change the optimizer behavior:

1. much lower LR
2. constant or gentler scheduler
3. possibly a simpler supervision signal (`bce`) if cosine contrastive continues to preserve recall bias
