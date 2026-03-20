# Run 15 - Local / GPU / KinFaceW-I / Manual stop at epoch 3 / ViT-Small trial

**Date:** 2026-03-20
**GPU:** AMD Radeon RX 6750 XT (11.98 GB VRAM, gfx1031, ROCm 5.7)
**Status:** INTERRUPTED - original checkpoint archive corrupted after manual stop

---

## Context

Run 14 improved specificity by simplifying the head, but it still could not beat run 13 overall. Run 15 tested whether the backbone itself was too large for the dataset by moving from ViT-Base to ViT-Small while keeping the simplified BCE setup.

### Changes implemented since Run 14

1. **Switched backbone from `vit_base_patch16_224` to `vit_small_patch16_224`** to reduce total capacity.
2. **Kept the 1-layer cross-attention head** because that was the intended simplification from run 14.
3. **Kept dropout at 0.2** for the same regularization pressure.
4. **Kept BCE, 4:1 train negatives, 1:1 evaluation, frozen ViT, and LR=1e-5** so the backbone size was the only intended change.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Method | Local (`AMD/run_pipeline.sh` + `.venv`) |
| Device | GPU - cuda:0 (gfx1031, ROCm 5.7, PyTorch 2.3.1+rocm5.7) |
| Dataset | KinFaceW-I |
| Split | 70/15/15 -> 1865 train pairs / 158 val / 162 test |
| Epochs | 3 completed before stop |
| Batch size | 32 |
| Learning rate | 1e-5 |
| Scheduler | none (warmup 1 epoch) |
| Loss | bce |
| Temperature | 0.3 |
| Train negative ratio | 4.0 |
| Eval negative ratio | 1.0 |
| ViT model | vit_small_patch16_224 |
| Cross-attn layers | 1 |
| Cross-attn heads | 8 |
| Dropout | 0.2 |
| Freeze ViT | Yes |
| Trainable params | 2.25M |
| AMP | Enabled (ROCm AMP) |

---

## Original Run Progress

| Epoch | Train Loss | Val Acc | Val AUC | LR | Notes |
|-------|-----------|---------|---------|-----|-------|
| 1 | 0.8590 | 0.5000 | 0.7622 | 1.00e-05 | Initial checkpoint |
| 2 | 0.7509 | 0.5127 | 0.7694 | 1.00e-05 | Slight improvement |
| **3** | **0.6971** | **0.5380** | **0.7701** | **1.00e-05** | **Best logged AUC before stop** |

The run was stopped manually after epoch 3 because it was still well below the better base-ViT configurations. The original `output/015/checkpoints/best.pt` file is incomplete and cannot be loaded for protocol evaluation, so there is no canonical test-set result for the interrupted artifact itself.

---

## Replay of Same Configuration

To estimate whether the **configuration** was worth continuing, the exact same hyperparameter setup was replayed separately for 3 epochs into `output/015/recovery_checkpoints`. That replay is **supporting evidence only**, not the official result of the interrupted original run, because it diverged from the original logged validation AUC.

### Replay validation-selected threshold

| Metric | Value |
|--------|-------|
| Threshold | **0.750** |
| Val Accuracy | 70.89% |
| Val F1 | 0.747 |
| Val ROC-AUC | 0.803 |

### Replay test metrics (supporting evidence only)

| Metric | Value |
|--------|-------|
| Accuracy | **60.49%** |
| Precision | 58.10% |
| Recall | 75.31% |
| F1 | 0.656 |
| ROC-AUC | 0.696 |
| TAR@FAR=0.001 | 0.00% |
| TAR@FAR=0.01 | 0.00% |
| TAR@FAR=0.1 | 34.57% |

### Comparison with Run 14

| Metric | Run 14 | Run 15 replay |
|--------|--------|---------------|
| Accuracy | **62.96%** | 60.49% |
| F1 | 0.620 | **0.656** |
| ROC-AUC | 0.666 | **0.696** |
| Negative accuracy | **65.43%** | 45.68% |

The replay suggests ViT-Small shifts the model back toward higher recall, but it still does not beat run 13 overall and still does not approach the CNN+FaCoR target.

---

## Analysis

### What the original interrupted run showed

Even before the checkpoint failure, the original run 15 validation AUC only reached 0.7701 by epoch 3, which was already below the stronger base-ViT setups. That was enough to justify stopping the run instead of spending more budget on it.

### What the replay suggests

The same configuration is not a breakthrough. Even when replayed cleanly, it lands in the same general performance band as the other frozen-backbone experiments and still loses to run 13 on accuracy and to run 09 on all headline metrics.

### Final conclusion after 5 new runs

After runs 11 through 15, **none of the new experiments beat either**:

1. the repo's earlier best ViT result from run 09
2. the README CNN+FaCoR benchmark

Among the new runs, **run 13** remains the strongest fully completed configuration.

---

## Next Step

The 5-run budget is exhausted. The evidence from this batch suggests that further gains probably need a larger change than incremental tuning, such as:

1. a different pairing/scoring head
2. cross-validation instead of a single split
3. stronger data protocol or relation-aware supervision
