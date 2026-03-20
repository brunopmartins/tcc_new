# Run 16 - Local / GPU / KinFaceW-I / Manual stop during epoch 3 / First real BCE classifier-head test

**Date:** 2026-03-20
**GPU:** AMD Radeon RX 6750 XT (11.98 GB VRAM, gfx1031, ROCm 5.7)
**Status:** MANUALLY STOPPED during epoch 3 - best checkpoint recovered manually

---

## Context

Run 15 suggested the smaller backbone was not the main issue. The next finding was more fundamental: the BCE path in model 02 was not actually using the classifier head. Run 16 was the first run after fixing that path so BCE could train a real decision head instead of only reading cosine similarity.

### Changes implemented since Run 15

1. **Switched back from `vit_small_patch16_224` to `vit_base_patch16_224`** because the smaller backbone did not improve the prior batch.
2. **Enabled the actual BCE classifier head** so the BCE objective finally trained the score head it was supposed to use.
3. **Kept the simplified 1-layer cross-attention head with dropout 0.2** from the run 14/15 direction.
4. **Kept frozen ViT, BCE, LR=1e-5, 4:1 train negatives, and 1:1 evaluation** so the main change was the BCE-path fix itself.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Method | Local (`AMD/run_pipeline.sh` + `.venv`) |
| Device | GPU - cuda:0 (gfx1031, ROCm 5.7, PyTorch 2.3.1+rocm5.7) |
| Dataset | KinFaceW-I |
| Split | 70/15/15 -> 1865 train pairs / 158 val / 162 test |
| Epochs | 2 completed / 100 planned |
| Batch size | 32 |
| Learning rate | 1e-5 |
| Scheduler | none (warmup 1 epoch) |
| Loss | bce |
| Temperature | 0.3 |
| Train negative ratio | 4.0 |
| Eval negative ratio | 1.0 |
| ViT model | vit_base_patch16_224 |
| Cross-attn layers | 1 |
| Cross-attn heads | 8 |
| Dropout | 0.2 |
| Freeze ViT | Yes |
| Classifier head | Yes |
| Trainable params | 8.34M |
| AMP | Enabled (ROCm AMP) |

---

## Training Progress

| Epoch | Train Loss | Val Acc | Val AUC | LR | Notes |
|-------|-----------|---------|---------|-----|-------|
| 1 | 0.6647 | 0.5000 | 0.5826 | 1.00e-05 | Weak start |
| **2** | **0.6320** | **0.5000** | **0.6078** | **1.00e-05** | **Best checkpoint saved** |

The run was stopped during epoch 3 because it was clearly collapsing into a predict-all-kin regime again. The saved best checkpoint was evaluated afterward with the shared validation-threshold protocol.

---

## Results

### Validation-selected threshold

| Metric | Value |
|--------|-------|
| Threshold | **0.100** |
| Val Accuracy | 50.00% |
| Val F1 | 0.667 |
| Val ROC-AUC | 0.563 |

### Test metrics (recovered protocol)

| Metric | Value |
|--------|-------|
| Accuracy | **50.00%** |
| Precision | 50.00% |
| Recall | 100.00% |
| F1 | 0.667 |
| ROC-AUC | 0.542 |
| TAR@FAR=0.001 | 0.00% |
| TAR@FAR=0.01 | 0.00% |
| TAR@FAR=0.1 | 9.88% |

### Comparison with Run 15 replay

| Metric | Run 15 replay | Run 16 |
|--------|---------------|--------|
| Accuracy | **60.49%** | 50.00% |
| F1 | 0.656 | **0.667** |
| ROC-AUC | **0.696** | 0.542 |

---

## Analysis

### What changed

The classifier-head fix was real, but it did not improve the run by itself. The model still learned a threshold-free signal that was too weak to separate non-kin pairs, and the validation-selected threshold collapsed to `0.10`.

### What failed

This run effectively predicted everything as kin. That keeps recall at 100% but destroys specificity and makes the 50% accuracy meaningless on a balanced split.

### Likely root cause

The BCE-path bug was not the only issue. Even after fixing the head, the loss still seemed too biased toward a trivial kin-favoring solution.

---

## Next Step

Keep the classifier-head fix, but test whether the problem is just head capacity:

1. restore the 2-layer head
2. lower dropout back to 0.1
3. keep BCE and the same data protocol
