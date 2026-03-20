# Run 06 — Local / GPU / KinFaceW-I / 1-epoch smoke test (AUC early-stop fix)

**Date:** 2026-03-01
**GPU:** AMD Radeon RX 6750 XT (11.98 GB VRAM, gfx1031, ROCm 5.7)
**Status:** SUCCESS — pipeline smoke test only (EPOCHS=1)

---

## Context

Run 05 exposed that early stopping monitored `val_accuracy` (stuck at 0.5000 for all epochs since the model predicts everything as kin at threshold=0.5) rather than `val_roc_auc`. This caused training to stop after 16 epochs despite the model learning well (AUC started at 0.82 and was still above 0.70).

Fix applied before this run:
- `shared/AMD/trainer.py`: added `monitor_metric` parameter (default `"accuracy"` — backward compatible)
- `02_vit_facor_crossattn/AMD/train.py`: passes `monitor_metric="roc_auc"` to the trainer

This run was a 1-epoch smoke test to confirm the fix works without waiting for a full training run.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Method | Local (`AMD/run_pipeline.sh` + `.venv`) |
| Device | GPU — cuda:0 (gfx1031, ROCm 5.7, PyTorch 2.3.1+rocm5.7) |
| Dataset | KinFaceW-I |
| Split | 70/15/15 → 746 train / 158 val / 162 test |
| Epochs | **1 (smoke test)** |
| Batch size | 32 |
| Learning rate | 1e-4 (cosine warmup from 2e-5) |
| Temperature | 0.07 |
| ViT model | vit_base_patch16_224 |
| Cross-attn layers | 2 |
| Cross-attn heads | 8 |
| Loss | cosine_contrastive |
| Freeze ViT | No |
| AMP | Enabled (ROCm AMP) |

---

## Training Progress

| Epoch | Train Loss | Val Acc | Val AUC | LR | Time |
|-------|-----------|---------|---------|-----|------|
| 1/1 | 4.0188 | 0.5000 | **0.8279** | 2.00e-05 | 18.5s |

Early stopping now correctly tracks AUC: `-> New best model! roc_auc: 0.8279`

---

## Results

### test.py (optimal threshold)

| Metric | Value |
|--------|-------|
| Accuracy | 62.96% |
| Precision | 58.68% |
| Recall | 87.65% |
| F1 | 0.703 |
| ROC-AUC | 0.762 |
| Threshold | 0.900 |

### evaluate.py (full analysis)

| Metric | Value |
|--------|-------|
| Accuracy | 61.73% |
| Precision | 57.72% |
| Recall | 87.65% |
| F1 | 0.696 |
| ROC-AUC | 0.734 |
| Threshold | 0.900 |

Recall=0.877 and threshold=0.900 indicate the model is still predicting nearly everything as kin. This is expected after 1 epoch of warmup-phase training (LR=2e-05).

---

## Lessons

- AUC early stopping fix confirmed working. The trainer correctly logs `roc_auc` as the monitored metric.
- 1-epoch AUC of 0.828 (val) is promising — pretrained ViT features already carry discriminative signal.
- Results from a 1-epoch run are not meaningful beyond confirming the pipeline works end-to-end.
