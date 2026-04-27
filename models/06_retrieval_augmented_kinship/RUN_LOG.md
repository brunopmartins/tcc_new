# Run Log — Model 06: Retrieval-Augmented Kinship

This file logs every training run for this model with the exact configuration and final metrics. The goal is reproducibility: anyone (human or agent) reading this should be able to recreate any run by copying its launch command.

Newest run on top.

---

## Run 001 — 2026-04-26 — Completed (early stop)

**Status:** Stopped (early stop, patience 10)
**Outcome:** Test ROC AUC = **0.776**, best Val AUC = 0.8361 at epoch 8

### Launch command

```bash
SKIP_INSTALL=1 EPOCHS=20 PATIENCE=10 NUM_WORKERS=4 \
  bash models/06_retrieval_augmented_kinship/AMD/run_pipeline.sh
```

### Configuration

| Param | Value |
|-------|-------|
| Dataset | fiw (66.4k train / 11.4k val / 13.4k test pairs) |
| Backbone | vit_base_patch16_224 (frozen) |
| Img size | 224 |
| Batch size | 8 |
| Grad accum | 4 (effective batch = 32) |
| LR | 1e-4 |
| Scheduler | cosine, warmup=3 ep, min_lr=1e-6 |
| Weight decay | 1e-4 |
| Epochs | 20 (early-stopped at 18) |
| Patience | 10 |
| Loss | combined (BCE + 0.3 × contrastive + 0.15 × relation-CE) |
| Temperature | 0.1 |
| Retrieval K | 32 |
| Cross-attention | 2 layers, 4 heads |
| Embedding dim | 512 |
| Dropout | 0.1 |
| Gallery cap | 200,000 |
| Gallery on CPU | False |
| Gallery refresh every | 0 (built once) |
| Max grad norm | 1.0 |
| AMP | on |
| Seed | 42 |
| Workers | 4 |

### Training trajectory

- Best Val AUC: **0.8361** at epoch 8 (best.pt saved here)
- Stopped at epoch 18 (patience 10 hit)
- Trainable params: **8,155,148 / 93,953,804 (8.68%)**
- Time per epoch: ~21 min (~6 hours total + ~7 min gallery build)
- Gallery: 33,207 positive train pairs

| Epoch | Val AUC | Train Loss | LR | Note |
|-------|---------|------------|-----|------|
| 1 | 0.7820 | 1.342 | 3.33e-5 | Warmup |
| 2 | 0.8026 | 1.226 | 6.67e-5 | |
| 3 | 0.8230 | 1.187 | 1.00e-4 | LR peak |
| 4 | 0.8261 | 1.146 | 9.92e-5 | |
| 5 | 0.8336 | 1.123 | 9.67e-5 | |
| 6 | 0.8295 | 1.106 | 9.26e-5 | Patience 1 |
| 7 | 0.8287 | 1.083 | 8.71e-5 | Patience 2 |
| **8** | **0.8361** | 1.061 | 8.03e-5 | **New best** |
| 9 | 0.8258 | 1.037 | 7.26e-5 | Patience 1 |
| 10 | 0.8202 | 1.018 | 6.40e-5 | Patience 2 |
| 11 | 0.8239 | 0.999 | 5.51e-5 | Patience 3 |
| 12 | 0.8173 | 0.977 | 4.59e-5 | Patience 4 |
| 13 | 0.8256 | 0.958 | 3.70e-5 | Patience 5 |
| 14 | 0.8248 | 0.941 | 2.84e-5 | Patience 6 |
| 15 | 0.8237 | 0.921 | 2.07e-5 | Patience 7 |
| 16 | 0.8191 | 0.906 | 1.39e-5 | Patience 8 |
| 17 | 0.8243 | 0.894 | 8.41e-6 | Patience 9 |
| 18 | 0.8258 | 0.888 | 4.34e-6 | Patience 10 → stop |

### Test metrics

Threshold = 0.40 (selected on validation, applied as-is to test).

| Metric | Value |
|--------|-------|
| Test ROC AUC | **0.776** |
| Test Accuracy | 69.8% |
| Balanced Accuracy | 70.3% |
| Test F1 | 0.722 |
| Test Precision | 64.5% |
| Test Recall | 82.0% |
| Average Precision | 0.735 |
| TAR@FAR=0.1 | 0.388 |
| TAR@FAR=0.01 | 0.062 |
| TAR@FAR=0.001 | 0.006 |

Protocol-internal Test AUC was 0.7637 (uses different evaluator) — the standalone `test.py` evaluator at AUC=0.776 is the canonical figure.

### Per-relation accuracy (FIW)

| Relation | Acc | N |
|----------|-----|---|
| sibs (mixed siblings) | 87.2% | 234 |
| bb (brothers) | 86.5% | 860 |
| ss (sisters) | 86.3% | 731 |
| md (mother-daughter) | 85.9% | 1,038 |
| ms (mother-son) | 83.9% | 1,036 |
| gfgs (grandfather-grandson) | 82.7% | 98 |
| fs (father-son) | 78.8% | 1,135 |
| fd (father-daughter) | 76.9% | 918 |
| gfgd (grandfather-granddaughter) | 75.4% | 138 |
| gmgd (grandmother-granddaughter) | 63.4% | 123 |
| gmgs (grandmother-grandson) | 61.2% | 121 |

### Notes

- **Big val→test gap:** Val AUC 0.836 → Test AUC 0.776 (Δ -0.06). Larger gap than Models 02/03 — retrieval is picking up patterns specific to the train gallery that don't generalize.
- **Per-relation is much more uniform than VLMs:** all 11 classes are at 61-87%. VLMs zero-shot got 0% on grandparent classes. Retrieval-augmentation does fix the rare-class problem.
- **AUC ceiling vs parametric models:** 0.776 vs 0.850. Most likely cause: encoder is frozen — the 85.8M backbone params never see kinship.
- **Mild overfitting after ep 8:** train loss kept falling 1.061 → 0.888 while val AUC plateaued. The 8M head capacity is enough to overfit eventually.

### Follow-up ideas (ranked)

1. **LoRA on q/v projections** of the encoder (~1-3M extra params, fits 12GB) — biggest expected gain (+0.04 to +0.07 AUC).
2. **Hard-negative supports:** mix in top-K positives for non-kin queries so cross-attention sees counterfactuals. Should close val→test gap.
3. **Multi-vector retrieval (ColBERT-style):** patch-level retrieval with MaxSim, would specifically help gmgs/gmgd (61-63%).
4. **Try DINOv2 encoder** (`vit_base_patch14_dinov2.lvd142m`) — drop-in swap, often +0.01-0.02 AUC for free.
5. **Larger K** (64 or 128) — frozen encoder makes retrieval the bottleneck. Cheap to test.
6. **Per-relation thresholds** — TAR@FAR=0.01=0.062 is very low; per-class calibration would help strict thresholds.

### Artifacts

- Checkpoints: `output/001/checkpoints/{best.pt, final.pt, epoch_{5,10,15}.pt}`
- Logs: `output/001/logs/{train,test,evaluate}.log`
- Results: `output/001/results/{test_metrics_rocm.json, metrics_rocm.json, per_relation.json, confusion_matrix_rocm.png, roc_curve_rocm.png, per_relation_rocm.png}`

---
