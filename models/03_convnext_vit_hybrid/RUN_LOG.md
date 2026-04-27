# Run Log — Model 03: ConvNeXt + ViT Hybrid

This file logs every training run for this model with the exact configuration and final metrics. The goal is reproducibility: anyone (human or agent) reading this should be able to recreate any run by copying its launch command.

Newest run on top.

---

## Run 007 — 2026-04-15 — Stopped manually (LR too conservative)

**Status:** Stopped manually (no convergence)
**Outcome:** Test ROC AUC = **0.828**, best Val AUC = 0.8678 at epoch 11

### Launch command

```bash
SKIP_INSTALL=1 EPOCHS=100 BATCH_SIZE=16 NUM_WORKERS=2 \
  FREEZE_BACKBONES=1 UNFREEZE_AFTER=10 \
  PARTIAL_UNFREEZE=1 BACKBONE_LR_FACTOR=0.001 \
  bash models/03_convnext_vit_hybrid/AMD/run_pipeline.sh
```

### Configuration

| Param | Value |
|-------|-------|
| Dataset | fiw |
| Backbones | convnext_base + vit_base_patch16_224 (freeze→partial unfreeze ep 11) |
| Batch size | 16 |
| LR | 1e-3 |
| Backbone LR factor | 0.001 (→ ~1e-6 backbone LR after unfreeze) |
| Partial unfreeze | ConvNeXt stages[2,3] + ViT blocks[8-11] (~117M of 176M) |
| Loss | contrastive (margin=0.5, temperature=0.3) |
| Negative ratio | 2:1 (relation_matched train, random eval) |
| Dropout | 0.30 |
| Epochs | 100 (stopped at 15) |
| Patience | 50 |
| Seed | 42 |

### Training trajectory

- Best Val AUC: **0.8678** at epoch 11 (first epoch after partial unfreeze)
- Stopped manually at epoch 15 — backbone LR too low to recover after the post-unfreeze dip
- Trainable params: ~2.9M (frozen phase) → ~117M (after partial unfreeze)

### Test metrics

| Metric | Value |
|--------|-------|
| Test ROC AUC | 0.828 |
| Test Accuracy | 49.2%* |
| Test F1 | 0.654* |
| Avg Precision | 0.798 |
| TAR@FAR=0.1 | 0.487 |
| TAR@FAR=0.01 | 0.115 |

*threshold=0.5 default — accuracy/F1 not directly comparable.

### Notes

- Hypothesis: partial unfreeze + low backbone LR would prevent the overfitting Run 006 hit.
- Result: LR too low — the model basically froze again. Recommend factor=0.005 next.

### Artifacts

- `output/007/checkpoints/best.pt` (1.6 GB)
- `output/007/logs/train.log`
- `output/007/results/test_metrics_rocm.txt`, `metrics_rocm.json`, `backbone_analysis.json`

---

## Run 006 — 2026-04-13 — Stopped manually (overfitting post-unfreeze)

**Status:** Stopped manually (overfitting)
**Outcome:** Test ROC AUC = **0.848**, best Val AUC = 0.8728 at epoch 11

### Launch command

```bash
SKIP_INSTALL=1 EPOCHS=100 BATCH_SIZE=16 NUM_WORKERS=2 \
  FREEZE_BACKBONES=1 UNFREEZE_AFTER=10 \
  BACKBONE_LR_FACTOR=0.01 \
  bash models/03_convnext_vit_hybrid/AMD/run_pipeline.sh
```

### Configuration

| Param | Value |
|-------|-------|
| Dataset | fiw |
| Backbones | convnext_base + vit_base_patch16_224 (freeze→full unfreeze ep 11) |
| Batch size | 16 |
| LR | 1e-3 |
| Backbone LR factor | 0.01 (→ ~1e-5 backbone LR after unfreeze) |
| Full unfreeze | All 176M params after epoch 10 |
| Loss | contrastive (margin=0.5, temperature=0.3) |
| Negative ratio | 2:1 (relation_matched train, random eval) |
| Dropout | 0.30 |
| Epochs | 100 (stopped at 17) |
| Patience | 50 |
| Seed | 42 |

### Training trajectory

- Best Val AUC: **0.8728** at epoch 11 (first epoch after full unfreeze)
- Train loss collapsed from 0.0133 → 0.0038 in 6 epochs after unfreeze
- Val AUC declined from 0.8728 peak — overfitting confirmed; stopped manually at ep 17

### Test metrics

| Metric | Value |
|--------|-------|
| Test ROC AUC | **0.848** (best of all freeze/unfreeze runs) |
| Test Accuracy | 50.5%* |
| Test F1 | 0.659* |
| Avg Precision | **0.816** |
| TAR@FAR=0.1 | **0.511** (best across all M03 runs) |
| TAR@FAR=0.01 | 0.132 |

Backbone separability — best fused delta of all M03 runs (+0.346).

### Notes

- Nearly matches the no-freeze baseline (0.848 vs 0.850).
- Full unfreeze + factor=0.01 is the sweet spot identified so far.

### Artifacts

- `output/006/checkpoints/best.pt` (2.0 GB)
- `output/006/logs/train.log`
- `output/006/results/test_metrics_rocm.txt`, `metrics_rocm.json`, `backbone_analysis.json`

---

## Run 005 — 2026-04-13 — OOM crash at unfreeze

**Status:** OOM crash
**Outcome:** Test ROC AUC = **0.813** (frozen-phase checkpoint at ep 10)

### Launch command

```bash
SKIP_INSTALL=1 EPOCHS=100 BATCH_SIZE=32 NUM_WORKERS=4 \
  FREEZE_BACKBONES=1 UNFREEZE_AFTER=10 \
  BACKBONE_LR_FACTOR=0.01 \
  bash models/03_convnext_vit_hybrid/AMD/run_pipeline.sh
```

### Configuration

| Param | Value |
|-------|-------|
| Dataset | fiw |
| Batch size | **32** (the OOM cause) |
| Backbones | freeze → full unfreeze ep 11 |
| Backbone LR factor | 0.01 |
| Other params | same as Run 006 |

### Outcome

- 10 epochs in frozen phase succeeded (Val AUC peaked 0.8638 at ep 10).
- Epoch 11 unfroze 176M params — `HIP out of memory: tried to allocate 30 MiB`.
- best.pt corresponds to the frozen-phase ep 10 checkpoint (so Run 005 ≈ Run 004 configuration-wise at evaluation time).

### Test metrics (frozen-phase checkpoint)

| Metric | Value |
|--------|-------|
| Test ROC AUC | 0.813 |
| Test Accuracy | 48.2%* |
| Avg Precision | 0.790 |
| TAR@FAR=0.1 | 0.468 |

### Notes

- Lesson: with 12 GB VRAM, full unfreeze of 176M params requires batch_size ≤ 16. Run 006 reproduced with batch=16 and succeeded.

### Artifacts

- `output/005/checkpoints/best.pt` (695 MB, frozen-phase only)
- `output/005/logs/train.log` ends with traceback

---

## Run 004 — 2026-04-06 — Completed (frozen-only, early stop)

**Status:** Early stop (patience)
**Outcome:** Test ROC AUC = **0.823**, best Val AUC = 0.8672 at epoch 11

### Launch command

```bash
SKIP_INSTALL=1 EPOCHS=100 BATCH_SIZE=32 NUM_WORKERS=4 \
  FREEZE_BACKBONES=1 \
  bash models/03_convnext_vit_hybrid/AMD/run_pipeline.sh
```

### Configuration

| Param | Value |
|-------|-------|
| Dataset | fiw |
| Backbones | **always frozen** (only head trains) |
| Batch size | 32 |
| LR | 1e-3 |
| Loss | contrastive |
| Negative ratio | 2:1 |
| Dropout | 0.30 |
| Epochs | 100 (stopped at 25) |
| Patience | (default) |
| Seed | 42 |

### Training trajectory

- Trainable params: ~2.9M (head only)
- Best Val AUC: 0.8672 at epoch 11
- Time per epoch: ~44 min (frozen → fast)

### Test metrics

| Metric | Value |
|--------|-------|
| Test ROC AUC | 0.823 |
| Test Accuracy | **70.6%** (best of M03 runs) |
| Test F1 | **0.748** |
| Avg Precision | 0.797 |
| TAR@FAR=0.1 | 0.489 |
| Threshold | 0.75 (val-optimized) |

### Notes

- Lower bound for the freeze/unfreeze experiments — head alone insufficient (-2.7pp vs no-freeze).

### Artifacts

- `output/004/checkpoints/best.pt` (695 MB)
- `output/004/logs/train.log`
- `output/004/results/...` (full eval done)

---

## Run 003 — 2026-04-01 — Stopped manually (overfitting)

**Status:** Stopped manually
**Outcome:** Test ROC AUC = **0.845**, best Val AUC = 0.8771 at epoch 5

### Launch command

```bash
# (Approximate — predates the partial-unfreeze flags)
SKIP_INSTALL=1 EPOCHS=50 BATCH_SIZE=8 NUM_WORKERS=4 \
  DROPOUT=0.40 NEGATIVE_RATIO=3.0 \
  bash models/03_convnext_vit_hybrid/AMD/run_pipeline.sh
```

### Configuration

| Param | Value |
|-------|-------|
| Dataset | fiw |
| Backbones | **no freeze** (trained from epoch 1) |
| Batch size | 8 |
| Dropout | 0.40 (higher regularization) |
| Negative ratio | 3:1 |
| Epochs | 50 (stopped at 10) |
| Seed | 42 |

### Test metrics

| Metric | Value |
|--------|-------|
| Test ROC AUC | 0.845 |
| Avg Precision | 0.811 |
| TAR@FAR=0.01 | **0.138** (best of M03 runs) |
| Val AUC peak | 0.8771 (ep 5) |

### Notes

- Higher dropout (0.40) gave best ViT separability (+0.311) but Test AUC slightly below Run 002.

---

## Run 002 — 2026-03-23 — Stopped manually (overfitting)

**Status:** Stopped manually
**Outcome:** Test ROC AUC = **0.850** (best M03 result), best Val AUC = 0.8851 at epoch 5

### Launch command

```bash
# (Approximate — predates the partial-unfreeze flags)
SKIP_INSTALL=1 EPOCHS=50 BATCH_SIZE=8 NUM_WORKERS=4 \
  DROPOUT=0.25 NEGATIVE_RATIO=2.0 \
  bash models/03_convnext_vit_hybrid/AMD/run_pipeline.sh
```

### Configuration

| Param | Value |
|-------|-------|
| Dataset | fiw |
| Backbones | **no freeze** (trained from epoch 1) |
| Batch size | 8 |
| Dropout | 0.25 |
| Negative ratio | 2:1 |
| Epochs | 50 (stopped at 13) |
| Seed | 42 |

### Test metrics

| Metric | Value |
|--------|-------|
| Test ROC AUC | **0.850** ← Best Model 03 result |
| Avg Precision | **0.816** |
| TAR@FAR=0.1 | 0.487 |
| Val AUC peak | 0.8851 (ep 5) |

### Notes

- Best M03 run overall — ties with Model 02 R031 at 0.850 AUC.
- Epoch 5 was the peak; train loss kept falling but val AUC declined → manual stop.

---

## Run 001 — 2026-03-23 — OOM crash (no checkpoint)

**Status:** OOM crash
**Outcome:** No checkpoint, no metrics — batch_size too large for the initial run.

### Notes

- First attempt at Model 03. Batch size config caused OOM before epoch 1 completed.
- Drove the lesson that 176M-param hybrid needs batch ≤ 16 with grad accumulation.

---
