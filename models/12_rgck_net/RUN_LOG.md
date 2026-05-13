# Run Log — Model 12: RGCK-Net (Region-Guided Cross Kinship Network)

This file logs every training run for this model with the exact configuration and final metrics. The goal is reproducibility: anyone (human or agent) reading this should be able to recreate any run by copying its launch command.

Newest run on top.

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
