# Run Log — Model 12: RGCK-Net (Region-Guided Cross Kinship Network)

This file logs every training run for this model with the exact configuration and final metrics. The goal is reproducibility: anyone (human or agent) reading this should be able to recreate any run by copying its launch command.

Newest run on top.

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
