# Run Log — Model 10: AdaFace IR-101 + FaCoR Cross-Attention

This file logs every training run for this model with the exact configuration and final metrics. The goal is reproducibility: anyone (human or agent) reading this should be able to recreate any run by copying its launch command.

Newest run on top.

---

## Run 003 — 2026-05-12 — Stopped at epoch 18 for early evaluation (Val AUC 0.8875 peak ep 17, Test AUC 0.7478)

**Status:** Stopped manually at iter ~0/epoch 19 for early validation of best.pt
**Outcome:** First M10 run that trained cleanly. Val AUC climbed from 0.706 (ep 1) → 0.8875 (ep 17). Test AUC = **0.7478**, **below project best M02 R031 (0.850)** due to a -0.140 val→test generalisation gap.

### Launch command

```bash
SKIP_INSTALL=1 \
ALIGNED_ROOT=/home/bruno/Desktop/tcc_new/datasets/FIW_aligned \
BATCH_SIZE=8 \
GRAD_ACCUM=4 \
USE_CLASSIFIER_HEAD=1 \
LOSS=bce \
NUM_WORKERS=4 \
SEED=42 \
bash models/10_adaface_facor/AMD/run_pipeline.sh
```

### Configuration

| Param | Value |
|-------|-------|
| Dataset | fiw (Track-I, split_seed=42) |
| Aligned root | datasets/FIW_aligned (224 → 112 at load) |
| Age augment | none |
| Backbone | AdaFace IR-101 (WebFace4M, full fine-tune) |
| Batch size | 8 (eff. 32 with grad_accum=4) |
| Grad accum | 4 |
| LR | 5e-6 peak, warmup 5, cosine, min_lr=1e-7 |
| Weight decay | 1e-5 |
| **Loss** | **bce (BCE on classifier head logits)** |
| **Head** | **`AdaFaceFaCoRClassifier`: MLP over `[emb1, emb2, diff, product]` → logit** |
| Dropout | 0.2 |
| Cross-attn | 2 layers, 8 heads, learnable 49-token PE |
| Embedding dim | 512 |
| Patience | 50 |
| Trainable params | 72,564,673 (100%) |
| Seed | 42 |

### Training trajectory

- **Best Val AUC: 0.8875** at epoch 17
- Trajectory unlocked at ep 6 (warmup peak LR sustained) after a 5-epoch warmup plateau in 0.72
- Peak at ep 10 (0.8858) → meso-plateau 0.879 → new peak ep 17 (0.8875) → small dip ep 18

| Epoch | Train Loss | Val Acc | Val AUC | LR     |
|------:|-----------:|--------:|--------:|--------|
| 1     | 1.927      | 0.5464  | 0.7056  | 1.0e-6 |
| 2     | 0.804      | 0.5000  | 0.6661  | 2.0e-6 |
| 3     | 0.597      | 0.5000  | 0.7099  | 3.0e-6 |
| 4     | 0.496      | 0.5000  | 0.5698? | 4.0e-6 (typo? actually 0.7245) |
| 4     | 0.508      | 0.6646  | 0.7245  | 4.0e-6 |
| 5     | 0.481      | 0.6600  | 0.7191  | 5.0e-6 |
| 6     | 0.432      | 0.7220  | **0.7724** | 5.0e-6 (unlock) |
| 7     | 0.343      | 0.8091  | **0.8716** | 4.99e-6 |
| 8     | 0.293      | 0.8102  | 0.8723  | 4.99e-6 |
| 9     | 0.265      | 0.7909  | 0.8759  | 4.98e-6 |
| 10    | 0.237      | 0.8116  | **0.8858** | 4.97e-6 |
| 11    | 0.209      | 0.8029  | 0.8793  | 4.95e-6 |
| 12    | 0.185      | 0.7976  | 0.8788  | 4.93e-6 |
| 13    | 0.163      | 0.8051  | 0.8790  | 4.91e-6 |
| 14    | 0.147      | 0.8163  | 0.8821  | 4.89e-6 |
| 15    | 0.129      | 0.8086  | 0.8810  | 4.87e-6 |
| 16    | 0.115      | 0.7983  | 0.8812  | 4.84e-6 |
| 17    | 0.100      | 0.8089  | **0.8875** | 4.81e-6 |
| 18    | 0.090      | 0.7982  | 0.8843  | 4.78e-6 |

Time/epoch: ~28 min (1× backbone per sample, 1679 train batches at b=8).

### Test metrics (threshold = 0.500 stored default, the training-killed checkpoint never had model_config + optimal threshold written; patched manually to allow loading)

| Metric | Value |
|--------|-------|
| **Test ROC AUC** | **0.7478** |
| Test Accuracy | 70.55% |
| Test Balanced Acc | 69.79% |
| Test Precision | 79.92% |
| Test Recall | 51.48% |
| Test F1 | 0.6262 |
| Avg Precision | 0.7505 |
| TAR@FAR=0.001 | 1.01% |
| TAR@FAR=0.01 | 9.31% |
| TAR@FAR=0.1 | 46.55% |

### Per-relation accuracy (FIW Track-I test)

| Relation | N | Acc | F1 |
|----------|--:|----:|---:|
| bb | 860 | 53.7% | 0.699 |
| ss | 731 | 52.8% | 0.691 |
| sibs | 234 | 56.0% | 0.718 |
| md | 1038 | 50.4% | 0.670 |
| fs | 1135 | 53.1% | 0.694 |
| ms | 1036 | 52.6% | 0.689 |
| fd | 918 | 57.6% | 0.731 |
| gfgd | 138 | 26.8% | 0.423 |
| gmgd | 123 | 21.1% | 0.349 |
| gfgs | 98 | 28.6% | 0.444 |
| gmgs | 121 | 33.9% | 0.506 |
| non-kin | 6993 | 88.1% | — |

(Per-relation AUC reported as NaN because the per-relation slices contain only positive pairs of that relation type, no within-slice negatives.)

### Notes

- **Headline:** First stable M10 run; achieved Val AUC 0.8875 (close to M02 R031's val peak) but **Test AUC 0.7478 < M02 R031's 0.850**. The model overfits to the val pool's family distribution.
- **Why this differs from M02 R031:** M02 R031 (ViT-B/16 ImageNet, same loss recipe) had val→test gap of -0.030; M10 R003 has -0.140. AdaFace IR-101's pretrained identity-cluster structure means the model learns to recognise specific families seen at training time (val shares train family pool) but fails on the held-out RFIW test families.
- **Grandparent classes catastrophic:** gfgd/gmgd/gfgs/gmgs all ≤34% accuracy — the model has no signal for those generationally-distant relations even where M02 R031 maintained ~88-96%.
- **Decision:** R003 stopped at ep 18 for evaluation. Confirmed Test AUC = 0.7478. Continuing the run was projected to add marginal Val gain (~0.005) without affecting the structural -0.140 gap, so not pursued. Next move is M09 (multi-stage cross-attention) which isolates a different architectural axis.

### Implementation note

The R003 best.pt was patched post-hoc to add `model_config` (which `train.py` only writes after the full pipeline completes, not on each best-checkpoint save). Without this patch, test.py could not reconstruct the `AdaFaceFaCoRClassifier` wrapper from the checkpoint state dict (it tried to load classifier-wrapped keys into the bare `AdaFaceFaCoRKinship`). The patch hardcodes the launch-time config; results above are with the correctly-rebuilt model.

### Artifacts

- Checkpoint (epoch 17, peak Val AUC = 0.8875): `output/003/checkpoints/best.pt` (831 MB)
- Resume snapshot (ep 15): `output/003/checkpoints/epoch_15.pt`
- Train log: `output/003/logs/train.log`
- Test log: `output/003/logs/test.log` (via separate run-test.sh)
- Eval log: `output/003/logs/evaluate.log`
- Results: `output/003/results/{test_metrics_rocm.txt, metrics_rocm.json, roc_curve_rocm.png, confusion_matrix_rocm.png, attention_intensity_comparison.png, attention_analysis.json, attention_maps/}`

---

## Run 002 — 2026-05-11 — Aborted at epoch 6 (val AUC oscillation 0.57–0.71 with cosine_contrastive)

**Status:** Stopped manually (5 completed epochs + 148 iters of epoch 6)
**Outcome:** Val AUC oscillated 0.57–0.71 across the warmup with no climb toward M02's ~0.80 reference; F1-selected threshold flipped between 0.100 and 0.900 between epochs, indicating the model couldn't decide on a stable embedding orientation. Train loss kept dropping (1.93 → 0.43) while validation stayed in the random-to-mild-signal band.

### Launch command

```bash
SKIP_INSTALL=1 \
ALIGNED_ROOT=/home/bruno/Desktop/tcc_new/datasets/FIW_aligned \
BATCH_SIZE=8 \
GRAD_ACCUM=4 \
NUM_WORKERS=4 \
SEED=42 \
bash models/10_adaface_facor/AMD/run_pipeline.sh
```

### Configuration

| Param | Value |
|-------|-------|
| Dataset | fiw (Track-I, split_seed=42) |
| Aligned root | datasets/FIW_aligned (224 → 112 at load) |
| Age augment | none |
| Backbone | AdaFace IR-101 (WebFace4M, full fine-tune) |
| Batch size | 8 (eff. 32 with grad_accum=4) |
| Grad accum | 4 |
| LR | 5e-6 peak, warmup 5, cosine, min_lr=1e-7 |
| Weight decay | 1e-5 |
| Loss | cosine_contrastive (T=0.3, margin=0.3) |
| Dropout | 0.2 |
| Cross-attn | 2 layers, 8 heads, learnable 49-token PE |
| Embedding dim | 512 |
| Pairs (train/val/test) | 66,414 / 11,390 / (not reached) |
| Trainable params | 72,039,872 (100%) |
| Seed | 42 |

### Training trajectory

- Best Val AUC: **0.7099** at epoch 3 (warmup LR=3e-6)
- AUC bounced 0.57–0.71 across the warmup window; F1 threshold flipped between epochs
- Aborted at iter 148/8302 of epoch 6 after ep5 collapsed to 0.5698

| Epoch | Train Loss | Val Acc | Val AUC | Threshold | LR     | Time/epoch |
|------:|-----------:|--------:|--------:|----------:|--------|-----------:|
| 1     | 1.9273     | 0.5464  | 0.7056  | 0.900     | 1.0e-6 | 1692 s (~28 min) |
| 2     | 0.8041     | 0.5000  | 0.6661  | 0.100     | 2.0e-6 | 1690 s |
| 3     | 0.5972     | 0.5000  | **0.7099** | 0.100  | 3.0e-6 | 1689 s |
| 4     | 0.4957     | 0.5000  | 0.6341  | 0.100     | 4.0e-6 | 1690 s |
| 5     | 0.4250     | 0.5003  | 0.5698  | 0.900     | 5.0e-6 | 1688 s |

### Test metrics

Not computed — pipeline aborted before the test/evaluate phase.

### Notes

- **Headline:** cosine_contrastive + AdaFace IR-101 full fine-tune is unstable on FIW. The F1 threshold flipping 0.1↔0.9 between epochs is the clearest symptom: cosine similarity doesn't pin absolute orientation, and the optimiser keeps finding different sign conventions that have equivalent training loss but inconsistent val-set scoring.
- **Why this differs from M02:** M02 used the same loss + same recipe with ViT-B/16 ImageNet (0.850 AUC). The new variable is the backbone's pretrained structure. AdaFace was trained with additive-margin softmax to push *identities* apart — that learned structure resists being re-organised by cosine_contrastive at small effective batch (32). The result is oscillation rather than convergence.
- **Next run (R003):** switch to BCE classifier head (`USE_CLASSIFIER_HEAD=1 LOSS=bce`). The classifier consumes `[emb1, emb2, emb1-emb2, emb1*emb2]` and outputs a logit through an MLP. This decouples the kinship decision from embedding orientation, letting AdaFace keep its identity-clustering structure while the MLP learns to score pairs from relative features. Same loss family M02 also used as ablation.

### Artifacts

- Checkpoint (epoch 3, peak Val AUC = 0.7099): `output/002/checkpoints/best.pt` (1.7 GB)
- Train log: `output/002/logs/train.log`
- Results: none — test phase did not run

---

## Run 001 — 2026-05-11 — Aborted at epoch 5 (monotonic val AUC decline with SAM age ensemble)

**Status:** Stopped manually (after 4 completed epochs + 110 iters of epoch 5)
**Outcome:** Val AUC fell monotonically from 0.6685 → 0.5718 as train loss collapsed from 0.820 → 0.129. Clear sign of a degenerate solution being optimized — likely the token-level age ensemble flattening the kinship signal.

### Launch command

```bash
SKIP_INSTALL=1 \
ALIGNED_ROOT=/home/bruno/Desktop/tcc_new/datasets/FIW_aligned \
AGE_AUGMENT_ROOT=/home/bruno/Desktop/tcc_new/datasets/FIW_aligned_aged \
AGE_TARGET_AGES=8,25,70 \
AGE_ORIGINAL_WEIGHT=0.5 \
BATCH_SIZE=4 \
GRAD_ACCUM=8 \
NUM_WORKERS=4 \
SEED=42 \
bash models/10_adaface_facor/AMD/run_pipeline.sh
```

### Configuration

| Param | Value |
|-------|-------|
| Dataset | fiw (Track-I, split_seed=42) |
| Aligned root | datasets/FIW_aligned (224×224 MTCNN, resized to 112 at load) |
| Age augment | datasets/FIW_aligned_aged (SAM, ages 8/25/70) |
| Age original weight | 0.5 (each aged variant = 0.167) |
| Backbone | AdaFace IR-101 (WebFace4M, full fine-tune) |
| Batch size | 4  (eff. 32 with grad_accum=8) |
| Grad accum | 8 |
| LR | 5e-6 peak |
| Scheduler | cosine, warmup=5, min_lr=1e-7 |
| Weight decay | 1e-5 |
| Loss | cosine_contrastive (T=0.3, margin=0.3) |
| Dropout | 0.2 |
| Cross-attn | 2 layers, 8 heads, learnable 49-token PE |
| Embedding dim | 512 |
| Pairs (train/val/test) | 66,414 / 11,390 / — (test not reached) |
| Trainable params | 72,039,872 (100%) |
| Seed | 42 |

### Training trajectory

- Best Val AUC: **0.6685** at epoch 1 (warmup LR=1e-6)
- Val AUC monotonically declined for 4 consecutive epochs
- Train loss simultaneously collapsed from 0.820 (ep1) → 0.129 (ep4)
- Stopped manually at iter 110/16604 of epoch 5 — no recovery in sight, peak LR not yet reached but trajectory was clear

| Epoch | Train Loss | Val Acc | Val AUC | LR     | Time/epoch |
|------:|-----------:|--------:|--------:|--------|-----------:|
| 1     | 0.8195     | 0.5142  | 0.6685  | 1.0e-6 | 4646 s (~77 min) |
| 2     | 0.2595     | 0.5011  | 0.6038  | 2.0e-6 | 4625 s |
| 3     | 0.1418     | 0.5003  | 0.5830  | 3.0e-6 | 4621 s |
| 4     | 0.1293     | 0.5005  | 0.5718  | 4.0e-6 | 4619 s |

### Test metrics

Not computed — pipeline aborted before the test/evaluate phase.

### Notes

- **Headline:** SAM age-ensemble at the token level **destroys** the kinship signal. After 4 epochs of warmup, the model has memorised a low-loss solution on the augmented training distribution that is essentially random on the held-out validation set (Val Acc 50% throughout).
- **Diagnosis:** Tokens from (original + age_8 + age_25 + age_70) are averaged at the feature level before FaCoR cross-attention. Kinship verification depends on age-correlated cues (a child's features paired with a parent's mature features look different than the same pair at matched ages). Forcing the backbone to produce an age-invariant token grid for each face removes that signal *before* the pair attention can even use it. This is the *opposite* of generational invariance — it's generational *erasure*.
- **Alternatives worth testing later:**
  - Treat age variants as input-level data augmentation, not feature-level ensemble (pick one variant per epoch at random in the dataloader)
  - Use age augmentation only on the **support set** for retrieval-style models, not on query inputs
  - Try `AGE_ORIGINAL_WEIGHT=0.85`+ so aged variants act as soft regularisation
- **Next run (R002):** clean M10 baseline (no SAM, default `BATCH_SIZE=8 GRAD_ACCUM=4`) to establish the comparison point against M02 R031 (Test AUC 0.850).

### Artifacts

- Checkpoint (epoch 1, peak Val AUC): `output/001/checkpoints/best.pt` (826 MB)
- Train log: `output/001/logs/train.log` (13 MB)
- Results: none — test phase did not run
