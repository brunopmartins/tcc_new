# Run Log — Model 11: AdaFace + FaCoRNet treatments

This file logs every training run for this model with the exact configuration and final metrics. The goal is reproducibility: anyone (human or agent) reading this should be able to recreate any run by copying its launch command.

Newest run on top.

---

## Run 001 — 2026-05-13 — Stopped at epoch 13 (peak Val AUC 0.8987 ep 9; Test AUC 0.7707, val→test gap -0.128)

**Status:** Stopped manually at end of epoch 13 (4 epochs below peak)
**Outcome:** Three earlier configurations (v1/v2/v3) collapsed to predict-all-kin due to a project-side bug in `cosine_contrastive` and `relation_guided` losses (they ignore batch labels — see [discussao.md](discussao.md)). The safe v4 path uses the M09 R001 stack (multistage + BCE + classifier head) with **`relation_matched` hard negatives** as the single FaCoRNet treatment applied. Val AUC reached **0.8987 (ep 9), the highest AdaFace-based Val AUC in the project**, surpassing M09 R001's lifetime peak (0.8919). **However, Test AUC was 0.7707 — WORSE than M09 R001 (0.7982) by -0.0275**. Val→test gap widened to -0.128 (vs M09 R001's -0.094). Hypothesis "hard negatives improve test generalisation" REJECTED.

### Attempts before v4 (all killed within 2 epochs, no test phase)

| Attempt | Config                                                    | Outcome                                              |
|---------|-----------------------------------------------------------|------------------------------------------------------|
| v1      | `relation_guided` loss, temp 0.07, multistage             | Train loss 0.80→0.02 in 1 ep; per-rel uniform 1.000 |
| v2      | `relation_guided` loss, temp 0.3, multistage              | Slower collapse but same degenerate state            |
| v3      | `cosine_contrastive` loss, temp 0.3, multistage           | Same degenerate state                                |
| **v4**  | **BCE + classifier head + multistage + `relation_matched` negs** | **Healthy training, peak Val AUC 0.8987**     |

Root cause of v1-v3 failure: `CosineContrastiveLoss.forward` and `RelationGuidedContrastiveLoss.forward` accept `labels` (or attention) but **never use them** in the loss computation. Every `(emb1_i, emb2_i)` pair is treated as a positive in the InfoNCE numerator, regardless of `label`. Combined with `relation_matched` hard negatives, the model was actively trained to pull non-kin pairs together.

### Launch command (v4)

```bash
SKIP_INSTALL=1 \
ALIGNED_ROOT=/home/bruno/Desktop/tcc_new/datasets/FIW_aligned \
BATCH_SIZE=4 \
GRAD_ACCUM=8 \
USE_MULTISTAGE=1 \
USE_CLASSIFIER_HEAD=1 \
LOSS=bce \
NUM_WORKERS=4 \
SEED=42 \
bash models/11_facornet_adaface/AMD/run_pipeline.sh
```

(Defaults `TRAIN_NEGATIVE_STRATEGY=relation_matched` and `EVAL_NEGATIVE_STRATEGY=relation_matched` come from the pipeline script.)

### Configuration

| Param | Value |
|-------|-------|
| Dataset | fiw (Track-I, split_seed=42) |
| Aligned root | datasets/FIW_aligned (224 → 112 at load) |
| Backbone | AdaFace IR-101 (WebFace4M, **full fine-tune**) |
| Architecture | Multi-stage cross-attn (stages 3 + 4 of IR-101) — same as M09 |
| Head | `AdaFaceMultiStageKinshipClassifier`: MLP over `[emb1, emb2, diff, product]` → logit |
| Loss | bce |
| **Train neg strategy** | **`relation_matched`** (FaCoRNet hard negatives) |
| **Eval neg strategy** | **`relation_matched`** |
| Batch | 4 (eff. 32 with grad_accum=8) |
| LR | 5e-6 peak, warmup 5, cosine, min_lr=1e-7 |
| Weight decay | 1e-5 |
| Dropout | 0.2 |
| Embedding dim | 512 |
| Patience | 50 |
| Trainable params | 70,876,353 (100% of total) |
| Time/epoch | ~40 min |
| Seed | 42 |

### Training trajectory (v4)

- Best Val AUC: **0.8987** at epoch 9 (best.pt) — highest AdaFace-based Val AUC in the project
- Stopped manually at epoch 13 (4 epochs below peak, train loss continuing to drop)
- Time per epoch: ~40 min (identical to M09 R001)

| Epoch | Train Loss | Val Acc | Val AUC | Thr | Note |
|------:|-----------:|--------:|--------:|-----|------|
| 1 | 0.6560 | 65.7 % | 0.7207 | 0.350 | warmup 1/5 |
| 2 | 0.6077 | 64.7 % | 0.7260 | 0.200 |  |
| 3 | 0.5705 | 64.5 % | 0.7174 | 0.100 | warmup drift |
| 4 | 0.5500 | 64.1 % | 0.7078 | 0.100 | pre-unlock dip |
| 5 | 0.5058 | 78.2 % | **0.8349** | 0.100 | **+0.127 unlock** (M09 R001 same ep: 0.8237) |
| 6 | 0.4996 | 82.0 % | 0.8833 | 0.100 | climb |
| 7 | 0.4552 | 82.7 % | 0.8913 | 0.100 | climb (effectively at M09 R001 lifetime peak) |
| 8 | 0.4218 | 81.4 % | 0.8895 | 0.100 | small dip |
| **9** | **0.3947** | **81.4 %** | **0.8987** | 0.100 | **lifetime peak — best.pt, surpasses M09 R001** |
| 10 | 0.3574 | 81.7 % | 0.8916 | 0.100 | dip |
| 11 | 0.3291 | 82.1 % | 0.8943 | 0.100 | bounce |
| 12 | 0.3032 | 82.2 % | 0.8917 | 0.100 | dip |
| 13 | 0.2725 | 80.7 % | 0.8856 | 0.100 | -0.013 from peak — **manual stop** |

### Test metrics (threshold 0.500)

Stored threshold on best.pt defaulted to 0.500 (training killed before `update_checkpoint_metadata`). The val-phase F1-optimal at ep 9 was 0.100. AUC/AP threshold-invariant.

| Metric | Value |
|--------|-------|
| **Test ROC AUC** | **0.7707** |
| Test Accuracy | 70.57 % |
| Test Balanced Acc | 69.87 % |
| Test Precision | 78.56 % |
| Test Recall | 53.05 % |
| Test F1 | 0.6333 |
| Test Avg Precision | 0.7629 |
| TAR @ FAR=0.001 | 1.26 % |
| TAR @ FAR=0.01 | 10.98 % |
| TAR @ FAR=0.1 | 45.12 % |
| **Val→test AUC gap** | **-0.128** (vs M09 R001 -0.094, M09 R002 -0.107) |

### Per-relation accuracy (FIW Track-I test)

| Relation | N | M09 R001 | M09 R002 | **M11 v4** | Δ M11 vs M09 R001 |
|----------|--:|---------:|---------:|-----------:|------------------:|
| bb | 860 | 64.2 % | 58.1 % | **56.6 %** | -7.6 pp |
| ss | 731 | 62.9 % | 59.9 % | 53.9 % | -9.0 pp |
| sibs | 234 | 64.5 % | 59.8 % | **50.0 %** | **-14.5 pp** ⚠ |
| md | 1038 | 53.9 % | 52.6 % | 53.1 % | -0.8 |
| fs | 1135 | 59.1 % | 56.7 % | 54.1 % | -5.0 |
| ms | 1036 | 57.3 % | 55.5 % | 51.5 % | -5.8 |
| fd | 918 | 63.6 % | 59.3 % | 61.7 % | -1.9 |
| **gfgd** | 138 | 31.2 % | 50.7 % | **37.7 %** | **+6.5** |
| **gfgs** | 98 | 30.6 % | 33.7 % | **36.7 %** | **+6.1** |
| gmgd | 123 | 31.7 % | 33.3 % | 22.0 % | **-9.7** ⚠ |
| gmgs | 121 | 37.2 % | 37.2 % | 28.9 % | **-8.3** ⚠ |
| non-kin | 6993 | 84.7 % | 86.2 % | 86.7 % | +2.0 |

`gfgd` and `gfgs` (grandfather classes) improved by +6 pp. All sibling classes (bb, ss, sibs) regressed by 8-15 pp. `gmgd` and `gmgs` (grandmother classes) regressed by 8-10 pp. Net Test AUC worse.

### Notes

- **Val→test gap pattern is now consistent across M09 R002 and M11 v4**: harder training distributions (balanced sampling in M09 R002, hard negatives in M11 v4) consistently raise Val AUC but *hurt* Test AUC. Two independent FaCoRNet-inspired interventions on M09 R001 have now reproduced this.
- **The M09 R001 stack (BCE + classifier head + multistage + random negs) is the AdaFace-based local maximum** as far as we can tell. Adding FaCoRNet treatments doesn't improve generalization.
- The bug discovery in `discussao.md` is documented. The pipeline defaults were updated post-discovery to use the safe stack (`LOSS=bce`, `USE_CLASSIFIER_HEAD=1`, `USE_MULTISTAGE=1`).
- A future faithful FaCoRNet implementation (label-aware Rel-Guide, paper-faithful `M(beta)/s` with s=500, positives-only batches with implicit negatives) is still pending.

### Artifacts

- Checkpoint (epoch 9, Val AUC 0.8987): `output/002/checkpoints/best.pt` (852 MB) — patched with `model_config` (use_multistage=True)
- Train log: `output/002/logs/train.log`
- Test log: `output/002/logs/test.log`
- Evaluate log: `output/002/logs/evaluate.log`
- Results: `output/002/results/{test_metrics_rocm.txt, metrics_rocm.json, roc_curve_rocm.png, confusion_matrix_rocm.png, attention_intensity_comparison.png, attention_analysis.json, attention_maps/}`

---
