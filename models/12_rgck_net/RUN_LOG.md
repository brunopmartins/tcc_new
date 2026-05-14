# Run Log — Model 12: RGCK-Net (Region-Guided Cross Kinship Network)

This file logs every training run for this model with the exact configuration and final metrics. The goal is reproducibility: anyone (human or agent) reading this should be able to recreate any run by copying its launch command.

Newest run on top.

---

## Run 004 — 2026-05-14 — Stopped at epoch 8 (peak Val AUC 0.9354 ep 4 — project max; Test AUC 0.8473. Intended hard-negs test — Errata: hypothesis NOT actually tested)

**Status:** Stopped manually at epoch 8 (Val AUC -0.010 from peak, train loss still descending, grandparent classes collapsing in val)

> ⚠ **Errata (post-run, 2026-05-14):** the `relation_matched` sampler at
> [models/shared/dataset.py:433](../shared/dataset.py#L433) does NOT preserve relation/role —
> line 464 picks a relation but line 465 hardcodes `"non-kin"`, so
> the algorithm is identical to the random sampler at
> [models/shared/dataset.py:512](../shared/dataset.py#L512) up to seed offset (270 vs 200).
> Measured: 0.09 % overlap on train negatives, 100 % on test (test
> pairs come from RFIW Track-I lists, not the sampler). R004 actually
> tested **negative-sampler reseed variance** with all else identical
> to R002 — not hard negatives. The original "hard negs rejected" /
> "cross-experiment confirmation with M11 v4" narrative below is
> withdrawn; both rest on the same broken sampler. The numerical
> results are valid; the causal interpretation is not. See
> [run-review/run-004.md](run-review/run-004.md) Errata.

**Outcome (original framing, kept for the record):** Phase 6 of
`proposta_rgck_net_kinship.md` §38 — same stack as R002 (Phase 2
partial unfreeze + classifier head + BCE) + `relation_matched` ~~hard
negatives~~ negatives-with-reseed on both train and eval. Val AUC
peaked at **0.9354** (ep 4) — highest in the project, +0.003 over
R002. Test AUC = 0.8473 — -0.009 below R002's 0.8564. All TAR@FAR
levels regressed. Val→test gap -0.088 vs R002 -0.076. All 11 kin
classes regressed; non-kin +1.2 pp.

**Corrected interpretation:** the R004 → R002 deltas measure
sensitivity of partial-FT to the specific draw of training negatives.
This sets a noise floor of roughly ±0.01 Test AUC for any future
single-knob comparison that doesn't control the negative-sampler seed.

### Launch command

```bash
SKIP_INSTALL=1 \
ALIGNED_ROOT=/home/bruno/Desktop/tcc_new/datasets/FIW_aligned \
BATCH_SIZE=8 \
GRAD_ACCUM=4 \
UNFREEZE_LAST_STAGE=1 \
LEARNING_RATE=1e-5 \
TRAIN_NEGATIVE_STRATEGY=relation_matched \
EVAL_NEGATIVE_STRATEGY=relation_matched \
NUM_WORKERS=4 \
SEED=42 \
bash models/12_rgck_net/AMD/run_pipeline.sh
```

### Changes from Run 002

| Parameter | R002 | **R004** |
|---|---|---|
| `TRAIN_NEGATIVE_STRATEGY` | random | **`relation_matched`** |
| `EVAL_NEGATIVE_STRATEGY` | random | **`relation_matched`** |
| All other knobs | — | identical |

Single-knob change: negative sampling. `relation_matched` constructs
negatives by drawing the partner of a positive `fs` pair from another
`fs` pair (different family but same role). This produces visually
plausible hard negatives that force the model to discriminate on
kinship-specific cues rather than role cues.

### Configuration

Same as R002:

| Param | Value |
|-------|-------|
| Backbone | AdaFace IR-101 (stages 1-3 frozen, stage 4 + output_layer trainable) |
| Trainable params | 31,554,818 / 70,740,674 (44.61%) |
| Loss | BCE on classifier logit (no SupCon) |
| Batch | 8 × grad-accum 4 (eff. 32) |
| LR | 1e-5 peak, warmup 5, cosine, min_lr 1e-6 |
| Weight decay | 1e-4 |
| Dropout | 0.2 |
| **Negative strategy** | **`relation_matched`** both train and eval |
| Time/epoch | ~25.8 min |

### Training trajectory

| Epoch | Train Loss | Val Acc | Val AUC | Thr | LR | R002 same ep | Δ R004-R002 |
|------:|-----------:|--------:|--------:|----:|-----|------:|------:|
| 1 | 0.5855 | 78.1 % | 0.8676 | 0.450 | 2.0e-6 | 0.8644 | +0.003 |
| 2 | 0.4043 | 84.0 % | 0.9274 | 0.350 | 4.0e-6 | 0.9259 | +0.001 |
| 3 | 0.3287 | 85.8 % | 0.9333 | 0.350 | 6.0e-6 | 0.9311 | +0.002 |
| **4** | **0.2855** | **86.2 %** | **0.9354 (peak)** | 0.150 | 8.0e-6 | 0.9323 | **+0.003 (project max)** |
| 5 | 0.2514 | 86.4 % | 0.9326 | 0.250 | 1.0e-5 | 0.9306 | +0.002 |
| 6 | 0.2253 | 86.2 % | 0.9350 | 0.150 | 1.0e-5 | 0.9284 | +0.007 |
| 7 | 0.1999 | 85.9 % | 0.9313 | 0.100 | 9.99e-6 | 0.9230 | +0.008 |
| 8 | 0.1798 | 85.4 % | 0.9259 | 0.100 | 9.98e-6 | (n/a) | — |

R004 stayed above R002 in every epoch (+0.001 to +0.008). Val AUC at
the peak (ep 4) was the highest the project has ever produced: **0.9354**.
But the train→val gap widened in later epochs as overfitting set in
faster than in R002.

### Test metrics (threshold 0.500)

| Metric | M12 R002 (HEADLINE) | **M12 R004** | Δ |
|---|---:|---:|---:|
| **Test ROC AUC** | **0.8564** | **0.8473** | **-0.009** |
| Test Accuracy | 76.79 % | 75.75 % | -1.0 pp |
| Test Balanced Acc | 76.48 % | 75.33 % | -1.2 pp |
| Test F1 | 0.7402 | 0.7211 | -0.019 |
| Test Precision | 79.82 % | 80.29 % | +0.5 pp |
| Test Recall | 69.00 % | 65.44 % | -3.6 pp |
| Test Avg Precision | 0.8389 | 0.8287 | -0.010 |
| TAR @ FAR=0.001 | 4.18 % | 2.01 % | -2.2 pp |
| TAR @ FAR=0.01 | 17.58 % | 14.71 % | -2.9 pp |
| TAR @ FAR=0.1 | 57.11 % | 56.06 % | -1.0 pp |
| **Val→test AUC gap** | **-0.076** | **-0.088** | gap widened |

### Per-relation accuracy (FIW Track-I test, threshold 0.500)

| Relation | N | M12 R002 | **M12 R004** | Δ |
|----------|--:|---------:|-------------:|---:|
| bb | 860 | 75.4 % | 72.4 % | -3.0 pp |
| ss | 731 | 75.9 % | 72.5 % | -3.4 pp |
| sibs | 234 | 79.5 % | 76.9 % | -2.6 pp |
| md | 1038 | 68.7 % | 64.8 % | -3.9 pp |
| fs | 1135 | 68.6 % | 64.4 % | -4.2 pp |
| ms | 1036 | 69.4 % | 66.1 % | -3.3 pp |
| fd | 918 | 68.4 % | 65.6 % | -2.8 pp |
| **gfgd** | 138 | 52.2 % | **46.4 %** | **-5.8 pp** |
| gfgs | 98 | 39.8 % | 38.8 % | -1.0 pp |
| gmgd | 123 | 36.6 % | 34.2 % | -2.4 pp |
| **gmgs** | 121 | 44.6 % | **33.9 %** | **-10.7 pp** ⚠ |
| **non-kin** | 6993 | 84.0 % | **85.2 %** | **+1.2 pp** |

**All 11 kin classes regressed by 1-11 pp.** Non-kin specificity went
up by 1.2 pp — the model became *more conservative* about predicting
kin. Grandmother-grandson (`gmgs`) had the largest drop (-10.7 pp).

The directionality is consistent: hard negatives push the kin/non-kin
decision boundary *toward* non-kin (the model needs to be more certain
to call something kin because hard negatives look kinship-like).

### Notes (revised after Errata, 2026-05-14)

- ~~**Phase 6 (`relation_matched` hard negatives) is REJECTED**~~ —
  withdrawn. The sampler does not produce hard negatives; R004 was
  effectively R002 with a different seed for negative sampling.
  Phase 6 hypothesis is **untested**.
- ~~**Cross-experiment robustness check confirmed**~~ — withdrawn.
  M11 v4 and M12 R004 both used the same broken sampler. The
  "hard negs hurt Test" pattern attributed to them is unverified
  until the sampler is fixed.
- **R003 (SupCon aux) is still REJECTED** on the basis of R003's own
  evidence (cross-generation classes regressed, not just a magnitude
  question). That conclusion is independent of the sampler bug.
- The Val AUC peak of **0.9354** is the project-wide maximum
  validation score, but it doesn't transfer — likely a combination
  of the same overfit dynamics R002 already shows + sampler-seed
  variance in this particular draw.
- **Noise floor under partial-FT:** R002 → R004 differs only in the
  negative-sampler seed (200 → 270), and Test AUC swung by ~0.009
  (and per-class by 1-11 pp). Future improvements smaller than this
  cannot be cleanly attributed to the intervention unless the seed
  is controlled.

### Artifacts

- Checkpoint (epoch 4, Val AUC 0.9354): `output/004/checkpoints/best.pt` (536 MB) — patched manually with `model_config` (freeze=True, unfreeze_last_stage=True, supcon_weight=0)
- Resume snapshot: `output/004/checkpoints/epoch_5.pt` (will be pruned)
- Train log: `output/004/logs/train.log`
- Test log: `output/004/logs/test.log` (also `/tmp/m12_r004_test.log`)
- Results: `output/004/results/test_metrics_rocm.txt`

---

## Run 003 — 2026-05-14 — Stopped at epoch 7 (peak Val AUC 0.9306 ep 4; Test AUC 0.8510, SupCon aux did NOT help)

**Status:** Stopped manually at epoch 7 (3 consecutive declines below peak)
**Outcome:** Phase 4 of `proposta_rgck_net_kinship.md` §28 — same stack as R002 (Phase 2 partial unfreeze + classifier head) + **supervised-contrastive auxiliary loss with λ=0.05** over the L2-normalised contextualised global tokens `(gA, gB)`. Hypothesis: aux contrastive organises the embedding space and lifts Val/Test AUC. **Rejected by experimental evidence:** Test AUC = **0.8510**, -0.005 below R002 (0.8564, current project headline). All three TAR@FAR levels also regressed. Tiny improvements in `sibs` (+3.4 pp) and `bb` (+1.7 pp), but regressions in `gmgs`, `gfgs`, non-kin, and lower precision overall. Net negative.

### Launch command

```bash
SKIP_INSTALL=1 \
ALIGNED_ROOT=/home/bruno/Desktop/tcc_new/datasets/FIW_aligned \
BATCH_SIZE=8 \
GRAD_ACCUM=4 \
UNFREEZE_LAST_STAGE=1 \
LEARNING_RATE=1e-5 \
SUPCON_WEIGHT=0.05 \
NUM_WORKERS=4 \
SEED=42 \
bash models/12_rgck_net/AMD/run_pipeline.sh
```

### Changes from Run 002

| Parameter | R002 | **R003** |
|---|---|---|
| `SUPCON_WEIGHT` | 0.0 (off) | **0.05** |
| `SUPCON_MARGIN` | — | 0.3 (default) |
| All other knobs | — | identical |

The SupCon term is a margin-style label-aware contrastive on the cosine
similarity of the contextualised global tokens (after the cross-region
adapter, L2-normalised):

- For label=1 pairs: pull `cos(gA, gB)` toward 1 via `(1 - cos)²`
- For label=0 pairs: push `cos(gA, gB)` below `1 - margin = 0.7` via `max(0, cos - 0.7)²`

`L_total = L_bce(classifier_logit, label) + 0.05 × L_supcon(gA, gB, label, margin=0.3)`

### Configuration

Same as R002, plus:

| Param | Value |
|-------|-------|
| **Loss** | **BCE + 0.05 × SupCon** (margin 0.3) |
| Trainable params | 31,554,818 / 70,740,674 (44.61%) — identical to R002 |
| Time/epoch | ~26 min (similar to R002, supcon term is cheap) |

### Training trajectory

| Epoch | Train Loss | Val Acc | Val AUC | Thr | LR | R002 same ep | Δ R003-R002 |
|------:|-----------:|--------:|--------:|----:|-----|------:|------:|
| 1 | 0.6020 | 77.7 % | 0.8626 | 0.400 | 2.0e-6 | 0.8644 | -0.002 |
| 2 | 0.4164 | 84.2 % | 0.9246 | 0.400 | 4.0e-6 | 0.9259 | -0.001 |
| 3 | 0.3446 | 85.2 % | 0.9305 | 0.350 | 6.0e-6 | 0.9311 | -0.001 |
| **4** | **0.3021** | **85.7 %** | **0.9306** | 0.300 | 8.0e-6 | 0.9323 | -0.002 (peak) |
| 5 | 0.2714 | 85.6 % | 0.9285 | 0.150 | 1.0e-5 | 0.9306 | -0.002 |
| 6 | 0.2397 | 85.6 % | 0.9295 | 0.200 | 1.0e-5 | 0.9284 | +0.001 (R003 ahead — only positive Δ) |
| 7 | 0.2197 | 85.3 % | 0.9203 | 0.100 | 9.99e-6 | 0.9230 | -0.003 (decline confirmed) |

R003 trajectory tracks R002 with a small offset of ~-0.001 to -0.002
throughout. The single positive Δ (ep 6) was within noise. Train loss is
consistently ~0.013 higher than R002 (the supcon contribution).

### Test metrics (threshold 0.500)

| Metric | M12 R002 (project headline) | **M12 R003** | Δ |
|---|---:|---:|---:|
| **Test ROC AUC** | **0.8564** | **0.8510** | **-0.005** |
| Test Accuracy | 76.79 % | 76.37 % | -0.4 pp |
| Test Balanced Acc | 76.48 % | 76.09 % | -0.4 pp |
| Test F1 | 0.7402 | 0.7373 | -0.003 |
| Test Precision | 79.82 % | 78.88 % | -0.9 pp |
| Test Recall | 69.00 % | 69.22 % | +0.2 pp |
| Test Avg Precision | 0.8389 | 0.8305 | -0.008 |
| TAR @ FAR=0.001 | 4.18 % | 2.67 % | -1.5 pp |
| TAR @ FAR=0.01 | 17.58 % | 16.57 % | -1.0 pp |
| TAR @ FAR=0.1 | 57.11 % | 55.43 % | -1.7 pp |
| Val→test AUC gap | -0.076 | -0.080 | wider |

### Per-relation accuracy (FIW Track-I test, threshold 0.500)

| Relation | N | M12 R002 | **M12 R003** | Δ |
|----------|--:|---------:|-------------:|---:|
| **sibs** | 234 | 79.5 % | **82.9 %** | **+3.4** |
| **bb** | 860 | 75.4 % | **77.1 %** | **+1.7** |
| ss | 731 | 75.9 % | 75.9 % | = |
| gfgd | 138 | 52.2 % | 52.9 % | +0.7 |
| md | 1038 | 68.7 % | 68.6 % | -0.1 |
| fs | 1135 | 68.6 % | 68.4 % | -0.2 |
| ms | 1036 | 69.4 % | 69.1 % | -0.3 |
| fd | 918 | 68.4 % | 68.4 % | = |
| gfgs | 98 | 39.8 % | 38.8 % | -1.0 |
| non-kin | 6993 | 84.0 % | 83.0 % | -1.0 |
| gmgs | 121 | 44.6 % | 43.0 % | -1.6 |
| gmgd | 123 | 36.6 % | 36.6 % | = |

R003 improves on the **same-generation classes** (sibs, bb — both with
high N) but slightly regresses on the **rare grandparent classes**
(gmgs, gfgs) and non-kin. The supcon term pulls positive pairs together
in embedding space, which helps siblings (visually similar within a
batch) but hurts the more conceptually distant grandparent pairs.

### Notes

- **SupCon λ=0.05 hypothesis rejected.** The proposal §28 caveat — "if
  contrastive loss is too strong, can force artificial closeness" — was
  prescient. λ=0.05 was already at the conservative end suggested by the
  proposal and still produced net regression.
- The siblings improvement (sibs +3.4, bb +1.7) is consistent with the
  supcon term's behaviour: it pulls positive pairs together, which works
  for classes where positives are visually similar.
- Grandparent classes regressed because their positives are
  cross-generational (large age gap = visually different), so forcing
  them closer in embedding space distorts the learned representation.
- This is a **second negative result** for sophistication on top of the
  M12 R002 stack — earlier, M09 R002 (balanced sampling) and M11 v4
  (relation_matched negatives) also lowered Test AUC relative to their
  no-aux M09 R001 baseline. The pattern: any auxiliary that constrains
  the embedding space tends to over-fit val-pool patterns at the cost
  of test generalisation.
- **M12 R002 (Test AUC 0.8564) remains the project headline.**

### Artifacts

- Checkpoint (epoch 4, Val AUC 0.9306): `output/003/checkpoints/best.pt` (536 MB) — patched manually with `model_config` (supcon_weight=0.05, supcon_margin=0.3)
- Resume snapshot: `output/003/checkpoints/epoch_5.pt` (will be pruned)
- Train log: `output/003/logs/train.log`
- Test log: `output/003/logs/test.log` (also `/tmp/m12_r003_test.log`)
- Results: `output/003/results/test_metrics_rocm.txt`

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
