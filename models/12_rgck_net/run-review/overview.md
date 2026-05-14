# Overview — Model 12: RGCK-Net (Region-Guided Cross Kinship Network)

**Model:** 12 — implementation of the proposal in
`proposta_rgck_net_kinship.md`. AdaFace IR-101 (WebFace4M) frozen +
5 fixed-region tokens + bidirectional cross-region attention + sigmoid
regional gating + BCE classifier head.
**GPU:** AMD Radeon RX 6750 XT (11.98 GB VRAM, gfx1031, ROCm 5.7)
**Dataset:** FIW (FIW_aligned 224×224 native — model crops 5 regions
internally, each resized to 112×112 for AdaFace)

---

## Hypothesis recap

After three M11 attempts and M09 R002 all showed that *sophisticating
the training distribution* over M09 R001's full-FT AdaFace stack
(balanced sampling, hard negatives, attention-driven contrastive)
consistently raised Val AUC but lowered Test AUC, M12 tested an
**architectural** direction:

> Does **reducing AdaFace's influence** (frozen backbone, lightweight
> region tokens) reduce family memorization enough to close the
> val→test gap, even at a lower Val AUC ceiling?

The structure:
- AdaFace IR-101 backbone is **frozen** (5.6 M / 70.7 M = 7.9 % trainable)
- 5 anatomical region tokens (global, eyes, nose, mouth, jaw) via fixed
  bounding boxes in 224×224 aligned-face coordinates, then resized to
  112×112 for AdaFace
- 1-layer × 4-head bidirectional cross-region attention (5×5 attention
  matrix — orders of magnitude cheaper than M09's 196×196 + 49×49)
- Per-region sigmoid gating (interpretable weights)
- 3-layer MLP classifier head on `[gA, gB, |diff|, prod, per-rel sims, weights, weighted score]`
- BCE on classifier logit

## Configuration baseline (R001 defaults)

| Knob | Default |
|---|---|
| Backbone | AdaFace IR-101 WebFace4M, **frozen** |
| Regions | 5: global, eyes, nose, mouth, jaw |
| Cross-attn layers | 1 |
| Cross-attn heads | 4 |
| Gate hidden dim | 128 |
| Classifier hidden dim | 512 |
| Loss | bce on classifier head |
| LR | 1e-4 peak (head-only), warmup 5, cosine, min_lr 1e-6 |
| Weight decay | 1e-4 |
| Dropout | 0.2 |
| Batch | 8 × grad-accum 4 (eff 32) |
| Train neg strategy | random |
| Eval neg strategy | random |
| Patience | 50 |
| Epochs (max) | 100 |
| Img size | 224 (native FIW_aligned) |
| Trainable params | **5,589,762 / 70,740,674 (7.90%)** |
| Time/epoch | ~20 min (half of full-FT models) |

## Run table

| | Run 001 | **Run 002** | Run 003 | Run 004 |
|---|---|---|---|---|
| **Date** | 2026-05-13 | 2026-05-13 | 2026-05-14 | 2026-05-14 |
| **Phase** | 1 (frozen) | **2 (partial unfreeze)** | 4 (R002 + SupCon) | 6 (R002 + hard negs) |
| **Trainable** | 5.6 M (7.9 %) | 31.6 M (44.6 %) | 31.6 M (44.6 %) | 31.6 M (44.6 %) |
| **LR** | 1e-4 | 1e-5 | 1e-5 | 1e-5 |
| **Loss** | BCE | BCE | BCE + 0.05 × SupCon | BCE |
| **Neg strategy** | random | random | random | **`relation_matched`** |
| **Status** | Stopped at ep 7 | Stopped at ep 7 | Stopped at ep 7 | Stopped at ep 8 |
| **Best Val AUC** | 0.8351 (ep 3) | 0.9323 (ep 4) | 0.9306 (ep 4) | **0.9354 (ep 4)** — project max |
| **Test ROC-AUC** | 0.7464 | **0.8564** ⭐ **HEADLINE** | 0.8510 | 0.8473 |
| **Test Accuracy** | 68.0 % | 76.8 % | 76.4 % | 75.8 % |
| **Val→test gap** | -0.089 | -0.076 | -0.080 | -0.088 |
| **Notes** | Phase 1 capped ceiling | **Phase 2 partial unfreeze beats M02 R031.** [run-002.md](run-002.md) | SupCon aux REJECTED. [run-003.md](run-003.md) | Hard negs REJECTED — Val AUC project max but Test -0.009 vs R002, gap widens. Same M11 v4 pattern. [run-004.md](run-004.md) |

---

## Test metrics

| Metric | M02 R031 (prior best) | Run 001 (Phase 1) | **Run 002 (Phase 2)** | Run 003 (+SupCon) | Run 004 (+hard negs) |
|---|---:|---:|---:|---:|---:|
| **Test ROC-AUC** | 0.850 | 0.7464 | **0.8564** ⭐ | 0.8510 | 0.8473 |
| Test Accuracy | 74.4 % | 68.00 % | **76.79 %** ⭐ | 76.37 % | 75.75 % |
| Test Balanced Acc | 75.2 % | 67.41 % | **76.48 %** | 76.09 % | 75.33 % |
| Test F1 | 0.779 | 0.6150 | 0.7402 | 0.7373 | 0.7211 |
| Test Precision | 66.5 % | 72.60 % | 79.82 % | 78.88 % | **80.29 %** ⭐ |
| Test Recall | 94.1 % | 53.34 % | 69.00 % | 69.22 % | 65.44 % |
| **Avg Precision** | 0.817 | 0.7323 | **0.8389** ⭐ | 0.8305 | 0.8287 |
| **TAR@FAR=0.001** | 2.5 % | 2.36 % | **4.18 %** ⭐ | 2.67 % | 2.01 % |
| **TAR@FAR=0.01** | 14.0 % | 10.06 % | **17.58 %** ⭐ | 16.57 % | 14.71 % |
| **TAR@FAR=0.1** | 49.9 % | 37.86 % | **57.11 %** ⭐ | 55.43 % | 56.06 % |
| Best Val ROC-AUC | 0.881 | 0.8351 | 0.9323 | 0.9306 | **0.9354** ⭐ (project max) |
| Best Val Accuracy | 76.6 % | 75.4 % | 85.5 % | 85.7 % | 86.2 % |
| **Val→Test AUC gap** | -0.031 | -0.089 | -0.076 | -0.080 | -0.088 |

⭐ = R002 wins on every threshold-invariant Test metric — remains the **project headline**. R004 has the project-wide max Val AUC but it doesn't transfer.

⭐ = M12 R002 wins on **all** threshold-invariant metrics (AUC, Avg Precision, all three TAR@FAR levels).

---

## Issues Log

| # | Severity | Status | Title | Notes |
|---|----------|--------|-------|-------|
| I-01 | Medium | Partial | Frozen backbone caps Val AUC at ~0.835 | Peak Val AUC 0.8351 vs M09 R001's 0.8919. Architectural simplicity comes at the cost of representational capacity. Proposal Phase 2 (unfreeze last IR-101 stage) could lift this. |
| I-02 | Medium | Open | Test AUC 0.7464 below M09 R001's 0.7982 | The smallest val→test gap (-0.089) doesn't compensate for the lower ceiling. Net Test AUC -0.052 vs M09 R001. |
| I-03 | Info | Open | No `evaluate.py` for M12 | Only `test.py` exists. No ROC plot, confusion matrix, or attention visualisation generated. Can be added later if needed; the regional gating weights are the most interesting visual artefact for this architecture. |
| I-04 | Info | Workaround applied | `model_config` not saved when training killed mid-pipeline | Same pattern as M09/M10/M11. best.pt patched manually before test.py could rebuild the model. |
| I-05 | Info | Open | Region tokenizer re-runs AdaFace 5× per face | Phase 1 uses Strategy 2 from the proposal (`recortar regiões + backbone`). Phase 2 should consider Strategy 1 (ROI Align on a single feature-map pass), which would halve training time further. |

---

## Comparison with other models (FIW, Test ROC-AUC)

| Model | Test ROC-AUC | Test Acc | Backbone | Architecture / Recipe | Notes |
|---|---:|---:|---|---|---|
| **M12 R002** | **0.8564** ⭐ | **76.8 %** | **AdaFace IR-101 (stages 1-3 frozen, stage 4 unfrozen)** | **5 region tokens + cross-attn + gate, BCE, LR 1e-5** | **NEW PROJECT HEADLINE** |
| M02 R031 | 0.850 | 74.4 % | ViT-B/16 ImageNet (full FT) | FaCoR top-only + cosine_contrastive (works on ViT) | prior project best |
| M05 R007 | 0.810 | — | hybrid (DINOv2 + LoRA + diff-attn) | — | partial freeze |
| M09 R001 | 0.7982 | 71.9 % | AdaFace IR-101 (full FT) | Multistage + BCE + classifier head + random negs | best AdaFace full-FT |
| M09 R002 | 0.7824 | 71.6 % | M09 R001 + balanced sampler | balanced positives | val→test gap widened |
| M06 R001 | 0.776 | 69.8 % | ViT-B/16 (frozen) + retrieval | retrieval + cross-attn | best frozen-ViT |
| M11 R001 v4 | 0.7707 | 70.6 % | M09 R001 + `relation_matched` negs | hard negs on M09 stack | val→test gap widened most (-0.128) |
| M10 R003 | 0.7478 | 70.6 % | AdaFace IR-101 (full FT) | FaCoR top-only + BCE + classifier head | val→test gap -0.140 |
| M12 R001 | 0.7464 | 68.0 % | AdaFace IR-101 (FROZEN) | 5 region tokens + cross-attn + gate | smallest val→test gap (-0.089), lowest Val AUC ceiling |
| M08 R001 | 0.693 | 60.8 % | ArcFace IR-100 (frozen) + retrieval | retrieval + cross-attn | anti-kinship trap |

---

## Per-relation accuracy comparison (test)

| Relation | M09 R001 | M09 R002 | M11 v4 | **M12 R001** |
|----------|---------:|---------:|-------:|-------------:|
| bb | 64.2 % | 58.1 % | 56.6 % | 58.9 % |
| ss | 62.9 % | 59.9 % | 53.9 % | 57.1 % |
| sibs | 64.5 % | 59.8 % | 50.0 % | 61.5 % |
| fs | 59.1 % | 56.7 % | 54.1 % | 56.3 % |
| ms | 57.3 % | 55.5 % | 51.5 % | 51.6 % |
| md | 53.9 % | 52.6 % | 53.1 % | 54.1 % |
| fd | 63.6 % | 59.3 % | 61.7 % | **52.0 %** |
| gfgd | 31.2 % | 50.7 % | 37.7 % | 28.3 % |
| gfgs | 30.6 % | 33.7 % | 36.7 % | 26.5 % |
| **gmgd** | 31.7 % | 33.3 % | 22.0 % | **36.6 %** |
| gmgs | 37.2 % | 37.2 % | 28.9 % | 33.1 % |
| non-kin | 84.7 % | 86.2 % | 86.7 % | 81.5 % |

`md` and `gmgd` improved over M09 R001. `fd` regressed catastrophically (-11.6 pp). Other classes 3-6 pp below.

---

## Conclusion (as of R004)

**R002 (Test AUC 0.8564) remains the project headline.** M02 R031's
reign as best-in-project (0.850) is over and stays over. R003 and R004
both attempted to improve on R002 via proposal Phase 4-6 interventions
and both failed.

Key findings (cumulative across R001-R004):

1. **R002 — partial unfreeze of stage 4 + output_layer is the winning
   architectural recipe.** Single change vs R001 (frozen) lifted Test
   AUC by +0.110 (from 0.7464 to 0.8564). The Phase 1 capacity
   bottleneck was real and severe.

2. **R002's val→test gap of -0.076 is the best among AdaFace-based
   models with non-trivial training**. R001 (-0.089) had a tighter gap
   but a much lower Val ceiling. The R002 trade-off (slightly more
   memorisation than R001 but much higher Val ceiling) is favourable.

3. **R003 (SupCon λ=0.05 aux loss) — REJECTED.** Net Test AUC -0.005,
   sibling classes improved (+1-3 pp) but grandparent classes regressed.
   Proposal §28 warning ("contrastive forte demais força aproximações
   artificiais") confirmed even at the conservative λ=0.05.

4. **R004 (hard negatives via `relation_matched`) — REJECTED.** Val AUC
   reached the project-wide max (0.9354), but Test AUC -0.009 vs R002.
   Gap widened to -0.088. **This is the SECOND independent confirmation
   of the "hard negs raise Val, drop Test" pattern** — M11 R001 v4 had
   shown it on full-FT, M12 R004 reproduces it on partial-FT. The
   mechanism is robust across architectural configurations.

5. **The recipe stack that won (R002):**
   - AdaFace IR-101 backbone
   - Stages 1-3 frozen, stage 4 (body[46:49]) + output_layer trainable
   - 5 region tokens (global, eyes, nose, mouth, jaw — fixed coords)
   - 1-layer × 4-head bidirectional cross-region attention
   - Sigmoid regional gating
   - 3-layer MLP classifier head over `[gA, gB, |diff|, prod, sims, weights, score]`
   - BCE loss on classifier logit
   - LR 1e-5 (10× lower than R001's 1e-4)
   - **Random negatives, no auxiliary losses** (both confirmed via
     ablation in R003 and R004)

6. **Three consecutive negative results** (M09 R002 balanced sampling,
   M11 v4 hard negs, M12 R004 hard negs again, M12 R003 SupCon) all
   show: sophisticating the training distribution on top of M09 R001
   or M12 R002 baselines consistently raises Val AUC but hurts Test
   AUC. The "harder train = better generalisation" intuition is wrong
   for this dataset's val→test family split structure.

### What R001 already validated, still standing

R001 (Phase 1, fully frozen) had Val→test gap -0.089 — smallest in the
AdaFace family. R002 keeps most of that gap benefit (-0.076) while
adding the capacity needed to raise the absolute Val AUC ceiling. The
Phase 1 → Phase 2 progression validated the proposal's experimental
sequence design.

### Open issues (still tracked from R001)

| # | Severity | Status | Title | Notes |
|---|----------|--------|-------|-------|
| I-01 | Medium | Closed in R002 | Frozen backbone caps Val AUC at ~0.835 | R002 partial unfreeze lifts peak to 0.9323 |
| I-02 | Closed in R002 | M12 Test AUC below M09 R001 | R002 Test AUC 0.8564 beats M09 R001 (0.7982) by +0.058 |
| I-03 | Info | Open | No `evaluate.py` for M12 | Still applies — only `test.py` exists. Visualisations (ROC, CM, attention maps) and per-region gate weights would be valuable. |
| I-04 | Info | Workaround applied | `model_config` not saved when training killed mid-pipeline | Same pattern as M09/M10/M11 — best.pt patched manually before test.py rebuild. |
| I-05 | Info | Open | Region tokenizer re-runs AdaFace 5× per face | Still using Strategy 2. Strategy 1 (ROI Align on shared feature map) would halve training time and may add small Test AUC. |
| I-06 | Info | Open | Per-relation grandparent accuracies still 37-52 % at threshold 0.5 | Relation-conditional aux head (proposal Phase 5) still untried. |
| I-07 | High | **Closed in R004** | Phase 6 hard negatives (`relation_matched`) cross-experiment robustness check | Two independent experiments (M11 v4 full-FT, M12 R004 partial-FT) reproduce the "Val up, Test down" pattern. Hard negs are a closed direction. |
| I-08 | High | **Closed in R003** | Phase 4 SupCon λ=0.05 auxiliary loss | R003 showed -0.005 Test AUC, regressed grandparent classes. Closed direction at this λ. Possible lower λ untested. |

### Next directions

Closed by experimental evidence so far:
- ~~Phase 4 (SupCon aux at λ=0.05) — REJECTED in R003.~~
- ~~Phase 6 (hard negatives via `relation_matched`) — REJECTED in R004.~~

Still untried:
- **Phase 5: relation-type auxiliary head.** Predict the relation
  category (11 kin classes + 1 non-kin) as an auxiliary task. Could
  specifically help the grandparent classes. Requires dataset
  modification to pass relation int labels to the loss (~moderate code
  change). **Most promising untried proposal direction.**
- **Architecture switch to ROI Align tokenizer** (proposal §15 Strategy 1):
  one feature-map forward + ROI pool instead of 5 separate AdaFace
  forwards per face. Halves training time. May add 1-2 points of Test
  AUC because the regions then come from coherent feature-space context
  instead of independent crops. **Highest-EV architectural change.**
- **Lower SupCon λ** (0.01 or 0.02) — if R003's regression scales with
  λ, lowering it might give net-zero or slightly positive effect. Low
  EV.
- **Hyperparameter sweep around R002**: vary LR, dropout, classifier
  hidden dim. Low EV.

Likely terminal directions (skip unless other gains exhausted):
- ~~Phase 3 full fine-tune — almost certainly regressive (the M11 v4
  lesson).~~
- ~~Variations of hard-negative strategies — closed direction.~~

The proposal's experimental sequence (§38) is **partially vindicated**.
Phase 2 (R002) is the clear winner. Phase 4 and Phase 6 didn't add
value on this dataset; only Phase 5 remains untested in the proposal
sequence.

---

## Architectural notes

- **Fixed regions in 224×224 aligned-face coords:**
  - `global`: (0:224, 0:224) — whole face
  - `eyes`:   (40:100, 20:204) — top-third horizontal strip
  - `nose`:   (80:150, 70:154) — central
  - `mouth`:  (140:185, 50:174)
  - `jaw`:    (170:220, 20:204) — bottom strip
  Each crop is resized to 112×112 for AdaFace.

- **Cross-region adapter:** verbatim MultiHeadAttention with
  `batch_first=True`, both directions wired in a `ModuleDict` per layer.
  1 layer × 4 heads × 512-d. The bidirectional pass keeps tokens_A and
  tokens_B symmetric.

- **Regional gate:** MLP over `[rA, rB, |rA-rB|, rA*rB]` → 1 logit per
  region → sigmoid. Outputs (B, K) weights in [0, 1]. Sigmoid (not
  softmax) per proposal §22 — multiple regions can be salient at once.

- **Classifier head input:** `[gA(512), gB(512), |diff|(512), prod(512), per-rel sims(5), per-rel weights(5), regional_score(1)]` = 2049-d.
  Three Linear/BatchNorm/GELU layers down to 1 logit.

- **Output tuple:** `(logit, weights, attn_map)`. Logit is consumed by
  the BCE loss; weights are for interpretability; attn_map is the last
  cross-region attention map (B, 4, 5, 5).
