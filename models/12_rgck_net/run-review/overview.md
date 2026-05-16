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

| | Run 001 | Run 002 | Run 003 | Run 004 | Run 005 | **Run 006** | Run 007 | Run 008 | Run 009 |
|---|---|---|---|---|---|---|---|---|---|
| **Date** | 2026-05-13 | 2026-05-13 | 2026-05-14 | 2026-05-14 | 2026-05-14 | 2026-05-14 | 2026-05-15 | 2026-05-16 | 2026-05-16 |
| **Phase** | 1 (frozen) | 2 (partial unfreeze) | 4 (R002 + SupCon) | 6 (R002 + hard negs — *intended; not tested*) | 5 (R002 + relation-type aux head) | **5 + symmetric forward (Option-B BCE)** | R006 + differential LR | R006 + L2-SP | **R006 + comparison-only fusion** |
| **Trainable** | 5.6 M (7.9 %) | 31.6 M (44.6 %) | 31.6 M (44.6 %) | 31.6 M (44.6 %) | 31.6 M (44.6 %) | 31.6 M (44.6 %) | 31.6 M (44.6 %) (3 param groups) | 31.6 M (44.6 %) | 31.0 M (44.2 %) — classifier smaller |
| **LR** | 1e-4 | 1e-5 | 1e-5 | 1e-5 | 1e-5 | 1e-5 | stage4 5e-6 / output 5e-6 / head 2e-5 | 1e-5 | 1e-5 |
| **Loss** | BCE | BCE | BCE + 0.05 × SupCon | BCE | BCE + 0.05 × CE_rel(pos, balanced) | 0.5·(BCE_AB + BCE_BA) + 0.05·avg(CE_rel_AB, CE_rel_BA)\|pos | same as R006 | R006 loss + **1e-3·L2SP(stage4+output_layer)** | same as R006 |
| **Neg strategy** | random | random | random | `relation_matched` *(no-op: misnamed sampler, see R004 Errata)* | random | random | random | random | random |
| **Status** | Stopped at ep 7 | Stopped at ep 7 | Stopped at ep 7 | Stopped at ep 8 | SAFEGUARD ep 16 | Manual stop ep 8 | SAFEGUARD ep 15 | SAFEGUARD ep 16 | SAFEGUARD ep 15 |
| **Best Val AUC** | 0.8351 (ep 3) | 0.9323 (ep 4) | 0.9306 (ep 4) | **0.9354 (ep 4)** — project max | 0.9318 (ep 4) | 0.9049 (ep 3) | 0.9093 (ep 4) | 0.9052 (ep 3) | 0.9012 (ep 3) — -0.004 vs R006 (predicted) |
| **Test ROC-AUC** | 0.7464 | 0.8564 | 0.8510 | 0.8473 | 0.8476 | **0.8788** ⭐ **HEADLINE** | 0.8730 | **0.8788** tied | 0.8739 |
| **Test Accuracy** | 68.0 % | 76.8 % | 76.4 % | 75.8 % | 76.5 % | **79.3 %** ⭐ | 78.8 % | 79.3 % | 78.3 % |
| **Val→test gap** | -0.089 | -0.076 | -0.080 | -0.088 | -0.084 | **-0.026** ⭐⭐ | -0.036 | **-0.026** tied | -0.027 (tied w/ R006) |
| **Notes** | Phase 1 capped ceiling | Phase 2 partial unfreeze beats M02 R031. [run-002.md](run-002.md) | SupCon aux REJECTED. [run-003.md](run-003.md) | Intended hard-negs test — actually tested negative-sampler reseed because `relation_matched` sampler does the same thing as `random` (different seed only). Phase 6 hypothesis remains UNTESTED. [run-004.md](run-004.md) | Phase 5 per-class hypothesis CONFIRMED: all 11 kin classes improved 2-7 pp vs R002. Global AUC -0.009 (within ±0.009 noise floor). [run-005.md](run-005.md) | **AUC HEADLINE.** R005 stack + symmetric forward closed the val→test gap from -0.084 to -0.026 and lifted Test AUC by +0.031 vs R005, +0.022 vs R002. Every kin class +13-43 pp vs R002; grandparents +31-43 pp. [run-006.md](run-006.md) | Diff-LR (head 4× backbone). **Hypothesis confirmed on Val (+0.004 peak), neutral on Test (-0.006 within noise).** [run-007.md](run-007.md) | L2-SP λ=1e-3. Penalty WAS applied but final solution **numerically tied with R006**. Per-rel ≤0.4 pp difference in 10 of 11 classes. [run-008.md](run-008.md) | **NEW STRICT-FAR CHAMPION.** Drop gA, gB from classifier. Aggregate AUC tied with R006 (within noise), **TAR@FAR=0.001 = 5.43 % +2.1 pp vs R006 — highest trained M12 model.** 9 of 11 kin classes gained 1-9 pp vs R006 (gfgs +9.2 standout). [run-009.md](run-009.md) |

---

## Test metrics

| Metric | M02 R031 | R001 | R002 | R003 | R004 | R005 | **R006** | R007 | R008 | R009 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Test ROC-AUC** | 0.850 | 0.7464 | 0.8564 | 0.8510 | 0.8473 | 0.8476 | **0.8788** ⭐ HEADLINE | 0.8730 | **0.8788** tied | 0.8739 |
| Test Accuracy | 74.4 % | 68.00 % | 76.79 % | 76.37 % | 75.75 % | 76.53 % | **79.33 %** ⭐ | 78.76 % | 79.28 % | 78.28 % |
| Test Balanced Acc | 75.2 % | 67.41 % | 76.48 % | 76.09 % | 75.33 % | 76.45 % | **79.65 %** ⭐ | 79.11 % | 79.61 % | 78.74 % |
| Test F1 | 0.779 | 0.6150 | 0.7402 | 0.7373 | 0.7211 | 0.7528 | **0.8017** ⭐ | 0.7980 | **0.8017** tied | 0.7984 |
| Test Precision | 66.5 % | 72.60 % | 79.82 % | 78.88 % | **80.29 %** ⭐ | 75.99 % | 74.17 % | 73.28 % | 74.05 % | 71.89 % |
| Test Recall | 94.1 % | 53.34 % | 69.00 % | 69.22 % | 65.44 % | 74.58 % | 87.24 % | 87.61 % | 87.39 % | **89.77 %** ⭐ |
| **Avg Precision** | 0.817 | 0.7323 | 0.8389 | 0.8305 | 0.8287 | 0.8288 | 0.8561 | 0.8521 | **0.8563** ⭐ | 0.8497 |
| **TAR@FAR=0.001** | 2.5 % | 2.36 % | 4.18 % | 2.67 % | 2.01 % | 2.94 % | 3.33 % | 4.49 % | 2.78 % | **5.43 %** ⭐⭐ |
| **TAR@FAR=0.01** | 14.0 % | 10.06 % | 17.58 % | 16.57 % | 14.71 % | 16.67 % | 18.61 % | **19.59 %** ⭐ | 19.12 % | 17.79 % |
| **TAR@FAR=0.1** | 49.9 % | 37.86 % | 57.11 % | 55.43 % | 56.06 % | 55.60 % | **59.93 %** ⭐ | 57.56 % | 59.65 % | 58.61 % |
| Best Val ROC-AUC | 0.881 | 0.8351 | 0.9323 | 0.9306 | **0.9354** ⭐ (project max) | 0.9318 | 0.9049 | 0.9093 | 0.9052 | 0.9012 |
| Best Val Accuracy | 76.6 % | 75.4 % | 85.5 % | 85.7 % | **86.2 %** | 86.0 % | 81.4 % | 81.7 % | 81.3 % | 80.1 % |
| **Val→Test AUC gap** | -0.031 | -0.089 | -0.076 | -0.080 | -0.088 | -0.084 | **-0.026** ⭐⭐ | -0.036 | **-0.026** tied | -0.027 |

⭐ = R006 wins on **every threshold-invariant Test metric** except TAR@FAR=0.001 — the **NEW PROJECT HEADLINE** (Test AUC 0.8788, +0.022 vs R002, +0.024 vs M02 R031). R004 has the project-wide max Val AUC but it doesn't transfer (note: R004's `relation_matched` sampler is misnamed — it does not produce hard negatives; see [run-004.md](run-004.md) Errata). R006's Val AUC is the lowest of the partial-FT runs *by design* — symmetric forward removes direction-specific shortcuts that inflate Val without transferring to Test.

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

## Conclusion (as of R009)

**R006 (Test AUC 0.8788) is the new project headline**, displacing
R002 (0.8564) and M02 R031 (0.850). R006 wins every threshold-
invariant Test metric except TAR@FAR=0.001. Val→test gap collapsed to
**-0.026** — smallest of any AdaFace-based model with non-trivial
training in the project. The decisive intervention is **symmetric
forward** (Option-B BCE on top of Phase 5): each pair is processed
in both (A,B) and (B,A) orders and the loss penalises each direction.
No new parameters; ~+2 % wall-clock per epoch.

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

4. **R004 (intended Phase 6 / hard negatives via `relation_matched`) —
   hypothesis NOT actually tested.** Post-run audit found the
   `_sample_fiw_rfiw_relation_matched_negatives` function does not
   preserve relation/role; it samples two random images from two
   different families and labels `"non-kin"`, identical to the
   `random` sampler except for the seed offset (270 vs 200). Measured
   overlap: 0.09 % on training negatives, 100 % on test (the test
   pairs come from RFIW Track-I lists, not the sampler). What R004
   actually measured was negative-sampler reseed variance with all
   else identical to R002: Val AUC +0.003 (project max 0.9354),
   Test AUC -0.009, gap widened by 0.012. **This sets a noise floor:
   any future improvement smaller than ~0.01 Test AUC is
   indistinguishable from sampler-seed variance unless we control
   the seed.** The earlier "M11 v4 confirms hard negs hurt" claim
   rests on the same broken sampler and is also withdrawn.

5. **R005 (Phase 5 relation-type CE aux head, λ=0.05, class-balanced)
   — per-class hypothesis CONFIRMED, AUC hypothesis NOT.** All 11 kin
   classes improved 2-7 pp vs R002. The 4 grandparent classes (the
   explicit Phase 5 targets) gained 2.0-7.3 pp: gmgd +7.3, gmgs +6.6,
   gfgd +3.6, gfgs +2.0. Test ROC AUC 0.8476 — -0.009 vs R002, **within
   the ±0.009 noise floor**. Trade-off: non-kin specificity -5.7 pp at
   the lower selected threshold (0.300 vs R002's ~0.500), TAR@FAR
   levels all down 0.9-1.5 pp. **First post-R002 intervention to not
   lose meaningfully on AUC and to improve every kin class
   simultaneously.** R005's per-class profile is the *opposite* of
   R003's (R003 hurt grandparents, R005 helps them) — different aux
   objectives produce opposite class signatures.

8. **R006 (R005 stack + symmetric forward, Option-B BCE) — NEW PROJECT
   HEADLINE.** Test ROC AUC **0.8788**, +0.022 vs R002 (well outside
   the ±0.009 noise floor). Val→test gap **-0.026** — smallest in the
   AdaFace family. Wins every threshold-invariant Test metric vs R002
   except TAR@FAR=0.001. **Every kin class improved 13-43 pp vs R002**;
   grandparents jumped 31-43 pp (gmgd 36.6→79.7, gmgs 44.6→80.2,
   gfgs 39.8→76.5, gfgd 52.2→83.3). Val AUC ceiling fell to 0.9049
   (-0.027 vs R005), exactly as the hypothesis predicted: forcing
   `f(A,B) ≈ f(B,A)` removes direction-specific shortcuts that
   inflated Val but didn't transfer to Test. **Symmetric forward
   contribution alone (R006 vs R005, identical otherwise)**: +0.031
   Test AUC, 0.058 reduction in val→test gap, +31.6 pp avg grandparent
   accuracy. ~+2 % wall-clock overhead per epoch; zero new parameters.
   **Phase 5 + symmetric forward are multiplicative**, not saturating:
   e.g. gmgd went 36.6 → 43.9 (Phase 5: +7.3) → 79.7 (symmetry on top:
   +35.8).

5. **The recipe stack that won (R002):**
   - AdaFace IR-101 backbone
   - Stages 1-3 frozen, stage 4 (body[46:49]) + output_layer trainable
   - 5 region tokens (global, eyes, nose, mouth, jaw — fixed coords)
   - 1-layer × 4-head bidirectional cross-region attention
   - Sigmoid regional gating
   - 3-layer MLP classifier head over `[gA, gB, |diff|, prod, sims, weights, score]`
   - BCE loss on classifier logit
   - LR 1e-5 (10× lower than R001's 1e-4)
   - Random negatives (default sampler), no auxiliary losses. R003
     directly tested SupCon λ=0.05 and rejected it. R004 *intended*
     to test hard negs but actually only tested another draw of the
     same random-negatives distribution (see point 4); the hard-negs
     hypothesis remains untested.

6. **"Sophisticated training distribution hurts" — partially supported,
   partially untested.** Confirmed for: M09 R002 balanced sampling,
   M12 R003 SupCon. *Not* confirmed for hard negatives: both M11 v4
   and M12 R004 used the broken `relation_matched` sampler that does
   not actually generate hard negatives. The "harder train hurts"
   intuition is correct for the auxiliary-loss and class-balancing
   variants tested, but the hard-negative variant remains open until
   the sampler is fixed.

7. **Phase 5 relation-type aux head is a viable alternative recipe.**
   R005 shows a *qualitatively different* model from R002: stronger
   kin recall and per-class balance, weaker non-kin specificity. The
   AUC trade-off is within noise floor. For applications that care
   about per-class fairness across kin types R005 is preferable; for
   strict-FAR deployment R002 wins. The proposal §38 narrative for
   Phase 5 ("relation-type aux helps the rare classes") is now
   empirically supported on FIW.

9. **The decisive insight (R006): the M12 architecture was learning
   direction-specific shortcuts.** The `cross_region` adapter has
   separate `attn_ab` / `attn_ba` / `ffn_a` / `ffn_b` / `norm_*`
   weights; the gate and classifier concatenate in `[A, B]` order.
   RFIW Track-I lists each pair in a single canonical order, so the
   model could learn to extract kinship signal *given a specific
   role ordering*. This shortcut didn't transfer across families
   because role-by-visual-cue ordering (age, pose) differs between
   train and test families. **R006's symmetric forward forced
   `f(A,B) ≈ f(B,A)`, removing the shortcut.** The result is the
   biggest single jump in Test AUC since R002's partial unfreeze
   (+0.022) and a 0.050 gap closure simultaneously. This is the
   most important finding of the post-R002 R&D cycle.

10. **R007 (R006 stack + differential LR: stage4 5e-6, output_layer
    5e-6, head 2e-5). Hypothesis confirmed on Val (+0.004 peak),
    neutral on Test (-0.006 within noise floor).** The 4×-higher
    head LR lets the head reach a higher Val ceiling, but the gain
    doesn't transfer to Test. Val→Test gap widened slightly to
    -0.036 (still 2nd best in AdaFace family). Diff-LR shifts the
    operating curve slightly toward strict-FAR: TAR@FAR=0.001
    +1.2 pp, FAR=0.01 +1.0 pp, FAR=0.1 -2.4 pp vs R006. The
    intervention is **structurally different** from R006's: R006
    was a *generalisation* regulariser (forced order-invariance);
    R007 is a *fit-quality* reallocation (lets head fit better). On
    a dataset where R006 already closed most of the train→test gap,
    R007's lever doesn't have anything to push against. **R006 stays
    headline.** R007 may be preferable for low-FAR deployment
    scenarios.

11. **R008 (R006 stack + L2-SP λ=1e-3 on stage 4 + output_layer,
    25.9 M params anchored to AdaFace pretrain). Numerically tied
    with R006 on every threshold-invariant Test metric.** Test ROC
    AUC 0.8788 (identical to R006), val→test gap -0.026 (identical),
    Test F1 0.8017 (identical). Per-relation accuracies within
    ±0.4 pp of R006 in 10 of 11 classes; only gmgd shifted by
    +1.6 pp. The penalty WAS applied (train loss +0.084 vs R006 at
    ep 7, smoother post-peak decline) but the final solution lands
    at the same Test discrimination. Two non-exclusive interpretations:
    λ=1e-3 may be too small to change the final solution, or R006's
    symmetric forward already exhausted the gap-closure available
    from generalisation regularisers (the residual -0.026 gap is
    near the irreducible noise for this dataset/architecture).
    Either way, **R006 stays headline.**

12. **Post-R006 R&D taxonomy** has crystallised:

    | Run | Intervention type | Effect on Test AUC | Effect on TAR@FAR=0.001 |
    |---|---|---:|---:|
    | R005 | per-class loss (CE_rel aux) | -0.009 within noise | -1.2 pp |
    | **R006** | **structural symmetry (shortcut removal)** | **+0.022 (decisive)** | -0.85 pp |
    | R007 | capacity reallocation (diff-LR) | -0.006 within noise | +1.2 pp |
    | R008 | parameter anchoring (L2-SP) | +0.000 tied | -0.55 pp |
    | **R009** | **identity leakage removal** | -0.005 within noise | **+2.1 pp ⭐** |

    Two distinct R&D phases have closed:
    - **Aggregate-AUC phase**: R002 (partial unfreeze) + R006
      (symmetric forward) = the two decisive moves. Other interventions
      are neutral. **R006 stays AUC headline at 0.8788.**
    - **Strict-FAR phase**: R009 (comparison-only fusion) is the
      decisive move. TAR@FAR=0.001 = 5.43 % is the highest of any
      trained M12 model and approaches the B0 frozen baseline's
      7.06 % (which has zero training and benefits from raw AdaFace
      identity discrimination).

13. **R009 (comparison-only fusion). Aggregate Test AUC -0.005 vs R006
    (within noise); strict-FAR +2.1 pp.** Dropping `gA, gB` from the
    classifier input forces the head to rely only on comparison
    features. The intervention behaves as predicted on Val (-0.004,
    less raw signal), is approximately neutral on aggregate Test AUC,
    but **massively shifts the operating curve toward strict-FAR**:
    TAR@FAR=0.001 jumped from 3.33 % to **5.43 %**, the largest
    single-step strict-FAR gain in the M12 R&D cycle. **9 of 11 kin
    classes improved 1-9 pp vs R006**, especially gfgs (+9.2 pp) —
    the historic worst grandparent class is now at 85.7 %. Non-kin
    specificity dropped 4.4 pp (more recall-oriented operating point).

    R006 stays AUC headline; **R009 is the new strict-FAR champion**.
    For deployment in low-FAR scenarios (forensic search, large-pool
    verification), R009 is the preferred M12 model.

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
| I-07 | High | **Reopened (Errata 2026-05-14)** | Phase 6 hard negatives via `relation_matched` | Originally closed citing M11 v4 + M12 R004. Both rely on `_sample_fiw_rfiw_relation_matched_negatives` which does NOT preserve relation/role — it samples uniformly across families like `random` does, only with a different seed. Phase 6 hypothesis is therefore untested. Requires sampler fix before re-running. |
| I-08 | High | **Closed in R003** | Phase 4 SupCon λ=0.05 auxiliary loss | R003 showed -0.005 Test AUC, regressed grandparent classes. Closed direction at this λ. Possible lower λ untested. |
| I-09 | High | **Open (Errata 2026-05-14)** | `relation_matched` sampler is misnamed | [models/shared/dataset.py:433](../../shared/dataset.py#L433): line 464 picks a relation but line 465 hardcodes `"non-kin"` — the chosen relation is discarded. Algorithm is identical to [models/shared/dataset.py:512](../../shared/dataset.py#L512) up to seed offset. Either rename the function to clarify it's a reseed, or implement actual role-matched sampling so Phase 6 can be tested. |
| I-10 | Info | **Open (Errata 2026-05-14)** | Noise floor under partial-FT ≈ ±0.01 Test AUC from negative-sampler reseed | R002 vs R004 differ only in negative-sampler seed (200 vs 270) and produced Test AUC 0.8564 vs 0.8473 — a ~0.009 swing. Any future "improvement" of smaller magnitude is indistinguishable from sampler variance. Control the seed for definitive comparisons. |
| I-11 | High | **CLOSED in R006** | Architecture is asymmetric in (A, B) | R006 tested symmetric forward (Option-B BCE: 0.5·(BCE_AB + BCE_BA)). Result: Test AUC +0.031 vs R005, gap closed by 0.058, +31.6 pp avg grandparent accuracy. Symmetric forward should be the default for M12 going forward. [model.py:148-174](../model.py#L148-L174), [model.py:234-242](../model.py#L234-L242), [model.py:374](../model.py#L374). |
| I-06 | Info | **Closed in R006** | Per-relation grandparent accuracies | R005 (Phase 5) lifted grandparents 2-7 pp (target confirmed). R006 (Phase 5 + symmetric forward) lifted grandparents another 28-36 pp — gmgd 36.6→79.7, gmgs 44.6→80.2, gfgs 39.8→76.5, gfgd 52.2→83.3. The combination is multiplicative. Per-rel weaknesses essentially resolved at thr=0.500. |

### Next directions (re-prioritised after R009, 2026-05-16)

Closed by experimental evidence so far:
- ~~Phase 4 (SupCon aux at λ=0.05) — REJECTED in R003.~~
- ~~Architecture asymmetry — CLOSED in R006 (now default).~~
- ~~Differential LR — neutral on Test in R007.~~
- ~~L2-SP λ=1e-3 — neutral on Test in R008.~~
- ~~Comparison-only fusion — neutral on Test AUC in R009 but
  decisive on strict-FAR (+2.1 pp).~~

The shortcut-removal intervention family has been exhausted on M12.
Both interventions in that family (R006 symmetric forward,
R009 comparison-only fusion) have been tested. R006 is the only one
that moved aggregate AUC; R009 is the only one that moved strict-FAR
meaningfully.

Reopened by R004 Errata:
- **Phase 6 (real hard negatives)** — requires implementing actual
  role-matched negative sampling. Untested under correct sampler.

Confirmed compounding interventions:
- **Phase 5 (relation-type aux head)** + **symmetric forward** —
  cumulative +0.022 Test AUC vs R002.

Two complementary M12 headlines now stand:
- **AUC headline: R006** (Test AUC 0.8788, val→test gap -0.026).
- **Strict-FAR headline: R009** (TAR@FAR=0.001 = 5.43 %, val→test
  gap -0.027).

Active priority order (post-R009):

1. **R010 — Stack R007 + R009** (differential LR + comparison-only
   fusion). Both R007 and R009 moved strict-FAR in the same
   direction (+1.2 pp and +2.1 pp respectively over R006).
   Combining them might compound (strict-FAR could reach 6-7 %)
   while staying near R006's aggregate AUC. Highest-EV next M12
   run if continuing the cycle.

2. **R011 candidate — L2-SP at higher λ (1e-2)** on top of R006.
   Distinguishes the two R008 interpretations (λ too small vs gap
   already closed). Lower EV given R008's null result.

3. **R012 candidate — Lower relation_aux λ** (0.02 or 0.03) on
   top of R006 — if reducing the aux weight recovers some
   precision without losing per-class balance.

4. **Architecture switch to ROI Align tokenizer** (proposal §15
   Strategy 1). Larger code change; defer until 1-3 above are
   exhausted or the project shifts to a different focus.

5. **Fix sampler + re-test Phase 6.** Open since R004 Errata.
   Lower priority.

**Strong consideration: shift focus from M12 R&D to the TCC
narrative.** The post-R002 cycle has now established a clear
mechanistic picture:
- *Aggregate AUC*: R006 captures the main signal available from
  generalisation regularisers on this architecture.
- *Strict-FAR*: R009 captures the main signal available from
  identity-leakage removal.
- *Per-class balance*: R005 + R006 produce the strongest balance,
  especially on grandparent classes.
- *Operating-point flexibility*: the project now has three
  qualitatively different operating points (R002 high-precision,
  R006 balanced, R009 high-recall/strict-FAR), each preferable in
  different deployment scenarios.

This is a strong story for a TCC. Further single-knob runs are
likely to produce diminishing returns; the next experimental moves
(R010 stack-R007+R009, R011 higher-λ L2-SP) have low expected EV
relative to the cost of consuming more compute and complicating
the narrative.

Likely terminal directions:
- ~~Phase 3 full fine-tune — almost certainly regressive.~~
- ~~Lower SupCon λ~~ (R003 wrong-direction signal).
- ~~Higher head LR / wider diff-LR ratios~~ (R007 neutral on Test).
- ~~Variants of L2-SP at λ < 1e-3~~ (smaller λ less likely to help than R008).

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
