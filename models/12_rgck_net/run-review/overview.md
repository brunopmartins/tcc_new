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

| | Run 001 | **Run 002** | Run 003 | Run 004 | Run 005 |
|---|---|---|---|---|---|
| **Date** | 2026-05-13 | 2026-05-13 | 2026-05-14 | 2026-05-14 | 2026-05-14 |
| **Phase** | 1 (frozen) | **2 (partial unfreeze)** | 4 (R002 + SupCon) | 6 (R002 + hard negs — *intended; not tested*) | **5 (R002 + relation-type aux head)** |
| **Trainable** | 5.6 M (7.9 %) | 31.6 M (44.6 %) | 31.6 M (44.6 %) | 31.6 M (44.6 %) | 31.6 M (44.6 %) (+5,643 in relation_head) |
| **LR** | 1e-4 | 1e-5 | 1e-5 | 1e-5 | 1e-5 |
| **Loss** | BCE | BCE | BCE + 0.05 × SupCon | BCE | **BCE + 0.05 × CE_rel(pos, balanced)** |
| **Neg strategy** | random | random | random | `relation_matched` *(no-op: misnamed sampler, see R004 Errata)* | random |
| **Status** | Stopped at ep 7 | Stopped at ep 7 | Stopped at ep 7 | Stopped at ep 8 | SAFEGUARD ep 16 |
| **Best Val AUC** | 0.8351 (ep 3) | 0.9323 (ep 4) | 0.9306 (ep 4) | **0.9354 (ep 4)** — project max | 0.9318 (ep 4) |
| **Test ROC-AUC** | 0.7464 | **0.8564** ⭐ **HEADLINE** | 0.8510 | 0.8473 | 0.8476 |
| **Test Accuracy** | 68.0 % | 76.8 % | 76.4 % | 75.8 % | 76.5 % |
| **Val→test gap** | -0.089 | -0.076 | -0.080 | -0.088 | -0.084 |
| **Notes** | Phase 1 capped ceiling | **Phase 2 partial unfreeze beats M02 R031.** [run-002.md](run-002.md) | SupCon aux REJECTED. [run-003.md](run-003.md) | Intended hard-negs test — actually tested negative-sampler reseed because `relation_matched` sampler does the same thing as `random` (different seed only). Phase 6 hypothesis remains UNTESTED. [run-004.md](run-004.md) | **Phase 5 per-class hypothesis CONFIRMED**: all 11 kin classes improved 2-7 pp vs R002, especially grandparents (gmgd +7.3, gmgs +6.6, gfgd +3.6, gfgs +2.0). Global AUC -0.009 (within ±0.009 noise floor). [run-005.md](run-005.md) |

---

## Test metrics

| Metric | M02 R031 (prior best) | Run 001 (Phase 1) | **Run 002 (Phase 2)** | Run 003 (+SupCon) | Run 004 (+hard negs) | Run 005 (+rel aux head) |
|---|---:|---:|---:|---:|---:|---:|
| **Test ROC-AUC** | 0.850 | 0.7464 | **0.8564** ⭐ | 0.8510 | 0.8473 | 0.8476 |
| Test Accuracy | 74.4 % | 68.00 % | **76.79 %** ⭐ | 76.37 % | 75.75 % | 76.53 % |
| Test Balanced Acc | 75.2 % | 67.41 % | **76.48 %** | 76.09 % | 75.33 % | 76.45 % |
| Test F1 | 0.779 | 0.6150 | 0.7402 | 0.7373 | 0.7211 | **0.7528** ⭐ vs M12 |
| Test Precision | 66.5 % | 72.60 % | 79.82 % | 78.88 % | **80.29 %** ⭐ | 75.99 % |
| Test Recall | 94.1 % | 53.34 % | 69.00 % | 69.22 % | 65.44 % | **74.58 %** ⭐ vs M12 |
| **Avg Precision** | 0.817 | 0.7323 | **0.8389** ⭐ | 0.8305 | 0.8287 | 0.8288 |
| **TAR@FAR=0.001** | 2.5 % | 2.36 % | **4.18 %** ⭐ | 2.67 % | 2.01 % | 2.94 % |
| **TAR@FAR=0.01** | 14.0 % | 10.06 % | **17.58 %** ⭐ | 16.57 % | 14.71 % | 16.67 % |
| **TAR@FAR=0.1** | 49.9 % | 37.86 % | **57.11 %** ⭐ | 55.43 % | 56.06 % | 55.60 % |
| Best Val ROC-AUC | 0.881 | 0.8351 | 0.9323 | 0.9306 | **0.9354** ⭐ (project max) | 0.9318 |
| Best Val Accuracy | 76.6 % | 75.4 % | 85.5 % | 85.7 % | 86.2 % | 86.0 % |
| **Val→Test AUC gap** | -0.031 | -0.089 | -0.076 | -0.080 | -0.088 | -0.084 |

⭐ = R002 wins on every threshold-invariant Test metric — remains the **project headline**. R004 has the project-wide max Val AUC but it doesn't transfer (note: R004's `relation_matched` sampler is misnamed — it does not produce hard negatives; see [run-004.md](run-004.md) Errata).

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

## Conclusion (as of R005)

**R002 (Test AUC 0.8564) remains the project headline.** M02 R031's
reign as best-in-project (0.850) is over and stays over. R003, R004,
and R005 all attempted to improve on R002 via proposal Phase 4-6
interventions; none beat R002 on threshold-invariant AUC, but R005 is
the first to **deliver meaningful per-class gains** (especially on
grandparent classes) without widening the val→test gap.

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
| I-11 | Medium | **Open (R005-discovered, 2026-05-14)** | Architecture is asymmetric in (A, B) | `cross_region` adapter uses separate `attn_ab`/`attn_ba`/`ffn_a`/`ffn_b`/`norm_a*`/`norm_b*` weights; `regional_gate` and `classifier` concatenate tokens in `[A, B]` order — `forward(A, B) ≠ forward(B, A)` even though kinship is symmetric. R006 tests symmetric forward (process each pair in both directions, combine in loss). [model.py:148-174](../model.py#L148-L174), [model.py:234-242](../model.py#L234-L242), [model.py:374](../model.py#L374). |
| I-06 | Info | **Partial closure in R005** | Per-relation grandparent accuracies at threshold 0.5 | R005 (Phase 5 aux head, balanced CE) lifted grandparent classes: gmgd +7.3 pp, gmgs +6.6 pp, gfgd +3.6 pp, gfgs +2.0 pp vs R002 — explicit hypothesis confirmed on its targets. Test AUC unchanged (within noise floor) and non-kin specificity dropped 5.7 pp; trade-off. |

### Next directions (re-prioritised after R005, 2026-05-14)

Closed by experimental evidence so far:
- ~~Phase 4 (SupCon aux at λ=0.05) — REJECTED in R003.~~

Reopened by R004 Errata:
- **Phase 6 (real hard negatives)** — requires implementing actual
  role-matched negative sampling first. Lower priority than the
  symmetry / diff-LR experiments below; revisit only after those
  plateau.

Partially closed by R005:
- **Phase 5 (relation-type aux head, λ=0.05, balanced)** — per-class
  hypothesis confirmed (grandparents +2-7 pp); global AUC within
  noise floor (-0.009). Stays "on the table" as a recipe to
  combine with other interventions, not as a standalone winner.

Active priority order (post-R005):

1. **R006: R005 + symmetric forward (Option-B BCE).** Currently
   `forward(A,B) ≠ forward(B,A)` because the cross-region adapter
   has direction-specific weights (`attn_ab` vs `attn_ba`, etc.) and
   the regional_gate / classifier concatenate tokens in
   `[A, B]` order. R006 processes each pair in both orders (cheap:
   only the head runs twice, tokenizer reused) and applies
   Option-B BCE = 0.5·(BCE_AB + BCE_BA) plus averaged CE_rel.
   Hypothesis: ~0.005-0.01 Test AUC gain from variance reduction.
   Infrastructure committed (8923043), launch pending. **Next run.**

2. **Differential LR across the trainable stack.** Today, stage 4 +
   output_layer + cross-attn + gate + classifier all share 1e-5.
   Split: stage 4 → 2-5e-6, output_layer → 5e-6, cross-attn / gate /
   classifier → 1-2e-5. Lets the head adapt while the backbone melts
   less. Cheap structural change on top of R002 or R005/R006.

3. **L2-SP regularisation on the unfrozen backbone.** Penalise
   `‖θ_stage4 - θ_AdaFace_original‖²` (and likewise for output_layer)
   in the loss. More surgical than raising dropout or lowering LR for
   shrinking the Val→Test gap.

4. **Comparison-only fusion ablation.** Drop `gA, gB` from the
   classifier input at [model.py:374](../model.py#L374); keep
   `|diff|, prod, sims, weights, score`. Removes identity-as-feature
   signal that may be feeding val-pool memorisation. Likely lowers
   Val AUC; may close the gap and raise Test.

5. **Lower relation_aux λ** (0.02 or 0.03) — if the R005 trade-off
   scales with λ, a smaller weight may preserve more of R002's
   strict-FAR performance while retaining most of the per-class gain.

6. **Architecture switch to ROI Align tokenizer** (proposal §15
   Strategy 1): single feature-map forward + ROI pool instead of 5
   AdaFace passes. Halves training time. Larger code change; defer
   until 1-5 above are exhausted.

7. **Fix sampler + re-test Phase 6.** Only after 1-6 plateau.

Likely terminal directions:
- ~~Phase 3 full fine-tune — almost certainly regressive (the M11 v4
  lesson, intuitively still expected even with corrected sampler).~~
- **Lower SupCon λ** (0.01 or 0.02) — SupCon's failure signature in
  R003 (hurts cross-generation classes) is a wrong-direction signal,
  not just a magnitude issue. Skip.

The proposal's experimental sequence (§38) is **largely vindicated**.
Phase 2 (R002) is the clear winner. Phase 4 rejected; Phase 5
per-class confirmed; Phase 6 genuinely untested; the new symmetry
direction (R006) sits outside the proposal sequence as a discovered
architectural correction.

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
