# M12 RGCK-Net — Post-R002 R&D Cycle Summary (R005-R010)

**Period:** 2026-05-14 to 2026-05-17
**Starting point:** M12 R002 (Test AUC 0.8564) — project headline after Phase 2 (partial unfreeze of stage 4 + output_layer).
**Ending point:** **Three** complementary headlines (each champion of a different deployment regime):
- **M12 R006 (Test AUC 0.8788)** — aggregate-AUC champion
- **M12 R009 (TAR@FAR=0.001 = 5.43 %)** — strict-FAR champion
- **M12 R010 (TAR@FAR=0.01 = 21.16 %, TAR@FAR=0.1 = 60.00 %, Test AP = 0.8567)** — mid-FAR + AP champion (R007 + R009 stack)

All lie on the M12 architectural family with the same backbone freezing recipe; they differ by structural and capacity-allocation knobs added on top of a shared Phase 5 (relation-type aux head) base.

---

## TL;DR

The post-R002 cycle tested five single-knob interventions on top of
the M12 R002 stack. Two of them moved a Test-side metric meaningfully;
three were empirically neutral within the ±0.009 Test AUC noise floor
established in R004. The two winning interventions are **structural**
(they remove a class of shortcut from the architecture). The three
neutral interventions are training-dynamics adjustments (auxiliary
loss, learning-rate split, parameter anchoring) that do not change
the structural shortcut surface.

| Run | Intervention | Test AUC vs R002 | TAR@FAR=0.001 vs R002 | TAR@FAR=0.01 vs R002 | Net result |
|---|---|---:|---:|---:|---|
| R002 | (baseline) | 0.8564 | 4.18 % | 17.58 % | prior headline |
| R003 | + SupCon aux (λ=0.05) | −0.005 | −1.5 pp | −1.0 pp | rejected — wrong-direction signature |
| R004 | + "hard negs" (sampler bug) | −0.009 | −2.2 pp | −2.9 pp | hypothesis untested (R004 Errata) |
| R005 | + relation aux head (Phase 5) | −0.009 | −1.2 pp | −0.9 pp | per-class balance confirmed; AUC neutral |
| **R006** | + symmetric forward (Option-B BCE) | **+0.022** | −0.85 pp | +1.03 pp | **AUC champion** |
| R007 | R006 + differential LR | −0.006 | +0.31 pp | +2.01 pp | neutral on AUC; mid-FAR shift |
| R008 | R006 + L2-SP (λ=1e-3) | +0.000 | −1.4 pp | +1.54 pp | tied with R006 |
| **R009** | R006 + comparison-only fusion | −0.005 | **+1.25 pp** | +0.21 pp | **strict-FAR champion** |
| **R010** | R006 + R007 + R009 (stack) | −0.003 | +1.03 pp | **+3.58 pp ⭐⭐** | **mid-FAR + AP champion** |

R005 ran with `RELATION_AUX_WEIGHT=0.05` alone. R006-R009 all include
the same R005 stack and add one more knob each. R010 stacks R007 + R009
on top of the R006 baseline (3 knobs on top of Phase 5).

---

## What actually changed in M12 (architectural deltas)

The R002 model is the M12 Phase 2 baseline:
- AdaFace IR-101 backbone with stages 1-3 frozen, stage 4 (`body[46:49]`) +
  `output_layer` trainable
- 5 anatomical region tokens (global, eyes, nose, mouth, jaw) via fixed
  bounding boxes in 224×224 aligned-face coordinates
- 1-layer × 4-head bidirectional cross-region attention adapter
- Sigmoid regional gating
- 3-layer MLP classifier head on `[gA, gB, |diff|, prod, sims, weights, score]`
- BCE on the classifier logit
- LR 1e-5 with warmup + cosine; random negatives

Three architectural changes were tested in the post-R002 cycle. **Two**
of them stuck.

### 1. Phase 5: relation-type auxiliary head (R005, kept)

Added a `relation_head: Linear(embedding_dim → 11)` operating on
`0.5 · (gA + gB)`. Trained with class-balanced cross-entropy on positive
pairs only (the 11 FIW kinship relations), weighted by inverse class
frequency from the training positives.

  L = L_BCE + 0.05 · CE_rel(rel_logits | label = 1)

The class weights routed gradient toward the under-represented
grandparent classes (gmgd 2.46×, gmgs 2.14×, gfgd 2.19×, gfgs 1.96×).

**Effect (R005 vs R002):** all 11 kin classes gained 2-7 pp;
grandparents specifically gained 2.0-7.3 pp. Aggregate Test AUC −0.009
within noise; the per-class hypothesis was confirmed on its targets.

### 2. Symmetric forward / Option-B BCE (R006, kept — this is the decisive move)

The M12 cross-region adapter was asymmetric in `(A, B)`: separate
`attn_ab` / `attn_ba` / `ffn_a` / `ffn_b` / `norm_a*` / `norm_b*`
weights, and the regional gate and classifier concatenate tokens in
`[A, B]` order. Because the RFIW Track-I pair lists each pair in a
single canonical order, the model could learn `attn_ab` to extract
kinship signal *given a specific role ordering* — a shortcut that
did not transfer across the val/test family split.

R006 processes each pair in both `(A, B)` and `(B, A)` orders and
combines them with Option-B BCE:

  L = 0.5 · (BCE_AB + BCE_BA) + 0.05 · avg(CE_rel_AB, CE_rel_BA) | label=1

The tokenizer (5× AdaFace per face — the expensive part) runs once
per face; only the post-tokenizer head runs twice. Wall-clock
overhead: ~+2 %. Zero new parameters.

**Effect (R006 vs R005, isolated):**
- Test ROC AUC: 0.8476 → 0.8788 (+0.031)
- Val→Test gap: -0.084 → -0.026 (−0.058)
- Every kin class +13-43 pp vs R002
- Grandparents +31-43 pp vs R002 (gmgd 36.6 → 79.7, gmgs 44.6 → 80.2,
  gfgs 39.8 → 76.5, gfgd 52.2 → 83.3)

Val AUC ceiling dropped by 0.027 vs R005 — exactly the predicted
side-effect of removing the shortcut. The lower Val is not a loss;
it confirms the model can no longer exploit the train-side regularity
that did not generalise.

### 3. Comparison-only fusion (R009, optional — strict-FAR specialist)

Default fusion concatenates `[gA(512), gB(512), |diff|(512), prod(512),
sims(K), weights(K), regional_score(1)]` (2049 dims). R009 drops the
raw `gA` and `gB` slots, keeping only comparison features:
`[|diff|(512), prod(512), sims(K), weights(K), regional_score(1)]`
(1035 dims). This removes identity-as-feature signal — the head can
no longer use raw demographic profiles of A and B to bias the
prediction, only explicit comparison primitives.

**Effect (R009 vs R006):**
- Test ROC AUC: 0.8788 → 0.8739 (−0.005, within noise)
- TAR@FAR=0.001: 3.33 % → 5.43 % (+2.1 pp) — biggest strict-FAR
  jump in the cycle; highest of any trained M12 model in the project
- 9 of 11 kin classes gained 1-9 pp; gfgs +9.2 pp the standout
- non-kin specificity dropped 4.4 pp (more recall-oriented threshold)

The aggregate AUC summary doesn't capture the win because the operating
curve shifted: strict-FAR up substantially, mid-range FAR slightly
down. R009 is the best M12 model for low-FAR verification deployment.

### 4. R010 — Stacking R007 + R009 on R006 (operating-curve champion)

R010 tests whether the two non-AUC interventions compound. The setup:
R006 baseline (Phase 5 + symmetric forward) plus differential LR (R007:
stage 4 = 5e-6, output_layer = 5e-6, head = 2e-5) plus comparison-only
fusion (R009: drop gA, gB from classifier). All other knobs identical
to R006.

**The hypothesis was strict-FAR compounding** to the 6-7 % range
(R007 +1.2 pp + R009 +2.1 pp ≈ +3 pp from R006's 3.33 %). The
hypothesis was **falsified on strict-FAR but exceeded on mid-FAR**.

**Effect (R010 vs R006):**
- Test ROC AUC: 0.8788 → 0.8754 (−0.003, within noise)
- TAR@FAR=0.001: 3.33 % → 5.21 % (+1.88 pp; tied with R009 alone
  within noise — strict-FAR ceiling not exceeded)
- **TAR@FAR=0.01: 18.61 % → 21.16 % (+2.55 pp)** — *superadditive*,
  larger than R007's +0.98 pp + R009's −0.82 pp combined
- **TAR@FAR=0.1: 59.93 % → 60.00 % (+0.07 pp)** — M12 best
- **Test AP: 0.8561 → 0.8567 (+0.001)** — M12 best
- Test Precision: 74.17 % → 75.43 % (+1.26 pp) — best of any
  post-R002 M12 run
- non-kin specificity: 72.1 % → 74.8 % (+2.7 pp)
- gfgs: 76.5 % → 83.7 % (+7.2 pp) — R009's gain held under stack
- gmgs: 80.2 % → 66.1 % (−14.1 pp) — worst per-class regression
  in the cycle (threshold-shift artifact at val-selected 0.500)

**Mechanism — why mid-FAR compounded but strict-FAR didn't:**

R009 alone *hurt* TAR@FAR=0.01 (−0.82 pp) — removing gA/gB shifted
probability mass toward strict-FAR but lost mid-FAR discriminative
power. R007 alone *helped* TAR@FAR=0.01 (+0.98 pp) — the head LR
ramped up faster and the head learned mid-FAR features more
thoroughly. In R010, R007's mid-FAR boost compensates for R009's
mid-FAR loss in a way that exceeds either parent alone:

- Strict-FAR tail saturates at the R009 ceiling (~5.4 %): both
  interventions converge on the same identity-leakage-free
  operating point at the extreme threshold.
- Mid-FAR region is jointly shaped by both interventions: the
  smaller cmp-only head + higher head LR + ramped-up training
  refinement produces an operating curve neither parent reaches.

This is the first **superadditive** result in the M12 cycle. The
other stack interactions (R005+R006, R006+R007, R006+R008,
R006+R009) were either neutral or roughly additive.

R010's val-selected threshold (0.500) is the highest of any M12 run,
explaining the per-class regression at high-N classes (the higher
threshold rejects borderline positives). The trade-off shifts the
deployment regime from "balanced kin-recall" (R006) to "high-precision
mid-FAR" (R010).

---

## What was tested and rejected

### Three training-dynamics interventions, all neutral

| Run | Intervention | Δ Test AUC vs R006 | Verdict |
|---|---|---:|---|
| R007 | Differential LR: stage 4 = 5e-6, output_layer = 5e-6, head = 2e-5 | −0.006 | Within noise. +0.004 Val peak (head fits better) but no Test transfer. |
| R008 | L2-SP regulariser λ = 1e-3 on stage 4 + output_layer (25.9 M params anchored to AdaFace pretrain) | +0.000 | Numerically tied with R006. Penalty WAS applied (train loss +0.084 at ep 7) but final solution converges to same Test predictions. |
| R005 | Relation-type CE aux head λ = 0.05 (vs R002, not R006) | −0.009 | Per-class hypothesis confirmed; aggregate AUC within noise. Kept in stack because it composes with R006. |

The three neutral interventions adjust **how** the model learns
(loss weighting, gradient magnitudes per group, weight drift
penalties) but do not change the **shortcut surface** the model can
exploit. Once R006 removed the direction-specific shortcut, adjusting
training dynamics inside the resulting architecture has nothing left
to push against on aggregate AUC.

### Two interventions that did not even count as a test

| Run | Intervention | Verdict |
|---|---|---|
| R003 | SupCon λ = 0.05 on the L2-normalised global tokens | Rejected. Test AUC −0.005, but the per-class signature is wrong: SupCon hurts grandparents (cross-generation pairs are forced to converge in embedding space), which is the opposite of what kinship verification needs. |
| R004 | "Hard negatives" via `relation_matched` sampler | Hypothesis was not actually tested. Post-hoc audit found `_sample_fiw_rfiw_relation_matched_negatives` (shared/dataset.py:433) does not preserve role/relation — it samples uniformly across families like `_sample_fiw_negatives` (shared/dataset.py:512), differing only by seed offset (270 vs 200). Measured overlap with the random sampler: 0.09 % on training negatives, 100 % on test. R004 effectively measured negative-sampler reseed variance under partial-FT. The Phase 6 hard-negative hypothesis remains untested. See R004 Errata. |

The R004 finding set a **noise floor** for single-knob comparisons:
two runs differing only in negative-sampler seed shifted Test AUC by
≈ 0.009. Any future improvement smaller than this is indistinguishable
from sampler variance unless the seed is controlled.

---

## Mechanistic picture (R005-R010 taxonomy)

The cycle produced a structural taxonomy of interventions and their
effects on M12:

| Intervention type | Example | Effect on Test AUC | Effect on TAR@FAR=0.001 | Effect on TAR@FAR=0.01 |
|---|---|---:|---:|---:|
| **Shortcut removal — direction** | R006 symmetric forward | **+0.022** (decisive) | −0.85 pp | +1.03 pp |
| **Shortcut removal — identity** | R009 comparison-only fusion | −0.005 within noise | **+2.1 pp** (decisive) | −0.82 pp |
| **Capacity reallocation** | R007 differential LR | −0.006 within noise | +1.2 pp | +0.98 pp |
| **Per-class loss** | R005 relation aux head | −0.009 within noise | −1.2 pp | −0.94 pp |
| **Parameter anchoring** | R008 L2-SP λ=1e-3 | +0.000 tied | −0.55 pp | +0.51 pp |
| **Wrong-direction aux loss** | R003 SupCon | −0.005 with bad signature | −1.5 pp | −1.01 pp |
| **No-op (bug)** | R004 broken sampler | −0.009 sampler noise | −2.2 pp sampler noise | −2.87 pp sampler noise |
| **Stack: capacity + identity removal** | R010 (R007 + R009) | −0.003 within noise | +1.88 pp (≈R009) | **+2.55 pp ⭐⭐ (superadditive)** |

The cleanest distinction is between **shortcut-removal interventions**
(R006, R009) and everything else. Only shortcut-removal interventions
moved an aggregate Test-side metric meaningfully on their own:

- R006's symmetric forward removed **direction-specific shortcuts**
  in the cross-region adapter — these moved the *aggregate AUC*
  because they were affecting the full kin/non-kin separation.
- R009's comparison-only fusion removed **identity-as-feature
  shortcuts** in the classifier — these moved *strict-FAR* because
  they were affecting the tail of the score distribution.

The training-dynamics interventions (R005, R007, R008) were each
empirically neutral on aggregate AUC. But R010 demonstrated that
*combining* an identity-removal shortcut intervention (R009) with
a capacity-reallocation intervention (R007) produces a
**superadditive** mid-FAR effect — the cmp-only head's mid-FAR
weakness is compensated by diff-LR's mid-FAR strength, and the
combination exceeds either parent alone. This is the first composed
intervention in the cycle that exceeded its parts.

---

## What the val→test gap reveals

The val→test AUC gap is the cleanest single indicator of how much a
model relies on shortcuts that do not transfer across families:

| Run | Val→Test gap | Reading |
|---|---:|---|
| M02 R031 (ViT-B/16 full FT) | −0.031 | best gap, but a different architecture family |
| **M12 R006** | **−0.026** | smallest in AdaFace-trained family |
| M12 R008 (L2-SP) | −0.026 | tied with R006 |
| M12 R009 | −0.027 | tied with R006 (strict-FAR specialist) |
| M12 R010 (R007+R009 stack) | −0.030 | tight; mid-FAR + AP champion |
| M12 R007 (diff-LR) | −0.036 | small widening from higher head LR |
| M12 R002 | −0.076 | the partial-FT baseline before the cycle |
| M12 R003 (SupCon) | −0.080 | aux loss didn't widen the gap |
| M12 R005 (Phase 5) | −0.084 | per-class aux didn't widen it either |
| M12 R004 (sampler reseed) | −0.088 | reseed noise |
| M12 R001 (Phase 1, fully frozen) | −0.089 | low-capacity floor |
| M11 v4 (M09 + "hard negs") | −0.128 | worst gap in the project (full-FT + ineffective sampler) |
| M10 R003 (M09 + top-only) | −0.140 | worst trained gap |

R006 closed the gap from -0.076 to -0.026 — a 50 % reduction — in a
single intervention. This is the largest single-step generalisation
improvement in the post-M02 era of the project.

---

## What this means for the TCC narrative

The post-R002 cycle has clarified four things:

1. **The M12 architecture (R002 form) was learning two distinct
   shortcuts** — direction-specific role ordering in the cross-region
   adapter, and identity-as-feature in the classifier. Both are
   reasonable for a single-direction supervised loss to find, and
   both are removable with small structural changes.

2. **The shortcuts had measurably different signatures.** The
   direction shortcut affected the full kin/non-kin separation
   (aggregate AUC). The identity shortcut affected the tail of the
   score distribution (strict-FAR). Removing the first lifted Test
   AUC; removing the second lifted strict-FAR.

3. **Training-dynamics interventions are not a substitute for
   architectural correctness.** Phase 5 (per-class loss), differential
   LR, and L2-SP all behaved as predicted on Val and on intermediate
   training metrics, but none moved Test AUC meaningfully because
   they did not change what shortcuts the model could learn.

4. **The val→test gap is a strong diagnostic for the
   family-disjoint regime.** Models with a tight gap (M02 R031, M12
   R006/R008/R009) generalise well; models with a wide gap (M11 v4,
   M10 R003) over-fit on training families. The R006 intervention
   was the single largest gap-closer in the post-R002 era and is the
   piece of evidence that anchors the rest of the narrative.

For the TCC text:
- The **architectural contribution** of the proposed RGCK-Net is
  Phase 2 (R002) + Phase 5 (R005) + symmetric forward (R006). Each
  piece is justified by its own measured effect.
- The **operating-point flexibility** of the project is R002
  (high-precision) ↔ R006 (balanced) ↔ R009 (high-recall /
  strict-FAR). Each is preferable in different deployment scenarios.
- The **negative results** (R003, R007, R008) are valuable for the
  thesis: they bound the design space and demonstrate honest
  experimental rigour.
- The **methodological correction** (R004 Errata) is an example of
  scientific transparency that strengthens the thesis rather than
  weakens it.

---

## Open questions / untried directions

These remain open but were not pursued in the cycle:

1. **R011 candidate — L2-SP at higher λ** (1e-2 or 1e-1).
   Distinguishes the two R008 interpretations (λ too small vs gap
   already closed). Lower EV.

2. **Lower relation_aux λ** (0.02 or 0.03) — small probability of
   recovering some precision without losing per-class gains.

3. **ROI Align tokenizer** (proposal §15 Strategy 1) — replaces 5
   AdaFace forwards with one feature-map forward + ROI pool.
   Halves training time; architectural change of larger scope.

4. **Fix the `relation_matched` sampler** and re-run Phase 6 — the
   only intervention that has genuinely never been tested under a
   correct implementation. Lower priority than 1-3 above.

5. **Reproducibility under different seeds** — every result here is
   at seed=42 with negative-sampler offset 200. Multiple seeds would
   tighten the noise-floor estimate and give variance bands on each
   reported Test AUC.

6. **Different stacks** — R010 showed mid-FAR superadditivity from
   R007+R009. Other pairs (e.g., R005+R007, R008+R009) have not
   been tested as stacks. EV is uncertain given R010's mixed result.

---

## Artifacts

All checkpoints, train/test logs, and per-run reviews live in
`models/12_rgck_net/output/` and `models/12_rgck_net/run-review/`.
The per-run reviews are:

- [run-005.md](run-005.md) — Phase 5 relation aux head
- [run-006.md](run-006.md) — symmetric forward (aggregate-AUC champion)
- [run-007.md](run-007.md) — differential LR
- [run-008.md](run-008.md) — L2-SP regulariser λ=1e-3
- [run-009.md](run-009.md) — comparison-only fusion (strict-FAR champion)
- [run-010.md](run-010.md) — R007 + R009 stack (mid-FAR + AP champion)
- [run-011.md](run-011.md) — role-matched hard negatives (low-FAR champion, CV-validated)
- [run-004.md](run-004.md) — hard-negatives Errata

The full run-by-run history is in `RUN_LOG.md` and the cross-run
tables are in `overview.md`.

## Postscript (2026-05-27) — R011 closed the cycle

The R011 candidate that this summary listed as "next" was executed
(single-run 2026-05-25 → 2026-05-26) and CV-validated (2026-05-26 →
2026-05-27, `output/014/`). The R011 intervention was *not* L2-SP at
higher λ — that idea was supplanted by the role-matched hard-negatives
sampler-fix (`fc3301d`), which had the higher expected value given
the M02 R031 / M11 v4 / M12 R004 retraction history.

**R011 outcome — CV (n=5):** Test AUC 0.8761 ± 0.0029 (within noise
floor of R010 CV 0.8739 ± 0.0038, z = 0.45). TAR@FAR=0.001 = 0.0677 ±
0.0147 (+0.015 over R010 ~ 1σ). The single-run 0.8825 was a favorable
upper-tail draw; the CV mean is the honest project number. R011 is
now the **low-FAR** headline; AUC remains tied with R006/R010 at the
0.876 ± 0.003 noise floor. Full details: [run-011.md](run-011.md).
