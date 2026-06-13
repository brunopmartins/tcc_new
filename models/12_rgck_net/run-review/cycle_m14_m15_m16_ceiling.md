# Post-R012 campaign — can anything break the 0.876 ceiling? (M14 / M15 / M16)

**Period:** 2026-06-10 → 2026-06-13
**Question:** R006/R010/R011/R012 all hit a 5-fold-CV Test-AUC ceiling of
**0.876 ± 0.003**; only *averaging* the R011 CV folds (the R011 ensemble, **0.8839**)
ever broke it. Three sibling models were built to attack that ceiling from three
different angles, each a clean single-variable change on the **M12 R011 recipe**
(symmetric forward + comparison-only fusion + relation-aux 0.05 + role-matched hard
negatives 30%):

| Model | Lever | Hypothesis |
|---|---|---|
| **M14** `14_rgck_lora` | LoRA backbone adaptation (r16) instead of stage-4 unfreeze | low-rank regularisation generalises better |
| **M15** `15_rgck_hires` | hi-res ROI-Align region tokenizer (224→14×14 / 160→10×10) | undistorted finer region grid carries more signal |
| **M16** `16_fa_rgck` | family-adversarial DANN (each FIW family = a domain) | directly removing family identity closes the val→test gap |

All evaluated on the **fixed RFIW Track-I test set (13 425 pairs)**, the same
protocol as the M12 line.

---

## Results (single runs unless noted)

| Run | Test AUC | AP | TAR@FAR .001 | TAR@FAR .01 | TAR@FAR .1 | Val→test gap |
|---|---:|---:|---:|---:|---:|---:|
| M12 R011 (CV mean) | 0.8761 ± .003 | 0.8562 | 0.0677 | 0.2069 | 0.5950 | −0.021 |
| M12 R011 (CV ensemble) | **0.8839** | 0.8657 | 0.0801 | 0.2172 | 0.6063 | — |
| M12 R012 (single) | 0.8813 | 0.8636 | 0.0878 | 0.2088 | 0.6105 | −0.026 |
| **M14** LoRA | 0.8700 | 0.8491 | 0.0636 | 0.1953 | 0.5763 | **−0.032** |
| **M16 R001** (λ≈0.15, w0.1) | 0.8774 | 0.8562 | 0.0379 | 0.2063 | 0.5906 | −0.021 |
| **M16 R002** (λ≈0.68, w0.3) | 0.8701 | 0.8473 | 0.0470 | 0.1715 | 0.5776 | −0.025 |
| **M15** hi-res @160 | 0.8803 | **0.8760** | **0.0992** | **0.2906** | **0.6465** | −0.021 |

Bold = best-in-project. M15's AP and full TAR@FAR curve beat **even the R011
ensemble** — from a single model.

---

## Per-lever verdict

### M14 — LoRA backbone: BELOW (negative)
AUC 0.8700, and the val→test gap **widened** to −0.032 (vs R011 −0.021). LoRA did
not regularise toward better generalisation; its ~0.9 M adapters underfit vs the
31 M stage-4 unfreeze. Train loss still fell to ~0.13 (it memorised the train set;
what it avoided was a *val collapse*). Confounds: best.pt = ep5 = the warmup
boundary; LoRA rank untuned (N=1). **Suggestive, not settled** → would need a rank
sweep + post-warmup checkpoint + CV.

### M16 — family-adversarial DANN: UNPRODUCTIVE (as configured)
- **R001** tied R011 (0.8774) but was an **invalid test**: the DANN λ ramp was
  tied to `--epochs`=100 while the run stopped at ep14, so the best-val checkpoint
  (ep3) saw **λ≈0.15** (effective weight ≈0.015) — essentially R011 + a 1.5% nudge.
- **R002** fixed it (`dann_max_epochs=6`, γ=5, weight 0.3 → **λ≈0.68 at the ep3
  checkpoint**) + per-term loss logging. Result: AUC **0.8701** (below R011), gap
  −0.025 (**not** tighter). The treatment, applied at strength, **mildly hurt** and
  did not close the gap.
- **Decomposition (R002):** `base(BCE+rel)` fell 0.74→0.28 (did not collapse to
  ~0.01), but **`CE_family(raw)` stayed at ~6.1 ≈ chance (ln 486 = 6.19) the whole
  run** — the 486-way discriminator never learned to predict family, so the GRL
  gradient was weak/noisy (it acted more as a gradient-noise regulariser than true
  invariance). The **486-way long-tailed** family target is the likely culprit.
  → If pursued: coarser domain buckets, or warm-start from R011 + constant λ.

### M15 — hi-res ROI-Align: POSITIVE on ranking (the one win)
AUC 0.8803 (top-of-band, within noise of R011 CV, below the 0.8839 ensemble — **no
aggregate-AUC break**), but **AP 0.8760 and the entire TAR@FAR curve are the best
in the project**, beating even the R011 ensemble (AP +0.010; TAR@FAR=0.001 0.099 vs
0.080; 0.01 0.291 vs 0.217; 0.1 0.647 vs 0.606). Threshold-invariant ⇒ genuine
ranking gains, not the thr-0.550 operating point. The undistorted, finer region
grid sharpens the **low-FAR tail** and positive-class ranking — exactly where
kinship verification cares. **This is the @160 *weakened* proxy** (the planned @224
/ 14×14 is GPU-blocked); it still won. Caveats: single run, best.pt = ep4 (warmup
boundary), manual finalize. **→ CV M15@160 next** to confirm; a CV-ensemble could
lift AUC too.

---

## Methodological lessons (apply to every future run here)

1. **Report the headline diagnostic.** Always log/report **Val AUC + val→test gap**
   — decision rules depend on it; the first M16 table omitted it.
2. **Decompose multi-term losses.** A high total loss is meaningless if a big
   auxiliary term (e.g. 0.1·CE_family ≈ 0.62 at chance) hides the BCE. Log each
   term separately (added to M16 train.py).
3. **Decouple the DANN/aux schedule from `--epochs`.** Safeguard stops ~ep14, so a
   ramp calibrated for 100 epochs never reaches strength at the selected checkpoint.
4. **Hardware: any backbone input > 112 is MIOpen-slow on this gfx1031 card**
   (~5 h/epoch @160, ~10 h @224, vs ~5 min @112) — the AdaFace conv body has no fast
   kernel for non-112 shapes. ROI-Align @224 is correct but impractical here.
5. **Early-peak + flat-val runs don't trip the decline-based safeguard** (M14 ran to
   ep22, M15 would've run ~25 h past peak). Manually stop + finalise on best.pt
   (transplant `model_config` + the peak-epoch threshold) to get the result cheaply.
6. **N=1 caution.** Single weight/seed/run + early/warmup-boundary checkpoints →
   verdicts are provisional until CV / multi-seed.

---

## Corrected conclusion

- **The aggregate-AUC ceiling (~0.876 CV) holds.** No lever produced a clean
  single-model AUC break; the R011 CV-ensemble (0.8839) remains the AUC headline.
- **But the ceiling is not the whole story for deployment.** M15's hi-res region
  tokenizer **materially improves AP and the low-FAR operating curve** — the most
  operationally relevant regime — beating even the ensemble there.
- **Family-memorisation removal (M16) did not help** even when finally tested at
  strength, and the discriminator-at-chance result suggests family identity is not
  cleanly separable / removable as configured — consistent with the wall being
  **representational/data-bound** rather than simple overfitting (though M14's
  memorisation + M16's discriminator failure mean this is *supported*, not proven).
- **LoRA (M14) is below** the unfreeze and widened the gap.

### Next steps (priority)
1. **CV M15 @160** (5 family-disjoint folds) — confirm the AP/low-FAR win; then
   CV-ensemble for a possible AUC lift. Highest EV.
2. M15 @224 if a faster GPU becomes available (full-strength hi-res).
3. (Lower EV) M16 with coarser domain buckets or warm-start+constant-λ; M14 rank
   sweep. Both look unpromising.

Per-model detail: [`../../14_rgck_lora/run-review/run-001.md`](../../14_rgck_lora/run-review/run-001.md),
[`../../15_rgck_hires/run-review/run-001.md`](../../15_rgck_hires/run-review/run-001.md),
[`../../16_fa_rgck/run-review/run-001.md`](../../16_fa_rgck/run-review/run-001.md),
[`../../16_fa_rgck/run-review/run-002.md`](../../16_fa_rgck/run-review/run-002.md).
