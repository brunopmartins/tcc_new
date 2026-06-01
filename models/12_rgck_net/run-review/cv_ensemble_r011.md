# CV-fold ensemble — R011 (NEW PROJECT HEADLINE, executed 2026-05-29)

**Status:** Executed 2026-05-29 against the 5 R011 CV checkpoints in
[`output/014/fold_{0..4}/checkpoints/best.pt`](../output/014/). Inference
ran in ~30 min on the RX 6750 XT.

**Outcome — new project headline on every threshold-invariant test metric:**

| Metric | **Ensemble** | Per-fold mean (CV) | Δ |
|---|---:|---:|---:|
| Test ROC AUC | **0.8839** | 0.8761 ± 0.0029 | **+0.0078** |
| Test AP | **0.8657** | 0.8562 ± 0.0031 | +0.0095 |
| TAR@FAR=0.001 | **0.0801** | 0.0677 ± 0.0147 | +0.0124 |
| TAR@FAR=0.01 | **0.2172** | 0.2069 ± 0.0107 | +0.0103 |
| TAR@FAR=0.1 | **0.6063** | 0.5951 ± 0.0083 | +0.0112 |
| F1 | **0.8054** | 0.7979 ± 0.0024 | +0.0075 |
| Accuracy | **0.7925** | 0.7858 ± 0.0036 | +0.0067 |
| Precision / Recall | 0.7313 / 0.8961 | 0.7282 / 0.8826 | +0.003 / +0.014 |

Ensemble threshold = mean of the 5 per-fold val-selected thresholds = 0.330
(per-fold thresholds: 0.40, 0.40, 0.30, 0.40, 0.15).

**Vs all prior headlines:**

- vs **R010 CV** (0.8739 ± 0.0038): +0.0100 AUC (≈2σ — first reproducible
  break of the 0.876 ceiling).
- vs **R011 single-run** (0.8825, upper-tail draw): +0.0014 AUC,
  +0.0050 TAR@FAR=0.001 — even the upper-tail draw is beaten.
- vs **R011 CV mean** (0.8761 ± 0.0029): +0.0078 AUC, +0.0124 low-FAR
  (~0.85σ on AUC, similar on low-FAR — meaningful when paired with the
  zero-cost methodology).
- vs **M02 R031 CV** (0.8462 ± 0.0040): +0.0377 AUC.

## Idea

R011's 5-fold CV produced 5 models trained on family-disjoint splits of
the training+val set, all evaluated on the *same* fixed RFIW Track-I test
set (13 425 pairs). The 5 test-set sigmoid prediction vectors are
already statistically independent in a useful sense: each model saw a
disjoint subset of training families, so the variance in their
predictions on test pairs reflects genuine model-disagreement rather
than a noise floor.

Averaging the 5 sigmoid outputs is the cheapest plausible way to push
past the 0.876 ± 0.003 AUC ceiling that R006/R010/R011 all hit in CV.
No new training; no new architecture; ~30 min of inference time.

## Method

[`AMD/cv_ensemble.py`](../AMD/cv_ensemble.py) +
[`AMD/cv_ensemble_r011.sh`](../AMD/cv_ensemble_r011.sh).

For each fold k ∈ {0..4}:

1. Load `output/014/fold_k/checkpoints/best.pt`.
2. Reconstruct the M12 model from the checkpoint's `model_config`
   (handles all R006–R012 flags via `model_config.get(...)` defaults).
3. Run the canonical test loader, record sigmoid probabilities.
4. Reproduce the per-fold test metrics (sanity-check against the
   `test_metrics_rocm.txt` file already written by that fold).
5. Read the fold's val-selected F1-optimal threshold from the
   checkpoint's `protocol` metadata.

Then:

- `ensemble_probs = mean(sigmoid_probs across 5 folds)` per test sample.
- `ensemble_threshold = mean(per_fold_thresholds)` — a calibration
  choice that avoids re-tuning on test data. (Each fold's val set is
  disjoint, so we cannot average val predictions; the mean threshold
  is the cleanest test-data-free aggregate.)
- Compute the full metric pack on the ensembled predictions.

Output:
- `output/014/ensemble/ensemble_metrics.txt` — single-page summary
  with ensemble metrics, per-fold reproduction, and the threshold table.
- `output/014/ensemble/ensemble_probs.npz` — `probs_per_fold (5, N)`,
  `ensemble_probs (N,)`, `labels`, `thresholds`, `ensemble_threshold`.
  Lets us slice into per-relation or per-fold disagreement later
  without re-running inference.

## Expected outcome

Soft ensemble of CV folds historically gives:
- **+0.005 to +0.010 Test AUC** above the per-fold mean.
- Bigger lift at strict FAR if the per-fold low-FAR predictions are
  uncorrelated. R011's per-fold TAR@FAR=0.001 varies 0.051–0.083
  (~0.03 swing) — meaningful diversity to exploit.
- Tiny or zero lift on threshold-dependent metrics (F1, recall) because
  threshold averaging is conservative.

Best case: ensemble Test AUC reaches 0.882–0.886, finally clearing the
"R006-family ceiling" headline with a CV-grade number (since each of the
5 models is itself a CV-validated draw). Worst case: matches the
per-fold mean (0.8761) — confirms the predictions are too correlated
for ensembling to help.

## Why this is defensible in the TCC

Reporting a CV-fold ensemble as the project headline is standard in
biometric verification and ML competitions; it shows the variance of
the single-model number is genuinely *model variance*, not just sampling
noise. The 5 best.pt files were produced by an honest 5-fold protocol;
the ensemble inherits the same protocol guarantees on the test set.

## Decision rule

After running the ensemble:
- **AUC ≥ 0.880**: report as the project's deployment headline number;
  cite the per-fold mean ± std alongside for the variance picture.
- **AUC in [0.876, 0.880]**: same headline but framed as "ensembling
  recovers ~half of the inter-fold variance"; not a breakthrough.
- **AUC < 0.876**: ensemble adds nothing — the folds learned the same
  function. File the result and move on.

In all three cases the TAR@FAR=0.001 lift is also reported, since R011
is already the project's low-FAR headline.

## Cost

- ~5 min inference per fold × 5 folds = ~30 min wall-clock.
- No training. GPU thermal load is low (test mode only).
- Disk: `ensemble_probs.npz` is ~600 KB (5 × 13 425 float64 + labels).

---

## Result (2026-05-29 execution)

The ensemble cleared the 0.876 ± 0.003 ceiling that R006/R010/R011 had
all hit individually. Best-case outcome from the pre-registered decision
rule: **AUC ≥ 0.880 → report as the project's deployment headline**.

**Project ranking after the ensemble (Test ROC AUC, single-number form):**

1. **M12 R011 CV ensemble: 0.8839** (NEW HEADLINE)
2. M12 R011 single-run: 0.8825 (upper-tail draw, retracted)
3. M12 R011 CV mean: 0.8761 ± 0.0029
4. M12 R006 single-run: 0.8788
5. M12 R010 CV mean: 0.8739 ± 0.0038
6. M12 R006 CV mean: 0.8733 ± 0.0038
7. M12 R002 single-run: 0.8564
8. M02 R031 CV mean: 0.8462 ± 0.0040
9. M02 R031 single-run: 0.850
10. M13 R002 single-run: 0.8526 (line stopped)
11. M13 R001 single-run: 0.8462 (line stopped)

The ensemble result is not a "free" win — it's the payoff for the
5-fold CV investment in `output/014/`. Reporting it as the headline
requires explicit framing: it is *not* a single model. The CV mean and
spread should be cited alongside whenever the ensemble number is the
quoted figure.

### Artifacts

- `output/014/ensemble/ensemble_metrics.txt` (full metric table +
  per-fold reproduction)
- `output/014/ensemble/ensemble_probs.npz` (probs_per_fold (5, N) +
  ensemble_probs (N,) + labels + thresholds)
- `output/014/ensemble/ensemble.log` (full run log)

### Per-fold reproduction (sanity check)

All 5 folds reproduced their original `test_metrics_rocm.txt` exactly:
0.8767 / 0.8754 / 0.8771 / 0.8796 / 0.8716 — confirming the ensemble
inference is operating on the same checkpoints that produced the CV
mean, with no drift.
