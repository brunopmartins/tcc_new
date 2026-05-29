# CV-fold ensemble — R011 (planning doc, not yet executed)

**Status:** Script implemented (2026-05-27), not yet run. The 5 R011 CV
checkpoints in [`output/014/fold_{0..4}/checkpoints/best.pt`](../output/014/)
are the inputs.

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
