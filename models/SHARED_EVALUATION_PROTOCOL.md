# Shared Evaluation Protocol

This repository now uses one common evaluation protocol across all kinship models.

## Core Rules

1. **Disjoint splits only**
   - `KinFaceW`: pair-disjoint splits by pair ID
   - `FIW`: family-disjoint splits by family ID

2. **Validation chooses the operating point**
   - Early stopping monitors validation `roc_auc`
   - The classification threshold is selected on the validation split only
   - Test scripts reuse the threshold stored in the checkpoint protocol metadata

3. **Best checkpoint, not last checkpoint**
   - Training scripts reload `best.pt` before final evaluation
   - Reported test metrics come from the validation-selected best checkpoint

4. **Shared metrics**
   - `accuracy`
   - `balanced_accuracy`
   - `precision`
   - `recall`
   - `f1`
   - `roc_auc`
   - `average_precision`
   - `tar@far`
   - per-relation metrics when available

5. **Cross-dataset reporting**
   - Keep train and test datasets explicit
   - Use the same stored validation threshold when running cross-dataset evaluation

6. **Reproducibility**
   - Seed Python, NumPy, and PyTorch
   - Store protocol metadata in checkpoints and `protocol_summary.json`

## Files Added / Updated

- `models/shared/dataset.py`
  - deterministic splits
  - family-disjoint FIW support
  - CV fold support with inner validation split

- `models/shared/evaluation.py`
  - shared score extraction for logits, embedding pairs, and dict outputs
  - richer metric reporting

- `models/shared/protocol.py`
  - seed setup
  - validation-calibrated evaluation
  - checkpoint metadata helpers

- `models/shared/Nvidia/trainer.py`
- `models/shared/AMD/trainer.py`
  - shared `monitor_metric`
  - best-checkpoint selection by validation metric

## Checkpoint Metadata

Every patched training entrypoint now stores:

- `model_config`
- `protocol.protocol_version`
- `protocol.train_dataset`
- `protocol.test_dataset`
- `protocol.split_seed`
- `protocol.negative_ratio`
- `protocol.selected_threshold`
- `protocol.threshold_metric`
- `protocol.threshold_source`
- `protocol.monitor_metric`
- `protocol.seed`

## Reporting Guidance

- Prefer `roc_auc` as the main model-selection metric
- Report threshold-free and thresholded metrics together
- For final comparisons, use:
  - repeated seeds, or
  - 5-fold CV with `mean ± std`

The existing `models/02_vit_facor_crossattn/AMD/train_cv.py` now follows this rule by keeping the held-out fold separate from the inner validation split used for early stopping and threshold calibration.

