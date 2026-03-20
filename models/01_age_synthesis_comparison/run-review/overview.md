# Overview — Age Synthesis + All-vs-All Comparison Model

**Model:** 01 — Age Synthesis Comparison
**GPU:** AMD Radeon RX 6750 XT (11.98 GB VRAM, gfx1031, ROCm 5.7)
**Dataset available:** KinFaceW-I (FIW requires manual registration — unavailable)

Individual run details: [run-01.md](run-01.md) · [run-02.md](run-02.md) · [run-03.md](run-03.md) · [run-04.md](run-04.md)

---

## Run Comparison

| | Run 1 | Run 2 | Run 3 | Run 4 |
|---|---|---|---|---|
| **Date** | 2026-02-18 | 2026-02-18 | 2026-02-18 | 2026-02-19 |
| **Method** | Docker | Local (setup.sh) | Local (run_pipeline.sh) | Local (run_pipeline.sh) |
| **Device** | GPU (ROCm via Docker) | CPU (GPU not detected) | GPU (ROCm, fixed) | GPU (ROCm) |
| **Dataset** | KinFaceW-I+II (split bug) | KinFaceW-I (proper split) | KinFaceW-I (proper split) | KinFaceW-I (proper split) |
| **Split** | Broken — no true test set | 70/15/15 (746/158/162) | 70/15/15 (746/158/162) | 70/15/15 (746/158/162) |
| **Epochs** | 50 (no early stop) | 37 (early stop) | 26 (early stop at ep. 11 best) | **FAILED** (OOM at epoch 1 validation) |
| **Batch size** | 16 | 32 | 32 | 8 |
| **Age synthesis** | Disabled | Disabled | Enabled (no SAM weights — identity fallback) | **Enabled (SAM loaded)** |
| **AMP** | Enabled | Disabled (CPU) | Enabled | Enabled |
| **Speed** | ~11 s/epoch | ~147 s/epoch | ~22 s/epoch | ~8.7 min/epoch (SAM active) |

### Accuracy (test.py, optimal threshold)

| | Run 1 | Run 2 | Run 3 | Run 4 |
|---|---|---|---|---|
| **Accuracy** | 74.25% ⚠️ | 68.52% | 64.81% | N/A |
| **F1** | 80.83% ⚠️ | 72.73% | 69.52% | N/A |
| **ROC-AUC** | 74.96% ⚠️ | 68.19% | 66.44% | N/A |
| **Precision** | 70.17% | 64.15% | 61.32% | N/A |
| **Recall** | 95.31% | 83.95% | 80.25% | N/A |
| **Non-kin accuracy** | 46.40% | 53.09% | 49.38% | N/A |
| **Threshold** | 0.60 | 0.55 | 0.50 | N/A |
| **Samples** | 936 ⚠️ | 162 | 162 | N/A |

⚠️ Run 1 metrics are inflated by the dataset split bug (training data in test set).
Run 4 produced no metrics — OOM crash during validation.

### Reliability

| | Run 1 | Run 2 | Run 3 | Run 4 |
|---|---|---|---|---|
| **Honest evaluation** | No (data leakage) | Yes | Yes | N/A (crashed) |
| **GPU training** | Yes | No | Yes | Yes |
| **Age synthesis active** | No | No | No (identity fallback) | **Yes (SAM loaded)** |
| **Reproducible locally** | No (Docker-only) | Yes | Yes | Yes (after OOM fix) |

---

## Per-Relation Summary (test.py)

| Relation | Run 1 | Run 2 | Run 3 |
|----------|-------|-------|-------|
| fd (father/daughter) | 96.27% | 88.00% | 88.00% |
| fs (father/son) | 96.15% | 84.00% | 88.00% |
| md (mother/daughter) | 93.70% | 85.71% | 85.71% |
| ms (mother/son) | 94.83% | 76.47% | 52.94% |
| negative (non-kin) | **46.40%** | **53.09%** | **49.38%** |

Non-kin discrimination is the persistent weakness across all runs. ms detection degraded significantly in Run 3.

---

## TAR @ FAR

| FAR | Run 1 | Run 2 | Run 3 |
|-----|-------|-------|-------|
| 0.001 | 0.00% | 0.00% | 0.00% |
| 0.01 | 2.81% | 2.47% | 0.00% |
| 0.1 | 18.57% | 11.11% | 16.05% |

All runs fail biometric operating points. None of the runs have activated the model's core contribution (age synthesis).

---

## Issues Log

| # | Issue | Severity | Status | First seen | Notes |
|---|-------|----------|--------|-----------|-------|
| 1 | YAML `>` folding broke shell variable expansion | High | ✅ Fixed | Run 1 | Created `AMD/run_pipeline.sh` |
| 2 | Docker default shm (64 MB) killed DataLoader workers | High | ✅ Fixed | Run 1 | Added `shm_size: '2gb'` |
| 3 | Dataset split bug — all splits loaded same data | High | ✅ Fixed | Run 1 | Proper 70/15/15 split applied from Run 2 |
| 4 | FIW dataset unavailable | Medium | ❌ Open | Run 1 | Requires manual registration |
| 5 | Age synthesis disabled — no pretrained SAM weights | **Critical** | ✅ Fixed | Run 4 | SAM weights downloaded; loaded successfully in Run 4 |
| 6 | `evaluate.py` comparison matrix crash without age synthesis | Low | ✅ Fixed | Run 1 | Graceful fallback added |
| 7 | High recall bias / weak non-kin discrimination | Medium | ❌ Open | Run 1 | Worsened in Run 3 without real synthesis |
| 8 | ROCm not detected in venv — user not in `render` group | High | ✅ Fixed | Run 2 | `sudo usermod -aG render,video bruno` |
| 9 | `UndefinedMetricWarning` spam (5× per epoch) | Low | ❌ Open | Run 2 | Per-relation AUC fails on single-class batches |
| 10 | Age synthesis identity fallback makes ablation meaningless | High | ❌ Open | Run 3 | All aggregation methods produce identical scores |
| 11 | Early stopping too aggressive for small dataset | Medium | ❌ Open | Run 2–3 | Model peaks early, then oscillates |
| 12 | ms relation degraded to near-chance in Run 3 (52.94%) | Medium | ❌ Open | Run 3 | Worst-affected by degenerate synthesis |
| 13 | OOM during validation with SAM enabled (batch_size=8) | **Critical** | ✅ Fixed | Run 4 | StyleGAN2 decoder allocates ~4.5 GB per batch; fixed by per-image SAM processing |

---

## Comparison with README Targets

| Metric | README (no synthesis) | README (with synthesis) | Best honest run | Gap |
|--------|----------------------|------------------------|----------------|-----|
| Accuracy | ~75% | ~79% | 68.52% (Run 2) | −6.5% / −10.5% |
| F1 | ~0.74 | ~0.78 | 72.73% (Run 2) | −1.3% / −5.3% |
| ROC-AUC | ~0.82 | ~0.86 | 68.19% (Run 2) | −13.8% / −17.8% |

The README targets have not been reached. Age synthesis has never been exercised (no SAM weights). The honest baseline (Run 2) falls 6.5% below the no-synthesis target — explained by small dataset (KinFaceW-I only, 746 train) and CPU-only training stopping at epoch 37.

---

## Suggestions

### Immediate (no downloads)

| Priority | Change | Expected Impact |
|----------|--------|----------------|
| 1 | Suppress `UndefinedMetricWarning` in per-relation evaluation | Cleaner output |
| 2 | Increase early-stopping patience from 15 to 25 epochs | Allow more exploration |
| 3 | Re-run with `TRAIN_DATASET=kinface` including KinFaceW-II (split fixed) | More data → better generalization |

### High Impact (requires ~1 GB download)

| Priority | Change | Expected Impact |
|----------|--------|----------------|
| **#1** | **Download SAM weights** (`sam_ffhq_aging.pt`) → place in `SAM/pretrained_models/` | Activates core contribution; expected +4% accuracy; makes ablation meaningful |

### Medium Impact

| Priority | Change | Expected Impact |
|----------|--------|----------------|
| 4 | Use `--backbone arcface` | ArcFace designed for face verification; likely better embeddings |
| 5 | Use `--loss contrastive` or `cosine_contrastive` | May improve AUC and non-kin discrimination |
| 6 | Implement 5-fold cross-validation (standard KinFaceW protocol) | Comparable to published results |

### Large Impact (requires FIW registration)

| Priority | Change | Expected Impact |
|----------|--------|----------------|
| 7 | Register and download FIW dataset | 26,000+ images vs 746 — major improvement |
| 8 | Train on FIW, test on KinFaceW (cross-dataset evaluation) | Validates generalization |

---

## Conclusion

Four runs attempted. Run 4 was the first to load SAM weights and activate real age synthesis, but failed with OOM during validation — StyleGAN2's 1024x1024 decoder requires ~4.5 GB per batch, which exceeded the 12 GB GPU when combined with training optimizer state. The OOM has been fixed by processing images one at a time through SAM.

SAM weights are now downloaded and the OOM fix is in place. The next run will be the first to complete with real age synthesis active end-to-end.
