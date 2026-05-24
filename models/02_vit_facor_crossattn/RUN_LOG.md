# Run Log — Model 02: ViT + FaCoR Cross-Attention

This file logs every training run for this model with the exact configuration and final metrics. The goal is reproducibility: anyone (human or agent) reading this should be able to recreate any run by copying its launch command.

Newest run on top.

---

## 5-fold CV study — 2026-05-22 → 2026-05-24 — Family-disjoint CV of R031

**Status:** Completed. 5 folds in `output/033/fold_{0..4}/`, ~48 h GPU wall-clock total.

**Outcome:**
- **R031 5-fold mean Test AUC: 0.8462 ± 0.0040** (single-run 0.850 was +0.96σ above mean — favorable but not an outlier).
- Test AP CV mean: 0.8131 ± 0.0045 (single-run 0.817, +0.87σ).
- TAR@FAR=0.001 CV mean: 2.19 ± 0.63 % (single 2.5 %, +0.49σ).
- TAR@FAR=0.01 CV mean: 13.49 ± 1.19 % (single 14.0 %, +0.43σ).
- TAR@FAR=0.1 CV mean: 49.64 ± 2.55 % (single 49.9 %, +0.10σ).
- σ_AUC = 0.0040 matches M12's σ_AUC = 0.0038 — same family-fold noise floor across architectures.

**Cross-model CV-vs-CV comparison (same methodology):**
- M02 R031 CV: AUC 0.8462 ± 0.0040
- M12 R006 CV: AUC 0.8733 ± 0.0038 (**+0.0271 vs M02, ~6.8σ**)
- M12 R010 CV: AUC 0.8739 ± 0.0038 (**+0.0277 vs M02, ~7.0σ**)

**The M12 architecture's advantage over M02 R031 is statistically massive (~7σ) and stable across folds.** This is the strongest cross-model claim in the project, anchored by CV rather than single-run draws.

### Setup

Same family-disjoint 5-fold splitter as M12 ([shared/dataset.py `create_fiw_5fold_train_val_loaders`](../../shared/dataset.py)). 571 unique training families → 5 disjoint groups of 114-115. Official RFIW Track-I test (13,425 face-level pairs) preserved across folds.

M02 train.py extended with `--fold` and `--num_folds` flags; run_pipeline.sh accepts FOLD/NUM_FOLDS/RUN_OVERRIDE env vars. Runner: `models/02_vit_facor_crossattn/AMD/cv_runner.sh` (5-fold sequential with 15-min cooldowns + skip-if-completed).

### Launch command (one fold)

```bash
SKIP_INSTALL=1 \
ALIGNED_ROOT=$PROJECT_ROOT/datasets/FIW_aligned \
TRAIN_DATASET=fiw \
EPOCHS=50 BATCH_SIZE=32 \
LEARNING_RATE=5e-6 WEIGHT_DECAY=1e-5 \
WARMUP_EPOCHS=5 MIN_LR=1e-7 SCHEDULER=cosine \
VIT_MODEL=vit_base_patch16_224 \
CROSS_ATTN_LAYERS=2 CROSS_ATTN_HEADS=8 \
LOSS=contrastive TEMPERATURE=0.3 MARGIN=0.3 \
DROPOUT=0.2 FREEZE_VIT=0 \
NEGATIVE_RATIO=2.0 EVAL_NEGATIVE_RATIO=1.0 \
TRAIN_NEGATIVE_STRATEGY=relation_matched EVAL_NEGATIVE_STRATEGY=random \
NUM_WORKERS=4 PATIENCE=20 SEED=42 \
RUN_OVERRIDE=033 FOLD=$K NUM_FOLDS=5 \
bash models/02_vit_facor_crossattn/AMD/run_pipeline.sh
```

### Per-fold Test ROC AUC

| Fold | Train fids | Val fids | Val peak AUC (ep) | Test AUC |
|---:|---:|---:|---|---:|
| 0 | 456 | 115 | 0.8807 (4) | 0.8445 |
| 1 | 457 | 114 | 0.8479 (3) | 0.8498 |
| 2 | 457 | 114 | 0.8472 (2) | 0.8477 |
| 3 | 457 | 114 | 0.8740 (4) | 0.8400 |
| 4 | 457 | 114 | 0.8743 (3) | 0.8489 |
| **Mean** | — | — | — | **0.8462** |
| **Std** | — | — | — | **0.0040** |

### Implementation notes

- `LOSS=contrastive` (not `cosine_contrastive`) was the recipe — caught by loss-scale mismatch on first fold 0 attempt.
- Fold 2 was interrupted mid-training by an `rm -rf` on the output dir while the train process was still saving checkpoints. The best.pt (ep 2, Val AUC 0.8472) survived and was tested manually with `--threshold 0.900` (val-selected threshold). AUC/AP/TAR are threshold-invariant.

Full breakdown in [run-review/cv_5x_R031.md](run-review/cv_5x_R031.md).

---

## Historical runs (026-032) — pre-RUN_LOG

These runs predate the RUN_LOG.md system. Their results are summarized in [docs/pt/09_visao_geral_pesquisa.md](../../docs/pt/09_visao_geral_pesquisa.md) under "Evolucao dos Resultados (Modelo 02 — FIW)". Configuration details and per-epoch trajectories are recoverable from `output/0XX/logs/train.log`.

Best historical run: **Run 031** — Test ROC AUC = **0.850** (single-run); **CV mean 0.8462 ± 0.0040** (n=5, family-disjoint over RFIW Track-I, see CV study above). Single-run was +0.96σ above CV mean — favorable but not an outlier.
- Config: no freeze, LR=5e-6, warmup=5, dropout=0.2
- This is the canonical Model 02 result used in all comparison tables. Quote `0.8462 ± 0.0040` when reporting alongside CV'd M12 numbers; quote `0.850` only when comparing to non-CV'd literature.

For new runs, follow the template in [prompts/train_model.md](../../prompts/train_model.md) (local) and append entries above this section.

---
